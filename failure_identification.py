"""Human-calibrated, resumable response-level failure identification (RQ1).

This module never invokes a model, assigns failure subtypes, or uses audit labels
for calibration. Unspecified scientific settings remain unresolved.

Executable policy contracts
===========================

The following mappings occupy existing policy fields in config.yaml. They must
be explicitly configured and documented through the configuration's provenance
mechanism before formal execution.

failure_identification.hard_label_rules::

    {
      "tasks": {
        "affected_systems": {"route": "hard"},
        "threat_report_alignment": {"route": "similarity"}
      },
      "annotation_protocols": {
        "anchor": <AnnotationService protocol>,
        "boundary": <AnnotationService protocol>
      }
    }

For a similarity-only configuration, annotation_protocols may instead be placed
inside pass_failure_range_estimator. Conflicting protocol declarations are
rejected.

evaluation.scoring.response_level_failure_criteria maps hard-routed task names
to explicit rules::

    {
      "affected_systems": {
        "pass_if": {
          "op": "equal",
          "left": {"source": "prediction", "pointer": "/answer/decisions"},
          "right": {"source": "reference", "pointer": "/answer/decisions"}
        },
        "on_missing": "review",
        "availability": {
          "transport_failure": "error",
          "refusal": "review",
          "invalid_schema": "review",
          "NEED_MORE_CONTEXT": "review",
          "UNSUPPORTED": "failure"
        }
      }
    }

Availability and on_missing actions are pass, failure, review, or error. They
are identification criteria, NOT task-metric replacement policies.

Predicate operators:
  equal, not_equal, subset, set_equal, all, any, not, length_equal,
  absolute_difference_le, between.

Operands are {"literal": <JSON>} or {"source": ..., "pointer": <RFC6901>}.
Sources are prediction, reference, instance, and contribution. Contribution
access is restricted to extracted targets/predictions/unit IDs; aggregate metric
values and metric-derived correctness shortcuts are unavailable.

similarity_measure_by_task maps task names to stored-score declarations::

    {
      "threat_report_alignment": {
        "source": "stored_score",
        "pointer": "/similarity_score",
        "score_version": "explicit-version",
        "version_pointer": "/similarity_version",
        "direction": "higher_is_better",
        "scale": [0, 100]
      }
    }

The declared scalar must already be persisted in a score record. This module
does not substitute sentence BLEU for corpus BLEU or recompute another measure.
Score records must identify response_id, instance_id, task, and config_hash.

tied_score_policy::

    {"mode": "split_by_response_id" | "keep_together",
     "partition": "rank_floor"}

Partitions use floor(rank / (population_size * quantile_width)), capped at the
last nominal bin. Empty bins are omitted. keep_together assigns all equal scores
to the bin of their first rank. Actual memberships and sizes are persisted.

inspection_order::

    {"group_order": "lexical", "score_order": "ascending" | "descending"}

pass_failure_range_estimator::

    {
      "method": "anchor_extrema",
      "lower_inclusive": true,
      "upper_inclusive": true,
      "one_class_policy": "review_all" | "observed_range",
      "contradictory_anchor_policy": "retain_overlap" | "review_all",
      "small_stratum_policy": "all" | "error",
      "annotation_protocols": {...}  # optional alternative location
    }

anchor_extrema fits only the observed within-class minimum/maximum interval.
It never extrapolates beyond observed scores. Contradictory anchors mean equal
scores receiving different final judgments. Overlap is always reviewed.

convergence.distribution_distance::

    {
      "metric": "total_variation" | "l1" | "linf",
      "distribution_fields": ["task_name"],
      "normalization": "failure_mass" | "population",
      "empty_distribution": "zero",
      "minimum_comparisons": 1,
      "no_new_failure_set": "zero_new_failures"
    }

The cumulative failure distribution is evaluated after complete strata only.
minimum_comparisons counts consecutive stable transitions.

convergence.unvisited_strata_policy:
    "continue" | "retain_unresolved"

Annotation protocols are complete existing AnnotationService protocols. Their
assignments and reference_applicability must cover requested responses. Scores,
ranges, and automatic suggestions are never added to visible annotation data.

Persistence
===========

Intermediate immutable manifests are stored as experiment_manifest records.
Strata and ranges are frozen after insertion. Decision revisions are append-only,
content-addressed FailureDecision records; manifests designate authoritative
decision IDs. Historical unresolved decisions are not overwritten.

A completed stage can explicitly report partial coverage after permitted early
termination. Unvisited responses remain unresolved, never implicitly passed.
"""

from __future__ import annotations

import copy
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import numpy as np

from annotation_app import AnnotationService
from config import ExperimentConfig
from schemas import (
    AnnotationPurpose,
    AnnotationTask,
    DecisionOrigin,
    FailureDecision,
    FailureStatus,
    ModelResponse,
    ResponseStatus,
    StageState,
    StageStatus,
    TaskInstance,
    _canonical_json,
    _content_hash,
    _identifier,
)
from storage import ArtifactStore


_MISSING: object = object()
_GROUP_FIELDS: frozenset[str] = frozenset(
    {
        "task_name",
        "stage",
        "model_id",
        "run_index",
        "source_dataset",
        "source_item_id",
        "case_id",
        "instance_id",
    }
)
_ACTIONS: frozenset[str] = frozenset({"pass", "failure", "review", "error"})
_IDENTIFICATION_PURPOSES: frozenset[str] = frozenset(
    {
        AnnotationPurpose.IDENTIFICATION_ANCHOR.value,
        AnnotationPurpose.IDENTIFICATION_BOUNDARY.value,
    }
)


class _AnnotationsPending(RuntimeError):
    """Internal control flow distinguishing pending human work from errors."""

    def __init__(self, task_ids: Sequence[str]) -> None:
        super().__init__("Required independent human references are incomplete.")
        self.task_ids: list[str] = sorted(set(task_ids))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an explicitly configured mapping.")
    return copy.deepcopy(value)


def _required(value: Mapping[str, Any], key: str, name: str) -> Any:
    if key not in value or value[key] is None:
        raise ValueError(f"{name}.{key} is unresolved.")
    return value[key]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite numeric value.")
    result: float = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _identities(values: Sequence[str]) -> list[str]:
    """Deduplicate repeated requests, preserving distinct response identities."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("response_ids must be a sequence of response identifiers.")
    return sorted({_identifier(value) for value in values})


def _pointer(value: Any, pointer: Any) -> Any:
    if not isinstance(pointer, str):
        raise ValueError("Extraction pointers must be explicit strings.")
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("Nonempty JSON Pointers must begin with '/'.")
    current: Any = value
    for component in pointer[1:].split("/"):
        for index, character in enumerate(component):
            if character == "~" and (
                index + 1 == len(component)
                or component[index + 1] not in {"0", "1"}
            ):
                raise ValueError("Malformed JSON Pointer escape.")
        token: str = component.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping) and token in current:
            current = current[token]
        elif (
            isinstance(current, list)
            and token.isdigit()
            and str(int(token)) == token
            and int(token) < len(current)
        ):
            current = current[int(token)]
        else:
            return _MISSING
    return current


class FailureIdentifier:
    """Identify substantive failures using fixed rules and human-calibrated strata."""

    def __init__(
        self,
        config: ExperimentConfig,
        store: ArtifactStore,
        annotations: AnnotationService,
    ) -> None:
        self._config: ExperimentConfig = config
        self._store: ArtifactStore = store
        self._annotations: AnnotationService = annotations
        self._started_at: datetime | None = None
        self._population_manifest: str | None = None
        self._execution_id: str | None = None

    @property
    def _version(self) -> str:
        return _content_hash(
            {
                "method": "rq1_human_calibrated_identification",
                "implementation_contract": "failure-identification-v1",
                "sampling_identity_contract": "root_source_response_v1",
                "config_hash": self._config.config_hash,
                "corpus_version": self._config.corpus_version,
                "failure_identification": self._config.failure_identification,
                "response_criteria": (
                    self._config.evaluation.scoring.response_level_failure_criteria
                ),
                "guideline_manifest": self._config.taxonomy.guideline_manifest,
            }
        )

    def _manifest(
        self, purpose: str, content: dict[str, Any]
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "purpose": purpose,
            "identification_version": self._version,
            "config_hash": self._config.config_hash,
            "corpus_version": self._config.corpus_version,
            **content,
        }
        identity: str = _content_hash(payload)
        record: dict[str, Any] = {"manifest_id": identity, **payload}
        self._store.create_record("experiment_manifest", identity, record)
        return record

    def _join(
        self, response_id: str
    ) -> tuple[ModelResponse, TaskInstance, list[dict[str, Any]]]:
        response = ModelResponse.model_validate(
            self._store.read_record("model_response", response_id)
        )
        instance = TaskInstance.model_validate(
            self._store.read_record("task_instance", response.instance_id),
            context={"require_content_hash": True},
        )
        if response.config_hash != self._config.config_hash:
            raise ValueError("Identification response belongs to another configuration.")
        if (
            instance.transformation_log.get("corpus_version")
            != self._config.corpus_version
        ):
            raise ValueError("Identification instance belongs to another corpus.")
        tasks = self._config.evaluation.tasks.get(instance.stage.value.lower(), {})
        if instance.task_name not in tasks:
            raise ValueError("Task/stage pair is not in the configured task inventory.")
        if not instance.reference.get("version") or not instance.reference.get("answer"):
            raise ValueError("Identification requires a versioned task reference.")
        scores: list[dict[str, Any]] = self._store.list_records(
            "score", {"response_id": response_id}
        )
        for score in scores:
            if (
                score.get("response_id") != response_id
                or score.get("instance_id") != instance.instance_id
                or score.get("task") != instance.task_name
                or score.get("config_hash") != self._config.config_hash
            ):
                raise ValueError("Stored extraction/score provenance is incompatible.")
            reference_hash: Any = score.get("reference_hash")
            if (
                reference_hash is not None
                and reference_hash != _content_hash(instance.reference)
            ):
                raise ValueError("Stored score uses a different task reference.")
        if not scores:
            raise ValueError(
                f"Response {response_id!r} lacks persisted extraction/score artifacts."
            )
        return response, instance, scores

    def _hard_settings(self) -> dict[str, Any]:
        raw: Any = self._config.failure_identification.hard_label_rules
        return {} if raw is None else _mapping(raw, "hard_label_rules")

    def _route(self, task_name: str) -> str:
        settings: dict[str, Any] = self._hard_settings()
        tasks: Any = settings.get("tasks", {})
        if not isinstance(tasks, dict):
            raise ValueError("hard_label_rules.tasks must be a mapping.")
        entry: Any = tasks.get(task_name)
        if entry is None:
            measures: Any = (
                self._config.failure_identification.stratification
                .similarity_measure_by_task
            )
            if isinstance(measures, dict) and task_name in measures:
                return "similarity"
            raise ValueError(f"Task {task_name!r} lacks an explicit identification route.")
        rule: dict[str, Any] = _mapping(entry, f"Route for {task_name}")
        route: Any = _required(rule, "route", task_name)
        if route not in {"hard", "similarity"}:
            raise ValueError("Identification route must be hard or similarity.")
        return route

    def _score(
        self, instance: TaskInstance, scores: list[dict[str, Any]]
    ) -> dict[str, Any]:
        measures: dict[str, Any] = _mapping(
            self._config.failure_identification.stratification.similarity_measure_by_task,
            "similarity_measure_by_task",
        )
        declaration: dict[str, Any] = _mapping(
            measures.get(instance.task_name), "Task similarity declaration"
        )
        if _required(declaration, "source", "Similarity") != "stored_score":
            raise ValueError("Only explicitly persisted similarity scores are supported.")
        pointer: Any = _required(declaration, "pointer", "Similarity")
        version_pointer: Any = _required(declaration, "version_pointer", "Similarity")
        expected_version: str = _identifier(
            _required(declaration, "score_version", "Similarity")
        )
        direction: Any = _required(declaration, "direction", "Similarity")
        if direction not in {"higher_is_better", "lower_is_better"}:
            raise ValueError("Similarity direction is unresolved or unsupported.")
        scale: Any = _required(declaration, "scale", "Similarity")
        if not isinstance(scale, list) or len(scale) != 2:
            raise ValueError("Similarity scale must contain two explicit bounds.")
        lower: float = _finite(scale[0], "Similarity lower bound")
        upper: float = _finite(scale[1], "Similarity upper bound")
        if lower >= upper:
            raise ValueError("Similarity scale bounds must be increasing.")
        found: list[dict[str, Any]] = []
        for artifact in scores:
            value: Any = _pointer(artifact, pointer)
            version: Any = _pointer(artifact, version_pointer)
            if value is _MISSING or version is _MISSING:
                continue
            if version != expected_version:
                continue
            number: float = _finite(value, "Similarity score")
            if not lower <= number <= upper:
                raise ValueError("Similarity score is outside its declared scale.")
            found.append(
                {
                    "score": number,
                    "score_id": _identifier(artifact.get("score_id")),
                    "score_version": expected_version,
                    "declaration_hash": _content_hash(declaration),
                    "direction": direction,
                    "scale": [lower, upper],
                }
            )
        if len(found) != 1:
            raise ValueError(
                "Similarity identification requires exactly one compatible stored "
                "score, not a missing or ambiguously selected artifact."
            )
        return found[0]

    def _decision(
        self,
        response_id: str,
        decision: FailureStatus,
        origin: DecisionOrigin,
        evidence_refs: Sequence[str],
        score: float | None = None,
        stratum_id: str | None = None,
    ) -> FailureDecision:
        record = FailureDecision(
            response_id=response_id,
            identification_version=self._version,
            similarity_score=score,
            stratum_id=stratum_id,
            decision=decision,
            decision_origin=origin,
            evidence_refs=sorted(set(evidence_refs)),
            requires_review=decision == FailureStatus.UNRESOLVED,
        )
        identity: str = _content_hash(record)
        self._store.create_record(
            "failure_decision", identity, record.model_dump(mode="json")
        )
        return record

    def _operand(self, value: Any, sources: dict[str, Any]) -> Any:
        operand: dict[str, Any] = _mapping(value, "Predicate operand")
        if set(operand) == {"literal"}:
            return operand["literal"]
        if set(operand) != {"source", "pointer"}:
            raise ValueError("Operands require literal, or source and pointer.")
        source: Any = operand["source"]
        if source not in sources:
            raise ValueError("Predicate uses an unsupported source.")
        pointer: Any = operand["pointer"]
        if source == "contribution":
            first: str = str(pointer).split("/")[1] if "/" in str(pointer) else ""
            if first not in {"targets", "predictions", "unit_ids"}:
                raise ValueError(
                    "Response rules cannot use aggregate metric values or derived scores."
                )
        result: Any = _pointer(sources[source], pointer)
        if result is _MISSING:
            raise LookupError("A required predicate operand is absent.")
        return result

    def _predicate(self, expression: Any, sources: dict[str, Any]) -> bool:
        node: dict[str, Any] = _mapping(expression, "Response correctness predicate")
        operator: Any = _required(node, "op", "Predicate")
        if operator in {"all", "any"}:
            children: Any = _required(node, "conditions", "Predicate")
            if not isinstance(children, list) or not children:
                raise ValueError("Compound predicates require nonempty conditions.")
            values: list[bool] = [
                self._predicate(child, sources) for child in children
            ]
            return all(values) if operator == "all" else any(values)
        if operator == "not":
            return not self._predicate(
                _required(node, "condition", "Predicate"), sources
            )
        left: Any = self._operand(_required(node, "left", "Predicate"), sources)
        if operator == "between":
            lower: float = _finite(
                self._operand(_required(node, "lower", "Predicate"), sources),
                "Predicate lower bound",
            )
            upper: float = _finite(
                self._operand(_required(node, "upper", "Predicate"), sources),
                "Predicate upper bound",
            )
            inclusive: Any = _required(node, "inclusive", "Predicate")
            if not isinstance(inclusive, bool) or lower > upper:
                raise ValueError("between requires ordered bounds and explicit inclusivity.")
            number: float = _finite(left, "Predicate number")
            return lower <= number <= upper if inclusive else lower < number < upper

        right: Any = self._operand(_required(node, "right", "Predicate"), sources)
        if operator in {"equal", "not_equal"}:
            equal: bool = _canonical_json(left) == _canonical_json(right)
            return equal if operator == "equal" else not equal
        if operator in {"subset", "set_equal"}:
            if not isinstance(left, list) or not isinstance(right, list):
                raise ValueError("Set predicates require arrays.")
            left_set: set[str] = {_canonical_json(item) for item in left}
            right_set: set[str] = {_canonical_json(item) for item in right}
            return left_set <= right_set if operator == "subset" else left_set == right_set
        if operator == "length_equal":
            if not isinstance(left, (list, dict, str)) or not isinstance(
                right, (list, dict, str)
            ):
                raise ValueError("length_equal requires arrays, objects, or strings.")
            return len(left) == len(right)
        if operator == "absolute_difference_le":
            tolerance: float = _finite(
                _required(node, "tolerance", "Predicate"), "Predicate tolerance"
            )
            if tolerance < 0:
                raise ValueError("Predicate tolerance cannot be negative.")
            return abs(
                _finite(left, "Left operand") - _finite(right, "Right operand")
            ) <= tolerance
        raise ValueError(f"Unsupported response-level predicate operator: {operator!r}.")

    def _action(
        self, action: Any, response_id: str, refs: list[str]
    ) -> FailureDecision:
        if action not in _ACTIONS:
            raise ValueError("Response-level identification action is unresolved.")
        if action == "error":
            raise ValueError(
                f"Response {response_id!r} is blocked by its fixed identification policy."
            )
        status: FailureStatus = {
            "pass": FailureStatus.PASS,
            "failure": FailureStatus.FAILURE,
            "review": FailureStatus.UNRESOLVED,
        }[action]
        return self._decision(
            response_id,
            status,
            DecisionOrigin.PENDING_REVIEW
            if status == FailureStatus.UNRESOLVED
            else DecisionOrigin.HARD_LABEL,
            refs,
        )

    def assign_hard_label_decisions(
        self, response_ids: list[str]
    ) -> list[FailureDecision]:
        """Apply explicit response-level rules, never aggregate task metrics."""
        criteria: dict[str, Any] = _mapping(
            self._config.evaluation.scoring.response_level_failure_criteria,
            "response_level_failure_criteria",
        )
        decisions: list[FailureDecision] = []
        for response_id in _identities(response_ids):
            response, instance, scores = self._join(response_id)
            if self._route(instance.task_name) != "hard":
                continue
            rule: dict[str, Any] = _mapping(
                criteria.get(instance.task_name), "Task response-level criterion"
            )
            rule_artifact: dict[str, Any] = self._manifest(
                "identification_hard_rule",
                {"task_name": instance.task_name, "rule": rule},
            )
            refs: list[str] = [
                rule_artifact["manifest_id"],
                *[item["score_id"] for item in scores],
            ]
            availability: str = response.status.value
            if response.status == ResponseStatus.SUCCESS:
                availability = str(response.parsed_answer.get("status", "invalid_schema"))
                if availability == "OK":
                    availability = "available"
            if availability != "available":
                actions: dict[str, Any] = _mapping(
                    _required(rule, "availability", "Hard rule"), "Availability actions"
                )
                decision: FailureDecision = self._action(
                    _required(actions, availability, "Availability actions"),
                    response_id,
                    refs,
                )
            else:
                contributions: list[dict[str, Any]] = [
                    item["contribution"]
                    for item in scores
                    if isinstance(item.get("contribution"), dict)
                ]
                if len(contributions) != 1:
                    raise ValueError(
                        "Hard-label rules require exactly one persisted extraction contribution."
                    )
                sources: dict[str, Any] = {
                    "prediction": response.parsed_answer,
                    "reference": instance.reference,
                    "instance": instance.model_dump(mode="json"),
                    "contribution": contributions[0],
                }
                try:
                    passed: bool = self._predicate(
                        _required(rule, "pass_if", "Hard rule"), sources
                    )
                    decision = self._action(
                        "pass" if passed else "failure", response_id, refs
                    )
                except LookupError:
                    decision = self._action(
                        _required(rule, "on_missing", "Hard rule"), response_id, refs
                    )
            decisions.append(decision)
        return decisions

    def _partition_settings(self) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
        settings = self._config.failure_identification.stratification
        fields: Any = settings.grouping_fields
        if not isinstance(fields, list) or not fields or len(fields) != len(set(fields)):
            raise ValueError("Stratification grouping_fields must be nonempty and unique.")
        if not set(fields).issubset(_GROUP_FIELDS):
            raise ValueError("Unsupported stratification grouping field.")
        ties: dict[str, Any] = _mapping(settings.tied_score_policy, "tied_score_policy")
        if (
            _required(ties, "mode", "Tie policy")
            not in {"split_by_response_id", "keep_together"}
            or _required(ties, "partition", "Tie policy") != "rank_floor"
        ):
            raise ValueError("Unsupported explicit tie/partition policy.")
        order: dict[str, Any] = _mapping(settings.inspection_order, "inspection_order")
        if (
            _required(order, "group_order", "Inspection order") != "lexical"
            or _required(order, "score_order", "Inspection order")
            not in {"ascending", "descending"}
        ):
            raise ValueError("Unsupported explicit stratum inspection order.")
        return fields, ties, order

    def build_strata(self, response_ids: list[str]) -> list[dict[str, Any]]:
        """Partition finite, version-compatible stored similarity scores."""
        fields, ties, order = self._partition_settings()
        width: float = self._config.quantile_width
        if not 0 < width <= 1:
            raise ValueError("Quantile width must lie in (0, 1].")
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        group_values: dict[str, dict[str, Any]] = {}
        from sampling import source_response_mapping
        sampling_sources = source_response_mapping(self._store, _identities(response_ids))
        for response_id in _identities(response_ids):
            response, instance, scores = self._join(response_id)
            if self._route(instance.task_name) != "similarity":
                continue
            score: dict[str, Any] = self._score(instance, scores)
            values: dict[str, Any] = {
                "task_name": instance.task_name,
                "stage": instance.stage.value,
                "model_id": response.model_id,
                "run_index": response.run_index,
                "source_dataset": instance.source_dataset,
                "source_item_id": instance.source_item_id,
                "case_id": instance.case_id,
                "instance_id": instance.instance_id,
            }
            selected: dict[str, Any] = {field: values[field] for field in fields}
            key: str = _canonical_json(selected)
            group_values[key] = selected
            groups[key].append(
                {"response_id": response_id, "sampling_source_response_id": sampling_sources[response_id],
                 "group_fields": values, **score}
            )
        strata: list[dict[str, Any]] = []
        nominal_bins: int = int(math.ceil(1.0 / width))
        for key in sorted(groups):
            rows: list[dict[str, Any]] = sorted(
                groups[key], key=lambda item: (item["score"], item["sampling_source_response_id"])
            )
            compatibility: set[str] = {
                _canonical_json(
                    {
                        "version": item["score_version"],
                        "direction": item["direction"],
                        "scale": item["scale"],
                        "declaration": item["declaration_hash"],
                    }
                )
                for item in rows
            }
            if len(compatibility) != 1:
                raise ValueError(
                    "A stratum group pools incompatible similarity measures or versions."
                )
            partitions: dict[int, list[dict[str, Any]]] = defaultdict(list)
            first_tied_rank: int = 0
            previous_score: float | None = None
            for rank, row in enumerate(rows):
                if previous_score is None or row["score"] != previous_score:
                    first_tied_rank = rank
                effective_rank: int = (
                    first_tied_rank if ties["mode"] == "keep_together" else rank
                )
                index: int = min(
                    nominal_bins - 1,
                    int(math.floor(effective_rank / (len(rows) * width))),
                )
                partitions[index].append(row)
                previous_score = row["score"]
            indices: list[int] = sorted(
                partitions, reverse=order["score_order"] == "descending"
            )
            for index in indices:
                content: dict[str, Any] = {
                    "identification_version": self._version,
                    "config_hash": self._config.config_hash,
                    "group": group_values[key],
                    "group_population_hash": _content_hash(rows),
                    "group_size": len(rows),
                    "nominal_bin_index": index,
                    "nominal_bin_count": nominal_bins,
                    "quantile_width": width,
                    "tie_policy": ties,
                    "sampling_identity_contract": "root_source_response_v1",
                    "inspection_order": order,
                    "members": partitions[index],
                    "effective_size": len(partitions[index]),
                }
                identity: str = _content_hash(content)
                record: dict[str, Any] = {"stratum_id": identity, **content}
                self._store.create_record("stratum", identity, record)
                self._store.freeze_records("stratum", [identity])
                strata.append(record)
        return strata

    def _stratum(self, stratum_id: str) -> dict[str, Any]:
        record: dict[str, Any] = self._store.read_record("stratum", stratum_id)
        if record.get("identification_version") != self._version:
            raise ValueError("Stratum belongs to another identification protocol.")
        return record

    def _estimator(self) -> dict[str, Any]:
        estimator: dict[str, Any] = _mapping(
            self._config.failure_identification.stratification.pass_failure_range_estimator,
            "pass_failure_range_estimator",
        )
        if _required(estimator, "method", "Range estimator") != "anchor_extrema":
            raise ValueError("Unsupported explicit range estimator.")
        for key in ("lower_inclusive", "upper_inclusive"):
            if not isinstance(_required(estimator, key, "Range estimator"), bool):
                raise ValueError("Range endpoint inclusion must be explicit.")
        if _required(estimator, "one_class_policy", "Range estimator") not in {
            "review_all", "observed_range"
        }:
            raise ValueError("Unsupported one-class anchor policy.")
        if _required(estimator, "contradictory_anchor_policy", "Range estimator") not in {
            "retain_overlap", "review_all"
        }:
            raise ValueError("Unsupported contradictory-anchor policy.")
        if _required(estimator, "small_stratum_policy", "Range estimator") not in {
            "all", "error"
        }:
            raise ValueError("Unsupported small-stratum sampling policy.")
        return estimator

    def _annotation_protocol(self, role: str) -> dict[str, Any]:
        hard: dict[str, Any] = self._hard_settings()
        range_raw: Any = (
            self._config.failure_identification.stratification.pass_failure_range_estimator
        )
        estimator: dict[str, Any] = range_raw if isinstance(range_raw, dict) else {}
        candidates: list[Any] = []
        for container in (hard, estimator):
            protocols: Any = container.get("annotation_protocols", {})
            if not isinstance(protocols, dict):
                raise ValueError("annotation_protocols must be a mapping.")
            if role in protocols:
                candidates.append(protocols[role])
        if not candidates or len({_canonical_json(item) for item in candidates}) != 1:
            raise ValueError(f"Missing or conflicting explicit {role} annotation protocol.")
        protocol: dict[str, Any] = _mapping(candidates[0], "Annotation protocol")
        # The configuration cannot contain its own digest. Bind this derived
        # value in the copied runtime protocol, preserving the hashed declaration.
        if protocol.get("config_hash") in {None, "$config_hash"}:
            protocol["config_hash"] = self._config.config_hash
        if (
            protocol.get("config_hash") != self._config.config_hash
            or protocol.get("judgment") != "identification"
            or protocol.get("cohort_id") is not None
        ):
            raise ValueError(
                "Operational identification requires its own non-audit correctness protocol."
            )
        return protocol

    def request_anchors(self, stratum_id: str) -> list[AnnotationTask]:
        """Persist deterministic anchor membership before requesting blinded tasks."""
        stratum: dict[str, Any] = self._stratum(stratum_id)
        estimator: dict[str, Any] = self._estimator()
        count: int = _positive_integer(
            self._config.failure_identification.stratification.anchor_count_per_stratum,
            "anchor_count_per_stratum",
        )
        seed: Any = (
            self._config.failure_identification.stratification.anchor_sampling_seed
        )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("anchor_sampling_seed is unresolved.")
        members: list[str] = sorted(
            item["response_id"] for item in stratum["members"]
        )
        if count > len(members):
            if estimator["small_stratum_policy"] == "error":
                raise ValueError("Requested anchor count exceeds this stratum's size.")
            count = len(members)
        from sampling import sample_response_ids
        sampled, sampling_sources = sample_response_ids(self._store, members, count=count, seed=seed)
        protocol: dict[str, Any] = self._annotation_protocol("anchor")
        self._manifest(
            "identification_anchor_sample",
            {
                "stratum_id": stratum_id,
                "response_ids": sampled,
                "seed": seed,
                "sampling_algorithm": "python_random_sample_sorted_root_source_response_ids",
                "source_response_ids": [sampling_sources[identity] for identity in sampled],
                "source_response_ids_by_response": sampling_sources,
                "requested_count": (
                    self._config.failure_identification.stratification.anchor_count_per_stratum
                ),
                "effective_count": count,
                "protocol_hash": _content_hash(protocol),
            },
        )
        tasks: list[AnnotationTask] = self._annotations.create_tasks(
            sampled, AnnotationPurpose.IDENTIFICATION_ANCHOR.value, protocol
        )
        self._manifest(
            "identification_anchor_tasks",
            {
                "stratum_id": stratum_id,
                "response_ids": sampled,
                "task_ids": sorted(item.annotation_task_id for item in tasks),
                "protocol_hash": _content_hash(protocol),
            },
        )
        return tasks

    def _sample(self, stratum_id: str) -> dict[str, Any]:
        matches: list[dict[str, Any]] = [
            item
            for item in self._store.list_records("experiment_manifest")
            if item.get("purpose") == "identification_anchor_sample"
            and item.get("stratum_id") == stratum_id
            and item.get("identification_version") == self._version
        ]
        if len(matches) != 1:
            raise ValueError("Exactly one designated anchor sample is required.")
        return matches[0]

    def _reference(
        self, response_id: str, purpose: str, tasks: list[AnnotationTask]
    ) -> dict[str, Any]:
        if purpose not in _IDENTIFICATION_PURPOSES:
            raise ValueError("Audit and subtype references cannot calibrate identification.")
        relevant: list[AnnotationTask] = [
            task for task in tasks
            if task.response_id == response_id and task.purpose.value == purpose
        ]
        if not relevant:
            raise ValueError("Human reference lacks designated identification tasks.")
        task_ids: set[str] = {item.annotation_task_id for item in relevant}
        submissions: list[dict[str, Any]] = [
            item for item in self._store.list_records("annotation")
            if item.get("annotation_task_id") in task_ids
            and item.get("purpose") == purpose
            and item.get("audit_only") is False
        ]
        submitted_tasks: set[str] = {
            item["annotation_task_id"] for item in submissions
        }
        if submitted_tasks != task_ids:
            raise _AnnotationsPending(sorted(task_ids - submitted_tasks))
        self._annotations.create_adjudications(sorted(task_ids))
        try:
            reference: dict[str, Any] = self._annotations.get_reference(
                response_id, purpose
            )
        except LookupError:
            pending: list[str] = [
                item["annotation_task_id"]
                for item in self._store.list_records("annotation_task")
                if item.get("response_id") == response_id
                and item.get("status") in {"pending", "in_progress", "draft"}
                and item.get("cohort_id") is None
            ]
            raise _AnnotationsPending(pending or sorted(task_ids)) from None
        if (
            reference.get("purpose") != purpose
            or reference.get("audit_only") is not False
            or reference.get("cohort_id") is not None
            or reference.get("uncertain", True)
            or reference.get("decision") not in {"pass", "failure"}
        ):
            raise ValueError("Human reference is not an eligible final operational judgment.")
        if reference.get("guideline_hash") != relevant[0].guideline_hash:
            raise ValueError("Human reference uses a different guideline.")
        protocol: dict[str, Any] = self._annotation_protocol(
            "anchor" if purpose == AnnotationPurpose.IDENTIFICATION_ANCHOR.value
            else "boundary"
        )
        if reference.get("protocol_hash") != _content_hash(protocol):
            raise ValueError("Human reference belongs to another identification protocol.")
        for annotation_id in reference.get("annotation_ids", []):
            found: list[dict[str, Any]] = []
            for kind in ("annotation", "adjudication"):
                try:
                    found.append(self._store.read_record(kind, annotation_id))
                except KeyError:
                    pass
            if len(found) != 1:
                raise ValueError("Human-reference lineage is missing or ambiguous.")
            item = found[0]
            if (
                item.get("audit_only") is not False
                or item.get("cohort_id") is not None
                or item.get("reference_purpose", item.get("purpose")) != purpose
            ):
                raise ValueError("Audit or unrelated annotations entered calibration lineage.")
        return reference

    def fit_ranges(self, stratum_id: str) -> dict[str, Any]:
        """Fit only the explicitly configured estimator to finalized anchor references."""
        stratum: dict[str, Any] = self._stratum(stratum_id)
        estimator: dict[str, Any] = self._estimator()
        sample: dict[str, Any] = self._sample(stratum_id)
        tasks: list[AnnotationTask] = self.request_anchors(stratum_id)
        scores: dict[str, float] = {
            item["response_id"]: item["score"] for item in stratum["members"]
        }
        anchors: list[dict[str, Any]] = []
        pending: list[str] = []
        for response_id in sample["response_ids"]:
            try:
                reference = self._reference(
                    response_id, AnnotationPurpose.IDENTIFICATION_ANCHOR.value, tasks
                )
                anchors.append(
                    {
                        "response_id": response_id,
                        "score": scores[response_id],
                        "decision": reference["decision"],
                        "reference": reference,
                    }
                )
            except _AnnotationsPending as exc:
                pending.extend(exc.task_ids)
        if pending:
            raise _AnnotationsPending(pending)
        by_class: dict[str, list[float]] = {"pass": [], "failure": []}
        outcomes_at_score: dict[float, set[str]] = defaultdict(set)
        for anchor in anchors:
            by_class[anchor["decision"]].append(anchor["score"])
            outcomes_at_score[anchor["score"]].add(anchor["decision"])
        contradictory: bool = any(
            len(outcomes) > 1 for outcomes in outcomes_at_score.values()
        )
        one_class: bool = any(not values for values in by_class.values())
        usable: bool = not (
            (one_class and estimator["one_class_policy"] == "review_all")
            or (
                contradictory
                and estimator["contradictory_anchor_policy"] == "review_all"
            )
        )
        ranges: dict[str, dict[str, Any] | None] = {}
        for label, values in by_class.items():
            ranges[label] = (
                {
                    "lower": min(values),
                    "upper": max(values),
                    "lower_inclusive": estimator["lower_inclusive"],
                    "upper_inclusive": estimator["upper_inclusive"],
                }
                if usable and values else None
            )
        content: dict[str, Any] = {
            "stratum_id": stratum_id,
            "identification_version": self._version,
            "config_hash": self._config.config_hash,
            "sample_manifest_id": sample["manifest_id"],
            "anchors": anchors,
            "estimator": estimator,
            "ranges": ranges,
            "usable": usable,
            "one_class_observed": one_class,
            "contradictory_equal_score_anchors": contradictory,
            "boundary_labels_used_for_fitting": False,
        }
        identity: str = _content_hash(content)
        record: dict[str, Any] = {"range_id": identity, **content}
        self._store.create_record("range", identity, record)
        self._store.freeze_records("range", [identity])
        return record

    @staticmethod
    def _contains(interval: dict[str, Any] | None, score: float) -> bool:
        if interval is None:
            return False
        lower: bool = (
            score >= interval["lower"]
            if interval["lower_inclusive"] else score > interval["lower"]
        )
        upper: bool = (
            score <= interval["upper"]
            if interval["upper_inclusive"] else score < interval["upper"]
        )
        return lower and upper

    def classify_stratum(self, stratum_id: str) -> list[FailureDecision]:
        """Preserve anchors and review all overlaps and uncovered scores."""
        stratum: dict[str, Any] = self._stratum(stratum_id)
        fitted: dict[str, Any] = self.fit_ranges(stratum_id)
        anchors: dict[str, dict[str, Any]] = {
            item["response_id"]: item for item in fitted["anchors"]
        }
        decisions: list[FailureDecision] = []
        for member in stratum["members"]:
            response_id: str = member["response_id"]
            score: float = member["score"]
            refs: list[str] = [member["score_id"], fitted["range_id"]]
            if response_id in anchors:
                anchor: dict[str, Any] = anchors[response_id]
                refs.extend(anchor["reference"]["annotation_ids"])
                decision: FailureStatus = FailureStatus(anchor["decision"])
                origin: DecisionOrigin = DecisionOrigin.HUMAN_ANCHOR
            else:
                failed: bool = self._contains(fitted["ranges"]["failure"], score)
                passed: bool = self._contains(fitted["ranges"]["pass"], score)
                if failed == passed:
                    decision = FailureStatus.UNRESOLVED
                    origin = DecisionOrigin.PENDING_REVIEW
                else:
                    decision = FailureStatus.FAILURE if failed else FailureStatus.PASS
                    origin = DecisionOrigin.AUTOMATIC_RANGE
            decisions.append(
                self._decision(
                    response_id, decision, origin, refs, score, stratum_id
                )
            )
        return decisions

    def request_boundary_reviews(
        self, decisions: list[FailureDecision]
    ) -> list[AnnotationTask]:
        """Request blinded correctness reviews, never reuse independent audits."""
        selected: list[FailureDecision] = [
            FailureDecision.model_validate(item)
            for item in decisions if item.requires_review
        ]
        if not selected:
            return []
        if (
            self._config.failure_identification.stratification
            .manually_review_boundary_cases is not True
        ):
            raise ValueError("Ambiguous identification requires human boundary review.")
        for decision in selected:
            if decision.identification_version != self._version:
                raise ValueError("Boundary decision belongs to another protocol.")
        response_ids: list[str] = _identities(
            [item.response_id for item in selected]
        )
        protocol: dict[str, Any] = self._annotation_protocol("boundary")
        tasks = self._annotations.create_tasks(
            response_ids, AnnotationPurpose.IDENTIFICATION_BOUNDARY.value, protocol
        )
        self._manifest(
            "identification_boundary_tasks",
            {
                "response_ids": response_ids,
                "source_decision_ids": sorted(_content_hash(item) for item in selected),
                "task_ids": sorted(item.annotation_task_id for item in tasks),
                "protocol_hash": _content_hash(protocol),
            },
        )
        return tasks

    def _resolve_boundaries(
        self, decisions: list[FailureDecision]
    ) -> list[FailureDecision]:
        tasks: list[AnnotationTask] = self.request_boundary_reviews(decisions)
        result: list[FailureDecision] = []
        pending: list[str] = []
        for decision in decisions:
            if not decision.requires_review:
                result.append(decision)
                continue
            try:
                reference = self._reference(
                    decision.response_id,
                    AnnotationPurpose.IDENTIFICATION_BOUNDARY.value,
                    tasks,
                )
                result.append(
                    self._decision(
                        decision.response_id,
                        FailureStatus(reference["decision"]),
                        DecisionOrigin.HUMAN_BOUNDARY,
                        [
                            *decision.evidence_refs,
                            _content_hash(decision),
                            *reference["annotation_ids"],
                        ],
                        decision.similarity_score,
                        decision.stratum_id,
                    )
                )
            except _AnnotationsPending as exc:
                pending.extend(exc.task_ids)
        if pending:
            raise _AnnotationsPending(pending)
        return result

    def _convergence_policy(self) -> dict[str, Any]:
        settings = self._config.failure_identification.stratification.convergence
        policy: dict[str, Any] = _mapping(
            settings.distribution_distance, "convergence.distribution_distance"
        )
        if _required(policy, "metric", "Distribution distance") not in {
            "total_variation", "l1", "linf"
        }:
            raise ValueError("Unsupported failure-distribution distance.")
        fields: Any = _required(policy, "distribution_fields", "Distribution distance")
        if (
            not isinstance(fields, list) or not fields
            or not set(fields).issubset(_GROUP_FIELDS)
            or len(fields) != len(set(fields))
        ):
            raise ValueError("Declare nonempty unique supported distribution fields.")
        if _required(policy, "normalization", "Distribution distance") not in {
            "failure_mass", "population"
        }:
            raise ValueError("Unsupported failure-distribution normalization.")
        if _required(policy, "empty_distribution", "Distribution distance") != "zero":
            raise ValueError("Unsupported explicit empty-distribution policy.")
        _positive_integer(
            _required(policy, "minimum_comparisons", "Distribution distance"),
            "minimum_comparisons",
        )
        if _required(policy, "no_new_failure_set", "Distribution distance") != (
            "zero_new_failures"
        ):
            raise ValueError("Unsupported definition of no new failure set.")
        if settings.stability_tolerance is None:
            raise ValueError("Failure-distribution stability tolerance is unresolved.")
        return policy

    def check_convergence(self, history: list[dict[str, Any]]) -> bool:
        """Evaluate only complete consecutive stratum checkpoints."""
        policy: dict[str, Any] = self._convergence_policy()
        settings = self._config.failure_identification.stratification.convergence
        comparisons: int = policy["minimum_comparisons"]
        converged: bool = False
        distances: list[float] = []
        if len(history) >= comparisons + 1:
            tail: list[dict[str, Any]] = history[-(comparisons + 1):]
            if all(
                item.get("complete") is True
                and not item.get("pending_response_ids")
                for item in tail
            ):
                stable: bool = True
                no_new: bool = True
                for previous, current in zip(tail[:-1], tail[1:], strict=True):
                    left: dict[str, Any] = _mapping(
                        previous.get("failure_counts"), "Historical failure counts"
                    )
                    right: dict[str, Any] = _mapping(
                        current.get("failure_counts"), "Historical failure counts"
                    )
                    keys: list[str] = sorted(set(left) | set(right))
                    vectors: list[np.ndarray] = []
                    for entry, counts in ((previous, left), (current, right)):
                        vector = np.asarray(
                            [_finite(counts.get(key, 0), "Failure count") for key in keys],
                            dtype=float,
                        )
                        if np.any(vector < 0):
                            raise ValueError("Failure-distribution counts cannot be negative.")
                        denominator: float = (
                            float(vector.sum())
                            if policy["normalization"] == "failure_mass"
                            else _finite(entry.get("processed_count"), "Processed count")
                        )
                        vectors.append(
                            vector / denominator
                            if denominator > 0 else np.zeros_like(vector)
                        )
                    difference: np.ndarray = np.abs(vectors[1] - vectors[0])
                    distance: float = (
                        float(difference.max()) if len(difference) else 0.0
                    )
                    if policy["metric"] in {"l1", "total_variation"}:
                        distance = float(difference.sum())
                        if policy["metric"] == "total_variation":
                            distance /= 2.0
                    distances.append(distance)
                    stable = stable and distance <= float(settings.stability_tolerance)
                    new_failures: Any = current.get("new_failure_ids")
                    if not isinstance(new_failures, list):
                        raise ValueError("Convergence history lacks new failure membership.")
                    no_new = no_new and not new_failures
                converged = (
                    (stable or not settings.require_stable_failure_distribution)
                    and (no_new or not settings.require_no_new_failure_set)
                )
        self._manifest(
            "identification_convergence_check",
            {
                "history": history,
                "distance_policy": policy,
                "stability_tolerance": settings.stability_tolerance,
                "distances": distances,
                "converged": converged,
            },
        )
        return converged

    def _history_entry(
        self,
        stratum_id: str,
        decisions: dict[str, FailureDecision],
        previous_failures: set[str],
    ) -> dict[str, Any]:
        policy: dict[str, Any] = self._convergence_policy()
        counts: Counter[str] = Counter()
        failures: set[str] = set()
        for response_id, decision in decisions.items():
            if decision.decision != FailureStatus.FAILURE:
                continue
            failures.add(response_id)
            response, instance, _ = self._join(response_id)
            fields: dict[str, Any] = {
                "task_name": instance.task_name,
                "stage": instance.stage.value,
                "model_id": response.model_id,
                "run_index": response.run_index,
                "source_dataset": instance.source_dataset,
                "source_item_id": instance.source_item_id,
                "case_id": instance.case_id,
                "instance_id": instance.instance_id,
            }
            key: str = _canonical_json(
                {name: fields[name] for name in policy["distribution_fields"]}
            )
            counts[key] += 1
        return {
            "stratum_id": stratum_id,
            "complete": True,
            "processed_count": len(decisions),
            "failure_counts": dict(sorted(counts.items())),
            "failure_ids": sorted(failures),
            "new_failure_ids": sorted(failures - previous_failures),
            "pending_response_ids": [],
        }

    def _workload(
        self, population: list[str], decisions: dict[str, FailureDecision]
    ) -> dict[str, Any]:
        tasks: list[dict[str, Any]] = [
            item for item in self._store.list_records("annotation_task")
            if item.get("response_id") in population
            and item.get("purpose") in _IDENTIFICATION_PURPOSES
            and item.get("cohort_id") is None
        ]
        # Exclude old protocol versions, even if they share the same responses.
        protocol_hashes: set[str] = set()
        for role in ("anchor", "boundary"):
            try:
                protocol_hashes.add(_content_hash(self._annotation_protocol(role)))
            except ValueError:
                # An unused review path need not have an annotation protocol.
                pass
        contexts: dict[str, dict[str, Any]] = {
            item.get("annotation_task_id", ""): item
            for item in self._store.list_records("experiment_manifest")
            if item.get("purpose") == "annotation_task_context"
            and item.get("protocol_hash") in protocol_hashes
        }
        tasks = [
            item for item in tasks if item["annotation_task_id"] in contexts
        ]
        task_ids: set[str] = {item["annotation_task_id"] for item in tasks}
        anchors: set[str] = {
            item["response_id"] for item in tasks
            if item["purpose"] == AnnotationPurpose.IDENTIFICATION_ANCHOR.value
        }
        boundaries: set[str] = {
            item["response_id"] for item in tasks
            if item["purpose"] == AnnotationPurpose.IDENTIFICATION_BOUNDARY.value
        }
        submissions: list[dict[str, Any]] = [
            item for item in self._store.list_records("annotation")
            if item.get("annotation_task_id") in task_ids
            and item.get("audit_only") is False
        ]
        adjudications: list[dict[str, Any]] = [
            item for item in self._store.list_records("adjudication")
            if item.get("response_id") in population
            and item.get("protocol_hash") in protocol_hashes
            and item.get("audit_only") is False
        ]
        origins: Counter[str] = Counter(
            item.decision_origin.value for item in decisions.values()
        )
        reviewed: set[str] = anchors | boundaries
        return {
            "anchor_response_ids": sorted(anchors),
            "boundary_response_ids": sorted(boundaries),
            "unique_reviewed_responses": len(reviewed),
            "annotation_assignments": len(tasks),
            "independent_submissions": len(submissions),
            "adjudications": len(adjudications),
            "decision_origin_counts": dict(origins),
            "review_fraction": len(reviewed) / len(population) if population else None,
            "review_fraction_denominator": len(population),
            "denominator_unit": "requested_model_responses",
            "paper_reported_comparison_upper_bound": (
                self._config.failure_identification
                .reported_manual_inspection_fraction_upper_bound
            ),
            "reported_bound_is_enforced_as_budget": False,
            "independent_audit_work_included": False,
        }

    def _checkpoint(
        self,
        population: list[str],
        decisions: dict[str, FailureDecision],
        pending_tasks: list[str],
        history: list[dict[str, Any]],
        completed: bool,
        output_manifest: str | None = None,
        error: str | None = None,
    ) -> None:
        finalized: list[str] = sorted(
            identity for identity, item in decisions.items()
            if not item.requires_review
        )
        timestamp: datetime = _now()
        status: StageStatus = (
            StageStatus.FAILED if error is not None
            else StageStatus.COMPLETED if completed
            else StageStatus.WAITING_FOR_ANNOTATIONS if pending_tasks
            else StageStatus.RUNNING
        )
        if self._execution_id is None or self._population_manifest is None:
            raise RuntimeError("Identification execution has not been initialized.")
        state = StageState(
            stage="identify",
            execution_id=self._execution_id,
            config_hash=self._config.config_hash,
            status=status,
            completed_work=finalized,
            pending_work=[] if completed else sorted(set(population) - set(finalized)),
            outstanding_annotation_task_ids=sorted(set(pending_tasks)),
            cursor={
                "history": history,
                "authoritative_decision_ids": {
                    key: _content_hash(value) for key, value in sorted(decisions.items())
                },
            },
            input_manifests={"population": self._population_manifest},
            output_manifests=(
                {"identification": output_manifest} if output_manifest else {}
            ),
            started_at=self._started_at or timestamp,
            updated_at=timestamp,
            completed_at=timestamp if completed else None,
            errors=[] if error is None else [{"exception_class": error}],
            metadata={
                "identification_version": self._version,
                "coverage_complete": len(finalized) == len(population),
                "unresolved_response_ids": sorted(set(population) - set(finalized)),
            },
        )
        self._store.set_stage_state("identify", state.model_dump(mode="json"))

    def run(self, response_ids: list[str]) -> dict[str, Any]:
        """Advance identification, returning rather than blocking for annotation."""
        self._config.validate("identify")
        population: list[str] = _identities(response_ids)
        if not population:
            raise ValueError("Failure identification requires a nonempty population.")
        # Resolve the full population before creating work or requesting humans.
        provenance: list[dict[str, Any]] = []
        for response_id in population:
            response, instance, scores = self._join(response_id)
            route: str = self._route(instance.task_name)
            if route == "similarity":
                self._score(instance, scores)
            provenance.append(
                {
                    "response_id": response_id,
                    "response_hash": _content_hash(response),
                    "instance_id": instance.instance_id,
                    "instance_hash": instance.content_hash,
                    "reference_hash": _content_hash(instance.reference),
                    "score_ids": sorted(item["score_id"] for item in scores),
                    "route": route,
                }
            )
        population_record: dict[str, Any] = self._manifest(
            "identification_population", {"members": provenance}
        )
        self._population_manifest = population_record["manifest_id"]
        self._execution_id = _content_hash(
            {"version": self._version, "population": self._population_manifest}
        )
        self._started_at = _now()

        previous: list[StageState] = [
            StageState.model_validate(item)
            for item in self._store.list_records("stage_state")
            if item.get("stage") == "identify"
            and item.get("execution_id") == self._execution_id
        ]
        if previous:
            starts: list[datetime] = [
                item.started_at for item in previous if item.started_at is not None
            ]
            if starts:
                self._started_at = min(starts)
            completed_states: list[StageState] = [
                item for item in previous if item.status == StageStatus.COMPLETED
            ]
            if completed_states:
                manifest_id: str = completed_states[-1].output_manifests["identification"]
                saved = self._store.read_record("experiment_manifest", manifest_id)
                return saved["result"]

        decisions: dict[str, FailureDecision] = {}
        history: list[dict[str, Any]] = []
        pending_tasks: list[str] = []
        converged: bool = False
        strata: list[dict[str, Any]] = []
        try:
            hard: list[FailureDecision] = self.assign_hard_label_decisions(population)
            decisions.update({item.response_id: item for item in hard})
            try:
                hard = self._resolve_boundaries(hard)
                decisions.update({item.response_id: item for item in hard})
            except _AnnotationsPending as exc:
                pending_tasks = exc.task_ids
                raise
            similarity_ids: list[str] = [
                item["response_id"] for item in provenance if item["route"] == "similarity"
            ]
            if similarity_ids:
                self._convergence_policy()
                policy: Any = (
                    self._config.failure_identification.stratification.convergence
                    .unvisited_strata_policy
                )
                if policy not in {"continue", "retain_unresolved"}:
                    raise ValueError("Unsupported explicit unvisited-strata policy.")
                strata = self.build_strata(similarity_ids)
                for stratum in strata:
                    self.request_anchors(stratum["stratum_id"])
                    current: list[FailureDecision] = self.classify_stratum(
                        stratum["stratum_id"]
                    )
                    decisions.update({item.response_id: item for item in current})
                    current = self._resolve_boundaries(current)
                    decisions.update({item.response_id: item for item in current})
                    old_failures: set[str] = (
                        set(history[-1]["failure_ids"]) if history else set()
                    )
                    history.append(
                        self._history_entry(
                            stratum["stratum_id"], decisions, old_failures
                        )
                    )
                    converged = self.check_convergence(history)
                    self._checkpoint(
                        population, decisions, [], history, completed=False
                    )
                    if converged and policy == "retain_unresolved":
                        break

            processed: set[str] = set(decisions)
            for stratum in strata:
                for member in stratum["members"]:
                    response_id = member["response_id"]
                    if response_id not in processed:
                        decisions[response_id] = self._decision(
                            response_id,
                            FailureStatus.UNRESOLVED,
                            DecisionOrigin.NOT_PROCESSED,
                            [member["score_id"], self._population_manifest],
                            member["score"],
                            stratum["stratum_id"],
                        )

            finalized: dict[str, FailureDecision] = {
                key: value for key, value in decisions.items()
                if not value.requires_review
            }
            decision_ids: list[str] = sorted(
                _content_hash(item) for item in finalized.values()
            )
            freeze_hash: str | None = (
                self._store.freeze_records("failure_decision", decision_ids)
                if decision_ids else None
            )
            unresolved: list[str] = sorted(set(population) - set(finalized))
            result: dict[str, Any] = {
                "status": "completed",
                "identification_version": self._version,
                "population_manifest": self._population_manifest,
                "decision_ids": decision_ids,
                "decision_ids_by_response": {
                    key: _content_hash(value) for key, value in sorted(finalized.items())
                },
                "failure_response_ids": sorted(
                    key for key, value in finalized.items()
                    if value.decision == FailureStatus.FAILURE
                ),
                "pending_annotation_task_ids": [],
                "unresolved_response_ids": unresolved,
                "unprocessed_response_ids": sorted(
                    key for key, value in decisions.items()
                    if value.decision_origin == DecisionOrigin.NOT_PROCESSED
                ),
                "unresolved_decision_ids": sorted(
                    _content_hash(value) for value in decisions.values()
                    if value.requires_review
                ),
                "coverage": {
                    "population": len(population),
                    "finalized": len(finalized),
                    "fraction": len(finalized) / len(population),
                    "complete": not unresolved,
                },
                "converged": converged,
                "history": history,
                "decision_freeze_manifest": freeze_hash,
                "workload": self._workload(population, decisions),
            }
            manifest: dict[str, Any] = self._manifest(
                "identification_final",
                {
                    "population_manifest": self._population_manifest,
                    "result": result,
                },
            )
            self._checkpoint(
                population, decisions, [], history,
                completed=True, output_manifest=manifest["manifest_id"],
            )
            return result
        except _AnnotationsPending as exc:
            pending_tasks = sorted(set([*pending_tasks, *exc.task_ids]))
            if not pending_tasks:
                raise RuntimeError("Waiting state lacks pending annotation identities.") from exc
            self._checkpoint(
                population, decisions, pending_tasks, history, completed=False
            )
            finalized_ids: dict[str, str] = {
                key: _content_hash(value)
                for key, value in sorted(decisions.items())
                if not value.requires_review
            }
            return {
                "status": "waiting_for_annotations",
                "identification_version": self._version,
                "population_manifest": self._population_manifest,
                "decision_ids": sorted(finalized_ids.values()),
                "decision_ids_by_response": finalized_ids,
                "failure_response_ids": sorted(
                    key for key, value in decisions.items()
                    if value.decision == FailureStatus.FAILURE
                ),
                "pending_annotation_task_ids": pending_tasks,
                "unresolved_response_ids": sorted(set(population) - set(finalized_ids)),
                "unprocessed_response_ids": sorted(set(population) - set(decisions)),
                "coverage": {
                    "population": len(population),
                    "finalized": len(finalized_ids),
                    "fraction": len(finalized_ids) / len(population),
                    "complete": False,
                },
                "converged": False,
                "history": history,
                "workload": self._workload(population, decisions),
            }
        except Exception as exc:
            self._checkpoint(
                population, decisions, [], history,
                completed=False, error=type(exc).__name__,
            )
            raise
