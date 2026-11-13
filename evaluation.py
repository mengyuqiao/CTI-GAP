"""Reproducible task metrics, independent audits, and taxonomy stability.

This module performs measurement only. It does not infer missing scientific
settings, repair predictions, adjudicate annotations, or modify audit references.

Executable policy contracts
===========================

Several scoring policies are configurable. This module consumes explicit
mappings in the existing configuration fields; it does not add configuration
sections.

evaluation.scoring.task_field_extraction_rules maps task names to declarations:

    {
        "prediction": "/answer/decisions",
        "reference": "/answer/decisions",
        "representation": "aligned",
        "unit_ids": "/candidates"
    }

Pointers follow RFC 6901. Prediction pointers address response.parsed_answer;
reference pointers address instance.reference; unit_ids addresses the instance.

Supported representations:
  * aligned: aligned arrays, or individual scalar decisions.
  * keyed: prediction/reference objects keyed by unit identifier. Keys must match.
  * text: strings, or reference arrays of strings.
  * ranking: prediction array of candidate IDs and reference object mapping
    candidate IDs to graded relevance.
  * ioc_type_value: IOC-normalization-only arrays of {raw, type, value} objects.
    Exact (raw, type, value) membership is aligned over the prediction/reference
    union: correct pairs are TP, substitutions are FP+FN, omissions are FN, and
    additional pairs are FP. Metadata such as tags is not a correctness target.
    Duplicate raw identifiers are rejected rather than silently deduplicated.
    This representation requires binary F1 averaging with positive class 1;
    pooling then preserves these pair counts across instances.
    This pair-F1 counting policy is opt-in.

Binary metrics require explicit 0/1 labels. Probability F1 extraction additionally
declares prediction_kind="probability"; configured thresholds are then required.

Accuracy policy, globally or per task:
    {"mode": "unit_exact" | "response_exact"}

Invalid-output and abstention policies, globally or per task, are:
    "error", "exclude", or {"action": "replace", "prediction": <explicit value>}

Policies may be keyed by availability category, including transport_failure,
refusal, invalid_schema, missing_prediction, NEED_MORE_CONTEXT, and UNSUPPORTED.
Replacement is explicit scoring, never a repair of the stored response.

F1 positive_class_definitions is 0/1, or a task-to-0/1 mapping.
F1 averaging supports binary, micro, macro, and weighted.
Probability thresholds are numbers or task-to-number mappings.

BLEU:
  aggregation_level is corpus or sentence_mean;
  smoothing is {"method": <SacreBLEU method>, "value": <number or null>,
                "effective_order": <boolean>};
  structured_output_serialization is "text", "canonical_json", or a task mapping.
An empty string is a valid hypothesis; an absent field is not.

NDCG:
  gain_function is linear or exponential;
  tie_policy is "ordered";
  reference objects must match the complete predicted candidate set.
All-zero ideal gain is reported as undefined, never perfect performance.

Audit manifest conventions
==========================

A locked AuditCohort.sampling_protocol binds frozen_prediction_ids:
    {response_id: prediction_record_id}
and, for label audits, taxonomy_id and guideline_hash.

Identification prediction IDs address failure_decision records. Label prediction
IDs address label_assignment, assignment, consensus, or final_assignment records.
The latter must still explicitly identify a pre-correction prediction origin.

Independent submissions are annotation records with cohort_id, response_id,
annotator_id, purpose, guideline_hash, status="submitted", and decision or labels.
Adjudications are separate adjudication records with the same binding fields.
An adjudication with purpose="audit_adjudication" additionally supplies
reference_purpose. Expanded cohorts may explicitly name reference_cohort_ids to
reuse the original independently annotated cohort without copying its labels.

Agreement annotations are never replaced by adjudications.

Stability cohort conventions
============================

sampling_protocol supplies taxonomy_assignment_ids:
    {taxonomy_id: {response_id: assignment_record_id}}
and discovery_runs:
    {taxonomy_id: {"condition": "...", "run_index": ..., ...}}

comparison_aggregation is "all_pairs" or
    {"mode": "reference", "reference_taxonomy_id": "..."}.
Supported unmatched_category_policy values are "zero" and "exclude".
empty_set_policy is "one", "zero", or "undefined".
prevalence_normalization is "label_mass".
All such choices must be explicitly configured before measurement.

Undefined macro metrics propagate an undefined component. They are never silently
computed over a reduced subtype population. Results retain full precision;
presentation rounding is intentionally left to downstream consumers.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import tempfile
import warnings
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import krippendorff
import numpy as np
from sacrebleu.metrics import BLEU
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import f1_score, roc_auc_score

from config import ExperimentConfig
from schemas import (
    AssignmentOrigin,
    AuditCohort,
    FailureDecision,
    FailureStatus,
    LabelAssignment,
    ModelResponse,
    ResponseStatus,
    TaskInstance,
    TaxonomyVersion,
    _canonical_json,
    _content_hash,
    _json_primitive,
)
from storage import ArtifactStore


_MISSING: object = object()
_ASSIGNMENT_KINDS: tuple[str, ...] = (
    "label_assignment",
    "assignment",
    "consensus",
    "final_assignment",
)
_TAXONOMY_KINDS: tuple[str, ...] = ("taxonomy", "taxonomy_version")
_PRECORRECTION_ORIGINS: frozenset[AssignmentOrigin] = frozenset(
    {
        AssignmentOrigin.AGENT,
        AssignmentOrigin.CONSENSUS,
        AssignmentOrigin.DISCOVERY,
    }
)
_AUDIT_PURPOSES: dict[str, frozenset[str]] = {
    "failure_identification": frozenset(
        {"failure_identification", "independent_identification_audit"}
    ),
    "failure_categorization": frozenset(
        {
            "failure_categorization",
            "independent_label_audit",
            "expanded_label_audit",
        }
    ),
    "human_agreement": frozenset({"human_agreement"}),
    "taxonomy_stability": frozenset({"taxonomy_stability", "stability"}),
}


def _required(value: Any, path: str) -> Any:
    """Reject unresolved values without treating zero or False as unresolved."""
    if value is None:
        raise ValueError(f"{path} is unresolved.")
    return value


def _pointer(value: Any, pointer: Any) -> Any:
    """Resolve an RFC 6901 pointer; absence is distinct from JSON null."""
    if not isinstance(pointer, str):
        raise ValueError("Extraction pointers must be explicit strings.")
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("Nonempty extraction pointers must begin with '/'.")
    current: Any = value
    for part in pointer[1:].split("/"):
        if any(
            part[index] == "~"
            and (index + 1 == len(part) or part[index + 1] not in "01")
            for index in range(len(part))
        ):
            raise ValueError("Malformed JSON Pointer escape.")
        token: str = part.replace("~1", "/").replace("~0", "~")
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


def _task_policy(value: Any, task: str, required: bool = True) -> Any:
    """Select a task override, otherwise retain an explicit global policy."""
    if isinstance(value, Mapping) and task in value:
        return value[task]
    if value is None and required:
        raise ValueError(f"Scoring policy for {task!r} is unresolved.")
    return value


def _finite(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} must be a finite number.")
    result: float = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite.")
    return result


def _binary(value: Any, description: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, int) or value not in (0, 1):
        raise ValueError(f"{description} requires an explicitly constructed binary label.")
    return value


def _probability(value: Any, description: str) -> float:
    result: float = _finite(value, description)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{description} must lie in [0, 1].")
    return result


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
    }


def _macro(values: Sequence[float | None]) -> float | None:
    """Propagate undefined components rather than changing the population."""
    if not values or any(value is None for value in values):
        return None
    return float(np.mean([float(value) for value in values if value is not None]))


def _unique_ids(values: Sequence[str], description: str) -> list[str]:
    result: list[str] = list(values)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{description} must contain nonempty string IDs.")
    if len(result) != len(set(result)):
        raise ValueError(f"{description} contains duplicate IDs.")
    return result


def _sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _count_binary(
    targets: Sequence[int], predictions: Sequence[int], positive: int
) -> dict[str, int]:
    counts: dict[str, int] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for target, prediction in zip(targets, predictions, strict=True):
        key: str
        if prediction == positive:
            key = "tp" if target == positive else "fp"
        else:
            key = "fn" if target == positive else "tn"
        counts[key] += 1
    return counts


class Evaluator:
    """Compute traceable measurements from stored, independently frozen inputs."""

    def __init__(self, config: ExperimentConfig, store: ArtifactStore) -> None:
        self._config: ExperimentConfig = config
        self._store: ArtifactStore = store

    def _persist(self, purpose: str, result: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "purpose": purpose,
            "config_hash": self._config.config_hash,
            "corpus_version": self._config.corpus_version,
            **result,
        }
        payload = json.loads(_canonical_json(payload))
        record_id: str = _content_hash(payload)
        self._store.create_record(
            "report", record_id, {"report_id": record_id, **payload}
        )
        return {"artifact_id": record_id, **payload}

    def _task(self, instance: TaskInstance) -> Any:
        tasks: Mapping[str, Any] = self._config.evaluation.tasks.get(
            instance.stage.value.lower(), {}
        )
        if instance.task_name not in tasks:
            raise ValueError("Task/stage pair is not configured.")
        return tasks[instance.task_name]

    def _rule(self, task: str) -> dict[str, Any]:
        rules: Any = _required(
            self._config.evaluation.scoring.task_field_extraction_rules,
            "evaluation.scoring.task_field_extraction_rules",
        )
        if not isinstance(rules, Mapping) or task not in rules:
            raise ValueError(f"No explicit extraction rule exists for {task!r}.")
        rule: Any = rules[task]
        if not isinstance(rule, dict):
            raise ValueError("Task extraction rules must be mappings.")
        for field in ("prediction", "reference", "representation"):
            if field not in rule:
                raise ValueError(f"Extraction rule lacks {field!r}.")
        return rule

    def _availability(self, response: ModelResponse) -> str:
        if response.status != ResponseStatus.SUCCESS:
            return response.status.value
        semantic_status: Any = response.parsed_answer.get("status")
        if semantic_status in {"NEED_MORE_CONTEXT", "UNSUPPORTED"}:
            return str(semantic_status)
        if semantic_status != "OK":
            raise ValueError("Successful task response lacks an allowed semantic status.")
        return "available"

    def _unavailable(
        self, task: str, category: str
    ) -> tuple[bool, Any]:
        scoring = self._config.evaluation.scoring
        abstained: bool = category in {"NEED_MORE_CONTEXT", "UNSUPPORTED"}
        policies: Any = (
            scoring.abstention_policy if abstained else scoring.invalid_output_policy
        )
        policy: Any = _task_policy(policies, task)
        if isinstance(policy, Mapping) and category in policy:
            policy = policy[category]
        action: Any = policy.get("action") if isinstance(policy, Mapping) else policy
        if action == "exclude":
            return True, _MISSING
        if action == "error":
            raise ValueError(
                f"Response availability {category!r} blocks scoring under its policy."
            )
        if action == "replace" and isinstance(policy, Mapping):
            if "prediction" not in policy:
                raise ValueError("Replacement scoring requires an explicit prediction.")
            return False, policy["prediction"]
        raise ValueError(
            f"Unsupported or unresolved availability policy for {task}/{category}."
        )

    def _reference_version(self, instance: TaskInstance) -> str:
        version: Any = instance.reference.get("version")
        if not isinstance(version, str) or not version:
            raise ValueError("Task reference requires an explicit version.")
        return version

    def _aligned(
        self,
        instance: TaskInstance,
        rule: dict[str, Any],
        prediction: Any,
        reference: Any,
    ) -> tuple[list[Any], list[Any], list[str]]:
        representation: str = rule["representation"]
        if representation == "ioc_type_value":
            if instance.task_name != "ioc_normalization":
                raise ValueError("IOC type/value extraction is specific to IOC normalization.")
            f1 = self._config.evaluation.scoring.f1
            if f1.averaging != "binary" or _task_policy(
                f1.positive_class_definitions, instance.task_name
            ) != 1:
                raise ValueError("IOC type/value pair F1 requires binary averaging and positive class 1.")
            def pairs(value: Any, description: str) -> set[str]:
                if not isinstance(value, list):
                    raise ValueError(f"{description} IOC normalization must be an array.")
                result: set[str] = set()
                raw_ids: set[str] = set()
                for item in value:
                    if not isinstance(item, dict) or any(
                        not isinstance(item.get(field), str) or not item[field]
                        for field in ("raw", "type", "value")
                    ):
                        raise ValueError(f"{description} IOC entries require raw, type, and value strings.")
                    if item["raw"] in raw_ids:
                        raise ValueError(f"{description} IOC entries have duplicate raw identifiers.")
                    raw_ids.add(item["raw"])
                    result.add(_canonical_json([item["raw"], item["type"], item["value"]]))
                return result
            predicted_pairs = pairs(prediction, "Predicted")
            reference_pairs = pairs(reference, "Reference")
            if not reference_pairs:
                raise ValueError("IOC type/value extraction requires nonempty gold forms.")
            identifiers = sorted(predicted_pairs | reference_pairs)
            return (
                [int(key in predicted_pairs) for key in identifiers],
                [int(key in reference_pairs) for key in identifiers],
                identifiers,
            )
        if representation == "keyed":
            if not isinstance(prediction, dict) or not isinstance(reference, dict):
                raise ValueError("Keyed extraction requires prediction/reference objects.")
            if set(prediction) != set(reference):
                raise ValueError(
                    "Missing or extra keyed predictions require a separately resolved policy."
                )
            identifiers: list[str] = sorted(reference)
            return (
                [prediction[key] for key in identifiers],
                [reference[key] for key in identifiers],
                identifiers,
            )
        if representation != "aligned":
            raise ValueError("This metric requires aligned or keyed extraction.")
        predictions: list[Any] = _sequence(prediction)
        targets: list[Any] = _sequence(reference)
        if len(predictions) != len(targets) or not targets:
            raise ValueError("Prediction/reference scoring units are empty or misaligned.")
        if "unit_ids" in rule:
            raw_ids: Any = _pointer(instance.model_dump(mode="json"), rule["unit_ids"])
            if raw_ids is _MISSING or not isinstance(raw_ids, list):
                raise ValueError("Configured unit_ids pointer does not select a list.")
            identifiers = [_canonical_json(item) for item in raw_ids]
            if len(identifiers) != len(targets):
                raise ValueError("Unit identifiers do not match prediction/reference lengths.")
            _unique_ids(identifiers, "Scoring unit IDs")
        else:
            identifiers = [str(index) for index in range(len(targets))]
        return predictions, targets, identifiers

    def _serialize_text(self, value: Any, task: str) -> str:
        serialization: Any = _task_policy(
            self._config.evaluation.scoring.bleu.structured_output_serialization,
            task,
        )
        if serialization == "text":
            if not isinstance(value, str):
                raise ValueError("Text serialization requires an actual string.")
            return value
        if serialization == "canonical_json":
            return _canonical_json(value)
        raise ValueError("Unsupported or unresolved BLEU serialization policy.")

    def _bleu(self) -> BLEU:
        settings = self._config.evaluation.scoring.bleu
        tokenizer: Any = _required(settings.tokenizer, "BLEU tokenizer")
        order: Any = _required(settings.maximum_ngram_order, "BLEU n-gram order")
        lowercase: Any = _required(settings.lowercase, "BLEU lowercase")
        smoothing: Any = _required(settings.smoothing, "BLEU smoothing")
        if not isinstance(smoothing, Mapping) or not {
            "method", "value", "effective_order"
        }.issubset(smoothing):
            raise ValueError(
                "BLEU smoothing must explicitly declare method/value/effective_order."
            )
        if not isinstance(smoothing["effective_order"], bool):
            raise ValueError("BLEU effective_order must be explicitly boolean.")
        if smoothing["value"] is not None:
            _finite(smoothing["value"], "BLEU smoothing value")
        return BLEU(
            lowercase=lowercase,
            tokenize=tokenizer,
            smooth_method=smoothing["method"],
            smooth_value=smoothing["value"],
            effective_order=smoothing["effective_order"],
            max_ngram_order=order,
        )

    def score_response(
        self, instance: TaskInstance, response: ModelResponse
    ) -> dict[str, Any]:
        """Extract validated metric contributions, not substantive failure labels."""
        if response.instance_id != instance.instance_id:
            raise ValueError("Response and task-instance identities differ.")
        if response.config_hash != self._config.config_hash:
            raise ValueError("Response belongs to a different experimental configuration.")
        corpus: Any = instance.transformation_log.get("corpus_version")
        if corpus != self._config.corpus_version:
            raise ValueError("Instance belongs to a different corpus.")
        definition: Any = self._task(instance)
        rule: dict[str, Any] = self._rule(instance.task_name)
        reference_version: str = self._reference_version(instance)
        scoring_version: str = _content_hash(
            {
                "rules": rule,
                "scoring": self._config.evaluation.scoring,
                "task": definition,
            }
        )
        reference: Any = _pointer(instance.reference, rule["reference"])
        if reference is _MISSING or reference is None:
            raise ValueError("Required reference metric input is absent.")

        category: str = self._availability(response)
        excluded: bool = False
        prediction: Any = _MISSING
        if category == "available":
            prediction = _pointer(response.parsed_answer, rule["prediction"])
            if prediction is _MISSING or prediction is None:
                category = "missing_prediction"
        if category != "available":
            excluded, prediction = self._unavailable(instance.task_name, category)

        context: Any = response.usage.get("evaluation_context", {})
        if not isinstance(context, dict):
            raise ValueError("evaluation_context must be a provenance object.")
        condition: Any = context.get("condition")
        if condition is None:
            if self._config.operational.selected_defense is not None:
                raise ValueError("Defense responses require explicit condition provenance.")
            condition = "original"
        if not isinstance(condition, str) or not condition:
            raise ValueError("Experimental condition must be a nonempty string.")

        result: dict[str, Any] = {
            "response_id": response.response_id,
            "instance_id": instance.instance_id,
            "case_id": instance.case_id,
            "task": instance.task_name,
            "stage": instance.stage.value,
            "model_id": response.model_id,
            "run_index": response.run_index,
            "condition": condition,
            "evaluation_context": context,
            "metric": definition.metric,
            "reference_version": reference_version,
            "reference_hash": _content_hash(instance.reference),
            "scoring_version": scoring_version,
            "availability": category,
            "excluded": excluded,
            "replacement_applied": category != "available" and not excluded,
            "scoring_units": 0,
            "contribution": {},
            "failure_decision": None,
        }
        if not excluded:
            metric: str = definition.metric
            contribution: dict[str, Any]
            if metric in {"f1", "accuracy", "roc_auc"}:
                predictions, targets, identifiers = self._aligned(
                    instance, rule, prediction, reference
                )
                if metric in {"f1", "roc_auc"}:
                    binary_targets: list[int] = [
                        _binary(value, "Reference target") for value in targets
                    ]
                    if metric == "roc_auc":
                        _required(
                            self._config.evaluation.scoring.roc_auc.target_definition_manifest,
                            "ROC-AUC target definition manifest",
                        )
                        _required(
                            self._config.evaluation.scoring.roc_auc.missing_probability_policy,
                            "ROC-AUC missing probability policy",
                        )
                        if instance.task_name in {
                            "exploit_likelihood", "campaign_escalation"
                        }:
                            horizon: Any = _required(
                                definition.horizon_days, "Forecast horizon"
                            )
                            construction: Any = _required(
                                definition.binary_target_construction,
                                "Forecast binary target construction",
                            )
                            if instance.reference.get("horizon_days") != horizon:
                                raise ValueError("Reference forecast horizon mismatch.")
                            if _canonical_json(
                                instance.reference.get("target_construction")
                            ) != _canonical_json(construction):
                                raise ValueError("Forecast target construction mismatch.")
                        binary_predictions: list[Any] = [
                            _probability(value, "Task probability")
                            for value in predictions
                        ]
                        if any(
                            component.lower() == "confidence"
                            for component in rule["prediction"].split("/")
                        ):
                            raise ValueError(
                                "Generic confidence cannot substitute for task probability."
                            )
                    else:
                        prediction_kind: Any = rule.get("prediction_kind", "binary")
                        if prediction_kind == "probability":
                            threshold: Any = _task_policy(
                                self._config.evaluation.scoring.f1.probability_thresholds,
                                instance.task_name,
                            )
                            threshold_value: float = _probability(
                                threshold, "F1 probability threshold"
                            )
                            binary_predictions = [
                                int(_probability(value, "F1 probability") >= threshold_value)
                                for value in predictions
                            ]
                        elif prediction_kind == "binary":
                            binary_predictions = [
                                _binary(value, "Predicted decision")
                                for value in predictions
                            ]
                        else:
                            raise ValueError("Unsupported F1 prediction_kind.")
                    contribution = {
                        "targets": binary_targets,
                        "predictions": binary_predictions,
                        "unit_ids": identifiers,
                    }
                    if metric == "f1":
                        positive: int = _binary(
                            _task_policy(
                                self._config.evaluation.scoring.f1.positive_class_definitions,
                                instance.task_name,
                            ),
                            "F1 positive class",
                        )
                        contribution["counts"] = _count_binary(
                            binary_targets, binary_predictions, positive
                        )
                else:
                    policy: Any = _task_policy(
                        self._config.evaluation.scoring.accuracy.exact_match_and_partial_credit_rules,
                        instance.task_name,
                    )
                    mode: Any = policy.get("mode") if isinstance(policy, Mapping) else policy
                    correct: list[int] = [
                        int(_canonical_json(left) == _canonical_json(right))
                        for left, right in zip(predictions, targets, strict=True)
                    ]
                    if mode == "response_exact":
                        correct = [int(all(correct))]
                        identifiers = ["response"]
                    elif mode != "unit_exact":
                        raise ValueError("Unsupported accuracy credit policy.")
                    contribution = {
                        "correct": sum(correct),
                        "total": len(correct),
                        "unit_correct": correct,
                        "unit_ids": identifiers,
                    }
                result["scoring_units"] = len(identifiers)
            elif metric == "bleu":
                hypothesis: str = self._serialize_text(prediction, instance.task_name)
                multiple: Any = rule.get("multiple_references", False)
                if not isinstance(multiple, bool):
                    raise ValueError("multiple_references must be explicitly boolean.")
                raw_references: list[Any] = (
                    reference if multiple and isinstance(reference, list) else [reference]
                )
                if not raw_references:
                    raise ValueError("BLEU requires at least one reference.")
                references: list[str] = [
                    self._serialize_text(item, instance.task_name)
                    for item in raw_references
                ]
                if any(not text for text in references):
                    raise ValueError("BLEU reference text must not be empty.")
                contribution = {"hypothesis": hypothesis, "references": references}
                result["scoring_units"] = 1
            elif metric == "ndcg":
                if rule["representation"] != "ranking":
                    raise ValueError("NDCG requires ranking representation.")
                if not isinstance(prediction, list) or not isinstance(reference, dict):
                    raise ValueError("NDCG requires ordered IDs and graded relevance.")
                identifiers = _unique_ids(prediction, "Predicted ranking")
                if set(identifiers) != set(reference):
                    raise ValueError(
                        "Rankings must contain exactly the declared reference candidates."
                    )
                relevance: dict[str, float] = {
                    key: _finite(value, "Graded relevance")
                    for key, value in reference.items()
                }
                if any(value < 0 for value in relevance.values()):
                    raise ValueError("NDCG relevance must be nonnegative.")
                _required(
                    self._config.evaluation.scoring.ndcg.relevance_manifest,
                    "NDCG relevance manifest",
                )
                contribution = {"ranking": identifiers, "relevance": relevance}
                result["scoring_units"] = 1
            else:
                raise ValueError(f"Unsupported metric: {metric}.")
            result["contribution"] = contribution

        score_id: str = _content_hash(result)
        self._store.create_record(
            "score",
            score_id,
            {
                "score_id": score_id,
                "config_hash": self._config.config_hash,
                **result,
            },
        )
        return {"score_id": score_id, **result}

    def _ndcg(self, contribution: dict[str, Any]) -> float | None:
        settings = self._config.evaluation.scoring.ndcg
        cutoff: int = _required(settings.cutoff_k, "NDCG cutoff")
        gain: Any = _required(settings.gain_function, "NDCG gain function")
        if _required(settings.tie_policy, "NDCG tie policy") != "ordered":
            raise ValueError("This evaluator requires an explicitly ordered ranking.")
        if gain not in {"linear", "exponential"}:
            raise ValueError("Unsupported NDCG gain function.")
        relevance: dict[str, float] = contribution["relevance"]
        ordered: list[float] = [
            relevance[key] for key in contribution["ranking"]
        ]
        ideal: list[float] = sorted(relevance.values(), reverse=True)

        def dcg(values: list[float]) -> float:
            gains: list[float] = [
                value if gain == "linear" else math.expm1(value * math.log(2))
                for value in values[:cutoff]
            ]
            if any(not math.isfinite(value) for value in gains):
                raise ValueError("NDCG gain overflow.")
            return float(
                sum(value / math.log2(index + 2) for index, value in enumerate(gains))
            )

        ideal_dcg: float = dcg(ideal)
        return None if ideal_dcg == 0 else dcg(ordered) / ideal_dcg

    def _metric(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        included: list[dict[str, Any]] = [
            record for record in records if not record["excluded"]
        ]
        metric: str = records[0]["metric"]
        task: str = records[0]["task"]
        result: dict[str, Any] = {
            "metric": metric,
            "native_scale": [0, 100] if metric == "bleu" else [0, 1],
            "value": None,
            "undefined_reason": None,
            "response_count": len(records),
            "included_responses": len(included),
            "excluded_responses": len(records) - len(included),
            "availability_counts": dict(Counter(record["availability"] for record in records)),
            "scoring_units": sum(record["scoring_units"] for record in included),
        }
        if not included:
            result["undefined_reason"] = "No included responses."
            return result
        contributions: list[dict[str, Any]] = [
            item["contribution"] for item in included
        ]
        if metric in {"f1", "roc_auc"}:
            targets: list[int] = [
                value for item in contributions for value in item["targets"]
            ]
            predictions: list[Any] = [
                value for item in contributions for value in item["predictions"]
            ]
            if metric == "roc_auc":
                if len(set(targets)) < 2:
                    result["undefined_reason"] = "ROC-AUC target population has only one class."
                else:
                    result["value"] = float(roc_auc_score(targets, predictions))
            else:
                averaging: Any = _required(
                    self._config.evaluation.scoring.f1.averaging, "F1 averaging"
                )
                if averaging not in {"binary", "micro", "macro", "weighted"}:
                    raise ValueError("Unsupported F1 averaging policy.")
                positive: int = _binary(
                    _task_policy(
                        self._config.evaluation.scoring.f1.positive_class_definitions,
                        task,
                    ),
                    "F1 positive class",
                )
                counts: dict[str, int] = _count_binary(targets, predictions, positive)
                result["counts"] = counts
                if averaging == "binary":
                    result["value"] = _prf(
                        counts["tp"], counts["fp"], counts["fn"]
                    )["f1"]
                else:
                    # Undefined class-wise F1 is not silently assigned zero.
                    class_f1: list[float | None] = []
                    for label in (0, 1):
                        class_counts = _count_binary(targets, predictions, label)
                        class_f1.append(
                            _prf(
                                class_counts["tp"],
                                class_counts["fp"],
                                class_counts["fn"],
                            )["f1"]
                        )
                    if averaging == "micro":
                        result["value"] = float(
                            f1_score(targets, predictions, average="micro", labels=[0, 1])
                        )
                    elif averaging == "weighted":
                        support = [targets.count(label) for label in (0, 1)]
                        # A class absent from the reference has exactly zero
                        # weight, so its undefined F1 is not an undefined
                        # component of this support-weighted average.
                        result["value"] = _ratio(
                            sum(
                                float(value) * weight
                                for value, weight in zip(class_f1, support, strict=True)
                                if weight and value is not None
                            ),
                            sum(support),
                        )
                    elif any(value is None for value in class_f1):
                        result["undefined_reason"] = "A configured class-wise F1 is undefined."
                    else:
                        result["value"] = float(
                            f1_score(
                                targets, predictions, average=averaging, labels=[0, 1]
                            )
                        )
                result["averaging"] = averaging
        elif metric == "accuracy":
            numerator: int = sum(item["correct"] for item in contributions)
            denominator: int = sum(item["total"] for item in contributions)
            result.update(
                value=_ratio(numerator, denominator),
                numerator=numerator,
                denominator=denominator,
            )
        elif metric == "bleu":
            scorer: BLEU = self._bleu()
            aggregation: Any = _required(
                self._config.evaluation.scoring.bleu.aggregation_level,
                "BLEU aggregation level",
            )
            if aggregation == "corpus":
                reference_counts: set[int] = {
                    len(item["references"]) for item in contributions
                }
                if len(reference_counts) != 1:
                    raise ValueError("Corpus BLEU requires explicitly aligned reference streams.")
                hypotheses: list[str] = [item["hypothesis"] for item in contributions]
                references: list[list[str]] = [
                    [item["references"][index] for item in contributions]
                    for index in range(next(iter(reference_counts)))
                ]
                score = scorer.corpus_score(hypotheses, references)
                result["value"] = float(score.score)
                result["bleu_statistics"] = {
                    "counts": list(score.counts),
                    "totals": list(score.totals),
                    "system_length": score.sys_len,
                    "reference_length": score.ref_len,
                }
            elif aggregation == "sentence_mean":
                scores: list[float] = [
                    float(
                        scorer.sentence_score(
                            item["hypothesis"], item["references"]
                        ).score
                    )
                    for item in contributions
                ]
                result["value"] = float(np.mean(scores))
            else:
                raise ValueError("Unsupported BLEU aggregation level.")
            result["aggregation"] = aggregation
            result["signature"] = str(scorer.get_signature())
        elif metric == "ndcg":
            scores = [self._ndcg(item) for item in contributions]
            result["query_scores"] = scores
            result["value"] = _macro(scores)
            if result["value"] is None:
                result["undefined_reason"] = "At least one query has zero ideal gain."
        if result["value"] is None and result["undefined_reason"] is None:
            result["undefined_reason"] = "Metric denominator is zero."
        if result["value"] is not None:
            value: float = float(result["value"])
            lower, upper = self._config.evaluation.reporting_scale
            fraction: float = value / 100 if metric == "bleu" else value
            result["display_value"] = lower + fraction * (upper - lower)
            result["display_scale"] = [lower, upper]
        else:
            result["display_value"] = None
        return result

    def _failure_rate(self, response_ids: list[str]) -> dict[str, Any]:
        # Decisions are append-only: a completed boundary review retains its
        # earlier pending decision. Completed identification manifests designate
        # the authoritative revision instead of making record insertion order a
        # scientific selection rule.
        designated: dict[str, set[str]] = defaultdict(set)
        for manifest in self._store.list_records("experiment_manifest"):
            if (
                manifest.get("purpose") != "identification_final"
                or manifest.get("config_hash") != self._config.config_hash
                or manifest.get("corpus_version") != self._config.corpus_version
            ):
                continue
            result: Any = manifest.get("result", {})
            if not isinstance(result, dict) or result.get("status") != "completed":
                continue
            binding: Any = result.get("decision_ids_by_response", {})
            if not isinstance(binding, dict):
                continue
            for response_id in response_ids:
                if response_id in binding:
                    designated[response_id].add(binding[response_id])
        decisions: list[dict[str, Any]] = self._store.list_records("failure_decision")
        matches: dict[str, list[FailureDecision]] = defaultdict(list)
        for payload in decisions:
            if payload.get("response_id") in response_ids:
                selected: set[str] = designated.get(payload["response_id"], set())
                if selected and _content_hash(payload) not in selected:
                    continue
                matches[payload["response_id"]].append(
                    FailureDecision.model_validate(payload)
                )
        versions: set[str] = {
            item.identification_version
            for records in matches.values()
            for item in records
        }
        if len(versions) != 1:
            return {
                "status": "incomplete",
                "reason": "No unique identification version for the exact population.",
                "response_count": len(response_ids),
            }
        finalized: list[FailureDecision] = []
        for response_id in response_ids:
            records: list[FailureDecision] = matches.get(response_id, [])
            if (
                len(designated.get(response_id, set())) > 1
                or len(records) != 1
                or records[0].decision == FailureStatus.UNRESOLVED
            ):
                return {
                    "status": "incomplete",
                    "reason": "Missing, ambiguous, or unresolved response failure decision.",
                    "response_count": len(response_ids),
                }
            finalized.append(records[0])
        failures: int = sum(
            item.decision == FailureStatus.FAILURE for item in finalized
        )
        return {
            "status": "complete",
            "failure_count": failures,
            "response_count": len(response_ids),
            "value": 100 * failures / len(response_ids) if response_ids else None,
            "unit": "percent_responses",
            "identification_version": next(iter(versions)),
            "source": "operational_failure_decisions_not_independent_audit",
        }

    def aggregate_task_metrics(self, response_ids: list[str]) -> dict[str, Any]:
        """Compute metrics per run, then summarize matched repeated populations."""
        identifiers: list[str] = _unique_ids(response_ids, "Response population")
        if not identifiers:
            raise ValueError("An empty response population cannot be aggregated.")
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for response_id in sorted(identifiers):
            response = ModelResponse.model_validate(
                self._store.read_record("model_response", response_id)
            )
            instance = TaskInstance.model_validate(
                self._store.read_record("task_instance", response.instance_id)
            )
            record: dict[str, Any] = self.score_response(instance, response)
            key: tuple[Any, ...] = (
                record["task"], record["model_id"], record["condition"],
                record["reference_version"], record["scoring_version"],
                _content_hash(record["evaluation_context"]),
            )
            groups[key].append(record)

        outputs: list[dict[str, Any]] = []
        for key, records in sorted(groups.items()):
            task, model, condition, reference_version, scoring_version, _ = key
            by_run: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for record in records:
                by_run[record["run_index"]].append(record)
            run_results: list[dict[str, Any]] = []
            manifests: list[tuple[tuple[str, str], ...]] = []
            included_manifests: list[tuple[str, ...]] = []
            for run, population in sorted(by_run.items()):
                instance_ids: list[str] = [item["instance_id"] for item in population]
                _unique_ids(instance_ids, "Task/model/run instance population")
                manifests.append(
                    tuple(sorted(
                        (item["instance_id"], item["reference_hash"])
                        for item in population
                    ))
                )
                included_manifests.append(tuple(sorted(
                    item["instance_id"] for item in population if not item["excluded"]
                )))
                members: list[str] = sorted(item["response_id"] for item in population)
                run_results.append(
                    {
                        "run_index": run,
                        "response_ids": members,
                        "instance_manifest_hash": _content_hash(manifests[-1]),
                        "included_instance_manifest_hash": _content_hash(included_manifests[-1]),
                        **self._metric(population),
                        "failure_rate": self._failure_rate(members),
                    }
                )

            context: dict[str, Any] = records[0]["evaluation_context"]
            protocol: str = context.get("protocol", "main")
            if protocol == "stronger_grounding":
                expected_runs: int = self._config.defenses.stronger_grounding.runs
                report_std: bool = (
                    self._config.defenses.stronger_grounding.report_standard_deviation
                )
            elif protocol == "main":
                expected_runs = self._config.evaluation.runs
                report_std = self._config.evaluation.report_standard_deviation
            elif protocol in {"counterfactual_consistency", "tailored_mitigation"}:
                settings = getattr(self._config.defenses, protocol)
                expected_runs = _required(
                    settings.repetition_count, f"defenses.{protocol}.repetition_count"
                )
                report_std = (
                    self._config.evaluation.report_standard_deviation and expected_runs > 1
                )
            else:
                raise ValueError(
                    "Optional experiments require an implemented explicit repetition protocol."
                )
            reasons: list[str] = []
            if set(by_run) != set(range(expected_runs)):
                reasons.append("Observed run IDs do not match configured repetition count.")
            if len(set(manifests)) != 1:
                reasons.append("Instance/reference populations differ across runs.")
            if len(set(included_manifests)) != 1:
                reasons.append("Included scoring populations differ across runs.")
            if any(item["display_value"] is None for item in run_results):
                reasons.append("At least one run metric is undefined.")
            summary: dict[str, Any] = {
                "status": "incomplete" if reasons else "complete",
                "reasons": reasons,
                "expected_runs": expected_runs,
                "observed_runs": len(by_run),
                "mean": None,
                "standard_deviation": None,
                "unit": "task_score_on_configured_reporting_scale",
            }
            if not reasons:
                values: list[float] = [
                    float(item["display_value"]) for item in run_results
                ]
                summary["mean"] = float(np.mean(values))
                if report_std:
                    ddof: int = _required(
                        self._config.evaluation.standard_deviation_ddof,
                        "evaluation.standard_deviation_ddof",
                    )
                    if ddof >= len(values):
                        raise ValueError("Standard-deviation ddof exceeds run population.")
                    summary["standard_deviation"] = float(np.std(values, ddof=ddof))
                    summary["standard_deviation_ddof"] = ddof
            outputs.append(
                {
                    "task": task,
                    "model_id": model,
                    "condition": condition,
                    "reference_version": reference_version,
                    "scoring_version": scoring_version,
                    "evaluation_context": context,
                    "runs": run_results,
                    "summary": summary,
                }
            )
        return self._persist(
            "task_metrics",
            {
                "response_ids": sorted(identifiers),
                "groups": outputs,
                "failure_rate": self._failure_rate(identifiers),
                "reporting_scale": list(self._config.evaluation.reporting_scale),
                "paired_differences": self._paired_differences(outputs),
            },
        )

    def _paired_differences(self, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Report differences only for explicitly bound matched comparisons."""
        comparisons: list[dict[str, Any]] = []
        originals: dict[tuple[str, str, str], dict[str, Any]] = {}
        for group in groups:
            comparison_id: Any = group["evaluation_context"].get("comparison_id")
            if comparison_id and group["condition"] == "original":
                originals[(group["task"], group["model_id"], comparison_id)] = group
        for group in groups:
            comparison_id = group["evaluation_context"].get("comparison_id")
            key: tuple[str, str, str] = (
                group["task"], group["model_id"], comparison_id
            )
            if not comparison_id or group["condition"] == "original" or key not in originals:
                continue
            original: dict[str, Any] = originals[key]
            controls: Any = group["evaluation_context"].get("comparison_controls_hash")
            if (
                not controls
                or controls != original["evaluation_context"].get("comparison_controls_hash")
                or group["reference_version"] != original["reference_version"]
                or group["scoring_version"] != original["scoring_version"]
            ):
                raise ValueError("Paired comparison lacks identical frozen controls.")
            left: dict[int, Any] = {run["run_index"]: run for run in original["runs"]}
            right: dict[int, Any] = {run["run_index"]: run for run in group["runs"]}
            if set(left) != set(right) or any(
                left[index]["instance_manifest_hash"] != right[index]["instance_manifest_hash"]
                for index in left
            ):
                raise ValueError("Paired comparison uses different instance populations.")
            if any(
                left[index]["included_instance_manifest_hash"]
                != right[index]["included_instance_manifest_hash"]
                for index in left
            ):
                raise ValueError("Paired comparison uses different included scoring populations.")
            if (
                original["summary"]["status"] != "complete"
                or group["summary"]["status"] != "complete"
            ):
                continue
            differences: list[dict[str, Any]] = []
            for index in sorted(left):
                original_fr = left[index]["failure_rate"]
                defended_fr = right[index]["failure_rate"]
                delta_fr: float | None = None
                if (
                    original_fr["status"] == defended_fr["status"] == "complete"
                    and original_fr["identification_version"]
                    == defended_fr["identification_version"]
                ):
                    delta_fr = original_fr["value"] - defended_fr["value"]
                differences.append(
                    {
                        "run_index": index,
                        "task_score_gain": (
                            right[index]["display_value"] - left[index]["display_value"]
                        ),
                        "failure_rate_reduction_percentage_points": delta_fr,
                    }
                )
            comparisons.append(
                {
                    "task": group["task"],
                    "model_id": group["model_id"],
                    "condition": group["condition"],
                    "comparison_id": comparison_id,
                    "runs": differences,
                }
            )
        return comparisons

    def _cohort(self, cohort: AuditCohort, purpose: str, size: int) -> AuditCohort:
        stored = AuditCohort.model_validate(
            self._store.read_record("audit_cohort", cohort.cohort_id)
        )
        if _canonical_json(stored) != _canonical_json(cohort):
            raise ValueError("Supplied cohort differs from its stored manifest.")
        if not cohort.locked or cohort.purpose not in _AUDIT_PURPOSES[purpose]:
            raise ValueError("A locked cohort with the appropriate purpose is required.")
        if len(cohort.response_ids) != size:
            raise ValueError("Audit membership differs from its configured sample size.")
        protocol: dict[str, Any] = cohort.sampling_protocol
        excluded: set[str] = set(protocol.get("excluded_response_ids", []))
        if excluded & set(cohort.response_ids):
            raise ValueError("Audit membership violates declared exclusions.")
        for response_id in cohort.response_ids:
            self._store.read_record("model_response", response_id)
        return stored

    def _prediction_ids(self, cohort: AuditCohort) -> dict[str, str]:
        binding: Any = cohort.sampling_protocol.get("frozen_prediction_ids")
        if not isinstance(binding, dict) or set(binding) != set(cohort.response_ids):
            raise ValueError("Audit requires exact frozen prediction ID bindings.")
        result: dict[str, str] = {}
        for response_id, prediction_id in binding.items():
            if not isinstance(prediction_id, str):
                raise ValueError("Audit scoring requires one designated prediction per response.")
            result[response_id] = prediction_id
        return result

    def _assignment(
        self, assignment_id: str, taxonomy: TaxonomyVersion
    ) -> LabelAssignment:
        found: list[dict[str, Any]] = []
        for kind in _ASSIGNMENT_KINDS:
            try:
                found.append(self._store.read_record(kind, assignment_id))
            except KeyError:
                pass
        if len(found) != 1:
            raise ValueError("Assignment identity is missing or ambiguous.")
        return LabelAssignment.model_validate(
            found[0],
            context={
                "taxonomy": taxonomy,
                "require_taxonomy": True,
                "require_evidence_for_each_label": (
                    self._config.taxonomy.require_evidence_for_each_label
                ),
            },
        )

    def _taxonomy(self, taxonomy_id: Any) -> TaxonomyVersion:
        if not isinstance(taxonomy_id, str):
            raise ValueError("An explicit taxonomy ID is required.")
        found: list[dict[str, Any]] = []
        for kind in _TAXONOMY_KINDS:
            try:
                found.append(self._store.read_record(kind, taxonomy_id))
            except KeyError:
                pass
        if len(found) != 1:
            raise ValueError("Taxonomy identity is missing or ambiguous.")
        return TaxonomyVersion.model_validate(found[0])

    def _annotations(
        self, cohort: AuditCohort, response_id: str, purposes: set[str]
    ) -> list[dict[str, Any]]:
        source_cohorts: set[str] = {
            cohort.cohort_id,
            *cohort.sampling_protocol.get("reference_cohort_ids", []),
        }
        guideline: Any = cohort.sampling_protocol.get("guideline_hash")
        if not isinstance(guideline, str):
            raise ValueError("Audit must bind a frozen guideline hash.")
        records: list[dict[str, Any]] = [
            item for item in self._store.list_records("annotation")
            if item.get("response_id") == response_id
            and item.get("cohort_id") in source_cohorts
            and item.get("purpose") in purposes
            and item.get("status", "submitted") == "submitted"
        ]
        if any(item.get("guideline_hash") != guideline for item in records):
            raise ValueError("Independent annotations use a different guideline.")
        actors: list[str] = [item["annotator_id"] for item in records]
        _unique_ids(actors, "Independent annotators for one response")
        return records

    def _human_reference(
        self,
        cohort: AuditCohort,
        response_id: str,
        purposes: set[str],
        field: str,
        minimum_raters: int,
    ) -> Any:
        records: list[dict[str, Any]] = self._annotations(cohort, response_id, purposes)
        if len(records) < minimum_raters:
            raise ValueError("Independent human reference annotation is incomplete.")
        if any(item.get("uncertain", False) for item in records):
            require_adjudication: bool = True
        else:
            require_adjudication = False
        values: list[Any] = []
        for record in records:
            if field not in record:
                raise ValueError("Independent annotation lacks its required result.")
            value: Any = record[field]
            if field == "labels":
                value = sorted(_unique_ids(value, "Human label assignment"))
            values.append(value)

        allowed_cohorts: set[str] = {
            cohort.cohort_id,
            *cohort.sampling_protocol.get("reference_cohort_ids", []),
        }
        adjudications: list[dict[str, Any]] = [
            item for item in self._store.list_records("adjudication")
            if item.get("response_id") == response_id
            and item.get("cohort_id") in allowed_cohorts
            and (
                item.get("purpose") in purposes
                or (
                    item.get("purpose") == "audit_adjudication"
                    and item.get("reference_purpose") in purposes
                )
            )
        ]
        if adjudications:
            if len(adjudications) != 1:
                raise ValueError("Audit reference adjudication is ambiguous.")
            adjudication: dict[str, Any] = adjudications[0]
            if (
                adjudication.get("guideline_hash")
                != cohort.sampling_protocol["guideline_hash"]
                or adjudication.get("uncertain", False)
                or field not in adjudication
            ):
                raise ValueError("Audit adjudication is incomplete or guideline-incompatible.")
            return adjudication[field]
        if require_adjudication or len({_canonical_json(value) for value in values}) != 1:
            raise ValueError("Human disagreement requires an explicit adjudicated reference.")
        return values[0]

    def score_failure_audit(self, cohort: AuditCohort) -> dict[str, Any]:
        """Score frozen automatic failure/pass decisions against blind references."""
        settings = self._config.audits.failure_identification
        self._cohort(cohort, "failure_identification", settings.instances)
        if settings.positive_class != "failure" or settings.zero_denominator_result != "N/A":
            raise ValueError("Unsupported identification-audit metric convention.")
        required_exclusions: set[str] = set(settings.exclude)
        if not required_exclusions.issubset(
            set(cohort.sampling_protocol.get("exclusion_categories", []))
        ):
            raise ValueError("Identification audit does not record all required exclusions.")
        bindings: dict[str, str] = self._prediction_ids(cohort)
        counts: Counter[str] = Counter({"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        for response_id in cohort.response_ids:
            prediction = FailureDecision.model_validate(
                self._store.read_record("failure_decision", bindings[response_id])
            )
            if (
                prediction.response_id != response_id
                or prediction.identification_version != cohort.frozen_prediction_version
                or prediction.decision == FailureStatus.UNRESOLVED
                or prediction.decision_origin.value not in {"hard_label", "automatic_range"}
            ):
                raise ValueError("Audit prediction is not a final frozen automatic assignment.")
            human: Any = self._human_reference(
                cohort,
                response_id,
                {"independent_identification_audit", "failure_identification"},
                "decision",
                settings.independent_annotators,
            )
            if human not in {"pass", "failure"}:
                raise ValueError("Audit reference must be a final failure/pass judgment.")
            counts.update(_count_binary(
                [int(human == "failure")],
                [int(prediction.decision == FailureStatus.FAILURE)],
                1,
            ))
        if sum(counts.values()) != len(cohort.response_ids):
            raise RuntimeError("Audit confusion counts do not exhaust membership.")
        return self._persist(
            "failure_identification_audit",
            {
                "cohort_id": cohort.cohort_id,
                "cohort_hash": _content_hash(cohort),
                "response_ids": sorted(cohort.response_ids),
                "prediction_version": cohort.frozen_prediction_version,
                "n": len(cohort.response_ids),
                "counts": dict(counts),
                "metrics": _prf(counts["tp"], counts["fp"], counts["fn"]),
                "unit": "native_fraction",
                "undefined_display": settings.zero_denominator_result,
                "status": "complete",
                "audit_labels_used_for_tuning": False,
            },
        )

    def score_label_audit(self, cohort: AuditCohort) -> dict[str, Any]:
        """Compare pre-correction multi-label predictions with independent humans."""
        settings = self._config.audits.failure_categorization
        self._cohort(cohort, "failure_categorization", settings.total_instances)
        protocol: dict[str, Any] = cohort.sampling_protocol
        original: list[str] = protocol.get("original_response_ids", [])
        additional: list[str] = protocol.get("additional_response_ids", [])
        if (
            len(original) != settings.original_agreement_instances
            or len(additional) != settings.additional_instances
            or set(original) & set(additional)
            or set(original + additional) != set(cohort.response_ids)
        ):
            raise ValueError("Expanded audit must preserve the original-plus-added composition.")
        if protocol.get("includes_stable_consensus") is not True:
            raise ValueError("Categorization audit must explicitly include stable-consensus cases.")
        taxonomy: TaxonomyVersion = self._taxonomy(protocol.get("taxonomy_id"))
        if not taxonomy.frozen:
            raise ValueError("Label audit requires the stabilized frozen taxonomy.")
        subtype_ids: list[str] = [
            subtype.subtype_id for subtype in taxonomy.subtype_definitions
        ]
        bindings: dict[str, str] = self._prediction_ids(cohort)
        counts: dict[str, Counter[str]] = {
            subtype: Counter({"tp": 0, "fp": 0, "fn": 0})
            for subtype in subtype_ids
        }
        exact: int = 0
        for response_id in cohort.response_ids:
            prediction: LabelAssignment = self._assignment(bindings[response_id], taxonomy)
            if (
                prediction.response_id != response_id
                or prediction.origin not in _PRECORRECTION_ORIGINS
            ):
                raise ValueError("Audit must evaluate designated pre-correction predictions.")
            human_raw: Any = self._human_reference(
                cohort,
                response_id,
                {"human_agreement", "independent_label_audit", "expanded_label_audit"},
                "labels",
                2,
            )
            human: set[str] = set(_unique_ids(human_raw, "Adjudicated human labels"))
            if not human.issubset(subtype_ids):
                raise ValueError("Human reference contains unknown subtype IDs.")
            predicted: set[str] = set(prediction.labels)
            exact += int(human == predicted)
            for subtype in subtype_ids:
                if subtype in human and subtype in predicted:
                    counts[subtype]["tp"] += 1
                elif subtype in predicted:
                    counts[subtype]["fp"] += 1
                elif subtype in human:
                    counts[subtype]["fn"] += 1
        rows: dict[str, Any] = {
            subtype: {
                **_prf(values["tp"], values["fp"], values["fn"]),
                "support": values["tp"] + values["fn"],
            }
            for subtype, values in counts.items()
        }
        pooled: Counter[str] = Counter()
        for values in counts.values():
            pooled.update(values)
        macro: dict[str, float | None] = {
            metric: _macro([row[metric] for row in rows.values()])
            for metric in ("precision", "recall", "f1")
        }
        return self._persist(
            "failure_categorization_audit",
            {
                "cohort_id": cohort.cohort_id,
                "cohort_hash": _content_hash(cohort),
                "response_ids": sorted(cohort.response_ids),
                "taxonomy_id": taxonomy.taxonomy_id,
                "guideline_hash": taxonomy.guideline_hash,
                "prediction_version": cohort.frozen_prediction_version,
                "n": len(cohort.response_ids),
                "per_subtype": rows,
                "macro": macro,
                "micro": _prf(pooled["tp"], pooled["fp"], pooled["fn"]),
                "exact_label_set_agreement": exact / len(cohort.response_ids),
                "macro_undefined_policy": "propagate_undefined",
                "unit": "native_fraction",
                "undefined_display": settings.zero_denominator_result,
                "predictions_before_human_correction": True,
                "status": "complete",
            },
        )

    def score_human_agreement(self, cohort: AuditCohort) -> dict[str, Any]:
        """Measure original independent annotations, never adjudicated labels."""
        settings = self._config.annotation.agreement_study
        self._cohort(cohort, "human_agreement", settings.instances)
        policy: Any = _required(
            settings.uncertain_label_handling, "Human-agreement uncertain-label handling"
        )
        if policy not in {"reject", "missing_annotation", "exclude_instance"}:
            raise ValueError("Unsupported uncertain-label handling policy.")
        if "calibration_examples" not in cohort.sampling_protocol.get(
            "exclusion_categories", []
        ):
            raise ValueError("Agreement cohort must record calibration exclusion.")
        taxonomy: TaxonomyVersion = self._taxonomy(
            cohort.sampling_protocol.get("taxonomy_id")
        )
        if not taxonomy.frozen:
            raise ValueError("Agreement requires frozen annotation definitions.")
        subtype_ids: list[str] = [
            item.subtype_id for item in taxonomy.subtype_definitions
        ]
        submissions: dict[str, list[dict[str, Any]]] = {}
        raters: set[str] = set()
        excluded: list[str] = []
        exact: int = 0
        exact_denominator: int = 0
        for response_id in cohort.response_ids:
            records = self._annotations(cohort, response_id, {"human_agreement"})
            if len(records) != settings.annotators_per_instance:
                raise ValueError("Agreement requires the configured independent rater count.")
            for record in records:
                labels = _unique_ids(record.get("labels", []), "Independent labels")
                if not set(labels).issubset(subtype_ids):
                    raise ValueError("Independent annotation contains unknown subtypes.")
                if "labels" not in record:
                    raise ValueError("Missing annotations are not empty label sets.")
            uncertain: bool = any(record.get("uncertain", False) for record in records)
            if uncertain and policy == "reject":
                raise ValueError("Uncertain agreement annotation requires protocol resolution.")
            if uncertain and policy == "exclude_instance":
                excluded.append(response_id)
                continue
            submissions[response_id] = records
            raters.update(record["annotator_id"] for record in records)
            if not uncertain:
                exact_denominator += 1
                exact += int(len({
                    tuple(sorted(record["labels"])) for record in records
                }) == 1)

        coder_ids: list[str] = sorted(raters)
        instance_ids: list[str] = sorted(submissions)
        alpha_results: dict[str, dict[str, Any]] = {}
        for subtype in subtype_ids:
            matrix: np.ndarray = np.full(
                (len(coder_ids), len(instance_ids)), np.nan, dtype=float
            )
            coder_index: dict[str, int] = {
                actor: index for index, actor in enumerate(coder_ids)
            }
            for column, response_id in enumerate(instance_ids):
                for record in submissions[response_id]:
                    if record.get("uncertain", False):
                        continue
                    matrix[coder_index[record["annotator_id"]], column] = float(
                        subtype in record["labels"]
                    )
            reason: str | None = None
            alpha: float | None = None
            finite: np.ndarray = matrix[np.isfinite(matrix)]
            if len(finite) == 0 or len(set(finite.tolist())) < 2:
                reason = "Constant or empty annotation population; alpha is undefined."
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", RuntimeWarning)
                        computed: float = float(
                            krippendorff.alpha(
                                reliability_data=matrix,
                                level_of_measurement="nominal",
                                value_domain=[0, 1],
                            )
                        )
                    if math.isfinite(computed):
                        alpha = computed
                    else:
                        reason = "Krippendorff alpha is nonfinite."
                except (ValueError, ZeroDivisionError, RuntimeWarning):
                    reason = "Insufficient comparable annotation information."
            alpha_results[subtype] = {"alpha": alpha, "undefined_reason": reason}
        return self._persist(
            "human_agreement",
            {
                "cohort_id": cohort.cohort_id,
                "taxonomy_id": taxonomy.taxonomy_id,
                "guideline_hash": taxonomy.guideline_hash,
                "n": len(cohort.response_ids),
                "included_instances": len(instance_ids),
                "excluded_instance_ids": excluded,
                "coder_ids": coder_ids,
                "per_subtype": alpha_results,
                "macro_alpha": _macro([
                    row["alpha"] for row in alpha_results.values()
                ]),
                "macro_undefined_policy": "propagate_undefined",
                "exact_label_set_agreement": _ratio(exact, exact_denominator),
                "exact_agreement_numerator": exact,
                "exact_agreement_denominator": exact_denominator,
                "uncertain_label_handling": policy,
                "pre_adjudication": True,
                "unit": "native_coefficient_or_fraction",
                "status": "complete",
            },
        )

    def score_taxonomy_stability(
        self, taxonomy_ids: list[str], cohort: AuditCohort
    ) -> dict[str, Any]:
        """Align independently discovered taxonomies using fixed audit assignments."""
        settings = self._config.taxonomy_stability
        self._cohort(cohort, "taxonomy_stability", settings.audit_instances)
        identifiers: list[str] = _unique_ids(taxonomy_ids, "Taxonomy comparison IDs")
        if len(identifiers) < 2:
            raise ValueError("Taxonomy stability requires at least two taxonomies.")
        threshold: float = _probability(
            _required(settings.recovery_match_threshold, "Recovery threshold"),
            "Recovery threshold",
        )
        unmatched: Any = _required(settings.unmatched_category_policy, "Unmatched policy")
        empty: Any = _required(settings.empty_set_policy, "Empty-set policy")
        normalization: Any = _required(
            settings.prevalence_normalization, "Prevalence normalization"
        )
        base: float = _finite(
            _required(settings.jsd_logarithm_base, "JSD logarithm base"),
            "JSD logarithm base",
        )
        schedule: Any = _required(
            settings.comparison_aggregation, "Stability comparison aggregation"
        )
        if unmatched not in {"zero", "exclude"}:
            raise ValueError("Unsupported unmatched-category policy.")
        if empty not in {"one", "zero", "undefined"}:
            raise ValueError("Unsupported empty-set policy.")
        if normalization != "label_mass":
            raise ValueError("Unsupported prevalence normalization.")
        if base <= 0 or base == 1:
            raise ValueError("Invalid JSD logarithm base.")
        if settings.category_matching != "maximum_weight_bipartite":
            raise ValueError("Unsupported category matching.")
        protocol: dict[str, Any] = cohort.sampling_protocol
        assignment_bindings: Any = protocol.get("taxonomy_assignment_ids")
        runs: Any = protocol.get("discovery_runs")
        if not isinstance(assignment_bindings, dict) or not isinstance(runs, dict):
            raise ValueError("Stability cohort requires assignments and discovery-run provenance.")
        taxonomies: dict[str, TaxonomyVersion] = {
            identity: self._taxonomy(identity) for identity in identifiers
        }
        assignments: dict[str, dict[str, set[str]]] = {}
        sets: dict[str, dict[str, set[str]]] = {}
        for identity, taxonomy in taxonomies.items():
            if not taxonomy.frozen or identity not in runs:
                raise ValueError("Stability requires completed, provenance-bound discovery runs.")
            binding: Any = assignment_bindings.get(identity)
            if not isinstance(binding, dict) or set(binding) != set(cohort.response_ids):
                raise ValueError("Every taxonomy requires complete fixed-audit assignments.")
            labels: dict[str, set[str]] = {}
            subtype_sets: dict[str, set[str]] = {
                subtype.subtype_id: set() for subtype in taxonomy.subtype_definitions
            }
            for response_id in cohort.response_ids:
                assignment = self._assignment(binding[response_id], taxonomy)
                if assignment.response_id != response_id or assignment.uncertain:
                    raise ValueError("Stability assignments are missing or unresolved.")
                labels[response_id] = set(assignment.labels)
                for subtype in assignment.labels:
                    subtype_sets[subtype].add(response_id)
            assignments[identity] = labels
            sets[identity] = subtype_sets

        if schedule == "all_pairs":
            pairs: list[tuple[str, str]] = list(itertools.combinations(identifiers, 2))
        elif isinstance(schedule, Mapping) and schedule.get("mode") == "reference":
            reference_id: Any = schedule.get("reference_taxonomy_id")
            if reference_id not in identifiers:
                raise ValueError("Reference taxonomy is outside the comparison population.")
            pairs = [
                (reference_id, identity)
                for identity in identifiers if identity != reference_id
            ]
        else:
            raise ValueError("Unsupported stability comparison schedule.")

        def overlap(left: set[str], right: set[str]) -> float | None:
            union: set[str] = left | right
            if union:
                return len(left & right) / len(union)
            return {"one": 1.0, "zero": 0.0, "undefined": None}[empty]

        comparisons: list[dict[str, Any]] = []
        for left_id, right_id in pairs:
            left_sets: dict[str, set[str]] = sets[left_id]
            right_sets: dict[str, set[str]] = sets[right_id]
            left_names: list[str] = sorted(left_sets)
            right_names: list[str] = sorted(right_sets)
            weights: np.ndarray = np.zeros((len(left_names), len(right_names)))
            for i, left_name in enumerate(left_names):
                for j, right_name in enumerate(right_names):
                    value: float | None = overlap(
                        left_sets[left_name], right_sets[right_name]
                    )
                    # Empty categories cannot establish semantic recovery.
                    weights[i, j] = (
                        value if value is not None
                        and left_sets[left_name] and right_sets[right_name] else 0.0
                    )
            row_indices, column_indices = linear_sum_assignment(-weights)
            mapping: dict[str, str] = {}
            matched: list[dict[str, Any]] = []
            recovered: int = 0
            for row, column in zip(row_indices, column_indices, strict=True):
                left_name = left_names[int(row)]
                right_name = right_names[int(column)]
                value = overlap(left_sets[left_name], right_sets[right_name])
                supported: bool = bool(
                    left_sets[left_name] and right_sets[right_name]
                )
                is_recovered: bool = (
                    supported and value is not None and value >= threshold and value > 0
                )
                recovered += int(is_recovered)
                mapping[right_name] = left_name
                matched.append(
                    {
                        "reference_subtype": left_name,
                        "alternative_subtype": right_name,
                        "jaccard": value,
                        "recovered": is_recovered,
                        "nonempty_support": supported,
                    }
                )
            unmatched_left: list[str] = sorted(set(left_names) - set(mapping.values()))
            unmatched_right: list[str] = sorted(set(right_names) - set(mapping))
            overlaps: list[float | None] = [item["jaccard"] for item in matched]
            if unmatched == "zero":
                overlaps.extend([0.0] * (len(unmatched_left) + len(unmatched_right)))
            instance_overlaps: list[float | None] = []
            for response_id in cohort.response_ids:
                left_labels: set[str] = assignments[left_id][response_id]
                right_labels: set[str] = {
                    mapping.get(label, f"unmatched:{right_id}:{label}")
                    for label in assignments[right_id][response_id]
                }
                instance_overlaps.append(overlap(left_labels, right_labels))

            # Alignment dimensions retain unmatched categories on separate axes.
            axes: list[tuple[str | None, str | None]] = [
                (item["reference_subtype"], item["alternative_subtype"])
                for item in matched
            ]
            axes += [(name, None) for name in unmatched_left]
            axes += [(None, name) for name in unmatched_right]
            left_mass: np.ndarray = np.asarray([
                len(left_sets[name]) if name is not None else 0 for name, _ in axes
            ], dtype=float)
            right_mass: np.ndarray = np.asarray([
                len(right_sets[name]) if name is not None else 0 for _, name in axes
            ], dtype=float)
            jsd: float | None = None
            if left_mass.sum() > 0 and right_mass.sum() > 0:
                left_mass /= left_mass.sum()
                right_mass /= right_mass.sum()
                jsd = float(jensenshannon(left_mass, right_mass, base=base) ** 2)
            comparisons.append(
                {
                    "reference_taxonomy_id": left_id,
                    "alternative_taxonomy_id": right_id,
                    "reference_run": runs[left_id],
                    "alternative_run": runs[right_id],
                    "alignment": matched,
                    "unmatched_reference": unmatched_left,
                    "unmatched_alternative": unmatched_right,
                    "taxonomy_recovery_rate": recovered / len(left_names),
                    "matched_category_jaccard": _macro(overlaps),
                    "instance_label_set_jaccard": _macro(instance_overlaps),
                    "jensen_shannon_divergence": jsd,
                    "reference_subtype_count": len(left_names),
                    "alternative_subtype_count": len(right_names),
                    "recovered_reference_subtypes": recovered,
                }
            )
        metrics: tuple[str, ...] = (
            "taxonomy_recovery_rate",
            "matched_category_jaccard",
            "instance_label_set_jaccard",
            "jensen_shannon_divergence",
        )
        return self._persist(
            "taxonomy_stability",
            {
                "cohort_id": cohort.cohort_id,
                "cohort_hash": _content_hash(cohort),
                "taxonomy_ids": identifiers,
                "n": len(cohort.response_ids),
                "comparisons": comparisons,
                "mean_across_declared_comparisons": {
                    name: _macro([item[name] for item in comparisons])
                    for name in metrics
                },
                "settings": settings.model_dump(mode="json"),
                "jsd_is_divergence_not_distance": True,
                "unit": "native_fraction_or_divergence",
                "status": "complete",
            },
        )

    def write_report(self, results: dict[str, Any], path: str) -> None:
        """Atomically write a full-precision canonical report without overwriting."""
        if not isinstance(results, dict):
            raise TypeError("Report results must be a dictionary.")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("An explicit report destination is required.")
        report: dict[str, Any] = {
            "study": self._config.study.title,
            "experiment_id": self._config.experiment_id,
            "corpus_version": self._config.corpus_version,
            "config_hash": self._config.config_hash,
            "scoring_configuration": self._config.evaluation.scoring.model_dump(
                mode="json"
            ),
            "implementation_choices": _json_primitive(
                self._config.implementation_choices
            ),
            "results": results,
            "undefined_representation": None,
            "notes": [
                "Null metric values are unavailable, not numerical zero.",
                "Task metrics, response failure rates, and audit metrics are distinct.",
                "Missing results are reported as unavailable, never substituted.",
                "Numerical values retain full precision.",
            ],
        }

        def check_secrets(value: Any) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    normalized: str = str(key).lower().replace("-", "_")
                    if normalized in {
                        "api_key", "authorization", "password", "access_token",
                        "refresh_token", "client_secret", "private_key", "credentials",
                    } and item is not None:
                        raise ValueError("Report contains a credential-bearing field.")
                    check_secrets(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    check_secrets(item)

        check_secrets(report)
        content: bytes = (_canonical_json(report) + "\n").encode("utf-8")
        destination: Path = Path(path).expanduser().absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".evaluation-", dir=destination.parent
        )
        temporary: Path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if destination.is_symlink() or not destination.is_file():
                    raise FileExistsError("Existing report destination is unsafe.") from None
                if destination.read_bytes() != content:
                    raise FileExistsError(
                        "A different report exists; select a new versioned destination."
                    ) from None
            self._store.save_blob(content)
        finally:
            temporary.unlink(missing_ok=True)
