"""Human-in-the-loop taxonomy discovery for RQ2.

Discovery is separate from applying an already supplied taxonomy. No published
subtypes are copied into the discovery initialization.

Operational protocol
====================

The following executable declarations use existing configuration policy
fields and must be documented as implementation choices before execution.

``taxonomy_discovery.coverage_definition`` is a mapping:

    {
        "mode": "fully_explained" | "at_least_one_known",
        "population": "non_initialization",
        "sampling_method": "simple_random_without_replacement",
        "task_input_projection": "evidence_as_documents",
        "assignment_round_index": 0,
        "output_policy": {
            "other_with_known_labels": true,
            "allow_empty_labels_when_uncertain": true
        },
        "annotation_protocols": {
            "initialization": <AnnotationService protocol>,
            "refinement": <AnnotationService protocol>
        }
    }

The annotation protocols are complete protocols accepted by AnnotationService.
Initialization requires judgment="discovery", taxonomy_id=null, and cohort_id=null.
Refinement requires judgment="discovery" and cohort_id=null; its taxonomy_id is
bound to the current discovered version when tasks are requested.

``taxonomy_discovery.revisit_prior_assignments_policy`` must explicitly select
``"all_non_initialization"`` or ``{"mode": "all_non_initialization"}``.
Every expansion therefore reclassifies the original non-initialization
population. Results are never relabeled into a new taxonomy without inference.

``taxonomy_discovery.persistent_other_policy`` is a mapping:

    {
        "mode": "retain_other",
        "uncertain_cases": "review" | "block",
        "on_insufficient_coverage": "block",
        "epsilon_usage": "not_operationalized"
    }

The supplied epsilon is recorded, but is not given an invented mathematical
meaning. Convergence uses completed human refinement, no new categories, and the
explicit coverage threshold.

RQ2 assignment round_index is the explicitly declared protocol value above,
not an RQ3 deliberation round. Discovery iteration lives in provenance manifests.

Storage
=======

Records are content-addressed and append-only once used. A taxonomy draft used
by assignments cannot be edited through the supplied ArtifactStore. Finalization
therefore creates a separate frozen snapshot if the converged draft is already
referenced. The finalization manifest explicitly connects that snapshot to its
converged source; historical assignments keep their original taxonomy identity.

No audit annotations are used for initialization, refinement, or convergence.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from annotation_app import AnnotationService
from config import ExperimentConfig
from generation_context import review_task_input
from model_client import ModelClient
from schemas import (
    AnnotationPurpose,
    AnnotationTask,
    AssignmentOrigin,
    FailureDecision,
    FailureStatus,
    LabelAssignment,
    ModelResponse,
    ResponseStatus,
    StageState,
    StageStatus,
    SubtypeDefinition,
    SubtypeRules,
    TaskInstance,
    TaxonomyVersion,
    _canonical_json,
    _content_hash,
    _identifier,
)
from storage import ArtifactStore


_TAXONOMY_KINDS: tuple[str, ...] = ("taxonomy", "taxonomy_version")
_ANNOTATION_KINDS: tuple[str, ...] = ("annotation", "adjudication")
_INITIALIZATION_PURPOSE: str = AnnotationPurpose.TAXONOMY_INITIALIZATION.value
_REFINEMENT_PURPOSE: str = AnnotationPurpose.TAXONOMY_REFINEMENT.value


class _PendingAnnotations(RuntimeError):
    """Internal nonblocking control flow for incomplete human references."""

    def __init__(self, task_ids: Sequence[str]) -> None:
        super().__init__("Discovery requires additional human annotation.")
        self.task_ids: list[str] = sorted(set(task_ids))


class _ProtocolBlocked(RuntimeError):
    """A scientifically unresolved condition that must not trigger more inference."""


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


def _ids(values: Any, name: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be an explicit identifier sequence.")
    result: list[str] = [_identifier(value) for value in values]
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate identities.")
    if not result and not allow_empty:
        raise ValueError(f"{name} must not be empty.")
    return sorted(result)


def _semantic_definition(definition: SubtypeDefinition) -> dict[str, Any]:
    """Compare category meaning independently of accumulated support references."""
    return definition.model_dump(
        mode="json",
        exclude={"supporting_annotation_ids", "provenance"},
    )


class TaxonomyManager:
    """Discover, version, and freeze human-supported CTI failure taxonomies."""

    def __init__(
        self,
        config: ExperimentConfig,
        store: ArtifactStore,
        models: ModelClient,
        annotations: AnnotationService,
        *, initialization: str = "human",
    ) -> None:
        if initialization not in {"human", "model"}:
            raise ValueError("Select human or explicit model initialization.")
        if initialization == "model" and (
            config.operational.taxonomy_mode != "discovery"
            or config.taxonomy.discovery_initial_taxonomy is not None
        ):
            raise ValueError("Model initialization requires discovery without a supplied taxonomy.")
        self._initialization = initialization
        self._config: ExperimentConfig = config
        self._store: ArtifactStore = store
        self._models: ModelClient = models
        self._annotations: AnnotationService = annotations
        self._population: list[str] = []
        self._population_id: str | None = None
        self._execution_id: str | None = None
        self._started_at: datetime | None = None
        self._current_taxonomy: TaxonomyVersion | None = None
        self._current_assignments: list[LabelAssignment] = []
        self._history: list[dict[str, Any]] = []

    @property
    def _protocol_id(self) -> str:
        return _content_hash(
            {
                "implementation_contract": "rq2-taxonomy-discovery-v1",
                "initialization_source": self._initialization,
                "config_hash": self._config.config_hash,
                "corpus_version": self._config.corpus_version,
                "discovery": self._config.taxonomy_discovery,
                "taxonomy_settings": {
                    "multiple_labels_allowed": (
                        self._config.taxonomy.multiple_labels_allowed
                    ),
                    "require_evidence_for_each_label": (
                        self._config.taxonomy.require_evidence_for_each_label
                    ),
                    "guideline_manifest": self._config.taxonomy.guideline_manifest,
                },
            }
        )

    def _event(self, purpose: str, content: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "purpose": purpose,
            "protocol_id": self._protocol_id,
            "config_hash": self._config.config_hash,
            "corpus_version": self._config.corpus_version,
            "population_manifest_id": self._population_id,
            **content,
        }
        identity: str = _content_hash(payload)
        record: dict[str, Any] = {"manifest_id": identity, **payload}
        self._store.create_record("experiment_manifest", identity, record)
        return record

    def _events(self, purpose: str) -> list[dict[str, Any]]:
        return [
            item
            for item in self._store.list_records("experiment_manifest")
            if item.get("purpose") == purpose
            and item.get("protocol_id") == self._protocol_id
            and (
                self._population_id is None
                or item.get("population_manifest_id") == self._population_id
            )
        ]

    def _find(
        self, kinds: tuple[str, ...], record_id: str
    ) -> tuple[str, dict[str, Any]]:
        found: list[tuple[str, dict[str, Any]]] = []
        for kind in kinds:
            try:
                found.append((kind, self._store.read_record(kind, record_id)))
            except KeyError:
                pass
        if len(found) != 1:
            raise ValueError(f"Missing or ambiguous record identity: {record_id}.")
        return found[0]

    def _taxonomy(self, taxonomy_id: str) -> TaxonomyVersion:
        _, payload = self._find(_TAXONOMY_KINDS, taxonomy_id)
        return TaxonomyVersion.model_validate(payload)

    def _immutable(self, kind: str, record_id: str) -> None:
        records: list[dict[str, Any]] = self._store.list_records(
            kind,
            {"record_id": record_id, "lifecycle": ["immutable", "frozen"]},
        )
        if len(records) != 1:
            raise ValueError(f"{kind}/{record_id} must be immutable.")

    def _settings(self) -> dict[str, Any]:
        settings: dict[str, Any] = _mapping(
            self._config.taxonomy_discovery.coverage_definition,
            "taxonomy_discovery.coverage_definition",
        )
        required_keys: set[str] = {
            "mode",
            "population",
            "sampling_method",
            "task_input_projection",
            "assignment_round_index",
            "output_policy",
            "annotation_protocols",
        }
        if set(settings) != required_keys:
            raise ValueError(
                "coverage_definition must contain exactly the documented "
                "executable protocol fields."
            )
        if settings["mode"] not in {"fully_explained", "at_least_one_known"}:
            raise ValueError("Unsupported discovery coverage definition.")
        if settings["population"] != "non_initialization":
            raise ValueError("Coverage population must explicitly be non_initialization.")
        if settings["sampling_method"] != "simple_random_without_replacement":
            raise ValueError("Unsupported initialization sampling method.")
        if settings["task_input_projection"] != "evidence_as_documents":
            raise ValueError("Unsupported classification input projection.")
        round_index: Any = settings["assignment_round_index"]
        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or round_index < 0
        ):
            raise ValueError("RQ2 assignment_round_index must be explicitly nonnegative.")
        output: dict[str, Any] = _mapping(settings["output_policy"], "output_policy")
        if set(output) != {
            "other_with_known_labels",
            "allow_empty_labels_when_uncertain",
        } or any(not isinstance(value, bool) for value in output.values()):
            raise ValueError("Discovery output edge-case policies must be explicit booleans.")
        protocols: dict[str, Any] = _mapping(
            settings["annotation_protocols"], "annotation_protocols"
        )
        if set(protocols) != {"initialization", "refinement"}:
            raise ValueError("Both discovery annotation protocols must be supplied.")
        revisit: Any = self._config.taxonomy_discovery.revisit_prior_assignments_policy
        revisit_mode: Any = revisit.get("mode") if isinstance(revisit, dict) else revisit
        if revisit_mode != "all_non_initialization":
            raise ValueError(
                "This implementation requires explicit all_non_initialization "
                "reclassification after expansion."
            )
        persistent: dict[str, Any] = self._persistent_policy()
        if persistent["epsilon_usage"] != "not_operationalized":
            raise ValueError("No mathematical interpretation of epsilon is implemented.")
        discovery = self._config.taxonomy_discovery
        if (
            discovery.human_review_other_cases is not True
            or discovery.add_human_supported_new_categories is not True
            or discovery.stop_when_no_new_categories is not True
        ):
            raise ValueError("Human-supported refinement and no-new-category stopping are required.")
        if discovery.stability_threshold_epsilon is None:
            raise ValueError("Record the unresolved epsilon setting before formal discovery.")
        return settings

    def _persistent_policy(self) -> dict[str, Any]:
        policy: dict[str, Any] = _mapping(
            self._config.taxonomy_discovery.persistent_other_policy,
            "persistent_other_policy",
        )
        if set(policy) != {
            "mode",
            "uncertain_cases",
            "on_insufficient_coverage",
            "epsilon_usage",
        }:
            raise ValueError("Persistent-other policy must use the documented fields.")
        if (
            policy["mode"] != "retain_other"
            or policy["uncertain_cases"] not in {"review", "block"}
            or policy["on_insufficient_coverage"] != "block"
            or policy["epsilon_usage"] != "not_operationalized"
        ):
            raise ValueError("Unsupported persistent-other or uncertainty policy.")
        return policy

    def _annotation_protocol(
        self, role: str, taxonomy: TaxonomyVersion | None = None
    ) -> dict[str, Any]:
        settings: dict[str, Any] = self._settings()
        protocol: dict[str, Any] = _mapping(
            settings["annotation_protocols"][role], f"{role} annotation protocol"
        )
        # The configuration cannot contain its own digest. Bind this derived
        # value in the copied runtime protocol, preserving the hashed declaration.
        if protocol.get("config_hash") in {None, "$config_hash"}:
            protocol["config_hash"] = self._config.config_hash
        if (
            protocol.get("config_hash") != self._config.config_hash
            or protocol.get("judgment") != "discovery"
            or protocol.get("cohort_id") is not None
        ):
            raise ValueError("Discovery requires its own non-audit human proposal protocol.")
        if role == "initialization":
            if protocol.get("taxonomy_id") is not None:
                raise ValueError("Initialization must not receive an existing taxonomy.")
        else:
            if taxonomy is None:
                raise ValueError("Refinement requires its exact parent taxonomy.")
            declared: Any = protocol.get("taxonomy_id")
            if declared not in {None, taxonomy.taxonomy_id}:
                raise ValueError("Refinement protocol names a different taxonomy.")
            protocol["taxonomy_id"] = taxonomy.taxonomy_id
            if protocol.get("guideline_hash") != taxonomy.guideline_hash:
                raise ValueError("Refinement cannot silently change the guideline.")
        return protocol

    def _response(self, response_id: str) -> tuple[ModelResponse, TaskInstance]:
        response = ModelResponse.model_validate(
            self._store.read_record("model_response", response_id)
        )
        instance = TaskInstance.model_validate(
            self._store.read_record("task_instance", response.instance_id),
            context={"require_content_hash": True},
        )
        if response.config_hash != self._config.config_hash:
            raise ValueError("Discovery response belongs to another configuration.")
        if (
            instance.transformation_log.get("corpus_version")
            != self._config.corpus_version
        ):
            raise ValueError("Discovery input belongs to another corpus.")
        if response.status == ResponseStatus.TRANSPORT_FAILURE:
            raise ValueError("Transport failures are not completed taxonomy-discovery inputs.")
        if not instance.reference.get("answer") or not instance.reference.get(
            "evidence_records"
        ):
            raise ValueError("Discovery requires applicable reference answers and evidence.")
        return response, instance

    def _establish_population(self, failure_ids: list[str]) -> list[str]:
        if not isinstance(failure_ids, list):
            raise TypeError("failure_ids must be model-response identifiers.")
        population: list[str] = sorted({_identifier(item) for item in failure_ids})
        if not population:
            raise ValueError("Taxonomy discovery requires a nonempty failure population.")
        if self._population and self._population != population:
            raise ValueError("A manager execution cannot change its failure population.")

        final_manifests: list[dict[str, Any]] = [
            item
            for item in self._store.list_records("experiment_manifest")
            if item.get("purpose") == "identification_final"
            and item.get("config_hash") == self._config.config_hash
            and set(population).issubset(
                item.get("result", {}).get("failure_response_ids", [])
            )
        ]
        bindings: list[dict[str, str]] = []
        for manifest in final_manifests:
            designated: Any = manifest["result"].get("decision_ids_by_response", {})
            if all(identity in designated for identity in population):
                bindings.append({identity: designated[identity] for identity in population})
        distinct: dict[str, dict[str, str]] = {
            _canonical_json(binding): binding for binding in bindings
        }
        if len(distinct) > 1:
            raise ValueError("Failure population has ambiguous identification snapshots.")
        designated_ids: dict[str, str] = next(iter(distinct.values()), {})

        from sampling import source_response_mapping
        sampling_sources = source_response_mapping(self._store, population)
        members: list[dict[str, Any]] = []
        versions: set[str] = set()
        for response_id in population:
            response, instance = self._response(response_id)
            if designated_ids:
                decision_id: str = designated_ids[response_id]
                decision = FailureDecision.model_validate(
                    self._store.read_record("failure_decision", decision_id)
                )
            else:
                candidates: list[dict[str, Any]] = [
                    item
                    for item in self._store.list_records(
                        "failure_decision", {"response_id": response_id}
                    )
                    if item.get("decision") in {"pass", "failure"}
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        "Each discovery response requires one designated final "
                        "failure decision, or an identification manifest."
                    )
                decision = FailureDecision.model_validate(candidates[0])
                decision_id = _content_hash(decision)
            self._immutable("failure_decision", decision_id)
            if (
                decision.response_id != response_id
                or decision.decision != FailureStatus.FAILURE
                or decision.requires_review
            ):
                raise ValueError("Discovery population contains an unresolved or passed response.")
            versions.add(decision.identification_version)
            members.append(
                {
                    "response_id": response_id,
                    "sampling_source_response_id": sampling_sources[response_id],
                    "response_hash": _content_hash(response),
                    "instance_id": instance.instance_id,
                    "instance_hash": instance.content_hash,
                    "reference_hash": _content_hash(instance.reference),
                    "failure_decision_id": decision_id,
                }
            )
        if len(versions) != 1:
            raise ValueError("Discovery cannot pool different identification versions.")
        self._population = population
        content: dict[str, Any] = {
            "purpose": "taxonomy_failure_population",
            "protocol_id": self._protocol_id,
            "config_hash": self._config.config_hash,
            "corpus_version": self._config.corpus_version,
            "identification_version": next(iter(versions)),
            "members": members,
        }
        identity: str = _content_hash(content)
        self._store.create_record(
            "experiment_manifest", identity, {"manifest_id": identity, **content}
        )
        self._population_id = identity
        self._execution_id = _content_hash(
            {"protocol_id": self._protocol_id, "population_manifest_id": identity}
        )
        return population

    def _initial_sample(self) -> dict[str, Any]:
        samples: list[dict[str, Any]] = self._events("taxonomy_initialization_sample")
        if len(samples) != 1:
            raise ValueError("Exactly one saved initialization sample is required.")
        return samples[0]

    def request_initialization(self, failure_ids: list[str]) -> list[AnnotationTask]:
        """Sample once, persist membership, then request independent human proposals."""
        self._config.validate("discover_taxonomy")
        if self._config.operational.taxonomy_mode != "discovery":
            raise ValueError("Initialization is available only in explicit discovery mode.")
        settings: dict[str, Any] = self._settings()
        population: list[str] = self._establish_population(failure_ids)
        count: int | None = self._config.taxonomy_discovery.initial_subset_size
        seed: int | None = self._config.taxonomy_discovery.initialization_seed
        if count is None or seed is None:
            raise ValueError("Initialization size and seed must be explicitly resolved.")
        if count > len(population):
            raise ValueError("Initialization sample exceeds the failure population.")
        if count == len(population):
            raise ValueError(
                "The configured non-initialization coverage population would be empty."
            )
        from sampling import sample_response_ids
        selected, sampling_sources = sample_response_ids(self._store, population, count=count, seed=seed)
        protocol: dict[str, Any] = self._annotation_protocol("initialization")
        sample: dict[str, Any] = self._event(
            "taxonomy_initialization_sample",
            {
                "response_ids": selected,
                "seed": seed,
                "sample_size": count,
                "sampling_method": settings["sampling_method"],
                "sampling_implementation": "python_random_sample_sorted_root_source_response_ids",
                "source_response_ids": [sampling_sources[identity] for identity in selected],
                "source_response_ids_by_response": sampling_sources,
                "annotation_protocol_hash": _content_hash(protocol),
            },
        )
        tasks: list[AnnotationTask] = self._annotations.create_tasks(
            selected, _INITIALIZATION_PURPOSE, protocol
        )
        self._event(
            "taxonomy_initialization_tasks",
            {
                "sample_manifest_id": sample["manifest_id"],
                "response_ids": selected,
                "task_ids": sorted(task.annotation_task_id for task in tasks),
                "annotation_protocol_hash": _content_hash(protocol),
            },
        )
        return tasks

    def _references(
        self, tasks: list[AnnotationTask], purpose: str,
        *, taxonomy: TaxonomyVersion | None = None,
    ) -> list[dict[str, Any]]:
        if purpose not in {_INITIALIZATION_PURPOSE, _REFINEMENT_PURPOSE}:
            raise ValueError("Only operational discovery references are eligible.")
        role = "initialization" if purpose == _INITIALIZATION_PURPOSE else "refinement"
        protocol_hash = _content_hash(self._annotation_protocol(role, taxonomy))
        grouped: dict[str, list[AnnotationTask]] = {}
        for task in tasks:
            if task.purpose.value != purpose or task.cohort_id is not None:
                raise ValueError("Audit or unrelated tasks cannot support discovery.")
            grouped.setdefault(task.response_id, []).append(task)
        pending: list[str] = []
        references: list[dict[str, Any]] = []
        for response_id, group in sorted(grouped.items()):
            task_ids: set[str] = {task.annotation_task_id for task in group}
            submissions: list[dict[str, Any]] = [
                item
                for item in self._store.list_records("annotation")
                if item.get("annotation_task_id") in task_ids
                and item.get("purpose") == purpose
                and item.get("audit_only") is False
                and item.get("cohort_id") is None
            ]
            submitted: set[str] = {
                item["annotation_task_id"] for item in submissions
            }
            if submitted != task_ids:
                pending.extend(task_ids - submitted)
                continue
            self._annotations.create_adjudications(sorted(task_ids))
            try:
                reference: dict[str, Any] = self._annotations.get_reference(
                    response_id, purpose, protocol_hash=protocol_hash,
                    taxonomy_id=taxonomy.taxonomy_id if taxonomy is not None else None,
                )
            except LookupError:
                outstanding: list[str] = [
                    item["annotation_task_id"]
                    for item in self._store.list_records(
                        "annotation_task", {"response_id": response_id}
                    )
                    if item.get("cohort_id") is None
                    and item.get("status") in {"pending", "in_progress", "draft"}
                    and item.get("purpose")
                    in {
                        purpose,
                        "operational_adjudication",
                        "senior_escalation",
                    }
                ]
                pending.extend(outstanding or sorted(task_ids))
                continue
            if (
                reference.get("audit_only") is not False
                or reference.get("cohort_id") is not None
                or reference.get("purpose") != purpose
                or reference.get("uncertain", True)
                or reference.get("protocol_hash") != protocol_hash
            ):
                raise ValueError("Discovery reference is unresolved or belongs to an audit.")
            if reference.get("guideline_hash") != group[0].guideline_hash:
                raise ValueError("Discovery reference guideline differs from its task.")
            annotation_ids: list[str] = _ids(
                reference.get("annotation_ids"), "Human reference annotation IDs"
            )
            for annotation_id in annotation_ids:
                kind, record = self._find(_ANNOTATION_KINDS, annotation_id)
                self._immutable(kind, annotation_id)
                if (
                    record.get("response_id") != response_id
                    or record.get("audit_only") is not False
                    or record.get("cohort_id") is not None
                    or record.get("reference_purpose", record.get("purpose")) != purpose
                    or record.get("guideline_hash") != reference["guideline_hash"]
                ):
                    raise ValueError("Human support has incompatible discovery provenance.")
            references.append(reference)
        if pending:
            raise _PendingAnnotations(pending)
        return references

    def _records_from_ids(
        self, annotation_ids: list[str], purpose: str, responses: set[str]
    ) -> list[dict[str, Any]]:
        identifiers: list[str] = _ids(
            annotation_ids, "Completed annotation IDs", allow_empty=not responses
        )
        records: list[dict[str, Any]] = []
        for identity in identifiers:
            kind, record = self._find(_ANNOTATION_KINDS, identity)
            self._immutable(kind, identity)
            if (
                record.get("response_id") not in responses
                or record.get("reference_purpose", record.get("purpose")) != purpose
                or record.get("cohort_id") is not None
                or record.get("audit_only") is not False
                or record.get("uncertain", True)
                or not isinstance(record.get("taxonomy_proposal"), dict)
            ):
                raise ValueError("Ineligible, incomplete, or audit-derived category proposal.")
            records.append(record)
        return records

    def _validate_definitions(
        self, taxonomy: TaxonomyVersion, fallback: str
    ) -> None:
        known: set[str] = {
            definition.subtype_id for definition in taxonomy.subtype_definitions
        }
        if fallback in known or "uncertain" in known:
            raise ValueError("Fallback and uncertainty sentinels are not substantive categories.")
        if set(taxonomy.inclusion_exclusion_rules) != known:
            raise ValueError("Every discovered category needs operational annotation rules.")
        for definition in taxonomy.subtype_definitions:
            rules: SubtypeRules = taxonomy.inclusion_exclusion_rules[definition.subtype_id]
            from model_initialization import has_model_support
            model_supported = has_model_support(self, definition)
            if (
                not definition.evidence_requirements
                or (not definition.supporting_annotation_ids and not model_supported)
                or not rules.inclusion
                or not rules.exclusion
            ):
                raise ValueError("Category definitions require human support and complete criteria.")

    def _build_version(
        self,
        parent: TaxonomyVersion | None,
        records: list[dict[str, Any]],
        guideline_hash: str,
    ) -> tuple[TaxonomyVersion, list[str]]:
        definitions: dict[str, SubtypeDefinition] = {
            item.subtype_id: item
            for item in (parent.subtype_definitions if parent is not None else [])
        }
        rules: dict[str, SubtypeRules] = (
            dict(parent.inclusion_exclusion_rules) if parent is not None else {}
        )
        additions: list[str] = []
        for record in records:
            if record["guideline_hash"] != guideline_hash:
                raise ValueError("Category proposals use a different frozen guideline.")
            proposal: dict[str, Any] = _mapping(
                record["taxonomy_proposal"], "Human taxonomy proposal"
            )
            if set(proposal) != {"new_subtypes", "rules", "no_new_categories"}:
                raise ValueError("Human proposals must explicitly state additions and convergence.")
            raw_definitions: Any = proposal["new_subtypes"]
            raw_rules: dict[str, Any] = _mapping(proposal["rules"], "Proposed rules")
            if not isinstance(raw_definitions, list):
                raise ValueError("Human new_subtypes must be an array.")
            if (
                not isinstance(proposal["no_new_categories"], bool)
                or proposal["no_new_categories"] != (not raw_definitions)
            ):
                raise ValueError("Human no-new-category statement contradicts its proposals.")
            parsed: list[SubtypeDefinition] = [
                SubtypeDefinition.model_validate(item) for item in raw_definitions
            ]
            proposed_ids: list[str] = [item.subtype_id for item in parsed]
            if len(proposed_ids) != len(set(proposed_ids)) or set(raw_rules) != set(
                proposed_ids
            ):
                raise ValueError("Proposed category IDs and rules are inconsistent.")
            if parsed and not record.get("evidence_spans"):
                raise ValueError("New categories require observable supporting evidence.")
            for definition in parsed:
                subtype_id: str = definition.subtype_id
                proposed_rule = SubtypeRules.model_validate(raw_rules[subtype_id])
                if subtype_id in definitions:
                    if (
                        _canonical_json(_semantic_definition(definitions[subtype_id]))
                        != _canonical_json(_semantic_definition(definition))
                        or _canonical_json(rules[subtype_id])
                        != _canonical_json(proposed_rule)
                    ):
                        raise _ProtocolBlocked(
                            "Human proposals conflict on an existing category identity. "
                            "Explicit human reconciliation is required; no automatic "
                            "merging, rewriting, or renaming was performed."
                        )
                    # Identical proposals are agreement, not a new category.
                    continue
                payload: dict[str, Any] = definition.model_dump(mode="json")
                payload["supporting_annotation_ids"] = [record["annotation_id"]]
                payload["provenance"] = {
                    "human_annotation_id": record["annotation_id"],
                    "supporting_response_id": record["response_id"],
                    "guideline_hash": guideline_hash,
                    "evidence_spans": record["evidence_spans"],
                    "protocol_id": self._protocol_id,
                }
                definitions[subtype_id] = SubtypeDefinition.model_validate(payload)
                rules[subtype_id] = proposed_rule
                additions.append(subtype_id)
        if parent is not None and not additions:
            return parent, []
        if not definitions:
            raise _ProtocolBlocked(
                "Human initialization established no categories; discovery cannot "
                "classify or claim a stabilized taxonomy."
            )
        content: dict[str, Any] = {
            "parent_id": parent.taxonomy_id if parent is not None else None,
            "iteration": parent.iteration + 1 if parent is not None else 0,
            "subtype_definitions": [
                definitions[identity].model_dump(mode="json")
                for identity in sorted(definitions)
            ],
            "inclusion_exclusion_rules": {
                identity: rules[identity].model_dump(mode="json")
                for identity in sorted(rules)
            },
            "guideline_hash": guideline_hash,
            "frozen": False,
        }
        identity: str = _content_hash(
            {"protocol_id": self._protocol_id, "taxonomy_content": content}
        )
        taxonomy = TaxonomyVersion(taxonomy_id=identity, **content)
        self._validate_definitions(
            taxonomy, self._config.taxonomy_discovery.fallback_label
        )
        self._store.create_record(
            "taxonomy", identity, taxonomy.model_dump(mode="json")
        )
        self._event(
            "taxonomy_version_support",
            {
                "taxonomy_id": identity,
                "parent_taxonomy_id": taxonomy.parent_id,
                "iteration": taxonomy.iteration,
                "new_subtype_ids": sorted(additions),
                "annotation_ids": sorted(record["annotation_id"] for record in records),
                "supporting_response_ids": sorted(
                    {record["response_id"] for record in records}
                ),
                "taxonomy_content_hash": _content_hash(taxonomy),
            },
        )
        return taxonomy, sorted(additions)

    def create_initial_taxonomy(self, annotation_ids: list[str]) -> TaxonomyVersion:
        """Construct the initial categories solely from completed human references."""
        sample: dict[str, Any] = self._initial_sample()
        tasks: list[AnnotationTask] = self.request_initialization(self._population)
        references: list[dict[str, Any]] = self._references(
            tasks, _INITIALIZATION_PURPOSE
        )
        designated: list[str] = sorted(
            {
                identity
                for reference in references
                for identity in reference["annotation_ids"]
            }
        )
        if _ids(annotation_ids, "Initialization annotation IDs") != designated:
            raise ValueError(
                "Initialization requires exactly the completed designated human "
                "reference records, not task IDs or unadjudicated alternatives."
            )
        records: list[dict[str, Any]] = self._records_from_ids(
            designated, _INITIALIZATION_PURPOSE, set(sample["response_ids"])
        )
        protocol: dict[str, Any] = self._annotation_protocol("initialization")
        taxonomy, additions = self._build_version(
            None, records, protocol["guideline_hash"]
        )
        self._event(
            "taxonomy_initialization_complete",
            {
                "sample_manifest_id": sample["manifest_id"],
                "taxonomy_id": taxonomy.taxonomy_id,
                "annotation_ids": designated,
                "new_subtype_ids": additions,
            },
        )
        return taxonomy

    def _model_id(self) -> str:
        configured: Any = self._config.taxonomy_discovery.assisting_model
        if isinstance(configured, str):
            model_id: str = configured
        elif isinstance(configured, dict):
            model_id = _identifier(configured.get("model_id"))
            if model_id not in self._config.models.evaluation:
                raise ValueError("Assisting model_id is not a configured evaluation model.")
            entry = self._config.models.evaluation[model_id]
            if (
                configured.get("endpoint_or_checkpoint") != entry.endpoint_or_checkpoint
                or configured.get("revision") != entry.revision
            ):
                raise ValueError("Assisting model endpoint/revision disagrees with its model ID.")
        else:
            raise ValueError("An explicitly configured assisting model is required.")
        panel_names: set[str] = {
            entry.name for entry in self._config.models.categorization_panel
        }
        if model_id not in self._config.models.evaluation and model_id not in panel_names:
            raise ValueError("ModelClient cannot resolve the declared assisting model.")
        return model_id

    def _packet(
        self, response_id: str, taxonomy: TaxonomyVersion
    ) -> dict[str, Any]:
        response, instance = self._response(response_id)
        role: str = "refinement"
        protocol: dict[str, Any] = self._annotation_protocol(role, taxonomy)
        applicability_by_response = protocol.get("reference_applicability", {})
        applicability: Any = applicability_by_response.get(
            response_id, applicability_by_response.get("*")
        )
        if not isinstance(applicability, dict) or not applicability:
            raise ValueError("Classification reference applicability is unresolved.")
        fields = ("evidence_id", "document_id", "source_id", "source_version", "snippet", "role",
                  "publication_time", "modification_time", "claim_valid_from", "claim_valid_until")

        def project(value: Mapping[str, Any]) -> dict[str, Any]:
            return {key: value[key] for key in fields if key in value}

        task_input = review_task_input(self._store, response, instance)
        return {
            "task_input": task_input,
            "evaluated_response": {
                "response_id": response.response_id,
                "raw_text": response.raw_text,
                "parsed_answer": response.parsed_answer,
                "status": response.status.value,
                "schema_errors": response.schema_errors,
            },
            "reference_material": {
                "version": instance.reference["version"],
                "answer": instance.reference["answer"],
                "evidence_records": [
                    project(item) for item in instance.reference["evidence_records"]
                ],
                "applicability": applicability,
            },
            "current_taxonomy": taxonomy.model_dump(mode="json"),
            "discovery_settings": {
                "fallback_label": self._config.taxonomy_discovery.fallback_label,
                "human_review_other_cases": (
                    self._config.taxonomy_discovery.human_review_other_cases
                ),
                "add_human_supported_new_categories": (
                    self._config.taxonomy_discovery.add_human_supported_new_categories
                ),
            },
            "taxonomy_settings": {
                "multiple_labels_allowed": self._config.taxonomy.multiple_labels_allowed,
                "require_evidence_for_each_label": (
                    self._config.taxonomy.require_evidence_for_each_label
                ),
            },
            "output_policy": self._settings()["output_policy"],
        }

    def _output_contract(self, taxonomy: TaxonomyVersion) -> dict[str, Any]:
        labels: list[str] = sorted(
            [
                item.subtype_id for item in taxonomy.subtype_definitions
            ]
            + [self._config.taxonomy_discovery.fallback_label]
        )
        return {
            "version": "rq2-classification-v1",
            "kind": "classification",
            "taxonomy_id": taxonomy.taxonomy_id,
            "discovery_mode": True,
            "allow_empty_labels_when_uncertain": (
                self._settings()["output_policy"]["allow_empty_labels_when_uncertain"]
            ),
            "json_schema": {
                "type": "object",
                "required": ["labels", "evidence_spans", "uncertain"],
                "additionalProperties": False,
                "properties": {
                    "labels": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"type": "string", "enum": labels},
                    },
                    "uncertain": {"type": "boolean"},
                    "evidence_spans": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["source_ref"],
                            "additionalProperties": False,
                            "properties": {
                                "source_ref": {"type": "string", "minLength": 1},
                                "quote": {"type": ["string", "null"]},
                                "start_offset": {
                                    "type": ["integer", "null"],
                                    "minimum": 0,
                                },
                                "end_offset": {
                                    "type": ["integer", "null"],
                                    "minimum": 0,
                                },
                                "subtype_id": {"type": ["string", "null"]},
                                "relation_type": {"type": ["string", "null"]},
                                "metadata": {"type": "object"},
                            },
                        },
                    },
                },
            },
        }

    def _validate_spans(
        self, assignment: LabelAssignment, packet: dict[str, Any]
    ) -> None:
        texts: dict[str, set[str]] = {}
        response: dict[str, Any] = packet["evaluated_response"]
        texts[response["response_id"]] = {response["raw_text"]}
        records: list[dict[str, Any]] = [
            *packet["task_input"]["source_documents"],
            *packet["reference_material"]["evidence_records"],
        ]
        for record in records:
            snippet: Any = record.get("snippet")
            if not isinstance(snippet, str):
                continue
            for key in ("evidence_id", "document_id"):
                identity: Any = record.get(key)
                if isinstance(identity, str):
                    texts.setdefault(identity, set()).add(snippet)
        for span in assignment.evidence_spans:
            sources: set[str] = texts.get(span.source_ref, set())
            if len(sources) != 1:
                raise ValueError("Classifier span cites an unavailable or ambiguous source.")
            text: str = next(iter(sources))
            if span.quote is not None and span.quote not in text:
                raise ValueError("Classifier evidence quotation is not verbatim.")
            if span.start_offset is not None and span.end_offset is not None:
                if span.end_offset > len(text):
                    raise ValueError("Classifier evidence offsets exceed the source.")
                if (
                    span.quote is not None
                    and text[span.start_offset:span.end_offset] != span.quote
                ):
                    raise ValueError("Classifier evidence offsets disagree with its quote.")

    def classify_remaining(
        self, failure_ids: list[str], taxonomy: TaxonomyVersion
    ) -> list[LabelAssignment]:
        """Classify the fixed non-initialization population against this version."""
        population: list[str] = self._establish_population(failure_ids)
        stored: TaxonomyVersion = self._taxonomy(taxonomy.taxonomy_id)
        if _canonical_json(stored) != _canonical_json(taxonomy):
            raise ValueError("Classification taxonomy differs from its stored version.")
        self._validate_definitions(
            taxonomy, self._config.taxonomy_discovery.fallback_label
        )
        initial_ids: set[str] = set(self._initial_sample()["response_ids"])
        selected: list[str] = sorted(set(population) - initial_ids)
        if not selected:
            raise ValueError("No non-initialization classification population exists.")
        model_id: str = self._model_id()
        template: str | None = self._config.taxonomy_discovery.prompt_template_path
        if template is None:
            raise ValueError("Discovery classification prompt is unresolved.")
        self._models.validate_endpoints([model_id])
        settings: dict[str, Any] = self._settings()
        packets: dict[str, dict[str, Any]] = {
            response_id: self._packet(response_id, taxonomy)
            for response_id in selected
        }
        requests: list[dict[str, Any]] = []
        for response_id in selected:
            response, _ = self._response(response_id)
            packet: dict[str, Any] = packets[response_id]
            prompt: str = self._models.render_prompt(template, packet)
            requests.append(
                {
                    "model_id": model_id,
                    "prompt": prompt,
                    "run_index": 0,
                    "schema_name": "rq2_taxonomy_classification",
                    "instance_id": response.instance_id,
                    "schema_contract": self._output_contract(taxonomy),
                }
            )
        outputs: list[ModelResponse] = self._models.generate_batch(requests)
        if len(outputs) != len(selected):
            raise RuntimeError("Classifier returned an incomplete request population.")
        assignments: list[LabelAssignment] = []
        exceptions: list[dict[str, Any]] = []
        for response_id, output in zip(selected, outputs, strict=True):
            if output.status != ResponseStatus.SUCCESS:
                exceptions.append(
                    {
                        "response_id": response_id,
                        "classifier_response_id": output.response_id,
                        "status": output.status.value,
                    }
                )
                continue
            parsed: dict[str, Any] = output.parsed_answer
            try:
                fallback: str = self._config.taxonomy_discovery.fallback_label
                proposed: list[str] = parsed["labels"]
                if (
                    fallback in proposed
                    and len(proposed) > 1
                    and not settings["output_policy"]["other_with_known_labels"]
                ):
                    raise ValueError("Mixed fallback/current labels violate the protocol.")
                if (
                    not self._config.taxonomy.multiple_labels_allowed
                    and len(set(proposed) - {fallback}) > 1
                ):
                    raise ValueError("Classifier violated the configured label multiplicity.")
                content: dict[str, Any] = {
                    "response_id": response_id,
                    "taxonomy_id": taxonomy.taxonomy_id,
                    "actor_id": model_id,
                    "round_index": settings["assignment_round_index"],
                    "labels": proposed,
                    "evidence_spans": parsed["evidence_spans"],
                    "uncertain": parsed["uncertain"],
                    "origin": AssignmentOrigin.DISCOVERY.value,
                    "supersedes_id": None,
                }
                identity: str = _content_hash(
                    {
                        "assignment": content,
                        "classifier_response_id": output.response_id,
                        "protocol_id": self._protocol_id,
                    }
                )
                assignment = LabelAssignment.model_validate(
                    {"assignment_id": identity, **content},
                    context={
                        "taxonomy": taxonomy,
                        "require_taxonomy": True,
                        "discovery_mode": True,
                        "fallback_label": fallback,
                        "require_evidence_for_each_label": (
                            self._config.taxonomy.require_evidence_for_each_label
                        ),
                    },
                )
                self._validate_spans(assignment, packets[response_id])
                self._store.create_record(
                    "label_assignment", identity, assignment.model_dump(mode="json")
                )
                self._event(
                    "taxonomy_assignment_provenance",
                    {
                        "assignment_id": identity,
                        "response_id": response_id,
                        "classifier_response_id": output.response_id,
                        "request_hash": output.usage.get("request_hash"),
                        "taxonomy_id": taxonomy.taxonomy_id,
                        "taxonomy_content_hash": _content_hash(taxonomy),
                        "guideline_hash": taxonomy.guideline_hash,
                        "discovery_iteration": taxonomy.iteration,
                        "packet_hash": _content_hash(packets[response_id]),
                    },
                )
                assignments.append(assignment)
            except (ValueError, KeyError, TypeError) as exc:
                exceptions.append(
                    {
                        "response_id": response_id,
                        "classifier_response_id": output.response_id,
                        "status": "invalid_classification",
                        "error_class": type(exc).__name__,
                    }
                )
        self._event(
            "taxonomy_classification_cycle",
            {
                "taxonomy_id": taxonomy.taxonomy_id,
                "iteration": taxonomy.iteration,
                "expected_response_ids": selected,
                "assignment_ids": sorted(item.assignment_id for item in assignments),
                "classifier_response_ids": sorted(item.response_id for item in outputs),
                "exceptions": exceptions,
                "complete": not exceptions and len(assignments) == len(selected),
            },
        )
        self._current_assignments = assignments
        if exceptions:
            raise _ProtocolBlocked(
                "Classification contains transport, schema, refusal, or evidence "
                "exceptions. They remain unclassified and are not coverage or "
                "evidence of no new categories. Resolve the recorded exceptions "
                "under an explicitly authorized protocol before proceeding."
            )
        return assignments

    def _assignment_set(
        self, assignments: list[LabelAssignment]
    ) -> tuple[TaxonomyVersion, list[LabelAssignment]]:
        if not assignments:
            raise ValueError("An explicit classification assignment population is required.")
        taxonomy_ids: set[str] = {item.taxonomy_id for item in assignments}
        response_ids: list[str] = [item.response_id for item in assignments]
        if len(taxonomy_ids) != 1 or len(response_ids) != len(set(response_ids)):
            raise ValueError("Assignment population mixes taxonomy versions or response duplicates.")
        taxonomy = self._taxonomy(next(iter(taxonomy_ids)))
        validated: list[LabelAssignment] = []
        for assignment in assignments:
            stored = self._store.read_record("label_assignment", assignment.assignment_id)
            if _canonical_json(stored) != _canonical_json(assignment):
                raise ValueError("Assignment differs from its persisted prediction.")
            parsed = LabelAssignment.model_validate(
                stored,
                context={
                    "taxonomy": taxonomy,
                    "require_taxonomy": True,
                    "discovery_mode": True,
                    "fallback_label": self._config.taxonomy_discovery.fallback_label,
                },
            )
            if parsed.origin != AssignmentOrigin.DISCOVERY:
                raise ValueError("RQ2 refinement requires discovery predictions.")
            validated.append(parsed)
        return taxonomy, sorted(validated, key=lambda item: item.response_id)

    def request_other_reviews(
        self, assignments: list[LabelAssignment]
    ) -> list[AnnotationTask]:
        """Request human refinement for fallback and explicitly routed uncertainty."""
        taxonomy, validated = self._assignment_set(assignments)
        fallback: str = self._config.taxonomy_discovery.fallback_label
        policy: dict[str, Any] = self._persistent_policy()
        uncertain_ids: list[str] = [
            item.response_id for item in validated if item.uncertain
        ]
        if uncertain_ids and policy["uncertain_cases"] == "block":
            raise _ProtocolBlocked(
                "Uncertain classification requires protocol resolution; the "
                "configured policy does not authorize refinement routing."
            )
        response_ids: list[str] = sorted(
            {
                item.response_id
                for item in validated
                if fallback in item.labels
                or (item.uncertain and policy["uncertain_cases"] == "review")
            }
        )
        tasks: list[AnnotationTask] = []
        if response_ids:
            protocol: dict[str, Any] = self._annotation_protocol("refinement", taxonomy)
            tasks = self._annotations.create_tasks(
                response_ids, _REFINEMENT_PURPOSE, protocol
            )
        self._event(
            "taxonomy_refinement_tasks",
            {
                "taxonomy_id": taxonomy.taxonomy_id,
                "iteration": taxonomy.iteration,
                "assignment_ids": sorted(item.assignment_id for item in validated),
                "response_ids": response_ids,
                "task_ids": sorted(item.annotation_task_id for item in tasks),
                "uncertain_response_ids": uncertain_ids,
                "fallback_response_ids": sorted(
                    item.response_id for item in validated if fallback in item.labels
                ),
            },
        )
        return tasks

    def _review_cycle(self, taxonomy_id: str) -> dict[str, Any]:
        cycles: list[dict[str, Any]] = [
            item
            for item in self._events("taxonomy_refinement_tasks")
            if item["taxonomy_id"] == taxonomy_id
        ]
        if len(cycles) != 1:
            raise ValueError("Exactly one designated refinement cycle is required.")
        return cycles[0]

    def refine_taxonomy(
        self, taxonomy: TaxonomyVersion, review_ids: list[str]
    ) -> TaxonomyVersion:
        """Add only completed, reconciled, human-supported categories."""
        stored: TaxonomyVersion = self._taxonomy(taxonomy.taxonomy_id)
        if stored.frozen or taxonomy.frozen:
            raise ValueError("A frozen taxonomy cannot be expanded by RQ2.")
        if _canonical_json(stored) != _canonical_json(taxonomy):
            raise ValueError("Refinement parent differs from its stored content.")
        cycle: dict[str, Any] = self._review_cycle(taxonomy.taxonomy_id)
        tasks: list[AnnotationTask] = [
            AnnotationTask.model_validate(
                self._store.read_record("annotation_task", identity)
            )
            for identity in cycle["task_ids"]
        ]
        references: list[dict[str, Any]] = (
            self._references(tasks, _REFINEMENT_PURPOSE, taxonomy=taxonomy) if tasks else []
        )
        designated: list[str] = sorted(
            {
                identity
                for reference in references
                for identity in reference["annotation_ids"]
            }
        )
        if _ids(review_ids, "Completed refinement IDs", allow_empty=True) != designated:
            raise ValueError("Refinement requires exactly the designated completed human records.")
        records: list[dict[str, Any]] = self._records_from_ids(
            designated, _REFINEMENT_PURPOSE, set(cycle["response_ids"])
        )
        for record in records:
            if record.get("taxonomy_id") != taxonomy.taxonomy_id:
                raise ValueError("Human refinement belongs to another parent taxonomy.")
        child, additions = self._build_version(
            taxonomy, records, taxonomy.guideline_hash
        )
        self._event(
            "taxonomy_refinement_complete",
            {
                "parent_taxonomy_id": taxonomy.taxonomy_id,
                "taxonomy_id": child.taxonomy_id,
                "iteration": taxonomy.iteration,
                "refinement_cycle_id": cycle["manifest_id"],
                "assignment_ids": cycle["assignment_ids"],
                "review_response_ids": cycle["response_ids"],
                "annotation_ids": designated,
                "new_subtype_ids": additions,
                "complete": True,
                "unresolved_review_ids": [],
            },
        )
        return child

    def _coverage(
        self, taxonomy: TaxonomyVersion, assignments: list[LabelAssignment]
    ) -> dict[str, Any]:
        settings: dict[str, Any] = self._settings()
        expected: set[str] = set(self._population) - set(
            self._initial_sample()["response_ids"]
        )
        known: set[str] = {
            item.subtype_id for item in taxonomy.subtype_definitions
        }
        fallback: str = self._config.taxonomy_discovery.fallback_label
        covered: list[str] = []
        for assignment in assignments:
            supported: bool = bool(set(assignment.labels) & known)
            fully_explained: bool = supported and fallback not in assignment.labels
            if (
                not assignment.uncertain
                and (
                    fully_explained
                    if settings["mode"] == "fully_explained"
                    else supported
                )
            ):
                covered.append(assignment.response_id)
        observed: set[str] = {item.response_id for item in assignments}
        complete: bool = observed == expected and len(assignments) == len(expected)
        threshold: float | None = self._config.taxonomy_discovery.coverage_threshold
        if threshold is None:
            raise ValueError("Discovery coverage threshold is unresolved.")
        return {
            "definition": settings["mode"],
            "population": settings["population"],
            "denominator": len(expected),
            "numerator": len(covered),
            "value": len(covered) / len(expected) if expected else None,
            "threshold": threshold,
            "covered_response_ids": sorted(covered),
            "unclassified_response_ids": sorted(expected - observed),
            "classification_complete": complete,
            "uncertain_response_ids": sorted(
                item.response_id for item in assignments if item.uncertain
            ),
            "other_response_ids": sorted(
                item.response_id for item in assignments if fallback in item.labels
            ),
        }

    def check_convergence(
        self, taxonomy: TaxonomyVersion, assignments: list[LabelAssignment]
    ) -> bool:
        """Require completed human refinement, no additions, and sufficient coverage."""
        current, validated = self._assignment_set(assignments)
        if current.taxonomy_id != taxonomy.taxonomy_id:
            raise ValueError("Convergence cannot mix taxonomy and assignment versions.")
        coverage: dict[str, Any] = self._coverage(taxonomy, validated)
        completed: list[dict[str, Any]] = [
            item
            for item in self._events("taxonomy_refinement_complete")
            if item["parent_taxonomy_id"] == taxonomy.taxonomy_id
            and item["assignment_ids"]
            == sorted(assignment.assignment_id for assignment in validated)
        ]
        review_complete: bool = len(completed) == 1 and completed[0]["complete"]
        additions: list[str] | None = (
            completed[0]["new_subtype_ids"] if review_complete else None
        )
        converged: bool = (
            review_complete
            and additions == []
            and coverage["classification_complete"]
            and coverage["value"] is not None
            and coverage["value"] >= coverage["threshold"]
        )
        self._event(
            "taxonomy_convergence",
            {
                "taxonomy_id": taxonomy.taxonomy_id,
                "taxonomy_content_hash": _content_hash(taxonomy),
                "assignment_ids": sorted(item.assignment_id for item in validated),
                "refinement_complete": review_complete,
                "refinement_manifest_id": (
                    completed[0]["manifest_id"] if review_complete else None
                ),
                "new_subtype_ids": additions,
                "coverage": coverage,
                "epsilon": self._config.taxonomy_discovery.stability_threshold_epsilon,
                "epsilon_usage": "not_operationalized",
                "converged": converged,
            },
        )
        return converged

    def freeze_taxonomy(self, taxonomy_id: str) -> TaxonomyVersion:
        """Finalize only a taxonomy with a complete successful convergence record."""
        taxonomy: TaxonomyVersion = self._taxonomy(taxonomy_id)
        existing: list[dict[str, Any]] = [
            item
            for item in self._events("taxonomy_discovery_finalization")
            if item["source_taxonomy_id"] == taxonomy_id
            or item["taxonomy_id"] == taxonomy_id
        ]
        if existing:
            if len(existing) != 1:
                raise ValueError("Ambiguous discovery finalization.")
            final = self._taxonomy(existing[0]["taxonomy_id"])
            if not final.frozen:
                raise ValueError("Finalization manifest references an unfrozen taxonomy.")
            return final
        convergence: list[dict[str, Any]] = [
            item
            for item in self._events("taxonomy_convergence")
            if item["taxonomy_id"] == taxonomy_id
            and item.get("converged") is True
            and item["taxonomy_content_hash"] == _content_hash(taxonomy)
        ]
        if len(convergence) != 1:
            raise ValueError("Freezing requires one completed successful convergence evaluation.")
        self._validate_definitions(
            taxonomy, self._config.taxonomy_discovery.fallback_label
        )
        if taxonomy.frozen:
            final = taxonomy
            finalization_mode: str = "already_frozen"
        else:
            payload: dict[str, Any] = taxonomy.model_dump(mode="json")
            payload["frozen"] = True
            TaxonomyVersion.model_validate(payload)
            kind, _ = self._find(_TAXONOMY_KINDS, taxonomy_id)
            try:
                self._store.update_draft(kind, taxonomy_id, {"frozen": True})
                final = self._taxonomy(taxonomy_id)
                finalization_mode = "draft_flag_finalized"
            except PermissionError:
                # ArtifactStore forbids mutating drafts referenced by assignments.
                # Preserve their identities and create a clearly linked final
                # snapshot, rather than weakening append-only storage guarantees.
                payload["parent_id"] = taxonomy.taxonomy_id
                payload["iteration"] = taxonomy.iteration + 1
                payload["taxonomy_id"] = _content_hash(
                    {
                        "final_snapshot_of": taxonomy.taxonomy_id,
                        "convergence_manifest_id": convergence[0]["manifest_id"],
                        "taxonomy_content": {
                            key: value
                            for key, value in payload.items()
                            if key != "taxonomy_id"
                        },
                    }
                )
                final = TaxonomyVersion.model_validate(payload)
                self._store.create_record(
                    "taxonomy", final.taxonomy_id, final.model_dump(mode="json")
                )
                finalization_mode = "immutable_final_snapshot"
        final_kind, _ = self._find(_TAXONOMY_KINDS, final.taxonomy_id)
        freeze_hash: str = self._store.freeze_records(
            final_kind, [final.taxonomy_id]
        )
        self._event(
            "taxonomy_discovery_finalization",
            {
                "source_taxonomy_id": taxonomy_id,
                "taxonomy_id": final.taxonomy_id,
                "convergence_manifest_id": convergence[0]["manifest_id"],
                "taxonomy_freeze_manifest": freeze_hash,
                "finalization_mode": finalization_mode,
                "category_changes_during_finalization": [],
                "historical_assignments_rewritten": False,
                "coverage": convergence[0]["coverage"],
                "discovery_claim": True,
            },
        )
        return final

    def _checkpoint(
        self,
        status: StageStatus,
        pending: list[str],
        result: dict[str, Any],
        *,
        output_id: str | None = None,
    ) -> None:
        if self._execution_id is None or self._population_id is None:
            raise RuntimeError("Discovery execution is not initialized.")
        previous: list[StageState] = [
            StageState.model_validate(item)
            for item in self._store.list_records("stage_state")
            if item.get("stage") == "discover_taxonomy"
            and item.get("execution_id") == self._execution_id
        ]
        completed: set[str] = {
            identity for state in previous for identity in state.completed_work
        }
        completed.update(
            assignment.response_id for assignment in self._current_assignments
        )
        timestamp: datetime = _now()
        started: datetime = min(
            [
                self._started_at or timestamp,
                *[
                    state.started_at
                    for state in previous
                    if state.started_at is not None
                ],
            ]
        )
        state = StageState(
            stage="discover_taxonomy",
            execution_id=self._execution_id,
            config_hash=self._config.config_hash,
            status=status,
            completed_work=sorted(completed),
            pending_work=(
                [] if status == StageStatus.COMPLETED
                else sorted(set(self._population) - completed)
            ),
            outstanding_annotation_task_ids=sorted(set(pending)),
            cursor={
                "taxonomy_id": (
                    self._current_taxonomy.taxonomy_id
                    if self._current_taxonomy is not None else None
                ),
                "iteration": (
                    self._current_taxonomy.iteration
                    if self._current_taxonomy is not None else None
                ),
                "assignment_ids": sorted(
                    item.assignment_id for item in self._current_assignments
                ),
                "history": self._history,
                "result": result,
            },
            input_manifests={"failure_population": self._population_id},
            output_manifests={"discovery": output_id} if output_id else {},
            started_at=started,
            updated_at=timestamp,
            completed_at=timestamp if status == StageStatus.COMPLETED else None,
            metadata={
                "protocol_id": self._protocol_id,
                "audit_annotations_used": False,
            },
        )
        self._store.set_stage_state(
            "discover_taxonomy", state.model_dump(mode="json")
        )

    def _result(
        self,
        status: StageStatus,
        pending: list[str] | None = None,
        reasons: list[str] | None = None,
    ) -> dict[str, Any]:
        coverage: dict[str, Any] | None = None
        if self._current_taxonomy is not None and self._current_assignments:
            if all(
                item.taxonomy_id == self._current_taxonomy.taxonomy_id
                for item in self._current_assignments
            ):
                coverage = self._coverage(
                    self._current_taxonomy, self._current_assignments
                )
        return {
            "status": status.value,
            "execution_id": self._execution_id,
            "population_manifest_id": self._population_id,
            "protocol_id": self._protocol_id,
            "taxonomy_id": (
                self._current_taxonomy.taxonomy_id
                if self._current_taxonomy is not None else None
            ),
            "iteration": (
                self._current_taxonomy.iteration
                if self._current_taxonomy is not None else None
            ),
            "pending_annotation_task_ids": sorted(set(pending or [])),
            "classified_count": len(self._current_assignments),
            "classification_assignment_ids": sorted(
                item.assignment_id for item in self._current_assignments
            ),
            "coverage": coverage,
            "history": copy.deepcopy(self._history),
            "blocking_reasons": reasons or [],
            "discovery_claim": (
                self._config.operational.taxonomy_mode == "discovery"
            ),
        }

    def _fixed_application(self) -> dict[str, Any]:
        declaration: Any = self._config.taxonomy.discovery_initial_taxonomy
        if not isinstance(declaration, dict):
            raise ValueError(
                "Fixed-taxonomy application requires a complete explicit "
                "TaxonomyVersion object, not published category names alone."
            )
        taxonomy = TaxonomyVersion.model_validate(declaration)
        if not taxonomy.frozen:
            raise ValueError("Fixed-taxonomy application requires a frozen taxonomy.")
        self._store.create_record(
            "taxonomy", taxonomy.taxonomy_id, taxonomy.model_dump(mode="json")
        )
        freeze_hash: str = self._store.freeze_records(
            "taxonomy", [taxonomy.taxonomy_id]
        )
        self._current_taxonomy = taxonomy
        result: dict[str, Any] = {
            "status": StageStatus.COMPLETED.value,
            "execution_mode": "fixed_taxonomy_application",
            "taxonomy_id": taxonomy.taxonomy_id,
            "iteration": taxonomy.iteration,
            "taxonomy_freeze_manifest": freeze_hash,
            "discovery_claim": False,
            "human_discovery_initialization_performed": False,
            "classification_delegated_to": "Categorizer",
            "pending_annotation_task_ids": [],
        }
        manifest: dict[str, Any] = self._event(
            "fixed_taxonomy_ready", {"result": result}
        )
        result["discovery_manifest_id"] = manifest["manifest_id"]
        self._checkpoint(
            StageStatus.COMPLETED, [], result, output_id=manifest["manifest_id"]
        )
        return result

    def run(self, failure_ids: list[str]) -> dict[str, Any]:
        """Advance discovery until completion, human waiting, or a protocol blocker."""
        self._config.validate("discover_taxonomy")
        self._started_at = _now()
        self._establish_population(failure_ids)

        previous: list[StageState] = [
            StageState.model_validate(item)
            for item in self._store.list_records("stage_state")
            if item.get("stage") == "discover_taxonomy"
            and item.get("execution_id") == self._execution_id
            and item.get("status") == StageStatus.COMPLETED.value
        ]
        if previous:
            return copy.deepcopy(previous[0].cursor["result"])

        try:
            if self._config.operational.taxonomy_mode == "fixed":
                return self._fixed_application()
            if self._config.operational.taxonomy_mode != "discovery":
                raise ValueError("Select discovery or fixed taxonomy mode explicitly.")
            if self._config.taxonomy.discovery_initial_taxonomy is not None:
                raise ValueError("Discovery cannot initialize from a supplied taxonomy.")
            self._settings()

            if self._initialization == "model":
                from model_initialization import initialize
                taxonomy = initialize(self)
            else:
                tasks: list[AnnotationTask] = self.request_initialization(self._population)
                initial_references: list[dict[str, Any]] = self._references(
                    tasks, _INITIALIZATION_PURPOSE
                )
                initial_ids: list[str] = sorted(
                    {
                        identity
                        for reference in initial_references
                        for identity in reference["annotation_ids"]
                    }
                )
                taxonomy: TaxonomyVersion = self.create_initial_taxonomy(initial_ids)
            self._current_taxonomy = taxonomy

            # Every non-converged automatic transition adds at least one
            # human-supported category. A completed no-addition cycle that fails
            # coverage blocks explicitly instead of repeating indefinitely.
            while True:
                self._current_taxonomy = taxonomy
                assignments: list[LabelAssignment] = self.classify_remaining(
                    self._population, taxonomy
                )
                self._current_assignments = assignments
                review_tasks: list[AnnotationTask] = self.request_other_reviews(
                    assignments
                )
                references: list[dict[str, Any]] = (
                    self._references(review_tasks, _REFINEMENT_PURPOSE, taxonomy=taxonomy)
                    if review_tasks else []
                )
                review_ids: list[str] = sorted(
                    {
                        identity
                        for reference in references
                        for identity in reference["annotation_ids"]
                    }
                )
                refined: TaxonomyVersion = self.refine_taxonomy(
                    taxonomy, review_ids
                )
                additions: list[str] = sorted(
                    {
                        item.subtype_id for item in refined.subtype_definitions
                    }
                    - {
                        item.subtype_id for item in taxonomy.subtype_definitions
                    }
                )
                self._history.append(
                    {
                        "parent_taxonomy_id": taxonomy.taxonomy_id,
                        "taxonomy_id": refined.taxonomy_id,
                        "iteration": taxonomy.iteration,
                        "new_subtype_ids": additions,
                        "review_annotation_ids": review_ids,
                        "coverage": self._coverage(taxonomy, assignments),
                    }
                )
                if refined.taxonomy_id != taxonomy.taxonomy_id:
                    self._checkpoint(
                        StageStatus.RUNNING,
                        [],
                        self._result(StageStatus.RUNNING),
                    )
                    taxonomy = refined
                    self._current_assignments = []
                    continue

                if not self.check_convergence(taxonomy, assignments):
                    raise _ProtocolBlocked(
                        "Human refinement is complete and found no new categories, "
                        "but the declared coverage condition is not satisfied. "
                        "Persistent other/uncertain cases remain visible; no "
                        "additional inference or forced convergence was performed."
                    )
                convergence_coverage: dict[str, Any] = self._coverage(
                    taxonomy, assignments
                )
                final: TaxonomyVersion = self.freeze_taxonomy(taxonomy.taxonomy_id)
                self._current_taxonomy = final
                result: dict[str, Any] = self._result(StageStatus.COMPLETED)
                result.update(
                    coverage=convergence_coverage,
                    converged_source_taxonomy_id=taxonomy.taxonomy_id,
                    taxonomy_id=final.taxonomy_id,
                    iteration=final.iteration,
                    frozen=True,
                    execution_mode="discovery",
                    unresolved_response_ids=sorted(
                        set(convergence_coverage["other_response_ids"])
                        | set(convergence_coverage["uncertain_response_ids"])
                    ),
                )
                manifest: dict[str, Any] = self._event(
                    "taxonomy_discovery_complete", {"result": result}
                )
                result["discovery_manifest_id"] = manifest["manifest_id"]
                self._checkpoint(
                    StageStatus.COMPLETED,
                    [],
                    result,
                    output_id=manifest["manifest_id"],
                )
                return result

        except _PendingAnnotations as exc:
            result = self._result(
                StageStatus.WAITING_FOR_ANNOTATIONS,
                pending=exc.task_ids,
            )
            self._checkpoint(
                StageStatus.WAITING_FOR_ANNOTATIONS, exc.task_ids, result
            )
            return result
        except _ProtocolBlocked as exc:
            result = self._result(StageStatus.BLOCKED, reasons=[str(exc)])
            event = self._event("taxonomy_discovery_blocked", {"result": result})
            result["discovery_manifest_id"] = event["manifest_id"]
            self._checkpoint(
                StageStatus.BLOCKED, [], result, output_id=event["manifest_id"]
            )
            return result
        except Exception as exc:
            result = self._result(
                StageStatus.BLOCKED,
                reasons=[
                    "Discovery stopped without scientific fallback: "
                    f"{type(exc).__name__}."
                ],
            )
            self._checkpoint(StageStatus.BLOCKED, [], result)
            raise
