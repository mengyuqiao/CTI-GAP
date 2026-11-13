"""Human annotation, independent audit sampling, and a thin Streamlit interface.

This module never executes models or computes agreement statistics. Configuration
is supplied through the ``protocol`` and ``sampling`` arguments because the
designed constructor accepts only an ArtifactStore.

Protocol contract
=================

create_tasks() requires an explicit protocol mapping containing:

    protocol_version: str
    config_hash: SHA-256
    guideline_hash: SHA-256
    guideline_content: JSON object whose canonical hash is guideline_hash
    judgment: "identification" | "labels" | "discovery"
    annotator_ids: list[str]
    annotators_per_instance: int
    assignments: {response_id: [annotator_id, ...]}
    queue_order_policy: "response_id"
    reference_establishment: "independent_agreement_or_adjudication"
    cohort_id: str | null
    taxonomy_id: str | null
    correctness_criteria: JSON object
    reference_applicability: {response_id: JSON object}
    adjudication:
        adjudicator_id: str
        senior_adjudicator_id: str
        escalate_unresolved_cases_to_senior: true
        adjudicator_must_be_independent: bool

Label judgments require a frozen taxonomy. Discovery may optionally supply a
current, non-frozen taxonomy; published categories are never automatically loaded.
Discovery proposals use SubtypeDefinition and SubtypeRules from schemas.py.

Agreement-study protocols additionally require:
    agreement_report_id: str | null

The report may be supplied later by creating adjudications with a protocol whose
report already exists, or by recording the report under the cohort's protocol
before task creation. An unresolved report blocks adjudication, not annotation.
The report is checked in storage rather than accepted as a boolean assertion.

Sampling contract
=================

sample_audit() requires:
    cohort_id, config_hash, guideline_hash, seed, sample_size,
    sampling_unit="response", sampling_method,
    eligible_population_version, eligible_response_ids,
    frozen_prediction_version, frozen_prediction_ids,
    exclusion_categories, excluded_response_ids, annotation_protocol

frozen_prediction_ids maps every eligible response to one stored immutable
prediction record ID. Additional fields depend on the cohort:

* Identification: sampling_method="simple_random_without_replacement";
  duplicate_key_by_response and duplicate_representatives are required.
* Agreement: sampling_method="stratified_without_replacement";
  stratum_by_response and allocation_by_stratum are required. Each response has
  exactly one explicitly assigned stratum; overlapping-category allocation must
  therefore be resolved before calling this module.
* Expanded label audit: original_cohort_id, additional_sampling_protocol, and
  stable_consensus_response_ids are required.
* Stability: an explicit 180-response population or stratified allocation.

Sample sizes come from the paper: 100, 320, 800 (320+480), and 180 respectively.
They are never reduced when eligible populations are too small.

Security boundary
=================

Service identity arguments are authenticated principal IDs supplied by a trusted
caller. They are not passwords or self-asserted UI names. The service enforces
assignment ownership and adjudicator authorization, but cannot authenticate an
arbitrary caller through an API that receives only a string.

The UI requires Streamlit OIDC authentication and an explicit environment-supplied
subject-to-annotator mapping. It never offers an annotator-name selector.

Atomicity boundary
==================

ArtifactStore exposes no multi-operation transaction or task-claim operation.
Tasks are preassigned; retrieval never claims or reassigns work. Immutable,
deterministically identified submissions are authoritative. Task status is a
recoverable projection updated after submission. A concurrent conflicting
submission fails through create_record uniqueness. Partial task creation is
resumable by repeating the identical request.
"""

from __future__ import annotations

import copy
import json
import os
import random
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from generation_context import review_task_input
from schemas import (
    AnnotationPurpose,
    AnnotationStatus,
    AnnotationTask,
    AssignmentOrigin,
    AuditCohort,
    EvidenceSpan,
    FailureDecision,
    FailureStatus,
    LabelAssignment,
    ModelResponse,
    ResponseStatus,
    SubtypeDefinition,
    SubtypeRules,
    TaskInstance,
    TaxonomyVersion,
    _canonical_json,
    _content_hash,
    _identifier,
)
from storage import ArtifactStore


_IDENTIFICATION_PURPOSES: frozenset[AnnotationPurpose] = frozenset(
    {
        AnnotationPurpose.IDENTIFICATION_ANCHOR,
        AnnotationPurpose.IDENTIFICATION_BOUNDARY,
        AnnotationPurpose.INDEPENDENT_IDENTIFICATION_AUDIT,
    }
)
_DISCOVERY_PURPOSES: frozenset[AnnotationPurpose] = frozenset(
    {
        AnnotationPurpose.TAXONOMY_INITIALIZATION,
        AnnotationPurpose.TAXONOMY_REFINEMENT,
    }
)
_AUDIT_PURPOSES: frozenset[AnnotationPurpose] = frozenset(
    {
        AnnotationPurpose.HUMAN_AGREEMENT,
        AnnotationPurpose.INDEPENDENT_IDENTIFICATION_AUDIT,
        AnnotationPurpose.INDEPENDENT_LABEL_AUDIT,
        AnnotationPurpose.EXPANDED_LABEL_AUDIT,
        AnnotationPurpose.AUDIT_ADJUDICATION,
    }
)
_RESOLUTION_PURPOSES: frozenset[AnnotationPurpose] = frozenset(
    {
        AnnotationPurpose.OPERATIONAL_ADJUDICATION,
        AnnotationPurpose.AUDIT_ADJUDICATION,
        AnnotationPurpose.SENIOR_ESCALATION,
    }
)
_ASSIGNMENT_KINDS: tuple[str, ...] = (
    "label_assignment", "assignment", "consensus", "final_assignment"
)
_TAXONOMY_KINDS: tuple[str, ...] = ("taxonomy", "taxonomy_version")
_AUTOMATED_ORIGINS: frozenset[AssignmentOrigin] = frozenset(
    {AssignmentOrigin.AGENT, AssignmentOrigin.CONSENSUS, AssignmentOrigin.DISCOVERY}
)
_BLIND_KEYS: frozenset[str] = frozenset(
    {
        "identification_score", "identification_scores", "similarity_score",
        "automatic_assignment", "automatic_assignments", "automatic_decision",
        "automatic_predictions", "agent_predictions", "peer_predictions",
        "peer_labels", "other_annotator_decisions", "independent_submissions",
        "adjudicated_labels", "final_assignment", "final_labels",
        "stratum_id", "stratum_ranges", "decision_thresholds",
        "stable_consensus", "review_trigger",
    }
)
_PROTOCOL_KEYS: frozenset[str] = frozenset(
    {
        "protocol_version", "config_hash", "guideline_hash", "guideline_content",
        "judgment", "annotator_ids", "annotators_per_instance", "assignments",
        "queue_order_policy", "reference_establishment", "cohort_id",
        "taxonomy_id", "correctness_criteria", "reference_applicability",
        "adjudication", "agreement_report_id",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an explicit mapping.")
    return copy.deepcopy(value)


def _ids(value: Any, name: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an explicit list.")
    result: list[str] = [_identifier(item) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate identities.")
    if nonempty and not result:
        raise ValueError(f"{name} must not be empty.")
    return result


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ValueError(f"Required setting {key!r} is unresolved.")
    return mapping[key]


def _blind_check(value: Any) -> None:
    """Reject internal classification fields without scanning source prose."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _BLIND_KEYS:
                raise ValueError(f"Internal field {key!r} cannot enter blinded materials.")
            _blind_check(item)
    elif isinstance(value, list):
        for item in value:
            _blind_check(item)


def _decision_key(result: Mapping[str, Any], judgment: str) -> str:
    if judgment == "identification":
        value: Any = result.get("decision")
    elif judgment == "labels":
        value = sorted(result.get("labels", []))
    else:
        value = result.get("taxonomy_proposal")
    return _canonical_json(
        {"result": value, "uncertain": result.get("uncertain", False)}
    )


class AnnotationService:
    """Manage immutable human judgments and independently sampled audits."""

    def __init__(self, store: ArtifactStore) -> None:
        self._store: ArtifactStore = store

    def _find(self, kinds: tuple[str, ...], record_id: str) -> tuple[str, dict[str, Any]]:
        found: list[tuple[str, dict[str, Any]]] = []
        for kind in kinds:
            try:
                found.append((kind, self._store.read_record(kind, record_id)))
            except KeyError:
                pass
        if len(found) != 1:
            raise ValueError(f"Missing or ambiguous stored identity: {record_id}.")
        return found[0]

    def _taxonomy(self, taxonomy_id: str) -> TaxonomyVersion:
        _, payload = self._find(_TAXONOMY_KINDS, taxonomy_id)
        return TaxonomyVersion.model_validate(payload)

    def _immutable(self, kind: str, record_id: str) -> None:
        records: list[dict[str, Any]] = self._store.list_records(
            kind, {"record_id": record_id, "lifecycle": ["immutable", "frozen"]}
        )
        if len(records) != 1:
            raise ValueError(f"{kind}/{record_id} must be immutable before use.")

    def _protocol(self, protocol: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = _mapping(protocol, "Annotation protocol")
        if set(result) - _PROTOCOL_KEYS:
            raise ValueError("Unknown annotation-protocol fields.")
        _identifier(_required(result, "protocol_version"))
        _digest(_required(result, "config_hash"), "config_hash")
        guideline_hash: str = _digest(
            _required(result, "guideline_hash"), "guideline_hash"
        )
        guideline: dict[str, Any] = _mapping(
            _required(result, "guideline_content"), "guideline_content"
        )
        if not guideline or _content_hash(guideline) != guideline_hash:
            raise ValueError("Guideline content does not match its frozen hash.")
        judgment: Any = _required(result, "judgment")
        if judgment not in {"identification", "labels", "discovery"}:
            raise ValueError("Unknown human-judgment type.")
        pool: list[str] = _ids(_required(result, "annotator_ids"), "annotator_ids")
        count: int = _integer(
            _required(result, "annotators_per_instance"), "annotators_per_instance"
        )
        if count > len(pool):
            raise ValueError("Required raters exceed the authorized pool.")
        assignments: dict[str, Any] = _mapping(
            _required(result, "assignments"), "assignments"
        )
        for response_id, actors in assignments.items():
            _identifier(response_id)
            selected: list[str] = _ids(actors, "Assigned raters")
            if len(selected) != count or not set(selected).issubset(pool):
                raise ValueError("Every assignment needs the required distinct authorized raters.")
        if _required(result, "queue_order_policy") != "response_id":
            raise ValueError("Only explicitly selected response_id queue order is supported.")
        if _required(result, "reference_establishment") != (
            "independent_agreement_or_adjudication"
        ):
            raise ValueError("Unsupported human-reference establishment protocol.")
        if "cohort_id" not in result or "taxonomy_id" not in result:
            raise ValueError("Explicit cohort_id and taxonomy_id values, including null, are required.")
        if result["cohort_id"] is not None:
            _identifier(result["cohort_id"])
        if result["taxonomy_id"] is not None:
            taxonomy: TaxonomyVersion = self._taxonomy(result["taxonomy_id"])
            if taxonomy.guideline_hash != guideline_hash:
                raise ValueError("Taxonomy and annotation guideline hashes differ.")
            if judgment == "labels" and not taxonomy.frozen:
                raise ValueError("Application judgments require a frozen taxonomy.")
        elif judgment == "labels":
            raise ValueError("Label annotation requires an explicit taxonomy.")
        criteria: dict[str, Any] = _mapping(
            _required(result, "correctness_criteria"), "correctness_criteria"
        )
        if judgment == "identification" and not criteria:
            raise ValueError("Identification requires fixed correctness criteria.")
        _mapping(_required(result, "reference_applicability"), "reference_applicability")
        adjudication: dict[str, Any] = _mapping(
            _required(result, "adjudication"), "adjudication"
        )
        required_keys: set[str] = {
            "adjudicator_id", "senior_adjudicator_id",
            "escalate_unresolved_cases_to_senior",
            "adjudicator_must_be_independent",
        }
        if set(adjudication) != required_keys:
            raise ValueError("Adjudication authorization and escalation must be explicit.")
        _identifier(adjudication["adjudicator_id"])
        _identifier(adjudication["senior_adjudicator_id"])
        if adjudication["escalate_unresolved_cases_to_senior"] is not True:
            raise ValueError("Unresolved adjudications must escalate to the named senior.")
        if not isinstance(adjudication["adjudicator_must_be_independent"], bool):
            raise ValueError("Adjudicator independence policy must be boolean.")
        _canonical_json(result)
        return result

    def _context_id(self, task_id: str) -> str:
        return _content_hash({"annotation_task_context": task_id})

    def _context(self, task_id: str) -> dict[str, Any]:
        return self._store.read_record(
            "experiment_manifest", self._context_id(task_id)
        )

    def _save_context(self, task_id: str, context: dict[str, Any]) -> None:
        identity: str = self._context_id(task_id)
        self._store.create_record(
            "experiment_manifest",
            identity,
            {
                **context,
                "manifest_id": identity,
                "purpose": "annotation_task_context",
                "annotation_task_id": task_id,
            },
        )

    def _materials(
        self, response_id: str, protocol: dict[str, Any]
    ) -> dict[str, Any]:
        response = ModelResponse.model_validate(
            self._store.read_record("model_response", response_id)
        )
        instance = TaskInstance.model_validate(
            self._store.read_record("task_instance", response.instance_id)
        )
        if response.status == ResponseStatus.TRANSPORT_FAILURE:
            raise ValueError("Transport failures require operational handling, not annotation.")
        reference: dict[str, Any] = instance.reference
        if not reference.get("answer") or not reference.get("evidence_records"):
            raise ValueError("Human annotation requires reference answers and evidence.")
        applicability: Any = protocol["reference_applicability"].get(
            response_id, protocol["reference_applicability"].get("*")
        )
        if not isinstance(applicability, dict) or not applicability:
            raise ValueError("Task-specific reference applicability is unresolved.")
        document_fields: tuple[str, ...] = (
            "evidence_id", "document_id", "source_id", "source_version", "snippet",
            "publication_time", "modification_time", "claim_valid_from",
            "claim_valid_until", "role",
        )

        def project(record: Mapping[str, Any]) -> dict[str, Any]:
            return {key: record[key] for key in document_fields if key in record}

        materials: dict[str, Any] = {
            "task_input": review_task_input(self._store, response, instance),
            "evaluated_response": {
                "response_id": response.response_id,
                "raw_text": response.raw_text,
                "parsed_answer": response.parsed_answer,
                "status": response.status.value,
                "schema_errors": response.schema_errors,
            },
            "reference_material": {
                "version": reference.get("version"),
                "answer": reference["answer"],
                "evidence_records": [
                    project(item) for item in reference["evidence_records"]
                ],
                "applicability": applicability,
            },
            "guideline": protocol["guideline_content"],
            "correctness_criteria": protocol["correctness_criteria"],
            "judgment": protocol["judgment"],
        }
        if protocol["taxonomy_id"] is not None:
            taxonomy = self._taxonomy(protocol["taxonomy_id"])
            materials["taxonomy"] = {
                "taxonomy_id": taxonomy.taxonomy_id,
                "guideline_hash": taxonomy.guideline_hash,
                "frozen": taxonomy.frozen,
                "subtype_definitions": [
                    {
                        "subtype_id": item.subtype_id,
                        "name": item.name,
                        "definition": item.definition,
                        "category": item.category,
                        "evidence_requirements": item.evidence_requirements,
                    }
                    for item in taxonomy.subtype_definitions
                ],
                "inclusion_exclusion_rules": {
                    key: item.model_dump(mode="json")
                    for key, item in taxonomy.inclusion_exclusion_rules.items()
                },
            }
        _blind_check(materials)
        return json.loads(_canonical_json(materials))

    def _submission_id(self, task_id: str) -> str:
        return _content_hash({"independent_annotation": task_id})

    def _resolution_id(self, task_id: str) -> str:
        return _content_hash({"human_resolution": task_id})

    def _exists(self, kind: str, record_id: str) -> dict[str, Any] | None:
        try:
            return self._store.read_record(kind, record_id)
        except KeyError:
            return None

    def create_tasks(
        self,
        response_ids: list[str],
        purpose: str,
        protocol: dict[str, Any],
    ) -> list[AnnotationTask]:
        """Create deterministic preassigned tasks without implicit randomization."""
        identities: list[str] = _ids(response_ids, "response_ids")
        selected_purpose = AnnotationPurpose(purpose)
        if selected_purpose in {
            AnnotationPurpose.AUDIT_ADJUDICATION,
            AnnotationPurpose.SENIOR_ESCALATION,
        }:
            raise ValueError("Use create_adjudications/resolve_task for derived review tasks.")
        settings: dict[str, Any] = self._protocol(protocol)
        if (
            not set(identities).issubset(settings["assignments"])
            and "*" not in settings["assignments"]
        ):
            raise ValueError("Every response requires an explicit assignment schedule.")
        if selected_purpose in _IDENTIFICATION_PURPOSES:
            if settings["judgment"] != "identification":
                raise ValueError("Identification purposes require identification judgments.")
        elif selected_purpose in _DISCOVERY_PURPOSES:
            if settings["judgment"] != "discovery":
                raise ValueError("Discovery purposes require discovery judgments.")
        elif selected_purpose != AnnotationPurpose.CALIBRATION:
            if settings["judgment"] != "labels":
                raise ValueError("This annotation purpose requires subtype labels.")
        expected_raters: int | None = {
            AnnotationPurpose.HUMAN_AGREEMENT: 3,
            AnnotationPurpose.INDEPENDENT_IDENTIFICATION_AUDIT: 2,
        }.get(selected_purpose)
        if expected_raters is not None and settings["annotators_per_instance"] != expected_raters:
            raise ValueError("Rater count differs from the paper's annotation protocol.")
        cohort_id: str | None = settings["cohort_id"]
        if selected_purpose in _AUDIT_PURPOSES:
            if cohort_id is None:
                raise ValueError("Audit work requires a locked cohort.")
            cohort = AuditCohort.model_validate(
                self._store.read_record("audit_cohort", cohort_id)
            )
            if not cohort.locked or not set(identities).issubset(cohort.response_ids):
                raise ValueError("Annotation responses are outside locked audit membership.")
            declared: Any = cohort.sampling_protocol["annotation_protocol"]
            if _canonical_json(declared) != _canonical_json(settings):
                raise ValueError("Task protocol differs from the frozen cohort protocol.")
        protocol_hash: str = _content_hash(settings)
        prepared: list[tuple[AnnotationTask, dict[str, Any]]] = []
        for response_id in sorted(identities):
            materials: dict[str, Any] = self._materials(response_id, settings)
            actors = settings["assignments"].get(
                response_id, settings["assignments"].get("*")
            )
            for slot, actor in enumerate(actors):
                identity: str = _content_hash(
                    {
                        "response_id": response_id,
                        "purpose": selected_purpose.value,
                        "cohort_id": cohort_id,
                        "protocol_hash": protocol_hash,
                        "slot": slot,
                        "assigned_annotator": actor,
                    }
                )
                task = AnnotationTask(
                    annotation_task_id=identity,
                    response_id=response_id,
                    purpose=selected_purpose,
                    cohort_id=cohort_id,
                    assigned_annotator=actor,
                    guideline_hash=settings["guideline_hash"],
                    visible_payload=materials,
                )
                context: dict[str, Any] = {
                    "protocol": settings,
                    "protocol_hash": protocol_hash,
                    "config_hash": settings["config_hash"],
                    "slot": slot,
                    "reference_purpose": selected_purpose.value,
                    "parent_task_ids": [],
                    "audit_only": selected_purpose in _AUDIT_PURPOSES,
                }
                prepared.append((task, context))
        result: list[AnnotationTask] = []
        for task, context in prepared:
            existing: dict[str, Any] | None = self._exists(
                "annotation_task", task.annotation_task_id
            )
            if existing is not None:
                stored = AnnotationTask.model_validate(existing)
                comparable: dict[str, Any] = stored.model_dump(mode="json")
                comparable["status"] = task.status.value
                if _canonical_json(comparable) != _canonical_json(task):
                    raise ValueError("Existing task identity has incompatible content.")
                task = stored
            self._save_context(task.annotation_task_id, context)
            self._store.create_record(
                "annotation_task", task.annotation_task_id,
                task.model_dump(mode="json"),
            )
            result.append(task)
        return result

    def get_next_task(self, annotator_id: str) -> AnnotationTask:
        """Return an existing authorized task; never claim another user's work."""
        actor: str = _identifier(annotator_id)
        tasks: list[AnnotationTask] = [
            AnnotationTask.model_validate(payload)
            for payload in self._store.list_records(
                "annotation_task", {"actor_id": None}
            )
            if payload.get("assigned_annotator") == actor
        ]
        available: list[AnnotationTask] = []
        for task in tasks:
            if task.status not in {
                AnnotationStatus.PENDING,
                AnnotationStatus.IN_PROGRESS,
                AnnotationStatus.DRAFT,
            }:
                continue
            if self._exists("annotation", self._submission_id(task.annotation_task_id)):
                continue
            if self._exists("adjudication", self._resolution_id(task.annotation_task_id)):
                continue
            context: dict[str, Any] = self._context(task.annotation_task_id)
            if context["protocol"]["queue_order_policy"] != "response_id":
                raise ValueError("Unsupported queue-order policy.")
            available.append(task)
        if not available:
            raise LookupError("No pending tasks are assigned to this authenticated annotator.")
        return sorted(
            available, key=lambda item: (item.response_id, item.annotation_task_id)
        )[0]

    def _source_texts(self, visible: dict[str, Any]) -> dict[str, str]:
        texts: dict[str, str] = {}
        ambiguous: set[str] = set()
        response: dict[str, Any] = visible["evaluated_response"]
        texts[response["response_id"]] = response["raw_text"]
        records: list[dict[str, Any]] = [
            *visible["task_input"]["source_documents"],
            *visible["reference_material"]["evidence_records"],
        ]
        for record in records:
            for field in ("evidence_id", "document_id"):
                identity: Any = record.get(field)
                snippet: Any = record.get("snippet")
                if isinstance(identity, str) and isinstance(snippet, str):
                    if identity in ambiguous:
                        continue
                    if identity in texts and texts[identity] != snippet:
                        # Ambiguous document IDs must be cited by evidence_id.
                        texts.pop(identity)
                        ambiguous.add(identity)
                    else:
                        texts[identity] = snippet
        return texts

    def _result(
        self,
        task: AnnotationTask,
        context: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        settings: dict[str, Any] = context["protocol"]
        supplied: dict[str, Any] = _mapping(payload, "Human annotation")
        allowed: set[str] = {
            "decision", "labels", "taxonomy_proposal", "uncertain",
            "evidence_spans", "notes", "guideline_hash", "taxonomy_id",
        }
        if set(supplied) - allowed:
            raise ValueError("Unknown human-annotation fields.")
        if not isinstance(supplied.get("uncertain"), bool):
            raise ValueError("An explicit uncertainty boolean is required.")
        if supplied.get("guideline_hash", task.guideline_hash) != task.guideline_hash:
            raise ValueError("Submission guideline differs from the task.")
        if supplied.get("taxonomy_id", settings["taxonomy_id"]) != settings["taxonomy_id"]:
            raise ValueError("Submission taxonomy differs from the task.")
        spans_raw: Any = supplied.get("evidence_spans")
        if not isinstance(spans_raw, list):
            raise ValueError("Evidence spans must be explicitly supplied.")
        spans: list[EvidenceSpan] = [
            EvidenceSpan.model_validate(item) for item in spans_raw
        ]
        source_texts: dict[str, str] = self._source_texts(task.visible_payload)
        for span in spans:
            if span.source_ref not in source_texts:
                raise ValueError("Evidence span cites an unavailable or ambiguous source.")
            text: str = source_texts[span.source_ref]
            if span.quote is not None and span.quote not in text:
                raise ValueError("Evidence quotation is not verbatim in the cited source.")
            if span.start_offset is not None and span.end_offset is not None:
                if span.end_offset > len(text):
                    raise ValueError("Evidence offsets exceed the cited source.")
                if span.quote is not None and text[
                    span.start_offset:span.end_offset
                ] != span.quote:
                    raise ValueError("Evidence offsets and quotation disagree.")
        result: dict[str, Any] = {
            "uncertain": supplied["uncertain"],
            "evidence_spans": [item.model_dump(mode="json") for item in spans],
            "notes": supplied.get("notes", ""),
        }
        if not isinstance(result["notes"], str):
            raise ValueError("Human notes must be a string.")
        judgment: str = settings["judgment"]
        if judgment == "identification":
            decision: Any = supplied.get("decision")
            if decision not in {"pass", "failure", "unresolved"}:
                raise ValueError("Identification decision must be pass, failure, or unresolved.")
            if decision == "unresolved" and not result["uncertain"]:
                raise ValueError("Unresolved judgments must retain uncertainty.")
            result["decision"] = decision
            if "labels" in supplied or "taxonomy_proposal" in supplied:
                raise ValueError("Identification submissions cannot contain subtype decisions.")
        elif judgment == "labels":
            labels: list[str] = _ids(
                supplied.get("labels"), "Human labels", nonempty=False
            )
            if not labels and not result["uncertain"]:
                raise ValueError("An empty subtype assignment must be explicitly uncertain.")
            taxonomy: TaxonomyVersion = self._taxonomy(settings["taxonomy_id"])
            assignment = LabelAssignment.model_validate(
                {
                    "assignment_id": task.annotation_task_id,
                    "response_id": task.response_id,
                    "taxonomy_id": taxonomy.taxonomy_id,
                    "actor_id": task.assigned_annotator,
                    "round_index": 0,
                    "origin": "independent_annotation",
                    "labels": labels,
                    "evidence_spans": spans,
                    "uncertain": result["uncertain"],
                },
                context={
                    "taxonomy": taxonomy,
                    "require_taxonomy": True,
                    "require_frozen_taxonomy": True,
                    "require_evidence_for_each_label": True,
                },
            )
            result["labels"] = assignment.labels
            if "decision" in supplied or "taxonomy_proposal" in supplied:
                raise ValueError("Frozen-taxonomy submissions cannot add categories.")
        else:
            proposal: dict[str, Any] = _mapping(
                supplied.get("taxonomy_proposal"), "taxonomy_proposal"
            )
            if set(proposal) != {"new_subtypes", "rules", "no_new_categories"}:
                raise ValueError("Discovery proposals require new_subtypes, rules, no_new_categories.")
            definitions: Any = proposal["new_subtypes"]
            if not isinstance(definitions, list):
                raise ValueError("new_subtypes must be a list.")
            parsed = [SubtypeDefinition.model_validate(item) for item in definitions]
            subtype_ids: list[str] = [item.subtype_id for item in parsed]
            if len(subtype_ids) != len(set(subtype_ids)):
                raise ValueError("Discovery proposal contains duplicate subtype IDs.")
            rules: dict[str, Any] = _mapping(proposal["rules"], "Discovery rules")
            if set(rules) != set(subtype_ids):
                raise ValueError("Each proposed subtype needs its own annotation rules.")
            for value in rules.values():
                SubtypeRules.model_validate(value)
            if not isinstance(proposal["no_new_categories"], bool):
                raise ValueError("no_new_categories must be boolean.")
            if proposal["no_new_categories"] != (not parsed):
                raise ValueError("Discovery convergence statement contradicts its additions.")
            if parsed and not spans:
                raise ValueError("New categories require observable supporting discrepancies.")
            result["taxonomy_proposal"] = proposal
            if "decision" in supplied or "labels" in supplied:
                raise ValueError("Discovery proposals are not fixed-taxonomy assignments.")
        if not result["uncertain"] and not spans:
            raise ValueError("Final human judgments require supporting evidence.")
        return result

    def _finish_task(self, task: AnnotationTask, status: AnnotationStatus) -> None:
        current = AnnotationTask.model_validate(
            self._store.read_record("annotation_task", task.annotation_task_id)
        )
        if current.status == status:
            return
        if current.status not in {
            AnnotationStatus.DRAFT, AnnotationStatus.PENDING, AnnotationStatus.IN_PROGRESS
        }:
            raise ValueError("Submitted annotation task cannot change lifecycle.")
        self._store.update_draft(
            "annotation_task", task.annotation_task_id, {"status": status.value}
        )

    def submit_annotation(
        self, task_id: str, annotator_id: str, payload: dict[str, Any]
    ) -> None:
        """Append an independent submission; identical retries are idempotent."""
        task = AnnotationTask.model_validate(
            self._store.read_record("annotation_task", _identifier(task_id))
        )
        actor: str = _identifier(annotator_id)
        if task.assigned_annotator != actor:
            raise PermissionError("This annotation task belongs to another annotator.")
        if task.purpose in _RESOLUTION_PURPOSES:
            raise ValueError("Adjudication tasks must be submitted through resolve_task.")
        context: dict[str, Any] = self._context(task_id)
        result: dict[str, Any] = self._result(task, context, payload)
        identity: str = self._submission_id(task_id)
        existing: dict[str, Any] | None = self._exists("annotation", identity)
        if existing is not None:
            if existing["result_hash"] != _content_hash(result):
                raise ValueError("A conflicting immutable submission already exists.")
            self._finish_task(task, AnnotationStatus.SUBMITTED)
            return
        if task.status not in {
            AnnotationStatus.DRAFT, AnnotationStatus.PENDING, AnnotationStatus.IN_PROGRESS
        }:
            raise ValueError("Task is not open for independent submission.")
        for item in self._store.list_records("annotation"):
            if (
                item.get("response_id") == task.response_id
                and item.get("purpose") == task.purpose.value
                and item.get("cohort_id") == task.cohort_id
                and item.get("protocol_hash") == context["protocol_hash"]
                and item.get("annotator_id") == actor
            ):
                raise ValueError("This rater already filled an independent slot.")
        record: dict[str, Any] = {
            "annotation_id": identity,
            "annotation_task_id": task_id,
            "response_id": task.response_id,
            "purpose": task.purpose.value,
            "cohort_id": task.cohort_id,
            "annotator_id": actor,
            "guideline_hash": task.guideline_hash,
            "taxonomy_id": context["protocol"]["taxonomy_id"],
            "config_hash": context["config_hash"],
            "protocol_hash": context["protocol_hash"],
            "submitted_at": _now(),
            "status": "submitted",
            "audit_only": context["audit_only"],
            "result_hash": _content_hash(result),
            **result,
        }
        self._store.create_record("annotation", identity, record)
        self._store.freeze_records("annotation", [identity])
        self._finish_task(task, AnnotationStatus.SUBMITTED)

    def _group_tasks(self, task: AnnotationTask, context: dict[str, Any]) -> list[AnnotationTask]:
        result: list[AnnotationTask] = []
        for payload in self._store.list_records(
            "annotation_task", {"response_id": task.response_id}
        ):
            other = AnnotationTask.model_validate(payload)
            if other.purpose != task.purpose or other.cohort_id != task.cohort_id:
                continue
            other_context: dict[str, Any] = self._context(other.annotation_task_id)
            if other_context["protocol_hash"] == context["protocol_hash"]:
                result.append(other)
        count: int = context["protocol"]["annotators_per_instance"]
        if len(result) != count or len({item.assigned_annotator for item in result}) != count:
            raise ValueError("Independent task group is incomplete or has repeated raters.")
        return sorted(result, key=lambda item: item.annotation_task_id)

    def _independent_records(
        self, tasks: list[AnnotationTask]
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for task in tasks:
            item: dict[str, Any] | None = self._exists(
                "annotation", self._submission_id(task.annotation_task_id)
            )
            if item is None:
                raise ValueError("All independent raters must submit before adjudication.")
            if item["annotator_id"] != task.assigned_annotator:
                raise ValueError("Independent submission ownership is inconsistent.")
            records.append(item)
        return records

    def _agreement_prerequisite(self, task: AnnotationTask, context: dict[str, Any]) -> None:
        if task.purpose != AnnotationPurpose.HUMAN_AGREEMENT:
            return
        reports: list[dict[str, Any]] = [
            report for report in self._store.list_records("report")
            if report.get("purpose") == "human_agreement"
            and report.get("cohort_id") == task.cohort_id
            and report.get("pre_adjudication") is True
            and report.get("status") == "complete"
            and report.get("guideline_hash") == task.guideline_hash
        ]
        requested: Any = context["protocol"].get("agreement_report_id")
        if requested is not None:
            reports = [item for item in reports if item["report_id"] == requested]
        if not reports:
            raise ValueError("Record complete pre-adjudication human agreement before adjudicating.")

    def _derived_task(
        self,
        parent: AnnotationTask,
        context: dict[str, Any],
        purpose: AnnotationPurpose,
        actor: str,
        parents: list[str],
        records: list[dict[str, Any]],
        *,
        escalation: bool = False,
    ) -> AnnotationTask:
        identity: str = _content_hash(
            {
                "derived_review": purpose.value,
                "parent_task_ids": sorted(parents),
                "assigned_annotator": actor,
                "protocol_hash": context["protocol_hash"],
                "escalation": escalation,
            }
        )
        visible: dict[str, Any] = copy.deepcopy(parent.visible_payload)
        visible["human_review_material"] = [
            {
                "annotation_id": item["annotation_id"],
                "decision": item.get("decision"),
                "labels": item.get("labels"),
                "taxonomy_proposal": item.get("taxonomy_proposal"),
                "uncertain": item.get("uncertain", False),
                "evidence_spans": item.get("evidence_spans", []),
                "notes": item.get("notes", ""),
            }
            for item in records
        ]
        task = AnnotationTask(
            annotation_task_id=identity,
            response_id=parent.response_id,
            purpose=purpose,
            cohort_id=parent.cohort_id,
            assigned_annotator=actor,
            guideline_hash=parent.guideline_hash,
            visible_payload=visible,
        )
        derived: dict[str, Any] = {
            **context,
            "parent_task_ids": sorted(parents),
            "reference_purpose": context.get("reference_purpose", parent.purpose.value),
            "escalation": escalation,
        }
        self._save_context(identity, derived)
        existing: dict[str, Any] | None = self._exists("annotation_task", identity)
        if existing is None:
            self._store.create_record("annotation_task", identity, task.model_dump(mode="json"))
            return task
        return AnnotationTask.model_validate(existing)

    def create_adjudications(self, task_ids: list[str]) -> list[AnnotationTask]:
        """Create human review only after complete independent submissions."""
        identities: list[str] = _ids(task_ids, "task_ids")
        groups: dict[tuple[str, str, str | None, str], AnnotationTask] = {}
        for identity in identities:
            task = AnnotationTask.model_validate(
                self._store.read_record("annotation_task", identity)
            )
            if task.purpose in _RESOLUTION_PURPOSES:
                raise ValueError("Cannot adjudicate an adjudication through this method.")
            context: dict[str, Any] = self._context(identity)
            groups[(
                task.response_id, task.purpose.value, task.cohort_id,
                context["protocol_hash"],
            )] = task
        prepared: list[tuple[AnnotationTask, dict[str, Any], list[AnnotationTask], list[dict[str, Any]]]] = []
        for parent in groups.values():
            context = self._context(parent.annotation_task_id)
            tasks: list[AnnotationTask] = self._group_tasks(parent, context)
            records: list[dict[str, Any]] = self._independent_records(tasks)
            judgment: str = context["protocol"]["judgment"]
            disagreement: bool = len({
                _decision_key(item, judgment) for item in records
            }) > 1
            uncertain: bool = any(item["uncertain"] for item in records)
            if not disagreement and not uncertain:
                continue
            self._agreement_prerequisite(parent, context)
            prepared.append((parent, context, tasks, records))
        result: list[AnnotationTask] = []
        for parent, context, tasks, records in prepared:
            actor: str = context["protocol"]["adjudication"]["adjudicator_id"]
            if (
                context["protocol"]["adjudication"]["adjudicator_must_be_independent"]
                and actor in {item.assigned_annotator for item in tasks}
            ):
                raise ValueError("The authorized adjudicator is not independent of this group.")
            purpose = (
                AnnotationPurpose.AUDIT_ADJUDICATION
                if context["audit_only"]
                else AnnotationPurpose.OPERATIONAL_ADJUDICATION
            )
            result.append(self._derived_task(
                parent, context, purpose, actor,
                [item.annotation_task_id for item in tasks], records,
            ))
        return result

    def resolve_task(
        self, task_id: str, adjudicator_id: str, payload: dict[str, Any]
    ) -> None:
        """Preserve discussion/escalation separately from the final reference."""
        task = AnnotationTask.model_validate(
            self._store.read_record("annotation_task", _identifier(task_id))
        )
        actor: str = _identifier(adjudicator_id)
        if task.assigned_annotator != actor or task.purpose not in _RESOLUTION_PURPOSES:
            raise PermissionError("Caller is not authorized to resolve this task.")
        context: dict[str, Any] = self._context(task_id)
        settings: dict[str, Any] = context["protocol"]
        required_actor: str = settings["adjudication"][
            "senior_adjudicator_id"
            if task.purpose == AnnotationPurpose.SENIOR_ESCALATION
            else "adjudicator_id"
        ]
        if actor != required_actor:
            raise PermissionError("Adjudicator identity differs from the frozen authorization.")
        if context["parent_task_ids"]:
            for parent_id in context["parent_task_ids"]:
                parent = AnnotationTask.model_validate(
                    self._store.read_record("annotation_task", parent_id)
                )
                if parent.purpose not in _RESOLUTION_PURPOSES:
                    if self._exists("annotation", self._submission_id(parent_id)) is None:
                        raise ValueError("Independent adjudication prerequisites are incomplete.")
        elif task.purpose != AnnotationPurpose.OPERATIONAL_ADJUDICATION:
            raise ValueError("Audit adjudication cannot bypass independent submissions.")

        result: dict[str, Any] = self._result(task, context, payload)
        if result["uncertain"]:
            if task.purpose == AnnotationPurpose.SENIOR_ESCALATION:
                raise ValueError("A senior unresolved judgment cannot become a final reference.")
            senior = settings["adjudication"]["senior_adjudicator_id"]
            if settings["adjudication"]["adjudicator_must_be_independent"]:
                original_raters = {
                    self._store.read_record("annotation_task", parent_id)["assigned_annotator"]
                    for parent_id in context["parent_task_ids"]
                }
                if senior in original_raters or senior == actor:
                    raise ValueError("The authorized senior adjudicator is not independent.")
            discussion_id: str = self._submission_id(task_id)
            existing: dict[str, Any] | None = self._exists("annotation", discussion_id)
            if existing is not None and existing["result_hash"] != _content_hash(result):
                raise ValueError("Conflicting immutable adjudication discussion exists.")
            discussion: dict[str, Any] = {
                "annotation_id": discussion_id,
                "annotation_task_id": task_id,
                "response_id": task.response_id,
                "purpose": "senior_escalation",
                "reference_purpose": context["reference_purpose"],
                "cohort_id": task.cohort_id,
                "annotator_id": actor,
                "guideline_hash": task.guideline_hash,
                "taxonomy_id": settings["taxonomy_id"],
                "config_hash": context["config_hash"],
                "protocol_hash": context["protocol_hash"],
                "submitted_at": existing["submitted_at"] if existing else _now(),
                "status": "submitted",
                "audit_only": context["audit_only"],
                "result_hash": _content_hash(result),
                **result,
            }
            self._store.create_record("annotation", discussion_id, discussion)
            self._store.freeze_records("annotation", [discussion_id])
            self._derived_task(
                task, context, AnnotationPurpose.SENIOR_ESCALATION,
                settings["adjudication"]["senior_adjudicator_id"],
                [task_id], [discussion], escalation=True,
            )
            self._finish_task(task, AnnotationStatus.SUBMITTED)
            return

        identity: str = self._resolution_id(task_id)
        existing = self._exists("adjudication", identity)
        if existing is not None:
            if existing["result_hash"] != _content_hash(result):
                raise ValueError("A conflicting final adjudication already exists.")
            self._finish_task(task, AnnotationStatus.ADJUDICATED)
            return
        if task.status not in {
            AnnotationStatus.DRAFT, AnnotationStatus.PENDING, AnnotationStatus.IN_PROGRESS
        }:
            raise ValueError("This adjudication is already closed or escalated.")
        purpose: str = (
            "audit_adjudication" if context["audit_only"]
            else context["reference_purpose"]
        )
        record: dict[str, Any] = {
            "annotation_id": identity,
            "annotation_task_id": task_id,
            "response_id": task.response_id,
            "purpose": purpose,
            "reference_purpose": context["reference_purpose"],
            "cohort_id": task.cohort_id,
            "annotator_id": actor,
            "guideline_hash": task.guideline_hash,
            "taxonomy_id": settings["taxonomy_id"],
            "config_hash": context["config_hash"],
            "protocol_hash": context["protocol_hash"],
            "submitted_at": _now(),
            "status": "adjudicated",
            "audit_only": context["audit_only"],
            "senior_adjudication": task.purpose == AnnotationPurpose.SENIOR_ESCALATION,
            "result_hash": _content_hash(result),
            **result,
        }
        self._store.create_record("adjudication", identity, record)
        self._store.freeze_records("adjudication", [identity])
        self._finish_task(task, AnnotationStatus.ADJUDICATED)

    def get_reference(
        self,
        response_id: str,
        purpose: str,
        *,
        protocol_hash: str | None = None,
        taxonomy_id: str | None = None,
    ) -> dict[str, Any]:
        """Return an unambiguous completed human reference, never a partial vote."""
        response: str = _identifier(response_id)
        selected = AnnotationPurpose(purpose)
        tasks: list[AnnotationTask] = [
            AnnotationTask.model_validate(item)
            for item in self._store.list_records(
                "annotation_task", {"response_id": response}
            )
            if item["purpose"] == selected.value
        ]
        if not tasks:
            raise LookupError("No human-reference task exists for this response/purpose.")
        groups: dict[tuple[str | None, str], AnnotationTask] = {}
        for task in tasks:
            context: dict[str, Any] = self._context(task.annotation_task_id)
            if protocol_hash is not None and context["protocol_hash"] != protocol_hash:
                continue
            if taxonomy_id is not None and context["protocol"]["taxonomy_id"] != taxonomy_id:
                continue
            groups[(task.cohort_id, context["protocol_hash"])] = task
        if not groups:
            raise LookupError("No human-reference task matches the selected protocol/taxonomy.")
        references: list[dict[str, Any]] = []
        for parent in groups.values():
            context = self._context(parent.annotation_task_id)
            if parent.purpose == AnnotationPurpose.OPERATIONAL_ADJUDICATION:
                group_tasks: list[AnnotationTask] = [parent]
                records: list[dict[str, Any]] = []
            else:
                group_tasks = self._group_tasks(parent, context)
                records = self._independent_records(group_tasks)
            adjudications: list[dict[str, Any]] = [
                item for item in self._store.list_records("adjudication")
                if item.get("response_id") == response
                and item.get("cohort_id") == parent.cohort_id
                and item.get("protocol_hash") == context["protocol_hash"]
                and item.get("reference_purpose", item.get("purpose")) == selected.value
                and item.get("uncertain") is False
            ]
            if len(adjudications) > 1:
                raise ValueError("Multiple final adjudications are ambiguous.")
            if adjudications:
                chosen: dict[str, Any] = adjudications[0]
                origin: str = "adjudicated"
            elif records and not any(item["uncertain"] for item in records) and len({
                _decision_key(item, context["protocol"]["judgment"]) for item in records
            }) == 1:
                chosen = records[0]
                origin = "independent_agreement"
            else:
                raise LookupError("Human reference is incomplete, disputed, or unresolved.")
            references.append({
                "response_id": response,
                "purpose": selected.value,
                "cohort_id": parent.cohort_id,
                "guideline_hash": parent.guideline_hash,
                "taxonomy_id": context["protocol"]["taxonomy_id"],
                "protocol_hash": context["protocol_hash"],
                "audit_only": context["audit_only"],
                "origin": origin,
                "annotation_ids": [
                    item["annotation_id"] for item in (adjudications or records)
                ],
                **{
                    key: chosen[key] for key in (
                        "decision", "labels", "taxonomy_proposal",
                        "uncertain", "evidence_spans", "notes",
                    ) if key in chosen
                },
            })
        if len(references) != 1:
            raise ValueError("Purpose alone does not identify an unambiguous cohort/version.")
        return references[0]

    def _prediction(
        self, response_id: str, prediction_id: str, purpose: str, version: str
    ) -> None:
        if purpose == "failure_identification":
            kind: str = "failure_decision"
            payload: dict[str, Any] = self._store.read_record(kind, prediction_id)
            decision = FailureDecision.model_validate(payload)
            if (
                decision.response_id != response_id
                or decision.identification_version != version
                or decision.decision == FailureStatus.UNRESOLVED
                or decision.decision_origin.value not in {"hard_label", "automatic_range"}
            ):
                raise ValueError("Identification audit requires frozen automatic decisions.")
        else:
            kind, payload = self._find(_ASSIGNMENT_KINDS, prediction_id)
            taxonomy: TaxonomyVersion = self._taxonomy(payload["taxonomy_id"])
            assignment = LabelAssignment.model_validate(
                payload, context={"taxonomy": taxonomy, "require_taxonomy": True}
            )
            if assignment.response_id != response_id or assignment.origin not in _AUTOMATED_ORIGINS:
                raise ValueError("Label audits require pre-correction automated predictions.")
        self._immutable(kind, prediction_id)

    def _sample(
        self,
        population: list[str],
        size: int,
        method: str,
        seed: int,
        sampling: dict[str, Any],
    ) -> list[str]:
        if len(population) < size:
            raise ValueError(
                f"Eligible audit population has {len(population)} records; {size} are required."
            )
        generator = random.Random(seed)
        if method == "simple_random_without_replacement":
            return sorted(generator.sample(sorted(population), size))
        if method == "explicit_membership":
            if len(population) != size:
                raise ValueError("Explicit audit membership must have exactly the requested size.")
            return sorted(population)
        if method != "stratified_without_replacement":
            raise ValueError("Unsupported audit sampling method.")
        strata: dict[str, Any] = _mapping(
            _required(sampling, "stratum_by_response"), "stratum_by_response"
        )
        allocation: dict[str, Any] = _mapping(
            _required(sampling, "allocation_by_stratum"), "allocation_by_stratum"
        )
        if not set(population).issubset(strata):
            raise ValueError("Every eligible response needs an explicit stratum assignment.")
        groups: dict[str, list[str]] = defaultdict(list)
        for response_id in sorted(population):
            groups[_identifier(strata[response_id])].append(response_id)
        if set(groups) != set(allocation):
            raise ValueError("Explicit stratum allocation must cover exactly the eligible strata.")
        counts: dict[str, int] = {
            key: _integer(value, "Stratum allocation", minimum=0)
            for key, value in allocation.items()
        }
        if sum(counts.values()) != size:
            raise ValueError("Stratum allocations do not sum to the requested sample size.")
        selected: list[str] = []
        for key in sorted(groups):
            if counts[key] > len(groups[key]):
                raise ValueError("An explicitly allocated stratum has insufficient eligible records.")
            selected.extend(generator.sample(groups[key], counts[key]))
        return sorted(selected)

    def sample_audit(self, purpose: str, sampling: dict[str, Any]) -> AuditCohort:
        """Create reproducible audit membership without consulting human audit labels."""
        aliases: dict[str, str] = {
            "independent_identification_audit": "failure_identification",
            "independent_label_audit": "failure_categorization",
            "expanded_label_audit": "failure_categorization",
            "stability": "taxonomy_stability",
        }
        selected: str = aliases.get(purpose, purpose)
        sizes: dict[str, int] = {
            "failure_identification": 100,
            "human_agreement": 320,
            "failure_categorization": 800,
            "taxonomy_stability": 180,
        }
        if selected not in sizes:
            raise ValueError("Unsupported audit cohort purpose.")
        settings: dict[str, Any] = _mapping(sampling, "Sampling protocol")
        cohort_id: str = _identifier(_required(settings, "cohort_id"))
        _digest(_required(settings, "config_hash"), "config_hash")
        guideline_hash: str = _digest(_required(settings, "guideline_hash"), "guideline_hash")
        seed: int = _integer(_required(settings, "seed"), "seed", minimum=0)
        size: int = _integer(_required(settings, "sample_size"), "sample_size")
        if size != sizes[selected]:
            raise ValueError("Requested audit size differs from the paper protocol.")
        if _required(settings, "sampling_unit") != "response":
            raise ValueError("Audit sampling unit must be a model-response pair.")
        method: str = _identifier(_required(settings, "sampling_method"))
        _identifier(_required(settings, "eligible_population_version"))
        version: str = _identifier(_required(settings, "frozen_prediction_version"))
        population: list[str] = _ids(
            _required(settings, "eligible_response_ids"), "eligible_response_ids"
        )
        exclusions: set[str] = set(_ids(
            _required(settings, "excluded_response_ids"), "excluded_response_ids",
            nonempty=False,
        ))
        categories: list[str] = _ids(
            _required(settings, "exclusion_categories"), "exclusion_categories",
            nonempty=False,
        )
        annotation_protocol: dict[str, Any] = self._protocol(
            _required(settings, "annotation_protocol")
        )
        if (
            annotation_protocol["cohort_id"] != cohort_id
            or annotation_protocol["guideline_hash"] != guideline_hash
            or annotation_protocol["config_hash"] != settings["config_hash"]
        ):
            raise ValueError("Sampling and annotation identities disagree.")
        bindings: dict[str, Any] = _mapping(
            _required(settings, "frozen_prediction_ids"), "frozen_prediction_ids"
        )
        if not set(population).issubset(bindings):
            raise ValueError("Every eligible response needs a designated frozen prediction.")

        all_tasks: list[dict[str, Any]] = self._store.list_records("annotation_task")
        calibration: set[str] = {
            item["response_id"] for item in all_tasks
            if item["purpose"] == "calibration"
        }
        if selected == "failure_identification":
            required_categories: set[str] = {
                "human_anchors", "calibration_examples",
                "previously_manually_reviewed_cases", "duplicate_records",
            }
            if not required_categories.issubset(categories):
                raise ValueError("Identification audit omits required exclusion categories.")
            if method != "simple_random_without_replacement":
                raise ValueError("Identification audit uses simple random sampling only.")
            reviewed: set[str] = {
                item["response_id"] for item in all_tasks
                if item["purpose"] not in {value.value for value in _AUDIT_PURPOSES}
            }
            exclusions.update(reviewed | calibration)
            duplicate_keys: dict[str, Any] = _mapping(
                _required(settings, "duplicate_key_by_response"), "duplicate_key_by_response"
            )
            representatives: dict[str, Any] = _mapping(
                _required(settings, "duplicate_representatives"), "duplicate_representatives"
            )
            duplicate_groups: dict[str, list[str]] = defaultdict(list)
            for response_id in population:
                if response_id not in duplicate_keys:
                    raise ValueError("Duplicate detection keys must cover the population.")
                duplicate_groups[_identifier(duplicate_keys[response_id])].append(response_id)
            for key, members in duplicate_groups.items():
                if len(members) > 1:
                    representative: Any = representatives.get(key)
                    if representative not in members:
                        raise ValueError("Duplicate groups require an explicit retained representative.")
                    exclusions.update(set(members) - {representative})
        elif selected == "human_agreement":
            if method != "stratified_without_replacement":
                raise ValueError("Agreement sampling requires explicit stage/category strata.")
            if "calibration_examples" not in categories:
                raise ValueError("Agreement sampling must exclude calibration examples.")
            exclusions.update(calibration)

        eligible: list[str] = sorted(set(population) - exclusions)
        original: list[str] = []
        original_cohort_id: str | None = None
        if selected == "failure_categorization":
            original_cohort_id = _identifier(_required(settings, "original_cohort_id"))
            original_cohort = AuditCohort.model_validate(
                self._store.read_record("audit_cohort", original_cohort_id)
            )
            if (
                not original_cohort.locked
                or original_cohort.purpose != "human_agreement"
                or len(original_cohort.response_ids) != 320
            ):
                raise ValueError("Expanded audit must preserve a locked 320-instance agreement cohort.")
            original = original_cohort.response_ids
            if not set(original).issubset(population) or set(original) & exclusions:
                raise ValueError("Expanded population must retain every original audit member.")
            additional_protocol: dict[str, Any] = _mapping(
                _required(settings, "additional_sampling_protocol"),
                "additional_sampling_protocol",
            )
            additional_method: str = _identifier(
                _required(additional_protocol, "sampling_method")
            )
            additions: list[str] = self._sample(
                sorted(set(eligible) - set(original)), 480,
                additional_method, seed, additional_protocol,
            )
            selected_ids: list[str] = sorted(original + additions)
            stable: list[str] = _ids(
                _required(settings, "stable_consensus_response_ids"),
                "stable_consensus_response_ids", nonempty=False,
            )
            if not set(stable) & set(selected_ids):
                raise ValueError("Selected categorization audit lacks stable-consensus cases.")
            old_bindings: Any = original_cohort.sampling_protocol.get("frozen_prediction_ids")
            if not isinstance(old_bindings, dict) or any(
                old_bindings.get(identity) != bindings[identity] for identity in original
            ):
                raise ValueError("Expanded audit must retain the original pre-correction snapshot.")
        else:
            selected_ids = self._sample(eligible, size, method, seed, settings)

        for response_id in selected_ids:
            response = ModelResponse.model_validate(
                self._store.read_record("model_response", response_id)
            )
            if response.status == ResponseStatus.TRANSPORT_FAILURE:
                raise ValueError("Transport failures are not completed response audit units.")
            self._prediction(response_id, _identifier(bindings[response_id]), selected, version)
        if not set(selected_ids).issubset(annotation_protocol["assignments"]):
            raise ValueError("Selected audit members lack explicit independent-rater assignments.")

        protocol: dict[str, Any] = {
            **settings,
            "membership_count": len(selected_ids),
            "excluded_response_ids": sorted(exclusions),
            "frozen_prediction_ids": {
                identity: bindings[identity] for identity in selected_ids
            },
            "annotation_protocol": annotation_protocol,
            "guideline_hash": guideline_hash,
            "taxonomy_id": annotation_protocol["taxonomy_id"],
            "sampling_implementation": "python_random_sample_sorted_population",
            "sampling_input_hash": _content_hash(settings),
        }
        if selected == "failure_categorization":
            protocol.update(
                original_response_ids=sorted(original),
                additional_response_ids=sorted(set(selected_ids) - set(original)),
                original_cohort_id=original_cohort_id,
                reference_cohort_ids=[original_cohort_id],
                includes_stable_consensus=True,
            )
        cohort = AuditCohort(
            cohort_id=cohort_id,
            purpose=selected,
            response_ids=selected_ids,
            sampling_protocol=protocol,
            seed=seed,
            frozen_prediction_version=version,
        )
        existing: dict[str, Any] | None = self._exists("audit_cohort", cohort_id)
        if existing is not None:
            previous = AuditCohort.model_validate(existing)
            comparable: dict[str, Any] = previous.model_dump(mode="json")
            comparable["locked"] = False
            if _canonical_json(comparable) != _canonical_json(cohort):
                raise ValueError("Existing cohort identity has a different sampling protocol.")
            return previous
        self._store.create_record("audit_cohort", cohort_id, cohort.model_dump(mode="json"))
        return cohort

    def lock_cohort(self, cohort_id: str) -> AuditCohort:
        """Freeze membership and prediction bindings, not annotation completion."""
        cohort = AuditCohort.model_validate(
            self._store.read_record("audit_cohort", _identifier(cohort_id))
        )
        protocol: dict[str, Any] = cohort.sampling_protocol
        if protocol["membership_count"] != len(cohort.response_ids):
            raise ValueError("Cohort membership count is inconsistent.")
        if set(cohort.response_ids) & set(protocol["excluded_response_ids"]):
            raise ValueError("Cohort membership violates its exclusions.")
        bindings: dict[str, Any] = protocol["frozen_prediction_ids"]
        for response_id in cohort.response_ids:
            self._store.read_record("model_response", response_id)
            self._prediction(
                response_id, bindings[response_id], cohort.purpose,
                cohort.frozen_prediction_version,
            )
        if not cohort.locked:
            payload: dict[str, Any] = cohort.model_dump(mode="json")
            payload["locked"] = True
            cohort = AuditCohort.model_validate(payload)
            self._store.update_draft("audit_cohort", cohort_id, {"locked": True})
        self._store.freeze_records("audit_cohort", [cohort_id])
        return cohort


def _run_ui() -> None:
    """Present only authenticated, preassigned work through the service API."""
    import streamlit as st

    st.set_page_config(page_title="CTI annotation", layout="wide")
    st.title("CTI evidence annotation")
    database_path: str | None = os.environ.get("CTI_ANNOTATION_DATABASE")
    artifact_root: str | None = os.environ.get("CTI_ANNOTATION_ARTIFACT_ROOT")
    identities_json: str | None = os.environ.get("CTI_ANNOTATION_IDENTITY_MAP")
    if not database_path or not artifact_root or not identities_json:
        st.error(
            "Deployment must explicitly configure database, artifact directory, "
            "and authenticated-subject identity mapping."
        )
        st.stop()
    user: Any = getattr(st, "user", None)
    if user is None or not getattr(user, "is_logged_in", False):
        st.info("Authenticate with the deployment's configured OIDC provider.")
        if st.button("Sign in"):
            st.login()
        st.stop()
    try:
        identity_map: Any = json.loads(identities_json)
        subject: Any = user.get("sub")
        if not isinstance(identity_map, dict) or subject not in identity_map:
            raise PermissionError("Authenticated subject is not authorized.")
        actor: str = _identifier(identity_map[subject])
        store = ArtifactStore(database_path, artifact_root)
        store.initialize()
        service = AnnotationService(store)
        task: AnnotationTask = service.get_next_task(actor)
    except LookupError:
        st.success("No pending tasks are assigned to you.")
        st.stop()
    except (ValueError, PermissionError, RuntimeError, OSError) as exc:
        st.error(str(exc))
        st.stop()

    st.caption(f"Assigned task: {task.annotation_task_id}")
    material: dict[str, Any] = task.visible_payload
    left, right = st.columns(2)
    with left:
        st.subheader("Original task")
        st.json(material["task_input"])
        st.subheader("Evaluated response")
        st.code(material["evaluated_response"]["raw_text"], language="text")
        if material["evaluated_response"]["schema_errors"]:
            st.write(material["evaluated_response"]["schema_errors"])
    with right:
        st.subheader("Applicable reference material")
        st.json(material["reference_material"])
        st.subheader("Frozen annotation guidance")
        st.json(material["guideline"])
        if material["correctness_criteria"]:
            st.json(material["correctness_criteria"])
        if "taxonomy" in material:
            st.json(material["taxonomy"])
    if "human_review_material" in material:
        st.subheader("Independent human submissions for authorized adjudication")
        st.json(material["human_review_material"])

    judgment: str = material["judgment"]
    session_key: str = f"annotation:{task.annotation_task_id}"
    with st.form(session_key, clear_on_submit=False):
        st.caption(
            "Draft inputs remain editable until submission. Evidence must cite "
            "exact supplied source identifiers and verbatim quotations."
        )
        uncertain: bool = st.checkbox(
            "The human judgment remains uncertain", value=False
        )
        payload: dict[str, Any] = {
            "uncertain": uncertain,
            "guideline_hash": task.guideline_hash,
        }
        if judgment == "identification":
            choice: str = st.selectbox(
                "Correctness judgment",
                ["Select a judgment", "pass", "failure", "unresolved"],
                index=0,
            )
            payload["decision"] = choice
        elif judgment == "labels":
            taxonomy_material: dict[str, Any] = material["taxonomy"]
            names: dict[str, str] = {
                item["subtype_id"]: item["name"]
                for item in taxonomy_material["subtype_definitions"]
            }
            labels: list[str] = st.multiselect(
                "Independently supported subtypes",
                sorted(names),
                format_func=lambda item: f"{item}: {names[item]}",
            )
            payload["labels"] = labels
            payload["taxonomy_id"] = taxonomy_material["taxonomy_id"]
        else:
            proposal_text: str = st.text_area(
                "Discovery proposal JSON",
                value="",
                height=180,
                help=(
                    "Provide new_subtypes, rules, and no_new_categories. "
                    "Definitions are human proposals, not automatic additions."
                ),
            )
        evidence_text: str = st.text_area(
            "Evidence-span JSON array", value="", height=180
        )
        payload["notes"] = st.text_area("Concise evidence-based notes", value="")
        submitted: bool = st.form_submit_button("Submit immutable human judgment")

    if submitted:
        try:
            payload["evidence_spans"] = json.loads(evidence_text)
            if judgment == "discovery":
                payload["taxonomy_proposal"] = json.loads(proposal_text)
            if task.purpose in _RESOLUTION_PURPOSES:
                service.resolve_task(task.annotation_task_id, actor, payload)
            else:
                service.submit_annotation(task.annotation_task_id, actor, payload)
            st.success(
                "Submission preserved. Independent labels and audit references "
                "do not overwrite operational predictions."
            )
            st.rerun()
        except (ValueError, TypeError, KeyError, PermissionError, LookupError) as exc:
            st.error(f"Submission was not accepted: {exc}")


if __name__ == "__main__":
    _run_ui()
