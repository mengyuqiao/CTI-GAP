"""Two-round, human-in-the-loop failure categorization (RQ3).

This module applies a frozen taxonomy. It does not identify failures, discover
categories, conduct independent audits, or infer correctness from consensus.

Executable policy declarations
==============================

The following declarations use existing configuration fields. They have no
implicit defaults.

categorization.label_set_comparison_policy::

    {"mode": "set_equality"}

categorization.incomplete_agent_output_policy::

    {
      "mode": "block",
      "explicit_uncertainty": "block",
      "empty_assignment": "block",
      "allow_empty_labels_when_uncertain": true,
      "assignment_run_index": 0,
      "workload_denominator": "frozen_requested_response_cohort",
      "task_input_projection": "evidence_as_documents",
      "annotation_protocol": <complete AnnotationService protocol>
    }

``allow_empty_labels_when_uncertain`` defines the model-output contract required
by the supplied prompts. Empty or explicitly uncertain classifications remain
operational exceptions; they are not silently finalized or added to the paper's
disagreement/change selector.

The annotation protocol must use judgment="labels", cohort_id=null, the frozen
taxonomy ID, and its guideline hash. An omitted/null taxonomy ID or the explicit
"$taxonomy_id" marker binds to the supplied frozen taxonomy in a runtime copy;
a concrete different identity is rejected. It preassigns one authorized adjudicator to
each potentially reviewed response. reference_applicability must cover the whole
requested cohort. Models receive applicability notes, never human judgments.

The peer-information mapping follows agent_round_two.jinja2's documented
contract. All four agents see the same immutable first-round snapshot.

Identity conventions
====================

Panel model names are stable actor IDs because ModelClient resolves those exact
names to the configured categorization endpoints. Assignment round_index is 1
or 2; assignment_run_index separately selects the explicitly configured inference
run/seed. Pipeline consensus has actor_id="rq3:consensus", an engineering identity,
not an additional agent.

Raw model responses, agent assignments, peer snapshots, selector records, human
references, and final assignments are retained separately. Finalization never
overwrites an automated prediction. Internal manifests designate the exact
records belonging to this protocol and population.

ArtifactStore provides immutable inserts, not a cross-provider distributed job
lease. Concurrent processes can make duplicate external requests, but its
authoritative request cache prevents duplicate persisted completions.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
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
    TaskInstance,
    TaxonomyVersion,
    _canonical_json,
    _content_hash,
    _identifier,
)
from storage import ArtifactStore


_PANEL_NAMES: frozenset[str] = frozenset(
    {"GPT-5", "Llama-4-17B", "Gemini 2.5", "Claude Sonnet 4"}
)
_TAXONOMY_KINDS: tuple[str, ...] = ("taxonomy", "taxonomy_version")
_OPERATIONAL_PURPOSE: str = AnnotationPurpose.OPERATIONAL_ADJUDICATION.value
_CONSENSUS_ACTOR: str = "rq3:consensus"


class _OperationalException(RuntimeError):
    """Incomplete or unusable agent execution, not a scientific review trigger."""

    def __init__(
        self,
        message: str,
        *,
        response_id: str = "",
        round_index: int = 0,
        artifact_ids: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.response_id: str = response_id
        self.round_index: int = round_index
        self.artifact_ids: list[str] = sorted(set(artifact_ids))


class _AdjudicationPending(LookupError):
    """No eligible completed operational human reference exists."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an explicitly configured mapping.")
    return copy.deepcopy(value)


def _ids(values: Any, name: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple of identifiers.")
    result: list[str] = [_identifier(value) for value in values]
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate identifiers.")
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty.")
    return sorted(result)


class Categorizer:
    """Apply a frozen taxonomy using four agents, two rounds, and human review."""

    def __init__(
        self,
        config: ExperimentConfig,
        store: ArtifactStore,
        models: ModelClient,
        annotations: AnnotationService,
        *,
        run_index: int | None = None,
        single_agent: str | None = None,
        reconsideration: dict[str, Any] | None = None,
    ) -> None:
        if run_index is not None and (
            isinstance(run_index, bool) or not isinstance(run_index, int) or run_index < 0
        ):
            raise ValueError("Categorization run_index must be a nonnegative integer.")
        self._config: ExperimentConfig = config
        self._store: ArtifactStore = store
        self._models: ModelClient = models
        self._annotations: AnnotationService = annotations
        self._run_index: int | None = run_index
        if single_agent is None and reconsideration is not None:
            raise ValueError("Single-agent reconsideration requires an explicitly selected agent.")
        if single_agent is not None and (
            single_agent not in _PANEL_NAMES
            or reconsideration not in ({"mode": "fresh_reassessment"}, {"mode": "self_review"})
        ):
            raise ValueError("Single-agent ablation requires explicit supported reconsideration.")
        self._single_agent = single_agent
        self._reconsideration = copy.deepcopy(reconsideration)

    @property
    def _protocol_id(self) -> str:
        return _content_hash(
            {
                "implementation_contract": "rq3-categorization-v1",
                "config_hash": self._config.config_hash,
                "corpus_version": self._config.corpus_version,
                "categorization": self._config.categorization,
                "panel": self._config.models.categorization_panel,
                "taxonomy_settings": self._config.taxonomy,
                "assignment_run_index": self._policy()["assignment_run_index"],
                "execution_mechanisms": self._mechanisms(),
            }
        )

    def _single_reconsideration_template(self) -> Path:
        return Path(__file__).with_name("prompts") / "agent_single_reconsideration.jinja2"

    def _mechanisms(self) -> dict[str, Any]:
        mechanisms = {
            "heterogeneous_panel": self._single_agent is None,
            "agent_ids": [self._single_agent] if self._single_agent else sorted(_PANEL_NAMES),
            "rounds": 2, "selective_human_verification": True,
            "reconsideration": self._reconsideration or {"mode": "first_round_peer_review"},
            "single_agent_variant": self._single_agent is not None,
        }
        if self._single_agent is not None:
            template = self._single_reconsideration_template()
            mechanisms["reconsideration_template"] = {
                "path": str(template), "sha256": hashlib.sha256(template.read_bytes()).hexdigest(),
            }
        return mechanisms

    def _policy(self) -> dict[str, Any]:
        comparison: Any = self._config.categorization.label_set_comparison_policy
        if comparison != {"mode": "set_equality"}:
            raise ValueError(
                "RQ3 requires an explicitly configured set_equality comparison policy."
            )
        policy: dict[str, Any] = _mapping(
            self._config.categorization.incomplete_agent_output_policy,
            "categorization.incomplete_agent_output_policy",
        )
        required: set[str] = {
            "mode",
            "explicit_uncertainty",
            "empty_assignment",
            "allow_empty_labels_when_uncertain",
            "assignment_run_index",
            "workload_denominator",
            "task_input_projection",
            "annotation_protocol",
        }
        if set(policy) != required:
            raise ValueError(
                "incomplete_agent_output_policy must contain exactly the "
                "documented executable policy fields."
            )
        if (
            policy["mode"] != "block"
            or policy["explicit_uncertainty"] != "block"
            or policy["empty_assignment"] != "block"
        ):
            raise ValueError(
                "This implementation blocks incomplete, empty, and explicitly "
                "uncertain agent outputs without changing the paper's selector."
            )
        if policy["allow_empty_labels_when_uncertain"] is not True:
            raise ValueError(
                "The supplied categorization prompts require an explicitly "
                "authorized empty uncertain model-output representation."
            )
        run_index: Any = policy["assignment_run_index"]
        if (
            isinstance(run_index, bool)
            or not isinstance(run_index, int)
            or run_index < 0
        ):
            raise ValueError("assignment_run_index must be explicitly nonnegative.")
        if self._run_index is not None:
            policy["assignment_run_index"] = self._run_index
        if policy["workload_denominator"] != "frozen_requested_response_cohort":
            raise ValueError("Unsupported operational workload denominator.")
        if policy["task_input_projection"] != "evidence_as_documents":
            raise ValueError("Unsupported categorization input projection.")
        _mapping(policy["annotation_protocol"], "Operational annotation protocol")

        peer: dict[str, Any] = _mapping(
            self._config.categorization.peer_payload_policy, "peer_payload_policy"
        )
        if set(peer) != {
            "mode", "show_actor_ids", "show_uncertainty", "restate_own_prediction"
        }:
            raise ValueError("Peer exposure must use the supplied template's full contract.")
        if peer["mode"] not in {"labels_only", "evidence_spans"}:
            raise ValueError("Unsupported peer exposure mode.")
        for key in ("show_actor_ids", "show_uncertainty", "restate_own_prediction"):
            if not isinstance(peer[key], bool):
                raise ValueError("Every peer exposure flag must be explicitly boolean.")
        return policy

    def _panel(self) -> list[dict[str, str]]:
        entries = self._config.models.categorization_panel
        if len(entries) != 4 or {entry.name for entry in entries} != _PANEL_NAMES:
            raise ValueError("RQ3 requires the four specified distinct categorization agents.")
        result: list[dict[str, str]] = []
        for entry in entries:
            if self._single_agent is not None and entry.name != self._single_agent:
                continue
            if not entry.endpoint_or_checkpoint or not entry.revision:
                raise ValueError("Categorization panel endpoint/revision is unresolved.")
            result.append(
                {
                    "actor_id": entry.name,
                    "name": entry.name,
                    "endpoint_or_checkpoint": entry.endpoint_or_checkpoint,
                    "revision": entry.revision,
                }
            )
        return result

    def _event(self, purpose: str, content: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "purpose": purpose,
            "protocol_id": self._protocol_id,
            "config_hash": self._config.config_hash,
            "corpus_version": self._config.corpus_version,
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
        ]

    def _immutable(self, kind: str, record_id: str) -> None:
        matches: list[dict[str, Any]] = self._store.list_records(
            kind,
            {"record_id": record_id, "lifecycle": ["immutable", "frozen"]},
        )
        if len(matches) != 1:
            raise ValueError(f"{kind}/{record_id} must be immutable.")

    def _taxonomy(self, taxonomy: TaxonomyVersion) -> TaxonomyVersion:
        found: list[tuple[str, dict[str, Any]]] = []
        for kind in _TAXONOMY_KINDS:
            try:
                found.append(
                    (kind, self._store.read_record(kind, taxonomy.taxonomy_id))
                )
            except KeyError:
                pass
        if len(found) != 1:
            raise ValueError("A unique stored frozen taxonomy is required.")
        kind, payload = found[0]
        stored = TaxonomyVersion.model_validate(payload)
        if _canonical_json(stored) != _canonical_json(taxonomy) or not stored.frozen:
            raise ValueError("Taxonomy is unfrozen or differs from its stored version.")
        self._immutable(kind, stored.taxonomy_id)
        known: set[str] = {item.subtype_id for item in stored.subtype_definitions}
        if (
            "other" in known
            or "uncertain" in known
            or self._config.taxonomy_discovery.fallback_label in known
        ):
            raise ValueError("Discovery/uncertainty sentinels cannot be RQ3 subtypes.")
        return stored

    def _annotation_protocol(self, taxonomy: TaxonomyVersion) -> dict[str, Any]:
        protocol: dict[str, Any] = _mapping(
            self._policy()["annotation_protocol"], "Operational annotation protocol"
        )
        # The configuration cannot contain its own digest. Bind this derived
        # value in the copied runtime protocol, preserving the hashed declaration.
        if protocol.get("config_hash") in {None, "$config_hash"}:
            protocol["config_hash"] = self._config.config_hash
        # Discovery produces this identity after the configuration is frozen.
        # Resolve only the unspecified/runtime forms; preserve concrete bindings.
        if protocol.get("taxonomy_id") in {None, "$taxonomy_id"}:
            protocol["taxonomy_id"] = taxonomy.taxonomy_id
        if (
            protocol.get("config_hash") != self._config.config_hash
            or protocol.get("taxonomy_id") != taxonomy.taxonomy_id
            or protocol.get("guideline_hash") != taxonomy.guideline_hash
            or protocol.get("judgment") != "labels"
            or protocol.get("cohort_id") is not None
        ):
            raise ValueError(
                "Operational annotation protocol must bind this configuration, "
                "frozen taxonomy, guideline, and a non-audit purpose."
            )
        guideline: Any = protocol.get("guideline_content")
        if not isinstance(guideline, dict) or _content_hash(guideline) != taxonomy.guideline_hash:
            raise ValueError("Operational guideline content does not match its frozen hash.")
        if protocol.get("annotators_per_instance") != 1:
            raise ValueError("Direct operational review requires one designated adjudicator.")
        adjudication: dict[str, Any] = _mapping(
            protocol.get("adjudication"), "Adjudication authorization"
        )
        actor: str = _identifier(adjudication.get("adjudicator_id"))
        assignments: dict[str, Any] = _mapping(
            protocol.get("assignments"), "Operational assignments"
        )
        for actors in assignments.values():
            if actors != [actor]:
                raise ValueError(
                    "Direct operational tasks must be assigned to their authorized adjudicator."
                )
        return protocol

    def _failure_decision(self, response_id: str) -> tuple[str, FailureDecision]:
        bindings: dict[str, str] = {}
        for manifest in self._store.list_records("experiment_manifest"):
            if (
                manifest.get("purpose") != "identification_final"
                or manifest.get("config_hash") != self._config.config_hash
            ):
                continue
            result: dict[str, Any] = manifest.get("result", {})
            if response_id in result.get("failure_response_ids", []):
                identity: Any = result.get("decision_ids_by_response", {}).get(response_id)
                if isinstance(identity, str):
                    bindings[identity] = identity
        if len(bindings) > 1:
            raise ValueError("Response has ambiguous designated identification snapshots.")
        if bindings:
            identity: str = next(iter(bindings))
            decision = FailureDecision.model_validate(
                self._store.read_record("failure_decision", identity)
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
                raise ValueError("Response requires one designated frozen failure decision.")
            decision = FailureDecision.model_validate(candidates[0])
            identity = _content_hash(decision)
        self._immutable("failure_decision", identity)
        if (
            decision.response_id != response_id
            or decision.decision != FailureStatus.FAILURE
            or decision.requires_review
        ):
            raise ValueError("RQ3 accepts only finalized identified failure responses.")
        return identity, decision

    def _input(
        self, response_id: str
    ) -> tuple[ModelResponse, TaskInstance, str]:
        response = ModelResponse.model_validate(
            self._store.read_record("model_response", _identifier(response_id))
        )
        instance = TaskInstance.model_validate(
            self._store.read_record("task_instance", response.instance_id),
            context={"require_content_hash": True},
        )
        if response.config_hash != self._config.config_hash:
            raise ValueError("Evaluated response belongs to a different configuration.")
        if instance.transformation_log.get("corpus_version") != self._config.corpus_version:
            raise ValueError("Task belongs to a different corpus.")
        self._immutable("model_response", response_id)
        self._immutable("task_instance", instance.instance_id)
        if response.status == ResponseStatus.TRANSPORT_FAILURE:
            raise _OperationalException(
                "Transport failure is not a completed categorizable response.",
                response_id=response_id,
            )
        identity, _ = self._failure_decision(response_id)
        if not instance.reference.get("answer") or not instance.reference.get("evidence_records"):
            raise ValueError("Categorization requires applicable reference material.")
        return response, instance, identity

    def _packet(self, response_id: str, taxonomy: TaxonomyVersion) -> dict[str, Any]:
        response, instance, _ = self._input(response_id)
        protocol: dict[str, Any] = self._annotation_protocol(taxonomy)
        applicability_by_response = protocol.get("reference_applicability", {})
        applicability: Any = applicability_by_response.get(
            response_id, applicability_by_response.get("*")
        )
        if not isinstance(applicability, dict) or not applicability:
            raise ValueError("Task-specific reference applicability must be resolved.")
        fields: tuple[str, ...] = (
            "evidence_id", "document_id", "source_id", "source_version",
            "snippet", "role", "publication_time", "modification_time",
            "claim_valid_from", "claim_valid_until",
        )

        def project(record: Mapping[str, Any]) -> dict[str, Any]:
            return {key: record[key] for key in fields if key in record}

        task_input = review_task_input(self._store, response, instance)

        taxonomy_projection: dict[str, Any] = {
            "taxonomy_id": taxonomy.taxonomy_id,
            "iteration": taxonomy.iteration,
            "guideline_hash": taxonomy.guideline_hash,
            "frozen": taxonomy.frozen,
            "subtype_definitions": [
                {
                    "subtype_id": item.subtype_id,
                    "name": item.name,
                    "definition": item.definition,
                    "evidence_requirements": item.evidence_requirements,
                }
                for item in taxonomy.subtype_definitions
            ],
            "inclusion_exclusion_rules": {
                key: value.model_dump(mode="json")
                for key, value in taxonomy.inclusion_exclusion_rules.items()
            },
        }
        return {
            "task_input": task_input,
            "evaluated_response": {
                "response_id": response.response_id,
                "instance_id": instance.instance_id,
                "status": response.status.value,
                "raw_text": response.raw_text,
                "parsed_answer": response.parsed_answer,
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
            "frozen_taxonomy": taxonomy_projection,
        }

    def _payload(self, packet: dict[str, Any]) -> dict[str, Any]:
        return {
            **copy.deepcopy(packet),
            "categorization_settings": (
                self._config.categorization.model_dump(mode="json")
            ),
            "taxonomy_settings": {
                "multiple_labels_allowed": self._config.taxonomy.multiple_labels_allowed,
                "require_evidence_for_each_label": (
                    self._config.taxonomy.require_evidence_for_each_label
                ),
                "guideline_manifest": self._config.taxonomy.guideline_manifest,
            },
            "reproducibility_settings": {
                "retain_pre_correction_predictions": (
                    self._config.reproducibility.retain_pre_correction_predictions
                ),
                "silently_fill_unspecified_settings": (
                    self._config.reproducibility.silently_fill_unspecified_settings
                ),
            },
            "output_policy": {
                "allow_empty_labels_when_uncertain": (
                    self._policy()["allow_empty_labels_when_uncertain"]
                )
            },
        }

    def _contract(self, taxonomy: TaxonomyVersion) -> dict[str, Any]:
        subtype_ids: list[str] = sorted(
            item.subtype_id for item in taxonomy.subtype_definitions
        )
        return {
            "version": "rq3-classification-v1",
            "kind": "classification",
            "taxonomy_id": taxonomy.taxonomy_id,
            "discovery_mode": False,
            "allow_empty_labels_when_uncertain": (
                self._policy()["allow_empty_labels_when_uncertain"]
            ),
            "json_schema": {
                "type": "object",
                "required": ["labels", "evidence_spans", "uncertain"],
                "additionalProperties": False,
                "properties": {
                    "labels": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"type": "string", "enum": subtype_ids},
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
                                    "type": ["integer", "null"], "minimum": 0
                                },
                                "end_offset": {
                                    "type": ["integer", "null"], "minimum": 0
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
        for record in [
            *packet["task_input"]["source_documents"],
            *packet["reference_material"]["evidence_records"],
        ]:
            snippet: Any = record.get("snippet")
            if not isinstance(snippet, str):
                continue
            for field in ("evidence_id", "document_id"):
                identity: Any = record.get(field)
                if isinstance(identity, str):
                    texts.setdefault(identity, set()).add(snippet)
        for span in assignment.evidence_spans:
            sources: set[str] = texts.get(span.source_ref, set())
            if len(sources) != 1:
                raise ValueError("Evidence cites an unavailable or ambiguous source.")
            source: str = next(iter(sources))
            if span.quote is not None and span.quote not in source:
                raise ValueError("Evidence quotation is not verbatim.")
            if span.start_offset is not None and span.end_offset is not None:
                if span.end_offset > len(source):
                    raise ValueError("Evidence offsets exceed the immutable source.")
                if (
                    span.quote is not None
                    and source[span.start_offset:span.end_offset] != span.quote
                ):
                    raise ValueError("Evidence quotation and offsets disagree.")

    def _validated_assignment(
        self, assignment: LabelAssignment, taxonomy: TaxonomyVersion
    ) -> LabelAssignment:
        return LabelAssignment.model_validate(
            assignment.model_dump(mode="python"),
            context={
                "taxonomy": taxonomy,
                "require_taxonomy": True,
                "require_frozen_taxonomy": True,
                "require_evidence_for_each_label": True,
            },
        )

    def _round(
        self,
        assignments: list[LabelAssignment],
        round_index: int,
    ) -> tuple[str, TaxonomyVersion, dict[str, LabelAssignment]]:
        self._policy()
        panel: set[str] = {item["actor_id"] for item in self._panel()}
        if not isinstance(assignments, list) or len(assignments) != len(panel):
            raise _OperationalException("A round requires exactly one valid assignment per selected agent.")
        responses: set[str] = {item.response_id for item in assignments}
        taxonomy_ids: set[str] = {item.taxonomy_id for item in assignments}
        actors: list[str] = [item.actor_id for item in assignments]
        if (
            len(responses) != 1
            or len(taxonomy_ids) != 1
            or set(actors) != panel
            or len(set(actors)) != len(actors)
        ):
            raise _OperationalException("Round membership, response, or taxonomy is inconsistent.")
        taxonomy_id: str = next(iter(taxonomy_ids))
        found: list[TaxonomyVersion] = []
        for kind in _TAXONOMY_KINDS:
            try:
                found.append(TaxonomyVersion.model_validate(
                    self._store.read_record(kind, taxonomy_id)
                ))
            except KeyError:
                pass
        if len(found) != 1:
            raise ValueError("Round taxonomy is missing or ambiguous.")
        taxonomy: TaxonomyVersion = self._taxonomy(found[0])
        validated: dict[str, LabelAssignment] = {}
        for item in assignments:
            assignment: LabelAssignment = self._validated_assignment(item, taxonomy)
            if (
                assignment.round_index != round_index
                or assignment.origin != AssignmentOrigin.AGENT
            ):
                raise _OperationalException("Wrong round index or non-agent round assignment.")
            if assignment.uncertain or not assignment.labels:
                raise _OperationalException(
                    "Explicit uncertainty or empty labels block selector evaluation.",
                    response_id=assignment.response_id,
                    round_index=round_index,
                    artifact_ids=[assignment.assignment_id],
                )
            validated[assignment.actor_id] = assignment
        return next(iter(responses)), taxonomy, validated

    def _stored_agent(
        self, provenance: dict[str, Any], taxonomy: TaxonomyVersion
    ) -> LabelAssignment:
        identity: str = provenance["assignment_id"]
        assignment = LabelAssignment.model_validate(
            self._store.read_record("label_assignment", identity)
        )
        self._immutable("label_assignment", identity)
        return self._validated_assignment(assignment, taxonomy)

    def _generate_round(
        self,
        response_id: str,
        taxonomy: TaxonomyVersion,
        round_index: int,
        packet: dict[str, Any],
        payloads: dict[str, dict[str, Any]],
        snapshot_id: str | None = None,
    ) -> list[LabelAssignment]:
        template: str | None = (
            self._config.categorization.prompt_templates.round_one
            if round_index == 1
            else self._config.categorization.prompt_templates.round_two
        )
        if round_index == 2 and self._single_agent is not None:
            template = str(self._single_reconsideration_template())
        if template is None:
            raise ValueError("Categorization prompt path is unresolved.")
        policy: dict[str, Any] = self._policy()
        panel: list[dict[str, str]] = self._panel()
        packet_hash: str = _content_hash(packet)
        requests: list[dict[str, Any]] = []
        missing: list[str] = []
        assignments: dict[str, LabelAssignment] = {}
        for actor in panel:
            actor_id: str = actor["actor_id"]
            candidates: list[dict[str, Any]] = [
                item for item in self._events("categorization_agent_prediction")
                if item["response_id"] == response_id
                and item["taxonomy_id"] == taxonomy.taxonomy_id
                and item["round_index"] == round_index
                and item["actor_id"] == actor_id
                and item["packet_hash"] == packet_hash
                and item.get("first_round_snapshot_id") == snapshot_id
            ]
            if len(candidates) > 1:
                raise ValueError("Agent prediction lineage is ambiguous.")
            if candidates:
                assignments[actor_id] = self._stored_agent(candidates[0], taxonomy)
                continue
            prompt: str = self._models.render_prompt(template, payloads[actor_id])
            requests.append(
                {
                    "model_id": actor_id,
                    "prompt": prompt,
                    "run_index": policy["assignment_run_index"],
                    "schema_name": f"rq3_round_{round_index}",
                    "instance_id": packet["task_input"]["instance_id"],
                    "schema_contract": self._contract(taxonomy),
                }
            )
            missing.append(actor_id)

        outputs: list[ModelResponse] = (
            self._models.generate_batch(requests) if requests else []
        )
        if len(outputs) != len(missing):
            raise _OperationalException(
                "Model client returned an incomplete round.",
                response_id=response_id,
                round_index=round_index,
            )
        errors: list[str] = []
        for actor_id, output in zip(missing, outputs, strict=True):
            if (
                output.status != ResponseStatus.SUCCESS
                or output.model_id != actor_id
                or output.instance_id != packet["task_input"]["instance_id"]
            ):
                event = self._event(
                    "categorization_agent_exception",
                    {
                        "response_id": response_id,
                        "taxonomy_id": taxonomy.taxonomy_id,
                        "round_index": round_index,
                        "actor_id": actor_id,
                        "packet_hash": packet_hash,
                        "model_response_id": output.response_id,
                        "status": output.status.value,
                    },
                )
                errors.append(event["manifest_id"])
                continue
            try:
                parsed: dict[str, Any] = output.parsed_answer
                if set(parsed) != {"labels", "evidence_spans", "uncertain"}:
                    raise ValueError("Agent output has an incompatible field set.")
                content: dict[str, Any] = {
                    "response_id": response_id,
                    "taxonomy_id": taxonomy.taxonomy_id,
                    "actor_id": actor_id,
                    "round_index": round_index,
                    "labels": parsed["labels"],
                    "evidence_spans": parsed["evidence_spans"],
                    "uncertain": parsed["uncertain"],
                    "origin": AssignmentOrigin.AGENT.value,
                    "supersedes_id": None,
                }
                identity: str = _content_hash(
                    {
                        "assignment": content,
                        "model_response_id": output.response_id,
                        "protocol_id": self._protocol_id,
                        "first_round_snapshot_id": snapshot_id,
                    }
                )
                assignment = self._validated_assignment(
                    LabelAssignment.model_validate({"assignment_id": identity, **content}),
                    taxonomy,
                )
                self._validate_spans(assignment, packet)
                self._store.create_record(
                    "label_assignment", identity, assignment.model_dump(mode="json")
                )
                self._store.freeze_records("label_assignment", [identity])
                self._event(
                    "categorization_agent_prediction",
                    {
                        "response_id": response_id,
                        "taxonomy_id": taxonomy.taxonomy_id,
                        "guideline_hash": taxonomy.guideline_hash,
                        "actor_id": actor_id,
                        "round_index": round_index,
                        "assignment_id": identity,
                        "model_response_id": output.response_id,
                        "request_hash": output.usage.get("request_hash"),
                        "packet_hash": packet_hash,
                        "first_round_snapshot_id": snapshot_id,
                    },
                )
                assignments[actor_id] = assignment
            except (ValueError, TypeError, KeyError) as exc:
                event = self._event(
                    "categorization_agent_exception",
                    {
                        "response_id": response_id,
                        "taxonomy_id": taxonomy.taxonomy_id,
                        "round_index": round_index,
                        "actor_id": actor_id,
                        "packet_hash": packet_hash,
                        "model_response_id": output.response_id,
                        "status": "invalid_assignment",
                        "error_class": type(exc).__name__,
                    },
                )
                errors.append(event["manifest_id"])
        if errors:
            raise _OperationalException(
                "Unusable agent outputs remain operational exceptions, not consensus.",
                response_id=response_id,
                round_index=round_index,
                artifact_ids=errors,
            )
        result: list[LabelAssignment] = [
            assignments[item["actor_id"]] for item in panel
        ]
        self._round(result, round_index)
        return result

    def run_round_one(
        self, response_id: str, taxonomy: TaxonomyVersion
    ) -> list[LabelAssignment]:
        """Generate missing independent predictions over one identical packet."""
        self._config.validate("categorize")
        self._policy()
        taxonomy = self._taxonomy(taxonomy)
        packet: dict[str, Any] = self._packet(response_id, taxonomy)
        shared: dict[str, Any] = self._payload(packet)
        actors: list[dict[str, str]] = self._panel()
        self._models.validate_endpoints([item["actor_id"] for item in actors])
        return self._generate_round(
            response_id,
            taxonomy,
            1,
            packet,
            {item["actor_id"]: copy.deepcopy(shared) for item in actors},
        )

    def run_round_two(
        self,
        response_id: str,
        taxonomy: TaxonomyVersion,
        first_round: list[LabelAssignment],
    ) -> list[LabelAssignment]:
        """Reconsider using a complete, immutable first-round peer snapshot."""
        self._config.validate("categorize")
        bound_response, stored_taxonomy, first = self._round(first_round, 1)
        if (
            bound_response != response_id
            or stored_taxonomy.taxonomy_id != taxonomy.taxonomy_id
        ):
            raise ValueError("Round-two inputs bind a different response or taxonomy.")
        taxonomy = self._taxonomy(taxonomy)
        packet: dict[str, Any] = self._packet(response_id, taxonomy)
        packet_hash: str = _content_hash(packet)
        assignment_ids: list[str] = sorted(item.assignment_id for item in first.values())
        for assignment in first.values():
            stored = self._store.read_record("label_assignment", assignment.assignment_id)
            if _canonical_json(stored) != _canonical_json(assignment):
                raise ValueError("First-round assignment differs from persisted content.")
            self._immutable("label_assignment", assignment.assignment_id)
            lineage = [
                event
                for event in self._events("categorization_agent_prediction")
                if event["assignment_id"] == assignment.assignment_id
                and event["packet_hash"] == packet_hash
                and event["round_index"] == 1
            ]
            if len(lineage) != 1:
                raise ValueError("Round-one packet lineage is missing or changed.")
        freeze_hash: str = self._store.freeze_records(
            "label_assignment", assignment_ids
        )
        snapshot: dict[str, Any] = self._event(
            "categorization_first_round_snapshot",
            {
                "response_id": response_id,
                "taxonomy_id": taxonomy.taxonomy_id,
                "guideline_hash": taxonomy.guideline_hash,
                "classification_packet_hash": packet_hash,
                "assignment_ids": assignment_ids,
                "assignment_freeze_manifest": freeze_hash,
            },
        )
        template: str | None = self._config.categorization.prompt_templates.round_two
        if template is None:
            raise ValueError("Round-two prompt path is unresolved.")
        payloads: dict[str, dict[str, Any]] = {}
        for actor in self._panel():
            payload: dict[str, Any] = self._payload(packet)
            if self._single_agent is not None:
                payload["single_agent_reconsideration"] = {
                    **self._reconsideration,
                    "actor_id": self._single_agent,
                    "first_round_manifest_id": snapshot["manifest_id"],
                }
                if self._reconsideration["mode"] == "self_review":
                    previous = first[self._single_agent]
                    payload["single_agent_reconsideration"]["previous_prediction"] = {
                        "labels": list(previous.labels),
                        "evidence_spans": [span.model_dump(mode="json") for span in previous.evidence_spans],
                        "uncertain": previous.uncertain,
                    }
                payloads[actor["actor_id"]] = payload
                continue
            payload["first_round"] = [
                item.model_dump(mode="json")
                for item in sorted(first.values(), key=lambda item: item.assignment_id)
            ]
            payload["round_two_context"] = {
                "recipient_actor_id": actor["actor_id"],
                "round_index": 2,
                "template_path": template,
                "first_round_manifest_hash": snapshot["manifest_id"],
                "classification_packet_hash": packet_hash,
                "first_round_classification_packet_hash": packet_hash,
                "first_round_records_verified_immutable": True,
                "panel": self._panel(),
            }
            payloads[actor["actor_id"]] = payload
        return self._generate_round(
            response_id, taxonomy, 2, packet, payloads, snapshot["manifest_id"]
        )

    def _selector(
        self,
        first_round: list[LabelAssignment],
        second_round: list[LabelAssignment],
    ) -> dict[str, Any]:
        response_id, taxonomy, first = self._round(first_round, 1)
        second_response, second_taxonomy, second = self._round(second_round, 2)
        if (
            response_id != second_response
            or taxonomy.taxonomy_id != second_taxonomy.taxonomy_id
            or set(first) != set(second)
        ):
            raise _OperationalException("Round identities do not match.")
        disagreement: bool = len({
            frozenset(item.labels) for item in first.values()
        }) > 1
        changed: list[str] = sorted(
            actor for actor in first
            if set(first[actor].labels) != set(second[actor].labels)
        )
        reasons: list[str] = []
        if disagreement:
            reasons.append("first_round_disagreement")
        if changed:
            reasons.append("cross_round_label_change")
        return {
            "response_id": response_id,
            "taxonomy_id": taxonomy.taxonomy_id,
            "guideline_hash": taxonomy.guideline_hash,
            "first_round_assignment_ids": sorted(
                item.assignment_id for item in first.values()
            ),
            "second_round_assignment_ids": sorted(
                item.assignment_id for item in second.values()
            ),
            "first_round_disagreement": disagreement,
            "changed_actor_ids": changed,
            "requires_review": disagreement or bool(changed),
            "review_reasons": reasons,
            "selector_definition": "round_one_set_disagreement_or_any_actor_set_change",
        }

    def requires_review(
        self,
        first_round: list[LabelAssignment],
        second_round: list[LabelAssignment],
    ) -> bool:
        """Apply exactly the paper's disagreement-or-cross-round-change selector."""
        return bool(self._selector(first_round, second_round)["requires_review"])

    def _human_reference(
        self, response_id: str, taxonomy: TaxonomyVersion
    ) -> dict[str, Any]:
        protocol: dict[str, Any] = self._annotation_protocol(taxonomy)
        try:
            reference: dict[str, Any] = self._annotations.get_reference(
                response_id, _OPERATIONAL_PURPOSE,
                protocol_hash=_content_hash(protocol), taxonomy_id=taxonomy.taxonomy_id,
            )
        except LookupError:
            raise _AdjudicationPending(
                "Operational adjudication is not complete."
            ) from None
        if (
            reference.get("response_id") != response_id
            or reference.get("purpose") != _OPERATIONAL_PURPOSE
            or reference.get("audit_only") is not False
            or reference.get("cohort_id") is not None
            or reference.get("taxonomy_id") != taxonomy.taxonomy_id
            or reference.get("guideline_hash") != taxonomy.guideline_hash
            or reference.get("protocol_hash") != _content_hash(protocol)
            or reference.get("uncertain", True)
            or reference.get("origin") != "adjudicated"
        ):
            raise ValueError("Human reference is not eligible operational adjudication.")
        annotation_ids: list[str] = _ids(
            reference.get("annotation_ids"), "Adjudication reference IDs"
        )
        for identity in annotation_ids:
            record = self._store.read_record("adjudication", identity)
            self._immutable("adjudication", identity)
            if (
                record.get("response_id") != response_id
                or record.get("audit_only") is not False
                or record.get("cohort_id") is not None
                or record.get("reference_purpose") != _OPERATIONAL_PURPOSE
                or record.get("taxonomy_id") != taxonomy.taxonomy_id
                or record.get("guideline_hash") != taxonomy.guideline_hash
            ):
                raise ValueError("Audit or incompatible records cannot correct operational labels.")
        return reference

    def finalize(
        self,
        response_id: str,
        first_round: list[LabelAssignment],
        second_round: list[LabelAssignment],
    ) -> LabelAssignment:
        """Create an append-only consensus or human-finalized label assignment."""
        selector: dict[str, Any] = self._selector(first_round, second_round)
        if selector["response_id"] != response_id:
            raise ValueError("Finalization response differs from the agent rounds.")
        _, taxonomy, first = self._round(first_round, 1)
        _, _, second = self._round(second_round, 2)
        packet: dict[str, Any] = self._packet(response_id, taxonomy)
        lineage: list[str] = [
            *selector["first_round_assignment_ids"],
            *selector["second_round_assignment_ids"],
        ]
        for assignment in [*first.values(), *second.values()]:
            stored = self._store.read_record("label_assignment", assignment.assignment_id)
            if _canonical_json(stored) != _canonical_json(assignment):
                raise ValueError("Cannot finalize an unpersisted or altered prediction.")
        self._store.freeze_records("label_assignment", sorted(lineage))

        reference: dict[str, Any] | None = None
        if selector["requires_review"]:
            reference = self._human_reference(response_id, taxonomy)
            labels: list[str] = reference["labels"]
            spans: list[Any] = reference["evidence_spans"]
            actor_ids: set[str] = {
                self._store.read_record("adjudication", identity)["annotator_id"]
                for identity in reference["annotation_ids"]
            }
            if len(actor_ids) != 1:
                raise ValueError("Final adjudicator identity is ambiguous.")
            actor_id: str = next(iter(actor_ids))
            origin = AssignmentOrigin.OPERATIONAL_ADJUDICATION
        else:
            labels = sorted(next(iter(first.values())).labels)
            unique_spans: dict[str, Any] = {}
            for assignment in [*first.values(), *second.values()]:
                for span in assignment.evidence_spans:
                    value: dict[str, Any] = span.model_dump(mode="json")
                    unique_spans[_canonical_json(value)] = value
            spans = [unique_spans[key] for key in sorted(unique_spans)]
            actor_id = _CONSENSUS_ACTOR
            origin = AssignmentOrigin.CONSENSUS

        content: dict[str, Any] = {
            "response_id": response_id,
            "taxonomy_id": taxonomy.taxonomy_id,
            "actor_id": actor_id,
            "round_index": 2,
            "labels": labels,
            "evidence_spans": spans,
            "uncertain": False,
            "origin": origin.value,
            "supersedes_id": None,
        }
        identity: str = _content_hash(
            {
                "final_assignment": content,
                "selector": selector,
                "protocol_id": self._protocol_id,
                "human_reference": reference,
            }
        )
        assignment = self._validated_assignment(
            LabelAssignment.model_validate({"assignment_id": identity, **content}),
            taxonomy,
        )
        if not assignment.labels:
            raise ValueError("An empty final categorization cannot be silently accepted.")
        self._validate_spans(assignment, packet)
        self._store.create_record(
            "final_assignment", identity, assignment.model_dump(mode="json")
        )
        self._store.freeze_records("final_assignment", [identity])
        self._event(
            "categorization_finalization",
            {
                "response_id": response_id,
                "taxonomy_id": taxonomy.taxonomy_id,
                "guideline_hash": taxonomy.guideline_hash,
                "executed_mechanisms": self._mechanisms(),
                "assignment_id": identity,
                "selector": selector,
                "packet_hash": _content_hash(packet),
                "human_reference": reference,
                "pre_correction_assignment_ids": sorted(lineage),
            },
        )
        return assignment

    def _review_tasks(
        self, response_id: str, taxonomy: TaxonomyVersion
    ) -> list[AnnotationTask]:
        protocol: dict[str, Any] = self._annotation_protocol(taxonomy)
        tasks: list[AnnotationTask] = self._annotations.create_tasks(
            [response_id], _OPERATIONAL_PURPOSE, protocol
        )
        self._event(
            "categorization_operational_review",
            {
                "response_id": response_id,
                "taxonomy_id": taxonomy.taxonomy_id,
                "guideline_hash": taxonomy.guideline_hash,
                "protocol_hash": _content_hash(protocol),
                "annotation_task_ids": sorted(
                    task.annotation_task_id for task in tasks
                ),
            },
        )
        return tasks

    def run(
        self, failure_ids: list[str], taxonomy: TaxonomyVersion
    ) -> dict[str, Any]:
        """Advance a frozen response cohort without blocking for human annotation."""
        self._config.validate("categorize")
        self._policy()
        taxonomy = self._taxonomy(taxonomy)
        if not isinstance(failure_ids, list):
            raise TypeError("failure_ids must contain evaluated response identifiers.")
        population: list[str] = sorted({_identifier(item) for item in failure_ids})
        if not population:
            raise ValueError("Categorization requires a nonempty failure population.")
        protocol: dict[str, Any] = self._annotation_protocol(taxonomy)
        self._models.validate_endpoints(
            [item["actor_id"] for item in self._panel()]
        )

        members: list[dict[str, Any]] = []
        for response_id in population:
            response, instance, decision_id = self._input(response_id)
            if (
                response_id not in protocol.get("assignments", {})
                and "*" not in protocol.get("assignments", {})
            ):
                raise ValueError("Operational adjudication assignments do not cover the cohort.")
            packet: dict[str, Any] = self._packet(response_id, taxonomy)
            members.append(
                {
                    "response_id": response_id,
                    "response_hash": _content_hash(response),
                    "instance_id": instance.instance_id,
                    "instance_hash": instance.content_hash,
                    "failure_decision_id": decision_id,
                    "classification_packet_hash": _content_hash(packet),
                }
            )
        population_manifest: dict[str, Any] = self._event(
            "categorization_population",
            {
                "taxonomy_id": taxonomy.taxonomy_id,
                "guideline_hash": taxonomy.guideline_hash,
                "members": members,
                "workload_denominator": "frozen_requested_response_cohort",
            },
        )
        execution_id: str = _content_hash(
            {
                "protocol_id": self._protocol_id,
                "population_manifest_id": population_manifest["manifest_id"],
            }
        )
        previous: list[StageState] = [
            StageState.model_validate(item)
            for item in self._store.list_records("stage_state")
            if item.get("stage") == "categorize"
            and item.get("execution_id") == execution_id
        ]
        completed_states: list[StageState] = [
            item for item in previous if item.status == StageStatus.COMPLETED
        ]
        if completed_states:
            return copy.deepcopy(completed_states[0].cursor["result"])

        starts: list[datetime] = [
            item.started_at for item in previous if item.started_at is not None
        ]
        started: datetime = min(starts) if starts else _now()
        finals: dict[str, str] = {}
        selectors: dict[str, dict[str, Any]] = {}
        waiting_humans: list[str] = []
        operational_errors: dict[str, dict[str, Any]] = {}
        pending_tasks: set[str] = set()
        human_finalized: set[str] = set()

        def result_snapshot() -> dict[str, Any]:
            flagged: list[str] = sorted(
                key for key, value in selectors.items() if value["requires_review"]
            )
            complete: bool = len(finals) == len(population)
            status: StageStatus = (
                StageStatus.COMPLETED if complete
                else StageStatus.WAITING_FOR_ANNOTATIONS if waiting_humans
                else StageStatus.BLOCKED if operational_errors
                else StageStatus.RUNNING
            )
            return {
                "status": status.value,
                "execution_id": execution_id,
                "protocol_id": self._protocol_id,
                "population_manifest_id": population_manifest["manifest_id"],
                "taxonomy_id": taxonomy.taxonomy_id,
                "guideline_hash": taxonomy.guideline_hash,
                "final_assignment_ids_by_response": dict(sorted(finals.items())),
                "finalized_response_ids": sorted(finals),
                "awaiting_model_response_ids": sorted(operational_errors),
                "awaiting_human_response_ids": sorted(waiting_humans),
                "pending_annotation_task_ids": sorted(pending_tasks),
                "operational_exceptions": copy.deepcopy(operational_errors),
                "review_triggers": {
                    key: value["review_reasons"]
                    for key, value in sorted(selectors.items())
                    if value["requires_review"]
                },
                "selector_manifest_ids": {
                    key: value["manifest_id"]
                    for key, value in sorted(selectors.items())
                },
                "coverage": {
                    "intended_responses": len(population),
                    "selector_evaluable_responses": len(selectors),
                    "finalized_responses": len(finals),
                    "complete": complete,
                },
                "workload": {
                    "denominator_definition": "frozen_requested_response_cohort",
                    "intended_cohort_size": len(population),
                    "selector_evaluable_responses": len(selectors),
                    "flagged_unique_responses": len(flagged),
                    "flagged_response_ids": flagged,
                    "completed_human_adjudications": len(human_finalized),
                    "pending_or_failed_model_responses": len(operational_errors),
                    "review_fraction": (
                        len(flagged) / len(population)
                        if len(selectors) == len(population) else None
                    ),
                    "provisional_review_fraction_among_evaluable": (
                        len(flagged) / len(selectors) if selectors else None
                    ),
                    "rate_is_provisional": len(selectors) != len(population),
                    "paper_reported_comparison_upper_bound": (
                        self._config.categorization
                        .reported_human_verification_fraction_upper_bound
                    ),
                    "reported_bound_enforced_as_cap": False,
                    "independent_audit_work_included": False,
                },
            }

        def checkpoint(result: dict[str, Any], output_id: str | None = None) -> None:
            status = StageStatus(result["status"])
            if status == StageStatus.COMPLETED and output_id is None:
                # Publish completion only after its immutable result manifest exists.
                return
            timestamp: datetime = _now()
            state = StageState(
                stage="categorize",
                execution_id=execution_id,
                config_hash=self._config.config_hash,
                status=status,
                completed_work=sorted(finals),
                pending_work=(
                    [] if status == StageStatus.COMPLETED
                    else sorted(set(population) - set(finals))
                ),
                outstanding_annotation_task_ids=sorted(pending_tasks),
                cursor={"result": result},
                input_manifests={
                    "population": population_manifest["manifest_id"],
                    "taxonomy": taxonomy.taxonomy_id,
                },
                output_manifests={"categorization": output_id} if output_id else {},
                started_at=started,
                updated_at=timestamp,
                completed_at=timestamp if status == StageStatus.COMPLETED else None,
                metadata={
                    "protocol_id": self._protocol_id,
                    "independent_audits_excluded": True,
                },
            )
            self._store.set_stage_state("categorize", state.model_dump(mode="json"))

        for response_id in population:
            try:
                first: list[LabelAssignment] = self.run_round_one(response_id, taxonomy)
                second: list[LabelAssignment] = self.run_round_two(
                    response_id, taxonomy, first
                )
                selector: dict[str, Any] = self._event(
                    "categorization_selector", self._selector(first, second)
                )
                selectors[response_id] = selector
                if selector["requires_review"]:
                    try:
                        self._human_reference(response_id, taxonomy)
                    except _AdjudicationPending:
                        tasks: list[AnnotationTask] = self._review_tasks(
                            response_id, taxonomy
                        )
                        waiting_humans.append(response_id)
                        pending_tasks.update(
                            task.annotation_task_id for task in tasks
                            if task.status.value in {"draft", "pending", "in_progress"}
                        )
                        # An unresolved first adjudication may have escalated.
                        for task_payload in self._store.list_records(
                            "annotation_task", {"response_id": response_id}
                        ):
                            if (
                                task_payload.get("purpose") == "senior_escalation"
                                and task_payload.get("cohort_id") is None
                                and task_payload.get("guideline_hash") == taxonomy.guideline_hash
                                and task_payload.get("status") in {"draft", "pending", "in_progress"}
                            ):
                                pending_tasks.add(task_payload["annotation_task_id"])
                        if not pending_tasks:
                            raise ValueError(
                                "Pending operational adjudication lacks an open review task."
                            )
                        checkpoint(result_snapshot())
                        continue
                final: LabelAssignment = self.finalize(response_id, first, second)
                finals[response_id] = final.assignment_id
                if final.origin == AssignmentOrigin.OPERATIONAL_ADJUDICATION:
                    human_finalized.add(response_id)
            except _OperationalException as exc:
                operational_errors[response_id] = {
                    "reason": str(exc),
                    "round_index": exc.round_index,
                    "artifact_ids": exc.artifact_ids,
                    "paper_selector_evaluated": False,
                }
            checkpoint(result_snapshot())

        result: dict[str, Any] = result_snapshot()
        output: dict[str, Any] = self._event(
            "categorization_execution_result", {"result": result}
        )
        result["categorization_manifest_id"] = output["manifest_id"]
        checkpoint(result, output["manifest_id"])
        return result
