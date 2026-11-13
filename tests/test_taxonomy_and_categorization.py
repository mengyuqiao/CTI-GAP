"""Offline regression tests for taxonomy discovery, categorization, and defenses.

Fixture policies are explicitly documented implementation choices, not settings
attributed to the paper. Real configuration readiness is tested separately from
component behavior.

Operational annotation protocols embed config_hash inside configuration policy
mappings. The implementation also hashes those mappings, creating a self-
referential identity requirement. Component fixtures therefore use an explicit
fixed configuration identity and mock only stage-readiness checks. They do not
claim to be executable scientific manifests.

Inference is scripted through the designed ModelClient interfaces. Real Jinja
rendering, schemas, persistence, taxonomy logic, selectors, metrics, and
counterfactual calculations remain under test. No network, credentials, GPU,
actual annotator, or external model is required.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm
import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from annotation_app import AnnotationService
from categorization import Categorizer
from config import ExperimentConfig
from evaluation import Evaluator
from experiments import ExperimentRunner
from model_client import ModelClient
from schemas import (
    AnnotationTask,
    AssignmentOrigin,
    AuditCohort,
    FailureDecision,
    LabelAssignment,
    ModelResponse,
    ResponseStatus,
    TaskInstance,
    TaxonomyVersion,
    _canonical_json,
    _content_hash,
    _task_content_projection,
    _text_hash,
)
from storage import ArtifactStore
from taxonomy import TaxonomyManager


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"
FIXTURE_CONFIG_HASH: str = "6" * 64
CORPUS_VERSION: str = "fictional-taxonomy-corpus-v1"
GUIDELINE: dict[str, Any] = {
    "version": "fictional-guideline-v1",
    "instruction": "Assign only independently evidenced fictional discrepancies.",
}
GUIDELINE_HASH: str = _content_hash(GUIDELINE)
REFERENCE_TEXT: str = "Reference actor is Correct."
INPUT_TEXT: str = "Mechanism evidence."
WRONG_TEXT: str = "Wrong actor"

Script: Any = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def _baseline() -> dict[str, Any]:
    assert CONFIG_PATH.is_file(), "The supplied config.yaml must be available."
    payload: Any = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return copy.deepcopy(payload)


def _resolve(payload: dict[str, Any], path: str, value: Any) -> None:
    """Declare a test-only resolution in the existing provenance mechanism."""
    current: dict[str, Any] = payload
    parts: list[str] = path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = copy.deepcopy(value)
    payload.setdefault("implementation_choices", {})[path] = {
        "configuration_path": path,
        "origin": "implementation",
        "rationale": (
            "Synthetic offline regression-test policy only; not an author setting."
        ),
        "affected_experiments": ["all"],
        "selected_value": copy.deepcopy(value),
    }


def _span(label: str | None = None) -> dict[str, Any]:
    return {
        "source_ref": "fixture-reference",
        "quote": REFERENCE_TEXT,
        "subtype_id": label,
        "relation_type": "fictional-supported-discrepancy",
    }


def _definition(label: str) -> dict[str, Any]:
    return {
        "subtype_id": label,
        "name": f"Fictional subtype {label}",
        "definition": f"Observable fictional discrepancy represented by {label}.",
        "evidence_requirements": ["A discrepancy against applicable reference evidence."],
        "supporting_annotation_ids": [],
    }


def _rules() -> dict[str, Any]:
    return {
        "inclusion": ["The specified discrepancy is observable."],
        "exclusion": ["The required discrepancy is absent."],
        "boundary_rules": ["Do not infer hidden internal causes."],
        "permitted_cooccurring_labels": [],
    }


def _taxonomy(
    identity: str = "fixture-frozen-taxonomy",
    labels: tuple[str, ...] = ("fixture-A", "fixture-B"),
    *,
    frozen: bool = True,
) -> TaxonomyVersion:
    return TaxonomyVersion.model_validate(
        {
            "taxonomy_id": identity,
            "iteration": 0,
            "guideline_hash": GUIDELINE_HASH,
            "subtype_definitions": [_definition(label) for label in labels],
            "inclusion_exclusion_rules": {label: _rules() for label in labels},
            "frozen": frozen,
        }
    )


def _proposal(labels: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "new_subtypes": [_definition(label) for label in labels],
        "rules": {label: _rules() for label in labels},
        "no_new_categories": not labels,
    }


class FixedIdentityConfig(ExperimentConfig):
    """Avoid a configuration-hash fixed point in isolated component fixtures."""

    @property
    def config_hash(self) -> str:
        return FIXTURE_CONFIG_HASH


@dataclass
class ScriptedAnnotations:
    """Small, persistent substitute for the designed AnnotationService interface."""

    store: ArtifactStore
    protocols: dict[str, dict[str, Any]] = field(default_factory=dict)
    references: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    requests: list[tuple[list[str], str]] = field(default_factory=list)

    def create_tasks(
        self, response_ids: list[str], purpose: str, protocol: dict[str, Any]
    ) -> list[AnnotationTask]:
        self.requests.append((sorted(response_ids), purpose))
        result: list[AnnotationTask] = []
        for response_id in sorted(response_ids):
            identity: str = _content_hash([response_id, purpose, protocol])
            try:
                task = AnnotationTask.model_validate(
                    self.store.read_record("annotation_task", identity)
                )
            except KeyError:
                response = ModelResponse.model_validate(
                    self.store.read_record("model_response", response_id)
                )
                instance = TaskInstance.model_validate(
                    self.store.read_record("task_instance", response.instance_id)
                )
                task = AnnotationTask(
                    annotation_task_id=identity,
                    response_id=response_id,
                    purpose=purpose,
                    assigned_annotator=protocol["adjudication"]["adjudicator_id"],
                    guideline_hash=protocol["guideline_hash"],
                    visible_payload={
                        "task_input": {
                            "instance_id": instance.instance_id,
                            "source_documents": [
                                item.model_dump(mode="json")
                                for item in instance.evidence_records
                            ],
                        },
                        "evaluated_response": {
                            "response_id": response_id,
                            "raw_text": response.raw_text,
                        },
                        "reference_material": instance.reference,
                    },
                )
                self.store.create_record(
                    "annotation_task", identity, task.model_dump(mode="json")
                )
            self.protocols[identity] = copy.deepcopy(protocol)
            result.append(task)
        return result

    def create_adjudications(self, task_ids: list[str]) -> list[AnnotationTask]:
        # Fixture groups have one scripted, designated reference. No majority
        # or automatic reconciliation is introduced by this substitute.
        for identity in task_ids:
            self.store.read_record("annotation_task", identity)
        return []

    def get_reference(
        self, response_id: str, purpose: str, *,
        protocol_hash: str | None = None, taxonomy_id: str | None = None,
    ) -> dict[str, Any]:
        key: tuple[str, str] = (response_id, purpose)
        if key not in self.references:
            raise LookupError("Scripted human reference is incomplete.")
        if protocol_hash is not None and self.references[key]["protocol_hash"] != protocol_hash:
            raise LookupError("Scripted human reference belongs to another protocol.")
        return copy.deepcopy(self.references[key])

    def submit_annotation(
        self, task_id: str, annotator_id: str, payload: dict[str, Any]
    ) -> None:
        self._submit(task_id, annotator_id, payload, adjudication=False)

    def resolve_task(
        self, task_id: str, adjudicator_id: str, payload: dict[str, Any]
    ) -> None:
        self._submit(task_id, adjudicator_id, payload, adjudication=True)

    def _submit(
        self,
        task_id: str,
        actor: str,
        payload: dict[str, Any],
        *,
        adjudication: bool,
    ) -> None:
        task = AnnotationTask.model_validate(
            self.store.read_record("annotation_task", task_id)
        )
        if actor != task.assigned_annotator:
            raise PermissionError("Scripted task belongs to another principal.")
        protocol: dict[str, Any] = self.protocols[task_id]
        identity: str = _content_hash([task_id, payload, adjudication])
        # Complete the mock task projection before inserting a record that
        # references it. The actual AnnotationService lifecycle is tested in the
        # dedicated annotation/audit test module.
        if task.status.value in {"pending", "in_progress", "draft"}:
            self.store.update_draft(
                "annotation_task",
                task_id,
                {"status": "adjudicated" if adjudication else "submitted"},
            )
        record: dict[str, Any] = {
            "annotation_id": identity,
            "annotation_task_id": task_id,
            "response_id": task.response_id,
            "purpose": task.purpose.value,
            "reference_purpose": task.purpose.value,
            "cohort_id": None,
            "annotator_id": actor,
            "guideline_hash": task.guideline_hash,
            "taxonomy_id": protocol["taxonomy_id"],
            "protocol_hash": _content_hash(protocol),
            "config_hash": FIXTURE_CONFIG_HASH,
            "audit_only": False,
            "uncertain": False,
            "evidence_spans": [_span(label) for label in payload.get("labels", [])]
            or [_span()],
            "status": "adjudicated" if adjudication else "submitted",
            **copy.deepcopy(payload),
        }
        kind: str = "adjudication" if adjudication else "annotation"
        self.store.create_record(kind, identity, record)
        self.store.freeze_records(kind, [identity])
        self.references[(task.response_id, task.purpose.value)] = {
            **record,
            "origin": "adjudicated" if adjudication else "independent_agreement",
            "annotation_ids": [identity],
        }


@dataclass
class ScriptedModels:
    """Use real templates and deterministic, request-bound synthetic completions."""

    config: ExperimentConfig
    store: ArtifactStore
    script: Script | None = None
    requests: list[dict[str, Any]] = field(default_factory=list)
    packets: dict[str, dict[str, Any]] = field(default_factory=dict)
    provider_calls: set[str] = field(default_factory=set)
    validated_models: list[list[str]] = field(default_factory=list)

    def validate_endpoints(self, model_ids: list[str]) -> None:
        self.validated_models.append(list(model_ids))

    def render_prompt(self, template: str, payload: dict[str, Any]) -> str:
        rendered: str = ModelClient(self.config, self.store).render_prompt(
            template, payload
        )
        self.packets[_text_hash(rendered)] = copy.deepcopy(payload)
        return rendered

    def generate(
        self, model_id: str, prompt: str, run_index: int, schema_name: str
    ) -> ModelResponse:
        matches: list[dict[str, Any]] = [
            item for item in self.requests
            if item["model_id"] == model_id
            and item["prompt"] == prompt
            and item["run_index"] == run_index
            and item["schema_name"] == schema_name
        ]
        if not matches:
            raise ValueError("Direct mock generation requires an explicit request binding.")
        return self._output(matches[-1])

    def generate_batch(self, requests: list[dict[str, Any]]) -> list[ModelResponse]:
        self.requests.extend(copy.deepcopy(requests))
        return [self._output(request) for request in requests]

    def _output(self, request: dict[str, Any]) -> ModelResponse:
        identity: str = _content_hash(request)
        self.provider_calls.add(identity)
        packet: dict[str, Any] = self.packets[_text_hash(request["prompt"])]
        if self.script is None:
            parsed: dict[str, Any] = {
                "labels": ["fixture-A"],
                "evidence_spans": [_span("fixture-A")],
                "uncertain": False,
            }
        else:
            parsed = self.script(request, packet)
        status: ResponseStatus = ResponseStatus(
            parsed.pop("__fixture_status__", "success")
        )
        probability: Any = parsed.pop("__fixture_probability__", None)
        usage: dict[str, Any] = {}
        errors: list[str] = []
        if probability is not None:
            usage["probability_artifact"] = probability
        if status == ResponseStatus.TRANSPORT_FAILURE:
            usage["execution_error"] = {"exception_class": "FictionalTransportError"}
        if status == ResponseStatus.INVALID_SCHEMA:
            errors = ["Fictional invalid classifier output."]
        output = ModelResponse(
            response_id=identity,
            instance_id=request["instance_id"],
            model_id=request["model_id"],
            run_index=request["run_index"],
            prompt_hash=_text_hash(request["prompt"]),
            config_hash=self.config.config_hash,
            status=status,
            raw_text=_canonical_json(parsed),
            parsed_answer=parsed if status == ResponseStatus.SUCCESS else {},
            schema_errors=errors,
            usage=usage,
        )
        self.store.create_record(
            "model_response", identity, output.model_dump(mode="json")
        )
        return output


@dataclass
class Harness:
    config: FixedIdentityConfig
    store: ArtifactStore
    models: ScriptedModels
    annotations: ScriptedAnnotations
    taxonomy: TaxonomyVersion
    failure_ids: list[str]
    instance_ids: list[str]

    def manager(self) -> TaxonomyManager:
        return TaxonomyManager(
            self.config, self.store, self.models, self.annotations  # type: ignore[arg-type]
        )

    def categorizer(self) -> Categorizer:
        return Categorizer(
            self.config, self.store, self.models, self.annotations  # type: ignore[arg-type]
        )

    def runner(self) -> ExperimentRunner:
        return ExperimentRunner(
            self.config,
            self.store,
            self.models,  # type: ignore[arg-type]
            Evaluator(self.config, self.store),
            self.annotations,  # type: ignore[arg-type]
        )


def _annotation_protocol(
    response_ids: list[str], taxonomy_id: str | None = None
) -> dict[str, Any]:
    return {
        "protocol_version": "fictional-human-protocol-v1",
        "config_hash": FIXTURE_CONFIG_HASH,
        "guideline_hash": GUIDELINE_HASH,
        "guideline_content": GUIDELINE,
        "judgment": "labels" if taxonomy_id is not None else "discovery",
        "cohort_id": None,
        "taxonomy_id": taxonomy_id,
        "annotator_ids": ["fixture-reviewer"],
        "annotators_per_instance": 1,
        "assignments": {
            identity: ["fixture-reviewer"] for identity in response_ids
        },
        "queue_order_policy": "response_id",
        "reference_establishment": "independent_agreement_or_adjudication",
        "correctness_criteria": {},
        "reference_applicability": {
            identity: {"scope": "Fictional contemporaneous attribution evidence."}
            for identity in response_ids
        },
        "adjudication": {
            "adjudicator_id": "fixture-reviewer",
            "senior_adjudicator_id": "fixture-senior",
            "escalate_unresolved_cases_to_senior": True,
            "adjudicator_must_be_independent": False,
        },
    }


def _component_config(
    tmp_path: Path, failures: list[str], taxonomy: TaxonomyVersion
) -> FixedIdentityConfig:
    payload: dict[str, Any] = _baseline()
    payload["experiment_id"] = "fictional-taxonomy-tests"
    payload["corpus_version"] = CORPUS_VERSION
    payload["operational"] = {
        "database_path": str(tmp_path / "records.sqlite"),
        "artifact_root": str(tmp_path / "blobs"),
        "output_path": str(tmp_path / "reports"),
        "taxonomy_mode": "discovery",
        "selected_models": ["G5"],
        "selected_tasks": ["threat_actor_linking"],
    }
    initialization = _annotation_protocol(failures)
    refinement = copy.deepcopy(initialization)
    operational = _annotation_protocol(failures, taxonomy.taxonomy_id)
    settings: dict[str, Any] = {
        "inference.temperature": 0.0,
        "inference.top_p": 1.0,
        "inference.max_output_tokens": 256,
        "inference.context_window": 100000,
        "inference.concurrency": 1,
        "inference.truncation_policy": "reject",
        "inference.retry_policy": {
            "max_attempts": 1,
            "timeout_seconds": 10.0,
            "retryable_status_codes": [],
            "retryable_exception_names": [],
            "backoff_seconds": [],
            "retry_ambiguous_timeouts": False,
        },
        "evaluation.run_seeds": [0, 1, 2, 3, 4],
        "taxonomy.guideline_manifest": "fixture-guideline-manifest",
        "taxonomy_discovery.initial_subset_size": 1,
        "taxonomy_discovery.initialization_seed": 29,
        "taxonomy_discovery.assisting_model": "G5",
        "taxonomy_discovery.prompt_template_path": str(
            PROJECT_ROOT / "prompts/taxonomy_classification.jinja2"
        ),
        "taxonomy_discovery.coverage_threshold": 1.0,
        "taxonomy_discovery.coverage_definition": {
            "mode": "fully_explained",
            "population": "non_initialization",
            "sampling_method": "simple_random_without_replacement",
            "task_input_projection": "evidence_as_documents",
            "assignment_round_index": 0,
            "output_policy": {
                "other_with_known_labels": True,
                "allow_empty_labels_when_uncertain": True,
            },
            "annotation_protocols": {
                "initialization": initialization,
                "refinement": refinement,
            },
        },
        "taxonomy_discovery.stability_threshold_epsilon": 0.0,
        "taxonomy_discovery.revisit_prior_assignments_policy": "all_non_initialization",
        "taxonomy_discovery.persistent_other_policy": {
            "mode": "retain_other",
            "uncertain_cases": "review",
            "on_insufficient_coverage": "block",
            "epsilon_usage": "not_operationalized",
        },
        "categorization.peer_payload_policy": {
            "mode": "labels_only",
            "show_actor_ids": False,
            "show_uncertainty": False,
            "restate_own_prediction": False,
        },
        "categorization.prompt_templates.round_one": str(
            PROJECT_ROOT / "prompts/agent_round_one.jinja2"
        ),
        "categorization.prompt_templates.round_two": str(
            PROJECT_ROOT / "prompts/agent_round_two.jinja2"
        ),
        "categorization.label_set_comparison_policy": {"mode": "set_equality"},
        "categorization.incomplete_agent_output_policy": {
            "mode": "block",
            "explicit_uncertainty": "block",
            "empty_assignment": "block",
            "allow_empty_labels_when_uncertain": True,
            "assignment_run_index": 0,
            "workload_denominator": "frozen_requested_response_cohort",
            "task_input_projection": "evidence_as_documents",
            "annotation_protocol": operational,
        },
        "taxonomy_stability.recovery_match_threshold": 0.5,
        "taxonomy_stability.unmatched_category_policy": "zero",
        "taxonomy_stability.empty_set_policy": "one",
        "taxonomy_stability.prevalence_normalization": "label_mass",
        "taxonomy_stability.jsd_logarithm_base": 2.0,
        "taxonomy_stability.comparison_aggregation": "all_pairs",
    }
    for model_id in payload["models"]["evaluation"]:
        settings[f"models.evaluation.{model_id}.endpoint_or_checkpoint"] = (
            f"fixture/{model_id}"
        )
        settings[f"models.evaluation.{model_id}.revision"] = "fictional-revision"
    for index, entry in enumerate(payload["models"]["categorization_panel"]):
        entry["endpoint_or_checkpoint"] = f"fixture/panel-{index}"
        entry["revision"] = "fictional-revision"
    settings["inference.reasoning_settings"] = {
        "parameters": {},
        "adapters": {
            "G5": {
                "supported_parameters": ["temperature", "top_p", "max_tokens"],
                "api_base": "http://127.0.0.1:1",
                "probability_interface": None,
            }
        },
    }
    for path, value in settings.items():
        _resolve(payload, path, value)
    return FixedIdentityConfig.model_validate(payload)


def _reconfigure(harness: Harness, changes: dict[str, Any]) -> None:
    payload: dict[str, Any] = harness.config.model_dump(mode="json")
    for path, value in changes.items():
        _resolve(payload, path, value)
    harness.config = FixedIdentityConfig.model_validate(payload)
    harness.models.config = harness.config


@pytest.fixture(autouse=True)
def no_live_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Offline tests must not call a live model provider.")

    monkeypatch.setattr(litellm, "completion", forbidden)


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Harness:
    # Component tests isolate execution logic from already separately tested
    # readiness gates. This does not turn unresolved production settings into
    # defaults.
    monkeypatch.setattr(FixedIdentityConfig, "validate", lambda self, stage: None)
    taxonomy = _taxonomy()
    failure_ids: list[str] = [f"fixture-response-{index}" for index in range(3)]
    config = _component_config(tmp_path, failure_ids, taxonomy)
    store = ArtifactStore(str(tmp_path / "records.sqlite"), str(tmp_path / "blobs"))
    store.initialize()
    store.create_record("taxonomy", taxonomy.taxonomy_id, taxonomy.model_dump(mode="json"))
    instance_ids: list[str] = []
    for index, response_id in enumerate(failure_ids):
        instance_id: str = f"fixture-instance-{index}"
        draft = TaskInstance(
            instance_id=instance_id,
            case_id=f"fixture-case-{index}",
            source_dataset="ctibench",
            source_item_id=f"fictional-source-{index}",
            task_name="threat_actor_linking",
            stage="ATTRIBUTION",
            snapshot_date="2024-06-01",
            evidence_records=[{
                "evidence_id": "fixture-input",
                "source_id": "fictional-source",
                "document_id": "fictional-input-document",
                "content_reference": "fixture-original-artifact",
                "role": "model_input",
                "snippet": INPUT_TEXT,
                "publication_time": "2024-05-01",
            }],
            environment={"report_label": "original", "platform": "fictional"},
            candidates=["A", "B"],
            reference={
                "version": "fixture-reference-v1",
                "answer": {"choice": "B"},
                "evidence_records": [{
                    "evidence_id": "fixture-reference",
                    "document_id": "fictional-reference-document",
                    "role": "reference_only",
                    "snippet": REFERENCE_TEXT,
                }],
            },
            output_profile={"type": "object"},
            transformation_log={
                "corpus_version": CORPUS_VERSION,
                "config_hash": FIXTURE_CONFIG_HASH,
                "task_guidance": "Choose an actor using the fictional evidence.",
            },
        )
        task_payload = draft.model_dump(mode="json")
        task_payload["content_hash"] = _content_hash(_task_content_projection(draft))
        instance = TaskInstance.model_validate(task_payload)
        store.create_record("task_instance", instance_id, instance.model_dump(mode="json"))
        store.create_record(
            "reference",
            f"reference-{index}",
            {"reference_id": f"reference-{index}", **instance.reference},
        )
        store.freeze_records("task_instance", [instance_id])
        response = ModelResponse(
            response_id=response_id,
            instance_id=instance_id,
            model_id="G5",
            run_index=0,
            prompt_hash=_text_hash(f"fictional-original-prompt-{index}"),
            config_hash=FIXTURE_CONFIG_HASH,
            status="success",
            raw_text=_canonical_json({"answer": WRONG_TEXT}),
            parsed_answer={"status": "OK", "answer": {"choice": "A"}},
        )
        store.create_record("model_response", response_id, response.model_dump(mode="json"))
        decision = FailureDecision(
            response_id=response_id,
            identification_version="fixture-identification-v1",
            decision="failure",
            decision_origin="hard_label",
        )
        store.create_record(
            "failure_decision", _content_hash(decision), decision.model_dump(mode="json")
        )
        instance_ids.append(instance_id)
    corpus = {
        "manifest_id": "fixture-corpus-manifest",
        "corpus_version": CORPUS_VERSION,
        "config_hash": FIXTURE_CONFIG_HASH,
        "members": [
            {
                "instance_id": identity,
                "content_hash": store.read_record("task_instance", identity)["content_hash"],
            }
            for identity in instance_ids
        ],
    }
    store.create_record("corpus_manifest", corpus["manifest_id"], corpus)
    store.freeze_records("corpus_manifest", [corpus["manifest_id"]])
    models = ScriptedModels(config, store)
    annotations = ScriptedAnnotations(store)
    return Harness(config, store, models, annotations, taxonomy, failure_ids, instance_ids)


def _complete_proposal(
    annotations: ScriptedAnnotations,
    task: AnnotationTask,
    labels: tuple[str, ...] = (),
) -> str:
    annotations.submit_annotation(
        task.annotation_task_id,
        task.assigned_annotator,
        {"taxonomy_proposal": _proposal(labels)},
    )
    return annotations.get_reference(
        task.response_id, task.purpose.value
    )["annotation_ids"][0]


def _initialize(harness: Harness, manager: TaxonomyManager) -> TaxonomyVersion:
    tasks = manager.request_initialization(harness.failure_ids)
    annotation_ids = [
        _complete_proposal(harness.annotations, task, ("fixture-A",))
        for task in tasks
    ]
    return manager.create_initial_taxonomy(annotation_ids)


def _assignment(
    harness: Harness,
    actor: str,
    labels: tuple[str, ...],
    round_index: int,
    *,
    response_id: str | None = None,
    taxonomy: TaxonomyVersion | None = None,
    uncertain: bool = False,
    persist: bool = False,
) -> LabelAssignment:
    active: TaxonomyVersion = taxonomy or harness.taxonomy
    response: str = response_id or harness.failure_ids[0]
    content: dict[str, Any] = {
        "response_id": response,
        "taxonomy_id": active.taxonomy_id,
        "actor_id": actor,
        "round_index": round_index,
        "origin": "agent",
        "labels": list(labels),
        "evidence_spans": [_span(label) for label in labels],
        "uncertain": uncertain,
    }
    assignment = LabelAssignment.model_validate(
        {"assignment_id": _content_hash(content), **content},
        context={"taxonomy": active, "require_taxonomy": True},
    )
    if persist:
        harness.store.create_record(
            "label_assignment",
            assignment.assignment_id,
            assignment.model_dump(mode="json"),
        )
    return assignment


def _rounds(
    harness: Harness,
    first_labels: list[tuple[str, ...]],
    second_labels: list[tuple[str, ...]],
) -> tuple[list[LabelAssignment], list[LabelAssignment]]:
    actors: list[str] = [
        entry.name for entry in harness.config.models.categorization_panel
    ]
    return (
        [
            _assignment(harness, actor, labels, 1)
            for actor, labels in zip(actors, first_labels, strict=True)
        ],
        [
            _assignment(harness, actor, labels, 2)
            for actor, labels in zip(actors, second_labels, strict=True)
        ],
    )


# ---------------------------------------------------------------------------
# Real readiness checks preserve unresolved settings.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("discover_taxonomy", "taxonomy_discovery"),
        ("categorize", "categorization"),
        ("defenses", "counterfactual"),
    ],
)
def test_unresolved_scientific_protocols_block_without_inference(
    tmp_path: Path, stage: str, expected: str
) -> None:
    payload = _baseline()
    payload["experiment_id"] = "readiness-fixture"
    payload["corpus_version"] = CORPUS_VERSION
    payload["operational"] = {
        "database_path": str(tmp_path / "records.sqlite"),
        "artifact_root": str(tmp_path / "blobs"),
        "output_path": str(tmp_path / "reports"),
        "taxonomy_mode": "discovery",
        "selected_defense": "counterfactual_consistency",
    }
    path: Path = tmp_path / "unresolved.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = ExperimentConfig.load(str(path))
    with pytest.raises(ValueError) as captured:
        config.validate(stage)
    assert expected in str(captured.value)
    assert config.defenses.counterfactual_consistency.stability_threshold_tau is None
    assert config.categorization.peer_payload_policy is None


# ---------------------------------------------------------------------------
# Human-supported, iterative RQ2 discovery.
# ---------------------------------------------------------------------------


def test_initialization_waits_for_humans_and_is_reproducible(harness: Harness) -> None:
    manager = harness.manager()
    waiting = manager.run(harness.failure_ids)
    assert waiting["status"] == "waiting_for_annotations"
    assert harness.models.requests == []
    first = manager.request_initialization(harness.failure_ids)
    repeated = manager.request_initialization(list(reversed(harness.failure_ids)))
    assert first == repeated
    assert len(first) == harness.config.taxonomy_discovery.initial_subset_size
    assert len(harness.store.list_records("annotation_task")) == len(first)
    assert first[0].response_id in harness.failure_ids
    assert INPUT_TEXT in _canonical_json(first[0].visible_payload)
    assert REFERENCE_TEXT in _canonical_json(first[0].visible_payload)


def test_initial_taxonomy_contains_only_human_supported_categories(harness: Harness) -> None:
    manager = harness.manager()
    initial = _initialize(harness, manager)
    assert {item.subtype_id for item in initial.subtype_definitions} == {"fixture-A"}
    assert not initial.frozen
    assert initial.parent_id is None
    assert initial.subtype_definitions[0].supporting_annotation_ids
    published = {
        subtype
        for group in harness.config.taxonomy.published_reference.values()
        for subtype in group
    }
    assert published.isdisjoint(
        item.subtype_id for item in initial.subtype_definitions
    )


def test_passed_response_cannot_enter_discovery(harness: Harness) -> None:
    original = harness.store.read_record("model_response", harness.failure_ids[0])
    original["response_id"] = "fixture-passed-response"
    harness.store.create_record("model_response", original["response_id"], original)
    decision = FailureDecision(
        response_id=original["response_id"],
        identification_version="fixture-identification-v1",
        decision="pass",
        decision_origin="hard_label",
    )
    harness.store.create_record(
        "failure_decision", _content_hash(decision), decision.model_dump(mode="json")
    )
    with pytest.raises(ValueError, match="passed"):
        harness.manager().request_initialization([original["response_id"]])
    assert harness.models.requests == []


def test_other_review_expands_versions_and_changes_model_requests(harness: Harness) -> None:
    manager = harness.manager()
    initial = _initialize(harness, manager)

    def classify(request: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
        labels = packet["current_taxonomy"]["subtype_definitions"]
        selected = "other" if len(labels) == 1 else "fixture-B"
        return {
            "labels": [selected],
            "evidence_spans": [_span(selected)],
            "uncertain": False,
        }

    harness.models.script = classify
    old_assignments = manager.classify_remaining(harness.failure_ids, initial)
    before = {
        item.assignment_id: harness.store.read_record(
            "label_assignment", item.assignment_id
        )
        for item in old_assignments
    }
    tasks = manager.request_other_reviews(old_assignments)
    assert tasks == manager.request_other_reviews(old_assignments)
    assert all("other" in item.labels for item in old_assignments)
    assert {item.subtype_id for item in initial.subtype_definitions} == {"fixture-A"}
    with pytest.raises(ValueError):
        manager.freeze_taxonomy(initial.taxonomy_id)
    with pytest.raises(RuntimeError):
        manager.refine_taxonomy(initial, [])

    reviews = [
        _complete_proposal(
            harness.annotations, task, ("fixture-B",) if index == 0 else ()
        )
        for index, task in enumerate(tasks)
    ]
    child = manager.refine_taxonomy(initial, reviews)
    assert child.parent_id == initial.taxonomy_id
    assert child.iteration == initial.iteration + 1
    assert {item.subtype_id for item in child.subtype_definitions} == {
        "fixture-A", "fixture-B"
    }
    requests_before: set[str] = set(harness.models.provider_calls)
    new_assignments = manager.classify_remaining(harness.failure_ids, child)
    assert requests_before < harness.models.provider_calls
    assert all(item.taxonomy_id == child.taxonomy_id for item in new_assignments)
    assert all(item.labels == ["fixture-B"] for item in new_assignments)
    for identity, payload in before.items():
        assert harness.store.read_record("label_assignment", identity) == payload
    discovery_prompts = [
        request["prompt"] for request in harness.models.requests
        if request["schema_name"] == "rq2_taxonomy_classification"
    ]
    assert initial.taxonomy_id in discovery_prompts[0]
    assert "Fictional subtype fixture-B" not in discovery_prompts[0]
    assert child.taxonomy_id in discovery_prompts[-1]


def test_uncertainty_and_other_have_distinct_review_provenance(harness: Harness) -> None:
    manager = harness.manager()
    initial = _initialize(harness, manager)
    counter: int = 0

    def classify(request: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
        nonlocal counter
        counter += 1
        label = "fixture-A" if counter == 1 else "other"
        return {
            "labels": [label],
            "evidence_spans": [_span(label)],
            "uncertain": counter == 1,
        }

    harness.models.script = classify
    assignments = manager.classify_remaining(harness.failure_ids, initial)
    manager.request_other_reviews(assignments)
    events = [
        item for item in harness.store.list_records("experiment_manifest")
        if item.get("purpose") == "taxonomy_refinement_tasks"
    ]
    assert len(events) == 1
    assert len(events[0]["uncertain_response_ids"]) == 1
    assert len(events[0]["fallback_response_ids"]) == 1
    assert set(events[0]["uncertain_response_ids"]).isdisjoint(
        events[0]["fallback_response_ids"]
    )
    assert len(initial.subtype_definitions) == 1


def test_no_new_categories_does_not_override_insufficient_coverage(harness: Harness) -> None:
    manager = harness.manager()
    initial = _initialize(harness, manager)
    harness.models.script = lambda request, packet: {
        "labels": ["other"],
        "evidence_spans": [_span("other")],
        "uncertain": False,
    }
    assignments = manager.classify_remaining(harness.failure_ids, initial)
    tasks = manager.request_other_reviews(assignments)
    assert manager.check_convergence(initial, assignments) is False
    reviews = [_complete_proposal(harness.annotations, task) for task in tasks]
    assert manager.refine_taxonomy(initial, reviews) == initial
    assert manager.check_convergence(initial, assignments) is False
    with pytest.raises(ValueError, match="convergence"):
        manager.freeze_taxonomy(initial.taxonomy_id)


def test_supported_coverage_converges_only_after_refinement_and_freezes(harness: Harness) -> None:
    manager = harness.manager()
    initial = _initialize(harness, manager)
    assignments = manager.classify_remaining(harness.failure_ids, initial)
    assert manager.check_convergence(initial, assignments) is False
    assert manager.request_other_reviews(assignments) == []
    assert manager.refine_taxonomy(initial, []) == initial
    assert manager.check_convergence(initial, assignments) is True
    frozen = manager.freeze_taxonomy(initial.taxonomy_id)
    assert frozen.frozen
    assert frozen.guideline_hash == initial.guideline_hash
    assert frozen.subtype_definitions == initial.subtype_definitions
    assert frozen.inclusion_exclusion_rules == initial.inclusion_exclusion_rules
    assert manager.freeze_taxonomy(initial.taxonomy_id) == frozen
    assert harness.store.read_record("taxonomy", initial.taxonomy_id)["frozen"] is False
    with pytest.raises(ValueError, match="frozen"):
        manager.refine_taxonomy(frozen, [])


def test_fixed_taxonomy_application_is_not_reported_as_discovery(harness: Harness) -> None:
    _reconfigure(
        harness,
        {
            "operational.taxonomy_mode": "fixed",
            "taxonomy.discovery_initial_taxonomy": harness.taxonomy.model_dump(mode="json"),
        },
    )
    result = harness.manager().run(harness.failure_ids)
    assert result["status"] == "completed"
    assert result["discovery_claim"] is False
    assert result["human_discovery_initialization_performed"] is False
    assert result["classification_delegated_to"] == "Categorizer"
    assert harness.models.requests == []
    assert harness.annotations.requests == []


# ---------------------------------------------------------------------------
# RQ3: exact panel membership and set-based disagreement/stability.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ([("fixture-A",)] * 4, [("fixture-A",)] * 4, False),
        (
            [("fixture-A", "fixture-B"), ("fixture-B", "fixture-A")] * 2,
            [("fixture-B", "fixture-A")] * 4,
            False,
        ),
        (
            [("fixture-A",)] * 3 + [("fixture-B",)],
            [("fixture-A",)] * 4,
            True,
        ),
        (
            [("fixture-A",)] * 4,
            [("fixture-A",)] * 3 + [("fixture-B",)],
            True,
        ),
        ([("fixture-A",)] * 4, [("fixture-B",)] * 4, True),
        (
            [("fixture-A",)] * 4,
            [("fixture-A",)] * 3 + [("fixture-A", "fixture-B")],
            True,
        ),
    ],
)
def test_review_selector_uses_round_one_disagreement_or_any_change(
    harness: Harness,
    first: list[tuple[str, ...]],
    second: list[tuple[str, ...]],
    expected: bool,
) -> None:
    round_one, round_two = _rounds(harness, first, second)
    assert harness.categorizer().requires_review(round_one, round_two) is expected


@pytest.mark.parametrize("invalid", ["missing", "duplicate", "wrong_response", "wrong_taxonomy"])
def test_incomplete_or_misbound_round_is_not_consensus(
    harness: Harness, invalid: str
) -> None:
    first, second = _rounds(
        harness, [("fixture-A",)] * 4, [("fixture-A",)] * 4
    )
    if invalid == "missing":
        first = first[:-1]
    elif invalid == "duplicate":
        first[-1] = first[0]
    elif invalid == "wrong_response":
        first[-1] = first[-1].model_copy(update={"response_id": harness.failure_ids[1]})
    else:
        first[-1] = first[-1].model_copy(update={"taxonomy_id": "another-taxonomy"})
    with pytest.raises((ValueError, RuntimeError)):
        harness.categorizer().requires_review(first, second)


def test_duplicate_and_unknown_subtypes_are_rejected_by_active_taxonomy(harness: Harness) -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        _assignment(harness, "GPT-5", ("fixture-A", "fixture-A"), 1)
    with pytest.raises(ValidationError, match="Unknown subtype"):
        _assignment(harness, "GPT-5", ("unpublished-label",), 1)


def test_round_one_is_independent_and_round_two_preserves_snapshot(harness: Harness) -> None:
    categorizer = harness.categorizer()
    first = categorizer.run_round_one(harness.failure_ids[0], harness.taxonomy)
    actors = {item.actor_id for item in first}
    assert actors == {
        "GPT-5", "Llama-4-17B", "Gemini 2.5", "Claude Sonnet 4"
    }
    assert "LL70" not in actors
    first_requests = harness.models.requests[:]
    assert len(first_requests) == 4
    assert len({item["prompt"] for item in first_requests}) == 1
    first_packet = harness.models.packets[_text_hash(first_requests[0]["prompt"])]
    assert "first_round" not in first_packet
    assert "final_labels" not in first_packet
    preserved = _canonical_json(first)
    second = categorizer.run_round_two(harness.failure_ids[0], harness.taxonomy, first)
    assert len(second) == 4
    assert _canonical_json(first) == preserved
    for request in harness.models.requests[4:]:
        packet = harness.models.packets[_text_hash(request["prompt"])]
        assert {item["assignment_id"] for item in packet["first_round"]} == {
            item.assignment_id for item in first
        }
        assert all(item["round_index"] == 1 for item in packet["first_round"])
        assert packet["round_two_context"]["recipient_actor_id"] == request["model_id"]
        assert packet["round_two_context"]["classification_packet_hash"] == (
            packet["round_two_context"]["first_round_classification_packet_hash"]
        )
        exposed = request["prompt"].split("AUTHORIZED FIRST-ROUND PROPOSALS", 1)[1]
        exposed = exposed.split("RECONSIDERATION PROCEDURE", 1)[0]
        assert '"evidence_spans"' not in exposed
        assert '"actor_id"' not in exposed


@pytest.mark.parametrize(
    "status",
    ["transport_failure", "refusal", "invalid_schema", "unknown_label", "uncertain"],
)
def test_agent_execution_exceptions_block_without_assigning_a_failure_subtype(
    harness: Harness, status: str
) -> None:
    def output(request: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
        if request["model_id"] != "GPT-5":
            return {
                "labels": ["fixture-A"],
                "evidence_spans": [_span("fixture-A")],
                "uncertain": False,
            }
        if status in {"transport_failure", "refusal", "invalid_schema"}:
            return {"__fixture_status__": status}
        label = "unknown-label" if status == "unknown_label" else "fixture-A"
        return {
            "labels": [label],
            "evidence_spans": [_span(label)],
            "uncertain": status == "uncertain",
        }

    harness.models.script = output
    result = harness.categorizer().run([harness.failure_ids[0]], harness.taxonomy)
    assert result["status"] == "blocked"
    assert result["coverage"]["finalized_responses"] == 0
    assert result["coverage"]["selector_evaluable_responses"] == 0
    assert result["workload"]["review_fraction"] is None
    assert harness.store.list_records("final_assignment") == []
    assert harness.annotations.requests == []
    assert len(harness.taxonomy.subtype_definitions) == 2


def test_stable_consensus_finalizes_without_human_review_and_resumes(harness: Harness) -> None:
    categorizer = harness.categorizer()
    result = categorizer.run([harness.failure_ids[0]], harness.taxonomy)
    assert result["status"] == "completed"
    identity = result["final_assignment_ids_by_response"][harness.failure_ids[0]]
    final = LabelAssignment.model_validate(
        harness.store.read_record("final_assignment", identity)
    )
    assert final.origin == AssignmentOrigin.CONSENSUS
    assert final.labels == ["fixture-A"]
    assert final.taxonomy_id == harness.taxonomy.taxonomy_id
    assert harness.annotations.requests == []
    assert len(harness.models.provider_calls) == 8
    assert result["workload"]["review_fraction"] == 0.0
    assert categorizer.run([harness.failure_ids[0]], harness.taxonomy) == result
    assert len(harness.store.list_records("final_assignment")) == 1
    assert len(harness.models.provider_calls) == 8


def test_human_adjudication_overrides_majority_without_overwriting_agents(harness: Harness) -> None:
    def classify(request: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
        label = "fixture-B" if request["model_id"] == "GPT-5" else "fixture-A"
        return {"labels": [label], "evidence_spans": [_span(label)], "uncertain": False}

    harness.models.script = classify
    categorizer = harness.categorizer()
    waiting = categorizer.run([harness.failure_ids[0]], harness.taxonomy)
    assert waiting["status"] == "waiting_for_annotations"
    assert waiting["coverage"]["finalized_responses"] == 0
    assert waiting["workload"]["review_fraction"] == 1.0
    assert waiting["workload"]["review_fraction"] > (
        harness.config.categorization.reported_human_verification_fraction_upper_bound
    )
    assert waiting["workload"]["reported_bound_enforced_as_cap"] is False
    before = harness.store.list_records("label_assignment")
    tasks = [
        AnnotationTask.model_validate(harness.store.read_record("annotation_task", identity))
        for identity in waiting["pending_annotation_task_ids"]
    ]
    assert len(tasks) == 1
    assert categorizer.run([harness.failure_ids[0]], harness.taxonomy)["status"] == (
        "waiting_for_annotations"
    )
    assert len(harness.store.list_records("annotation_task")) == 1
    harness.annotations.resolve_task(
        tasks[0].annotation_task_id,
        tasks[0].assigned_annotator,
        {"labels": ["fixture-B"]},
    )
    final_result = categorizer.run([harness.failure_ids[0]], harness.taxonomy)
    assert final_result["status"] == "completed"
    identity = final_result["final_assignment_ids_by_response"][harness.failure_ids[0]]
    final = LabelAssignment.model_validate(
        harness.store.read_record("final_assignment", identity)
    )
    assert final.labels == ["fixture-B"]
    assert final.origin == AssignmentOrigin.OPERATIONAL_ADJUDICATION
    assert harness.store.list_records("label_assignment") == before
    assert harness.store.read_record("taxonomy", harness.taxonomy.taxonomy_id) == (
        harness.taxonomy.model_dump(mode="json")
    )
    assert final_result["workload"]["independent_audit_work_included"] is False
    assert categorizer.run([harness.failure_ids[0]], harness.taxonomy) == final_result


# ---------------------------------------------------------------------------
# Real parser contracts: no repair, fallback label, or unknown-taxonomy coercion.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "valid"),
    [
        ('{"labels":["fixture-A"],"evidence_spans":[],"uncertain":false}', False),
        ('{"labels":["unknown"],"evidence_spans":[],"uncertain":false}', False),
        ("```json\n{}\n```", False),
        ('{"labels":[],"labels":[],"evidence_spans":[],"uncertain":true}', False),
        ('{"labels":[],"evidence_spans":[],"uncertain":true}', True),
    ],
)
def test_real_classifier_parser_preserves_invalid_raw_text(
    harness: Harness, raw: str, valid: bool
) -> None:
    client = ModelClient(harness.config, harness.store)
    contract = {
        "version": "fixture-parser-schema-v1",
        "kind": "classification",
        "taxonomy_id": harness.taxonomy.taxonomy_id,
        "discovery_mode": False,
        "allow_empty_labels_when_uncertain": True,
        "json_schema": {
            "type": "object",
            "required": ["labels", "evidence_spans", "uncertain"],
            "additionalProperties": False,
            "properties": {
                "labels": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
                "evidence_spans": {"type": "array", "items": {"type": "object"}},
                "uncertain": {"type": "boolean"},
            },
        },
    }
    prompt: str = "Fictional parser request."
    request = {
        "model_id": "G5",
        "endpoint_or_checkpoint": "fixture/G5",
        "revision": "fictional-revision",
        "prompt_hash": _text_hash(prompt),
        "generation_controls": {},
        "seed_behavior": {},
        "schema_name": "fixture-classifier",
        "schema_version": "fixture-parser-schema-v1",
        "run_index": 0,
        "config_hash": FIXTURE_CONFIG_HASH,
        "provider_options": {"schema_contract": contract},
        "instance_id": harness.instance_ids[0],
        "prompt": prompt,
    }
    request["request_hash"] = _content_hash(request)
    harness.store.create_record("request", request["request_hash"], request)
    response = ModelResponse(
        response_id=_content_hash(raw),
        instance_id=harness.instance_ids[0],
        model_id="G5",
        run_index=0,
        prompt_hash=request["prompt_hash"],
        config_hash=FIXTURE_CONFIG_HASH,
        status="invalid_schema",
        schema_errors=["Not parsed yet."],
        raw_text=raw,
        usage={"request_hash": request["request_hash"]},
    )
    parsed = client.parse_response(response, "fixture-classifier")
    assert parsed.raw_text == raw
    assert (parsed.status == ResponseStatus.SUCCESS) is valid
    if not valid:
        assert parsed.parsed_answer == {}
        assert parsed.schema_errors
    assert harness.models.requests == []


# ---------------------------------------------------------------------------
# Global taxonomy matching on the formal-sized fixed audit population.
# ---------------------------------------------------------------------------


def _stability_cohort(
    harness: Harness,
    memberships: dict[str, dict[str, set[int]]],
) -> AuditCohort:
    count: int = harness.config.taxonomy_stability.audit_instances
    assert count == 180
    responses: list[str] = []
    original = harness.store.read_record("model_response", harness.failure_ids[0])
    for index in range(count):
        identity = f"stability-response-{index:03d}"
        payload = {**copy.deepcopy(original), "response_id": identity}
        harness.store.create_record("model_response", identity, payload)
        responses.append(identity)
    bindings: dict[str, dict[str, str]] = {}
    for taxonomy_id, categories in memberships.items():
        taxonomy = _taxonomy(taxonomy_id, tuple(categories))
        harness.store.create_record(
            "taxonomy", taxonomy_id, taxonomy.model_dump(mode="json")
        )
        bindings[taxonomy_id] = {}
        for index, response_id in enumerate(responses):
            labels = tuple(
                label for label, members in categories.items() if index in members
            )
            assignment = _assignment(
                harness, "fixture-stability-coder", labels, 0,
                response_id=response_id, taxonomy=taxonomy, persist=True,
            )
            bindings[taxonomy_id][response_id] = assignment.assignment_id
    first_id: str = next(iter(memberships))
    cohort = AuditCohort(
        cohort_id=_content_hash(memberships_as_json(memberships)),
        purpose="taxonomy_stability",
        response_ids=responses,
        seed=37,
        frozen_prediction_version="fixture-stability-predictions-v1",
        locked=True,
        sampling_protocol={
            "membership_count": count,
            "eligible_population_version": "fixture-stability-population",
            "sampling_method": "explicit-fixture-membership",
            "exclusion_categories": [],
            "excluded_response_ids": [],
            "annotation_protocol": {"fixture_only": True},
            "frozen_prediction_ids": bindings[first_id],
            "taxonomy_assignment_ids": bindings,
            "discovery_runs": {
                identity: {"condition": "fictional", "run_index": index}
                for index, identity in enumerate(memberships)
            },
        },
    )
    harness.store.create_record(
        "audit_cohort", cohort.cohort_id, cohort.model_dump(mode="json")
    )
    return cohort


def memberships_as_json(
    memberships: dict[str, dict[str, set[int]]]
) -> dict[str, dict[str, list[int]]]:
    return {
        taxonomy: {label: sorted(values) for label, values in categories.items()}
        for taxonomy, categories in memberships.items()
    }


def test_renamed_categories_align_by_membership_not_names(harness: Harness) -> None:
    memberships = {
        "left-renamed": {"A": {0, 1}, "B": {2, 3}},
        "right-renamed": {"unrelated-Y": {2, 3}, "unrelated-X": {0, 1}},
    }
    cohort = _stability_cohort(harness, memberships)
    evaluator = Evaluator(harness.config, harness.store)
    result = evaluator.score_taxonomy_stability(list(memberships), cohort)
    pair = result["comparisons"][0]
    assert pair["taxonomy_recovery_rate"] == 1.0
    assert pair["matched_category_jaccard"] == 1.0
    assert pair["instance_label_set_jaccard"] == 1.0
    assert pair["jensen_shannon_divergence"] == 0.0
    reversed_result = evaluator.score_taxonomy_stability(
        list(reversed(memberships)), cohort
    )
    assert reversed_result["comparisons"][0]["matched_category_jaccard"] == 1.0


def test_matching_is_global_rather_than_greedy(harness: Harness) -> None:
    left = {"A": set(range(6)), "B": {6, 7, 8}}
    right = {"X": {0, 1, 2, 3, 6, 7, 8}, "Y": {4, 5}}
    cohort = _stability_cohort(harness, {"global-left": left, "global-right": right})
    result = Evaluator(harness.config, harness.store).score_taxonomy_stability(
        ["global-left", "global-right"], cohort
    )
    pair = result["comparisons"][0]
    alignment = {
        item["reference_subtype"]: item["alternative_subtype"]
        for item in pair["alignment"]
    }
    assert alignment == {"A": "Y", "B": "X"}

    def jaccard(a: set[int], b: set[int]) -> float:
        return len(a & b) / len(a | b)

    possible = [
        sum(jaccard(left[label], right[target]) for label, target in zip(
            sorted(left), permutation, strict=True
        ))
        for permutation in itertools.permutations(sorted(right))
    ]
    expected = max(possible) / 2
    assert pair["matched_category_jaccard"] == pytest.approx(expected)
    assert expected > (jaccard(left["A"], right["X"]) + jaccard(left["B"], right["Y"])) / 2


def test_split_unmatched_categories_and_jsd_use_declared_policies(harness: Harness) -> None:
    cohort = _stability_cohort(
        harness,
        {
            "split-left": {"whole": {0, 1, 2, 3}},
            "split-right": {"part-A": {0, 1}, "part-B": {2, 3}},
        },
    )
    result = Evaluator(harness.config, harness.store).score_taxonomy_stability(
        ["split-left", "split-right"], cohort
    )
    pair = result["comparisons"][0]
    assert len(pair["unmatched_alternative"]) == 1
    assert pair["matched_category_jaccard"] == pytest.approx(0.25)
    # P=(1,0), Q=(1/2,1/2), base-2 divergence:
    # KL(P||M)=log2(4/3); KL(Q||M)=.5 log2(2/3)+.5.
    expected_jsd = 0.5 * np.log2(4 / 3) + 0.25 * np.log2(2 / 3) + 0.25
    assert pair["jensen_shannon_divergence"] == pytest.approx(expected_jsd)
    assert pair["jensen_shannon_divergence"] != pytest.approx(np.sqrt(expected_jsd))
    assert result["jsd_is_divergence_not_distance"] is True


# ---------------------------------------------------------------------------
# Counterfactual preservation and capability gates.
# ---------------------------------------------------------------------------


def fixture_candidate_output(
    *, payload: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    """Trusted local fixture adapter; never receives a reference answer."""
    assert parameters == {}
    result = copy.deepcopy(payload["original_model_output"])
    result["answer"]["choice"] = payload["selected_label"]
    return result


def _configure_c3(harness: Harness, tmp_path: Path) -> None:
    guidance_path = tmp_path / "guidance.json"
    profiles_path = tmp_path / "profiles.json"
    guidance_path.write_text(
        _canonical_json({
            "version": "fixture-v1",
            "tasks": {
                "threat_actor_linking": {
                    "guidance": "Choose the fictional actor.",
                    "candidate_policy": "include",
                }
            },
        }),
        encoding="utf-8",
    )
    profiles_path.write_text(
        _canonical_json({
            "version": "fixture-v1",
            "tasks": {
                "threat_actor_linking": {
                    "output_profile": {"type": "object"},
                    "schema_contract": {
                        "version": "fixture-task-v1",
                        "kind": "task",
                        "json_schema": {"type": "object"},
                    },
                }
            },
        }),
        encoding="utf-8",
    )
    instance = TaskInstance.model_validate(
        harness.store.read_record("task_instance", harness.instance_ids[0])
    )
    items: list[dict[str, Any]] = []
    decisions: dict[str, Any] = {}
    for index in range(4):
        operator_id = f"fixture-incidental-report-label-{index}"
        items.append({
            "operator_id": operator_id,
            "family": "I_spur",
            "instance_ids": [instance.instance_id],
            "edits": [{
                "pointer": "/environment/report_label",
                "before": "original",
                "after": f"variant-{index}",
            }],
            "preservation_justification": (
                "Only an incidental report label changes; attribution evidence is fixed."
            ),
            "preservation_evidence_refs": ["fixture-input"],
        })
        decisions[operator_id] = {
            "decision": "admissible",
            "instance_id": instance.instance_id,
            "original_content_hash": instance.content_hash,
            "reviewer_id": "fixture-independent-reviewer",
            "justification": "The actor judgment and its supplied mechanism evidence remain fixed.",
            "evidence_refs": ["fixture-input"],
            "answer_preserved": True,
            "causal_attributes_preserved": True,
            "chronology_valid": True,
        }
    adapter = {
        "callable": f"{__name__}:fixture_candidate_output",
        "version": "fixture-adapter-v1",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameters": {},
    }
    _reconfigure(harness, {
        "operational.selected_defense": "counterfactual_consistency",
        "evaluation.prompt.template_path": str(
            PROJECT_ROOT / "prompts/task_evaluation.jinja2"
        ),
        "evaluation.prompt.task_guidance_manifest": str(guidance_path),
        "evaluation.prompt.output_profile_manifest": str(profiles_path),
        "evaluation.scoring.task_field_extraction_rules": {
            "threat_actor_linking": {
                "prediction": "/answer/choice",
                "reference": "/answer/choice",
                "representation": "aligned",
            }
        },
        "evaluation.scoring.invalid_output_policy": "error",
        "evaluation.scoring.abstention_policy": "error",
        "evaluation.scoring.accuracy.exact_match_and_partial_credit_rules": {
            "mode": "response_exact"
        },
        "defenses.counterfactual_consistency.operators": {
            "protocol_version": "fixture-interventions-v1",
            "items": items,
            "preservation_protocol": {"mode": "manifest", "decisions": decisions},
            "accepted_count_policy": "use_admissible",
        },
        "defenses.counterfactual_consistency.counterfactuals_per_instance": 4,
        "defenses.counterfactual_consistency.stability_threshold_tau": 0.75,
        "defenses.counterfactual_consistency.output_equivalence_rule": "candidate_label_exact",
        "defenses.counterfactual_consistency.probability_extraction": {
            "interface": "token_logprobs",
            "aggregation_space": "tokens",
            "token_to_label": {"A": "A", "B": "B"},
            "sum_tolerance": 1e-10,
            "candidate_output_adapter": adapter,
        },
        "defenses.counterfactual_consistency.entropy_weight_formula": "exp_negative_entropy",
        "defenses.counterfactual_consistency.normalize_weights": True,
        "defenses.counterfactual_consistency.include_original_input": False,
        "defenses.counterfactual_consistency.tie_policy": "block",
        "defenses.counterfactual_consistency.repetition_count": 1,
    })
    runner = harness.runner()
    guidance, profiles = runner._task_manifests()
    original, _ = runner._task_payload(instance, guidance, profiles)
    for operator in items:
        modified = copy.deepcopy(original)
        modified["environment"]["report_label"] = operator["edits"][0]["after"]
        decisions[operator["operator_id"]]["modified_model_input_hash"] = _content_hash(modified)
    operators = copy.deepcopy(harness.config.defenses.counterfactual_consistency.operators)
    operators["preservation_protocol"]["decisions"] = decisions
    _reconfigure(harness, {"defenses.counterfactual_consistency.operators": operators})


def test_probability_capability_cannot_be_replaced_with_json_confidence(
    harness: Harness,
) -> None:
    _reconfigure(harness, {
        "operational.selected_defense": "counterfactual_consistency",
        "defenses.counterfactual_consistency.probability_extraction": {
            "interface": "token_logprobs",
            "aggregation_space": "tokens",
        },
    })
    with pytest.raises(ValueError, match="probability"):
        ModelClient(harness.config, harness.store).validate_endpoints(["G5"])
    assert harness.models.requests == []


def test_missing_preservation_review_is_unresolved_not_consistent(
    harness: Harness, tmp_path: Path
) -> None:
    _configure_c3(harness, tmp_path)
    operators = harness.config.defenses.counterfactual_consistency.operators
    assert isinstance(operators, dict)
    changed = copy.deepcopy(operators)
    changed["preservation_protocol"]["decisions"] = {}
    _reconfigure(harness, {"defenses.counterfactual_consistency.operators": changed})
    result = harness.runner().run_counterfactual_defense([harness.instance_ids[0]])
    assert result["status"] == "blocked"
    assert len(result["unresolved_preservation_variant_ids"]) == 4
    assert harness.models.requests == []
    assert "omega" not in result


def test_answer_changing_variants_are_excluded_without_fabricating_scores(
    harness: Harness, tmp_path: Path
) -> None:
    _configure_c3(harness, tmp_path)
    operators = copy.deepcopy(
        harness.config.defenses.counterfactual_consistency.operators
    )
    for decision in operators["preservation_protocol"]["decisions"].values():
        decision.update(
            decision="rejected",
            answer_preserved=False,
            causal_attributes_preserved=False,
            justification="The proposed edit would invalidate the requested judgment.",
        )
    _reconfigure(harness, {"defenses.counterfactual_consistency.operators": operators})
    with pytest.raises(ValueError, match="without admissible"):
        harness.runner().run_counterfactual_defense([harness.instance_ids[0]])
    assert harness.models.requests == []
    variants = harness.store.list_records("counterfactual")
    assert all(item["preservation_status"] == "rejected" for item in variants)


def test_justification_cannot_override_explicit_failed_preservation_flags(
    harness: Harness, tmp_path: Path
) -> None:
    _configure_c3(harness, tmp_path)
    operators = copy.deepcopy(
        harness.config.defenses.counterfactual_consistency.operators
    )
    first_decision = next(iter(
        operators["preservation_protocol"]["decisions"].values()
    ))
    first_decision["answer_preserved"] = False
    first_decision["justification"] = "A prose claim does not override this false flag."
    _reconfigure(harness, {"defenses.counterfactual_consistency.operators": operators})
    runner = harness.runner()
    instance = TaskInstance.model_validate(
        harness.store.read_record("task_instance", harness.instance_ids[0])
    )
    variants = runner.build_counterfactuals(instance)
    with pytest.raises(ValueError, match="answer"):
        runner.validate_preservation(instance, variants)
    assert harness.models.requests == []


def test_c3_uses_weighted_probabilities_not_majority_and_strict_tau(
    harness: Harness, tmp_path: Path
) -> None:
    _configure_c3(harness, tmp_path)

    def probabilities(
        request: dict[str, Any], packet: dict[str, Any]
    ) -> dict[str, Any]:
        context = request["template_provenance"]["experiment_context"]
        if context["condition"] == "original":
            probability_a = 0.6
        elif packet["environment"]["report_label"] == "variant-3":
            probability_a = 0.01
        else:
            probability_a = 0.51
        return {
            "status": "OK",
            "task": packet["task_name"],
            "case_id": packet["case_id"],
            "snapshot_date": packet["snapshot_date"],
            "answer": {"choice": "A" if probability_a > 0.5 else "B"},
            "confidence": 0.99,
            "justification": "Fictional probability fixture.",
            "evidence_refs": ["fixture-input"],
            "metadata": {},
            "__fixture_probability__": {
                "interface": "token_logprobs",
                "data": {"content": [{
                    "top_logprobs": [
                        {"token": "A", "logprob": float(np.log(probability_a))},
                        {"token": "B", "logprob": float(np.log(1 - probability_a))},
                    ]
                }]},
            },
        }

    harness.models.script = probabilities
    result = harness.runner().run_counterfactual_defense([harness.instance_ids[0]])
    assert result["status"] == "completed"
    artifacts = [
        item for item in harness.store.list_records("experiment_manifest")
        if item.get("purpose") == "counterfactual_consistency_results"
    ]
    diagnostic = artifacts[0]["diagnostics"][0]
    assert diagnostic["omega"] == 0.75
    assert diagnostic["vulnerable"] is False  # Strict Omega < tau.
    assert len(diagnostic["variant_response_ids"]) == 4
    corrected = harness.store.read_record(
        "model_response", diagnostic["corrected_response_id"]
    )
    assert corrected["parsed_answer"]["answer"]["choice"] == "B"
    variants = [
        harness.store.read_record("model_response", identity)
        for identity in diagnostic["variant_response_ids"]
    ]
    assert sum(
        item["parsed_answer"]["answer"]["choice"] == "A" for item in variants
    ) == 3
    assert diagnostic["weighted_candidate_totals"]["B"] > (
        diagnostic["weighted_candidate_totals"]["A"]
    )
    assert artifacts[0]["consistency_is_correctness"] is False
    assert artifacts[0]["unique_failure_mechanism_inferred"] is False


def test_unsupported_counterfactual_weighting_never_invokes_voting(
    harness: Harness, tmp_path: Path
) -> None:
    _configure_c3(harness, tmp_path)
    _reconfigure(harness, {
        "defenses.counterfactual_consistency.entropy_weight_formula": "majority_vote"
    })
    with pytest.raises(ValueError, match="exp_negative_entropy"):
        harness.runner().run_counterfactual_defense([harness.instance_ids[0]])
    assert harness.models.requests == []
