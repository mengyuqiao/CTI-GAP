"""Release-blocking tests for independent-audit isolation.

The supplied config.yaml defines sample sizes, raters, exclusions, and controls.
Synthetic records and explicitly declared test-only policies exercise unresolved
interfaces without treating those policies as defaults.

Large audit fixtures insert independent submissions through ArtifactStore to keep
tests offline and deterministic. Separate tests exercise AnnotationService's
submission, authorization, adjudication, and lifecycle behavior directly.

No audit labels are passed to model execution, taxonomy discovery, or scientific
configuration tuning. A failing lifecycle or isolation assertion is a regression,
not a reason to bypass the corresponding production guard.
"""

from __future__ import annotations

import copy
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from annotation_app import AnnotationService
from config import ExperimentConfig
from evaluation import Evaluator
from failure_identification import FailureIdentifier
from schemas import (
    AnnotationTask,
    AuditCohort,
    FailureDecision,
    LabelAssignment,
    ModelResponse,
    TaskInstance,
    TaxonomyVersion,
    _canonical_json,
    _content_hash,
    _task_content_projection,
    _text_hash,
)
from storage import ArtifactStore


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"

GUIDELINE: dict[str, Any] = {
    "version": "offline-audit-guideline-v1",
    "instruction": "Use only observable discrepancies and supplied evidence.",
}
GUIDELINE_HASH: str = _content_hash(GUIDELINE)
REFERENCE_TEXT: str = "Fictional reference: asset A is affected; actor B is supported."
RESPONSE_TEXT: str = "Fictional evaluated answer: asset B is affected."
CORPUS_VERSION: str = "offline-audit-corpus"
PREDICTION_VERSION: str = "offline-frozen-identification-v1"
SAMPLING_SEED: int = 173  # Test-only deterministic sampling, not an author setting.
LABELS: tuple[str, str] = ("fixture-A", "fixture-B")


def _resolve(payload: dict[str, Any], path: str, value: Any) -> None:
    """Record an explicit test-only resolution of an existing policy field."""
    current: dict[str, Any] = payload
    components: list[str] = path.split(".")
    for component in components[:-1]:
        current = current.setdefault(component, {})
    current[components[-1]] = copy.deepcopy(value)
    payload.setdefault("implementation_choices", {})[path] = {
        "configuration_path": path,
        "origin": "implementation",
        "rationale": "Offline isolation fixture only; not a scientific setting.",
        "affected_experiments": ["all"],
        "selected_value": copy.deepcopy(value),
    }


def _span(label: str | None = None) -> dict[str, Any]:
    return {
        "source_ref": "fixture-reference",
        "quote": REFERENCE_TEXT,
        "subtype_id": label,
        "relation_type": "observable-reference-discrepancy",
    }


def _taxonomy() -> TaxonomyVersion:
    return TaxonomyVersion.model_validate(
        {
            "taxonomy_id": "offline-frozen-taxonomy",
            "iteration": 0,
            "guideline_hash": GUIDELINE_HASH,
            "frozen": True,
            "subtype_definitions": [
                {
                    "subtype_id": label,
                    "name": f"Fictional category {label}",
                    "definition": f"Observable fictional discrepancy {label}.",
                    "evidence_requirements": ["Exact applicable evidence."],
                }
                for label in LABELS
            ],
            "inclusion_exclusion_rules": {
                label: {
                    "inclusion": ["The specified discrepancy is evidenced."],
                    "exclusion": ["The discrepancy is not evidenced."],
                    "boundary_rules": ["Do not infer hidden internal causes."],
                    "permitted_cooccurring_labels": [],
                }
                for label in LABELS
            },
        }
    )


@dataclass
class AuditWorld:
    """Isolated persistence and explicit synthetic experimental units."""

    config: ExperimentConfig
    store: ArtifactStore
    service: AnnotationService
    evaluator: Evaluator
    taxonomy: TaxonomyVersion
    task: TaskInstance

    def population(
        self,
        count: int,
        prefix: str = "response",
    ) -> tuple[list[str], dict[str, str], dict[str, str]]:
        """Create distinct model/run responses with frozen automatic predictions."""
        responses: list[str] = []
        decisions: dict[str, str] = {}
        assignments: dict[str, str] = {}
        for index in range(count):
            identity: str = f"{prefix}-{index:04d}"
            response = ModelResponse(
                response_id=identity,
                instance_id=self.task.instance_id,
                model_id="G5" if index % 2 == 0 else "CLD",
                run_index=index % self.config.evaluation_runs,
                prompt_hash=_text_hash(f"Fictional prompt {identity}"),
                config_hash=self.config.config_hash,
                status="success",
                raw_text=RESPONSE_TEXT,
                parsed_answer={
                    "status": "OK",
                    "answer": {"affected_assets": ["B"]},
                    "confidence": 0.5,
                },
            )
            self.store.create_record(
                "model_response", identity, response.model_dump(mode="json")
            )
            decision = FailureDecision(
                response_id=identity,
                identification_version=PREDICTION_VERSION,
                decision="failure" if index % 2 == 0 else "pass",
                decision_origin="hard_label",
            )
            decision_id: str = _content_hash(decision)
            self.store.create_record(
                "failure_decision", decision_id, decision.model_dump(mode="json")
            )
            assignment = LabelAssignment(
                assignment_id=f"prediction:{identity}",
                response_id=identity,
                taxonomy_id=self.taxonomy.taxonomy_id,
                actor_id="fixture:stable-consensus",
                round_index=2,
                labels=[LABELS[0]],
                evidence_spans=[_span(LABELS[0])],
                uncertain=False,
                origin="consensus",
            )
            self.store.create_record(
                "consensus",
                assignment.assignment_id,
                assignment.model_dump(mode="json"),
            )
            responses.append(identity)
            decisions[identity] = decision_id
            assignments[identity] = assignment.assignment_id
        return responses, decisions, assignments

    def protocol(
        self,
        response_ids: list[str],
        raters: int,
        judgment: str,
        cohort_id: str | None = None,
    ) -> dict[str, Any]:
        actors: list[str] = [f"independent-rater-{index}" for index in range(raters)]
        return {
            "protocol_version": "offline-human-protocol-v1",
            "config_hash": self.config.config_hash,
            "guideline_hash": GUIDELINE_HASH,
            "guideline_content": GUIDELINE,
            "judgment": judgment,
            "annotator_ids": actors,
            "annotators_per_instance": raters,
            "assignments": {identity: actors[:] for identity in response_ids},
            "queue_order_policy": "response_id",
            "reference_establishment": "independent_agreement_or_adjudication",
            "cohort_id": cohort_id,
            "taxonomy_id": self.taxonomy.taxonomy_id if judgment == "labels" else None,
            "correctness_criteria": {
                "fixture": "Compare the requested answer with applicable evidence."
            },
            "reference_applicability": {
                identity: {"scope": "Fictional contemporaneous evidence."}
                for identity in response_ids
            },
            "adjudication": {
                "adjudicator_id": "independent-adjudicator",
                "senior_adjudicator_id": "independent-senior",
                "escalate_unresolved_cases_to_senior": True,
                "adjudicator_must_be_independent": True,
            },
            "agreement_report_id": None,
        }

    def sampling(
        self,
        response_ids: list[str],
        prediction_ids: dict[str, str],
        purpose: str = "failure_identification",
        cohort_id: str = "identification-audit",
    ) -> dict[str, Any]:
        identification: bool = purpose == "failure_identification"
        size: int = (
            self.config.audits.failure_identification.instances
            if identification
            else self.config.annotation.agreement_study.instances
        )
        raters: int = (
            self.config.audits.failure_identification.independent_annotators
            if identification
            else self.config.annotation.agreement_study.annotators_per_instance
        )
        method: str = (
            self.config.audits.failure_identification.sampling
            if identification
            else "stratified_without_replacement"
        )
        result: dict[str, Any] = {
            "cohort_id": cohort_id,
            "config_hash": self.config.config_hash,
            "guideline_hash": GUIDELINE_HASH,
            "seed": SAMPLING_SEED,
            "sample_size": size,
            "sampling_unit": "response",
            "sampling_method": method,
            "eligible_population_version": "offline-eligible-population-v1",
            "eligible_response_ids": response_ids,
            "frozen_prediction_version": PREDICTION_VERSION,
            "frozen_prediction_ids": prediction_ids,
            "exclusion_categories": (
                self.config.audits.failure_identification.exclude
                if identification else ["calibration_examples"]
            ),
            "excluded_response_ids": [],
            "annotation_protocol": self.protocol(
                response_ids, raters,
                "identification" if identification else "labels",
                cohort_id,
            ),
        }
        if identification:
            result["duplicate_key_by_response"] = {
                identity: f"unique-observation:{identity}" for identity in response_ids
            }
            result["duplicate_representatives"] = {}
        else:
            result["stratum_by_response"] = {
                identity: "explicit-fixture-stratum" for identity in response_ids
            }
            result["allocation_by_stratum"] = {"explicit-fixture-stratum": size}
        return result

    def locked_cohort(
        self,
        response_ids: list[str],
        predictions: dict[str, str],
        purpose: str,
        cohort_id: str,
        protocol: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> AuditCohort:
        """Persist exact controlled membership; this helper does not sample."""
        sampling: dict[str, Any] = {
            "membership_count": len(response_ids),
            "eligible_population_version": "offline-population-v1",
            "sampling_method": "explicit-offline-fixture-membership",
            "exclusion_categories": self.config.audits.failure_identification.exclude,
            "excluded_response_ids": [],
            "annotation_protocol": protocol,
            "guideline_hash": GUIDELINE_HASH,
            "taxonomy_id": self.taxonomy.taxonomy_id,
            "frozen_prediction_ids": {
                identity: predictions[identity] for identity in response_ids
            },
            **(extra or {}),
        }
        cohort = AuditCohort(
            cohort_id=cohort_id,
            purpose=purpose,
            response_ids=response_ids,
            sampling_protocol=sampling,
            seed=SAMPLING_SEED,
            frozen_prediction_version=PREDICTION_VERSION,
            locked=True,
        )
        self.store.create_record(
            "audit_cohort", cohort_id, cohort.model_dump(mode="json")
        )
        self.store.freeze_records("audit_cohort", [cohort_id])
        return cohort

    def annotation(
        self,
        response_id: str,
        purpose: str,
        cohort_id: str | None,
        actor: str,
        result: dict[str, Any],
        protocol: dict[str, Any] | None = None,
        task: AnnotationTask | None = None,
        adjudication: bool = False,
    ) -> str:
        """Insert a controlled independent submission or separate adjudication."""
        identity: str = (
            _content_hash({"independent_annotation": task.annotation_task_id})
            if task is not None and not adjudication
            else _content_hash([response_id, purpose, cohort_id, actor, result])
        )
        labels: list[str] = result.get("labels", [])
        record: dict[str, Any] = {
            "annotation_id": identity,
            "response_id": response_id,
            "purpose": "audit_adjudication" if adjudication else purpose,
            "reference_purpose": purpose,
            "cohort_id": cohort_id,
            "annotator_id": actor,
            "guideline_hash": GUIDELINE_HASH,
            "taxonomy_id": self.taxonomy.taxonomy_id if "labels" in result else None,
            "status": "adjudicated" if adjudication else "submitted",
            "audit_only": cohort_id is not None,
            "uncertain": False,
            "evidence_spans": [_span(label) for label in labels] or [_span()],
            **result,
        }
        if task is not None:
            record["annotation_task_id"] = task.annotation_task_id
        if protocol is not None:
            record["protocol_hash"] = _content_hash(protocol)
        kind: str = "adjudication" if adjudication else "annotation"
        self.store.create_record(kind, identity, record)
        self.store.freeze_records(kind, [identity])
        return identity


@pytest.fixture
def paper_config(tmp_path: Path) -> ExperimentConfig:
    assert CONFIG_PATH.is_file(), "The supplied config.yaml is required."
    return ExperimentConfig.load(str(CONFIG_PATH))


@pytest.fixture
def world(tmp_path: Path, paper_config: ExperimentConfig) -> AuditWorld:
    payload: dict[str, Any] = paper_config.model_dump(mode="json")
    payload["experiment_id"] = "offline-audit-isolation"
    payload["corpus_version"] = CORPUS_VERSION
    payload["operational"].update(
        database_path=str(tmp_path / "audit.sqlite"),
        artifact_root=str(tmp_path / "blobs"),
        output_path=str(tmp_path / "reports"),
    )
    _resolve(
        payload, "annotation.agreement_study.uncertain_label_handling", "reject"
    )
    config = ExperimentConfig.model_validate(payload)
    store = ArtifactStore(str(tmp_path / "audit.sqlite"), str(tmp_path / "blobs"))
    store.initialize()
    taxonomy: TaxonomyVersion = _taxonomy()
    store.create_record(
        "taxonomy", taxonomy.taxonomy_id, taxonomy.model_dump(mode="json")
    )
    draft = TaskInstance(
        instance_id="shared-fictional-task",
        case_id="shared-fictional-case",
        source_dataset="ctibench",
        source_item_id="fictional-source-item",
        task_name="affected_systems",
        stage="CONTEXTUALIZATION",
        snapshot_date="2024-06-01",
        evidence_records=[{
            "evidence_id": "fixture-input",
            "document_id": "fixture-input-document",
            "source_id": "fixture-source",
            "content_reference": "fictional-artifact",
            "role": "model_input",
            "snippet": "Fictional asset inventory and contemporaneous evidence.",
            "publication_time": "2024-05-01",
        }],
        candidates=["A", "B"],
        reference={
            "version": "offline-reference-v1",
            "answer": {"affected_assets": ["A"]},
            "evidence_records": [{
                "evidence_id": "fixture-reference",
                "document_id": "fixture-reference-document",
                "role": "reference_only",
                "snippet": REFERENCE_TEXT,
            }],
        },
        output_profile={"type": "object"},
        transformation_log={
            "corpus_version": CORPUS_VERSION,
            "config_hash": config.config_hash,
            "task_guidance": "Determine affected assets from the supplied evidence.",
        },
    )
    task_payload: dict[str, Any] = draft.model_dump(mode="json")
    task_payload["content_hash"] = _content_hash(_task_content_projection(draft))
    task = TaskInstance.model_validate(task_payload)
    store.create_record("task_instance", task.instance_id, task.model_dump(mode="json"))
    store.freeze_records("task_instance", [task.instance_id])
    return AuditWorld(
        config, store, AnnotationService(store), Evaluator(config, store), taxonomy, task
    )


def _assert_blinded(value: Any, forbidden: set[str]) -> None:
    if isinstance(value, dict):
        assert not (set(key.lower() for key in value) & forbidden)
        for item in value.values():
            _assert_blinded(item, forbidden)
    elif isinstance(value, list):
        for item in value:
            _assert_blinded(item, forbidden)


def _forbidden_fields(config: ExperimentConfig) -> set[str]:
    return {
        *config.audits.failure_identification.blind_to,
        "similarity_score",
        "automatic_decision",
        "agent_predictions",
        "peer_predictions",
        "final_labels",
        "adjudicated_labels",
        "stable_consensus",
        "review_trigger",
    }


def test_protocol_controls_and_unresolved_settings_are_not_silently_resolved(
    paper_config: ExperimentConfig,
) -> None:
    identification = paper_config.audits.failure_identification
    categorization = paper_config.audits.failure_categorization
    assert identification.population == "automatically_labeled_responses"
    assert identification.freeze_identification_before_sampling is True
    assert identification.use_audit_labels_for_tuning is False
    assert categorization.include_stable_consensus_cases is True
    assert categorization.evaluate_predictions_before_human_correction is True
    assert categorization.count_audit_work_as_operational_verification is False
    assert (
        categorization.total_instances
        == categorization.original_agreement_instances
        + categorization.additional_instances
    )
    assert identification.sampling_seed is None
    assert categorization.additional_sampling_protocol is None
    assert paper_config.annotation.agreement_study.uncertain_label_handling is None
    assert paper_config.taxonomy.guideline_manifest is None
    with pytest.raises(ValueError, match="unresolved"):
        paper_config.validate("audit")


def test_identification_sampling_excludes_union_and_preserves_simple_random_semantics(
    world: AuditWorld,
) -> None:
    size: int = world.config.audits.failure_identification.instances
    response_ids, predictions, _ = world.population(size + 9)
    forbidden: set[str] = set(response_ids[-5:-2])
    protocol = world.protocol(response_ids, 1, "identification")
    world.service.create_tasks(
        [response_ids[-5]], "identification_anchor", protocol
    )
    world.service.create_tasks(
        [response_ids[-5], response_ids[-4]], "calibration", protocol
    )
    world.service.create_tasks(
        [response_ids[-3]], "identification_boundary", protocol
    )
    # The last two records are explicitly duplicate copies of one observation.
    duplicate: str = response_ids[-1]
    representative: str = response_ids[-2]
    forbidden.add(duplicate)
    sampling = world.sampling(response_ids, predictions)
    sampling["duplicate_key_by_response"][duplicate] = "declared-duplicate"
    sampling["duplicate_key_by_response"][representative] = "declared-duplicate"
    sampling["duplicate_representatives"] = {"declared-duplicate": representative}

    cohort = world.service.sample_audit("failure_identification", sampling)
    eligible: list[str] = sorted(set(response_ids) - forbidden)
    expected = sorted(random.Random(SAMPLING_SEED).sample(eligible, size))
    assert cohort.response_ids == expected
    assert len(cohort.response_ids) == len(set(cohort.response_ids)) == size
    assert not set(cohort.response_ids) & forbidden
    assert world.service.sample_audit("failure_identification", sampling) == cohort

    # Shared case/task IDs and repeated raw text do not collapse distinct runs.
    assert len({world.store.read_record("model_response", item)["instance_id"]
                for item in eligible}) == 1
    assert set(eligible) != {item for item in eligible if
                            world.store.read_record(
                                "failure_decision", predictions[item]
                            )["decision"] == "failure"}


def test_insufficient_eligible_population_is_not_filled_from_reviewed_cases(
    world: AuditWorld,
) -> None:
    size = world.config.audits.failure_identification.instances
    response_ids, predictions, _ = world.population(size)
    protocol = world.protocol(response_ids, 1, "identification")
    world.service.create_tasks([response_ids[0]], "calibration", protocol)
    with pytest.raises(ValueError, match="required"):
        world.service.sample_audit(
            "failure_identification", world.sampling(response_ids, predictions)
        )


@pytest.mark.parametrize("decision", ["unresolved", "human_anchor"])
def test_formal_identification_audit_requires_final_automatic_predictions(
    world: AuditWorld, decision: str
) -> None:
    size = world.config.audits.failure_identification.instances
    responses, predictions, _ = world.population(size)
    revised = FailureDecision(
        response_id=responses[0],
        identification_version=PREDICTION_VERSION,
        decision="unresolved" if decision == "unresolved" else "failure",
        decision_origin="pending_review" if decision == "unresolved" else "human_anchor",
        requires_review=decision == "unresolved",
    )
    identity = _content_hash(revised)
    world.store.create_record(
        "failure_decision", identity, revised.model_dump(mode="json")
    )
    predictions[responses[0]] = identity
    with pytest.raises(ValueError, match="automatic"):
        world.service.sample_audit(
            "failure_identification", world.sampling(responses, predictions)
        )


def test_locked_cohort_retains_membership_and_prediction_snapshot(
    world: AuditWorld,
) -> None:
    size = world.config.audits.failure_identification.instances
    responses, predictions, _ = world.population(size)
    sampling = world.sampling(responses, predictions)
    cohort = world.service.sample_audit("failure_identification", sampling)
    locked = world.service.lock_cohort(cohort.cohort_id)
    before = world.store.read_record("audit_cohort", locked.cohort_id)
    later = FailureDecision(
        response_id=responses[0],
        identification_version="later-identification-version",
        decision="pass",
        decision_origin="hard_label",
    )
    world.store.create_record(
        "failure_decision", _content_hash(later), later.model_dump(mode="json")
    )
    world.population(1, prefix="newly-arrived-response")
    assert world.store.read_record("audit_cohort", locked.cohort_id) == before
    assert locked.frozen_prediction_version == PREDICTION_VERSION
    assert locked.sampling_protocol["frozen_prediction_ids"] == predictions
    with pytest.raises(PermissionError):
        world.store.update_draft(
            "audit_cohort", locked.cohort_id, {"response_ids": responses[:-1]}
        )
    with pytest.raises(PermissionError):
        world.store.delete_draft("audit_cohort", locked.cohort_id)
    with pytest.raises(ValueError, match="incomplete"):
        world.evaluator.score_failure_audit(locked)


def test_blinded_tasks_preserve_evidence_but_not_automatic_or_peer_judgments(
    world: AuditWorld,
) -> None:
    size = world.config.audits.failure_identification.instances
    responses, predictions, _ = world.population(size)
    sampling = world.sampling(responses, predictions)
    cohort = world.service.lock_cohort(
        world.service.sample_audit("failure_identification", sampling).cohort_id
    )
    tasks = world.service.create_tasks(
        [responses[0]],
        "independent_identification_audit",
        cohort.sampling_protocol["annotation_protocol"],
    )
    assert len(tasks) == world.config.audits.failure_identification.independent_annotators
    first = world.service.get_next_task(tasks[0].assigned_annotator)
    second_before = world.service.get_next_task(tasks[1].assigned_annotator)
    _assert_blinded(first.visible_payload, _forbidden_fields(world.config))
    assert RESPONSE_TEXT in _canonical_json(first.visible_payload)
    assert REFERENCE_TEXT in _canonical_json(first.visible_payload)
    assert "task_guidance" in first.visible_payload["task_input"]

    # Persist one independent result to inspect the other rater's view without
    # conflating annotation lifecycle behavior with blinding behavior.
    world.annotation(
        first.response_id, first.purpose.value, first.cohort_id,
        first.assigned_annotator, {"decision": "failure"},
        cohort.sampling_protocol["annotation_protocol"], first,
    )
    second_after = world.service.get_next_task(tasks[1].assigned_annotator)
    assert second_after.visible_payload == second_before.visible_payload
    _assert_blinded(second_after.visible_payload, _forbidden_fields(world.config))
    with pytest.raises(LookupError):
        world.service.get_next_task(first.assigned_annotator)


def test_submission_authorization_idempotence_and_immutable_independent_labels(
    world: AuditWorld,
) -> None:
    responses, _, _ = world.population(1)
    protocol = world.protocol(responses, 2, "identification")
    tasks = world.service.create_tasks(
        responses, "identification_anchor", protocol
    )
    task = tasks[0]
    payload = {
        "decision": "failure",
        "uncertain": False,
        "evidence_spans": [_span()],
        "notes": "Independent fixture judgment.",
    }
    with pytest.raises(PermissionError):
        world.service.submit_annotation(
            task.annotation_task_id, tasks[1].assigned_annotator, payload
        )
    world.service.submit_annotation(
        task.annotation_task_id, task.assigned_annotator, payload
    )
    preserved = world.store.list_records("annotation")
    world.service.submit_annotation(
        task.annotation_task_id, task.assigned_annotator, payload
    )
    assert world.store.list_records("annotation") == preserved
    with pytest.raises(ValueError, match="conflicting"):
        world.service.submit_annotation(
            task.annotation_task_id, task.assigned_annotator,
            {**payload, "decision": "pass"},
        )
    assert world.store.list_records("annotation") == preserved


def test_identification_metrics_use_frozen_predictions_not_later_corrections(
    world: AuditWorld,
) -> None:
    size = world.config.audits.failure_identification.instances
    responses, predictions, _ = world.population(size)
    protocol = world.protocol(
        responses,
        world.config.audits.failure_identification.independent_annotators,
        "identification",
        "identification-metrics",
    )
    cohort = world.locked_cohort(
        responses, predictions, "failure_identification",
        "identification-metrics", protocol,
    )
    expected: Counter[str] = Counter(tp=0, fp=0, fn=0, tn=0)
    for index, response_id in enumerate(responses):
        predicted: bool = index % 2 == 0
        human: bool = not predicted if index < 2 else predicted
        expected[
            "tp" if predicted and human else "fp" if predicted
            else "fn" if human else "tn"
        ] += 1
        for actor in protocol["annotator_ids"]:
            world.annotation(
                response_id, "independent_identification_audit", cohort.cohort_id,
                actor, {"decision": "failure" if human else "pass"},
            )
    corrected = FailureDecision(
        response_id=responses[0],
        identification_version="post-audit-corrected-version",
        decision="pass",
        decision_origin="human_boundary",
    )
    world.store.create_record(
        "failure_decision", _content_hash(corrected), corrected.model_dump(mode="json")
    )
    scored = world.evaluator.score_failure_audit(cohort)
    assert scored["counts"] == dict(expected)
    assert scored["counts"]["fp"] == scored["counts"]["fn"] == 1
    assert sum(scored["counts"].values()) == size
    assert scored["metrics"]["precision"] == pytest.approx(
        expected["tp"] / (expected["tp"] + expected["fp"])
    )
    assert scored["prediction_version"] == PREDICTION_VERSION
    assert scored["audit_labels_used_for_tuning"] is False


@pytest.mark.parametrize("corruption", ["wrong_response", "missing_prediction", "wrong_version"])
def test_identification_scoring_rejects_ambiguous_or_broken_lineage(
    world: AuditWorld, corruption: str
) -> None:
    size = world.config.audits.failure_identification.instances
    responses, predictions, _ = world.population(size)
    if corruption == "wrong_response":
        predictions[responses[0]] = predictions[responses[1]]
    elif corruption == "missing_prediction":
        predictions[responses[0]] = "nonexistent-prediction"
    protocol = world.protocol(responses, 2, "identification", "broken-lineage")
    cohort = world.locked_cohort(
        responses, predictions, "failure_identification", "broken-lineage", protocol
    )
    if corruption == "wrong_version":
        payload = cohort.model_dump(mode="json")
        payload["cohort_id"] = "wrong-version-cohort"
        payload["frozen_prediction_version"] = "nonexistent-version"
        cohort = AuditCohort.model_validate(payload)
        world.store.create_record(
            "audit_cohort", cohort.cohort_id, cohort.model_dump(mode="json")
        )
    with pytest.raises((ValueError, KeyError)):
        world.evaluator.score_failure_audit(cohort)


def test_missing_prediction_version_is_not_a_valid_audit_manifest() -> None:
    with pytest.raises(ValidationError):
        AuditCohort.model_validate({
            "cohort_id": "missing-version",
            "purpose": "failure_identification",
            "response_ids": ["response"],
            "sampling_protocol": {},
            "seed": SAMPLING_SEED,
            "locked": False,
        })


def _agreement_fixture(
    world: AuditWorld,
) -> tuple[AuditCohort, list[AnnotationTask], list[str]]:
    count = world.config.annotation.agreement_study.instances
    responses, _, predictions = world.population(count, prefix="agreement")
    sampling = world.sampling(
        responses, predictions, "human_agreement", "agreement-cohort"
    )
    cohort = world.service.lock_cohort(
        world.service.sample_audit("human_agreement", sampling).cohort_id
    )
    protocol = cohort.sampling_protocol["annotation_protocol"]
    tasks = world.service.create_tasks(
        [responses[0]], "human_agreement", protocol
    )
    task_by_actor = {task.assigned_annotator: task for task in tasks}
    for index, response_id in enumerate(responses):
        for rater_index, actor in enumerate(protocol["annotator_ids"]):
            label = LABELS[index % 2]
            if index == 0 and rater_index == 0:
                label = LABELS[1]
            world.annotation(
                response_id, "human_agreement", cohort.cohort_id, actor,
                {"labels": [label]},
                protocol,
                task_by_actor.get(actor) if index == 0 else None,
            )
    return cohort, tasks, responses


def test_agreement_precedes_adjudication_and_uses_original_labels(
    world: AuditWorld,
) -> None:
    cohort, tasks, responses = _agreement_fixture(world)
    before_annotations = world.store.list_records("annotation")
    with pytest.raises(ValueError, match="pre-adjudication"):
        world.service.create_adjudications([
            task.annotation_task_id for task in tasks
        ])
    before = world.evaluator.score_human_agreement(cohort)
    assert before["pre_adjudication"] is True
    assert before["exact_agreement_denominator"] == len(responses)
    assert before["exact_label_set_agreement"] == pytest.approx(
        (len(responses) - 1) / len(responses)
    )
    adjudications = world.service.create_adjudications([
        task.annotation_task_id for task in tasks
    ])
    assert len(adjudications) == 1
    review = adjudications[0]
    world.service.resolve_task(
        review.annotation_task_id, review.assigned_annotator,
        {
            "labels": [LABELS[0]],
            "uncertain": False,
            "evidence_spans": [_span(LABELS[0])],
            "notes": "Separate adjudicated reference.",
        },
    )
    assert world.store.list_records("annotation") == before_annotations
    after = world.evaluator.score_human_agreement(cohort)
    assert after == before


def test_missing_independent_ratings_cannot_be_replaced_by_adjudication(
    world: AuditWorld,
) -> None:
    size = world.config.annotation.agreement_study.instances
    responses, _, predictions = world.population(size, prefix="incomplete-agreement")
    protocol = world.protocol(
        responses, world.config.annotation.agreement_study.annotators_per_instance,
        "labels", "incomplete-agreement",
    )
    cohort = world.locked_cohort(
        responses, predictions, "human_agreement", "incomplete-agreement", protocol
    )
    world.annotation(
        responses[0], "human_agreement", cohort.cohort_id, "independent-adjudicator",
        {"labels": [LABELS[0]]}, adjudication=True,
    )
    with pytest.raises(ValueError, match="rater count"):
        world.evaluator.score_human_agreement(cohort)
    tasks = world.service.create_tasks(
        [responses[0]], "human_agreement", protocol
    )
    with pytest.raises(ValueError, match="independent raters"):
        world.service.create_adjudications([
            task.annotation_task_id for task in tasks
        ])


def test_unresolved_uncertain_label_policy_blocks_agreement(
    world: AuditWorld, paper_config: ExperimentConfig
) -> None:
    cohort, _, _ = _agreement_fixture(world)
    payload = world.config.model_dump(mode="python")
    payload["annotation"]["agreement_study"]["uncertain_label_handling"] = (
        paper_config.annotation.agreement_study.uncertain_label_handling
    )
    unresolved = ExperimentConfig.model_validate(payload)
    with pytest.raises(ValueError, match="unresolved"):
        Evaluator(unresolved, world.store).score_human_agreement(cohort)


def test_expanded_audit_requires_every_additional_independent_reference(
    world: AuditWorld,
) -> None:
    settings = world.config.audits.failure_categorization
    responses, _, predictions = world.population(settings.total_instances, prefix="expanded")
    original_ids = responses[:settings.original_agreement_instances]
    additions = responses[settings.original_agreement_instances:]
    original_protocol = world.protocol(
        original_ids,
        world.config.annotation.agreement_study.annotators_per_instance,
        "labels", "expanded-original",
    )
    original = world.locked_cohort(
        original_ids, predictions, "human_agreement",
        "expanded-original", original_protocol,
    )
    # Two independent raters are an explicit local fixture protocol, not an
    # inferred expanded-audit setting.
    protocol = world.protocol(responses, 2, "labels", "expanded-combined")
    cohort = world.locked_cohort(
        responses, predictions, "failure_categorization",
        "expanded-combined", protocol,
        {
            "original_cohort_id": original.cohort_id,
            "original_response_ids": original_ids,
            "additional_response_ids": additions,
            "reference_cohort_ids": [original.cohort_id],
            "includes_stable_consensus": True,
        },
    )

    def add_reference(response_id: str, purpose: str, cohort_id: str) -> None:
        index = responses.index(response_id)
        labels = [LABELS[1]] if index == 0 else list(LABELS) if index == 1 else [LABELS[0]]
        for actor in protocol["annotator_ids"]:
            world.annotation(
                response_id, purpose, cohort_id, actor, {"labels": labels}
            )

    for identity in original_ids:
        add_reference(identity, "human_agreement", original.cohort_id)
    with pytest.raises(ValueError, match="incomplete"):
        world.evaluator.score_label_audit(cohort)

    for identity in additions[:-1]:
        add_reference(identity, "expanded_label_audit", cohort.cohort_id)
    last = additions[-1]
    # Operational correction cannot fill the final independent-reference slot.
    world.annotation(
        last, "operational_adjudication", None, "operational-reviewer",
        {"labels": [LABELS[0]]},
    )
    with pytest.raises(ValueError, match="incomplete"):
        world.evaluator.score_label_audit(cohort)
    add_reference(last, "expanded_label_audit", cohort.cohort_id)

    # The stable automated consensus is wrong on the first instance; its later
    # correction must not replace the frozen original in audit scoring.
    correction = LabelAssignment(
        assignment_id="operational-correction",
        response_id=responses[0],
        taxonomy_id=world.taxonomy.taxonomy_id,
        actor_id="operational-reviewer",
        round_index=2,
        labels=[LABELS[1]],
        evidence_spans=[_span(LABELS[1])],
        origin="operational_adjudication",
        supersedes_id=None,
    )
    world.store.create_record(
        "final_assignment", correction.assignment_id,
        correction.model_dump(mode="json"),
    )
    scored = world.evaluator.score_label_audit(cohort)
    expected_tp: int = settings.total_instances - 1
    assert scored["n"] == settings.total_instances
    assert scored["micro"]["tp"] == expected_tp
    assert scored["micro"]["fp"] == 1
    assert scored["micro"]["fn"] == 2
    assert scored["micro"]["precision"] == pytest.approx(
        expected_tp / settings.total_instances
    )
    assert scored["micro"]["recall"] == pytest.approx(
        expected_tp / (settings.total_instances + 1)
    )
    support = sum(row["support"] for row in scored["per_subtype"].values())
    assert support == settings.total_instances + 1
    assert scored["predictions_before_human_correction"] is True

    tasks = world.service.create_tasks(
        [additions[0]], "expanded_label_audit", protocol
    )
    _assert_blinded(tasks[0].visible_payload, _forbidden_fields(world.config))
    assert "Fictional category" in _canonical_json(tasks[0].visible_payload)
    assert "fixture:stable-consensus" not in _canonical_json(tasks[0].visible_payload)


def test_expanded_sampling_blocks_unresolved_additional_protocol(
    world: AuditWorld,
) -> None:
    settings = world.config.audits.failure_categorization
    responses, _, predictions = world.population(settings.total_instances)
    original_ids = responses[:settings.original_agreement_instances]
    original_protocol = world.protocol(
        original_ids,
        world.config.annotation.agreement_study.annotators_per_instance,
        "labels", "original-for-unresolved",
    )
    original = world.locked_cohort(
        original_ids, predictions, "human_agreement",
        "original-for-unresolved", original_protocol,
    )
    sampling = world.sampling(
        responses, predictions, "human_agreement", "unresolved-expanded"
    )
    sampling.update(
        sample_size=settings.total_instances,
        original_cohort_id=original.cohort_id,
        additional_sampling_protocol=settings.additional_sampling_protocol,
        stable_consensus_response_ids=[responses[0]],
    )
    with pytest.raises(ValueError, match="additional_sampling_protocol"):
        world.service.sample_audit("failure_categorization", sampling)


def test_audit_references_cannot_fall_back_to_operational_purposes(
    world: AuditWorld,
) -> None:
    count = world.config.audits.failure_identification.instances
    responses, predictions, _ = world.population(count)
    sampling = world.sampling(responses, predictions)
    cohort = world.service.lock_cohort(
        world.service.sample_audit("failure_identification", sampling).cohort_id
    )
    protocol = cohort.sampling_protocol["annotation_protocol"]
    tasks = world.service.create_tasks(
        [responses[0]], "independent_identification_audit", protocol
    )
    for task in tasks:
        world.annotation(
            task.response_id, task.purpose.value, task.cohort_id,
            task.assigned_annotator, {"decision": "pass"}, protocol, task,
        )
    audit_reference = world.service.get_reference(
        responses[0], "independent_identification_audit"
    )
    assert audit_reference["audit_only"] is True
    assert audit_reference["decision"] == "pass"
    for purpose in ("identification_anchor", "identification_boundary",
                    "operational_adjudication"):
        with pytest.raises(LookupError):
            world.service.get_reference(responses[0], purpose)
    assert world.store.list_records("final_assignment") == []


def test_audit_labels_do_not_change_operational_anchor_ranges_or_workload(
    world: AuditWorld, tmp_path: Path
) -> None:
    """Exercise real RQ1 range fitting with purpose-scoped human references."""
    responses, _, _ = world.population(4, prefix="range-response")
    anchor_protocol = world.protocol(responses, 1, "identification")
    anchor_protocol["config_hash"] = "$config_hash"
    anchor_protocol["assignments"] = {"*": ["independent-rater-0"]}
    anchor_protocol["reference_applicability"] = {"*": {"scope": "Fixed operational scope."}}
    payload = world.config.model_dump(mode="json")
    explicit: dict[str, Any] = {
        "failure_identification.hard_label_rules": {
            "tasks": {"affected_systems": {"route": "similarity"}},
            "annotation_protocols": {
                "anchor": anchor_protocol,
                "boundary": anchor_protocol,
            },
        },
        "failure_identification.stratification.similarity_measure_by_task": {
            "affected_systems": {
                "source": "stored_score",
                "pointer": "/similarity_score",
                "version_pointer": "/similarity_version",
                "score_version": "offline-similarity-v1",
                "direction": "higher_is_better",
                "scale": [0, 100],
            },
        },
        "failure_identification.stratification.grouping_fields": ["task_name"],
        "failure_identification.stratification.anchor_count_per_stratum": 2,
        "failure_identification.stratification.anchor_sampling_seed": SAMPLING_SEED,
        "failure_identification.stratification.tied_score_policy": {
            "mode": "keep_together", "partition": "rank_floor",
        },
        "failure_identification.stratification.inspection_order": {
            "group_order": "lexical", "score_order": "ascending",
        },
        "failure_identification.stratification.pass_failure_range_estimator": {
            "method": "anchor_extrema",
            "lower_inclusive": True,
            "upper_inclusive": True,
            "one_class_policy": "observed_range",
            "contradictory_anchor_policy": "retain_overlap",
            "small_stratum_policy": "all",
        },
    }
    for path, value in explicit.items():
        _resolve(payload, path, value)

    # Bind the production configuration at runtime and persist distinct response
    # records for that configuration without modifying any original artifacts.
    range_config = ExperimentConfig.model_validate(payload)
    operational_responses: list[str] = []
    for response_id in responses:
        projected = world.store.read_record("model_response", response_id)
        projected["response_id"] = response_id + ":operational"
        projected["config_hash"] = range_config.config_hash
        world.store.create_record("model_response", projected["response_id"], projected)
        operational_responses.append(projected["response_id"])
    responses = operational_responses
    operational_predictions: dict[str, str] = {}
    for response_id in responses:
        prediction = FailureDecision(
            response_id=response_id, identification_version=PREDICTION_VERSION,
            decision="failure", decision_origin="hard_label",
        )
        prediction_id = _content_hash(prediction)
        operational_predictions[response_id] = prediction_id
        world.store.create_record(
            "failure_decision", prediction_id, prediction.model_dump(mode="json")
        )
        score_id = f"score:{response_id}"
        world.store.create_record(
            "score", score_id,
            {
                "score_id": score_id,
                "response_id": response_id,
                "instance_id": world.task.instance_id,
                "task": world.task.task_name,
                "config_hash": range_config.config_hash,
                "similarity_score": 50.0,
                "similarity_version": "offline-similarity-v1",
            },
        )
    identifier = FailureIdentifier(range_config, world.store, world.service)
    stratum = identifier.build_strata(responses)[0]
    tasks = identifier.request_anchors(stratum["stratum_id"])
    protocol = identifier._annotation_protocol("anchor")
    for task in tasks:
        world.annotation(
            task.response_id, "identification_anchor", None,
            task.assigned_annotator, {"decision": "failure"}, protocol, task,
        )
    before = identifier.fit_ranges(stratum["stratum_id"])
    decisions_before = world.store.list_records("failure_decision")
    operational_before = [
        item for item in world.store.list_records("annotation")
        if item.get("audit_only") is False
    ]

    separate_audit = AuditCohort(
        cohort_id="separate-audit", purpose="failure_identification",
        response_ids=responses, sampling_protocol={
            "membership_count": len(responses),
            "eligible_population_version": "purpose-scoping-fixture-v1",
            "sampling_method": "explicit-purpose-scoping-fixture",
            "exclusion_categories": [], "excluded_response_ids": [],
            "annotation_protocol": {"fixture_only": True},
            "frozen_prediction_ids": operational_predictions,
        },
        seed=SAMPLING_SEED, frozen_prediction_version=PREDICTION_VERSION, locked=True,
    )
    world.store.create_record(
        "audit_cohort", separate_audit.cohort_id, separate_audit.model_dump(mode="json")
    )
    for response_id in responses:
        world.annotation(
            response_id, "independent_identification_audit", "separate-audit",
            "blind-audit-rater", {"decision": "pass"},
        )
        world.annotation(
            response_id, "independent_identification_audit", "separate-audit",
            "blind-audit-adjudicator", {"decision": "pass"}, adjudication=True,
        )
    after = identifier.fit_ranges(stratum["stratum_id"])
    assert after == before
    assert after["ranges"]["failure"] is not None
    assert after["ranges"]["pass"] is None
    assert world.store.list_records("failure_decision") == decisions_before
    operational_after = [
        item for item in world.store.list_records("annotation")
        if item.get("audit_only") is False
    ]
    assert operational_after == operational_before
    audit_count = sum(
        item.get("audit_only") is True
        for item in world.store.list_records("annotation")
    )
    assert audit_count == len(responses)
    assert len(tasks) / len(responses) > (
        world.config.failure_identification.reported_manual_inspection_fraction_upper_bound
    )

    destination = tmp_path / "isolated-workload.json"
    world.evaluator.write_report(
        {
            "operational_anchor_submissions": len(operational_after),
            "independent_audit_submissions": audit_count,
            "operational_denominator": len(responses),
            "audit_included_in_operational_work": False,
        },
        str(destination),
    )
    reported = json.loads(destination.read_text(encoding="utf-8"))["results"]
    assert reported["operational_anchor_submissions"] == len(tasks)
    assert reported["independent_audit_submissions"] == len(responses)
    assert reported["audit_included_in_operational_work"] is False
