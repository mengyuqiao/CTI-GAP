"""Offline regressions for measurement provenance and real annotation lifecycles."""

from __future__ import annotations

from pathlib import Path

import pytest

from annotation_app import AnnotationService
from evaluation import Evaluator
from schemas import FailureDecision, ModelResponse, _content_hash
from test_audit_isolation import (  # Reuse complete real-store annotation fixtures.
    AuditWorld,
    _span,
    paper_config,
    world,
)
from test_metrics_and_identification import (
    MetricHarness,
    _harness,
    _resolve,
    metric_harness,
    metric_payload,
)


def test_failure_rate_uses_completed_identification_decision_bindings(
    metric_harness: MetricHarness,
) -> None:
    """Historical pending/boundary decisions must not hide a completed judgment."""
    harness = metric_harness
    _, response = harness.add([1], [0])
    old = FailureDecision(
        response_id=response.response_id,
        identification_version="active-identification",
        decision="unresolved",
        decision_origin="pending_review",
        requires_review=True,
    )
    final = FailureDecision(
        response_id=response.response_id,
        identification_version="active-identification",
        decision="failure",
        decision_origin="human_boundary",
    )
    for decision in (old, final):
        harness.store.create_record(
            "failure_decision", _content_hash(decision), decision.model_dump(mode="json")
        )
    payload = {
        "purpose": "identification_final",
        "config_hash": harness.config.config_hash,
        "corpus_version": harness.config.corpus_version,
        "identification_version": "active-identification",
        "result": {
            "status": "completed",
            "identification_version": "active-identification",
            "decision_ids_by_response": {response.response_id: _content_hash(final)},
        },
    }
    manifest_id = _content_hash(payload)
    harness.store.create_record(
        "experiment_manifest", manifest_id, {"manifest_id": manifest_id, **payload}
    )
    rate = harness.aggregate([response])["failure_rate"]
    assert rate["status"] == "complete"
    assert rate["failure_count"] == rate["response_count"] == 1
    assert rate["value"] == 100.0
    assert len(harness.store.list_records("failure_decision")) == 2


def _context_response(
    harness: MetricHarness, response: ModelResponse, protocol: str, condition: str
) -> ModelResponse:
    payload = response.model_dump(mode="json")
    payload["response_id"] += ":context"
    payload["usage"] = {
        "evaluation_context": {
            "protocol": protocol,
            "condition": condition,
            "comparison_id": "matched-fixture",
            "comparison_controls_hash": "same-controls",
        }
    }
    contextual = ModelResponse.model_validate(payload)
    harness.store.create_record(
        "model_response", contextual.response_id, contextual.model_dump(mode="json")
    )
    return contextual


def test_paired_task_scores_require_identical_included_populations(
    tmp_path: Path, metric_payload: dict,
) -> None:
    """Dropping the defended method's wrong answer cannot create a paired gain."""
    _resolve(metric_payload, "evaluation.scoring.invalid_output_policy", "exclude")
    harness = _harness(tmp_path, metric_payload)
    responses = []
    for run in range(5):
        for condition in ("original", "strong_rag"):
            for index in range(2):
                _, response = harness.add(
                    [1], [1] if index == 0 else [0], run=run,
                    instance_id=f"matched-{index}",
                    status="refusal" if condition == "strong_rag" and index == 1 else "success",
                )
                responses.append(_context_response(harness, response, "main", condition))
    with pytest.raises(ValueError, match="[Ii]ncluded|scoring population"):
        harness.aggregate(responses)


def test_ambiguous_document_ids_remain_unavailable_after_three_snippets() -> None:
    service = AnnotationService.__new__(AnnotationService)
    visible = {
        "evaluated_response": {"response_id": "answer", "raw_text": "response"},
        "task_input": {"source_documents": [
            {"evidence_id": "e1", "document_id": "shared", "snippet": "one"},
            {"evidence_id": "e2", "document_id": "shared", "snippet": "two"},
            {"evidence_id": "e3", "document_id": "shared", "snippet": "three"},
        ]},
        "reference_material": {"evidence_records": []},
    }
    texts = service._source_texts(visible)
    assert "shared" not in texts
    assert texts["e1"] == "one"
    assert texts["e2"] == "two"
    assert texts["e3"] == "three"


def test_real_submission_closes_referenced_annotation_task(world: AuditWorld) -> None:
    """Persisting a submission must also finish its task without mutating evidence."""
    responses, _, _ = world.population(1)
    protocol = world.protocol(responses, 1, "identification")
    task = world.service.create_tasks(responses, "identification_anchor", protocol)[0]
    judgment = {"decision": "failure", "uncertain": False, "evidence_spans": [_span()]}
    world.service.submit_annotation(task.annotation_task_id, task.assigned_annotator, judgment)
    stored = world.store.read_record("annotation_task", task.annotation_task_id)
    assert stored["status"] == "submitted"
    assert stored["visible_payload"] == task.visible_payload
    world.service.submit_annotation(task.annotation_task_id, task.assigned_annotator, judgment)
    assert len(world.store.list_records("annotation")) == 1
    with pytest.raises(PermissionError):
        world.store.update_draft("annotation_task", task.annotation_task_id, {"visible_payload": {}})


def test_reference_lookup_can_select_current_protocol(world: AuditWorld) -> None:
    """Old discovery/annotation iterations must remain distinct but selectable."""
    responses, _, _ = world.population(1)
    protocols = []
    for index, decision in enumerate(("pass", "failure")):
        protocol = world.protocol(responses, 1, "identification")
        protocol["protocol_version"] = f"iteration-{index}"
        task = world.service.create_tasks(responses, "identification_anchor", protocol)[0]
        world.annotation(
            responses[0], "identification_anchor", None, task.assigned_annotator,
            {"decision": decision}, protocol, task,
        )
        protocols.append(protocol)
    with pytest.raises(ValueError, match="ambiguous"):
        world.service.get_reference(responses[0], "identification_anchor")
    chosen = world.service.get_reference(
        responses[0], "identification_anchor", protocol_hash=_content_hash(protocols[1])
    )
    assert chosen["decision"] == "failure"


def test_operational_protocol_supports_explicit_future_response_schedule(world: AuditWorld) -> None:
    """A wildcard schedule avoids embedding future hash-derived IDs into config."""
    responses, _, _ = world.population(1)
    protocol = world.protocol(responses, 1, "identification")
    protocol["assignments"] = {"*": protocol["assignments"][responses[0]]}
    protocol["reference_applicability"] = {"*": {"scope": "Same fixed correctness scope."}}
    tasks = world.service.create_tasks(responses, "identification_anchor", protocol)
    assert len(tasks) == 1
    assert tasks[0].response_id == responses[0]
    assert tasks[0].assigned_annotator == "independent-rater-0"
    assert tasks[0].visible_payload["reference_material"]["applicability"] == {
        "scope": "Same fixed correctness scope."
    }
    assert set(protocol["assignments"]) == {"*"}


@pytest.mark.parametrize("protocol", ["counterfactual_consistency", "tailored_mitigation"])
def test_optional_defenses_use_their_configured_repetition_count(
    tmp_path: Path, metric_payload: dict, protocol: str,
) -> None:
    _resolve(metric_payload, f"defenses.{protocol}.repetition_count", 1)
    harness = _harness(tmp_path, metric_payload)
    _, response = harness.add([1], [1])
    response = _context_response(harness, response, protocol, "corrected")
    group = harness.aggregate([response])["groups"][0]
    assert group["summary"]["status"] == "complete"
    assert group["summary"]["expected_runs"] == 1
    assert group["summary"]["mean"] == 100.0


def test_weighted_f1_ignores_a_class_with_zero_reference_support(
    tmp_path: Path, metric_payload: dict,
) -> None:
    _resolve(metric_payload, "evaluation.scoring.f1.averaging", "weighted")
    harness = _harness(tmp_path, metric_payload)
    _, response = harness.add([1, 1], [1, 1])
    scored = harness.aggregate([response])["groups"][0]["runs"][0]
    assert scored["value"] == 1.0


def test_independent_senior_cannot_be_an_original_rater(world: AuditWorld) -> None:
    responses, _, _ = world.population(1)
    protocol = world.protocol(responses, 2, "identification")
    protocol["adjudication"]["senior_adjudicator_id"] = "independent-rater-0"
    tasks = world.service.create_tasks(responses, "identification_anchor", protocol)
    for index, task in enumerate(tasks):
        world.service.submit_annotation(
            task.annotation_task_id, task.assigned_annotator,
            {"decision": "pass" if index == 0 else "failure", "uncertain": False,
             "evidence_spans": [_span()]},
        )
    review = world.service.create_adjudications([task.annotation_task_id for task in tasks])[0]
    with pytest.raises(ValueError, match="[Ii]ndependent"):
        world.service.resolve_task(
            review.annotation_task_id, review.assigned_annotator,
            {"decision": "unresolved", "uncertain": True, "evidence_spans": [_span()]},
        )
    # Reject before persisting discussion or assigning a conflicted senior.
    assert len(world.store.list_records("annotation")) == 2
    assert len(world.store.list_records("annotation_task")) == 3
