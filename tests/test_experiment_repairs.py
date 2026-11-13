"""Offline regressions for experiment orchestration and scientific bindings."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest

from schemas import FailureDecision, LabelAssignment, ModelResponse, StageStatus, TaskInstance, _content_hash
from test_taxonomy_and_categorization import (
    Harness,
    _configure_c3,
    _reconfigure,
    _span,
    harness,
    no_live_inference,
)


def _adapter(function: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "callable": f"{__name__}:{function}",
        "version": "offline-regression-v1",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameters": parameters or {},
    }


def retrieval_by_budget(*, payload: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    """A deterministic local adapter makes dropped configuration observable."""
    budget = payload.get("retrieval_configuration", {}).get("evidence_context_budget", 2)
    return {
        "evidence": [{
            "evidence_id": f"retrieved-{index}",
            "document_id": f"document-{index}",
            "snippet": "Historically available evidence.",
            "role": "model_input",
            "publication_time": "2024-05-01",
            "snapshot_admission_verified": True,
        } for index in range(budget)],
        "provenance": {"version": "offline-v1"},
    }


def candidate_probabilities(*, payload: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    assert "ROLE AND GLOBAL CONSTRAINTS" in payload["prompt"]
    assert payload["message_mapping"] == "single_user_message"
    assert payload["endpoint_or_checkpoint"]
    assert payload["revision"]
    changed = payload["model_input"]["environment"]["report_label"] != "original"
    return {
        "probabilities": {"A": 0.1 if changed else 0.8, "B": 0.9 if changed else 0.2},
        "provenance": {"method": "explicit offline finite-candidate fixture"},
    }


def corrected_candidate(*, payload: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    answer = copy.deepcopy(payload["original_model_output"])
    answer["answer"]["choice"] = payload["selected_label"]
    if parameters.get("wrong_task"):
        answer["task"] = "another_task"
    if parameters.get("wrong_label"):
        answer["answer"]["choice"] = "A"
    return answer


def _configured(harness: Harness, tmp_path: Path) -> TaskInstance:
    _configure_c3(harness, tmp_path)
    instance = TaskInstance.model_validate(
        harness.store.read_record("task_instance", harness.instance_ids[0])
    )
    # Reviews certify the exact modified input, independently of model output.
    operators = copy.deepcopy(harness.config.defenses.counterfactual_consistency.operators)
    for variant in harness.runner().build_counterfactuals(instance):
        operators["preservation_protocol"]["decisions"][variant["operator_id"]][
            "modified_model_input_hash"
        ] = variant["modified_model_input_hash"]
    _reconfigure(harness, {"defenses.counterfactual_consistency.operators": operators})
    return instance


def test_review_of_an_incidental_edit_cannot_approve_a_replaced_platform_edit(
    harness: Harness, tmp_path: Path,
) -> None:
    instance = _configured(harness, tmp_path)
    operators = copy.deepcopy(harness.config.defenses.counterfactual_consistency.operators)
    operators["items"][0]["edits"] = [{
        "pointer": "/environment/platform", "before": "fictional", "after": "other-platform",
    }]
    _reconfigure(harness, {"defenses.counterfactual_consistency.operators": operators})
    runner = harness.runner()
    with pytest.raises(ValueError, match="modified|reviewed|preservation"):
        runner.validate_preservation(instance, runner.build_counterfactuals(instance))
    assert harness.models.requests == []


def test_retrieval_budget_reaches_the_pinned_adapter(harness: Harness) -> None:
    instance = TaskInstance.model_validate(
        harness.store.read_record("task_instance", harness.instance_ids[0])
    )
    evidence = harness.runner()._retrieve(instance, {
        "adapter": _adapter("retrieval_by_budget"),
        "retrieval_configuration": {"evidence_context_budget": 1},
    })
    assert [item["evidence_id"] for item in evidence] == ["retrieved-0"]


def _finite_c3(harness: Harness, tmp_path: Path, **adapter_parameters: Any) -> None:
    _configured(harness, tmp_path)
    _reconfigure(harness, {
        "defenses.counterfactual_consistency.probability_extraction": {
            "interface": "candidate_log_likelihoods",
            "aggregation_space": "finite_candidates",
            "adapter": _adapter("candidate_probabilities", {"candidates": {"A": "A", "B": "B"}}),
            "label_pointer": "/answer/choice",
            "sum_tolerance": 1e-10,
            "candidate_output_adapter": _adapter("corrected_candidate", adapter_parameters),
        },
    })

    def task_answer(request: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "OK", "task": packet["task_name"], "case_id": packet["case_id"],
            "snapshot_date": packet["snapshot_date"], "answer": {"choice": "A"},
            "justification": "Fictional supplied evidence.", "evidence_refs": ["fixture-input"],
        }

    harness.models.script = task_answer


@pytest.mark.parametrize("invalid", ["wrong_task", "wrong_label"])
def test_c3_rejects_invalid_algorithmic_outputs(
    harness: Harness, tmp_path: Path, invalid: str,
) -> None:
    _finite_c3(harness, tmp_path, **{invalid: True})
    with pytest.raises(ValueError, match="identity|selected|label"):
        harness.runner().run_counterfactual_defense([harness.instance_ids[0]])


def test_finite_candidate_c3_scores_actual_task_prompts_and_reports_metrics(
    harness: Harness, tmp_path: Path,
) -> None:
    _finite_c3(harness, tmp_path)
    outcome = harness.runner().run_counterfactual_defense([harness.instance_ids[0]])
    assert outcome["status"] == "completed"
    corrected = harness.store.read_record("model_response", outcome["corrected_response_ids"][0])
    assert corrected["parsed_answer"]["answer"]["choice"] == "B"
    assert outcome["task_metrics"]["groups"]
    results = [item for item in harness.store.list_records("experiment_manifest")
               if item.get("purpose") == "counterfactual_consistency_results"]
    assert results[0]["diagnostics"][0]["omega"] == 0
    assert all(request["probability_interface"] is None for request in harness.models.requests)


def test_mitigation_group_prevalence_uses_all_responses_and_keeps_multiple_labels(
    harness: Harness,
) -> None:
    third = harness.store.read_record("model_response", harness.failure_ids[2])
    third.update(response_id="fixture-claude-response", model_id="CLD")
    response = ModelResponse.model_validate(third)
    harness.store.create_record("model_response", response.response_id, third)
    population = [*harness.failure_ids[:2], response.response_id]
    bindings = {}
    for response_id, labels in (
        (population[0], ["fixture-A", "fixture-B"]),
        (population[2], ["fixture-A"]),
    ):
        assignment = LabelAssignment(
            assignment_id=f"labels-{response_id}", response_id=response_id,
            taxonomy_id=harness.taxonomy.taxonomy_id, actor_id="fixture-coder",
            round_index=0, origin="consensus", labels=labels,
            evidence_spans=[_span(label) for label in labels],
        )
        harness.store.create_record("label_assignment", assignment.assignment_id,
                                    assignment.model_dump(mode="json"))
        harness.store.freeze_records("label_assignment", [assignment.assignment_id])
        bindings[response_id] = assignment.assignment_id
    result = harness.runner()._mitigation_prevalence(population, {
        "identification": {"failure_response_ids": [population[0], population[2]]},
        "categorization": {"final_assignment_ids_by_response": bindings},
    }, {"mode": "pooled_response_counts", "groups": {"general": ["G5"], "agent": ["CLD"]}})
    assert result["general"]["response_count"] == 2
    assert result["general"]["percentages"] == {"fixture-A": 50.0, "fixture-B": 50.0}
    assert result["agent"]["percentages"] == {"fixture-A": 100.0, "fixture-B": 0.0}


def test_repetition_ablation_runs_one_real_round_and_preserves_agreed_labels(harness: Harness) -> None:
    outcome = harness.runner()._categorization_ablation(
        "repetition", harness.failure_ids[:1], harness.taxonomy,
        {"strategy": "one_round_with_review"}, 0,
    )
    assert outcome["status"] == "completed"
    assert outcome["labels_by_response"] == {harness.failure_ids[0]: ["fixture-A"]}
    assert len(harness.models.requests) == 4
    assert harness.annotations.requests == []


def test_repetition_ablation_keeps_human_review_when_first_round_disagrees(harness: Harness) -> None:
    def disagreement(request: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
        label = "fixture-B" if request["model_id"] == "Claude Sonnet 4" else "fixture-A"
        return {"labels": [label], "evidence_spans": [_span(label)], "uncertain": False}

    harness.models.script = disagreement
    outcome = harness.runner()._categorization_ablation(
        "repetition", harness.failure_ids[:1], harness.taxonomy,
        {"strategy": "one_round_with_review"}, 0,
    )
    assert StageStatus(outcome["status"]) == StageStatus.WAITING_FOR_ANNOTATIONS
    assert len(harness.models.requests) == 4
    assert outcome["pending_annotation_task_ids"]
    assert outcome["labels_by_response"] == {}


def test_human_verification_ablation_uses_explicit_second_round_plurality(harness: Harness) -> None:
    def disagreement(request: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
        label = "fixture-B" if request["model_id"] == "Claude Sonnet 4" else "fixture-A"
        return {"labels": [label], "evidence_spans": [_span(label)], "uncertain": False}

    harness.models.script = disagreement
    outcome = harness.runner()._categorization_ablation(
        "human_verification", harness.failure_ids[:1], harness.taxonomy,
        {"strategy": "two_round_automated", "consensus_policy": {
            "mode": "label_set_plurality", "tie_policy": "block",
        }}, 0,
    )
    assert outcome["status"] == "completed"
    assert outcome["labels_by_response"] == {harness.failure_ids[0]: ["fixture-A"]}
    assert len(harness.models.requests) == 8
    assert harness.annotations.requests == []


def test_human_initialization_ablation_uses_declared_frozen_taxonomy(harness: Harness) -> None:
    result = harness.runner()._categorization_ablation(
        "human_initiation", harness.failure_ids[:1], harness.taxonomy,
        {"strategy": "fixed_initial_taxonomy"}, 0,
    )
    assert result["status"] == "completed"
    assert result["taxonomy_discovery_executed"] is False
    assert result["taxonomy_id"] == harness.taxonomy.taxonomy_id
    assert len(harness.models.requests) == 8
    assert harness.annotations.requests == []


def test_stability_rejects_runs_that_repeat_identical_sampling_seeds(harness: Harness) -> None:
    conditions = [{"condition_id": "init", "perturbations": ["initial_subset_seed"], "runs": [{}] * 5}]
    with pytest.raises(ValueError, match="independent|seed|perturbation"):
        harness.runner()._validate_stability_design(
            conditions, {("init", index): harness.config for index in range(5)}
        )


@pytest.mark.parametrize("defect", [None, "missing_width", "repeated_initialization", "changed_guideline"])
def test_stability_validates_the_effective_perturbation_schedule(
    harness: Harness, defect: str | None,
) -> None:
    configurations = {}
    widths = [0.025, 0.05, 0.1, 0.025, 0.05]
    for index, width in enumerate(widths):
        payload = harness.config.model_dump(mode="json")
        payload["failure_identification"]["stratification"]["quantile_width"] = 0.05 if defect == "missing_width" else width
        payload["failure_identification"]["stratification"]["anchor_sampling_seed"] = index + 20
        payload["taxonomy_discovery"]["initialization_seed"] = 1 if defect == "repeated_initialization" else index + 1
        if defect == "changed_guideline" and index == 4:
            payload["taxonomy"]["guideline_manifest"] = "another-guideline"
        effective = type(harness.config).model_validate(payload)
        configurations[("joint", index)] = effective
    conditions = [{
        "condition_id": "joint", "perturbations": ["initial_subset_seed", "anchor_selection", "quantile_width"],
        "runs": [{}] * 5,
    }]
    if defect is None:
        assert harness.runner()._validate_stability_design(conditions, configurations) is None
    else:
        with pytest.raises(ValueError, match="width|seed|guideline"):
            harness.runner()._validate_stability_design(conditions, configurations)


def retrieval_with_normalized_facts(*, payload: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    output = retrieval_by_budget(payload=payload, parameters=parameters)
    output["evidence"][0].update(
        canonical_entities=[{"surface": "an-alias", "canonical": "resolved-fixture-entity"}],
        normalized_record={"technique_id": "T-FIXTURE"}, claim_group="source-view-1",
        unrelated_internal_metadata="must-never-enter-task-prompt",
    )
    return output


def test_tailored_retrieval_places_normalized_facts_in_the_actual_task_prompt(
    harness: Harness, tmp_path: Path,
) -> None:
    _configure_c3(harness, tmp_path)
    policy_path = tmp_path / "subtype-policies.yaml"
    policy_path.write_text("version: offline-policies-v1\npolicies: {}\n", encoding="utf-8")
    _reconfigure(harness, {
        "operational.selected_defense": "tailored_mitigation",
        "operational.selected_mitigation_mode": "retrieval_only",
        "defenses.tailored_mitigation.subtype_policy_manifest": str(policy_path),
        "defenses.tailored_mitigation.routing_policy": {"mode": "universal", "uses_gold_failure_labels": False},
        "defenses.tailored_mitigation.model_group_aggregation": {
            "mode": "pooled_response_counts", "groups": {"general": ["G5"]},
        },
        "defenses.tailored_mitigation.retrieval_models": {"adapter": _adapter("retrieval_with_normalized_facts")},
        "defenses.tailored_mitigation.repetition_count": 1,
        "defenses.tailored_mitigation.evidence_context_budget": 1,
        "failure_identification.hard_label_rules": {"tasks": {"threat_actor_linking": {"route": "hard"}}},
        "evaluation.scoring.response_level_failure_criteria": {"threat_actor_linking": {
            "pass_if": {"op": "equal", "left": {"source": "prediction", "pointer": "/answer/choice"},
                        "right": {"source": "reference", "pointer": "/answer/choice"}},
            "on_missing": "error", "availability": {},
        }},
    })

    def correct(request: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
        return {"status": "OK", "task": packet["task_name"], "case_id": packet["case_id"],
                "snapshot_date": packet["snapshot_date"], "answer": {"choice": "B"},
                "justification": "Supplied fictional evidence."}

    harness.models.script = correct
    outcome = harness.runner().run_tailored_mitigation("retrieval_only", harness.instance_ids[:1])
    assert outcome["status"] == "completed"
    rendered = harness.models.requests[0]["prompt"]
    assert "resolved-fixture-entity" in rendered
    assert "T-FIXTURE" in rendered
    assert "source-view-1" in rendered
    assert "must-never-enter-task-prompt" not in rendered


def test_stability_builtin_assigner_runs_rq3_and_projects_back_to_original_audit_ids(
    harness: Harness,
) -> None:
    protocol = copy.deepcopy(harness.config.categorization.incomplete_agent_output_policy)
    annotation = protocol["annotation_protocol"]
    annotation["assignments"] = {"*": ["fixture-reviewer"]}
    annotation["reference_applicability"] = {
        "*": next(iter(annotation["reference_applicability"].values())),
    }
    _reconfigure(harness, {"categorization.incomplete_agent_output_policy": protocol})
    source_id = harness.failure_ids[0]
    harness.annotations.references[(source_id, "expanded_label_audit")] = {
        "labels": ["fixture-B"], "audit_only": True,
    }
    outcome = harness.runner()._stability_audit_assignments(
        harness.config, [source_id], harness.taxonomy,
        {"manifest_id": "fixture-stability-condition"}, 0,
        models=harness.models, annotations=harness.annotations,
    )
    assert outcome["status"] == "completed"
    assert set(outcome["assignment_ids_by_response"]) == {source_id}
    assignment = harness.store.read_record(
        "label_assignment", outcome["assignment_ids_by_response"][source_id]
    )
    assert assignment["response_id"] == source_id
    assert assignment["labels"] == ["fixture-A"]
    assert len(harness.models.requests) == 8
    assert harness.annotations.requests == []
    projections = [item for item in harness.store.list_records("experiment_manifest")
                   if item.get("purpose") == "stability_audit_assignment_projection"]
    assert projections[0]["source_audit_response_id"] == source_id
    assert projections[0]["condition_response_id"] != source_id


def embed_failed_responses(*, payload: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    assert all("synthetic passed response" not in text for text in payload["texts"])
    return {"vectors": [[float(index), 1.0] for index, _ in enumerate(payload["texts"])]}


def test_baseline_classifies_only_failures_but_counts_the_complete_identified_population(
    harness: Harness,
) -> None:
    passed = harness.store.read_record("model_response", harness.failure_ids[1])
    passed.update(response_id="synthetic-passed-response", raw_text="synthetic passed response")
    harness.store.create_record("model_response", passed["response_id"], passed)
    passed_decision = FailureDecision(
        response_id=passed["response_id"], identification_version="fixture-identification-v1",
        decision="pass", decision_origin="hard_label",
    )
    passed_decision_id = _content_hash(passed_decision)
    harness.store.create_record("failure_decision", passed_decision_id, passed_decision.model_dump(mode="json"))
    harness.store.freeze_records("failure_decision", [passed_decision_id])
    failure_id = harness.failure_ids[0]
    failure_decision = harness.store.list_records("failure_decision", {"response_id": failure_id})[0]
    population = {
        "manifest_id": "complete-identified-population", "purpose": "identification_population",
        "config_hash": harness.config.config_hash, "corpus_version": harness.config.corpus_version,
        "members": [{"response_id": failure_id}, {"response_id": passed["response_id"]}],
    }
    harness.store.create_record("experiment_manifest", population["manifest_id"], population)
    final = {
        "manifest_id": "complete-identification-final", "purpose": "identification_final",
        "config_hash": harness.config.config_hash, "corpus_version": harness.config.corpus_version,
        "result": {
            "status": "completed", "coverage": {"complete": True},
            "population_manifest": population["manifest_id"],
            "failure_response_ids": [failure_id],
            "decision_ids_by_response": {failure_id: _content_hash(failure_decision),
                                         passed["response_id"]: passed_decision_id},
        },
    }
    harness.store.create_record("experiment_manifest", final["manifest_id"], final)
    _reconfigure(harness, {
        "operational.selected_baseline": "kmeans", "baselines.repetition_count": 1,
        "baselines.prevalence_denominator": {"mode": "identified_responses"},
        "baselines.kmeans.embedding_model": {"adapter": _adapter("embed_failed_responses")},
        "baselines.kmeans.clusters": 1,
        "baselines.kmeans.input_representation": {"parts": [{
            "source": "response", "pointer": "/raw_text", "serialization": "text",
        }], "separator": "\n"},
        "baselines.kmeans.initialization": {
            "parameters": {"init": "k-means++", "n_init": 1, "max_iter": 10, "tol": 1e-4,
                           "verbose": 0, "copy_x": True, "algorithm": "lloyd"},
            "run_seeds": [0], "summary_ddof": 0,
        },
        "baselines.kmeans.cluster_to_subtype_mapping": {
            "frozen_before_evaluation": True, "uses_held_out_gold_labels": False,
            "labels": {"0": ["fixture-A"]},
        },
    })
    result = harness.runner().run_baseline("kmeans", [failure_id, passed["response_id"]])
    predictions = harness.store.read_record("experiment_manifest", result["prediction_manifest_ids"][0])
    assert predictions["labels_by_response"] == {failure_id: ["fixture-A"]}
    assert predictions["prevalence"]["denominator"] == 2
    assert predictions["prevalence"]["percentages"]["fixture-A"] == 50.0
    assert result["excluded_pass_response_ids"] == [passed["response_id"]]
