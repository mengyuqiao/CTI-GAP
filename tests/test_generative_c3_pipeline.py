"""Offline end-to-end generative C3 with the real exact token-tree backend.

The observed original completion is scripted through the existing model-client
test interface. Probability artifacts are never scripted: a source-pinned tiny
autoregressive grammar supplies full next-token distributions to the production
enumerator. Real prompt rendering, JSON schema validation, preservation binding,
artifact persistence, task projection, entropy aggregation and BLEU all execute.
These are declared test fixtures, not reproduced paper measurements.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

import generative_probabilities
from schemas import TaskInstance, _canonical_json, _content_hash, _task_content_projection
from storage import ArtifactStore
from test_taxonomy_and_categorization import Harness, _configure_c3, _reconfigure, harness, no_live_inference


CHECKPOINT = "a" * 64


def grammar_oracle(*, payload: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    """A full-vocabulary autoregressive grammar, with branching at two steps."""
    assert "ROLE AND GLOBAL CONSTRAINTS" in payload["prompt"]
    assert "fixture-reference" not in payload["prompt"]
    packet = payload["model_input"]
    assert packet["environment"]["report_label"] in payload["prompt"]
    report = packet["environment"]["report_label"]
    values = parameters["values"]
    # The field and justification are separate random decisions. Different full
    # JSON completions can therefore encode exactly the same task judgment.
    prefix = _canonical_json({
        "status": "OK", "task": packet["task_name"], "case_id": packet["case_id"],
        "snapshot_date": packet["snapshot_date"],
    })[:-1] + ',"answer":{"value":'
    fragments = {
        0: "", 1: prefix, 2: _canonical_json(values["original"]),
        3: _canonical_json(values["corrected"]), 4: _canonical_json(values["other"]),
        5: '},"justification":"First supported explanation.","evidence_refs":["fixture-input"]}',
        6: '},"justification":"Second supported explanation.","evidence_refs":["fixture-input"]}',
        7: "broken-output-one", 8: "broken-output-two",
    }
    invalid_kind = parameters.get("scenario")
    if report != "original":
        if invalid_kind in {"wrong_task", "wrong_case", "wrong_snapshot"}:
            field = {"wrong_task": "task", "wrong_case": "case_id", "wrong_snapshot": "snapshot_date"}[invalid_kind]
            identity = json.loads(prefix.split(',"answer":')[0] + "}")
            identity[field] = "another-task" if field != "snapshot_date" else "2023-01-01"
            fragments[1] = _canonical_json(identity)[:-1] + ',"answer":{"value":'
        elif invalid_kind == "schema_violation":
            fragments.update({2: "7", 3: "8", 4: "9"})
        elif invalid_kind == "missing_field":
            fragments[1] = prefix.replace('"value":', '"missing":')
        elif invalid_kind == "duplicate_keys":
            fragments[1] = prefix.replace('"value":', '"value":"silently replaced","value":')
        elif invalid_kind == "long_justification":
            long_text = " ".join(["word"] * 41)
            fragments[5] = fragments[6] = '},"justification":' + _canonical_json(long_text) + "}"

    def next_token_probabilities(tokens: tuple[int, ...]) -> dict[int, float]:
        probabilities = {token: 0.0 for token in fragments}
        if not tokens:
            probabilities[1] = 1.0
        elif len(tokens) == 1:
            mode = parameters.get("scenario", "weighted")
            if mode == "invalid_maximum" and report != "original":
                probabilities.update({7: 0.6, 3: 0.3, 8: 0.1})
            elif mode == "distinct_invalids" and report != "original":
                probabilities.update({7: 0.26, 8: 0.25, 3: 0.49})
            elif report == "original":
                probabilities.update({2: 0.75, 4: 0.25})
            elif report == "variant-3":
                probabilities.update({3: 0.99, 4: 0.01})
            else:
                probabilities.update({2: 0.51, 3: 0.49})
        elif len(tokens) == 2 and tokens[1] in {2, 3, 4}:
            probabilities.update({5: 0.6, 6: 0.4})
        else:
            probabilities[0] = 1.0
        return probabilities

    return {
        "next_token_probabilities": next_token_probabilities,
        "decode": lambda tokens: "".join(fragments[token] for token in tokens),
        "eos_token_ids": [0],
        "provenance": {
            **{field: payload[field] for field in ("model_id", "endpoint_or_checkpoint", "revision", "message_mapping")},
            "checkpoint_sha256": payload["generation_checkpoint_sha256"],
            "chat_template_hash": "b" * 64,
            "oracle": "offline two-decision full-vocabulary grammar",
        },
    }


def _configure_generative(
    harness: Harness, tmp_path: Path, *, task: str = "threat_report_alignment",
    values: dict[str, Any] | None = None, scenario: str = "weighted", max_nodes: int = 100,
    horizon: int = 4,
) -> TaskInstance:
    _configure_c3(harness, tmp_path)
    source = TaskInstance.model_validate(harness.store.read_record("task_instance", harness.instance_ids[0]))
    values = values or {
        "original": "unsupported red smoke narrative", "corrected": "alpha beta gamma delta",
        "other": "independent third judgment",
    }
    payload = source.model_dump(mode="json")
    payload.update(task_name=task, stage="ATTRIBUTION" if task == "evidence_weighting" else "CONTEXTUALIZATION",
                   reference={"version": "fixture-reference-v1", "answer": {"value": values["corrected"]}})
    payload.pop("content_hash")
    draft = TaskInstance.model_validate(payload)
    payload["content_hash"] = _content_hash(_task_content_projection(draft))
    instance = TaskInstance.model_validate(payload)
    # A separate frozen corpus avoids editing the reusable attribution fixture.
    store = ArtifactStore(str(tmp_path / "generative.sqlite"), str(tmp_path / "generative-blobs"))
    store.initialize()
    store.create_record("task_instance", instance.instance_id, instance.model_dump(mode="json"))
    store.freeze_records("task_instance", [instance.instance_id])
    corpus = {"manifest_id": "generative-corpus", "corpus_version": harness.config.corpus_version,
              "members": [{"instance_id": instance.instance_id, "content_hash": instance.content_hash}]}
    store.create_record("corpus_manifest", corpus["manifest_id"], corpus)
    store.freeze_records("corpus_manifest", [corpus["manifest_id"]])
    harness.store = harness.models.store = harness.annotations.store = store
    harness.instance_ids = [instance.instance_id]
    guidance_path = tmp_path / "generative-guidance.json"
    profiles_path = tmp_path / "generative-profiles.json"
    schema = {
        "type": "object", "required": ["status", "task", "case_id", "snapshot_date", "answer", "justification"],
        "properties": {
            "status": {"type": "string"}, "task": {"type": "string"},
            "case_id": {"type": "string"}, "snapshot_date": {"type": "string"},
            "justification": {"type": "string"},
            "answer": {"type": "object", "required": ["value"], "properties": {
                "value": {"anyOf": [{"type": "string"}, {"type": "array"}]},
            }}, "evidence_refs": {"type": "array", "items": {"type": "string"}},
        },
    }
    guidance_path.write_text(_canonical_json({"version": "offline-v1", "tasks": {
        task: {"guidance": "Answer using only supplied evidence.", "candidate_policy": "include"},
    }}))
    profiles_path.write_text(_canonical_json({"version": "offline-v1", "tasks": {
        task: {"output_profile": instance.output_profile, "schema_contract": {
            "version": "offline-generic-task-v1", "kind": "task", "json_schema": schema,
        }},
    }}))
    oracle_spec = {
        "callable": f"{__name__}:grammar_oracle", "version": "offline-grammar-v1",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameters": {"values": values, "scenario": scenario},
    }
    adapter = {
        "callable": "generative_probabilities:complete_output_distribution", "version": "exact-eos-horizon-v1",
        "source_sha256": hashlib.sha256(Path(generative_probabilities.__file__).read_bytes()).hexdigest(),
        "parameters": {
            "mode": "oracle_adapter", "oracle_adapter": oracle_spec, "model_id": "G5",
            "endpoint_or_checkpoint": harness.config.models.evaluation["G5"].endpoint_or_checkpoint,
            "revision": harness.config.models.evaluation["G5"].revision,
            "checkpoint_sha256": CHECKPOINT, "context_window": 100000,
            "max_new_tokens": horizon, "max_nodes": max_nodes, "sum_tolerance": 1e-10,
        },
    }
    operators = copy.deepcopy(harness.config.defenses.counterfactual_consistency.operators)
    for decision in operators["preservation_protocol"]["decisions"].values():
        decision["original_content_hash"] = instance.content_hash
    _reconfigure(harness, {
        "operational.selected_tasks": [task], "inference.max_output_tokens": horizon,
        "evaluation.standard_deviation_ddof": 0,
        "evaluation.prompt.task_guidance_manifest": str(guidance_path),
        "evaluation.prompt.output_profile_manifest": str(profiles_path),
        "evaluation.scoring.task_field_extraction_rules": {task: {
            "prediction": "/answer/value", "reference": "/answer/value", "representation": "text",
            "multiple_references": False,
        }},
        "evaluation.scoring.bleu.aggregation_level": "corpus",
        "evaluation.scoring.bleu.tokenizer": "none", "evaluation.scoring.bleu.maximum_ngram_order": 2,
        "evaluation.scoring.bleu.smoothing": {"method": "none", "value": None, "effective_order": True},
        "evaluation.scoring.bleu.lowercase": False,
        "evaluation.scoring.bleu.structured_output_serialization": "text" if isinstance(values["corrected"], str) else "canonical_json",
        "defenses.counterfactual_consistency.operators": operators,
        "defenses.counterfactual_consistency.output_equivalence_rule": "task_field_exact",
        "defenses.counterfactual_consistency.probability_extraction": {
            "interface": "autoregressive_complete", "aggregation_space": "task_outputs",
            "label_pointer": "/answer/value", "sum_tolerance": 1e-10,
            "generation_checkpoint_sha256": CHECKPOINT, "adapter": adapter,
        },
    })
    for variant in harness.runner().build_counterfactuals(instance):
        operators["preservation_protocol"]["decisions"][variant["operator_id"]][
            "modified_model_input_hash"] = variant["modified_model_input_hash"]
    _reconfigure(harness, {"defenses.counterfactual_consistency.operators": operators})

    def observed_output(request: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
        return {"status": "OK", "task": packet["task_name"], "case_id": packet["case_id"],
                "snapshot_date": packet["snapshot_date"], "answer": {"value": values["original"]},
                "justification": "Observed offline model completion.", "evidence_refs": ["fixture-input"]}
    harness.models.script = observed_output
    return instance


def _artifacts(harness: Harness, purpose: str) -> list[dict[str, Any]]:
    return [item for item in harness.store.list_records("experiment_manifest") if item.get("purpose") == purpose]


def test_bleu_c3_uses_exact_tree_projection_union_and_persists_provenance(harness: Harness, tmp_path: Path) -> None:
    _configure_generative(harness, tmp_path)
    outcome = harness.runner().run_counterfactual_defense(harness.instance_ids)
    assert outcome["status"] == "completed"
    assert outcome["probability_enumeration_status"] == "complete"
    corrected = harness.store.read_record("model_response", outcome["corrected_response_ids"][0])
    assert corrected["parsed_answer"]["answer"]["value"] == "alpha beta gamma delta"
    assert corrected["parsed_answer"]["justification"] == "First supported explanation."
    diagnostic = _artifacts(harness, "counterfactual_consistency_results")[0]["diagnostics"][0]
    assert diagnostic["omega"] == 0.75
    assert diagnostic["vulnerable"] is False
    populations = _artifacts(harness, "counterfactual_complete_output_probabilities")
    assert len(populations) == 5
    original = next(item for item in populations if item["request"]["model_input"]["environment"]["report_label"] == "original")
    selected = _canonical_json("alpha beta gamma delta")
    assert selected not in original["task_distribution"]  # Union support must introduce this judgment.
    assert selected in diagnostic["weighted_task_totals"]
    entropy = -(0.51 * math.log(0.51) + 0.49 * math.log(0.49))
    assert diagnostic["log_weights"][0] == pytest.approx(-entropy)
    for population in populations:
        assert population["total_probability_mass"] == pytest.approx(1)
        assert len(population["backend_output"]["outcomes"]) == 4
        assert len(population["task_distribution"]) == 2  # Two justification variants merge per judgment.
        assert population["request"]["prompt"] == population["prompt"]
        assert population["backend_output"]["provenance"]["complete"] is True
        assert population["request"]["generation_checkpoint_sha256"] == CHECKPOINT
        assert population["request"]["message_mapping"] == "single_user_message"
    scores = harness.store.list_records("score")
    assert {score["condition"] for score in scores} == {"original", "corrected"}
    scored_original = harness.store.read_record("model_response", outcome["original_response_ids"][0])
    assert scored_original["usage"]["probability_artifact"]["rendered_prompt"] == original["prompt"]
    groups = {group["condition"]: group for group in outcome["task_metrics"]["groups"]}
    assert groups["corrected"]["summary"]["mean"] == pytest.approx(100)
    assert groups["original"]["summary"]["mean"] == 0
    assert corrected["usage"]["representative_outcome"]["probability"] == pytest.approx(0.99 * 0.6)


@pytest.mark.parametrize("task,values", [
    ("event_timeline_construction", {"original": [{"event": "wrong"}], "corrected": [{"event": "entry"}, {"event": "exit"}], "other": []}),
    ("evidence_weighting", {"original": [0.1, 0.9], "corrected": [0.8, 0.2], "other": [0.5, 0.5]}),
])
def test_c3_preserves_structured_task_field_types(harness: Harness, tmp_path: Path, task: str, values: dict[str, Any]) -> None:
    _configure_generative(harness, tmp_path, task=task, values=values)
    outcome = harness.runner().run_counterfactual_defense(harness.instance_ids)
    assert outcome["status"] == "completed"
    corrected = harness.store.read_record("model_response", outcome["corrected_response_ids"][0])
    assert corrected["parsed_answer"]["answer"]["value"] == values["corrected"]
    assert corrected["usage"]["selected_task_judgment"] == _canonical_json(values["corrected"])
    assert next(group for group in outcome["task_metrics"]["groups"] if group["condition"] == "corrected")["summary"]["mean"] == pytest.approx(100)


def test_distinct_invalid_raw_outputs_keep_mass_and_do_not_form_a_single_bucket(harness: Harness, tmp_path: Path) -> None:
    _configure_generative(harness, tmp_path, scenario="distinct_invalids")
    outcome = harness.runner().run_counterfactual_defense(harness.instance_ids)
    assert outcome["status"] == "completed"
    population = next(item for item in _artifacts(harness, "counterfactual_complete_output_probabilities")
                      if item["request"]["model_input"]["environment"]["report_label"] != "original")
    invalid = {key: value for key, value in population["task_distribution"].items() if key.startswith("invalid_raw:")}
    assert sorted(invalid.values()) == pytest.approx([0.25, 0.26])
    assert population["task_distribution"][_canonical_json("alpha beta gamma delta")] == pytest.approx(0.49)
    assert sum(population["task_distribution"].values()) == pytest.approx(1)


def test_invalid_maximum_blocks_correction_without_renormalization(harness: Harness, tmp_path: Path) -> None:
    _configure_generative(harness, tmp_path, scenario="invalid_maximum")
    outcome = harness.runner().run_counterfactual_defense(harness.instance_ids)
    assert outcome["status"] == "blocked"
    assert "invalid task JSON" in outcome["reason"]
    assert outcome["probability_enumeration_status"] == "complete"
    assert _artifacts(harness, "counterfactual_consistency_results") == []
    assert not any(item["usage"].get("evaluation_context", {}).get("condition") == "corrected"
                   for item in harness.store.list_records("model_response"))


@pytest.mark.parametrize("invalid_kind", ["wrong_task", "wrong_case", "wrong_snapshot", "schema_violation", "missing_field", "long_justification", "duplicate_keys"])
def test_complete_raw_json_still_requires_original_task_contract(harness: Harness, tmp_path: Path, invalid_kind: str) -> None:
    _configure_generative(harness, tmp_path, scenario=invalid_kind)
    outcome = harness.runner().run_counterfactual_defense(harness.instance_ids)
    assert outcome["status"] == "blocked"
    assert "invalid task JSON" in outcome["reason"]
    population = next(item for item in _artifacts(harness, "counterfactual_complete_output_probabilities")
                      if item["request"]["model_input"]["environment"]["report_label"] != "original")
    assert all(not item["schema_and_task_valid"] for item in population["projected_outcomes"])
    assert sum(population["task_distribution"].values()) == pytest.approx(1)


def test_repeated_generative_c3_has_one_complete_metric_group_per_condition(harness: Harness, tmp_path: Path) -> None:
    _configure_generative(harness, tmp_path)
    _reconfigure(harness, {"defenses.counterfactual_consistency.repetition_count": 2})
    outcome = harness.runner().run_counterfactual_defense(harness.instance_ids)
    assert outcome["status"] == "completed"
    groups = outcome["task_metrics"]["groups"]
    assert len(groups) == 2
    assert {group["condition"] for group in groups} == {"original", "corrected"}
    assert all(group["summary"]["status"] == "complete" for group in groups)
    assert all(group["summary"]["observed_runs"] == 2 for group in groups)
    assert all(len(group["runs"]) == 2 for group in groups)
    assert all(set(group["evaluation_context"]) == {"protocol", "condition", "experiment_manifest_id"}
               for group in groups)


def test_omega_compares_variant_argmax_with_observed_original_field(harness: Harness, tmp_path: Path) -> None:
    _configure_generative(harness, tmp_path)
    observed = harness.models.script

    def supported_nonmodal_original(request: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
        answer = observed(request, packet)
        answer["answer"]["value"] = "independent third judgment"
        return answer

    harness.models.script = supported_nonmodal_original
    outcome = harness.runner().run_counterfactual_defense(harness.instance_ids)
    assert outcome["status"] == "completed"
    diagnostic = _artifacts(harness, "counterfactual_consistency_results")[0]["diagnostics"][0]
    assert diagnostic["omega"] == 0


def test_budget_exhaustion_persists_blocked_artifact_without_partial_distribution(harness: Harness, tmp_path: Path) -> None:
    _configure_generative(harness, tmp_path, max_nodes=2)
    outcome = harness.runner().run_counterfactual_defense(harness.instance_ids)
    assert outcome["status"] == "blocked"
    assert outcome["probability_enumeration_status"] == "blocked"
    assert outcome["reason"] == "max_nodes_exceeded"
    population = _artifacts(harness, "counterfactual_complete_output_probabilities")[0]
    assert population["backend_output"]["outcomes"] == []
    assert population["backend_output"]["provenance"]["complete"] is False
    assert "task_distribution" not in population
    assert _artifacts(harness, "counterfactual_consistency_results") == []


def test_generative_horizon_must_match_task_generation(harness: Harness, tmp_path: Path) -> None:
    _configure_generative(harness, tmp_path)
    _reconfigure(harness, {"inference.max_output_tokens": 5})
    with pytest.raises(ValueError, match="max_new_tokens.*max_output_tokens"):
        harness.runner().run_counterfactual_defense(harness.instance_ids)
    assert harness.models.requests == []


def test_max_token_leaves_keep_original_probability_and_truncation_provenance(harness: Harness, tmp_path: Path) -> None:
    _configure_generative(harness, tmp_path, horizon=3)
    outcome = harness.runner().run_counterfactual_defense(harness.instance_ids)
    assert outcome["status"] == "completed"
    population = _artifacts(harness, "counterfactual_complete_output_probabilities")[0]
    assert all(item["truncated"] for item in population["backend_output"]["outcomes"])
    assert sum(item["probability"] for item in population["backend_output"]["outcomes"]) == pytest.approx(1)
