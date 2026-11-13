"""F2: production C3 uses the counterfactual-only published estimator.

The exact grammar and finite/token adapters below are explicit offline
probability fixtures. Prompt rendering, enumeration, projection, correction,
schema validation, scoring, and persistence use the production runner.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from config import ExperimentConfig
from schemas import _canonical_json
from test_generative_c3_pipeline import _configure_generative
from test_taxonomy_and_categorization import (
    FixedIdentityConfig, Harness, _baseline, _configure_c3, _reconfigure,
    harness, no_live_inference,
)


def audit_candidate_probabilities(*, payload: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    """Known probabilities at the genuine external scoring boundary."""
    assert "ROLE AND GLOBAL CONSTRAINTS" in payload["prompt"]
    original = payload["model_input"]["environment"]["report_label"] == "original"
    return {
        "probabilities": {"A": 0.999 if original else 0.4, "B": 0.001 if original else 0.6},
        "provenance": {"method": "audit-f2-offline-counterexample-v1"},
    }


def audit_grammar_oracle(*, payload: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    """Full next-token distributions; the production enumerator multiplies paths."""
    assert "ROLE AND GLOBAL CONSTRAINTS" in payload["prompt"]
    packet = payload["model_input"]
    original = packet["environment"]["report_label"] == "original"
    fragments = {0: ""}
    for token, value in [(1, "A"), (2, "B")]:
        fragments[token] = _canonical_json({
            "status": "OK", "task": packet["task_name"], "case_id": packet["case_id"],
            "snapshot_date": packet["snapshot_date"], "answer": {"value": value},
            "justification": "Offline estimator counterexample.", "evidence_refs": ["fixture-input"],
        })

    def next_token_probabilities(tokens: tuple[int, ...]) -> dict[int, float]:
        if not tokens:
            return {0: 0.0, 1: 0.999 if original else 0.4, 2: 0.001 if original else 0.6}
        return {0: 1.0, 1: 0.0, 2: 0.0}

    return {
        "next_token_probabilities": next_token_probabilities,
        "decode": lambda tokens: "".join(fragments[token] for token in tokens),
        "eos_token_ids": [0],
        "provenance": {
            **{key: payload[key] for key in ("model_id", "endpoint_or_checkpoint", "revision", "message_mapping")},
            "checkpoint_sha256": payload["generation_checkpoint_sha256"],
            "chat_template_hash": "b" * 64, "oracle": "audit-f2-offline-counterexample-v1",
        },
    }


def _configure_counterexample(harness: Harness, tmp_path: Path, interface: str) -> None:
    if interface == "autoregressive_complete":
        _configure_generative(harness, tmp_path, values={"original": "A", "corrected": "B", "other": "C"})
        extraction = copy.deepcopy(harness.config.defenses.counterfactual_consistency.probability_extraction)
        oracle = extraction["adapter"]["parameters"]["oracle_adapter"]
        oracle.update({
            "callable": f"{__name__}:audit_grammar_oracle",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "parameters": {},
        })
    else:
        _configure_c3(harness, tmp_path)
        extraction = copy.deepcopy(harness.config.defenses.counterfactual_consistency.probability_extraction)
        if interface == "candidate_log_likelihoods":
            extraction.update({
                "interface": interface, "aggregation_space": "finite_candidates",
                "label_pointer": "/answer/choice",
                "adapter": {
                    "callable": f"{__name__}:audit_candidate_probabilities", "version": "audit-f2-v1",
                    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "parameters": {"candidates": {"A": "A", "B": "B"}},
                },
            })
            extraction.pop("token_to_label")

        def observed_output(request: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
            original = packet["environment"]["report_label"] == "original"
            answer = {
                "status": "OK", "task": packet["task_name"], "case_id": packet["case_id"],
                "snapshot_date": packet["snapshot_date"], "answer": {"choice": "A" if original else "B"},
                "justification": "Offline estimator counterexample.", "evidence_refs": ["fixture-input"],
            }
            if interface == "token_logprobs":
                answer["__fixture_probability__"] = {
                    "interface": "token_logprobs", "data": {"content": [{"top_logprobs": [
                        {"token": "A", "logprob": math.log(0.999 if original else 0.4)},
                        {"token": "B", "logprob": math.log(0.001 if original else 0.6)},
                    ]}]},
                }
            return answer

        harness.models.script = observed_output

    operators = copy.deepcopy(harness.config.defenses.counterfactual_consistency.operators)
    operators["items"] = operators["items"][:1]
    _reconfigure(harness, {
        "defenses.counterfactual_consistency.operators": operators,
        "defenses.counterfactual_consistency.counterfactuals_per_instance": 1,
        "defenses.counterfactual_consistency.probability_extraction": extraction,
        "defenses.counterfactual_consistency.normalize_weights": False,
    })


@pytest.mark.parametrize("interface", ["token_logprobs", "candidate_log_likelihoods", "autoregressive_complete"])
def test_legacy_original_augmentation_cannot_reverse_the_published_correction(
    harness: Harness, tmp_path: Path, interface: str,
) -> None:
    """An enabled original append used to select A instead of the paper's B."""
    _configure_counterexample(harness, tmp_path, interface)
    _reconfigure(harness, {"defenses.counterfactual_consistency.include_original_input": True})
    try:
        outcome = harness.runner().run_counterfactual_defense([harness.instance_ids[0]])
    except ValueError as error:
        assert "counterfactual-only" in str(error)
        assert harness.models.requests == []
        return
    results = [record for record in harness.store.list_records("experiment_manifest")
               if record.get("purpose") == "counterfactual_consistency_results"]
    corrected = harness.store.read_record("model_response", outcome["corrected_response_ids"][0])
    pytest.fail(f"Original augmentation ran and selected {corrected['parsed_answer']['answer']}; "
                f"counterexample diagnostic: {results[0]['diagnostics'][0]}")


@pytest.mark.parametrize("interface", ["token_logprobs", "candidate_log_likelihoods", "autoregressive_complete"])
@pytest.mark.parametrize("normalized", [False, True])
def test_default_c3_corrects_from_counterfactuals_and_preserves_exact_entropy_weights(
    harness: Harness, tmp_path: Path, interface: str, normalized: bool,
) -> None:
    """A high-confidence original must not enter either production estimator."""
    _configure_counterexample(harness, tmp_path, interface)
    _reconfigure(harness, {"defenses.counterfactual_consistency.normalize_weights": normalized})
    payload = harness.config.model_dump(mode="json")
    payload["defenses"]["counterfactual_consistency"].pop("include_original_input")
    payload["implementation_choices"].pop("defenses.counterfactual_consistency.include_original_input")
    harness.config = FixedIdentityConfig.model_validate(payload)
    harness.models.config = harness.config
    outcome = harness.runner().run_counterfactual_defense([harness.instance_ids[0]])
    assert outcome["status"] == "completed"
    corrected = harness.store.read_record("model_response", outcome["corrected_response_ids"][0])
    field = "value" if interface == "autoregressive_complete" else "choice"
    assert corrected["parsed_answer"]["answer"][field] == "B"
    results = next(record for record in harness.store.list_records("experiment_manifest")
                   if record.get("purpose") == "counterfactual_consistency_results")
    diagnostic = results["diagnostics"][0]
    assert diagnostic["omega"] == 0
    assert len(diagnostic["variant_response_ids"]) == 1
    # Decimal arithmetic evaluated independently at 45 digits from .4 and .6.
    assert diagnostic["weights"] == pytest.approx([1.0 if normalized else 0.5101698002503163])
    key = "weighted_task_totals" if interface == "autoregressive_complete" else "weighted_candidate_totals"
    label_a = '"A"' if interface == "autoregressive_complete" else "A"
    label_b = '"B"' if interface == "autoregressive_complete" else "B"
    assert diagnostic[key] == pytest.approx({
        label_a: 0.4 if normalized else 0.20406792010012653,
        label_b: 0.6 if normalized else 0.3061018801501898,
    })
    for artifact in [outcome, results, corrected["usage"], diagnostic]:
        assert artifact["correction_estimator"] == "counterfactual_only"
    experiment = harness.store.read_record("experiment_manifest", outcome["experiment_manifest_id"])
    assert experiment["correction_estimator"] == "counterfactual_only"
    assert experiment["original_input_role"] == "consistency_target_and_baseline"
    assert diagnostic["aggregation_response_ids"] == diagnostic["variant_response_ids"]
    assert corrected["usage"]["aggregation_response_ids"] == diagnostic["variant_response_ids"]
    assert outcome["original_response_ids"][0] not in diagnostic["aggregation_response_ids"]
    if interface == "autoregressive_complete":
        original = harness.store.read_record("model_response", outcome["original_response_ids"][0])
        original_probability_id = original["usage"]["probability_artifact"]["manifest_id"]
        assert original_probability_id in diagnostic["probability_manifest_ids"]
        assert original_probability_id not in diagnostic["aggregation_probability_manifest_ids"]
        assert len(diagnostic["aggregation_probability_manifest_ids"]) == 1
        assert corrected["usage"]["aggregation_probability_manifest_ids"] == diagnostic["aggregation_probability_manifest_ids"]


@pytest.mark.parametrize("requested", [True, None])
def test_production_config_rejects_original_augmentation_before_execution(tmp_path: Path, requested: Any) -> None:
    payload = _baseline()
    payload["defenses"]["counterfactual_consistency"]["include_original_input"] = requested
    config_path = tmp_path / "unsupported-augmentation.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="include_original_input"):
        ExperimentConfig.load(str(config_path))
