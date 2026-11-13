"""Offline measurement and RQ1 regression tests.

All executable policies below are explicitly marked test-only implementation
choices derived from the supplied config.yaml. Fictional observations are test
data, not reproduced experimental results.

Identification fixtures use the production configuration hash and its runtime
protocol binding. No configuration validation or provenance hashing is bypassed.

ScriptedAnnotations implements the AnnotationService methods consumed by RQ1.
It persists independent human-reference artifacts but performs no inference,
sampling, adjudication heuristics, or interaction with real annotators.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from annotation_app import AnnotationService
from config import ExperimentConfig
from evaluation import Evaluator
from failure_identification import FailureIdentifier
from schemas import (
    AnnotationPurpose,
    AnnotationTask,
    AuditCohort,
    DecisionOrigin,
    FailureDecision,
    FailureStatus,
    ModelResponse,
    ResponseStatus,
    TaskInstance,
    _canonical_json,
    _content_hash,
    _task_content_projection,
    _text_hash,
)
from storage import ArtifactStore


PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"
CORPUS_VERSION: str = "fictional-metrics-corpus-v1"
REFERENCE_VERSION: str = "fictional-reference-v1"
SIMILARITY_VERSION: str = "fictional-similarity-v1"

TASK_STAGES: dict[str, str] = {
    "affected_systems": "CONTEXTUALIZATION",
    "threat_actor_linking": "ATTRIBUTION",
    "exploit_likelihood": "PREDICTION",
    "threat_report_alignment": "CONTEXTUALIZATION",
    "event_timeline_construction": "CONTEXTUALIZATION",
    "rule_generation_yara": "MITIGATION",
    "countermeasure_ranking": "MITIGATION",
}


def _resolve(
    payload: dict[str, Any],
    path: str,
    value: Any,
    stages: Sequence[str] = ("identify", "evaluate"),
) -> None:
    """Set an explicit fixture policy and record its implementation-choice provenance."""
    current: dict[str, Any] = payload
    components: list[str] = path.split(".")
    for component in components[:-1]:
        current = current.setdefault(component, {})
    current[components[-1]] = copy.deepcopy(value)
    choices: dict[str, Any] = payload.setdefault("implementation_choices", {})
    choices[path] = {
        "configuration_path": path,
        "origin": "implementation",
        "rationale": (
            "Offline numerical fixture only; not a setting reported by the paper."
        ),
        "affected_experiments": list(stages),
        "selected_value": copy.deepcopy(value),
    }


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _baseline() -> dict[str, Any]:
    assert CONFIG_PATH.is_file(), "The supplied config.yaml is required."
    result: Any = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(result, dict)
    return copy.deepcopy(result)


def _load(tmp_path: Path, payload: dict[str, Any], name: str = "fixture") -> ExperimentConfig:
    path: Path = tmp_path / f"{name}.yaml"
    _dump(path, payload)
    return ExperimentConfig.load(str(path))


def _resolved_payload(tmp_path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = _baseline()
    payload["experiment_id"] = "fictional-metrics-tests"
    payload["corpus_version"] = CORPUS_VERSION
    prerequisite: Path = tmp_path / "prerequisite.json"
    prerequisite.write_text('{"version":"fictional-v1"}\n', encoding="utf-8")
    payload["operational"] = {
        "database_path": str(tmp_path / "records.sqlite"),
        "artifact_root": str(tmp_path / "blobs"),
        "output_path": str(tmp_path / "reports"),
        "selected_tasks": ["affected_systems"],
        "prerequisite_manifests": {
            "responses": str(prerequisite),
            "references": str(prerequisite),
        },
    }
    settings: dict[str, Any] = {
        "evaluation.standard_deviation_ddof": 0,
        "evaluation.scoring.task_field_extraction_rules": {
            "affected_systems": {
                "prediction": "/answer/value",
                "reference": "/answer/value",
                "representation": "aligned",
                "unit_ids": "/candidates",
                "prediction_kind": "binary",
            },
            "threat_actor_linking": {
                "prediction": "/answer/value",
                "reference": "/answer/value",
                "representation": "aligned",
            },
            "exploit_likelihood": {
                "prediction": "/answer/value",
                "reference": "/answer/value",
                "representation": "aligned",
            },
            "threat_report_alignment": {
                "prediction": "/answer/value",
                "reference": "/answer/value",
                "representation": "text",
                "multiple_references": False,
            },
            "event_timeline_construction": {
                "prediction": "/answer/value",
                "reference": "/answer/value",
                "representation": "text",
                "multiple_references": False,
            },
            "rule_generation_yara": {
                "prediction": "/answer/value",
                "reference": "/answer/value",
                "representation": "text",
                "multiple_references": False,
            },
            "countermeasure_ranking": {
                "prediction": "/answer/value",
                "reference": "/answer/value",
                "representation": "ranking",
            },
        },
        "evaluation.scoring.invalid_output_policy": "error",
        "evaluation.scoring.abstention_policy": "error",
        "evaluation.scoring.f1.averaging": "binary",
        "evaluation.scoring.f1.positive_class_definitions": 1,
        "evaluation.scoring.f1.probability_thresholds": 0.5,
        "evaluation.scoring.accuracy.exact_match_and_partial_credit_rules": {
            "mode": "unit_exact"
        },
        "evaluation.scoring.bleu.aggregation_level": "corpus",
        "evaluation.scoring.bleu.tokenizer": "none",
        "evaluation.scoring.bleu.maximum_ngram_order": 2,
        "evaluation.scoring.bleu.smoothing": {
            "method": "none",
            "value": None,
            "effective_order": True,
        },
        "evaluation.scoring.bleu.lowercase": False,
        "evaluation.scoring.bleu.structured_output_serialization": "text",
        "evaluation.scoring.roc_auc.target_definition_manifest": str(prerequisite),
        "evaluation.scoring.roc_auc.missing_probability_policy": "error",
        "evaluation.scoring.ndcg.cutoff_k": 2,
        "evaluation.scoring.ndcg.gain_function": "linear",
        "evaluation.scoring.ndcg.relevance_manifest": str(prerequisite),
        "evaluation.scoring.ndcg.tie_policy": "ordered",
        "evaluation.tasks.prediction.exploit_likelihood.horizon_days": 30,
        "evaluation.tasks.prediction.exploit_likelihood.binary_target_construction": {
            "source": "fictional-observed-binary-outcome",
        },
        "taxonomy.guideline_manifest": str(prerequisite),
    }
    for path, value in settings.items():
        _resolve(payload, path, value)
    return payload


@dataclass
class MetricHarness:
    config: ExperimentConfig
    store: ArtifactStore
    evaluator: Evaluator
    serial: int = 0

    def add(
        self,
        target: Any,
        prediction: Any,
        task: str = "affected_systems",
        run: int = 0,
        model: str = "G5",
        instance_id: str | None = None,
        candidates: list[Any] | None = None,
        status: ResponseStatus = ResponseStatus.SUCCESS,
        semantic_status: str = "OK",
        omit_prediction: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> tuple[TaskInstance, ModelResponse]:
        """Persist one fictional task/response pair with an explicit reference."""
        self.serial += 1
        identity: str = instance_id or f"fixture-instance-{self.serial}"
        unit_values: list[Any] = (
            candidates
            if candidates is not None
            else [f"unit-{index}" for index in range(
                len(target) if isinstance(target, list) else 1
            )]
        )
        reference: dict[str, Any] = {
            "version": REFERENCE_VERSION,
            "answer": {"value": target},
            "evidence_records": [{
                "evidence_id": "fixture-reference-evidence",
                "role": "reference_only",
                "snippet": "Fictional reference evidence.",
            }],
        }
        if task == "exploit_likelihood":
            definition = self.config.evaluation.tasks["prediction"][task]
            reference.update(
                horizon_days=definition.horizon_days,
                target_construction=definition.binary_target_construction,
            )
        draft: TaskInstance = TaskInstance(
            instance_id=identity,
            case_id=f"case:{identity}",
            source_dataset="ctibench",
            source_item_id=identity,
            task_name=task,
            stage=TASK_STAGES[task],
            snapshot_date="2024-06-01",
            candidates=unit_values,
            reference=reference,
            output_profile={"type": "object"},
            transformation_log={
                "corpus_version": CORPUS_VERSION,
                "config_hash": self.config.config_hash,
                "task_guidance": "Use the fictional supplied evidence.",
            },
        )
        instance_payload: dict[str, Any] = draft.model_dump(mode="json")
        instance_payload["content_hash"] = _content_hash(_task_content_projection(draft))
        instance = TaskInstance.model_validate(instance_payload)
        self.store.create_record(
            "task_instance", identity, instance.model_dump(mode="json")
        )
        parsed: dict[str, Any] = {
            "status": semantic_status,
            "answer": {} if omit_prediction else {"value": prediction},
            "confidence": 0.5,
            "justification": "Fictional justification.",
            "metadata": {},
            "evidence_refs": [],
        }
        if extra:
            parsed.update(copy.deepcopy(extra))
        usage: dict[str, Any] = {}
        errors: list[str] = []
        if status != ResponseStatus.SUCCESS:
            parsed = {}
        if status == ResponseStatus.TRANSPORT_FAILURE:
            usage["execution_error"] = {"exception_class": "FictionalTransportError"}
        if status == ResponseStatus.INVALID_SCHEMA:
            errors = ["Fictional invalid output."]
        response = ModelResponse(
            response_id=f"fixture-response-{self.serial}",
            instance_id=identity,
            model_id=model,
            run_index=run,
            prompt_hash=_text_hash(f"fictional prompt:{identity}"),
            config_hash=self.config.config_hash,
            status=status,
            raw_text=_canonical_json(parsed) if parsed else "Fictional invalid/refused output.",
            parsed_answer=parsed,
            usage=usage,
            schema_errors=errors,
        )
        self.store.create_record(
            "model_response", response.response_id, response.model_dump(mode="json")
        )
        return instance, response

    def aggregate(self, responses: Sequence[ModelResponse]) -> dict[str, Any]:
        return self.evaluator.aggregate_task_metrics(
            [response.response_id for response in responses]
        )

    def decide(
        self,
        response: ModelResponse,
        failure: bool,
        version: str = "fictional-identification-v1",
    ) -> None:
        decision = FailureDecision(
            response_id=response.response_id,
            identification_version=version,
            decision="failure" if failure else "pass",
            decision_origin="hard_label",
        )
        self.store.create_record(
            "failure_decision", _content_hash(decision), decision.model_dump(mode="json")
        )


@pytest.fixture
def metric_payload(tmp_path: Path) -> dict[str, Any]:
    return _resolved_payload(tmp_path)


@pytest.fixture
def metric_harness(
    tmp_path: Path, metric_payload: dict[str, Any]
) -> MetricHarness:
    config = _load(tmp_path, metric_payload)
    store = ArtifactStore(
        str(tmp_path / "records.sqlite"), str(tmp_path / "blobs")
    )
    store.initialize()
    return MetricHarness(config, store, Evaluator(config, store))


def _harness(tmp_path: Path, payload: dict[str, Any]) -> MetricHarness:
    config = _load(tmp_path, payload, f"configuration-{len(list(tmp_path.glob('*.yaml')))}")
    store = ArtifactStore(
        str(tmp_path / f"records-{len(list(tmp_path.glob('*.sqlite')))}.sqlite"),
        str(tmp_path / "blobs"),
    )
    store.initialize()
    return MetricHarness(config, store, Evaluator(config, store))


def _run_metric(result: dict[str, Any], group: int = 0, run: int = 0) -> dict[str, Any]:
    return result["groups"][group]["runs"][run]


# ---------------------------------------------------------------------------
# Scientific configuration and binary/classification metrics.
# ---------------------------------------------------------------------------


def test_reported_settings_and_unresolved_policies_are_preserved(tmp_path: Path) -> None:
    config = _load(tmp_path, _baseline())
    assert config.evaluation.runs == 5
    assert config.evaluation.reporting_scale == (0.0, 100.0)
    assert config.quantile_width == 0.05
    assert config.failure_identification.stratification.manually_review_boundary_cases
    convergence = config.failure_identification.stratification.convergence
    assert convergence.require_stable_failure_distribution
    assert convergence.require_no_new_failure_set
    assert config.failure_identification.reported_manual_inspection_fraction_upper_bound == 0.015
    assert config.audits.failure_identification.zero_denominator_result == "N/A"
    assert config.evaluation.scoring.f1.averaging is None
    assert config.evaluation.scoring.f1.probability_thresholds is None
    assert config.evaluation.standard_deviation_ddof is None
    with pytest.raises(ValueError, match="unresolved"):
        config.validate("identify")


def test_candidate_f1_uses_aligned_units(metric_harness: MetricHarness) -> None:
    instance, response = metric_harness.add([1, 1, 1, 0], [1, 1, 0, 1])
    extracted = metric_harness.evaluator.score_response(instance, response)
    assert extracted["contribution"]["counts"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 0}
    assert len(extracted["contribution"]["unit_ids"]) == 4
    scored = _run_metric(metric_harness.aggregate([response]))
    assert scored["value"] == pytest.approx(4 / 6)
    assert scored["display_value"] == pytest.approx(100 * 4 / 6)


@pytest.mark.parametrize("prediction", [[1, 0, 1], [1], [1, 1, 1, 1]])
def test_extra_or_omitted_aligned_predictions_are_not_discarded(
    metric_harness: MetricHarness, prediction: list[int]
) -> None:
    instance, response = metric_harness.add([1, 0], prediction)
    with pytest.raises(ValueError, match="misaligned"):
        metric_harness.evaluator.score_response(instance, response)


def test_duplicate_unit_identifiers_cannot_inflate_credit(metric_harness: MetricHarness) -> None:
    instance, response = metric_harness.add([1, 1], [1, 1], candidates=["same", "same"])
    with pytest.raises(ValueError, match="duplicate"):
        metric_harness.evaluator.score_response(instance, response)


def test_wrong_asset_slot_is_not_a_true_positive(metric_harness: MetricHarness) -> None:
    instance, response = metric_harness.add([1, 0], [0, 1])
    extracted = metric_harness.evaluator.score_response(instance, response)
    assert extracted["contribution"]["counts"] == {"tp": 0, "fp": 1, "fn": 1, "tn": 0}


def test_pooled_f1_is_not_mean_response_f1(metric_harness: MetricHarness) -> None:
    _, first = metric_harness.add([1], [1])
    _, second = metric_harness.add([1] * 9, [0] * 9)
    result = _run_metric(metric_harness.aggregate([first, second]))
    assert result["value"] == pytest.approx(2 / 11)
    assert result["value"] != pytest.approx(0.5)


def test_macro_f1_is_over_configured_binary_classes(
    tmp_path: Path, metric_payload: dict[str, Any]
) -> None:
    _resolve(metric_payload, "evaluation.scoring.f1.averaging", "macro")
    harness = _harness(tmp_path, metric_payload)
    _, response = harness.add([1, 1, 0, 0], [1, 0, 0, 0])
    # Class 1 F1 = 2/3; class 0 F1 = 4/5.
    result = _run_metric(harness.aggregate([response]))
    assert result["value"] == pytest.approx(((2 / 3) + (4 / 5)) / 2)


def test_positive_class_is_configured(
    tmp_path: Path, metric_payload: dict[str, Any]
) -> None:
    _resolve(metric_payload, "evaluation.scoring.f1.positive_class_definitions", 0)
    harness = _harness(tmp_path, metric_payload)
    _, response = harness.add([0, 0, 1], [0, 1, 1])
    result = _run_metric(harness.aggregate([response]))
    assert result["counts"] == {"tp": 1, "fp": 0, "fn": 1, "tn": 1}
    assert result["value"] == pytest.approx(2 / 3)


def test_explicit_probability_threshold_includes_equality(
    tmp_path: Path, metric_payload: dict[str, Any]
) -> None:
    rules = metric_payload["evaluation"]["scoring"]["task_field_extraction_rules"]
    rules["affected_systems"]["prediction_kind"] = "probability"
    # The implemented extraction protocol explicitly uses >= threshold.
    _resolve(metric_payload, "evaluation.scoring.task_field_extraction_rules", rules)
    _resolve(metric_payload, "evaluation.scoring.f1.probability_thresholds", 0.6)
    harness = _harness(tmp_path, metric_payload)
    instance, response = harness.add([0, 1, 1], [0.59, 0.60, 0.61])
    extracted = harness.evaluator.score_response(instance, response)
    assert extracted["contribution"]["predictions"] == [0, 1, 1]


@pytest.mark.parametrize("probability", [-0.1, 1.1, "0.7", None])
def test_f1_probability_validation(
    tmp_path: Path, metric_payload: dict[str, Any], probability: Any
) -> None:
    rules = metric_payload["evaluation"]["scoring"]["task_field_extraction_rules"]
    rules["affected_systems"]["prediction_kind"] = "probability"
    _resolve(metric_payload, "evaluation.scoring.task_field_extraction_rules", rules)
    harness = _harness(tmp_path, metric_payload)
    instance, response = harness.add([1], [probability])
    with pytest.raises(ValueError):
        harness.evaluator.score_response(instance, response)


@pytest.mark.parametrize("missing", ["averaging", "probability_thresholds"])
def test_missing_f1_policy_never_uses_library_default(
    tmp_path: Path, metric_payload: dict[str, Any], missing: str
) -> None:
    rules = metric_payload["evaluation"]["scoring"]["task_field_extraction_rules"]
    rules["affected_systems"]["prediction_kind"] = "probability"
    _resolve(metric_payload, "evaluation.scoring.task_field_extraction_rules", rules)
    _resolve(metric_payload, f"evaluation.scoring.f1.{missing}", None)
    harness = _harness(tmp_path, metric_payload)
    instance, response = harness.add([1], [0.8])
    with pytest.raises(ValueError, match="unresolved"):
        if missing == "averaging":
            harness.aggregate([response])
        else:
            harness.evaluator.score_response(instance, response)


def test_five_runs_are_scored_then_summarized(metric_harness: MetricHarness) -> None:
    responses: list[ModelResponse] = []
    expected: list[float] = []
    for run in range(metric_harness.config.evaluation_runs):
        predictions: list[int] = [1] * (run + 1) + [0] * (4 - run)
        _, response = metric_harness.add(
            [1] * 5, predictions, run=run, instance_id="shared-five-run-instance"
        )
        responses.append(response)
        expected.append(100 * 2 * (run + 1) / (2 * (run + 1) + 4 - run))
    group = metric_harness.aggregate(responses)["groups"][0]
    assert len(group["runs"]) == 5
    assert group["summary"]["status"] == "complete"
    assert group["summary"]["mean"] == pytest.approx(np.mean(expected))
    assert group["summary"]["standard_deviation"] == pytest.approx(np.std(expected, ddof=0))
    pooled = 100 * 2 * sum(range(1, 6)) / (2 * sum(range(1, 6)) + sum(range(5)))
    assert group["summary"]["mean"] != pytest.approx(pooled)


def test_missing_run_and_undefined_run_do_not_produce_complete_summary(
    metric_harness: MetricHarness,
) -> None:
    _, response = metric_harness.add([0], [0])
    group = metric_harness.aggregate([response])["groups"][0]
    assert group["summary"]["status"] == "incomplete"
    assert group["summary"]["mean"] is None
    assert group["runs"][0]["value"] is None
    assert group["runs"][0]["undefined_reason"]


def test_null_ddof_blocks_complete_summary(
    tmp_path: Path, metric_payload: dict[str, Any]
) -> None:
    _resolve(metric_payload, "evaluation.standard_deviation_ddof", None)
    harness = _harness(tmp_path, metric_payload)
    responses = [
        harness.add([1], [1], run=run, instance_id="ddof-instance")[1]
        for run in range(harness.config.evaluation.runs)
    ]
    with pytest.raises(ValueError, match="ddof.*unresolved"):
        harness.aggregate(responses)


def test_duplicate_response_population_is_rejected(metric_harness: MetricHarness) -> None:
    _, response = metric_harness.add([1], [1])
    with pytest.raises(ValueError, match="duplicate"):
        metric_harness.aggregate([response, response])


def test_accuracy_reports_correct_unit_denominator(metric_harness: MetricHarness) -> None:
    _, response = metric_harness.add(
        ["actor-A", "actor-B"], ["actor-A", "actor-C"], task="threat_actor_linking"
    )
    result = _run_metric(metric_harness.aggregate([response]))
    assert result["numerator"] == 1
    assert result["denominator"] == 2
    assert result["value"] == 0.5
    assert result["display_value"] == 50.0


# ---------------------------------------------------------------------------
# ROC-AUC and explicit forecasting references.
# ---------------------------------------------------------------------------


def test_auc_pools_continuous_scores_across_responses(metric_harness: MetricHarness) -> None:
    responses = [
        metric_harness.add(target, prediction, task="exploit_likelihood")[1]
        for target, prediction in [(0, 0.1), (0, 0.4), (1, 0.35), (1, 0.8)]
    ]
    for response in responses:
        single = _run_metric(metric_harness.aggregate([response]))
        assert single["value"] is None
    combined = _run_metric(metric_harness.aggregate(responses))
    assert combined["value"] == pytest.approx(0.75)
    assert combined["display_value"] == pytest.approx(75.0)


def test_continuous_epss_is_not_a_binary_target(metric_harness: MetricHarness) -> None:
    instance, response = metric_harness.add(0.42, 0.7, task="exploit_likelihood")
    with pytest.raises(ValueError, match="binary label"):
        metric_harness.evaluator.score_response(instance, response)


@pytest.mark.parametrize(
    "path",
    [
        "evaluation.tasks.prediction.exploit_likelihood.horizon_days",
        "evaluation.tasks.prediction.exploit_likelihood.binary_target_construction",
        "evaluation.scoring.roc_auc.target_definition_manifest",
    ],
)
def test_forecasting_requires_resolved_target_protocol(
    tmp_path: Path, metric_payload: dict[str, Any], path: str
) -> None:
    _resolve(metric_payload, path, None)
    harness = _harness(tmp_path, metric_payload)
    instance, response = harness.add(1, 0.8, task="exploit_likelihood")
    with pytest.raises(ValueError, match="unresolved"):
        harness.evaluator.score_response(instance, response)
    unrelated, unrelated_response = harness.add([1], [1])
    assert harness.evaluator.score_response(unrelated, unrelated_response)["scoring_units"] == 1


@pytest.mark.parametrize("prediction", [-0.01, 1.01, "0.8"])
def test_auc_rejects_invalid_probabilities(
    metric_harness: MetricHarness, prediction: Any
) -> None:
    instance, response = metric_harness.add(1, prediction, task="exploit_likelihood")
    with pytest.raises(ValueError):
        metric_harness.evaluator.score_response(instance, response)


def test_missing_auc_probability_is_not_replaced_by_confidence(
    metric_harness: MetricHarness,
) -> None:
    instance, response = metric_harness.add(
        1, None, task="exploit_likelihood", omit_prediction=True
    )
    with pytest.raises(ValueError, match="missing_prediction"):
        metric_harness.evaluator.score_response(instance, response)


def test_empty_population_is_not_a_zero_metric(metric_harness: MetricHarness) -> None:
    with pytest.raises(ValueError, match="empty"):
        metric_harness.evaluator.aggregate_task_metrics([])


# ---------------------------------------------------------------------------
# BLEU field extraction, serialization, and aggregation.
# ---------------------------------------------------------------------------


def test_bleu_ignores_json_envelope(metric_harness: MetricHarness) -> None:
    reference = "alpha beta gamma delta"
    first_instance, first = metric_harness.add(
        reference, reference, task="threat_report_alignment",
        extra={"confidence": 0.1, "justification": "unrelated", "evidence_refs": ["B", "A"]},
    )
    second_instance, second = metric_harness.add(
        reference, reference, task="threat_report_alignment",
        extra={"confidence": 0.9, "justification": reference, "metadata": {"different": True}},
    )
    first_score = metric_harness.evaluator.score_response(first_instance, first)
    second_score = metric_harness.evaluator.score_response(second_instance, second)
    assert first_score["contribution"] == second_score["contribution"]
    assert _run_metric(metric_harness.aggregate([first, second]))["display_value"] == pytest.approx(100.0)


def test_reference_in_justification_does_not_improve_wrong_answer(
    metric_harness: MetricHarness,
) -> None:
    reference = "alpha beta gamma delta"
    _, response = metric_harness.add(
        reference, "unrelated tokens entirely different",
        task="threat_report_alignment", extra={"justification": reference},
    )
    result = _run_metric(metric_harness.aggregate([response]))
    assert result["value"] == 0.0


def test_wrong_bleu_pointer_does_not_fall_back_to_envelope(
    tmp_path: Path, metric_payload: dict[str, Any]
) -> None:
    rules = metric_payload["evaluation"]["scoring"]["task_field_extraction_rules"]
    rules["threat_report_alignment"]["prediction"] = "/answer/missing"
    _resolve(metric_payload, "evaluation.scoring.task_field_extraction_rules", rules)
    harness = _harness(tmp_path, metric_payload)
    instance, response = harness.add("alpha beta", "alpha beta", task="threat_report_alignment")
    with pytest.raises(ValueError, match="missing_prediction"):
        harness.evaluator.score_response(instance, response)


def test_structured_timeline_serialization_preserves_event_order(
    tmp_path: Path, metric_payload: dict[str, Any]
) -> None:
    _resolve(
        metric_payload,
        "evaluation.scoring.bleu.structured_output_serialization",
        "canonical_json",
    )
    harness = _harness(tmp_path, metric_payload)
    timeline = [{"t": 1, "action": "phish"}, {"t": 2, "action": "beacon"}]
    instance, response = harness.add(
        timeline, list(reversed(timeline)), task="event_timeline_construction"
    )
    extracted = harness.evaluator.score_response(instance, response)["contribution"]
    assert extracted["hypothesis"] == _canonical_json(list(reversed(timeline)))
    assert extracted["references"] == [_canonical_json(timeline)]
    assert extracted["hypothesis"] != extracted["references"][0]
    assert harness.evaluator.score_response(instance, response)["contribution"] == extracted


def test_yara_scores_rule_body_not_rule_name(metric_harness: MetricHarness) -> None:
    body = 'rule fixture { strings: $a = "fictional" condition: $a }'
    instance, response = metric_harness.add(body, body, task="rule_generation_yara")
    extracted = metric_harness.evaluator.score_response(instance, response)
    assert extracted["contribution"]["hypothesis"] == body
    assert _run_metric(metric_harness.aggregate([response]))["value"] == pytest.approx(100.0)


def test_corpus_bleu_is_not_sentence_mean(
    tmp_path: Path, metric_payload: dict[str, Any]
) -> None:
    corpus_harness = _harness(tmp_path, metric_payload)
    pairs = [
        ("alpha beta gamma delta epsilon zeta eta theta", "alpha beta gamma delta epsilon zeta eta theta"),
        ("one two", "unrelated words"),
    ]
    corpus_responses = [
        corpus_harness.add(target, prediction, task="threat_report_alignment")[1]
        for target, prediction in pairs
    ]
    corpus_value = _run_metric(corpus_harness.aggregate(corpus_responses))["value"]
    _resolve(metric_payload, "evaluation.scoring.bleu.aggregation_level", "sentence_mean")
    sentence_harness = _harness(tmp_path, metric_payload)
    sentence_responses = [
        sentence_harness.add(target, prediction, task="threat_report_alignment")[1]
        for target, prediction in pairs
    ]
    sentence_value = _run_metric(sentence_harness.aggregate(sentence_responses))["value"]
    assert sentence_value == pytest.approx(50.0)
    assert corpus_value != pytest.approx(sentence_value)


@pytest.mark.parametrize(
    "field_name",
    ["tokenizer", "maximum_ngram_order", "smoothing", "lowercase", "structured_output_serialization"],
)
def test_bleu_requires_explicit_policy(
    tmp_path: Path, metric_payload: dict[str, Any], field_name: str
) -> None:
    _resolve(metric_payload, f"evaluation.scoring.bleu.{field_name}", None)
    harness = _harness(tmp_path, metric_payload)
    _, response = harness.add("alpha beta", "alpha beta", task="threat_report_alignment")
    with pytest.raises(ValueError):
        harness.aggregate([response])


# ---------------------------------------------------------------------------
# NDCG and status-specific scoring policies.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gain", ["linear", "exponential"])
def test_ndcg_uses_declared_gain_and_cutoff(
    tmp_path: Path, metric_payload: dict[str, Any], gain: str
) -> None:
    _resolve(metric_payload, "evaluation.scoring.ndcg.gain_function", gain)
    harness = _harness(tmp_path, metric_payload)
    relevance = {"A": 3, "B": 2, "C": 1, "D": 0}
    responses = [
        harness.add(relevance, ranking, task="countermeasure_ranking")[1]
        for ranking in [
            ["A", "B", "C", "D"],
            ["B", "A", "C", "D"],
            ["A", "B", "D", "C"],
        ]
    ]
    values = [
        _run_metric(harness.aggregate([response]))["value"] for response in responses
    ]
    transform = (lambda value: value) if gain == "linear" else (lambda value: 2**value - 1)
    ideal = transform(3) + transform(2) / math.log2(3)
    swapped = transform(2) + transform(3) / math.log2(3)
    assert values[0] == 1.0
    assert values[1] == pytest.approx(swapped / ideal)
    assert values[1] < values[0]
    assert values[2] == values[0]


@pytest.mark.parametrize("ranking", [["A", "A", "C"], ["A", "B"], ["A", "B", "X"]])
def test_ndcg_candidate_alignment_is_strict(
    metric_harness: MetricHarness, ranking: list[str]
) -> None:
    instance, response = metric_harness.add(
        {"A": 3, "B": 2, "C": 0}, ranking, task="countermeasure_ranking"
    )
    with pytest.raises(ValueError):
        metric_harness.evaluator.score_response(instance, response)


def test_zero_ideal_ndcg_is_undefined(metric_harness: MetricHarness) -> None:
    _, response = metric_harness.add(
        {"A": 0, "B": 0}, ["A", "B"], task="countermeasure_ranking"
    )
    result = _run_metric(metric_harness.aggregate([response]))
    assert result["value"] is None
    assert "zero ideal" in result["undefined_reason"]


@pytest.mark.parametrize("setting", ["cutoff_k", "gain_function", "tie_policy"])
def test_ndcg_null_controls_do_not_select_defaults(
    tmp_path: Path, metric_payload: dict[str, Any], setting: str
) -> None:
    _resolve(metric_payload, f"evaluation.scoring.ndcg.{setting}", None)
    harness = _harness(tmp_path, metric_payload)
    _, response = harness.add({"A": 1, "B": 0}, ["A", "B"], task="countermeasure_ranking")
    with pytest.raises(ValueError, match="unresolved"):
        harness.aggregate([response])


@pytest.mark.parametrize(
    ("status", "semantic", "category"),
    [
        (ResponseStatus.SUCCESS, "NEED_MORE_CONTEXT", "NEED_MORE_CONTEXT"),
        (ResponseStatus.SUCCESS, "UNSUPPORTED", "UNSUPPORTED"),
        (ResponseStatus.INVALID_SCHEMA, "OK", "invalid_schema"),
        (ResponseStatus.REFUSAL, "OK", "refusal"),
        (ResponseStatus.TRANSPORT_FAILURE, "OK", "transport_failure"),
    ],
)
def test_statuses_remain_distinct_under_exclusion_policy(
    tmp_path: Path,
    metric_payload: dict[str, Any],
    status: ResponseStatus,
    semantic: str,
    category: str,
) -> None:
    _resolve(metric_payload, "evaluation.scoring.invalid_output_policy", "exclude")
    _resolve(metric_payload, "evaluation.scoring.abstention_policy", "exclude")
    harness = _harness(tmp_path, metric_payload)
    instance, response = harness.add([1], [1], status=status, semantic_status=semantic)
    extracted = harness.evaluator.score_response(instance, response)
    assert extracted["availability"] == category
    assert extracted["excluded"] is True
    assert extracted["failure_decision"] is None
    assert response.status == status


def test_unresolved_decisions_do_not_yield_complete_failure_rate(
    metric_harness: MetricHarness,
) -> None:
    _, first = metric_harness.add([1], [1])
    _, second = metric_harness.add([1], [0])
    metric_harness.decide(first, False)
    result = metric_harness.aggregate([first, second])
    assert result["failure_rate"]["status"] == "incomplete"


# ---------------------------------------------------------------------------
# Scripted independent annotations for RQ1, without model execution.
# ---------------------------------------------------------------------------


@dataclass
class ScriptedAnnotations:
    """Minimal deterministic AnnotationService substitute with persisted lineage."""

    store: ArtifactStore
    references: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)

    def create_tasks(
        self, response_ids: list[str], purpose: str, protocol: dict[str, Any]
    ) -> list[AnnotationTask]:
        tasks: list[AnnotationTask] = []
        for response_id in sorted(response_ids):
            identity = _content_hash([purpose, response_id, protocol])
            task = AnnotationTask(
                annotation_task_id=identity,
                response_id=response_id,
                purpose=purpose,
                assigned_annotator="fixture-human",
                guideline_hash=protocol["guideline_hash"],
                visible_payload={"fixture": "blinded human evidence packet"},
            )
            self.store.create_record(
                "annotation_task", identity, task.model_dump(mode="json")
            )
            context_id = _content_hash({"annotation_task_context": identity})
            self.store.create_record(
                "experiment_manifest", context_id,
                {
                    "manifest_id": context_id,
                    "purpose": "annotation_task_context",
                    "annotation_task_id": identity,
                    "protocol_hash": _content_hash(protocol),
                },
            )
            tasks.append(task)
        return tasks

    def create_adjudications(self, task_ids: list[str]) -> list[AnnotationTask]:
        return []

    def get_reference(self, response_id: str, purpose: str) -> dict[str, Any]:
        key = (response_id, purpose)
        if key not in self.references:
            raise LookupError("Scripted human reference is not yet complete.")
        return copy.deepcopy(self.references[key])

    def submit_annotation(
        self, task_id: str, annotator_id: str, payload: dict[str, Any]
    ) -> None:
        task = AnnotationTask.model_validate(
            self.store.read_record("annotation_task", task_id)
        )
        assert task.assigned_annotator == annotator_id
        protocol = payload["protocol"]
        identity = _content_hash(["scripted-reference", task_id, payload["decision"]])
        record: dict[str, Any] = {
            "annotation_id": identity,
            "annotation_task_id": task_id,
            "response_id": task.response_id,
            "purpose": task.purpose.value,
            "annotator_id": annotator_id,
            "guideline_hash": task.guideline_hash,
            "protocol_hash": _content_hash(protocol),
            "cohort_id": None,
            "audit_only": False,
            "uncertain": False,
            "decision": payload["decision"],
            "status": "submitted",
        }
        self.store.create_record("annotation", identity, record)
        self.references[(task.response_id, task.purpose.value)] = {
            **record,
            "annotation_ids": [identity],
        }


@dataclass
class IdentificationHarness:
    metrics: MetricHarness
    identifier: FailureIdentifier
    annotations: ScriptedAnnotations
    protocol: dict[str, Any]

    def similarity(self, score: float, model: str = "G5") -> ModelResponse:
        instance, response = self.metrics.add(
            "fictional reference", "fictional answer",
            task="threat_report_alignment", model=model,
        )
        record: dict[str, Any] = {
            "score_id": f"similarity:{response.response_id}",
            "response_id": response.response_id,
            "instance_id": instance.instance_id,
            "task": instance.task_name,
            "config_hash": self.metrics.config.config_hash,
            "reference_hash": _content_hash(instance.reference),
            "similarity_score": score,
            "similarity_version": SIMILARITY_VERSION,
        }
        self.metrics.store.create_record("score", record["score_id"], record)
        return response

    def label(self, task: AnnotationTask, decision: str) -> None:
        self.annotations.submit_annotation(
            task.annotation_task_id,
            "fixture-human",
            {"decision": decision, "protocol": self.protocol},
        )


def _identification_harness(
    tmp_path: Path,
    payload: dict[str, Any],
    tie_mode: str = "keep_together",
    unvisited: str = "continue",
) -> IdentificationHarness:
    payload = copy.deepcopy(payload)
    payload["operational"]["selected_tasks"] = ["threat_report_alignment"]
    protocol: dict[str, Any] = {
        "config_hash": "$config_hash",
        "guideline_hash": _content_hash({"fixture": "correctness-guideline"}),
        "judgment": "identification",
        "cohort_id": None,
    }
    settings: dict[str, Any] = {
        "failure_identification.hard_label_rules": {
            "tasks": {
                "affected_systems": {"route": "hard"},
                "threat_report_alignment": {"route": "similarity"},
            },
            "annotation_protocols": {"anchor": protocol, "boundary": protocol},
        },
        "evaluation.scoring.response_level_failure_criteria": {
            "affected_systems": {
                "pass_if": {
                    "op": "equal",
                    "left": {"source": "prediction", "pointer": "/answer/value"},
                    "right": {"source": "reference", "pointer": "/answer/value"},
                },
                "on_missing": "review",
                "availability": {
                    "transport_failure": "error",
                    "refusal": "review",
                    "invalid_schema": "review",
                    "NEED_MORE_CONTEXT": "review",
                    "UNSUPPORTED": "failure",
                },
            }
        },
        "failure_identification.stratification.similarity_measure_by_task": {
            "threat_report_alignment": {
                "source": "stored_score",
                "pointer": "/similarity_score",
                "score_version": SIMILARITY_VERSION,
                "version_pointer": "/similarity_version",
                "direction": "higher_is_better",
                "scale": [0, 100],
            }
        },
        "failure_identification.stratification.grouping_fields": ["task_name", "model_id"],
        "failure_identification.stratification.anchor_count_per_stratum": 2,
        "failure_identification.stratification.anchor_sampling_seed": 17,
        "failure_identification.stratification.tied_score_policy": {
            "mode": tie_mode, "partition": "rank_floor",
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
        "failure_identification.stratification.convergence.distribution_distance": {
            "metric": "total_variation",
            "distribution_fields": ["task_name"],
            "normalization": "failure_mass",
            "empty_distribution": "zero",
            "minimum_comparisons": 1,
            "no_new_failure_set": "zero_new_failures",
        },
        "failure_identification.stratification.convergence.stability_tolerance": 0.1,
        "failure_identification.stratification.convergence.unvisited_strata_policy": unvisited,
    }
    for path, value in settings.items():
        _resolve(payload, path, value)
    config = _load(tmp_path, payload, f"id-config-{tie_mode}-{unvisited}")
    config.validate("identify")
    store = ArtifactStore(
        str(tmp_path / f"id-{tie_mode}-{unvisited}.sqlite"), str(tmp_path / "id-blobs")
    )
    store.initialize()
    metrics = MetricHarness(config, store, Evaluator(config, store))
    annotations = ScriptedAnnotations(store)
    identifier = FailureIdentifier(config, store, annotations)  # type: ignore[arg-type]
    return IdentificationHarness(
        metrics, identifier, annotations, identifier._annotation_protocol("anchor")
    )


@pytest.fixture
def identification_harness(
    tmp_path: Path, metric_payload: dict[str, Any]
) -> IdentificationHarness:
    return _identification_harness(tmp_path, metric_payload)


# ---------------------------------------------------------------------------
# Response failures, strata, anchors, review, convergence, and resumption.
# ---------------------------------------------------------------------------


def test_response_failure_rate_is_not_complement_of_f1(
    identification_harness: IdentificationHarness,
) -> None:
    harness = identification_harness
    first_instance, first = harness.metrics.add([1] * 4, [1] * 4)
    second_instance, second = harness.metrics.add([1] * 4, [1, 1, 1, 0])
    for instance, response in [(first_instance, first), (second_instance, second)]:
        harness.metrics.evaluator.score_response(instance, response)
    decisions = harness.identifier.assign_hard_label_decisions(
        [first.response_id, second.response_id]
    )
    assert [item.decision for item in decisions].count(FailureStatus.FAILURE) == 1
    result = harness.metrics.aggregate([first, second])
    assert result["failure_rate"]["status"] == "complete"
    assert result["failure_rate"]["value"] == 50.0
    f1 = _run_metric(result)["display_value"]
    assert f1 == pytest.approx(100 * 14 / 15)
    assert result["failure_rate"]["value"] != pytest.approx(100 - f1)


@pytest.mark.parametrize("tie_mode", ["keep_together", "split_by_response_id"])
def test_quantile_membership_is_complete_and_order_independent(
    tmp_path: Path, metric_payload: dict[str, Any], tie_mode: str
) -> None:
    harness = _identification_harness(tmp_path, metric_payload, tie_mode)
    responses = [harness.similarity(float(index // 3)) for index in range(60)]
    ids = [response.response_id for response in responses]
    first = harness.identifier.build_strata(ids)
    second = harness.identifier.build_strata(list(reversed(ids)))
    assert first == second
    members = [item["response_id"] for stratum in first for item in stratum["members"]]
    assert len(members) == len(ids)
    assert set(members) == set(ids)
    assert len(members) == len(set(members))
    assert all(stratum["quantile_width"] == harness.metrics.config.quantile_width for stratum in first)
    if tie_mode == "keep_together":
        locations: dict[float, set[str]] = {}
        for stratum in first:
            for item in stratum["members"]:
                locations.setdefault(item["score"], set()).add(stratum["stratum_id"])
        assert all(len(value) == 1 for value in locations.values())


def test_identical_scores_and_group_isolation(
    identification_harness: IdentificationHarness,
) -> None:
    harness = identification_harness
    responses = [
        harness.similarity(50.0, model=model)
        for model in ("G5", "CLD") for _ in range(5)
    ]
    strata = harness.identifier.build_strata([item.response_id for item in responses])
    assert len(strata) == 2
    assert {item["group"]["model_id"] for item in strata} == {"G5", "CLD"}
    assert all(item["effective_size"] == 5 for item in strata)


def test_unresolved_tie_policy_blocks_stratification(
    tmp_path: Path, metric_payload: dict[str, Any]
) -> None:
    harness = _identification_harness(tmp_path, metric_payload)
    payload = harness.metrics.config.model_dump(mode="python")
    payload["failure_identification"]["stratification"]["tied_score_policy"] = None
    config = ExperimentConfig.model_validate(payload)
    identifier = FailureIdentifier(config, harness.metrics.store, harness.annotations)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tied_score_policy"):
        identifier.build_strata([])


def test_anchor_requests_are_idempotent_and_incomplete_labels_block_ranges(
    identification_harness: IdentificationHarness,
) -> None:
    harness = identification_harness
    responses = [harness.similarity(50.0) for _ in range(6)]
    stratum = harness.identifier.build_strata([item.response_id for item in responses])[0]
    tasks = harness.identifier.request_anchors(stratum["stratum_id"])
    repeated = harness.identifier.request_anchors(stratum["stratum_id"])
    assert tasks == repeated
    assert len(tasks) == 2
    assert len(harness.metrics.store.list_records("annotation_task")) == 2
    with pytest.raises(RuntimeError, match="human"):
        harness.identifier.fit_ranges(stratum["stratum_id"])
    harness.label(tasks[0], "failure")
    with pytest.raises(RuntimeError):
        harness.identifier.fit_ranges(stratum["stratum_id"])


@pytest.mark.parametrize("decisions", [("pass", "pass"), ("failure", "failure")])
def test_one_class_anchors_do_not_invent_opposite_class_range(
    identification_harness: IdentificationHarness, decisions: tuple[str, str]
) -> None:
    harness = identification_harness
    responses = [harness.similarity(50.0) for _ in range(6)]
    stratum = harness.identifier.build_strata([item.response_id for item in responses])[0]
    tasks = harness.identifier.request_anchors(stratum["stratum_id"])
    for task, decision in zip(tasks, decisions, strict=True):
        harness.label(task, decision)
    fitted = harness.identifier.fit_ranges(stratum["stratum_id"])
    present = decisions[0]
    absent = "failure" if present == "pass" else "pass"
    assert fitted["ranges"][present]["lower"] == 50.0
    assert fitted["ranges"][present]["upper"] == 50.0
    assert fitted["ranges"][absent] is None
    assert fitted["boundary_labels_used_for_fitting"] is False


def test_contradictory_equal_score_anchors_route_overlap_to_review(
    identification_harness: IdentificationHarness,
) -> None:
    harness = identification_harness
    responses = [harness.similarity(50.0) for _ in range(6)]
    stratum = harness.identifier.build_strata([item.response_id for item in responses])[0]
    tasks = harness.identifier.request_anchors(stratum["stratum_id"])
    harness.label(tasks[0], "failure")
    harness.label(tasks[1], "pass")
    fitted = harness.identifier.fit_ranges(stratum["stratum_id"])
    assert fitted["contradictory_equal_score_anchors"] is True
    decisions = harness.identifier.classify_stratum(stratum["stratum_id"])
    anchor_ids = {task.response_id for task in tasks}
    anchors = [item for item in decisions if item.response_id in anchor_ids]
    others = [item for item in decisions if item.response_id not in anchor_ids]
    assert {item.decision for item in anchors} == {FailureStatus.PASS, FailureStatus.FAILURE}
    assert all(item.decision_origin == DecisionOrigin.HUMAN_ANCHOR for item in anchors)
    assert all(item.requires_review for item in others)
    assert all(item.decision == FailureStatus.UNRESOLVED for item in others)
    review = harness.identifier.request_boundary_reviews(decisions)
    assert len(review) == 4
    assert review == harness.identifier.request_boundary_reviews(decisions)


def test_separated_anchor_extrema_preserve_gap_for_review(
    tmp_path: Path, metric_payload: dict[str, Any]
) -> None:
    harness = _identification_harness(tmp_path, metric_payload, "split_by_response_id")
    # With 80 records, a 5% rank bin contains four distinct scores.
    responses = [harness.similarity(float(index)) for index in range(80)]
    strata = harness.identifier.build_strata([item.response_id for item in responses])
    stratum = strata[0]
    tasks = harness.identifier.request_anchors(stratum["stratum_id"])
    scores = {item["response_id"]: item["score"] for item in stratum["members"]}
    ordered = sorted(tasks, key=lambda task: scores[task.response_id])
    harness.label(ordered[0], "failure")
    harness.label(ordered[1], "pass")
    fitted = harness.identifier.fit_ranges(stratum["stratum_id"])
    assert fitted["ranges"]["failure"]["lower"] == scores[ordered[0].response_id]
    assert fitted["ranges"]["failure"]["upper"] == scores[ordered[0].response_id]
    decisions = harness.identifier.classify_stratum(stratum["stratum_id"])
    nonanchors = [item for item in decisions if item.response_id not in {task.response_id for task in tasks}]
    assert all(item.requires_review for item in nonanchors)


@pytest.mark.parametrize(
    ("stable", "no_new", "expected"),
    [(False, False, False), (False, True, False), (True, False, False), (True, True, True)],
)
def test_convergence_requires_both_conditions(
    identification_harness: IdentificationHarness,
    stable: bool,
    no_new: bool,
    expected: bool,
) -> None:
    identifier = identification_harness.identifier
    history = [
        {
            "complete": True,
            "failure_counts": {"A": 10},
            "processed_count": 20,
            "new_failure_ids": ["old"],
            "pending_response_ids": [],
        },
        {
            "complete": True,
            "failure_counts": {"A": 10} if stable else {"B": 10},
            "processed_count": 30,
            "new_failure_ids": [] if no_new else ["new"],
            "pending_response_ids": [],
        },
    ]
    assert identifier.check_convergence(history) is expected
    assert identifier.check_convergence(history[:1]) is False
    history[1]["pending_response_ids"] = ["pending-review"]
    assert identifier.check_convergence(history) is False


def test_convergence_tolerance_endpoint_is_inclusive(
    identification_harness: IdentificationHarness,
) -> None:
    history = [
        {"complete": True, "failure_counts": {"A": 10}, "processed_count": 10,
         "new_failure_ids": [], "pending_response_ids": []},
        {"complete": True, "failure_counts": {"A": 9, "B": 1}, "processed_count": 10,
         "new_failure_ids": [], "pending_response_ids": []},
    ]
    assert identification_harness.identifier.check_convergence(history) is True


def test_wait_resume_and_workload_above_reported_bound(
    identification_harness: IdentificationHarness,
) -> None:
    harness = identification_harness
    responses = [harness.similarity(50.0) for _ in range(6)]
    ids = [item.response_id for item in responses]
    waiting = harness.identifier.run(ids)
    assert waiting["status"] == "waiting_for_annotations"
    assert waiting["workload"]["review_fraction"] > 0.015
    assert waiting["workload"]["reported_bound_is_enforced_as_budget"] is False
    assert waiting["workload"]["independent_audit_work_included"] is False
    task_ids = waiting["pending_annotation_task_ids"]
    assert task_ids
    for index, task_id in enumerate(task_ids):
        task = AnnotationTask.model_validate(
            harness.metrics.store.read_record("annotation_task", task_id)
        )
        harness.label(task, "failure" if index == 0 else "pass")
    boundary_waiting = harness.identifier.run(ids)
    assert boundary_waiting["status"] == "waiting_for_annotations"
    assert len(boundary_waiting["pending_annotation_task_ids"]) == 4
    for task_id in boundary_waiting["pending_annotation_task_ids"]:
        task = AnnotationTask.model_validate(
            harness.metrics.store.read_record("annotation_task", task_id)
        )
        harness.label(task, "pass")
    completed = harness.identifier.run(ids)
    assert completed["status"] == "completed"
    assert completed["coverage"]["complete"] is True
    assert completed["coverage"]["finalized"] == 6
    assert completed["workload"]["unique_reviewed_responses"] == 6
    assert completed["workload"]["review_fraction"] == 1.0
    before = len(harness.metrics.store.list_records("failure_decision"))
    assert harness.identifier.run(ids) == completed
    assert len(harness.metrics.store.list_records("failure_decision")) == before
    assert len(completed["decision_ids_by_response"]) == 6
    rate = harness.metrics.evaluator._failure_rate(ids)
    assert rate["status"] == "complete"
    assert rate["failure_count"] == 1
    assert rate["value"] == pytest.approx(100 / 6)


def test_early_stop_retains_unvisited_responses_as_unresolved(
    tmp_path: Path, metric_payload: dict[str, Any]
) -> None:
    harness = _identification_harness(
        tmp_path, metric_payload, "split_by_response_id", "retain_unresolved"
    )
    responses = [harness.similarity(float(index)) for index in range(20)]
    ids = [item.response_id for item in responses]
    first = harness.identifier.run(ids)
    for task_id in first["pending_annotation_task_ids"]:
        task = AnnotationTask.model_validate(
            harness.metrics.store.read_record("annotation_task", task_id)
        )
        harness.label(task, "pass")
    second = harness.identifier.run(ids)
    for task_id in second["pending_annotation_task_ids"]:
        task = AnnotationTask.model_validate(
            harness.metrics.store.read_record("annotation_task", task_id)
        )
        harness.label(task, "pass")
    final = harness.identifier.run(ids)
    assert final["status"] == "completed"
    assert final["converged"] is True
    assert final["coverage"]["complete"] is False
    assert final["coverage"]["finalized"] == 2
    assert len(final["unprocessed_response_ids"]) == 18
    assert set(final["unprocessed_response_ids"]).isdisjoint(
        final["decision_ids_by_response"]
    )
    for identity in final["unresolved_decision_ids"]:
        record = harness.metrics.store.read_record("failure_decision", identity)
        assert record["decision"] == "unresolved"
        assert record["decision_origin"] == "not_processed"


# ---------------------------------------------------------------------------
# Independent identification-audit arithmetic; no audit tuning or sampling.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("automatic_failure", "human_failure", "precision", "recall", "f1"),
    [
        (False, False, None, None, None),
        (True, False, 0.0, None, 0.0),
        (False, True, None, 0.0, 0.0),
        (True, True, 1.0, 1.0, 1.0),
    ],
)
def test_failure_audit_zero_denominators_are_independent(
    metric_harness: MetricHarness,
    automatic_failure: bool,
    human_failure: bool,
    precision: float | None,
    recall: float | None,
    f1: float | None,
) -> None:
    harness = metric_harness
    size = harness.config.audits.failure_identification.instances
    assert size == 100
    response_ids: list[str] = []
    bindings: dict[str, str] = {}
    guideline_hash = _content_hash({"fixture": "blind-audit-guideline"})
    cohort_id = "fixture-identification-audit"
    version = "fixture-frozen-identification"
    annotations: list[dict[str, Any]] = []
    for index in range(size):
        _, response = harness.add([1], [1])
        response_ids.append(response.response_id)
        decision = FailureDecision(
            response_id=response.response_id,
            identification_version=version,
            decision="failure" if automatic_failure else "pass",
            decision_origin="hard_label",
        )
        identity = _content_hash(decision)
        bindings[response.response_id] = identity
        harness.store.create_record(
            "failure_decision", identity, decision.model_dump(mode="json")
        )
        for actor in ("fixture-rater-A", "fixture-rater-B"):
            annotation_id = f"audit:{index}:{actor}"
            annotations.append(
                {
                    "annotation_id": annotation_id,
                    "response_id": response.response_id,
                    "purpose": "independent_identification_audit",
                    "cohort_id": cohort_id,
                    "annotator_id": actor,
                    "guideline_hash": guideline_hash,
                    "status": "submitted",
                    "audit_only": True,
                    "uncertain": False,
                    "decision": "failure" if human_failure else "pass",
                }
            )
    cohort = AuditCohort(
        cohort_id=cohort_id,
        purpose="failure_identification",
        response_ids=response_ids,
        seed=41,
        frozen_prediction_version=version,
        locked=True,
        sampling_protocol={
            "membership_count": size,
            "eligible_population_version": "fictional-population-v1",
            "sampling_method": "explicit-fixture-membership",
            "exclusion_categories": harness.config.audits.failure_identification.exclude,
            "excluded_response_ids": [],
            "annotation_protocol": {"fixture_only": True},
            "guideline_hash": guideline_hash,
            "frozen_prediction_ids": bindings,
        },
    )
    harness.store.create_record(
        "audit_cohort", cohort_id, cohort.model_dump(mode="json")
    )
    for annotation in annotations:
        harness.store.create_record("annotation", annotation["annotation_id"], annotation)
    scored = harness.evaluator.score_failure_audit(cohort)
    assert scored["n"] == size
    assert sum(scored["counts"].values()) == size
    assert scored["metrics"]["precision"] == precision
    assert scored["metrics"]["recall"] == recall
    assert scored["metrics"]["f1"] == f1
    assert scored["undefined_display"] == "N/A"
    assert scored["audit_labels_used_for_tuning"] is False
