"""Offline production-contract regressions with real hashes and readiness checks."""

import copy
import hashlib
import json
from pathlib import Path

import litellm
import pytest
import yaml

from annotation_app import AnnotationService
from config import ExperimentConfig
from model_client import ModelClient
from schemas import ResponseStatus, TaskInstance, _content_hash, _task_content_projection
from storage import ArtifactStore
from test_taxonomy_and_categorization import (
    GUIDELINE, Harness, ScriptedModels, _proposal, _resolve, _span, harness,
)


@pytest.fixture
def production_harness(tmp_path, harness, request):
    """Use actual configuration identities, native human service, and stage gates."""
    payload = harness.config.model_dump(mode="json")
    guideline = tmp_path / "guideline.json"
    guideline.write_text(json.dumps(GUIDELINE))
    for path, value in {
        "taxonomy.guideline_manifest": str(guideline),
        "operational.prerequisite_manifests.identification": "urn:fixture:identification",
        "operational.prerequisite_manifests.taxonomy": "urn:fixture:taxonomy",
        "inference.precision": "fixture-provider-managed",
        "inference.quantization": "fixture-none",
        "inference.batch_size": 1,
    }.items():
        _resolve(payload, path, value)
    for index, entry in enumerate(payload["models"]["categorization_panel"]):
        for field in ("endpoint_or_checkpoint", "revision"):
            path = f"models.categorization_panel.{index}.{field}"
            payload["implementation_choices"][path] = {
                "configuration_path": path,
                "origin": "implementation",
                "rationale": "Offline provider fixture.",
                "affected_experiments": ["all"],
                "selected_value": entry[field],
            }
    coverage = payload["taxonomy_discovery"]["coverage_definition"]
    for protocol in coverage["annotation_protocols"].values():
        protocol["config_hash"] = "$config_hash"
        if getattr(request, "param", False):
            protocol["assignments"] = {"*": ["fixture-reviewer"]}
            protocol["reference_applicability"] = {"*": {"scope": "Fictional evidence."}}
    _resolve(payload, "taxonomy_discovery.coverage_definition", coverage)
    policy = payload["categorization"]["incomplete_agent_output_policy"]
    policy["annotation_protocol"]["config_hash"] = "$config_hash"
    if getattr(request, "param", None) == "dynamic_taxonomy":
        policy["annotation_protocol"]["taxonomy_id"] = "$taxonomy_id"
    if getattr(request, "param", False):
        policy["annotation_protocol"]["assignments"] = {"*": ["fixture-reviewer"]}
        policy["annotation_protocol"]["reference_applicability"] = {"*": {"scope": "Fictional evidence."}}
    _resolve(payload, "categorization.incomplete_agent_output_policy", policy)
    if getattr(request, "param", None) == "native_seed":
        reasoning = payload["inference"]["reasoning_settings"]
        reasoning["adapters"]["G5"]["seed_policy"] = "omit_if_unsupported"
        _resolve(payload, "inference.reasoning_settings", reasoning)
    path = tmp_path / "production-ready.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = ExperimentConfig.load(str(path))
    assert type(config) is ExperimentConfig
    config.validate("discover_taxonomy")
    config.validate("categorize")
    store = ArtifactStore(str(tmp_path / "production.sqlite"), str(tmp_path / "production-blobs"))
    store.initialize()
    store.create_record("taxonomy", harness.taxonomy.taxonomy_id, harness.taxonomy.model_dump(mode="json"))
    for identity in harness.instance_ids:
        raw = harness.store.read_record("task_instance", identity)
        raw["transformation_log"]["config_hash"] = config.config_hash
        raw["content_hash"] = None
        instance = TaskInstance.model_validate(raw)
        raw["content_hash"] = _content_hash(_task_content_projection(instance))
        store.create_record("task_instance", identity, raw)
        store.freeze_records("task_instance", [identity])
    for identity in harness.failure_ids:
        raw = harness.store.read_record("model_response", identity)
        raw["config_hash"] = config.config_hash
        store.create_record("model_response", identity, raw)
    for decision in harness.store.list_records("failure_decision"):
        store.create_record("failure_decision", _content_hash(decision), decision)
    return Harness(config, store, ScriptedModels(config, store), AnnotationService(store),
                   harness.taxonomy, harness.failure_ids, harness.instance_ids)


def _submit_proposals(world, tasks, additions=()):
    identities = []
    for index, task in enumerate(tasks):
        world.annotations.submit_annotation(task.annotation_task_id, task.assigned_annotator, {
            "taxonomy_proposal": _proposal(additions if index == 0 else ()),
            "uncertain": False,
            "evidence_spans": [_span()],
        })
        identities.extend(item["annotation_id"] for item in world.store.list_records("annotation")
                          if item["annotation_task_id"] == task.annotation_task_id)
    return identities


@pytest.mark.parametrize("production_harness", [False, True], indirect=True)
def test_discovery_reuses_response_across_refinement_versions(production_harness):
    world = production_harness
    manager = world.manager()
    initial = manager.create_initial_taxonomy(_submit_proposals(
        world, manager.request_initialization(world.failure_ids), ("fixture-A",)))
    world.models.script = lambda request, packet: {
        "labels": ["other"], "evidence_spans": [_span("other")], "uncertain": False,
    }
    first = manager.classify_remaining(world.failure_ids, initial)
    child = manager.refine_taxonomy(initial, _submit_proposals(
        world, manager.request_other_reviews(first), ("fixture-B",)))
    second = manager.classify_remaining(world.failure_ids, child)
    assert manager.refine_taxonomy(child, _submit_proposals(
        world, manager.request_other_reviews(second))) == child
    assert child.iteration == initial.iteration + 1


@pytest.mark.parametrize("production_harness", [True], indirect=True)
def test_wildcard_protocol_covers_categorization(production_harness):
    world = production_harness
    result = world.categorizer().run([world.failure_ids[0]], world.taxonomy)
    assert result["status"] == "completed"


@pytest.mark.parametrize("production_harness", ["dynamic_taxonomy"], indirect=True)
def test_discovered_taxonomy_is_applied_without_changing_config_identity(production_harness):
    from schemas import TaxonomyVersion
    world = production_harness
    original_hash = world.config.config_hash
    manager = world.manager()
    _submit_proposals(world, manager.request_initialization(world.failure_ids), ("fixture-A",))
    discovery = manager.run(world.failure_ids)
    assert discovery["status"] == "completed"
    taxonomy = TaxonomyVersion.model_validate(world.store.read_record("taxonomy", discovery["taxonomy_id"]))
    result = world.categorizer().run(world.failure_ids, taxonomy)
    assert result["status"] == "completed"
    assert result["taxonomy_id"] == taxonomy.taxonomy_id
    assert world.config.config_hash == original_hash
    declaration = world.config.categorization.incomplete_agent_output_policy["annotation_protocol"]
    assert declaration["taxonomy_id"] == "$taxonomy_id"


@pytest.mark.parametrize("declaration", [None, "omitted", "foreign-taxonomy"])
def test_runtime_taxonomy_binding_preserves_explicit_mismatch_checks(production_harness, declaration):
    from categorization import Categorizer
    world = production_harness
    payload = world.config.model_dump(mode="json")
    policy = payload["categorization"]["incomplete_agent_output_policy"]
    if declaration == "omitted":
        policy["annotation_protocol"].pop("taxonomy_id")
    else:
        policy["annotation_protocol"]["taxonomy_id"] = declaration
    _resolve(payload, "categorization.incomplete_agent_output_policy", policy)
    config = ExperimentConfig.model_validate(payload)
    categorizer = Categorizer(config, world.store, world.models, world.annotations)
    if declaration == "foreign-taxonomy":
        with pytest.raises(ValueError, match="must bind"):
            categorizer._annotation_protocol(world.taxonomy)
    else:
        protocol = categorizer._annotation_protocol(world.taxonomy)
        assert protocol["taxonomy_id"] == world.taxonomy.taxonomy_id
        assert protocol["config_hash"] == config.config_hash


def test_discovery_resumes_through_real_human_initialization_and_refinement(production_harness):
    world = production_harness
    world.models.script = lambda request, packet: {
        "labels": ["other" if len(packet["current_taxonomy"]["subtype_definitions"]) == 1 else "fixture-B"],
        "evidence_spans": [_span("other" if len(packet["current_taxonomy"]["subtype_definitions"]) == 1 else "fixture-B")],
        "uncertain": False,
    }
    waiting = world.manager().run(world.failure_ids)
    assert waiting["status"] == "waiting_for_annotations"
    from schemas import AnnotationTask
    tasks = [AnnotationTask.model_validate(world.store.read_record("annotation_task", identity))
             for identity in waiting["pending_annotation_task_ids"]]
    _submit_proposals(world, tasks, ("fixture-A",))
    waiting = world.manager().run(world.failure_ids)
    assert waiting["status"] == "waiting_for_annotations"
    tasks = [AnnotationTask.model_validate(world.store.read_record("annotation_task", identity))
             for identity in waiting["pending_annotation_task_ids"]]
    _submit_proposals(world, tasks, ("fixture-B",))
    final = world.manager().run(world.failure_ids)
    assert final["status"] == "completed"
    assert final["frozen"] is True
    calls = len(world.models.provider_calls)
    assert world.manager().run(world.failure_ids) == final
    assert len(world.models.provider_calls) == calls


@pytest.mark.parametrize("exception", [TimeoutError, ValueError])
def test_resume_does_not_retry_a_terminal_transport_failure(harness, monkeypatch, exception):
    payload = harness.config.model_dump(mode="json")
    _resolve(payload, "inference.retry_policy", {
        "max_attempts": 3, "timeout_seconds": 10,
        "retryable_status_codes": [], "retryable_exception_names": ["TimeoutError"],
        "backoff_seconds": [0, 0], "retry_ambiguous_timeouts": False,
    })
    adapters = copy.deepcopy(payload["inference"]["reasoning_settings"])
    adapters["adapters"]["G5"]["supported_parameters"].append("seed")
    _resolve(payload, "inference.reasoning_settings", adapters)
    config = ExperimentConfig.model_validate(payload)
    monkeypatch.setattr(litellm, "token_counter", lambda **kwargs: 1)
    calls = []
    def failure(**kwargs):
        calls.append(kwargs)
        raise exception("offline failure")
    monkeypatch.setattr(litellm, "completion", failure)
    request = {
        "model_id": "G5", "prompt": "offline retry contract", "run_index": 0,
        "schema_name": "retry-contract", "instance_id": harness.instance_ids[0],
        "schema_contract": harness.categorizer()._contract(harness.taxonomy),
        "template_provenance": {"fixture": "offline"},
    }
    first = ModelClient(config, harness.store).generate_batch([request])[0]
    second = ModelClient(config, harness.store).generate_batch([request])[0]
    assert second.response_id == first.response_id
    assert len(calls) == 1


def test_distinct_categorization_runs_have_distinct_predictions(production_harness):
    from categorization import Categorizer
    world = production_harness
    before = world.config.config_hash
    first = Categorizer(world.config, world.store, world.models, world.annotations, run_index=0)
    second = Categorizer(world.config, world.store, world.models, world.annotations, run_index=1)
    initial = first.run_round_one(world.failure_ids[0], world.taxonomy)
    repeated = second.run_round_one(world.failure_ids[0], world.taxonomy)
    assert {item.assignment_id for item in initial}.isdisjoint(item.assignment_id for item in repeated)
    assert {request["run_index"] for request in world.models.requests} == {0, 1}
    assert world.config.config_hash == before


def test_finite_candidate_backend_does_not_require_generation_logprobs(production_harness):
    world = production_harness
    payload = world.config.model_dump(mode="json")
    _resolve(payload, "operational.selected_defense", "counterfactual_consistency")
    _resolve(payload, "defenses.counterfactual_consistency.probability_extraction", {
        "interface": "candidate_log_likelihoods",
        "aggregation_space": "finite_candidates",
        "adapter": {
            "callable": "test_experiment_repairs:candidate_probabilities",
            "version": "offline-v1",
            "source_sha256": hashlib.sha256(Path(__file__).with_name(
                "test_experiment_repairs.py").read_bytes()).hexdigest(),
            "parameters": {"candidates": {"A": "A", "B": "B"}},
        },
    })
    config = ExperimentConfig.model_validate(payload)
    ModelClient(config, world.store).validate_endpoints(["G5"])


def test_provider_object_content_is_not_repaired_into_valid_json(harness, monkeypatch):
    payload = harness.config.model_dump(mode="json")
    adapters = copy.deepcopy(payload["inference"]["reasoning_settings"])
    adapters["adapters"]["G5"]["supported_parameters"].append("seed")
    _resolve(payload, "inference.reasoning_settings", adapters)
    config = ExperimentConfig.model_validate(payload)
    monkeypatch.setattr(litellm, "token_counter", lambda **kwargs: 1)
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: {
        "choices": [{"message": {"content": {
            "labels": ["fixture-A"], "evidence_spans": [_span("fixture-A")],
            "uncertain": False,
        }}, "finish_reason": "stop"}],
    })
    request = {
        "model_id": "G5", "prompt": "offline malformed container", "run_index": 0,
        "schema_name": "container-contract", "instance_id": harness.instance_ids[0],
        "schema_contract": harness.categorizer()._contract(harness.taxonomy),
        "template_provenance": {"fixture": "offline"},
    }
    output = ModelClient(config, harness.store).generate_batch([request])[0]
    assert output.status == ResponseStatus.INVALID_SCHEMA
    assert output.parsed_answer == {}
    assert "text string" in " ".join(output.schema_errors)
