"""Independent review boundaries for algorithmic C3 and RAG revisions."""
import copy
import json
from pathlib import Path

import pytest

from generation_context import review_task_input
from schemas import FailureDecision, ModelResponse, TaskInstance, _content_hash, _text_hash
from test_audit_c3_estimator import _configure_counterexample
from test_audit_generation_context import RETRIEVED, generated
from test_discovery_repairs import production_harness
from test_taxonomy_and_categorization import Harness, harness, no_live_inference


@pytest.mark.parametrize("interface", ["token_logprobs", "candidate_log_likelihoods", "autoregressive_complete"])
def test_algorithmic_c3_correction_retains_resolvable_generation_lineage(
    harness: Harness, tmp_path: Path, interface: str,
) -> None:
    _configure_counterexample(harness, tmp_path, interface)
    result = harness.runner().run_counterfactual_defense([harness.instance_ids[0]])
    assert result["status"] == "completed"
    instance = TaskInstance.model_validate(harness.store.read_record("task_instance", harness.instance_ids[0]))
    original = ModelResponse.model_validate(harness.store.read_record("model_response", result["original_response_ids"][0]))
    corrected = ModelResponse.model_validate(harness.store.read_record("model_response", result["corrected_response_ids"][0]))
    original_material = review_task_input(harness.store, original, instance)
    assert original_material["generation_context"]["manifest_id"]
    material = review_task_input(harness.store, corrected, instance)
    assert material["generation_context"]["manifest_id"]
    assert material["environment"]["report_label"] == "original"
    assert corrected.usage["generation_context_id"] == original.usage["generation_context_id"]
    derivation = corrected.usage["output_derivation"]
    assert material["output_derivation"]["kind"] == "counterfactual_probability_aggregation"
    assert len(material["output_derivation"]["counterfactual_inputs"]) == len(derivation["source_response_ids"])
    assert derivation["kind"] == "counterfactual_probability_aggregation"
    assert derivation["source_response_ids"] == corrected.usage["aggregation_response_ids"]
    sources = [ModelResponse.model_validate(harness.store.read_record("model_response", identity))
               for identity in derivation["source_response_ids"]]
    assert derivation["source_generation_context_ids"] == [
        source.usage["generation_context_id"] for source in sources
    ]
    assert all(review_task_input(harness.store, source, instance)["environment"]["report_label"] == "variant-0"
               for source in sources)
    if interface == "autoregressive_complete":
        assert derivation["aggregation_probability_manifest_ids"] == corrected.usage["aggregation_probability_manifest_ids"]
    assert corrected.usage["evaluation_context"]["condition"] == "corrected"
    assert corrected.usage["algorithm"] == "weighted_counterfactual_probability_aggregation"


@pytest.mark.parametrize("production_harness", [True], indirect=True)
def test_strong_rag_revision_retains_exact_prompt_and_separate_retrieved_citations(production_harness):
    """Exercise one real revision template, then native AI/human evidence validation."""
    world = production_harness
    candidate, original_payload = generated(world, mitigation=False)
    runner = world.runner()
    instance = TaskInstance.model_validate(world.store.read_record("task_instance", candidate.instance_id))
    template = str(Path(__file__).resolve().parents[1] / "prompts/defense_revision.jinja2")
    retrieved = [copy.deepcopy(RETRIEVED)]
    evidence_hash = _content_hash(retrieved)
    task_input = {key: copy.deepcopy(original_payload[key]) for key in (
        "case_id", "task_name", "snapshot_date", "task_guidance", "source_documents",
        "structured_feeds", "candidates", "candidate_policy", "environment", "output_profile",
    )}
    task_input.update(instance_id=instance.instance_id, stage=instance.stage.value)
    settings = world.config.defenses.stronger_grounding.model_dump(mode="json")
    settings["retrieval"]["evidence_context_budget"] = 1000
    settings["evidence_verification"].update({
        "checkpoint": "offline-revision-verifier-v1", "claim_segmentation": "offline-single-claim-v1",
        "alignment_threshold": 0.7, "revision_prompt_template_path": template,
    })
    verifier = settings["evidence_verification"]
    feedback = {
        "candidate_response_id": candidate.response_id, "evidence_package_hash": evidence_hash,
        **{key: verifier[key] for key in ("method", "checkpoint", "claim_segmentation", "alignment_threshold")},
        "claims": [{
            "claim_id": "offline-revision-claim-v1",
            "candidate_quote": candidate.parsed_answer["justification"],
            "evidence_ids": [RETRIEVED["evidence_id"]], "alignment_score": 0.125,
            "feedback_text": "Offline verifier checked only the supplied incident relation.",
        }],
    }
    artifact = runner._manifest("grounding_verifier_feedback", {
        "feedback": copy.deepcopy(feedback), "candidate_response_id": candidate.response_id,
    })
    feedback["artifact_id"] = artifact["manifest_id"]
    controls = {"temperature": world.config.inference.temperature, "top_p": world.config.inference.top_p}
    contract_hash = _content_hash(task_input["output_profile"])
    controls_hash = _content_hash(controls)
    payload = {
        "task_input": task_input, "candidate_response": candidate.model_dump(mode="json"),
        "retrieved_evidence": retrieved, "verification_feedback": feedback,
        "prompt_settings": world.config.evaluation.prompt.model_dump(mode="json"),
        "generation_settings": controls, "grounding_settings": settings,
        "revision_context": {
            "configuration": "strong_rag_with_evidence_verification", "comparison_id": "offline-revision-comparison",
            "instance_id": instance.instance_id, "model_id": "G5", "revision_index": 1,
            "template_path": template,
            "evidence_package_hash": evidence_hash, "candidate_evidence_package_hash": evidence_hash,
            "output_contract_hash": contract_hash, "candidate_output_contract_hash": contract_hash,
            "generation_settings_hash": controls_hash, "candidate_generation_settings_hash": controls_hash,
            "snapshot_admission_verified": True, "feedback_snapshot_admission_verified": True,
        },
    }
    world.models.script = lambda request, packet: {
        "status": "OK", "task": packet["task_input"]["task_name"],
        "case_id": packet["task_input"]["case_id"], "snapshot_date": packet["task_input"]["snapshot_date"],
        "answer": {"choice": "A"}, "justification": "Offline revised answer remains incorrect.",
        "evidence_refs": [RETRIEVED["evidence_id"]],
    }
    revised = runner._generate_task(instance, "G5", 0, payload,
        {"kind": "task", "version": "f1-revision-context-v1", "json_schema": {"type": "object"}},
        {"protocol": "stronger_grounding", "condition": "strong_rag_with_evidence_verification",
         "comparison_id": "offline-revision-comparison"}, template=template)
    revision_prompt = world.models.requests[-1]["prompt"]
    saved = world.store.read_record("experiment_manifest", revised.usage["generation_context_id"])
    assert saved["payload"] == payload
    assert saved["rendered_prompt"] == revision_prompt
    assert saved["prompt_hash"] == revised.prompt_hash == _text_hash(revision_prompt)
    assert json.dumps(candidate.raw_text) in revision_prompt
    assert feedback["claims"][0]["feedback_text"] in revision_prompt
    assert '"alignment_score": 0.125' in revision_prompt
    assert "fixture-reference" not in revision_prompt
    assert "Reference actor is Correct." not in revision_prompt
    assert RETRIEVED["evidence_id"] not in {item["evidence_id"] for item in task_input["source_documents"]}

    decision_payload = world.store.list_records("failure_decision", {"response_id": candidate.response_id})[0]
    decision_payload["response_id"] = revised.response_id
    decision = FailureDecision.model_validate(decision_payload)
    decision_id = _content_hash(decision)
    world.store.create_record("failure_decision", decision_id, decision.model_dump(mode="json"))
    world.store.freeze_records("failure_decision", [decision_id])
    classifier = world.categorizer()
    packet = classifier._packet(revised.response_id, world.taxonomy)
    materials = world.annotations._materials(revised.response_id, classifier._annotation_protocol(world.taxonomy))
    for material in (packet["task_input"], materials["task_input"]):
        assert material["generation_context"]["rendered_prompt"] == revision_prompt
        assert material["retrieved_evidence"] == retrieved
        assert {item["evidence_id"] for item in material["source_documents"]} == {"fixture-input", RETRIEVED["evidence_id"]}
        assert material["task_guidance"] == task_input["task_guidance"]

    judgment = {
        "labels": ["fixture-A"], "uncertain": False,
        "evidence_spans": [{"source_ref": RETRIEVED["evidence_id"], "quote": RETRIEVED["snippet"],
                            "subtype_id": "fixture-A"}],
    }
    world.models.script = lambda request, packet: copy.deepcopy(judgment)
    result = classifier.run([revised.response_id], world.taxonomy)
    assert result["status"] == "completed", result
    tasks = classifier._review_tasks(revised.response_id, world.taxonomy)
    world.annotations.resolve_task(tasks[0].annotation_task_id, "fixture-reviewer", judgment)
    reference = world.annotations.get_reference(revised.response_id, "operational_adjudication")
    assert reference["labels"] == ["fixture-A"]
    assert reference["evidence_spans"][0]["source_ref"] == RETRIEVED["evidence_id"]
