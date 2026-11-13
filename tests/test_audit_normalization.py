"""F6: real acquisition -> task projection -> rendering/measurement boundaries.

All sources and responses are fictional offline fixtures. Tests preserve the real
loader, prompt renderer, evaluator, and persistence; no model service is called.
The optional pair-F1 policy is an explicit opt-in setting.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from config import ExperimentConfig
from dataset_loader import DatasetLoader
from evaluation import Evaluator
from experiments import ExperimentRunner
from model_client import ModelClient
from schemas import ModelResponse, _canonical_json
from test_data_and_storage import (
    REFERENCE_SENTINEL,
    BundleFixture,
    _read_json,
    _resolve_fixture_setting,
    _write_json,
    _write_yaml,
    baseline_yaml,
    local_bundle,
    store_fixture,
)


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = "HIDDEN_GOLD_METADATA_71EA"


def _alias_resource(bundle, **changes):
    acquisition = _read_json(bundle.acquisition_path)
    resource = acquisition["normalization_resources"]["alias_table"]
    aliases = _read_json(Path(resource["path"]))
    aliases["entries"][0].update(changes)
    _write_json(Path(resource["path"]), aliases)
    resource["sha256"] = hashlib.sha256(Path(resource["path"]).read_bytes()).hexdigest()
    _write_json(bundle.acquisition_path, acquisition)


def _auxiliary(bundle: BundleFixture, text: str, observation: dict) -> dict:
    """Add an actual second acquisition source, not a synthetic join annotation."""
    source = bundle.source_path.parent / "independent-database.jsonl"
    source.write_text(_canonical_json({"id": "other-db-item", "text": text}) + "\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    acquisition = _read_json(bundle.acquisition_path)
    acquisition["sources"].append({
        "source_id": "second-database", "version": "second-db-v1",
        "dataset": None, "publisher": "Independent database", "format": "jsonl",
        "encoding": "utf-8", "id_field": "/id", "records_pointer": "",
        "parser_options": {}, "location": {
            "kind": "local", "path": str(source), "sha256": digest},
    })
    acquisition["evidence_records"].append({
        "evidence_id": "second-db-evidence", "source_id": "second-database",
        "document_id": "other-db-item", "content_reference": digest,
        "content_hash": digest, "source_version": "second-db-v1",
        "role": "model_input", "snippet": text,
        "publication_time": "2024-05-02", "provenance": {
            "selection_pointer": "/text", "hidden_gold": PRIVATE},
    })
    _write_json(bundle.acquisition_path, acquisition)
    snapshots = _read_json(bundle.snapshots_path)
    snapshots["records"].append({
        "evidence_id": "second-db-evidence", "source_id": "second-database",
        "document_id": "other-db-item", "content_hash": digest,
        "version_id": "second-db-v1", "available_at": "2024-05-02",
    })
    _write_json(bundle.snapshots_path, snapshots)
    return {
        "evidence_id": "second-db-evidence", "raw": text,
        "start_offset": 0, "end_offset": len(text),
        "unresolved_policy": "reject", **observation,
    }


def _prepare(bundle: BundleFixture, *, task="affected_systems", stage="CONTEXTUALIZATION",
             edit=None, reference=None, extraction=None):
    conversions = _read_json(bundle.conversions_path)
    conversion = copy.deepcopy(conversions["conversions"][0])
    conversion.update(task_name=task, stage=stage)
    if edit:
        edit(conversion)
    conversions["conversions"] = [conversion]
    _write_json(bundle.conversions_path, conversions)
    selection_path = Path(bundle.config.data.benchmarks["ctibench"].selection_manifest)
    selection = _read_json(selection_path)
    selection["item_ids"] = ["shared-case-item"]
    _write_json(selection_path, selection)
    if reference is not None:
        references = _read_json(bundle.references_path)
        references["references"][0]["answer"] = reference
        _write_json(bundle.references_path, references)
        # Keep the acquired, withheld reference source consistent with the gold
        # answer. Its checksum/lineage must still pass the genuine loader.
        rows = [json.loads(line) for line in bundle.source_path.read_text().splitlines()]
        rows[0]["affected_reference"] = REFERENCE_SENTINEL + ": " + _canonical_json(reference)
        bundle.source_path.write_text("".join(_canonical_json(row) + "\n" for row in rows))
        digest = hashlib.sha256(bundle.source_path.read_bytes()).hexdigest()
        acquisition = _read_json(bundle.acquisition_path)
        acquisition["sources"][0]["location"]["sha256"] = digest
        for record in acquisition["evidence_records"]:
            if record["source_id"] == "fixture-acquisition":
                record.update(content_hash=digest, content_reference=digest)
                if record["evidence_id"] == "fixture-affected-reference":
                    record["snippet"] = rows[0]["affected_reference"]
        _write_json(bundle.acquisition_path, acquisition)
        snapshots = _read_json(bundle.snapshots_path)
        for record in snapshots["records"]:
            if record["source_id"] == "fixture-acquisition":
                record["content_hash"] = digest
        _write_json(bundle.snapshots_path, snapshots)
    payload = bundle.config.model_dump(mode="json")
    payload["operational"]["selected_tasks"] = [task]
    policies = {
        "inference.temperature": 0.0,
        "inference.top_p": 1.0,
        "evaluation.scoring.task_field_extraction_rules": {
            task: extraction or {
                "prediction": "/answer/decisions", "reference": "/answer/decisions",
                "representation": "aligned", "unit_ids": "/candidates",
            }},
        "evaluation.scoring.invalid_output_policy": "error",
        "evaluation.scoring.abstention_policy": "error",
        "evaluation.scoring.f1.positive_class_definitions": 1,
        "evaluation.scoring.f1.averaging": "binary",
        "evaluation.scoring.accuracy.exact_match_and_partial_credit_rules": {"mode": "unit_exact"},
    }
    for path, value in policies.items():
        _resolve_fixture_setting(payload, path, value, stages=("prepare", "evaluate"))
    _write_yaml(bundle.config_path, payload)
    config = ExperimentConfig.load(str(bundle.config_path))
    store = bundle.store_fixture.store
    loader = DatasetLoader(config, store)
    loader.ingest_sources(str(bundle.acquisition_path))
    instance, = loader.build_instances()
    models = ModelClient(config, store)
    evaluator = Evaluator(config, store)
    runner = ExperimentRunner(config, store, models, evaluator, None)
    guidance = {"tasks": {task: {"guidance": "Use the supplied observations.", "candidate_policy": "exclude"}}}
    profiles = {"tasks": {task: {
        "output_profile": instance.output_profile,
        "schema_contract": {"kind": "task", "version": "offline-task-v1", "json_schema": {"type": "object"}},
    }}}
    task_input, _ = runner._task_payload(instance, guidance, profiles)
    prompt = models.render_prompt(str(ROOT / "prompts/task_evaluation.jinja2"), task_input)
    return instance, task_input, prompt, evaluator, config


def _response(instance, config, answer, identity="offline-normalization-output"):
    parsed = {"status": "OK", "answer": answer}
    return ModelResponse(
        response_id=identity, instance_id=instance.instance_id, model_id="G5",
        run_index=0, prompt_hash="a" * 64, config_hash=config.config_hash,
        status="success", raw_text=_canonical_json(parsed), parsed_answer=parsed,
        usage={}, schema_errors=[],
    )


def test_input_normalization_joins_actual_sources_and_reaches_model_without_gold(local_bundle):
    """Missing preprocessing consumption loses source equivalence at model input."""
    second = _auxiliary(local_bundle, "CDN-Fixture.Example", {
        "kind": "ioc", "ioc_type": "domain", "role": "input_normalization",
        "operations": [{"operation": "lowercase"}], "hidden_gold": PRIVATE,
    })
    def edit(conversion):
        for item in conversion["identifier_observations"]:
            item["role"] = "input_normalization" if item["kind"] == "ioc" else "provenance_only"
            item["hidden_gold"] = PRIVATE
        conversion["identifier_observations"].append(second)
        conversion["input_evidence_ids"].append("second-db-evidence")
    instance, payload, prompt, evaluator, config = _prepare(local_bundle, edit=edit)
    assert len(payload["structured_feeds"]) == 1
    ioc = next(item for item in payload["structured_feeds"] if item["kind"] == "ioc")
    assert ioc["canonical"] == "cdn-fixture.example"
    assert ioc["ioc_type"] == "domain"
    assert [(item["source_id"], item["evidence_id"], item["raw"]) for item in ioc["mentions"]] == [
        ("fixture-acquisition", "fixture-narrative", "cdn-fixture[.]example"),
        ("second-database", "second-db-evidence", "CDN-Fixture.Example"),
    ]
    assert "CanonicalFixtureTool" not in prompt
    assert "cdn-fixture.example" in prompt
    assert "cdn-fixture[.]example" in prompt
    assert PRIVATE not in prompt
    assert REFERENCE_SENTINEL not in prompt
    assert PRIVATE in _canonical_json(instance.transformation_log)
    assert instance.transformation_log["normalization_contract"] == "task-identifier-roles-v1"
    score = evaluator.score_response(instance, _response(instance, config, {"decisions": [1, 0]}))
    assert score["contribution"]["counts"] == {"tp": 1, "fp": 0, "fn": 0, "tn": 1}


def test_legacy_normalization_is_explicitly_provenance_only(local_bundle):
    """Old observations must not become new answer-bearing prompt inputs."""
    instance, payload, prompt, _, _ = _prepare(local_bundle)
    assert {item["role"] for item in instance.transformation_log["normalization"]} == {"provenance_only"}
    assert payload["structured_feeds"] == []
    assert "CanonicalFixtureTool" not in prompt
    assert "cdn-fixture.example" not in prompt


def test_input_aliases_require_visible_source_support_and_project_only_that_support(local_bundle):
    _auxiliary(local_bundle, "AliasFixture and CanonicalFixtureTool are names of the same fictional tool.", {})
    _alias_resource(local_bundle, evidence_refs=["second-db-evidence"], hidden_gold=PRIVATE)
    def edit(conversion):
        conversion["identifier_observations"] = [conversion["identifier_observations"][1]]
        conversion["identifier_observations"][0].update(role="input_normalization", hidden_gold=PRIVATE)
        conversion["input_evidence_ids"].append("second-db-evidence")
    instance, payload, prompt, _, _ = _prepare(local_bundle, edit=edit)
    feed, = payload["structured_feeds"]
    assert feed["canonical"] == "CanonicalFixtureTool"
    assert feed["supporting_evidence_ids"] == ["second-db-evidence"]
    assert feed["mentions"][0]["evidence_id"] == "fixture-narrative"
    assert PRIVATE not in prompt
    assert PRIVATE in _canonical_json(instance.transformation_log)


@pytest.mark.parametrize("support", ["missing-alias-support", "fixture-affected-reference"])
def test_input_aliases_cannot_use_hidden_or_unavailable_mapping_support(local_bundle, support):
    _alias_resource(local_bundle, evidence_refs=[support])
    def edit(conversion):
        conversion["identifier_observations"][1]["role"] = "input_normalization"
    with pytest.raises(ValueError, match="model-visible alias support"):
        _prepare(local_bundle, edit=edit)


@pytest.mark.parametrize("task,stage,kind,entity_type", [
    ("ioc_normalization", "CONTEXTUALIZATION", "ioc", None),
    ("vulnerability_linking", "CONTEXTUALIZATION", "alias", "vulnerability"),
    ("malware_family_mapping", "CONTEXTUALIZATION", "alias", "malware_family"),
    ("threat_actor_linking", "ATTRIBUTION", "alias", "actor"),
    ("campaign_attribution", "ATTRIBUTION", "alias", "campaign"),
    ("ttp_extraction", "ATTRIBUTION", "alias", "technique"),
    ("mitigation_ttp_mapping", "MITIGATION", "alias", "mitigation"),
])
def test_input_normalization_cannot_resolve_task_targets(local_bundle, task, stage, kind, entity_type):
    """A manifest role must not authorize revealing the quantity being evaluated."""
    if entity_type:
        _alias_resource(local_bundle, entity_type=entity_type)
    def edit(conversion):
        item = conversion["identifier_observations"][0 if kind == "ioc" else 1]
        item.update(role="input_normalization", kind=kind)
        if entity_type:
            item["entity_type"] = entity_type
    with pytest.raises(ValueError, match="task target"):
        _prepare(local_bundle, task=task, stage=stage, edit=edit)


def test_literal_cve_formatting_preserves_linking_selection_and_scoring(local_bundle):
    """Formatting a quoted distractor is permitted; it does not supply the selected CVE."""
    second = _auxiliary(local_bundle, "cve-2099-5678", {"kind": "cve", "role": "input_normalization"})
    def edit(conversion):
        conversion["identifier_observations"] = [second]
        conversion["input_evidence_ids"].append("second-db-evidence")
    instance, payload, prompt, evaluator, config = _prepare(
        local_bundle, task="vulnerability_linking", edit=edit,
        reference={"selected": "CVE-2099-1234"}, extraction={
            "prediction": "/answer/selected", "reference": "/answer/selected", "representation": "aligned"},
    )
    assert payload["structured_feeds"][0]["canonical"] == "CVE-2099-5678"
    assert "CVE-2099-5678" in prompt
    assert "CVE-2099-1234" not in prompt
    wrong = evaluator.score_response(instance, _response(instance, config, {"selected": "CVE-2099-5678"}))
    right = evaluator.score_response(instance, _response(instance, config, {"selected": "CVE-2099-1234"}, "correct-cve"))
    assert wrong["contribution"]["unit_correct"] == [0]
    assert right["contribution"]["unit_correct"] == [1]


def test_incident_ticket_identifier_inputs_preserve_hidden_ticket_field_decisions(local_bundle):
    """An IOC identity can support ticket generation without supplying ticket labels."""
    def edit(conversion):
        conversion["identifier_observations"][0]["role"] = "input_normalization"
    instance, payload, prompt, evaluator, config = _prepare(
        local_bundle, task="incident_ticket_generation", stage="MITIGATION", edit=edit,
        reference={"ticket": ["malware", "P2", "SOC-triage"]}, extraction={
            "prediction": "/answer/ticket", "reference": "/answer/ticket", "representation": "aligned"},
    )
    assert payload["structured_feeds"][0]["canonical"] == "cdn-fixture.example"
    assert "cdn-fixture.example" in prompt
    assert "P2" not in prompt
    assert "SOC-triage" not in prompt
    wrong = evaluator.score_response(instance, _response(instance, config, {
        "ticket": ["malware", "P2", "wrong-assignment-group"]}))
    assert wrong["contribution"]["unit_correct"] == [1, 1, 0]


@pytest.mark.parametrize("role", ["model_input", "gold_reference", "", None])
def test_unknown_normalization_roles_fail_closed(local_bundle, role):
    def edit(conversion):
        conversion["identifier_observations"][0]["role"] = role
    with pytest.raises(ValueError, match="normalization role"):
        _prepare(local_bundle, edit=edit)


def test_ioc_gold_is_withheld_and_real_type_value_pair_f1_detects_wrong_type(local_bundle):
    """Canonical IOC answers need a scoring path, without copying gold into generation."""
    gold = [
        {"raw": "cdn-fixture[.]example", "type": "domain", "value": "cdn-fixture.example", "tags": [PRIVATE]},
        {"raw": "other[.]example", "type": "domain", "value": "other.example"},
    ]
    second = _auxiliary(local_bundle, "other[.]example", {
        "kind": "ioc", "ioc_type": "domain",
        "operations": [{"operation": "replace", "old": "[.]", "new": "."}],
    })
    def edit(conversion):
        conversion["identifier_observations"].append(second)
        conversion["input_evidence_ids"].append("second-db-evidence")
    instance, payload, prompt, evaluator, config = _prepare(
        local_bundle, task="ioc_normalization", edit=edit,
        reference={"ioc_normalization": gold}, extraction={
            "prediction": "/answer/ioc_normalization", "reference": "/answer/ioc_normalization",
            "representation": "ioc_type_value"},
    )
    assert payload["structured_feeds"] == []
    assert "cdn-fixture.example" not in prompt
    assert "other.example" not in prompt
    assert PRIVATE not in prompt
    predicted = [
        {"raw": "cdn-fixture[.]example", "type": "domain", "value": "cdn-fixture.example", "tags": ["different metadata"]},
        {"raw": "other[.]example", "type": "url", "value": "other.example"},
    ]
    score = evaluator.score_response(instance, _response(instance, config, {"ioc_normalization": predicted}))
    assert score["contribution"]["counts"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 0}
    assert evaluator._metric([score])["value"] == pytest.approx(0.5)


@pytest.mark.parametrize("predicted,counts", [
    ([], {"tp": 0, "fp": 0, "fn": 1, "tn": 0}),
    ([{"raw": "cdn-fixture[.]example", "type": "domain", "value": "wrong.example"}],
     {"tp": 0, "fp": 1, "fn": 1, "tn": 0}),
    ([{"raw": "different-raw", "type": "domain", "value": "cdn-fixture.example"}],
     {"tp": 0, "fp": 1, "fn": 1, "tn": 0}),
])
def test_ioc_pair_f1_keeps_raw_identity_and_missing_predictions(local_bundle, predicted, counts):
    instance, _, _, evaluator, config = _prepare(
        local_bundle, task="ioc_normalization",
        reference={"ioc_normalization": [{"raw": "cdn-fixture[.]example", "type": "domain", "value": "cdn-fixture.example"}]},
        extraction={"prediction": "/answer/ioc_normalization", "reference": "/answer/ioc_normalization",
                    "representation": "ioc_type_value"},
    )
    score = evaluator.score_response(instance, _response(instance, config, {"ioc_normalization": predicted}))
    assert score["contribution"]["counts"] == counts


@pytest.mark.parametrize("path,value", [
    ("evaluation.scoring.f1.averaging", "micro"),
    ("evaluation.scoring.f1.positive_class_definitions", 0),
])
def test_ioc_pair_policy_rejects_incompatible_binary_membership_metrics(local_bundle, path, value):
    gold = [{"raw": "cdn-fixture[.]example", "type": "domain", "value": "cdn-fixture.example"}]
    instance, _, _, _, config = _prepare(
        local_bundle, task="ioc_normalization", reference={"ioc_normalization": gold},
        extraction={"prediction": "/answer/ioc_normalization", "reference": "/answer/ioc_normalization",
                    "representation": "ioc_type_value"},
    )
    payload = config.model_dump(mode="json")
    _resolve_fixture_setting(payload, path, value, stages=("evaluate",))
    changed = ExperimentConfig.model_validate(payload)
    evaluator = Evaluator(changed, local_bundle.store_fixture.store)
    with pytest.raises(ValueError, match="binary averaging and positive class 1"):
        evaluator.score_response(instance, _response(instance, changed, {"ioc_normalization": gold}))
