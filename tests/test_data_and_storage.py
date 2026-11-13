"""Offline tests for CTI schemas, configuration, data preparation, and storage.

Fixture policies are explicit engineering test inputs, not inferred paper
settings. The repository's config.yaml is read but is never modified. No model,
HTTP, or benchmark-download endpoint is invoked.

The tests use public service interfaces. Shared canonicalization helpers follow
the contracts implemented in schemas.py. Read-only SQLite inspection is limited
to integrity, lifecycle, and checkpoint properties without public readers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml
from pydantic import ValidationError

from config import ExperimentConfig
from dataset_loader import DatasetLoader
from schemas import (
    EvidenceRecord,
    LabelAssignment,
    ModelResponse,
    ResponseStatus,
    StageState,
    StageStatus,
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

REFERENCE_SENTINEL: str = "FIXTURE_WITHHELD_REFERENCE_7C419"
OUTCOME_SENTINEL: str = "FIXTURE_FUTURE_OUTCOME_8DB62"
FIXTURE_SNAPSHOT: str = "2024-06-01"
FIXTURE_VERSION: str = "fictional-test-source-v1"
FIXTURE_ALIAS_VERSION: str = "fictional-test-aliases-v1"


@dataclass(frozen=True)
class StoreFixture:
    """Explicit local persistence paths and their initialized store."""

    store: ArtifactStore
    database_path: Path
    artifact_root: Path


@dataclass(frozen=True)
class BundleFixture:
    """Local acquisition, conversion, reference, and configuration artifacts."""

    config: ExperimentConfig
    config_path: Path
    acquisition_path: Path
    snapshots_path: Path
    conversions_path: Path
    references_path: Path
    source_path: Path
    source_bytes: bytes
    store_fixture: StoreFixture
    loader: DatasetLoader


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(payload) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value: Any = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _put(payload: dict[str, Any], path: str, value: Any) -> None:
    current: dict[str, Any] = payload
    components: list[str] = path.split(".")
    for component in components[:-1]:
        child: Any = current.setdefault(component, {})
        assert isinstance(child, dict)
        current = child
    current[components[-1]] = copy.deepcopy(value)


def _resolve_fixture_setting(
    payload: dict[str, Any],
    path: str,
    value: Any,
    stages: tuple[str, ...] = ("prepare",),
) -> None:
    """Resolve a setting with test-only provenance rather than hidden defaults."""
    _put(payload, path, value)
    choices: dict[str, Any] = payload.setdefault("implementation_choices", {})
    choices[path] = {
        "configuration_path": path,
        "origin": "implementation",
        "rationale": (
            "Offline infrastructure fixture only; this value is not an author "
            "setting and must not be used for scientific reproduction."
        ),
        "affected_experiments": list(stages),
        "selected_value": copy.deepcopy(value),
    }


def _rehash_task(payload: dict[str, Any]) -> TaskInstance:
    """Recompute the implemented task-content digest after an explicit edit."""
    edited: dict[str, Any] = copy.deepcopy(payload)
    edited["content_hash"] = None
    draft: TaskInstance = TaskInstance.model_validate(edited)
    edited["content_hash"] = _content_hash(_task_content_projection(draft))
    return TaskInstance.model_validate(
        edited, context={"require_content_hash": True}
    )


def _minimal_task(config_hash: str = "a" * 64) -> TaskInstance:
    """Build a structural storage fixture, not an acquired scientific instance."""
    return _rehash_task(
        {
            "instance_id": "fixture-storage-instance",
            "case_id": "fixture-storage-case",
            "source_dataset": "ctibench",
            "source_item_id": "fixture-storage-source-item",
            "task_name": "affected_systems",
            "stage": "CONTEXTUALIZATION",
            "snapshot_date": FIXTURE_SNAPSHOT,
            "evidence_records": [],
            "candidates": ["fictional-asset-A", "fictional-asset-B"],
            "environment": {"platform": "fictional-platform"},
            "reference": {
                "version": "fixture-reference-v1",
                "answer": {"decisions": [1, 0]},
            },
            "transformation_log": {
                "corpus_version": "fixture-corpus",
                "config_hash": config_hash,
                "history": [{"operation": "explicit-test-fixture"}],
            },
            "output_profile": {"type": "object"},
            "content_hash": None,
        }
    )


def _request_payload(
    task: TaskInstance,
    run_index: int = 0,
    *,
    endpoint: str = "fixture/local-endpoint",
    revision: str = "fixture-checkpoint-v1",
    prompt: str = "Fictional offline task. Return one JSON object.",
    temperature: float = 0.0,
    schema_version: str = "fixture-output-schema-v1",
) -> dict[str, Any]:
    """Construct the exact supplementary storage request contract."""
    request: dict[str, Any] = {
        "model_id": "G5",
        "endpoint_or_checkpoint": endpoint,
        "revision": revision,
        "prompt_hash": _text_hash(prompt),
        "generation_controls": {
            "temperature": temperature,
            "top_p": 1.0,
            "max_tokens": 32,
        },
        "seed_behavior": {
            "requested": None,
            "supported": False,
            "transmitted": False,
        },
        "schema_name": "fixture-task",
        "schema_version": schema_version,
        "run_index": run_index,
        "config_hash": task.transformation_log["config_hash"],
        "provider_options": {
            "fixture_only": True,
            "message_mapping": "single_user_message",
        },
        "instance_id": task.instance_id,
        "prompt": prompt,
    }
    request["request_hash"] = _content_hash(request)
    return request


def _response_payload(
    task: TaskInstance,
    request: dict[str, Any],
    response_id: str = "fixture-response",
    status: ResponseStatus = ResponseStatus.SUCCESS,
) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "request_hash": request["request_hash"],
        "started_at": "2024-06-02T10:00:00Z",
        "completed_at": "2024-06-02T10:00:01Z",
    }
    parsed: dict[str, Any] = {
        "status": "OK",
        "answer": {"decisions": [1, 0]},
        "confidence": 0.5,
    }
    errors: list[str] = []
    raw: str = _canonical_json(parsed)
    if status == ResponseStatus.TRANSPORT_FAILURE:
        usage["execution_error"] = {
            "exception_class": "FixtureTransportError",
            "message": "Fictional offline transport failure.",
        }
        parsed = {}
        raw = ""
    elif status == ResponseStatus.INVALID_SCHEMA:
        parsed = {}
        raw = "{fictional invalid JSON"
        errors = ["Fixture schema validation failed."]
    elif status == ResponseStatus.REFUSAL:
        parsed = {}
        raw = "Fictional provider refusal."

    return ModelResponse(
        response_id=response_id,
        instance_id=task.instance_id,
        model_id=request["model_id"],
        run_index=request["run_index"],
        prompt_hash=request["prompt_hash"],
        config_hash=request["config_hash"],
        status=status,
        raw_text=raw,
        parsed_answer=parsed,
        usage=usage,
        schema_errors=errors,
    ).model_dump(mode="json")


@pytest.fixture
def baseline_yaml() -> dict[str, Any]:
    """Read the actual supplied manifest; never resolve study nulls implicitly."""
    assert CONFIG_PATH.is_file(), "The supplied config.yaml must be present."
    payload: Any = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return copy.deepcopy(payload)


@pytest.fixture
def baseline_config(baseline_yaml: dict[str, Any], tmp_path: Path) -> ExperimentConfig:
    path: Path = tmp_path / "baseline.yaml"
    _write_yaml(path, baseline_yaml)
    return ExperimentConfig.load(str(path))


@pytest.fixture
def store_fixture(tmp_path: Path) -> StoreFixture:
    database: Path = tmp_path / "storage" / "artifacts.sqlite"
    artifacts: Path = tmp_path / "storage" / "blobs"
    store: ArtifactStore = ArtifactStore(str(database), str(artifacts))
    store.initialize()
    return StoreFixture(store, database, artifacts)


@pytest.fixture
def stored_task(store_fixture: StoreFixture) -> TaskInstance:
    task: TaskInstance = _minimal_task()
    store_fixture.store.create_record(
        "task_instance", task.instance_id, task.model_dump(mode="json")
    )
    return task


@pytest.fixture
def fixture_taxonomy() -> TaxonomyVersion:
    """A human-style fixture taxonomy, deliberately not the published 15 labels."""
    return TaxonomyVersion.model_validate(
        {
            "taxonomy_id": "fixture-discovered-taxonomy",
            "parent_id": None,
            "iteration": 0,
            "guideline_hash": _content_hash({"fixture": "guideline"}),
            "frozen": True,
            "subtype_definitions": [
                {
                    "subtype_id": subtype_id,
                    "name": f"Fictional subtype {subtype_id}",
                    "definition": "A fictional discrepancy used only for tests.",
                    "evidence_requirements": ["An exact source discrepancy."],
                    "supporting_annotation_ids": [],
                }
                for subtype_id in ("fixture-A", "fixture-B")
            ],
            "inclusion_exclusion_rules": {
                subtype_id: {
                    "inclusion": ["The fictional discrepancy is evidenced."],
                    "exclusion": ["No discrepancy is present."],
                    "boundary_rules": [],
                    "permitted_cooccurring_labels": [],
                }
                for subtype_id in ("fixture-A", "fixture-B")
            },
        }
    )


@pytest.fixture
def local_bundle(
    baseline_yaml: dict[str, Any],
    store_fixture: StoreFixture,
    tmp_path: Path,
) -> BundleFixture:
    """Create heterogeneous evidence and explicit benchmark-to-task conversions."""
    root: Path = tmp_path / "bundle"
    root.mkdir()

    rows: list[dict[str, Any]] = [
        {
            "id": "shared-case-item",
            "structured": {
                "indicator": "cdn-fixture[.]example",
                "mechanism": "fictional-mechanism-A",
            },
            "text": (
                "Fictional advisory: cdn-fixture[.]example is associated with "
                "the fictional mechanism. Tool AliasFixture is mentioned."
            ),
            "old_claim": "Fictional old claim: delivery used channel A.",
            "correction": "Fictional correction: delivery used channel B, not A.",
            "affected_reference": REFERENCE_SENTINEL + ": only asset A is affected.",
            "link_reference": "Fictional reference: candidate CVE-2099-1234 is selected.",
        },
        {
            "id": "forecast-item",
            "text": "Fictional pre-snapshot telemetry describes mechanism Z.",
            "outcome": OUTCOME_SENTINEL + ": exploitation was observed after the snapshot.",
        },
    ]
    source_path: Path = root / "fictional-benchmark.jsonl"
    source_bytes: bytes = (
        "\n".join(_canonical_json(row) for row in rows) + "\n"
    ).encode("utf-8")
    source_path.write_bytes(source_bytes)
    source_digest: str = hashlib.sha256(source_bytes).hexdigest()

    alias_path: Path = root / "aliases.json"
    _write_json(
        alias_path,
        {
            "version": FIXTURE_ALIAS_VERSION,
            "entries": [
                {
                    "name": "AliasFixture",
                    "canonical": "CanonicalFixtureTool",
                    "entity_type": "tool",
                    "evidence_refs": ["fixture-alias-support"],
                }
            ],
        },
    )
    alias_digest: str = hashlib.sha256(alias_path.read_bytes()).hexdigest()

    def evidence(
        evidence_id: str,
        document_id: str,
        pointer: str,
        snippet: str,
        role: str = "model_input",
        published: str = "2024-05-01",
        modified: str | None = None,
        conflicts_with: list[str] | None = None,
    ) -> dict[str, Any]:
        provenance: dict[str, Any] = {"selection_pointer": pointer}
        if conflicts_with is not None:
            provenance["conflicts_with"] = conflicts_with
        return {
            "evidence_id": evidence_id,
            "source_id": "fixture-acquisition",
            "document_id": document_id,
            "content_reference": source_digest,
            "content_hash": source_digest,
            "source_version": FIXTURE_VERSION,
            "role": role,
            "snippet": snippet,
            "publication_time": published,
            "modification_time": modified,
            "retrieval_time": "2024-11-01T12:00:00Z",
            "claim_valid_from": published,
            "claim_valid_until": None,
            "provenance": provenance,
        }

    evidence_records: list[dict[str, Any]] = [
        evidence("fixture-narrative", "shared-case-item", "/text", rows[0]["text"]),
        evidence(
            "fixture-structured",
            "shared-case-item",
            "/structured/indicator",
            rows[0]["structured"]["indicator"],
        ),
        evidence(
            "fixture-old", "shared-case-item", "/old_claim", rows[0]["old_claim"],
            published="2024-04-01",
            conflicts_with=["fixture-correction"],
        ),
        evidence(
            "fixture-correction", "shared-case-item", "/correction",
            rows[0]["correction"], published="2024-05-15",
            conflicts_with=["fixture-old"],
        ),
        evidence(
            "fixture-affected-reference", "shared-case-item", "/affected_reference",
            rows[0]["affected_reference"], role="reference_only",
        ),
        evidence(
            "fixture-link-reference", "shared-case-item", "/link_reference",
            rows[0]["link_reference"], role="reference_only",
        ),
        evidence("fixture-forecast-input", "forecast-item", "/text", rows[1]["text"]),
        evidence(
            "fixture-outcome", "forecast-item", "/outcome", rows[1]["outcome"],
            role="outcome_reference", published="2024-06-20",
        ),
    ]
    acquisition_path: Path = root / "acquisition.json"
    _write_json(
        acquisition_path,
        {
            "version": "fixture-acquisition-manifest-v1",
            "sources": [
                {
                    "source_id": "fixture-acquisition",
                    "version": FIXTURE_VERSION,
                    "dataset": "ctibench",
                    "publisher": "Fictional fixture publisher",
                    "format": "jsonl",
                    "encoding": "utf-8",
                    "id_field": "/id",
                    "records_pointer": "",
                    "parser_options": {},
                    "location": {
                        "kind": "local",
                        "path": str(source_path),
                        "sha256": source_digest,
                    },
                }
            ],
            "evidence_records": evidence_records,
            "normalization_resources": {
                "alias_table": {
                    "version": FIXTURE_ALIAS_VERSION,
                    "path": str(alias_path),
                    "sha256": alias_digest,
                }
            },
        },
    )

    snapshots_path: Path = root / "snapshots.json"
    _write_json(
        snapshots_path,
        {
            "version": "fixture-historical-snapshots-v1",
            "records": [
                {
                    "evidence_id": item["evidence_id"],
                    "source_id": item["source_id"],
                    "document_id": item["document_id"],
                    "content_hash": source_digest,
                    "version_id": "fixture-proven-historical-version",
                    "available_at": item["publication_time"],
                }
                for item in evidence_records
                if item["role"] == "model_input"
            ],
        },
    )

    selection_path: Path = root / "selection.json"
    _write_json(
        selection_path,
        {
            "version": "fixture-selection-v1",
            "dataset": "ctibench",
            "dataset_version": FIXTURE_VERSION,
            "source_ids": ["fixture-acquisition"],
            "item_ids": ["shared-case-item", "forecast-item"],
        },
    )

    target_construction: dict[str, Any] = {
        "fixture_only": True,
        "label_source": "explicit-fictional-observed-outcome",
    }
    profile: dict[str, Any] = {
        "type": "object",
        "description": "Fictional fixture output contract, not an answer.",
    }

    def conversion(
        identity: str,
        item_id: str,
        case_id: str,
        task_name: str,
        stage: str,
        reference_id: str,
        inputs: list[str],
        observations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "instance_id": identity,
            "case_id": case_id,
            "source_dataset": "ctibench",
            "source_item_id": item_id,
            "source_id": "fixture-acquisition",
            "task_name": task_name,
            "stage": stage,
            "snapshot_date": FIXTURE_SNAPSHOT,
            "input_evidence_ids": inputs,
            "reference_id": reference_id,
            "candidates": (
                ["CVE-2099-1234", "CVE-2099-5678"]
                if task_name == "vulnerability_linking"
                else ["fictional-asset-A", "fictional-asset-B"]
                if task_name == "affected_systems"
                else []
            ),
            "environment": {"platform": "fictional-test-environment"},
            "output_profile": profile,
            "task_guidance": "Use only the fictional supplied evidence.",
            "rule_id": "fixture-explicit-conversion",
            "rule_version": "fixture-conversion-rule-v1",
            "transformation_history": [
                {"operation": "explicit-fixture-conversion", "fixture_only": True}
            ],
            "identifier_observations": observations or [],
        }

    narrative: str = rows[0]["text"]
    ioc_raw: str = "cdn-fixture[.]example"
    alias_raw: str = "AliasFixture"
    observations: list[dict[str, Any]] = [
        {
            "kind": "ioc",
            "raw": ioc_raw,
            "evidence_id": "fixture-narrative",
            "start_offset": narrative.index(ioc_raw),
            "end_offset": narrative.index(ioc_raw) + len(ioc_raw),
            "unresolved_policy": "reject",
            "ioc_type": "domain",
            "operations": [
                {"operation": "replace", "old": "[.]", "new": "."},
                {"operation": "lowercase"},
            ],
        },
        {
            "kind": "alias",
            "raw": alias_raw,
            "entity_type": "tool",
            "evidence_id": "fixture-narrative",
            "start_offset": narrative.index(alias_raw),
            "end_offset": narrative.index(alias_raw) + len(alias_raw),
            "unresolved_policy": "reject",
        },
    ]
    common_inputs: list[str] = [
        "fixture-narrative", "fixture-structured",
        "fixture-old", "fixture-correction",
    ]
    conversions_path: Path = root / "conversions.json"
    _write_json(
        conversions_path,
        {
            "version": "fixture-conversions-v1",
            "conversions": [
                conversion(
                    "fixture-affected", "shared-case-item", "fixture-shared-case",
                    "affected_systems", "CONTEXTUALIZATION",
                    "fixture-ref-affected", common_inputs, observations,
                ),
                conversion(
                    "fixture-linking", "shared-case-item", "fixture-shared-case",
                    "vulnerability_linking", "CONTEXTUALIZATION",
                    "fixture-ref-linking", common_inputs,
                ),
                conversion(
                    "fixture-forecast", "forecast-item", "fixture-forecast-case",
                    "exploit_likelihood", "PREDICTION", "fixture-ref-forecast",
                    ["fixture-forecast-input"],
                ),
            ],
        },
    )
    references_path: Path = root / "references.json"
    _write_json(
        references_path,
        {
            "version": "fixture-references-v1",
            "references": [
                {
                    "reference_id": "fixture-ref-affected",
                    "instance_id": "fixture-affected",
                    "answer": {
                        "decisions": [1, 0],
                        "withheld_marker": REFERENCE_SENTINEL,
                    },
                    "evidence_ids": ["fixture-affected-reference"],
                },
                {
                    "reference_id": "fixture-ref-linking",
                    "instance_id": "fixture-linking",
                    "answer": {"selected": "CVE-2099-1234"},
                    "evidence_ids": ["fixture-link-reference"],
                },
                {
                    "reference_id": "fixture-ref-forecast",
                    "instance_id": "fixture-forecast",
                    "answer": {"exploited": 1},
                    "evidence_ids": ["fixture-outcome"],
                    "horizon_days": 30,
                    "target_construction": target_construction,
                },
            ],
        },
    )

    payload: dict[str, Any] = copy.deepcopy(baseline_yaml)
    payload["experiment_id"] = "fixture-data-storage-test"
    payload["corpus_version"] = "fixture-local-corpus-v1"
    payload["operational"] = {
        "database_path": str(store_fixture.database_path),
        "artifact_root": str(store_fixture.artifact_root),
        "output_path": str(root / "reports"),
        "selected_benchmarks": ["ctibench"],
        "selected_tasks": [
            "affected_systems", "vulnerability_linking", "exploit_likelihood"
        ],
    }
    settings: dict[str, Any] = {
        "data.benchmarks.ctibench.version": FIXTURE_VERSION,
        "data.benchmarks.ctibench.source_path": str(source_path),
        "data.benchmarks.ctibench.selection_manifest": str(selection_path),
        "data.evidence_manifest": str(acquisition_path),
        "data.historical_snapshot_manifest": str(snapshots_path),
        "data.task_conversion_manifest": str(conversions_path),
        "data.reference_answer_manifest": str(references_path),
        "data.preprocessing.alias_table_version": FIXTURE_ALIAS_VERSION,
        "data.preprocessing.taxonomy_versions": {"fixture": "no-attack-inputs"},
        "data.preprocessing.deduplication_policy": "preserve_source_versions",
        "evaluation.tasks.prediction.exploit_likelihood.horizon_days": 30,
        "evaluation.tasks.prediction.exploit_likelihood.binary_target_construction": (
            target_construction
        ),
    }
    for path, value in settings.items():
        _resolve_fixture_setting(payload, path, value)

    config_path: Path = root / "fixture-config.yaml"
    _write_yaml(config_path, payload)
    config: ExperimentConfig = ExperimentConfig.load(str(config_path))
    loader: DatasetLoader = DatasetLoader(config, store_fixture.store)
    return BundleFixture(
        config, config_path, acquisition_path, snapshots_path,
        conversions_path, references_path, source_path, source_bytes,
        store_fixture, loader,
    )


def _build(bundle: BundleFixture) -> list[TaskInstance]:
    bundle.loader.ingest_sources(str(bundle.acquisition_path))
    return bundle.loader.build_instances()


def _changed_temporal_loader(
    bundle: BundleFixture,
    change: Callable[[dict[str, Any], dict[str, Any]], None],
) -> DatasetLoader:
    """Create an explicitly reloaded protocol after changing fixture evidence."""
    acquisition: dict[str, Any] = _read_json(bundle.acquisition_path)
    snapshots: dict[str, Any] = _read_json(bundle.snapshots_path)
    change(acquisition, snapshots)
    _write_json(bundle.acquisition_path, acquisition)
    _write_json(bundle.snapshots_path, snapshots)
    config: ExperimentConfig = ExperimentConfig.load(str(bundle.config_path))
    loader: DatasetLoader = DatasetLoader(config, bundle.store_fixture.store)
    loader.ingest_sources(str(bundle.acquisition_path))
    return loader


# ---------------------------------------------------------------------------
# Schemas and canonical content.
# ---------------------------------------------------------------------------


def test_task_round_trip_preserves_nested_scientific_content() -> None:
    original: TaskInstance = _minimal_task()
    serialized: str = _canonical_json(original)
    restored: TaskInstance = TaskInstance.model_validate_json(serialized)
    assert _canonical_json(restored) == serialized
    assert restored.content_hash == original.content_hash
    assert isinstance(restored.reference["answer"], dict)
    assert isinstance(restored.transformation_log["history"], list)
    assert restored.snapshot_date.isoformat() == FIXTURE_SNAPSHOT


def test_mapping_order_is_irrelevant_but_ordered_sequences_are_not() -> None:
    first: dict[str, Any] = {"b": {"y": 2, "x": 1}, "a": [1, 2]}
    second: dict[str, Any] = {"a": [1, 2], "b": {"x": 1, "y": 2}}
    assert _canonical_json(first) == _canonical_json(second)
    assert _content_hash(first) == _content_hash(second)
    second["a"] = [2, 1]
    assert _content_hash(first) != _content_hash(second)


def test_set_like_labels_are_canonical_and_duplicates_rejected(
    fixture_taxonomy: TaxonomyVersion,
) -> None:
    payload: dict[str, Any] = {
        "assignment_id": "fixture-assignment",
        "response_id": "fixture-response",
        "taxonomy_id": fixture_taxonomy.taxonomy_id,
        "actor_id": "fixture-annotator",
        "round_index": 1,
        "origin": "agent",
        "labels": ["fixture-B", "fixture-A"],
        "evidence_spans": [],
        "uncertain": False,
    }
    context: dict[str, Any] = {"taxonomy": fixture_taxonomy}
    first: LabelAssignment = LabelAssignment.model_validate(payload, context=context)
    payload["labels"] = ["fixture-A", "fixture-B"]
    second: LabelAssignment = LabelAssignment.model_validate(payload, context=context)
    assert first == second
    assert _content_hash(first) == _content_hash(second)

    payload["labels"] = ["fixture-A", "fixture-A"]
    with pytest.raises(ValidationError):
        LabelAssignment.model_validate(payload, context=context)
    payload["labels"] = ["not-discovered"]
    with pytest.raises(ValidationError, match="Unknown subtype"):
        LabelAssignment.model_validate(payload, context=context)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("snapshot_date", "2024-06-02"),
        ("reference", {"version": "changed", "answer": {"decisions": [0, 1]}}),
        ("candidates", ["fictional-asset-B", "fictional-asset-A"]),
        ("environment", {"platform": "another-fictional-platform"}),
        ("transformation_log", {"history": [{"operation": "changed"}]}),
    ],
)
def test_scientific_task_changes_change_content_hash(field: str, value: Any) -> None:
    original: TaskInstance = _minimal_task()
    payload: dict[str, Any] = original.model_dump(mode="json")
    payload[field] = value
    changed: TaskInstance = _rehash_task(payload)
    assert changed.content_hash != original.content_hash


def test_stale_task_content_digest_is_rejected() -> None:
    payload: dict[str, Any] = _minimal_task().model_dump(mode="json")
    payload["environment"]["platform"] = "changed-without-rehashing"
    with pytest.raises(ValidationError, match="content_hash"):
        TaskInstance.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("instance_id", ""), ("stage", "INVALID_STAGE"), ("snapshot_date", "2024-99-01")],
)
def test_malformed_task_fields_are_rejected(field: str, value: Any) -> None:
    payload: dict[str, Any] = _minimal_task().model_dump(mode="json")
    payload["content_hash"] = None
    payload[field] = value
    with pytest.raises(ValidationError):
        TaskInstance.model_validate(payload)


def test_task_stage_membership_requires_configuration_context(
    baseline_config: ExperimentConfig,
) -> None:
    payload: dict[str, Any] = _minimal_task().model_dump(mode="json")
    payload["content_hash"] = None
    payload["stage"] = "MITIGATION"
    context: dict[str, Any] = {
        "evaluation_tasks": baseline_config.evaluation.tasks
    }
    with pytest.raises(ValidationError, match="not configured"):
        TaskInstance.model_validate(payload, context=context)


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), float("inf"), "0.5"])
def test_declared_probability_fields_reject_malformed_values(probability: Any) -> None:
    task: TaskInstance = _minimal_task()
    request: dict[str, Any] = _request_payload(task)
    response: dict[str, Any] = _response_payload(task, request)
    response["parsed_answer"]["answer"] = {
        "exploit_likelihood": {"prob_exploit": probability}
    }
    with pytest.raises(ValidationError):
        ModelResponse.model_validate(response)


def test_reference_numeric_values_are_not_all_probabilities() -> None:
    payload: dict[str, Any] = _minimal_task().model_dump(mode="json")
    payload["reference"]["severity_score"] = 9.8
    task: TaskInstance = _rehash_task(payload)
    assert task.reference["severity_score"] == 9.8


# ---------------------------------------------------------------------------
# Configuration fidelity and stage-local readiness.
# ---------------------------------------------------------------------------


def test_supplied_configuration_preserves_reported_values_and_nulls(
    baseline_config: ExperimentConfig,
) -> None:
    assert baseline_config.evaluation_runs == 5
    assert baseline_config.quantile_width == 0.05
    assert baseline_config.data.task_counts_are_approximate is True
    assert baseline_config.data.task_counts_may_overlap is True
    assert baseline_config.data.preprocessing.preserve_source_provenance is True
    assert baseline_config.evaluation.prompt.supplied_evidence_only is True
    assert baseline_config.evaluation.prompt.enforce_snapshot_date is True
    assert baseline_config.reproducibility.silently_fill_unspecified_settings is False
    assert baseline_config.study.random_seed is None
    assert baseline_config.data.evidence_manifest is None
    assert baseline_config.inference.temperature is None
    assert baseline_config.inference.retry_policy is None
    assert baseline_config.evaluation.run_seeds is None
    assert baseline_config.data.preprocessing.deduplication_policy is None


def test_configuration_hash_ignores_yaml_formatting(
    baseline_yaml: dict[str, Any], tmp_path: Path
) -> None:
    first_path: Path = tmp_path / "first.yaml"
    second_path: Path = tmp_path / "second.yaml"
    _write_yaml(first_path, baseline_yaml)
    second_path.write_text(
        "# A comment changes presentation, not the experiment.\n"
        + yaml.safe_dump(baseline_yaml, sort_keys=True),
        encoding="utf-8",
    )
    first: ExperimentConfig = ExperimentConfig.load(str(first_path))
    second: ExperimentConfig = ExperimentConfig.load(str(second_path))
    assert first.config_hash == second.config_hash


def test_experimental_setting_changes_configuration_hash(
    baseline_yaml: dict[str, Any], tmp_path: Path
) -> None:
    first_path: Path = tmp_path / "first.yaml"
    second_path: Path = tmp_path / "second.yaml"
    _write_yaml(first_path, baseline_yaml)
    changed: dict[str, Any] = copy.deepcopy(baseline_yaml)
    _resolve_fixture_setting(
        changed, "inference.temperature", 0.1, stages=("evaluate",)
    )
    _write_yaml(second_path, changed)
    assert (
        ExperimentConfig.load(str(first_path)).config_hash
        != ExperimentConfig.load(str(second_path)).config_hash
    )


def test_unresolved_preparation_fails_actionably(
    baseline_config: ExperimentConfig,
) -> None:
    with pytest.raises(ValueError) as captured:
        baseline_config.validate("prepare")
    message: str = str(captured.value)
    assert "prepare" in message
    assert "data.evidence_manifest" in message
    assert "unresolved" in message


def test_resolved_preparation_does_not_require_counterfactual_settings(
    local_bundle: BundleFixture,
) -> None:
    local_bundle.config.validate("prepare")
    assert (
        local_bundle.config.defenses.counterfactual_consistency.stability_threshold_tau
        is None
    )
    assert local_bundle.config.inference.temperature is None


def test_referenced_file_change_is_detected_after_configuration_load(
    local_bundle: BundleFixture,
) -> None:
    original: bytes = local_bundle.references_path.read_bytes()
    local_bundle.references_path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="changed"):
        local_bundle.config.validate("prepare")


# ---------------------------------------------------------------------------
# Local provenance-aware dataset preparation.
# ---------------------------------------------------------------------------


def test_local_ingestion_preserves_provenance_normalization_and_conflicts(
    local_bundle: BundleFixture,
) -> None:
    instances: list[TaskInstance] = _build(local_bundle)
    assert len(instances) == 3
    affected: TaskInstance = next(
        item for item in instances if item.task_name == "affected_systems"
    )
    store: ArtifactStore = local_bundle.store_fixture.store
    source_record: dict[str, Any] = store.read_record(
        "source", affected.transformation_log["source_record_id"]
    )
    assert source_record["original_item_id"] == "shared-case-item"
    assert source_record["dataset"] == "ctibench"
    assert source_record["original"]["structured"]["mechanism"] == "fictional-mechanism-A"
    assert "cdn-fixture[.]example" in source_record["original"]["text"]
    assert source_record["original"]["old_claim"] != source_record["original"]["correction"]

    normalizations: list[dict[str, Any]] = affected.transformation_log["normalization"]
    canonical_values: set[str] = {item["canonical"] for item in normalizations}
    assert "cdn-fixture.example" in canonical_values
    assert "CanonicalFixtureTool" in canonical_values
    assert "cdn-fixture[.]example" in affected.evidence_records[0].snippet
    assert affected.transformation_log["rule_version"] == "fixture-conversion-rule-v1"
    assert affected.transformation_log["history"]

    evidence_ids: set[str] = {item.evidence_id for item in affected.evidence_records}
    assert {"fixture-old", "fixture-correction", "fixture-structured"} <= evidence_ids
    findings: list[dict[str, str]] = local_bundle.loader.validate_instance(affected)
    assert not [item for item in findings if item["severity"] == "error"]
    assert any(item["code"] == "declared_conflict" for item in findings)

    raw_digest: str = source_record["raw_blob_hash"]
    raw_path: Path = (
        local_bundle.store_fixture.artifact_root / raw_digest[:2] / raw_digest[2:]
    )
    assert raw_path.read_bytes() == local_bundle.source_bytes


def test_input_reference_separation_and_legitimate_candidates(
    local_bundle: BundleFixture,
) -> None:
    instances: list[TaskInstance] = _build(local_bundle)
    affected: TaskInstance = next(
        item for item in instances if item.instance_id == "fixture-affected"
    )
    linking: TaskInstance = next(
        item for item in instances if item.instance_id == "fixture-linking"
    )
    forecast: TaskInstance = next(
        item for item in instances if item.instance_id == "fixture-forecast"
    )
    assert REFERENCE_SENTINEL in _canonical_json(affected.reference)
    assert OUTCOME_SENTINEL in _canonical_json(forecast.reference)

    for instance in instances:
        visible: str = _canonical_json(
            {
                "evidence": instance.evidence_records,
                "candidates": instance.candidates,
                "environment": instance.environment,
            }
        )
        assert REFERENCE_SENTINEL not in visible
        assert OUTCOME_SENTINEL not in visible
        assert all(item.role.value == "model_input" for item in instance.evidence_records)

    # A correct candidate may be offered; an answer-selection field may not.
    assert linking.reference["answer"]["selected"] in linking.candidates
    assert all(isinstance(candidate, str) for candidate in linking.candidates)
    outcome: dict[str, Any] = forecast.reference["evidence_records"][0]
    assert outcome["role"] == "outcome_reference"
    assert outcome["publication_time"] > FIXTURE_SNAPSHOT
    assert not [
        item for item in local_bundle.loader.validate_instance(forecast)
        if item["severity"] == "error"
    ]


def test_shared_cases_are_not_deduplicated_across_tasks(
    local_bundle: BundleFixture,
) -> None:
    instances: list[TaskInstance] = _build(local_bundle)
    shared: list[TaskInstance] = [
        item for item in instances if item.case_id == "fixture-shared-case"
    ]
    assert len(shared) == 2
    assert len({item.instance_id for item in shared}) == 2
    assert len({item.task_name for item in shared}) == 2
    assert len({item.reference["reference_id"] for item in shared}) == 2
    assert len({item.source_item_id for item in shared}) == 1

    reports: list[dict[str, Any]] = local_bundle.store_fixture.store.list_records("report")
    count_report: dict[str, Any] = next(
        item for item in reports if "task_assignments" in item
    )
    assert count_report["task_assignments"] == 3
    assert count_report["unique_cases"] == 2
    assert count_report["approximate_count_discrepancies"]
    assert count_report["counts_are_selection_quotas"] is False


def test_late_retrieval_does_not_replace_historical_availability(
    local_bundle: BundleFixture,
) -> None:
    instances: list[TaskInstance] = _build(local_bundle)
    for instance in instances:
        assert all(
            item.retrieval_time is not None
            and item.retrieval_time.date() > instance.snapshot_date
            for item in instance.evidence_records
        )
        assert not [
            item for item in local_bundle.loader.validate_instance(instance)
            if item["severity"] == "error"
        ]


@pytest.mark.parametrize(
    ("violation", "expected_message"),
    [
        ("future_publication", "published after"),
        ("future_modification", "Post-snapshot modification"),
        ("future_availability", "unavailable at the snapshot"),
        ("missing_availability", "available_at"),
        ("missing_publication", "publication metadata"),
    ],
)
def test_invalid_temporal_provenance_is_reported_and_blocks_construction(
    local_bundle: BundleFixture,
    violation: str,
    expected_message: str,
) -> None:
    valid_instances: list[TaskInstance] = _build(local_bundle)
    original: TaskInstance = next(
        item for item in valid_instances if item.instance_id == "fixture-affected"
    )

    def change(acquisition: dict[str, Any], snapshots: dict[str, Any]) -> None:
        record: dict[str, Any] = next(
            item for item in acquisition["evidence_records"]
            if item["evidence_id"] == "fixture-narrative"
        )
        proof: dict[str, Any] = next(
            item for item in snapshots["records"]
            if item["evidence_id"] == "fixture-narrative"
        )
        if violation == "future_publication":
            record["publication_time"] = "2024-06-10"
        elif violation == "future_modification":
            record["modification_time"] = "2024-06-10T00:00:00Z"
        elif violation == "future_availability":
            proof["available_at"] = "2024-06-10"
        elif violation == "missing_availability":
            proof.pop("available_at")
        elif violation == "missing_publication":
            record["publication_time"] = None
        else:
            raise AssertionError("Unexpected parameterized fixture violation.")

    invalid_loader: DatasetLoader = _changed_temporal_loader(local_bundle, change)
    findings: list[dict[str, str]] = invalid_loader.validate_instance(original)
    errors: list[dict[str, str]] = [
        item for item in findings if item["severity"] == "error"
    ]
    assert errors
    assert any(expected_message in item["message"] for item in errors)
    with pytest.raises(ValueError):
        invalid_loader.build_instances()
    assert local_bundle.store_fixture.store.list_records("corpus_manifest") == []


def test_reference_role_cannot_be_declared_as_input(
    local_bundle: BundleFixture,
) -> None:
    conversions: dict[str, Any] = _read_json(local_bundle.conversions_path)
    conversions["conversions"][0]["input_evidence_ids"].append(
        "fixture-affected-reference"
    )
    _write_json(local_bundle.conversions_path, conversions)
    config: ExperimentConfig = ExperimentConfig.load(str(local_bundle.config_path))
    loader: DatasetLoader = DatasetLoader(config, local_bundle.store_fixture.store)
    loader.ingest_sources(str(local_bundle.acquisition_path))
    with pytest.raises(ValueError, match="Reference-only evidence"):
        loader.build_instances()


def test_freezing_and_reopening_preserve_exact_corpus_membership(
    local_bundle: BundleFixture,
) -> None:
    built: list[TaskInstance] = _build(local_bundle)
    digest: str = local_bundle.loader.freeze_corpus()
    assert len(digest) == 64
    assert local_bundle.loader.freeze_corpus() == digest

    reopened_store: ArtifactStore = ArtifactStore(
        str(local_bundle.store_fixture.database_path),
        str(local_bundle.store_fixture.artifact_root),
    )
    reopened_store.initialize()
    reopened_loader: DatasetLoader = DatasetLoader(local_bundle.config, reopened_store)
    loaded: list[TaskInstance] = reopened_loader.load_instances()
    assert [item.instance_id for item in loaded] == [item.instance_id for item in built]
    assert [item.content_hash for item in loaded] == [item.content_hash for item in built]
    assert len(reopened_loader.load_instances({"case_id": "fixture-shared-case"})) == 2

    with pytest.raises(PermissionError):
        reopened_store.update_draft(
            "task_instance", loaded[0].instance_id,
            {"environment": {"platform": "silently-changed"}},
        )
    assert reopened_loader.load_instances() == loaded


# ---------------------------------------------------------------------------
# Drafts, immutable records, byte artifacts, and append-only history.
# ---------------------------------------------------------------------------


def test_draft_crud_preserves_unpatched_fields(store_fixture: StoreFixture) -> None:
    store: ArtifactStore = store_fixture.store
    payload: dict[str, Any] = {
        "config_id": "fixture-draft",
        "description": "original",
        "nested": {"unchanged": [1, 2]},
    }
    store.create_record("configuration", "fixture-draft", payload)
    store.update_draft("configuration", "fixture-draft", {"description": "edited"})
    assert store.read_record("configuration", "fixture-draft") == {
        **payload, "description": "edited"
    }
    store.delete_draft("configuration", "fixture-draft")
    with pytest.raises(KeyError):
        store.read_record("configuration", "fixture-draft")
    assert store.list_records("configuration") == []


def test_freeze_is_order_independent_and_prevents_mutation(
    store_fixture: StoreFixture,
) -> None:
    store: ArtifactStore = store_fixture.store
    for identity in ("fixture-first", "fixture-second"):
        store.create_record("configuration", identity, {"config_id": identity, "value": 1})
    digest: str = store.freeze_records(
        "configuration", ["fixture-first", "fixture-second"]
    )
    assert digest == store.freeze_records(
        "configuration", ["fixture-second", "fixture-first"]
    )
    with pytest.raises(PermissionError):
        store.update_draft("configuration", "fixture-first", {"value": 2})
    with pytest.raises(PermissionError):
        store.delete_draft("configuration", "fixture-first")
    assert store.read_record("configuration", "fixture-first")["value"] == 1


def test_submitted_annotations_are_append_only(
    store_fixture: StoreFixture, stored_task: TaskInstance
) -> None:
    store: ArtifactStore = store_fixture.store
    request: dict[str, Any] = _request_payload(stored_task)
    store.create_record("request", request["request_hash"], request)
    response: dict[str, Any] = _response_payload(stored_task, request)
    store.create_record("model_response", response["response_id"], response)
    annotation: dict[str, Any] = {
        "annotation_id": "fixture-submitted-annotation",
        "response_id": response["response_id"],
        "purpose": "identification_anchor",
        "annotator_id": "fixture-human",
        "guideline_hash": "b" * 64,
        "status": "submitted",
        "decision": "failure",
    }
    store.create_record("annotation", annotation["annotation_id"], annotation)
    with pytest.raises(PermissionError):
        store.update_draft("annotation", annotation["annotation_id"], {"decision": "pass"})
    with pytest.raises(PermissionError):
        store.delete_draft("annotation", annotation["annotation_id"])
    assert store.read_record("annotation", annotation["annotation_id"]) == annotation


def test_corrections_preserve_original_assignment_history(
    store_fixture: StoreFixture,
    stored_task: TaskInstance,
    fixture_taxonomy: TaxonomyVersion,
) -> None:
    store: ArtifactStore = store_fixture.store
    request: dict[str, Any] = _request_payload(stored_task)
    store.create_record("request", request["request_hash"], request)
    response: dict[str, Any] = _response_payload(stored_task, request)
    store.create_record("model_response", response["response_id"], response)
    store.create_record(
        "taxonomy", fixture_taxonomy.taxonomy_id,
        fixture_taxonomy.model_dump(mode="json"),
    )
    first: dict[str, Any] = {
        "assignment_id": "fixture-original-assignment",
        "response_id": response["response_id"],
        "taxonomy_id": fixture_taxonomy.taxonomy_id,
        "actor_id": "fixture-human",
        "round_index": 0,
        "origin": "independent_annotation",
        "labels": ["fixture-A"],
        "evidence_spans": [],
        "uncertain": False,
        "supersedes_id": None,
    }
    store.create_record("label_assignment", first["assignment_id"], first)
    store.freeze_records("label_assignment", [first["assignment_id"]])
    corrected: dict[str, Any] = {
        **first,
        "assignment_id": "fixture-corrected-assignment",
        "labels": ["fixture-B"],
        "supersedes_id": first["assignment_id"],
    }
    store.create_record("label_assignment", corrected["assignment_id"], corrected)
    assert store.read_record("label_assignment", first["assignment_id"]) == first
    assert store.read_record("label_assignment", corrected["assignment_id"]) == corrected
    assert len(store.list_records("label_assignment")) == 2


def test_identical_insert_is_idempotent_but_conflicting_insert_is_rejected(
    store_fixture: StoreFixture,
) -> None:
    store: ArtifactStore = store_fixture.store
    original: dict[str, Any] = {"config_id": "fixture-identity", "value": 1}
    store.create_record("configuration", "fixture-identity", original)
    store.create_record("configuration", "fixture-identity", original)
    with pytest.raises(sqlite3.IntegrityError):
        store.create_record(
            "configuration", "fixture-identity", {**original, "value": 2}
        )
    assert store.read_record("configuration", "fixture-identity") == original
    assert len(store.list_records("configuration")) == 1


def test_blob_identity_is_exact_bytes_and_survives_restart(
    store_fixture: StoreFixture,
) -> None:
    content: bytes = b"fictional fixture\r\n"
    digest: str = store_fixture.store.save_blob(content)
    assert digest == hashlib.sha256(content).hexdigest()
    assert store_fixture.store.save_blob(content) == digest
    assert store_fixture.store.save_blob(b"fictional fixture\n") != digest

    reopened: ArtifactStore = ArtifactStore(
        str(store_fixture.database_path), str(store_fixture.artifact_root)
    )
    reopened.initialize()
    assert reopened.save_blob(content) == digest
    blob_path: Path = store_fixture.artifact_root / digest[:2] / digest[2:]
    assert blob_path.read_bytes() == content
    assert len([
        path for path in store_fixture.artifact_root.rglob("*") if path.is_file()
    ]) == 2


# ---------------------------------------------------------------------------
# Transactions, concurrency, checkpoints, and deterministic exports.
# ---------------------------------------------------------------------------


def test_initialization_is_repeatable_and_sqlite_integrity_is_valid(
    store_fixture: StoreFixture,
) -> None:
    store: ArtifactStore = store_fixture.store
    store.create_record("configuration", "fixture-survivor", {"config_id": "fixture-survivor"})
    blob: str = store.save_blob(b"fixture-surviving-blob")
    store.initialize()
    assert store.read_record("configuration", "fixture-survivor")
    assert store.save_blob(b"fixture-surviving-blob") == blob

    with sqlite3.connect(
        f"{store_fixture.database_path.as_uri()}?mode=ro", uri=True
    ) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_rejected_response_write_rolls_back_record_and_cache(
    store_fixture: StoreFixture, stored_task: TaskInstance
) -> None:
    store: ArtifactStore = store_fixture.store
    request: dict[str, Any] = _request_payload(stored_task)
    store.create_record("request", request["request_hash"], request)
    response: dict[str, Any] = _response_payload(stored_task, request)
    response["run_index"] = 1  # Deliberately disagrees with the bound request.
    with pytest.raises(ValueError, match="disagree on run_index"):
        store.create_record("model_response", response["response_id"], response)
    with pytest.raises(KeyError):
        store.read_record("model_response", response["response_id"])
    assert store.get_cached_response(request["request_hash"]) is None
    assert store.list_records("model_response") == []


def test_missing_instance_foreign_reference_does_not_leave_partial_response(
    store_fixture: StoreFixture,
) -> None:
    task: TaskInstance = _minimal_task()
    request: dict[str, Any] = _request_payload(task)
    response: dict[str, Any] = _response_payload(task, request)
    # No request_hash link is needed to exercise the required instance link.
    response["usage"].pop("request_hash")
    with pytest.raises(KeyError):
        store_fixture.store.create_record(
            "model_response", response["response_id"], response
        )
    assert store_fixture.store.list_records("model_response") == []


def test_concurrent_conflicting_completions_have_one_authoritative_winner(
    store_fixture: StoreFixture, stored_task: TaskInstance
) -> None:
    request: dict[str, Any] = _request_payload(stored_task)
    store_fixture.store.create_record("request", request["request_hash"], request)
    barrier: threading.Barrier = threading.Barrier(2)

    def compete(index: int) -> tuple[bool, str]:
        worker: ArtifactStore = ArtifactStore(
            str(store_fixture.database_path), str(store_fixture.artifact_root)
        )
        worker.initialize()
        identity: str = f"fixture-competing-response-{index}"
        payload: dict[str, Any] = _response_payload(stored_task, request, identity)
        barrier.wait(timeout=10)
        try:
            worker.create_record("model_response", identity, payload)
        except sqlite3.IntegrityError:
            return False, identity
        return True, identity

    with ThreadPoolExecutor(max_workers=2) as executor:
        results: list[tuple[bool, str]] = list(executor.map(compete, [0, 1]))

    winners: list[str] = [identity for success, identity in results if success]
    assert len(winners) == 1
    cached: ModelResponse | None = store_fixture.store.get_cached_response(
        request["request_hash"]
    )
    assert cached is not None
    assert cached.response_id == winners[0]
    assert len(store_fixture.store.list_records("model_response")) == 1
    with sqlite3.connect(
        f"{store_fixture.database_path.as_uri()}?mode=ro", uri=True
    ) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_checkpoints_survive_restart_and_cannot_discard_completed_work(
    store_fixture: StoreFixture,
) -> None:
    start: datetime = datetime(2024, 6, 2, 10, 0, tzinfo=timezone.utc)
    first: StageState = StageState(
        stage="fixture-evaluate",
        execution_id="fixture-execution",
        config_hash="c" * 64,
        status=StageStatus.RUNNING,
        completed_work=["job-1"],
        pending_work=["job-2"],
        cursor={"next_job": "job-2"},
        input_manifests={"corpus": "fixture-frozen-corpus"},
        started_at=start,
        updated_at=start,
    )
    store_fixture.store.set_stage_state(first.stage, first.model_dump(mode="json"))

    reopened: ArtifactStore = ArtifactStore(
        str(store_fixture.database_path), str(store_fixture.artifact_root)
    )
    reopened.initialize()
    states: list[dict[str, Any]] = reopened.list_records(
        "stage_state", {"config_hash": "c" * 64}
    )
    assert states == [first.model_dump(mode="json")]

    bad: dict[str, Any] = first.model_dump(mode="json")
    bad["completed_work"] = []
    bad["pending_work"] = ["job-1", "job-2"]
    with pytest.raises(ValueError, match="discard completed work"):
        reopened.set_stage_state(first.stage, bad)

    final_payload: dict[str, Any] = first.model_dump(mode="json")
    final_payload.update(
        status="completed",
        completed_work=["job-1", "job-2"],
        pending_work=[],
        cursor={"next_job": None},
        updated_at="2024-06-02T10:01:00Z",
        completed_at="2024-06-02T10:01:00Z",
    )
    reopened.set_stage_state(first.stage, final_payload)
    assert len(reopened.list_records("stage_state")) == 2

    with sqlite3.connect(
        f"{store_fixture.database_path.as_uri()}?mode=ro", uri=True
    ) as connection:
        row: tuple[Any, ...] | None = connection.execute(
            "SELECT state_id FROM stage_checkpoints WHERE stage=?",
            (first.stage,),
        ).fetchone()
    assert row is not None
    assert reopened.read_record("stage_state", row[0]) == final_payload


def test_exports_are_repeatable_and_do_not_overwrite_conflicting_content(
    store_fixture: StoreFixture, tmp_path: Path
) -> None:
    store: ArtifactStore = store_fixture.store
    store.create_record("configuration", "fixture-export", {"config_id": "fixture-export"})
    destination: Path = tmp_path / "export" / "configurations.jsonl"
    store.export_records("configuration", str(destination))
    original: bytes = destination.read_bytes()
    store.export_records("configuration", str(destination))
    assert destination.read_bytes() == original

    entry: dict[str, Any] = json.loads(original.decode("utf-8"))
    assert entry["record_id"] == "fixture-export"
    assert entry["payload"]["config_id"] == "fixture-export"
    assert entry["content_hash"] == _content_hash(entry["payload"])

    store.create_record("configuration", "another-export", {"config_id": "another-export"})
    with pytest.raises(FileExistsError):
        store.export_records("configuration", str(destination))
    assert destination.read_bytes() == original


# ---------------------------------------------------------------------------
# Cache identities and retained content outcomes.
# ---------------------------------------------------------------------------


def test_all_configured_evaluation_runs_have_distinct_request_identities(
    baseline_config: ExperimentConfig, stored_task: TaskInstance
) -> None:
    requests: list[dict[str, Any]] = [
        _request_payload(stored_task, run_index=index)
        for index in range(baseline_config.evaluation_runs)
    ]
    assert len(requests) == 5
    assert len({item["request_hash"] for item in requests}) == len(requests)
    assert len({item["prompt_hash"] for item in requests}) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"endpoint": "fixture/another-endpoint"},
        {"revision": "fixture-checkpoint-v2"},
        {"prompt": "A changed fictional offline prompt."},
        {"temperature": 0.1},
        {"schema_version": "fixture-output-schema-v2"},
        {"run_index": 1},
    ],
)
def test_cache_identity_changes_for_request_defining_fields(
    stored_task: TaskInstance, change: dict[str, Any]
) -> None:
    original: dict[str, Any] = _request_payload(stored_task)
    changed: dict[str, Any] = _request_payload(stored_task, **change)
    assert changed["request_hash"] != original["request_hash"]


def test_cached_response_retains_original_artifact_across_restart(
    store_fixture: StoreFixture, stored_task: TaskInstance
) -> None:
    request: dict[str, Any] = _request_payload(stored_task)
    store: ArtifactStore = store_fixture.store
    store.create_record("request", request["request_hash"], request)
    payload: dict[str, Any] = _response_payload(stored_task, request)
    store.create_record("model_response", payload["response_id"], payload)

    reopened: ArtifactStore = ArtifactStore(
        str(store_fixture.database_path), str(store_fixture.artifact_root)
    )
    reopened.initialize()
    cached: ModelResponse | None = reopened.get_cached_response(request["request_hash"])
    assert cached is not None
    assert cached.model_dump(mode="json") == payload
    assert cached.run_index == 0
    assert cached.raw_text == payload["raw_text"]
    assert cached.prompt_hash == request["prompt_hash"]
    assert cached.config_hash == request["config_hash"]
    assert cached.schema_errors == []


@pytest.mark.parametrize(
    "status",
    [ResponseStatus.SUCCESS, ResponseStatus.INVALID_SCHEMA, ResponseStatus.REFUSAL],
)
def test_content_outcomes_are_retained_and_cacheable_by_storage_contract(
    store_fixture: StoreFixture,
    stored_task: TaskInstance,
    status: ResponseStatus,
) -> None:
    store: ArtifactStore = store_fixture.store
    request: dict[str, Any] = _request_payload(stored_task)
    store.create_record("request", request["request_hash"], request)
    payload: dict[str, Any] = _response_payload(stored_task, request, status=status)
    store.create_record("model_response", payload["response_id"], payload)
    cached: ModelResponse | None = store.get_cached_response(request["request_hash"])
    assert cached is not None
    assert cached.status == status
    assert cached.raw_text == payload["raw_text"]
    assert cached.schema_errors == payload["schema_errors"]


def test_valid_insufficient_context_is_not_a_transport_failure(
    store_fixture: StoreFixture, stored_task: TaskInstance
) -> None:
    store: ArtifactStore = store_fixture.store
    request: dict[str, Any] = _request_payload(stored_task)
    store.create_record("request", request["request_hash"], request)
    payload: dict[str, Any] = _response_payload(stored_task, request)
    payload["parsed_answer"] = {
        "status": "NEED_MORE_CONTEXT",
        "answer": {},
        "metadata": {"missing_fields": ["fictional-asset-version"]},
    }
    payload["raw_text"] = _canonical_json(payload["parsed_answer"])
    response: ModelResponse = ModelResponse.model_validate(
        payload,
        context={
            "allowed_answer_statuses": ["OK", "NEED_MORE_CONTEXT", "UNSUPPORTED"]
        },
    )
    store.create_record(
        "model_response", response.response_id, response.model_dump(mode="json")
    )
    cached: ModelResponse | None = store.get_cached_response(request["request_hash"])
    assert cached is not None
    assert cached.status == ResponseStatus.SUCCESS
    assert cached.parsed_answer["status"] == "NEED_MORE_CONTEXT"


def test_transport_failure_attempt_is_retained_but_not_cached(
    store_fixture: StoreFixture, stored_task: TaskInstance
) -> None:
    store: ArtifactStore = store_fixture.store
    request: dict[str, Any] = _request_payload(stored_task)
    store.create_record("request", request["request_hash"], request)
    payload: dict[str, Any] = _response_payload(
        stored_task, request, status=ResponseStatus.TRANSPORT_FAILURE
    )
    store.create_record("model_response", payload["response_id"], payload)
    attempt: dict[str, Any] = {
        "attempt_id": "fixture-failed-attempt",
        "request_hash": request["request_hash"],
        "outcome": "transport_failure",
        "started_at": "2024-06-02T10:00:00Z",
        "completed_at": "2024-06-02T10:00:01Z",
        "response_id": payload["response_id"],
        "error": {"exception_class": "FixtureTransportError"},
        "metadata": {"fixture_only": True},
    }
    store.create_record("request_attempt", attempt["attempt_id"], attempt)
    assert store.get_cached_response(request["request_hash"]) is None
    assert store.read_record("request_attempt", attempt["attempt_id"]) == attempt
    assert store.read_record("model_response", payload["response_id"]) == payload


def test_request_hash_must_cover_the_complete_declared_request(
    store_fixture: StoreFixture, stored_task: TaskInstance
) -> None:
    request: dict[str, Any] = _request_payload(stored_task)
    request["revision"] = "changed-without-rehashing"
    with pytest.raises(ValueError, match="Request hash"):
        store_fixture.store.create_record("request", request["request_hash"], request)
    assert store_fixture.store.list_records("request") == []
