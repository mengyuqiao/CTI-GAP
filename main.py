"""CLI orchestration for the CTI failure-mode reproduction pipeline.

All scientific settings come from the explicitly selected configuration.
Missing settings remain unresolved; this entry point never supplies experimental
defaults, runs optional experiments implicitly, or tunes procedures using audits.

Runtime declarations are those supported by config.py's ``operational`` section.
Prerequisite manifests may identify stored manifests/records or local JSON/YAML
files. Bare hashes must resolve to a stored record or freeze manifest; this
module does not read ArtifactStore's private blob directory.

Audit sampling declarations
===========================

``operational.prerequisite_manifests.audit_sampling`` identifies a versioned
manifest with an ``audits`` mapping keyed by the configured selected_audit:

    {
      "version": "explicit-version",
      "audits": {
        "failure_identification": {
          "sampling": { ... AnnotationService.sample_audit arguments ... }
        },
        "human_agreement": {
          "sampling": { ... }
        },
        "failure_categorization": {
          "sampling": { ... }
        }
      }
    }

A declaration may additionally specify ``calibration_response_ids``. Otherwise,
the calibration population must be unambiguous among stored calibration tasks.
Sampling, annotation assignments, applicability notes, seeds, and prediction
bindings must already be explicitly resolved.

Independent agreement is scored before disagreement adjudication. Expanded
categorization audits preserve the original agreement cohort and create new
independent annotation tasks only for the additional cohort members.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from annotation_app import AnnotationService
from categorization import Categorizer
from config import ExperimentConfig
from dataset_loader import DatasetLoader
from evaluation import Evaluator
from experiments import ExperimentRunner
from failure_identification import FailureIdentifier
from model_client import ModelClient
from schemas import (
    AnnotationStatus,
    AnnotationTask,
    AuditCohort,
    FailureDecision,
    FailureStatus,
    ModelResponse,
    StageState,
    StageStatus,
    TaskInstance,
    TaxonomyVersion,
    _canonical_json,
    _content_hash,
    _identifier,
)
from storage import ArtifactStore
from taxonomy import TaxonomyManager


_LOGGER = logging.getLogger(__name__)

_STAGE_ALIASES: dict[str, str] = {
    "prepare": "prepare",
    "evaluate": "evaluate",
    "identify": "identify",
    "discover-taxonomy": "discover_taxonomy",
    "discover_taxonomy": "discover_taxonomy",
    "categorize": "categorize",
    "audit": "audit",
    "baselines": "baselines",
    "ablations": "ablations",
    "stability": "stability",
    "defenses": "defenses",
    "report": "report",
}

_EXPORT_KINDS: tuple[str, ...] = (
    "configuration",
    "config_manifest",
    "source",
    "evidence",
    "evidence_record",
    "task_instance",
    "reference",
    "prompt",
    "request",
    "inference_request",
    "request_attempt",
    "model_response",
    "score",
    "scoring",
    "stratum",
    "range",
    "failure_decision",
    "taxonomy",
    "taxonomy_version",
    "label_assignment",
    "assignment",
    "consensus",
    "final_assignment",
    "annotation_task",
    "annotation",
    "adjudication",
    "audit_cohort",
    "metric",
    "metric_result",
    "experiment_manifest",
    "corpus_manifest",
    "reference_manifest",
    "sampling_manifest",
    "discovery_review",
    "counterfactual",
    "stage_state",
    "report",
)

_MANIFEST_KINDS: tuple[str, ...] = (
    "experiment_manifest",
    "corpus_manifest",
    "reference_manifest",
    "sampling_manifest",
    "config_manifest",
    "scoring",
    "report",
    "audit_cohort",
)

_SECRET_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "passwd",
        "access_token",
        "refresh_token",
        "auth_token",
        "client_secret",
        "private_key",
        "credentials",
        "bearer_token",
    }
)


class _UniqueLoader(yaml.SafeLoader):
    """Safe loader rejecting ambiguous manifest keys."""


_UniqueLoader.yaml_implicit_resolvers = {
    key: [
        (tag, expression)
        for tag, expression in values
        if tag != "tag:yaml.org,2002:timestamp"
    ]
    for key, values in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _yaml_mapping(
    loader: _UniqueLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise ValueError("Manifest merge keys are not permitted.")
        key: Any = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError("Manifest keys must be unique strings.")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping
)


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Manifest contains duplicate JSON keys.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON number is forbidden: {value}.")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _mapping(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an explicitly supplied mapping.")
    return copy.deepcopy(value)


def _ids(value: Any, description: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{description} must be an explicit identifier list.")
    result: list[str] = [_identifier(item) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{description} contains duplicate identifiers.")
    if not result and not allow_empty:
        raise ValueError(f"{description} must not be empty.")
    return sorted(result)


def _check_secrets(value: Any) -> None:
    """Reject structured credentials without rewriting evidence or model prose."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized: str = str(key).lower().replace("-", "_")
            if normalized in _SECRET_KEYS and item is not None:
                raise ValueError("Credentials cannot enter configuration provenance.")
            _check_secrets(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_secrets(item)


class Main:
    """Initialize services once and execute explicitly requested pipeline stages."""

    def __init__(self, config_path: str) -> None:
        """Load configuration and open explicitly configured runtime storage."""
        self._config: ExperimentConfig = ExperimentConfig.load(config_path)
        self._config_path: Path = Path(config_path).expanduser().resolve()
        self._base: Path = self._config_path.parent

        missing: list[str] = []
        if self._config.experiment_id is None:
            missing.append("experiment_id")
        if self._config.operational.database_path is None:
            missing.append("operational.database_path")
        if self._config.operational.artifact_root is None:
            missing.append("operational.artifact_root")
        if missing:
            raise ValueError(
                "Runtime initialization is unresolved. Supply explicit values for: "
                + ", ".join(missing)
                + ". No storage location or experiment identity was selected."
            )

        database: Path = self._path(
            self._config.operational.database_path, "operational.database_path"
        )
        artifacts: Path = self._path(
            self._config.operational.artifact_root, "operational.artifact_root"
        )
        self._store: ArtifactStore = ArtifactStore(str(database), str(artifacts))
        self._store.initialize()

        self._loader: DatasetLoader = DatasetLoader(self._config, self._store)
        self._models: ModelClient = ModelClient(self._config, self._store)
        self._evaluator: Evaluator = Evaluator(self._config, self._store)
        self._annotations: AnnotationService = AnnotationService(self._store)
        self._identifier: FailureIdentifier = FailureIdentifier(
            self._config, self._store, self._annotations
        )
        self._taxonomy_manager: TaxonomyManager = TaxonomyManager(
            self._config, self._store, self._models, self._annotations
        )
        self._categorizer: Categorizer = Categorizer(
            self._config, self._store, self._models, self._annotations
        )
        self._runner: ExperimentRunner = ExperimentRunner(
            self._config,
            self._store,
            self._models,
            self._evaluator,
            self._annotations,
        )

        configuration: dict[str, Any] = self._config.model_dump(mode="json")
        _check_secrets(configuration)
        self._configuration_id: str = self._config.config_hash
        record: dict[str, Any] = {
            "config_id": self._configuration_id,
            "experiment_id": self._config.experiment_id,
            "config_hash": self._config.config_hash,
            "configuration": configuration,
            "implementation_choices": configuration["implementation_choices"],
        }
        self._store.create_record("configuration", self._configuration_id, record)
        self._store.freeze_records("configuration", [self._configuration_id])

    def _path(self, value: str | None, description: str) -> Path:
        if value is None or not value.strip():
            raise ValueError(f"{description} is unresolved.")
        if "://" in value or value.startswith(("sha256:", "urn:")):
            raise ValueError(f"{description} must identify an explicit local path.")
        path: Path = Path(value).expanduser()
        return (path if path.is_absolute() else self._base / path).resolve()

    @staticmethod
    def _stage(stage: str) -> str:
        normalized: str = stage.strip().lower()
        if normalized not in _STAGE_ALIASES:
            raise ValueError(f"Unknown pipeline stage: {stage!r}.")
        return _STAGE_ALIASES[normalized]

    def _selection(self, stage: str) -> dict[str, Any]:
        operational = self._config.operational
        return {
            "baseline": operational.selected_baseline if stage == "baselines" else None,
            "ablation": operational.selected_ablation if stage == "ablations" else None,
            "defense": operational.selected_defense if stage == "defenses" else None,
            "mitigation_mode": (
                operational.selected_mitigation_mode if stage == "defenses" else None
            ),
            "audit": operational.selected_audit if stage == "audit" else None,
        }

    def _inputs(self, stage: str) -> dict[str, str]:
        """Bind declared prerequisites and available local contents for resumption."""
        manifests: dict[str, str] = {
            "configuration": self._configuration_id,
            "selection": _content_hash(self._selection(stage)),
        }
        for name, declaration in sorted(
            self._config.operational.prerequisite_manifests.items()
        ):
            value: dict[str, Any] = {"declaration": declaration}
            if "://" not in declaration and not declaration.startswith(
                ("sha256:", "urn:")
            ):
                local: Path = Path(declaration)
                local = local if local.is_absolute() else self._base / local
                if local.is_file():
                    value["content_sha256"] = hashlib.sha256(
                        local.read_bytes()
                    ).hexdigest()
            manifests[f"prerequisite:{name}"] = _content_hash(value)
        return manifests

    def _execution_id(self, stage: str, inputs: dict[str, str]) -> str:
        return _content_hash(
            {
                "orchestrator": "main-v1",
                "experiment_id": self._config.experiment_id,
                "stage": stage,
                "config_hash": self._config.config_hash,
                "inputs": inputs,
                "selection": self._selection(stage),
            }
        )

    def _states(self, stage: str) -> list[StageState]:
        result: list[StageState] = [
            StageState.model_validate(item)
            for item in self._store.list_records("stage_state")
            if item.get("stage") == f"main:{stage}"
            and item.get("metadata", {}).get("experiment_id")
            == self._config.experiment_id
            and item.get("metadata", {}).get("selection") == self._selection(stage)
        ]
        return sorted(
            result,
            key=lambda state: (
                state.updated_at
                or state.started_at
                or datetime.min.replace(tzinfo=timezone.utc),
                state.execution_id,
            ),
        )

    def _checkpoint(
        self,
        stage: str,
        inputs: dict[str, str],
        status: StageStatus,
        result: dict[str, Any],
        *,
        pending_tasks: list[str] | None = None,
        error_class: str | None = None,
    ) -> None:
        execution_id: str = self._execution_id(stage, inputs)
        previous: list[StageState] = [
            item for item in self._states(stage)
            if item.execution_id == execution_id
        ]
        timestamp: datetime = _now()
        completed_work: list[str] = []
        if previous:
            completed_work = previous[-1].completed_work
        if status == StageStatus.COMPLETED:
            completed_work = sorted(set([*completed_work, stage]))
        state = StageState(
            stage=f"main:{stage}",
            execution_id=execution_id,
            config_hash=self._config.config_hash,
            status=status,
            completed_work=completed_work,
            pending_work=[] if status == StageStatus.COMPLETED else [stage],
            outstanding_annotation_task_ids=sorted(set(pending_tasks or [])),
            cursor={"result": result, "scientific_substep": result.get("substep")},
            input_manifests=inputs,
            output_manifests={
                key: value
                for key, value in result.items()
                if isinstance(value, str)
                and (
                    key.endswith("_manifest_id")
                    or key in {"artifact_id", "corpus_manifest_hash"}
                )
            },
            started_at=(
                previous[0].started_at if previous else timestamp
            ),
            updated_at=timestamp,
            completed_at=timestamp if status == StageStatus.COMPLETED else None,
            errors=[] if error_class is None else [{"exception_class": error_class}],
            metadata={
                "experiment_id": self._config.experiment_id,
                "selection": self._selection(stage),
                "configuration_id": self._configuration_id,
            },
        )
        self._store.set_stage_state(
            f"main:{stage}", state.model_dump(mode="json")
        )

    def _immutable(self, kind: str, record_id: str) -> None:
        records: list[dict[str, Any]] = self._store.list_records(
            kind,
            {"record_id": record_id, "lifecycle": ["immutable", "frozen"]},
        )
        if len(records) != 1:
            raise ValueError(f"{kind}/{record_id} is not an immutable prerequisite.")

    def _manifest(self, name: str) -> dict[str, Any]:
        declaration: str | None = (
            self._config.operational.prerequisite_manifests.get(name)
        )
        if declaration is None:
            raise ValueError(f"Prerequisite manifest {name!r} is unresolved.")
        return self._resolve_manifest(declaration)

    def _resolve_manifest(self, declaration: str) -> dict[str, Any]:
        # Stage references may be declared before derived artifact IDs exist.
        # Resolution is scoped to the exact configuration and one completed
        # execution, avoiding a configuration/response hash fixed point.
        if declaration.startswith("urn:stage:"):
            stage = self._stage(declaration.removeprefix("urn:stage:"))
            completed = [state for state in self._states(stage)
                         if state.status == StageStatus.COMPLETED
                         and state.config_hash == self._config.config_hash]
            if len(completed) != 1:
                raise ValueError("Stage prerequisite requires one completed execution for this configuration.")
            state = completed[0]
            return {"config_hash": state.config_hash,
                    "corpus_version": self._config.corpus_version,
                    "execution_id": state.execution_id,
                    "result": copy.deepcopy(state.cursor["result"])}
        identity: str = declaration.removeprefix("sha256:")
        matches: list[tuple[str, dict[str, Any]]] = []
        for kind in _MANIFEST_KINDS:
            try:
                matches.append((kind, self._store.read_record(kind, identity)))
            except KeyError:
                pass
        if matches:
            if len(matches) != 1:
                raise ValueError("Prerequisite manifest identity is ambiguous.")
            kind, payload = matches[0]
            self._immutable(kind, identity)
            self._compatible(payload)
            return payload

        if "://" not in declaration and not declaration.startswith(
            ("sha256:", "urn:")
        ):
            local: Path = Path(declaration)
            local = local if local.is_absolute() else self._base / local
            if local.is_file():
                raw: bytes = local.read_bytes()
                if local.suffix.lower() == ".json":
                    value: Any = json.loads(
                        raw.decode("utf-8"),
                        object_pairs_hook=_json_pairs,
                        parse_constant=_reject_constant,
                    )
                else:
                    documents: list[Any] = list(
                        yaml.load_all(raw.decode("utf-8"), Loader=_UniqueLoader)
                    )
                    if len(documents) != 1:
                        raise ValueError("Exactly one manifest document is required.")
                    value = documents[0]
                payload = _mapping(value, "Prerequisite manifest")
                _canonical_json(payload)
                self._compatible(payload)
                self._store.save_blob(raw)
                return payload

        # freeze_records returns a hash but exposes no dedicated manifest reader.
        # Reconstruct candidate memberships through public record APIs only.
        for kind in (
            "task_instance", "model_response", "failure_decision",
            "reference", "taxonomy", "taxonomy_version", "score",
        ):
            payloads: list[dict[str, Any]] = self._store.list_records(
                kind, {"lifecycle": ["immutable", "frozen"]}
            )
            identity_field: str = {
                "task_instance": "instance_id",
                "model_response": "response_id",
                "failure_decision": "",
                "reference": "reference_id",
                "taxonomy": "taxonomy_id",
                "taxonomy_version": "taxonomy_id",
                "score": "score_id",
            }[kind]
            members: list[dict[str, str]] = []
            for payload in payloads:
                record_id: str = (
                    payload[identity_field] if identity_field else _content_hash(payload)
                )
                members.append({
                    "record_id": record_id,
                    "content_hash": _content_hash(payload),
                })
            members.sort(key=lambda item: item["record_id"])
            reconstructed: dict[str, Any] = {"kind": kind, "members": members}
            if members and _content_hash(reconstructed) == identity:
                return reconstructed
        raise ValueError(
            f"Prerequisite {declaration!r} cannot be resolved. Supply a local "
            "membership manifest or a unique stored manifest ID; opaque blob "
            "references cannot be read through the public storage interface."
        )

    def _compatible(self, manifest: dict[str, Any]) -> None:
        config_hash: Any = manifest.get("config_hash")
        if config_hash is not None and config_hash != self._config.config_hash:
            raise ValueError("Prerequisite uses a different configuration hash.")
        corpus: Any = manifest.get("corpus_version")
        if corpus is not None and corpus != self._config.corpus_version:
            raise ValueError("Prerequisite uses a different corpus version.")

    def _response_ids(self) -> list[str]:
        manifest: dict[str, Any] = self._manifest("responses")
        nested: Any = manifest.get("result", manifest)
        result: dict[str, Any] = _mapping(nested, "Response manifest result")
        raw: Any = result.get("response_ids")
        if raw is None and manifest.get("kind") == "model_response":
            raw = [item["record_id"] for item in manifest.get("members", [])]
        if raw is None and isinstance(result.get("members"), list):
            raw = [item["response_id"] for item in result["members"]]
        identifiers: list[str] = _ids(raw, "Frozen response population")
        for identity in identifiers:
            response = ModelResponse.model_validate(
                self._store.read_record("model_response", identity)
            )
            self._immutable("model_response", identity)
            if response.config_hash != self._config.config_hash:
                raise ValueError("Response manifest contains another configuration.")
            instance = TaskInstance.model_validate(
                self._store.read_record("task_instance", response.instance_id)
            )
            if not instance.reference.get("answer"):
                raise ValueError("Response manifest contains an unreferenced task.")
        return identifiers

    def _failure_ids(self) -> list[str]:
        manifest: dict[str, Any] = self._manifest("identification")
        result: dict[str, Any] = _mapping(
            manifest.get("result", manifest), "Identification result"
        )
        if result.get("status") != "completed":
            raise ValueError("Failure identification has not completed.")
        coverage: Any = result.get("coverage", {})
        if not isinstance(coverage, dict) or coverage.get("complete") is not True:
            raise ValueError(
                "Identification coverage is incomplete; unresolved responses cannot "
                "be silently excluded from the downstream failure population."
            )
        failures: list[str] = _ids(
            result.get("failure_response_ids"),
            "Failure population",
            allow_empty=True,
        )
        bindings: dict[str, Any] = _mapping(
            result.get("decision_ids_by_response"), "Failure decision bindings"
        )
        for response_id in failures:
            if response_id not in bindings:
                raise ValueError("Failure population lacks designated decision identities.")
            decision_id: str = _identifier(bindings[response_id])
            decision = FailureDecision.model_validate(
                self._store.read_record("failure_decision", decision_id)
            )
            self._immutable("failure_decision", decision_id)
            if (
                decision.response_id != response_id
                or decision.decision != FailureStatus.FAILURE
                or decision.requires_review
            ):
                raise ValueError("Failure population contains an unresolved or passed response.")
        return failures

    def _frozen_taxonomy(self) -> TaxonomyVersion:
        manifest: dict[str, Any] = self._manifest("taxonomy")
        result: dict[str, Any] = _mapping(
            manifest.get("result", manifest), "Taxonomy prerequisite"
        )
        identity: Any = result.get("taxonomy_id")
        if not isinstance(identity, str):
            raise ValueError("Taxonomy manifest must designate a frozen taxonomy ID.")
        found: list[tuple[str, TaxonomyVersion]] = []
        for kind in ("taxonomy", "taxonomy_version"):
            try:
                found.append((
                    kind,
                    TaxonomyVersion.model_validate(
                        self._store.read_record(kind, identity)
                    ),
                ))
            except KeyError:
                pass
        if len(found) != 1:
            raise ValueError("Stored taxonomy is missing or ambiguous.")
        kind, taxonomy = found[0]
        self._immutable(kind, identity)
        if not taxonomy.frozen:
            raise ValueError("Categorization requires a stabilized frozen taxonomy.")
        return taxonomy

    def _instance_filters(self) -> dict[str, Any]:
        filters: dict[str, Any] = {}
        if self._config.operational.selected_tasks is not None:
            filters["task_name"] = self._config.operational.selected_tasks
        return filters

    def _audit_declaration(self) -> dict[str, Any]:
        selected: str | None = self._config.operational.selected_audit
        if selected is None:
            raise ValueError("operational.selected_audit is unresolved.")
        manifest: dict[str, Any] = self._manifest("audit_sampling")
        audits: dict[str, Any] = _mapping(
            manifest.get("audits"), "Audit sampling manifest.audits"
        )
        return _mapping(audits.get(selected), f"Audit declaration for {selected}")

    def _calibration(self, declaration: dict[str, Any], guideline_hash: str) -> None:
        tasks: list[AnnotationTask] = [
            AnnotationTask.model_validate(item)
            for item in self._store.list_records("annotation_task")
            if item.get("purpose") == "calibration"
            and item.get("guideline_hash") == guideline_hash
        ]
        declared: Any = declaration.get("calibration_response_ids")
        identities: list[str] = (
            _ids(declared, "Calibration response IDs")
            if declared is not None
            else sorted({task.response_id for task in tasks})
        )
        if len(identities) != self._config.annotation.calibration.instances:
            raise ValueError(
                "Formal audits require the completed configured calibration "
                "population under the frozen guideline."
            )
        for identity in identities:
            reference: dict[str, Any] = self._annotations.get_reference(
                identity, "calibration"
            )
            if (
                reference.get("guideline_hash") != guideline_hash
                or reference.get("uncertain", True)
            ):
                raise ValueError("Calibration is incomplete or guideline-incompatible.")

    def _audit_tasks(
        self,
        cohort: AuditCohort,
        purpose: str,
        response_ids: list[str],
    ) -> list[AnnotationTask]:
        protocol: dict[str, Any] = _mapping(
            cohort.sampling_protocol.get("annotation_protocol"),
            "Frozen audit annotation protocol",
        )
        return self._annotations.create_tasks(response_ids, purpose, protocol)

    def _independent_pending(self, tasks: list[AnnotationTask]) -> list[str]:
        submissions: dict[str, list[dict[str, Any]]] = {}
        for record in self._store.list_records("annotation"):
            task_id: Any = record.get("annotation_task_id")
            if isinstance(task_id, str):
                submissions.setdefault(task_id, []).append(record)
        pending: list[str] = []
        for task in tasks:
            records: list[dict[str, Any]] = [
                record for record in submissions.get(task.annotation_task_id, [])
                if record.get("purpose") == task.purpose.value
                and record.get("cohort_id") == task.cohort_id
                and record.get("annotator_id") == task.assigned_annotator
                and record.get("guideline_hash") == task.guideline_hash
                and record.get("status") == "submitted"
                and record.get("audit_only") is True
            ]
            if len(records) > 1:
                raise ValueError("An independent audit slot has ambiguous submissions.")
            if not records:
                pending.append(task.annotation_task_id)
        return sorted(pending)

    def _references_pending(
        self,
        cohort: AuditCohort,
        purpose: str,
        response_ids: list[str],
    ) -> tuple[list[str], list[str]]:
        unresolved: list[str] = []
        pending_tasks: set[str] = set()
        for response_id in response_ids:
            try:
                reference = self._annotations.get_reference(response_id, purpose)
            except LookupError:
                unresolved.append(response_id)
                continue
            if (
                reference.get("cohort_id") != cohort.cohort_id
                or reference.get("audit_only") is not True
                or reference.get("guideline_hash")
                != cohort.sampling_protocol["guideline_hash"]
                or reference.get("uncertain", True)
            ):
                raise ValueError("Audit reference has incompatible lineage.")
        if unresolved:
            for payload in self._store.list_records("annotation_task"):
                if (
                    payload.get("response_id") in unresolved
                    and payload.get("cohort_id") == cohort.cohort_id
                    and payload.get("status") in {"draft", "pending", "in_progress"}
                ):
                    pending_tasks.add(payload["annotation_task_id"])
            if not pending_tasks:
                raise ValueError(
                    "Unresolved audit references have no open independent or "
                    "adjudication tasks. Inspect the preserved workflow lineage."
                )
        return unresolved, sorted(pending_tasks)

    def _waiting(
        self,
        cohort: AuditCohort,
        substep: str,
        task_ids: list[str],
    ) -> dict[str, Any]:
        if not task_ids:
            raise ValueError("Annotation waiting must identify outstanding tasks.")
        return {
            "status": StageStatus.WAITING_FOR_ANNOTATIONS.value,
            "substep": substep,
            "cohort_id": cohort.cohort_id,
            "cohort_hash": _content_hash(cohort),
            "pending_annotation_task_ids": sorted(set(task_ids)),
            "audit_work_is_operational_verification": False,
        }

    def _run_audit(self) -> dict[str, Any]:
        selected: str = _identifier(self._config.operational.selected_audit)
        declaration: dict[str, Any] = self._audit_declaration()
        sampling: dict[str, Any] = _mapping(
            declaration.get("sampling"), "Resolved audit sampling protocol"
        )
        if sampling.get("config_hash") != self._config.config_hash:
            raise ValueError("Audit sampling declaration uses another configuration.")
        guideline_hash: str = _identifier(sampling.get("guideline_hash"))
        self._calibration(declaration, guideline_hash)

        cohort: AuditCohort = self._annotations.sample_audit(selected, sampling)
        cohort = self._annotations.lock_cohort(cohort.cohort_id)

        if selected == "failure_identification":
            purpose: str = "independent_identification_audit"
            tasks = self._audit_tasks(cohort, purpose, cohort.response_ids)
            pending = self._independent_pending(tasks)
            if pending:
                return self._waiting(cohort, "independent_identification", pending)
            self._annotations.create_adjudications([
                task.annotation_task_id for task in tasks
            ])
            unresolved, pending = self._references_pending(
                cohort, purpose, cohort.response_ids
            )
            if unresolved:
                return self._waiting(cohort, "identification_adjudication", pending)
            metrics = self._evaluator.score_failure_audit(cohort)
            return {
                "status": StageStatus.COMPLETED.value,
                "substep": "identification_audit_scored",
                "cohort_id": cohort.cohort_id,
                "metrics": metrics,
                "artifact_id": metrics.get("artifact_id"),
                "audit_labels_used_for_tuning": False,
            }

        if selected == "human_agreement":
            purpose = "human_agreement"
            tasks = self._audit_tasks(cohort, purpose, cohort.response_ids)
            pending = self._independent_pending(tasks)
            if pending:
                return self._waiting(cohort, "independent_human_agreement", pending)
            # This must precede creation of disagreement adjudications.
            agreement = self._evaluator.score_human_agreement(cohort)
            self._annotations.create_adjudications([
                task.annotation_task_id for task in tasks
            ])
            unresolved, pending = self._references_pending(
                cohort, purpose, cohort.response_ids
            )
            if unresolved:
                waiting = self._waiting(cohort, "agreement_adjudication", pending)
                waiting["pre_adjudication_agreement"] = agreement
                return waiting
            return {
                "status": StageStatus.COMPLETED.value,
                "substep": "agreement_and_references_complete",
                "cohort_id": cohort.cohort_id,
                "pre_adjudication_agreement": agreement,
                "artifact_id": agreement.get("artifact_id"),
            }

        if selected != "failure_categorization":
            raise ValueError("Unsupported configured audit selection.")

        protocol = cohort.sampling_protocol
        original_id: str = _identifier(protocol.get("original_cohort_id"))
        original = AuditCohort.model_validate(
            self._store.read_record("audit_cohort", original_id)
        )
        original_ids: list[str] = _ids(
            protocol.get("original_response_ids"), "Original agreement membership"
        )
        additional_ids: list[str] = _ids(
            protocol.get("additional_response_ids"), "Additional audit membership"
        )
        sizes = self._config.audits.failure_categorization
        if (
            not original.locked
            or original.purpose != "human_agreement"
            or set(original.response_ids) != set(original_ids)
            or len(original_ids) != sizes.original_agreement_instances
            or len(additional_ids) != sizes.additional_instances
            or set(original_ids) & set(additional_ids)
            or set(original_ids + additional_ids) != set(cohort.response_ids)
            or original.sampling_protocol.get("guideline_hash") != guideline_hash
        ):
            raise ValueError("Expanded audit does not preserve compatible distinct membership.")

        agreement_reports: list[dict[str, Any]] = [
            item for item in self._store.list_records("report")
            if item.get("purpose") == "human_agreement"
            and item.get("cohort_id") == original_id
            and item.get("pre_adjudication") is True
            and item.get("status") == "complete"
            and item.get("guideline_hash") == guideline_hash
        ]
        if not agreement_reports:
            raise ValueError(
                "Expanded categorization audit requires the original recorded "
                "pre-adjudication agreement study."
            )
        unresolved, pending = self._references_pending(
            original, "human_agreement", original_ids
        )
        if unresolved:
            return self._waiting(original, "original_audit_references", pending)

        purpose = "expanded_label_audit"
        tasks = self._audit_tasks(cohort, purpose, additional_ids)
        pending = self._independent_pending(tasks)
        if pending:
            return self._waiting(cohort, "additional_independent_references", pending)
        self._annotations.create_adjudications([
            task.annotation_task_id for task in tasks
        ])
        unresolved, pending = self._references_pending(cohort, purpose, additional_ids)
        if unresolved:
            return self._waiting(cohort, "additional_reference_adjudication", pending)

        metrics = self._evaluator.score_label_audit(cohort)
        return {
            "status": StageStatus.COMPLETED.value,
            "substep": "expanded_categorization_audit_scored",
            "cohort_id": cohort.cohort_id,
            "original_cohort_id": original_id,
            "metrics": metrics,
            "artifact_id": metrics.get("artifact_id"),
            "pre_correction_predictions_evaluated": True,
            "audit_labels_used_for_operational_correction": False,
        }

    def _stored_results(self) -> dict[str, Any]:
        reports: list[dict[str, Any]] = [
            item for item in self._store.list_records("report")
            if item.get("config_hash") == self._config.config_hash
        ]
        states: list[dict[str, Any]] = [
            item for item in self._store.list_records("stage_state")
            if item.get("config_hash") == self._config.config_hash
            and item.get("stage") not in {"main:report", "main:export"}
        ]
        workload: dict[str, Any] = {}
        submissions: list[dict[str, Any]] = self._store.list_records("annotation")
        adjudications: list[dict[str, Any]] = self._store.list_records("adjudication")
        for label, records in (
            ("independent_submissions", submissions),
            ("adjudications", adjudications),
        ):
            relevant: list[dict[str, Any]] = [
                item for item in records
                if item.get("config_hash") == self._config.config_hash
            ]
            counts: Counter[str] = Counter(
                item.get("purpose", "unspecified") for item in relevant
            )
            workload[label] = {
                "by_purpose": dict(sorted(counts.items())),
                "audit_count": sum(item.get("audit_only") is True for item in relevant),
                "operational_count": sum(
                    item.get("audit_only") is False for item in relevant
                ),
                "unclassified_count": sum(
                    not isinstance(item.get("audit_only"), bool) for item in relevant
                ),
            }
        return {
            "reports": reports,
            "stage_states": states,
            "annotation_workload": workload,
            "implementation_choices": self._config.model_dump(mode="json")[
                "implementation_choices"
            ],
            "missing_results_are_zero": False,
            "source_blobs": (
                "Content-addressed source artifacts are referenced by exported "
                "manifests; no unsupported blob-export interface is invoked."
            ),
        }

    def _dispatch(self, stage: str) -> dict[str, Any]:
        if stage == "prepare":
            manifest_path: str = _identifier(self._config.data.evidence_manifest)
            self._loader.ingest_sources(manifest_path)
            instances: list[TaskInstance] = self._loader.build_instances()
            for instance in instances:
                errors: list[dict[str, str]] = [
                    item for item in self._loader.validate_instance(instance)
                    if item.get("severity") == "error"
                ]
                if errors:
                    raise ValueError(
                        f"Preparation validation failed for {instance.instance_id}."
                    )
            corpus_hash: str = self._loader.freeze_corpus()
            return {
                "status": StageStatus.COMPLETED.value,
                "corpus_manifest_hash": corpus_hash,
                "instance_ids": [item.instance_id for item in instances],
                "task_assignment_count": len(instances),
                "unique_case_count": len({item.case_id for item in instances}),
            }

        if stage == "evaluate":
            self._manifest("corpus")
            self._manifest("references")
            model_ids: list[str] = (
                self._config.operational.selected_models
                or list(self._config.models.evaluation)
            )
            self._models.validate_endpoints(model_ids)
            instances = self._loader.load_instances(self._instance_filters())
            response_ids: list[str] = self._runner.run_task_evaluation(instances)
            self._store.freeze_records("model_response", response_ids)
            metrics = self._evaluator.aggregate_task_metrics(response_ids)
            return {
                "status": StageStatus.COMPLETED.value,
                "response_ids": response_ids,
                "task_metrics": metrics,
                "artifact_id": metrics.get("artifact_id"),
            }

        if stage == "identify":
            return self._identifier.run(self._response_ids())

        if stage == "discover_taxonomy":
            return self._taxonomy_manager.run(self._failure_ids())

        if stage == "categorize":
            failures: list[str] = self._failure_ids()
            taxonomy = self._frozen_taxonomy()
            if not failures:
                return {
                    "status": StageStatus.COMPLETED.value,
                    "taxonomy_id": taxonomy.taxonomy_id,
                    "final_assignment_ids_by_response": {},
                    "failure_population_size": 0,
                    "operational_review_fraction": None,
                }
            return self._categorizer.run(failures, taxonomy)

        if stage == "audit":
            return self._run_audit()

        if stage == "baselines":
            name: str = _identifier(self._config.operational.selected_baseline)
            return self._runner.run_baseline(name, self._failure_ids())

        if stage == "ablations":
            component: str = _identifier(self._config.operational.selected_ablation)
            return self._runner.run_ablation(component, self._response_ids())

        if stage == "stability":
            declaration: str = _identifier(self._config.taxonomy_stability.audit_manifest)
            manifest = self._resolve_manifest(declaration)
            cohort_payload: Any = manifest.get("cohort", manifest)
            if isinstance(cohort_payload, dict) and "response_ids" in cohort_payload:
                cohort = AuditCohort.model_validate(cohort_payload)
            else:
                cohort_id: str = _identifier(manifest.get("cohort_id"))
                cohort = AuditCohort.model_validate(
                    self._store.read_record("audit_cohort", cohort_id)
                )
            self._immutable("audit_cohort", cohort.cohort_id)
            return self._runner.run_stability_study(self._failure_ids(), cohort)

        if stage == "defenses":
            selected: str = _identifier(self._config.operational.selected_defense)
            if selected == "stronger_grounding":
                declaration = _identifier(
                    self._config.defenses.stronger_grounding.matched_instance_manifest
                )
                matched = self._resolve_manifest(declaration)
                identifiers = _ids(matched.get("instance_ids"), "Matched grounding instances")
                return self._runner.run_grounding_comparison(identifiers)
            instances = self._loader.load_instances(self._instance_filters())
            identifiers = [item.instance_id for item in instances]
            if selected == "counterfactual_consistency":
                return self._runner.run_counterfactual_defense(identifiers)
            if selected == "tailored_mitigation":
                mode: str = _identifier(
                    self._config.operational.selected_mitigation_mode
                )
                return self._runner.run_tailored_mitigation(mode, identifiers)
            raise ValueError("Unknown explicitly selected defense.")

        if stage == "report":
            self._manifest("results")
            self._manifest("scoring")
            directory: Path = self._path(
                self._config.operational.output_path, "operational.output_path"
            )
            results: dict[str, Any] = self._stored_results()
            report_id: str = _content_hash(results)
            destination: Path = directory / f"report-{report_id}.json"
            self._evaluator.write_report(results, str(destination))
            return {
                "status": StageStatus.COMPLETED.value,
                "report_path": str(destination),
                "report_content_identity": report_id,
                "inference_executed": False,
            }

        raise ValueError(f"Stage is not implemented: {stage}.")

    def run_stage(self, stage: str) -> dict[str, Any]:
        """Validate and advance one stage, preserving human waiting states."""
        selected: str = self._stage(stage)
        self._config.validate(selected)
        inputs: dict[str, str] = self._inputs(selected)
        execution_id: str = self._execution_id(selected, inputs)
        previous: list[StageState] = [
            state for state in self._states(selected)
            if state.execution_id == execution_id
        ]
        completed: list[StageState] = [
            state for state in previous if state.status == StageStatus.COMPLETED
        ]
        # Reports intentionally observe newly stored artifacts on each request.
        if completed and selected != "report":
            return copy.deepcopy(completed[-1].cursor["result"])

        if selected != "report":
            self._checkpoint(
                selected, inputs, StageStatus.RUNNING,
                {"status": StageStatus.RUNNING.value},
            )
        try:
            result: dict[str, Any] = self._dispatch(selected)
            if not isinstance(result, dict):
                raise TypeError("Stage components must return a structured result.")
            status: StageStatus = StageStatus(result.get("status", "blocked"))
            pending: list[str] = []
            if status == StageStatus.WAITING_FOR_ANNOTATIONS:
                pending = _ids(
                    result.get("pending_annotation_task_ids"),
                    "Outstanding annotation tasks",
                )
            if selected != "report":
                self._checkpoint(
                    selected, inputs, status, result, pending_tasks=pending
                )
            return result
        except Exception as exc:
            if selected != "report":
                self._checkpoint(
                    selected,
                    inputs,
                    StageStatus.FAILED,
                    {
                        "status": StageStatus.FAILED.value,
                        "exception_class": type(exc).__name__,
                        "scientific_fallback_applied": False,
                    },
                    error_class=type(exc).__name__,
                )
            raise

    def resume(self, stage: str) -> dict[str, Any]:
        """Resume only an existing, configuration-compatible execution."""
        selected: str = self._stage(stage)
        self._config.validate(selected)
        states: list[StageState] = self._states(selected)
        if not states:
            raise ValueError(
                f"No orchestration checkpoint exists for {selected!r}. "
                "Use its explicit stage command to begin a new execution."
            )
        inputs: dict[str, str] = self._inputs(selected)
        expected: str = self._execution_id(selected, inputs)
        compatible: list[StageState] = [
            state for state in states
            if state.config_hash == self._config.config_hash
            and state.input_manifests == inputs
            and state.execution_id == expected
        ]
        if not compatible:
            raise ValueError(
                "Existing checkpoints use different configuration or prerequisite "
                "identities. Resume is prohibited; declare a new documented execution."
            )
        latest: StageState = compatible[-1]
        if latest.status == StageStatus.COMPLETED:
            return copy.deepcopy(latest.cursor["result"])
        return self.run_stage(selected)

    def export_results(self, output_path: str) -> None:
        """Export immutable records and stored results without invoking inference."""
        destination: Path = self._path(output_path, "export destination")
        destination.mkdir(parents=True, exist_ok=True)
        for kind in _EXPORT_KINDS:
            self._store.export_records(kind, str(destination / f"{kind}.jsonl"))
        self._evaluator.write_report(
            self._stored_results(), str(destination / "results.json")
        )
        _LOGGER.info(
            "Exported records to %s. Source blobs remain referenced by their "
            "content-addressed provenance manifests.",
            destination,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce CTI failure identification and categorization.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Explicit experiment YAML path; no configuration fallback is used.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Engineering logging verbosity; does not alter experiments.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "prepare", "evaluate", "identify", "discover-taxonomy", "categorize",
        "audit", "stability", "report",
    ):
        command = commands.add_parser(name, allow_abbrev=False)
        if name == "audit":
            command.add_argument(
                "--audit",
                required=True,
                choices=(
                    "failure_identification", "human_agreement",
                    "failure_categorization",
                ),
                help="Must match operational.selected_audit.",
            )
        if name == "report":
            command.add_argument(
                "--export",
                default=None,
                metavar="DIRECTORY",
                help="Optionally export stored records to this explicit directory.",
            )

    baseline = commands.add_parser("baselines", allow_abbrev=False)
    baseline.add_argument(
        "--name",
        required=True,
        choices=("kmeans", "lda", "gpt_judge", "gemini_judge", "multi_agent_debate"),
    )
    ablation = commands.add_parser("ablations", allow_abbrev=False)
    ablation.add_argument(
        "--component",
        required=True,
        choices=(
            "stratification", "human_initiation", "multi_agent_decision",
            "repetition", "human_verification",
        ),
    )
    defense = commands.add_parser("defenses", allow_abbrev=False)
    defense.add_argument(
        "--defense",
        required=True,
        choices=(
            "counterfactual_consistency", "tailored_mitigation",
            "stronger_grounding",
        ),
    )
    defense.add_argument(
        "--mode",
        default=None,
        choices=("retrieval_only", "inference_only", "full_fledged"),
    )
    resume = commands.add_parser("resume", allow_abbrev=False)
    resume.add_argument(
        "stage",
        choices=tuple(name for name in _STAGE_ALIASES if "_" not in name),
    )
    return parser


def _entrypoint(argv: list[str] | None = None) -> int:
    parser: argparse.ArgumentParser = _parser()
    arguments: argparse.Namespace = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        # Check CLI selectors before opening storage or creating provenance.
        config: ExperimentConfig = ExperimentConfig.load(arguments.config)
        operational = config.operational
        if arguments.command == "baselines":
            if arguments.name != operational.selected_baseline:
                raise ValueError("--name must match operational.selected_baseline.")
        elif arguments.command == "ablations":
            if arguments.component != operational.selected_ablation:
                raise ValueError("--component must match operational.selected_ablation.")
        elif arguments.command == "audit":
            if arguments.audit != operational.selected_audit:
                raise ValueError("--audit must match operational.selected_audit.")
        elif arguments.command == "defenses":
            if arguments.defense != operational.selected_defense:
                raise ValueError("--defense must match operational.selected_defense.")
            if arguments.defense == "tailored_mitigation":
                if (
                    arguments.mode is None
                    or arguments.mode != operational.selected_mitigation_mode
                ):
                    raise ValueError(
                        "Tailored mitigation requires --mode matching "
                        "operational.selected_mitigation_mode."
                    )
            elif arguments.mode is not None:
                raise ValueError("--mode applies only to tailored_mitigation.")

        application = Main(arguments.config)
        result: dict[str, Any] = (
            application.resume(arguments.stage)
            if arguments.command == "resume"
            else application.run_stage(arguments.command)
        )
        if arguments.command == "report" and arguments.export is not None:
            application.export_results(arguments.export)
        print(_canonical_json(result))
        status: Any = result.get("status")
        if status == StageStatus.COMPLETED.value:
            return 0
        if status == StageStatus.WAITING_FOR_ANNOTATIONS.value:
            return 3
        return 2
    except KeyboardInterrupt:
        _LOGGER.error("Interrupted; persisted artifacts remain available for resumption.")
        return 130
    except Exception as exc:
        # Provider exceptions can contain request headers or credentials. Detailed
        # model diagnostics belong to the sanitized persisted attempt records.
        if isinstance(exc, (ValueError, FileNotFoundError, PermissionError, LookupError)):
            _LOGGER.error("%s", exc)
        else:
            _LOGGER.error(
                "Execution stopped (%s). Inspect preserved stage/attempt artifacts; "
                "no scientific fallback was applied.",
                type(exc).__name__,
            )
        return 2


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
