"""Configuration loading, provenance, hashing, and stage-local readiness checks.

An incomplete configuration is loadable. Execution is authorized
separately by ``ExperimentConfig.validate(stage)``. No unresolved experimental
value is replaced with a library default.

Additional operational declarations may be supplied alongside the original YAML:

    experiment_id: "explicit-study-id"
    corpus_version: "explicit-frozen-corpus-version"
    operational:
      database_path: "artifacts/study.sqlite"
      artifact_root: "artifacts/blobs"
      output_path: "artifacts/reports"
      selected_tasks: null
      selected_models: null
      selected_benchmarks: null
      selected_baseline: null
      selected_ablation: null
      selected_defense: null
      selected_mitigation_mode: null
      selected_audit: null
      taxonomy_mode: null
      prerequisite_manifests: {}
      applicability: {}
      artifact_hashes: {}
    implementation_choices:
      inference.temperature:
        configuration_path: "inference.temperature"
        origin: "implementation"
        rationale: "Describe why this value was selected."
        affected_experiments: ["evaluate"]
        selected_value: 0.1
        source: null
        manifest_reference: null
        applicability: null

``operational.applicability`` maps configuration paths to ``required`` or
``not_applicable``. A not-applicable declaration also needs a corresponding
provenance record; it cannot silently waive a required experimental setting.
Secrets must be supplied through environment variables, not this manifest.

Only the public methods specified by the project design are provided: ``load``
and ``validate``. Properties expose the designed flattened configuration view,
while inherited fields retain every original YAML section.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import parse_qsl, urlsplit

import yaml
from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    StrictStr,
    ValidationError,
    model_validator,
)

from schemas import (
    ConfigurationInput,
    Digest,
    Identifier,
    JsonObject,
    SchemaModel,
    _canonical_json,
    _content_hash,
    _json_primitive,
)


_EVALUATION_MODELS: dict[str, str] = {
    "G5": "GPT-5",
    "Go4": "GPT-o4 mini",
    "CLD": "Claude Sonnet 4",
    "GEM": "Gemini 2.5",
    "LL70": "Llama-3.1-70B-Instruct",
    "MIX": "Mixtral-8x7B-Instruct-v0.1",
    "QWN": "Qwen2.5-14B-Instruct",
    "GRK": "Grok-2",
    "FSC": "Foundation-Sec-8B",
    "CB0": "Cyber-Zero",
    "ZYS": "ZySec-AI-SecurityLLM",
    "LLY": "Lily-Cybersecurity-7B-v0.2",
    "CBS": "CyberBase-13b",
    "SPT": "clouditera-secgpt",
    "DHT": "DeepHat-V1-7B",
}

_PANEL_NAMES: frozenset[str] = frozenset(
    {"GPT-5", "Llama-4-17B", "Gemini 2.5", "Claude Sonnet 4"}
)

_TASK_METRICS: dict[str, dict[str, str]] = {
    "contextualization": {
        "affected_systems": "f1",
        "attack_infrastructure": "f1",
        "vulnerability_linking": "accuracy",
        "malware_family_mapping": "f1",
        "ioc_normalization": "f1",
        "threat_report_alignment": "bleu",
        "event_timeline_construction": "bleu",
        "graph_population": "accuracy",
        "source_reliability_scoring": "roc_auc",
    },
    "attribution": {
        "threat_actor_linking": "accuracy",
        "ttp_extraction": "f1",
        "campaign_attribution": "accuracy",
        "infrastructure_reuse": "f1",
        "language_style_profiling": "accuracy",
        "false_flag_detection": "f1",
        "evidence_weighting": "bleu",
        "relation_graph_building": "f1",
    },
    "prediction": {
        "exploit_likelihood": "roc_auc",
        "impact_forecast": "bleu",
        "target_sector_prediction": "accuracy",
        "campaign_escalation": "roc_auc",
    },
    "mitigation": {
        "patch_recommendation": "f1",
        "rule_generation_yara": "bleu",
        "response_summarization": "bleu",
        "mitigation_ttp_mapping": "accuracy",
        "defensive_playbook_generation": "bleu",
        "countermeasure_ranking": "ndcg",
        "incident_ticket_generation": "accuracy",
    },
}

_STAGE_ALIASES: dict[str, str] = {
    "prepare": "prepare",
    "evaluate": "evaluate",
    "identify": "identify",
    "discover_taxonomy": "discover_taxonomy",
    "discover-taxonomy": "discover_taxonomy",
    "categorize": "categorize",
    "audit": "audit",
    "baselines": "baselines",
    "ablations": "ablations",
    "stability": "stability",
    "defenses": "defenses",
    "report": "report",
}

_SECRET_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "auth_token",
        "authorization",
        "password",
        "passwd",
        "secret",
        "client_secret",
        "credentials",
        "private_key",
        "bearer_token",
    }
)

# Exact requirements fixed by the supplied paper-reproduction protocol.
_FIXED_VALUES: dict[str, Any] = {
    "study.execution_mode": "inference_and_analysis",
    "training.required": False,
    "evaluation.runs": 5,
    "evaluation.reporting_scale": [0, 100],
    "data.task_counts_are_approximate": True,
    "data.task_counts_may_overlap": True,
    "categorization.rounds": 2,
    "categorization.first_round_independent": True,
    "categorization.second_round_observes_first_round_peers": True,
    "categorization.human_review.trigger_on_first_round_disagreement": True,
    "categorization.human_review.trigger_on_any_agent_cross_round_change": True,
    "categorization.human_review.introduce_new_categories": False,
    "annotation.pool.annotators": 7,
    "annotation.pool.graduate_researchers": 4,
    "annotation.pool.professional_cybersecurity_researchers": 3,
    "annotation.calibration.instances": 45,
    "annotation.agreement_study.instances": 320,
    "annotation.agreement_study.annotators_per_instance": 3,
    "annotation.agreement_study.preserve_independent_labels": True,
    "annotation.adjudication.use_adjudicated_labels_for_agreement_statistics": False,
    "audits.failure_identification.instances": 100,
    "audits.failure_identification.independent_annotators": 2,
    "audits.failure_identification.freeze_identification_before_sampling": True,
    "audits.failure_identification.use_audit_labels_for_tuning": False,
    "audits.failure_categorization.original_agreement_instances": 320,
    "audits.failure_categorization.additional_instances": 480,
    "audits.failure_categorization.total_instances": 800,
    "audits.failure_categorization.include_stable_consensus_cases": True,
    "audits.failure_categorization.independent_human_references": True,
    "audits.failure_categorization.evaluate_predictions_before_human_correction": True,
    "audits.failure_categorization.count_audit_work_as_operational_verification": False,
    "taxonomy_stability.audit_instances": 180,
    "taxonomy_stability.independent_runs_per_condition": 5,
    "defenses.counterfactual_consistency.preserve_answer_determining_attributes": True,
    "defenses.counterfactual_consistency.require_task_dependent_preservation_justification": True,
    "defenses.counterfactual_consistency.consistency_is_not_correctness": True,
    "defenses.counterfactual_consistency.include_original_input": False,
    "defenses.counterfactual_consistency.use_self_reported_confidence_as_probability": False,
    "defenses.stronger_grounding.instances_per_task": 100,
    "defenses.stronger_grounding.distinct_task_instances": 300,
    "defenses.stronger_grounding.model_instance_pairs_per_configuration": 600,
    "defenses.stronger_grounding.runs": 1,
    "defenses.stronger_grounding.report_standard_deviation": False,
    "defenses.stronger_grounding.evidence_verification.revision_passes": 1,
    "defenses.stronger_grounding.evidence_verification.use_verifier_scores_as_final_labels": False,
    "reproducibility.freeze_corpus_and_reference_versions": True,
    "reproducibility.freeze_prompts_and_generation_settings": True,
    "reproducibility.retain_raw_model_outputs": True,
    "reproducibility.retain_pre_correction_predictions": True,
    "reproducibility.retain_independent_human_labels": True,
    "reproducibility.separate_operational_review_from_audits": True,
    "reproducibility.record_implementation_choices": True,
    "reproducibility.silently_substitute_models": False,
    "reproducibility.silently_fill_unspecified_settings": False,
    "evaluation.prompt.external_browsing": False,
    "evaluation.prompt.supplied_evidence_only": True,
    "evaluation.prompt.enforce_snapshot_date": True,
    "evaluation.prompt.offensive_guidance_allowed": False,
}


class _UniqueSafeLoader(yaml.SafeLoader):
    """Safe YAML loader rejecting duplicate keys and preserving date strings."""


# Avoid implicit conversion of ISO dates into Python date objects in JSON fields.
_UniqueSafeLoader.yaml_implicit_resolvers = {
    key: [
        (tag, expression)
        for tag, expression in resolvers
        if tag != "tag:yaml.org,2002:timestamp"
    ]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _construct_mapping(
    loader: _UniqueSafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    """Reject duplicate keys, merge keys, and non-string configuration keys."""
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise ValueError("YAML merge keys are not permitted in experiment manifests.")
        key: Any = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("Configuration mapping keys must be strings.")
        if key in result:
            raise ValueError("Duplicate configuration mapping key.")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


class _Choice(SchemaModel):
    """Provenance for one explicit resolution or applicability decision."""

    configuration_path: Identifier
    origin: Literal["author_specified", "implementation"]
    rationale: Identifier
    affected_experiments: list[Identifier]
    selected_value: JsonValue = None
    manifest_reference: StrictStr | None = None
    source: StrictStr | None = None
    applicability: Literal["required", "not_applicable"] | None = None

    @model_validator(mode="after")
    def _check_choice(self) -> Self:
        if not self.affected_experiments:
            raise ValueError("A provenance record must identify affected experiments.")
        if self.origin == "author_specified" and not self.source:
            raise ValueError("An author-specified choice requires a supporting source.")
        if (
            "selected_value" not in self.model_fields_set
            and self.manifest_reference is None
            and self.applicability is None
        ):
            raise ValueError("Record the selected value, artifact, or applicability.")
        return self


class _Operational(SchemaModel):
    """Engineering and selection declarations for a concrete run."""

    database_path: StrictStr | None = None
    artifact_root: StrictStr | None = None
    output_path: StrictStr | None = None
    selected_tasks: list[Identifier] | None = None
    selected_models: list[Identifier] | None = None
    selected_benchmarks: list[Identifier] | None = None
    selected_baseline: Identifier | None = None
    selected_ablation: Identifier | None = None
    selected_defense: Identifier | None = None
    selected_mitigation_mode: Identifier | None = None
    selected_audit: Identifier | None = None
    taxonomy_mode: Literal["discovery", "fixed"] | None = None
    prerequisite_manifests: dict[Identifier, Identifier] = Field(default_factory=dict)
    applicability: dict[
        Identifier, Literal["required", "not_applicable"]
    ] = Field(default_factory=dict)
    artifact_hashes: dict[Identifier, Digest] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_selections(self) -> Self:
        for name in (
            "selected_tasks",
            "selected_models",
            "selected_benchmarks",
        ):
            values: list[str] | None = getattr(self, name)
            if values is not None and (
                not values or len(values) != len(set(values))
            ):
                raise ValueError(f"{name} must be nonempty and duplicate-free.")
        return self


def _get(value: Any, path: str, default: Any = None) -> Any:
    """Read a dotted path, including numeric positions in sequences."""
    current: Any = value
    for component in path.split("."):
        if isinstance(current, Mapping):
            if component not in current:
                return default
            current = current[component]
        elif isinstance(current, (tuple, list)) and component.isdigit():
            index: int = int(component)
            if index >= len(current):
                return default
            current = current[index]
        else:
            return default
    return current


def _is_unresolved(value: Any) -> bool:
    """False and zero are resolved values; empty policy containers are not."""
    return value is None or value == "" or value == {} or value == []


def _check_secrets(value: Any, path: str = "") -> None:
    """Reject secret-bearing fields/URLs without including their values in errors."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized: str = str(key).lower().replace("-", "_")
            child_path: str = f"{path}.{key}" if path else str(key)
            if (
                normalized in _SECRET_KEYS
                or normalized.endswith("_api_key")
                or normalized.endswith("_password")
                or normalized.endswith("_client_secret")
            ) and item is not None:
                raise ValueError(
                    f"Secret-valued configuration is forbidden at {child_path}; "
                    "use an environment-variable reference."
                )
            _check_secrets(item, child_path)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_secrets(item, f"{path}.{index}")
    elif isinstance(value, str) and "://" in value:
        try:
            parsed = urlsplit(value)
            has_credentials: bool = (
                parsed.username is not None or parsed.password is not None
            )
            query_secrets: bool = any(
                key.lower().replace("-", "_") in _SECRET_KEYS
                for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
            )
        except ValueError as exc:
            raise ValueError(f"Malformed URL at {path}.") from exc
        if has_credentials or query_secrets:
            raise ValueError(
                f"Credential-bearing URL is forbidden at {path}; "
                "supply credentials outside the manifest."
            )


def _reference_items(value: Any, path: str = "") -> list[tuple[str, str]]:
    """Find declared artifact locations without interpreting arbitrary prose."""
    result: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child: str = f"{path}.{key}" if path else str(key)
            if isinstance(item, str):
                is_reference: bool = (
                    key.endswith("_manifest")
                    or key.endswith("_path")
                    or key in {"round_one", "round_two", "manifest_reference"}
                    or path == "operational.prerequisite_manifests"
                )
                is_output: bool = child in {
                    "operational.database_path",
                    "operational.artifact_root",
                    "operational.output_path",
                }
                if is_reference and not is_output:
                    result.append((child, item))
            if key not in {"artifact_hashes"}:
                result.extend(_reference_items(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.extend(_reference_items(item, f"{path}.{index}"))
    return result


def _local_reference(value: str, base: Path) -> Path | None:
    """Resolve local paths privately; URI/version identifiers stay declarations."""
    if "://" in value or value.startswith(("sha256:", "urn:")):
        return None
    path: Path = Path(value)
    return path if path.is_absolute() else base / path


class ExperimentConfig(ConfigurationInput):
    """Validated full manifest with the designed flattened configuration view."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        allow_inf_nan=False,
        revalidate_instances="always",
        hide_input_in_errors=True,
    )

    experiment_id: Identifier | None = None
    corpus_version: Identifier | None = None
    implementation_choices: dict[Identifier, _Choice] = Field(default_factory=dict)
    operational: _Operational = Field(default_factory=_Operational)

    _source_path: Path | None = PrivateAttr(default=None)
    _reference_digests: dict[str, str] = PrivateAttr(default_factory=dict)

    @property
    def config_hash(self) -> str:
        """Hash the complete manifest and available referenced-file contents."""
        payload: dict[str, Any] = self.model_dump(mode="python")
        return _content_hash(
            {
                "configuration": payload,
                "referenced_content_sha256": self._reference_digests,
            }
        )

    @property
    def model_settings(self) -> JsonObject:
        return self._view(
            {"models": self.models, "inference": self.inference}
        )

    @property
    def task_settings(self) -> JsonObject:
        return self._view({"evaluation": self.evaluation, "data": self.data})

    @property
    def identification_settings(self) -> JsonObject:
        return self._view(self.failure_identification)

    @property
    def discovery_settings(self) -> JsonObject:
        return self._view(
            {
                "taxonomy_discovery": self.taxonomy_discovery,
                "taxonomy": self.taxonomy,
            }
        )

    @property
    def optional_experiments(self) -> JsonObject:
        return self._view(
            {
                "baselines": self.baselines,
                "ablations": self.ablations,
                "taxonomy_stability": self.taxonomy_stability,
                "defenses": self.defenses,
            }
        )

    @property
    def evaluation_runs(self) -> int:
        return self.evaluation.runs

    @property
    def quantile_width(self) -> float:
        return self.failure_identification.stratification.quantile_width

    @property
    def seeds(self) -> list[int | None] | None:
        """Preserve unresolved seeds, including explicit per-run null entries."""
        return copy.deepcopy(self.evaluation.run_seeds)

    @staticmethod
    def _view(value: Any) -> JsonObject:
        result: Any = _json_primitive(value)
        if not isinstance(result, dict):
            raise TypeError("A configuration view must be an object.")
        return copy.deepcopy(result)

    @classmethod
    def load(cls, path: str) -> Self:
        """Load one safe, duplicate-free YAML document without resolving nulls.

        Args:
            path: Explicit YAML path. Relative artifact paths are relative to it.

        Returns:
            A structurally valid, possibly scientifically incomplete manifest.

        Raises:
            ValueError: YAML, schema, security, provenance, or protocol violation.
            OSError: The explicitly selected configuration file cannot be read.
        """
        source_path: Path = Path(path).expanduser().resolve()
        text: str = source_path.read_text(encoding="utf-8")
        try:
            documents: list[Any] = list(
                yaml.load_all(text, Loader=_UniqueSafeLoader)
            )
        except (yaml.YAMLError, ValueError, RecursionError) as exc:
            # Parser snippets can contain secrets, so do not reproduce them.
            raise ValueError(
                "Configuration YAML is invalid; require one document, unique "
                "string mapping keys, and no merge keys."
            ) from None
        if len(documents) != 1 or not isinstance(documents[0], dict):
            raise ValueError("Configuration must contain exactly one mapping document.")

        raw: dict[str, Any] = documents[0]
        _check_secrets(raw)
        try:
            config: Self = cls.model_validate(raw)
        except ValidationError as exc:
            paths: list[str] = sorted(
                {
                    ".".join(str(part) for part in item["loc"]) or "<root>"
                    for item in exc.errors(include_input=False)
                }
            )
            raise ValueError(
                "Invalid configuration structure or scalar type at: "
                + ", ".join(paths)
                + ". Check schemas.py; unknown fields are not accepted."
            ) from None
        config._source_path = source_path
        config._check_protocol()
        config._check_provenance()
        config._capture_references()
        return config

    def _payload(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    def _check_protocol(self) -> None:
        """Check paper-fixed invariants without requiring execution readiness."""
        payload: dict[str, Any] = self._payload()
        errors: list[str] = []
        for path, expected in _FIXED_VALUES.items():
            actual: Any = _json_primitive(_get(payload, path))
            if actual != expected:
                errors.append(f"{path}: inconsistent with the fixed reproduction protocol")

        models: dict[str, Any] = payload["models"]["evaluation"]
        if set(models) != set(_EVALUATION_MODELS):
            errors.append("models.evaluation: require exactly the 15 specified model IDs")
        for model_id, name in _EVALUATION_MODELS.items():
            if model_id in models and models[model_id]["name"] != name:
                errors.append(
                    f"models.evaluation.{model_id}.name: model substitution is forbidden"
                )
        panel: list[Any] = payload["models"]["categorization_panel"]
        if len(panel) != 4 or {item["name"] for item in panel} != _PANEL_NAMES:
            errors.append(
                "models.categorization_panel: require the four specified distinct "
                "agents, including Llama-4-17B"
            )

        tasks: dict[str, Any] = payload["evaluation"]["tasks"]
        if set(tasks) != set(_TASK_METRICS):
            errors.append("evaluation.tasks: require the four specified CTI stages")
        for stage, expected_tasks in _TASK_METRICS.items():
            actual_tasks: dict[str, Any] = tasks.get(stage, {})
            if set(actual_tasks) != set(expected_tasks):
                errors.append(f"evaluation.tasks.{stage}: task inventory is inconsistent")
            for task, metric in expected_tasks.items():
                if task in actual_tasks and actual_tasks[task]["metric"] != metric:
                    errors.append(
                        f"evaluation.tasks.{stage}.{task}.metric: incorrect paper metric"
                    )

        grounding: Any = self.defenses.stronger_grounding
        if set(grounding.tasks) != {
            "affected_systems", "threat_actor_linking", "patch_recommendation"
        } or set(grounding.models) != {"G5", "CLD"}:
            errors.append("defenses.stronger_grounding: matched task/model inventory differs")
        if set(grounding.configurations) != {
            "original", "strong_rag", "strong_rag_with_evidence_verification"
        }:
            errors.append("defenses.stronger_grounding.configurations: require all three settings")
        if not all(
            grounding.comparison_controls.model_dump().values()
        ) or not grounding.retrieval.same_corpus_snapshot_between_grounding_variants:
            errors.append("defenses.stronger_grounding: matched controls must remain enabled")

        expected_blinding: set[str] = {
            "identification_scores",
            "automatic_assignments",
            "other_annotator_decisions",
        }
        audit: Any = self.audits.failure_identification
        if not expected_blinding.issubset(audit.blind_to):
            errors.append("audits.failure_identification.blind_to: required blinding is missing")
        if not {
            "human_anchors", "calibration_examples",
            "previously_manually_reviewed_cases", "duplicate_records",
        }.issubset(audit.exclude):
            errors.append("audits.failure_identification.exclude: required exclusions are missing")
        if audit.positive_class != "failure" or audit.zero_denominator_result != "N/A":
            errors.append("audits.failure_identification: failure-positive/N/A policy must be retained")

        all_tasks: set[str] = {
            task for stage_tasks in _TASK_METRICS.values() for task in stage_tasks
        }
        for field, allowed in (
            ("selected_tasks", all_tasks),
            ("selected_models", set(_EVALUATION_MODELS)),
            ("selected_benchmarks", set(self.data.benchmarks)),
        ):
            selection: list[str] | None = getattr(self.operational, field)
            if selection is not None and not set(selection).issubset(allowed):
                errors.append(f"operational.{field}: references unknown inventory entries")

        if errors:
            raise ValueError("Configuration protocol violations:\n- " + "\n- ".join(errors))

    def _check_provenance(self) -> None:
        """Check supplied provenance; missing provenance is a readiness issue."""
        payload: dict[str, Any] = self._payload()
        sentinel: object = object()
        errors: list[str] = []
        for key, choice in self.implementation_choices.items():
            if key != choice.configuration_path:
                errors.append(f"implementation_choices.{key}: path must match its mapping key")
                continue
            actual: Any = _get(payload, key, sentinel)
            if actual is sentinel:
                errors.append(f"implementation_choices.{key}: configuration path does not exist")
                continue
            if "selected_value" in choice.model_fields_set:
                if _canonical_json(actual) != _canonical_json(choice.selected_value):
                    errors.append(f"implementation_choices.{key}: selected value differs from configuration")
        for path, applicability in self.operational.applicability.items():
            if _get(payload, path, sentinel) is sentinel:
                errors.append(f"operational.applicability.{path}: unknown configuration path")
            choice: _Choice | None = self.implementation_choices.get(path)
            if choice is not None and choice.applicability != applicability:
                errors.append(f"operational.applicability.{path}: provenance disagrees")
        if errors:
            raise ValueError("Invalid implementation-choice provenance:\n- " + "\n- ".join(errors))

    def _capture_references(self) -> None:
        """Hash available local files without making unavailable files load errors."""
        base: Path = (
            self._source_path.parent if self._source_path is not None else Path.cwd()
        )
        digests: dict[str, str] = {}
        for path, reference in _reference_items(self._payload()):
            local: Path | None = _local_reference(reference, base)
            if local is not None and local.is_file():
                digests[path] = hashlib.sha256(local.read_bytes()).hexdigest()
        self._reference_digests = digests

    def validate(self, stage: str) -> None:
        """Authorize only the requested stage; aggregate all readiness failures.

        Database freezing, actual annotation completion, endpoint capabilities,
        and artifact ownership remain checks of the respective runtime services.
        This method checks declarations and local referenced-file integrity only.
        """
        normalized: str = stage.strip().lower()
        if normalized not in _STAGE_ALIASES:
            raise ValueError(
                "Unknown stage. Select one of: "
                + ", ".join(sorted(set(_STAGE_ALIASES.values())))
                + ". Resume must specify the original stage."
            )
        selected_stage: str = _STAGE_ALIASES[normalized]
        self._check_protocol()
        self._check_provenance()
        _check_secrets(self._payload())
        check: _Readiness = _Readiness(self, selected_stage)
        check.run()
        if check.errors:
            raise ValueError(
                f"Stage '{selected_stage}' is not ready:\n- "
                + "\n- ".join(dict.fromkeys(check.errors))
            )


class _Readiness:
    """Private dependency-aware accumulator; never performs experimental work."""

    def __init__(self, config: ExperimentConfig, stage: str) -> None:
        self.config: ExperimentConfig = config
        self.stage: str = stage
        self.payload: dict[str, Any] = config._payload()
        self.errors: list[str] = []
        self._required: set[str] = set()

    def _error(self, path: str, reason: str) -> None:
        self.errors.append(f"{path}: {reason}")

    def _provenance(self, path: str) -> None:
        choice: _Choice | None = self.config.implementation_choices.get(path)
        if choice is None:
            self._error(
                path,
                "record the selected value in "
                f"implementation_choices.{path}",
            )

    def _need(
        self,
        path: str,
        *,
        provenance: bool = True,
        artifact: bool = False,
        allow_not_applicable: bool = False,
    ) -> Any:
        self._required.add(path)
        value: Any = _get(self.payload, path)
        applicability: str | None = self.config.operational.applicability.get(path)
        if applicability == "not_applicable":
            choice: _Choice | None = self.config.implementation_choices.get(path)
            if (
                not allow_not_applicable
                or choice is None
                or choice.applicability != "not_applicable"
            ):
                self._error(path, "not-applicable declaration cannot waive this prerequisite")
            return value
        if _is_unresolved(value):
            self._error(path, f"unresolved; supply an explicit setting required by {self.stage}")
            return value
        if provenance:
            self._provenance(path)
        if artifact:
            self._artifact(path, value)
        return value

    def _many(self, prefix: str, names: tuple[str, ...]) -> None:
        for name in names:
            self._need(f"{prefix}.{name}")

    def _artifact(self, path: str, value: Any) -> None:
        if not isinstance(value, str):
            self._error(path, "artifact reference must be a string")
            return
        base: Path = (
            self.config._source_path.parent
            if self.config._source_path is not None
            else Path.cwd()
        )
        local: Path | None = _local_reference(value, base)
        declared: str | None = self.config.operational.artifact_hashes.get(path)
        if local is None:
            if not value.startswith(("sha256:", "urn:")) and declared is None:
                self._error(path, "remote artifact requires an immutable content hash")
            return
        if not local.is_file():
            self._error(path, "referenced local file is missing or is not a regular file")
            return
        digest: str = hashlib.sha256(local.read_bytes()).hexdigest()
        captured: str | None = self.config._reference_digests.get(path)
        if captured != digest:
            self._error(path, "artifact changed or appeared after load; reload the manifest")
        if declared is not None and declared != digest:
            self._error(path, "artifact content differs from its declared SHA-256")

    def _prerequisite(self, name: str) -> None:
        self._need(
            f"operational.prerequisite_manifests.{name}",
            provenance=False,
            artifact=True,
        )

    def _tasks(self) -> dict[str, tuple[str, Any]]:
        selected: list[str] | None = self.config.operational.selected_tasks
        return {
            name: (stage, task)
            for stage, tasks in self.config.evaluation.tasks.items()
            for name, task in tasks.items()
            if selected is None or name in selected
        }

    def _models(self, model_ids: list[str] | None = None) -> None:
        selected: list[str] = (
            model_ids
            if model_ids is not None
            else (
                self.config.operational.selected_models
                or list(self.config.models.evaluation)
            )
        )
        for model_id in selected:
            for field in ("endpoint_or_checkpoint", "revision"):
                self._need(f"models.evaluation.{model_id}.{field}")

    def _generation(self) -> None:
        self._many(
            "inference",
            (
                "temperature", "top_p", "max_output_tokens", "context_window",
                "concurrency", "truncation_policy", "retry_policy",
            ),
        )
        # Hosting-dependent options have no universal meaning for API providers.
        for field in ("precision", "quantization", "reasoning_settings", "batch_size"):
            self._need(
                f"inference.{field}", allow_not_applicable=True
            )

    def _metric_policies(self, *, summary: bool = False) -> None:
        prefix: str = "evaluation.scoring"
        self._many(
            prefix,
            ("task_field_extraction_rules", "invalid_output_policy", "abstention_policy"),
        )
        metrics: set[str] = {task.metric for _, task in self._tasks().values()}
        if "f1" in metrics:
            self._many(
                f"{prefix}.f1", ("averaging", "positive_class_definitions")
            )
            self._need(
                f"{prefix}.f1.probability_thresholds",
                allow_not_applicable=True,
            )
        if "accuracy" in metrics:
            self._need(f"{prefix}.accuracy.exact_match_and_partial_credit_rules")
        if "bleu" in metrics:
            self._many(
                f"{prefix}.bleu",
                (
                    "aggregation_level", "tokenizer", "maximum_ngram_order",
                    "smoothing", "lowercase", "structured_output_serialization",
                ),
            )
        if "roc_auc" in metrics:
            self._need(f"{prefix}.roc_auc.target_definition_manifest", artifact=True)
            self._need(f"{prefix}.roc_auc.missing_probability_policy")
            for name, (stage, task) in self._tasks().items():
                if name in {"exploit_likelihood", "campaign_escalation"}:
                    self._need(f"evaluation.tasks.{stage}.{name}.horizon_days")
                    self._need(
                        f"evaluation.tasks.{stage}.{name}.binary_target_construction"
                    )
        if "ndcg" in metrics:
            self._many(
                f"{prefix}.ndcg", ("cutoff_k", "gain_function", "tie_policy")
            )
            self._need(f"{prefix}.ndcg.relevance_manifest", artifact=True)
        if summary:
            self._need("evaluation.standard_deviation_ddof")

    def _prepare(self) -> None:
        selected: list[str] = (
            self.config.operational.selected_benchmarks
            or list(self.config.data.benchmarks)
        )
        for benchmark in selected:
            prefix: str = f"data.benchmarks.{benchmark}"
            self._need(f"{prefix}.version")
            self._need(f"{prefix}.source_path", artifact=True)
            self._need(f"{prefix}.selection_manifest", artifact=True)
        for field in (
            "evidence_manifest",
            "historical_snapshot_manifest",
            "task_conversion_manifest",
            "reference_answer_manifest",
        ):
            self._need(f"data.{field}", artifact=True)
        self._many(
            "data.preprocessing",
            ("alias_table_version", "taxonomy_versions", "deduplication_policy"),
        )

    def _evaluate(self) -> None:
        self._prerequisite("corpus")
        self._prerequisite("references")
        self._models()
        self._generation()
        for field in ("template_path", "task_guidance_manifest", "output_profile_manifest"):
            self._need(f"evaluation.prompt.{field}", artifact=True)
        seeds: Any = self._need("evaluation.run_seeds", allow_not_applicable=True)
        if isinstance(seeds, list) and any(seed is None for seed in seeds):
            for index, seed in enumerate(seeds):
                if seed is None:
                    self._need(
                        f"evaluation.run_seeds.{index}", allow_not_applicable=True
                    )
        self._metric_policies(summary=True)

    def _identify(self) -> None:
        self._prerequisite("responses")
        self._prerequisite("references")
        self._need("evaluation.scoring.response_level_failure_criteria")
        tasks: dict[str, tuple[str, Any]] = self._tasks()
        hard = self.config.failure_identification.hard_label_rules
        declared = hard.get("tasks", {}) if isinstance(hard, dict) else {}
        routes = {name: declared.get(name, {}).get("route", "similarity" if task.metric == "bleu" else "hard")
                  for name, (_, task) in tasks.items()}
        if "hard" in routes.values():
            self._need("failure_identification.hard_label_rules")
        if "similarity" in routes.values():
            prefix: str = "failure_identification.stratification"
            self._many(
                prefix,
                (
                    "similarity_measure_by_task", "grouping_fields",
                    "anchor_count_per_stratum", "anchor_sampling_seed",
                    "tied_score_policy", "inspection_order",
                    "pass_failure_range_estimator",
                ),
            )
            self._many(
                f"{prefix}.convergence",
                ("distribution_distance", "stability_tolerance", "unvisited_strata_policy"),
            )
            if (
                self.config.quantile_width != 0.05
                and self.stage != "stability"
            ):
                self._provenance(f"{prefix}.quantile_width")
        self._need("taxonomy.guideline_manifest", artifact=True)
        self._metric_policies()

    def _discover(self) -> None:
        self._prerequisite("identification")
        self._need("taxonomy.guideline_manifest", artifact=True)
        mode: Any = self._need("operational.taxonomy_mode", provenance=False)
        if mode == "fixed":
            self._need("taxonomy.discovery_initial_taxonomy")
            self._provenance("operational.taxonomy_mode")
            return
        if self.config.taxonomy.discovery_initial_taxonomy is not None:
            self._error(
                "taxonomy.discovery_initial_taxonomy",
                "de novo discovery cannot start from a supplied fixed taxonomy",
            )
        prefix: str = "taxonomy_discovery"
        self._many(
            prefix,
            (
                "initial_subset_size", "initialization_seed", "assisting_model",
                "coverage_threshold", "coverage_definition",
                "stability_threshold_epsilon", "revisit_prior_assignments_policy",
                "persistent_other_policy",
            ),
        )
        self._need(f"{prefix}.prompt_template_path", artifact=True)
        assisting: Any = self.config.taxonomy_discovery.assisting_model
        if isinstance(assisting, str):
            if assisting in self.config.models.evaluation:
                self._models([assisting])
            else:
                self._error(
                    f"{prefix}.assisting_model",
                    "use a configured model ID or an explicit endpoint/revision object",
                )
        elif isinstance(assisting, dict):
            for key in ("endpoint_or_checkpoint", "revision"):
                if _is_unresolved(assisting.get(key)):
                    self._error(f"{prefix}.assisting_model.{key}", "unresolved model identity")
        self._generation()

    def _categorize(self, *, selected_agent: str | None = None) -> None:
        self._prerequisite("identification")
        self._prerequisite("taxonomy")
        self._need("taxonomy.guideline_manifest", artifact=True)
        for index in range(len(self.config.models.categorization_panel)):
            if selected_agent is not None and self.config.models.categorization_panel[index].name != selected_agent:
                continue
            self._many(
                f"models.categorization_panel.{index}",
                ("endpoint_or_checkpoint", "revision"),
            )
        self._many(
            "categorization",
            ("peer_payload_policy", "label_set_comparison_policy", "incomplete_agent_output_policy"),
        )
        for round_name in (("round_one",) if selected_agent else ("round_one", "round_two")):
            self._need(
                f"categorization.prompt_templates.{round_name}", artifact=True
            )
        self._generation()

    def _audit(self) -> None:
        self._need("taxonomy.guideline_manifest", artifact=True)
        self._prerequisite("predictions")
        audit: Any = self._need("operational.selected_audit", provenance=False)
        if audit == "failure_identification":
            self._prerequisite("identification")
            self._need("audits.failure_identification.sampling_seed")
        elif audit in {"human_agreement", "failure_categorization"}:
            self._need("annotation.agreement_study.uncertain_label_handling")
            self._prerequisite("audit_sampling")
            if audit == "failure_categorization":
                self._need(
                    "audits.failure_categorization.additional_sampling_protocol"
                )
        elif audit is not None:
            self._error("operational.selected_audit", "unknown audit selection")

    def _baseline(self) -> None:
        self._prerequisite("identification")
        self._prerequisite("taxonomy")
        self._many("baselines", ("repetition_count", "prevalence_denominator"))
        name: Any = self._need("operational.selected_baseline", provenance=False)
        protocols: dict[str, tuple[str, ...]] = {
            "kmeans": (
                "embedding_model", "input_representation", "clusters",
                "initialization", "cluster_to_subtype_mapping",
            ),
            "lda": (
                "preprocessing", "topic_count", "priors", "iterations",
                "topic_to_subtype_mapping",
            ),
            "multi_agent_debate": ("rounds", "consensus_policy", "stopping_rule"),
        }
        if name in protocols:
            self._many(f"baselines.{name}", protocols[name])
            if name == "multi_agent_debate":
                self._categorize()
        elif name in {"gpt_judge", "gemini_judge"}:
            model_id: str = "G5" if name == "gpt_judge" else "GEM"
            self._models([model_id])
            self._generation()
            self._need(f"baselines.{name}.prompt_template_path", artifact=True)
        elif name is not None:
            self._error("operational.selected_baseline", "unknown baseline")

    def _ablation(self) -> None:
        component: Any = self._need("operational.selected_ablation", provenance=False)
        self._need("ablations.repetition_count")
        replacements: Any = self._need("ablations.replacement_behavior_by_component")
        if component not in self.config.ablations.components_to_remove:
            self._error("operational.selected_ablation", "select a declared ablation component")
        if isinstance(replacements, dict) and component not in replacements:
            self._error(
                "ablations.replacement_behavior_by_component",
                "selected component lacks an explicit replacement protocol",
            )
        self._prerequisite("responses")
        self._prerequisite("references")
        # Replacement protocols must declare which lower-level stages execute.
        if isinstance(replacements, dict):
            protocol: Any = replacements.get(component)
            affected: Any = protocol.get("affected_stages") if isinstance(protocol, dict) else None
            allowed: dict[str, Any] = {
                "identify": self._identify,
                "discover_taxonomy": self._discover,
                "categorize": self._categorize,
            }
            if not isinstance(affected, list) or not affected:
                self._error(
                    "ablations.replacement_behavior_by_component",
                    "replacement must declare nonempty affected_stages",
                )
            else:
                for stage in affected:
                    if stage not in allowed:
                        self._error(
                            "ablations.replacement_behavior_by_component",
                            "unsupported affected stage",
                        )
                    elif (stage == "categorize" and component == "multi_agent_decision"
                          and protocol.get("strategy") == "single_judge"):
                        judge = protocol.get("judge")
                        if judge not in {"gpt_judge", "gemini_judge"}:
                            self._error("ablations.replacement_behavior_by_component", "select an explicit judge")
                        else:
                            if protocol.get("reconsideration") not in (
                                {"mode": "fresh_reassessment"}, {"mode": "self_review"}
                            ):
                                self._error("ablations.replacement_behavior_by_component", "declare single-agent reconsideration")
                            self._categorize(selected_agent="GPT-5" if judge == "gpt_judge" else "Gemini 2.5")
                    else:
                        allowed[stage]()

    def _stability(self) -> None:
        self._identify()
        self._discover()
        prefix: str = "taxonomy_stability"
        self._many(
            prefix,
            (
                "condition_combination_design", "recovery_match_threshold",
                "unmatched_category_policy", "empty_set_policy",
                "prevalence_normalization", "jsd_logarithm_base",
                "comparison_aggregation",
            ),
        )
        self._need(f"{prefix}.audit_manifest", artifact=True)
        if self.config.operational.taxonomy_mode != "discovery":
            self._error(
                "operational.taxonomy_mode",
                "taxonomy-stability discovery requires discovery mode",
            )
        widths: list[float] = self.config.taxonomy_stability.quantile_widths
        if set(widths) != {0.025, 0.05, 0.10}:
            self._error(f"{prefix}.quantile_widths", "require the three paper-specified widths")

    def _defense(self) -> None:
        name: Any = self._need("operational.selected_defense", provenance=False)
        if name == "counterfactual_consistency":
            self._c3()
        elif name == "tailored_mitigation":
            self._tailored()
        elif name == "stronger_grounding":
            self._grounding()
        elif name is not None:
            self._error(
                "operational.selected_defense",
                "select counterfactual_consistency, tailored_mitigation, or stronger_grounding",
            )

    def _defense_common(self, model_ids: list[str] | None = None, *, require_identification: bool = True) -> None:
        self._prerequisite("corpus")
        self._prerequisite("references")
        self._models(model_ids)
        self._generation()
        self._metric_policies()
        if require_identification:
            self._need("evaluation.scoring.response_level_failure_criteria")

    def _c3(self) -> None:
        self._defense_common(require_identification=False)
        prefix: str = "defenses.counterfactual_consistency"
        self._many(
            prefix,
            (
                "operators", "counterfactuals_per_instance",
                "stability_threshold_tau", "output_equivalence_rule",
                "probability_extraction", "entropy_weight_formula",
                "normalize_weights", "tie_policy",
                "repetition_count",
            ),
        )
        # This domain is fixed by both C3 equations, rather than a configurable
        # implementation choice requiring user-supplied provenance.
        if self._need(f"{prefix}.include_original_input", provenance=False) is not False:
            self._error(f"{prefix}.include_original_input", "C3 correction must be counterfactual-only")
        if any(task.metric == "bleu" for _, task in self._tasks().values()):
            self._need(f"{prefix}.free_text_aggregation")
        operators: Any = self.config.defenses.counterfactual_consistency.operators
        if isinstance(operators, dict) and _is_unresolved(
            operators.get("preservation_protocol")
        ):
            self._error(
                f"{prefix}.operators.preservation_protocol",
                "declare task-dependent answer-preservation validation",
            )
        probability: Any = self.config.defenses.counterfactual_consistency.probability_extraction
        if isinstance(probability, dict) and probability.get("interface") == "autoregressive_complete":
            if probability.get("aggregation_space") != "task_outputs":
                self._error(f"{prefix}.probability_extraction", "complete autoregression must use task_outputs")
            if not isinstance(probability.get("label_pointer"), str) or not probability["label_pointer"].startswith("/"):
                self._error(f"{prefix}.probability_extraction", "specify the exact task-field JSON pointer")
            if self.config.defenses.counterfactual_consistency.output_equivalence_rule != "task_field_exact":
                self._error(f"{prefix}.output_equivalence_rule", "complete outputs require task_field_exact")
            adapter = probability.get("adapter")
            parameters = adapter.get("parameters", {}) if isinstance(adapter, dict) else {}
            if parameters.get("max_new_tokens") != self.config.inference.max_output_tokens:
                self._error(f"{prefix}.probability_extraction", "probability horizon must match generation max_output_tokens")
            budget = parameters.get("max_nodes")
            if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
                self._error(f"{prefix}.probability_extraction", "complete enumeration requires a positive max_nodes budget")
            if "candidate_output_adapter" in probability:
                self._error(f"{prefix}.probability_extraction", "complete outcomes supply their own validated outputs")
        forbidden: re.Pattern[str] = re.compile(
            r"self[_ -]?reported|json[_ -]?confidence|majority[_ -]?vot",
            re.IGNORECASE,
        )
        if probability is not None and forbidden.search(_canonical_json(probability)):
            self._error(
                f"{prefix}.probability_extraction",
                "confidence/voting is not the required model probability interface",
            )

    def _tailored(self) -> None:
        self._defense_common()
        prefix: str = "defenses.tailored_mitigation"
        mode: Any = self._need("operational.selected_mitigation_mode", provenance=False)
        if mode not in self.config.defenses.tailored_mitigation.configurations:
            self._error("operational.selected_mitigation_mode", "select a declared mitigation mode")
        self._many(prefix, ("routing_policy", "model_group_aggregation", "repetition_count"))
        self._need(f"{prefix}.subtype_policy_manifest", artifact=True)
        if mode in {"retrieval_only", "full_fledged"}:
            self._many(
                prefix,
                (
                    "retrieval_models", "chunking_policy", "evidence_context_budget",
                    "source_diversity_parameters", "freshness_parameters",
                    "ioc_reuse_thresholds",
                ),
            )
        if mode in {"inference_only", "full_fledged"}:
            self._need(f"{prefix}.prompt_manifest", artifact=True)
        routing: Any = self.config.defenses.tailored_mitigation.routing_policy
        if isinstance(routing, dict) and routing.get("uses_gold_failure_labels") is not False:
            self._error(
                f"{prefix}.routing_policy",
                "explicitly declare uses_gold_failure_labels: false",
            )

    def _grounding(self) -> None:
        self._defense_common(["G5", "CLD"])
        prefix: str = "defenses.stronger_grounding"
        self._need(f"{prefix}.sampling_seed")
        self._need(f"{prefix}.matched_instance_manifest", artifact=True)
        self._many(
            f"{prefix}.retrieval",
            (
                "sparse_retriever", "dense_retriever", "fusion_rule",
                "retrieval_depth", "cross_encoder_reranker", "reranking_depth",
                "evidence_context_budget",
            ),
        )
        self._many(
            f"{prefix}.evidence_verification",
            ("checkpoint", "claim_segmentation", "alignment_threshold"),
        )
        self._need(
            f"{prefix}.evidence_verification.revision_prompt_template_path",
            artifact=True,
        )

    def run(self) -> None:
        """Collect readiness errors for one selected execution graph."""
        self._need("experiment_id", provenance=False)
        for path in (
            "operational.database_path",
            "operational.artifact_root",
            "operational.output_path",
        ):
            self._need(path, provenance=False)
        if self.stage not in {"report", "audit"}:
            self._need("corpus_version", provenance=False)

        handlers: dict[str, Any] = {
            "prepare": self._prepare,
            "evaluate": self._evaluate,
            "identify": self._identify,
            "discover_taxonomy": self._discover,
            "categorize": self._categorize,
            "audit": self._audit,
            "baselines": self._baseline,
            "ablations": self._ablation,
            "stability": self._stability,
            "defenses": self._defense,
        }
        if self.stage == "report":
            # Reports consume scored artifacts, not live inference credentials.
            self._prerequisite("results")
            self._prerequisite("scoring")
        else:
            handlers[self.stage]()

        # Every changed experimental setting is represented in the hash. Paths
        # that are null in the template require provenance when this stage
        # consumes them; unrelated optional nulls are deliberately not visited.
        for path in self._required:
            choice: _Choice | None = self.config.implementation_choices.get(path)
            if choice is not None:
                affected: set[str] = {
                    _STAGE_ALIASES.get(item, item)
                    for item in choice.affected_experiments
                }
                if self.stage not in affected and "all" not in affected:
                    self._error(
                        path,
                        "provenance does not list the requested affected experiment",
                    )
