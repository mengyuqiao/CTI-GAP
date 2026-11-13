"""Validated configuration inputs and immutable experimental record snapshots.

Only Pydantic and the Python standard library are used here. Configuration
readiness, persistence, inference, scoring, and annotation access control belong
to their respective service modules.

Configuration ``None`` values mean unresolved, not an implementation default.
Record status vocabularies and canonical JSON conventions are engineering
conventions.

FailureDecision explicitly permits None for similarity_score and stratum_id.
This resolves the design's hard-label representation mismatch without inventing
a score or stratum: these fields are inapplicable to hard-label decisions.
Similarity-based decisions must provide both fields.

Frozen Pydantic models are shallowly immutable. Incoming containers are copied,
but authoritative append-only guarantees must be enforced by ArtifactStore.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import date, datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal, Self, TypeAlias

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Shared primitive validation and canonical serialization.
# ---------------------------------------------------------------------------


def _identifier(value: Any) -> str:
    """Validate identifiers without stripping or changing their spelling."""
    if not isinstance(value, str) or not value or not value.strip():
        raise ValueError("An identifier must be a nonempty string.")
    if value != value.strip():
        raise ValueError("Identifiers must not contain surrounding whitespace.")
    return value


def _finite_number(value: Any) -> float:
    """Accept real JSON numbers, but not booleans or numeric strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a number, not a boolean or numeric string.")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("Expected a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError("NaN and infinity are not valid experimental values.")
    return number


def _probability(value: Any) -> float:
    number = _finite_number(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError("Probability must be between zero and one.")
    return number


def _digest(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA-256 hexadecimal digest.")
    return value


def _calendar_date(value: Any) -> date:
    if isinstance(value, datetime):
        raise ValueError("A calendar date must not contain a time.")
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError("Expected an ISO calendar date, YYYY-MM-DD.")
    return date.fromisoformat(value)


def _aware_datetime(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("Expected an ISO timezone-aware timestamp.") from exc
    if not isinstance(value, datetime):
        raise ValueError("Expected a timezone-aware timestamp.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Naive timestamps are not permitted.")
    return value.astimezone(timezone.utc)


def _source_time(value: Any) -> date | datetime:
    """Preserve date-only precision instead of inventing midnight timestamps."""
    if isinstance(value, datetime):
        return _aware_datetime(value)
    if isinstance(value, date):
        return value
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return _calendar_date(value)
    return _aware_datetime(value)


Identifier = Annotated[str, BeforeValidator(_identifier)]
Digest = Annotated[str, BeforeValidator(_digest)]
FiniteFloat = Annotated[float, BeforeValidator(_finite_number)]
Probability = Annotated[float, BeforeValidator(_probability)]
NonNegativeFloat = Annotated[FiniteFloat, Field(ge=0)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]
CalendarDate = Annotated[date, BeforeValidator(_calendar_date)]
UtcTimestamp = Annotated[datetime, BeforeValidator(_aware_datetime)]
SourceTime = Annotated[date | datetime, BeforeValidator(_source_time)]
JsonObject: TypeAlias = dict[str, JsonValue]
PolicyValue: TypeAlias = StrictStr | JsonObject | list[JsonValue] | None


def _json_primitive(value: Any) -> JsonValue:
    """Convert supported values to deterministic JSON-compatible primitives."""
    if isinstance(value, Enum):
        return _json_primitive(value.value)
    if isinstance(value, BaseModel):
        return _json_primitive(value.model_dump(mode="python"))
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Canonical JSON does not permit NaN or infinity.")
        return value
    if isinstance(value, datetime):
        normalized = _aware_datetime(value)
        return normalized.isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Canonical JSON object keys must be strings.")
        return {key: _json_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_primitive(item) for item in value]
    raise ValueError(f"Unsupported canonical JSON type: {type(value).__name__}.")


def _canonical_json(value: Any) -> str:
    """Serialize deterministically; ordered sequences retain their ordering."""
    return json.dumps(
        _json_primitive(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _content_hash(value: Any) -> str:
    """Hash canonical structured content with SHA-256."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _text_hash(text: str) -> str:
    """Hash exact UTF-8 text, without JSON quoting or newline normalization."""
    if not isinstance(text, str):
        raise TypeError("Exact text hashing requires a string.")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_unique(values: list[Any], description: str) -> None:
    identities = [_canonical_json(value) for value in values]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{description} must not contain duplicates.")


def _ordered_times(
    start: date | datetime | None,
    end: date | datetime | None,
    description: str,
) -> None:
    """Compare exact timestamps, or dates when either source has date precision."""
    if start is None or end is None:
        return
    if isinstance(start, datetime) and isinstance(end, datetime):
        invalid = start > end
    else:
        start_date = start.date() if isinstance(start, datetime) else start
        end_date = end.date() if isinstance(end, datetime) else end
        invalid = start_date > end_date
    if invalid:
        raise ValueError(f"{description}: start must not be later than end.")


class SchemaModel(BaseModel):
    """Strict, copied-input model with shallowly frozen validated attributes."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    @model_validator(mode="before")
    @classmethod
    def _copy_input(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return copy.deepcopy(dict(value))
        return value

    @model_validator(mode="after")
    def _validate_json_values(self) -> Self:
        # This also rejects nonfinite values buried inside otherwise open JSON.
        _canonical_json(self.model_dump(mode="python"))
        return self


# ---------------------------------------------------------------------------
# Configuration input schemas. ExperimentConfig itself belongs in config.py.
# ---------------------------------------------------------------------------


class StudyInput(SchemaModel):
    title: Identifier
    execution_mode: Literal["inference_and_analysis"]
    random_seed: StrictInt | None = None


class TrainingInput(SchemaModel):
    required: StrictBool = False
    learning_rate: NonNegativeFloat | None = None
    batch_size: PositiveInt | None = None
    epochs: NonNegativeInt | None = None
    optimizer: StrictStr | None = None
    weight_decay: NonNegativeFloat | None = None
    learning_rate_scheduler: StrictStr | None = None
    gradient_accumulation_steps: PositiveInt | None = None


class BenchmarkInput(SchemaModel):
    reported_original_instances: NonNegativeInt
    version: StrictStr | None = None
    source_path: StrictStr | None = None
    selection_manifest: StrictStr | None = None


class PreprocessingInput(SchemaModel):
    normalize_identifiers: StrictBool
    align_prediction_chronology: StrictBool
    construct_reference_entity_mappings: StrictBool
    preserve_source_provenance: StrictBool
    alias_table_version: StrictStr | None = None
    taxonomy_versions: PolicyValue = None
    deduplication_policy: PolicyValue = None


class DataInput(SchemaModel):
    benchmarks: dict[Identifier, BenchmarkInput]
    evidence_sources: list[Identifier]
    preprocessing: PreprocessingInput
    task_counts_are_approximate: StrictBool
    task_counts_may_overlap: StrictBool
    evidence_manifest: StrictStr | None = None
    historical_snapshot_manifest: StrictStr | None = None
    task_conversion_manifest: StrictStr | None = None
    reference_answer_manifest: StrictStr | None = None


class ModelEndpointInput(SchemaModel):
    name: Identifier
    endpoint_or_checkpoint: StrictStr | None = None
    revision: StrictStr | None = None


class ModelsInput(SchemaModel):
    evaluation: dict[Identifier, ModelEndpointInput]
    categorization_panel: list[ModelEndpointInput]


class TemplateRangesInput(SchemaModel):
    temperature: tuple[FiniteFloat, FiniteFloat]
    top_p: tuple[Probability, Probability]

    @model_validator(mode="after")
    def _validate_ranges(self) -> Self:
        if self.temperature[0] > self.temperature[1]:
            raise ValueError("Temperature range must be ordered.")
        if self.top_p[0] > self.top_p[1]:
            raise ValueError("Top-p range must be ordered.")
        return self


class InferenceInput(SchemaModel):
    paper_template_ranges: TemplateRangesInput
    temperature: NonNegativeFloat | None = None
    top_p: Probability | None = None
    max_output_tokens: PositiveInt | None = None
    context_window: PositiveInt | None = None
    batch_size: PositiveInt | None = None
    concurrency: PositiveInt | None = None
    precision: StrictStr | None = None
    quantization: PolicyValue = None
    reasoning_settings: PolicyValue = None
    truncation_policy: PolicyValue = None
    retry_policy: PolicyValue = None


class EvaluationPromptInput(SchemaModel):
    json_only: StrictBool
    supplied_evidence_only: StrictBool
    external_browsing: StrictBool
    enforce_snapshot_date: StrictBool
    require_evidence_references: StrictBool
    invent_identifiers: StrictBool
    insufficient_context_status: Identifier
    allowed_statuses: list[Identifier]
    maximum_justification_words: PositiveInt
    offensive_guidance_allowed: StrictBool
    template_path: StrictStr | None = None
    task_guidance_manifest: StrictStr | None = None
    output_profile_manifest: StrictStr | None = None

    @model_validator(mode="after")
    def _validate_statuses(self) -> Self:
        _check_unique(self.allowed_statuses, "Allowed answer statuses")
        if self.insufficient_context_status not in self.allowed_statuses:
            raise ValueError("Insufficient-context status must be allowed.")
        return self


MetricName: TypeAlias = Literal["f1", "accuracy", "bleu", "roc_auc", "ndcg"]


class TaskDefinitionInput(SchemaModel):
    approximate_instances: NonNegativeInt
    metric: MetricName
    horizon_days: PositiveInt | None = None
    binary_target_construction: PolicyValue = None


class F1ScoringInput(SchemaModel):
    averaging: StrictStr | None = None
    positive_class_definitions: PolicyValue | StrictInt = None
    probability_thresholds: PolicyValue | FiniteFloat = None


class AccuracyScoringInput(SchemaModel):
    exact_match_and_partial_credit_rules: PolicyValue = None


class BleuScoringInput(SchemaModel):
    aggregation_level: StrictStr | None = None
    tokenizer: StrictStr | None = None
    maximum_ngram_order: PositiveInt | None = None
    smoothing: PolicyValue = None
    lowercase: StrictBool | None = None
    structured_output_serialization: PolicyValue = None


class AucScoringInput(SchemaModel):
    target_definition_manifest: StrictStr | None = None
    missing_probability_policy: PolicyValue = None


class NdcgScoringInput(SchemaModel):
    cutoff_k: PositiveInt | None = None
    gain_function: PolicyValue = None
    relevance_manifest: StrictStr | None = None
    tie_policy: PolicyValue = None


class ScoringInput(SchemaModel):
    f1: F1ScoringInput
    accuracy: AccuracyScoringInput
    bleu: BleuScoringInput
    roc_auc: AucScoringInput
    ndcg: NdcgScoringInput
    task_field_extraction_rules: PolicyValue = None
    response_level_failure_criteria: PolicyValue = None
    invalid_output_policy: PolicyValue = None
    abstention_policy: PolicyValue = None


class EvaluationInput(SchemaModel):
    runs: PositiveInt
    report_mean: StrictBool
    report_standard_deviation: StrictBool
    reporting_scale: tuple[FiniteFloat, FiniteFloat]
    prompt: EvaluationPromptInput
    tasks: dict[Identifier, dict[Identifier, TaskDefinitionInput]]
    scoring: ScoringInput
    run_seeds: list[StrictInt | None] | None = None
    standard_deviation_ddof: NonNegativeInt | None = None

    @model_validator(mode="after")
    def _validate_relationships(self) -> Self:
        if self.reporting_scale[0] >= self.reporting_scale[1]:
            raise ValueError("Reporting scale must have increasing bounds.")
        if self.run_seeds is not None and len(self.run_seeds) != self.runs:
            raise ValueError("run_seeds must have one entry per evaluation run.")
        if (
            self.standard_deviation_ddof is not None
            and self.standard_deviation_ddof >= self.runs
        ):
            raise ValueError("Standard-deviation ddof must be smaller than runs.")
        task_names: list[str] = [
            name for tasks in self.tasks.values() for name in tasks
        ]
        _check_unique(task_names, "Task names across stages")
        return self


class ConvergenceInput(SchemaModel):
    require_stable_failure_distribution: StrictBool
    require_no_new_failure_set: StrictBool
    distribution_distance: PolicyValue = None
    stability_tolerance: NonNegativeFloat | None = None
    unvisited_strata_policy: PolicyValue = None


class StratificationInput(SchemaModel):
    quantile_width: Probability
    manually_review_boundary_cases: StrictBool
    convergence: ConvergenceInput
    similarity_measure_by_task: PolicyValue = None
    grouping_fields: list[Identifier] | None = None
    anchor_count_per_stratum: PositiveInt | None = None
    anchor_sampling_seed: StrictInt | None = None
    tied_score_policy: PolicyValue = None
    inspection_order: PolicyValue = None
    pass_failure_range_estimator: PolicyValue = None

    @field_validator("quantile_width")
    @classmethod
    def _positive_width(cls, value: float) -> float:
        if value == 0:
            raise ValueError("Quantile width must be positive.")
        return value


class FailureIdentificationInput(SchemaModel):
    stratification: StratificationInput
    reported_manual_inspection_fraction_upper_bound: Probability
    hard_label_rules: PolicyValue = None


class TaxonomyDiscoveryInput(SchemaModel):
    fallback_label: Identifier
    human_review_other_cases: StrictBool
    add_human_supported_new_categories: StrictBool
    stop_when_no_new_categories: StrictBool
    initial_subset_size: PositiveInt | None = None
    initialization_seed: StrictInt | None = None
    assisting_model: PolicyValue = None
    prompt_template_path: StrictStr | None = None
    coverage_threshold: Probability | None = None
    coverage_definition: PolicyValue = None
    stability_threshold_epsilon: NonNegativeFloat | None = None
    revisit_prior_assignments_policy: PolicyValue = None
    persistent_other_policy: PolicyValue = None


class TaxonomyInput(SchemaModel):
    multiple_labels_allowed: StrictBool
    require_evidence_for_each_label: StrictBool
    published_reference: dict[Identifier, dict[Identifier, Identifier]]
    guideline_manifest: StrictStr | None = None
    discovery_initial_taxonomy: PolicyValue = None

    @model_validator(mode="after")
    def _validate_reference_ids(self) -> Self:
        subtype_ids = [
            subtype_id
            for subtypes in self.published_reference.values()
            for subtype_id in subtypes
        ]
        _check_unique(subtype_ids, "Published subtype identifiers")
        return self


class CategorizationTemplatesInput(SchemaModel):
    round_one: StrictStr | None = None
    round_two: StrictStr | None = None


class CategorizationReviewInput(SchemaModel):
    trigger_on_first_round_disagreement: StrictBool
    trigger_on_any_agent_cross_round_change: StrictBool
    introduce_new_categories: StrictBool


class CategorizationInput(SchemaModel):
    rounds: PositiveInt
    first_round_independent: StrictBool
    second_round_observes_first_round_peers: StrictBool
    prompt_templates: CategorizationTemplatesInput
    human_review: CategorizationReviewInput
    reported_human_verification_fraction_upper_bound: Probability
    peer_payload_policy: PolicyValue = None
    label_set_comparison_policy: PolicyValue = None
    incomplete_agent_output_policy: PolicyValue = None


class AnnotatorPoolInput(SchemaModel):
    annotators: PositiveInt
    graduate_researchers: NonNegativeInt
    professional_cybersecurity_researchers: NonNegativeInt
    experience_years_range: tuple[NonNegativeInt, NonNegativeInt]
    senior_adjudicator_experience_years: NonNegativeInt

    @model_validator(mode="after")
    def _validate_pool(self) -> Self:
        total = (
            self.graduate_researchers
            + self.professional_cybersecurity_researchers
        )
        if self.annotators != total:
            raise ValueError("Annotator pool counts do not sum to annotators.")
        if self.experience_years_range[0] > self.experience_years_range[1]:
            raise ValueError("Experience range must be ordered.")
        return self


class CalibrationInput(SchemaModel):
    instances: PositiveInt
    cover_four_stages: StrictBool
    cover_three_high_level_categories: StrictBool
    independent_initial_annotation: StrictBool
    discuss_disagreements: StrictBool
    exclude_from_agreement_statistics: StrictBool
    freeze_guideline_before_formal_reliability_study: StrictBool


class AgreementStudyInput(SchemaModel):
    instances: PositiveInt
    annotators_per_instance: PositiveInt
    stratify_by: list[Identifier]
    preserve_independent_labels: StrictBool
    metrics: list[Identifier]
    uncertain_label_handling: PolicyValue = None


class AdjudicationInput(SchemaModel):
    review_reference_evidence: StrictBool
    apply_frozen_inclusion_exclusion_criteria: StrictBool
    escalate_unresolved_cases_to_senior: StrictBool
    use_adjudicated_labels_for_agreement_statistics: StrictBool


class AnnotationInput(SchemaModel):
    pool: AnnotatorPoolInput
    calibration: CalibrationInput
    agreement_study: AgreementStudyInput
    adjudication: AdjudicationInput


class IdentificationAuditInput(SchemaModel):
    instances: PositiveInt
    sampling: Identifier
    population: Identifier
    exclude: list[Identifier]
    independent_annotators: PositiveInt
    disagreement_adjudicator: StrictBool
    blind_to: list[Identifier]
    freeze_identification_before_sampling: StrictBool
    use_audit_labels_for_tuning: StrictBool
    positive_class: Identifier
    metrics: list[Identifier]
    zero_denominator_result: StrictStr
    sampling_seed: StrictInt | None = None


class CategorizationAuditInput(SchemaModel):
    original_agreement_instances: PositiveInt
    additional_instances: NonNegativeInt
    total_instances: PositiveInt
    include_stable_consensus_cases: StrictBool
    independent_human_references: StrictBool
    evaluate_predictions_before_human_correction: StrictBool
    count_audit_work_as_operational_verification: StrictBool
    metrics: list[Identifier]
    zero_denominator_result: StrictStr
    additional_sampling_protocol: PolicyValue = None

    @model_validator(mode="after")
    def _validate_counts(self) -> Self:
        if self.total_instances != (
            self.original_agreement_instances + self.additional_instances
        ):
            raise ValueError("Expanded audit size must equal original plus added.")
        return self


class AuditsInput(SchemaModel):
    failure_identification: IdentificationAuditInput
    failure_categorization: CategorizationAuditInput


class KMeansInput(SchemaModel):
    embedding_model: PolicyValue = None
    input_representation: PolicyValue = None
    clusters: PositiveInt | None = None
    initialization: PolicyValue = None
    cluster_to_subtype_mapping: PolicyValue = None


class LdaInput(SchemaModel):
    preprocessing: PolicyValue = None
    topic_count: PositiveInt | None = None
    priors: PolicyValue = None
    iterations: PositiveInt | None = None
    topic_to_subtype_mapping: PolicyValue = None


class JudgeInput(SchemaModel):
    model: Identifier
    prompt_template_path: StrictStr | None = None


class DebateInput(SchemaModel):
    panel: Identifier
    rounds: PositiveInt | None = None
    consensus_policy: PolicyValue = None
    stopping_rule: PolicyValue = None


class BaselinesInput(SchemaModel):
    kmeans: KMeansInput
    lda: LdaInput
    gpt_judge: JudgeInput
    gemini_judge: JudgeInput
    multi_agent_debate: DebateInput
    repetition_count: PositiveInt | None = None
    prevalence_denominator: PolicyValue = None


class AblationsInput(SchemaModel):
    components_to_remove: list[Identifier]
    replacement_behavior_by_component: PolicyValue = None
    repetition_count: PositiveInt | None = None


class TaxonomyStabilityInput(SchemaModel):
    perturbations: list[Identifier]
    quantile_widths: list[Probability]
    independent_runs_per_condition: PositiveInt
    audit_instances: PositiveInt
    category_matching: Identifier
    matching_weight: Identifier
    metrics: list[Identifier]
    condition_combination_design: PolicyValue = None
    audit_manifest: StrictStr | None = None
    recovery_match_threshold: Probability | None = None
    unmatched_category_policy: PolicyValue = None
    empty_set_policy: PolicyValue = None
    prevalence_normalization: PolicyValue = None
    jsd_logarithm_base: FiniteFloat | None = None
    comparison_aggregation: PolicyValue = None

    @model_validator(mode="after")
    def _validate_parameters(self) -> Self:
        if any(width == 0 for width in self.quantile_widths):
            raise ValueError("Stability quantile widths must be positive.")
        _check_unique(self.quantile_widths, "Stability quantile widths")
        if (
            self.jsd_logarithm_base is not None
            and (
                self.jsd_logarithm_base <= 0
                or self.jsd_logarithm_base == 1
            )
        ):
            raise ValueError("Logarithm base must be positive and not one.")
        return self


class CounterfactualInput(SchemaModel):
    intervention_families: list[Identifier]
    preserve_answer_determining_attributes: StrictBool
    require_task_dependent_preservation_justification: StrictBool
    consistency_is_not_correctness: StrictBool
    use_self_reported_confidence_as_probability: StrictBool
    operators: PolicyValue = None
    counterfactuals_per_instance: PositiveInt | None = None
    stability_threshold_tau: Probability | None = None
    output_equivalence_rule: PolicyValue = None
    probability_extraction: PolicyValue = None
    entropy_weight_formula: PolicyValue = None
    normalize_weights: StrictBool | None = None
    # The published correction sums over generated counterfactuals only.
    include_original_input: StrictBool | None = False
    tie_policy: PolicyValue = None
    free_text_aggregation: PolicyValue = None
    repetition_count: PositiveInt | None = None


class TailoredMitigationInput(SchemaModel):
    configurations: list[Identifier]
    subtype_policy_manifest: StrictStr | None = None
    retrieval_models: PolicyValue = None
    chunking_policy: PolicyValue = None
    evidence_context_budget: PositiveInt | None = None
    source_diversity_parameters: PolicyValue = None
    freshness_parameters: PolicyValue = None
    ioc_reuse_thresholds: PolicyValue = None
    prompt_manifest: StrictStr | None = None
    routing_policy: PolicyValue = None
    model_group_aggregation: PolicyValue = None
    repetition_count: PositiveInt | None = None


class RetrievalInput(SchemaModel):
    same_corpus_snapshot_between_grounding_variants: StrictBool
    sparse_retriever: PolicyValue = None
    dense_retriever: PolicyValue = None
    fusion_rule: PolicyValue = None
    retrieval_depth: PositiveInt | None = None
    cross_encoder_reranker: PolicyValue = None
    reranking_depth: PositiveInt | None = None
    evidence_context_budget: PositiveInt | None = None


class EvidenceVerificationInput(SchemaModel):
    method: Identifier
    revision_passes: PositiveInt
    use_verifier_scores_as_final_labels: StrictBool
    checkpoint: StrictStr | None = None
    claim_segmentation: PolicyValue = None
    alignment_threshold: Probability | None = None
    revision_prompt_template_path: StrictStr | None = None


class GroundingControlsInput(SchemaModel):
    same_instances: StrictBool
    same_task_inputs: StrictBool
    same_references: StrictBool
    same_decoding_settings: StrictBool
    same_failure_identification_criteria: StrictBool


class StrongerGroundingInput(SchemaModel):
    tasks: list[Identifier]
    models: list[Identifier]
    instances_per_task: PositiveInt
    distinct_task_instances: PositiveInt
    model_instance_pairs_per_configuration: PositiveInt
    runs: PositiveInt
    report_standard_deviation: StrictBool
    configurations: list[Identifier]
    retrieval: RetrievalInput
    evidence_verification: EvidenceVerificationInput
    comparison_controls: GroundingControlsInput
    metrics: list[Identifier]
    sampling_seed: StrictInt | None = None
    matched_instance_manifest: StrictStr | None = None

    @model_validator(mode="after")
    def _validate_sample_arithmetic(self) -> Self:
        _check_unique(self.tasks, "Grounding tasks")
        _check_unique(self.models, "Grounding models")
        _check_unique(self.configurations, "Grounding configurations")
        if self.distinct_task_instances != len(self.tasks) * self.instances_per_task:
            raise ValueError("Grounding task-instance count is inconsistent.")
        expected_pairs = self.distinct_task_instances * len(self.models)
        if self.model_instance_pairs_per_configuration != expected_pairs:
            raise ValueError("Grounding model-instance pair count is inconsistent.")
        return self


class DefensesInput(SchemaModel):
    counterfactual_consistency: CounterfactualInput
    tailored_mitigation: TailoredMitigationInput
    stronger_grounding: StrongerGroundingInput


class ReproducibilityInput(SchemaModel):
    freeze_corpus_and_reference_versions: StrictBool
    freeze_prompts_and_generation_settings: StrictBool
    retain_raw_model_outputs: StrictBool
    retain_pre_correction_predictions: StrictBool
    retain_independent_human_labels: StrictBool
    separate_operational_review_from_audits: StrictBool
    record_implementation_choices: StrictBool
    silently_substitute_models: StrictBool
    silently_fill_unspecified_settings: StrictBool


class ConfigurationInput(SchemaModel):
    """Actual config.yaml structure, with no service methods or readiness logic."""

    study: StudyInput
    training: TrainingInput
    data: DataInput
    models: ModelsInput
    inference: InferenceInput
    evaluation: EvaluationInput
    failure_identification: FailureIdentificationInput
    taxonomy_discovery: TaxonomyDiscoveryInput
    taxonomy: TaxonomyInput
    categorization: CategorizationInput
    annotation: AnnotationInput
    audits: AuditsInput
    baselines: BaselinesInput
    ablations: AblationsInput
    taxonomy_stability: TaxonomyStabilityInput
    defenses: DefensesInput
    reproducibility: ReproducibilityInput

    @model_validator(mode="after")
    def _validate_cross_section_relationships(self) -> Self:
        original_size = (
            self.audits.failure_categorization.original_agreement_instances
        )
        if original_size != self.annotation.agreement_study.instances:
            raise ValueError("Original label audit must match agreement-study size.")

        known_tasks = {
            name
            for stage_tasks in self.evaluation.tasks.values()
            for name in stage_tasks
        }
        grounding = self.defenses.stronger_grounding
        if not set(grounding.tasks).issubset(known_tasks):
            raise ValueError("Grounding references undefined evaluation tasks.")
        if not set(grounding.models).issubset(self.models.evaluation):
            raise ValueError("Grounding references undefined evaluation models.")
        return self


# ---------------------------------------------------------------------------
# Shared experimental enums.
# ---------------------------------------------------------------------------


class CTIStage(str, Enum):
    CONTEXTUALIZATION = "CONTEXTUALIZATION"
    ATTRIBUTION = "ATTRIBUTION"
    PREDICTION = "PREDICTION"
    MITIGATION = "MITIGATION"


class EvidenceRole(str, Enum):
    MODEL_INPUT = "model_input"
    REFERENCE_ONLY = "reference_only"
    OUTCOME_REFERENCE = "outcome_reference"


class ResponseStatus(str, Enum):
    SUCCESS = "success"
    TRANSPORT_FAILURE = "transport_failure"
    REFUSAL = "refusal"
    INVALID_SCHEMA = "invalid_schema"


class FailureStatus(str, Enum):
    PASS = "pass"
    FAILURE = "failure"
    UNRESOLVED = "unresolved"


class DecisionOrigin(str, Enum):
    HARD_LABEL = "hard_label"
    AUTOMATIC_RANGE = "automatic_range"
    HUMAN_ANCHOR = "human_anchor"
    HUMAN_BOUNDARY = "human_boundary"
    PENDING_REVIEW = "pending_review"
    NOT_PROCESSED = "not_processed"


class AssignmentOrigin(str, Enum):
    AGENT = "agent"
    DISCOVERY = "discovery"
    HUMAN_INITIALIZATION = "human_initialization"
    HUMAN_REFINEMENT = "human_refinement"
    CONSENSUS = "consensus"
    OPERATIONAL_ADJUDICATION = "operational_adjudication"
    INDEPENDENT_ANNOTATION = "independent_annotation"
    AUDIT_ADJUDICATION = "audit_adjudication"


class AnnotationPurpose(str, Enum):
    CALIBRATION = "calibration"
    IDENTIFICATION_ANCHOR = "identification_anchor"
    IDENTIFICATION_BOUNDARY = "identification_boundary"
    TAXONOMY_INITIALIZATION = "taxonomy_initialization"
    TAXONOMY_REFINEMENT = "taxonomy_refinement"
    OPERATIONAL_ADJUDICATION = "operational_adjudication"
    INDEPENDENT_IDENTIFICATION_AUDIT = "independent_identification_audit"
    HUMAN_AGREEMENT = "human_agreement"
    INDEPENDENT_LABEL_AUDIT = "independent_label_audit"
    EXPANDED_LABEL_AUDIT = "expanded_label_audit"
    AUDIT_ADJUDICATION = "audit_adjudication"
    SENIOR_ESCALATION = "senior_escalation"


class AnnotationStatus(str, Enum):
    DRAFT = "draft"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    ADJUDICATED = "adjudicated"
    CANCELLED = "cancelled"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_FOR_ANNOTATIONS = "waiting_for_annotations"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


# ---------------------------------------------------------------------------
# Provenance-aware task and evidence records.
# ---------------------------------------------------------------------------


class EvidenceRecord(SchemaModel):
    evidence_id: Identifier
    source_id: Identifier
    document_id: Identifier
    content_reference: Identifier
    role: EvidenceRole
    source_url: StrictStr | None = None
    source_version: StrictStr | None = None
    content_hash: Digest | None = None
    snippet: StrictStr | None = None
    publication_time: SourceTime | None = None
    modification_time: SourceTime | None = None
    retrieval_time: UtcTimestamp | None = None
    claim_valid_from: SourceTime | None = None
    claim_valid_until: SourceTime | None = None
    original_time_strings: dict[Identifier, StrictStr] = Field(default_factory=dict)
    aliases: list[Identifier] = Field(default_factory=list)
    claims: list[JsonObject] = Field(default_factory=list)
    provenance: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_temporal_scope(self) -> Self:
        _ordered_times(
            self.claim_valid_from,
            self.claim_valid_until,
            "Claim validity",
        )
        # Publication and retrieval do not establish historical eligibility.
        return self


class TaskInstance(SchemaModel):
    instance_id: Identifier
    case_id: Identifier
    source_dataset: Identifier
    source_item_id: Identifier
    task_name: Identifier
    stage: CTIStage
    snapshot_date: CalendarDate
    evidence_records: list[EvidenceRecord] = Field(default_factory=list)
    candidates: list[JsonValue] = Field(default_factory=list)
    environment: JsonObject = Field(default_factory=dict)
    reference: JsonObject = Field(default_factory=dict)
    transformation_log: JsonObject = Field(default_factory=dict)
    output_profile: JsonObject = Field(default_factory=dict)
    content_hash: Digest | None = None

    @model_validator(mode="after")
    def _validate_instance(self, info: ValidationInfo) -> Self:
        _check_unique(
            [record.evidence_id for record in self.evidence_records],
            "Evidence identifiers",
        )
        context = info.context or {}
        configured_tasks = context.get("evaluation_tasks")
        if configured_tasks is not None:
            stage_tasks = configured_tasks.get(self.stage.value.lower(), {})
            if self.task_name not in stage_tasks:
                raise ValueError("Task is not configured for this CTI stage.")
        if self.content_hash is not None:
            expected = _content_hash(_task_content_projection(self))
            if self.content_hash != expected:
                raise ValueError("Task content_hash does not match its content.")
        if context.get("require_content_hash", False) and self.content_hash is None:
            raise ValueError("Finalized task instances require content_hash.")
        return self


def _task_content_projection(instance: TaskInstance) -> JsonObject:
    """Define task hashing, excluding only the self-referential digest field."""
    payload = instance.model_dump(mode="python", exclude={"content_hash"})
    return _json_primitive(payload)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Model responses and declared probability fields.
# ---------------------------------------------------------------------------


# Only paths explicitly identified as probabilities in the paper's contract.
# "*" matches array elements, not arbitrary dictionary keys.
_PROBABILITY_PATHS: tuple[tuple[str, ...], ...] = (
    ("confidence",),
    ("answer", "vuln_linking", "cve_candidates", "*", "score"),
    ("answer", "malware_mapping", "family_candidates", "*", "score"),
    ("answer", "actor_linking", "actor_candidates", "*", "score"),
    ("answer", "campaign_attribution", "score"),
    ("answer", "false_flag", "likelihood"),
    ("answer", "exploit_likelihood", "prob_exploit"),
    ("answer", "impact_forecast", "uncertainty"),
    ("answer", "target_sector", "*", "prob"),
    ("answer", "escalation", "prob"),
)


def _validate_probability_path(value: Any, path: tuple[str, ...]) -> None:
    if not path:
        _probability(value)
        return
    head, *tail = path
    if head == "*":
        if not isinstance(value, list):
            raise ValueError("A declared probability array must be a list.")
        for item in value:
            _validate_probability_path(item, tuple(tail))
    elif isinstance(value, Mapping) and head in value:
        _validate_probability_path(value[head], tuple(tail))


class ModelResponse(SchemaModel):
    response_id: Identifier
    instance_id: Identifier
    model_id: Identifier
    run_index: NonNegativeInt
    prompt_hash: Digest
    config_hash: Digest
    status: ResponseStatus
    raw_text: StrictStr = ""
    parsed_answer: JsonObject = Field(default_factory=dict)
    usage: JsonObject = Field(default_factory=dict)
    schema_errors: list[StrictStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_response(self, info: ValidationInfo) -> Self:
        context = info.context or {}
        if self.status == ResponseStatus.TRANSPORT_FAILURE:
            if not self.usage.get("execution_error"):
                raise ValueError("Transport failure requires usage.execution_error.")
        elif self.status == ResponseStatus.INVALID_SCHEMA:
            if not self.schema_errors:
                raise ValueError("Invalid schema requires recorded schema_errors.")
        elif self.status == ResponseStatus.SUCCESS:
            if self.schema_errors:
                raise ValueError("Successful responses cannot have schema_errors.")
            if not self.parsed_answer:
                raise ValueError("Success requires a parsed output object.")

            # A schema may be supplied for task outputs or agent assignments.
            output_schema = context.get("output_schema")
            if output_schema is not None:
                if not (
                    isinstance(output_schema, type)
                    and issubclass(output_schema, BaseModel)
                ):
                    raise ValueError("output_schema must be a Pydantic model class.")
                output_schema.model_validate(
                    self.parsed_answer,
                    context=context.get("output_context"),
                )
            elif "allowed_answer_statuses" in context:
                answer_status = self.parsed_answer.get("status")
                if answer_status not in context["allowed_answer_statuses"]:
                    raise ValueError("Parsed semantic answer status is not allowed.")
                if not isinstance(self.parsed_answer.get("answer"), dict):
                    raise ValueError("Task output requires an answer object.")

            for path in _PROBABILITY_PATHS:
                _validate_probability_path(self.parsed_answer, path)
            for custom_path in context.get("probability_paths", ()):
                if not isinstance(custom_path, (list, tuple)) or not all(
                    isinstance(part, str) for part in custom_path
                ):
                    raise ValueError("Probability paths must be string sequences.")
                _validate_probability_path(
                    self.parsed_answer, tuple(custom_path)
                )

        # Invalid output is deliberately not validated as successful content:
        # malformed probabilities remain available in raw_text for analysis.
        for time_key in ("started_at", "completed_at"):
            timestamp = self.usage.get(time_key)
            if timestamp is not None:
                _aware_datetime(timestamp)
        return self


class FailureDecision(SchemaModel):
    response_id: Identifier
    identification_version: Identifier
    decision: FailureStatus
    decision_origin: DecisionOrigin
    similarity_score: FiniteFloat | None = None
    stratum_id: Identifier | None = None
    evidence_refs: list[Identifier] = Field(default_factory=list)
    requires_review: StrictBool = False

    @model_validator(mode="after")
    def _validate_decision(self) -> Self:
        if (self.similarity_score is None) != (self.stratum_id is None):
            raise ValueError("Similarity score and stratum must be supplied together.")
        if self.decision_origin == DecisionOrigin.AUTOMATIC_RANGE:
            if self.similarity_score is None:
                raise ValueError("Range propagation requires score and stratum.")
        if self.decision_origin == DecisionOrigin.HARD_LABEL:
            if self.similarity_score is not None or self.stratum_id is not None:
                raise ValueError("Hard-label decisions do not have similarity strata.")
        pending_origins = {
            DecisionOrigin.PENDING_REVIEW,
            DecisionOrigin.NOT_PROCESSED,
        }
        if self.decision_origin in pending_origins:
            if self.decision != FailureStatus.UNRESOLVED:
                raise ValueError("Pending/unprocessed decisions must be unresolved.")
        if self.decision == FailureStatus.UNRESOLVED:
            if not self.requires_review:
                raise ValueError("Unresolved decisions must remain review-required.")
        elif self.requires_review:
            raise ValueError("Final pass/failure decisions cannot require review.")
        return self


# ---------------------------------------------------------------------------
# Taxonomy and assignment records.
# ---------------------------------------------------------------------------


class SubtypeDefinition(SchemaModel):
    subtype_id: Identifier
    name: Identifier
    definition: Identifier
    category: Identifier | None = None
    evidence_requirements: list[Identifier] = Field(default_factory=list)
    supporting_annotation_ids: list[Identifier] = Field(default_factory=list)
    positive_examples: list[JsonValue] = Field(default_factory=list)
    negative_examples: list[JsonValue] = Field(default_factory=list)
    provenance: JsonObject = Field(default_factory=dict)


class SubtypeRules(SchemaModel):
    inclusion: list[Identifier] = Field(default_factory=list)
    exclusion: list[Identifier] = Field(default_factory=list)
    boundary_rules: list[Identifier] = Field(default_factory=list)
    permitted_cooccurring_labels: list[Identifier] = Field(default_factory=list)


class TaxonomyVersion(SchemaModel):
    taxonomy_id: Identifier
    iteration: NonNegativeInt
    guideline_hash: Digest
    parent_id: Identifier | None = None
    subtype_definitions: list[SubtypeDefinition] = Field(default_factory=list)
    inclusion_exclusion_rules: dict[Identifier, SubtypeRules] = Field(
        default_factory=dict
    )
    frozen: StrictBool = False

    @model_validator(mode="after")
    def _validate_taxonomy(self) -> Self:
        if self.parent_id == self.taxonomy_id:
            raise ValueError("A taxonomy cannot be its own parent.")
        subtype_ids = [item.subtype_id for item in self.subtype_definitions]
        _check_unique(subtype_ids, "Subtype identifiers")
        known_ids = set(subtype_ids)
        if not set(self.inclusion_exclusion_rules).issubset(known_ids):
            raise ValueError("Taxonomy rules reference an unknown subtype.")
        for rules in self.inclusion_exclusion_rules.values():
            if not set(rules.permitted_cooccurring_labels).issubset(known_ids):
                raise ValueError("Co-occurrence rules reference unknown subtypes.")
        if self.frozen:
            if not subtype_ids:
                raise ValueError("A frozen taxonomy must contain subtypes.")
            if set(self.inclusion_exclusion_rules) != known_ids:
                raise ValueError("Every frozen subtype requires annotation rules.")
            for subtype in self.subtype_definitions:
                if not subtype.evidence_requirements:
                    raise ValueError(
                        "Frozen subtypes require explicit evidence requirements."
                    )
                rules = self.inclusion_exclusion_rules[subtype.subtype_id]
                if not rules.inclusion or not rules.exclusion:
                    raise ValueError(
                        "Frozen subtypes require inclusion and exclusion criteria."
                    )
        return self


class EvidenceSpan(SchemaModel):
    source_ref: Identifier
    quote: StrictStr | None = None
    start_offset: NonNegativeInt | None = None
    end_offset: NonNegativeInt | None = None
    subtype_id: Identifier | None = None
    relation_type: Identifier | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_span(self) -> Self:
        if (self.start_offset is None) != (self.end_offset is None):
            raise ValueError("Span offsets must be supplied together.")
        if self.start_offset is not None and self.end_offset is not None:
            if self.end_offset <= self.start_offset:
                raise ValueError("Span offsets use a nonempty half-open interval.")
        if self.quote is None and self.start_offset is None:
            raise ValueError("An evidence span requires a quote or offsets.")
        if self.quote is not None and not self.quote.strip():
            raise ValueError("Evidence quotes must not be empty.")
        # Offset units and quote matching require the referenced source content.
        return self


class LabelAssignment(SchemaModel):
    assignment_id: Identifier
    response_id: Identifier
    taxonomy_id: Identifier
    actor_id: Identifier
    round_index: NonNegativeInt
    origin: AssignmentOrigin
    labels: list[Identifier] = Field(default_factory=list)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    uncertain: StrictBool = False
    supersedes_id: Identifier | None = None

    @field_validator("labels")
    @classmethod
    def _canonical_labels(cls, labels: list[str]) -> list[str]:
        _check_unique(labels, "Assignment labels")
        return sorted(labels)

    @model_validator(mode="after")
    def _validate_assignment(self, info: ValidationInfo) -> Self:
        if self.supersedes_id == self.assignment_id:
            raise ValueError("An assignment cannot supersede itself.")
        context = info.context or {}
        taxonomy = context.get("taxonomy")
        if taxonomy is None:
            if context.get("require_taxonomy", False):
                raise ValueError("Taxonomy-aware assignment validation is required.")
            # Structural parsing alone does not certify taxonomy membership.
            return self
        if not isinstance(taxonomy, TaxonomyVersion):
            taxonomy = TaxonomyVersion.model_validate(taxonomy)
        if taxonomy.taxonomy_id != self.taxonomy_id:
            raise ValueError("Assignment references a different taxonomy version.")
        known_ids = {item.subtype_id for item in taxonomy.subtype_definitions}
        discovery_mode = context.get("discovery_mode", False)
        allowed_ids = set(known_ids)
        if discovery_mode:
            fallback_label = context.get("fallback_label")
            if fallback_label is None:
                raise ValueError("Discovery requires the configured fallback label.")
            allowed_ids.add(_identifier(fallback_label))
        elif context.get("require_frozen_taxonomy", False) and not taxonomy.frozen:
            raise ValueError("Final categorization requires a frozen taxonomy.")

        unknown = set(self.labels) - allowed_ids
        if unknown:
            raise ValueError(f"Unknown subtype identifiers: {sorted(unknown)}.")
        for span in self.evidence_spans:
            if span.subtype_id is not None and span.subtype_id not in self.labels:
                raise ValueError("Evidence span subtype is absent from assignment.")
        if context.get("require_evidence_for_each_label", False):
            supported = {
                span.subtype_id
                for span in self.evidence_spans
                if span.subtype_id is not None
            }
            if not (set(self.labels) & known_ids).issubset(supported):
                raise ValueError("Each assigned subtype requires linked evidence.")
        return self


# ---------------------------------------------------------------------------
# Human annotation and frozen audit manifests.
# ---------------------------------------------------------------------------


_INDEPENDENT_PURPOSES = frozenset(
    {
        AnnotationPurpose.CALIBRATION,
        AnnotationPurpose.IDENTIFICATION_ANCHOR,
        AnnotationPurpose.IDENTIFICATION_BOUNDARY,
        AnnotationPurpose.TAXONOMY_INITIALIZATION,
        AnnotationPurpose.HUMAN_AGREEMENT,
        AnnotationPurpose.INDEPENDENT_IDENTIFICATION_AUDIT,
        AnnotationPurpose.INDEPENDENT_LABEL_AUDIT,
        AnnotationPurpose.EXPANDED_LABEL_AUDIT,
    }
)

_BLINDED_FORBIDDEN_KEYS = frozenset(
    {
        "identification_score",
        "identification_scores",
        "similarity_score",
        "automatic_assignment",
        "automatic_assignments",
        "automatic_decision",
        "automatic_predictions",
        "agent_predictions",
        "peer_predictions",
        "peer_labels",
        "other_annotator_decisions",
        "independent_submissions",
        "adjudicated_labels",
        "final_assignment",
        "final_labels",
    }
)


def _check_blinded_payload(payload: JsonValue) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key.lower() in _BLINDED_FORBIDDEN_KEYS:
                raise ValueError(f"Blinded annotation payload contains {key!r}.")
            _check_blinded_payload(value)
    elif isinstance(payload, list):
        for value in payload:
            _check_blinded_payload(value)


class AnnotationTask(SchemaModel):
    annotation_task_id: Identifier
    response_id: Identifier
    purpose: AnnotationPurpose
    assigned_annotator: Identifier
    guideline_hash: Digest
    cohort_id: Identifier | None = None
    visible_payload: JsonObject = Field(default_factory=dict)
    status: AnnotationStatus = AnnotationStatus.PENDING

    @model_validator(mode="after")
    def _validate_visibility(self) -> Self:
        if self.purpose in _INDEPENDENT_PURPOSES:
            _check_blinded_payload(self.visible_payload)
        audit_purposes = {
            AnnotationPurpose.HUMAN_AGREEMENT,
            AnnotationPurpose.INDEPENDENT_IDENTIFICATION_AUDIT,
            AnnotationPurpose.INDEPENDENT_LABEL_AUDIT,
            AnnotationPurpose.EXPANDED_LABEL_AUDIT,
            AnnotationPurpose.AUDIT_ADJUDICATION,
        }
        if self.purpose in audit_purposes and self.cohort_id is None:
            raise ValueError("Independent audit tasks require cohort membership.")
        return self


class AuditCohort(SchemaModel):
    cohort_id: Identifier
    purpose: Identifier
    seed: StrictInt
    frozen_prediction_version: Identifier
    response_ids: list[Identifier] = Field(default_factory=list)
    sampling_protocol: JsonObject = Field(default_factory=dict)
    locked: StrictBool = False

    @model_validator(mode="after")
    def _validate_cohort(self) -> Self:
        _check_unique(self.response_ids, "Audit response identifiers")
        protocol = self.sampling_protocol
        declared_count = protocol.get("membership_count")
        if declared_count is not None:
            if isinstance(declared_count, bool) or not isinstance(
                declared_count, int
            ):
                raise ValueError("membership_count must be an integer.")
            if declared_count != len(self.response_ids):
                raise ValueError("Audit membership count is inconsistent.")

        excluded_ids = protocol.get("excluded_response_ids")
        if excluded_ids is not None:
            if not isinstance(excluded_ids, list) or not all(
                isinstance(item, str) for item in excluded_ids
            ):
                raise ValueError("excluded_response_ids must be a string list.")
            _check_unique(excluded_ids, "Excluded response identifiers")
            if set(excluded_ids) & set(self.response_ids):
                raise ValueError("Audit membership contains excluded responses.")

        frozen_ids = protocol.get("frozen_prediction_ids")
        if frozen_ids is not None:
            if not isinstance(frozen_ids, dict):
                raise ValueError(
                    "frozen_prediction_ids must map each response to prediction IDs."
                )
            if set(frozen_ids) != set(self.response_ids):
                raise ValueError("Frozen predictions must exactly cover membership.")
            for prediction_ids in frozen_ids.values():
                if isinstance(prediction_ids, str):
                    _identifier(prediction_ids)
                elif isinstance(prediction_ids, list) and prediction_ids:
                    for prediction_id in prediction_ids:
                        _identifier(prediction_id)
                    _check_unique(prediction_ids, "Frozen prediction identifiers")
                else:
                    raise ValueError("Frozen prediction references cannot be empty.")

        original = protocol.get("original_response_ids")
        additions = protocol.get("additional_response_ids")
        if original is not None or additions is not None:
            if not isinstance(original, list) or not isinstance(additions, list):
                raise ValueError("Expanded cohorts require original and added IDs.")
            for response_id in original + additions:
                _identifier(response_id)
            _check_unique(original + additions, "Expanded audit membership")
            if set(original + additions) != set(self.response_ids):
                raise ValueError("Expanded membership must retain all original IDs.")
            _identifier(protocol.get("original_cohort_id"))

        if self.locked:
            required = {
                "membership_count",
                "eligible_population_version",
                "sampling_method",
                "exclusion_categories",
                "excluded_response_ids",
                "annotation_protocol",
            }
            missing = required - set(protocol)
            if missing:
                raise ValueError(
                    f"Locked cohort lacks sampling metadata: {sorted(missing)}."
                )
            _identifier(protocol["eligible_population_version"])
            _identifier(protocol["sampling_method"])
            if not isinstance(protocol["exclusion_categories"], list):
                raise ValueError("exclusion_categories must be a list.")
            if protocol["annotation_protocol"] is None:
                raise ValueError("Locked cohort needs an annotation protocol.")
            if frozen_ids is None and not protocol.get(
                "frozen_prediction_manifest"
            ):
                raise ValueError(
                    "Locked cohort needs frozen predictions or their manifest."
                )
            if not self.response_ids:
                raise ValueError("Locked audit cohorts must not be empty.")
        return self


# ---------------------------------------------------------------------------
# Metric artifacts and resumable stage state.
# ---------------------------------------------------------------------------


class MetricScope(str, Enum):
    RESPONSE = "response"
    TASK_RUN = "task_run"
    ACROSS_RUNS = "across_runs"
    FAILURE_IDENTIFICATION_AUDIT = "failure_identification_audit"
    LABEL_AUDIT = "label_audit"
    HUMAN_AGREEMENT = "human_agreement"
    TAXONOMY_STABILITY = "taxonomy_stability"
    RESPONSE_FAILURE_RATE = "response_failure_rate"


class MetricResult(SchemaModel):
    metric_id: Identifier
    metric_name: Identifier
    scope: MetricScope
    scoring_version: Identifier
    reference_version: Identifier
    reporting_scale: tuple[FiniteFloat, FiniteFloat]
    value: FiniteFloat | None = None
    undefined_reason: Identifier | None = None
    task_name: Identifier | None = None
    model_id: Identifier | None = None
    run_index: NonNegativeInt | None = None
    cohort_id: Identifier | None = None
    response_ids: list[Identifier] = Field(default_factory=list)
    numerator: FiniteFloat | None = None
    denominator: NonNegativeFloat | None = None
    support: NonNegativeInt | None = None
    sufficient_statistics: JsonObject = Field(default_factory=dict)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_metric(self) -> Self:
        lower, upper = self.reporting_scale
        if lower >= upper:
            raise ValueError("Metric reporting scale must have increasing bounds.")
        if self.value is None:
            if self.undefined_reason is None:
                raise ValueError("Undefined metrics require an explicit reason.")
        elif self.undefined_reason is not None:
            raise ValueError("Defined metrics cannot have an undefined reason.")
        elif not lower <= self.value <= upper:
            raise ValueError("Metric value is outside its declared reporting scale.")
        if self.denominator == 0 and self.value is not None:
            raise ValueError("A zero-denominator metric must be undefined.")
        if self.scope == MetricScope.TASK_RUN:
            if (
                self.run_index is None
                or self.task_name is None
                or self.model_id is None
            ):
                raise ValueError("Task-run metrics require task, model, and run.")
        if self.scope == MetricScope.ACROSS_RUNS and self.run_index is not None:
            raise ValueError("Across-run summaries do not identify a single run.")
        if self.metric_name.lower() in {"std", "standard_deviation"}:
            ddof = self.metadata.get("standard_deviation_ddof")
            run_count = self.metadata.get("run_count")
            if (
                isinstance(ddof, bool)
                or not isinstance(ddof, int)
                or ddof < 0
                or isinstance(run_count, bool)
                or not isinstance(run_count, int)
                or run_count <= ddof
            ):
                raise ValueError("Standard deviation requires explicit ddof and N.")
        _check_unique(self.response_ids, "Metric response membership")
        return self


class StageState(SchemaModel):
    stage: Identifier
    execution_id: Identifier
    config_hash: Digest
    status: StageStatus = StageStatus.PENDING
    completed_work: list[Identifier] = Field(default_factory=list)
    pending_work: list[Identifier] = Field(default_factory=list)
    cursor: JsonObject = Field(default_factory=dict)
    outstanding_annotation_task_ids: list[Identifier] = Field(default_factory=list)
    input_manifests: dict[Identifier, Identifier] = Field(default_factory=dict)
    output_manifests: dict[Identifier, Identifier] = Field(default_factory=dict)
    started_at: UtcTimestamp | None = None
    updated_at: UtcTimestamp | None = None
    completed_at: UtcTimestamp | None = None
    errors: list[JsonObject] = Field(default_factory=list)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_state(self) -> Self:
        _check_unique(self.completed_work, "Completed work")
        _check_unique(self.pending_work, "Pending work")
        _check_unique(
            self.outstanding_annotation_task_ids, "Outstanding annotation tasks"
        )
        if set(self.completed_work) & set(self.pending_work):
            raise ValueError("Completed and pending work must be disjoint.")
        _ordered_times(self.started_at, self.updated_at, "Stage update")
        _ordered_times(self.started_at, self.completed_at, "Stage completion")
        _ordered_times(self.updated_at, self.completed_at, "Final stage update")
        if self.status == StageStatus.COMPLETED:
            if self.pending_work or self.outstanding_annotation_task_ids:
                raise ValueError("Completed stages cannot have outstanding work.")
            if self.completed_at is None:
                raise ValueError("Completed stages require a completion timestamp.")
        elif self.completed_at is not None:
            raise ValueError("Only completed stages have completed_at.")
        if self.status == StageStatus.WAITING_FOR_ANNOTATIONS:
            if not self.outstanding_annotation_task_ids:
                raise ValueError("Waiting stages must identify pending annotations.")
        if self.status == StageStatus.FAILED and not self.errors:
            raise ValueError("Failed stages require recorded errors.")
        return self
