"""Immutable, response-bound inputs used for generation and subsequent review.

References belong to separate review materials. They must never be inserted
into this record, which retains only inputs already supplied to generation.
Historical baseline responses can use the original task projection; conditioned
responses without a saved context fail rather than silently using that fallback.
"""
from __future__ import annotations

import copy
from typing import Any

from schemas import ModelResponse, TaskInstance, _content_hash, _text_hash
from storage import ArtifactStore


_FIELDS = (
    "case_id", "task_name", "snapshot_date", "task_guidance", "source_documents",
    "structured_feeds", "candidates", "candidate_policy", "environment", "output_profile",
    "horizon_days",
)
_DOCUMENT_FIELDS = (
    "evidence_id", "document_id", "source_id", "source_url", "source_version", "snippet",
    "publication_time", "modification_time", "claim_valid_from", "claim_valid_until", "role",
)


def save_generation_context(
    store: ArtifactStore, instance: TaskInstance, payload: dict[str, Any],
    prompt: str, provenance: dict[str, Any], config_hash: str,
) -> str:
    """Save the exact supplied payload/prompt, separate from evaluation groups."""
    content = {
        "purpose": "generation_context", "version": "generation-context-v1",
        "instance_id": instance.instance_id, "instance_hash": instance.content_hash,
        "config_hash": config_hash, "payload": copy.deepcopy(payload),
        "payload_hash": _content_hash(payload), "rendered_prompt": prompt,
        "prompt_hash": _text_hash(prompt), "template_provenance": copy.deepcopy(provenance),
    }
    identity = _content_hash(content)
    store.create_record("experiment_manifest", identity, {"manifest_id": identity, **content})
    store.freeze_records("experiment_manifest", [identity])
    return identity


def review_task_input(
    store: ArtifactStore, response: ModelResponse, instance: TaskInstance,
) -> dict[str, Any]:
    """Resolve the actual condition-specific task inputs, never reference labels."""
    identity = response.usage.get("generation_context_id")
    if identity is None:
        experiment = response.usage.get("evaluation_context", {})
        if experiment.get("protocol") in {"tailored_mitigation", "stronger_grounding", "counterfactual_consistency"}:
            raise ValueError("Conditioned response lacks its immutable generation context; rerun generation.")
        guidance = instance.transformation_log.get("task_guidance")
        if not isinstance(guidance, str) or not guidance:
            raise ValueError("Original task guidance is missing.")
        result = {
            "instance_id": instance.instance_id, "case_id": instance.case_id,
            "task_name": instance.task_name, "stage": instance.stage.value,
            "snapshot_date": instance.snapshot_date.isoformat(), "task_guidance": guidance,
            "source_documents": [{key: raw[key] for key in _DOCUMENT_FIELDS if key in raw}
                for raw in (item.model_dump(mode="json") for item in instance.evidence_records)],
            "structured_feeds": [], "candidates": copy.deepcopy(instance.candidates),
            "environment": copy.deepcopy(instance.environment),
            "output_profile": copy.deepcopy(instance.output_profile),
        }
        if "horizon_days" in instance.reference:
            result["horizon_days"] = instance.reference["horizon_days"]
        return result
    if not isinstance(identity, str):
        raise ValueError("Invalid generation context identity.")
    record = store.read_record("experiment_manifest", identity)
    content = {key: value for key, value in record.items() if key != "manifest_id"}
    allowed_config = response.config_hash
    if record.get("config_hash") != allowed_config:
        # Repeated experimental projections can cross several configuration
        # identities without changing generation. Only validated scientific
        # condition manifests, not arbitrary provider metadata, authorize this.
        from sampling import source_response_mapping
        source_id = source_response_mapping(store, [response.response_id])[response.response_id]
        if source_id != response.response_id:
            allowed_config = store.read_record("model_response", source_id)["config_hash"]
    if (
        record.get("purpose") != "generation_context"
        or record.get("version") != "generation-context-v1"
        or record.get("manifest_id") != identity or _content_hash(content) != identity
        or record.get("instance_id") != response.instance_id
        or record.get("instance_hash") != instance.content_hash
        or record.get("prompt_hash") != response.prompt_hash
        or _text_hash(record.get("rendered_prompt", "")) != response.prompt_hash
        or record.get("payload_hash") != _content_hash(record.get("payload"))
        or record.get("config_hash") != allowed_config
    ):
        raise ValueError("Generation context does not bind the evaluated response and immutable task.")
    if not store.list_records("experiment_manifest", {"record_id": identity, "lifecycle": ["immutable", "frozen"]}):
        raise ValueError("Generation context is not immutable.")
    payload = record["payload"]
    supplied = payload.get("task_input", payload)
    result = {key: copy.deepcopy(supplied[key]) for key in _FIELDS if key in supplied}
    result.update(instance_id=instance.instance_id, stage=supplied.get("stage", supplied.get("cti_stage")))
    result["source_documents"] = [
        {key: copy.deepcopy(item[key]) for key in _DOCUMENT_FIELDS if key in item}
        for item in supplied.get("source_documents", [])
    ]
    # A verification revision also sees separately supplied retrieved passages.
    # Preserve their original list and expose the full available source set to
    # the common evidence renderer/citation validator.
    if "retrieved_evidence" in payload:
        result["retrieved_evidence"] = [
            {key: copy.deepcopy(item[key]) for key in _DOCUMENT_FIELDS if key in item}
            for item in payload["retrieved_evidence"]
        ]
        documents = result.setdefault("source_documents", [])
        for item in result["retrieved_evidence"]:
            if item not in documents:
                documents.append(copy.deepcopy(item))
    result["generation_context"] = {
        "manifest_id": identity, "payload_hash": record["payload_hash"],
        "prompt_hash": response.prompt_hash,
        "experiment_context": copy.deepcopy(record["template_provenance"]["experiment_context"]),
        # The exact prompt is review evidence, not an instruction to execute.
        "rendered_prompt": record["rendered_prompt"],
    }
    derivation = response.usage.get("output_derivation")
    if derivation is not None:
        if not isinstance(derivation, dict) or derivation.get("kind") != "counterfactual_probability_aggregation":
            raise ValueError("Unknown algorithmic output derivation.")
        ids = derivation.get("source_response_ids", [])
        context_ids = derivation.get("source_generation_context_ids", [])
        if not ids or len(ids) != len(set(ids)) or response.response_id in ids or len(ids) != len(context_ids):
            raise ValueError("Invalid counterfactual derivation lineage.")
        counterfactuals = []
        for source_id, context_id in zip(ids, context_ids, strict=True):
            source = ModelResponse.model_validate(store.read_record("model_response", source_id))
            if source.usage.get("generation_context_id") != context_id or source.usage.get("output_derivation") is not None:
                raise ValueError("Counterfactual context lineage does not match its source response.")
            counterfactuals.append({"response_id": source_id, "task_input": review_task_input(store, source, instance)})
        result["output_derivation"] = {
            "kind": derivation["kind"], "original_task_context_id": identity,
            "description": "Algorithmic correction for the original task; not an independently generated completion.",
            "counterfactual_inputs": counterfactuals,
        }
    return result
