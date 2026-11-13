"""Task-specific, opt-in model visibility for E.2 identifier preprocessing.

Identifiers are normalized across databases. ``task-identifier-roles-v1``
controls which normalization artifacts enter which prompts.

An observation with no role, or ``role='provenance_only'``, is retained for
validation and audit only. ``role='input_normalization'`` declares a source-side
identifier suitable for input linking. Resolved observations with that role are
joined by exact (kind, type, canonical identifier), retaining every source span.
This join establishes identifier equivalence, never actor/campaign/causal edges.
No reference answer, alias-resource support metadata, or arbitrary annotation is
projected. Unresolved and disabled transformations remain provenance only.
Opted-in alias mappings must cite snapshot-admitted model-input evidence; those
support identifiers accompany the normalized input, without resource metadata.
Historical availability is checked by DatasetLoader, and source publication /
version dates by ExperimentRunner._task_payload before this projector runs.
The projector checks roles, source identities, and exact spans; it does not
independently establish historical availability.

Literal CVE/ATT&CK case formatting is allowed even in linking/extraction tasks:
it does not infer a missing identifier. IOC canonicalization is withheld for IOC
normalization; aliases that resolve the task's actor, campaign, vulnerability,
malware, technique, sector, style, or mitigation target are rejected. Other
source-side roles require explicit opt-in. Unknown tasks/types/roles fail closed.
Reliability scores, ticket labels, membership decisions, and rankings are not
alias identity types and cannot be introduced as normalization observations.
"""
from __future__ import annotations

from typing import Any

from schemas import EvidenceRole, TaskInstance


NORMALIZATION_CONTRACT = "task-identifier-roles-v1"

# E.3 task vocabulary; adding a task requires an explicit visibility review.
_TASKS = frozenset({
    "affected_systems", "attack_infrastructure", "vulnerability_linking",
    "malware_family_mapping", "ioc_normalization", "threat_report_alignment",
    "event_timeline_construction", "graph_population", "source_reliability_scoring",
    "threat_actor_linking", "ttp_extraction", "campaign_attribution",
    "infrastructure_reuse", "language_style_profiling", "false_flag_detection",
    "evidence_weighting", "relation_graph_building", "exploit_likelihood",
    "impact_forecast", "target_sector_prediction", "campaign_escalation",
    "patch_recommendation", "rule_generation_yara", "response_summarization",
    "mitigation_ttp_mapping", "defensive_playbook_generation", "countermeasure_ranking",
    "incident_ticket_generation",
})
_ALIAS_TYPES = frozenset({
    "actor", "threat_actor", "intrusion_set", "malware", "malware_family",
    "tool", "campaign", "vulnerability", "cve", "technique", "ttp",
    "attack_technique", "mitigation", "countermeasure", "defense", "patch",
    "sector", "target_sector", "style", "language_style", "report", "host",
    "asset", "product",
})
_TARGET_ALIASES = {
    "vulnerability_linking": {"vulnerability", "cve"},
    "malware_family_mapping": {"malware", "malware_family"},
    "threat_actor_linking": {"actor", "threat_actor", "intrusion_set"},
    "campaign_attribution": {"campaign"},
    "ttp_extraction": {"technique", "ttp", "attack_technique"},
    "target_sector_prediction": {"sector", "target_sector"},
    "language_style_profiling": {"style", "language_style"},
    "mitigation_ttp_mapping": {"mitigation", "countermeasure", "defense"},
    "patch_recommendation": {"patch"},
    "countermeasure_ranking": {"mitigation", "countermeasure", "defense"},
}


def observation_role(task: str, observation: dict[str, Any]) -> str:
    """Validate an explicit source-role declaration before resolving its value."""
    role = observation.get("role", "provenance_only")
    if role not in ("provenance_only", "input_normalization"):
        raise ValueError("Unknown identifier normalization role.")
    if role == "provenance_only":
        return role
    if task not in _TASKS:
        raise ValueError("Input normalization role lacks a task-specific contract.")
    kind = observation.get("kind")
    if kind not in {"cve", "ioc", "attack", "alias"}:
        raise ValueError("Input normalization role has an unknown identifier kind.")
    if task == "ioc_normalization" and kind == "ioc":
        raise ValueError("Input normalization would disclose the IOC task target.")
    if kind == "alias":
        entity_type = observation.get("entity_type")
        if entity_type not in _ALIAS_TYPES:
            raise ValueError("Input normalization role requires a known alias entity_type.")
        if entity_type in _TARGET_ALIASES.get(task, set()):
            raise ValueError("Input normalization would disclose an alias-resolved task target.")
    return role


def model_normalization_feeds(instance: TaskInstance) -> list[dict[str, Any]]:
    """Join admitted identifier mentions into allowlisted model-visible records."""
    records = instance.transformation_log.get("normalization", [])
    if not isinstance(records, list):
        raise ValueError("Identifier normalization provenance must be a list.")
    evidence = {item.evidence_id: item for item in instance.evidence_records}
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Identifier normalization records must be objects.")
        role = record.get("role", "provenance_only")
        if role == "provenance_only":
            continue
        observation = record.get("observation")
        if not isinstance(observation, dict):
            raise ValueError("Model-visible normalization lacks its source observation.")
        if observation_role(instance.task_name, observation) != role:
            raise ValueError("Identifier normalization role differs from its source declaration.")
        if instance.transformation_log.get("normalization_contract") != NORMALIZATION_CONTRACT:
            raise ValueError("Model-visible normalization lacks a supported contract version.")
        if record.get("status") in {"unresolved", "preserved"}:
            continue
        if record.get("status") not in {"validated", "syntax_validated", "normalized"}:
            raise ValueError("Model-visible normalization has an unknown resolution status.")
        canonical = record.get("canonical")
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError("Resolved normalization requires a canonical identifier.")
        evidence_id = observation.get("evidence_id")
        source = evidence.get(evidence_id)
        start, end = observation.get("start_offset"), observation.get("end_offset")
        raw = observation.get("raw")
        if (
            source is None or source.role != EvidenceRole.MODEL_INPUT
            or not isinstance(raw, str) or not raw
            or isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, int) or not isinstance(end, int)
            or start < 0 or end <= start or (source.snippet or "")[start:end] != raw
            or record.get("raw") != raw or record.get("evidence_id") != evidence_id
        ):
            raise ValueError("Model-visible normalization lacks an exact input source span.")
        kind = observation["kind"]
        type_field = "ioc_type" if kind == "ioc" else "entity_type" if kind == "alias" else None
        identifier_type = observation.get(type_field, "") if type_field else ""
        if type_field and (not isinstance(identifier_type, str) or not identifier_type):
            raise ValueError("Typed input normalization lacks its identifier type.")
        support = record.get("evidence_refs", []) if kind == "alias" else []
        if kind == "alias" and (
            not isinstance(support, list) or not support
            or any(not isinstance(ref, str) or ref not in evidence
                   or evidence[ref].role != EvidenceRole.MODEL_INPUT for ref in support)
        ):
            raise ValueError("Input normalization requires model-visible alias support.")
        key = (kind, identifier_type, canonical)
        if key not in groups:
            groups[key] = {"record_type": "normalized_identifier", "kind": kind,
                           "canonical": canonical, "mentions": []}
            if type_field:
                groups[key][type_field] = identifier_type
            if kind == "alias":
                groups[key]["supporting_evidence_ids"] = []
        for ref in support:
            if ref not in groups[key]["supporting_evidence_ids"]:
                groups[key]["supporting_evidence_ids"].append(ref)
        mention = {
            "raw": raw, "evidence_id": evidence_id, "source_id": source.source_id,
            "document_id": source.document_id, "start_offset": start, "end_offset": end,
        }
        if mention not in groups[key]["mentions"]:
            groups[key]["mentions"].append(mention)
    return list(groups.values())
