"""Independent F5 review of execution identity under prompt content drift."""
from pathlib import Path

import pytest

from test_audit_single_agent_ablation import PROTOCOL
from test_discovery_repairs import production_harness
from test_taxonomy_and_categorization import harness, no_live_inference


@pytest.mark.parametrize("production_harness", [True], indirect=True)
def test_single_agent_resume_never_reuses_predictions_after_reconsideration_prompt_changes(
    production_harness, monkeypatch,
):
    world = production_harness
    first = world.runner()._categorization_ablation(
        "multi_agent_decision", world.failure_ids[:1], world.taxonomy, PROTOCOL, 0
    )
    assert first["status"] == "completed"
    assert len(world.models.requests) == 2
    # Simulate different installed prompt bytes without editing a production
    # file. All other template/config reads retain the original exact behavior.
    read_bytes = Path.read_bytes

    def reconsideration_variant(path):
        original = read_bytes(path)
        if path.name == "agent_single_reconsideration.jinja2":
            return original + b"\nIndependent review fixture: reassess every subtype from its evidence.\n"
        return original

    monkeypatch.setattr(Path, "read_bytes", reconsideration_variant)
    try:
        second = world.runner()._categorization_ablation(
            "multi_agent_decision", world.failure_ids[:1], world.taxonomy, PROTOCOL, 0
        )
    except ValueError:
        return  # An explicit frozen-template mismatch is also a safe outcome.
    assert second["protocol_id"] != first["protocol_id"], (
        "Changed built-in reconsideration prompt silently reused the completed "
        f"single-agent execution {first['execution_id']} with {len(world.models.requests)} total requests."
    )
    assert "Independent review fixture: reassess every subtype from its evidence." in world.models.requests[-1]["prompt"]
