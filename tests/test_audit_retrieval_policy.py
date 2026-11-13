"""F3: controller declarations must match the policy the pinned backend runs.

Only model completions use the existing offline harness. Retrieval runs through
the real controller, digest-checked adapter, hybrid ranking and H.2 selection.
"""
from __future__ import annotations

import copy
import hashlib
from collections import Counter
from pathlib import Path

import pytest

import research_backends
from schemas import TaskInstance, _content_hash
from test_tailored_fidelity import configured, passage
from test_taxonomy_and_categorization import Harness, _configure_c3, _reconfigure, harness


def configure_controller(harness: Harness, tmp_path, *, conflict=None, poison_ranking=False):
    _configure_c3(harness, tmp_path)
    instance = TaskInstance.model_validate(
        harness.store.read_record('task_instance', harness.instance_ids[0]))
    records = [
        passage('old', 'A', 'outdated January observation'),
        passage('reused', 'B', 'commonly reused observable', ioc_reuse_count=2),
        passage('context', 'C', 'incidental regional context', evidence_kind='context_only'),
        passage('good-A', 'A', 'specific exploit delivery mechanism'),
        passage('extra-A', 'A', 'second independent same-publisher report'),
        passage('good-B', 'B', 'independent infrastructure ownership observation',
                publication_time='2024-05-31'),
        passage('good-C', 'C', 'contrary behavioral evidence for an alternative',
                evidence_relation='counterevidence'),
    ]
    for record in records:
        record['environment'] = instance.environment
        record.setdefault('evidence_kind', 'causal')
        if record['evidence_id'] not in {'old', 'good-B'}:
            record['publication_time'] = '2024-06-01'
    _, parameters = configured(records, minimum_publishers=3, cap=1)
    parameters['frozen_scores']['query_hash'] = _content_hash(research_backends.retrieval_query({
        'model_input_evidence': [item.model_dump(mode='json') for item in instance.evidence_records],
        'environment': instance.environment}))
    parameters['tailored'].update({
        'maximum_age_days': 1, 'maximum_ioc_reuse': 1, 'environment_fields': [],
        'evidence_policies': {
            'causal_context': {'maximum_context_only': 0, 'require_classification': True},
            'counterevidence': {'minimum_records': 1},
        },
    })
    evidence_policies = copy.deepcopy(parameters['tailored']['evidence_policies'])
    if conflict is not None:
        field, value = conflict
        parameters['tailored'][field] = value
    if poison_ranking:
        # A mismatch must be caught before the backend even reads replay scores.
        parameters['frozen_scores']['query_hash'] = 'ranking-must-not-start'
    policies = tmp_path / 'audit-retrieval-policies.yaml'
    policies.write_text('version: offline-source-policies-v1\npolicies: {}\n', encoding='utf-8')
    declaration = {
        'callable': 'research_backends:tailored_retrieval', 'version': 'audit-f3-v1',
        'source_sha256': hashlib.sha256(Path(research_backends.__file__).read_bytes()).hexdigest(),
        'parameters': parameters,
    }
    _reconfigure(harness, {
        'operational.selected_defense': 'tailored_mitigation',
        'operational.selected_mitigation_mode': 'retrieval_only',
        'defenses.tailored_mitigation.subtype_policy_manifest': str(policies),
        'defenses.tailored_mitigation.routing_policy': {'mode': 'universal', 'uses_gold_failure_labels': False},
        'defenses.tailored_mitigation.model_group_aggregation': {
            'mode': 'pooled_response_counts', 'groups': {'general': ['G5']}},
        'defenses.tailored_mitigation.retrieval_models': {
            'adapter': declaration, 'evidence_policies': evidence_policies},
        'defenses.tailored_mitigation.source_diversity_parameters': {
            'minimum_publishers': 3, 'maximum_per_publisher': 1},
        'defenses.tailored_mitigation.freshness_parameters': {
            'maximum_age_days': 1, 'recency_weight': 0, 'source_quality_weight': 0},
        'defenses.tailored_mitigation.ioc_reuse_thresholds': {'maximum_ioc_reuse': 1},
        'defenses.tailored_mitigation.repetition_count': 1,
        'defenses.tailored_mitigation.evidence_context_budget': 10000,
        'failure_identification.hard_label_rules': {'tasks': {'threat_actor_linking': {'route': 'hard'}}},
        'evaluation.scoring.response_level_failure_criteria': {'threat_actor_linking': {
            'pass_if': {'op': 'equal', 'left': {'source': 'prediction', 'pointer': '/answer/choice'},
                        'right': {'source': 'reference', 'pointer': '/answer/choice'}},
            'on_missing': 'error', 'availability': {}}},
    })
    harness.models.script = lambda request, packet: {
        'status': 'OK', 'task': packet['task_name'], 'case_id': packet['case_id'],
        'snapshot_date': packet['snapshot_date'], 'answer': {'choice': 'B'},
        'justification': 'Uses the supplied fictional evidence.',
    }


@pytest.mark.parametrize(('field', 'conflicting_value'), [
    ('minimum_publishers', 1),
    ('maximum_per_publisher', 2),
    ('maximum_age_days', 366),
    ('recency_weight', .5),
    ('source_quality_weight', .5),
    ('maximum_ioc_reuse', 10),
    ('evidence_policies', {'causal_context': {'maximum_context_only': 1}}),
])
def test_real_controller_rejects_conflicting_policy_before_ranking(
    harness: Harness, tmp_path, field, conflicting_value,
):
    # Removing reconciliation would hit the intentionally invalid score binding.
    configure_controller(harness, tmp_path, conflict=(field, conflicting_value), poison_ranking=True)
    with pytest.raises(ValueError, match='[Ff]orwarded.*' + field + '.*disagree'):
        harness.runner().run_tailored_mitigation('retrieval_only', harness.instance_ids[:1])
    assert harness.models.requests == []
    assert not [item for item in harness.store.list_records('experiment_manifest')
                if item.get('purpose') == 'retrieval_evidence_package']


def test_matching_controller_policy_selects_only_eligible_diverse_evidence_and_is_persisted(
    harness: Harness, tmp_path,
):
    # Ignoring any gate changes this literal, independently chosen evidence set.
    configure_controller(harness, tmp_path)
    result = harness.runner().run_tailored_mitigation('retrieval_only', harness.instance_ids[:1])
    assert result['status'] == 'completed'
    packages = [item for item in harness.store.list_records('experiment_manifest')
                if item.get('purpose') == 'retrieval_evidence_package']
    output = packages[0]['result']
    assert [item['evidence_id'] for item in output['evidence']] == ['good-A', 'good-B', 'good-C']
    assert output['provenance']['incident_publishers'] == ['A', 'B', 'C']
    policy = output['provenance']['effective_tailored_policy']
    assert {key: policy[key] for key in (
        'minimum_publishers', 'maximum_per_publisher', 'maximum_age_days',
        'maximum_ioc_reuse', 'recency_weight', 'source_quality_weight', 'evidence_policies')} == {
            'minimum_publishers': 3, 'maximum_per_publisher': 1, 'maximum_age_days': 1,
            'maximum_ioc_reuse': 1, 'recency_weight': 0., 'source_quality_weight': 0.,
            'evidence_policies': {
                'causal_context': {'maximum_context_only': 0, 'require_classification': True},
                'counterevidence': {'minimum_records': 1},
            },
        }
    assert output['provenance']['tailored_policy_hash'] == _content_hash(policy)


def test_conflicting_publisher_age_and_reuse_controls_cannot_complete_generation(harness: Harness, tmp_path):
    configure_controller(harness, tmp_path)
    retrieval = copy.deepcopy(harness.config.defenses.tailored_mitigation.retrieval_models)
    retrieval['adapter']['parameters']['tailored'].update({
        'minimum_publishers': 1, 'maximum_per_publisher': 10,
        'maximum_age_days': 366, 'maximum_ioc_reuse': 10,
    })
    _reconfigure(harness, {'defenses.tailored_mitigation.retrieval_models': retrieval})
    with pytest.raises(ValueError, match='[Ff]orwarded.*disagree'):
        harness.runner().run_tailored_mitigation('retrieval_only', harness.instance_ids[:1])
    assert harness.models.requests == []


def test_unrecognized_controller_control_is_rejected_before_ranking(harness: Harness, tmp_path):
    configure_controller(harness, tmp_path, poison_ranking=True)
    _reconfigure(harness, {'defenses.tailored_mitigation.freshness_parameters': {'max_age_typo': 1}})
    with pytest.raises(ValueError, match='[Uu]nknown.*freshness_parameters'):
        harness.runner().run_tailored_mitigation('retrieval_only', harness.instance_ids[:1])


def test_direct_pinned_backend_rejects_disagreement_before_ranking():
    # The adapter also protects callers other than this controller.
    from test_tailored_fidelity import invoke
    payload, parameters = configured([passage('a', 'A', 'independent observation')])
    payload['retrieval_configuration'] = {'source_diversity_parameters': {'minimum_publishers': 3}}
    parameters['frozen_scores']['query_hash'] = 'ranking-must-not-start'
    with pytest.raises(ValueError, match='[Ff]orwarded.*minimum_publishers.*disagree'):
        invoke(payload, parameters)


def evidence_bundle_case(kind, *, alternate=False):
    incident = passage('incident-A', 'A', 'specific incident evidence')
    if kind == 'reference':
        incident['standard_mappings'] = {'technique_id': 'T-TEST'}
        support = passage('standard-A', 'A', 'declared permitted technique', incident_roles=[],
                          authoritative_options={'technique_id': ['T-TEST']})
        evidence_policy = {'standards': {
            'reference_evidence_ids': ['standard-A'], 'mapping_fields': ['technique_id'],
            'on_invalid': 'exclude'}}
    else:
        incident['stage_transitions'] = [{
            'from_stage': 'recon', 'to_stage': 'execution', 'artifact_evidence_ids': ['log-A']}]
        support = passage('log-A', 'A', 'historical telemetry', artifact_type='log', incident_roles=[])
        evidence_policy = {'attack_chain': {'unbacked_stage_weight': 1.}}
    records = [incident, support]
    if alternate:
        records.append(passage('alternate-B', 'B', 'independent alternate incident evidence',
                               standard_mappings={'technique_id': 'T-TEST'}))
    payload, parameters = configured(records, cap=1)
    payload['retrieval_configuration'] = {
        'source_diversity_parameters': {'minimum_publishers': 1, 'maximum_per_publisher': 1}}
    parameters['tailored']['evidence_policies'] = evidence_policy
    return payload, parameters


@pytest.mark.parametrize('kind', ['reference', 'artifact'])
def test_publisher_passage_cap_rejects_an_unavoidable_mandatory_bundle_overflow(kind):
    from test_tailored_fidelity import invoke
    payload, parameters = evidence_bundle_case(kind)
    with pytest.raises(ValueError, match='[Pp]ublisher|[Dd]iversity|[Cc]ap|[Ii]ncompatible|[Uu]nsupported'):
        invoke(payload, parameters)


@pytest.mark.parametrize('kind', ['reference', 'artifact'])
def test_package_cap_selects_a_feasible_alternate_publisher(kind):
    from test_tailored_fidelity import invoke
    payload, parameters = evidence_bundle_case(kind, alternate=True)
    result = invoke(payload, parameters)
    counts = Counter(item['publisher'] for item in result['evidence'])
    assert max(counts.values()) <= 1
    assert [item['evidence_id'] for item in result['evidence']] == (
        ['alternate-B', 'standard-A'] if kind == 'reference' else ['alternate-B'])
    assert result['provenance']['incident_publisher_counts'] == {'B': 1}
    assert result['provenance']['publisher_counts'] == ({'A': 1, 'B': 1} if kind == 'reference' else {'B': 1})
    assert result['provenance']['publisher_cap_scope'] == 'all_emitted_evidence_passages'


def test_shared_artifact_counts_once_for_the_package_cap_and_preserves_feasible_diversity():
    from test_tailored_fidelity import invoke
    rows = [passage('top-A', 'A', 'top incident A evidence'),
            passage('top-B', 'B', 'top incident B evidence'),
            passage('extra-A', 'A', 'extra incident A evidence'),
            passage('shared-log', 'A', 'a shared source telemetry log', artifact_type='log', incident_roles=[])]
    for row in rows[:3]:
        row['stage_transitions'] = [{
            'from_stage': 'recon', 'to_stage': 'execution', 'artifact_evidence_ids': ['shared-log']}]
    payload, parameters = configured(rows, minimum_publishers=2, cap=2)
    parameters['tailored']['evidence_policies'] = {'attack_chain': {'unbacked_stage_weight': 1.}}
    result = invoke(payload, parameters)
    assert {row['evidence_id'] for row in result['evidence']} == {'top-A', 'top-B', 'shared-log'}
    assert [row['evidence_id'] for row in result['evidence']].count('shared-log') == 1
    assert result['provenance']['incident_publisher_counts'] == {'A': 1, 'B': 1}
    assert result['provenance']['publisher_counts'] == {'A': 2, 'B': 1}
