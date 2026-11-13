"""Offline paper-H.2 regressions through the source-pinned production adapter.

Encoder scores below are deliberately declared replay fixtures. Expectations are
hand-checked source selections and metadata, not scores computed by the backend.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

import research_backends
from experiments import ExperimentRunner
from schemas import _content_hash
from test_taxonomy_and_categorization import Harness, _configure_c3, _reconfigure, harness


def passage(identity, publisher, text, **annotations):
    return {
        'evidence_id': identity, 'document_id': 'doc-'+identity,
        'role': 'model_input', 'snippet': text, 'publisher': publisher,
        'publication_time': '2024-01-01', 'available_at': '2024-01-01',
        'source_version': 'source-v1', 'content_hash': _content_hash(text),
        'incident_roles': ['infrastructure'], 'environment': {'platform': 'linux'},
        **annotations,
    }


def configured(passages, *, budget=10000, minimum_publishers=1, cap=10, **preprocessing):
    payload = {
        'instance_id': 'offline-case', 'snapshot_date': '2024-06-01',
        'model_input_evidence': [{'snippet': 'infrastructure evidence'}],
        'environment': {'platform': 'linux'},
    }
    parameters = {
        'passages': passages, 'mode': 'frozen_scores', 'bm25': {'k1': 1.2, 'b': .75},
        'retrieval_depth': len(passages), 'reranking_depth': len(passages),
        'context_budget_chars': budget, 'rrf_rank_constant': 60,
        'frozen_scores': {
            'query_hash': _content_hash('infrastructure evidence\n{"platform":"linux"}'),
            'passage_hash': _content_hash(passages),
            'dense_scores': list(range(len(passages), 0, -1)),
            'cross_encoder_scores': list(range(len(passages), 0, -1)),
            'model_identity': {'dense': 'recorded-fixture', 'cross_encoder': 'recorded-fixture'},
        },
        'tailored': {
            'require_incident_roles': True, 'minimum_publishers': minimum_publishers,
            'maximum_per_publisher': cap, 'environment_fields': ['platform'],
            'maximum_ioc_reuse': 10, 'source_quality_weight': 0., 'recency_weight': 0.,
            'maximum_age_days': 366, 'required_metadata': [],
            'preprocessing': preprocessing,
        },
    }
    return payload, parameters


def invoke(payload, parameters):
    # _invoke checks the actual module's source digest and JSON boundary, exactly
    # as _retrieve does. No model service or retrieval helper is replaced.
    runner = object.__new__(ExperimentRunner)
    return runner._invoke({
        'callable': 'research_backends:tailored_retrieval', 'version': 'offline-h2-v1',
        'source_sha256': hashlib.sha256(Path(research_backends.__file__).read_bytes()).hexdigest(),
        'parameters': parameters,
    }, payload, 'retrieval.adapter')


@pytest.mark.parametrize('defect', ['same_publisher', 'wrong_environment'])
def test_tailored_gates_and_publisher_caps_precede_the_final_context_budget(defect):
    first = passage('a', 'publisher-A', 'a report')
    second = passage('b', 'publisher-A', 'b report')
    third = passage('c', 'publisher-B', 'c report')
    if defect == 'wrong_environment':
        first['environment'] = {'platform': 'windows'}
    payload, parameters = configured([first, second, third], budget=16, minimum_publishers=2, cap=1)
    output = invoke(payload, parameters)
    assert [item['evidence_id'] for item in output['evidence']] == (
        ['a', 'c'] if defect == 'same_publisher' else ['b', 'c'])
    assert output['provenance']['context_characters'] == 16
    assert output['provenance']['publishers'] == ['publisher-A', 'publisher-B']


def test_diversity_reserves_space_for_shorter_lower_ranked_whole_spans():
    records = [passage('long', 'A', 'a'*14), passage('short', 'A', 'b'*5), passage('other', 'B', 'c'*10)]
    payload, parameters = configured(records, budget=15, minimum_publishers=2, cap=1)
    output = invoke(payload, parameters)
    assert [item['evidence_id'] for item in output['evidence']] == ['short', 'other']
    assert [item['snippet'] for item in output['evidence']] == ['bbbbb', 'cccccccccc']


def test_near_duplicate_reports_keep_source_lineage_without_false_corroboration():
    text = 'the observed infrastructure provided a payload delivery channel for the incident '
    records = [passage('original', 'A', text+'today'), passage('syndicated', 'B', text+'yesterday'),
               passage('independent', 'C', 'independent telemetry establishes a separate observation')]
    payload, parameters = configured(records, minimum_publishers=2, near_duplicate_threshold=.7)
    before = copy.deepcopy(parameters)
    output = invoke(payload, parameters)
    assert [item['evidence_id'] for item in output['evidence']] == ['original', 'independent']
    original = output['evidence'][0]
    assert original['snippet'] == text+'today'
    assert original['content_hash'] == records[0]['content_hash']
    assert original['normalized_record']['duplicate_sources'][0]['evidence_id'] == 'syndicated'
    assert original['normalized_record']['duplicate_sources'][0]['publisher'] == 'B'
    assert output['provenance']['publishers'] == ['A', 'C']
    assert parameters == before
    parameters['tailored']['minimum_publishers'] = 3
    with pytest.raises(ValueError, match='diversity'):
        invoke(payload, parameters)


def test_ioc_profiles_use_all_admitted_annotations_and_exclude_future_reports():
    records = [
        passage('current', 'A', 'current bad.test infrastructure', ioc_observations=[
            {'ioc': 'bad.test', 'observed_at': '2024-04-01', 'campaign_id': 'campaign-2'}]),
        passage('historical', 'B', 'historical bad.test campaign', ioc_observations=[
            {'ioc': 'bad.test', 'observed_at': '2023-12-01', 'campaign_id': 'campaign-1'}]),
        passage('future', 'C', 'future bad.test campaign', publication_time='2025-01-01',
                available_at='2025-01-01', ioc_observations=[
                    {'ioc': 'bad.test', 'observed_at': '2025-01-01', 'campaign_id': 'campaign-3'}]),
    ]
    payload, parameters = configured(records)
    parameters['reranking_depth'] = 1  # The profile must still see the historical report.
    output = invoke(payload, parameters)
    profile = output['evidence'][0]['normalized_record']['ioc_reuse_profiles'][0]
    assert profile['ioc'] == 'bad.test'
    assert profile['first_seen'] == '2023-12-01'
    assert profile['last_seen'] == '2024-04-01'
    assert profile['frequency'] == 2
    assert profile['campaign_count'] == 2
    assert profile['campaign_ids'] == ['campaign-1', 'campaign-2']
    assert profile['supporting_evidence_ids'] == ['current', 'historical']
    assert output['evidence'][0]['ioc_reuse_count'] == 2


def test_derived_ioc_campaign_reuse_drives_the_real_gate_without_keyword_role_invention():
    records = [
        passage('reused', 'A', 'bad.test', ioc_observations=[
            {'ioc': 'bad.test', 'observed_at': '2024-04-01', 'campaign_id': 'c2'}]),
        passage('old', 'B', 'older bad.test observation', ioc_observations=[
            {'ioc': 'bad.test', 'observed_at': '2023-12-01', 'campaign_id': 'c1'}]),
        passage('specific', 'C', 'a distinct explicitly linked delivery behavior', campaign_specific_evidence=True),
        passage('co-mention', 'D', 'operator victim malware named here', incident_roles=[]),
    ]
    payload, parameters = configured(records)
    parameters['tailored']['maximum_ioc_reuse'] = 1
    assert [item['evidence_id'] for item in invoke(payload, parameters)['evidence']] == ['specific']


def test_exact_aliases_and_typed_schema_maps_reach_the_production_structured_feed():
    records = [passage('alias', 'A', 'FictionAlias met Fictionalias and FictionAliasExtra',
        entities=[{'surface': 'FictionAlias'}, {'surface': 'Fictionalias'}, {'surface': 'FictionAliasExtra'}],
        source_schema='vendor', source_record={'actor_name': 'FictionAlias', 'version_text': '2.1'})]
    payload, parameters = configured(records, alias_table={'FictionAlias': 'CanonicalActor'},
        schema_maps={'vendor': {'actor': 'actor_name', 'version': 'version_text'}},
        field_definitions={'actor': 'Reported actor name', 'version': 'Reported tool version'})
    evidence = invoke(payload, parameters)['evidence']
    feeds = ExperimentRunner._retrieval_structured_feeds(evidence)
    assert feeds[0]['canonical_entities'] == [
        {'surface': 'FictionAlias', 'canonical': 'CanonicalActor'},
        {'surface': 'Fictionalias', 'canonical': 'Fictionalias'},
        {'surface': 'FictionAliasExtra', 'canonical': 'FictionAliasExtra'},
    ]
    normalized = feeds[0]['normalized_record']
    assert normalized['fields'] == {'actor': 'CanonicalActor', 'version': '2.1'}
    assert normalized['field_definitions'] == {'actor': 'Reported actor name', 'version': 'Reported tool version'}
    assert evidence[0]['source_record'] == {'actor_name': 'FictionAlias', 'version_text': '2.1'}
    assert evidence[0]['snippet'] == records[0]['snippet']


def test_alias_table_also_normalizes_preexisting_canonical_entity_annotations():
    records = [passage('entity', 'A', 'AnnotatedAlias appears in the source',
                       canonical_entities=[{'surface':'AnnotatedAlias','canonical':'AnnotatedAlias'}])]
    payload, parameters = configured(records, alias_table={'AnnotatedAlias':'CanonicalEntity'})
    assert invoke(payload, parameters)['evidence'][0]['canonical_entities'] == [
        {'surface':'AnnotatedAlias','canonical':'CanonicalEntity'}]


def test_views_are_contiguous_chronological_bundles_with_grounding_links():
    records = [passage('new-A', 'A', 'new A observation', publication_time='2024-05-01'),
               passage('mid-B', 'B', 'middle B observation', publication_time='2024-03-01'),
               passage('old-A', 'A', 'old A observation', publication_time='2024-01-01')]
    payload, parameters = configured(records)
    evidence = invoke(payload, parameters)['evidence']
    assert [item['evidence_id'] for item in evidence] == ['old-A', 'new-A', 'mid-B']
    feeds = ExperimentRunner._retrieval_structured_feeds(evidence)
    assert [item['claim_group'] for item in feeds] == ['source:A', 'source:A', 'source:B']
    assert feeds[0]['normalized_record']['view']['evidence_ids'] == ['old-A', 'new-A']
    assert feeds[0]['normalized_record']['grounding'] == {
        'evidence_id': 'old-A', 'publisher': 'A', 'publication_time': '2024-01-01',
        'as_of': '2024-06-01', 'span': [0, 17],
    }


def test_curated_distinct_claim_views_are_never_lexically_deduplicated():
    text = 'same quoted infrastructure with distinct curated interpretations'
    records = [passage('supports', 'A', text, claim_group='positive-view'),
               passage('disputes', 'B', text, claim_group='negative-view')]
    payload, parameters = configured(records, minimum_publishers=2)
    output = invoke(payload, parameters)
    assert [item['evidence_id'] for item in output['evidence']] == ['supports', 'disputes']


def test_duplicate_representative_choice_does_not_preclude_feasible_publisher_diversity():
    records = [passage('top-A', 'A', 'a syndicated quoted report'),
               passage('copy-B', 'B', 'a syndicated quoted report'),
               passage('independent-A', 'A', 'an independent observation')]
    payload, parameters = configured(records, minimum_publishers=2, cap=1)
    output = invoke(payload, parameters)
    assert [item['evidence_id'] for item in output['evidence']] == ['copy-B', 'independent-A']
    assert output['evidence'][0]['normalized_record']['duplicate_sources'][0]['evidence_id'] == 'top-A'


def test_diversity_that_cannot_fit_the_budget_blocks_without_truncation():
    payload, parameters = configured([passage('a', 'A', 'a'*9), passage('b', 'B', 'b'*9)],
                                     budget=17, minimum_publishers=2)
    with pytest.raises(ValueError, match='diversity'):
        invoke(payload, parameters)


def test_causal_context_controls_reweight_and_cap_only_explicit_classifications():
    records = [passage('context', 'A', 'context!', evidence_kind='context_only'),
               passage('causal', 'B', 'causal!!', evidence_kind='causal')]
    payload, parameters = configured(records, budget=8)
    parameters['tailored']['evidence_policies'] = {
        'causal_context': {'context_only_weight': .01, 'maximum_context_only': 1,
                           'require_classification': True}}
    result = invoke(payload, parameters)
    assert [item['evidence_id'] for item in result['evidence']] == ['causal']
    assert result['evidence'][0]['normalized_record']['evidence_kind'] == 'causal'
    parameters['context_budget_chars'] = 16
    parameters['tailored']['evidence_policies']['causal_context']['maximum_context_only'] = 0
    assert [item['evidence_id'] for item in invoke(payload, parameters)['evidence']] == ['causal']


def test_missing_causal_classification_is_reported_without_inference_from_words():
    payload, parameters = configured([passage('unknown', 'A', 'causal exploit mechanism telemetry')])
    parameters['tailored']['evidence_policies'] = {
        'causal_context': {'context_only_weight': .1, 'require_classification': True}}
    with pytest.raises(ValueError, match='classification'):
        invoke(payload, parameters)


def test_stage_confidence_notes_use_admitted_artifacts_and_include_their_spans():
    records = [passage('report', 'A', 'reported stage sequence', stage_transitions=[
        {'from_stage': 'recon', 'to_stage': 'execution', 'artifact_evidence_ids': ['log']},
        {'from_stage': 'execution', 'to_stage': 'persistence', 'artifact_evidence_ids': ['future-log']},
    ]), passage('log', 'telemetry', 'retained historical log', artifact_type='log', incident_roles=[]),
        passage('future-log', 'telemetry', 'unavailable future log', artifact_type='log',
                incident_roles=[], publication_time='2025-01-01', available_at='2025-01-01')]
    payload, parameters = configured(records)
    parameters['tailored']['evidence_policies'] = {'attack_chain': {'unbacked_stage_weight': .5}}
    result = invoke(payload, parameters)
    assert [item['evidence_id'] for item in result['evidence']] == ['report', 'log']
    transitions = result['evidence'][0]['normalized_record']['stage_transitions']
    assert transitions[0]['confidence_note'] == 'artifact_backed'
    assert transitions[0]['artifact_evidence_ids'] == ['log']
    assert transitions[1]['confidence_note'] == 'soft_hint_without_admitted_artifacts'
    assert transitions[1]['artifact_evidence_ids'] == []
    assert transitions[1]['unavailable_artifact_evidence_ids'] == ['future-log']
    assert 'unavailable future log' not in str(result)


def test_authoritative_mapping_checks_attach_actual_snapshot_bounded_references():
    records = [passage('invalid', 'A', 'unrecognized technique', standard_mappings={'technique_id': 'T-BAD'}),
               passage('valid', 'B', 'recognized technique', standard_mappings={'technique_id': 'T-TEST'}),
               passage('standard', 'authority', 'T-TEST is the declared permitted identifier',
                       authoritative_options={'technique_id': ['T-TEST']}, incident_roles=[])]
    payload, parameters = configured(records)
    parameters['tailored']['evidence_policies'] = {'standards': {
        'reference_evidence_ids': ['standard'], 'mapping_fields': ['technique_id'], 'on_invalid': 'exclude'}}
    result = invoke(payload, parameters)
    assert [item['evidence_id'] for item in result['evidence']] == ['valid', 'standard']
    check = result['evidence'][0]['normalized_record']['standard_validation']['technique_id']
    assert check == {'value': 'T-TEST', 'valid': True, 'reference_evidence_ids': ['standard']}
    assert result['evidence'][1]['normalized_record']['authoritative_options'] == {'technique_id': ['T-TEST']}
    parameters['passages'][2]['available_at'] = '2025-01-01'
    parameters['frozen_scores']['passage_hash'] = _content_hash(parameters['passages'])
    with pytest.raises(ValueError, match='authoritative.*admitted'):
        invoke(payload, parameters)


def test_geography_tooling_quotas_and_counterevidence_are_selected_before_budgeting():
    records = [passage('common-A', 'A', 'first!!!', diversity_metadata={'geography': 'common', 'tooling': 'X'}),
               passage('common-B', 'B', 'second!!', diversity_metadata={'geography': 'common', 'tooling': 'X'}),
               passage('rare-C', 'C', 'third!!!', diversity_metadata={'geography': 'rare', 'tooling': 'Y'},
                       evidence_relation='counterevidence')]
    payload, parameters = configured(records, budget=16, minimum_publishers=2)
    parameters['tailored']['evidence_policies'] = {
        'diversity': {'fields': ['geography', 'tooling'], 'inverse_frequency_weight': 0.,
                      'minimum_distinct': {'geography': 2}, 'maximum_per_value': {'tooling': 1}},
        'counterevidence': {'priority_bonus': 0., 'minimum_records': 1}}
    result = invoke(payload, parameters)
    assert [item['evidence_id'] for item in result['evidence']] == ['common-A', 'rare-C']
    assert result['evidence'][1]['normalized_record']['diversity_metadata'] == {'geography': 'rare', 'tooling': 'Y'}
    assert result['evidence'][1]['normalized_record']['evidence_relation'] == 'counterevidence'
    parameters['tailored']['evidence_policies']['diversity']['minimum_distinct']['geography'] = 3
    with pytest.raises(ValueError, match='diversity|quota'):
        invoke(payload, parameters)


def test_explicit_counterevidence_priority_can_displace_a_higher_ranked_shortcut():
    records = [passage('shortcut', 'A', 'support!', evidence_relation='support'),
               passage('counter', 'B', 'counter!', evidence_relation='counterevidence')]
    payload, parameters = configured(records, budget=8)
    parameters['tailored']['evidence_policies'] = {'counterevidence': {'priority_bonus': 2., 'minimum_records': 1}}
    result = invoke(payload, parameters)
    assert [item['evidence_id'] for item in result['evidence']] == ['counter']


def test_geography_reweighting_uses_admitted_metadata_frequencies():
    records = [passage('common-A', 'A', 'first!!!', diversity_metadata={'geography': 'common'}),
               passage('common-B', 'B', 'second!!', diversity_metadata={'geography': 'common'}),
               passage('rare-C', 'C', 'third!!!', diversity_metadata={'geography': 'rare'})]
    payload, parameters = configured(records, budget=8)
    parameters['tailored']['evidence_policies'] = {
        'diversity': {'fields': ['geography'], 'inverse_frequency_weight': 2.}}
    assert [item['evidence_id'] for item in invoke(payload, parameters)['evidence']] == ['rare-C']


def test_metadata_quotas_cannot_override_the_incident_publisher_requirement():
    payload, parameters = configured([passage('one', 'A', 'only one actual publisher')], minimum_publishers=2)
    parameters['tailored']['evidence_policies'] = {
        'diversity':{'fields':['publisher'], 'minimum_distinct':{'publisher':1}}}
    with pytest.raises(ValueError, match='publisher|diversity'):
        invoke(payload, parameters)


def test_unselected_stage_artifacts_do_not_spend_the_final_context_budget():
    records = [passage('expensive', 'A', 'a'*9, stage_transitions=[
        {'from_stage':'one','to_stage':'two','artifact_evidence_ids':['log']}]),
        passage('independent', 'B', 'b'*8),
        passage('log', 'logs', 'c'*8, artifact_type='log', incident_roles=[])]
    payload, parameters = configured(records, budget=16)
    parameters['tailored']['evidence_policies'] = {'attack_chain':{'unbacked_stage_weight':1.}}
    result = invoke(payload, parameters)
    assert [item['evidence_id'] for item in result['evidence']] == ['independent']
    assert result['provenance']['context_characters'] == 8


def test_real_tailored_runner_receives_preprocessed_sources_in_the_task_prompt(harness: Harness, tmp_path):
    from schemas import TaskInstance
    _configure_c3(harness, tmp_path)
    instance = TaskInstance.model_validate(harness.store.read_record('task_instance',harness.instance_ids[0]))
    report = passage('alias-report', 'source-A', 'FictionAlias explicitly linked to this incident',
        entities=[{'surface':'FictionAlias'}], environment=instance.environment,
        evidence_kind='causal', evidence_relation='counterevidence',
        diversity_metadata={'geography':'rare'},
        source_schema='vendor', source_record={'actor_name':'FictionAlias'})
    _, parameters = configured([report], alias_table={'FictionAlias':'ResolvedFixtureActor'},
        schema_maps={'vendor':{'actor':'actor_name'}}, field_definitions={'actor':'Reported actor name'})
    parameters['frozen_scores']['query_hash'] = _content_hash(research_backends.retrieval_query({
        'model_input_evidence':[item.model_dump(mode='json') for item in instance.evidence_records],
        'environment':instance.environment}))
    parameters['tailored']['environment_fields'] = []
    parameters['tailored']['evidence_policies'] = {
        'causal_context':{'context_only_weight':.1, 'require_classification':True},
        'counterevidence':{'priority_bonus':1., 'minimum_records':1},
        'diversity':{'fields':['geography'], 'minimum_distinct':{'geography':1}}}
    policies = tmp_path/'retrieval-policies.yaml'
    policies.write_text('version: offline-source-policies-v1\npolicies: {}\n',encoding='utf-8')
    declaration = {
        'callable':'research_backends:tailored_retrieval', 'version':'offline-h2-v1',
        'source_sha256':hashlib.sha256(Path(research_backends.__file__).read_bytes()).hexdigest(),
        'parameters':parameters}
    _reconfigure(harness, {
        'operational.selected_defense':'tailored_mitigation',
        'operational.selected_mitigation_mode':'retrieval_only',
        'defenses.tailored_mitigation.subtype_policy_manifest':str(policies),
        'defenses.tailored_mitigation.routing_policy':{'mode':'universal','uses_gold_failure_labels':False},
        'defenses.tailored_mitigation.model_group_aggregation':{
            'mode':'pooled_response_counts','groups':{'general':['G5']}},
        'defenses.tailored_mitigation.retrieval_models':{'adapter':declaration},
        'defenses.tailored_mitigation.repetition_count':1,
        'defenses.tailored_mitigation.evidence_context_budget':10000,
        'failure_identification.hard_label_rules':{'tasks':{'threat_actor_linking':{'route':'hard'}}},
        'evaluation.scoring.response_level_failure_criteria':{'threat_actor_linking':{
            'pass_if':{'op':'equal','left':{'source':'prediction','pointer':'/answer/choice'},
                       'right':{'source':'reference','pointer':'/answer/choice'}},
            'on_missing':'error','availability':{}}},
    })
    def answer(request, packet):
        return {'status':'OK','task':packet['task_name'],'case_id':packet['case_id'],
                'snapshot_date':packet['snapshot_date'],'answer':{'choice':'B'},
                'justification':'Supplied fictional evidence.'}
    harness.models.script = answer
    result = harness.runner().run_tailored_mitigation('retrieval_only',harness.instance_ids[:1])
    assert result['status'] == 'completed'
    prompt = harness.models.requests[0]['prompt']
    for expected in ('ResolvedFixtureActor','source:source-A','Reported actor name',
                     'counterevidence','causal','rare','alias-report'):
        assert expected in prompt
    retained = [item for item in harness.store.list_records('experiment_manifest')
                if item.get('purpose') == 'retrieval_evidence_package']
    assert retained[0]['result']['evidence'][0]['snippet'] == report['snippet']
