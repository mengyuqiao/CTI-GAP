"""F5: removing heterogeneous agents retains reconsideration and human review."""
from pathlib import Path

import pytest

from annotation_app import AnnotationService
from test_discovery_repairs import production_harness
from test_taxonomy_and_categorization import harness, _reconfigure, _span


PROTOCOL = {'strategy': 'single_judge', 'judge': 'gpt_judge',
            'reconsideration': {'mode': 'fresh_reassessment'}}


def make_old_baseline_executable(world, tmp_path):
    # Makes the pre-repair one-pass implementation execute, exposing its real
    # methodological error instead of an unrelated unresolved-template error.
    template = tmp_path/'one-pass-baseline.jinja2'
    template.write_text('{{ task_input | tojson }}')
    _reconfigure(world, {'baselines.gpt_judge.prompt_template_path': str(template)})
    world.annotations = AnnotationService(world.store)


def test_single_agent_arm_keeps_two_rounds_and_resume_without_other_agents(harness, tmp_path):
    make_old_baseline_executable(harness, tmp_path)
    result = harness.runner()._categorization_ablation(
        'multi_agent_decision', harness.failure_ids, harness.taxonomy, PROTOCOL, 0)
    assert result['status'] == 'completed'
    assert len(harness.models.requests) == 6
    assert {item['model_id'] for item in harness.models.requests} == {'GPT-5'}
    assert {item['schema_name'] for item in harness.models.requests} == {'rq3_round_1', 'rq3_round_2'}
    assert set(result['labels_by_response']) == set(harness.failure_ids)
    assert result['executed_mechanisms']['rounds'] == 2
    assert result['executed_mechanisms']['selective_human_verification'] is True
    calls = len(harness.models.provider_calls)
    resumed = harness.runner()._categorization_ablation(
        'multi_agent_decision', harness.failure_ids, harness.taxonomy, PROTOCOL, 0)
    assert resumed == result
    assert len(harness.models.provider_calls) == calls


def test_label_change_in_single_agent_reconsideration_still_requires_native_human_review(harness, tmp_path):
    make_old_baseline_executable(harness, tmp_path)
    def predict(request, packet):
        label = 'fixture-B' if request['schema_name'] == 'rq3_round_2' else 'fixture-A'
        return {'labels': [label], 'evidence_spans': [_span(label)], 'uncertain': False}
    harness.models.script = predict
    result = harness.runner()._categorization_ablation(
        'multi_agent_decision', harness.failure_ids[:1], harness.taxonomy, PROTOCOL, 0)
    assert result['status'] == 'waiting_for_annotations'
    assert result['review_triggers'] == {harness.failure_ids[0]: ['cross_round_label_change']}
    tasks = result['pending_annotation_task_ids']
    assert len(tasks) == 1
    harness.annotations.resolve_task(tasks[0], 'fixture-reviewer', {
        'labels': ['fixture-B'], 'evidence_spans': [_span('fixture-B')], 'uncertain': False})
    final = harness.runner()._categorization_ablation(
        'multi_agent_decision', harness.failure_ids[:1], harness.taxonomy, PROTOCOL, 0)
    assert final['status'] == 'completed'
    assert final['labels_by_response'] == {harness.failure_ids[0]: ['fixture-B']}
    assert final['human_review_performed'] is True
    assert len(harness.models.requests) == 2


@pytest.mark.parametrize('production_harness', [True], indirect=True)
@pytest.mark.parametrize('mode', ['fresh_reassessment', 'self_review'])
def test_single_agent_arm_runs_with_real_readiness_and_config_identity(production_harness, mode):
    world = production_harness
    protocol = {**PROTOCOL, 'reconsideration': {'mode': mode}}
    result = world.runner()._categorization_ablation(
        'multi_agent_decision', world.failure_ids[:1], world.taxonomy, protocol, 1)
    assert result['status'] == 'completed'
    assert len(world.models.requests) == 2
    assert {item['run_index'] for item in world.models.requests} == {1}
    second = world.models.packets[next(key for key,value in world.models.packets.items()
        if 'single_agent_reconsideration' in value)]
    assert second['single_agent_reconsideration']['mode'] == mode
    assert ('previous_prediction' in second['single_agent_reconsideration']) == (mode == 'self_review')


def test_single_agent_reconsideration_must_be_explicit(harness):
    with pytest.raises(ValueError, match='reconsideration'):
        harness.runner()._categorization_ablation('multi_agent_decision', harness.failure_ids[:1],
            harness.taxonomy, {'strategy': 'single_judge', 'judge': 'gpt_judge'}, 0)
    assert harness.models.requests == []


@pytest.mark.parametrize('production_harness', [True], indirect=True)
def test_public_ablation_controller_preserves_repetitions_and_matched_source_population(
    production_harness, tmp_path, monkeypatch,
):
    import hashlib
    import json
    import litellm
    import research_backends
    import yaml
    from config import ExperimentConfig
    from model_client import ModelClient
    from schemas import FailureDecision, ModelResponse, _content_hash
    from test_taxonomy_and_categorization import _resolve
    world = production_harness
    payload = world.config.model_dump(mode='json')
    conditions = [{'preparation_adapter': {
        'callable': 'research_backends:prepare_condition', 'version': 'f5-matched-v1',
        'source_sha256': hashlib.sha256(Path(research_backends.__file__).read_bytes()).hexdigest(),
        'parameters': {'changes': [{'pointer': '/operational/output_path',
            'before': payload['operational']['output_path'], 'after': str(tmp_path/f'condition-{index}')}],
            'rationale': 'Separate execution artifacts; preserve scientific controls.'},
    }} for index in range(2)]
    for path,value in {
        'operational.selected_ablation': 'multi_agent_decision',
        'operational.prerequisite_manifests.responses': 'urn:fixture:responses',
        'operational.prerequisite_manifests.references': 'urn:fixture:references',
        'inference.reasoning_settings': {'parameters':{}, 'adapters': {entry.name: {
            'supported_parameters': ['temperature','top_p','max_tokens','seed'],
            'api_base': 'http://127.0.0.1:1', 'probability_interface': None,
        } for entry in world.config.models.categorization_panel}},
        'ablations.repetition_count': 2,
        'ablations.replacement_behavior_by_component': {'multi_agent_decision': {
            **PROTOCOL, 'affected_stages': ['categorize'], 'conditions': conditions,
            'comparison_denominator': 'frozen_matched_failure_population',
        }},
    }.items():
        _resolve(payload,path,value)
    config_file=tmp_path/'single-agent-controller.yaml'
    config_file.write_text(yaml.safe_dump(payload))
    config=ExperimentConfig.load(str(config_file))
    config.validate('ablations')
    source_ids=[]
    for identity in world.failure_ids:
        raw=world.store.read_record('model_response',identity)
        derived_id=_content_hash({'source':identity,'config_hash':config.config_hash})
        raw.update(response_id=derived_id,config_hash=config.config_hash)
        response=ModelResponse.model_validate(raw)
        world.store.create_record('model_response',derived_id,response.model_dump(mode='json'))
        original=world.store.list_records('failure_decision',{'response_id':identity})[0]
        decision=FailureDecision.model_validate({**original,'response_id':derived_id})
        world.store.create_record('failure_decision',_content_hash(decision),decision.model_dump(mode='json'))
        source_ids.append(derived_id)
    world.config=config
    world.models=ModelClient(config,world.store)
    calls=[]
    def complete(**request):
        calls.append(request)
        output={'labels':['fixture-A'],'evidence_spans':[_span('fixture-A')],'uncertain':False}
        return {'choices':[{'message':{'content':json.dumps(output)}}], 'usage':{},'model':'offline-fixture'}
    monkeypatch.setattr(litellm,'completion',complete)
    monkeypatch.setattr(litellm,'token_counter',lambda **kwargs:1)
    result=world.runner().run_ablation('multi_agent_decision',source_ids)
    assert result['status']=='completed',result
    assert len(result['conditions'])==2
    assert len(calls)==12
    cohorts=[]
    for condition in result['conditions']:
        execution=condition['result']
        assert execution['executed_mechanisms']['agent_ids']==['GPT-5']
        assert execution['executed_mechanisms']['rounds']==2
        assert execution['executed_mechanisms']['selective_human_verification'] is True
        projection=world.store.read_record('experiment_manifest',condition['condition_manifest_id'])
        cohorts.append(projection['source_response_ids'])
    assert cohorts==[sorted(source_ids),sorted(source_ids)]
    assert world.runner().run_ablation('multi_agent_decision',source_ids)==result
    assert len(calls)==12


def test_single_agent_execution_does_not_require_removed_agent_endpoints(harness):
    payload=harness.config.model_dump(mode='json')
    for entry in payload['models']['categorization_panel']:
        if entry['name']!='GPT-5':
            entry['endpoint_or_checkpoint']=None
            entry['revision']=None
    harness.config=type(harness.config).model_validate(payload)
    harness.models.config=harness.config
    result=harness.runner()._categorization_ablation('multi_agent_decision',harness.failure_ids[:1],
        harness.taxonomy,PROTOCOL,0)
    assert result['status']=='completed',result
    assert len(harness.models.requests)==2
    assert {request['model_id'] for request in harness.models.requests}=={'GPT-5'}
