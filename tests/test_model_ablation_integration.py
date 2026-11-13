"""Production ablation routing with synthetic transport, not empirical targets.

The model seed is the ablation that removes human initialization.
All readiness gates, condition projection, discovery, and RQ3 remain real.
"""
import copy
import hashlib
import json
from pathlib import Path

import litellm
import pytest
import yaml

from config import ExperimentConfig
from model_client import ModelClient
from schemas import FailureDecision, ModelResponse, TaxonomyVersion, _content_hash
from test_discovery_repairs import production_harness
from test_taxonomy_and_categorization import _proposal, _resolve, _span, harness


def _configure(world, tmp_path, *, mixed_versions=False, fixed_mode=False):
    """Give every real service explicit, hash-bound local test declarations."""
    import research_backends
    payload = world.config.model_dump(mode='json')
    condition = {'preparation_adapter':{
        'callable':'research_backends:prepare_condition', 'version':'offline-integration-v1',
        'source_sha256':hashlib.sha256(Path(research_backends.__file__).read_bytes()).hexdigest(),
        'parameters':{
            'changes':[{'pointer':'/operational/output_path',
                        'before':payload['operational']['output_path'],
                        'after':str(tmp_path/'model-seeded-condition')}],
            'rationale':'Isolate the ablation output directory; all scientific controls stay fixed.'}}}
    model_ids = ['G5', *(entry.name for entry in world.config.models.categorization_panel)]
    reasoning = {'parameters':{}, 'adapters':{identity:{
        'supported_parameters':['temperature','top_p','max_tokens','seed'],
        'api_base':'http://127.0.0.1:1', 'probability_interface':None,
    } for identity in model_ids}}
    for path, value in {
        'operational.selected_ablation':'human_initiation',
        'operational.prerequisite_manifests.responses':'urn:fixture:responses',
        'operational.prerequisite_manifests.references':'urn:fixture:references',
        'inference.reasoning_settings':reasoning,
        'ablations.repetition_count':1,
        'ablations.replacement_behavior_by_component':{'human_initiation':{
            'strategy':'model_initialization',
            'affected_stages':['discover_taxonomy','categorize'],
            'comparison_denominator':'frozen_matched_failure_population',
            'conditions':[condition],
        }},
    }.items():
        _resolve(payload, path, value)
    if fixed_mode:
        _resolve(payload, 'operational.taxonomy_mode', 'fixed')
        _resolve(payload, 'taxonomy.discovery_initial_taxonomy',
                 world.taxonomy.model_dump(mode='json'))
    filename = tmp_path/'ablation-controller.yaml'
    filename.write_text(yaml.safe_dump(payload))
    config = ExperimentConfig.load(str(filename))
    config.validate('ablations')
    config.validate('discover_taxonomy')
    config.validate('categorize')
    assert type(config) is ExperimentConfig
    # The source cohort was evaluated under this exact base configuration.
    # Create new immutable records, preserving the original production fixture.
    response_ids = []
    for index,identity in enumerate(world.failure_ids):
        raw = world.store.read_record('model_response', identity)
        response_id = _content_hash({'source':identity, 'base_config_hash':config.config_hash})
        raw.update(response_id=response_id, config_hash=config.config_hash)
        response = ModelResponse.model_validate(raw)
        world.store.create_record('model_response', response_id, response.model_dump(mode='json'))
        original = world.store.list_records('failure_decision', {'response_id':identity})
        assert len(original) == 1
        decision_raw = {**original[0], 'response_id':response_id}
        if mixed_versions and index == len(world.failure_ids)-1:
            decision_raw['identification_version'] = 'incompatible-source-identification-version'
        decision = FailureDecision.model_validate(decision_raw)
        world.store.create_record('failure_decision', _content_hash(decision),
                                 decision.model_dump(mode='json'))
        response_ids.append(response_id)
    world.config = config
    world.models = ModelClient(config, world.store)
    return sorted(response_ids)


def _synthetic_transport(monkeypatch):
    calls = []
    def complete(**request):
        prompt = request['messages'][0]['content']
        calls.append(prompt)
        if 'deliberately model-initialized ablation' in prompt:
            output = {**_proposal(('fixture-A',)), 'evidence_spans':[_span('fixture-A')]}
        else:
            output = {'labels':['fixture-A'], 'evidence_spans':[_span('fixture-A')],
                      'uncertain':False}
        return {'choices':[{'message':{'content':json.dumps(output)}, 'finish_reason':'stop'}],
                'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}}
    monkeypatch.setattr(litellm, 'completion', complete)
    monkeypatch.setattr(litellm, 'token_counter', lambda **kwargs:100)
    return calls


@pytest.mark.parametrize('production_harness', ['dynamic_taxonomy'], indirect=True)
def test_actual_ablation_controller_discovers_model_seed_and_categorizes_matched_cohort(
        production_harness, tmp_path, monkeypatch):
    world = production_harness
    base_ids = _configure(world, tmp_path)
    originals = {identity:copy.deepcopy(world.store.read_record('model_response',identity))
                 for identity in base_ids}
    calls = _synthetic_transport(monkeypatch)
    result = world.runner().run_ablation('human_initiation', base_ids)
    assert result['status'] == 'completed'
    assert result['component'] == 'human_initiation'
    assert len(result['conditions']) == 1
    condition = result['conditions'][0]
    effective = world.store.read_record('experiment_manifest', condition['condition_manifest_id'])
    assert effective['source_response_ids'] == base_ids
    derived_ids = effective['derived_response_ids']
    assert len(derived_ids) == len(base_ids) == 3
    assert not set(derived_ids) & set(base_ids)
    for identity in derived_ids:
        projected = world.store.read_record('model_response', identity)
        source = originals[projected['usage']['source_response_id']]
        assert projected['raw_text'] == source['raw_text']
        assert projected['parsed_answer'] == source['parsed_answer']
        assert projected['instance_id'] == source['instance_id']
        assert projected['model_id'] == source['model_id']
        assert projected['config_hash'] == effective['effective_config_hash']
    assert all(world.store.read_record('model_response',identity) == raw
               for identity,raw in originals.items())
    outcome = condition['result']
    assert outcome['status'] == 'completed'
    assert outcome['strategy'] == 'model_initialization'
    assert outcome['initialization'] == 'model'
    assert outcome['taxonomy_discovery_executed'] is True
    assert outcome['taxonomy_discovery']['execution_mode'] == 'discovery'
    assert outcome['comparison_scope'] == 'model_initialized_discovery_and_standard_categorization'
    assert set(outcome['final_assignment_ids_by_response']) == set(derived_ids)
    taxonomy = TaxonomyVersion.model_validate(world.store.read_record('taxonomy',outcome['taxonomy_id']))
    assert taxonomy.frozen
    assert {definition.subtype_id for definition in taxonomy.subtype_definitions} == {'fixture-A'}
    assert taxonomy.subtype_definitions[0].supporting_annotation_ids == []
    assert taxonomy.subtype_definitions[0].provenance['origin'] == 'model_initialization'
    requests = world.store.list_records('request')
    proposal_requests = [request for request in requests
                         if request['schema_name'] == 'taxonomy_initialization']
    assert len(proposal_requests) == 1
    assert proposal_requests[0]['provider_options']['schema_contract']['kind'] == 'taxonomy_proposal'
    assert not any(task['purpose'] == 'taxonomy_initialization'
                   for task in world.store.list_records('annotation_task'))
    final_assignments = [world.store.read_record('final_assignment',identity)
                         for identity in outcome['final_assignment_ids_by_response'].values()]
    assert all(assignment['labels'] == ['fixture-A'] for assignment in final_assignments)
    rounds = [assignment for assignment in world.store.list_records('label_assignment')
               if assignment['response_id'] in derived_ids and assignment['origin'] == 'agent']
    assert {assignment['round_index'] for assignment in rounds} == {1,2}
    assert len(rounds) == 3 * 4 * 2
    before_resume = len(calls)
    assert world.runner().run_ablation('human_initiation',base_ids) == result
    assert len(calls) == before_resume


@pytest.mark.parametrize('production_harness', ['dynamic_taxonomy'], indirect=True)
def test_condition_projection_preserves_rejection_of_mixed_identification_versions(
        production_harness, tmp_path, monkeypatch):
    world = production_harness
    base_ids = _configure(world, tmp_path, mixed_versions=True)
    calls = _synthetic_transport(monkeypatch)
    with pytest.raises(ValueError, match='different identification versions'):
        world.runner().run_ablation('human_initiation', base_ids)
    assert calls == []
    assert not world.store.list_records('request')


@pytest.mark.parametrize('production_harness', ['dynamic_taxonomy'], indirect=True)
def test_model_initialization_ablation_rejects_a_fixed_taxonomy_comparator(
        production_harness, tmp_path, monkeypatch):
    world = production_harness
    base_ids = _configure(world, tmp_path, fixed_mode=True)
    calls = _synthetic_transport(monkeypatch)
    with pytest.raises(ValueError, match='discovery|fixed'):
        world.runner().run_ablation('human_initiation', base_ids)
    assert calls == []
