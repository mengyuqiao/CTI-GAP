"""Single-component human-initialization ablation preserves later RQ2 steps."""
import copy
import pytest
from taxonomy import TaxonomyManager
from schemas import TaxonomyVersion
from test_discovery_repairs import production_harness, _submit_proposals
from test_taxonomy_and_categorization import harness, _proposal, _span


def _script(request, packet):
    if request['schema_name'] == 'taxonomy_initialization':
        return {**_proposal(('fixture-A',)), 'evidence_spans':[_span('fixture-A')]}
    return {'labels':['fixture-A'],'evidence_spans':[_span('fixture-A')],'uncertain':False}


def test_model_seed_uses_no_human_initialization_and_retains_discovery(production_harness):
    world = production_harness
    world.models.script = _script
    manager = TaxonomyManager(world.config,world.store,world.models,world.annotations,initialization='model')
    result = manager.run(world.failure_ids)
    assert result['status'] == 'completed'
    assert result['execution_mode'] == 'discovery'
    tasks = world.store.list_records('annotation_task')
    assert not any(t['purpose'] == 'taxonomy_initialization' for t in tasks)
    taxonomy = TaxonomyVersion.model_validate(world.store.read_record('taxonomy',result['taxonomy_id']))
    definition = taxonomy.subtype_definitions[0]
    assert definition.supporting_annotation_ids == []
    assert definition.provenance['origin'] == 'model_initialization'
    assert definition.provenance['model_proposal_response_id']
    assert manager.run(world.failure_ids) == result


def test_model_seed_still_routes_other_to_real_human_refinement(production_harness):
    world = production_harness
    def script(request, packet):
        if request['schema_name'] == 'taxonomy_initialization':
            return _script(request,packet)
        label = 'other' if len(packet['current_taxonomy']['subtype_definitions']) == 1 else 'fixture-B'
        return {'labels':[label],'evidence_spans':[_span(label)],'uncertain':False}
    world.models.script = script
    manager = TaxonomyManager(world.config,world.store,world.models,world.annotations,initialization='model')
    waiting = manager.run(world.failure_ids)
    assert waiting['status'] == 'waiting_for_annotations'
    from schemas import AnnotationTask
    tasks = [AnnotationTask.model_validate(world.store.read_record('annotation_task',identity))
             for identity in waiting['pending_annotation_task_ids']]
    assert tasks and all(task.purpose.value == 'taxonomy_refinement' for task in tasks)
    _submit_proposals(world,tasks,('fixture-B',))
    result = manager.run(world.failure_ids)
    assert result['status'] == 'completed'
    final = TaxonomyVersion.model_validate(world.store.read_record('taxonomy',result['taxonomy_id']))
    assert {d.subtype_id for d in final.subtype_definitions} == {'fixture-A','fixture-B'}


def test_model_seed_rejects_unsupported_evidence_without_human_provenance(production_harness):
    world = production_harness
    world.models.script = lambda req,pkt: {**_proposal(('fixture-A',)),
        'evidence_spans':[{**_span('fixture-A'),'quote':'not in any source'}]}
    with pytest.raises(ValueError,match='evidence'):
        TaxonomyManager(world.config,world.store,world.models,world.annotations,
                        initialization='model').run(world.failure_ids)


@pytest.mark.parametrize("production_harness", ["native_seed"], indirect=True)
def test_model_initialization_uses_real_model_client_schema_and_request_path(production_harness, monkeypatch):
    import json
    import litellm
    from model_client import ModelClient
    world = production_harness
    calls = []
    def complete(**request):
        prompt = request['messages'][0]['content']
        calls.append(prompt)
        output = ({**_proposal(('fixture-A',)), 'evidence_spans':[_span('fixture-A')]}
                  if 'deliberately model-initialized ablation' in prompt else
                  {'labels':['fixture-A'],'evidence_spans':[_span('fixture-A')],'uncertain':False})
        return {'choices':[{'message':{'content':json.dumps(output)},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':100,'completion_tokens':20,'total_tokens':120}}
    monkeypatch.setattr(litellm,'completion',complete)
    monkeypatch.setattr(litellm,'token_counter',lambda **kwargs:100)
    model = ModelClient(world.config,world.store)
    result = TaxonomyManager(world.config,world.store,model,world.annotations,initialization='model').run(world.failure_ids)
    assert result['status'] == 'completed'
    proposals = [request for request in world.store.list_records('request')
                 if request['schema_name'] == 'taxonomy_initialization']
    assert len(proposals) == 1
    assert proposals[0]['provider_options']['schema_contract']['kind'] == 'taxonomy_proposal'
    calls_before = len(calls)
    resumed = TaxonomyManager(world.config,world.store,model,world.annotations,initialization='model').run(world.failure_ids)
    assert resumed == result
    assert len(calls) == calls_before


def test_model_seed_checks_offsets_as_well_as_quote_substring(production_harness):
    world = production_harness
    world.models.script = lambda req,pkt: {**_proposal(('fixture-A',)),
        'evidence_spans':[{**_span('fixture-A'),'start_offset':1,'end_offset':5}]}
    with pytest.raises(ValueError,match='evidence'):
        TaxonomyManager(world.config,world.store,world.models,world.annotations,
                        initialization='model').run(world.failure_ids)


def test_model_initialization_cannot_silently_apply_a_fixed_taxonomy(production_harness):
    from config import ExperimentConfig
    world = production_harness
    payload = world.config.model_dump(mode='json')
    payload['operational']['taxonomy_mode'] = 'fixed'
    config = ExperimentConfig.model_validate(payload)
    with pytest.raises(ValueError,match='discovery'):
        TaxonomyManager(config,world.store,world.models,world.annotations,initialization='model')
