"""F1: actual generation inputs survive classification and human review."""
import copy
from pathlib import Path

import pytest

from annotation_app import AnnotationService
from schemas import FailureDecision, TaskInstance, _content_hash
from test_discovery_repairs import production_harness
from test_taxonomy_and_categorization import harness, _span


RETRIEVED = {
    'evidence_id': 'independent-retrieved', 'document_id': 'independent-retrieved',
    'snippet': 'Retrieved incident explicitly links the indicator to this event.',
    'source_id': 'independent-publisher', 'role': 'model_input',
    'publication_time': '2024-05-25',
}


def generated(world, *, mitigation=True):
    """Use the actual production generation entry point and immutable artifacts."""
    runner = world.runner()
    instance = TaskInstance.model_validate(world.store.read_record('task_instance', world.instance_ids[0]))
    payload = {
        'case_id': instance.case_id, 'task_name': instance.task_name,
        'cti_stage': instance.stage.value, 'snapshot_date': instance.snapshot_date.isoformat(),
        'source_documents': [copy.deepcopy(RETRIEVED)] if mitigation else [
            item.model_dump(mode='json') for item in instance.evidence_records],
        'structured_feeds': [{'evidence_id': RETRIEVED['evidence_id'],
            'normalized_record': {'relation': 'incident-linked'}}] if mitigation else [],
        'candidates': [], 'candidate_policy': 'exclude', 'environment': instance.environment,
        'task_guidance': 'Assess only the incident-linked evidence.' if mitigation else 'Baseline guidance.',
        'output_profile': instance.output_profile,
        'temperature': 0., 'top_p': 1.,
        'prompt_settings': world.config.evaluation.prompt.model_dump(mode='json'),
    }
    world.models.script = lambda request, packet: {
        'status': 'OK', 'task': packet['task_name'], 'case_id': packet['case_id'],
        'snapshot_date': packet['snapshot_date'], 'answer': {'choice': 'A'},
        'justification': 'An intentionally incorrect controlled fixture.',
    }
    response = runner._generate_task(instance, 'G5', 0, payload,
        {'kind': 'task', 'version': 'f1-context-v1', 'json_schema': {'type': 'object'}},
        {'protocol': 'tailored_mitigation' if mitigation else 'main_task_evaluation',
         'condition': 'retrieval_only' if mitigation else 'original'},
        template=str(Path(__file__).resolve().parents[1] / 'prompts/task_evaluation.jinja2'))
    source = world.store.list_records('failure_decision', {'response_id': world.failure_ids[0]})[0]
    source['response_id'] = response.response_id
    decision = FailureDecision.model_validate(source)
    world.store.create_record('failure_decision', _content_hash(decision), decision.model_dump(mode='json'))
    world.store.freeze_records('failure_decision', [_content_hash(decision)])
    return response, payload


@pytest.mark.parametrize('production_harness', [True], indirect=True)
def test_classifier_and_native_human_review_use_actual_generation_materials(production_harness):
    world = production_harness
    response, payload = generated(world)
    classifier = world.categorizer()
    packet = classifier._packet(response.response_id, world.taxonomy)
    materials = world.annotations._materials(response.response_id, classifier._annotation_protocol(world.taxonomy))
    for task_input in (packet['task_input'], materials['task_input']):
        assert task_input['task_guidance'] == 'Assess only the incident-linked evidence.'
        assert [item['evidence_id'] for item in task_input['source_documents']] == ['independent-retrieved']
        assert task_input['structured_feeds'] == payload['structured_feeds']
        assert task_input['candidates'] == []
        assert task_input['candidate_policy'] == 'exclude'
    assert 'fixture-reference' not in world.models.requests[0]['prompt']
    assert 'Reference actor is Correct.' not in world.models.requests[0]['prompt']
    context_id = response.usage['generation_context_id']
    context = world.store.read_record('experiment_manifest', context_id)
    assert context['payload'] == payload
    assert context['payload_hash'] == _content_hash(payload)
    assert world.store.list_records('experiment_manifest', {'record_id': context_id, 'lifecycle': ['immutable','frozen']})


@pytest.mark.parametrize('production_harness', [True], indirect=True)
def test_independent_reviewers_retrieved_quote_passes_real_classifier_and_human_validation(production_harness):
    world = production_harness
    response, _ = generated(world)
    world.models.script = lambda request, packet: {
        'labels': ['fixture-A'], 'uncertain': False,
        'evidence_spans': [{'source_ref': 'independent-retrieved', 'quote': RETRIEVED['snippet'],
                            'subtype_id': 'fixture-A'}],
    }
    result = world.categorizer().run([response.response_id], world.taxonomy)
    assert result['status'] == 'completed', result
    tasks = world.categorizer()._review_tasks(response.response_id, world.taxonomy)
    world.annotations.resolve_task(tasks[0].annotation_task_id, 'fixture-reviewer', {
        'labels': ['fixture-A'], 'uncertain': False,
        'evidence_spans': [{'source_ref': 'independent-retrieved', 'quote': RETRIEVED['snippet'],
                            'subtype_id': 'fixture-A'}],
    })
    assert world.annotations.get_reference(response.response_id, 'operational_adjudication')['labels'] == ['fixture-A']


@pytest.mark.parametrize('production_harness', [True], indirect=True)
def test_condition_contexts_remain_distinct_and_retrieved_citations_do_not_validate_for_baseline(production_harness):
    world = production_harness
    mitigated, _ = generated(world)
    baseline, _ = generated(world, mitigation=False)
    packet = world.categorizer()._packet(baseline.response_id, world.taxonomy)
    assert packet['task_input']['task_guidance'] == 'Baseline guidance.'
    assert packet['task_input']['structured_feeds'] == []
    assert mitigated.usage['generation_context_id'] != baseline.usage['generation_context_id']
    tasks = world.categorizer()._review_tasks(baseline.response_id, world.taxonomy)
    with pytest.raises(ValueError, match='unavailable|source'):
        world.annotations.resolve_task(tasks[0].annotation_task_id, 'fixture-reviewer', {
            'labels': ['fixture-A'], 'uncertain': False,
            'evidence_spans': [{'source_ref': 'independent-retrieved', 'quote': RETRIEVED['snippet'],
                               'subtype_id': 'fixture-A'}],
        })


@pytest.mark.parametrize('production_harness', [True], indirect=True)
def test_wrong_context_identity_and_missing_mitigation_context_fail_closed(production_harness):
    from generation_context import review_task_input
    from schemas import ModelResponse
    world = production_harness
    mitigated, _ = generated(world)
    baseline, _ = generated(world, mitigation=False)
    instance = TaskInstance.model_validate(world.store.read_record('task_instance', mitigated.instance_id))
    forged = mitigated.model_dump(mode='json')
    forged['usage']['generation_context_id'] = baseline.usage['generation_context_id']
    with pytest.raises(ValueError, match='bind'):
        review_task_input(world.store, ModelResponse.model_validate(forged), instance)
    forged['usage'].pop('generation_context_id')
    with pytest.raises(ValueError, match='lacks.*context'):
        review_task_input(world.store, ModelResponse.model_validate(forged), instance)


@pytest.mark.parametrize('production_harness', [True], indirect=True)
def test_reference_only_fields_are_rejected_before_any_generation(production_harness):
    world = production_harness
    response, payload = generated(world)
    instance = TaskInstance.model_validate(world.store.read_record('task_instance', response.instance_id))
    calls = len(world.models.requests)
    payload['structured_feeds'].append({'reference_answer': 'must not reach inference'})
    with pytest.raises(ValueError, match='internal field'):
        world.runner()._generate_task(instance, 'G5', 1, payload,
            {'kind':'task','version':'f1-context-v1','json_schema':{'type':'object'}},
            {'protocol':'tailored_mitigation','condition':'retrieval_only'},
            template=str(Path(__file__).resolve().parents[1]/'prompts/task_evaluation.jinja2'))
    assert len(world.models.requests) == calls


@pytest.mark.parametrize('production_harness', [True], indirect=True)
def test_review_projection_does_not_expose_unrendered_document_metadata(production_harness):
    world = production_harness
    old = RETRIEVED.get('evaluator_diagnostic')
    try:
        RETRIEVED['evaluator_diagnostic'] = 'must not enter review or inference'
        response, _ = generated(world)
    finally:
        if old is None:
            RETRIEVED.pop('evaluator_diagnostic')
        else:
            RETRIEVED['evaluator_diagnostic'] = old
    packet = world.categorizer()._packet(response.response_id, world.taxonomy)
    assert 'evaluator_diagnostic' not in packet['task_input']['source_documents'][0]
    assert 'must not enter review or inference' not in world.models.requests[0]['prompt']


def test_actual_tailored_failure_reaches_native_human_adjudication_with_retrieved_citation(harness, tmp_path):
    from test_audit_retrieval_policy import configure_controller
    from test_taxonomy_and_categorization import _reconfigure
    configure_controller(harness, tmp_path)
    policy=copy.deepcopy(harness.config.categorization.incomplete_agent_output_policy)
    policy['annotation_protocol']['assignments']={'*':['fixture-reviewer']}
    policy['annotation_protocol']['reference_applicability']={'*':{'scope':'Supplied fictional incident.'}}
    _reconfigure(harness, {'categorization.incomplete_agent_output_policy':policy})
    harness.annotations=AnnotationService(harness.store)
    quote='specific exploit delivery mechanism'
    def predict(request, packet):
        if request['schema_name']=='cti_task_evaluation':
            return {'status':'OK','task':packet['task_name'],'case_id':packet['case_id'],
                'snapshot_date':packet['snapshot_date'],'answer':{'choice':'A'},
                'justification':'Controlled wrong answer requiring categorization.'}
        label='fixture-B' if request['model_id']=='Claude Sonnet 4' else 'fixture-A'
        return {'labels':[label],'uncertain':False,'evidence_spans':[
            {'source_ref':'good-A','quote':quote,'subtype_id':label}]}
    harness.models.script=predict
    result=harness.runner().run_tailored_mitigation('retrieval_only',harness.instance_ids[:1])
    assert result['status']=='waiting_for_annotations',result
    assert result['categorization']['operational_exceptions']=={}
    pending=result['pending_annotation_task_ids']
    assert len(pending)==1
    task=harness.store.read_record('annotation_task',pending[0])
    assert [item['evidence_id'] for item in task['visible_payload']['task_input']['source_documents']]==[
        'good-A','good-B','good-C']
    harness.annotations.resolve_task(pending[0],'fixture-reviewer',{
        'labels':['fixture-A'],'uncertain':False,
        'evidence_spans':[{'source_ref':'good-A','quote':quote,'subtype_id':'fixture-A'}]})
    final=harness.runner().run_tailored_mitigation('retrieval_only',harness.instance_ids[:1])
    assert final['status']=='completed',final
    assert final['failure_count']==1
    assert final['model_group_subtype_prevalence']['general']['percentages']['fixture-A']==100.


@pytest.mark.parametrize('production_harness', [True], indirect=True)
def test_model_seed_discovery_uses_generated_example_context_and_evidence(production_harness):
    from taxonomy import TaxonomyManager
    from test_taxonomy_and_categorization import _proposal
    world=production_harness
    generated_response,_=generated(world)
    def script(request,packet):
        if request['schema_name']=='taxonomy_initialization':
            example=packet['initialization_examples'][0]
            assert example['task_guidance']=='Assess only the incident-linked evidence.'
            assert example['structured_feeds']
            return {**_proposal(('fixture-A',)),'evidence_spans':[
                {'source_ref':'independent-retrieved','quote':RETRIEVED['snippet'],'subtype_id':'fixture-A'}]}
        return {'labels':['fixture-A'],'evidence_spans':[_span('fixture-A')],'uncertain':False}
    world.models.script=script
    manager=TaxonomyManager(world.config,world.store,world.models,world.annotations,initialization='model')
    result=manager.run([generated_response.response_id,world.failure_ids[0]])
    assert result['status']=='completed',result


@pytest.mark.parametrize('production_harness', [True], indirect=True)
def test_generation_context_survives_two_validated_condition_projections(production_harness, tmp_path):
    import hashlib
    import research_backends
    from evaluation import Evaluator
    from experiments import ExperimentRunner
    from generation_context import review_task_input
    from schemas import ModelResponse
    world=production_harness
    response,_=generated(world)
    def declaration(config,index):
        return {'preparation_adapter':{'callable':'research_backends:prepare_condition','version':'f1-condition-v1',
            'source_sha256':hashlib.sha256(Path(research_backends.__file__).read_bytes()).hexdigest(),
            'parameters':{'changes':[{'pointer':'/operational/output_path',
                'before':config.operational.output_path,'after':str(tmp_path/f'projection-{index}')}],
                'rationale':'Change execution artifacts; preserve generation and scientific inputs.'}}}
    first,ids,_=world.runner()._condition(declaration(world.config,1),[response.response_id],'projection-1')
    runner=ExperimentRunner(first,world.store,world.models,Evaluator(first,world.store),world.annotations)
    second,ids,_=runner._condition(declaration(first,2),ids,'projection-2')
    projected=ModelResponse.model_validate(world.store.read_record('model_response',ids[0]))
    instance=TaskInstance.model_validate(world.store.read_record('task_instance',response.instance_id))
    actual=review_task_input(world.store,projected,instance)
    assert actual['task_guidance']=='Assess only the incident-linked evidence.'
    assert [item['evidence_id'] for item in actual['source_documents']]==['independent-retrieved']


def test_baseline_judge_classifies_mitigated_output_using_actual_context(harness, tmp_path):
    from test_taxonomy_and_categorization import _reconfigure
    template=tmp_path/'baseline-classification.jinja2'
    template.write_text('{{ task_input | tojson }}')
    _reconfigure(harness, {
        'operational.selected_baseline':'gpt_judge',
        'baselines.gpt_judge.prompt_template_path':str(template),
        'baselines.prevalence_denominator':'classified_responses',
        'baselines.repetition_count':1,
    })
    response,_=generated(harness)
    def classify(request,packet):
        assert packet['task_input']['task_guidance']=='Assess only the incident-linked evidence.'
        assert packet['task_input']['structured_feeds']
        return {'labels':['fixture-A'],'uncertain':False,'evidence_spans':[
            {'source_ref':'independent-retrieved','quote':RETRIEVED['snippet'],'subtype_id':'fixture-A'}]}
    harness.models.script=classify
    result=harness.runner().run_baseline('gpt_judge',[response.response_id])
    assert result['status']=='completed'
    prediction=harness.store.read_record('experiment_manifest',result['prediction_manifest_ids'][0])
    assert prediction['labels_by_response']=={response.response_id:['fixture-A']}
