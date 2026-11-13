"""Real CLI initialization/configuration/SQLite contracts (no hash overrides)."""
import copy
import json
from pathlib import Path
import pytest
import yaml
from config import ExperimentConfig
from main import Main
from schemas import StageStatus


def _config(tmp_path):
    payload = yaml.safe_load(Path('config.yaml').read_text())
    payload.update(experiment_id='offline-cli-smoke',corpus_version='fixture-v1')
    payload['operational'] = {'database_path':str(tmp_path/'db.sqlite'),
        'artifact_root':str(tmp_path/'blobs'),'output_path':str(tmp_path/'reports'),
        'prerequisite_manifests':{'results':'urn:stage:evaluate','scoring':'urn:stage:evaluate'}}
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(payload))
    return path


def test_stage_reference_resolves_current_config_completed_result_without_hash_cycle(tmp_path):
    path = _config(tmp_path)
    app = Main(str(path))
    digest = app._config.config_hash
    app._checkpoint('evaluate',app._inputs('evaluate'),StageStatus.COMPLETED,
                    {'status':'completed','response_ids':['frozen-response']})
    reopened = Main(str(path))
    assert reopened._config.config_hash == digest
    manifest = reopened._resolve_manifest('urn:stage:evaluate')
    assert manifest['result']['response_ids'] == ['frozen-response']
    assert manifest['config_hash'] == digest
    reopened._config.validate('report')
    report = reopened.run_stage('report')
    assert report['status'] == 'completed'
    assert Path(report['report_path']).is_file()


def test_unfinished_stage_reference_cannot_be_consumed(tmp_path):
    app = Main(str(_config(tmp_path)))
    with pytest.raises(ValueError,match='completed'):
        app._resolve_manifest('urn:stage:evaluate')

from test_data_and_storage import baseline_yaml, store_fixture, local_bundle
from test_metrics_and_identification import _resolved_payload
from test_taxonomy_and_categorization import _resolve


def test_prepare_evaluate_identify_report_and_resume_with_real_config(local_bundle, tmp_path, monkeypatch):
    """Production pipeline; only external model transport/tokenizer is simulated."""
    import litellm
    bundle = local_bundle
    cfg = bundle.config.model_dump(mode='json')
    conversions = json.loads(bundle.conversions_path.read_text())
    conversion = conversions['conversions'][0]
    conversions['conversions'] = [conversion]
    bundle.conversions_path.write_text(json.dumps(conversions))
    selection_path = Path(cfg['data']['benchmarks']['ctibench']['selection_manifest'])
    selection = json.loads(selection_path.read_text())
    selection['item_ids'] = ['shared-case-item']
    selection_path.write_text(json.dumps(selection))
    cfg['operational'].update(selected_tasks=['affected_systems'], selected_models=['G5'],
        prerequisite_manifests={'corpus':'urn:stage:prepare','references':'urn:stage:prepare',
          'responses':'urn:stage:evaluate','identification':'urn:stage:identify',
          'results':'urn:stage:identify','scoring':'urn:stage:evaluate'})
    # Fixture policies are declared before any derived response identities exist.
    metric = _resolved_payload(tmp_path)
    for key,value in metric['evaluation']['scoring'].items():
        _resolve(cfg, f'evaluation.scoring.{key}', value)
    for key,value in metric['evaluation']['scoring']['f1'].items():
        _resolve(cfg, f'evaluation.scoring.f1.{key}', value)
    _resolve(cfg,'evaluation.scoring.task_field_extraction_rules',{'affected_systems':{
        'prediction':'/answer/decisions','reference':'/answer/decisions',
        'representation':'aligned','unit_ids':'/candidates','prediction_kind':'binary'}})
    _resolve(cfg,'evaluation.scoring.response_level_failure_criteria',{'affected_systems':{
        'pass_if':{'op':'equal','left':{'source':'prediction','pointer':'/answer/decisions'},
                   'right':{'source':'reference','pointer':'/answer/decisions'}},'on_missing':'unresolved',
        'availability':{name:'failure' for name in ['NEED_MORE_CONTEXT','UNSUPPORTED','refusal','invalid_schema','transport_failure']}}})
    _resolve(cfg,'failure_identification.hard_label_rules',{'tasks':{'affected_systems':{'route':'hard'}}})
    guidance_path = tmp_path/'guidance.json'
    guidance_path.write_text(json.dumps({'version':'fixture-v1','tasks':{'affected_systems':{
        'guidance':'Judge each asset using supplied evidence.','candidate_policy':'include'}}}))
    profile_path = tmp_path/'profiles.json'
    profile_path.write_text(json.dumps({'version':'fixture-v1','tasks':{'affected_systems':{
        'output_profile':conversion['output_profile'],'schema_contract':{'version':'fixture-v1','kind':'task',
        'json_schema':{'type':'object','required':['status','task','case_id','snapshot_date','answer','justification'],
                       'properties':{'answer':{'type':'object','required':['decisions'],
                         'properties':{'decisions':{'type':'array','items':{'type':'integer'}}}}}}}}}}))
    guideline_path = tmp_path/'guideline.json'
    guideline_path.write_text(json.dumps({'version':'fixture-v1','criteria':'Any wrong asset decision fails.'}))
    resolved = {
      'models.evaluation.G5.endpoint_or_checkpoint':'openai/offline-fixture',
      'models.evaluation.G5.revision':'fixture-v1',
      'inference.temperature':0.,'inference.top_p':1.,'inference.max_output_tokens':256,
      'inference.context_window':100000,'inference.concurrency':1,'inference.batch_size':1,
      'inference.truncation_policy':'reject','inference.precision':'fixture-not-applicable',
      'inference.quantization':'fixture-not-applicable',
      'inference.reasoning_settings':{'parameters':{},'adapters':{'G5':{
          'supported_parameters':['temperature','top_p','max_tokens','seed'],
          'api_base':'http://127.0.0.1:1','probability_interface':None}}},
      'inference.retry_policy':{'max_attempts':1,'timeout_seconds':1.,'retryable_status_codes':[],
          'retryable_exception_names':[],'backoff_seconds':[],'retry_ambiguous_timeouts':False},
      'evaluation.run_seeds':[0,1,2,3,4], 'evaluation.standard_deviation_ddof':0,
      'evaluation.prompt.template_path':str(Path('prompts/task_evaluation.jinja2').resolve()),
      'evaluation.prompt.task_guidance_manifest':str(guidance_path),
      'evaluation.prompt.output_profile_manifest':str(profile_path),
      'taxonomy.guideline_manifest':str(guideline_path)}
    for key,value in resolved.items():
        _resolve(cfg,key,value)
    path = tmp_path/'pipeline.yaml'
    path.write_text(yaml.safe_dump(cfg))
    app = Main(str(path))
    app.run_stage('prepare')
    calls = []
    def complete(**request):
        prompt = request['messages'][0]['content']
        assert 'WITHHELD_REFERENCE' not in prompt
        calls.append(request)
        output = {'status':'OK','task':'affected_systems','case_id':conversion['case_id'],
                  'snapshot_date':conversion['snapshot_date'],'answer':{'decisions':[1,0]},
                  'justification':'Fixture response based on input.', 'evidence_refs':['fixture-narrative']}
        return {'id':'fixture','choices':[{'message':{'content':json.dumps(output)},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':100,'completion_tokens':50,'total_tokens':150}}
    monkeypatch.setattr(litellm,'completion',complete)
    monkeypatch.setattr(litellm,'token_counter',lambda **kwargs:100)
    evaluated = app.run_stage('evaluate')
    assert len(evaluated['response_ids']) == 5
    assert len(calls) == 5
    identified = app.run_stage('identify')
    assert identified['coverage']['complete']
    assert identified['failure_response_ids'] == []
    assert Main(str(path)).resume('evaluate') == evaluated
    assert len(calls) == 5
    report = app.run_stage('report')
    assert Path(report['report_path']).is_file()
    # The complete metric uses five identical correct candidate sets, not a published result.
    group = evaluated['task_metrics']['groups'][0]
    assert group['summary']['mean'] == 100
    assert group['summary']['standard_deviation'] == 0


def test_baseline_dispatch_categorizes_identified_failures_only(tmp_path, monkeypatch):
    path = _config(tmp_path)
    cfg = yaml.safe_load(path.read_text())
    cfg['operational']['selected_baseline'] = 'kmeans'
    path.write_text(yaml.safe_dump(cfg))
    app = Main(str(path))
    monkeypatch.setattr(app,'_response_ids',lambda:['pass','failure'])
    monkeypatch.setattr(app,'_failure_ids',lambda:['failure'])
    seen = []
    def run(name, population):
        seen.extend(population)
        return {'status':'completed'}
    monkeypatch.setattr(app._runner,'run_baseline',run)
    app._dispatch('baselines')
    assert seen == ['failure']
