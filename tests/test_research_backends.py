"""Independent numerical and admission checks for shipped offline backends."""
import copy
import math
import pytest
from schemas import _content_hash
from research_backends import bm25_scores, fuse_rankings, normalize_log_likelihoods, hybrid_retrieval, inference_constraints


def test_bm25_distinguishes_frequency_length_and_no_overlap():
    scores = bm25_scores('cve patch', ['cve cve patch', 'unrelated', 'patch'], k1=1.2, b=0.75)
    # N=3, df(cve)=1, df(patch)=2, average length=5/3.
    idf_cve = math.log(1 + 2.5/1.5)
    idf_patch = math.log(1 + 1.5/2.5)
    expected = idf_cve * (2*2.2)/(2+1.2*(.25+.75*3/(5/3)))
    expected += idf_patch*2.2/(1+1.2*(.25+.75*3/(5/3)))
    assert scores[0] == pytest.approx(expected)
    assert scores[1] == 0
    assert scores[0] > scores[2] > 0


def test_rrf_combines_rank_positions_and_candidate_logsumexp_is_stable():
    assert fuse_rankings([[0, 1], [1, 2]], rank_constant=1) == pytest.approx({0:.5, 1:5/6, 2:1/3})
    assert normalize_log_likelihoods({'a': -10000, 'b': -10000-math.log(3)}) == pytest.approx({'a': .75,'b': .25})
    with pytest.raises(ValueError):
        normalize_log_likelihoods({'a': float('nan')})


def _retrieval():
    payload = {'instance_id':'case', 'snapshot_date':'2024-06-01',
               'model_input_evidence':[{'snippet':'cve patch'}], 'environment':{}}
    passages = [{'evidence_id':f'e{i}', 'document_id':f'd{i}', 'role':'model_input',
                 'snippet': text, 'publication_time':'2024-01-01',
                 'available_at':'2024-01-01', 'content_hash':_content_hash(text),
                 'source_version':'v1'} for i,text in enumerate(['cve patch','unrelated','patch'])]
    from research_backends import retrieval_query
    parameters = {'passages':passages,'mode':'frozen_scores', 'bm25':{'k1':1.2,'b':.75},
                  'retrieval_depth':3, 'reranking_depth':3, 'context_budget_chars':40,
                  'rrf_rank_constant':60,
                  'frozen_scores':{'query_hash':_content_hash(retrieval_query(payload)),
                                   'passage_hash':_content_hash(passages),
                                   'dense_scores':[.1,.9,.2], 'cross_encoder_scores':[.8,.1,.7],
                                   'model_identity':{'dense':'fixture-recorded','cross_encoder':'fixture-recorded'}}}
    return payload, parameters


def test_hybrid_pipeline_reranks_fused_candidates_and_enforces_snapshot_and_budget():
    payload, parameters = _retrieval()
    out = hybrid_retrieval(payload=payload, parameters=parameters)
    assert [p['evidence_id'] for p in out['evidence']] == ['e0','e2','e1']
    assert out['provenance']['context_characters'] <= 40
    assert all(p['snapshot_admission_verified'] for p in out['evidence'])
    changed = copy.deepcopy(parameters)
    changed['passages'][0]['available_at'] = '2025-01-01'
    # Exact persisted scores cannot be reused after corpus edits.
    with pytest.raises(ValueError):
        hybrid_retrieval(payload=payload, parameters=changed)


def test_inference_constraints_use_policy_without_gold_routing():
    out = inference_constraints(payload={'original_guidance':'Do task.',
        'subtype_policy_manifest':{'policies':{'1.1':{'inference':'Require relation evidence.'}}}}, parameters={})
    assert out['task_guidance'] == 'Do task.\nRequire relation evidence.'


def test_forwarded_retrieval_controls_cannot_disagree_with_backend_parameters():
    payload, parameters = _retrieval()
    payload['retrieval_configuration'] = {'retrieval_depth':1}
    with pytest.raises(ValueError,match='disagree'):
        hybrid_retrieval(payload=payload,parameters=parameters)


def test_alignscore_checks_verbatim_pretty_json_claims_without_evaluation_labels(tmp_path, monkeypatch):
    import sys
    import types
    from research_backends import alignscore_verification
    calls = []
    class Scorer:
        def __init__(self, **kwargs):
            assert kwargs['device'] == 'cpu'
        def score(self, *, contexts, claims):
            calls.append((contexts,claims))
            return [.75] * len(claims)
    monkeypatch.setitem(sys.modules, 'alignscore',types.SimpleNamespace(AlignScore=Scorer))
    checkpoint = tmp_path/'checkpoint'; checkpoint.touch()
    backbone = tmp_path/'backbone'; backbone.mkdir()
    raw = '{"answer": {"decisions": [1, 0]}, "status": "OK"}'
    result = alignscore_verification(payload={'candidate_response_id':'candidate',
        'candidate_raw_text':raw,'task_input':{'candidates':['asset A','asset B']},'retrieved_evidence':[{'evidence_id':'e','snippet':'asset A affected'}]},
        parameters={'checkpoint':str(checkpoint),'backbone':str(backbone),'batch_size':1,
            'evaluation_mode':'nli_sp','alignment_threshold':.5,'claim_rules':[{'pointer':'/answer/decisions',
                'mode':'aligned_candidates','positive':'affected','negative':'unaffected',
                'template':'{candidate} is {decision}'}]})
    assert result['claims'][0]['candidate_quote'] == '1'
    assert result['claims'][0]['alignment_score'] == .75
    assert calls == [(['asset A affected']*2, ['asset A is affected','asset B is unaffected'])]
    assert 'decision' not in result


def test_tailored_retrieval_requires_incident_role_and_diverse_environment_matched_evidence():
    from research_backends import tailored_retrieval
    payload, params = _retrieval()
    params['mode'] = 'frozen_scores'
    for i,p in enumerate(params['passages']):
        p.update(publisher='one' if i < 2 else 'two',
                 incident_roles=['infrastructure'] if i != 1 else [],
                 environment={'platform':'linux'}, ioc_reuse_count=1,
                 campaign_specific_evidence=True, claim_group='view-'+str(i),
                 canonical_entities=[], normalized_record={})
    payload['environment'] = {'platform':'linux'}
    from research_backends import retrieval_query
    params['frozen_scores']['query_hash'] = _content_hash(retrieval_query(payload))
    params['frozen_scores']['passage_hash'] = _content_hash(params['passages'])
    params['tailored'] = {'require_incident_roles':True,'minimum_publishers':2,
        'maximum_per_publisher':1,'environment_fields':['platform'],
        'maximum_ioc_reuse':2,'source_quality_weight':0.,'recency_weight':0.,
        'maximum_age_days':366,'required_metadata':['claim_group','normalized_record','canonical_entities']}
    out = tailored_retrieval(payload=payload,parameters=params)
    assert [p['evidence_id'] for p in out['evidence']] == ['e0','e2']
    assert out['provenance']['publishers'] == ['one','two']
    params['tailored']['minimum_publishers'] = 3
    with pytest.raises(ValueError,match='diversity'):
        tailored_retrieval(payload=payload,parameters=params)


def test_native_cpu_candidate_likelihoods_include_eos_and_length(tmp_path):
    """A real tiny uniform causal LM gives independently known probabilities."""
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    from tokenizers import Tokenizer, models, pre_tokenizers
    from research_backends import candidate_probabilities
    vocab = {'[UNK]':0, '[EOS]':1, 'query':2, 'A':3, 'B':4, '[PAD]':5}
    token_engine = Tokenizer(models.WordLevel(vocab,unk_token='[UNK]'))
    token_engine.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=token_engine,
        unk_token='[UNK]',eos_token='[EOS]',pad_token='[PAD]')
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.save_pretrained(tmp_path)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=len(vocab),n_positions=32,n_embd=8,n_layer=1,n_head=1,
        bos_token_id=1,eos_token_id=1,pad_token_id=5))
    with torch.no_grad():
        for value in model.parameters():
            value.zero_()
    model.save_pretrained(tmp_path)
    from research_backends import checkpoint_digest
    digest = checkpoint_digest(tmp_path)
    out = candidate_probabilities(payload={'model_id':'tiny-fixture','prompt':'query','endpoint_or_checkpoint':str(tmp_path),
                 'revision':'all-zero-weights-fixture','message_mapping':'single_user_message',
                 'generation_checkpoint_sha256':digest},
        parameters={'model_id':'tiny-fixture','checkpoint':str(tmp_path),'endpoint_or_checkpoint':str(tmp_path),
                    'revision':'all-zero-weights-fixture','checkpoint_sha256':digest,'context_window':32,
                    'candidates':{'short':'A','long':'A B'}})
    assert out['probabilities'] == pytest.approx({'short':6/7,'long':1/7},abs=1e-7)
    assert out['provenance']['log_likelihoods'] == pytest.approx({
        'short':-2*math.log(6),'long':-3*math.log(6)},abs=1e-6)
    assert out['provenance']['device'] == 'cpu'


def test_hybrid_accepts_budget_forwarded_by_real_configuration_schema():
    payload, parameters = _retrieval()
    payload['retrieval_configuration'] = {'retrieval_depth':3,'reranking_depth':3,'evidence_context_budget':40}
    assert hybrid_retrieval(payload=payload,parameters=parameters)['provenance']['context_budget_chars'] == 40


def test_future_corpus_does_not_change_historical_bm25_statistics():
    from research_backends import retrieval_query
    payload = {'snapshot_date':'2024-01-01','environment':{},'model_input_evidence':[{'snippet':'a b'}]}
    passages = [{'evidence_id':f'e{i:02}','document_id':f'd{i}','role':'model_input','snippet':text,
        'publication_time':'2023-01-01','available_at':'2023-01-01','source_version':'v1','content_hash':_content_hash(text)}
        for i,text in enumerate(['a','b'])]
    params = {'passages':passages,'mode':'frozen_scores','bm25':{'k1':0,'b':0},
        'retrieval_depth':1,'reranking_depth':1,'context_budget_chars':40,'rrf_rank_constant':60,
        'frozen_scores':{'query_hash':_content_hash(retrieval_query(payload)),
            'passage_hash':_content_hash(passages),'dense_scores':[0,1], 'cross_encoder_scores':[1,1],
            'model_identity':{'fixture':'recorded'}}}
    before = hybrid_retrieval(payload=payload,parameters=params)['evidence']
    future = [{**passages[0],'evidence_id':f'future-{i}','publication_time':'2025-01-01'} for i in range(10)]
    params['passages'] = passages + future
    params['frozen_scores'].update(passage_hash=_content_hash(params['passages']),
        dense_scores=[0,1]+[0]*10,cross_encoder_scores=[1]*12)
    after = hybrid_retrieval(payload=payload,parameters=params)['evidence']
    assert [p['evidence_id'] for p in before] == [p['evidence_id'] for p in after]


def test_candidate_scorer_rejects_different_revision_before_model_load(tmp_path):
    from research_backends import candidate_probabilities
    with pytest.raises(ValueError,match='binding'):
        candidate_probabilities(payload={'model_id':'m','prompt':'q','endpoint_or_checkpoint':'server/m',
            'revision':'original','message_mapping':'single_user_message'},
            parameters={'model_id':'m','checkpoint':str(tmp_path),'revision':'different',
                        'endpoint_or_checkpoint':'server/m','candidates':{'a':'A'}})


def test_condition_adapter_declares_every_change_and_keeps_population():
    from research_backends import prepare_condition
    original = {'taxonomy_discovery':{'initialization_seed':1},'implementation_choices':{}}
    out = prepare_condition(payload={'configuration':original,'response_ids':['b','a'],'condition':'seed2'},
        parameters={'changes':[{'pointer':'/taxonomy_discovery/initialization_seed','before':1,'after':2}],
                    'rationale':'Explicit independent seed perturbation.'})
    assert out['configuration']['taxonomy_discovery']['initialization_seed'] == 2
    assert original['taxonomy_discovery']['initialization_seed'] == 1
    assert out['response_ids'] == ['a','b']
    assert len(out['changes']) == 2  # scientific value and its implementation-choice provenance
    assert out['configuration']['implementation_choices']['taxonomy_discovery.initialization_seed']['selected_value'] == 2


def test_candidate_scorer_rejects_generation_scoring_weight_mismatch(tmp_path):
    from research_backends import candidate_probabilities
    with pytest.raises(ValueError,match='checkpoint'):
        candidate_probabilities(payload={'model_id':'m','prompt':'q','endpoint_or_checkpoint':'local/m',
            'revision':'v1','message_mapping':'single_user_message','generation_checkpoint_sha256':'a'*64},
            parameters={'model_id':'m','checkpoint':str(tmp_path),'revision':'v1',
                'endpoint_or_checkpoint':'local/m','checkpoint_sha256':'b'*64,'candidates':{'a':'A'}})
