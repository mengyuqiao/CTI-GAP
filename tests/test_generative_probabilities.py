"""Exact finite-horizon output probabilities, including a real local causal LM."""
import copy
import hashlib
import importlib
import math

import pytest

from schemas import _content_hash


def _backend():
    assert importlib.util.find_spec('generative_probabilities') is not None, (
        'The exact bounded output-distribution backend has not been implemented.'
    )
    return importlib.import_module('generative_probabilities')


def _enumerate(oracle, **overrides):
    options = dict(eos_token_ids=[0], max_new_tokens=2, max_nodes=20, tolerance=1e-12)
    options.update(overrides)
    return _backend().enumerate_outputs(
        oracle, lambda ids: ''.join({0:'', 1:'A', 2:'B'}[i] for i in ids), **options)


def test_exact_tree_includes_eos_and_horizon_leaves_with_original_mass():
    # Hand calculation: EOS=.2; A-EOS=.8*.25=.2; A-A=.8*.75=.6.
    def oracle(prefix):
        return {0:.2, 1:.8, 2:0.} if not prefix else {0:.25, 1:.75, 2:0.}
    result = _enumerate(oracle)
    assert result['status'] == 'complete'
    outcomes = {tuple(row['token_ids']):row for row in result['outcomes']}
    assert set(outcomes) == {(0,), (1,0), (1,1)}
    assert {key:row['probability'] for key,row in outcomes.items()} == pytest.approx(
        {(0,):.2, (1,0):.2, (1,1):.6})
    assert [outcomes[key]['raw_text'] for key in [(0,), (1,0), (1,1)]] == ['', 'A', 'AA']
    assert not outcomes[(1,0)]['truncated']
    assert outcomes[(1,1)]['truncated']
    assert outcomes[(1,1)]['log_probability'] == pytest.approx(math.log(.6))
    assert result['provenance']['total_probability'] == pytest.approx(1.)
    assert result['provenance']['renormalized'] is False


def test_duplicate_decoded_strings_keep_all_token_paths_and_mass():
    result = _backend().enumerate_outputs(
        lambda prefix:{0:.25, 1:.25, 2:.5}, lambda ids:'same',
        eos_token_ids=[0,2], max_new_tokens=1, max_nodes=4, tolerance=1e-12)
    assert result['status'] == 'complete'
    assert len(result['outcomes']) == 3
    assert sum(row['probability'] for row in result['outcomes']) == pytest.approx(1.)
    assert {tuple(row['token_ids']) for row in result['outcomes']} == {(0,), (1,), (2,)}
    assert [row['truncated'] for row in result['outcomes']] == [False, True, False]


def test_zero_horizon_is_one_empty_truncated_output_without_an_oracle_query():
    def forbidden(prefix):
        raise AssertionError('The zero-token horizon must not invoke the oracle.')
    result = _enumerate(forbidden, max_new_tokens=0, max_nodes=1)
    assert result['status'] == 'complete'
    assert result['outcomes'] == [dict(raw_text='', probability=1., log_probability=0.,
                                      token_ids=[], truncated=True)]


def test_node_budget_blocks_without_publishing_a_partial_distribution():
    calls = []
    def oracle(prefix):
        calls.append(prefix)
        return {0:.5, 1:.5}
    result = _enumerate(oracle, max_nodes=3)
    assert result['status'] == 'blocked'
    assert result['outcomes'] == []
    assert result['provenance']['complete'] is False
    assert result['provenance']['reason'] == 'max_nodes_exceeded'
    assert result['provenance']['nodes_visited'] <= 3
    assert calls == [(), (1,)]


@pytest.mark.parametrize('distribution', [
    {}, {0:.2, 1:.2}, {0:.5, 1:-.5}, {0:float('nan'), 1:1.},
    {0:float('inf'), 1:0.}, {0:True, 1:0.}, {'0':.5, 1:.5},
    {False:.5, 1:.5}, {0:1.01, 1:0.},
])
def test_malformed_node_distributions_are_rejected(distribution):
    with pytest.raises(ValueError):
        _enumerate(lambda prefix:distribution)


def test_missing_eos_and_changing_vocabulary_cannot_claim_full_support():
    with pytest.raises(ValueError, match='EOS'):
        _enumerate(lambda prefix:{1:1.})
    with pytest.raises(ValueError, match='vocabulary'):
        _enumerate(lambda prefix:{0:.5, 1:.5} if not prefix else {0:.5, 2:.5})


@pytest.mark.parametrize('overrides', [
    {'max_new_tokens':-1}, {'max_new_tokens':True}, {'max_nodes':0},
    {'max_nodes':2.5}, {'tolerance':float('nan')}, {'tolerance':1.},
    {'tolerance':-1.}, {'eos_token_ids':[False]}, {'eos_token_ids':[-1]},
])
def test_invalid_bounds_are_rejected(overrides):
    with pytest.raises(ValueError):
        _enumerate(lambda prefix:{0:1.}, **overrides)


def test_tiny_positive_support_is_preserved_without_renormalization():
    result = _enumerate(lambda prefix:{0:1., 1:1e-200}, max_new_tokens=1)
    assert result['status'] == 'complete'
    assert len(result['outcomes']) == 2
    assert result['outcomes'][1]['probability'] == pytest.approx(1e-200, rel=1e-12, abs=0.)


def test_positive_leaf_mass_outside_float_range_blocks_instead_of_disappearing():
    result = _enumerate(lambda prefix:{0:1., 1:1e-200}, max_new_tokens=2)
    assert result['status'] == 'blocked'
    assert result['outcomes'] == []
    assert result['provenance']['reason'] == 'probability_underflow'
    assert result['provenance']['unrepresentable_log_probability'] < -900


def test_tolerated_node_roundoff_is_never_renormalized():
    result = _enumerate(lambda prefix:{0:.5000000000001, 1:.5}, max_new_tokens=1)
    assert result['status'] == 'complete'
    assert result['outcomes'][0]['probability'] == .5000000000001
    assert result['outcomes'][1]['probability'] == .5


def _binding():
    payload = {'model_id':'tiny-fixture', 'endpoint_or_checkpoint':'local/tiny-fixture',
        'revision':'fixture-v1', 'message_mapping':'single_user_message',
        'generation_checkpoint_sha256':'a'*64, 'prompt':'query',
        'instance_id':'example', 'run_index':0}
    parameters = {'model_id':'tiny-fixture', 'endpoint_or_checkpoint':'local/tiny-fixture',
        'revision':'fixture-v1', 'checkpoint_sha256':'a'*64, 'context_window':32,
        'max_new_tokens':2, 'max_nodes':100, 'sum_tolerance':1e-12}
    return payload, parameters


@pytest.fixture
def pinned_oracle(tmp_path, monkeypatch):
    source = tmp_path/'cti_gap_probability_oracle.py'
    source.write_text('''
def make_oracle(*, payload, parameters):
    assert payload['message_mapping'] == 'single_user_message'
    assert parameters == {'fixture': 'exact-full-vocabulary'}
    def next_token_probabilities(prefix):
        return {0: .5, 1: .25, 2: .25}
    def decode(ids):
        return ''.join({0: '', 1: 'A', 2: 'B'}[i] for i in ids)
    return {'next_token_probabilities': next_token_probabilities,
        'decode': decode, 'eos_token_ids': [0], 'provenance': {
            'model_id': payload['model_id'],
            'endpoint_or_checkpoint': payload['endpoint_or_checkpoint'],
            'revision': payload['revision'],
            'checkpoint_sha256': payload['generation_checkpoint_sha256'],
            'message_mapping': payload['message_mapping'],
            'chat_template_hash': 'b'*64}}
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(__import__('sys').modules, 'cti_gap_probability_oracle', raising=False)
    return {'callable':'cti_gap_probability_oracle:make_oracle', 'version':'v1',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'parameters':{'fixture':'exact-full-vocabulary'}}


def test_pinned_oracle_adapter_executes_exact_full_tree(pinned_oracle):
    payload, parameters = _binding()
    parameters.update(mode='oracle_adapter', oracle_adapter=pinned_oracle)
    result = _backend().complete_output_distribution(payload=payload, parameters=parameters)
    assert result['status'] == 'complete'
    assert len(result['outcomes']) == 7
    masses = {row['raw_text']:row['probability'] for row in result['outcomes']}
    assert masses == pytest.approx({'':.5, 'A':.125, 'B':.125,
                                   'AA':.0625, 'AB':.0625, 'BA':.0625, 'BB':.0625})
    provenance = result['provenance']
    assert provenance['checkpoint_sha256'] == 'a'*64
    assert provenance['prompt_hash'] == _content_hash('query')
    assert provenance['chat_template_hash'] == 'b'*64
    assert provenance['oracle_adapter']['source_sha256'] == pinned_oracle['source_sha256']


def test_oracle_source_and_identity_are_bound_before_enumeration(pinned_oracle):
    payload, parameters = _binding()
    parameters.update(mode='oracle_adapter', oracle_adapter=copy.deepcopy(pinned_oracle))
    parameters['oracle_adapter']['source_sha256'] = '0'*64
    with pytest.raises(ValueError, match='source'):
        _backend().complete_output_distribution(payload=payload, parameters=parameters)
    parameters['oracle_adapter'] = pinned_oracle
    payload['generation_chat_template_hash'] = 'c'*64
    with pytest.raises(ValueError, match='template'):
        _backend().complete_output_distribution(payload=payload, parameters=parameters)


@pytest.mark.parametrize('field,value', [
    ('model_id','other'), ('endpoint_or_checkpoint','different'), ('revision','other'),
    ('generation_checkpoint_sha256','c'*64), ('message_mapping','system_and_user'),
])
def test_generation_binding_mismatch_is_rejected_before_model_load(field, value):
    payload, parameters = _binding()
    payload[field] = value
    with pytest.raises(ValueError):
        _backend().complete_output_distribution(payload=payload, parameters=parameters)


@pytest.fixture
def uniform_checkpoint(tmp_path):
    torch = pytest.importorskip('torch')
    transformers = pytest.importorskip('transformers')
    from tokenizers import Tokenizer, models, pre_tokenizers
    from research_backends import checkpoint_digest
    vocab = {'[UNK]':0, '[EOS]':1, 'query':2, 'A':3, 'B':4, '[PAD]':5}
    engine = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    engine.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=engine,
        unk_token='[UNK]', eos_token='[EOS]', pad_token='[PAD]')
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.save_pretrained(tmp_path)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=6, n_positions=32, n_embd=8, n_layer=1, n_head=1,
        bos_token_id=1, eos_token_id=1, pad_token_id=5))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.save_pretrained(tmp_path)
    payload, parameters = _binding()
    digest = checkpoint_digest(tmp_path)
    payload.update(endpoint_or_checkpoint=str(tmp_path), generation_checkpoint_sha256=digest)
    parameters.update(endpoint_or_checkpoint=str(tmp_path), checkpoint=str(tmp_path),
                      checkpoint_sha256=digest, sum_tolerance=1e-10)
    return payload, parameters


def test_real_local_causal_lm_uses_every_softmax_token_without_candidate_conditioning(uniform_checkpoint):
    payload, parameters = uniform_checkpoint
    result = _backend().complete_output_distribution(payload=payload, parameters=parameters)
    assert result['status'] == 'complete'
    # Six first tokens, with five non-EOS prefixes each having six leaves.
    assert len(result['outcomes']) == 31
    first_eos = next(row for row in result['outcomes'] if row['token_ids'] == [1])
    assert first_eos['probability'] == pytest.approx(1/6)
    assert first_eos['raw_text'] == ''
    assert not first_eos['truncated']
    rest = [row for row in result['outcomes'] if row is not first_eos]
    assert all(row['probability'] == pytest.approx(1/36) for row in rest)
    assert sum(row['probability'] for row in result['outcomes']) == pytest.approx(1.)
    assert result['provenance']['device'] == 'cpu'
    assert result['provenance']['local_files_only'] is True
    assert result['provenance']['chat_template_hash'] == _content_hash("{{ messages[0]['content'] }}")


def test_native_tree_blocks_for_budget_and_rejects_context_overflow(uniform_checkpoint):
    payload, parameters = uniform_checkpoint
    parameters['max_nodes'] = 6
    result = _backend().complete_output_distribution(payload=payload, parameters=parameters)
    assert result['status'] == 'blocked'
    assert result['outcomes'] == []
    parameters['context_window'] = 2
    with pytest.raises(ValueError, match='context'):
        _backend().complete_output_distribution(payload=payload, parameters=parameters)


def test_cached_local_model_does_not_bypass_checkpoint_revalidation(uniform_checkpoint):
    payload, parameters = uniform_checkpoint
    parameters['max_new_tokens'] = 1
    assert _backend().complete_output_distribution(payload=payload, parameters=parameters)['status'] == 'complete'
    from pathlib import Path
    (Path(parameters['checkpoint'])/'tampered.txt').write_text('changed checkpoint bytes')
    with pytest.raises(ValueError, match='checkpoint'):
        _backend().complete_output_distribution(payload=payload, parameters=parameters)
