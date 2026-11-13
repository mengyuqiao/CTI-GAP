"""Exact bounded autoregressive output distributions for Appendix H's C3.

This backend enumerates free-text P(y|x) exactly: the outcome space contains *all* token paths ending
at an EOS token or the declared max_new_tokens horizon. A horizon leaf retains
its original probability and is marked truncated. No beam, top-k, sampling,
candidate conditioning, or probability renormalization is performed.

Most real-model trees exceed a practical node budget. Such a tree is BLOCKED;
its partial leaves are never returned as a complete distribution. Positive
mass that cannot be represented by a Python float similarly blocks the result.
Only local CPU checkpoints or an explicitly source-pinned oracle are used.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

from schemas import _content_hash


class _EnumerationBlocked(Exception):
    def __init__(self, reason, **details):
        self.reason = reason
        self.details = details


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}.')
    return value


def _tolerance(value):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 <= value < 1):
        raise ValueError('Probability sum tolerance must be finite and lie in [0, 1).')
    return float(value)


def _eos_ids(values):
    if isinstance(values, (str, bytes)):
        raise ValueError('EOS token IDs must be an iterable of nonnegative integers.')
    try:
        return frozenset(_integer(value, 'EOS token ID') for value in values)
    except TypeError as error:
        raise ValueError('EOS token IDs must be an iterable of integers.') from error


def enumerate_outputs(next_token_probabilities, decode, *, eos_token_ids,
                      max_new_tokens, max_nodes, tolerance=1e-10):
    """Enumerate every positive-probability EOS/horizon path of an exact oracle.

    The oracle receives only generated tokens, as a tuple, and must return a
    normalized mapping over the same complete vocabulary at every queried node.
    Include zero-probability vocabulary entries: they validate support but do
    not create paths. Every declared EOS must belong to that vocabulary. The
    decoder receives a complete token path, including EOS where present.

    Nodes count the root and every positive-probability child, including leaves.
    A blocked result has no outcomes; diagnostics cannot be consumed as a
    distribution. Malformed parameters/probabilities raise ValueError. Duplicate
    decoded strings remain separate paths so token and termination provenance
    are lossless; a downstream text projection can sum their probabilities.
    """
    horizon = _integer(max_new_tokens, 'max_new_tokens')
    budget = _integer(max_nodes, 'max_nodes', minimum=1)
    tolerance = _tolerance(tolerance)
    eos = _eos_ids(eos_token_ids)
    if not callable(next_token_probabilities) or not callable(decode):
        raise ValueError('The next-token oracle and decoder must be callable.')
    provenance = {
        'method':'exact_bounded_autoregressive_enumeration',
        'implementation':'all_EOS_or_max_new_tokens_token_paths',
        'max_new_tokens':horizon, 'max_nodes':budget, 'sum_tolerance':tolerance,
        'eos_token_ids':sorted(eos), 'nodes_visited':1, 'nodes_expanded':0,
        'leaves_completed':0, 'renormalized':False, 'complete':False,
        'probability_arithmetic':'log_path_accumulation_float64',
        'duplicate_decoded_strings':'retained_as_separate_token_paths',
    }
    stack, outcomes, vocabulary = [((), 0.)], [], None
    try:
        while stack:
            prefix, log_probability = stack.pop()
            ended = bool(prefix) and prefix[-1] in eos
            if ended or len(prefix) == horizon:
                probability = math.exp(log_probability)
                if probability == 0.:
                    raise _EnumerationBlocked('probability_underflow',
                        unrepresentable_log_probability=log_probability,
                        unrepresentable_token_ids=list(prefix))
                text = decode(prefix)
                if not isinstance(text, str):
                    raise ValueError('The output decoder must return raw text.')
                outcomes.append({'raw_text':text, 'probability':probability,
                    'log_probability':log_probability, 'token_ids':list(prefix),
                    'truncated':not ended})
                provenance['leaves_completed'] += 1
                continue
            distribution = next_token_probabilities(prefix)
            if not isinstance(distribution, Mapping) or not distribution:
                raise ValueError('Each oracle distribution must be a nonempty full-vocabulary mapping.')
            probabilities = {}
            for token_id, value in distribution.items():
                _integer(token_id, 'vocabulary token ID')
                if (isinstance(value, bool) or not isinstance(value, Real)
                        or not math.isfinite(value) or not 0 <= value <= 1):
                    raise ValueError('Token probabilities must be finite numbers in [0, 1].')
                probabilities[token_id] = float(value)
            support = frozenset(probabilities)
            if not eos <= support:
                raise ValueError('Every EOS token ID must appear in the full vocabulary.')
            if vocabulary is None:
                vocabulary = support
                provenance['vocabulary_size'] = len(vocabulary)
                provenance['vocabulary_hash'] = _content_hash(sorted(vocabulary))
            elif support != vocabulary:
                raise ValueError('The oracle vocabulary support changed between prefixes.')
            total = math.fsum(probabilities.values())
            if abs(total - 1.) > tolerance:
                raise ValueError('A full next-token distribution must sum to one within tolerance.')
            provenance['nodes_expanded'] += 1
            children = sorted(token_id for token_id, value in probabilities.items() if value > 0.)
            if provenance['nodes_visited'] + len(children) > budget:
                raise _EnumerationBlocked('max_nodes_exceeded',
                    next_expansion_children=len(children),
                    minimum_nodes_required=provenance['nodes_visited'] + len(children))
            provenance['nodes_visited'] += len(children)
            for token_id in reversed(children):
                stack.append((prefix + (token_id,),
                              log_probability + math.log(probabilities[token_id])))
        total = math.fsum(row['probability'] for row in outcomes)
        if abs(total - 1.) > tolerance:
            raise _EnumerationBlocked('leaf_probability_sum_outside_tolerance',
                                      observed_total_probability=total)
    except _EnumerationBlocked as error:
        return {'status':'blocked', 'outcomes':[], 'provenance':{
            **provenance, 'reason':error.reason, **error.details}}
    provenance.update(complete=True, total_probability=total)
    return {'status':'complete', 'outcomes':outcomes, 'provenance':provenance}


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be an explicitly declared nonempty string.')
    return value


def _digest(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in '0123456789abcdef' for character in value)):
        raise ValueError(f'{name} must be a lowercase SHA-256 digest.')
    return value


def _binding(payload, parameters):
    for field in ('model_id', 'endpoint_or_checkpoint', 'revision'):
        _nonempty(parameters.get(field), field)
        if payload.get(field) != parameters[field]:
            raise ValueError('Output probability model binding differs from task generation.')
    if payload.get('message_mapping') != 'single_user_message':
        raise ValueError('Output probabilities require the actual generation message mapping.')
    expected = _digest(parameters.get('checkpoint_sha256'), 'checkpoint_sha256')
    if payload.get('generation_checkpoint_sha256') != expected:
        raise ValueError('Generation and probability evaluation must bind the same checkpoint digest.')
    if not isinstance(payload.get('prompt'), str):
        raise ValueError('The exact generation prompt must be supplied.')
    _integer(parameters.get('context_window'), 'context_window', minimum=1)
    _integer(parameters.get('max_new_tokens'), 'max_new_tokens')
    _integer(parameters.get('max_nodes'), 'max_nodes', minimum=1)
    _tolerance(parameters.get('sum_tolerance'))
    return expected


def _check_template(payload, parameters, actual_hash):
    _digest(actual_hash, 'chat_template_hash')
    for declared in (payload.get('generation_chat_template_hash'),
                     payload.get('chat_template_hash'), parameters.get('chat_template_hash')):
        if declared is not None and declared != actual_hash:
            raise ValueError('Probability evaluation chat template differs from generation.')


def _oracle_adapter(specification, payload):
    """Apply the repository's source-pinned module:function adapter contract."""
    if not isinstance(specification, Mapping) or set(specification) != {
            'callable', 'version', 'source_sha256', 'parameters'}:
        raise ValueError('oracle_adapter requires callable, version, source_sha256, and parameters.')
    target = _nonempty(specification['callable'], 'oracle_adapter.callable')
    _nonempty(specification['version'], 'oracle_adapter.version')
    _digest(specification['source_sha256'], 'oracle_adapter.source_sha256')
    if not isinstance(specification['parameters'], Mapping) or target.count(':') != 1:
        raise ValueError('The oracle adapter must use module:function and mapping parameters.')
    module_name, function_name = target.split(':')
    if function_name.startswith('_') or '.' in function_name:
        raise ValueError('The oracle adapter must name a public module-level function.')
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise ValueError('The oracle adapter is not callable.')
    source = inspect.getsourcefile(function)
    if source is None or not Path(source).is_file():
        raise ValueError('The oracle adapter source cannot be verified.')
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != specification['source_sha256']:
        raise ValueError('The oracle adapter source differs from the declared digest.')
    oracle = function(payload=copy.deepcopy(payload),
                      parameters=copy.deepcopy(specification['parameters']))
    if not isinstance(oracle, Mapping):
        raise ValueError('The oracle adapter must return callbacks and model provenance.')
    for field in ('next_token_probabilities', 'decode'):
        if not callable(oracle.get(field)):
            raise ValueError(f'The oracle adapter requires a callable {field}.')
    identity = oracle.get('provenance')
    if not isinstance(identity, Mapping):
        raise ValueError('The oracle adapter requires explicit model provenance.')
    for field in ('model_id', 'endpoint_or_checkpoint', 'revision', 'message_mapping'):
        if identity.get(field) != payload[field]:
            raise ValueError('The oracle model identity differs from generation binding.')
    if identity.get('checkpoint_sha256') != payload['generation_checkpoint_sha256']:
        raise ValueError('The oracle checkpoint differs from generation binding.')
    return oracle


def complete_output_distribution(*, payload, parameters):
    """Production adapter for an exact EOS/max-token output distribution.

    Common parameters: model_id, endpoint_or_checkpoint, revision,
    checkpoint_sha256, context_window, max_new_tokens, max_nodes, sum_tolerance.
    The default local_model mode also requires an existing checkpoint directory.
    CPU float32 model logits receive the full float64 softmax, without generation
    warpers. The exact chat template is covered by the checkpoint digest and its
    hash is recorded; any separately declared template hash must match.

    mode=oracle_adapter instead requires a standard source-pinned oracle_adapter
    declaration {callable, version, source_sha256, parameters}. Its function gets
    keyword payload/parameters and returns next_token_probabilities, decode,
    eos_token_ids, and provenance binding the model fields, checkpoint_sha256,
    message_mapping and chat_template_hash. The caller explicitly vouches for
    that oracle implementation; no hosted provider or paid/GPU call is built in.
    """
    expected = _binding(payload, parameters)
    common = {'model_id':payload['model_id'],
        'endpoint_or_checkpoint':payload['endpoint_or_checkpoint'],
        'revision':payload['revision'], 'checkpoint_sha256':expected,
        'message_mapping':payload['message_mapping'],
        'prompt_hash':_content_hash(payload['prompt']),
        'instance_id':payload.get('instance_id'), 'run_index':payload.get('run_index')}
    mode = parameters.get('mode', 'local_model')
    if mode == 'oracle_adapter':
        oracle = _oracle_adapter(parameters.get('oracle_adapter'), payload)
        _check_template(payload, parameters, oracle['provenance'].get('chat_template_hash'))
        result = enumerate_outputs(oracle['next_token_probabilities'], oracle['decode'],
            eos_token_ids=oracle.get('eos_token_ids'), max_new_tokens=parameters['max_new_tokens'],
            max_nodes=parameters['max_nodes'], tolerance=parameters['sum_tolerance'])
        result['provenance'].update(common, mode=mode,
            chat_template_hash=oracle['provenance']['chat_template_hash'],
            oracle_provenance=copy.deepcopy(dict(oracle['provenance'])),
            oracle_adapter=copy.deepcopy(parameters['oracle_adapter']))
        return result
    if mode != 'local_model':
        raise ValueError('Select local_model or an explicitly pinned oracle_adapter mode.')
    from research_backends import _load_causal_checkpoint, _local_model, checkpoint_digest
    import torch
    checkpoint = _local_model(parameters['checkpoint'])
    # Revalidate even when _load_causal_checkpoint returns a cached model.
    if checkpoint_digest(checkpoint) != expected:
        raise ValueError('Local checkpoint content differs from the declared generation checkpoint.')
    tokenizer, model = _load_causal_checkpoint(checkpoint, parameters['revision'], expected)
    if not tokenizer.chat_template:
        raise ValueError('Exact output probabilities require the generation chat template.')
    template_hash = _content_hash(tokenizer.chat_template)
    _check_template(payload, parameters, template_hash)
    prefix = tokenizer.apply_chat_template([{'role':'user', 'content':payload['prompt']}],
                                          tokenize=True, add_generation_prompt=True)
    if not prefix:
        raise ValueError('The generation chat template must produce prompt tokens.')
    required = len(prefix) + parameters['max_new_tokens']
    model_window = getattr(model.config, 'max_position_embeddings', None)
    if (required > parameters['context_window']
            or isinstance(model_window, int) and required > model_window):
        raise ValueError('Exact output enumeration exceeds the declared/model context window.')
    eos = getattr(model.generation_config, 'eos_token_id', None)
    if eos is None:
        eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError('The local generation model must declare EOS token IDs.')
    eos = [eos] if isinstance(eos, int) else eos

    def next_token_probabilities(generated):
        inputs = torch.tensor([prefix + list(generated)], dtype=torch.long, device='cpu')
        with torch.inference_mode():
            logits = model(input_ids=inputs, use_cache=False).logits[0, -1].double()
            if not bool(torch.isfinite(logits).all()):
                raise ValueError('The causal model returned nonfinite vocabulary logits.')
            log_probabilities = torch.log_softmax(logits, dim=-1).tolist()
        probabilities = {}
        for token_id, log_probability in enumerate(log_probabilities):
            probability = math.exp(log_probability)
            if probability == 0.:
                raise _EnumerationBlocked('probability_underflow',
                    unrepresentable_log_probability=log_probability,
                    unrepresentable_token_ids=list(generated) + [token_id])
            probabilities[token_id] = probability
        return probabilities

    result = enumerate_outputs(next_token_probabilities,
        lambda ids:tokenizer.decode(list(ids), skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False),
        eos_token_ids=eos, max_new_tokens=parameters['max_new_tokens'],
        max_nodes=parameters['max_nodes'], tolerance=parameters['sum_tolerance'])
    result['provenance'].update(common, mode=mode, checkpoint=checkpoint,
        chat_template_hash=template_hash, prompt_token_ids=list(prefix),
        context_window=parameters['context_window'], device='cpu', local_files_only=True,
        model_dtype='float32', softmax_dtype='float64',
        next_token_distribution='unmodified_full_vocabulary_softmax',
        decode_options={'skip_special_tokens':True, 'clean_up_tokenization_spaces':False})
    return result
