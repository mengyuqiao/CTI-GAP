"""Seeded sampling over verified source responses, independent of condition IDs.

Only content-addressed condition manifests establish projection lineage. Generic
provider/cache metadata such as usage.source_response_id is never a sampling key.
"""
from __future__ import annotations

import random

from schemas import _content_hash, _identifier


SOURCE_IDENTITY_CONTRACT = 'root_source_response_v1'


def source_response_mapping(store, response_ids):
    """Map current execution IDs to root sources, checking projection integrity.

    Legacy condition projections can be resolved from their recorded manifest
    and original digest formula. New manifests explicitly bind parent and root.
    Repeated projections retain the same roots; unrelated raw/provider response
    ancestry does not merge scientifically distinct evaluation responses.
    """
    bindings, listed_as_derived = {}, set()
    for manifest in store.list_records('experiment_manifest'):
        if manifest.get('purpose') != 'effective_experimental_condition':
            continue
        derived = set(manifest.get('derived_response_ids', []))
        listed_as_derived.update(derived)
        if manifest.get('sampling_identity_contract') == SOURCE_IDENTITY_CONTRACT:
            pairs = manifest.get('sampling_parent_response_ids_by_response', {})
        else:
            pairs = {}
            for source in manifest.get('source_response_ids', []):
                child = _content_hash({
                    'source_response_id': source,
                    'effective_config_hash': manifest.get('effective_config_hash'),
                    'condition': manifest.get('condition'),
                })
                if child in derived:
                    pairs[child] = source
        for child, parent in pairs.items():
            if child in derived:
                bindings.setdefault(child, []).append((parent, manifest))
    resolved, visiting = {}, set()

    def resolve(response_id):
        if response_id in resolved:
            return resolved[response_id]
        if response_id in visiting:
            raise ValueError('Cyclic condition source-response sampling lineage.')
        visiting.add(response_id)
        matches = bindings.get(response_id, [])
        response = store.read_record('model_response', response_id)
        if not matches:
            if response_id in listed_as_derived or 'sampling_source_response_id' in response.get('usage', {}):
                raise ValueError('Condition sampling lineage lacks an authoritative source binding.')
            root = response_id
        else:
            validated = set()
            for parent, manifest in matches:
                if manifest.get('manifest_id') != _content_hash({
                        key: value for key, value in manifest.items() if key != 'manifest_id'}):
                    raise ValueError('Condition sampling lineage manifest has invalid content identity.')
                identity = {
                    'source_response_id': parent,
                    'effective_config_hash': manifest['effective_config_hash'],
                    'condition': manifest['condition'],
                }
                declared_root = None
                if manifest.get('sampling_identity_contract') == SOURCE_IDENTITY_CONTRACT:
                    declared_root = manifest['sampling_source_response_ids_by_response'][response_id]
                    identity.update(sampling_source_response_id=declared_root,
                                    sampling_identity_contract=SOURCE_IDENTITY_CONTRACT)
                if response_id != _content_hash(identity) or response['config_hash'] != manifest['effective_config_hash']:
                    raise ValueError('Condition sampling lineage disagrees with the projected response identity.')
                validated.add((parent, declared_root))
            if len(validated) != 1:
                raise ValueError('Ambiguous condition source-response sampling lineage.')
            parent, declared_root = next(iter(validated))
            source = store.read_record('model_response', parent)
            invariant_fields = ('instance_id', 'model_id', 'run_index', 'prompt_hash',
                                'status', 'raw_text', 'parsed_answer', 'schema_errors')
            if any(response.get(field) != source.get(field) for field in invariant_fields):
                raise ValueError('Condition projection changed the source response scientific payload.')
            root = resolve(parent)
            if declared_root is not None and declared_root != root:
                raise ValueError('Condition sampling lineage declares an incorrect root source response.')
        visiting.remove(response_id)
        resolved[response_id] = root
        return root

    result = {_identifier(response_id): resolve(response_id) for response_id in response_ids}
    if len(set(result.values())) != len(result):
        raise ValueError('Matched sampling population contains duplicate source response identities.')
    return result


def sample_response_ids(store, response_ids, *, count, seed):
    """Sample sorted root IDs, then return current execution IDs in root order."""
    sources = source_response_mapping(store, response_ids)
    by_source = {source: response_id for response_id, source in sources.items()}
    selected_sources = sorted(random.Random(seed).sample(sorted(by_source), count))
    return [by_source[source] for source in selected_sources], sources
