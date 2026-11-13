"""F4: matched seeded cohorts use source identities across condition namespaces."""
from __future__ import annotations

import copy
import hashlib
import random
from datetime import datetime, timezone
from pathlib import Path

import pytest

import model_initialization
import research_backends
from config import ExperimentConfig
from evaluation import Evaluator
from experiments import ExperimentRunner
from failure_identification import FailureIdentifier
from schemas import FailureDecision, StageState, _content_hash
from taxonomy import TaxonomyManager
from test_discovery_repairs import production_harness
from test_metrics_and_identification import _identification_harness, metric_payload
from test_taxonomy_and_categorization import _proposal, _resolve, _span, harness


SOURCE_IDS = [f'r{index}' for index in range(8)]


def config_with(config, changes):
    payload = config.model_dump(mode='json')
    for path, value in changes.items():
        _resolve(payload, path, value)
    result = ExperimentConfig.model_validate(payload)
    result._source_path = config._source_path
    result._reference_digests = copy.deepcopy(config._reference_digests)
    return result


def condition(world, config, response_ids, label, *, seed=None, preserve=False):
    before = config.taxonomy_discovery.initialization_seed
    # An unchanged seed explicitly freezes membership while the ablation changes
    # initialization source outside configuration (human versus model).
    protocol = {'preparation_adapter': {
        'callable': 'research_backends:prepare_condition', 'version': 'audit-f4-v1',
        'source_sha256': hashlib.sha256(Path(research_backends.__file__).read_bytes()).hexdigest(),
        'parameters': {
            'changes': [{'pointer': '/taxonomy_discovery/initialization_seed',
                         'before': before, 'after': before if seed is None else seed}],
            'rationale': 'Offline matched sampling condition; fixed or explicitly perturbed seed.',
        },
    }}
    runner = ExperimentRunner(config, world.store, world.models,
                              Evaluator(config, world.store), world.annotations)
    return runner._condition(protocol, response_ids, label, preserve_identification=preserve)


def source_members(store, response_ids, *, ordered=False):
    """Independent fixture lookup: roots are explicitly named r0 through r7."""
    result = []
    for response_id in response_ids:
        while response_id not in SOURCE_IDS:
            response_id = store.read_record('model_response', response_id)['usage']['source_response_id']
        result.append(response_id)
    return result if ordered else sorted(result)


def prepare_discovery(world):
    world.config = config_with(world.config, {
        'taxonomy_discovery.initial_subset_size': 3,
        'taxonomy_discovery.initialization_seed': 7,
    })
    world.models.config = world.config
    raw = world.store.read_record('model_response', world.failure_ids[0])
    for index, response_id in enumerate(SOURCE_IDS):
        response = copy.deepcopy(raw)
        response.update(response_id=response_id, config_hash=world.config.config_hash,
                        usage={'source_response_id': f'provider-decoy-{7-index}'})
        world.store.create_record('model_response', response_id, response)
        decision = FailureDecision(response_id=response_id, identification_version='audit-f4-source-v1',
                                   decision='failure', decision_origin='hard_label')
        world.store.create_record('failure_decision', _content_hash(decision), decision.model_dump(mode='json'))


def model_proposal(request, packet):
    return {**_proposal(('fixture-A',)), 'evidence_spans': [_span('fixture-A')]}


def test_exact_independent_audit_counterexample():
    assert sorted(random.Random(7).sample(SOURCE_IDS, 3)) == ['r1', 'r3', 'r5']
    projected = {_content_hash({'source_response_id': response_id,
                               'effective_config_hash': 'a'*64,
                               'condition': 'ablation:human_initiation:0'}): response_id
                 for response_id in SOURCE_IDS}
    assert sorted(projected[item] for item in random.Random(7).sample(sorted(projected), 3)) == [
        'r2', 'r5', 'r7']


def legacy_projection(world, *, broken=None):
    """Persist the old controller's projection contract for backward lineage."""
    derived = []
    for response_id in SOURCE_IDS:
        raw = world.store.read_record('model_response', response_id)
        raw['config_hash'] = 'a'*64
        raw['usage']['source_response_id'] = response_id
        raw['response_id'] = _content_hash({
            'source_response_id': response_id, 'effective_config_hash': 'a'*64,
            'condition': 'ablation:human_initiation:0'})
        if broken == 'scientific_payload':
            raw['raw_text'] = 'changed output must not be grouped with its source'
        world.store.create_record('model_response', raw['response_id'], raw)
        derived.append(raw['response_id'])
    content = {
        'condition': 'ablation:human_initiation:0', 'effective_config_hash': 'a'*64,
        'source_response_ids': SOURCE_IDS if broken != 'missing_binding' else ['unrelated-source'],
        'derived_response_ids': sorted(derived), 'changes': [], 'adapter_provenance': {},
    }
    world.runner()._manifest('effective_experimental_condition', content)
    return derived


@pytest.mark.parametrize('production_harness', [True], indirect=True)
def test_legacy_condition_manifest_recovers_exact_audit_sources(production_harness):
    from sampling import sample_response_ids
    world = production_harness
    prepare_discovery(world)
    projected = legacy_projection(world)
    selected, sources = sample_response_ids(world.store, projected, count=3, seed=7)
    assert sorted(sources[item] for item in selected) == ['r1', 'r3', 'r5']


@pytest.mark.parametrize('production_harness', [True], indirect=True)
@pytest.mark.parametrize('broken', ['scientific_payload', 'missing_binding'])
def test_condition_lineage_never_silently_accepts_changed_outputs_or_missing_bindings(
    production_harness, broken,
):
    from sampling import source_response_mapping
    world = production_harness
    prepare_discovery(world)
    projected = legacy_projection(world, broken=broken)
    with pytest.raises(ValueError, match='[Cc]ondition.*(scientific payload|source binding)'):
        source_response_mapping(world.store, projected)


@pytest.mark.parametrize('production_harness', [True], indirect=True)
def test_consistent_repeated_condition_manifests_do_not_make_source_identity_ambiguous(production_harness):
    from sampling import source_response_mapping
    world = production_harness
    prepare_discovery(world)
    _, projected, _ = condition(world, world.config, SOURCE_IDS, 'condition:retry')
    _, repeated, _ = condition(world, world.config, SOURCE_IDS[:4], 'condition:retry')
    assert set(repeated).issubset(projected)
    assert set(source_response_mapping(world.store, projected).values()) == set(SOURCE_IDS)


def test_old_completed_identification_cannot_bypass_the_new_sampling_contract(tmp_path, metric_payload):
    world = _identification_harness(tmp_path, metric_payload)
    ids = [world.similarity(50.).response_id for _ in range(3)]
    config, store = world.metrics.config, world.metrics.store
    # Construct an actual saved completion under the pre-repair protocol hash.
    legacy_version = _content_hash({
        'method': 'rq1_human_calibrated_identification',
        'implementation_contract': 'failure-identification-v1',
        'config_hash': config.config_hash, 'corpus_version': config.corpus_version,
        'failure_identification': config.failure_identification,
        'response_criteria': config.evaluation.scoring.response_level_failure_criteria,
        'guideline_manifest': config.taxonomy.guideline_manifest,
    })
    members = []
    for response_id in sorted(ids):
        response, instance, scores = world.identifier._join(response_id)
        members.append({
            'response_id': response_id, 'response_hash': _content_hash(response),
            'instance_id': instance.instance_id, 'instance_hash': instance.content_hash,
            'reference_hash': _content_hash(instance.reference),
            'score_ids': sorted(item['score_id'] for item in scores), 'route': 'similarity',
        })
    population_id = _content_hash({
        'purpose': 'identification_population', 'identification_version': legacy_version,
        'config_hash': config.config_hash, 'corpus_version': config.corpus_version, 'members': members,
    })
    execution_id = _content_hash({'version': legacy_version, 'population': population_id})
    store.create_record('experiment_manifest', 'legacy-completed-result', {
        'manifest_id': 'legacy-completed-result', 'purpose': 'identification_final',
        'result': {'status': 'completed', 'legacy_sampling_result': True},
    })
    now = datetime.now(timezone.utc)
    store.set_stage_state('identify', StageState(
        stage='identify', execution_id=execution_id, config_hash=config.config_hash,
        status='completed', completed_work=ids,
        input_manifests={'population': population_id},
        output_manifests={'identification': 'legacy-completed-result'},
        started_at=now, updated_at=now, completed_at=now,
    ).model_dump(mode='json'))
    outcome = world.identifier.run(ids)
    assert outcome['status'] == 'waiting_for_annotations'
    assert 'legacy_sampling_result' not in outcome


@pytest.mark.parametrize('production_harness', [True], indirect=True)
def test_human_and_model_initialization_keep_the_same_source_subset_after_condition_projection(
    production_harness,
):
    world = production_harness
    prepare_discovery(world)
    human = TaxonomyManager(world.config, world.store, world.models, world.annotations)
    human.request_initialization(SOURCE_IDS)
    assert human._initial_sample()['response_ids'] == ['r1', 'r3', 'r5']
    effective, projected, manifest = condition(
        world, world.config, SOURCE_IDS, 'ablation:human_initiation:0', preserve=True)
    world.models.config = effective
    world.models.script = model_proposal
    model = TaxonomyManager(effective, world.store, world.models, world.annotations, initialization='model')
    model._establish_population(projected)
    model_initialization.initialize(model)
    assert source_members(world.store, model._initial_sample()['response_ids']) == ['r1', 'r3', 'r5']
    examples = world.models.packets[next(iter(world.models.packets))]['initialization_examples']
    assert source_members(world.store, [item['response_id'] for item in examples], ordered=True) == ['r1', 'r3', 'r5']
    assert set(projected).isdisjoint(SOURCE_IDS)
    assert model._initial_sample()['source_response_ids'] == ['r1', 'r3', 'r5']
    assert set(manifest['sampling_source_response_ids_by_response'].values()) == set(SOURCE_IDS)


@pytest.mark.parametrize('production_harness', [True], indirect=True)
@pytest.mark.parametrize(('seed', 'expected'), [(7, ['r1', 'r3', 'r5']), (11, ['r4', 'r6', 'r7'])])
def test_repeated_real_condition_projection_preserves_roots_and_applies_declared_seed(
    production_harness, seed, expected,
):
    world = production_harness
    prepare_discovery(world)
    first_config, first_ids, _ = condition(world, world.config, SOURCE_IDS, 'condition:first', preserve=True)
    second_config, second_ids, manifest = condition(
        world, first_config, first_ids, 'condition:second', seed=seed, preserve=True)
    manager = TaxonomyManager(second_config, world.store, world.models, world.annotations)
    manager.request_initialization(second_ids)
    assert source_members(world.store, manager._initial_sample()['response_ids']) == expected
    assert manager._initial_sample()['source_response_ids'] == expected
    assert set(first_ids).isdisjoint(second_ids)
    assert set(manifest['sampling_source_response_ids_by_response'].values()) == set(SOURCE_IDS)


@pytest.mark.parametrize(('width', 'seed', 'expected'), [
    (1., 7, [['r1', 'r3', 'r5']]),
    (1., 11, [['r4', 'r6', 'r7']]),
    (.5, 7, [['r0', 'r1', 'r2'], ['r4', 'r5', 'r6']]),
])
def test_tied_strata_and_anchor_membership_are_matched_by_source_with_width_and_seed_controls(
    tmp_path, metric_payload, width, seed, expected,
):
    world = _identification_harness(tmp_path, metric_payload, tie_mode='split_by_response_id')
    config = config_with(world.metrics.config, {
        'failure_identification.stratification.quantile_width': width,
        'failure_identification.stratification.anchor_count_per_stratum': 3,
        'failure_identification.stratification.anchor_sampling_seed': seed,
    })
    world.metrics.config = config
    # The real condition helper only needs these two world interfaces.
    world.store = world.metrics.store
    world.models = None
    for index, response_id in enumerate(SOURCE_IDS):
        original = world.similarity(50.)
        raw = world.store.read_record('model_response', original.response_id)
        raw.update(response_id=response_id, usage={'source_response_id': f'provider-decoy-{7-index}'})
        world.store.create_record('model_response', response_id, raw)
        score = world.store.list_records('score', {'response_id': original.response_id})[0]
        score.update(score_id='score-'+response_id, response_id=response_id)
        world.store.create_record('score', score['score_id'], score)
    effective, projected, _ = condition(world, config, SOURCE_IDS, 'stability:anchor_selection:0')
    for response_id in projected:
        source_id = world.store.read_record('model_response', response_id)['usage']['source_response_id']
        score = world.store.list_records('score', {'response_id': source_id})[0]
        score.update(score_id='score-'+response_id, response_id=response_id, config_hash=effective.config_hash)
        world.store.create_record('score', score['score_id'], score)
    identifier = FailureIdentifier(effective, world.store, world.annotations)
    strata = identifier.build_strata(list(reversed(projected)))
    assert [source_members(world.store, [item['response_id'] for item in row['members']], ordered=True)
            for row in strata] == ([SOURCE_IDS] if width == 1. else [SOURCE_IDS[:4], SOURCE_IDS[4:]])
    selected = []
    for row in strata:
        tasks = identifier.request_anchors(row['stratum_id'])
        selected.append(source_members(world.store, [task.response_id for task in tasks]))
    assert selected == expected
