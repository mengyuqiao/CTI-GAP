"""Method fidelity gates; no empirical result values are acceptance targets."""
import copy
from pathlib import Path
import pytest

from config import ExperimentConfig, _Readiness
from model_client import ModelClient
from test_discovery_repairs import production_harness
from test_taxonomy_and_categorization import _resolve, harness


def test_complete_autoregressive_backend_is_a_valid_model_interface(production_harness):
    world = production_harness
    payload = world.config.model_dump(mode='json')
    _resolve(payload,'operational.selected_defense','counterfactual_consistency')
    extraction = {'interface':'autoregressive_complete','aggregation_space':'task_outputs',
        'label_pointer':'/answer/value','sum_tolerance':1e-10,'generation_checkpoint_sha256':'a'*64,
        'adapter':{'callable':'generative_probabilities:complete_output_distribution',
                   'version':'v1','source_sha256':'a'*64,'parameters':{
                       'max_new_tokens':payload['inference']['max_output_tokens'],'max_nodes':100}}}
    _resolve(payload,'defenses.counterfactual_consistency.probability_extraction',extraction)
    config = ExperimentConfig.model_validate(payload)
    ModelClient(config,world.store).validate_endpoints(['G5'])


def test_generative_readiness_requires_exact_field_and_output_horizon(production_harness):
    payload = production_harness.config.model_dump(mode='json')
    _resolve(payload,'defenses.counterfactual_consistency.probability_extraction',{
        'interface':'autoregressive_complete','aggregation_space':'task_outputs','label_pointer':'/answer/value',
        'sum_tolerance':1e-10,'adapter':{'callable':'generative_probabilities:complete_output_distribution',
        'version':'v1','source_sha256':'a'*64,'parameters':{'max_new_tokens':999,'max_nodes':100}}})
    config = ExperimentConfig.model_validate(payload)
    checker = _Readiness(config,'defenses')
    checker._c3()
    assert any('horizon' in error for error in checker.errors)


def test_c3_readiness_does_not_require_unused_human_failure_judgments(production_harness):
    checker = _Readiness(production_harness.config,'defenses')
    checker._c3()
    assert 'evaluation.scoring.response_level_failure_criteria' not in checker._required
