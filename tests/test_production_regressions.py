"""Regressions for production boundaries previously bypassed by fixtures."""
import copy

import pytest

from config import ExperimentConfig
from categorization import Categorizer
from failure_identification import FailureIdentifier
from taxonomy import TaxonomyManager
from test_taxonomy_and_categorization import _component_config, _taxonomy, _resolve


@pytest.mark.parametrize('stage', ['identify', 'discover', 'categorize'])
def test_operational_protocol_binds_real_config_hash_at_runtime(tmp_path, stage):
    taxonomy = _taxonomy()
    payload = _component_config(tmp_path, ['response'], taxonomy).model_dump(mode='json')
    protocol = copy.deepcopy(payload['taxonomy_discovery']['coverage_definition']['annotation_protocols']['initialization'])
    protocol['config_hash'] = '$config_hash'
    if stage == 'identify':
        protocol['judgment'] = 'identification'
        _resolve(payload, 'failure_identification.hard_label_rules', {'annotation_protocols': {'anchor': protocol}})
    elif stage == 'discover':
        settings = payload['taxonomy_discovery']['coverage_definition']
        settings['annotation_protocols'] = {'initialization': protocol, 'refinement': protocol}
        _resolve(payload, 'taxonomy_discovery.coverage_definition', settings)
    else:
        protocol['judgment'] = 'labels'
        protocol['taxonomy_id'] = taxonomy.taxonomy_id
        settings = payload['categorization']['incomplete_agent_output_policy']
        settings['annotation_protocol'] = protocol
        _resolve(payload, 'categorization.incomplete_agent_output_policy', settings)
    config = ExperimentConfig.model_validate(payload)
    before = config.config_hash
    if stage == 'identify':
        result = FailureIdentifier(config, None, None)._annotation_protocol('anchor')
    elif stage == 'discover':
        result = TaxonomyManager(config, None, None, None)._annotation_protocol('initialization')
    else:
        result = Categorizer(config, None, None, None)._annotation_protocol(taxonomy)
    assert result['config_hash'] == before
    assert config.config_hash == before
    assert protocol['config_hash'] == '$config_hash'


def test_explicit_hard_route_for_text_task_does_not_require_unused_stratification(tmp_path):
    from config import _Readiness
    payload = _component_config(tmp_path,['response'],_taxonomy()).model_dump(mode='json')
    payload['operational']['selected_tasks'] = ['threat_report_alignment']
    _resolve(payload,'failure_identification.hard_label_rules',{'tasks':{'threat_report_alignment':{'route':'hard'}}})
    config = ExperimentConfig.model_validate(payload)
    check = _Readiness(config,'identify')
    check._identify()
    assert not any('stratification' in error for error in check.errors)


def test_single_judge_ablation_readiness_keeps_rq3_controls_and_only_selected_agent(tmp_path):
    from config import _Readiness
    payload = _component_config(tmp_path,['response'],_taxonomy()).model_dump(mode='json')
    payload['operational']['selected_ablation'] = 'multi_agent_decision'
    _resolve(payload,'ablations.replacement_behavior_by_component',{
        'multi_agent_decision':{'strategy':'single_judge','judge':'gpt_judge',
            'reconsideration':{'mode':'fresh_reassessment'},'affected_stages':['categorize']}})
    for entry in payload['models']['categorization_panel']:
        if entry['name'] != 'GPT-5':
            entry['endpoint_or_checkpoint'] = None
            entry['revision'] = None
    config = ExperimentConfig.model_validate(payload)
    check = _Readiness(config,'ablations')
    check._ablation()
    assert 'categorization.incomplete_agent_output_policy' in check._required
    assert 'categorization.prompt_templates.round_one' in check._required
    assert 'baselines.gpt_judge.prompt_template_path' not in check._required
    selected_index=next(i for i,entry in enumerate(config.models.categorization_panel) if entry.name=='GPT-5')
    assert f'models.categorization_panel.{selected_index}.endpoint_or_checkpoint' in check._required
    assert not any(path.startswith('models.categorization_panel.') and
                   not path.startswith(f'models.categorization_panel.{selected_index}.')
                   for path in check._required)


from test_taxonomy_and_categorization import harness, _configure_c3, _reconfigure
from schemas import TaskInstance, _content_hash


def test_preservation_can_bind_inputs_and_reference_without_recursive_config_hash(harness, tmp_path):
    _configure_c3(harness,tmp_path)
    instance = TaskInstance.model_validate(harness.store.read_record('task_instance',harness.instance_ids[0]))
    variants = harness.runner().build_counterfactuals(instance)
    operators = copy.deepcopy(harness.config.defenses.counterfactual_consistency.operators)
    for variant in variants:
        decision = operators['preservation_protocol']['decisions'][variant['operator_id']]
        decision['original_content_hash'] = '$original_content_hash'
        decision['original_model_input_hash'] = variant['original_model_input_hash']
        decision['reference_hash'] = _content_hash(instance.reference)
    _reconfigure(harness,{'defenses.counterfactual_consistency.operators':operators})
    reviewed = harness.runner().validate_preservation(instance,variants)
    assert all(v['preservation_status'] == 'admissible' for v in reviewed)
    # The runtime marker never allows the review of a different model input.
    operators['preservation_protocol']['decisions'][variants[0]['operator_id']]['original_model_input_hash'] = 'a'*64
    _reconfigure(harness,{'defenses.counterfactual_consistency.operators':operators})
    with pytest.raises(ValueError,match='source content'):
        harness.runner().validate_preservation(instance,variants)
