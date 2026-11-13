"""Model-seeded RQ2 initialization for an explicitly selected ablation.

This replaces only the initial human proposal. Sampling, subsequent human review
of `other`, reclassification, convergence and final RQ3 are unchanged. Model
support is kept as model provenance and never recorded as human annotation.
"""
from pathlib import Path
from generation_context import review_task_input
from schemas import (SubtypeDefinition, SubtypeRules, TaxonomyVersion, EvidenceSpan,
                     ResponseStatus, _canonical_json, _content_hash)


def initialize(manager):
    from taxonomy import _ProtocolBlocked
    config = manager._config
    population = manager._population
    count = config.taxonomy_discovery.initial_subset_size
    seed = config.taxonomy_discovery.initialization_seed
    if count is None or seed is None or not 0 < count < len(population):
        raise ValueError('Model initialization requires a nonempty proper seeded subset.')
    from sampling import sample_response_ids
    selected, sampling_sources = sample_response_ids(manager._store,population,count=count,seed=seed)
    protocol = manager._annotation_protocol('initialization')
    sample = manager._event('taxonomy_initialization_sample',{
        'response_ids':selected,'seed':seed,'sample_size':count,
        'sampling_method':manager._settings()['sampling_method'],
        'sampling_implementation':'python_random_sample_sorted_root_source_response_ids',
        'source_response_ids':[sampling_sources[identity] for identity in selected],
        'source_response_ids_by_response':sampling_sources,
        'annotation_protocol_hash':_content_hash(protocol),'initialization_source':'model'})
    known = manager._events('taxonomy_model_initialization_complete')
    if known:
        if len(known) != 1:
            raise ValueError('Ambiguous model initialization lineage.')
        return manager._taxonomy(known[0]['taxonomy_id'])
    examples, texts = [], {}
    ambiguous = set()
    for response_id in selected:
        response, instance = manager._response(response_id)
        task_input = review_task_input(manager._store, response, instance)
        examples.append({'response_id':response_id,'task':instance.task_name,
            'task_guidance':task_input['task_guidance'],
            'model_output':response.raw_text,
            'input_evidence':task_input['source_documents'],
            'structured_feeds':task_input['structured_feeds'],
            'task_input':task_input,
            'reference_answer':instance.reference['answer'],
            'reference_evidence':instance.reference.get('evidence_records',[])})
        records = [{'evidence_id':response_id,'snippet':response.raw_text},
                   *examples[-1]['input_evidence'],*examples[-1]['reference_evidence']]
        for item in records:
            for field in ('evidence_id','document_id'):
                identity, text = item.get(field),item.get('snippet')
                if not isinstance(identity,str) or not isinstance(text,str) or identity in ambiguous:
                    continue
                if identity in texts and texts[identity] != text:
                    texts.pop(identity);ambiguous.add(identity)
                else:
                    texts[identity]=text
    payload = {'initialization_examples':examples,'guideline':protocol['guideline_content']}
    path = Path(__file__).with_name('prompts')/'taxonomy_initialization.jinja2'
    prompt = manager._models.render_prompt(str(path),payload)
    contract = {'version':'model-taxonomy-proposal-v1','kind':'taxonomy_proposal',
        'json_schema':{'type':'object','additionalProperties':False,
            'required':['new_subtypes','rules','no_new_categories','evidence_spans'],
            'properties':{'new_subtypes':{'type':'array','items':{'type':'object'}},
                'rules':{'type':'object'},'no_new_categories':{'type':'boolean'},
                'evidence_spans':{'type':'array','items':{'type':'object'}}}}}
    output = manager._models.generate_batch([{
        'model_id':manager._model_id(),'prompt':prompt,'run_index':0,
        'schema_name':'taxonomy_initialization','instance_id':manager._response(selected[0])[1].instance_id,
        'schema_contract':contract,'template_provenance':{
            'template_path':str(path),'template_hash':__import__('hashlib').sha256(path.read_bytes()).hexdigest(),
            'payload_hash':_content_hash(payload),'protocol_id':manager._protocol_id}}])[0]
    if output.status != ResponseStatus.SUCCESS:
        raise _ProtocolBlocked('Model initialization produced no usable category proposal.')
    proposal = output.parsed_answer
    if set(proposal) != {'new_subtypes','rules','no_new_categories','evidence_spans'}:
        raise ValueError('Model initialization requires explicit proposal fields.')
    definitions = [SubtypeDefinition.model_validate(item) for item in proposal['new_subtypes']]
    if not definitions:
        raise _ProtocolBlocked('Model initialization found no categories; convergence is not established.')
    ids = [d.subtype_id for d in definitions]
    if {config.taxonomy_discovery.fallback_label,'uncertain'} & set(ids):
        raise ValueError('Fallback/uncertainty sentinels are not discovered categories.')
    if len(ids) != len(set(ids)) or set(ids) != set(proposal['rules']) or proposal['no_new_categories'] is not False:
        raise ValueError('Model category IDs, rules and no-new-category statement conflict.')
    spans = [EvidenceSpan.model_validate(item) for item in proposal['evidence_spans']]
    for span in spans:
        if not span.quote or span.source_ref not in texts or span.quote not in texts[span.source_ref]:
            raise ValueError('Model category evidence is not a verbatim supplied span.')
        if span.start_offset is not None and texts[span.source_ref][span.start_offset:span.end_offset] != span.quote:
            raise ValueError('Model category evidence offsets disagree with the supplied quote.')
        if span.subtype_id not in ids:
            raise ValueError('Model evidence must identify its proposed category.')
    manager._store.freeze_records('model_response',[output.response_id])
    rules = {key:SubtypeRules.model_validate(value) for key,value in proposal['rules'].items()}
    supported = []
    for definition in definitions:
        evidence = [span.model_dump(mode='json') for span in spans if span.subtype_id == definition.subtype_id]
        if not evidence or definition.supporting_annotation_ids or not definition.evidence_requirements:
            raise ValueError('Model proposal needs evidence and cannot claim human annotations.')
        if not rules[definition.subtype_id].inclusion or not rules[definition.subtype_id].exclusion:
            raise ValueError('Model proposal lacks operational category criteria.')
        raw = definition.model_dump(mode='json')
        raw['provenance'] = {'origin':'model_initialization',
            'model_proposal_response_id':output.response_id,'seed_response_ids':selected,
            'evidence_spans':evidence,'protocol_id':manager._protocol_id}
        supported.append(raw)
    content = {'iteration':0,'parent_id':None,'guideline_hash':protocol['guideline_hash'],
               'subtype_definitions':supported,'inclusion_exclusion_rules':rules,'frozen':False}
    identity = _content_hash({'protocol_id':manager._protocol_id,'content':content})
    taxonomy = TaxonomyVersion(taxonomy_id=identity,**content)
    manager._store.create_record('taxonomy',identity,taxonomy.model_dump(mode='json'))
    manager._event('taxonomy_model_initialization_complete',{
        'taxonomy_id':identity,'model_proposal_response_id':output.response_id,
        'sample_manifest_id':sample['manifest_id'],'supporting_response_ids':selected,
        'human_initialization_performed':False})
    manager._validate_definitions(taxonomy,config.taxonomy_discovery.fallback_label)
    return taxonomy


def has_model_support(manager, definition):
    """A model proposal is eligible only in this arm and this exact protocol."""
    provenance = definition.provenance
    if manager._initialization != 'model' or provenance.get('origin') != 'model_initialization':
        return False
    source = provenance.get('model_proposal_response_id')
    events = [event for event in manager._events('taxonomy_model_initialization_complete')
              if event['model_proposal_response_id'] == source]
    if len(events) != 1 or provenance.get('protocol_id') != manager._protocol_id:
        return False
    manager._immutable('model_response',source)
    from taxonomy import _semantic_definition
    from schemas import ModelResponse
    response = ModelResponse.model_validate(manager._store.read_record('model_response',source))
    return any(_semantic_definition(SubtypeDefinition.model_validate(raw)) == _semantic_definition(definition)
               for raw in response.parsed_answer.get('new_subtypes',[]))
