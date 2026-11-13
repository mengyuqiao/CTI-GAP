"""Local CPU backends for the explicit ExperimentRunner adapter contract.

Models/checkpoints must already exist locally. Nothing downloads or calls hosted
services. Frozen scores are exact replay artifacts, never synthetic replacements
for missing encoders. Numerical choices (BM25, cosine, RRF and character budget)
are implementation choices and must be recorded in the experiment configuration.
"""
from __future__ import annotations

import copy
import hashlib
import math
import re
from collections import Counter
from datetime import date
from pathlib import Path
from functools import lru_cache

import numpy as np
from schemas import _canonical_json, _content_hash


def _number(value, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('Expected a finite numeric backend parameter.')
    if minimum is not None and value < minimum:
        raise ValueError('Backend parameter below allowed minimum.')
    return float(value)


def _positive_int(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError('Expected a positive integer backend parameter.')
    return value


def _nonnegative_int(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('Expected a nonnegative integer backend parameter.')
    return value


def _tokens(text):
    return re.findall(r'\w+', text.lower())


def bm25_scores(query, documents, *, k1, b):
    """Robertson BM25 with positive log IDF; repeated query terms count once."""
    k1 = _number(k1, minimum=0)
    b = _number(b, minimum=0)
    if b > 1:
        raise ValueError('BM25 b must lie in [0,1].')
    counts = [Counter(_tokens(text)) for text in documents]
    lengths = [sum(count.values()) for count in counts]
    average = sum(lengths) / len(counts) if counts else 0
    scores = np.zeros(len(counts))
    if not average:
        return scores.tolist()
    for term in set(_tokens(query)):
        df = sum(term in count for count in counts)
        idf = math.log1p((len(counts)-df+.5)/(df+.5))
        for index, count in enumerate(counts):
            frequency = count[term]
            if frequency:
                scores[index] += idf * frequency*(k1+1)/(frequency+k1*(1-b+b*lengths[index]/average))
    return scores.tolist()


def fuse_rankings(rankings, *, rank_constant):
    constant = _number(rank_constant, minimum=0)
    result = {}
    for ranking in rankings:
        if len(set(ranking)) != len(ranking):
            raise ValueError('A ranking contains duplicate candidates.')
        for rank, index in enumerate(ranking, 1):
            result[index] = result.get(index, 0.) + 1/(constant+rank)
    return result


def normalize_log_likelihoods(log_likelihoods):
    """Exact stable conditioning on a declared finite candidate response space."""
    if not log_likelihoods:
        raise ValueError('Candidate space is empty.')
    values = {key: _number(value) for key, value in log_likelihoods.items()}
    maximum = max(values.values())
    mass = {key: math.exp(value-maximum) for key,value in values.items()}
    total = sum(mass.values())
    return {key:value/total for key,value in mass.items()}


def retrieval_query(payload):
    """Input-only query. Evaluation references and automated labels are excluded."""
    texts = [item['snippet'] for item in payload['model_input_evidence']]
    return '\n'.join(texts + [_canonical_json(payload.get('environment', {}))])


def _local_model(path):
    local = Path(path).expanduser().resolve()
    if not local.exists():
        raise FileNotFoundError(f'Local model/checkpoint unavailable: {local}')
    return str(local)


def _day(value):
    return date.fromisoformat(value[:10])


def _rank_hybrid(*, payload, parameters):
    """Rank historically admitted evidence before any final context selection."""
    declared = payload.get('retrieval_configuration', {})
    for field in ('retrieval_depth', 'reranking_depth'):
        if field in declared and declared[field] != parameters[field]:
            raise ValueError('Forwarded retrieval controls disagree with backend parameters.')
    budget = declared.get('evidence_context_budget')
    if budget is not None:
        expected = {'unit':'characters', 'maximum':parameters['context_budget_chars']}
        if budget != expected and budget != parameters['context_budget_chars']:
            raise ValueError('Forwarded context budget disagrees with backend parameters.')
    passages = copy.deepcopy(parameters['passages'])
    identities = [item['evidence_id'] for item in passages]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate retrieval evidence IDs.')
    query = retrieval_query(payload)
    snapshot = _day(payload['snapshot_date'])
    eligible = []
    for i, item in enumerate(passages):
        if item.get('role') != 'model_input':
            raise ValueError('Withheld references cannot enter the retrieval corpus.')
        if not item.get('content_hash') or not item.get('source_version'):
            raise ValueError('Retrieval requires immutable source/version provenance.')
        if _day(item['publication_time']) > snapshot or _day(item['available_at']) > snapshot:
            continue
        if item.get('modification_time') and _day(item['modification_time']) > snapshot:
            continue
        eligible.append(i)
    documents = [p['snippet'] for p in passages]
    admitted_sparse = bm25_scores(query, [documents[i] for i in eligible], **parameters['bm25'])
    sparse = [0.] * len(passages)
    for i, score in zip(eligible, admitted_sparse, strict=True):
        sparse[i] = score
    mode = parameters['mode']
    if mode == 'frozen_scores':
        frozen = parameters['frozen_scores']
        if frozen['query_hash'] != _content_hash(query) or frozen['passage_hash'] != _content_hash(passages):
            raise ValueError('Recorded retrieval scores belong to a different query/corpus.')
        dense, cross = frozen['dense_scores'], frozen['cross_encoder_scores']
        if not frozen['model_identity']:
            raise ValueError('Recorded encoder scores need model identities.')
        provenance = {'recorded_scores_hash':_content_hash(frozen), 'models':frozen['model_identity']}
    elif mode == 'local_models':
        from sentence_transformers import SentenceTransformer, CrossEncoder
        dense_path = _local_model(parameters['dense_checkpoint'])
        cross_path = _local_model(parameters['cross_encoder_checkpoint'])
        encoder = SentenceTransformer(dense_path, device='cpu', local_files_only=True)
        vectors = encoder.encode([query, *documents], normalize_embeddings=True)
        dense = np.asarray(vectors[1:]) @ np.asarray(vectors[0])
        reranker = CrossEncoder(cross_path, device='cpu', local_files_only=True)
        cross = None
        provenance = {'models':{'dense':dense_path,'cross_encoder':cross_path},
                      'revisions':parameters['model_revisions']}
    else:
        raise ValueError('Select local_models or exact frozen_scores replay.')
    if len(dense) != len(passages):
        raise ValueError('Dense score population mismatch.')
    dense = [_number(value) for value in np.asarray(dense).tolist()]
    depth = _positive_int(parameters['retrieval_depth'])
    rank = lambda scores: sorted(eligible, key=lambda i:(-scores[i],identities[i]))[:depth]
    fused = fuse_rankings([rank(sparse),rank(dense)], rank_constant=parameters['rrf_rank_constant'])
    candidate_ids = sorted(fused, key=lambda i:(-fused[i],identities[i]))[:_positive_int(parameters['reranking_depth'])]
    if cross is None:
        cross_values = reranker.predict([(query,documents[i]) for i in candidate_ids]) if candidate_ids else []
        reranked_scores = {i:_number(float(v)) for i,v in zip(candidate_ids,cross_values,strict=True)}
    else:
        if len(cross) != len(passages):
            raise ValueError('Cross-encoder score population mismatch.')
        reranked_scores = {i:_number(cross[i]) for i in candidate_ids}
    ordered = sorted(candidate_ids,key=lambda i:(-reranked_scores[i],identities[i]))
    return passages, ordered, {
        **provenance, 'query_hash':_content_hash(query), 'corpus_hash':_content_hash(passages),
        'sparse_scores':sparse, 'dense_scores':dense, 'fused_scores':{identities[i]:v for i,v in fused.items()},
        'cross_encoder_scores':{identities[i]:v for i,v in reranked_scores.items()},
        'admitted_evidence_ids':[identities[i] for i in eligible]}


def hybrid_retrieval(*, payload, parameters):
    """BM25 + cosine dense retrieval, RRF fusion, cross-encoder reranking.

    parameters: passages, bm25{k1,b}, retrieval_depth, reranking_depth,
    context_budget_chars, rrf_rank_constant, mode. local_models additionally needs
    dense_checkpoint and cross_encoder_checkpoint directories; frozen_scores
    needs query_hash, passage_hash, dense_scores, cross_encoder_scores and
    model_identity. Replay scores cover ALL supplied passages in original order.
    """
    passages, ordered, provenance = _rank_hybrid(payload=payload, parameters=parameters)
    budget = _positive_int(parameters['context_budget_chars'])
    evidence, used = [], 0
    for index in ordered:
        item = passages[index]
        # Preserve complete evidence spans. Never truncate to change their meaning.
        if used + len(item['snippet']) <= budget:
            item['snapshot_admission_verified'] = True
            evidence.append(item)
            used += len(item['snippet'])
    return {'evidence':evidence, 'provenance':{
        **provenance, 'context_characters':used, 'context_budget_chars':budget}}


def inference_constraints(*, payload, parameters):
    """Apply universally declared subtype policies without gold-label routing."""
    policies = payload['subtype_policy_manifest']['policies']
    rules = [policies[key]['inference'] for key in sorted(policies)]
    if any(not isinstance(rule,str) or not rule.strip() for rule in rules):
        raise ValueError('Every subtype inference constraint needs explicit text.')
    return {'task_guidance':'\n'.join([payload['original_guidance'],*rules])}


def candidate_probabilities(*, payload, parameters):
    """Teacher-force complete continuations through the supplied local causal LM.

    Candidates are explicit distinct strings under parameters.candidates. Scores
    sum token log probabilities INCLUDING EOS, without length normalization.
    Returned probabilities condition on that finite response space. This is a
    candidate-conditioned P(y|x), not an unrestricted free-text density.
    Model identity must match the experimental model; no substitute model.
    """
    if any(payload.get(key) != parameters.get(key) for key in ('endpoint_or_checkpoint','revision')):
        raise ValueError('Candidate scoring model binding differs from task generation.')
    if payload.get('message_mapping') != 'single_user_message':
        raise ValueError('Candidate scorer requires the actual generation message binding.')
    if payload['model_id'] != parameters['model_id']:
        raise ValueError('Candidate scorer must use the evaluated model identity.')
    checkpoint = _local_model(parameters['checkpoint'])
    candidates = parameters['candidates']
    if not candidates or len(set(candidates.values())) != len(candidates):
        raise ValueError('Provide distinct, nonempty finite candidate completions.')
    if any(not isinstance(text,str) or not text for text in candidates.values()):
        raise ValueError('Candidate continuations must be nonempty strings.')
    expected = parameters.get('checkpoint_sha256')
    if not expected or expected != payload.get('generation_checkpoint_sha256'):
        raise ValueError('Generation and scoring must bind the same checkpoint digest.')
    import torch
    tokenizer, model = _load_causal_checkpoint(checkpoint, parameters['revision'], expected)
    if not tokenizer.chat_template:
        raise ValueError('Candidate scoring requires the generation model chat template.')
    prefix = tokenizer.apply_chat_template([{'role':'user','content':payload['prompt']}],
                                            tokenize=True, add_generation_prompt=True)
    if not prefix or tokenizer.eos_token_id is None:
        raise ValueError('Candidate scoring requires prompt tokens and an EOS token.')
    scores, token_sequences = {}, {}
    with torch.inference_mode():
        for label, text in candidates.items():
            continuation = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
            sequence = prefix + continuation
            if len(sequence) > parameters['context_window']:
                raise ValueError('Candidate scoring exceeds the declared context window.')
            inputs = torch.tensor([sequence], dtype=torch.long)
            logp = torch.log_softmax(model(input_ids=inputs).logits[0].float(),dim=-1)
            positions = torch.arange(len(prefix)-1,len(sequence)-1)
            scores[label] = float(logp[positions,torch.tensor(continuation)].sum())
            token_sequences[label] = continuation
    if len({tuple(v) for v in token_sequences.values()}) != len(candidates):
        raise ValueError('Distinct candidates tokenize to identical outcomes.')
    return {'probabilities':normalize_log_likelihoods(scores),'provenance':{
        'method':'teacher_forced_finite_candidate_likelihood', 'model_id':payload['model_id'],
        'checkpoint':checkpoint, 'revision':parameters['revision'], 'checkpoint_sha256':expected,
        'prompt_hash':_content_hash(payload['prompt']), 'candidates_hash':_content_hash(candidates),
        'log_likelihoods':scores, 'candidate_token_ids':token_sequences,
        'includes_eos':True, 'length_normalization':False, 'device':'cpu',
        'endpoint_or_checkpoint':payload['endpoint_or_checkpoint'],
        'message_mapping':payload['message_mapping'],
        'chat_template_hash':_content_hash(tokenizer.chat_template)}}


def alignscore_verification(*, payload, parameters):
    """Actual AlignScore claim–evidence alignment; fixed JSON-pointer claims.

    Claims are serialized JSON field values located in the candidate's raw JSON.
    Parameters require explicit claim_pointers, checkpoint, backbone, batch_size,
    evaluation_mode and alignment_threshold. No evaluation reference is read.
    """
    from alignscore import AlignScore
    checkpoint = _local_model(parameters['checkpoint'])
    backbone = _local_model(parameters['backbone'])
    candidate = payload.get('candidate_response')
    if candidate is None:
        import json
        candidate = {'response_id':payload['candidate_response_id'],
                     'raw_text':payload['candidate_raw_text'],
                     'parsed_answer':json.loads(payload['candidate_raw_text'])}
    evidence = payload['retrieved_evidence']
    spans = _json_value_spans(candidate['raw_text'])
    claims, quotes, pointers = [], [], []
    rules = parameters.get('claim_rules')
    if rules is None:
        # Text-only legacy pointers have semantic content by themselves.
        import json
        rules = [{'pointer':pointer, 'mode':'text'} for pointer in parameters['claim_pointers']]
    for rule in rules:
        pointer = rule['pointer']
        if pointer not in spans:
            raise ValueError('Claim pointer is missing from retained candidate JSON.')
        import json
        start, end = spans[pointer]
        value = json.loads(candidate['raw_text'][start:end])
        if rule['mode'] == 'aligned_candidates':
            candidates = payload['task_input']['candidates']
            if not isinstance(value,list) or len(value) != len(candidates):
                raise ValueError('Task claims must align with the complete candidate list.')
            for i,(entity,decision) in enumerate(zip(candidates,value,strict=True)):
                if type(decision) is not int or decision not in (0,1):
                    raise ValueError('Aligned task claims require binary decisions.')
                token = pointer+'/'+str(i)
                begin, finish = spans[token]
                quotes.append(candidate['raw_text'][begin:finish])
                pointers.append(token)
                claims.append(rule['template'].format(candidate=entity,
                    decision=rule['positive'] if decision else rule['negative']))
        elif rule['mode'] in {'text','scalar'}:
            if not isinstance(value,str):
                raise ValueError('Nontext claims require task-specific semantic segmentation.')
            quotes.append(candidate['raw_text'][start:end])
            pointers.append(pointer)
            claims.append(value if rule['mode'] == 'text' else rule['template'].format(value=value))
        else:
            raise ValueError('Unsupported explicit claim segmentation mode.')
    if not claims or not evidence:
        raise ValueError('Verification requires declared claims and retrieved evidence.')
    scorer = AlignScore(model=backbone,batch_size=_positive_int(parameters['batch_size']),
                        device='cpu',ckpt_path=checkpoint,evaluation_mode=parameters['evaluation_mode'])
    context = '\n'.join(item['snippet'] for item in evidence)
    scores = scorer.score(contexts=[context]*len(claims),claims=claims)
    feedback = {'candidate_response_id':candidate['response_id'],
                'evidence_package_hash':_content_hash(evidence), 'method':'AlignScore',
                'checkpoint':checkpoint, 'claim_segmentation':{'rules':rules},
                'alignment_threshold':_number(parameters['alignment_threshold']),
                'claims':[{'claim_id':f'claim-{i}','candidate_quote':quotes[i], 'candidate_pointer':pointers[i],
                           'claim_text':quote, 'feedback_text':'Checked assertion: '+quote,
                           'evidence_ids':[item['evidence_id'] for item in evidence],
                           'alignment_score':_number(float(score))}
                          for i,(quote,score) in enumerate(zip(claims,scores,strict=True))]}
    return {'artifact_id':_content_hash(feedback),**feedback}


def _json_value_spans(text):
    """Locate exact JSON value spans using the standard decoder, not reserialization."""
    import json
    decoder = json.JSONDecoder()
    spans = {}
    def whitespace(index):
        while index < len(text) and text[index].isspace():
            index += 1
        return index
    def visit(index, pointer):
        index = whitespace(index)
        value, end = decoder.raw_decode(text,index)
        spans[pointer] = (index,end)
        if isinstance(value,dict):
            cursor = whitespace(index+1)
            for _ in value:
                key, cursor = decoder.raw_decode(text,cursor)
                cursor = whitespace(cursor)
                if text[cursor] != ':':
                    raise ValueError('Invalid JSON claim source.')
                token = key.replace('~','~0').replace('/','~1')
                cursor = whitespace(visit(cursor+1,pointer+'/'+token))
                if text[cursor] == ',':
                    cursor = whitespace(cursor+1)
        elif isinstance(value,list):
            cursor = whitespace(index+1)
            for i in range(len(value)):
                cursor = whitespace(visit(cursor,pointer+'/'+str(i)))
                if text[cursor] == ',':
                    cursor = whitespace(cursor+1)
        return end
    end = visit(0,'')
    if text[end:].strip():
        raise ValueError('Trailing content in candidate JSON.')
    return spans


def _duplicate_clusters(passages, threshold):
    """Deterministic lexical report deduplication; no semantic claim inference.

    Near-identical reports are detected with token trigram Jaccard; the
    threshold is configurable. Different
    explicit claim views are protected even when their quoted text is identical.
    Cluster membership requires similarity to every member, avoiding transitive
    chains of increasingly different reports. Raw reports remain in the pinned
    corpus; the selected representative retains every duplicate's source lineage.
    """
    threshold = _number(threshold, minimum=0)
    if not 0 < threshold <= 1:
        raise ValueError('Near-duplicate threshold must lie in (0,1].')
    clusters, shingles = [], {}
    for passage in sorted(passages, key=lambda item:item['evidence_id']):
        identity = passage['evidence_id']
        tokens = _tokens(passage['snippet'])
        width = min(3, len(tokens))
        shingles[identity] = ({tuple(tokens[i:i+width]) for i in range(len(tokens)-width+1)}
                              if width else {passage['snippet']})
        def matches(other):
            left, right = passage.get('claim_group'), other.get('claim_group')
            if left and right and left != right:
                return False
            a, b = shingles[identity], shingles[other['evidence_id']]
            return len(a & b)/len(a | b) >= threshold
        for cluster in clusters:
            if all(matches(other) for other in cluster):
                cluster.append(passage)
                break
        else:
            clusters.append([passage])
    return clusters


def _normalized_field(record, name, value):
    # Derived records must not silently replace different curated assertions.
    if name in record and record[name] != value:
        raise ValueError('Conflicting curated and derived normalized field: '+name)
    record[name] = value


def _prepare_tailored_corpus(passages, *, snapshot, options):
    """Process only admitted source annotations, never labels or guessed roles.

    Optional preprocessing keys: near_duplicate_threshold (default .95), an
    exact case-sensitive alias_table (surface -> canonical string, or an object
    with canonical and optional disambiguation_note), schema_maps (source_schema
    -> {normalized_field: source_record_field}), and field_definitions. Source
    entities use explicit ``entities: [{surface: ...}]`` annotations; IOC events
    use ``ioc_observations: [{ioc, observed_at, campaign_id?}]``. Defanged IOC
    quotes can declare a separate ``surface``. No keyword role/campaign detector
    or fuzzy entity merger is introduced. Original source records stay intact.
    """
    if not isinstance(options, dict) or set(options)-{
            'near_duplicate_threshold', 'alias_table', 'schema_maps', 'field_definitions'}:
        raise ValueError('Unknown tailored corpus preprocessing configuration.')
    aliases = options.get('alias_table', {})
    schemas = options.get('schema_maps', {})
    definitions = options.get('field_definitions', {})
    if not all(isinstance(value, dict) for value in (aliases, schemas, definitions)):
        raise ValueError('Alias tables, schema maps and definitions must be objects.')
    def resolve(surface):
        if not isinstance(surface, str) or not surface:
            raise ValueError('Entity and IOC annotations require nonempty strings.')
        canonical, seen = surface, set()
        while canonical in aliases:
            if canonical in seen:
                raise ValueError('Cyclic alias table.')
            seen.add(canonical)
            entry = aliases[canonical]
            target = entry.get('canonical') if isinstance(entry, dict) else entry
            if not isinstance(target, str) or not target:
                raise ValueError('Alias table entries require a canonical entity string.')
            if target == canonical:
                break
            canonical = target
        return canonical
    # Validate the whole declared table, including entries not used by this query.
    for surface in aliases:
        resolve(surface)
    threshold = options.get('near_duplicate_threshold', .95)
    clusters = _duplicate_clusters(passages, threshold)
    cluster_by_id = {item['evidence_id']: cluster for cluster in clusters for item in cluster}
    # Multiple passages of a document and syndicated near-identical documents
    # count as one report. Union document IDs so separate spans cannot reinflate
    # an already deduplicated report's IOC frequency.
    parent = {item['document_id']:item['document_id'] for item in passages}
    def report_root(identity):
        while parent[identity] != identity:
            identity = parent[identity]
        return identity
    for cluster in clusters:
        root = min(report_root(item['document_id']) for item in cluster)
        for item in cluster:
            parent[report_root(item['document_id'])] = root
    observations, profiles_by_evidence = {}, {}
    for passage in passages:
        if not isinstance(passage.get('publisher'), str) or not passage['publisher']:
            raise ValueError('Tailored evidence requires explicit publisher provenance.')
        record = passage.setdefault('normalized_record', {})
        if not isinstance(record, dict):
            raise ValueError('Normalized evidence records must be objects.')
        annotated_entities = copy.deepcopy(passage.get('canonical_entities', []))
        if not isinstance(annotated_entities, list):
            raise ValueError('Canonical entities must be a list of explicit annotations.')
        entities = []
        for annotation in [*annotated_entities, *passage.get('entities', [])]:
            entity = {'surface':annotation} if isinstance(annotation, str) else copy.deepcopy(annotation)
            surface = entity['surface']
            canonical = resolve(entity.get('canonical',surface))
            if surface in aliases and canonical != resolve(surface):
                raise ValueError('Curated entity conflicts with the declared alias table.')
            entity['canonical'] = canonical
            alias = aliases.get(surface)
            if isinstance(alias, dict) and alias.get('disambiguation_note'):
                entity['disambiguation_note'] = alias['disambiguation_note']
            if entity not in entities:
                entities.append(entity)
        passage['canonical_entities'] = entities
        if 'incident_roles' not in passage:
            # This projects explicitly labeled roles; it does not label raw text.
            passage['incident_roles'] = sorted({entity['incident_role'] for entity in entities
                                                if entity.get('incident_role')})
        schema = passage.get('source_schema')
        if schema in schemas:
            source = passage.get('source_record')
            if not isinstance(source, dict) or not isinstance(schemas[schema], dict):
                raise ValueError('Schema normalization requires an explicit source record and field map.')
            fields = {}
            for field, source_field in schemas[schema].items():
                if source_field not in source:
                    raise ValueError('Mapped source record field is missing: '+source_field)
                value = source[source_field]
                fields[field] = resolve(value) if isinstance(value, str) else copy.deepcopy(value)
            _normalized_field(record, 'fields', fields)
            _normalized_field(record, 'field_definitions', {
                field:definitions[field] for field in fields if field in definitions})
        for observation in passage.get('ioc_observations', []):
            observed = _day(observation['observed_at'])
            if observed > snapshot:
                continue
            surface = observation.get('surface', observation['ioc'])
            if not isinstance(surface, str) or not surface or (
                    surface not in passage['snippet'] and
                    surface not in _canonical_json(passage.get('source_record', {}))):
                raise ValueError('IOC observations must identify a source-grounded surface.')
            ioc = resolve(observation['ioc'])
            profile = observations.setdefault(ioc, {
                'dates':set(), 'reports':set(), 'campaigns':set(), 'evidence':set()})
            profile['dates'].add(observed.isoformat())
            profile['reports'].add(report_root(passage['document_id']))
            campaign = observation.get('campaign_id')
            if campaign is not None:
                if not isinstance(campaign, str) or not campaign:
                    raise ValueError('Explicit campaign identities must be nonempty strings.')
                profile['campaigns'].add(campaign)
            profile['evidence'].add(passage['evidence_id'])
            profiles_by_evidence.setdefault(passage['evidence_id'], set()).add(ioc)
    profiles = {ioc:{
        'ioc':ioc, 'first_seen':min(values['dates']), 'last_seen':max(values['dates']),
        'frequency':len(values['reports']),
        'frequency_unit':'distinct_admitted_reports_after_deduplication',
        'campaign_count':len(values['campaigns']), 'campaign_ids':sorted(values['campaigns']),
        'supporting_evidence_ids':sorted(values['evidence']),
    } for ioc, values in observations.items()}
    for passage in passages:
        own = [copy.deepcopy(profiles[ioc])
               for ioc in sorted(profiles_by_evidence.get(passage['evidence_id'], set()))]
        if own:
            _normalized_field(passage['normalized_record'], 'ioc_reuse_profiles', own)
            # The qualitative paper leaves the threshold statistic unspecified.
            # Conservatively use the larger of report frequency and campaign
            # count, retaining a higher explicitly curated count when supplied.
            original = _number(passage.get('ioc_reuse_count', 0), minimum=0)
            derived = max(max(profile['frequency'], profile['campaign_count']) for profile in own)
            if original and original != derived:
                _normalized_field(passage['normalized_record'], 'provided_ioc_reuse_count', original)
            passage['ioc_reuse_count'] = max(original, derived)
    return cluster_by_id, {'method':'explicit_annotations_and_lexical_trigram_jaccard',
        'near_duplicate_threshold':threshold, 'alias_table_hash':_content_hash(aliases),
        'schema_maps_hash':_content_hash(schemas),
        'ioc_profile_population':'all_snapshot_admitted_passages',
        'ioc_reuse_statistic':'max_distinct_report_frequency_campaign_count_and_curated_count',
        'ioc_profiles':profiles,
        'duplicate_groups':[[item['evidence_id'] for item in cluster] for cluster in clusters if len(cluster)>1]}


def _tailored_evidence_features(admitted, policy):
    """Operationalize H.2 only from explicitly typed evidence annotations.

    Optional evidence_policies sections:
      causal_context: context_only_weight, maximum_context_only,
                      require_classification (evidence_kind is source annotated)
      attack_chain: unbacked_stage_weight (stage_transitions link artifact IDs)
      standards: reference_evidence_ids, mapping_fields, on_invalid
                 (references carry authoritative_options; claims standard_mappings)
      diversity: fields, inverse_frequency_weight, minimum_distinct,
                 maximum_per_value, require_metadata (diversity_metadata fields)
      counterevidence: priority_bonus, minimum_records (evidence_relation)
    selection_search_limit bounds exact feasibility work, failing explicitly if
    exhausted. No source annotation is manufactured from snippet keywords.
    """
    if not isinstance(policy, dict) or set(policy)-{
        'causal_context','attack_chain','standards','diversity','counterevidence','selection_search_limit'}:
        raise ValueError('Unknown tailored evidence policy.')
    allowed = {
        'causal_context':{'context_only_weight','maximum_context_only','require_classification'},
        'attack_chain':{'unbacked_stage_weight'},
        'standards':{'reference_evidence_ids','mapping_fields','on_invalid'},
        'diversity':{'fields','inverse_frequency_weight','minimum_distinct','maximum_per_value','require_metadata'},
        'counterevidence':{'priority_bonus','minimum_records'},
    }
    for name, fields in allowed.items():
        if name in policy and (not isinstance(policy[name], dict) or set(policy[name])-fields):
            raise ValueError('Unknown tailored evidence controls: '+name)
    by_id = {item['evidence_id']:item for item in admitted}
    causal = policy.get('causal_context', {})
    chain = policy.get('attack_chain', {})
    diversity = policy.get('diversity', {})
    counter = policy.get('counterevidence', {})
    context_weight = _number(causal.get('context_only_weight', 1), minimum=0)
    stage_weight = _number(chain.get('unbacked_stage_weight', 1), minimum=0)
    if max(context_weight, stage_weight) > 1:
        raise ValueError('Context-only and unbacked-stage weights must lie in [0,1].')
    fields = diversity.get('fields', [])
    minima = diversity.get('minimum_distinct', {})
    caps = diversity.get('maximum_per_value', {})
    if not isinstance(fields, list) or len(set(fields)) != len(fields) or any(
            not isinstance(field, str) or not field for field in fields):
        raise ValueError('Diversity fields must be distinct explicit metadata field names.')
    if 'publisher' in fields:
        raise ValueError('Metadata diversity cannot override the separate incident publisher requirement.')
    if not isinstance(minima, dict) or not isinstance(caps, dict) or (set(minima)|set(caps))-set(fields):
        raise ValueError('Diversity quotas must name configured metadata fields.')
    for value in minima.values():
        _nonnegative_int(value)
    for value in caps.values():
        _positive_int(value)
    controls = {
        'minimum_distinct':minima, 'maximum_per_value':caps,
        'maximum_context_only':(_nonnegative_int(causal['maximum_context_only'])
                                if 'maximum_context_only' in causal else None),
        'minimum_counterevidence':_nonnegative_int(counter.get('minimum_records',0)),
        'search_limit':_positive_int(policy.get('selection_search_limit',100000)),
    }
    # Frequency-based metadata balancing is configurable. Use
    # snapshot-admitted evidence only; never training distributions or gold labels.
    frequencies = {field:Counter() for field in fields}
    for passage in admitted:
        metadata = passage.get('diversity_metadata', {})
        if not isinstance(metadata, dict):
            raise ValueError('Diversity metadata must be explicitly annotated fields.')
        for field in fields:
            value = metadata.get(field)
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise ValueError('Diversity metadata values must be nonempty strings.')
                frequencies[field][value] += 1
    exponent = _number(diversity.get('inverse_frequency_weight',0), minimum=0)
    bonus = _number(counter.get('priority_bonus',0), minimum=0)
    references, options, option_sources = set(), {}, {}
    standards = policy.get('standards')
    if standards is not None:
        reference_ids = standards.get('reference_evidence_ids', [])
        mapping_fields = standards.get('mapping_fields', [])
        if (not isinstance(reference_ids, list) or not reference_ids or
                not isinstance(mapping_fields, list) or not mapping_fields or
                standards.get('on_invalid') not in {'exclude','annotate'}):
            raise ValueError('Declare authoritative references, mapping fields and invalid-mapping behavior.')
        for identity in reference_ids:
            if identity not in by_id:
                raise ValueError('Required authoritative reference is not snapshot admitted: '+identity)
            reference = by_id[identity]
            declared = reference.get('authoritative_options')
            if not isinstance(declared, dict):
                raise ValueError('Authoritative references require explicitly declared valid options.')
            references.add(identity)
            _normalized_field(reference['normalized_record'], 'authoritative_options', copy.deepcopy(declared))
            for field in mapping_fields:
                if field not in declared:
                    continue
                if not isinstance(declared[field], list):
                    raise ValueError('Authoritative options must be explicit lists.')
                for value in declared[field]:
                    key = _canonical_json(value)
                    options.setdefault(field,set()).add(key)
                    option_sources.setdefault((field,key),set()).add(identity)
        if any(field not in options for field in mapping_fields):
            raise ValueError('Authoritative references do not cover all configured mapping fields.')
    weights, bonuses, excluded, dependencies = {}, {}, set(), {}
    for passage in admitted:
        identity, record = passage['evidence_id'], passage['normalized_record']
        weight = 1.
        if 'causal_context' in policy and identity not in references and not passage.get('artifact_type'):
            kind = passage.get('evidence_kind', 'unclassified')
            if kind not in {'causal','context_only','mixed','unclassified'}:
                raise ValueError('Unsupported explicit evidence classification.')
            if causal.get('require_classification',False) and kind == 'unclassified':
                raise ValueError('Required causal/context classification is missing.')
            _normalized_field(record, 'evidence_kind', kind)
            if kind == 'context_only':
                weight *= context_weight
        if 'attack_chain' in policy and passage.get('stage_transitions'):
            notes, linked = [], set()
            for transition in passage['stage_transitions']:
                note = copy.deepcopy(transition)
                requested = transition.get('artifact_evidence_ids', [])
                valid = sorted({key for key in requested if key in by_id and by_id[key].get('artifact_type')})
                linked.update(valid)
                note['artifact_evidence_ids'] = valid
                note['unavailable_artifact_evidence_ids'] = sorted(set(requested)-set(valid))
                note['confidence_note'] = ('artifact_backed' if valid and len(valid)==len(set(requested))
                                           else 'soft_hint_without_admitted_artifacts')
                if 'confidence' in note and not 0 <= _number(note['confidence']) <= 1:
                    raise ValueError('Explicit stage confidence must lie in [0,1].')
                notes.append(note)
            if any(note['confidence_note'] != 'artifact_backed' for note in notes):
                weight *= stage_weight
            _normalized_field(record, 'stage_transitions', notes)
            dependencies[identity] = linked
        if 'attack_chain' in policy and passage.get('kill_chain_stage'):
            _normalized_field(record, 'kill_chain_stage', {
                'stage':passage['kill_chain_stage'], 'confidence_note':'soft_hint_requires_explicit_artifacts'})
        if 'diversity' in policy:
            metadata = {field:passage.get('diversity_metadata',{}).get(field) for field in fields}
            if diversity.get('require_metadata',False) and any(value is None for value in metadata.values()):
                raise ValueError('Required geography/tooling diversity metadata is missing.')
            _normalized_field(record, 'diversity_metadata', metadata)
            for field, value in metadata.items():
                if value is not None:
                    weight /= frequencies[field][value]**exponent
        relation = passage.get('evidence_relation', 'unclassified')
        if 'counterevidence' in policy:
            if relation not in {'support','counterevidence','alternative','unclassified'}:
                raise ValueError('Unsupported explicit evidence relationship.')
            _normalized_field(record, 'evidence_relation', relation)
        bonuses[identity] = bonus if relation in {'counterevidence','alternative'} else 0.
        if standards is not None and identity not in references:
            mappings = passage.get('standard_mappings', {})
            if not isinstance(mappings, dict):
                raise ValueError('Standard mappings require explicit field/value annotations.')
            checks = {}
            for field in standards['mapping_fields']:
                if field not in mappings:
                    continue
                value, key = mappings[field], _canonical_json(mappings[field])
                valid = key in options[field]
                checks[field] = {'value':copy.deepcopy(value), 'valid':valid,
                                 'reference_evidence_ids':sorted(option_sources.get((field,key), references))}
                if not valid and standards['on_invalid'] == 'exclude':
                    excluded.add(identity)
            _normalized_field(record, 'standard_validation', checks)
        weights[identity] = weight
    return weights, bonuses, excluded, dependencies, references, controls, {
        'policy':copy.deepcopy(policy), 'effective_selection_controls':controls,
        'diversity_frequencies':{field:dict(sorted(counts.items())) for field,counts in frequencies.items()},
        'authoritative_reference_ids':sorted(references),
        'authoritative_references_count_toward_incident_diversity':False,
        'excluded_invalid_mapping_ids':sorted(excluded)}


def _constrained_evidence_selection(ranked, *, budget, minimum_publishers,
                                    maximum_per_publisher, controls, dependencies, by_id, duplicate_groups,
                                    mandatory_references):
    """Exact feasibility for incident diversity and whole-package publisher caps.

    Greedy ranking decisions are accepted only if a feasible completion exists.
    Search branches on an unmet requirement, adding only candidates that cover
    it; it never has to enumerate irrelevant optional evidence. A declared work
    limit fails honestly rather than claiming a feasible policy is impossible.
    """
    states = 0
    lengths = [len(item['snippet']) for item in ranked]
    goals = {'publisher':minimum_publishers, **controls['minimum_distinct']}
    caps = controls['maximum_per_value']
    def summary(chosen):
        categories = {field:Counter() for field in set(goals)|set(caps)}
        context, counter, artifacts, clusters = 0, 0, set(), Counter()
        for index in chosen:
            item = ranked[index]
            for field in categories:
                value = item['publisher'] if field == 'publisher' else item.get('diversity_metadata',{}).get(field)
                if value is not None:
                    categories[field][value] += 1
            context += item.get('evidence_kind') == 'context_only'
            counter += item.get('evidence_relation') in {'counterevidence','alternative'}
            artifacts.update(dependencies.get(item['evidence_id'], set()))
            clusters[duplicate_groups.get(item['evidence_id'],item['evidence_id'])] += 1
        package_ids = (set(mandatory_references) | artifacts |
                       {ranked[index]['evidence_id'] for index in chosen})
        package_publishers = Counter(by_id[identity]['publisher'] for identity in package_ids)
        used = sum(lengths[index] for index in chosen)+sum(len(by_id[key]['snippet']) for key in artifacts)
        valid = used <= budget and all(all(count <= caps[field] for count in categories[field].values())
                                      for field in caps)
        valid = valid and all(count <= maximum_per_publisher for count in package_publishers.values())
        valid = valid and all(count <= 1 for count in clusters.values())
        if controls['maximum_context_only'] is not None:
            valid = valid and context <= controls['maximum_context_only']
        return valid, categories, counter, used, artifacts
    def feasible(chosen, available):
        failed = set()
        def search(current, remaining):
            nonlocal states
            states += 1
            if states > controls['search_limit']:
                raise ValueError('Tailored selection search limit reached; quota feasibility is unresolved.')
            key = frozenset(current)
            if key in failed:
                return False
            valid, categories, counter, used, _ = summary(current)
            if not valid:
                return False
            unmet = {field:target-len(categories[field]) for field,target in goals.items()
                     if len(categories[field]) < target}
            counter_missing = max(0,controls['minimum_counterevidence']-counter)
            if not unmet and not counter_missing:
                return True
            candidates = [index for index in remaining if summary((*current,index))[0]]
            choices = []
            for field, missing in unmet.items():
                values, matching = {}, []
                for index in candidates:
                    item = ranked[index]
                    value = item['publisher'] if field == 'publisher' else item.get('diversity_metadata',{}).get(field)
                    if value is not None and value not in categories[field]:
                        values[value] = min(values.get(value,math.inf),lengths[index])
                        matching.append(index)
                if len(values) < missing or sum(sorted(values.values())[:missing]) > budget-used:
                    return False
                choices.append(matching)
            if counter_missing:
                matching = [index for index in candidates if ranked[index].get('evidence_relation')
                            in {'counterevidence','alternative'}]
                if len(matching) < counter_missing:
                    return False
                choices.append(matching)
            for index in sorted(min(choices,key=len),key=lambda i:(lengths[i],i)):
                if search((*current,index), [other for other in remaining if other != index]):
                    return True
            failed.add(key)
            return False
        return search(tuple(chosen), list(available))
    if not feasible([], range(len(ranked))):
        raise ValueError('Publisher cap, source diversity or metadata quotas cannot fit the admitted whole evidence spans.')
    chosen = []
    for index in range(len(ranked)):
        if feasible([*chosen,index], range(index+1,len(ranked))):
            chosen.append(index)
    _, categories, _, used, artifacts = summary(chosen)
    selected = [ranked[index] for index in chosen]
    for item in selected:
        item['snapshot_admission_verified'] = True
    return selected, used, categories['publisher'], artifacts


def _budget_diverse_evidence(ranked, *, budget, minimum_publishers, maximum_per_publisher,
                             controls=None, dependencies=None, by_id=None, duplicate_groups=None,
                             mandatory_references=None):
    """Greedy relevance selection with exact cheapest-publisher feasibility.

    Reserving each missing publisher's cheapest remaining whole span makes the
    minimum diversity constraint feasible even when top passages are too long.
    This is a greedy heuristic, not an optimal relevance knapsack.
    """
    duplicate_groups = duplicate_groups or {}
    has_duplicates = len({duplicate_groups.get(item['evidence_id'],item['evidence_id'])
                          for item in ranked}) < len(ranked)
    if controls and (mandatory_references or has_duplicates or controls['minimum_distinct'] or controls['maximum_per_value'] or
                     controls['maximum_context_only'] is not None or
                     controls['minimum_counterevidence'] or any((dependencies or {}).values())):
        return _constrained_evidence_selection(
            ranked, budget=budget, minimum_publishers=minimum_publishers,
            maximum_per_publisher=maximum_per_publisher, controls=controls,
            dependencies=dependencies or {}, by_id=by_id or {}, duplicate_groups=duplicate_groups,
            mandatory_references=mandatory_references or set())
    counts, selected, used = Counter(), [], 0
    def feasible(rows, chosen, remaining):
        missing = max(0, minimum_publishers-len(chosen))
        cheapest = {}
        for passage in rows:
            publisher = passage['publisher']
            if publisher not in chosen:
                cheapest[publisher] = min(cheapest.get(publisher, math.inf),len(passage['snippet']))
        return len(cheapest) >= missing and sum(sorted(cheapest.values())[:missing]) <= remaining
    if not feasible(ranked, counts, budget):
        raise ValueError('Source diversity requirement cannot fit admitted evidence within the context budget.')
    for index, passage in enumerate(ranked):
        publisher, length = passage['publisher'], len(passage['snippet'])
        if counts.get(publisher, 0) >= maximum_per_publisher or used+length > budget:
            continue
        chosen = dict(counts)
        chosen[publisher] = chosen.get(publisher,0)+1
        if not feasible(ranked[index+1:], chosen, budget-used-length):
            continue
        passage['snapshot_admission_verified'] = True
        selected.append(passage)
        counts[publisher] += 1
        used += length
    if len(counts) < minimum_publishers:
        raise ValueError('Source diversity requirement cannot be satisfied by admitted evidence.')
    return selected, used, counts, set()


def _bundle_tailored_views(selected, snapshot):
    views = {}
    for passage in selected:
        group = passage.get('claim_group') or 'source:'+passage['publisher']
        if not isinstance(group, str):
            raise ValueError('Explicit claim groups must be strings.')
        basis = 'curated_claim_group' if passage.get('claim_group') else 'publisher'
        passage['claim_group'] = group
        views.setdefault(group, []).append((passage, basis))
    bundled = []
    for group, members in views.items():
        members.sort(key=lambda item:(_day(item[0]['publication_time']),item[0]['evidence_id']))
        identities = [item['evidence_id'] for item, _ in members]
        for passage, basis in members:
            record = passage['normalized_record']
            _normalized_field(record, 'view', {'group_id':group, 'basis':basis, 'evidence_ids':identities})
            _normalized_field(record, 'grounding', {
                'evidence_id':passage['evidence_id'], 'publisher':passage['publisher'],
                'publication_time':passage['publication_time'], 'as_of':snapshot.isoformat(),
                'span':[0,len(passage['snippet'])]})
            bundled.append(passage)
    return bundled


def _resolve_tailored_policy(*, payload, parameters):
    """Bind controller H.2 declarations to the executed adapter policy.

    ``parameters.tailored`` is authoritative. Each non-null controller mapping
    must explicitly agree with it; unknown controls fail closed. Supported keys
    are minimum_publishers/maximum_per_publisher in source_diversity_parameters,
    maximum_age_days/recency_weight/source_quality_weight in freshness_parameters,
    maximum_ioc_reuse in ioc_reuse_thresholds, and the complete evidence_policies
    mapping. Omitted controls add no override. The resolved policy is persisted.
    Validate before hybrid ranking, including any local encoder execution.
    """
    policy = copy.deepcopy(parameters['tailored'])
    if not isinstance(policy, dict):
        raise ValueError('Tailored backend policy must be an explicit mapping.')
    normalizers = {
        'minimum_publishers': _positive_int,
        'maximum_per_publisher': _positive_int,
        'maximum_age_days': _nonnegative_int,
        'maximum_ioc_reuse': _nonnegative_int,
        'recency_weight': lambda value: _number(value, minimum=0),
        'source_quality_weight': lambda value: _number(value, minimum=0),
    }
    for field, normalize in normalizers.items():
        policy[field] = normalize(policy[field])
    policy.setdefault('preprocessing', {})
    policy.setdefault('evidence_policies', {})
    declared = payload.get('retrieval_configuration', {})
    if not isinstance(declared, dict):
        raise ValueError('Forwarded retrieval configuration must be a mapping.')
    groups = {
        'source_diversity_parameters': {'minimum_publishers', 'maximum_per_publisher'},
        'freshness_parameters': {'maximum_age_days', 'recency_weight', 'source_quality_weight'},
        'ioc_reuse_thresholds': {'maximum_ioc_reuse'},
    }
    forwarded = {}
    for group, fields in groups.items():
        controls = declared.get(group)
        if controls is None:
            continue
        if not isinstance(controls, dict) or set(controls)-fields:
            raise ValueError('Unknown or malformed forwarded tailored controls: '+group)
        for field, value in controls.items():
            normalized = normalizers[field](value)
            if normalized != policy[field]:
                raise ValueError('Forwarded tailored '+field+' controls disagree with backend parameters.')
            forwarded[field] = normalized
    evidence_policies = declared.get('evidence_policies')
    if evidence_policies is not None:
        # These nested policies contain boolean gates as well as numeric quotas;
        # compare canonical JSON so e.g. True cannot masquerade as the count 1.
        if (not isinstance(evidence_policies, dict) or
                _canonical_json(evidence_policies) != _canonical_json(policy['evidence_policies'])):
            raise ValueError('Forwarded tailored evidence_policies controls disagree with backend parameters.')
        forwarded['evidence_policies'] = copy.deepcopy(evidence_policies)
    return policy, forwarded


def tailored_retrieval(*, payload, parameters):
    """Apply source-grounded H.2 processing and gates before the final budget.

    Incident roles, campaign-specific behavior, environment and source quality
    remain explicit source annotations. Optional tailored.preprocessing resolves
    exact aliases, maps declared schemas, computes snapshot-bounded IOC profiles,
    and deduplicates lexical near-copies. Grouped/dated normalized records reach
    the runner's structured feeds alongside unmodified quoted evidence spans.
    All numerical/lexical choices are configurable. The original hybrid API and frozen-score binding are kept.
    """
    policy, forwarded = _resolve_tailored_policy(payload=payload, parameters=parameters)
    passages, ordered, provenance = _rank_hybrid(payload=payload, parameters=parameters)
    snapshot = _day(payload['snapshot_date'])
    admitted_ids = set(provenance['admitted_evidence_ids'])
    admitted = [passage for passage in passages if passage['evidence_id'] in admitted_ids]
    clusters, preprocessing = _prepare_tailored_corpus(
        admitted, snapshot=snapshot, options=policy.get('preprocessing', {}))
    weights, bonuses, excluded, dependencies, references, controls, feature_provenance = (
        _tailored_evidence_features(admitted, policy.get('evidence_policies', {})))
    by_id = {passage['evidence_id']:passage for passage in admitted}
    artifact_ids = set().union(*dependencies.values()) if dependencies else set()
    eligible = []
    for rank, index in enumerate(ordered):
        passage = passages[index]
        identity = passage['evidence_id']
        if identity in excluded or identity in references or identity in artifact_ids:
            continue
        for field in policy['required_metadata']:
            if field not in passage:
                raise ValueError('Missing required tailored evidence metadata: '+field)
        if policy['require_incident_roles'] and not passage.get('incident_roles'):
            continue
        reuse = _number(passage.get('ioc_reuse_count',0),minimum=0)
        if reuse > policy['maximum_ioc_reuse'] and not passage.get('campaign_specific_evidence',False):
            continue
        if any(field not in passage.get('environment',{}) or field not in payload['environment'] or
               passage['environment'][field] != payload['environment'][field]
               for field in policy['environment_fields']):
            continue
        age = (snapshot-_day(passage['publication_time'])).days
        if age > policy['maximum_age_days']:
            continue
        quality = _number(passage.get('source_quality',0),minimum=0)
        score = 1/(rank+1) + _number(policy['source_quality_weight'],minimum=0)*quality
        score += _number(policy['recency_weight'],minimum=0)/(age+1)
        score = score*weights[identity]+bonuses[identity]
        eligible.append((score,passage))
    eligible.sort(key=lambda pair:(-pair[0],pair[1]['evidence_id']))
    ranked = []
    for _, passage in eligible:
        cluster = clusters[passage['evidence_id']]
        duplicates = [{field:item[field] for field in (
            'evidence_id','document_id','publisher','publication_time','available_at','source_version','content_hash',
            'source_id','source_url','content_reference','modification_time') if field in item}
            for item in cluster if item['evidence_id'] != passage['evidence_id']]
        if duplicates:
            _normalized_field(passage['normalized_record'], 'duplicate_sources', duplicates)
        ranked.append(passage)
    budget = _positive_int(parameters['context_budget_chars'])
    reference_cost = sum(len(by_id[identity]['snippet']) for identity in references)
    if reference_cost > budget:
        raise ValueError('Required authoritative references exceed the whole-span context budget.')
    # References are mandatory context. Shared artifact dependencies cost space
    # only when their associated incident evidence is actually selected.
    dependencies = {identity:keys-references for identity,keys in dependencies.items()}
    selected, used, counts, artifacts = _budget_diverse_evidence(
        ranked, budget=budget-reference_cost, minimum_publishers=_positive_int(policy['minimum_publishers']),
        maximum_per_publisher=_positive_int(policy['maximum_per_publisher']),
        controls=controls, dependencies=dependencies, by_id=by_id,
        mandatory_references=references,
        duplicate_groups={identity:cluster[0]['evidence_id'] for identity,cluster in clusters.items()})
    for identity in sorted(references|artifacts):
        by_id[identity]['snapshot_admission_verified'] = True
        selected.append(by_id[identity])
    used += reference_cost
    selected = _bundle_tailored_views(selected, snapshot)
    package_counts = Counter(passage['publisher'] for passage in selected)
    return {'evidence':selected,'provenance':{**provenance,
        'preprocessing':preprocessing,
        'evidence_policies':feature_provenance,
        'incident_publishers':sorted(counts),
        'incident_publisher_counts':dict(sorted(counts.items())),
        'publisher_counts':dict(sorted(package_counts.items())),
        'publisher_cap_scope':'all_emitted_evidence_passages',
        'effective_tailored_policy':copy.deepcopy(policy),
        'tailored_policy_authority':'backend_parameters_with_controller_equivalence',
        'normalized_forwarded_tailored_controls':forwarded,
        'tailored_policy_hash':_content_hash(policy),
        'publishers':sorted({passage['publisher'] for passage in selected}),
        'selected_evidence_ids':[p['evidence_id'] for p in selected],
        'context_characters':used, 'context_budget_chars':budget}}


def dense_embeddings(*, payload, parameters):
    """Actual local sentence embeddings for the KMeans comparison."""
    checkpoint = _local_model(parameters['checkpoint'])
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(checkpoint,device='cpu',local_files_only=True)
    vectors = encoder.encode(payload['texts'],batch_size=_positive_int(parameters['batch_size']),
                             normalize_embeddings=parameters['normalize_embeddings'])
    return {'vectors':np.asarray(vectors).tolist(),'provenance':{
        'checkpoint':checkpoint,'revision':parameters['revision'],'device':'cpu'}}


def prepare_condition(*, payload, parameters):
    """Construct an explicit matched ablation/stability configuration; no inference."""
    from experiments import _get_pointer, _replace_pointer
    configuration = copy.deepcopy(payload['configuration'])
    patches = copy.deepcopy(parameters['changes'])
    before_provenance = copy.deepcopy(configuration['implementation_choices'])
    for patch in patches:
        if _get_pointer(configuration,patch['pointer']) != patch['before']:
            raise ValueError('Condition patch does not match the frozen original value.')
        _replace_pointer(configuration,patch['pointer'],patch['after'])
        path = '.'.join(token.replace('~1','/').replace('~0','~')
                        for token in patch['pointer'][1:].split('/'))
        if path.startswith('implementation_choices'):
            raise ValueError('Declare scientific condition changes rather than editing provenance directly.')
        configuration['implementation_choices'][path] = {
            'configuration_path':path,'origin':'implementation',
            'rationale':parameters['rationale'],'affected_experiments':['all'],
            'selected_value':patch['after'], 'manifest_reference':None,
            'source':None, 'applicability':None}
    if before_provenance != configuration['implementation_choices']:
        patches.append({'pointer':'/implementation_choices','before':before_provenance,
                        'after':copy.deepcopy(configuration['implementation_choices'])})
    return {'configuration':configuration,'response_ids':sorted(payload['response_ids']),
            'changes':patches,'provenance':{'condition':payload['condition'],
            'parameters_hash':_content_hash(parameters),'inference_executed':False}}


def checkpoint_digest(checkpoint):
    """Digest exact local checkpoint directory bytes, including tokenizer/config."""
    root = Path(_local_model(checkpoint))
    entries = []
    for path in sorted(root.rglob('*')):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):
                    digest.update(chunk)
            entries.append((str(path.relative_to(root)),digest.hexdigest()))
    if not entries:
        raise ValueError('The local checkpoint contains no files.')
    return _content_hash(entries)


@lru_cache(maxsize=1)
def _load_causal_checkpoint(checkpoint, revision, expected_digest):
    # One pinned model at a time limits CPU memory; repeated variants reuse it.
    if checkpoint_digest(checkpoint) != expected_digest:
        raise ValueError('Local checkpoint content differs from the declared generation checkpoint.')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
                                               torch_dtype=torch.float32).to('cpu').eval()
    return tokenizer, model
