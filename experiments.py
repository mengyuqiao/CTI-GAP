"""Configuration-driven experiment orchestration.

This module coordinates existing inference, annotation, and evaluation services.
It does not train models, repair model outputs, infer missing scientific settings,
or use audit references to configure experimental methods.

Unresolved scientific settings have an engineering default of ``None`` and block
the affected experiment. They are never replaced with library defaults.

Optional adapter contract
========================

Some optional experiments require external implementations/checkpoints.
Existing open policy mappings may contain an explicit adapter:

    {
        "adapter": {
            "callable": "package.module:function",
            "version": "immutable-version",
            "source_sha256": "<SHA-256 of the defining Python source file>",
            "parameters": {}
        }
    }

Adapters are trusted, local, in-process functions, not remote services. Their
source digest is checked before execution. Adapter declarations and returned
artifacts are included in experimental manifests. Importing an adapter is
explicit; no implementation or checkpoint is selected automatically.

Data adapters receive keyword arguments ``payload`` and ``parameters`` and return
JSON-compatible objects. They may use explicitly declared local embedding,
retrieval, verification, or candidate-scoring models. They must not invoke hosted
task-generation services, access audit annotations, or retrieve undeclared evidence.

The supported roles are:

* embedding: return {"vectors": [[...], ...]} in supplied text order.
* retrieval: return {"evidence": [passage, ...], "provenance": {...}}.
* verification: return the claim-feedback contract required by
  prompts/defense_revision.jinja2.
* preservation: return per-variant, externally established preservation decisions.
* inference constraints: return task guidance, not a completed answer.
* condition preparation: explicitly construct an effective configuration and
  response population for an ablation or full RQ1--RQ2 stability rerun.

Condition preparation adapters are different from data adapters. They receive
only the sanitized configuration and declared response IDs, and return:

    {
        "configuration": <complete ExperimentConfig-compatible mapping>,
        "response_ids": [...],
        "changes": [{"pointer": "/...", "before": ..., "after": ...}],
        "provenance": {...}
    }

The runner, not the adapter, executes the lower-level pipeline. It validates
effective configurations, copies immutable evaluated responses with explicit
configuration lineage when necessary, and never rewrites audit records.

Task prompt manifests
=====================

The configured task-guidance manifest uses:

    {"version": "...", "tasks": {
      "<task>": {"guidance": "...", "candidate_policy": "include|exclude"}
    }}

The output-profile manifest uses:

    {"version": "...", "tasks": {
      "<task>": {"output_profile": {...},
                 "schema_contract": {
                   "version": "...", "kind": "task", "json_schema": {...}
                 }}
    }}

A task's original instance output_profile must agree with the manifest contract.
No populated reference answer is accepted as an output profile.

Counterfactual operators
=======================

The supported deterministic operator mapping is:

    {
      "protocol_version": "...",
      "items": [{
        "operator_id": "...",
        "family": "I_spur|I_contra|I_gen",
        "instance_ids": [...],
        "edits": [{"pointer": "/environment/...", "before": ..., "after": ...}],
        "preservation_justification": "...",
        "preservation_evidence_refs": [...]
      }],
      "preservation_protocol": {
        "mode": "manifest",
        "decisions": {
          "<operator_id>": {
            "decision": "admissible|rejected|unresolved",
            "instance_id": "...",
            "original_content_hash": "...",
            "modified_model_input_hash": "...",
            "reviewer_id": "...",
            "justification": "...",
            "evidence_refs": [...],
            "answer_preserved": true,
            "causal_attributes_preserved": true,
            "chronology_valid": true
          }
        }
      },
      "accepted_count_policy": "require_requested|use_admissible"
    }

Edits address only the model-visible prompt projection. Snapshot edits must
already be justified for the requested task; consistency never validates them.

C3 probability support
=====================

The current ModelClient retains token log probabilities. Its token interface
supports only an explicitly complete, single-token candidate space with actual
normalized probabilities recoverable from that artifact. Structured task JSON
uses a separate, explicit ``candidate_log_likelihoods`` scoring adapter instead:
``aggregation_space="finite_candidates"``, ``label_pointer="/answer/..."``,
``sum_tolerance``, ``adapter``, and ``candidate_output_adapter``. Scoring adapters
receive the exact rendered task prompt, single-user-message mapping, configured
model endpoint/revision, model ID, run index, and sanitized model inputs. Their
explicit parameters declare the finite candidate completions. The local causal
backend additionally binds ``generation_checkpoint_sha256`` to its checkpoint
content digest. Returned ``probabilities`` must cover exactly that candidate
space and include scoring ``provenance``.
Truncated top-token lists, self-reported confidence, arbitrary candidate scores,
and free-text voting are not probability distributions.

The explicit extraction mapping is:

    {
      "interface": "token_logprobs",
      "aggregation_space": "tokens",
      "token_to_label": {"<exact token>": "<candidate label>", ...},
      "sum_tolerance": <finite nonnegative number>,
      "candidate_output_adapter": <versioned data adapter>
    }

An admissible full-distribution artifact must expose a list of token/logprob
objects at data.content[0].top_logprobs. Missing support blocks correction.
No token list is renormalized to conceal missing probability mass.

Generative C3 explicitly selects ``interface="autoregressive_complete"``,
``aggregation_space="task_outputs"`` and ``output_equivalence_rule="task_field_exact"``.
The pinned ``generative_probabilities:complete_output_distribution`` backend
enumerates every EOS-terminated or output-horizon token sequence, or blocks when
the declared exact-enumeration budget is insufficient. Schema-valid outputs are
projected onto the explicit ``label_pointer`` and equal canonical JSON task
judgments share probability mass. Invalid raw outputs remain distinct outcomes;
their probability mass is neither discarded nor renormalized. Applying
``exp_negative_entropy`` to this projected distribution implements the
entropy-based weighting rule for generative task outputs.

Built-in optional protocols
===========================

Stability conditions declare ``perturbations`` and five explicit ``runs``. The
design may select ``audit_assignment_adapter: {mode: rq3}`` to apply each newly
discovered taxonomy through the existing categorization pipeline. Its operational
annotation protocol uses explicit ``$taxonomy_id``/``$config_hash`` bindings and
wildcard annotator assignments/reference applicability. Audit labels are not read.

Declared ablation strategies are ``one_round_with_review`` (repetition),
``two_round_automated`` (human verification, explicit label-set plurality and
blocked ties), ``single_judge`` (multi-agent decision, explicitly chosen judge),
``hard_rules`` (stratification), and ``model_initialization`` (human initiation).
The separately declared ``fixed_initial_taxonomy`` human-initiation comparator
is a fixed-taxonomy categorization comparison; it does not claim
to execute or evaluate de novo taxonomy discovery.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from annotation_app import AnnotationService
from categorization import Categorizer
from config import ExperimentConfig
from evaluation import Evaluator
from failure_identification import FailureIdentifier
from generation_context import review_task_input, save_generation_context
from model_client import ModelClient, _ValidatedOutput, _strict_object
from schemas import (
    AssignmentOrigin,
    AuditCohort,
    EvidenceRole,
    FailureDecision,
    LabelAssignment,
    ModelResponse,
    ResponseStatus,
    StageState,
    StageStatus,
    TaskInstance,
    TaxonomyVersion,
    _canonical_json,
    _content_hash,
    _identifier,
    _json_primitive,
)
from storage import ArtifactStore
from taxonomy import TaxonomyManager


_UNSET: object = object()
_TAXONOMY_KINDS: tuple[str, ...] = ("taxonomy", "taxonomy_version")
_ASSIGNMENT_KINDS: tuple[str, ...] = (
    "label_assignment", "assignment", "consensus", "final_assignment"
)
_RESERVED_INPUT_KEYS: frozenset[str] = frozenset(
    {
        "reference", "reference_answer", "adjudicated_labels", "final_labels",
        "automatic_assignments", "audit_labels", "human_labels",
    }
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _required(value: Any, path: str) -> Any:
    if value is None or value == "" or value == {} or value == []:
        raise ValueError(f"{path}: unresolved; an explicit protocol is required.")
    return value


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an explicit mapping.")
    return copy.deepcopy(value)


def _ids(value: Any, path: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{path}: expected an identifier sequence.")
    result: list[str] = [_identifier(item) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{path}: duplicate identifiers are not permitted.")
    if not result and not allow_empty:
        raise ValueError(f"{path}: an empty population is not permitted.")
    return sorted(result)


def _integer(value: Any, path: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{path}: expected an integer >= {minimum}.")
    return value


def _number(value: Any, path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path}: expected a finite number.")
    result: float = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{path}: invalid finite numeric value.")
    return result


def _pointer_tokens(pointer: str) -> list[str]:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("Edit pointers must be nonempty RFC 6901 pointers.")
    result: list[str] = []
    for component in pointer[1:].split("/"):
        for index, character in enumerate(component):
            if character == "~" and (
                index + 1 >= len(component) or component[index + 1] not in "01"
            ):
                raise ValueError("Malformed JSON Pointer escape.")
        result.append(component.replace("~1", "/").replace("~0", "~"))
    return result


def _get_pointer(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    current: Any = value
    for token in _pointer_tokens(pointer):
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif (
            isinstance(current, list)
            and token.isdigit()
            and str(int(token)) == token
            and int(token) < len(current)
        ):
            current = current[int(token)]
        else:
            raise ValueError(f"Pointer does not resolve: {pointer}.")
    return current


def _replace_pointer(value: dict[str, Any], pointer: str, replacement: Any) -> None:
    tokens: list[str] = _pointer_tokens(pointer)
    current: Any = value
    for token in tokens[:-1]:
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            raise ValueError(f"Edit pointer does not resolve: {pointer}.")
    final: str = tokens[-1]
    if isinstance(current, dict) and final in current:
        current[final] = copy.deepcopy(replacement)
    elif isinstance(current, list) and final.isdigit() and int(final) < len(current):
        current[int(final)] = copy.deepcopy(replacement)
    else:
        raise ValueError(f"Edit pointer does not address an existing field: {pointer}.")


def _reject_internal_fields(value: Any) -> None:
    """Inspect structured projections, not the words in retained evidence prose."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _RESERVED_INPUT_KEYS:
                raise ValueError(f"Model-visible data contains internal field {key!r}.")
            _reject_internal_fields(item)
    elif isinstance(value, list):
        for item in value:
            _reject_internal_fields(item)


class ExperimentRunner:
    """Compose existing pipeline components into matched, persistent experiments."""

    def __init__(
        self,
        config: ExperimentConfig,
        store: ArtifactStore,
        models: ModelClient,
        evaluator: Evaluator,
        annotations: AnnotationService,
    ) -> None:
        self._config: ExperimentConfig = config
        self._store: ArtifactStore = store
        self._models: ModelClient = models
        self._evaluator: Evaluator = evaluator
        self._annotations: AnnotationService = annotations
        source_path: Path | None = getattr(config, "_source_path", None)
        self._base: Path = source_path.parent if source_path is not None else Path.cwd()

    def _manifest(self, purpose: str, content: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "purpose": purpose,
            "config_hash": self._config.config_hash,
            "corpus_version": self._config.corpus_version,
            **content,
        }
        identity: str = _content_hash(payload)
        result: dict[str, Any] = {"manifest_id": identity, **payload}
        self._store.create_record("experiment_manifest", identity, result)
        return result

    def _load_manifest(self, declaration: Any, path: str) -> dict[str, Any]:
        location: str = _identifier(_required(declaration, path))
        if location.startswith("sha256:"):
            raise ValueError(
                f"{path}: supply a local manifest or an existing storage manifest ID; "
                "ArtifactStore has no public blob-reading interface."
            )
        try:
            return self._store.read_record("experiment_manifest", location)
        except KeyError:
            pass
        local: Path = Path(location)
        local = local if local.is_absolute() else self._base / local
        if not local.is_file():
            raise FileNotFoundError(f"{path}: declared manifest is missing.")
        raw: bytes = local.read_bytes()
        if local.suffix.lower() == ".json":
            result: Any = json.loads(raw.decode("utf-8"))
        else:
            result = yaml.safe_load(raw.decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError(f"{path}: manifest must be an object.")
        _identifier(_required(result.get("version"), f"{path}.version"))
        _canonical_json(result)
        self._store.save_blob(raw)
        return result

    def _adapter(self, declaration: Any, path: str) -> Callable[..., Any]:
        """Resolve an explicitly versioned local adapter, checking its source."""
        specification: dict[str, Any] = _mapping(declaration, path)
        if set(specification) != {
            "callable", "version", "source_sha256", "parameters"
        }:
            raise ValueError(
                f"{path}: declare callable, version, source_sha256, and parameters."
            )
        target: str = _identifier(specification["callable"])
        _identifier(specification["version"])
        _mapping(specification["parameters"], f"{path}.parameters")
        if target.count(":") != 1:
            raise ValueError(f"{path}.callable: use module:function.")
        module_name, function_name = target.split(":", 1)
        if function_name.startswith("_") or "." in function_name:
            raise ValueError(f"{path}: adapter must name an explicit module-level function.")
        module = importlib.import_module(module_name)
        function: Any = getattr(module, function_name, None)
        if not callable(function):
            raise ValueError(f"{path}: configured adapter is not callable.")
        source: str | None = inspect.getsourcefile(function)
        if source is None or not Path(source).is_file():
            raise ValueError(f"{path}: adapter source cannot be verified.")
        digest: str = hashlib.sha256(Path(source).read_bytes()).hexdigest()
        if digest != specification["source_sha256"]:
            raise ValueError(f"{path}: adapter source differs from its declared digest.")
        return function

    def _invoke(self, declaration: Any, payload: dict[str, Any], path: str) -> Any:
        function: Callable[..., Any] = self._adapter(declaration, path)
        result: Any = function(
            payload=copy.deepcopy(payload),
            parameters=copy.deepcopy(declaration["parameters"]),
        )
        return json.loads(_canonical_json(result))

    def _immutable(self, kind: str, record_id: str) -> None:
        records: list[dict[str, Any]] = self._store.list_records(
            kind, {"record_id": record_id, "lifecycle": ["immutable", "frozen"]}
        )
        if len(records) != 1:
            raise ValueError(f"{kind}/{record_id}: immutable prerequisite required.")

    def _instances(self, identifiers: list[str]) -> list[TaskInstance]:
        ids: list[str] = _ids(identifiers, "instance_ids")
        manifests: list[dict[str, Any]] = [
            record for record in self._store.list_records("corpus_manifest")
            if record.get("corpus_version") == self._config.corpus_version
        ]
        if len(manifests) != 1:
            raise ValueError("Exactly one frozen corpus manifest is required.")
        manifest: dict[str, Any] = manifests[0]
        self._immutable("corpus_manifest", manifest["manifest_id"])
        members: dict[str, str] = {
            item["instance_id"]: item["content_hash"] for item in manifest["members"]
        }
        if not set(ids).issubset(members):
            raise ValueError("Requested task instances are outside frozen corpus membership.")
        result: list[TaskInstance] = []
        for identity in ids:
            instance = TaskInstance.model_validate(
                self._store.read_record("task_instance", identity),
                context={"require_content_hash": True},
            )
            self._immutable("task_instance", identity)
            if (
                instance.content_hash != members[identity]
                or instance.transformation_log.get("corpus_version")
                != self._config.corpus_version
            ):
                raise ValueError("Task content does not match its frozen corpus.")
            if not instance.reference.get("version") or not instance.reference.get("answer"):
                raise ValueError("Versioned reference answers are required before evaluation.")
            result.append(instance)
        return result

    def _responses(self, identifiers: list[str]) -> list[tuple[ModelResponse, TaskInstance]]:
        result: list[tuple[ModelResponse, TaskInstance]] = []
        for identity in _ids(identifiers, "response_ids"):
            response = ModelResponse.model_validate(
                self._store.read_record("model_response", identity)
            )
            self._immutable("model_response", identity)
            if response.config_hash != self._config.config_hash:
                raise ValueError("Response belongs to a different configuration.")
            instance = TaskInstance.model_validate(
                self._store.read_record("task_instance", response.instance_id)
            )
            if instance.transformation_log.get("corpus_version") != self._config.corpus_version:
                raise ValueError("Response population belongs to a different corpus.")
            result.append((response, instance))
        return result

    def _taxonomy(self) -> TaxonomyVersion:
        protocol: Any = self._config.categorization.incomplete_agent_output_policy
        taxonomy_id: Any = None
        if isinstance(protocol, dict):
            taxonomy_id = protocol.get("annotation_protocol", {}).get("taxonomy_id")
        if not isinstance(taxonomy_id, str):
            raise ValueError(
                "categorization.incomplete_agent_output_policy.annotation_protocol."
                "taxonomy_id: a frozen taxonomy binding is required."
            )
        found: list[tuple[str, dict[str, Any]]] = []
        for kind in _TAXONOMY_KINDS:
            try:
                found.append((kind, self._store.read_record(kind, taxonomy_id)))
            except KeyError:
                pass
        if len(found) != 1:
            raise ValueError("Frozen taxonomy identity is missing or ambiguous.")
        kind, payload = found[0]
        taxonomy = TaxonomyVersion.model_validate(payload)
        if not taxonomy.frozen:
            raise ValueError("Optional categorization experiments require a frozen taxonomy.")
        self._immutable(kind, taxonomy_id)
        return taxonomy

    def _checkpoint(
        self,
        stage: str,
        manifest: dict[str, Any],
        work: list[str],
        completed: dict[str, Any],
        result: dict[str, Any],
        *,
        pending_annotations: list[str] | None = None,
    ) -> None:
        execution_id: str = manifest["manifest_id"]
        prior: list[StageState] = [
            StageState.model_validate(item)
            for item in self._store.list_records("stage_state")
            if item.get("stage") == stage and item.get("execution_id") == execution_id
        ]
        starts: list[datetime] = [
            state.started_at for state in prior if state.started_at is not None
        ]
        timestamp: datetime = _now()
        status: StageStatus = StageStatus(result["status"])
        pending: list[str] = sorted(set(work) - set(completed))
        state = StageState(
            stage=stage,
            execution_id=execution_id,
            config_hash=self._config.config_hash,
            status=status,
            completed_work=sorted(completed),
            pending_work=[] if status == StageStatus.COMPLETED else pending,
            outstanding_annotation_task_ids=sorted(set(pending_annotations or [])),
            cursor={"completed": completed, "result": result},
            input_manifests={"experiment": execution_id},
            output_manifests={},
            started_at=min(starts) if starts else timestamp,
            updated_at=timestamp,
            completed_at=timestamp if status == StageStatus.COMPLETED else None,
            errors=[],
        )
        self._store.set_stage_state(stage, state.model_dump(mode="json"))

    def _saved(self, stage: str, manifest_id: str) -> dict[str, Any]:
        states: list[StageState] = [
            StageState.model_validate(item)
            for item in self._store.list_records("stage_state")
            if item.get("stage") == stage
            and item.get("execution_id") == manifest_id
            and item.get("config_hash") == self._config.config_hash
        ]
        if not states:
            return {}
        states.sort(key=lambda state: state.updated_at or state.started_at or _now())
        return copy.deepcopy(states[-1].cursor)

    def _task_manifests(self) -> tuple[dict[str, Any], dict[str, Any]]:
        guidance: dict[str, Any] = self._load_manifest(
            self._config.evaluation.prompt.task_guidance_manifest,
            "evaluation.prompt.task_guidance_manifest",
        )
        profiles: dict[str, Any] = self._load_manifest(
            self._config.evaluation.prompt.output_profile_manifest,
            "evaluation.prompt.output_profile_manifest",
        )
        _mapping(guidance.get("tasks"), "task guidance tasks")
        _mapping(profiles.get("tasks"), "output profile tasks")
        return guidance, profiles

    def _task_payload(
        self,
        instance: TaskInstance,
        guidance: dict[str, Any],
        profiles: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from identifier_normalization import model_normalization_feeds

        task_guidance: dict[str, Any] = _mapping(
            guidance["tasks"].get(instance.task_name), "Task guidance"
        )
        profile: dict[str, Any] = _mapping(
            profiles["tasks"].get(instance.task_name), "Task output profile"
        )
        text: str = _identifier(task_guidance.get("guidance"))
        policy: Any = task_guidance.get("candidate_policy")
        if policy not in {"include", "exclude"}:
            raise ValueError("Task candidate visibility requires an explicit include/exclude policy.")
        output_profile: dict[str, Any] = _mapping(
            profile.get("output_profile"), "Resolved output contract"
        )
        if _canonical_json(output_profile) != _canonical_json(instance.output_profile):
            raise ValueError("Instance and configured task output contracts differ.")
        schema: dict[str, Any] = _mapping(
            profile.get("schema_contract"), "Task schema contract"
        )
        if schema.get("kind") != "task":
            raise ValueError("Main task generation requires a task schema contract.")
        _identifier(schema.get("version"))
        _mapping(schema.get("json_schema"), "Task JSON schema")
        documents: list[dict[str, Any]] = []
        metadata_fields: tuple[str, ...] = (
            "evidence_id", "document_id", "source_id", "source_url",
            "source_version", "snippet", "publication_time", "modification_time",
            "claim_valid_from", "claim_valid_until",
        )
        for evidence in instance.evidence_records:
            if evidence.role != EvidenceRole.MODEL_INPUT:
                raise ValueError("Reference-only or outcome evidence entered task inputs.")
            if evidence.publication_time is None:
                raise ValueError("Model-visible evidence lacks publication metadata.")
            publication = evidence.publication_time
            publication_date = publication.date() if isinstance(publication, datetime) else publication
            if publication_date > instance.snapshot_date:
                raise ValueError("Post-snapshot evidence entered task inputs.")
            if evidence.modification_time is not None:
                modification = evidence.modification_time
                modification_date = (
                    modification.date() if isinstance(modification, datetime) else modification
                )
                if modification_date > instance.snapshot_date:
                    raise ValueError("Post-snapshot source version entered task inputs.")
            raw: dict[str, Any] = evidence.model_dump(mode="json")
            documents.append({key: raw[key] for key in metadata_fields if key in raw})
        payload: dict[str, Any] = {
            "case_id": instance.case_id,
            "task_name": instance.task_name,
            "cti_stage": instance.stage.value,
            "snapshot_date": instance.snapshot_date.isoformat(),
            "source_documents": documents,
            "structured_feeds": model_normalization_feeds(instance),
            "candidates": instance.candidates if policy == "include" else [],
            "candidate_policy": policy,
            "environment": instance.environment,
            "task_guidance": text,
            "output_profile": output_profile,
            "temperature": self._config.inference.temperature,
            "top_p": self._config.inference.top_p,
            "prompt_settings": self._config.evaluation.prompt.model_dump(mode="json"),
        }
        _reject_internal_fields(payload)
        return payload, schema

    def _tag_response(
        self,
        response: ModelResponse,
        context: dict[str, Any],
        generation_context_id: str,
    ) -> ModelResponse:
        """Append a metadata projection without editing raw model output or cache."""
        payload: dict[str, Any] = response.model_dump(mode="json")
        usage: dict[str, Any] = copy.deepcopy(payload["usage"])
        request_hash: Any = usage.pop("request_hash", None)
        usage["source_request_hash"] = request_hash or usage.get("source_request_hash")
        usage["source_response_id"] = response.response_id
        usage["evaluation_context"] = copy.deepcopy(context)
        usage["generation_context_id"] = generation_context_id
        payload["usage"] = usage
        payload["response_id"] = _content_hash(
            {
                "source_response_id": response.response_id,
                "evaluation_context": context,
                "config_hash": self._config.config_hash,
                "generation_context_id": generation_context_id,
            }
        )
        tagged = ModelResponse.model_validate(payload)
        self._store.create_record(
            "model_response", tagged.response_id, tagged.model_dump(mode="json")
        )
        return tagged

    def _generate_task(
        self,
        instance: TaskInstance,
        model_id: str,
        run_index: int,
        payload: dict[str, Any],
        schema: dict[str, Any],
        context: dict[str, Any],
        *,
        template: str | None = None,
        probability_interface: str | None = None,
    ) -> ModelResponse:
        _reject_internal_fields(payload)
        prompt_path: str = _identifier(
            template or self._config.evaluation.prompt.template_path
        )
        prompt: str = self._models.render_prompt(prompt_path, payload)
        # Provenance is part of ModelClient's binding/request hash. Conditions
        # remain distinct without modifying task wording to defeat the cache.
        path: Path = Path(prompt_path)
        path = path if path.is_absolute() else self._base / path
        provenance: dict[str, Any] = {
            "template_path": prompt_path,
            "template_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
            "payload_hash": _content_hash(payload),
            "experiment_context": context,
        }
        context_id = save_generation_context(
            self._store, instance, payload, prompt, provenance, self._config.config_hash
        )
        requests: list[dict[str, Any]] = [{
            "model_id": model_id,
            "prompt": prompt,
            "run_index": run_index,
            "schema_name": "cti_task_evaluation",
            "instance_id": instance.instance_id,
            "schema_contract": schema,
            "template_provenance": provenance,
            "probability_interface": probability_interface,
        }]
        outputs: list[ModelResponse] = self._models.generate_batch(requests)
        if len(outputs) != 1:
            raise RuntimeError("Model client returned an incomplete task request.")
        return self._tag_response(outputs[0], context, context_id)

    def run_task_evaluation(self, instances: list[TaskInstance]) -> list[str]:
        """Run original inference over frozen instances, models, and five runs."""
        self._config.validate("evaluate")
        if not isinstance(instances, list) or not instances:
            raise ValueError("Main evaluation requires a nonempty instance list.")
        supplied_ids: list[str] = _ids(
            [instance.instance_id for instance in instances], "Task-instance population"
        )
        frozen: list[TaskInstance] = self._instances(supplied_ids)
        supplied: dict[str, TaskInstance] = {
            instance.instance_id: instance for instance in instances
        }
        for instance in frozen:
            if _canonical_json(instance) != _canonical_json(supplied[instance.instance_id]):
                raise ValueError("Supplied instance differs from its frozen stored record.")
        model_ids: list[str] = (
            self._config.operational.selected_models
            or list(self._config.models.evaluation)
        )
        self._models.validate_endpoints(model_ids)
        guidance, profiles = self._task_manifests()
        packets: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
            instance.instance_id: self._task_payload(instance, guidance, profiles)
            for instance in frozen
        }
        manifest = self._manifest(
            "main_task_evaluation",
            {
                "instance_ids": supplied_ids,
                "instance_hashes": {
                    instance.instance_id: instance.content_hash for instance in frozen
                },
                "models": {
                    identity: self._config.models.evaluation[identity].model_dump(mode="json")
                    for identity in model_ids
                },
                "runs": self._config.evaluation.runs,
                "guidance_hash": _content_hash(guidance),
                "output_profiles_hash": _content_hash(profiles),
                "scoring_hash": _content_hash(self._config.evaluation.scoring),
                "implementation_choices": _json_primitive(self._config.implementation_choices),
            },
        )
        work: dict[str, tuple[TaskInstance, str, int]] = {}
        for instance in frozen:
            for model_id in model_ids:
                for run_index in range(self._config.evaluation.runs):
                    identity: str = _content_hash(
                        [instance.instance_id, model_id, run_index]
                    )
                    work[identity] = (instance, model_id, run_index)
        saved: dict[str, Any] = self._saved("evaluate", manifest["manifest_id"])
        completed: dict[str, str] = saved.get("completed", {})
        if saved.get("result", {}).get("status") == "completed":
            return [completed[key] for key in sorted(work)]
        for identity in sorted(work):
            if identity in completed:
                self._store.read_record("model_response", completed[identity])
                continue
            instance, model_id, run_index = work[identity]
            payload, schema = packets[instance.instance_id]
            response: ModelResponse = self._generate_task(
                instance, model_id, run_index, payload, schema,
                {
                    "protocol": "main",
                    "condition": "original",
                    "experiment_manifest_id": manifest["manifest_id"],
                },
            )
            if response.status != ResponseStatus.TRANSPORT_FAILURE:
                completed[identity] = response.response_id
            result: dict[str, Any] = {
                "status": (
                    "completed" if len(completed) == len(work)
                    else "blocked" if response.status == ResponseStatus.TRANSPORT_FAILURE
                    else "running"
                ),
                "response_ids": [completed[key] for key in sorted(completed)],
                "transport_failure_response_id": (
                    response.response_id
                    if response.status == ResponseStatus.TRANSPORT_FAILURE else None
                ),
            }
            self._checkpoint("evaluate", manifest, sorted(work), completed, result)
            if response.status == ResponseStatus.TRANSPORT_FAILURE:
                raise RuntimeError(
                    "Main evaluation paused after an exhausted/nonretryable transport "
                    "failure. Its artifact was retained; no successful job was recorded."
                )
        return [completed[key] for key in sorted(work)]

    def _baseline_text(
        self,
        response: ModelResponse,
        instance: TaskInstance,
        representation: Any,
    ) -> str:
        policy: dict[str, Any] = _mapping(representation, "Baseline input representation")
        parts: Any = policy.get("parts")
        if not isinstance(parts, list) or not parts:
            raise ValueError("Baseline representation requires explicit ordered parts.")
        separator: Any = policy.get("separator")
        if not isinstance(separator, str):
            raise ValueError("Baseline representation.separator must be explicit.")
        sources: dict[str, Any] = {
            "response": response.model_dump(mode="json"),
            "instance": instance.model_dump(mode="json"),
        }
        text: list[str] = []
        for declaration in parts:
            part: dict[str, Any] = _mapping(declaration, "Representation part")
            if set(part) != {"source", "pointer", "serialization"}:
                raise ValueError("Representation parts need source/pointer/serialization.")
            if part["source"] not in sources:
                raise ValueError("Unknown baseline representation source.")
            value: Any = _get_pointer(sources[part["source"]], part["pointer"])
            if part["serialization"] == "text":
                if not isinstance(value, str):
                    raise ValueError("Text representation selected non-text data.")
                text.append(value)
            elif part["serialization"] == "canonical_json":
                text.append(_canonical_json(value))
            else:
                raise ValueError("Unsupported baseline serialization.")
        return separator.join(text)

    def _mapping_labels(
        self, declaration: Any, taxonomy: TaxonomyVersion
    ) -> dict[str, list[str]]:
        policy: dict[str, Any] = _mapping(declaration, "Cluster/topic mapping")
        if policy.get("frozen_before_evaluation") is not True:
            raise ValueError("Cluster/topic mappings must be frozen before evaluation.")
        if policy.get("uses_held_out_gold_labels") is not False:
            raise ValueError("Held-out audit labels cannot optimize baseline mappings.")
        labels: dict[str, Any] = _mapping(policy.get("labels"), "Mapping labels")
        known: set[str] = {item.subtype_id for item in taxonomy.subtype_definitions}
        result: dict[str, list[str]] = {}
        for key, value in labels.items():
            assigned: list[str] = _ids(value, "Mapped labels", allow_empty=True)
            if not set(assigned).issubset(known):
                raise ValueError("Baseline mapping contains unknown subtype IDs.")
            result[str(key)] = assigned
        return result

    def _prevalence(
        self, predictions: dict[str, list[str]], taxonomy: TaxonomyVersion, *,
        identified_population: tuple[list[str], dict[str, str], str] | None = None,
    ) -> dict[str, Any]:
        policy: Any = self._config.baselines.prevalence_denominator
        if policy == "classified_responses":
            denominator: int = len(predictions)
            identification_manifest_id: str | None = None
        elif isinstance(policy, dict) and policy.get("mode") == "identified_responses":
            population, _, identification_manifest_id = (
                identified_population or self._identified_population(list(predictions))
            )
            if not set(predictions).issubset(population):
                raise ValueError("Baseline predictions are outside the identified denominator population.")
            denominator = len(population)
        elif isinstance(policy, dict) and policy.get("mode") == "explicit_population":
            population: list[str] = _ids(policy.get("response_ids"), "Prevalence population")
            if not set(predictions).issubset(population):
                raise ValueError("Predictions are outside the declared prevalence denominator.")
            for identity in population:
                self._store.read_record("model_response", identity)
            denominator = len(population)
            identification_manifest_id = None
        else:
            raise ValueError(
                "baselines.prevalence_denominator: use identified_responses, classified_responses, or "
                "an explicit complete population; no denominator is inferred."
            )
        counts: Counter[str] = Counter(
            label for labels in predictions.values() for label in set(labels)
        )
        return {
            "denominator": denominator,
            "denominator_policy": policy,
            "denominator_identification_manifest_id": identification_manifest_id,
            "counts": {
                item.subtype_id: counts[item.subtype_id]
                for item in taxonomy.subtype_definitions
            },
            "percentages": {
                item.subtype_id: 100 * counts[item.subtype_id] / denominator
                for item in taxonomy.subtype_definitions
            },
            "labels_may_overlap": True,
        }

    def _identified_population(
        self, response_ids: list[str],
    ) -> tuple[list[str], dict[str, str], str]:
        """Resolve one complete, frozen RQ1 cohort, including identified passes."""
        requested = set(_ids(response_ids, "Baseline classified response IDs"))
        candidates: dict[str, tuple[list[str], dict[str, str], str]] = {}
        for manifest in self._store.list_records("experiment_manifest"):
            if (
                manifest.get("purpose") != "identification_final"
                or manifest.get("config_hash") != self._config.config_hash
                or manifest.get("corpus_version") != self._config.corpus_version
            ):
                continue
            result = manifest.get("result", {})
            if result.get("status") != "completed" or not result.get("coverage", {}).get("complete"):
                continue
            population_id = result.get("population_manifest")
            if not isinstance(population_id, str):
                continue
            population_manifest = self._store.read_record("experiment_manifest", population_id)
            if (
                population_manifest.get("purpose") != "identification_population"
                or population_manifest.get("config_hash") != self._config.config_hash
                or population_manifest.get("corpus_version") != self._config.corpus_version
            ):
                continue
            population = _ids(
                [member["response_id"] for member in population_manifest.get("members", [])],
                "Completed identification population",
            )
            if not requested.issubset(population):
                continue
            bindings = _mapping(result.get("decision_ids_by_response"), "Completed identification decisions")
            if set(bindings) != set(population):
                raise ValueError("Completed identification does not bind every population decision.")
            failures: set[str] = set()
            for response_id, identity in bindings.items():
                decision = FailureDecision.model_validate(self._store.read_record("failure_decision", identity))
                self._immutable("failure_decision", identity)
                if decision.response_id != response_id or decision.requires_review or decision.decision.value == "unresolved":
                    raise ValueError("Baseline denominator contains an unresolved or misbound decision.")
                if decision.decision.value == "failure":
                    failures.add(response_id)
            if failures != set(result.get("failure_response_ids", [])):
                raise ValueError("Completed identification failure membership disagrees with its frozen decisions.")
            self._immutable("experiment_manifest", population_id)
            self._immutable("experiment_manifest", manifest["manifest_id"])
            self._responses(population)
            identity = _content_hash({"population": population, "decisions": bindings})
            candidates[identity] = (population, bindings, manifest["manifest_id"])
        if len(candidates) != 1:
            raise ValueError(
                "identified_responses requires exactly one same-configuration completed "
                "identification population with complete frozen pass/failure decisions."
            )
        return next(iter(candidates.values()))

    def _statistical_baseline(
        self,
        name: str,
        population: list[tuple[ModelResponse, TaskInstance]],
        taxonomy: TaxonomyVersion,
        run_index: int,
    ) -> dict[str, list[str]]:
        if name == "kmeans":
            settings = self._config.baselines.kmeans
            model: dict[str, Any] = _mapping(settings.embedding_model, "embedding_model")
            initialization: dict[str, Any] = _mapping(
                settings.initialization, "kmeans.initialization"
            )
            parameters: dict[str, Any] = _mapping(
                initialization.get("parameters"), "KMeans explicit parameters"
            )
            required: set[str] = {
                "init", "n_init", "max_iter", "tol", "verbose",
                "copy_x", "algorithm",
            }
            if set(parameters) != required:
                raise ValueError("KMeans parameters must explicitly resolve all documented controls.")
            seeds: list[Any] = initialization.get("run_seeds", [])
            if run_index >= len(seeds):
                raise ValueError("KMeans run seed is unresolved.")
            seed: int = _integer(seeds[run_index], "KMeans seed", minimum=0)
            mapping = self._mapping_labels(settings.cluster_to_subtype_mapping, taxonomy)
            clusters: int = _integer(settings.clusters, "KMeans clusters")
            if set(mapping) != {str(index) for index in range(clusters)}:
                raise ValueError("Cluster mapping must cover every configured cluster.")
            texts: list[str] = [
                self._baseline_text(response, instance, settings.input_representation)
                for response, instance in population
            ]
            embedded: dict[str, Any] = _mapping(
                self._invoke(
                    model.get("adapter"), {"texts": texts},
                    "baselines.kmeans.embedding_model.adapter",
                ),
                "Embedding result",
            )
            vectors: np.ndarray = np.asarray(embedded.get("vectors"), dtype=float)
            if (
                vectors.ndim != 2 or vectors.shape[0] != len(population)
                or vectors.shape[1] == 0 or not np.isfinite(vectors).all()
            ):
                raise ValueError("Embedding adapter returned invalid or misaligned vectors.")
            self._store.save_blob(
                _canonical_json(
                    {"texts_hash": _content_hash(texts), "vectors": vectors.tolist()}
                ).encode("utf-8")
            )
            fitted = KMeans(
                n_clusters=clusters, random_state=seed, **parameters
            ).fit_predict(vectors)
            return {
                response.response_id: mapping[str(int(cluster))]
                for (response, _), cluster in zip(population, fitted, strict=True)
            }

        settings = self._config.baselines.lda
        preprocessing: dict[str, Any] = _mapping(settings.preprocessing, "LDA preprocessing")
        vectorizer_parameters: dict[str, Any] = _mapping(
            preprocessing.get("vectorizer_parameters"), "LDA vectorizer parameters"
        )
        required_vectorizer: set[str] = {
            "input", "encoding", "decode_error", "strip_accents", "lowercase",
            "preprocessor", "tokenizer", "stop_words", "token_pattern",
            "ngram_range", "analyzer", "max_df", "min_df", "max_features",
            "vocabulary", "binary",
        }
        if set(vectorizer_parameters) != required_vectorizer:
            raise ValueError("LDA vectorizer controls must be explicitly specified.")
        if vectorizer_parameters["tokenizer"] is not None or vectorizer_parameters["preprocessor"] is not None:
            raise ValueError("Callable text preprocessing requires a separately versioned adapter.")
        vectorizer_parameters["ngram_range"] = tuple(vectorizer_parameters["ngram_range"])
        priors: dict[str, Any] = _mapping(settings.priors, "LDA priors/procedure")
        parameters: dict[str, Any] = _mapping(priors.get("parameters"), "LDA parameters")
        required_lda: set[str] = {
            "doc_topic_prior", "topic_word_prior", "learning_method",
            "learning_decay", "learning_offset", "batch_size", "evaluate_every",
            "total_samples", "perp_tol", "mean_change_tol", "max_doc_update_iter",
            "n_jobs", "verbose",
        }
        if set(parameters) != required_lda:
            raise ValueError("LDA controls must be explicitly specified.")
        seeds: Any = priors.get("run_seeds")
        if not isinstance(seeds, list) or run_index >= len(seeds):
            raise ValueError("LDA run seed is unresolved.")
        topics: int = _integer(settings.topic_count, "LDA topic_count")
        mapping = self._mapping_labels(settings.topic_to_subtype_mapping, taxonomy)
        if set(mapping) != {str(index) for index in range(topics)}:
            raise ValueError("Topic mapping must cover every configured topic.")
        assignment: dict[str, Any] = _mapping(
            priors.get("assignment_policy"), "LDA assignment policy"
        )
        if assignment.get("mode") != "threshold":
            raise ValueError("LDA requires an explicit topic-proportion threshold policy.")
        threshold: float = _number(assignment.get("threshold"), "LDA threshold", minimum=0)
        if threshold > 1:
            raise ValueError("LDA threshold must not exceed one.")
        texts = [
            self._baseline_text(response, instance, preprocessing.get("input_representation"))
            for response, instance in population
        ]
        matrix = CountVectorizer(**vectorizer_parameters).fit_transform(texts)
        proportions: np.ndarray = LatentDirichletAllocation(
            n_components=topics,
            max_iter=_integer(settings.iterations, "LDA iterations"),
            random_state=_integer(seeds[run_index], "LDA seed", minimum=0),
            **parameters,
        ).fit_transform(matrix)
        return {
            response.response_id: sorted({
                label
                for topic, probability in enumerate(row)
                if probability >= threshold
                for label in mapping[str(topic)]
            })
            for (response, _), row in zip(population, proportions, strict=True)
        }

    def _judge_baseline(
        self,
        name: str,
        population: list[tuple[ModelResponse, TaskInstance]],
        taxonomy: TaxonomyVersion,
        run_index: int,
    ) -> dict[str, list[str]]:
        settings = (
            self._config.baselines.gpt_judge
            if name == "gpt_judge" else self._config.baselines.gemini_judge
        )
        model_id: str = "G5" if name == "gpt_judge" else "GEM"
        template: str = _identifier(settings.prompt_template_path)
        self._models.validate_endpoints([model_id])
        requests: list[dict[str, Any]] = []
        known: list[str] = sorted(item.subtype_id for item in taxonomy.subtype_definitions)
        for response, instance in population:
            payload: dict[str, Any] = {
                "task_input": review_task_input(self._store, response, instance),
                "evaluated_response": response.model_dump(mode="json"),
                "reference_material": instance.reference,
                "frozen_taxonomy": taxonomy.model_dump(mode="json"),
            }
            prompt: str = self._models.render_prompt(template, payload)
            requests.append({
                "model_id": model_id,
                "prompt": prompt,
                "run_index": run_index,
                "schema_name": f"baseline_{name}",
                "instance_id": instance.instance_id,
                "schema_contract": {
                    "version": "baseline-multilabel-v1",
                    "kind": "classification",
                    "taxonomy_id": taxonomy.taxonomy_id,
                    "discovery_mode": False,
                    "allow_empty_labels_when_uncertain": True,
                    "json_schema": {
                        "type": "object",
                        "required": ["labels", "evidence_spans", "uncertain"],
                        "additionalProperties": False,
                        "properties": {
                            "labels": {
                                "type": "array", "uniqueItems": True,
                                "items": {"type": "string", "enum": known},
                            },
                            "evidence_spans": {"type": "array", "items": {"type": "object"}},
                            "uncertain": {"type": "boolean"},
                        },
                    },
                },
            })
        outputs: list[ModelResponse] = self._models.generate_batch(requests)
        result: dict[str, list[str]] = {}
        for (original, instance), output in zip(population, outputs, strict=True):
            if output.status != ResponseStatus.SUCCESS:
                raise RuntimeError("Baseline classification has an unresolved model exception.")
            parsed: dict[str, Any] = output.parsed_answer
            assignment = LabelAssignment.model_validate(
                {
                    "assignment_id": _content_hash(
                        [name, run_index, original.response_id, output.response_id]
                    ),
                    "response_id": original.response_id,
                    "taxonomy_id": taxonomy.taxonomy_id,
                    "actor_id": model_id,
                    "round_index": run_index,
                    "origin": AssignmentOrigin.AGENT,
                    "labels": parsed["labels"],
                    "evidence_spans": parsed["evidence_spans"],
                    "uncertain": parsed["uncertain"],
                },
                context={
                    "taxonomy": taxonomy,
                    "require_taxonomy": True,
                    "require_evidence_for_each_label": True,
                },
            )
            Categorizer(self._config, self._store, self._models, self._annotations)._validate_spans(
                assignment, {
                    "task_input": review_task_input(self._store, original, instance),
                    "evaluated_response": original.model_dump(mode="json"),
                    "reference_material": instance.reference,
                },
            )
            self._store.create_record(
                "label_assignment", assignment.assignment_id,
                assignment.model_dump(mode="json"),
            )
            if assignment.uncertain:
                raise RuntimeError(
                    "Baseline uncertainty needs an explicit finalization policy; "
                    "it is not silently counted as a supported empty label set."
                )
            result[original.response_id] = assignment.labels
        return result

    def _debate_baseline(
        self,
        population: list[tuple[ModelResponse, TaskInstance]],
        taxonomy: TaxonomyVersion,
        run_index: int,
    ) -> dict[str, list[str]]:
        settings = self._config.baselines.multi_agent_debate
        if settings.rounds != 2 or settings.stopping_rule != "fixed_rounds":
            raise ValueError("Implemented debate uses explicitly selected two fixed rounds.")
        policy: dict[str, Any] = _mapping(settings.consensus_policy, "Debate consensus policy")
        if policy.get("mode") != "label_set_plurality" or policy.get("tie_policy") != "block":
            raise ValueError("Debate must explicitly authorize label-set plurality with blocked ties.")
        categorizer = Categorizer(
            self._config, self._store, self._models, self._annotations,
            run_index=run_index,
        )
        predictions: dict[str, list[str]] = {}
        for response, _ in population:
            first = categorizer.run_round_one(response.response_id, taxonomy)
            second = categorizer.run_round_two(response.response_id, taxonomy, first)
            votes: Counter[tuple[str, ...]] = Counter(
                tuple(sorted(item.labels)) for item in second
            )
            maximum: int = max(votes.values())
            winners: list[tuple[str, ...]] = [
                labels for labels, count in votes.items() if count == maximum
            ]
            if len(winners) != 1 or any(item.uncertain for item in second):
                raise RuntimeError("Automated debate remains unresolved under its frozen policy.")
            predictions[response.response_id] = list(winners[0])
        return predictions

    def run_baseline(self, name: str, response_ids: list[str]) -> dict[str, Any]:
        """Run explicitly parameterized statistical or automated judge baselines."""
        if name != self._config.operational.selected_baseline:
            raise ValueError("Requested baseline differs from operational.selected_baseline.")
        self._config.validate("baselines")
        if name not in {"kmeans", "lda", "gpt_judge", "gemini_judge", "multi_agent_debate"}:
            raise ValueError("Unknown baseline.")
        requested_population = self._responses(response_ids)
        policy = self._config.baselines.prevalence_denominator
        designated: dict[str, str] = {}
        identified_population = None
        if isinstance(policy, dict) and policy.get("mode") == "identified_responses":
            identified_population = self._identified_population(response_ids)
            _, designated, _ = identified_population
        population = []
        excluded_passes: list[str] = []
        for response, instance in requested_population:
            if response.response_id in designated:
                identity = designated[response.response_id]
                decision = FailureDecision.model_validate(self._store.read_record("failure_decision", identity))
            else:
                decisions = [item for item in self._store.list_records(
                    "failure_decision", {"response_id": response.response_id}
                ) if item.get("decision") in {"pass", "failure"} and not item.get("requires_review", True)]
                if len(decisions) != 1:
                    raise ValueError("Every baseline input needs one finalized frozen identification decision.")
                decision = FailureDecision.model_validate(decisions[0])
                identity = _content_hash(decision)
            self._immutable("failure_decision", identity)
            if decision.response_id != response.response_id or decision.requires_review:
                raise ValueError("Baseline failure decision is misbound or unresolved.")
            if decision.decision.value == "failure":
                population.append((response, instance))
            else:
                excluded_passes.append(response.response_id)
        if not population:
            raise ValueError("Categorization baselines require a nonempty identified failure population.")
        taxonomy: TaxonomyVersion = self._taxonomy()
        repetitions: int = _integer(
            self._config.baselines.repetition_count, "baselines.repetition_count"
        )
        # Validate denominator before invoking any embedding or model adapter.
        self._prevalence(
            {response.response_id: [] for response, _ in population}, taxonomy,
            identified_population=identified_population,
        )
        manifest = self._manifest(
            "categorization_baseline",
            {
                "method": name,
                "response_ids": sorted(response.response_id for response, _ in population),
                "input_response_ids": sorted(response_ids),
                "excluded_pass_response_ids": sorted(excluded_passes),
                "taxonomy_id": taxonomy.taxonomy_id,
                "taxonomy_hash": _content_hash(taxonomy),
                "repetitions": repetitions,
                "baseline_settings": self._config.baselines.model_dump(mode="json"),
                "mapping_optimized_on_audit": False,
            },
        )
        work: list[str] = [str(index) for index in range(repetitions)]
        saved = self._saved(f"baseline:{name}", manifest["manifest_id"])
        completed: dict[str, Any] = saved.get("completed", {})
        for index in range(repetitions):
            if str(index) in completed:
                continue
            if name in {"kmeans", "lda"}:
                predictions = self._statistical_baseline(name, population, taxonomy, index)
            elif name in {"gpt_judge", "gemini_judge"}:
                predictions = self._judge_baseline(name, population, taxonomy, index)
            else:
                predictions = self._debate_baseline(population, taxonomy, index)
            artifact = self._manifest(
                "baseline_predictions",
                {
                    "parent_manifest_id": manifest["manifest_id"],
                    "method": name,
                    "run_index": index,
                    "taxonomy_id": taxonomy.taxonomy_id,
                    "labels_by_response": predictions,
                    "prevalence": self._prevalence(
                        predictions, taxonomy, identified_population=identified_population
                    ),
                },
            )
            completed[str(index)] = artifact["manifest_id"]
            result = {
                "status": "completed" if len(completed) == repetitions else "running",
                "method": name,
                "experiment_manifest_id": manifest["manifest_id"],
                "run_prediction_manifest_ids": dict(completed),
            }
            self._checkpoint(f"baseline:{name}", manifest, work, completed, result)
        runs: list[dict[str, Any]] = [
            self._store.read_record("experiment_manifest", completed[str(index)])
            for index in range(repetitions)
        ]
        ddof: Any = None
        initialization: Any = (
            self._config.baselines.kmeans.initialization if name == "kmeans"
            else self._config.baselines.lda.priors if name == "lda" else None
        )
        if isinstance(initialization, dict):
            ddof = initialization.get("summary_ddof")
        summary: dict[str, Any] = {}
        for subtype in taxonomy.subtype_definitions:
            values: list[float] = [
                run["prevalence"]["percentages"][subtype.subtype_id] for run in runs
            ]
            summary[subtype.subtype_id] = {
                "mean": float(np.mean(values)),
                "standard_deviation": (
                    float(np.std(values, ddof=_integer(ddof, "summary_ddof", minimum=0)))
                    if ddof is not None and int(ddof) < repetitions else None
                ),
                "standard_deviation_ddof": ddof,
            }
        return {
            "status": "completed",
            "method": name,
            "experiment_manifest_id": manifest["manifest_id"],
            "prediction_manifest_ids": [completed[str(index)] for index in range(repetitions)],
            "subtype_prevalence_summary": summary,
            "independent_audit_metrics_computed": False,
            "excluded_pass_response_ids": sorted(excluded_passes),
            "prevalence_denominator_policy": policy,
        }

    def _originating_population(self, failure_ids: list[str]) -> list[str]:
        failures: list[str] = _ids(failure_ids, "Failure population")
        candidates: list[list[str]] = []
        for manifest in self._store.list_records("experiment_manifest"):
            if manifest.get("purpose") != "identification_final":
                continue
            result: dict[str, Any] = manifest.get("result", {})
            if not set(failures).issubset(result.get("failure_response_ids", [])):
                continue
            population_id: Any = result.get("population_manifest")
            if not isinstance(population_id, str):
                continue
            population = self._store.read_record("experiment_manifest", population_id)
            members: list[str] = sorted(
                item["response_id"] for item in population.get("members", [])
            )
            if members:
                candidates.append(members)
        distinct: dict[str, list[str]] = {
            _canonical_json(item): item for item in candidates
        }
        if len(distinct) != 1:
            raise ValueError(
                "Cannot recover one full originating RQ1 response population. "
                "A fixed failure subset is insufficient for selection perturbations."
            )
        return next(iter(distinct.values()))

    def _condition(
        self,
        declaration: Any,
        response_ids: list[str],
        label: str,
        *, preserve_identification: bool = False,
    ) -> tuple[ExperimentConfig, list[str], dict[str, Any]]:
        specification: dict[str, Any] = _mapping(declaration, "Condition protocol")
        adapter: Any = _required(specification.get("preparation_adapter"), "preparation_adapter")
        prepared: dict[str, Any] = _mapping(
            self._invoke(
                adapter,
                {
                    "configuration": self._config.model_dump(mode="json"),
                    "response_ids": response_ids,
                    "condition": label,
                },
                "condition.preparation_adapter",
            ),
            "Prepared condition",
        )
        configuration: dict[str, Any] = _mapping(
            prepared.get("configuration"), "Effective condition configuration"
        )
        effective = ExperimentConfig.model_validate(configuration)
        # Preserve explicit relative-path semantics and artifact identities.
        effective._source_path = getattr(self._config, "_source_path", None)
        effective._reference_digests = copy.deepcopy(
            getattr(self._config, "_reference_digests", {})
        )
        requested: list[str] = _ids(prepared.get("response_ids"), "Condition response IDs")
        if requested != sorted(response_ids):
            raise ValueError("Condition preparation cannot silently change the matched population.")
        changes: Any = prepared.get("changes")
        if not isinstance(changes, list) or not changes:
            raise ValueError("An effective condition requires explicit configuration changes.")
        original = self._config.model_dump(mode="json")
        alternative = effective.model_dump(mode="json")
        declared_paths: set[str] = set()
        for change in changes:
            item: dict[str, Any] = _mapping(change, "Declared condition change")
            pointer: str = _identifier(item.get("pointer"))
            if _get_pointer(original, pointer) != item.get("before"):
                raise ValueError("Condition change has an incorrect original value.")
            if _get_pointer(alternative, pointer) != item.get("after"):
                raise ValueError("Condition change has an incorrect effective value.")
            declared_paths.add(pointer)
        reconstructed: dict[str, Any] = copy.deepcopy(original)
        for change in changes:
            _replace_pointer(reconstructed, change["pointer"], change["after"])
        if _canonical_json(reconstructed) != _canonical_json(alternative):
            raise ValueError("Effective configuration contains undeclared changes.")
        if effective.corpus_version != self._config.corpus_version:
            raise ValueError("Matched conditions cannot replace the corpus version.")
        # Existing lower-level components require response.config_hash equality.
        # Preserve original raw output in an explicitly derived metadata record.
        from sampling import SOURCE_IDENTITY_CONTRACT, source_response_mapping
        source_identities = source_response_mapping(self._store, response_ids)
        derived: list[str] = []
        parents_by_derived: dict[str, str] = {}
        sources_by_derived: dict[str, str] = {}
        for response, _ in self._responses(response_ids):
            payload: dict[str, Any] = response.model_dump(mode="json")
            payload["config_hash"] = effective.config_hash
            payload["usage"] = {
                **payload["usage"],
                "source_response_id": response.response_id,
                "source_config_hash": response.config_hash,
                "condition": label,
                "sampling_source_response_id": source_identities[response.response_id],
            }
            payload["usage"].pop("request_hash", None)
            payload["response_id"] = _content_hash(
                {
                    "source_response_id": response.response_id,
                    "effective_config_hash": effective.config_hash,
                    "condition": label,
                    "sampling_source_response_id": source_identities[response.response_id],
                    "sampling_identity_contract": SOURCE_IDENTITY_CONTRACT,
                }
            )
            projected = ModelResponse.model_validate(payload)
            self._store.create_record(
                "model_response", projected.response_id, projected.model_dump(mode="json")
            )
            if preserve_identification:
                source_id, source_decision = Categorizer(
                    self._config, self._store, self._models, self._annotations
                )._failure_decision(response.response_id)
                decision_payload = source_decision.model_dump(mode="json")
                decision_payload.update(
                    response_id=projected.response_id,
                    identification_version=_content_hash({
                        "method": "fixed_failure_population_projection",
                        "source_identification_version": source_decision.identification_version,
                        "effective_config_hash": effective.config_hash,
                        "condition": label,
                    }),
                    evidence_refs=[*source_decision.evidence_refs, source_id],
                )
                decision = FailureDecision.model_validate(decision_payload)
                decision_id = _content_hash(decision)
                self._store.create_record("failure_decision", decision_id, decision.model_dump(mode="json"))
                self._store.freeze_records("failure_decision", [decision_id])
            derived.append(projected.response_id)
            parents_by_derived[projected.response_id] = response.response_id
            sources_by_derived[projected.response_id] = source_identities[response.response_id]
        provenance = self._manifest(
            "effective_experimental_condition",
            {
                "condition": label,
                "effective_config_hash": effective.config_hash,
                "source_response_ids": sorted(response_ids),
                "derived_response_ids": sorted(derived),
                "sampling_identity_contract": SOURCE_IDENTITY_CONTRACT,
                "sampling_parent_response_ids_by_response": parents_by_derived,
                "sampling_source_response_ids_by_response": sources_by_derived,
                "changes": changes,
                "adapter_provenance": prepared.get("provenance"),
            },
        )
        return effective, sorted(derived), provenance

    def _categorization_ablation(
        self, component: str, response_ids: list[str], taxonomy: TaxonomyVersion,
        protocol: dict[str, Any], run_index: int,
    ) -> dict[str, Any]:
        """Execute explicitly declared RQ3 ablation arms without changing core invariants."""
        expected = {
            "human_initiation": "fixed_initial_taxonomy",
            "repetition": "one_round_with_review",
            "human_verification": "two_round_automated",
            "multi_agent_decision": "single_judge",
        }
        if component not in expected or protocol.get("strategy") != expected[component]:
            raise ValueError("Ablation needs an explicit supported component-replacement strategy.")
        if component == "multi_agent_decision":
            judge = protocol.get("judge")
            if judge not in {"gpt_judge", "gemini_judge"}:
                raise ValueError("single_judge ablation must explicitly select gpt_judge or gemini_judge.")
            reconsideration = protocol.get("reconsideration")
            if reconsideration not in ({"mode": "fresh_reassessment"}, {"mode": "self_review"}):
                raise ValueError("single_judge ablation requires explicit reconsideration mode.")
            categorizer = Categorizer(
                self._config, self._store, self._models, self._annotations,
                run_index=run_index,
                single_agent="GPT-5" if judge == "gpt_judge" else "Gemini 2.5",
                reconsideration=reconsideration,
            )
            current = categorizer.run(response_ids, taxonomy)
            result = {
                **current, "removed_component": component, "strategy": protocol["strategy"],
                "labels_by_response": {
                    identity: self._store.read_record("final_assignment", assignment_id)["labels"]
                    for identity, assignment_id in current["final_assignment_ids_by_response"].items()
                },
                "executed_mechanisms": categorizer._mechanisms(),
                "human_review_performed": current["workload"]["completed_human_adjudications"] > 0,
                "comparison_scope": "single_agent_two_round_selective_human_review",
            }
            self._manifest("categorization_ablation_results", result)
            return result
        if component == "human_verification" and protocol.get("consensus_policy") != {
            "mode": "label_set_plurality", "tie_policy": "block",
        }:
            raise ValueError("Automated ablation requires explicit label-set plurality and blocked ties.")
        categorizer = Categorizer(
            self._config, self._store, self._models, self._annotations, run_index=run_index
        )
        if component == "human_initiation":
            result = {
                **categorizer.run(response_ids, taxonomy),
                "removed_component": component,
                "strategy": protocol["strategy"],
                "taxonomy_discovery_executed": False,
                "comparison_scope": "categorization_with_explicit_fixed_taxonomy",
            }
            self._manifest("categorization_ablation_results", result)
            return result
        predictions: dict[str, list[str]] = {}
        bindings: dict[str, str] = {}
        pending: list[str] = []
        for response_id in _ids(response_ids, "Ablation response IDs"):
            first = categorizer.run_round_one(response_id, taxonomy)
            active = (
                categorizer.run_round_two(response_id, taxonomy, first)
                if component == "human_verification" else first
            )
            if any(item.uncertain or not item.labels for item in active):
                raise RuntimeError("Ablation has unresolved classifier outputs.")
            votes = Counter(tuple(sorted(item.labels)) for item in active)
            reference = None
            if component == "repetition" and len(votes) > 1:
                try:
                    reference = categorizer._human_reference(response_id, taxonomy)
                except LookupError:
                    tasks = categorizer._review_tasks(response_id, taxonomy)
                    pending.extend(task.annotation_task_id for task in tasks)
                    continue
                labels = reference["labels"]
                spans = reference["evidence_spans"]
                actors = {
                    self._store.read_record("adjudication", identity)["annotator_id"]
                    for identity in reference["annotation_ids"]
                }
                if len(actors) != 1:
                    raise ValueError("Ablation adjudicator identity is ambiguous.")
                actor_id = next(iter(actors))
                origin = AssignmentOrigin.OPERATIONAL_ADJUDICATION
            else:
                maximum = max(votes.values())
                winners = [label_set for label_set, count in votes.items() if count == maximum]
                if len(winners) != 1:
                    raise RuntimeError("Ablation remains tied under its explicit consensus policy.")
                labels = list(winners[0])
                spans = [span.model_dump(mode="json") for item in active
                         if tuple(sorted(item.labels)) == winners[0] for span in item.evidence_spans]
                actor_id = "ablation_automated_consensus"
                origin = AssignmentOrigin.CONSENSUS
            content = {
                "response_id": response_id, "taxonomy_id": taxonomy.taxonomy_id,
                "actor_id": actor_id, "round_index": 1 if component == "repetition" else 2,
                "labels": labels, "evidence_spans": spans, "uncertain": False,
                "origin": origin.value,
            }
            assignment = LabelAssignment.model_validate({
                "assignment_id": _content_hash({
                    "component": component, "protocol": protocol,
                    "config_hash": self._config.config_hash, "run_index": run_index,
                    "assignment": content, "human_reference": reference,
                    "source_assignments": [item.assignment_id for item in [*first, *active]],
                }), **content,
            }, context={
                "taxonomy": taxonomy, "require_taxonomy": True,
                "require_evidence_for_each_label": True,
            })
            categorizer._validate_spans(assignment, categorizer._packet(response_id, taxonomy))
            self._store.create_record("final_assignment", assignment.assignment_id,
                                      assignment.model_dump(mode="json"))
            self._store.freeze_records("final_assignment", [assignment.assignment_id])
            predictions[response_id] = assignment.labels
            bindings[response_id] = assignment.assignment_id
        result = {
            "status": StageStatus.WAITING_FOR_ANNOTATIONS.value if pending else "completed",
            "removed_component": component, "strategy": protocol["strategy"],
            "labels_by_response": predictions, "final_assignment_ids_by_response": bindings,
            "pending_annotation_task_ids": sorted(set(pending)),
            "rounds_executed": 1 if component == "repetition" else 2,
        }
        self._manifest("categorization_ablation_results", result)
        return result

    def run_ablation(self, component: str, response_ids: list[str]) -> dict[str, Any]:
        """Execute a fully declared replacement procedure in an isolated condition."""
        if component != self._config.operational.selected_ablation:
            raise ValueError("Ablation differs from operational.selected_ablation.")
        self._config.validate("ablations")
        if component not in self._config.ablations.components_to_remove:
            raise ValueError("Unknown ablation component.")
        replacements: dict[str, Any] = _mapping(
            self._config.ablations.replacement_behavior_by_component,
            "ablations.replacement_behavior_by_component",
        )
        protocol: dict[str, Any] = _mapping(replacements.get(component), "Ablation protocol")
        strategies = {
            "human_initiation": {"fixed_initial_taxonomy", "model_initialization"},
            "stratification": {"hard_rules"},
            "repetition": {"one_round_with_review"},
            "human_verification": {"two_round_automated"},
            "multi_agent_decision": {"single_judge"},
        }
        if component not in strategies or protocol.get("strategy") not in strategies[component]:
            raise ValueError(
                "Declare an executable ablation strategy. Supported replacements: hard_rules "
                "for stratification, one_round_with_review for repetition, two_round_automated "
                "for human_verification, single_judge for multi_agent_decision, and "
                "model_initialization for human initiation, or the separately labeled "
                "fixed_initial_taxonomy comparator."
            )
        repetitions: int = _integer(
            self._config.ablations.repetition_count, "ablations.repetition_count"
        )
        conditions: Any = protocol.get("conditions")
        if not isinstance(conditions, list) or len(conditions) != repetitions:
            raise ValueError("Ablation requires one explicitly declared condition per repetition.")
        population: list[str] = _ids(response_ids, "Ablation population")
        if component == "stratification":
            population = self._originating_population(population)
        manifest = self._manifest(
            "component_ablation",
            {"component": component, "response_ids": population, "protocol": protocol},
        )
        results: list[dict[str, Any]] = []
        affected = protocol.get("affected_stages")
        if not isinstance(affected, list) or not affected:
            raise ValueError("Ablation must declare its affected stages.")
        if component == "stratification" and "identify" not in affected:
            raise ValueError("Stratification ablation must rerun identification.")
        model_initialization = component == "human_initiation" and protocol["strategy"] == "model_initialization"
        if model_initialization and set(affected) != {"discover_taxonomy", "categorize"}:
            raise ValueError("Model-initialization ablation must rerun taxonomy discovery and categorization.")
        if component != "stratification" and not model_initialization and set(affected) != {"categorize"}:
            raise ValueError("RQ3 component ablations operate on a fixed identified population and taxonomy.")
        for index, declaration in enumerate(conditions):
            effective, derived, provenance = self._condition(
                declaration, population, f"ablation:{component}:{index}",
                preserve_identification="identify" not in affected,
            )
            if model_initialization and (
                effective.operational.taxonomy_mode != "discovery"
                or effective.taxonomy.discovery_initial_taxonomy is not None
            ):
                raise ValueError(
                    "Model-initialization ablation requires de novo discovery without a supplied fixed taxonomy."
                )
            models = ModelClient(effective, self._store)
            evaluator = Evaluator(effective, self._store)
            annotations = AnnotationService(self._store)
            affected: Any = protocol.get("affected_stages")
            if not isinstance(affected, list) or not affected:
                raise ValueError("Ablation must declare its affected stages.")
            current: dict[str, Any] = {}
            failures: list[str] = derived
            if "identify" in affected:
                if component == "stratification":
                    identifier = FailureIdentifier(effective, self._store, annotations)
                    for _, instance in ExperimentRunner(
                        effective, self._store, models, evaluator, annotations
                    )._responses(derived):
                        if identifier._route(instance.task_name) != "hard":
                            raise ValueError("hard_rules replacement cannot retain similarity stratification.")
                for response, instance in ExperimentRunner(
                    effective, self._store, models, evaluator, annotations
                )._responses(derived):
                    evaluator.score_response(instance, response)
                current = FailureIdentifier(effective, self._store, annotations).run(derived)
                if current["status"] != "completed":
                    return {
                        "status": current["status"],
                        "experiment_manifest_id": manifest["manifest_id"],
                        "condition_manifest_id": provenance["manifest_id"],
                        "condition_result": current,
                        "pending_annotation_task_ids": current.get("pending_annotation_task_ids", []),
                        "completed_conditions": results,
                    }
                if not current.get("coverage", {}).get("complete"):
                    raise ValueError("Ablation failure selection has unresolved population coverage.")
                failures = current["failure_response_ids"]
            taxonomy: TaxonomyVersion | None = None
            discovery_result: dict[str, Any] | None = None
            if "discover_taxonomy" in affected:
                current = TaxonomyManager(
                    effective, self._store, models, annotations,
                    **({"initialization": "model"} if model_initialization else {}),
                ).run(failures)
                if current["status"] != "completed":
                    return {
                        "status": current["status"],
                        "experiment_manifest_id": manifest["manifest_id"],
                        "condition_result": current,
                        "pending_annotation_task_ids": current.get("pending_annotation_task_ids", []),
                        "completed_conditions": results,
                    }
                for kind in _TAXONOMY_KINDS:
                    try:
                        taxonomy = TaxonomyVersion.model_validate(
                            self._store.read_record(kind, current["taxonomy_id"])
                        )
                        break
                    except KeyError:
                        pass
                discovery_result = copy.deepcopy(current)
            if "categorize" in affected:
                if taxonomy is None:
                    taxonomy = ExperimentRunner(
                        effective, self._store, models, evaluator, annotations
                    )._taxonomy()
                runner = ExperimentRunner(effective, self._store, models, evaluator, annotations)
                current = (
                    runner._categorization_ablation(component, failures, taxonomy, protocol, index)
                    if component != "stratification" and not model_initialization
                    else Categorizer(effective, self._store, models, annotations).run(failures, taxonomy)
                )
                if model_initialization:
                    current = {
                        **current, "removed_component": component, "strategy": "model_initialization",
                        "taxonomy_discovery_executed": True, "initialization": "model",
                        "taxonomy_discovery": discovery_result,
                        "comparison_scope": "model_initialized_discovery_and_standard_categorization",
                    }
            results.append({
                "run_index": index,
                "condition_manifest_id": provenance["manifest_id"],
                "result": current,
                "scoring_config_hash": _content_hash(effective.evaluation.scoring),
            })
            if current.get("status") not in {None, "completed"}:
                return {
                    "status": current["status"],
                    "experiment_manifest_id": manifest["manifest_id"],
                    "conditions": results,
                    "pending_annotation_task_ids": current.get("pending_annotation_task_ids", []),
                }
            del evaluator  # Scoring is not inferred from categorization prevalence.
        return {
            "status": "completed",
            "experiment_manifest_id": manifest["manifest_id"],
            "component": component,
            "conditions": results,
            "comparison_denominator": protocol.get("comparison_denominator"),
        }

    def _validate_stability_design(
        self, conditions: list[dict[str, Any]],
        configurations: dict[tuple[str, int], ExperimentConfig],
    ) -> None:
        """Check that the declared study actually changes its sampling variables."""
        settings = self._config.taxonomy_stability
        declared: set[str] = set()
        observed_widths: set[float] = set()
        names: set[str] = set()
        for condition in conditions:
            name = _identifier(condition.get("condition_id"))
            if name in names:
                raise ValueError("Stability condition IDs must be unique.")
            names.add(name)
            perturbations = condition.get("perturbations")
            if perturbations is None and condition.get("perturbation") is not None:
                perturbations = [condition["perturbation"]]
            selected = set(_ids(perturbations, "Stability condition perturbations"))
            if not selected.issubset(settings.perturbations):
                raise ValueError("Stability condition includes an undeclared perturbation.")
            declared.update(selected)
            initialization_seeds: list[int | None] = []
            anchor_seeds: list[int | None] = []
            signatures: set[tuple[Any, ...]] = set()
            for index in range(settings.independent_runs_per_condition):
                effective = configurations[(name, index)]
                if effective.taxonomy.guideline_manifest != self._config.taxonomy.guideline_manifest:
                    raise ValueError("Stability must retain the frozen annotation guideline.")
                width = effective.quantile_width
                if width not in settings.quantile_widths:
                    raise ValueError("Condition quantile width is outside the declared study.")
                initialization = effective.taxonomy_discovery.initialization_seed
                anchor = effective.failure_identification.stratification.anchor_sampling_seed
                if initialization is None or anchor is None:
                    raise ValueError("Independent stability runs require explicit initialization and anchor seeds.")
                signature = (width, initialization, anchor)
                if signature in signatures:
                    raise ValueError("Independent stability runs cannot repeat identical sampling seeds and width.")
                signatures.add(signature)
                initialization_seeds.append(initialization)
                anchor_seeds.append(anchor)
                observed_widths.add(width)
            if "initial_subset_seed" in selected and len(set(initialization_seeds)) != len(initialization_seeds):
                raise ValueError("Initialization perturbation requires independent initialization seeds.")
            if "anchor_selection" in selected and len(set(anchor_seeds)) != len(anchor_seeds):
                raise ValueError("Anchor-selection perturbation requires independent anchor-sampling seeds.")
        if declared != set(settings.perturbations):
            raise ValueError("Stability design must cover every declared perturbation family.")
        if observed_widths != set(settings.quantile_widths):
            raise ValueError("Stability design must evaluate every declared quantile width.")

    def _stability_audit_assignments(
        self, effective: ExperimentConfig, response_ids: list[str], taxonomy: TaxonomyVersion,
        provenance: dict[str, Any], run_index: int, *,
        models: ModelClient | None = None, annotations: AnnotationService | None = None,
    ) -> dict[str, Any]:
        """Apply the discovered taxonomy with RQ3, without consulting audit labels."""
        policy = _mapping(effective.categorization.incomplete_agent_output_policy, "Stability categorization policy")
        annotation_protocol = _mapping(policy.get("annotation_protocol"), "Stability operational annotation protocol")
        if "*" not in annotation_protocol.get("assignments", {}) or "*" not in (
            annotation_protocol.get("reference_applicability", {})
        ):
            raise ValueError(
                "Built-in stability assignments need explicit wildcard operational annotator "
                "assignments and reference applicability for condition-derived response IDs."
            )
        derived: dict[str, str] = {}
        source_classifier = Categorizer(self._config, self._store, self._models, self._annotations)
        for source, _ in self._responses(response_ids):
            payload = source.model_dump(mode="json")
            payload["config_hash"] = effective.config_hash
            payload["response_id"] = _content_hash({
                "purpose": "stability_audit_response_projection",
                "source_response_id": source.response_id,
                "condition_manifest_id": provenance["manifest_id"],
                "taxonomy_id": taxonomy.taxonomy_id,
                "effective_config_hash": effective.config_hash,
                "run_index": run_index,
            })
            payload["usage"] = {
                **payload["usage"], "source_response_id": source.response_id,
                "source_config_hash": source.config_hash,
                "condition_manifest_id": provenance["manifest_id"],
                "stability_taxonomy_id": taxonomy.taxonomy_id,
            }
            payload["usage"].pop("request_hash", None)
            response = ModelResponse.model_validate(payload)
            self._store.create_record("model_response", response.response_id, response.model_dump(mode="json"))
            source_id, source_decision = source_classifier._failure_decision(source.response_id)
            decision_payload = source_decision.model_dump(mode="json")
            decision_payload.update(
                response_id=response.response_id,
                identification_version=_content_hash({
                    "method": "fixed_stability_failure_population_projection",
                    "source_decision_id": source_id,
                    "condition_manifest_id": provenance["manifest_id"],
                    "taxonomy_id": taxonomy.taxonomy_id,
                }),
                evidence_refs=[*source_decision.evidence_refs, source_id],
            )
            decision = FailureDecision.model_validate(decision_payload)
            decision_id = _content_hash(decision)
            self._store.create_record("failure_decision", decision_id, decision.model_dump(mode="json"))
            self._store.freeze_records("failure_decision", [decision_id])
            derived[source.response_id] = response.response_id
        classifier = Categorizer(
            effective, self._store, models or ModelClient(effective, self._store),
            annotations or AnnotationService(self._store), run_index=run_index,
        )
        result = classifier.run(sorted(derived.values()), taxonomy)
        if result["status"] != "completed":
            return {"status": result["status"], "categorization": result,
                    "pending_annotation_task_ids": result.get("pending_annotation_task_ids", []),
                    "condition_response_ids_by_audit_response": derived}
        assignments: dict[str, str] = {}
        for source_id, condition_id in sorted(derived.items()):
            assignment_id = result["final_assignment_ids_by_response"][condition_id]
            assignment = LabelAssignment.model_validate(
                self._store.read_record("final_assignment", assignment_id),
                context={"taxonomy": taxonomy, "require_taxonomy": True,
                         "require_evidence_for_each_label": True},
            )
            payload = assignment.model_dump(mode="json")
            payload.update(response_id=source_id, supersedes_id=None, assignment_id=_content_hash({
                "purpose": "stability_audit_assignment_projection",
                "source_assignment_id": assignment_id, "audit_response_id": source_id,
                "condition_manifest_id": provenance["manifest_id"],
            }))
            projected = LabelAssignment.model_validate(payload, context={
                "taxonomy": taxonomy, "require_taxonomy": True,
                "require_evidence_for_each_label": True,
            })
            self._store.create_record("label_assignment", projected.assignment_id,
                                      projected.model_dump(mode="json"))
            self._store.freeze_records("label_assignment", [projected.assignment_id])
            self._manifest("stability_audit_assignment_projection", {
                "source_audit_response_id": source_id, "condition_response_id": condition_id,
                "source_assignment_id": assignment_id, "assignment_id": projected.assignment_id,
                "condition_manifest_id": provenance["manifest_id"],
                "effective_config_hash": effective.config_hash, "taxonomy_id": taxonomy.taxonomy_id,
                "audit_reference_labels_used": False,
            })
            assignments[source_id] = projected.assignment_id
        return {"status": "completed", "assignment_ids_by_response": assignments}

    def run_stability_study(
        self, failure_ids: list[str], cohort: AuditCohort
    ) -> dict[str, Any]:
        """Rerun full RQ1--RQ2 conditions, then compare fixed-audit assignments."""
        self._config.validate("stability")
        settings = self._config.taxonomy_stability
        if not cohort.locked or len(cohort.response_ids) != settings.audit_instances:
            raise ValueError("Stability requires the locked 180-instance audit cohort.")
        full_population: list[str] = self._originating_population(failure_ids)
        design: dict[str, Any] = _mapping(
            settings.condition_combination_design, "Stability condition design"
        )
        conditions: Any = design.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            raise ValueError("Stability condition design requires explicit conditions.")
        assignment_adapter: Any = _required(
            design.get("audit_assignment_adapter"), "Stability audit_assignment_adapter"
        )
        builtin_assignments = assignment_adapter == {"mode": "rq3"}
        if not builtin_assignments:
            self._adapter(assignment_adapter, "stability.audit_assignment_adapter")
        if set(settings.quantile_widths) != {0.025, 0.05, 0.1}:
            raise ValueError("Stability requires the three paper-specified quantile widths.")
        for condition in conditions:
            if not isinstance(condition, dict):
                raise ValueError("Each stability condition must be an object.")
            runs: Any = condition.get("runs")
            if not isinstance(runs, list) or len(runs) != settings.independent_runs_per_condition:
                raise ValueError("Every stability condition requires five explicitly specified runs.")
        manifest = self._manifest(
            "taxonomy_stability_experiment",
            {
                "full_response_ids": full_population,
                "starting_failure_ids": sorted(failure_ids),
                "audit_cohort_id": cohort.cohort_id,
                "audit_cohort_hash": _content_hash(cohort),
                "design": design,
            },
        )
        taxonomy_ids: list[str] = []
        assignments: dict[str, dict[str, str]] = {}
        run_provenance: dict[str, Any] = {}
        prepared_conditions: dict[tuple[str, int], tuple[ExperimentConfig, list[str], dict[str, Any]]] = {}
        for condition in conditions:
            name = _identifier(condition.get("condition_id"))
            for index, declaration in enumerate(condition["runs"]):
                prepared_conditions[(name, index)] = self._condition(
                    declaration, full_population, f"stability:{name}:{index}"
                )
        self._validate_stability_design(
            conditions, {key: prepared[0] for key, prepared in prepared_conditions.items()}
        )
        for condition in conditions:
            name: str = _identifier(condition.get("condition_id"))
            for index, declaration in enumerate(condition["runs"]):
                effective, derived, provenance = prepared_conditions[(name, index)]
                if effective.quantile_width not in settings.quantile_widths:
                    raise ValueError("Condition quantile width is outside the declared study.")
                models = ModelClient(effective, self._store)
                annotations = AnnotationService(self._store)
                evaluator = Evaluator(effective, self._store)
                for response, instance in ExperimentRunner(
                    effective, self._store, models, evaluator, annotations
                )._responses(derived):
                    evaluator.score_response(instance, response)
                identification = FailureIdentifier(
                    effective, self._store, annotations
                ).run(derived)
                if (
                    identification["status"] != "completed"
                    or not identification.get("coverage", {}).get("complete")
                ):
                    return {
                        "status": identification["status"] if identification["status"] != "completed" else "blocked",
                        "experiment_manifest_id": manifest["manifest_id"],
                        "condition_manifest_id": provenance["manifest_id"],
                        "condition": name,
                        "run_index": index,
                        "identification": identification,
                        "pending_annotation_task_ids": identification.get("pending_annotation_task_ids", []),
                    }
                discovery = TaxonomyManager(
                    effective, self._store, models, annotations
                ).run(identification["failure_response_ids"])
                if discovery["status"] != "completed":
                    return {
                        "status": discovery["status"],
                        "experiment_manifest_id": manifest["manifest_id"],
                        "condition": name,
                        "run_index": index,
                        "discovery": discovery,
                        "pending_annotation_task_ids": discovery.get("pending_annotation_task_ids", []),
                    }
                taxonomy_id: str = discovery["taxonomy_id"]
                if builtin_assignments:
                    found_taxonomies = []
                    for kind in _TAXONOMY_KINDS:
                        try:
                            found_taxonomies.append(TaxonomyVersion.model_validate(
                                self._store.read_record(kind, taxonomy_id)
                            ))
                        except KeyError:
                            pass
                    if len(found_taxonomies) != 1:
                        raise ValueError("Discovered taxonomy identity is missing or ambiguous.")
                    audit_result = self._stability_audit_assignments(
                        effective, cohort.response_ids, found_taxonomies[0], provenance, index,
                        models=models, annotations=annotations,
                    )
                    if audit_result["status"] != "completed":
                        return {
                            "status": audit_result["status"], "audit_assignment": audit_result,
                            "experiment_manifest_id": manifest["manifest_id"],
                            "condition": name, "run_index": index,
                            "pending_annotation_task_ids": audit_result.get("pending_annotation_task_ids", []),
                        }
                    assigned = audit_result["assignment_ids_by_response"]
                else:
                    assigned = self._invoke(
                        assignment_adapter,
                        {
                            "taxonomy_id": taxonomy_id,
                            "audit_response_ids": cohort.response_ids,
                            "condition_manifest_id": provenance["manifest_id"],
                            "effective_config_hash": effective.config_hash,
                        },
                        "stability.audit_assignment_adapter",
                    )
                binding: dict[str, Any] = _mapping(assigned, "Stability audit assignments")
                if set(binding) != set(cohort.response_ids):
                    raise ValueError("Every discovered taxonomy needs complete fixed-audit assignments.")
                for response_id, assignment_id in binding.items():
                    found: list[dict[str, Any]] = []
                    for kind in _ASSIGNMENT_KINDS:
                        try:
                            found.append(self._store.read_record(kind, assignment_id))
                        except KeyError:
                            pass
                    if len(found) != 1:
                        raise ValueError("Stability assignment identity is missing or ambiguous.")
                    if (
                        found[0]["response_id"] != response_id
                        or found[0]["taxonomy_id"] != taxonomy_id
                        or found[0]["uncertain"]
                    ):
                        raise ValueError("Stability assignment is misbound or unresolved.")
                taxonomy_ids.append(taxonomy_id)
                assignments[taxonomy_id] = binding
                run_provenance[taxonomy_id] = {
                    "condition": name,
                    "run_index": index,
                    "condition_manifest_id": provenance["manifest_id"],
                }
        # Audit membership stays fixed; a new append-only comparison cohort binds
        # the newly generated taxonomy-specific predictions.
        payload: dict[str, Any] = cohort.model_dump(mode="json")
        payload["sampling_protocol"] = {
            **payload["sampling_protocol"],
            "taxonomy_assignment_ids": assignments,
            "discovery_runs": run_provenance,
            "source_cohort_id": cohort.cohort_id,
        }
        payload["cohort_id"] = _content_hash(
            {"source_cohort": cohort.cohort_id, "experiment": manifest["manifest_id"],
             "assignments": assignments}
        )
        comparison_cohort = AuditCohort.model_validate(payload)
        self._store.create_record(
            "audit_cohort", comparison_cohort.cohort_id,
            comparison_cohort.model_dump(mode="json"),
        )
        self._store.freeze_records("audit_cohort", [comparison_cohort.cohort_id])
        measured = self._evaluator.score_taxonomy_stability(
            taxonomy_ids, comparison_cohort
        )
        return {
            "status": "completed",
            "experiment_manifest_id": manifest["manifest_id"],
            "taxonomy_ids": taxonomy_ids,
            "comparison_cohort_id": comparison_cohort.cohort_id,
            "metrics": measured,
        }

    def build_counterfactuals(self, instance: TaskInstance) -> list[dict[str, Any]]:
        """Apply exact declared model-visible edits without asserting preservation."""
        settings = self._config.defenses.counterfactual_consistency
        operators: dict[str, Any] = _mapping(
            settings.operators, "defenses.counterfactual_consistency.operators"
        )
        _identifier(operators.get("protocol_version"))
        count: int = _integer(
            settings.counterfactuals_per_instance, "counterfactuals_per_instance"
        )
        items: Any = operators.get("items")
        if not isinstance(items, list):
            raise ValueError("Counterfactual operators.items must be explicitly supplied.")
        guidance, profiles = self._task_manifests()
        original, _ = self._task_payload(instance, guidance, profiles)
        permitted: set[str] = {
            "source_documents", "structured_feeds", "environment",
            "snapshot_date", "candidates",
        }
        result: list[dict[str, Any]] = []
        for declaration in items:
            operator: dict[str, Any] = _mapping(declaration, "Counterfactual operator")
            applies: list[str] = _ids(operator.get("instance_ids"), "Operator instance_ids")
            if instance.instance_id not in applies:
                continue
            identity: str = _identifier(operator.get("operator_id"))
            family: str = _identifier(operator.get("family"))
            if family not in settings.intervention_families:
                raise ValueError("Counterfactual operator has an undeclared intervention family.")
            edits: Any = operator.get("edits")
            if not isinstance(edits, list) or not edits:
                raise ValueError("Counterfactual operators require exact nonempty edits.")
            modified: dict[str, Any] = copy.deepcopy(original)
            for raw_edit in edits:
                edit: dict[str, Any] = _mapping(raw_edit, "Counterfactual edit")
                if set(edit) != {"pointer", "before", "after"}:
                    raise ValueError("Counterfactual edits require pointer/before/after.")
                pointer: str = _identifier(edit["pointer"])
                if _pointer_tokens(pointer)[0] not in permitted:
                    raise ValueError("Counterfactual edits cannot change task identity, schema, or references.")
                before: Any = _get_pointer(modified, pointer)
                if _canonical_json(before) != _canonical_json(edit["before"]):
                    raise ValueError("Counterfactual edit does not match its declared before value.")
                if _canonical_json(before) == _canonical_json(edit["after"]):
                    raise ValueError("A counterfactual edit must actually change the input.")
                _replace_pointer(modified, pointer, edit["after"])
            _reject_internal_fields(modified)
            content: dict[str, Any] = {
                "instance_id": instance.instance_id,
                "original_content_hash": instance.content_hash,
                "operator_id": identity,
                "family": family,
                "edits": edits,
                "modified_model_input": modified,
                "original_model_input_hash": _content_hash(original),
                "modified_model_input_hash": _content_hash(modified),
                "preservation_justification": _identifier(
                    operator.get("preservation_justification")
                ),
                "preservation_evidence_refs": _ids(
                    operator.get("preservation_evidence_refs"), "Preservation evidence"
                ),
                "config_hash": self._config.config_hash,
                "preservation_status": "candidate",
            }
            variant_id: str = _content_hash(content)
            result.append({"variant_id": variant_id, **content})
        if len(result) != count or len({item["variant_id"] for item in result}) != count:
            raise ValueError(
                "Applicable, unique counterfactual operators do not match the "
                "explicit requested variant count."
            )
        return result

    def validate_preservation(
        self, instance: TaskInstance, variants: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Check externally supported answer preservation independently of inference."""
        settings = self._config.defenses.counterfactual_consistency
        if (
            not settings.preserve_answer_determining_attributes
            or not settings.require_task_dependent_preservation_justification
        ):
            raise ValueError("Task-dependent preservation requirements must remain enabled.")
        operators: dict[str, Any] = _mapping(settings.operators, "Counterfactual operators")
        protocol: dict[str, Any] = _mapping(
            operators.get("preservation_protocol"), "Preservation protocol"
        )
        if protocol.get("mode") != "manifest":
            raise ValueError(
                "Preservation requires an explicit independent decision manifest. "
                "Model consistency cannot certify answer preservation."
            )
        decisions: dict[str, Any] = _mapping(
            protocol.get("decisions"), "Preservation decisions"
        )
        result: list[dict[str, Any]] = []
        for variant in variants:
            candidate: dict[str, Any] = _mapping(variant, "Counterfactual variant")
            if (
                candidate.get("instance_id") != instance.instance_id
                or candidate.get("original_content_hash") != instance.content_hash
            ):
                raise ValueError("Variant is bound to a different original task.")
            decision: Any = decisions.get(candidate["operator_id"])
            if decision is None:
                validation: dict[str, Any] = {
                    "decision": "unresolved",
                    "justification": "No independent preservation decision was supplied.",
                }
            else:
                validation = _mapping(decision, "Independent preservation decision")
                bound_content = validation.get("original_content_hash") == instance.content_hash
                if validation.get("original_content_hash") == "$original_content_hash":
                    # Task hashes contain configuration lineage. Exact model-input
                    # and reference hashes avoid embedding a recursive task hash
                    # in that same configuration, while binding the reviewed pair.
                    guidance, profiles = self._task_manifests()
                    original_input, _ = self._task_payload(instance, guidance, profiles)
                    bound_content = (
                        validation.get("original_model_input_hash") == _content_hash(original_input)
                        and candidate.get("original_model_input_hash") == _content_hash(original_input)
                        and validation.get("reference_hash") == _content_hash(instance.reference)
                    )
                if validation.get("instance_id") != instance.instance_id or not bound_content:
                    raise ValueError("Preservation decision refers to different source content.")
                if (
                    candidate.get("modified_model_input_hash")
                    != _content_hash(candidate.get("modified_model_input"))
                    or validation.get("modified_model_input_hash")
                    != candidate.get("modified_model_input_hash")
                ):
                    raise ValueError(
                        "Preservation decision does not certify this exact modified model input."
                    )
                if validation.get("decision") not in {"admissible", "rejected", "unresolved"}:
                    raise ValueError("Unknown preservation decision.")
                _identifier(validation.get("reviewer_id"))
                _identifier(validation.get("justification"))
                _ids(validation.get("evidence_refs"), "Preservation decision evidence")
                if validation["decision"] == "admissible":
                    for flag in (
                        "answer_preserved", "causal_attributes_preserved", "chronology_valid"
                    ):
                        if validation.get(flag) is not True:
                            raise ValueError(
                                "Admissible variants require affirmative, independently "
                                "supported answer, mechanism, and chronology preservation."
                            )
            validated: dict[str, Any] = {
                **candidate,
                "preservation_status": validation["decision"],
                "preservation_validation": validation,
            }
            self._store.create_record(
                "counterfactual", validated["variant_id"], validated
            )
            result.append(validated)
        return result

    def _distribution(self, response: ModelResponse, extraction: dict[str, Any]) -> dict[str, float]:
        artifact: Any = response.usage.get("probability_artifact")
        if not isinstance(artifact, dict) or artifact.get("interface") != "token_logprobs":
            raise ValueError("A genuine retained token-probability artifact is required.")
        mapping: dict[str, Any] = _mapping(
            extraction.get("token_to_label"), "C3 token_to_label"
        )
        if len(set(mapping.values())) != len(mapping):
            raise ValueError("C3 token-to-candidate mapping must be one-to-one.")
        data: Any = artifact.get("data")
        content: Any = data.get("content") if isinstance(data, dict) else None
        if not isinstance(content, list) or len(content) != 1:
            raise ValueError("Only complete single-token candidate distributions are supported.")
        top: Any = content[0].get("top_logprobs")
        if not isinstance(top, list):
            raise ValueError("Provider did not retain a complete candidate token distribution.")
        probabilities: dict[str, float] = {}
        for item in top:
            if not isinstance(item, dict) or item.get("token") not in mapping:
                raise ValueError("Probability artifact includes an undeclared candidate token.")
            token: str = item["token"]
            label: str = _identifier(mapping[token])
            if label in probabilities:
                raise ValueError("Probability artifact contains duplicate candidate labels.")
            probabilities[label] = math.exp(_number(item.get("logprob"), "Token log probability"))
        if set(probabilities) != set(mapping.values()):
            raise ValueError("Probability artifact omits declared candidates.")
        tolerance: float = _number(
            extraction.get("sum_tolerance"), "Probability sum tolerance", minimum=0
        )
        if abs(sum(probabilities.values()) - 1.0) > tolerance:
            raise ValueError(
                "Candidate probability mass is incomplete. Truncated token lists "
                "are not renormalized into invented distributions."
            )
        return probabilities

    def _candidate_probabilities(
        self, response: ModelResponse, model_input: dict[str, Any],
        extraction: dict[str, Any],
    ) -> dict[str, float]:
        """Score the explicitly declared candidate space against the actual prompt."""
        if extraction["interface"] == "token_logprobs":
            return self._distribution(response, extraction)
        declaration: dict[str, Any] = _mapping(extraction.get("adapter"), "C3 scoring adapter")
        candidates: dict[str, Any] = _mapping(
            declaration.get("parameters", {}).get("candidates"), "C3 finite candidates"
        )
        if not candidates:
            raise ValueError("C3 finite candidate space cannot be empty.")
        prompt = self._models.render_prompt(
            _identifier(self._config.evaluation.prompt.template_path), model_input
        )
        output = _mapping(self._invoke(
            declaration,
            {
                "model_id": response.model_id,
                "endpoint_or_checkpoint": self._config.models.evaluation[response.model_id].endpoint_or_checkpoint,
                "revision": self._config.models.evaluation[response.model_id].revision,
                "message_mapping": "single_user_message",
                "generation_checkpoint_sha256": extraction.get("generation_checkpoint_sha256"),
                "instance_id": response.instance_id,
                "run_index": response.run_index,
                "prompt": prompt,
                "model_input": model_input,
                "experiment_context": response.usage.get("evaluation_context", {}),
            },
            "C3 scoring adapter",
        ), "C3 candidate scoring output")
        raw = _mapping(output.get("probabilities"), "C3 candidate probabilities")
        if set(raw) != set(candidates):
            raise ValueError("C3 probabilities must cover exactly the declared finite candidates.")
        probabilities = {
            _identifier(label): _number(probability, "Candidate probability", minimum=0)
            for label, probability in raw.items()
        }
        tolerance = _number(extraction.get("sum_tolerance"), "Probability sum tolerance", minimum=0)
        if any(value > 1 for value in probabilities.values()) or abs(
            sum(probabilities.values()) - 1.0
        ) > tolerance:
            raise ValueError("C3 candidate probabilities must form a normalized distribution.")
        provenance = _mapping(output.get("provenance"), "C3 scoring provenance")
        if not provenance:
            raise ValueError("C3 candidate scoring requires explicit provenance.")
        self._manifest("counterfactual_candidate_probabilities", {
            "response_id": response.response_id,
            "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "adapter": declaration,
            "probabilities": probabilities,
            "provenance": provenance,
        })
        return probabilities

    def _validate_corrected_answer(
        self, answer: Any, instance: TaskInstance, schema: dict[str, Any],
        selected_label: Any, extraction: dict[str, Any],
    ) -> dict[str, Any]:
        result = _mapping(answer, "C3 corrected task output")
        _ValidatedOutput.model_validate(
            {"content": result}, context={"json_schema": schema["json_schema"]}
        )
        for key, expected in (
            ("task", instance.task_name), ("case_id", instance.case_id),
            ("snapshot_date", instance.snapshot_date.isoformat()),
        ):
            if result.get(key) != expected:
                raise ValueError(f"Corrected task output identity mismatch: {key}.")
        if result.get("status") not in self._config.evaluation.prompt.allowed_statuses:
            raise ValueError("Corrected task output has an unauthorized answer status.")
        justification = result.get("justification")
        if not isinstance(justification, str) or len(justification.split()) > (
            self._config.evaluation.prompt.maximum_justification_words
        ):
            raise ValueError("Corrected task justification violates its configured limit.")
        pointer: Any = extraction.get("label_pointer")
        if pointer is None:
            rules = _mapping(
                self._config.evaluation.scoring.task_field_extraction_rules,
                "Task field extraction rules",
            )
            pointer = _mapping(rules.get(instance.task_name), "Task extraction").get("prediction")
        if _canonical_json(_get_pointer(result, _identifier(pointer))) != _canonical_json(selected_label):
            raise ValueError("Corrected task answer does not represent the selected candidate label.")
        return result

    def _complete_task_distribution(
        self, response: ModelResponse, model_input: dict[str, Any],
        instance: TaskInstance, schema: dict[str, Any], extraction: dict[str, Any],
    ) -> dict[str, Any]:
        """Retain an exact token population, then project its full task judgments."""
        declaration = _mapping(extraction.get("adapter"), "C3 generative adapter")
        prompt = self._models.render_prompt(
            _identifier(self._config.evaluation.prompt.template_path), model_input
        )
        model = self._config.models.evaluation[response.model_id]
        request = {
            "model_id": response.model_id,
            "endpoint_or_checkpoint": model.endpoint_or_checkpoint,
            "revision": model.revision,
            "message_mapping": "single_user_message",
            "generation_checkpoint_sha256": extraction.get("generation_checkpoint_sha256"),
            "instance_id": instance.instance_id,
            "run_index": response.run_index,
            "prompt": prompt,
            "model_input": model_input,
            "experiment_context": response.usage.get("evaluation_context", {}),
        }
        output = _mapping(
            self._invoke(declaration, request, "C3 generative adapter"),
            "C3 complete output distribution",
        )
        provenance = _mapping(output.get("provenance"), "C3 generative provenance")
        if not provenance:
            raise ValueError("C3 complete enumeration requires nonempty provenance.")
        status = output.get("status")
        if status not in {"complete", "blocked"}:
            raise ValueError("C3 generative enumeration must be explicitly complete or blocked.")
        if status == "complete" and provenance.get("complete") is not True:
            raise ValueError("Complete C3 enumeration requires affirmative complete provenance.")
        artifact_content: dict[str, Any] = {
            "response_id": response.response_id, "request": request,
            "prompt": prompt, "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "adapter": declaration, "backend_output": output,
            "probability_enumeration_status": status,
            "output_equivalence_rule": "task_field_exact",
            "label_pointer": extraction["label_pointer"],
        }
        if status == "blocked":
            if output.get("outcomes"):
                raise ValueError("Blocked exact enumeration cannot expose a partial distribution.")
            artifact = self._manifest("counterfactual_complete_output_probabilities", artifact_content)
            return {
                "status": "blocked", "manifest_id": artifact["manifest_id"],
                "reason": output.get("reason") or provenance.get("reason") or "Exact output enumeration is blocked.",
                "provenance": provenance, "prompt": prompt,
            }
        outcomes = output.get("outcomes")
        if not isinstance(outcomes, list) or not outcomes:
            raise ValueError("Complete C3 enumeration must retain a nonempty output population.")
        projected_masses: dict[str, list[float]] = defaultdict(list)
        representatives: dict[str, list[dict[str, Any]]] = defaultdict(list)
        projected: list[dict[str, Any]] = []
        masses: list[float] = []
        for index, item in enumerate(outcomes):
            outcome = _mapping(item, "C3 enumerated output")
            raw = outcome.get("raw_text")
            token_ids = outcome.get("token_ids")
            if not isinstance(raw, str) or not isinstance(outcome.get("truncated"), bool):
                raise ValueError("Enumerated outputs require exact raw text and a truncation flag.")
            if not isinstance(token_ids, list) or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in token_ids
            ):
                raise ValueError("Enumerated outputs require exact nonnegative token IDs.")
            probability = _number(outcome.get("probability"), "Output probability", minimum=0)
            if probability > 1:
                raise ValueError("Output probability exceeds one.")
            masses.append(probability)
            try:
                answer = _strict_object(raw)
                selected = _get_pointer(answer, extraction["label_pointer"])
                answer = self._validate_corrected_answer(answer, instance, schema, selected, extraction)
                label = _canonical_json(selected)
                valid = True
                representatives[label].append({
                    "answer": answer, "raw_text": raw, "probability": probability,
                    "token_ids": token_ids, "truncated": outcome["truncated"],
                    "outcome_index": index,
                })
            except (ValueError, TypeError, KeyError) as error:
                # Distinct raw failures never collapse into a convenient invalid
                # bucket. Equal invalid raw text is the same observable outcome.
                label = "invalid_raw:" + _canonical_json(raw)
                valid = False
                invalid_reason = str(error)
            projected_masses[label].append(probability)
            projected.append({
                "outcome_index": index, "task_judgment": label,
                "schema_and_task_valid": valid, "probability": probability,
                **({} if valid else {"validation_error": invalid_reason}),
            })
        tolerance = _number(extraction.get("sum_tolerance"), "Probability sum tolerance", minimum=0)
        total_mass = math.fsum(masses)
        distribution = {label: math.fsum(values) for label, values in projected_masses.items()}
        if abs(total_mass - 1.0) > tolerance:
            raise ValueError("Complete C3 output probabilities must sum to one; missing mass cannot be renormalized.")
        artifact = self._manifest("counterfactual_complete_output_probabilities", {
            **artifact_content, "projected_outcomes": projected,
            "task_distribution": dict(distribution), "total_probability_mass": total_mass,
        })
        return {
            "status": "complete", "distribution": dict(distribution),
            "representatives": dict(representatives), "manifest_id": artifact["manifest_id"],
            "provenance": provenance, "prompt": prompt,
        }

    def _run_generative_counterfactual_defense(
        self, instances: list[TaskInstance], extraction: dict[str, Any],
    ) -> dict[str, Any]:
        """Run C3 over complete autoregressive outputs without finite candidates."""
        settings = self._config.defenses.counterfactual_consistency
        if extraction.get("aggregation_space") != "task_outputs":
            raise ValueError("Generative C3 requires task_outputs aggregation.")
        _pointer_tokens(_identifier(extraction.get("label_pointer")))
        declaration = _mapping(extraction.get("adapter"), "C3 generative adapter")
        if declaration.get("callable") != "generative_probabilities:complete_output_distribution":
            raise ValueError("Generative C3 requires the pinned complete autoregressive backend.")
        self._adapter(declaration, "C3 generative adapter")
        if declaration["parameters"].get("max_new_tokens") != self._config.inference.max_output_tokens:
            raise ValueError("C3 max_new_tokens must equal inference.max_output_tokens.")
        _number(extraction.get("sum_tolerance"), "Probability sum tolerance", minimum=0)
        if settings.output_equivalence_rule != "task_field_exact":
            raise ValueError("Generative C3 output equivalence must select task_field_exact.")
        if settings.entropy_weight_formula != "exp_negative_entropy":
            raise ValueError("Only explicitly selected exp_negative_entropy weights are implemented.")
        if settings.tie_policy != "block":
            raise ValueError("C3 aggregation ties require the explicit block policy.")
        tau = _number(settings.stability_threshold_tau, "C3 tau", minimum=0)
        if tau > 1:
            raise ValueError("C3 tau must lie in [0,1].")
        repetitions = _integer(settings.repetition_count, "C3 repetitions")
        model_ids = self._config.operational.selected_models or list(self._config.models.evaluation)
        self._models.validate_endpoints(model_ids)
        guidance, profiles = self._task_manifests()
        variants_by_instance = {
            instance.instance_id: self.validate_preservation(instance, self.build_counterfactuals(instance))
            for instance in instances
        }
        operators = _mapping(settings.operators, "C3 operators")
        count_policy = operators.get("accepted_count_policy")
        if count_policy not in {"require_requested", "use_admissible"}:
            raise ValueError("C3 accepted_count_policy is unresolved.")
        unresolved = [
            item["variant_id"] for variants in variants_by_instance.values()
            for item in variants if item["preservation_status"] == "unresolved"
        ]
        if unresolved:
            return {"status": "blocked", "unresolved_preservation_variant_ids": unresolved,
                    "reason": "Independent preservation review is incomplete."}
        for variants in variants_by_instance.values():
            accepted = [item for item in variants if item["preservation_status"] == "admissible"]
            if not accepted:
                raise ValueError("C3 consistency is undefined without admissible variants.")
            if count_policy == "require_requested" and len(accepted) != settings.counterfactuals_per_instance:
                raise ValueError("Rejected variants violate the configured accepted-count policy.")
        manifest = self._manifest("counterfactual_consistency", {
            "instance_ids": sorted(instance.instance_id for instance in instances),
            "variant_ids": {key: [item["variant_id"] for item in values]
                            for key, values in variants_by_instance.items()},
            "settings": settings.model_dump(mode="json"),
            "correction_estimator": "counterfactual_only",
            "correction_estimator_source": "C3 equation (Appendix H.1)",
            "original_input_role": "consistency_target_and_baseline",
            "generative_weighting": {
                "origin": "implementation", "formula": "exp(-H(projected_task_distribution))",
                "projection": "canonical JSON at label_pointer; distinct invalid raw outputs retain mass",
            },
        })
        diagnostics: list[dict[str, Any]] = []
        original_ids: list[str] = []
        corrected_ids: list[str] = []

        def blocked(reason: str, **details: Any) -> dict[str, Any]:
            result = self._manifest("counterfactual_consistency_blocked", {
                "parent_manifest_id": manifest["manifest_id"], "status": "blocked",
                "reason": reason, "diagnostics": diagnostics,
                "original_response_ids": original_ids, "corrected_response_ids": corrected_ids,
                **details,
            })
            return {"status": "blocked", "reason": reason,
                    "experiment_manifest_id": manifest["manifest_id"],
                    "result_manifest_id": result["manifest_id"], **details}

        for instance in instances:
            original_payload, schema = self._task_payload(instance, guidance, profiles)
            accepted = [item for item in variants_by_instance[instance.instance_id]
                        if item["preservation_status"] == "admissible"]
            for model_id in model_ids:
                for run_index in range(repetitions):
                    context = {"protocol": "counterfactual_consistency", "condition": "original",
                               "experiment_manifest_id": manifest["manifest_id"]}
                    original = self._generate_task(
                        instance, model_id, run_index, original_payload, schema, context
                    )
                    if original.status != ResponseStatus.SUCCESS:
                        return blocked("Original C3 task output is invalid.", original_response_id=original.response_id)
                    try:
                        selected = _get_pointer(original.parsed_answer, extraction["label_pointer"])
                        self._validate_corrected_answer(original.parsed_answer, instance, schema, selected, extraction)
                        target = _canonical_json(selected)
                    except (ValueError, TypeError, KeyError) as error:
                        return blocked("Original C3 task output is invalid.", validation_error=str(error))
                    original_population = self._complete_task_distribution(
                        original, original_payload, instance, schema, extraction
                    )
                    if original_population["status"] == "blocked":
                        return blocked(original_population["reason"], probability_enumeration_status="blocked",
                                       blocked_probability_manifest_id=original_population["manifest_id"])
                    if original_population["distribution"].get(target, 0.0) <= 0:
                        return blocked("Original task judgment has no support in its exact model distribution.",
                                       original_response_id=original.response_id,
                                       probability_manifest_id=original_population["manifest_id"])
                    # Per-response probability provenance belongs in usage.
                    # Evaluation context defines matched aggregation groups and
                    # must remain equal across instances and repetitions.
                    original_payload_record = original.model_dump(mode="json")
                    original_payload_record["response_id"] = _content_hash({
                        "source_response_id": original.response_id,
                        "probability_manifest_id": original_population["manifest_id"],
                    })
                    original_payload_record["usage"] = {
                        **original_payload_record["usage"], "source_response_id": original.response_id,
                        "probability_artifact": {
                            "interface": "autoregressive_complete",
                            "manifest_id": original_population["manifest_id"],
                            "provenance": original_population["provenance"],
                            "rendered_prompt": original_population["prompt"],
                        },
                    }
                    original = ModelResponse.model_validate(original_payload_record)
                    self._store.create_record("model_response", original.response_id, original.model_dump(mode="json"))
                    original_ids.append(original.response_id)
                    self._evaluator.score_response(instance, original)
                    populations: list[dict[str, Any]] = []
                    variant_response_ids: list[str] = []
                    variant_generation_context_ids: list[str] = []
                    matches = 0
                    for variant in accepted:
                        variant_context = {**context, "condition": "counterfactual",
                                           "variant_id": variant["variant_id"]}
                        response = self._generate_task(
                            instance, model_id, run_index, variant["modified_model_input"], schema,
                            variant_context,
                        )
                        population = self._complete_task_distribution(
                            response, variant["modified_model_input"], instance, schema, extraction
                        )
                        if population["status"] == "blocked":
                            return blocked(population["reason"], probability_enumeration_status="blocked",
                                           blocked_probability_manifest_id=population["manifest_id"])
                        best = max(population["distribution"].values())
                        winners = [key for key, value in population["distribution"].items() if value == best]
                        if len(winners) != 1:
                            return blocked("Counterfactual task-distribution argmax is tied.")
                        matches += int(winners[0] == target)
                        population["response_id"] = response.response_id
                        populations.append(population)
                        variant_response_ids.append(response.response_id)
                        variant_generation_context_ids.append(response.usage["generation_context_id"])
                    omega = matches / len(populations)
                    distributions = [item["distribution"] for item in populations]
                    log_weights = [sum(p * math.log(p) for p in item.values() if p > 0)
                                   for item in distributions]
                    # A shared scale leaves the selected correction unchanged.
                    # Subtracting the maximum also avoids 0/0 on normalization.
                    log_scale = max(log_weights)
                    scaled_weights = [math.exp(value - log_scale) for value in log_weights]
                    if settings.normalize_weights:
                        normalizer = math.fsum(scaled_weights)
                        weights = [value / normalizer for value in scaled_weights]
                        applied_log_scale = None
                    else:
                        weights = [math.exp(value) for value in log_weights]
                        applied_log_scale = None
                        if any(value == 0 for value in weights):
                            weights = scaled_weights
                            applied_log_scale = log_scale
                    support = sorted(set().union(*(item.keys() for item in distributions)))
                    totals = {label: math.fsum(weight * item.get(label, 0.0)
                              for weight, item in zip(weights, distributions, strict=True))
                              for label in support}
                    best = max(totals.values())
                    winners = [label for label, value in totals.items() if value == best]
                    if len(winners) != 1:
                        return blocked("Weighted C3 task correction is tied.", weighted_task_totals=totals)
                    winner = winners[0]
                    supporting = [
                        (weight * outcome["probability"], population, outcome)
                        for weight, population in zip(weights, populations, strict=True)
                        for outcome in population["representatives"].get(winner, [])
                    ]
                    if not supporting:
                        return blocked("The maximum-weight C3 outcome is invalid task JSON.",
                                       selected_task_judgment=winner, weighted_task_totals=totals,
                                       probability_enumeration_status="complete")
                    _, source_population, representative = max(supporting, key=lambda item: item[0])
                    answer = self._validate_corrected_answer(
                        representative["answer"], instance, schema, json.loads(winner), extraction
                    )
                    probability_ids = [original_population["manifest_id"],
                                       *[item["manifest_id"] for item in populations
                                         if item["manifest_id"] != original_population["manifest_id"]]]
                    payload = original.model_dump(mode="json")
                    payload["response_id"] = _content_hash([
                        manifest["manifest_id"], instance.instance_id, model_id, run_index,
                        representative["raw_text"], probability_ids,
                    ])
                    payload["raw_text"] = representative["raw_text"]
                    payload["parsed_answer"] = answer
                    payload["usage"] = {
                        "source_response_id": original.response_id,
                        "generation_context_id": original.usage["generation_context_id"],
                        "output_derivation": {
                            "kind": "counterfactual_probability_aggregation",
                            "source_response_ids": variant_response_ids,
                            "source_generation_context_ids": variant_generation_context_ids,
                            "aggregation_probability_manifest_ids": [item["manifest_id"] for item in populations],
                        },
                        "algorithm": "weighted_counterfactual_probability_aggregation",
                        "correction_estimator": "counterfactual_only",
                        "aggregation_response_ids": variant_response_ids,
                        "aggregation_probability_manifest_ids": [item["manifest_id"] for item in populations],
                        "schema_contract": schema, "selected_task_judgment": winner,
                        "probability_interface": "autoregressive_complete",
                        "probability_manifest_ids": probability_ids,
                        "rendered_prompt": original_population["prompt"],
                        "probability_provenance": source_population["provenance"],
                        "representative_outcome": {
                            "probability_manifest_id": source_population["manifest_id"],
                            "response_id": source_population["response_id"],
                            "token_ids": representative["token_ids"],
                            "probability": representative["probability"],
                            "truncated": representative["truncated"],
                        },
                        "evaluation_context": {
                            **context, "condition": "corrected",
                        },
                    }
                    corrected = ModelResponse.model_validate(payload, context={
                        "allowed_answer_statuses": self._config.evaluation.prompt.allowed_statuses,
                    })
                    self._store.create_record("model_response", corrected.response_id, corrected.model_dump(mode="json"))
                    self._evaluator.score_response(instance, corrected)
                    corrected_ids.append(corrected.response_id)
                    diagnostics.append({
                        "instance_id": instance.instance_id, "model_id": model_id, "run_index": run_index,
                        "omega": omega, "vulnerable": omega < tau,
                        "original_response_id": original.response_id,
                        "variant_response_ids": variant_response_ids,
                        "corrected_response_id": corrected.response_id,
                        "correction_estimator": "counterfactual_only",
                        "aggregation_response_ids": variant_response_ids,
                        "aggregation_probability_manifest_ids": [item["manifest_id"] for item in populations],
                        "weights": weights, "log_weights": log_weights,
                        "weight_log_scale": applied_log_scale, "weighted_task_totals": totals,
                        "probability_manifest_ids": probability_ids,
                        "probability_enumeration_status": "complete",
                    })
        task_metrics = self._evaluator.aggregate_task_metrics([*original_ids, *corrected_ids])
        result = self._manifest("counterfactual_consistency_results", {
            "parent_manifest_id": manifest["manifest_id"], "diagnostics": diagnostics,
            "corrected_response_ids": corrected_ids, "original_response_ids": original_ids,
            "task_metrics": task_metrics, "probability_enumeration_status": "complete",
            "correction_estimator": "counterfactual_only",
            "consistency_is_correctness": False, "unique_failure_mechanism_inferred": False,
        })
        return {"status": "completed", "experiment_manifest_id": manifest["manifest_id"],
                "result_manifest_id": result["manifest_id"], "corrected_response_ids": corrected_ids,
                "original_response_ids": original_ids, "task_metrics": task_metrics,
                "correction_estimator": "counterfactual_only",
                "probability_enumeration_status": "complete", "metric_contributions_persisted": True,
                "response_failure_rates": "require_fixed_identification"}

    def run_counterfactual_defense(self, instance_ids: list[str]) -> dict[str, Any]:
        """Diagnose admissible sensitivity and apply explicit probability correction."""
        if self._config.operational.selected_defense != "counterfactual_consistency":
            raise ValueError("Counterfactual defense must be explicitly selected.")
        settings = self._config.defenses.counterfactual_consistency
        if settings.include_original_input is not False:
            raise ValueError(
                "C3 correction is counterfactual-only; include_original_input must be false. "
                "Original-input augmentation is unsupported."
            )
        self._config.validate("defenses")
        instances = self._instances(instance_ids)
        extraction: dict[str, Any] = _mapping(
            settings.probability_extraction, "C3 probability extraction"
        )
        probability_interface = extraction.get("interface")
        if probability_interface == "autoregressive_complete":
            return self._run_generative_counterfactual_defense(instances, extraction)
        if (probability_interface, extraction.get("aggregation_space")) not in {
            ("token_logprobs", "tokens"),
            ("candidate_log_likelihoods", "finite_candidates"),
        }:
            raise ValueError("C3 requires an implemented explicit candidate probability interface.")
        if probability_interface == "candidate_log_likelihoods":
            _identifier(extraction.get("label_pointer"))
            self._adapter(extraction.get("adapter"), "C3 scoring adapter")
        if settings.output_equivalence_rule != "candidate_label_exact":
            raise ValueError("C3 output equivalence must explicitly select candidate_label_exact.")
        if settings.entropy_weight_formula != "exp_negative_entropy":
            raise ValueError("Only explicitly selected exp_negative_entropy weights are implemented.")
        if settings.tie_policy != "block":
            raise ValueError("C3 aggregation ties require the explicit block policy.")
        adapter: Any = _required(
            extraction.get("candidate_output_adapter"), "C3 candidate_output_adapter"
        )
        self._adapter(adapter, "C3 candidate_output_adapter")
        tau: float = _number(settings.stability_threshold_tau, "C3 tau", minimum=0)
        if tau > 1:
            raise ValueError("C3 tau must lie in [0,1].")
        repetitions: int = _integer(settings.repetition_count, "C3 repetitions")
        model_ids: list[str] = (
            self._config.operational.selected_models or list(self._config.models.evaluation)
        )
        self._models.validate_endpoints(model_ids)
        for instance in instances:
            metric = self._config.evaluation.tasks[
                instance.stage.value.lower()
            ][instance.task_name].metric
            if metric == "bleu":
                raise ValueError(
                    "Free-text C3 probability aggregation is not implemented by the "
                    "available token-probability interface; no voting or summarizer substitute is used."
                )
        guidance, profiles = self._task_manifests()
        variants_by_instance: dict[str, list[dict[str, Any]]] = {}
        for instance in instances:
            variants = self.validate_preservation(instance, self.build_counterfactuals(instance))
            variants_by_instance[instance.instance_id] = variants
        operators: dict[str, Any] = _mapping(settings.operators, "C3 operators")
        count_policy: Any = operators.get("accepted_count_policy")
        if count_policy not in {"require_requested", "use_admissible"}:
            raise ValueError("C3 accepted_count_policy is unresolved.")
        unresolved: list[str] = [
            item["variant_id"] for variants in variants_by_instance.values()
            for item in variants if item["preservation_status"] == "unresolved"
        ]
        if unresolved:
            return {
                "status": "blocked",
                "unresolved_preservation_variant_ids": unresolved,
                "reason": "Independent preservation review is incomplete.",
            }
        for variants in variants_by_instance.values():
            accepted = [item for item in variants if item["preservation_status"] == "admissible"]
            if not accepted:
                raise ValueError("C3 consistency is undefined without admissible variants.")
            if count_policy == "require_requested" and len(accepted) != settings.counterfactuals_per_instance:
                raise ValueError("Rejected variants violate the configured accepted-count policy.")
        manifest = self._manifest(
            "counterfactual_consistency",
            {
                "instance_ids": sorted(instance_ids),
                "variant_ids": {
                    key: [item["variant_id"] for item in values]
                    for key, values in variants_by_instance.items()
                },
                "settings": settings.model_dump(mode="json"),
                "correction_estimator": "counterfactual_only",
                "correction_estimator_source": "C3 equation (Appendix H.1)",
                "original_input_role": "consistency_target_and_baseline",
            },
        )
        diagnostics: list[dict[str, Any]] = []
        original_ids: list[str] = []
        corrected_ids: list[str] = []
        for instance in instances:
            original_payload, schema = self._task_payload(instance, guidance, profiles)
            accepted = [
                item for item in variants_by_instance[instance.instance_id]
                if item["preservation_status"] == "admissible"
            ]
            for model_id in model_ids:
                for run_index in range(repetitions):
                    base_context: dict[str, Any] = {
                        "protocol": "counterfactual_consistency",
                        "condition": "original",
                        "experiment_manifest_id": manifest["manifest_id"],
                    }
                    original = self._generate_task(
                        instance, model_id, run_index, original_payload, schema,
                        base_context, probability_interface=(
                            "token_logprobs" if probability_interface == "token_logprobs" else None
                        ),
                    )
                    if original.status != ResponseStatus.SUCCESS:
                        raise RuntimeError("Original C3 prediction lacks a usable probability artifact.")
                    original_ids.append(original.response_id)
                    self._evaluator.score_response(instance, original)
                    original_distribution = self._candidate_probabilities(original, original_payload, extraction)
                    maximum = max(original_distribution.values())
                    original_winners = [
                        label for label, probability in original_distribution.items()
                        if probability == maximum
                    ]
                    if len(original_winners) != 1:
                        raise ValueError("Original C3 candidate prediction is tied.")
                    target: str = (
                        _identifier(_get_pointer(original.parsed_answer, extraction["label_pointer"]))
                        if probability_interface == "candidate_log_likelihoods" else original_winners[0]
                    )
                    if target not in original_distribution:
                        raise ValueError("Original task answer is outside the declared candidate space.")
                    distributions: list[dict[str, float]] = []
                    variant_response_ids: list[str] = []
                    variant_generation_context_ids: list[str] = []
                    matches: int = 0
                    for variant in accepted:
                        response = self._generate_task(
                            instance, model_id, run_index,
                            variant["modified_model_input"], schema,
                            {
                                **base_context,
                                "condition": "counterfactual",
                                "variant_id": variant["variant_id"],
                            },
                            probability_interface=(
                                "token_logprobs" if probability_interface == "token_logprobs" else None
                            ),
                        )
                        if response.status != ResponseStatus.SUCCESS:
                            raise RuntimeError("A counterfactual probability response is unusable.")
                        distribution = self._candidate_probabilities(
                            response, variant["modified_model_input"], extraction
                        )
                        best: float = max(distribution.values())
                        winners = [label for label, value in distribution.items() if value == best]
                        if len(winners) != 1:
                            raise ValueError("Counterfactual candidate prediction is tied.")
                        matches += int(winners[0] == target)
                        distributions.append(distribution)
                        variant_response_ids.append(response.response_id)
                        variant_generation_context_ids.append(response.usage["generation_context_id"])
                    omega: float = matches / len(distributions)
                    weights: list[float] = [
                        math.exp(sum(
                            probability * math.log(probability)
                            for probability in distribution.values() if probability > 0
                        ))
                        for distribution in distributions
                    ]
                    if settings.normalize_weights:
                        total: float = sum(weights)
                        weights = [weight / total for weight in weights]
                    totals: dict[str, float] = {
                        label: sum(
                            weight * distribution[label]
                            for weight, distribution in zip(weights, distributions, strict=True)
                        )
                        for label in original_distribution
                    }
                    best = max(totals.values())
                    winners = [label for label, value in totals.items() if value == best]
                    if len(winners) != 1:
                        raise ValueError("Weighted C3 correction is tied.")
                    answer: Any = self._invoke(
                        adapter,
                        {
                            "selected_label": winners[0],
                            "original_model_output": original.parsed_answer,
                            "output_profile": instance.output_profile,
                            "instance_id": instance.instance_id,
                        },
                        "C3 candidate_output_adapter",
                    )
                    answer = self._validate_corrected_answer(
                        answer, instance, schema, winners[0], extraction
                    )
                    # A correction is an algorithmic output, not a new model call.
                    # Preserve its provenance and use the same schema contract.
                    payload = original.model_dump(mode="json")
                    payload["response_id"] = _content_hash(
                        [manifest["manifest_id"], instance.instance_id, model_id, run_index, answer]
                    )
                    payload["raw_text"] = _canonical_json(answer)
                    payload["parsed_answer"] = answer
                    payload["usage"] = {
                        "source_response_id": original.response_id,
                        "generation_context_id": original.usage["generation_context_id"],
                        "output_derivation": {
                            "kind": "counterfactual_probability_aggregation",
                            "source_response_ids": variant_response_ids,
                            "source_generation_context_ids": variant_generation_context_ids,
                        },
                        "algorithm": "weighted_counterfactual_probability_aggregation",
                        "correction_estimator": "counterfactual_only",
                        "aggregation_response_ids": variant_response_ids,
                        "schema_contract": schema,
                        "selected_label": winners[0],
                        "probability_interface": probability_interface,
                        "evaluation_context": {
                            "protocol": "counterfactual_consistency",
                            "condition": "corrected",
                            "experiment_manifest_id": manifest["manifest_id"],
                        },
                    }
                    corrected = ModelResponse.model_validate(payload, context={
                        "allowed_answer_statuses": self._config.evaluation.prompt.allowed_statuses,
                    })
                    self._store.create_record(
                        "model_response", corrected.response_id, corrected.model_dump(mode="json")
                    )
                    self._evaluator.score_response(instance, corrected)
                    corrected_ids.append(corrected.response_id)
                    diagnostics.append({
                        "instance_id": instance.instance_id,
                        "model_id": model_id,
                        "run_index": run_index,
                        "omega": omega,
                        "vulnerable": omega < tau,
                        "original_response_id": original.response_id,
                        "variant_response_ids": variant_response_ids,
                        "corrected_response_id": corrected.response_id,
                        "correction_estimator": "counterfactual_only",
                        "aggregation_response_ids": variant_response_ids,
                        "weights": weights,
                        "weighted_candidate_totals": totals,
                    })
        task_metrics = self._evaluator.aggregate_task_metrics([*original_ids, *corrected_ids])
        result = self._manifest(
            "counterfactual_consistency_results",
            {
                "parent_manifest_id": manifest["manifest_id"],
                "diagnostics": diagnostics,
                "corrected_response_ids": corrected_ids,
                "original_response_ids": original_ids,
                "task_metrics": task_metrics,
                "correction_estimator": "counterfactual_only",
                "consistency_is_correctness": False,
                "unique_failure_mechanism_inferred": False,
            },
        )
        return {
            "status": "completed",
            "experiment_manifest_id": manifest["manifest_id"],
            "result_manifest_id": result["manifest_id"],
            "corrected_response_ids": corrected_ids,
            "original_response_ids": original_ids,
            "task_metrics": task_metrics,
            "correction_estimator": "counterfactual_only",
            "metric_contributions_persisted": True,
            "response_failure_rates": "require_fixed_identification",
        }

    def _retrieve(
        self, instance: TaskInstance, settings: Any
    ) -> list[dict[str, Any]]:
        declaration: dict[str, Any] = _mapping(settings, "Retrieval protocol")
        output: dict[str, Any] = _mapping(
            self._invoke(
                declaration.get("adapter"),
                {
                    "instance_id": instance.instance_id,
                    "snapshot_date": instance.snapshot_date.isoformat(),
                    "model_input_evidence": [
                        item.model_dump(mode="json") for item in instance.evidence_records
                    ],
                    "environment": instance.environment,
                    "corpus_version": self._config.corpus_version,
                    "task_name": instance.task_name,
                    "retrieval_configuration": copy.deepcopy(
                        declaration.get("retrieval_configuration", {
                            key: value for key, value in declaration.items() if key != "adapter"
                        })
                    ),
                },
                "retrieval.adapter",
            ),
            "Retrieval output",
        )
        evidence: Any = output.get("evidence")
        if not isinstance(evidence, list):
            raise ValueError("Retrieval output must supply an ordered evidence list.")
        for passage in evidence:
            if not isinstance(passage, dict) or passage.get("role") != "model_input":
                raise ValueError("Retrieval cannot introduce reference-only evidence.")
            for field in ("evidence_id", "document_id", "snippet", "publication_time"):
                _identifier(passage.get(field))
            if passage.get("snapshot_admission_verified") is not True:
                raise ValueError("Retrieved evidence lacks verified historical-version admission.")
            publication = datetime.fromisoformat(
                passage["publication_time"].replace("Z", "+00:00")
            ).date() if "T" in passage["publication_time"] else datetime.strptime(
                passage["publication_time"], "%Y-%m-%d"
            ).date()
            if publication > instance.snapshot_date:
                raise ValueError("Retrieved evidence was published after the task snapshot.")
        _reject_internal_fields(evidence)
        self._manifest(
            "retrieval_evidence_package",
            {
                "instance_id": instance.instance_id,
                "snapshot_date": instance.snapshot_date.isoformat(),
                "declaration": declaration,
                "result": output,
                "evidence_hash": _content_hash(evidence),
            },
        )
        return evidence

    def _identify_and_categorize(
        self, response_ids: list[str], *, require_subtypes: bool
    ) -> dict[str, Any]:
        for response, instance in self._responses(response_ids):
            self._evaluator.score_response(instance, response)
        identification = FailureIdentifier(
            self._config, self._store, self._annotations
        ).run(response_ids)
        if identification["status"] != "completed":
            return {"status": identification["status"], "identification": identification,
                    "pending_annotation_task_ids": identification.get("pending_annotation_task_ids", [])}
        if not identification.get("coverage", {}).get("complete"):
            return {
                "status": "blocked",
                "identification": identification,
                "reason": "Response-level failure coverage is incomplete.",
            }
        categorization: dict[str, Any] | None = None
        failures: list[str] = identification["failure_response_ids"]
        if require_subtypes and failures:
            categorization = Categorizer(
                self._config, self._store, self._models, self._annotations
            ).run(failures, self._taxonomy())
            if categorization["status"] != "completed":
                return {
                    "status": categorization["status"],
                    "identification": identification,
                    "categorization": categorization,
                    "pending_annotation_task_ids": categorization.get("pending_annotation_task_ids", []),
                }
        return {
            "status": "completed",
            "identification": identification,
            "categorization": categorization,
            "failure_count": len(failures),
            "response_count": len(response_ids),
            "failure_rate": 100 * len(failures) / len(response_ids),
        }

    @staticmethod
    def _retrieval_structured_feeds(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Expose authorized normalization and grouping facts to task generation."""
        feeds: list[dict[str, Any]] = []
        for passage in evidence:
            facts = {
                field: copy.deepcopy(passage[field])
                for field in ("canonical_entities", "normalized_record", "claim_group")
                if field in passage and passage[field] is not None
            }
            if facts:
                feeds.append({
                    "evidence_id": passage["evidence_id"],
                    "document_id": passage["document_id"], **facts,
                })
        _reject_internal_fields(feeds)
        return feeds

    def run_tailored_mitigation(
        self, mode: str, instance_ids: list[str]
    ) -> dict[str, Any]:
        """Apply explicit retrieval/inference modules without oracle subtype routing."""
        if (
            self._config.operational.selected_defense != "tailored_mitigation"
            or self._config.operational.selected_mitigation_mode != mode
        ):
            raise ValueError("Tailored mitigation mode differs from the selected configuration.")
        self._config.validate("defenses")
        settings = self._config.defenses.tailored_mitigation
        if mode not in settings.configurations:
            raise ValueError("Unknown tailored-mitigation mode.")
        instances = self._instances(instance_ids)
        routing: dict[str, Any] = _mapping(settings.routing_policy, "Mitigation routing")
        if routing.get("uses_gold_failure_labels") is not False:
            raise ValueError("Mitigation routing cannot use gold failure labels.")
        if routing.get("mode") != "universal":
            raise ValueError(
                "Only explicitly selected universal mitigation is implemented; "
                "a label-aware router requires a separately resolved protocol."
            )
        policies = self._load_manifest(settings.subtype_policy_manifest, "Subtype policies")
        group_policy: dict[str, Any] = _mapping(
            settings.model_group_aggregation, "Model-group aggregation"
        )
        if group_policy.get("mode") != "pooled_response_counts":
            raise ValueError("Model-group aggregation must explicitly select pooled_response_counts.")
        groups: dict[str, Any] = _mapping(group_policy.get("groups"), "Model groups")
        models: list[str] = (
            self._config.operational.selected_models or list(self._config.models.evaluation)
        )
        flattened: list[str] = [model for members in groups.values() for model in members]
        if len(flattened) != len(set(flattened)) or not set(models).issubset(flattened):
            raise ValueError("Model groups must be disjoint and cover evaluated models.")
        repetitions: int = _integer(settings.repetition_count, "Mitigation repetitions")
        guidance, profiles = self._task_manifests()
        constraint_manifest: dict[str, Any] | None = None
        if mode in {"inference_only", "full_fledged"}:
            constraint_manifest = self._load_manifest(
                settings.prompt_manifest, "Mitigation prompt manifest"
            )
            self._adapter(constraint_manifest.get("adapter"), "mitigation.constraints.adapter")
        if mode in {"retrieval_only", "full_fledged"}:
            retrieval = _mapping(settings.retrieval_models, "Mitigation retrieval")
            self._adapter(retrieval.get("adapter"), "mitigation.retrieval.adapter")
        self._models.validate_endpoints(models)
        manifest = self._manifest(
            "tailored_mitigation",
            {
                "mode": mode,
                "instance_ids": sorted(instance_ids),
                "settings": settings.model_dump(mode="json"),
                "subtype_policies_hash": _content_hash(policies),
                "uses_gold_routing": False,
            },
        )
        response_ids: list[str] = []
        for instance in instances:
            payload, schema = self._task_payload(instance, guidance, profiles)
            if mode in {"retrieval_only", "full_fledged"}:
                retrieved = self._retrieve(instance, {
                    **_mapping(settings.retrieval_models, "Mitigation retrieval"),
                    "retrieval_configuration": {
                        **{key: value for key, value in retrieval.items() if key != "adapter"},
                        "retrieval_models": settings.retrieval_models,
                        "chunking_policy": settings.chunking_policy,
                        "evidence_context_budget": settings.evidence_context_budget,
                        "source_diversity_parameters": settings.source_diversity_parameters,
                        "freshness_parameters": settings.freshness_parameters,
                        "ioc_reuse_thresholds": settings.ioc_reuse_thresholds,
                        "subtype_policy_manifest": policies,
                    },
                })
                payload["source_documents"] = retrieved
                payload["structured_feeds"] = self._retrieval_structured_feeds(retrieved)
            if constraint_manifest is not None:
                constrained: Any = self._invoke(
                    constraint_manifest["adapter"],
                    {
                        "task_name": instance.task_name,
                        "original_guidance": payload["task_guidance"],
                        "model_visible_inputs": payload,
                        "subtype_policy_manifest": policies,
                    },
                    "mitigation.constraints.adapter",
                )
                text: str = _identifier(
                    _mapping(constrained, "Inference constraints").get("task_guidance")
                )
                payload["task_guidance"] = text
            for model_id in models:
                for run_index in range(repetitions):
                    response = self._generate_task(
                        instance, model_id, run_index, payload, schema,
                        {
                            "protocol": "tailored_mitigation",
                            "condition": mode,
                            "experiment_manifest_id": manifest["manifest_id"],
                        },
                    )
                    response_ids.append(response.response_id)
        outcome = self._identify_and_categorize(response_ids, require_subtypes=True)
        metrics = self._evaluator.aggregate_task_metrics(response_ids)
        prevalence = (
            self._mitigation_prevalence(response_ids, outcome, group_policy)
            if outcome["status"] == "completed" else None
        )
        results = self._manifest("tailored_mitigation_results", {
            "parent_manifest_id": manifest["manifest_id"],
            "mode": mode, "outcome": outcome,
            "task_metrics": metrics, "model_group_subtype_prevalence": prevalence,
        })
        return {
            **outcome,
            "mode": mode,
            "experiment_manifest_id": manifest["manifest_id"],
            "response_ids": response_ids,
            "model_group_policy": group_policy,
            "result_manifest_id": results["manifest_id"],
            "task_metrics": metrics,
            "model_group_subtype_prevalence": prevalence,
            "task_metric_contributions_persisted": True,
        }

    def _mitigation_prevalence(
        self, response_ids: list[str], outcome: dict[str, Any],
        group_policy: dict[str, Any],
    ) -> dict[str, Any]:
        """Count finalized multilabel failures over every evaluated group response."""
        taxonomy = self._taxonomy()
        known = {item.subtype_id for item in taxonomy.subtype_definitions}
        failures = set(outcome["identification"]["failure_response_ids"])
        categorization = outcome.get("categorization") or {}
        bindings = categorization.get("final_assignment_ids_by_response", {})
        if set(bindings) != failures or not failures.issubset(response_ids):
            raise ValueError("Mitigation prevalence requires complete finalized failure assignments.")
        labels: dict[str, set[str]] = {}
        for response_id, identity in bindings.items():
            found = []
            for kind in _ASSIGNMENT_KINDS:
                try:
                    found.append((kind, self._store.read_record(kind, identity)))
                except KeyError:
                    pass
            if len(found) != 1:
                raise ValueError("Mitigation assignment identity is missing or ambiguous.")
            kind, raw = found[0]
            self._immutable(kind, identity)
            assignment = LabelAssignment.model_validate(raw, context={
                "taxonomy": taxonomy, "require_taxonomy": True,
                "require_evidence_for_each_label": True,
            })
            if assignment.response_id != response_id or assignment.uncertain:
                raise ValueError("Mitigation assignment is misbound or unresolved.")
            labels[response_id] = set(assignment.labels)
        groups = _mapping(group_policy.get("groups"), "Model groups")
        inverse = {model: name for name, members in groups.items() for model in members}
        if sum(len(members) for members in groups.values()) != len(inverse):
            raise ValueError("Mitigation model groups must be disjoint.")
        denominators: Counter[str] = Counter()
        counts: dict[str, Counter[str]] = {name: Counter() for name in groups}
        for response, _ in self._responses(response_ids):
            if response.model_id not in inverse:
                raise ValueError("Mitigation model groups do not cover evaluated responses.")
            group = inverse[response.model_id]
            denominators[group] += 1
            counts[group].update(labels.get(response.response_id, set()))
        return {
            name: {
                "response_count": denominators[name],
                "denominator_policy": "all_evaluated_group_responses",
                "counts": {label: counts[name][label] for label in sorted(known)},
                "percentages": {
                    label: 100 * counts[name][label] / denominators[name]
                    if denominators[name] else None
                    for label in sorted(known)
                },
                "labels_may_overlap": True,
            }
            for name in groups
        }

    def run_grounding_comparison(self, instance_ids: list[str]) -> dict[str, Any]:
        """Evaluate matched original/RAG/RAG+verification arms with one revision."""
        if self._config.operational.selected_defense != "stronger_grounding":
            raise ValueError("Stronger grounding must be explicitly selected.")
        self._config.validate("defenses")
        settings = self._config.defenses.stronger_grounding
        instances = self._instances(instance_ids)
        if len(instances) != settings.distinct_task_instances:
            raise ValueError("Grounding comparison requires exactly 300 selected task instances.")
        counts: Counter[str] = Counter(instance.task_name for instance in instances)
        if set(counts) != set(settings.tasks) or any(
            count != settings.instances_per_task for count in counts.values()
        ):
            raise ValueError("Grounding comparison requires 100 instances for each declared task.")
        matched = self._load_manifest(
            settings.matched_instance_manifest, "Grounding matched-instance manifest"
        )
        if _ids(matched.get("instance_ids"), "Matched instance IDs") != sorted(instance_ids):
            raise ValueError("Supplied instances differ from the frozen matched sampling manifest.")
        if matched.get("sampling_seed") != settings.sampling_seed:
            raise ValueError("Matched sampling seed differs from configuration.")
        retrieval_declaration: dict[str, Any] = _mapping(
            settings.retrieval.fusion_rule, "Grounding fusion/retrieval implementation"
        )
        retrieval_adapter: Any = _required(
            retrieval_declaration.get("adapter"), "Grounding retrieval adapter"
        )
        verifier_policy: dict[str, Any] = _mapping(
            settings.evidence_verification.claim_segmentation,
            "Grounding segmentation/verifier implementation",
        )
        verifier_adapter: Any = _required(
            verifier_policy.get("adapter"), "AlignScore verifier adapter"
        )
        self._adapter(retrieval_adapter, "grounding.retrieval.adapter")
        self._adapter(verifier_adapter, "grounding.verifier.adapter")
        self._models.validate_endpoints(settings.models)
        guidance, profiles = self._task_manifests()
        controls_hash: str = _content_hash({
            "instance_hashes": {
                instance.instance_id: instance.content_hash for instance in instances
            },
            "decoding": self._config.inference,
            "scoring": self._config.evaluation.scoring,
            "failure_criteria": self._config.evaluation.scoring.response_level_failure_criteria,
            "retrieval": settings.retrieval,
        })
        manifest = self._manifest(
            "stronger_grounding_comparison",
            {
                "instance_ids": sorted(instance_ids),
                "matched_manifest_hash": _content_hash(matched),
                "settings": settings.model_dump(mode="json"),
                "comparison_controls_hash": controls_hash,
            },
        )
        final_ids: list[str] = []
        candidates: list[str] = []
        for instance in instances:
            original_payload, schema = self._task_payload(instance, guidance, profiles)
            retrieval = self._retrieve(
                instance,
                {
                    "adapter": retrieval_adapter,
                    "retrieval_configuration": settings.retrieval.model_dump(mode="json"),
                },
            )
            evidence_hash: str = _content_hash(retrieval)
            for model_id in settings.models:
                common: dict[str, Any] = {
                    "protocol": "stronger_grounding",
                    "comparison_id": manifest["manifest_id"],
                    "comparison_controls_hash": controls_hash,
                }
                original = self._generate_task(
                    instance, model_id, 0, original_payload, schema,
                    {**common, "condition": "original"},
                )
                final_ids.append(original.response_id)
                rag_payload: dict[str, Any] = copy.deepcopy(original_payload)
                rag_payload["source_documents"] = [
                    *rag_payload["source_documents"], *retrieval
                ]
                rag_payload["structured_feeds"] = [
                    *rag_payload["structured_feeds"], *self._retrieval_structured_feeds(retrieval),
                ]
                rag = self._generate_task(
                    instance, model_id, 0, rag_payload, schema,
                    {**common, "condition": "strong_rag"},
                )
                final_ids.append(rag.response_id)
                candidate = self._generate_task(
                    instance, model_id, 0, rag_payload, schema,
                    {**common, "condition": "verification_candidate"},
                )
                candidates.append(candidate.response_id)
                if candidate.status != ResponseStatus.SUCCESS:
                    raise RuntimeError(
                        "Grounding verification candidate is invalid. An explicit "
                        "invalid-candidate policy is required; no repair revision was substituted."
                    )
                feedback: dict[str, Any] = _mapping(
                    self._invoke(
                        verifier_adapter,
                        {
                            "candidate_response_id": candidate.response_id,
                            "candidate_raw_text": candidate.raw_text,
                            "task_input": rag_payload,
                            "retrieved_evidence": retrieval,
                            "evidence_package_hash": evidence_hash,
                            "method": settings.evidence_verification.method,
                            "checkpoint": settings.evidence_verification.checkpoint,
                            "claim_segmentation": (
                                settings.evidence_verification.claim_segmentation
                            ),
                            "alignment_threshold": (
                                settings.evidence_verification.alignment_threshold
                            ),
                        },
                        "grounding.verifier.adapter",
                    ),
                    "AlignScore feedback",
                )
                feedback_record = self._manifest(
                    "grounding_verifier_feedback",
                    {"feedback": feedback, "candidate_response_id": candidate.response_id},
                )
                feedback["artifact_id"] = feedback_record["manifest_id"]
                contract_hash: str = _content_hash(instance.output_profile)
                generation_settings: dict[str, Any] = {
                    "temperature": self._config.inference.temperature,
                    "top_p": self._config.inference.top_p,
                }
                generation_hash: str = _content_hash(generation_settings)
                documents: list[dict[str, Any]] = []
                for evidence in instance.evidence_records:
                    value = evidence.model_dump(mode="json")
                    documents.append(value)
                revision_template: str = _identifier(
                    settings.evidence_verification.revision_prompt_template_path
                )
                revision_payload: dict[str, Any] = {
                    "task_input": {
                        "instance_id": instance.instance_id,
                        "case_id": instance.case_id,
                        "task_name": instance.task_name,
                        "stage": instance.stage.value,
                        "snapshot_date": instance.snapshot_date.isoformat(),
                        "task_guidance": original_payload["task_guidance"],
                        "source_documents": documents,
                        "structured_feeds": rag_payload["structured_feeds"],
                        "candidates": original_payload["candidates"],
                        "candidate_policy": original_payload["candidate_policy"],
                        "environment": instance.environment,
                        "output_profile": instance.output_profile,
                    },
                    "candidate_response": candidate.model_dump(mode="json"),
                    "retrieved_evidence": retrieval,
                    "verification_feedback": feedback,
                    "prompt_settings": self._config.evaluation.prompt.model_dump(mode="json"),
                    "generation_settings": generation_settings,
                    "grounding_settings": settings.model_dump(mode="json"),
                    "revision_context": {
                        "configuration": "strong_rag_with_evidence_verification",
                        "comparison_id": manifest["manifest_id"],
                        "instance_id": instance.instance_id,
                        "model_id": model_id,
                        "revision_index": 1,
                        "template_path": revision_template,
                        "evidence_package_hash": evidence_hash,
                        "candidate_evidence_package_hash": evidence_hash,
                        "output_contract_hash": contract_hash,
                        "candidate_output_contract_hash": contract_hash,
                        "generation_settings_hash": generation_hash,
                        "candidate_generation_settings_hash": generation_hash,
                        "snapshot_admission_verified": True,
                        "feedback_snapshot_admission_verified": True,
                    },
                }
                revised = self._generate_task(
                    instance, model_id, 0, revision_payload, schema,
                    {**common, "condition": "strong_rag_with_evidence_verification"},
                    template=revision_template,
                )
                final_ids.append(revised.response_id)
        expected: int = (
            settings.model_instance_pairs_per_configuration * len(settings.configurations)
        )
        if len(final_ids) != expected:
            raise RuntimeError("Grounding comparison produced an incomplete final population.")
        outcome = self._identify_and_categorize(final_ids, require_subtypes=False)
        if outcome["status"] != "completed":
            return {
                **outcome,
                "experiment_manifest_id": manifest["manifest_id"],
                "final_response_ids": final_ids,
                "candidate_response_ids_excluded_from_final_metrics": candidates,
            }
        metrics: dict[str, Any] = self._evaluator.aggregate_task_metrics(final_ids)
        counts_by_condition: dict[str, list[int]] = {
            condition: [0, 0] for condition in settings.configurations
        }
        failed: set[str] = set(outcome["identification"]["failure_response_ids"])
        for identity in final_ids:
            response = ModelResponse.model_validate(
                self._store.read_record("model_response", identity)
            )
            condition: str = response.usage["evaluation_context"]["condition"]
            counts_by_condition[condition][0] += int(identity in failed)
            counts_by_condition[condition][1] += 1
        rates: dict[str, Any] = {}
        original_count, original_n = counts_by_condition["original"]
        original_rate: float = 100 * original_count / original_n
        for condition, (failures, denominator) in counts_by_condition.items():
            rate: float = 100 * failures / denominator
            rates[condition] = {
                "failure_count": failures,
                "response_count": denominator,
                "failure_rate_percent": rate,
                "absolute_reduction_percentage_points": original_rate - rate,
            }
        return {
            "status": "completed",
            "experiment_manifest_id": manifest["manifest_id"],
            "final_response_ids": final_ids,
            "candidate_response_ids_excluded_from_final_metrics": candidates,
            "task_metrics": metrics,
            "pooled_failure_rates": rates,
            "runs": settings.runs,
            "standard_deviations_reported": False,
            "verifier_scores_used_as_final_labels": False,
        }
