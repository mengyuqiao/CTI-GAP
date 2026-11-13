"""Strict, resumable inference over explicitly configured LiteLLM endpoints.

The provider message mapping is fixed: one user message containing the exact
rendered prompt. This adapter never repairs output, regenerates content-invalid
answers, substitutes models, or enables provider-native constrained decoding.

Request-binding contract
========================
generate_batch() accepts dictionaries containing:

    model_id, prompt, run_index, schema_name, instance_id, schema_contract

schema_contract is an explicitly supplied, versioned validation declaration:

    {
        "version": "v1",
        "json_schema": {...},
        "kind": "task" | "classification",
        "taxonomy_id": "...",                 # classification only
        "discovery_mode": false,              # classification only
        "allow_empty_labels_when_uncertain": true
    }

Optional request keys are probability_interface and template_provenance.
Bindings are persisted as prompt records. Direct generate() requires exactly
one matching persisted binding; it never extracts instance IDs from prose.

JSON Schema support is intentionally explicit and fail-closed. Unsupported
keywords are rejected before inference, rather than partially enforced.

Retry policy
============
inference.retry_policy must explicitly declare:
    max_attempts, timeout_seconds, retryable_status_codes,
    retryable_exception_names, backoff_seconds, retry_ambiguous_timeouts

backoff_seconds contains one delay per permitted retry. Names in
retryable_exception_names are exact exception class names. Content validation
failures never retry.

Optional provider adapter declarations
======================================
When provider metadata is unavailable, capability declarations may be supplied
explicitly inside inference.reasoning_settings:

    {
      "parameters": {"reasoning_effort": "low"},
      "adapters": {
        "G5": {
          "supported_parameters": ["temperature", "top_p", "max_tokens"],
          "api_base": null,
          "api_key_env": "OPENAI_API_KEY",
          "seed_policy": "require_support",
          "probability_interface": null
        }
      }
    }

Without the wrapper, reasoning_settings is treated as provider reasoning
parameters. Adapter declarations are hashed with the effective request. Secrets
are read from environment variables and never stored.

No model calls are performed during construction or endpoint validation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import litellm
from jinja2 import FileSystemLoader, StrictUndefined
from jinja2.sandbox import SandboxedEnvironment
from pydantic import ValidationError, model_validator

from config import ExperimentConfig
from schemas import (
    EvidenceSpan,
    LabelAssignment,
    ModelResponse,
    ResponseStatus,
    SchemaModel,
    TaskInstance,
    TaxonomyVersion,
    _canonical_json,
    _content_hash,
    _identifier,
    _text_hash,
)
from storage import ArtifactStore


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "null": (type(None),),
}
_SCHEMA_KEYWORDS: frozenset[str] = frozenset(
    {
        "$schema", "title", "description", "type", "properties", "required",
        "additionalProperties", "items", "enum", "const", "minimum", "maximum",
        "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength",
        "pattern", "minItems", "maxItems", "uniqueItems", "anyOf", "oneOf",
        "allOf", "minProperties", "maxProperties",
    }
)
_BATCH_KEYS: frozenset[str] = frozenset(
    {
        "model_id", "prompt", "run_index", "schema_name", "instance_id",
        "schema_contract", "probability_interface", "template_provenance",
    }
)
_CLASSIFICATION_KEYS: frozenset[str] = frozenset(
    {"labels", "evidence_spans", "uncertain"}
)
_SECRET_KEYS: frozenset[str] = frozenset(
    {
        "api_key", "authorization", "password", "access_token", "refresh_token",
        "client_secret", "credentials", "private_key",
    }
)
_FORBIDDEN_PROVIDER_PARAMETERS: frozenset[str] = frozenset(
    {
        "model", "messages", "api_key", "api_base", "base_url", "timeout",
        "num_retries", "max_retries", "fallbacks", "caching", "cache",
        "drop_params", "response_format", "tools", "tool_choice", "stream",
        "temperature", "top_p", "max_tokens", "seed", "n",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _positive_number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number.")
    return float(value)


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return value


def _safe_error(error: BaseException) -> dict[str, Any]:
    """Retain operational diagnostics without serializing exception messages."""
    status: Any = getattr(error, "status_code", None)
    return {
        "exception_class": type(error).__name__,
        "status_code": status if isinstance(status, int) else None,
        "message": "Provider exception text withheld to protect credentials.",
        "completion_may_have_occurred": (
            isinstance(error, TimeoutError)
            or "timeout" in type(error).__name__.lower()
        ),
    }


def _retry_permitted(error: Mapping[str, Any], policy: Mapping[str, Any]) -> bool:
    """Apply the same transport decision during execution and process resumption."""
    if error.get("completion_may_have_occurred") and not policy["retry_ambiguous_timeouts"]:
        return False
    return (
        error.get("status_code") in policy["retryable_status_codes"]
        or error.get("exception_class") in policy["retryable_exception_names"]
    )


def _reject_secrets(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower().replace("-", "_") in _SECRET_KEYS:
                raise ValueError("Credentials must be supplied through environment variables.")
            _reject_secrets(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_secrets(item)


def _strict_object(text: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON keys are forbidden.")
            result[key] = value
        return result

    def constant(_: str) -> None:
        raise ValueError("Nonfinite JSON numbers are forbidden.")

    result: Any = json.loads(
        text, object_pairs_hook=pairs, parse_constant=constant
    )
    if not isinstance(result, dict):
        raise ValueError("The output must be exactly one JSON object.")
    # Also reject finite-looking literals overflowing Python's float.
    _canonical_json(result)
    return result


def _check_schema(schema: Any) -> None:
    """Validate the supported JSON Schema dialect before expensive requests."""
    if not isinstance(schema, dict):
        raise ValueError("Every validation schema must be an object.")
    unknown: set[str] = set(schema) - _SCHEMA_KEYWORDS
    if unknown:
        raise ValueError(f"Unsupported JSON Schema keywords: {sorted(unknown)}.")
    declared: Any = schema.get("type")
    types: list[Any] = declared if isinstance(declared, list) else [declared]
    if declared is not None and (
        not types or any(item not in _JSON_TYPES for item in types)
    ):
        raise ValueError("Unsupported JSON Schema type.")
    properties: Any = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("Schema properties must be an object.")
    for child in properties.values():
        _check_schema(child)
    required: Any = schema.get("required", [])
    if (
        not isinstance(required, list)
        or any(not isinstance(item, str) for item in required)
        or len(required) != len(set(required))
    ):
        raise ValueError("Schema required must be a duplicate-free string array.")
    additional: Any = schema.get("additionalProperties", True)
    if not isinstance(additional, (bool, dict)):
        raise ValueError("additionalProperties must be boolean or a schema.")
    if isinstance(additional, dict):
        _check_schema(additional)
    if "items" in schema:
        _check_schema(schema["items"])
    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            alternatives: Any = schema[key]
            if not isinstance(alternatives, list) or not alternatives:
                raise ValueError(f"{key} requires nonempty schema alternatives.")
            for alternative in alternatives:
                _check_schema(alternative)
    for key in (
        "minLength", "maxLength", "minItems", "maxItems",
        "minProperties", "maxProperties",
    ):
        if key in schema:
            _nonnegative_integer(schema[key], key)
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if key in schema:
            value: Any = schema[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{key} requires a finite number.")
    if "pattern" in schema:
        re.compile(schema["pattern"])
    if "enum" in schema and (
        not isinstance(schema["enum"], list) or not schema["enum"]
    ):
        raise ValueError("enum requires a nonempty array.")
    if "uniqueItems" in schema and not isinstance(schema["uniqueItems"], bool):
        raise ValueError("uniqueItems must be boolean.")


def _matches_type(value: Any, name: str) -> bool:
    if name in {"integer", "number"} and isinstance(value, bool):
        return False
    return isinstance(value, _JSON_TYPES[name])


def _validate_value(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate without coercion, insertion, identifier repair, or clamping."""
    declared: Any = schema.get("type")
    if declared is not None:
        types: list[str] = declared if isinstance(declared, list) else [declared]
        if not any(_matches_type(value, name) for name in types):
            raise ValueError(f"{path}: incorrect JSON type.")
    if "enum" in schema and not any(
        _canonical_json(value) == _canonical_json(option)
        for option in schema["enum"]
    ):
        raise ValueError(f"{path}: value is not an allowed enum member.")
    if "const" in schema and _canonical_json(value) != _canonical_json(schema["const"]):
        raise ValueError(f"{path}: value differs from the required constant.")

    for keyword in ("anyOf", "oneOf", "allOf"):
        if keyword not in schema:
            continue
        accepted: int = 0
        for alternative in schema[keyword]:
            try:
                _validate_value(value, alternative, path)
                accepted += 1
            except ValueError:
                pass
        required_count: int = len(schema[keyword])
        if (
            (keyword == "anyOf" and accepted == 0)
            or (keyword == "oneOf" and accepted != 1)
            or (keyword == "allOf" and accepted != required_count)
        ):
            raise ValueError(f"{path}: {keyword} constraint failed.")

    if isinstance(value, dict):
        required: list[str] = schema.get("required", [])
        if any(key not in value for key in required):
            raise ValueError(f"{path}: required fields are missing.")
        properties: dict[str, Any] = schema.get("properties", {})
        additional: Any = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                _validate_value(item, properties[key], f"{path}.{key}")
            elif additional is False:
                raise ValueError(f"{path}: additional fields are forbidden.")
            elif isinstance(additional, dict):
                _validate_value(item, additional, f"{path}.{key}")
        for keyword, violation in (
            ("minProperties", len(value) < schema.get("minProperties", 0)),
            ("maxProperties", len(value) > schema.get("maxProperties", len(value))),
        ):
            if violation:
                raise ValueError(f"{path}: {keyword} constraint failed.")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{path}: too few array items.")
        if len(value) > schema.get("maxItems", len(value)):
            raise ValueError(f"{path}: too many array items.")
        if schema.get("uniqueItems", False):
            serialized: list[str] = [_canonical_json(item) for item in value]
            if len(serialized) != len(set(serialized)):
                raise ValueError(f"{path}: duplicate array items.")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate_value(item, schema["items"], f"{path}[{index}]")
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"{path}: string is too short.")
        if len(value) > schema.get("maxLength", len(value)):
            raise ValueError(f"{path}: string is too long.")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ValueError(f"{path}: string pattern mismatch.")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise ValueError(f"{path}: nonfinite number.")
        comparisons: tuple[tuple[str, bool], ...] = (
            ("minimum", value < schema.get("minimum", value)),
            ("maximum", value > schema.get("maximum", value)),
            ("exclusiveMinimum", value <= schema.get("exclusiveMinimum", value - 1)),
            ("exclusiveMaximum", value >= schema.get("exclusiveMaximum", value + 1)),
        )
        for keyword, violation in comparisons:
            if keyword in schema and violation:
                raise ValueError(f"{path}: {keyword} constraint failed.")


class _ValidatedOutput(SchemaModel):
    """Pydantic validation boundary for an explicitly supplied JSON contract."""

    content: dict[str, Any]

    @model_validator(mode="after")
    def _check_contract(self, info: Any) -> "_ValidatedOutput":
        context: dict[str, Any] = info.context or {}
        schema: Any = context.get("json_schema")
        if not isinstance(schema, dict):
            raise ValueError("An explicit JSON schema is required.")
        _validate_value(self.content, schema)
        return self


class ModelClient:
    """Shared, bounded, persistent adapter for task and classification inference."""

    def __init__(self, config: ExperimentConfig, store: ArtifactStore) -> None:
        self._config: ExperimentConfig = config
        self._store: ArtifactStore = store
        source: Path | None = getattr(config, "_source_path", None)
        self._base: Path = source.parent if source is not None else Path.cwd()
        self._local: threading.local = threading.local()
        self._locks_guard: threading.Lock = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._render_lock: threading.Lock = threading.Lock()
        self._rendered: dict[str, dict[str, Any]] = {}

    def _model(self, model_id: str) -> tuple[str, str, str]:
        _identifier(model_id)
        if model_id in self._config.models.evaluation:
            entry = self._config.models.evaluation[model_id]
            path: str = f"models.evaluation.{model_id}"
        else:
            matches = [
                (index, entry)
                for index, entry in enumerate(self._config.models.categorization_panel)
                if model_id == entry.name
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Unknown model ID. Use an evaluation abbreviation or the exact "
                    "configured categorization-panel name."
                )
            index, entry = matches[0]
            path = f"models.categorization_panel.{index}"
        if not entry.endpoint_or_checkpoint or not entry.revision:
            raise ValueError(f"{path}: endpoint/checkpoint and revision are unresolved.")
        return entry.endpoint_or_checkpoint, entry.revision, path

    def _reasoning(self, model_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        settings: Any = self._config.inference.reasoning_settings
        if settings is None:
            applicability: Any = self._config.operational.applicability.get(
                "inference.reasoning_settings"
            )
            if applicability != "not_applicable":
                raise ValueError("inference.reasoning_settings is unresolved.")
            return {}, {}
        if not isinstance(settings, dict):
            raise ValueError("reasoning_settings must be a mapping or explicitly inapplicable.")
        if "adapters" in settings or "parameters" in settings:
            if set(settings) - {"adapters", "parameters"}:
                raise ValueError("Unknown reasoning adapter wrapper fields.")
            parameters: Any = settings.get("parameters", {})
            adapters: Any = settings.get("adapters", {})
            if not isinstance(parameters, dict) or not isinstance(adapters, dict):
                raise ValueError("Reasoning parameters and adapters must be mappings.")
            adapter: Any = adapters.get(model_id, {})
        else:
            parameters = settings
            adapter = {}
        if not isinstance(adapter, dict):
            raise ValueError("Provider adapter declaration must be a mapping.")
        if set(parameters) & _FORBIDDEN_PROVIDER_PARAMETERS:
            raise ValueError("Reasoning settings cannot override fixed request controls.")
        _reject_secrets(parameters)
        _reject_secrets(adapter)
        return copy.deepcopy(parameters), copy.deepcopy(adapter)

    def _retry_policy(self) -> dict[str, Any]:
        policy: Any = self._config.inference.retry_policy
        required: set[str] = {
            "max_attempts", "timeout_seconds", "retryable_status_codes",
            "retryable_exception_names", "backoff_seconds",
            "retry_ambiguous_timeouts",
        }
        if not isinstance(policy, dict) or set(policy) != required:
            raise ValueError(
                "inference.retry_policy must explicitly declare the documented retry fields."
            )
        attempts: int = _nonnegative_integer(policy["max_attempts"], "max_attempts")
        if attempts < 1:
            raise ValueError("max_attempts must be positive.")
        _positive_number(policy["timeout_seconds"], "timeout_seconds")
        delays: Any = policy["backoff_seconds"]
        if not isinstance(delays, list) or len(delays) != attempts - 1:
            raise ValueError("Provide one explicit backoff delay per allowed retry.")
        for delay in delays:
            if (
                isinstance(delay, bool)
                or not isinstance(delay, (int, float))
                or not math.isfinite(float(delay))
                or delay < 0
            ):
                raise ValueError("Retry delays must be finite nonnegative numbers.")
        codes: Any = policy["retryable_status_codes"]
        names: Any = policy["retryable_exception_names"]
        if not isinstance(codes, list) or any(
            isinstance(code, bool) or not isinstance(code, int) for code in codes
        ):
            raise ValueError("Retry status codes must be an explicit integer list.")
        if not isinstance(names, list) or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise ValueError("Retry exception names must be an explicit string list.")
        if not isinstance(policy["retry_ambiguous_timeouts"], bool):
            raise ValueError("retry_ambiguous_timeouts must be explicitly boolean.")
        return copy.deepcopy(policy)

    def _capabilities(
        self, endpoint: str, adapter: dict[str, Any]
    ) -> set[str]:
        declared: Any = adapter.get("supported_parameters")
        if declared is not None:
            if not isinstance(declared, list) or any(
                not isinstance(item, str) for item in declared
            ):
                raise ValueError("supported_parameters must be a string list.")
            return set(declared)
        try:
            supported: Any = litellm.get_supported_openai_params(model=endpoint)
        except Exception:
            supported = None
        if not supported:
            raise ValueError(
                "Provider capabilities are unresolved; supply a versioned explicit "
                "supported_parameters declaration. No live inference probe was made."
            )
        return set(supported)

    def validate_endpoints(self, model_ids: list[str]) -> None:
        """Validate declarations and local capabilities without a generation probe."""
        if not isinstance(model_ids, list) or not model_ids:
            raise ValueError("model_ids must be an explicit nonempty list.")
        inference = self._config.inference
        for field in (
            "temperature", "top_p", "max_output_tokens", "context_window",
            "concurrency", "truncation_policy",
        ):
            if getattr(inference, field) is None:
                raise ValueError(f"inference.{field} is unresolved.")
        truncation: Any = inference.truncation_policy
        truncation_mode: Any = (
            truncation.get("mode") if isinstance(truncation, dict) else truncation
        )
        if truncation_mode != "reject":
            raise ValueError(
                "This adapter supports truncation_policy: reject only; it never edits prompts."
            )
        self._retry_policy()
        for model_id in model_ids:
            endpoint, _, _ = self._model(model_id)
            parameters, adapter = self._reasoning(model_id)
            supported: set[str] = self._capabilities(endpoint, adapter)
            required: set[str] = {"temperature", "top_p", "max_tokens", *parameters}
            if required - supported:
                raise ValueError(
                    f"{model_id}: unsupported configured parameters "
                    f"{sorted(required - supported)}."
                )
            base: Any = adapter.get("api_base")
            if base is not None:
                parsed = urlsplit(base)
                if (
                    parsed.scheme not in {"https", "http"}
                    or not parsed.hostname
                    or parsed.username is not None
                    or parsed.password is not None
                    or parsed.query
                ):
                    raise ValueError("api_base must be an explicit credential-free service URL.")
            key_env: Any = adapter.get("api_key_env")
            if key_env is not None:
                _identifier(key_env)
                if not os.environ.get(key_env):
                    raise ValueError(f"{model_id}: required credential environment is unavailable.")
            elif base is None:
                validator: Any = getattr(litellm, "validate_environment", None)
                if validator is not None:
                    try:
                        environment: Any = validator(model=endpoint)
                    except Exception:
                        raise ValueError(
                            f"{model_id}: credential metadata could not be validated."
                        ) from None
                    if isinstance(environment, dict) and environment.get(
                        "keys_in_environment"
                    ) is False:
                        raise ValueError(
                            f"{model_id}: provider credential environment is incomplete."
                        )

            if self._config.operational.selected_defense == "counterfactual_consistency":
                probability = self._config.defenses.counterfactual_consistency
                extraction: Any = probability.probability_extraction
                if not isinstance(extraction, dict):
                    raise ValueError("C3 requires an explicit probability extraction mapping.")
                interface: Any = extraction.get("interface")
                if interface in {"candidate_log_likelihoods", "autoregressive_complete"}:
                    declaration = extraction.get("adapter")
                    expected_space = ("finite_candidates" if interface == "candidate_log_likelihoods"
                                      else "task_outputs")
                    if (
                        extraction.get("aggregation_space") != expected_space
                        or not isinstance(declaration, dict)
                        or set(declaration) != {
                            "callable", "version", "source_sha256", "parameters"
                        }
                    ):
                        raise ValueError("C3 independent probability scoring requires an explicit pinned adapter.")
                    # The separate scoring adapter verifies exact model/prompt
                    # provenance; ordinary JSON generation need not expose logits.
                    continue
                if not interface or interface != adapter.get("probability_interface"):
                    raise ValueError("C3 probability capability does not match its configured interface.")
                if interface != "token_logprobs" or "logprobs" not in supported:
                    raise ValueError(
                        "This adapter exposes token log probabilities only. Candidate or "
                        "full-output probability scoring requires a separately resolved adapter."
                    )
                # Token probabilities cannot masquerade as candidate probabilities.
                if extraction.get("aggregation_space") != "tokens":
                    raise ValueError("Token log probabilities do not certify candidate probabilities.")

    def render_prompt(self, template: str, payload: dict[str, Any]) -> str:
        """Render exact text under StrictUndefined and retain content provenance."""
        _identifier(template)
        if not isinstance(payload, dict):
            raise TypeError("Prompt payload must be a dictionary.")
        if any(key.startswith("__") and key.endswith("invalid_payload__") for key in payload):
            raise ValueError("Reserved template failure symbols cannot be supplied.")
        path: Path = Path(template)
        path = path if path.is_absolute() else self._base / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError("The explicitly selected prompt template is missing.")
        raw: bytes = path.read_bytes()
        environment = SandboxedEnvironment(
            loader=FileSystemLoader(str(path.parent)),
            undefined=StrictUndefined,
            autoescape=False,
            keep_trailing_newline=True,
        )
        # No dynamic dates, model outputs, references, or configuration are injected.
        normalized_payload: dict[str, Any] = json.loads(_canonical_json(payload))
        rendered: str = environment.from_string(raw.decode("utf-8")).render(
            **normalized_payload
        )
        if not rendered.strip():
            raise ValueError("Rendered prompt is empty.")
        provenance: dict[str, Any] = {
            "template_path": template,
            "template_hash": hashlib.sha256(raw).hexdigest(),
            "payload_hash": _content_hash(normalized_payload),
            "prompt_hash": _text_hash(rendered),
        }
        with self._render_lock:
            self._rendered[provenance["prompt_hash"]] = provenance
        return rendered

    def _validate_contract(self, contract: Any) -> dict[str, Any]:
        if not isinstance(contract, dict):
            raise ValueError("A versioned schema_contract must be supplied.")
        allowed: set[str] = {
            "version", "json_schema", "kind", "taxonomy_id",
            "discovery_mode", "allow_empty_labels_when_uncertain",
        }
        if set(contract) - allowed:
            raise ValueError("Unknown schema_contract fields.")
        _identifier(contract.get("version"))
        if contract.get("kind") not in {"task", "classification", "taxonomy_proposal"}:
            raise ValueError("schema_contract.kind must be task, classification or taxonomy_proposal.")
        _check_schema(contract.get("json_schema"))
        if contract["kind"] == "classification":
            taxonomy_id: str = _identifier(contract.get("taxonomy_id"))
            self._taxonomy(taxonomy_id)
            for key in ("discovery_mode", "allow_empty_labels_when_uncertain"):
                if not isinstance(contract.get(key), bool):
                    raise ValueError(f"Classification contract requires explicit {key}.")
        return copy.deepcopy(contract)

    def _taxonomy(self, taxonomy_id: str) -> TaxonomyVersion:
        matches: list[dict[str, Any]] = []
        for kind in ("taxonomy", "taxonomy_version"):
            try:
                matches.append(self._store.read_record(kind, taxonomy_id))
            except KeyError:
                pass
        if len(matches) != 1:
            raise ValueError("Classification requires one unambiguous stored taxonomy.")
        return TaxonomyVersion.model_validate(matches[0])

    def _binding(self, model_id: str, prompt: str, run_index: int, schema_name: str) -> dict[str, Any]:
        current: Any = getattr(self._local, "binding", None)
        prompt_hash: str = _text_hash(prompt)
        if current is not None:
            candidates: list[dict[str, Any]] = [current]
        else:
            candidates = [
                item for item in self._store.list_records("prompt")
                if item.get("record_type") == "inference_binding"
                and item.get("model_id") == model_id
                and item.get("run_index") == run_index
                and item.get("schema_name") == schema_name
                and item.get("prompt_hash") == prompt_hash
                and item.get("config_hash") == self._config.config_hash
            ]
        if len(candidates) != 1:
            raise ValueError("Missing or ambiguous explicit task-instance request binding.")
        binding: dict[str, Any] = candidates[0]
        for key, expected in (
            ("model_id", model_id), ("run_index", run_index),
            ("schema_name", schema_name), ("prompt_hash", prompt_hash),
            ("config_hash", self._config.config_hash),
        ):
            if binding.get(key) != expected:
                raise ValueError("Request binding differs from the requested inference.")
        TaskInstance.model_validate(
            self._store.read_record("task_instance", binding["instance_id"])
        )
        self._validate_contract(binding["schema_contract"])
        return binding

    def _seed(self, run_index: int, supported: set[str], adapter: dict[str, Any]) -> dict[str, Any]:
        seeds: list[int | None] | None = self._config.evaluation.run_seeds
        if seeds is None:
            if self._config.operational.applicability.get(
                "evaluation.run_seeds"
            ) != "not_applicable":
                raise ValueError("evaluation.run_seeds is unresolved.")
            return {"requested": None, "supported": "seed" in supported, "transmitted": False}
        if run_index >= len(seeds):
            raise ValueError("No explicit seed entry exists for this run index.")
        seed: int | None = seeds[run_index]
        if seed is None:
            path: str = f"evaluation.run_seeds.{run_index}"
            if self._config.operational.applicability.get(path) != "not_applicable":
                raise ValueError(f"{path} is unresolved.")
        supported_seed: bool = "seed" in supported
        if seed is not None and not supported_seed:
            if adapter.get("seed_policy") != "omit_if_unsupported":
                raise ValueError("Provider seed support is absent; an explicit omission policy is required.")
        return {
            "requested": seed,
            "supported": supported_seed,
            "transmitted": seed is not None and supported_seed,
        }

    def _request(self, model_id: str, prompt: str, run_index: int, schema_name: str) -> dict[str, Any]:
        self.validate_endpoints([model_id])
        _nonnegative_integer(run_index, "run_index")
        _identifier(schema_name)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be nonempty exact text.")
        binding: dict[str, Any] = self._binding(model_id, prompt, run_index, schema_name)
        endpoint, revision, _ = self._model(model_id)
        parameters, adapter = self._reasoning(model_id)
        supported: set[str] = self._capabilities(endpoint, adapter)
        seed: dict[str, Any] = self._seed(run_index, supported, adapter)
        inference = self._config.inference
        controls: dict[str, Any] = {
            "temperature": inference.temperature,
            "top_p": inference.top_p,
            "max_tokens": inference.max_output_tokens,
            **parameters,
        }
        if seed["transmitted"]:
            controls["seed"] = seed["requested"]
        probability: Any = binding.get("probability_interface")
        if probability is not None:
            if probability != "token_logprobs" or adapter.get("probability_interface") != probability:
                raise ValueError("Unsupported probability extraction interface.")
            if "logprobs" not in supported:
                raise ValueError("Provider does not support requested log probabilities.")
            controls["logprobs"] = True

        try:
            tokens: int = litellm.token_counter(
                model=endpoint,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception:
            raise ValueError(
                "Context accounting is unavailable for this endpoint; no heuristic "
                "token estimate or silent truncation is permitted."
            ) from None
        if tokens + int(inference.max_output_tokens or 0) > int(inference.context_window or 0):
            raise ValueError("Request exceeds the explicitly configured context budget.")

        contract: dict[str, Any] = binding["schema_contract"]
        request: dict[str, Any] = {
            "model_id": model_id,
            "endpoint_or_checkpoint": endpoint,
            "revision": revision,
            "prompt_hash": _text_hash(prompt),
            "generation_controls": controls,
            "seed_behavior": seed,
            "schema_name": schema_name,
            "schema_version": contract["version"] + ":" + _content_hash(contract),
            "run_index": run_index,
            "config_hash": self._config.config_hash,
            "provider_options": {
                "adapter": adapter,
                "binding_id": binding["prompt_id"],
                "message_mapping": "single_user_message",
                "retry_policy": self._retry_policy(),
                "schema_contract": contract,
                "prompt_provenance": binding["template_provenance"],
                "context_input_tokens": tokens,
                "context_token_origin": "litellm_token_counter",
                "probability_interface": probability,
            },
            "instance_id": binding["instance_id"],
            "prompt": prompt,
        }
        request["request_hash"] = _content_hash(request)
        return request

    def _lock(self, request_hash: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(request_hash, threading.Lock())

    def generate(self, model_id: str, prompt: str, run_index: int, schema_name: str) -> ModelResponse:
        """Execute or recover one exactly bound experimental request."""
        request: dict[str, Any] = self._request(model_id, prompt, run_index, schema_name)
        request_hash: str = request["request_hash"]
        with self._lock(request_hash):
            cached: ModelResponse | None = self._store.get_cached_response(request_hash)
            if cached is not None:
                return cached
            self._store.save_blob(prompt.encode("utf-8"))
            self._store.create_record("request", request_hash, request)
            policy: dict[str, Any] = request["provider_options"]["retry_policy"]
            adapter: dict[str, Any] = request["provider_options"]["adapter"]
            previous: list[dict[str, Any]] = [
                item for item in self._store.list_records("request_attempt")
                if item["request_hash"] == request_hash
            ]
            if previous:
                latest = max(previous, key=lambda item: (
                    item.get("metadata", {}).get("attempt_index", -1),
                    item["completed_at"],
                ))
                if (
                    latest["outcome"] == "transport_failure"
                    and not _retry_permitted(latest["error"], policy)
                ):
                    return ModelResponse.model_validate(self._store.read_record(
                        "model_response", latest["response_id"]
                    ))
            completed_attempts: int = len(previous)
            if completed_attempts >= policy["max_attempts"]:
                failures = [
                    item for item in self._store.list_records(
                        "model_response", {"instance_id": request["instance_id"]}
                    )
                    if item.get("usage", {}).get("request_hash") == request_hash
                    and item["status"] == ResponseStatus.TRANSPORT_FAILURE.value
                ]
                if failures:
                    return ModelResponse.model_validate(failures[-1])
                raise RuntimeError("Retry budget exhausted; no additional provider call is authorized.")

            for attempt_index in range(completed_attempts, policy["max_attempts"]):
                attempt_id: str = uuid4().hex
                started: str = _now()
                clock: float = time.monotonic()
                response_id: str = _content_hash(
                    {"request_hash": request_hash, "attempt_id": attempt_id}
                )
                options: dict[str, Any] = {
                    **request["generation_controls"],
                    "model": request["endpoint_or_checkpoint"],
                    "messages": [{"role": "user", "content": prompt}],
                    "num_retries": 0,
                    "fallbacks": [],
                    "caching": False,
                    "drop_params": False,
                    "stream": False,
                    "timeout": policy["timeout_seconds"],
                }
                if adapter.get("api_base") is not None:
                    options["api_base"] = adapter["api_base"]
                if adapter.get("api_key_env") is not None:
                    options["api_key"] = os.environ[adapter["api_key_env"]]
                try:
                    provider: Any = litellm.completion(**options)
                except Exception as exc:
                    error: dict[str, Any] = _safe_error(exc)
                    ended: str = _now()
                    response = ModelResponse(
                        response_id=response_id,
                        instance_id=request["instance_id"],
                        model_id=model_id,
                        run_index=run_index,
                        prompt_hash=request["prompt_hash"],
                        config_hash=self._config.config_hash,
                        status=ResponseStatus.TRANSPORT_FAILURE,
                        usage={
                            "request_hash": request_hash,
                            "attempt_id": attempt_id,
                            "started_at": started,
                            "completed_at": ended,
                            "latency_seconds": time.monotonic() - clock,
                            "execution_error": error,
                        },
                    )
                    self._store.create_record(
                        "model_response", response_id, response.model_dump(mode="json")
                    )
                    self._store.create_record(
                        "request_attempt", attempt_id,
                        {
                            "attempt_id": attempt_id,
                            "request_hash": request_hash,
                            "outcome": "transport_failure",
                            "started_at": started,
                            "completed_at": ended,
                            "response_id": response_id,
                            "error": error,
                            "metadata": {"attempt_index": attempt_index},
                        },
                    )
                    retryable: bool = _retry_permitted(error, policy)
                    if not retryable or attempt_index + 1 >= policy["max_attempts"]:
                        return response
                    time.sleep(policy["backoff_seconds"][attempt_index])
                    continue

                ended = _now()
                response = self._provider_response(
                    provider, request, response_id, attempt_id, started, ended,
                    time.monotonic() - clock,
                )
                response = self.parse_response(response, schema_name)
                # Resolve cross-process completion races through authoritative storage.
                try:
                    self._store.create_record(
                        "model_response", response_id, response.model_dump(mode="json")
                    )
                except Exception:
                    authoritative = self._store.get_cached_response(request_hash)
                    if authoritative is None:
                        raise
                    self._store.save_blob(
                        _canonical_json(response.model_dump(mode="json")).encode("utf-8")
                    )
                    response = authoritative
                outcome: str = {
                    ResponseStatus.SUCCESS: "completed",
                    ResponseStatus.REFUSAL: "refusal",
                    ResponseStatus.INVALID_SCHEMA: "invalid_schema",
                }[response.status]
                self._store.create_record(
                    "request_attempt", attempt_id,
                    {
                        "attempt_id": attempt_id,
                        "request_hash": request_hash,
                        "outcome": outcome,
                        "started_at": started,
                        "completed_at": ended,
                        "response_id": response.response_id,
                        "metadata": {"attempt_index": attempt_index},
                    },
                )
                return response
            raise RuntimeError("No authorized inference attempt remains.")

    def _provider_response(
        self, provider: Any, request: dict[str, Any], response_id: str,
        attempt_id: str, started: str, ended: str, latency: float,
    ) -> ModelResponse:
        if hasattr(provider, "model_dump"):
            data: Any = provider.model_dump(mode="json")
        elif isinstance(provider, dict):
            data = provider
        else:
            raise RuntimeError("Provider returned an unsupported completion container.")
        if not isinstance(data, dict):
            raise RuntimeError("Provider completion must be an object.")
        # Persist a safe response allowlist, not provider headers/hidden SDK state.
        retained: dict[str, Any] = {
            key: data[key]
            for key in ("id", "model", "created", "choices", "usage", "system_fingerprint")
            if key in data
        }
        provider_blob: str = self._store.save_blob(
            _canonical_json(retained).encode("utf-8")
        )
        choices: Any = data.get("choices", [])
        errors: list[str] = []
        message: dict[str, Any] = {}
        finish: Any = None
        if not isinstance(choices, list) or len(choices) != 1:
            errors.append("Expected exactly one provider choice.")
        else:
            choice: Any = choices[0]
            if isinstance(choice, dict):
                supplied_message = choice.get("message")
                if not isinstance(supplied_message, dict):
                    errors.append("Provider message was not an object.")
                else:
                    message = supplied_message
                finish = choice.get("finish_reason")
            else:
                errors.append("Provider choice was not an object.")
        raw: Any = message.get("content", "")
        if raw is None:
            raw = ""
        if not isinstance(raw, str):
            errors.append("Provider content was not a text string.")
            raw = _canonical_json(raw)
        refusal: bool = bool(message.get("refusal")) or finish == "content_filter"
        usage: dict[str, Any] = {
            "request_hash": request["request_hash"],
            "attempt_id": attempt_id,
            "started_at": started,
            "completed_at": ended,
            "latency_seconds": latency,
            "provider_payload_blob": provider_blob,
            "provider_request_id": data.get("id"),
            "provider_reported_model": data.get("model"),
            "finish_reason": finish,
            "provider_usage": data.get("usage"),
            "usage_origin": "provider_reported",
            "provider_refusal": refusal,
            "provider_content_errors": errors,
            "schema_contract": request["provider_options"]["schema_contract"],
        }
        if (
            request["provider_options"]["probability_interface"] is not None
            and isinstance(choices, list) and len(choices) == 1
            and isinstance(choices[0], dict)
        ):
            usage["probability_artifact"] = {
                "interface": "token_logprobs",
                "data": choices[0].get("logprobs"),
            }
        return ModelResponse(
            response_id=response_id,
            instance_id=request["instance_id"],
            model_id=request["model_id"],
            run_index=request["run_index"],
            prompt_hash=request["prompt_hash"],
            config_hash=request["config_hash"],
            status=ResponseStatus.REFUSAL if refusal else ResponseStatus.INVALID_SCHEMA,
            raw_text=raw,
            usage=usage,
            schema_errors=errors or ([] if refusal else ["Output not yet parsed."]),
        )

    def parse_response(self, response: ModelResponse, schema_name: str) -> ModelResponse:
        """Validate the exact raw completion without repair or regeneration."""
        _identifier(schema_name)
        if response.status == ResponseStatus.TRANSPORT_FAILURE:
            return response
        if response.usage.get("provider_refusal") is True:
            return response.model_copy(
                update={"status": ResponseStatus.REFUSAL, "parsed_answer": {}, "schema_errors": []}
            )
        if response.usage.get("provider_content_errors"):
            return response.model_copy(update={
                "status": ResponseStatus.INVALID_SCHEMA,
                "parsed_answer": {},
                "schema_errors": response.usage["provider_content_errors"],
            })
        request_hash: Any = response.usage.get("request_hash")
        if not isinstance(request_hash, str):
            raise ValueError("Parsing requires the original stored request binding.")
        request: dict[str, Any] = self._store.read_record("request", request_hash)
        if request["schema_name"] != schema_name:
            raise ValueError("Parser schema differs from the stored request.")
        contract: dict[str, Any] = self._validate_contract(
            request["provider_options"]["schema_contract"]
        )
        payload: dict[str, Any] = response.model_dump(mode="python")
        try:
            parsed: dict[str, Any] = _strict_object(response.raw_text)
            _ValidatedOutput.model_validate(
                {"content": parsed},
                context={"json_schema": contract["json_schema"]},
            )
            if contract["kind"] == "classification":
                if set(parsed) != _CLASSIFICATION_KEYS:
                    raise ValueError("Classification output requires exactly three fields.")
                labels: Any = parsed["labels"]
                if not isinstance(labels, list):
                    raise ValueError("Classification labels must be an array.")
                taxonomy: TaxonomyVersion = self._taxonomy(contract["taxonomy_id"])
                if not labels and (
                    not contract["allow_empty_labels_when_uncertain"]
                    or parsed["uncertain"] is not True
                ):
                    raise ValueError("Empty classifications require authorized uncertainty.")
                spans: list[EvidenceSpan] = [
                    EvidenceSpan.model_validate(item) for item in parsed["evidence_spans"]
                ]
                assignment = LabelAssignment.model_validate(
                    {
                        "assignment_id": response.response_id,
                        "response_id": response.response_id,
                        "taxonomy_id": taxonomy.taxonomy_id,
                        "actor_id": response.model_id,
                        "round_index": 0,
                        "origin": "discovery" if contract["discovery_mode"] else "agent",
                        "labels": labels,
                        "evidence_spans": spans,
                        "uncertain": parsed["uncertain"],
                    },
                    context={
                        "taxonomy": taxonomy,
                        "require_taxonomy": True,
                        "require_frozen_taxonomy": not contract["discovery_mode"],
                        "discovery_mode": contract["discovery_mode"],
                        "fallback_label": self._config.taxonomy_discovery.fallback_label,
                        "require_evidence_for_each_label": True,
                    },
                )
                parsed["labels"] = assignment.labels
            elif contract["kind"] == "task":
                instance = TaskInstance.model_validate(
                    self._store.read_record("task_instance", response.instance_id)
                )
                for key, expected in (
                    ("task", instance.task_name),
                    ("case_id", instance.case_id),
                    ("snapshot_date", instance.snapshot_date.isoformat()),
                ):
                    if parsed.get(key) != expected:
                        raise ValueError(f"Task output identity mismatch: {key}.")
                justification: Any = parsed.get("justification")
                if not isinstance(justification, str) or len(justification.split()) > (
                    self._config.evaluation.prompt.maximum_justification_words
                ):
                    raise ValueError("Task justification violates its configured limit.")
            payload.update(
                status=ResponseStatus.SUCCESS,
                parsed_answer=parsed,
                schema_errors=[],
            )
            context: dict[str, Any] = {}
            if contract["kind"] == "task":
                context["allowed_answer_statuses"] = (
                    self._config.evaluation.prompt.allowed_statuses
                )
            return ModelResponse.model_validate(payload, context=context)
        except (ValueError, TypeError, KeyError, ValidationError) as exc:
            # Errors record locations/types without echoing potentially sensitive text.
            if isinstance(exc, ValidationError):
                errors: list[str] = [
                    ".".join(str(part) for part in item["loc"]) + ": " + item["type"]
                    for item in exc.errors(include_input=False, include_context=False)
                ]
            else:
                errors = [str(exc)] if not isinstance(exc, KeyError) else [
                    "Required output field is missing."
                ]
            payload.update(
                status=ResponseStatus.INVALID_SCHEMA,
                parsed_answer={},
                schema_errors=errors or ["Output validation failed."],
            )
            return ModelResponse.model_validate(payload)

    def _prepare_binding(self, envelope: Any) -> dict[str, Any]:
        if not isinstance(envelope, dict) or set(envelope) - _BATCH_KEYS:
            raise ValueError("Batch requests must use the documented envelope fields.")
        required: set[str] = {
            "model_id", "prompt", "run_index", "schema_name",
            "instance_id", "schema_contract",
        }
        if not required.issubset(envelope):
            raise ValueError("Batch request is missing an explicit inference binding field.")
        for name in ("model_id", "schema_name", "instance_id"):
            _identifier(envelope[name])
        _nonnegative_integer(envelope["run_index"], "run_index")
        prompt: Any = envelope["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Batch prompt must be a nonempty string.")
        contract: dict[str, Any] = self._validate_contract(envelope["schema_contract"])
        TaskInstance.model_validate(
            self._store.read_record("task_instance", envelope["instance_id"])
        )
        prompt_hash: str = _text_hash(prompt)
        provenance: Any = envelope.get("template_provenance")
        if provenance is None:
            with self._render_lock:
                provenance = copy.deepcopy(self._rendered.get(prompt_hash))
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError(
                "Prompt template provenance is required; render through render_prompt "
                "or provide the exact frozen provenance."
            )
        _reject_secrets(provenance)
        binding: dict[str, Any] = {
            "record_type": "inference_binding",
            "instance_id": envelope["instance_id"],
            "model_id": envelope["model_id"],
            "run_index": envelope["run_index"],
            "schema_name": envelope["schema_name"],
            "schema_contract": contract,
            "prompt_hash": prompt_hash,
            "config_hash": self._config.config_hash,
            "probability_interface": envelope.get("probability_interface"),
            "template_provenance": provenance,
        }
        binding["prompt_id"] = _content_hash(binding)
        return binding

    def generate_batch(self, requests: list[dict[str, Any]]) -> list[ModelResponse]:
        """Execute a prevalidated batch with bounded concurrency and stable order."""
        if not isinstance(requests, list):
            raise TypeError("requests must be a list.")
        if not requests:
            return []
        bindings: list[dict[str, Any]] = [
            self._prepare_binding(envelope) for envelope in requests
        ]
        self.validate_endpoints(list(dict.fromkeys(item["model_id"] for item in bindings)))
        # Preflight every request, including context and probability capabilities,
        # before any provider call.
        try:
            for envelope, binding in zip(requests, bindings, strict=True):
                self._local.binding = binding
                self._request(
                    envelope["model_id"], envelope["prompt"],
                    envelope["run_index"], envelope["schema_name"],
                )
        finally:
            self._local.binding = None
        for binding in bindings:
            self._store.create_record("prompt", binding["prompt_id"], binding)

        workers: int | None = self._config.inference.concurrency
        if workers is None:
            raise ValueError("inference.concurrency is unresolved.")

        def execute(index: int) -> ModelResponse:
            envelope: dict[str, Any] = requests[index]
            self._local.binding = bindings[index]
            try:
                return self.generate(
                    envelope["model_id"], envelope["prompt"],
                    envelope["run_index"], envelope["schema_name"],
                )
            finally:
                self._local.binding = None

        # Submit at most one window of futures. No application batching default
        # or GPU/provider-side batching behavior is inferred from batch_size.
        results: list[ModelResponse] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for start in range(0, len(requests), workers):
                stop: int = min(start + workers, len(requests))
                futures: list[Future[ModelResponse]] = [
                    executor.submit(execute, index) for index in range(start, stop)
                ]
                results.extend(future.result() for future in futures)
        return results
