"""Build provenance-complete, temporally controlled CTI task corpora.

This module implements only dataset preparation. It never generates scenarios,
infers benchmark selections, constructs prediction targets, or accesses models.

Manifest protocol
=================

All manifests are UTF-8 JSON objects or safe YAML objects with string keys.
Every manifest has an explicit nonempty ``version``.

``data.evidence_manifest`` is the acquisition manifest passed to
``ingest_sources``. It contains::

    {
      "version": "...",
      "sources": [{
        "source_id": "...",
        "version": "...",
        "dataset": "ctibench",       // null for auxiliary evidence
        "publisher": "...",
        "format": "jsonl",
        "encoding": "utf-8",
        "id_field": "/id",           // JSON Pointer into each parsed row
        "records_pointer": "",       // required for JSON; "" means root
        "location": {
          "kind": "local",
          "path": "...",
          "sha256": "..."            // optional expected raw-byte checksum
        },
        "parser_options": {}
      }],
      "evidence_records": [ /* schemas.EvidenceRecord objects */ ],
      "normalization_resources": {
        "alias_table": {"version": "...", "path": "...", "sha256": "..."},
        "attack": {"version": "...", "path": "...", "sha256": "..."}
      }
    }

Evidence ``source_id`` identifies an acquisition source; ``document_id`` is an
original item ID within that source. ``content_reference`` and ``content_hash``
must identify the exact acquired raw artifact's SHA-256 digest. Evidence snippets
must occur verbatim in their original parsed document. Input snippets additionally
declare ``provenance.selection_pointer``: a JSON Pointer identifying the original
field from which the snippet was selected. Text sources use the empty pointer.

Supported formats are json, jsonl, csv, parquet, hf_arrow, hf_parquet, and text.
CSV requires explicit pandas parser options, including ``sep``. Text requires
``item_id``. Hugging Face adapters operate on explicitly acquired local files;
they do not select repositories, configurations, or splits automatically.

HTTP locations must explicitly declare method, URL, params, optional JSON body,
``headers_from_env``, all four httpx timeout components, ``follow_redirects``,
``retry_policy: none``, ``pagination: none``, and ``maximum_bytes``. Credentials
are never saved. This adapter performs one declared request, not source-specific
pagination or historical-data reconstruction.

Each benchmark selection manifest contains::

    {"version": "...", "dataset": "...", "dataset_version": "...",
     "source_ids": ["..."], "item_ids": ["..."]}

The historical snapshot manifest contains::

    {"version": "...", "records": [{
      "evidence_id": "...", "source_id": "...", "document_id": "...",
      "content_hash": "...", "version_id": "...",
      "available_at": "YYYY-MM-DD or timezone-aware ISO timestamp"
    }]}

Availability is an assertion about these exact bytes, not the current URL.
Snapshot dates include the entire specified calendar day. Publication and actual
version modification dates must also be at or before the snapshot.

The conversion manifest contains ``conversions``. Every conversion explicitly
supplies instance_id, case_id, source_dataset, source_item_id, source_id,
task_name, stage, snapshot_date, input_evidence_ids, reference_id, candidates,
environment, output_profile, task_guidance, rule_id, rule_version,
transformation_history, and identifier_observations. No conversion is inferred.

References are supplied in a manifest's ``references`` list::

    {"reference_id": "...", "instance_id": "...", "answer": {...},
     "evidence_ids": ["..."],
     "horizon_days": 30, "target_construction": {...}}

The last two fields are required for forecasting tasks and must match the
configured reference protocol; no EPSS threshold is invented here.

Optional identifier observations declare kind, raw, evidence_id, start_offset,
end_offset, and unresolved_policy (reject or retain_unresolved). IOC observations
also declare ioc_type and an explicit ordered ``operations`` list. Supported
operations are replace, lowercase, url_host_lowercase, and strip_trailing_dot.
Normalization is recorded separately; original evidence is never rewritten.
Identifier observations may explicitly declare ``role: input_normalization``
for task-safe source linking. Omitted roles become ``provenance_only``. The
versioned identifier_normalization contract rejects task-target disclosure;
only successful, opted-in source transformations may enter structured feeds.

Alias resources contain ``version`` and ``entries`` with exact name, canonical,
entity_type, and evidence_refs fields. ATT&CK resources contain ``version`` and
``techniques`` with id, parent_id, revoked, and deprecated fields.

The supported deduplication policy is ``preserve_source_versions`` (a string or
an object with that ``mode``). It deduplicates storage and repeated acquisition,
not independent publishers, historical versions, or task instances.

A loaded TaskInstance includes evaluation references. It is NOT a prompt payload.
Prompt construction must use an explicit model-visible allowlist.
"""

from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import logging
import os
import re
from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import httpx
import yaml

from config import ExperimentConfig
from identifier_normalization import NORMALIZATION_CONTRACT, observation_role
from schemas import (
    CTIStage,
    EvidenceRecord,
    EvidenceRole,
    StageState,
    StageStatus,
    TaskInstance,
    _aware_datetime,
    _calendar_date,
    _canonical_json,
    _content_hash,
    _identifier,
    _json_primitive,
    _source_time,
    _task_content_projection,
)
from storage import ArtifactStore


_LOGGER = logging.getLogger(__name__)
_SUPPORTED_FORMATS: frozenset[str] = frozenset(
    {"json", "jsonl", "csv", "parquet", "hf_arrow", "hf_parquet", "text"}
)
_FORECAST_TASKS: frozenset[str] = frozenset(
    {"exploit_likelihood", "campaign_escalation"}
)
_SECRET_KEYS: frozenset[str] = frozenset(
    {
        "api_key", "apikey", "authorization", "password", "access_token",
        "refresh_token", "client_secret", "private_key", "credentials",
        "bearer_token",
    }
)
_CVE_PATTERN = re.compile(r"CVE-[0-9]{4}-[0-9]{4,}", re.IGNORECASE)
_ATTACK_PATTERN = re.compile(r"T[0-9]{4}(?:\.[0-9]{3})?", re.IGNORECASE)
_HASH_PATTERN = re.compile(r"[0-9a-fA-F]+")
_DOMAIN_PATTERN = re.compile(
    r"(?=.{1,253}\Z)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.?"
)


class _ManifestLoader(yaml.SafeLoader):
    """Safe YAML preserving ISO date strings and rejecting duplicate keys."""


_ManifestLoader.yaml_implicit_resolvers = {
    key: [
        (tag, expression)
        for tag, expression in resolvers
        if tag != "tag:yaml.org,2002:timestamp"
    ]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _yaml_mapping(
    loader: _ManifestLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise ValueError("Manifest YAML merge keys are not supported.")
        key: Any = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError("Manifest keys must be unique strings.")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_ManifestLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _yaml_mapping
)


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key.")
        result[key] = value
    return result


def _parse_json(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Nonfinite JSON number is forbidden: {value}.")

    return json.loads(
        text, object_pairs_hook=_json_pairs, parse_constant=reject_constant
    )


def _required(record: Mapping[str, Any], key: str) -> Any:
    if key not in record or record[key] is None:
        raise ValueError(f"Missing required manifest field: {key}.")
    return record[key]


def _object(value: Any, description: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object.")
    return value


def _objects(value: Any, description: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(
        not isinstance(item, dict) for item in value
    ):
        raise ValueError(f"{description} must be a list of objects.")
    return value


def _strings(value: Any, description: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{description} must be a list.")
    result: list[str] = [_identifier(item) for item in value]
    if len(result) != len(set(result)):
        raise ValueError(f"{description} must not contain duplicates.")
    return result


def _pointer(value: Any, pointer: str) -> Any:
    """Resolve RFC 6901 JSON Pointers without fuzzy field matching."""
    if not isinstance(pointer, str):
        raise ValueError("A JSON Pointer must be a string.")
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("A nonempty JSON Pointer must start with '/'.")
    current: Any = value
    for part in pointer[1:].split("/"):
        if re.search(r"~(?![01])", part):
            raise ValueError("Malformed JSON Pointer escape.")
        token: str = part.replace("~1", "/").replace("~0", "~")
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
            raise ValueError(f"JSON Pointer does not resolve: {pointer}.")
    return current


def _original_id(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("Original source IDs must be strings or integers.")
    return _identifier(str(value))


def _record_id(source_id: str, version: str, item_id: str) -> str:
    return _content_hash(
        {"source_id": source_id, "version": version, "original_item_id": item_id}
    )


def _date_of(value: Any) -> date:
    parsed: date | datetime = _source_time(value)
    return parsed.date() if isinstance(parsed, datetime) else parsed


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _check_credentials(value: Any) -> None:
    """Inspect structured acquisition settings, never rewrite source documents."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower().replace("-", "_") in _SECRET_KEYS and item is not None:
                raise ValueError("Literal credentials are forbidden in manifests.")
            _check_credentials(item)
    elif isinstance(value, list):
        for item in value:
            _check_credentials(item)


class DatasetLoader:
    """Prepare immutable source artifacts and explicit CTI task conversions."""

    def __init__(self, config: ExperimentConfig, store: ArtifactStore) -> None:
        self._config: ExperimentConfig = config
        self._store: ArtifactStore = store
        # config.py owns configuration-relative path semantics.
        source_path: Path | None = getattr(config, "_source_path", None)
        self._base: Path = (
            source_path.parent if source_path is not None else Path.cwd()
        )

    def _path(self, value: Any, base: Path | None = None) -> Path:
        path_text: str = _identifier(value)
        if "://" in path_text or path_text.startswith(("sha256:", "urn:")):
            raise ValueError(
                "This adapter requires a local manifest or an explicitly "
                "declared HTTP source; no remote-path fallback is performed."
            )
        path: Path = Path(path_text).expanduser()
        return (path if path.is_absolute() else (base or self._base) / path).resolve()

    def _manifest(self, path: Path) -> tuple[dict[str, Any], str]:
        if not path.is_file():
            raise FileNotFoundError(f"Manifest is missing: {path}.")
        raw: bytes = path.read_bytes()
        try:
            text: str = raw.decode("utf-8")
            if path.suffix.lower() == ".json":
                payload: Any = _parse_json(text)
            else:
                documents: list[Any] = list(
                    yaml.load_all(text, Loader=_ManifestLoader)
                )
                if len(documents) != 1:
                    raise ValueError("Exactly one manifest document is required.")
                payload = documents[0]
            result: dict[str, Any] = _object(payload, "Manifest")
            _identifier(_required(result, "version"))
            _canonical_json(result)
        except (ValueError, UnicodeError, yaml.YAMLError, RecursionError):
            raise ValueError(f"Invalid manifest structure: {path}.") from None
        digest: str = self._store.save_blob(raw)
        return result, digest

    def _bundle(self) -> dict[str, Any]:
        """Read declared manifests, preserving each exact artifact as a blob."""
        data = self._config.data
        paths: dict[str, str | None] = {
            "acquisition": data.evidence_manifest,
            "snapshots": data.historical_snapshot_manifest,
            "conversions": data.task_conversion_manifest,
            "references": data.reference_answer_manifest,
        }
        selected: list[str] = (
            self._config.operational.selected_benchmarks
            or list(data.benchmarks)
        )
        for dataset in selected:
            paths[f"selection:{dataset}"] = data.benchmarks[
                dataset
            ].selection_manifest
        bundle: dict[str, Any] = {}
        for name, value in paths.items():
            if value is None:
                raise ValueError(f"Unresolved preparation manifest: {name}.")
            path: Path = self._path(value)
            content, digest = self._manifest(path)
            bundle[name] = {
                "path": str(path),
                "sha256": digest,
                "content": content,
            }
        return bundle

    def _execution_id(self, phase: str) -> str:
        return _content_hash(
            {
                "phase": phase,
                "corpus_version": self._config.corpus_version,
                "config_hash": self._config.config_hash,
            }
        )

    def _checkpoint(
        self,
        phase: str,
        status: StageStatus,
        completed: list[str],
        pending: list[str],
        manifests: dict[str, str],
        error: str | None = None,
    ) -> None:
        stage: str = f"prepare.{phase}"
        execution_id: str = self._execution_id(phase)
        previous: list[StageState] = [
            StageState.model_validate(payload)
            for payload in self._store.list_records("stage_state")
            if payload.get("stage") == stage
            and payload.get("execution_id") == execution_id
        ]
        previous.sort(
            key=lambda item: item.updated_at or item.started_at
            or datetime.min.replace(tzinfo=timezone.utc)
        )
        old: StageState | None = previous[-1] if previous else None
        if old is not None and old.status == StageStatus.COMPLETED:
            return
        timestamp: datetime = _now()
        completed_set: set[str] = set(completed)
        if old is not None:
            completed_set.update(old.completed_work)
        state = StageState(
            stage=stage,
            execution_id=execution_id,
            config_hash=self._config.config_hash,
            status=status,
            completed_work=sorted(completed_set),
            pending_work=sorted(set(pending) - completed_set),
            input_manifests=manifests,
            started_at=old.started_at if old is not None else timestamp,
            updated_at=timestamp,
            completed_at=timestamp if status == StageStatus.COMPLETED else None,
            errors=[] if error is None else [{"operation": phase, "error": error}],
            metadata={"corpus_version": self._config.corpus_version},
        )
        self._store.set_stage_state(stage, state.model_dump(mode="json"))

    def _persist_manifest(
        self, purpose: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "manifest_id": self._execution_id(purpose),
            "purpose": purpose,
            "corpus_version": self._config.corpus_version,
            "config_hash": self._config.config_hash,
            **payload,
        }
        self._store.create_record(
            "experiment_manifest", record["manifest_id"], record
        )
        return record

    def _ingestion(self) -> dict[str, Any]:
        try:
            return self._store.read_record(
                "experiment_manifest", self._execution_id("ingestion")
            )
        except KeyError:
            raise ValueError(
                "Source ingestion is incomplete for this configuration; "
                "run ingest_sources(data.evidence_manifest) first."
            ) from None

    def _assert_bundle(self, bundle: dict[str, Any]) -> dict[str, Any]:
        ingestion: dict[str, Any] = self._ingestion()
        hashes: dict[str, str] = {
            key: value["sha256"] for key, value in bundle.items()
        }
        if ingestion["manifest_hashes"] != hashes:
            raise ValueError(
                "Preparation manifests changed after ingestion. Use a new "
                "configuration/corpus version and re-ingest."
            )
        return ingestion

    def _acquire(
        self, source: dict[str, Any], base: Path
    ) -> tuple[bytes, dict[str, Any]]:
        location: dict[str, Any] = _object(
            _required(source, "location"), "Source location"
        )
        _check_credentials(location)
        kind: Any = _required(location, "kind")
        provenance: dict[str, Any]
        if kind == "local":
            path: Path = self._path(_required(location, "path"), base)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Source {source['source_id']!r}: local artifact is missing."
                )
            content: bytes = path.read_bytes()
            provenance = {"kind": "local", "path": str(path)}
        elif kind == "http":
            url: str = _identifier(_required(location, "url"))
            parsed = urlsplit(url)
            query_keys: set[str] = {
                key.lower().replace("-", "_")
                for key, _ in parse_qsl(parsed.query)
            }
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or query_keys & _SECRET_KEYS
            ):
                raise ValueError("HTTP source URL is invalid or credential-bearing.")
            if _required(location, "retry_policy") != "none":
                raise ValueError("This acquisition adapter supports retry_policy: none.")
            if _required(location, "pagination") != "none":
                raise ValueError("Paginated acquisitions require a resolved adapter.")
            follow: Any = _required(location, "follow_redirects")
            if not isinstance(follow, bool):
                raise ValueError("follow_redirects must be explicitly boolean.")
            maximum: Any = _required(location, "maximum_bytes")
            if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
                raise ValueError("maximum_bytes must be an explicit positive integer.")
            timeout_values: dict[str, Any] = _object(
                _required(location, "timeout"), "HTTP timeout"
            )
            if set(timeout_values) != {"connect", "read", "write", "pool"}:
                raise ValueError("Declare connect/read/write/pool timeouts explicitly.")
            for value in timeout_values.values():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("HTTP timeouts must be positive finite numbers.")
                if not 0 < float(value) < float("inf"):
                    raise ValueError("HTTP timeouts must be positive finite numbers.")
            method: str = _identifier(_required(location, "method")).upper()
            if method not in {"GET", "POST"}:
                raise ValueError("Only explicitly declared GET/POST acquisition is supported.")
            params: dict[str, Any] = _object(
                _required(location, "params"), "HTTP params"
            )
            header_env: dict[str, Any] = _object(
                _required(location, "headers_from_env"), "HTTP environment headers"
            )
            headers: dict[str, str] = {}
            for header, environment_name in header_env.items():
                variable: str = _identifier(environment_name)
                if variable not in os.environ:
                    raise ValueError(
                        f"Source {source['source_id']!r}: required credential "
                        "environment variable is unavailable."
                    )
                headers[header] = os.environ[variable]
            try:
                timeout = httpx.Timeout(**timeout_values)
                with httpx.Client(
                    timeout=timeout,
                    follow_redirects=follow,
                    transport=httpx.HTTPTransport(retries=0),
                    trust_env=False,
                ) as client:
                    with client.stream(
                        method,
                        url,
                        params=params,
                        json=location.get("json_body"),
                        headers=headers,
                    ) as response:
                        response.raise_for_status()
                        buffer = io.BytesIO()
                        for chunk in response.iter_bytes():
                            if buffer.tell() + len(chunk) > maximum:
                                raise ValueError("Acquired source exceeds maximum_bytes.")
                            buffer.write(chunk)
                        content = buffer.getvalue()
            except httpx.HTTPError:
                raise RuntimeError(
                    f"Acquisition failed for source {source['source_id']!r}; "
                    "provider error details were withheld to protect credentials."
                ) from None
            provenance = {
                "kind": "http",
                "url": url,
                "method": method,
                "params": params,
                "json_body": location.get("json_body"),
                "headers_from_env": header_env,
                "timeout": timeout_values,
                "follow_redirects": follow,
                "retry_policy": "none",
                "pagination": "none",
                "maximum_bytes": maximum,
            }
        else:
            raise ValueError(f"Unsupported acquisition kind: {kind!r}.")
        expected: Any = location.get("sha256")
        actual: str = hashlib.sha256(content).hexdigest()
        if expected is not None and expected != actual:
            raise ValueError(f"Source {source['source_id']!r}: checksum mismatch.")
        return content, provenance

    def _parse_source(
        self, source: dict[str, Any], content: bytes
    ) -> list[tuple[str, Any]]:
        format_name: str = _identifier(_required(source, "format"))
        if format_name not in _SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported source format: {format_name!r}.")
        encoding: str = _identifier(_required(source, "encoding"))
        options: dict[str, Any] = _object(
            _required(source, "parser_options"), "Parser options"
        )
        rows: list[Any]
        if format_name == "text":
            if options:
                raise ValueError("Text parser_options must be empty.")
            return [
                (
                    _original_id(_required(source, "item_id")),
                    content.decode(encoding),
                )
            ]
        if format_name == "json":
            if options:
                raise ValueError("JSON parser_options must be empty.")
            root: Any = _parse_json(content.decode(encoding))
            rows_value: Any = _pointer(
                root, _required(source, "records_pointer")
            )
            if not isinstance(rows_value, list):
                raise ValueError("JSON records_pointer must select a list.")
            rows = rows_value
        elif format_name == "jsonl":
            if options:
                raise ValueError("JSONL parser_options must be empty.")
            rows = []
            for line_number, line in enumerate(content.decode(encoding).splitlines(), 1):
                if not line.strip():
                    raise ValueError(
                        f"Blank JSONL record at line {line_number}; "
                        "no silent record omission is allowed."
                    )
                rows.append(_parse_json(line))
        elif format_name == "csv":
            import pandas as pd

            if "sep" not in options:
                raise ValueError("CSV parser_options must explicitly supply sep.")
            forbidden: set[str] = {
                "dtype", "converters", "encoding", "keep_default_na",
                "na_filter", "on_bad_lines", "skiprows", "nrows",
            }
            if set(options) & forbidden:
                raise ValueError(
                    "CSV options cannot coerce values or silently skip/subsample rows."
                )
            frame = pd.read_csv(
                io.StringIO(content.decode(encoding)),
                dtype=str,
                keep_default_na=False,
                na_filter=False,
                on_bad_lines="error",
                **options,
            )
            rows = frame.to_dict(orient="records")
        elif format_name == "parquet":
            import pandas as pd

            if options:
                raise ValueError("Parquet parser_options must be empty.")
            try:
                frame = pd.read_parquet(io.BytesIO(content))
            except ImportError:
                raise RuntimeError(
                    "The declared Parquet adapter requires a separately installed "
                    "and versioned pandas Parquet engine."
                ) from None
            # JSON serialization exposes unsupported/nonfinite values rather
            # than replacing them with null or coercing source semantics.
            rows = frame.to_dict(orient="records")
        else:
            from datasets import Dataset

            if options:
                raise ValueError("Hugging Face file parser_options must be empty.")
            digest: str = hashlib.sha256(content).hexdigest()
            root: Path = Path(
                self._config.operational.artifact_root or ""
            ).expanduser()
            if not root.is_absolute():
                root = self._base / root
            # save_blob uses the shared storage layout; create an explicit
            # temporary parse file because Arrow/Parquet readers require paths.
            import tempfile

            suffix: str = ".arrow" if format_name == "hf_arrow" else ".parquet"
            with tempfile.TemporaryDirectory(prefix="cti-source-") as directory:
                temporary: Path = Path(directory) / f"{digest}{suffix}"
                temporary.write_bytes(content)
                dataset = (
                    Dataset.from_file(str(temporary))
                    if format_name == "hf_arrow"
                    else Dataset.from_parquet(str(temporary))
                )
                rows = list(dataset)

        id_pointer: str = _required(source, "id_field")
        result: list[tuple[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            _canonical_json(row)
            item_id: str = _original_id(_pointer(row, id_pointer))
            if item_id in seen:
                raise ValueError(
                    f"Source {source['source_id']!r}: duplicate original item ID."
                )
            seen.add(item_id)
            result.append((item_id, row))
        return result

    def _resources(
        self, acquisition: dict[str, Any], base: Path
    ) -> dict[str, Any]:
        declarations: dict[str, Any] = _object(
            acquisition.get("normalization_resources", {}),
            "Normalization resources",
        )
        result: dict[str, Any] = {}
        for name, declaration_value in declarations.items():
            if name not in {"alias_table", "attack"}:
                raise ValueError(f"Unsupported normalization resource: {name}.")
            declaration: dict[str, Any] = _object(
                declaration_value, "Normalization resource"
            )
            version: str = _identifier(_required(declaration, "version"))
            payload, digest = self._manifest(
                self._path(_required(declaration, "path"), base)
            )
            if digest != _required(declaration, "sha256"):
                raise ValueError(f"{name} resource checksum mismatch.")
            if payload["version"] != version:
                raise ValueError(f"{name} resource version mismatch.")
            result[name] = {"version": version, "sha256": digest, "content": payload}

        configured_alias: str | None = (
            self._config.data.preprocessing.alias_table_version
        )
        if configured_alias is not None:
            if result.get("alias_table", {}).get("version") != configured_alias:
                raise ValueError("Configured alias_table_version lacks its exact resource.")
        versions: Any = self._config.data.preprocessing.taxonomy_versions
        attack_version: Any = (
            versions.get("attack") if isinstance(versions, dict) else None
        )
        if attack_version is not None:
            if not isinstance(attack_version, str):
                raise ValueError("taxonomy_versions.attack must be an explicit version string.")
            if result.get("attack", {}).get("version") != attack_version:
                raise ValueError("Configured ATT&CK version lacks its exact resource.")
        return result

    def ingest_sources(self, manifest_path: str) -> None:
        """Ingest declared sources and archive all preparation manifests.

        Repeated ingestion reuses immutable acquired bytes. It does not refresh a
        live endpoint under an existing source identity.
        """
        self._config.validate("prepare")
        configured: str | None = self._config.data.evidence_manifest
        if configured is None or self._path(manifest_path) != self._path(configured):
            raise ValueError(
                "ingest_sources must use the configured data.evidence_manifest."
            )
        bundle: dict[str, Any] = self._bundle()
        acquisition: dict[str, Any] = bundle["acquisition"]["content"]
        sources: list[dict[str, Any]] = _objects(
            _required(acquisition, "sources"), "Acquisition sources"
        )
        source_ids: list[str] = [
            _identifier(_required(source, "source_id")) for source in sources
        ]
        if not source_ids or len(source_ids) != len(set(source_ids)):
            raise ValueError("Acquisition source IDs must be nonempty and unique.")
        policy: Any = self._config.data.preprocessing.deduplication_policy
        mode: Any = policy.get("mode") if isinstance(policy, dict) else policy
        if mode != "preserve_source_versions":
            raise ValueError(
                "Unsupported deduplication_policy; this adapter implements "
                "preserve_source_versions only."
            )
        hashes: dict[str, str] = {
            name: item["sha256"] for name, item in bundle.items()
        }
        completed: list[str] = []
        source_records: list[str] = []
        source_hashes: dict[str, str] = {}
        base: Path = self._path(manifest_path).parent
        try:
            self._checkpoint(
                "ingest", StageStatus.RUNNING, completed, source_ids, hashes
            )
            for source in sources:
                source_id: str = source["source_id"]
                version: str = _identifier(_required(source, "version"))
                _identifier(_required(source, "publisher"))
                dataset: Any = source.get("dataset")
                if dataset is not None:
                    if dataset not in self._config.data.benchmarks:
                        raise ValueError(f"Unknown benchmark dataset: {dataset!r}.")
                    benchmark = self._config.data.benchmarks[dataset]
                    if benchmark.version != version:
                        raise ValueError("Acquired benchmark version differs from configuration.")
                    location: dict[str, Any] = _object(
                        _required(source, "location"), "Source location"
                    )
                    if location.get("kind") == "local":
                        if self._path(location["path"], base) != self._path(
                            benchmark.source_path
                        ):
                            raise ValueError("Benchmark source path differs from configuration.")
                try:
                    existing: dict[str, Any] = self._store.read_record("source", source_id)
                except KeyError:
                    existing = {}
                if existing:
                    if (
                        existing.get("record_type") != "acquisition"
                        or existing.get("declaration_hash") != _content_hash(source)
                    ):
                        raise ValueError(
                            f"Immutable source identity/content conflict: {source_id}."
                        )
                    raw_hash: str = existing["raw_blob_hash"]
                    item_ids: list[str] = existing["parsed_record_ids"]
                    if source["location"]["kind"] == "local":
                        current: bytes = self._path(
                            source["location"]["path"], base
                        ).read_bytes()
                        if hashlib.sha256(current).hexdigest() != raw_hash:
                            raise ValueError(
                                f"Local source changed under immutable identity: {source_id}."
                            )
                    for item_record_id in item_ids:
                        self._store.read_record("source", item_record_id)
                else:
                    content, provenance = self._acquire(source, base)
                    raw_hash = self._store.save_blob(content)
                    rows: list[tuple[str, Any]] = self._parse_source(source, content)
                    item_ids = []
                    for item_id, original in rows:
                        identity: str = _record_id(source_id, version, item_id)
                        parsed_record: dict[str, Any] = {
                            "source_id": identity,
                            "record_type": "original_item",
                            "acquisition_source_id": source_id,
                            "source_version": version,
                            "dataset": dataset,
                            "original_item_id": item_id,
                            "raw_blob_hash": raw_hash,
                            "original": original,
                        }
                        self._store.create_record("source", identity, parsed_record)
                        item_ids.append(identity)
                    acquisition_record: dict[str, Any] = {
                        "source_id": source_id,
                        "record_type": "acquisition",
                        "source_version": version,
                        "dataset": dataset,
                        "publisher": source["publisher"],
                        "format": source["format"],
                        "raw_blob_hash": raw_hash,
                        "declaration_hash": _content_hash(source),
                        "parsed_record_ids": sorted(item_ids),
                        "acquisition": provenance,
                    }
                    self._store.create_record("source", source_id, acquisition_record)
                source_records.extend([source_id, *item_ids])
                source_hashes[source_id] = raw_hash
                completed.append(source_id)
                self._checkpoint(
                    "ingest", StageStatus.RUNNING, completed, source_ids, hashes
                )

            resources: dict[str, Any] = self._resources(acquisition, base)
            source_freeze: str = self._store.freeze_records(
                "source", sorted(set(source_records))
            )
            self._persist_manifest(
                "ingestion",
                {
                    "manifest_hashes": hashes,
                    "source_records": sorted(set(source_records)),
                    "source_hashes": source_hashes,
                    "source_freeze_hash": source_freeze,
                    "normalization_resources": resources,
                },
            )
            self._checkpoint(
                "ingest", StageStatus.COMPLETED, source_ids, [], hashes
            )
        except Exception as exc:
            self._checkpoint(
                "ingest", StageStatus.FAILED, completed, source_ids, hashes,
                error=type(exc).__name__,
            )
            raise

    def _indexes(
        self, bundle: dict[str, Any]
    ) -> tuple[
        dict[str, EvidenceRecord],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
    ]:
        acquisition: dict[str, Any] = bundle["acquisition"]["content"]
        evidence: dict[str, EvidenceRecord] = {}
        for payload in _objects(
            _required(acquisition, "evidence_records"), "Evidence records"
        ):
            record = EvidenceRecord.model_validate(payload)
            if record.evidence_id in evidence:
                raise ValueError("Duplicate evidence_id in acquisition manifest.")
            evidence[record.evidence_id] = record

        def index(name: str, list_key: str, id_key: str) -> dict[str, dict[str, Any]]:
            result: dict[str, dict[str, Any]] = {}
            for entry in _objects(
                _required(bundle[name]["content"], list_key), list_key
            ):
                identity: str = _identifier(_required(entry, id_key))
                if identity in result:
                    raise ValueError(f"Duplicate {id_key} in {name} manifest.")
                result[identity] = entry
            return result

        return (
            evidence,
            index("snapshots", "records", "evidence_id"),
            index("references", "references", "reference_id"),
            index("conversions", "conversions", "instance_id"),
        )

    def _original(
        self, source_id: str, item_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        source: dict[str, Any] = self._store.read_record("source", source_id)
        if source.get("record_type") != "acquisition":
            raise ValueError("Evidence source_id must identify an acquisition.")
        identity: str = _record_id(source_id, source["source_version"], item_id)
        original: dict[str, Any] = self._store.read_record("source", identity)
        if original.get("acquisition_source_id") != source_id:
            raise ValueError("Original item does not belong to the declared source.")
        return source, original

    def _check_evidence(
        self,
        evidence: EvidenceRecord,
        snapshots: dict[str, dict[str, Any]],
        snapshot: date | None,
    ) -> None:
        source, original = self._original(evidence.source_id, evidence.document_id)
        raw_hash: str = source["raw_blob_hash"]
        if evidence.content_hash != raw_hash or evidence.content_reference not in {
            raw_hash, f"sha256:{raw_hash}"
        }:
            raise ValueError("Evidence does not identify its exact acquired artifact.")
        if evidence.source_version != source["source_version"]:
            raise ValueError("Evidence source_version differs from its acquired version.")
        if evidence.snippet is None or not evidence.snippet.strip():
            raise ValueError("Selected evidence must contain an explicit nonempty snippet.")

        selection_pointer: Any = evidence.provenance.get("selection_pointer")
        if not isinstance(selection_pointer, str):
            raise ValueError("Evidence requires provenance.selection_pointer.")
        selected: Any = _pointer(original["original"], selection_pointer)
        if not isinstance(selected, str) or evidence.snippet not in selected:
            raise ValueError("Evidence snippet is not verbatim in its selected source field.")

        if snapshot is None:
            return
        proof: dict[str, Any] | None = snapshots.get(evidence.evidence_id)
        if proof is None:
            raise ValueError("Model-visible evidence lacks historical-version proof.")
        for field, expected in (
            ("source_id", evidence.source_id),
            ("document_id", evidence.document_id),
            ("content_hash", raw_hash),
        ):
            if _required(proof, field) != expected:
                raise ValueError("Historical proof identifies different source content.")
        _identifier(_required(proof, "version_id"))
        if _date_of(_required(proof, "available_at")) > snapshot:
            raise ValueError("Evidence version was unavailable at the snapshot.")
        if evidence.publication_time is None:
            raise ValueError("Visible evidence lacks publication metadata.")
        if _date_of(evidence.publication_time) > snapshot:
            raise ValueError("Visible evidence was published after the snapshot.")
        if (
            evidence.modification_time is not None
            and _date_of(evidence.modification_time) > snapshot
        ):
            raise ValueError(
                "Post-snapshot modification cannot masquerade as a historical version."
            )

    def _selection(
        self, bundle: dict[str, Any]
    ) -> dict[str, set[tuple[str, str]]]:
        selections: dict[str, set[tuple[str, str]]] = {}
        for name, wrapper in bundle.items():
            if not name.startswith("selection:"):
                continue
            dataset: str = name.split(":", 1)[1]
            content: dict[str, Any] = wrapper["content"]
            if _required(content, "dataset") != dataset:
                raise ValueError("Selection manifest dataset mismatch.")
            if _required(content, "dataset_version") != self._config.data.benchmarks[
                dataset
            ].version:
                raise ValueError("Selection manifest benchmark version mismatch.")
            source_ids: list[str] = _strings(
                _required(content, "source_ids"), "Selection source_ids"
            )
            item_ids: list[str] = _strings(
                _required(content, "item_ids"), "Selection item_ids"
            )
            selected: set[tuple[str, str]] = set()
            for item_id in item_ids:
                matches: list[str] = []
                for source_id in source_ids:
                    try:
                        source, original = self._original(source_id, item_id)
                    except KeyError:
                        continue
                    if source.get("dataset") != dataset:
                        raise ValueError("Selected source belongs to another dataset.")
                    if original["dataset"] == dataset:
                        matches.append(source_id)
                if len(matches) != 1:
                    raise ValueError(
                        "Each selected item must resolve to exactly one declared source."
                    )
                selected.add((matches[0], item_id))
            selections[dataset] = selected
        return selections

    def _conversions(
        self, bundle: dict[str, Any]
    ) -> list[dict[str, Any]]:
        _, _, _, conversions = self._indexes(bundle)
        selected_tasks: list[str] | None = self._config.operational.selected_tasks
        selections: dict[str, set[tuple[str, str]]] = self._selection(bundle)
        result: list[dict[str, Any]] = []
        covered: dict[str, set[tuple[str, str]]] = {
            name: set() for name in selections
        }
        for conversion in conversions.values():
            dataset: Any = _required(conversion, "source_dataset")
            task: Any = _required(conversion, "task_name")
            if dataset not in selections:
                continue
            if selected_tasks is not None and task not in selected_tasks:
                continue
            pair: tuple[str, str] = (
                _identifier(_required(conversion, "source_id")),
                _original_id(_required(conversion, "source_item_id")),
            )
            if pair not in selections[dataset]:
                raise ValueError("Conversion references an item outside the frozen selection.")
            covered[dataset].add(pair)
            result.append(conversion)
        for dataset, selected in selections.items():
            if selected - covered[dataset]:
                raise ValueError(
                    f"Selected {dataset} items lack conversions for this preparation "
                    "scope. Supply a matching explicit selection manifest."
                )
        if not result:
            raise ValueError("No declared task conversions match the preparation scope.")
        return sorted(result, key=lambda item: item["instance_id"])

    def _normalize_ioc(self, observation: dict[str, Any]) -> str:
        value: str = observation["raw"]
        operations: list[dict[str, Any]] = _objects(
            _required(observation, "operations"), "IOC operations"
        )
        for operation in operations:
            name: Any = _required(operation, "operation")
            if name == "replace":
                old: Any = _required(operation, "old")
                new: Any = _required(operation, "new")
                if not isinstance(old, str) or not old or not isinstance(new, str):
                    raise ValueError("IOC replacements require nonempty old and string new.")
                value = value.replace(old, new)
            elif name == "lowercase":
                if observation["ioc_type"] not in {"domain", "hash"}:
                    raise ValueError("Whole-value lowercasing is forbidden for URLs.")
                value = value.lower()
            elif name == "strip_trailing_dot":
                if observation["ioc_type"] != "domain":
                    raise ValueError("Trailing-dot normalization is domain-specific.")
                value = value[:-1] if value.endswith(".") else value
            elif name == "url_host_lowercase":
                parts = urlsplit(value)
                if parts.username is not None or parts.password is not None:
                    raise ValueError("Credential-bearing IOC URLs require a separate protocol.")
                if not parts.hostname:
                    raise ValueError("URL lacks a hostname.")
                host: str = parts.hostname.lower()
                host = f"[{host}]" if ":" in host else host
                netloc: str = host + (f":{parts.port}" if parts.port is not None else "")
                value = urlunsplit(
                    (parts.scheme, netloc, parts.path, parts.query, parts.fragment)
                )
            else:
                raise ValueError(f"Unsupported IOC normalization operation: {name!r}.")
        ioc_type: str = _identifier(_required(observation, "ioc_type"))
        if ioc_type in {"ipv4", "ipv6"}:
            address = ipaddress.ip_address(value)
            if address.version != (4 if ioc_type == "ipv4" else 6):
                raise ValueError("IOC IP type and address disagree.")
            value = str(address)
        elif ioc_type == "domain":
            if _DOMAIN_PATTERN.fullmatch(value) is None:
                raise ValueError("Ambiguous or malformed domain IOC.")
        elif ioc_type == "url":
            parts = urlsplit(value)
            if not parts.scheme or not parts.hostname:
                raise ValueError("URL IOC requires an explicit scheme and hostname.")
            _ = parts.port
        elif ioc_type == "hash":
            lengths: Any = _required(observation, "allowed_hash_lengths")
            if (
                not isinstance(lengths, list)
                or any(isinstance(item, bool) or not isinstance(item, int) for item in lengths)
                or len(value) not in lengths
                or _HASH_PATTERN.fullmatch(value) is None
            ):
                raise ValueError("Hash IOC violates the declared length/alphabet policy.")
        else:
            raise ValueError(f"Unsupported IOC type: {ioc_type!r}.")
        return value

    def _normalizations(
        self,
        conversion: dict[str, Any],
        evidence: dict[str, EvidenceRecord],
        resources: dict[str, Any],
    ) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = _objects(
            _required(conversion, "identifier_observations"),
            "Identifier observations",
        )
        result: list[dict[str, Any]] = []
        for observation in observations:
            role: str = observation_role(conversion["task_name"], observation)
            kind: str = _identifier(_required(observation, "kind"))
            raw: str = _identifier(_required(observation, "raw"))
            evidence_id: str = _identifier(_required(observation, "evidence_id"))
            if evidence_id not in evidence:
                raise ValueError("Identifier observation references non-visible evidence.")
            record: EvidenceRecord = evidence[evidence_id]
            start: Any = _required(observation, "start_offset")
            end: Any = _required(observation, "end_offset")
            if (
                isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, int) or not isinstance(end, int)
                or start < 0 or end <= start
                or (record.snippet or "")[start:end] != raw
            ):
                raise ValueError("Identifier observation offsets do not match raw evidence.")
            unresolved: Any = _required(observation, "unresolved_policy")
            if unresolved not in {"reject", "retain_unresolved"}:
                raise ValueError("Declare reject or retain_unresolved for identifiers.")
            normalized: dict[str, Any] = {
                "observation": observation,
                "role": role,
                "raw": raw,
                "evidence_id": evidence_id,
                "status": "preserved",
                "canonical": raw,
            }
            try:
                if kind == "attack":
                    if _ATTACK_PATTERN.fullmatch(raw) is None:
                        raise ValueError("Malformed ATT&CK identifier.")
                    resource: Any = resources.get("attack")
                    if resource is None:
                        raise ValueError("ATT&CK validation requires its configured resource.")
                    techniques: list[dict[str, Any]] = _objects(
                        _required(resource["content"], "techniques"), "ATT&CK techniques"
                    )
                    registry: dict[str, dict[str, Any]] = {}
                    for technique in techniques:
                        identifier: str = _identifier(_required(technique, "id"))
                        if identifier in registry:
                            raise ValueError("Duplicate ATT&CK resource identifier.")
                        registry[identifier] = technique
                    canonical: str = raw.upper()
                    if canonical not in registry:
                        raise ValueError("ATT&CK identifier is absent from the configured version.")
                    technique = registry[canonical]
                    expected_parent: str | None = (
                        canonical.split(".")[0] if "." in canonical else None
                    )
                    if technique.get("parent_id") != expected_parent:
                        raise ValueError("ATT&CK resource parent-child relationship is invalid.")
                    if expected_parent is not None and expected_parent not in registry:
                        raise ValueError("ATT&CK sub-technique parent is missing.")
                    for flag in ("revoked", "deprecated"):
                        if not isinstance(_required(technique, flag), bool):
                            raise ValueError("ATT&CK status flags must be explicit booleans.")
                    normalized.update(
                        canonical=canonical,
                        status="validated",
                        resource_version=resource["version"],
                        revoked=technique["revoked"],
                        deprecated=technique["deprecated"],
                    )
                elif not self._config.data.preprocessing.normalize_identifiers:
                    pass
                elif kind == "cve":
                    if _CVE_PATTERN.fullmatch(raw) is None:
                        raise ValueError("Malformed CVE identifier; no repair is inferred.")
                    normalized.update(
                        canonical=raw.upper(), status="syntax_validated",
                        authoritative_membership_verified=False,
                    )
                elif kind == "ioc":
                    normalized.update(
                        canonical=self._normalize_ioc(observation), status="normalized"
                    )
                elif kind == "alias":
                    resource = resources.get("alias_table")
                    if resource is None:
                        raise ValueError("Alias resolution requires the configured resource.")
                    entity_type: str = _identifier(
                        _required(observation, "entity_type")
                    )
                    matches: list[dict[str, Any]] = [
                        item
                        for item in _objects(
                            _required(resource["content"], "entries"), "Alias entries"
                        )
                        if item.get("name") == raw
                        and item.get("entity_type") == entity_type
                    ]
                    if len(matches) != 1:
                        raise ValueError("Alias mapping is absent or ambiguous.")
                    canonical = _identifier(_required(matches[0], "canonical"))
                    support: list[str] = _strings(
                        _required(matches[0], "evidence_refs"), "Alias evidence_refs"
                    )
                    if not support:
                        raise ValueError("Alias mappings require evidence.")
                    if role == "input_normalization" and any(
                        ref not in evidence or evidence[ref].role != EvidenceRole.MODEL_INPUT
                        for ref in support
                    ):
                        raise ValueError("Input normalization requires model-visible alias support.")
                    normalized.update(
                        canonical=canonical, status="validated",
                        resource_version=resource["version"], evidence_refs=support,
                    )
                else:
                    raise ValueError(f"Unsupported observation kind: {kind!r}.")
            except ValueError as exc:
                if unresolved == "reject":
                    raise
                normalized.update(
                    status="unresolved", canonical=None, reason=str(exc)
                )
            result.append(normalized)
        return result

    def _reference(
        self,
        conversion: dict[str, Any],
        references: dict[str, dict[str, Any]],
        evidence: dict[str, EvidenceRecord],
        snapshots: dict[str, dict[str, Any]],
        version: str,
    ) -> dict[str, Any]:
        reference_id: str = _identifier(_required(conversion, "reference_id"))
        if reference_id not in references:
            raise ValueError("Conversion reference_id does not resolve.")
        declaration: dict[str, Any] = references[reference_id]
        if _required(declaration, "instance_id") != conversion["instance_id"]:
            raise ValueError("Reference belongs to another task instance.")
        answer: dict[str, Any] = _object(
            _required(declaration, "answer"), "Reference answer"
        )
        if not answer:
            raise ValueError("Reference answer must not be empty.")
        ids: list[str] = _strings(
            _required(declaration, "evidence_ids"), "Reference evidence_ids"
        )
        records: list[dict[str, Any]] = []
        for evidence_id in ids:
            if evidence_id not in evidence:
                raise ValueError("Reference evidence_id does not resolve.")
            record: EvidenceRecord = evidence[evidence_id]
            if record.role == EvidenceRole.MODEL_INPUT:
                raise ValueError("Reference records must have a reference-only role.")
            if (
                record.role == EvidenceRole.OUTCOME_REFERENCE
                and conversion["task_name"] not in _FORECAST_TASKS
            ):
                raise ValueError("Future outcome references require a forecasting task.")
            self._check_evidence(record, snapshots, snapshot=None)
            records.append(record.model_dump(mode="json"))
        if not records:
            raise ValueError("Reference answers require supporting evidence.")
        reference: dict[str, Any] = {
            "reference_id": reference_id,
            "instance_id": conversion["instance_id"],
            "version": version,
            "answer": answer,
            "evidence_records": records,
            "declaration": declaration,
        }
        if conversion["task_name"] in _FORECAST_TASKS:
            task = self._config.evaluation.tasks["prediction"][conversion["task_name"]]
            if task.horizon_days is None or task.binary_target_construction is None:
                raise ValueError(
                    "Forecast horizon and binary_target_construction are unresolved."
                )
            if _required(declaration, "horizon_days") != task.horizon_days:
                raise ValueError("Reference forecast horizon differs from configuration.")
            target: Any = _required(declaration, "target_construction")
            if _canonical_json(target) != _canonical_json(task.binary_target_construction):
                raise ValueError("Reference target construction differs from configuration.")
            reference["horizon_days"] = task.horizon_days
            reference["target_construction"] = target
        return reference

    def _construct(
        self,
        conversion: dict[str, Any],
        bundle: dict[str, Any],
        ingestion: dict[str, Any],
    ) -> TaskInstance:
        evidence, snapshots, references, _ = self._indexes(bundle)
        source, original = self._original(
            _identifier(_required(conversion, "source_id")),
            _original_id(_required(conversion, "source_item_id")),
        )
        if source["dataset"] != _required(conversion, "source_dataset"):
            raise ValueError("Conversion source dataset does not match original item.")
        stage = CTIStage(_required(conversion, "stage"))
        task_name: str = _identifier(_required(conversion, "task_name"))
        stage_tasks = self._config.evaluation.tasks.get(stage.value.lower(), {})
        if task_name not in stage_tasks:
            raise ValueError("Conversion task does not belong to its declared stage.")
        snapshot: date = _calendar_date(_required(conversion, "snapshot_date"))
        input_ids: list[str] = _strings(
            _required(conversion, "input_evidence_ids"), "Input evidence IDs"
        )
        if not input_ids:
            raise ValueError("Explicit model-visible evidence is required.")
        visible: list[EvidenceRecord] = []
        for evidence_id in input_ids:
            if evidence_id not in evidence:
                raise ValueError("Conversion input evidence_id does not resolve.")
            record: EvidenceRecord = evidence[evidence_id]
            if record.role != EvidenceRole.MODEL_INPUT:
                raise ValueError("Reference-only evidence cannot enter model inputs.")
            self._check_evidence(record, snapshots, snapshot)
            visible.append(record)
        reference: dict[str, Any] = self._reference(
            conversion, references, evidence, snapshots,
            bundle["references"]["content"]["version"],
        )
        reference_ids: set[str] = {
            item["evidence_id"] for item in reference["evidence_records"]
        }
        if set(input_ids) & reference_ids:
            raise ValueError("Withheld reference records overlap model-visible records.")
        profile: dict[str, Any] = _object(
            _required(conversion, "output_profile"), "Output profile"
        )
        if not profile:
            raise ValueError("An explicit task output profile is required.")
        candidates: Any = _required(conversion, "candidates")
        if not isinstance(candidates, list):
            raise ValueError("Candidates must be an explicitly declared list.")
        history: Any = _required(conversion, "transformation_history")
        if not isinstance(history, list) or not history:
            raise ValueError("Conversion requires nonempty transformation_history.")
        rule_id: str = _identifier(_required(conversion, "rule_id"))
        rule_version: str = _identifier(_required(conversion, "rule_version"))
        guidance: str = _identifier(_required(conversion, "task_guidance"))
        normalized: list[dict[str, Any]] = self._normalizations(
            conversion,
            {item.evidence_id: item for item in visible},
            ingestion["normalization_resources"],
        )
        log: dict[str, Any] = {
            "corpus_version": self._config.corpus_version,
            "config_hash": self._config.config_hash,
            "source_record_id": original["source_id"],
            "acquisition_source_id": source["source_id"],
            "source_version": source["source_version"],
            "source_raw_blob_hash": source["raw_blob_hash"],
            "conversion_manifest_version": bundle["conversions"]["content"]["version"],
            "conversion_manifest_hash": bundle["conversions"]["sha256"],
            "reference_manifest_hash": bundle["references"]["sha256"],
            "snapshot_manifest_hash": bundle["snapshots"]["sha256"],
            "conversion": conversion,
            "rule_id": rule_id,
            "rule_version": rule_version,
            "history": history,
            "task_guidance": guidance,
            "normalization": normalized,
            "normalization_contract": NORMALIZATION_CONTRACT,
            "normalization_resource_hashes": {
                name: item["sha256"]
                for name, item in ingestion["normalization_resources"].items()
            },
        }
        instance = TaskInstance(
            instance_id=_identifier(_required(conversion, "instance_id")),
            case_id=_identifier(_required(conversion, "case_id")),
            source_dataset=conversion["source_dataset"],
            source_item_id=_original_id(conversion["source_item_id"]),
            task_name=task_name,
            stage=stage,
            snapshot_date=snapshot,
            evidence_records=visible,
            candidates=candidates,
            environment=_object(
                _required(conversion, "environment"), "Environment"
            ),
            reference=reference,
            transformation_log=log,
            output_profile=profile,
        )
        payload: dict[str, Any] = instance.model_dump(mode="python")
        payload["content_hash"] = _content_hash(_task_content_projection(instance))
        return TaskInstance.model_validate(
            payload,
            context={
                "evaluation_tasks": self._config.evaluation.tasks,
                "require_content_hash": True,
            },
        )

    def build_instances(self) -> list[TaskInstance]:
        """Construct only explicitly converted, fully validated task instances."""
        self._config.validate("prepare")
        bundle: dict[str, Any] = self._bundle()
        ingestion: dict[str, Any] = self._assert_bundle(bundle)
        conversions: list[dict[str, Any]] = self._conversions(bundle)
        intended: list[str] = [item["instance_id"] for item in conversions]
        completed: list[str] = []
        instances: list[TaskInstance] = []
        hashes: dict[str, str] = ingestion["manifest_hashes"]
        try:
            self._checkpoint(
                "build", StageStatus.RUNNING, completed, intended, hashes
            )
            for conversion in conversions:
                instance: TaskInstance = self._construct(conversion, bundle, ingestion)
                findings: list[dict[str, str]] = self.validate_instance(instance)
                blocking: list[dict[str, str]] = [
                    finding for finding in findings if finding["severity"] == "error"
                ]
                if blocking:
                    raise ValueError(
                        f"Instance {instance.instance_id!r} failed validation: "
                        + "; ".join(item["message"] for item in blocking)
                    )
                self._store.create_record(
                    "reference",
                    instance.reference["reference_id"],
                    instance.reference,
                )
                self._store.create_record(
                    "task_instance",
                    instance.instance_id,
                    instance.model_dump(mode="json"),
                )
                instances.append(instance)
                completed.append(instance.instance_id)
                self._checkpoint(
                    "build", StageStatus.RUNNING, completed, intended, hashes
                )
            self._persist_manifest(
                "construction",
                {
                    "manifest_hashes": hashes,
                    "members": [
                        {"instance_id": item.instance_id, "content_hash": item.content_hash}
                        for item in instances
                    ],
                },
            )
            self._write_counts(instances)
            self._checkpoint(
                "build", StageStatus.COMPLETED, intended, [], hashes
            )
            return instances
        except Exception as exc:
            self._checkpoint(
                "build", StageStatus.FAILED, completed, intended, hashes,
                error=type(exc).__name__,
            )
            raise

    def validate_instance(self, instance: TaskInstance) -> list[dict[str, str]]:
        """Return blocking errors and review warnings without repairing content."""
        findings: list[dict[str, str]] = []

        def error(code: str, message: str) -> None:
            findings.append({"severity": "error", "code": code, "message": message})

        try:
            instance = TaskInstance.model_validate(
                instance.model_dump(mode="python"),
                context={"require_content_hash": True},
            )
        except ValueError:
            error("schema_or_hash", "Instance schema or canonical content hash is invalid.")
            return findings
        if instance.transformation_log.get("corpus_version") != self._config.corpus_version:
            error("corpus_version", "Instance belongs to a different corpus version.")
        if instance.transformation_log.get("config_hash") != self._config.config_hash:
            error("config_version", "Instance belongs to a different configuration.")
        try:
            bundle: dict[str, Any] = self._bundle()
            ingestion: dict[str, Any] = self._assert_bundle(bundle)
            _, _, _, conversions = self._indexes(bundle)
            declaration: dict[str, Any] | None = conversions.get(instance.instance_id)
            if declaration is None:
                error("conversion", "Instance lacks a declared source conversion.")
            else:
                expected: TaskInstance = self._construct(
                    declaration, bundle, ingestion
                )
                if _canonical_json(expected) != _canonical_json(instance):
                    error(
                        "conversion_integrity",
                        "Instance differs from its declared source, reference, "
                        "normalization, profile, or temporal conversion.",
                    )
        except (ValueError, KeyError, OSError) as exc:
            error("provenance_or_temporal", str(exc))
        for record in instance.evidence_records:
            if record.role != EvidenceRole.MODEL_INPUT:
                error("reference_leakage", "A model-visible record has a withheld role.")
        for normalized in instance.transformation_log.get("normalization", []):
            if normalized.get("status") == "unresolved":
                findings.append(
                    {
                        "severity": "warning",
                        "code": "unresolved_identifier",
                        "message": (
                            "A source identifier remains unresolved under the "
                            "explicit retain_unresolved policy."
                        ),
                    }
                )
        # Contradictions are not resolved or discarded by data preparation.
        if any(
            record.provenance.get("conflicts_with")
            for record in instance.evidence_records
        ):
            findings.append(
                {
                    "severity": "warning",
                    "code": "declared_conflict",
                    "message": "Conflicting source claims were intentionally preserved.",
                }
            )
        return findings

    def _write_counts(self, instances: list[TaskInstance]) -> None:
        task_counts: dict[str, int] = {}
        case_ids: set[str] = set()
        for instance in instances:
            task_counts[instance.task_name] = task_counts.get(instance.task_name, 0) + 1
            case_ids.add(instance.case_id)
        discrepancies: list[dict[str, Any]] = []
        for tasks in self._config.evaluation.tasks.values():
            for name, definition in tasks.items():
                observed: int = task_counts.get(name, 0)
                if observed != definition.approximate_instances:
                    discrepancies.append(
                        {
                            "task": name,
                            "observed": observed,
                            "paper_approximate": definition.approximate_instances,
                            "severity": "warning",
                        }
                    )
        report: dict[str, Any] = {
            "report_id": self._execution_id("counts"),
            "corpus_version": self._config.corpus_version,
            "config_hash": self._config.config_hash,
            "task_assignments": len(instances),
            "unique_cases": len(case_ids),
            "task_counts": task_counts,
            "approximate_count_discrepancies": discrepancies,
            "counts_are_selection_quotas": False,
        }
        self._store.create_record("report", report["report_id"], report)
        if discrepancies:
            _LOGGER.warning(
                "Corpus counts differ from rounded paper counts; discrepancies "
                "were recorded without quota-based sampling or duplication."
            )

    def freeze_corpus(self) -> str:
        """Freeze exact validated membership and return its binding manifest hash."""
        self._config.validate("prepare")
        bundle: dict[str, Any] = self._bundle()
        ingestion: dict[str, Any] = self._assert_bundle(bundle)
        try:
            construction: dict[str, Any] = self._store.read_record(
                "experiment_manifest", self._execution_id("construction")
            )
        except KeyError:
            raise ValueError("Instance construction is incomplete; corpus cannot freeze.") from None
        intended: list[str] = [
            item["instance_id"] for item in self._conversions(bundle)
        ]
        if intended != [item["instance_id"] for item in construction["members"]]:
            raise ValueError("Constructed membership differs from intended conversion membership.")
        instances: list[TaskInstance] = []
        for member in construction["members"]:
            instance = TaskInstance.model_validate(
                self._store.read_record("task_instance", member["instance_id"]),
                context={"require_content_hash": True},
            )
            if instance.content_hash != member["content_hash"]:
                raise ValueError("Constructed instance changed before freezing.")
            errors: list[dict[str, str]] = [
                item for item in self.validate_instance(instance)
                if item["severity"] == "error"
            ]
            if errors:
                raise ValueError(
                    f"Cannot freeze invalid instance {instance.instance_id!r}: "
                    + "; ".join(item["message"] for item in errors)
                )
            instances.append(instance)
        if not instances:
            raise ValueError("An empty corpus cannot be frozen.")

        existing: list[dict[str, Any]] = [
            item for item in self._store.list_records("corpus_manifest")
            if item.get("corpus_version") == self._config.corpus_version
        ]
        membership: list[dict[str, Any]] = [
            {"instance_id": item.instance_id, "content_hash": item.content_hash}
            for item in instances
        ]
        for previous in existing:
            if (
                previous.get("config_hash") != self._config.config_hash
                or previous.get("members") != membership
                or previous.get("manifest_hashes") != ingestion["manifest_hashes"]
            ):
                raise ValueError(
                    "This corpus version is already frozen with different content. "
                    "Declare a new corpus version."
                )
            return _content_hash(previous)

        instance_freeze: str = self._store.freeze_records("task_instance", intended)
        reference_ids: list[str] = sorted(
            {item.reference["reference_id"] for item in instances}
        )
        reference_freeze: str = self._store.freeze_records("reference", reference_ids)
        manifest: dict[str, Any] = {
            "manifest_id": _content_hash(
                {"corpus_version": self._config.corpus_version}
            ),
            "corpus_version": self._config.corpus_version,
            "config_hash": self._config.config_hash,
            "members": membership,
            "manifest_hashes": ingestion["manifest_hashes"],
            "sources": ingestion["source_records"],
            "source_hashes": ingestion["source_hashes"],
            "source_freeze_hash": ingestion["source_freeze_hash"],
            "instance_freeze_hash": instance_freeze,
            "reference_freeze_hash": reference_freeze,
            "normalization_resources": ingestion["normalization_resources"],
            "implementation_choices": _json_primitive(
                self._config.implementation_choices
            ),
        }
        digest: str = _content_hash(manifest)
        self._store.save_blob(_canonical_json(manifest).encode("utf-8"))
        self._store.create_record(
            "corpus_manifest", manifest["manifest_id"], manifest
        )
        self._store.freeze_records("corpus_manifest", [manifest["manifest_id"]])
        self._checkpoint(
            "freeze", StageStatus.COMPLETED, intended, [],
            ingestion["manifest_hashes"],
        )
        return digest

    def load_instances(
        self, filters: dict[str, Any] | None = None
    ) -> list[TaskInstance]:
        """Load deterministic frozen membership; drafts require explicit opt-in.

        Supported filters: instance_id, case_id, source_dataset, source_item_id,
        task_name, stage, corpus_version, and include_drafts. Values may be scalar
        or lists except include_drafts, which must be boolean. An empty list
        matches no instances. Loading never reruns normalization or downloads data.
        """
        selected: dict[str, Any] = {} if filters is None else dict(filters)
        allowed: set[str] = {
            "instance_id", "case_id", "source_dataset", "source_item_id",
            "task_name", "stage", "corpus_version", "include_drafts",
        }
        unknown: set[str] = set(selected) - allowed
        if unknown:
            raise ValueError(f"Unsupported instance filters: {sorted(unknown)}.")
        include_drafts: Any = selected.pop("include_drafts", False)
        if not isinstance(include_drafts, bool):
            raise ValueError("include_drafts must be explicitly boolean.")
        version: Any = selected.pop("corpus_version", self._config.corpus_version)
        if version is None or version != self._config.corpus_version:
            raise ValueError("Requested corpus version differs from configuration.")

        for key, value in selected.items():
            values: list[Any] = value if isinstance(value, list) else [value]
            for item in values:
                if key == "stage":
                    CTIStage(item)
                else:
                    _identifier(item)

        expected_members: dict[str, str] = {}
        if not include_drafts:
            manifests: list[dict[str, Any]] = [
                item for item in self._store.list_records("corpus_manifest")
                if item.get("corpus_version") == version
            ]
            if len(manifests) != 1:
                raise ValueError("Exactly one frozen manifest is required for this corpus.")
            manifest: dict[str, Any] = manifests[0]
            expected_members = {
                item["instance_id"]: item["content_hash"]
                for item in manifest["members"]
            }
            # lifecycle is storage metadata, not a TaskInstance payload field.
            frozen_ids: set[str] = {
                item["instance_id"]
                for item in self._store.list_records(
                    "task_instance", {"lifecycle": "frozen"}
                )
            }
            if not set(expected_members).issubset(frozen_ids):
                raise ValueError("Corpus manifest contains nonfrozen task records.")

        payloads: list[dict[str, Any]] = self._store.list_records("task_instance")
        result: list[TaskInstance] = []
        found: set[str] = set()
        for payload in payloads:
            if payload.get("transformation_log", {}).get("corpus_version") != version:
                continue
            if not include_drafts and payload["instance_id"] not in expected_members:
                continue
            instance = TaskInstance.model_validate(
                payload, context={"require_content_hash": True}
            )
            found.add(instance.instance_id)
            if (
                not include_drafts
                and instance.content_hash != expected_members[instance.instance_id]
            ):
                raise ValueError("Frozen corpus instance hash differs from its manifest.")
            matches: bool = True
            for key, value in selected.items():
                actual: Any = getattr(instance, key)
                if isinstance(actual, CTIStage):
                    actual = actual.value
                options: list[Any] = value if isinstance(value, list) else [value]
                if actual not in options:
                    matches = False
                    break
            if matches:
                result.append(instance)
        if not include_drafts and found != set(expected_members):
            raise ValueError("Frozen corpus membership is incomplete in storage.")
        return sorted(result, key=lambda item: item.instance_id)
