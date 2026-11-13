"""Transactional, append-only experimental artifact storage.

The database path and artifact directory are supplied explicitly by the caller.
No experimental settings are inferred from missing configuration values.

Storage conventions:
  * Known scientific records use schemas.py models.
  * Supplementary artifact kinds have an explicitly registered JSON-object
    contract and an identity field.
  * Draft mutation is shallow field replacement, never recursive merging.
  * Model responses are cached only when usage.request_hash identifies a stored,
    validated inference request. Invalid completions and refusals are cacheable;
    transport failures are not.
  * Exports are canonical JSON Lines containing storage metadata and payloads.
  * Raw blobs use SHA-256 over exact bytes. Structured records and manifests use
    the shared canonical JSON hashing implementation.

Supplementary inference-request contract:
    request_hash, model_id, endpoint_or_checkpoint, revision, prompt_hash,
    generation_controls, seed_behavior, schema_name, schema_version, run_index,
    config_hash, provider_options; optional instance_id and prompt.

The request hash covers the entire validated request except request_hash itself.
This deliberately includes experimental context as well as inference controls.

SQLite guarantees one authoritative persisted completion per request, not
exactly one external provider call. No transaction spans inference or annotation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, StrictStr

from schemas import (
    AnnotationStatus,
    AnnotationTask,
    AuditCohort,
    Digest,
    EvidenceRecord,
    FailureDecision,
    Identifier,
    JsonObject,
    LabelAssignment,
    MetricResult,
    ModelResponse,
    NonNegativeInt,
    ResponseStatus,
    SchemaModel,
    StageState,
    StageStatus,
    TaskInstance,
    TaxonomyVersion,
    UtcTimestamp,
    _canonical_json,
    _content_hash,
    _identifier,
    _json_primitive,
    _text_hash,
)


_SCHEMA_VERSION: int = 1
_APPLICATION_ID: int = 0x43544941  # Engineering identifier: "CTIA".
_METADATA_FIELDS: tuple[str, ...] = (
    "instance_id",
    "response_id",
    "case_id",
    "model_id",
    "run_index",
    "config_hash",
    "taxonomy_id",
    "cohort_id",
    "purpose",
    "actor_id",
    "round_index",
    "supersedes_id",
    "status",
)
_FILTER_FIELDS: frozenset[str] = frozenset(
    (*_METADATA_FIELDS, "record_id", "content_hash", "lifecycle")
)

_MODEL_CONTRACTS: dict[str, type[SchemaModel]] = {
    "evidence": EvidenceRecord,
    "evidence_record": EvidenceRecord,
    "task_instance": TaskInstance,
    "model_response": ModelResponse,
    "failure_decision": FailureDecision,
    "taxonomy": TaxonomyVersion,
    "taxonomy_version": TaxonomyVersion,
    "label_assignment": LabelAssignment,
    "assignment": LabelAssignment,
    "final_assignment": LabelAssignment,
    "consensus": LabelAssignment,
    "annotation_task": AnnotationTask,
    "audit_cohort": AuditCohort,
    "metric": MetricResult,
    "metric_result": MetricResult,
    "stage_state": StageState,
}

_IDENTITY_FIELDS: dict[str, str] = {
    "evidence": "evidence_id",
    "evidence_record": "evidence_id",
    "task_instance": "instance_id",
    "model_response": "response_id",
    "taxonomy": "taxonomy_id",
    "taxonomy_version": "taxonomy_id",
    "label_assignment": "assignment_id",
    "assignment": "assignment_id",
    "final_assignment": "assignment_id",
    "consensus": "assignment_id",
    "annotation_task": "annotation_task_id",
    "audit_cohort": "cohort_id",
    "metric": "metric_id",
    "metric_result": "metric_id",
    "request": "request_hash",
    "inference_request": "request_hash",
    "request_attempt": "attempt_id",
    "annotation": "annotation_id",
    "adjudication": "annotation_id",
    "source": "source_id",
    "reference": "reference_id",
    "configuration": "config_id",
    "config_manifest": "config_id",
    "prompt": "prompt_id",
    "scoring": "scoring_id",
    "score": "score_id",
    "stratum": "stratum_id",
    "range": "range_id",
    "discovery_review": "review_id",
    "experiment_manifest": "manifest_id",
    "corpus_manifest": "manifest_id",
    "reference_manifest": "manifest_id",
    "sampling_manifest": "manifest_id",
    "report": "report_id",
    "counterfactual": "variant_id",
}

# These contracts allow extension fields, but only JSON-compatible values.
# Their fixed identity and lifecycle policies are not caller-controlled.
_JSON_KINDS: frozenset[str] = frozenset(
    {
        "source",
        "reference",
        "configuration",
        "config_manifest",
        "prompt",
        "scoring",
        "score",
        "stratum",
        "range",
        "discovery_review",
        "experiment_manifest",
        "corpus_manifest",
        "reference_manifest",
        "sampling_manifest",
        "report",
        "counterfactual",
    }
)
_JSON_DRAFT_KINDS: frozenset[str] = frozenset(
    {
        "configuration",
        "config_manifest",
        "prompt",
        "scoring",
        "stratum",
        "range",
    }
)
_ASSIGNMENT_KINDS: frozenset[str] = frozenset(
    {"label_assignment", "assignment", "final_assignment", "consensus"}
)
_REQUEST_KINDS: tuple[str, ...] = ("request", "inference_request")
_TAXONOMY_KINDS: tuple[str, ...] = ("taxonomy", "taxonomy_version")

_SECRET_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "password",
        "passwd",
        "authorization",
        "access_token",
        "refresh_token",
        "auth_token",
        "client_secret",
        "private_key",
        "credentials",
        "bearer_token",
    }
)


class _JsonDocument(SchemaModel):
    """Explicit contract for registered supplementary JSON artifacts."""

    content: JsonObject


class _InferenceRequest(SchemaModel):
    request_hash: Digest
    model_id: Identifier
    endpoint_or_checkpoint: Identifier
    revision: Identifier
    prompt_hash: Digest
    generation_controls: JsonObject
    seed_behavior: JsonObject
    schema_name: Identifier
    schema_version: Identifier
    run_index: NonNegativeInt
    config_hash: Digest
    provider_options: JsonObject
    instance_id: Identifier | None = None
    prompt: StrictStr | None = None


class _RequestAttempt(SchemaModel):
    attempt_id: Identifier
    request_hash: Digest
    outcome: Literal[
        "transport_failure", "completed", "refusal", "invalid_schema"
    ]
    started_at: UtcTimestamp
    completed_at: UtcTimestamp
    response_id: Identifier | None = None
    error: JsonObject | None = None
    metadata: JsonObject = Field(default_factory=dict)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _reject_credentials(value: Any) -> None:
    """Reject credential fields outside retained model/evidence text.

    Free text is not inspected or rewritten: threat reports and evaluated model
    responses can legitimately contain words such as 'password'. Callers remain
    responsible for sanitizing provider errors, headers, and configuration.
    """
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized: str = str(key).lower().replace("-", "_")
            if (
                normalized in _SECRET_FIELD_NAMES
                or normalized.endswith("_api_key")
                or normalized.endswith("_client_secret")
            ) and item is not None:
                raise ValueError(
                    f"Credential field {key!r} must not be persisted; "
                    "use an environment-variable reference."
                )
            _reject_credentials(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_credentials(item)


def _same_files(first: Path, second: Path) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            left_chunk: bytes = left.read(1024 * 1024)
            right_chunk: bytes = right.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _sync_directory(directory: Path) -> None:
    descriptor: int = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ArtifactStore:
    """SQLite records and content-addressed files shared by CLI and UI."""

    def __init__(self, database_path: str, artifact_root: str) -> None:
        """Retain explicit paths without selecting filesystem defaults."""
        if not isinstance(database_path, str) or not database_path.strip():
            raise ValueError("database_path must be supplied explicitly.")
        if not isinstance(artifact_root, str) or not artifact_root.strip():
            raise ValueError("artifact_root must be supplied explicitly.")
        if database_path == ":memory:" or database_path.startswith("file:"):
            raise ValueError("A local file database is required for verified WAL.")
        self._database_path: Path = Path(database_path).expanduser().absolute()
        self._artifact_root: Path = Path(artifact_root).expanduser().absolute()
        self._initialized: bool = False

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        if not self._initialized:
            raise RuntimeError("Call ArtifactStore.initialize() before use.")
        connection: sqlite3.Connection = sqlite3.connect(
            str(self._database_path), isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
                raise RuntimeError("SQLite foreign-key enforcement is unavailable.")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create or verify the storage schema; never migrate incompatible data."""
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._artifact_root.mkdir(parents=True, exist_ok=True)
        connection: sqlite3.Connection = sqlite3.connect(
            str(self._database_path), isolation_level=None
        )
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            journal: str = str(
                connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            ).lower()
            if journal != "wal":
                raise RuntimeError(
                    "WAL could not be enabled. Resolve filesystem/locking support."
                )
            application_id: int = int(
                connection.execute("PRAGMA application_id").fetchone()[0]
            )
            version: int = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            existing: set[str] = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if version == 0:
                if existing or application_id not in {0, _APPLICATION_ID}:
                    raise RuntimeError("Refusing to adopt an unrecognized database.")
            elif version != _SCHEMA_VERSION or application_id != _APPLICATION_ID:
                raise RuntimeError("Incompatible artifact database schema.")

            connection.execute("BEGIN IMMEDIATE")
            self._create_schema(connection)
            connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            connection.commit()

            required_columns: set[str] = {
                "kind", "record_id", "payload", "content_hash",
                "created_at", "lifecycle", *_METADATA_FIELDS,
            }
            actual: set[str] = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(records)")
            }
            if actual != required_columns:
                raise RuntimeError("Record table does not match the declared schema.")
            violations: list[Any] = connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()
            if violations:
                raise RuntimeError("Database contains broken foreign-key links.")
            self._initialized = True
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        metadata_sql: str = ",\n".join(
            f"{name} {'INTEGER' if name in {'run_index', 'round_index'} else 'TEXT'}"
            for name in _METADATA_FIELDS
        )
        statements: list[str] = [
            f"""
            CREATE TABLE IF NOT EXISTS records (
                kind TEXT NOT NULL,
                record_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                lifecycle TEXT NOT NULL
                    CHECK(lifecycle IN ('draft','immutable','frozen')),
                {metadata_sql},
                PRIMARY KEY(kind, record_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS freeze_manifests (
                manifest_hash TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS freeze_members (
                manifest_hash TEXT NOT NULL,
                kind TEXT NOT NULL,
                record_id TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                PRIMARY KEY(manifest_hash, kind, record_id),
                FOREIGN KEY(manifest_hash)
                    REFERENCES freeze_manifests(manifest_hash) ON DELETE RESTRICT,
                FOREIGN KEY(kind, record_id)
                    REFERENCES records(kind, record_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS record_links (
                source_kind TEXT NOT NULL,
                source_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                target_kind TEXT NOT NULL,
                target_id TEXT NOT NULL,
                PRIMARY KEY(source_kind, source_id, relation, target_kind, target_id),
                FOREIGN KEY(source_kind, source_id)
                    REFERENCES records(kind, record_id) ON DELETE CASCADE,
                FOREIGN KEY(target_kind, target_id)
                    REFERENCES records(kind, record_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS response_cache (
                request_hash TEXT PRIMARY KEY,
                request_kind TEXT NOT NULL,
                response_kind TEXT NOT NULL DEFAULT 'model_response'
                    CHECK(response_kind='model_response'),
                response_id TEXT NOT NULL UNIQUE,
                FOREIGN KEY(request_kind, request_hash)
                    REFERENCES records(kind, record_id) ON DELETE RESTRICT,
                FOREIGN KEY(response_kind, response_id)
                    REFERENCES records(kind, record_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS stage_checkpoints (
                stage TEXT PRIMARY KEY,
                state_kind TEXT NOT NULL DEFAULT 'stage_state'
                    CHECK(state_kind='stage_state'),
                state_id TEXT NOT NULL,
                FOREIGN KEY(state_kind, state_id)
                    REFERENCES records(kind, record_id) ON DELETE RESTRICT
            )
            """,
            """
            CREATE TRIGGER IF NOT EXISTS immutable_record_update
            BEFORE UPDATE ON records
            WHEN OLD.lifecycle != 'draft'
              OR EXISTS (
                SELECT 1 FROM freeze_members
                WHERE kind=OLD.kind AND record_id=OLD.record_id
              )
            BEGIN
                SELECT RAISE(ABORT, 'immutable record');
            END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS immutable_record_delete
            BEFORE DELETE ON records
            WHEN OLD.lifecycle != 'draft'
              OR EXISTS (
                SELECT 1 FROM freeze_members
                WHERE kind=OLD.kind AND record_id=OLD.record_id
              )
            BEGIN
                SELECT RAISE(ABORT, 'immutable record');
            END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS record_identity_update
            BEFORE UPDATE ON records
            WHEN OLD.kind != NEW.kind OR OLD.record_id != NEW.record_id
              OR OLD.created_at != NEW.created_at
            BEGIN
                SELECT RAISE(ABORT, 'record identity is immutable');
            END
            """,
        ]
        for statement in statements:
            connection.execute(statement)
        for name in _METADATA_FIELDS:
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS records_{name}_idx "
                f"ON records(kind, {name})"
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS links_target_idx "
            "ON record_links(target_kind, target_id)"
        )
        for table in ("freeze_manifests", "freeze_members", "response_cache"):
            for operation in ("UPDATE", "DELETE"):
                connection.execute(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS {table}_{operation.lower()}_guard
                    BEFORE {operation} ON {table}
                    BEGIN SELECT RAISE(ABORT, 'append-only storage table'); END
                    """
                )
        for operation in ("UPDATE", "DELETE"):
            connection.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS links_{operation.lower()}_guard
                BEFORE {operation} ON record_links
                WHEN EXISTS (
                    SELECT 1 FROM records
                    WHERE kind=OLD.source_kind AND record_id=OLD.source_id
                      AND lifecycle != 'draft'
                )
                BEGIN SELECT RAISE(ABORT, 'immutable record links'); END
                """
            )

    @staticmethod
    def _known_kind(kind: str) -> None:
        _identifier(kind)
        allowed: set[str] = (
            set(_MODEL_CONTRACTS)
            | set(_JSON_KINDS)
            | set(_REQUEST_KINDS)
            | {"request_attempt", "annotation", "adjudication"}
        )
        if kind not in allowed:
            raise ValueError(f"Unregistered artifact kind: {kind!r}.")

    @staticmethod
    def _row(
        connection: sqlite3.Connection, kind: str, record_id: str
    ) -> sqlite3.Row:
        row: sqlite3.Row | None = connection.execute(
            "SELECT * FROM records WHERE kind=? AND record_id=?",
            (kind, record_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"Record not found: {kind}/{record_id}.")
        return row

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        payload: dict[str, Any] = json.loads(row["payload"])
        if _content_hash(payload) != row["content_hash"]:
            raise sqlite3.DatabaseError("Stored record content hash is inconsistent.")
        return payload

    def _find(
        self,
        connection: sqlite3.Connection,
        kinds: tuple[str, ...],
        record_id: str,
    ) -> tuple[str, dict[str, Any]]:
        matches: list[tuple[str, dict[str, Any]]] = []
        for kind in kinds:
            row: sqlite3.Row | None = connection.execute(
                "SELECT * FROM records WHERE kind=? AND record_id=?",
                (kind, record_id),
            ).fetchone()
            if row is not None:
                matches.append((kind, self._decode(row)))
        if not matches:
            raise KeyError(f"Referenced record not found: {record_id}.")
        if len(matches) != 1:
            raise ValueError(f"Ambiguous referenced identity: {record_id}.")
        return matches[0]

    def _validate(
        self,
        connection: sqlite3.Connection,
        kind: str,
        record_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        self._known_kind(kind)
        _identifier(record_id)
        if not isinstance(payload, dict):
            raise TypeError("Record payload must be a dictionary.")

        if kind in _ASSIGNMENT_KINDS:
            taxonomy_id: Any = payload.get("taxonomy_id")
            _, taxonomy_payload = self._find(
                connection, _TAXONOMY_KINDS, _identifier(taxonomy_id)
            )
            taxonomy: TaxonomyVersion = TaxonomyVersion.model_validate(
                taxonomy_payload
            )
            discovery: bool = payload.get("origin") == "discovery"
            context: dict[str, Any] = {
                "taxonomy": taxonomy,
                "require_taxonomy": True,
                "require_frozen_taxonomy": kind in {"consensus", "final_assignment"},
                "discovery_mode": discovery,
                "fallback_label": "other",
            }
            validated: Any = LabelAssignment.model_validate(payload, context=context)
        elif kind in _MODEL_CONTRACTS:
            validated = _MODEL_CONTRACTS[kind].model_validate(payload)
        elif kind in _REQUEST_KINDS:
            validated = _InferenceRequest.model_validate(payload)
            request_data: dict[str, Any] = validated.model_dump(mode="python")
            expected: str = _content_hash(
                {key: value for key, value in request_data.items()
                 if key != "request_hash"}
            )
            if validated.request_hash != expected:
                raise ValueError("Request hash does not cover the full request.")
            if (
                validated.prompt is not None
                and _text_hash(validated.prompt) != validated.prompt_hash
            ):
                raise ValueError("Exact prompt does not match prompt_hash.")
            _reject_credentials(request_data)
        elif kind == "request_attempt":
            validated = _RequestAttempt.model_validate(payload)
            if validated.completed_at < validated.started_at:
                raise ValueError("Attempt completion precedes its start.")
            if validated.outcome == "transport_failure" and validated.error is None:
                raise ValueError("Transport attempts require error details.")
            _reject_credentials(validated.model_dump(mode="python"))
        else:
            # This includes supplementary submitted annotations. Their common
            # persistence contract is explicit; subtype semantics belong to the
            # annotation service and LabelAssignment records.
            validated = _JsonDocument(content=payload).content
            if kind in {"annotation", "adjudication"}:
                required: tuple[str, ...] = (
                    "annotation_id", "response_id", "purpose",
                    "annotator_id", "guideline_hash",
                )
                for field in required:
                    _identifier(validated.get(field))
                if not any(
                    field in validated
                    for field in ("labels", "decision", "taxonomy_proposal")
                ):
                    raise ValueError("Submitted annotation lacks an annotation result.")
                if validated.get("status", "submitted") not in {
                    "submitted", "adjudicated"
                }:
                    raise ValueError("Annotation records represent locked submissions.")
            if kind in {
                "configuration", "config_manifest", "experiment_manifest",
                "corpus_manifest", "reference_manifest", "sampling_manifest",
            }:
                _reject_credentials(validated)

        result: Any = _json_primitive(validated)
        if not isinstance(result, dict):
            raise ValueError("Validated payload must be a JSON object.")
        identity_field: str | None = _IDENTITY_FIELDS.get(kind)
        if identity_field is not None:
            if result.get(identity_field) != record_id:
                raise ValueError(
                    f"{kind}.{identity_field} must equal the storage record ID."
                )
        return result

    @staticmethod
    def _lifecycle(kind: str, payload: dict[str, Any]) -> str:
        if kind == "task_instance" or kind in _JSON_DRAFT_KINDS:
            return "draft"
        if kind in _TAXONOMY_KINDS:
            return "immutable" if payload["frozen"] else "draft"
        if kind == "audit_cohort":
            return "immutable" if payload["locked"] else "draft"
        if kind == "annotation_task":
            status: str = payload["status"]
            if status in {
                AnnotationStatus.DRAFT.value,
                AnnotationStatus.PENDING.value,
                AnnotationStatus.IN_PROGRESS.value,
            }:
                return "draft"
        return "immutable"

    @staticmethod
    def _metadata(payload: dict[str, Any]) -> dict[str, Any]:
        return {field: payload.get(field) for field in _METADATA_FIELDS}

    def _links(
        self,
        connection: sqlite3.Connection,
        kind: str,
        record_id: str,
        payload: dict[str, Any],
    ) -> list[tuple[str, str, str]]:
        links: list[tuple[str, str, str]] = []

        def add(relation: str, kinds: tuple[str, ...], target_id: str) -> None:
            target_kind, _ = self._find(connection, kinds, target_id)
            links.append((relation, target_kind, target_id))

        if kind == "model_response":
            add("instance", ("task_instance",), payload["instance_id"])
            request_hash: Any = payload["usage"].get("request_hash")
            if request_hash is not None:
                add("request", _REQUEST_KINDS, _identifier(request_hash))
        elif kind in _ASSIGNMENT_KINDS:
            add("response", ("model_response",), payload["response_id"])
            add("taxonomy", _TAXONOMY_KINDS, payload["taxonomy_id"])
        elif kind == "failure_decision":
            add("response", ("model_response",), payload["response_id"])
        elif kind == "annotation_task":
            add("response", ("model_response",), payload["response_id"])
            if payload["cohort_id"] is not None:
                add("cohort", ("audit_cohort",), payload["cohort_id"])
        elif kind in {"annotation", "adjudication"}:
            add("response", ("model_response",), payload["response_id"])
            task_id: Any = payload.get(
                "annotation_task_id", payload.get("task_id")
            )
            if task_id is not None:
                add("annotation_task", ("annotation_task",), _identifier(task_id))
            cohort_id: Any = payload.get("cohort_id")
            if cohort_id is not None:
                add("cohort", ("audit_cohort",), _identifier(cohort_id))
        elif kind == "audit_cohort":
            for response_id in payload["response_ids"]:
                add("member", ("model_response",), response_id)
        elif kind == "request_attempt":
            add("request", _REQUEST_KINDS, payload["request_hash"])
            if payload["response_id"] is not None:
                add("response", ("model_response",), payload["response_id"])
        elif kind in _REQUEST_KINDS and payload["instance_id"] is not None:
            add("instance", ("task_instance",), payload["instance_id"])

        supersedes: Any = payload.get("supersedes_id")
        if supersedes is not None:
            if supersedes == record_id:
                raise ValueError("A record cannot supersede itself.")
            add("supersedes", (kind,), _identifier(supersedes))
            if kind in {"annotation", "adjudication"} and str(
                payload.get("purpose", "")
            ).startswith(("independent_", "expanded_", "audit_")):
                raise ValueError(
                    "Audit submissions do not supersede operational annotations."
                )
        return links

    def _insert(
        self,
        connection: sqlite3.Connection,
        kind: str,
        record_id: str,
        payload: dict[str, Any],
    ) -> None:
        serialized: str = _canonical_json(payload)
        content_hash: str = _content_hash(payload)
        existing: sqlite3.Row | None = connection.execute(
            "SELECT * FROM records WHERE kind=? AND record_id=?",
            (kind, record_id),
        ).fetchone()
        if existing is not None:
            if existing["payload"] != serialized:
                raise sqlite3.IntegrityError(
                    f"Conflicting content for {kind}/{record_id}."
                )
            self._decode(existing)
            return

        links: list[tuple[str, str, str]] = self._links(
            connection, kind, record_id, payload
        )
        metadata: dict[str, Any] = self._metadata(payload)
        columns: tuple[str, ...] = (
            "kind", "record_id", "payload", "content_hash",
            "created_at", "lifecycle", *_METADATA_FIELDS,
        )
        values: list[Any] = [
            kind, record_id, serialized, content_hash, _utc_now(),
            self._lifecycle(kind, payload),
            *(metadata[field] for field in _METADATA_FIELDS),
        ]
        placeholders: str = ",".join("?" for _ in columns)
        connection.execute(
            f"INSERT INTO records ({','.join(columns)}) VALUES ({placeholders})",
            values,
        )
        for relation, target_kind, target_id in links:
            connection.execute(
                "INSERT INTO record_links VALUES (?,?,?,?,?)",
                (kind, record_id, relation, target_kind, target_id),
            )
        if kind == "model_response":
            self._register_cache(connection, payload)

    def create_record(
        self, kind: str, record_id: str, payload: dict[str, Any]
    ) -> None:
        """Insert atomically; identical replays are harmless, conflicts are errors."""
        with self._connection(write=True) as connection:
            validated: dict[str, Any] = self._validate(
                connection, kind, record_id, payload
            )
            self._insert(connection, kind, record_id, validated)

    def read_record(self, kind: str, record_id: str) -> dict[str, Any]:
        """Read exactly one identity, including historical superseded versions."""
        self._known_kind(kind)
        _identifier(record_id)
        with self._connection() as connection:
            return self._decode(self._row(connection, kind, record_id))

    def list_records(
        self, kind: str, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """List payloads in record-ID order.

        Scalar filters use equality; None uses IS NULL. Lists/tuples specify
        membership, optionally including None. An empty list matches nothing.
        No supersession, status, or lifecycle exclusions are implicit.
        """
        self._known_kind(kind)
        selected_filters: dict[str, Any] = {} if filters is None else filters
        if not isinstance(selected_filters, dict):
            raise TypeError("filters must be a dictionary.")
        unknown: set[str] = set(selected_filters) - _FILTER_FIELDS
        if unknown:
            raise ValueError(f"Unsupported filters: {sorted(unknown)}.")
        clauses: list[str] = ["kind=?"]
        parameters: list[Any] = [kind]
        for field in sorted(selected_filters):
            value: Any = selected_filters[field]
            options: list[Any] = (
                list(value) if isinstance(value, (list, tuple)) else [value]
            )
            if any(
                item is not None
                and (isinstance(item, bool) or not isinstance(item, (str, int)))
                for item in options
            ):
                raise TypeError("Filter values must be strings, integers, or null.")
            nonnull: list[Any] = [item for item in options if item is not None]
            alternatives: list[str] = []
            if nonnull:
                alternatives.append(
                    f"{field} IN ({','.join('?' for _ in nonnull)})"
                )
                parameters.extend(nonnull)
            if None in options:
                alternatives.append(f"{field} IS NULL")
            clauses.append(
                "(" + " OR ".join(alternatives) + ")" if alternatives else "0"
            )
        query: str = (
            "SELECT * FROM records WHERE "
            + " AND ".join(clauses)
            + " ORDER BY record_id"
        )
        with self._connection() as connection:
            return [self._decode(row) for row in connection.execute(query, parameters)]

    def update_draft(
        self, kind: str, record_id: str, patch: dict[str, Any]
    ) -> None:
        """Replace patched top-level fields, validating the complete new record."""
        self._known_kind(kind)
        if not isinstance(patch, dict):
            raise TypeError("patch must be a dictionary.")
        with self._connection(write=True) as connection:
            row: sqlite3.Row = self._row(connection, kind, record_id)
            if row["lifecycle"] != "draft":
                raise PermissionError("Only editable drafts may be updated.")
            current: dict[str, Any] = self._decode(row)
            identity: str | None = _IDENTITY_FIELDS.get(kind)
            if identity in patch and patch[identity] != current.get(identity):
                raise ValueError("Record identity cannot be changed.")
            updated: dict[str, Any] = dict(current)
            updated.update(patch)
            validated: dict[str, Any] = self._validate(
                connection, kind, record_id, updated
            )
            # Submission is immutable and links to its task before the task's
            # recoverable status projection is closed. That status-only change
            # preserves the referenced scientific materials and authorization.
            submitted_closure: bool = False
            if kind == "annotation_task" and set(patch) == {"status"}:
                source_kind = {
                    "submitted": "annotation", "adjudicated": "adjudication"
                }.get(patch["status"])
                if source_kind is not None:
                    submitted_closure = connection.execute(
                        "SELECT 1 FROM record_links AS links JOIN records AS source "
                        "ON source.kind=links.source_kind AND source.record_id=links.source_id "
                        "WHERE links.target_kind=? AND links.target_id=? "
                        "AND links.relation='annotation_task' AND source.kind=? "
                        "AND source.lifecycle IN ('immutable','frozen') LIMIT 1",
                        (kind, record_id, source_kind),
                    ).fetchone() is not None
            # References to a draft prevent changing its scientific meaning.
            if connection.execute(
                "SELECT 1 FROM record_links "
                "WHERE target_kind=? AND target_id=? LIMIT 1",
                (kind, record_id),
            ).fetchone() and _content_hash(validated) != row["content_hash"] and not submitted_closure:
                raise PermissionError("Referenced drafts cannot be changed.")
            links: list[tuple[str, str, str]] = self._links(
                connection, kind, record_id, validated
            )
            connection.execute(
                "DELETE FROM record_links WHERE source_kind=? AND source_id=?",
                (kind, record_id),
            )
            metadata: dict[str, Any] = self._metadata(validated)
            assignments: list[str] = [
                "payload=?", "content_hash=?", "lifecycle=?",
                *(f"{field}=?" for field in _METADATA_FIELDS),
            ]
            parameters: list[Any] = [
                _canonical_json(validated), _content_hash(validated),
                self._lifecycle(kind, validated),
                *(metadata[field] for field in _METADATA_FIELDS),
                kind, record_id,
            ]
            connection.execute(
                f"UPDATE records SET {','.join(assignments)} "
                "WHERE kind=? AND record_id=?",
                parameters,
            )
            for relation, target_kind, target_id in links:
                connection.execute(
                    "INSERT INTO record_links VALUES (?,?,?,?,?)",
                    (kind, record_id, relation, target_kind, target_id),
                )

    def delete_draft(self, kind: str, record_id: str) -> None:
        """Delete an unreferenced draft; never garbage-collect source blobs."""
        self._known_kind(kind)
        with self._connection(write=True) as connection:
            row: sqlite3.Row = self._row(connection, kind, record_id)
            if row["lifecycle"] != "draft":
                raise PermissionError("Only editable drafts may be deleted.")
            if connection.execute(
                "SELECT 1 FROM record_links WHERE target_kind=? AND target_id=?",
                (kind, record_id),
            ).fetchone():
                raise PermissionError("Referenced drafts cannot be deleted.")
            connection.execute(
                "DELETE FROM records WHERE kind=? AND record_id=?",
                (kind, record_id),
            )

    def freeze_records(self, kind: str, record_ids: list[str]) -> str:
        """Freeze an order-independent, duplicate-free membership atomically."""
        self._known_kind(kind)
        if not isinstance(record_ids, list) or not record_ids:
            raise ValueError("Freezing requires a nonempty list of record IDs.")
        for record_id in record_ids:
            _identifier(record_id)
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("Freeze membership contains duplicate IDs.")
        with self._connection(write=True) as connection:
            members: list[dict[str, str]] = []
            rows: list[sqlite3.Row] = []
            for record_id in sorted(record_ids):
                row: sqlite3.Row = self._row(connection, kind, record_id)
                payload: dict[str, Any] = self._decode(row)
                if kind == "task_instance":
                    TaskInstance.model_validate(
                        payload, context={"require_content_hash": True}
                    )
                if kind in _TAXONOMY_KINDS and not payload["frozen"]:
                    raise ValueError("Finalize taxonomy semantics before freezing.")
                if kind == "audit_cohort" and not payload["locked"]:
                    raise ValueError("Lock audit membership before freezing.")
                if kind == "annotation_task" and row["lifecycle"] == "draft":
                    raise ValueError("Pending annotation tasks cannot be frozen.")
                members.append(
                    {"record_id": record_id, "content_hash": row["content_hash"]}
                )
                rows.append(row)
            manifest: dict[str, Any] = {"kind": kind, "members": members}
            manifest_hash: str = _content_hash(manifest)
            existing: sqlite3.Row | None = connection.execute(
                "SELECT payload FROM freeze_manifests WHERE manifest_hash=?",
                (manifest_hash,),
            ).fetchone()
            if existing is not None:
                if existing["payload"] != _canonical_json(manifest):
                    raise sqlite3.IntegrityError("Freeze manifest hash conflict.")
                return manifest_hash

            connection.execute(
                "INSERT INTO freeze_manifests VALUES (?,?,?)",
                (manifest_hash, _canonical_json(manifest), _utc_now()),
            )
            for row in rows:
                if row["lifecycle"] == "draft":
                    connection.execute(
                        "UPDATE records SET lifecycle='frozen' "
                        "WHERE kind=? AND record_id=?",
                        (kind, row["record_id"]),
                    )
                connection.execute(
                    "INSERT INTO freeze_members VALUES (?,?,?,?)",
                    (manifest_hash, kind, row["record_id"], row["content_hash"]),
                )
            return manifest_hash

    def save_blob(self, content: bytes) -> str:
        """Atomically publish exact bytes under their raw SHA-256 identity."""
        if not self._initialized:
            raise RuntimeError("Call ArtifactStore.initialize() before use.")
        if not isinstance(content, bytes):
            raise TypeError("Blob content must be bytes.")
        digest: str = hashlib.sha256(content).hexdigest()
        directory: Path = self._artifact_root / digest[:2]
        directory.mkdir(parents=True, exist_ok=True)
        destination: Path = directory / digest[2:]
        descriptor: int
        temporary_name: str
        descriptor, temporary_name = tempfile.mkstemp(prefix=".blob-", dir=directory)
        temporary: Path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, destination)
                _sync_directory(directory)
            except FileExistsError:
                if destination.is_symlink() or not destination.is_file():
                    raise sqlite3.IntegrityError("Unsafe existing blob destination.")
                if not _same_files(temporary, destination):
                    raise sqlite3.IntegrityError("Conflicting bytes at blob hash.")
            return digest
        finally:
            temporary.unlink(missing_ok=True)

    def _register_cache(
        self, connection: sqlite3.Connection, payload: dict[str, Any]
    ) -> None:
        request_hash: Any = payload["usage"].get("request_hash")
        if request_hash is None or payload["status"] == ResponseStatus.TRANSPORT_FAILURE:
            return
        request_kind, request_payload = self._find(
            connection, _REQUEST_KINDS, request_hash
        )
        request: dict[str, Any] = self._validate(
            connection, request_kind, request_hash, request_payload
        )
        for field in ("model_id", "run_index", "prompt_hash", "config_hash"):
            if request[field] != payload[field]:
                raise ValueError(f"Response and request disagree on {field}.")
        if (
            request["instance_id"] is not None
            and request["instance_id"] != payload["instance_id"]
        ):
            raise ValueError("Response references a different task instance.")
        existing: sqlite3.Row | None = connection.execute(
            "SELECT response_id FROM response_cache WHERE request_hash=?",
            (request_hash,),
        ).fetchone()
        if existing is not None:
            if existing["response_id"] != payload["response_id"]:
                raise sqlite3.IntegrityError(
                    "A completed response already owns this request hash."
                )
            return
        connection.execute(
            "INSERT INTO response_cache "
            "(request_hash,request_kind,response_kind,response_id) VALUES (?,?,?,?)",
            (request_hash, request_kind, "model_response", payload["response_id"]),
        )

    def get_cached_response(self, request_hash: str) -> ModelResponse | None:
        """Return the authoritative completion, including retained invalid output."""
        # Validate digest using the shared schema primitive.
        _JsonDocument(content={"request_hash": request_hash})
        if (
            not isinstance(request_hash, str)
            or len(request_hash) != 64
            or any(character not in "0123456789abcdef" for character in request_hash)
        ):
            raise ValueError("request_hash must be a lowercase SHA-256 digest.")
        with self._connection() as connection:
            cached: sqlite3.Row | None = connection.execute(
                "SELECT * FROM response_cache WHERE request_hash=?",
                (request_hash,),
            ).fetchone()
            if cached is None:
                return None
            payload: dict[str, Any] = self._decode(
                self._row(connection, "model_response", cached["response_id"])
            )
            response: ModelResponse = ModelResponse.model_validate(payload)
            if response.status == ResponseStatus.TRANSPORT_FAILURE:
                raise sqlite3.DatabaseError("Cache incorrectly references transport failure.")
            if response.usage.get("request_hash") != request_hash:
                raise sqlite3.DatabaseError("Cached response request link is inconsistent.")
            _, request_payload = self._find(connection, _REQUEST_KINDS, request_hash)
            request: _InferenceRequest = _InferenceRequest.model_validate(request_payload)
            for field in ("model_id", "run_index", "prompt_hash", "config_hash"):
                if getattr(request, field) != getattr(response, field):
                    raise sqlite3.DatabaseError("Cache identity does not match its request.")
            return response

    def set_stage_state(self, stage: str, state: dict[str, Any]) -> None:
        """Append a validated checkpoint snapshot and move the current pointer."""
        _identifier(stage)
        validated: StageState = StageState.model_validate(state)
        if validated.stage != stage:
            raise ValueError("Checkpoint stage differs from the requested stage.")
        payload: dict[str, Any] = validated.model_dump(mode="json")
        state_id: str = _content_hash(payload)
        with self._connection(write=True) as connection:
            previous: sqlite3.Row | None = connection.execute(
                "SELECT r.* FROM stage_checkpoints AS s "
                "JOIN records AS r ON r.kind=s.state_kind AND r.record_id=s.state_id "
                "WHERE s.stage=?",
                (stage,),
            ).fetchone()
            if previous is not None:
                old: StageState = StageState.model_validate(self._decode(previous))
                if old.execution_id == validated.execution_id:
                    if old.config_hash != validated.config_hash:
                        raise ValueError("Resume cannot change configuration identity.")
                    if old.input_manifests != validated.input_manifests:
                        raise ValueError("Resume cannot change prerequisite manifests.")
                    if not set(old.completed_work).issubset(validated.completed_work):
                        raise ValueError("Checkpoint cannot discard completed work.")
                    if old.status == StageStatus.COMPLETED and old != validated:
                        raise ValueError("A completed execution requires a new execution ID.")
            self._insert(connection, "stage_state", state_id, payload)
            connection.execute(
                "INSERT INTO stage_checkpoints(stage,state_kind,state_id) "
                "VALUES (?,'stage_state',?) "
                "ON CONFLICT(stage) DO UPDATE SET state_id=excluded.state_id",
                (stage, state_id),
            )

    def export_records(self, kind: str, path: str) -> None:
        """Stream a deterministic, consistent-snapshot canonical JSONL export.

        Identical existing exports are accepted. Different existing exports are
        never overwritten. Payloads are not redacted or otherwise transformed.
        """
        self._known_kind(kind)
        if not isinstance(path, str) or not path.strip():
            raise ValueError("An explicit export path is required.")
        destination: Path = Path(path).expanduser().absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor: int
        temporary_name: str
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".export-", dir=destination.parent
        )
        temporary: Path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                with self._connection() as connection:
                    cursor: sqlite3.Cursor = connection.execute(
                        "SELECT * FROM records WHERE kind=? ORDER BY record_id",
                        (kind,),
                    )
                    for row in cursor:
                        payload: dict[str, Any] = self._decode(row)
                        manifests: list[str] = [
                            item[0]
                            for item in connection.execute(
                                "SELECT manifest_hash FROM freeze_members "
                                "WHERE kind=? AND record_id=? ORDER BY manifest_hash",
                                (kind, row["record_id"]),
                            )
                        ]
                        links: list[dict[str, str]] = [
                            {
                                "relation": item["relation"],
                                "target_kind": item["target_kind"],
                                "target_id": item["target_id"],
                            }
                            for item in connection.execute(
                                "SELECT relation,target_kind,target_id FROM record_links "
                                "WHERE source_kind=? AND source_id=? "
                                "ORDER BY relation,target_kind,target_id",
                                (kind, row["record_id"]),
                            )
                        ]
                        exported: dict[str, Any] = {
                            "kind": kind,
                            "record_id": row["record_id"],
                            "content_hash": row["content_hash"],
                            "created_at": row["created_at"],
                            "lifecycle": row["lifecycle"],
                            "freeze_manifests": manifests,
                            "links": links,
                            "payload": payload,
                        }
                        stream.write(_canonical_json(exported) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, destination)
                _sync_directory(destination.parent)
            except FileExistsError:
                if destination.is_symlink() or not destination.is_file():
                    raise FileExistsError("Existing export destination is unsafe.")
                if not _same_files(temporary, destination):
                    raise FileExistsError(
                        "A different export already exists; choose a new path."
                    )
        finally:
            temporary.unlink(missing_ok=True)
