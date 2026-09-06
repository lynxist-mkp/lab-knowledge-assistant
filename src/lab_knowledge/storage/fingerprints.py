from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from lab_knowledge.config import Settings
from lab_knowledge.storage.paths import collection_storage_bindings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingestion_fingerprints (
    source_path TEXT PRIMARY KEY,
    sha256 TEXT NOT NULL,
    document_id TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class FingerprintRecord:
    source_path: str
    sha256: str
    document_id: str
    status: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


class FingerprintStore:
    def __init__(self, db_path: Path, *, write_path: Path | None = None) -> None:
        self._read_path = db_path
        self._write_path = write_path or db_path
        self._read_conn = self._open(self._read_path)
        self._write_conn = (
            self._read_conn
            if self._write_path == self._read_path
            else self._open(self._write_path)
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> FingerprintStore:
        bindings = collection_storage_bindings(settings)
        return cls(
            bindings.ingestion_history_read_path(),
            write_path=bindings.ingestion_history_path,
        )

    @staticmethod
    def _open(path: Path) -> sqlite3.Connection:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute(_SCHEMA)
        conn.commit()
        return conn

    def get_by_source_path(self, source_path: str) -> FingerprintRecord | None:
        row = self._read_conn.execute(
            """
            SELECT source_path, sha256, document_id, status
            FROM ingestion_fingerprints
            WHERE source_path = ?
            """,
            (source_path,),
        ).fetchone()
        if row is None:
            return None
        return FingerprintRecord(
            source_path=row["source_path"],
            sha256=row["sha256"],
            document_id=row["document_id"],
            status=row["status"],
        )

    def upsert(
        self,
        source_path: str,
        sha256: str,
        document_id: str,
        status: str,
    ) -> None:
        self._write_conn.execute(
            """
            INSERT INTO ingestion_fingerprints (
                source_path, sha256, document_id, status, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                sha256 = excluded.sha256,
                document_id = excluded.document_id,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            (source_path, sha256, document_id, status, _now()),
        )
        self._write_conn.commit()

    def delete_by_document_id(self, document_id: str) -> None:
        self._write_conn.execute(
            "DELETE FROM ingestion_fingerprints WHERE document_id = ?",
            (document_id,),
        )
        self._write_conn.commit()

    def delete_by_source_path(self, source_path: str) -> None:
        self._write_conn.execute(
            "DELETE FROM ingestion_fingerprints WHERE source_path = ?",
            (source_path,),
        )
        self._write_conn.commit()

    def close(self) -> None:
        self._read_conn.close()
        if self._write_conn is not self._read_conn:
            self._write_conn.close()
