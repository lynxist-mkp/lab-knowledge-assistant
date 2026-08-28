from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from wenmai.config import Settings
from wenmai.storage.paths import store_path


@dataclass(frozen=True)
class FingerprintRecord:
    source_path: str
    sha256: str
    document_id: str
    status: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


class FingerprintStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_fingerprints (
                source_path TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL,
                document_id TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    @classmethod
    def from_settings(cls, settings: Settings) -> FingerprintStore:
        return cls(store_path(settings, "ingestion_history"))

    def get_by_source_path(self, source_path: str) -> FingerprintRecord | None:
        row = self._conn.execute(
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
        self._conn.execute(
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
        self._conn.commit()

    def delete_by_document_id(self, document_id: str) -> None:
        self._conn.execute(
            "DELETE FROM ingestion_fingerprints WHERE document_id = ?",
            (document_id,),
        )
        self._conn.commit()

    def delete_by_source_path(self, source_path: str) -> None:
        self._conn.execute(
            "DELETE FROM ingestion_fingerprints WHERE source_path = ?",
            (source_path,),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
