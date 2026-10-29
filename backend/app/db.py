"""SQLite：参与者、会谈、事件日志、话语、场景版本。"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS participants (
    id TEXT PRIMARY KEY,               -- 编号，如 P001
    created_at REAL NOT NULL,
    consent INTEGER NOT NULL DEFAULT 0,
    consent_at REAL,
    region_style TEXT,
    scene_id TEXT,
    notes TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    participant_id TEXT NOT NULL REFERENCES participants(id) ON DELETE CASCADE,
    condition TEXT NOT NULL,
    duration_min INTEGER NOT NULL,
    region_style TEXT,
    scene_id TEXT,
    started_at REAL,
    ended_at REAL,
    end_reason TEXT,
    status TEXT NOT NULL DEFAULT 'created'
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    ts REAL NOT NULL,
    t_rel REAL,
    type TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, ts);
CREATE TABLE IF NOT EXISTS utterances (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    speaker TEXT NOT NULL,             -- elder | companion
    text TEXT NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    labels TEXT DEFAULT '[]',
    turn_id TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_utt_session ON utterances(session_id, start_s);
CREATE TABLE IF NOT EXISTS scene_versions (
    scene_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    manifest TEXT NOT NULL,
    cause TEXT NOT NULL,               -- {"kind": edit|narration|system|undo|create, ...}
    diff TEXT,                         -- 用于撤销的实体级差异
    undone INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    PRIMARY KEY (scene_id, version)
);
"""


class Database:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)
        self._lock = threading.RLock()

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def query(self, sql: str, params: tuple | dict = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def one(self, sql: str, params: tuple | dict = ()) -> Optional[dict]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def transaction(self):
        return _Tx(self)

    # ---- 事件日志 ----
    def log_event(self, session_id: Optional[str], type_: str, payload: dict[str, Any],
                  t_rel: Optional[float] = None) -> None:
        self.execute("INSERT INTO events (session_id, ts, t_rel, type, payload) VALUES (?,?,?,?,?)",
                     (session_id, time.time(), t_rel, type_, json.dumps(payload, ensure_ascii=False)))

    def events(self, session_id: str) -> list[dict]:
        rows = self.query("SELECT * FROM events WHERE session_id=? ORDER BY ts, id", (session_id,))
        for r in rows:
            r["payload"] = json.loads(r["payload"])
        return rows

    def close(self) -> None:
        self._conn.close()


class _Tx:
    def __init__(self, db: Database):
        self.db = db

    def __enter__(self):
        self.db._lock.acquire()
        self.db._conn.execute("BEGIN")
        return self.db

    def __exit__(self, exc_type, exc, tb):
        try:
            self.db._conn.execute("ROLLBACK" if exc_type else "COMMIT")
        finally:
            self.db._lock.release()
        return False
