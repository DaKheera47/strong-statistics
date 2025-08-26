"""Database utilities for the Lifting Pipeline.

Responsible for initializing the SQLite schema and providing a context
manager for connections.
"""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path
import threading

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "lifting.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# SQLite connections are NOT thread-safe by default when shared;
# we create short-lived connections via contextmanager.
_lock = threading.Lock()

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sets (
  id INTEGER PRIMARY KEY,
  date TEXT NOT NULL,
  workout_name TEXT,
  duration_min REAL,
  exercise TEXT NOT NULL,
  set_order INTEGER,
  weight REAL,
  reps REAL,
  distance REAL,
  seconds REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_sets_row
  ON sets(date, workout_name, exercise, set_order, weight, reps, seconds);
CREATE INDEX IF NOT EXISTS idx_sets_date_exercise
  ON sets(date, exercise);

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT
);
"""

def init_db() -> None:
    """Initialize database schema if not present."""
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)


def set_meta(key: str, value: str) -> None:
    with get_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, value))
        conn.commit()


def get_meta(key: str) -> str | None:
    with get_conn() as conn:
        cur = conn.execute("SELECT value FROM meta WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

@contextmanager
def get_conn():
    """Yield a SQLite connection with row factory as dict-like tuples."""
    with _lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            conn.close()
