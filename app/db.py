"""Database utilities for strong-statistics.

Responsible for initializing the SQLite schema and providing a context
manager for connections.
"""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path
import threading
import time
import logging

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# New canonical DB filename after project rename
DB_PATH = DATA_DIR / "strong.db"
LEGACY_DB_PATH = DATA_DIR / "lifting.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# SQLite connections are NOT thread-safe by default when shared;
# we create short-lived connections via contextmanager.
_lock = threading.Lock()
_LOCK_MAX_WAIT_SEC = 5  # fail fast if something holds the lock too long
logger = logging.getLogger("strong.db")

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

def _migrate_legacy_db_if_needed() -> None:
  """If the old lifting.db exists and strong.db does not, rename it.

  This provides a seamless upgrade path for existing deployments.
  Safe to call multiple times (idempotent).
  """
  try:
    if LEGACY_DB_PATH.exists() and not DB_PATH.exists():
      LEGACY_DB_PATH.rename(DB_PATH)
      logger.info("Migrated legacy database %s -> %s", LEGACY_DB_PATH.name, DB_PATH.name)
  except Exception as e:  # pragma: no cover - best effort
    logger.error("Legacy DB migration failed: %s", e)


def init_db() -> None:
  """Initialize database schema if not present (and migrate legacy file)."""
  _migrate_legacy_db_if_needed()
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
  """Yield a SQLite connection with row factory as dict-like tuples.

  Adds defensive diagnostics so if the global lock is ever held
  unexpectedly long (e.g. a crashed thread), we surface an error
  instead of silently hanging the request.
  """
  start_wait = time.time()
  acquired = _lock.acquire(timeout=_LOCK_MAX_WAIT_SEC)
  if not acquired:  # pragma: no cover - only triggers on pathological hang
    waited = time.time() - start_wait
    logger.error("DB lock acquisition failed after %.2fs (possible deadlock)", waited)
    raise RuntimeError("database busy: internal lock timeout")
  try:
    waited = time.time() - start_wait
    if waited > 0.25:
      logger.warning("DB lock waited %.3fs (unexpected contention)", waited)
    conn = sqlite3.connect(DB_PATH, timeout=2)
    try:
      conn.row_factory = sqlite3.Row
      yield conn
    finally:
      conn.close()
  finally:
    _lock.release()
