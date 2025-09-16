"""Database utilities for strong-statistics.

Responsible for initializing the SQLite schema, providing a context
manager for connections, and enforcing deduplication.
"""

from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path
import threading
import time
import logging
import math
from typing import Any, Iterable, Sequence, Optional

import pandas as pd  # used for backfill/migration

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
  seconds REAL,
  -- canonical duplication guard (unique per logical row)
  dedupe_key TEXT
);

-- Keep the legacy unique for defense-in-depth (NULLs don't collide in SQLite).
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
    """If the old lifting.db exists and strong.db does not, rename it."""
    try:
        if LEGACY_DB_PATH.exists() and not DB_PATH.exists():
            LEGACY_DB_PATH.rename(DB_PATH)
            logger.info(
                "Migrated legacy database %s -> %s", LEGACY_DB_PATH.name, DB_PATH.name
            )
    except Exception as e:  # pragma: no cover
        logger.error("Legacy DB migration failed: %s", e)


def init_db() -> None:
    """Initialize database schema if not present, then ensure dedupe is enforced."""
    _migrate_legacy_db_if_needed()
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)
    _ensure_dedupe_column_and_index()


def set_meta(key: str, value: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, value)
        )
        conn.commit()


def get_meta(key: str) -> str | None:
    with get_conn() as conn:
        cur = conn.execute("SELECT value FROM meta WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None


@contextmanager
def get_conn():
    """Yield a SQLite connection with row factory as dict-like tuples."""
    start_wait = time.time()
    acquired = _lock.acquire(timeout=_LOCK_MAX_WAIT_SEC)
    if not acquired:  # pragma: no cover
        waited = time.time() - start_wait
        logger.error(
            "DB lock acquisition failed after %.2fs (possible deadlock)", waited
        )
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


# ---------- Canonicalization helpers ----------


def _is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


def canon_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def canon_num(x: Any) -> str:
    """Deterministic numeric rendering for key parts."""
    if x is None or _is_nan(x):
        return ""
    try:
        xf = float(x)
    except Exception:
        return canon_str(x)
    if abs(xf - round(xf)) < 1e-9:
        return str(int(round(xf)))
    return f"{xf:.6f}".rstrip("0").rstrip(".")


def _compose_key(
    date: str, workout: str | None, ex: str, occ: int, w: str, r: str, d: str, s: str
) -> str:
    return "|".join(
        [canon_str(date), canon_str(workout), canon_str(ex), str(int(occ)), w, r, d, s]
    )


# ---------- Public helper for processors ----------


def build_dedupe_keys_for_frame(
    df: pd.DataFrame, tiebreaker_col: Optional[str] = None
) -> pd.Series:
    """
    Given a normalized DataFrame with columns at least:
      date, workout_name, exercise, weight, reps, distance, seconds
    return a pandas Series of dedupe_key strings aligned with df.index.

    Algorithm:
      - Canonicalize numeric strings for (weight,reps,distance,seconds)
      - Make a value-tuple string K = "w|r|d|s"
      - Group by (date, workout_name, exercise, K)
      - Assign occurrence index = cumcount() + 1 using a deterministic sort:
          * If tiebreaker_col is present, sort by it (e.g., DB 'id')
          * Else sort by the original row position
      - Build key = date|workout|exercise|occ|w|r|d|s
    """
    work = df.copy()

    # Keep a stable original row position for alignment at the end
    work["_orig_pos"] = range(len(work))

    # Canonicalized numeric strings
    w = work["weight"].map(canon_num) if "weight" in work.columns else ""
    r = work["reps"].map(canon_num) if "reps" in work.columns else ""
    d = work["distance"].map(canon_num) if "distance" in work.columns else ""
    s = work["seconds"].map(canon_num) if "seconds" in work.columns else ""

    work["_w"] = w
    work["_r"] = r
    work["_d"] = d
    work["_s"] = s
    work["_tuple"] = work["_w"] + "|" + work["_r"] + "|" + work["_d"] + "|" + work["_s"]

    grp_cols = ["date", "workout_name", "exercise", "_tuple"]

    # Choose deterministic tiebreaker
    if tiebreaker_col and tiebreaker_col in work.columns:
        sort_cols = grp_cols + [tiebreaker_col]
    else:
        sort_cols = grp_cols + ["_orig_pos"]

    # Stable sort so equal keys keep order by tiebreaker
    work = work.sort_values(sort_cols, kind="mergesort")

    # Occurrence per identical tuple inside a workout/exercise
    work["_occ"] = work.groupby(grp_cols).cumcount() + 1

    # Compose the final keys in the sorted frame
    keys_sorted = [
        _compose_key(
            date=row["date"],
            workout=row.get("workout_name"),
            ex=row["exercise"],
            occ=int(row["_occ"]),
            w=row["_w"],
            r=row["_r"],
            d=row["_d"],
            s=row["_s"],
        )
        for _, row in work.iterrows()
    ]

    # Map keys back to original row order
    keys_series = pd.Series(keys_sorted, index=work["_orig_pos"]).sort_index()
    # Reindex to caller's original index shape/order
    keys_series.index = df.index  # align index labels
    return keys_series


# ---------- Migration to enforce dedupe_key uniqueness ----------


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row["name"] == column for row in cur.fetchall())


def _ensure_dedupe_column_and_index() -> None:
    """Add dedupe_key, backfill with multiset-based keys, drop dupes, and make UNIQUE."""
    with get_conn() as conn:
        # 1) Add column if needed
        if not _column_exists(conn, "sets", "dedupe_key"):
            try:
                conn.execute("ALTER TABLE sets ADD COLUMN dedupe_key TEXT")
                conn.commit()
                logger.info("Added column sets.dedupe_key")
            except Exception as e:  # pragma: no cover
                logger.error("Adding dedupe_key column failed: %s", e)

        # 2) Backfill all keys using the NEW algorithm
        _backfill_all_keys_with_multiset_algo(conn)

        # 3) Remove any duplicates by key (keep lowest id)
        _remove_duplicate_rows_by_key(conn)

        # 4) Create UNIQUE index on the dedupe key
        try:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_sets_dedupe ON sets(dedupe_key)"
            )
            conn.commit()
        except Exception as e:  # pragma: no cover
            logger.error("Creating unique index ux_sets_dedupe failed: %s", e)


def _backfill_all_keys_with_multiset_algo(conn: sqlite3.Connection) -> None:
    # Pull minimal columns
    df = pd.read_sql_query(
        """
        SELECT id, date, workout_name, exercise, set_order, weight, reps, distance, seconds
        FROM sets
        """,
        conn,
    )

    if df.empty:
        return

    # Use DB id as deterministic tiebreaker for backfill
    keys = build_dedupe_keys_for_frame(df, tiebreaker_col="id")

    updates = [(k, int(i)) for k, i in zip(keys.tolist(), df["id"].tolist())]
    conn.executemany("UPDATE sets SET dedupe_key=? WHERE id=?", updates)
    conn.commit()
    logger.info("Backfilled dedupe_key for %d rows (multiset algo)", len(updates))


def _remove_duplicate_rows_by_key(conn: sqlite3.Connection) -> None:
    """Keep the lowest id per dedupe_key, delete the rest."""
    dup_ids = conn.execute(
        """
        SELECT id
        FROM (
          SELECT id,
                 ROW_NUMBER() OVER (PARTITION BY dedupe_key ORDER BY id ASC) AS rn
          FROM sets
          WHERE dedupe_key IS NOT NULL AND dedupe_key <> ''
        )
        WHERE rn > 1
    """
    ).fetchall()

    if not dup_ids:
        return

    to_delete = [(row[0],) for row in dup_ids]
    # Chunk deletes
    for i in range(0, len(to_delete), 500):
        conn.executemany("DELETE FROM sets WHERE id=?", to_delete[i : i + 500])
    conn.commit()
    logger.info("Removed %d duplicate rows by dedupe_key", len(to_delete))
