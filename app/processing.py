"""CSV processing and aggregation logic."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import re
from datetime import datetime
from .db import get_conn, set_meta

DATE_COL = "Date"
EXPECTED_COLUMNS = [
    "Date","Workout Name","Duration","Exercise Name","Set Order","Weight","Reps","Distance","Seconds","RPE"
]

NORMALIZED_COLUMNS = [
    "date","workout_name","duration_min","exercise","set_order","weight","reps","distance","seconds"
]

DURATION_RE = re.compile(r"^(?P<mins>\d+)(m)?$")


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw Strong CSV DataFrame to schema columns.

    - Validates expected columns presence (subset check).
    - Parses Date into ISO string (naive) preserving order.
    - Extracts integer minutes from Duration like '35m'.
    - Renames columns & selects standardized subset.
    - Coerces numeric columns to floats (NaNs allowed).
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Parse date strings
    def parse_dt(s: str) -> str:
        s = str(s).strip()
        if not s:
            return s
        # Provided format: YYYY-MM-DD HH:MM:SS
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    df["date"] = df[DATE_COL].map(parse_dt)

    # Duration minutes
    def parse_duration(s):
        if pd.isna(s):
            return None
        s = str(s).strip()
        if not s:
            return None
        m = DURATION_RE.match(s)
        if m:
            return float(m.group("mins"))
        return None

    df["duration_min"] = df["Duration"].map(parse_duration)

    df["workout_name"] = df["Workout Name"].astype(str).where(~df["Workout Name"].isna(), None)
    df["exercise"] = df["Exercise Name"].astype(str)

    numeric_map = {
        "Set Order": "set_order",
        "Weight": "weight",
        "Reps": "reps",
        "Distance": "distance",
        "Seconds": "seconds"
    }
    for src, dst in numeric_map.items():
        df[dst] = pd.to_numeric(df[src], errors="coerce")

    normalized = df[NORMALIZED_COLUMNS].copy()
    return normalized


def upsert_sets(normalized: pd.DataFrame) -> int:
    """Insert normalized rows into DB using INSERT OR IGNORE.

    Returns number of newly inserted rows.
    """
    if normalized.empty:
        return 0

    rows = [
        (
            r.date,
            r.workout_name,
            r.duration_min,
            r.exercise,
            int(r.set_order) if pd.notna(r.set_order) else None,
            r.weight,
            r.reps,
            r.distance,
            r.seconds,
        )
        for r in normalized.itertuples(index=False)
    ]

    with get_conn() as conn:
        cur = conn.executemany(
            """
            INSERT OR IGNORE INTO sets
            (date, workout_name, duration_min, exercise, set_order, weight, reps, distance, seconds)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        conn.commit()
        return cur.rowcount  # number attempted; not exact inserted (SQLite sets to # of executions). We'll compute inserted via changes().


def count_sets() -> int:
    with get_conn() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM sets")
        return cur.fetchone()[0]


def process_csv_to_db(csv_path: Path) -> int:
    df = pd.read_csv(csv_path)
    normalized = normalize_df(df)
    before = count_sets()
    upsert_sets(normalized)
    after = count_sets()
    inserted = after - before
    set_meta("last_ingested_at", datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
    return inserted


def compute_session_agg():
    """Return list of dict session aggregates (date-level volume & duration)."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT date(substr(date,1,10)) as session_date,
                   SUM(COALESCE(weight,0) * COALESCE(reps,0)) as volume,
                   AVG(duration_min) as duration
            FROM sets
            GROUP BY session_date
            ORDER BY session_date
            """
        )
        return [dict(r) for r in cur.fetchall()]


def top_exercises(limit: int = 15):
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT exercise,
                   SUM(COALESCE(weight,0) * COALESCE(reps,0)) AS volume
            FROM sets
            GROUP BY exercise
            ORDER BY volume DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def prs(exercise: str):
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT date, weight, reps
            FROM sets
            WHERE exercise = ? AND weight IS NOT NULL
            ORDER BY weight DESC, reps DESC
            LIMIT 1
            """,
            (exercise,),
        )
        best = cur.fetchone()
        if not best:
            return {}
        return {
            "exercise": exercise,
            "max_weight": best[1],
            "best_set": {"date": best[0][:10], "weight": best[1], "reps": best[2]},
        }


def tuesday_strength():
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT substr(date,1,10) as d, exercise, MAX(weight) as max_weight
            FROM sets
            WHERE weight IS NOT NULL AND strftime('%w', date) = '2'  -- Tuesday (0=Sun)
            GROUP BY d, exercise
            ORDER BY d, exercise
            """
        )
        return [ {"date": row[0], "exercise": row[1], "max_weight": row[2]} for row in cur.fetchall() ]


def exercise_progress(exercise: str):
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT substr(date,1,10) as d,
                   MAX(weight) as max_weight,
                   MAX(COALESCE(weight,0) * COALESCE(reps,0)) as top_set_volume
            FROM sets
            WHERE exercise = ?
            GROUP BY d
            ORDER BY d
            """,
            (exercise,),
        )
        return [ {"date": r[0], "max_weight": r[1], "top_set_volume": r[2]} for r in cur.fetchall() ]


def list_exercises():
    with get_conn() as conn:
        cur = conn.execute("SELECT DISTINCT exercise FROM sets ORDER BY exercise")
        return [r[0] for r in cur.fetchall()]
