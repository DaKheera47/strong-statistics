"""Strong app CSV processing and aggregation logic."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import re
from datetime import datetime, timezone
from ..db import get_conn, set_meta

DATE_COL = "Date"
EXPECTED_COLUMNS = [
    "Date",
    "Workout Name",
    "Duration",
    "Exercise Name",
    "Set Order",
    "Weight",
    "Reps",
    "Distance",
    "Seconds",
    "RPE",
]

NORMALIZED_COLUMNS = [
    "date",
    "workout_name",
    "duration_min",
    "exercise",
    "set_order",
    "weight",
    "reps",
    "distance",
    "seconds",
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

    df["workout_name"] = (
        df["Workout Name"].astype(str).where(~df["Workout Name"].isna(), None)
    )
    df["exercise"] = df["Exercise Name"].astype(str)

    numeric_map = {
        "Set Order": "set_order",
        "Weight": "weight",
        "Reps": "reps",
        "Distance": "distance",
        "Seconds": "seconds",
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
        return (
            cur.rowcount
        )  # number attempted; not exact inserted (SQLite sets to # of executions). We'll compute inserted via changes().


def count_sets() -> int:
    with get_conn() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM sets")
        return cur.fetchone()[0]


def process_strong_csv(csv_path: Path) -> int:
    """Process Strong app CSV file and insert into database."""
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Convert "Duration" like "35m" or "1h 3m" -> integer minutes
    if "Duration" in df.columns:
        s = df["Duration"].astype("string").str.lower().str.strip()
        hrs = pd.to_numeric(s.str.extract(r"(\d+)\s*h", expand=False), errors="coerce")
        mins = pd.to_numeric(s.str.extract(r"(\d+)\s*m", expand=False), errors="coerce")

        total_min = (hrs.fillna(0) * 60 + mins.fillna(0)).astype("Int64")
        # if neither h nor m was present, keep it null
        total_min[hrs.isna() & mins.isna()] = pd.NA

        df["Duration"] = total_min

    normalized = normalize_df(df)
    before = count_sets()
    upsert_sets(normalized)
    after = count_sets()
    inserted = after - before
    set_meta(
        "last_ingested_at",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return inserted
