"""Strong app CSV processing and aggregation logic."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone
from ..db import get_conn, set_meta, build_dedupe_keys_for_frame

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


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw Strong CSV DataFrame to schema columns."""
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

    # Duration minutes - handle both old format (seconds) and new format (e.g., "35m")
    def parse_duration(s):
        if pd.isna(s):
            return None
        try:
            s_str = str(s).strip()
            # New format: "35m" (minutes with 'm' suffix)
            if s_str.endswith('m'):
                return float(s_str[:-1])
            # Old format: numeric seconds
            seconds = float(s_str)
            return seconds / 60.0
        except (ValueError, TypeError):
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
    """Insert normalized rows using OR IGNORE with a multiset-based dedupe_key."""
    if normalized.empty:
        return 0

    # Compute dedupe keys using stable multiset algorithm
    dedupe_keys = build_dedupe_keys_for_frame(
        normalized[
            [
                "date",
                "workout_name",
                "exercise",
                "weight",
                "reps",
                "distance",
                "seconds",
            ]
        ]
    )

    rows = []
    for r, k in zip(normalized.itertuples(index=False), dedupe_keys.tolist()):
        rows.append(
            (
                r.date,
                r.workout_name,
                r.duration_min,
                r.exercise,
                int(r.set_order) if pd.notna(r.set_order) else None,
                r.weight if pd.notna(r.weight) else None,
                r.reps if pd.notna(r.reps) else None,
                r.distance if pd.notna(r.distance) else None,
                r.seconds if pd.notna(r.seconds) else None,
                k,
            )
        )

    with get_conn() as conn:
        cur = conn.executemany(
            """
            INSERT OR IGNORE INTO sets
            (date, workout_name, duration_min, exercise, set_order, weight, reps, distance, seconds, dedupe_key)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        conn.commit()
        return cur.rowcount


def count_sets() -> int:
    with get_conn() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM sets")
        return cur.fetchone()[0]


def process_strong_csv(csv_path: Path) -> int:
    """Process Strong app CSV file and insert into database."""
    df = pd.read_csv(csv_path, sep=None, engine="python")

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
