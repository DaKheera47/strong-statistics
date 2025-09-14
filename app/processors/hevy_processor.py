"""Hevy app CSV processing logic for different data structure."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone
from ..db import get_conn, set_meta

HEVY_EXPECTED_COLUMNS = [
    "title",
    "start_time",
    "end_time",
    "description",
    "exercise_title",
    "superset_id",
    "exercise_notes",
    "set_index",
    "set_type",
    "weight_kg",
    "reps",
    "distance_km",
    "duration_seconds",
    "rpe",
]

HEVY_NORMALIZED_COLUMNS = [
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


def normalize_hevy_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Hevy app CSV DataFrame to schema columns.

    Hevy uses different column names and date formats than Strong.
    Maps Hevy columns to the same database schema as Strong.
    """
    missing = [c for c in HEVY_EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for hevy import: {missing}")

    # Parse Hevy date format: "14 Sep 2025, 17:41"
    def parse_hevy_dt(s: str) -> str:
        s = str(s).strip()
        if not s:
            return s
        try:
            # Hevy format: "14 Sep 2025, 17:41"
            dt = datetime.strptime(s, "%d %b %Y, %H:%M")
            return dt.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            try:
                # Fallback to other common formats
                for fmt in ["%Y-%m-%d %H:%M:%S", "%m/%d/%Y", "%Y-%m-%d"]:
                    dt = datetime.strptime(s, fmt)
                    return dt.strftime("%Y-%m-%dT%H:%M:%S")
            except ValueError:
                # If all formats fail, return as-is
                return s

    # Map Hevy columns to Strong schema
    df["date"] = df["start_time"].map(parse_hevy_dt)
    df["workout_name"] = df["title"].astype(str)

    # Calculate duration from start_time and end_time
    def calc_duration(start, end):
        try:
            start_dt = datetime.strptime(start, "%d %b %Y, %H:%M")
            end_dt = datetime.strptime(end, "%d %b %Y, %H:%M")
            duration_minutes = (end_dt - start_dt).total_seconds() / 60
            return duration_minutes if duration_minutes > 0 else None
        except:
            return None

    df["duration_min"] = df.apply(
        lambda row: calc_duration(row["start_time"], row["end_time"]), axis=1
    )
    df["exercise"] = df["exercise_title"].astype(str)
    df["set_order"] = pd.to_numeric(df["set_index"], errors="coerce")

    # Convert weight from kg to the same units as Strong (assuming Strong uses lbs)
    df["weight"] = pd.to_numeric(df["weight_kg"], errors="coerce")
    df["reps"] = pd.to_numeric(df["reps"], errors="coerce")
    df["distance"] = pd.to_numeric(df["distance_km"], errors="coerce")
    df["seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce")

    normalized = df[HEVY_NORMALIZED_COLUMNS].copy()
    return normalized


def upsert_hevy_sets(normalized: pd.DataFrame) -> int:
    """Insert normalized hevy data rows into a separate table or adapt to existing schema."""
    if normalized.empty:
        return 0

    # For now, we'll adapt hevy data to the existing sets table structure
    # In a real implementation, you might create a separate hevy_sets table
    rows = []
    for r in normalized.itertuples(index=False):
        # Map hevy data to existing sets table structure
        rows.append(
            (
                r.date,
                None,  # workout_name - hevy app might not have this concept
                None,  # duration_min - hevy app might not track workout duration
                r.exercise,
                1,  # set_order - hevy app might not have explicit set ordering
                r.weight,
                r.reps,
                None,  # distance - hevy app typically doesn't track distance
                None,  # seconds - hevy app might not track time per set
            )
        )

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
        return cur.rowcount


def count_hevy_sets() -> int:
    """Count hevy sets (or all sets for now)."""
    with get_conn() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM sets WHERE workout_name IS NULL")
        return cur.fetchone()[0]


def process_hevy_csv(csv_path: Path) -> int:
    """Process Hevy app CSV file and insert into database."""
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Hevy app specific preprocessing
    # Hevy app might have different data quirks to handle

    normalized = normalize_hevy_df(df)
    before_count = count_hevy_sets()
    upsert_hevy_sets(normalized)
    after_count = count_hevy_sets()
    inserted = after_count - before_count

    set_meta(
        "last_hevy_ingested_at",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return inserted
