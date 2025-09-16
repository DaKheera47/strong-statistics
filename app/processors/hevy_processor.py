"""Hevy app CSV processing mapped to the Strong schema."""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from ..db import get_conn, set_meta, build_dedupe_keys_for_frame

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

# Columns that match the Strong schema
HEVY_NORMALIZED_COLUMNS = [
    "date",  # ISO string from start_time (same as Strong "Date")
    "workout_name",  # Hevy "title"
    "duration_min",  # (end_time - start_time) in minutes
    "exercise",  # Hevy "exercise_title"
    "set_order",  # Hevy "set_index" + 1 (Strong is 1-based)
    "weight",  # Hevy "weight_kg" (keep kg)
    "reps",  # Hevy "reps"
    "distance",  # Hevy "distance_km" -> meters
    "seconds",  # Hevy "duration_seconds" per set
]


def _parse_hevy_dt(s: str) -> Optional[str]:
    """Parse Hevy timestamps like '14 Sep 2025, 17:41' -> 'YYYY-%m-%dT%H:%M:%S'."""
    if pd.isna(s):
        return None
    s = str(s).strip()
    if not s:
        return None
    # Primary Hevy format
    try:
        dt = datetime.strptime(s, "%d %b %Y, %H:%M")
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        pass
    # Fallbacks
    for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
    return None


def _calc_duration_min(start: str, end: str) -> Optional[float]:
    """(end_time - start_time) in minutes; None if parse fails or non-positive."""
    try:
        sdt = datetime.strptime(str(start), "%d %b %Y, %H:%M")
        edt = datetime.strptime(str(end), "%d %b %Y, %H:%M")
        mins = (edt - sdt).total_seconds() / 60
        return mins if mins > 0 else None
    except Exception:
        return None


def normalize_hevy_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Hevy CSV rows to the Strong schema columns."""
    missing = [c for c in HEVY_EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for hevy import: {missing}")

    # Date (use workout start_time like Strong "Date")
    df["date"] = df["start_time"].map(_parse_hevy_dt)

    # Workout name is the "title"
    df["workout_name"] = df["title"].astype(str)

    # Workout duration in minutes (same for all rows in the workout)
    df["duration_min"] = df.apply(
        lambda r: _calc_duration_min(r["start_time"], r["end_time"]), axis=1
    )

    # Exercise & set order (Strong is 1-based)
    df["exercise"] = df["exercise_title"].astype(str)
    df["set_order"] = (
        pd.to_numeric(df["set_index"], errors="coerce").astype("Int64") + 1
    )

    # Numbers (keep kg). Convert distance_km -> meters to align with UI "m".
    df["weight"] = pd.to_numeric(df["weight_kg"], errors="coerce")
    df["reps"] = pd.to_numeric(df["reps"], errors="coerce")
    km = pd.to_numeric(df["distance_km"], errors="coerce")
    df["distance"] = (km * 1000).where(~km.isna(), other=pd.NA)
    df["seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce")

    # Select standardized subset
    normalized = df[HEVY_NORMALIZED_COLUMNS].copy()

    # Optional: convert NaN/NA to None so sqlite gets real NULLs
    normalized = normalized.where(~normalized.isna(), None)

    return normalized


def upsert_hevy_sets(normalized: pd.DataFrame) -> int:
    """Insert normalized rows into the existing Strong 'sets' table."""
    if normalized.empty:
        return 0

    # Compute dedupe keys with the same multiset algorithm
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
                float(r.duration_min) if r.duration_min is not None else None,
                r.exercise,
                int(r.set_order) if r.set_order is not None else None,
                float(r.weight) if r.weight is not None else None,
                float(r.reps) if r.reps is not None else None,
                float(r.distance) if r.distance is not None else None,
                float(r.seconds) if r.seconds is not None else None,
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


def count_all_sets() -> int:
    with get_conn() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM sets")
        return int(cur.fetchone()[0])


def process_hevy_csv(csv_path: Path) -> int:
    """Process Hevy CSV into the Strong 'sets' table."""
    df = pd.read_csv(csv_path)

    normalized = normalize_hevy_df(df)
    before = count_all_sets()
    upsert_hevy_sets(normalized)
    after = count_all_sets()
    inserted = after - before

    set_meta(
        "last_ingested_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    set_meta(
        "last_hevy_ingested_at",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    return inserted
