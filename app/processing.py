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


def progressive_overload_data():
    """Get strength progression for top exercises over time."""
    with get_conn() as conn:
        # Get top 6 exercises by total volume
        top_exercises_cur = conn.execute(
            """
            SELECT exercise
            FROM sets
            WHERE weight IS NOT NULL
            GROUP BY exercise
            ORDER BY SUM(COALESCE(weight,0) * COALESCE(reps,0)) DESC
            LIMIT 6
            """
        )
        top_exercises = [row[0] for row in top_exercises_cur.fetchall()]
        
        result = {}
        for exercise in top_exercises:
            cur = conn.execute(
                """
                SELECT date(substr(date,1,10)) as workout_date,
                       MAX(weight) as max_weight,
                       MAX(COALESCE(weight,0) * COALESCE(reps,0)) as best_set_volume,
                       -- Estimate 1RM using Epley formula: weight * (1 + reps/30)
                       MAX(COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0)) as estimated_1rm
                FROM sets
                WHERE exercise = ? AND weight IS NOT NULL
                GROUP BY workout_date
                ORDER BY workout_date
                """,
                (exercise,)
            )
            
            data = []
            for row in cur.fetchall():
                data.append({
                    "date": row[0],
                    "max_weight": row[1],
                    "best_set_volume": row[2],
                    "estimated_1rm": round(row[3], 1)
                })
            result[exercise] = data
        
        return result


def volume_progression():
    """Get training volume progression over time with weekly rolling averages."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT date(substr(date,1,10)) as workout_date,
                   SUM(COALESCE(weight,0) * COALESCE(reps,0)) as total_volume,
                   COUNT(DISTINCT exercise) as exercises_count,
                   AVG(duration_min) as avg_duration
            FROM sets
            GROUP BY workout_date
            ORDER BY workout_date
            """
        )
        
        data = []
        volumes = []
        for row in cur.fetchall():
            volume = row[1]
            volumes.append(volume)
            
            # Calculate 7-day rolling average
            rolling_avg = sum(volumes[-7:]) / min(len(volumes), 7)
            
            data.append({
                "date": row[0],
                "volume": volume,
                "volume_7day_avg": round(rolling_avg, 1),
                "exercises_count": row[2],
                "duration": row[3]
            })
        
        return data


def personal_records_timeline():
    """Get timeline of personal records (new max weights achieved)."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            WITH ranked_sets AS (
                SELECT exercise, date, weight, reps,
                       LAG(weight) OVER (PARTITION BY exercise ORDER BY date, weight) as prev_weight
                FROM sets
                WHERE weight IS NOT NULL
                ORDER BY exercise, date, weight
            )
            SELECT exercise, 
                   date(substr(date,1,10)) as pr_date,
                   weight,
                   reps,
                   COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0) as estimated_1rm
            FROM ranked_sets
            WHERE prev_weight IS NULL OR weight > prev_weight
            ORDER BY date DESC
            """
        )
        
        return [{
            "exercise": row[0],
            "date": row[1], 
            "weight": row[2],
            "reps": row[3],
            "estimated_1rm": round(row[4], 1)
        } for row in cur.fetchall()]


def training_consistency():
    """Get training frequency and consistency metrics."""
    with get_conn() as conn:
        # Get workout frequency by week
        cur = conn.execute(
            """
            SELECT strftime('%Y-W%W', date) as week,
                   COUNT(DISTINCT date(substr(date,1,10))) as workouts_per_week,
                   SUM(COALESCE(weight,0) * COALESCE(reps,0)) as weekly_volume,
                   AVG(duration_min) as avg_duration
            FROM sets
            GROUP BY week
            ORDER BY week
            """
        )
        
        weekly_data = []
        for row in cur.fetchall():
            weekly_data.append({
                "week": row[0],
                "workouts": row[1],
                "volume": row[2],
                "avg_duration": row[3]
            })
        
        # Get monthly summary
        cur = conn.execute(
            """
            SELECT strftime('%Y-%m', date) as month,
                   COUNT(DISTINCT date(substr(date,1,10))) as workouts_per_month,
                   SUM(COALESCE(weight,0) * COALESCE(reps,0)) as monthly_volume,
                   COUNT(DISTINCT exercise) as unique_exercises
            FROM sets
            GROUP BY month
            ORDER BY month
            """
        )
        
        monthly_data = []
        for row in cur.fetchall():
            monthly_data.append({
                "month": row[0],
                "workouts": row[1],
                "volume": row[2], 
                "unique_exercises": row[3]
            })
        
        return {
            "weekly": weekly_data,
            "monthly": monthly_data
        }


def strength_balance():
    """Analyze strength balance across muscle groups/movement patterns."""
    with get_conn() as conn:
        # Categorize exercises by movement pattern
        movement_patterns = {
            'Push': ['Chest Press', 'Shoulder Press', 'Push'],
            'Pull': ['Seated Row', 'Lat Pulldown', 'Bent Over Row', 'Pull'],
            'Squat': ['Squat', 'Leg Press'],
            'Deadlift': ['Deadlift', 'Romanian Deadlift'],
            'Accessory': ['Bicep Curl', 'Pec Deck', 'Leg Extension', 'Glute Kickback']
        }
        
        result = {}
        for pattern, keywords in movement_patterns.items():
            like_conditions = ' OR '.join([f"exercise LIKE '%{keyword}%'" for keyword in keywords])
            
            cur = conn.execute(f"""
                SELECT date(substr(date,1,10)) as workout_date,
                       MAX(weight) as max_weight,
                       SUM(COALESCE(weight,0) * COALESCE(reps,0)) as total_volume,
                       MAX(COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0)) as best_estimated_1rm
                FROM sets
                WHERE ({like_conditions}) AND weight IS NOT NULL
                GROUP BY workout_date
                HAVING COUNT(*) > 0
                ORDER BY workout_date
            """)
            
            pattern_data = []
            for row in cur.fetchall():
                pattern_data.append({
                    "date": row[0],
                    "max_weight": row[1],
                    "volume": row[2],
                    "estimated_1rm": round(row[3], 1)
                })
            
            if pattern_data:  # Only include if we have data
                result[pattern] = pattern_data
        
        return result


def exercise_analysis(exercise: str):
    """Detailed analysis for a specific exercise."""
    with get_conn() as conn:
        # Get all sets for this exercise with trend analysis
        cur = conn.execute(
            """
            SELECT date(substr(date,1,10)) as workout_date,
                   weight, reps,
                   COALESCE(weight,0) * COALESCE(reps,0) as volume,
                   COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0) as estimated_1rm
            FROM sets
            WHERE exercise = ? AND weight IS NOT NULL
            ORDER BY date, set_order
            """,
            (exercise,)
        )
        
        all_sets = []
        for row in cur.fetchall():
            all_sets.append({
                "date": row[0],
                "weight": row[1],
                "reps": row[2], 
                "volume": row[3],
                "estimated_1rm": round(row[4], 1)
            })
        
        # Get session-level summary
        cur = conn.execute(
            """
            SELECT date(substr(date,1,10)) as workout_date,
                   MAX(weight) as max_weight,
                   SUM(COALESCE(weight,0) * COALESCE(reps,0)) as session_volume,
                   COUNT(*) as sets_count,
                   AVG(reps) as avg_reps,
                   MAX(COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0)) as best_estimated_1rm
            FROM sets
            WHERE exercise = ? AND weight IS NOT NULL
            GROUP BY workout_date
            ORDER BY workout_date
            """,
            (exercise,)
        )
        
        session_summary = []
        for row in cur.fetchall():
            session_summary.append({
                "date": row[0],
                "max_weight": row[1],
                "session_volume": row[2],
                "sets_count": row[3],
                "avg_reps": round(row[4], 1),
                "estimated_1rm": round(row[5], 1)
            })
        
        return {
            "all_sets": all_sets,
            "session_summary": session_summary
        }


def list_exercises():
    with get_conn() as conn:
        cur = conn.execute("SELECT DISTINCT exercise FROM sets ORDER BY exercise")
        return [r[0] for r in cur.fetchall()]
