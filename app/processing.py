"""CSV processing and aggregation logic."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import re
from datetime import datetime
import sqlite3
from contextlib import contextmanager
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


# Strong App inspired analytics functions

def get_exercise_records():
    """Get personal records for all exercises - like Strong's Records screen"""
    with get_conn() as conn:
        cur = conn.execute("""
            SELECT 
                exercise,
                MAX(weight) as max_weight,
                MAX(reps) as max_reps,
                MAX(COALESCE(weight,0) * COALESCE(reps,0)) as max_volume_set,
                COUNT(*) as total_sets,
                MIN(date) as first_performed,
                MAX(date) as last_performed,
                MAX(COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0)) as estimated_1rm
            FROM sets
            WHERE weight IS NOT NULL AND reps > 0
            GROUP BY exercise
            ORDER BY max_weight DESC
        """)
        
        records = []
        for row in cur.fetchall():
            records.append({
                'exercise': row[0],
                'max_weight': row[1],
                'max_reps': row[2],
                'max_volume_set': row[3],
                'estimated_1rm': round(row[7], 1),
                'total_sets': row[4],
                'first_performed': row[5],
                'last_performed': row[6]
            })
        
        return records


def get_body_measurements():
    """Get body measurements data - like Strong's measurements feature"""
    # For now, we'll simulate this since we don't have a measurements table
    # In a real app, you'd have a separate measurements table
    with get_conn() as conn:
        # Get workout dates to simulate body weight tracking
        cur = conn.execute("""
            SELECT date(substr(date,1,10)) as workout_date, COUNT(*) as workout_count
            FROM sets
            GROUP BY workout_date
            ORDER BY workout_date
        """)
        
        result = cur.fetchall()
        
        # Simulate body weight data (in a real app, this would come from a measurements table)
        measurements = []
        base_weight = 80  # kg
        
        for i, row in enumerate(result):
            # Simulate gradual weight changes over time
            weight_change = (i * 0.1) - 2  # Slight weight loss trend
            measurements.append({
                'date': row[0],
                'weight': round(base_weight + weight_change, 1),
                'type': 'weight'
            })
        
        return measurements


def get_workout_calendar():
    """Get workout calendar data - like Strong's workout frequency tracking"""
    with get_conn() as conn:
        cur = conn.execute("""
            SELECT 
                date(substr(date,1,10)) as workout_date,
                COUNT(DISTINCT exercise) as exercises_performed,
                SUM(COALESCE(weight,0) * COALESCE(reps,0)) as total_volume,
                COUNT(*) as total_sets,
                AVG(duration_min) as avg_duration_minutes
            FROM sets
            WHERE weight IS NOT NULL AND reps > 0
            GROUP BY workout_date
            ORDER BY workout_date
        """)
        
        calendar_data = []
        for row in cur.fetchall():
            calendar_data.append({
                'date': row[0],
                'exercises_performed': row[1],
                'total_volume': round(row[2], 1),
                'total_sets': row[3],
                'duration_minutes': round(row[4], 1) if row[4] else 0
            })
        
        return calendar_data


def get_exercise_detail(exercise_name):
    """Get detailed statistics for a specific exercise - like Strong's Exercise Detail Screen"""
    with get_conn() as conn:
        # Get all sets for this exercise
        cur = conn.execute("""
            SELECT 
                date(substr(date,1,10)) as workout_date,
                weight,
                reps,
                COALESCE(weight,0) * COALESCE(reps,0) as volume,
                COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0) as estimated_1rm
            FROM sets
            WHERE exercise = ? AND weight IS NOT NULL AND reps > 0
            ORDER BY date, set_order
        """, (exercise_name,))
        
        result = cur.fetchall()
        
        if not result:
            return None
        
        # Calculate progression over time
        sets_data = []
        max_weight = 0
        max_reps = 0
        max_volume = 0
        max_1rm = 0
        
        for row in result:
            set_data = {
                'date': row[0],
                'weight': row[1],
                'reps': row[2],
                'volume': row[3],
                'estimated_1rm': row[4]
            }
            sets_data.append(set_data)
            
            max_weight = max(max_weight, row[1])
            max_reps = max(max_reps, row[2])
            max_volume = max(max_volume, row[3])
            max_1rm = max(max_1rm, row[4])
        
        # Get workout frequency
        frequency_cur = conn.execute("""
            SELECT COUNT(DISTINCT date(substr(date,1,10))) as workout_days
            FROM sets
            WHERE exercise = ?
        """, (exercise_name,))
        
        workout_days = frequency_cur.fetchone()[0]
        
        # Calculate days since first and last workout
        first_date = result[0][0]
        last_date = result[-1][0]
        
        return {
            'exercise_name': exercise_name,
            'total_sets': len(sets_data),
            'workout_days': workout_days,
            'max_weight': max_weight,
            'max_reps': max_reps,
            'max_volume': max_volume,
            'max_1rm': round(max_1rm, 1),
            'first_workout': first_date,
            'last_workout': last_date,
            'sets_history': sets_data
        }


def get_muscle_group_balance():
    """Analyze strength balance across muscle groups - Strong's muscle balance feature"""
    with get_conn() as conn:
        # Categorize exercises by muscle groups (more detailed than movement patterns)
        muscle_groups = {
            'Chest': ['Chest Press', 'Bench Press', 'Pec Deck', 'Incline'],
            'Back': ['Seated Row', 'Lat Pulldown', 'Bent Over Row', 'Pull-up', 'Chin'],
            'Shoulders': ['Shoulder Press', 'Lateral Raise', 'Military Press', 'Overhead Press'],
            'Legs': ['Squat', 'Leg Press', 'Leg Extension', 'Leg Curl', 'Calf'],
            'Arms': ['Bicep Curl', 'Tricep', 'Hammer Curl'],
            'Core': ['Plank', 'Crunch', 'Russian Twist', 'Dead Bug']
        }
        
        balance_data = {}
        
        for muscle_group, keywords in muscle_groups.items():
            like_conditions = ' OR '.join([f"exercise LIKE '%{keyword}%'" for keyword in keywords])
            
            cur = conn.execute(f"""
                SELECT 
                    MAX(COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0)) as max_estimated_1rm,
                    SUM(COALESCE(weight,0) * COALESCE(reps,0)) as total_volume,
                    COUNT(*) as total_sets,
                    COUNT(DISTINCT date(substr(date,1,10))) as workout_days,
                    MAX(weight) as max_weight
                FROM sets
                WHERE ({like_conditions}) AND weight IS NOT NULL
            """)
            
            result = cur.fetchone()
            
            if result and result[0]:  # Only include if we have data
                balance_data[muscle_group] = {
                    'max_estimated_1rm': round(result[0], 1),
                    'total_volume': round(result[1], 1),
                    'total_sets': result[2],
                    'workout_days': result[3],
                    'max_weight': result[4]
                }
        
        return balance_data


def get_training_streak():
    """Calculate current and longest training streaks - like Strong's consistency tracking"""
    with get_conn() as conn:
        cur = conn.execute("""
            SELECT DISTINCT date(substr(date,1,10)) as workout_date
            FROM sets
            WHERE weight IS NOT NULL
            ORDER BY workout_date DESC
        """)
        
        workout_dates = [row[0] for row in cur.fetchall()]
        
        if not workout_dates:
            return {'current_streak': 0, 'longest_streak': 0, 'last_workout': None}
        
        # Calculate current streak
        current_streak = 0
        today = datetime.now().date()
        
        for date_str in workout_dates:
            workout_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            days_ago = (today - workout_date).days
            
            if days_ago <= 7:  # Within a week
                current_streak += 1
            else:
                break
        
        # Calculate longest streak (simplified - count consecutive weeks with workouts)
        from datetime import timedelta
        
        longest_streak = 1
        temp_streak = 1
        
        for i in range(1, len(workout_dates)):
            current_date = datetime.strptime(workout_dates[i-1], '%Y-%m-%d').date()
            next_date = datetime.strptime(workout_dates[i], '%Y-%m-%d').date()
            
            # If within 7 days, continue streak
            if (current_date - next_date).days <= 7:
                temp_streak += 1
                longest_streak = max(longest_streak, temp_streak)
            else:
                temp_streak = 1
        
        return {
            'current_streak': current_streak,
            'longest_streak': longest_streak,
            'last_workout': workout_dates[0],
            'total_workout_days': len(workout_dates)
        }


# Additional Strong-inspired analytics for maximum data visualization

def get_weekly_volume_heatmap():
    """Generate heatmap data for weekly training volume - like GitHub contribution graph"""
    with get_conn() as conn:
        cur = conn.execute("""
            SELECT 
                date(substr(date,1,10)) as workout_date,
                strftime('%w', date(substr(date,1,10))) as day_of_week,
                strftime('%W', date(substr(date,1,10))) as week_number,
                SUM(COALESCE(weight,0) * COALESCE(reps,0)) as daily_volume
            FROM sets
            WHERE weight IS NOT NULL AND reps > 0
            GROUP BY workout_date
            ORDER BY workout_date
        """)
        
        heatmap_data = []
        for row in cur.fetchall():
            heatmap_data.append({
                'date': row[0],
                'day_of_week': int(row[1]),  # 0=Sunday, 1=Monday, etc.
                'week_number': row[2],
                'volume': round(row[3], 1),
                'intensity': min(5, max(1, int(row[3] / 500)))  # Scale 1-5 for color intensity
            })
        
        return heatmap_data


def get_rep_range_distribution():
    """Analyze rep range distribution to see training focus"""
    with get_conn() as conn:
        cur = conn.execute("""
            SELECT 
                CASE 
                    WHEN reps BETWEEN 1 AND 3 THEN '1-3 (Strength)'
                    WHEN reps BETWEEN 4 AND 6 THEN '4-6 (Power)'
                    WHEN reps BETWEEN 7 AND 12 THEN '7-12 (Hypertrophy)'
                    WHEN reps BETWEEN 13 AND 20 THEN '13-20 (Endurance)'
                    ELSE '20+ (High Endurance)'
                END as rep_range,
                COUNT(*) as set_count,
                SUM(COALESCE(weight,0) * COALESCE(reps,0)) as total_volume
            FROM sets
            WHERE reps > 0 AND weight IS NOT NULL
            GROUP BY rep_range
            ORDER BY 
                CASE 
                    WHEN reps BETWEEN 1 AND 3 THEN 1
                    WHEN reps BETWEEN 4 AND 6 THEN 2
                    WHEN reps BETWEEN 7 AND 12 THEN 3
                    WHEN reps BETWEEN 13 AND 20 THEN 4
                    ELSE 5
                END
        """)
        
        return [
            {
                'rep_range': row[0],
                'set_count': row[1],
                'total_volume': round(row[2], 1),
                'percentage': 0  # Will calculate in frontend
            }
            for row in cur.fetchall()
        ]


def get_exercise_frequency():
    """Track how frequently each exercise is performed"""
    with get_conn() as conn:
        cur = conn.execute("""
            SELECT 
                exercise,
                COUNT(DISTINCT date(substr(date,1,10))) as workout_days,
                COUNT(*) as total_sets,
                AVG(COALESCE(weight,0)) as avg_weight,
                MAX(date(substr(date,1,10))) as last_performed,
                MIN(date(substr(date,1,10))) as first_performed
            FROM sets
            WHERE weight IS NOT NULL
            GROUP BY exercise
            ORDER BY workout_days DESC, total_sets DESC
        """)
        
        return [
            {
                'exercise': row[0],
                'workout_days': row[1],
                'total_sets': row[2],
                'avg_weight': round(row[3], 1),
                'last_performed': row[4],
                'first_performed': row[5],
                'frequency_score': row[1] * row[2]  # Days * Sets for ranking
            }
            for row in cur.fetchall()
        ]


def get_strength_ratios():
    """Calculate strength ratios between major lifts"""
    with get_conn() as conn:
        # Define major lift patterns
        lift_patterns = {
            'Squat': ['Squat', 'Leg Press'],
            'Bench': ['Chest Press', 'Bench Press'],
            'Deadlift': ['Deadlift'],
            'Row': ['Seated Row', 'Bent Over Row'],
            'Press': ['Shoulder Press', 'Overhead Press']
        }
        
        ratios = {}
        max_weights = {}
        
        for lift_name, patterns in lift_patterns.items():
            like_conditions = ' OR '.join([f"exercise LIKE '%{pattern}%'" for pattern in patterns])
            
            cur = conn.execute(f"""
                SELECT MAX(COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0)) as max_1rm
                FROM sets
                WHERE ({like_conditions}) AND weight IS NOT NULL
            """)
            
            result = cur.fetchone()
            if result and result[0]:
                max_weights[lift_name] = round(result[0], 1)
        
        # Calculate ratios (using typical strength standards)
        if 'Squat' in max_weights and 'Bench' in max_weights:
            ratios['Squat_to_Bench'] = round(max_weights['Squat'] / max_weights['Bench'], 2)
        
        if 'Deadlift' in max_weights and 'Squat' in max_weights:
            ratios['Deadlift_to_Squat'] = round(max_weights['Deadlift'] / max_weights['Squat'], 2)
        
        if 'Bench' in max_weights and 'Row' in max_weights:
            ratios['Bench_to_Row'] = round(max_weights['Bench'] / max_weights['Row'], 2)
        
        return {
            'max_lifts': max_weights,
            'ratios': ratios,
            'ideal_ratios': {
                'Squat_to_Bench': 1.3,  # Squat should be ~30% higher than bench
                'Deadlift_to_Squat': 1.2,  # Deadlift should be ~20% higher than squat
                'Bench_to_Row': 1.0  # Bench and row should be roughly equal
            }
        }


def get_recovery_tracking():
    """Track recovery time between sessions for each muscle group"""
    with get_conn() as conn:
        # Group exercises by muscle groups
        muscle_groups = {
            'Chest': ['Chest Press', 'Bench Press', 'Pec Deck'],
            'Back': ['Seated Row', 'Lat Pulldown', 'Bent Over Row', 'Pull'],
            'Shoulders': ['Shoulder Press', 'Military Press', 'Lateral Raise'],
            'Legs': ['Squat', 'Leg Press', 'Leg Extension', 'Leg Curl'],
            'Arms': ['Bicep Curl', 'Tricep', 'Hammer Curl']
        }
        
        recovery_data = {}
        
        for muscle_group, exercises in muscle_groups.items():
            like_conditions = ' OR '.join([f"exercise LIKE '%{ex}%'" for ex in exercises])
            
            cur = conn.execute(f"""
                SELECT 
                    date(substr(date,1,10)) as workout_date,
                    LAG(date(substr(date,1,10))) OVER (ORDER BY date) as prev_date
                FROM sets
                WHERE ({like_conditions}) AND weight IS NOT NULL
                GROUP BY workout_date
                ORDER BY workout_date
            """)
            
            recovery_times = []
            for row in cur.fetchall():
                if row[1]:  # If there's a previous date
                    current = datetime.strptime(row[0], '%Y-%m-%d')
                    previous = datetime.strptime(row[1], '%Y-%m-%d')
                    days_between = (current - previous).days
                    recovery_times.append(days_between)
            
            if recovery_times:
                recovery_data[muscle_group] = {
                    'avg_recovery': round(sum(recovery_times) / len(recovery_times), 1),
                    'min_recovery': min(recovery_times),
                    'max_recovery': max(recovery_times),
                    'total_sessions': len(recovery_times) + 1
                }
        
        return recovery_data


def get_progressive_overload_rate():
    """Calculate the rate of strength progression per exercise"""
    with get_conn() as conn:
        cur = conn.execute("""
            SELECT DISTINCT exercise FROM sets WHERE weight IS NOT NULL
            ORDER BY exercise
        """)
        
        exercises = [row[0] for row in cur.fetchall()]
        progression_rates = []
        
        for exercise in exercises:
            cur = conn.execute("""
                SELECT 
                    date(substr(date,1,10)) as workout_date,
                    MAX(COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0)) as estimated_1rm
                FROM sets
                WHERE exercise = ? AND weight IS NOT NULL
                GROUP BY workout_date
                ORDER BY workout_date
            """, (exercise,))
            
            sessions = cur.fetchall()
            
            if len(sessions) >= 2:
                first_1rm = sessions[0][1]
                last_1rm = sessions[-1][1]
                first_date = datetime.strptime(sessions[0][0], '%Y-%m-%d')
                last_date = datetime.strptime(sessions[-1][0], '%Y-%m-%d')
                
                days_elapsed = (last_date - first_date).days
                if days_elapsed > 0 and first_1rm > 0:
                    # Calculate weekly progression rate
                    total_gain = last_1rm - first_1rm
                    weekly_rate = (total_gain / days_elapsed) * 7
                    percentage_gain = (total_gain / first_1rm) * 100
                    
                    progression_rates.append({
                        'exercise': exercise,
                        'first_1rm': round(first_1rm, 1),
                        'last_1rm': round(last_1rm, 1),
                        'total_gain': round(total_gain, 1),
                        'weekly_rate': round(weekly_rate, 2),
                        'percentage_gain': round(percentage_gain, 1),
                        'days_tracked': days_elapsed,
                        'sessions': len(sessions)
                    })
        
        # Sort by percentage gain
        return sorted(progression_rates, key=lambda x: x['percentage_gain'], reverse=True)


def get_workout_duration_trends():
    """Analyze workout duration patterns over time"""
    with get_conn() as conn:
        cur = conn.execute("""
            SELECT 
                date(substr(date,1,10)) as workout_date,
                AVG(duration_min) as avg_duration,
                COUNT(DISTINCT exercise) as exercises_performed,
                COUNT(*) as total_sets,
                SUM(COALESCE(weight,0) * COALESCE(reps,0)) as total_volume
            FROM sets
            WHERE duration_min IS NOT NULL
            GROUP BY workout_date
            ORDER BY workout_date
        """)
        
        duration_data = []
        for row in cur.fetchall():
            # Calculate efficiency metrics
            volume_per_minute = row[4] / row[1] if row[1] > 0 else 0
            sets_per_minute = row[3] / row[1] if row[1] > 0 else 0
            
            duration_data.append({
                'date': row[0],
                'duration': round(row[1], 1),
                'exercises': row[2],
                'total_sets': row[3],
                'volume': round(row[4], 1),
                'volume_per_minute': round(volume_per_minute, 1),
                'sets_per_minute': round(sets_per_minute, 2),
                'efficiency_score': round(volume_per_minute / 10, 1)  # Arbitrary scale
            })
        
        return duration_data


def get_best_sets_analysis():
    """Find the single best set for each exercise"""
    with get_conn() as conn:
        cur = conn.execute("""
            WITH ranked_sets AS (
                SELECT 
                    exercise,
                    date(substr(date,1,10)) as workout_date,
                    weight,
                    reps,
                    COALESCE(weight,0) * COALESCE(reps,0) as volume,
                    COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0) as estimated_1rm,
                    ROW_NUMBER() OVER (PARTITION BY exercise ORDER BY COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0) DESC) as rn
                FROM sets
                WHERE weight IS NOT NULL AND reps > 0
            )
            SELECT 
                exercise,
                workout_date,
                weight,
                reps,
                volume,
                estimated_1rm
            FROM ranked_sets
            WHERE rn = 1
            ORDER BY estimated_1rm DESC
        """)
        
        return [
            {
                'exercise': row[0],
                'date': row[1],
                'weight': row[2],
                'reps': row[3],
                'volume': row[4],
                'estimated_1rm': round(row[5], 1)
            }
            for row in cur.fetchall()
        ]


def get_plateau_detection():
    """Detect potential plateaus in strength progression"""
    with get_conn() as conn:
        cur = conn.execute("""
            SELECT DISTINCT exercise FROM sets 
            WHERE weight IS NOT NULL 
            ORDER BY exercise
        """)
        
        exercises = [row[0] for row in cur.fetchall()]
        plateau_analysis = []
        
        for exercise in exercises:
            cur = conn.execute("""
                SELECT 
                    date(substr(date,1,10)) as workout_date,
                    MAX(COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0)) as estimated_1rm
                FROM sets
                WHERE exercise = ? AND weight IS NOT NULL
                GROUP BY workout_date
                ORDER BY workout_date DESC
                LIMIT 5
            """, (exercise,))
            
            recent_sessions = cur.fetchall()
            
            if len(recent_sessions) >= 3:
                # Check if last 3 sessions show no improvement
                last_3_1rms = [session[1] for session in recent_sessions[:3]]
                
                if len(set(last_3_1rms)) == 1:  # All the same
                    status = 'Plateau'
                elif max(last_3_1rms) == last_3_1rms[0]:  # Best is most recent
                    status = 'Progressing'
                elif last_3_1rms[0] < max(last_3_1rms):  # Recent performance declined
                    status = 'Declining'
                else:
                    status = 'Variable'
                
                plateau_analysis.append({
                    'exercise': exercise,
                    'status': status,
                    'current_1rm': round(last_3_1rms[0], 1),
                    'best_recent_1rm': round(max(last_3_1rms), 1),
                    'sessions_analyzed': len(recent_sessions),
                    'last_session': recent_sessions[0][0] if recent_sessions else None
                })
        
        return sorted(plateau_analysis, key=lambda x: x['current_1rm'], reverse=True)
