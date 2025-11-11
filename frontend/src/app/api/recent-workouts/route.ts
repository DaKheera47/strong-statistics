export const runtime = "nodejs";

import { NextResponse } from "next/server";
import { DatabaseSync } from "node:sqlite";

export interface RecentWorkoutData {
  date: string;
  workout_name: string;
  total_sets: number;
  total_volume: number;
  duration_minutes: number | null;
  exercises_count: number;
  prs: number;
}

export interface WorkoutDetailData {
  date: string;
  workout_name: string;
  duration_minutes: number | null;
  exercises: {
    exercise: string;
    sets: {
      set_order: number;
      weight: number | null;
      reps: number | null;
      distance: number | null;
      seconds: number | null;
      estimated_1rm: number | null;
    }[];
  }[];
}

// Row shape returned for detailed workout query
interface RawSetRow {
  date: string;
  workout_name: string;
  duration_min: number | null;
  exercise: string;
  set_order: number;
  weight: number | null;
  reps: number | null;
  distance: number | null;
  seconds: number | null;
  id: number;
  estimated_1rm: number | null;
}

// Internal structure for grouping sets per exercise
interface SetEntry {
  set_order: number;
  weight: number | null;
  reps: number | null;
  distance: number | null;
  seconds: number | null;
  estimated_1rm: number | null;
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const workoutDate = searchParams.get("date");
    const workoutName = searchParams.get("workout_name");

    const dbPath = process.env.DB_FILE || "/data/strong.db"
    const db = new DatabaseSync(dbPath);

    if (workoutDate && workoutName) {
      // Fetch detailed workout data
      const query = `
        SELECT 
          date,
          workout_name,
          duration_min,
          exercise,
          set_order,
          weight,
          reps,
          distance,
          seconds,
          id,
          CASE 
            WHEN weight IS NOT NULL AND reps IS NOT NULL AND reps > 0 
            THEN weight * (1 + reps / 30.0)
            ELSE NULL
          END as estimated_1rm
        FROM sets
        WHERE date = ? AND workout_name = ?
        ORDER BY id
      `;

      const rows = db
        .prepare(query)
        .all(workoutDate, workoutName) as unknown as RawSetRow[];

      if (rows.length === 0) {
        db.close();
        return NextResponse.json(
          { error: "Workout not found" },
          { status: 404 }
        );
      }

      const workoutDetail: WorkoutDetailData = {
        date: rows[0].date,
        workout_name: rows[0].workout_name,
        duration_minutes: rows[0].duration_min,
        exercises: [],
      };

      // Group by exercise while maintaining order
      const exerciseMap = new Map<string, SetEntry[]>();
      const exerciseOrder: string[] = [];

      rows.forEach((row: RawSetRow) => {
        if (!exerciseMap.has(row.exercise)) {
          exerciseMap.set(row.exercise, []);
          exerciseOrder.push(row.exercise);
        }
        exerciseMap.get(row.exercise)!.push({
          set_order: row.set_order,
          weight: row.weight,
          reps: row.reps,
          distance: row.distance,
          seconds: row.seconds,
          estimated_1rm: row.estimated_1rm,
        });
      });

      // Add exercises in the order they were performed
      exerciseOrder.forEach((exercise) => {
        const sets = exerciseMap.get(exercise)!;
        // Sort sets by set_order within each exercise
        sets.sort((a, b) => a.set_order - b.set_order);
        workoutDetail.exercises.push({
          exercise,
          sets,
        });
      });

      db.close();
      return NextResponse.json(workoutDetail);
    } else {
      // Fetch recent workouts list with PR calculations
      const query = `
        WITH meaningful AS (
          SELECT * FROM sets
          WHERE (
            (weight IS NOT NULL AND reps IS NOT NULL AND weight > 0 AND reps > 0)
            OR (distance IS NOT NULL AND distance > 0)
            OR (seconds IS NOT NULL AND seconds > 0)
          )
        ),
        workout_summary AS (
          SELECT 
            date,
            workout_name,
            COUNT(*) as total_sets,
            /* volume only from meaningful sets with weight & reps */
            SUM(CASE 
                  WHEN weight IS NOT NULL AND reps IS NOT NULL AND weight > 0 AND reps > 0 
                  THEN weight * reps 
                  ELSE 0 
                END) as total_volume,
            CASE 
              WHEN COUNT(duration_min) > 0 THEN AVG(duration_min)
              ELSE NULL
            END as duration_minutes,
            COUNT(DISTINCT exercise) as exercises_count
          FROM meaningful
          GROUP BY date, workout_name
        ),
        exercise_prs AS (
          SELECT DISTINCT
            s1.date,
            s1.workout_name,
            s1.exercise,
            MAX(s1.weight * (1 + s1.reps / 30.0)) as workout_e1rm
          FROM meaningful s1
          WHERE s1.weight IS NOT NULL AND s1.reps IS NOT NULL
          GROUP BY s1.date, s1.workout_name, s1.exercise
        ),
        historical_prs AS (
          SELECT 
            pr.date,
            pr.workout_name,
            pr.exercise,
            pr.workout_e1rm,
            CASE
              WHEN pr.workout_e1rm > COALESCE((
                SELECT MAX(h.weight * (1 + h.reps / 30.0))
                FROM meaningful h 
                WHERE h.exercise = pr.exercise 
                AND h.date < pr.date
                AND h.weight IS NOT NULL AND h.reps IS NOT NULL
              ), 0) THEN 1
              ELSE 0
            END as is_pr
          FROM exercise_prs pr
        )
        SELECT 
          ws.*,
          COALESCE(pr_count.prs, 0) as prs
        FROM workout_summary ws
        LEFT JOIN (
          SELECT date, workout_name, SUM(is_pr) as prs
          FROM historical_prs
          GROUP BY date, workout_name
        ) pr_count ON ws.date = pr_count.date AND ws.workout_name = pr_count.workout_name
        ORDER BY ws.date DESC, ws.workout_name
        LIMIT 50
      `;

      const rows = db.prepare(query).all() as unknown as RecentWorkoutData[];
      db.close();

      return NextResponse.json(rows);
    }
  } catch (error) {
    console.error("Database error:", error);
    return NextResponse.json(
      { error: "Failed to fetch recent workouts data" },
      { status: 500 }
    );
  }
}
