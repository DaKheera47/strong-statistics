export const runtime = "nodejs";

import { NextResponse } from "next/server";
import { DatabaseSync } from "node:sqlite";
import path from "path";

export interface SessionVolumeTrendData {
  date: string;
  volume: number;
  sets: number;
  duration_minutes: number | null;
}

export async function GET() {
  try {
    const dbPath = "/data/strong.db"
    const db = new DatabaseSync(dbPath);

    const query = `
      WITH meaningful AS (
        SELECT * FROM sets
        WHERE (
          (weight IS NOT NULL AND reps IS NOT NULL AND weight > 0 AND reps > 0)
          OR (distance IS NOT NULL AND distance > 0)
          OR (seconds IS NOT NULL AND seconds > 0)
        )
      )
      SELECT 
        date,
        SUM(CASE 
              WHEN weight IS NOT NULL AND reps IS NOT NULL AND weight > 0 AND reps > 0 
              THEN weight * reps 
              ELSE 0 END) as volume,
        COUNT(*) as sets,
        AVG(duration_min) as duration_minutes
      FROM meaningful
      GROUP BY date 
      ORDER BY date
    `;

    const rows = db.prepare(query).all() as unknown as SessionVolumeTrendData[];
    db.close();

    return NextResponse.json(rows);
  } catch (error) {
    console.error("Database error:", error);
    return NextResponse.json(
      { error: "Failed to fetch session volume trend data" },
      { status: 500 }
    );
  }
}
