export const runtime = "nodejs"; // ensure not running on Edge

import { NextResponse } from "next/server";
import { DatabaseSync } from "node:sqlite";
import path from "path";

export interface VolumeSparklineData {
  exercise: string;
  date: string;
  volume: number;
  sets: number;
}

export async function GET() {
  try {
    const dbPath = "/data/strong.db"
    const db = new DatabaseSync(dbPath);

    const query = `
      SELECT 
        exercise, 
        date, 
        SUM(COALESCE(weight, 0) * COALESCE(reps, 0)) as volume,
        COUNT(*) as sets 
      FROM sets 
      WHERE weight IS NOT NULL AND reps IS NOT NULL 
      GROUP BY exercise, date 
      ORDER BY exercise, date
    `;

    const rows = db.prepare(query).all() as unknown as VolumeSparklineData[];
    db.close();

    return NextResponse.json(rows);
  } catch (error) {
    console.error("Database error:", error);
    return NextResponse.json(
      { error: "Failed to fetch volume data" },
      { status: 500 }
    );
  }
}
