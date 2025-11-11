export const runtime = "nodejs";

import { NextResponse } from "next/server";
import { DatabaseSync } from "node:sqlite";

export interface MaxWeightSparklineData {
  exercise: string;
  date: string;
  maxWeight: number;
  reps: number | null;
  distance: number | null;
}

export async function GET() {
  try {
    const dbPath = process.env.DB_FILE || "/data/strong.db"
    const db = new DatabaseSync(dbPath);

    // We keep reps as the actual reps count even for distance-based exercises like carries,
    // and expose distance separately so the UI can decide which to display.
    const query = `
      SELECT 
        exercise, 
        date, 
        CASE 
          WHEN exercise = 'Pull Up (Assisted)' THEN MIN(weight)
          ELSE MAX(weight)
        END as maxWeight,
        MAX(reps) as reps,
        MAX(distance) as distance
      FROM sets 
      WHERE weight IS NOT NULL AND 
        (reps IS NOT NULL OR distance IS NOT NULL)
      GROUP BY exercise, date 
      ORDER BY exercise, date
    `;

  const rows = db.prepare(query).all() as unknown as MaxWeightSparklineData[];
    db.close();

    return NextResponse.json(rows);
  } catch (error) {
    console.error("Database error:", error);
    return NextResponse.json(
      { error: "Failed to fetch max weight data" },
      { status: 500 }
    );
  }
}