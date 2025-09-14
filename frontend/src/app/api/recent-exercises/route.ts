export const runtime = "nodejs";

import { NextResponse } from "next/server";
import { DatabaseSync } from "node:sqlite";
import path from "path";

export interface ExerciseWithLastActivity {
  name: string;
  lastActivityDate: string;
}

export async function GET() {
  try {
    const dbPath = process.env.DB_FILE || "/data/strong.db"
    const db = new DatabaseSync(dbPath);

    // Get exercises with their most recent activity date
    const query = `
      SELECT 
        exercise as name,
        MAX(date) as lastActivityDate
      FROM sets 
      WHERE weight IS NOT NULL AND reps IS NOT NULL 
      GROUP BY exercise
      ORDER BY lastActivityDate DESC
    `;

    const rows = db.prepare(query).all() as unknown as ExerciseWithLastActivity[];
    db.close();

    return NextResponse.json(rows);
  } catch (error) {
    console.error("Database error:", error);
    return NextResponse.json(
      { error: "Failed to fetch recent exercises" },
      { status: 500 }
    );
  }
}