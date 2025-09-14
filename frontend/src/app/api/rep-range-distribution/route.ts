export const runtime = "nodejs";

import { NextResponse } from "next/server";
import { DatabaseSync } from "node:sqlite";
import path from "path";

export interface RepRangeDistributionData {
  exercise: string;
  date: string;
  range_1_5: number;
  range_6_12: number;
  range_13_20: number;
  range_20_plus: number;
}

export async function GET() {
  try {
    const dbPath = process.env.DB_FILE || "/data/strong.db"
    const db = new DatabaseSync(dbPath);

    const query = `
      SELECT 
        exercise,
        date,
        SUM(CASE WHEN 
          CASE 
            WHEN exercise = 'Kettlebell Carry' AND distance IS NOT NULL THEN distance
            ELSE reps
          END >= 1 AND 
          CASE 
            WHEN exercise = 'Kettlebell Carry' AND distance IS NOT NULL THEN distance
            ELSE reps
          END <= 5 THEN 1 ELSE 0 END) as range_1_5,
        SUM(CASE WHEN 
          CASE 
            WHEN exercise = 'Kettlebell Carry' AND distance IS NOT NULL THEN distance
            ELSE reps
          END >= 6 AND 
          CASE 
            WHEN exercise = 'Kettlebell Carry' AND distance IS NOT NULL THEN distance
            ELSE reps
          END <= 12 THEN 1 ELSE 0 END) as range_6_12,
        SUM(CASE WHEN 
          CASE 
            WHEN exercise = 'Kettlebell Carry' AND distance IS NOT NULL THEN distance
            ELSE reps
          END >= 13 AND 
          CASE 
            WHEN exercise = 'Kettlebell Carry' AND distance IS NOT NULL THEN distance
            ELSE reps
          END <= 20 THEN 1 ELSE 0 END) as range_13_20,
        SUM(CASE WHEN 
          CASE 
            WHEN exercise = 'Kettlebell Carry' AND distance IS NOT NULL THEN distance
            ELSE reps
          END > 20 THEN 1 ELSE 0 END) as range_20_plus
      FROM sets 
      WHERE weight IS NOT NULL AND 
        (reps IS NOT NULL OR (exercise = 'Kettlebell Carry' AND distance IS NOT NULL))
      GROUP BY exercise, date 
      ORDER BY exercise, date
    `;

    const rows = db.prepare(query).all() as unknown as RepRangeDistributionData[];
    db.close();

    return NextResponse.json(rows);
  } catch (error) {
    console.error("Database error:", error);
    return NextResponse.json(
      { error: "Failed to fetch rep range distribution data" },
      { status: 500 }
    );
  }
}