export const runtime = "nodejs";

import { NextResponse } from "next/server";
import { DatabaseSync } from "node:sqlite";

export interface Exercise {
  name: string;
}

export async function GET() {
  try {
    const dbPath = process.env.DB_FILE || "/data/strong.db";
    const db = new DatabaseSync(dbPath);

    const query = `
      SELECT DISTINCT exercise as name
      FROM sets 
      WHERE weight IS NOT NULL AND reps IS NOT NULL 
      ORDER BY exercise
    `;

    const rows = db.prepare(query).all() as unknown as Exercise[];
    db.close();

    return NextResponse.json(rows);
  } catch (error) {
    console.error("Database error:", error);
    return NextResponse.json(
      { error: "Failed to fetch exercises" },
      { status: 500 }
    );
  }
}
