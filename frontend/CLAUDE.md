Feature catalog (what the UI does)
Global structure and UX
- Header status
  - Last ingested timestamp or latest workout date.
  - Quick KPIs: total sessions in range, total volume, average session duration, current training streak.
- Filters panel
  - Date range selector (absolute or quick ranges).
  - Exercise selector (multi-select, search).
  - Optional toggles: unit (kg/lb), aggregation granularity (day/week/month), primary metric (e.g., volume vs. e1RM).
- Responsive dashboard grid
  - A 12-column responsive layout that arranges “widgets” (charts or lists) into rows.
  - Each widget has a title, an optional description/legend, and a content area.
- Interactions common to all charts
  - Hover tooltips show detailed values with contextual units.
  - Click-to-focus an exercise: clicking a legend/bar/line applies a temporary exercise focus across relevant widgets.
  - Date zoom/brush on time-series charts updates the global date filter.
  - Hide/show series from legends without changing the global filter.
  - Empty states for no data in the selected range.

Dashboard widgets
- Volume sparklines (per-exercise small multiples)
  - A matrix of compact line charts showing daily (or weekly) training volume over time for each selected exercise.
  - Each sparkline highlights the latest value and shows a small delta vs. previous period.
  - Hover reveals date, volume, and count of sets.
- Volume trend (global)
  - Time-series showing total volume per day or per week across all selected exercises.
  - Optional overlay or secondary axis for session duration or set count.
  - A rolling average line (e.g., 7-day) to visualize trend.
- Exercise volume (ranking)
  - Horizontal bar chart ranking exercises by total volume in the filtered period.
  - Bars display absolute volume; optional labels for percentage contribution.
- Rep distribution
  - Histogram showing the distribution of reps per set within the filtered range.
  - Supports binning (e.g., 1–5, 6–10, 11–15) and a density/percentage toggle.
- Best sets (leaderboard)
  - A top-N leaderboard by estimated 1RM (or weight for 1-rep sets).
  - Each entry shows exercise, estimated 1RM, actual weight, reps, and date; optionally a bar visualization.
- Exercise progression
  - Per-exercise lines showing performance over time using best set of the day (estimated 1RM or top weight).
  - Highlights lifetime PR and current period PR; shows delta vs. previous period.
  - Option to display small multiples or overlay multiple exercises in one chart.
- Calendar heatmap
  - A year-style calendar view with color intensity proportional to volume or duration.
  - Hover shows date, sessions count, and volume; clicking narrows the date range to that day/week.
- Weekly split balance (P/P/L or custom categories)
  - Stacked weekly bars showing distribution of work by training category (Push/Pull/Legs or similar).
  - Each week aggregates either session counts or volume per category.
- Muscle group balance
  - Donut or bar chart showing contribution by muscle group (primary and/or secondary), based on an exercise→muscle mapping.
- Recent workouts
  - A chronological list of recent sessions with: date, workout name/title, number of sets, total volume, and duration.
  - Expand a session to see its exercises and top sets for that day.
- Training frequency by weekday
  - Bar chart of sessions or volume by day of week to reveal routine biases.
- Session duration trend
  - Time-series of session duration; optionally correlates duration with volume via color/size encoding.
- Personal records and milestones
  - Table or badges for lifetime PRs per exercise (max weight and max estimated 1RM) and the date achieved.
  - Milestones such as “first time squatting 100 kg” with date.

Global behaviors and flows
- Default state
  - On first open, shows a meaningful date window (e.g., last 90 days) and a sensible default selection of exercises (e.g., most trained).
- Synchronized filters
  - Changing date range or exercises refreshes all widgets for consistency.
- Quick comparisons
  - A “compare to previous period” toggle surfaces % change for key widgets (volume trend, sparklines, exercise volume).
- Export
  - Buttons to export a chart as an image and to copy the underlying data slice as CSV/JSON.
- Accessibility and theming
  - Dark and light themes.
  - Keyboard navigation for legends and focusable elements.
  - Color palettes with adequate contrast and colorblind-safe defaults.

Data slices each widget expects (frontend-facing, backend-agnostic)
- Volume sparklines: per exercise time-series
  - [{ exercise, date, volume, sets }]
- Volume trend: global time-series
  - [{ date, volume, sets, duration_minutes }]
- Exercise volume: ranking
  - [{ exercise, volume, sets, sessions }]
- Rep distribution: histogram-ready rows
  - [{ reps, count }] or raw set rows with reps to bin client-side
- Best sets: leaderboard
  - [{ exercise, estimated_1rm, weight, reps, date }]
- Exercise progression: per-exercise “best of day”
  - [{ exercise, date, e1rm, weight, reps }]
- Calendar heatmap: per-day summary
  - [{ date, sessions, volume, duration_minutes }]
- Weekly split balance: per-week category distribution
  - [{ week, category, sessions, volume }]
- Muscle group balance: totals by muscle group
  - [{ muscle_group, volume, sessions }]
- Recent workouts: session list
  - [{ date, workout_name, total_sets, volume, duration_minutes }]

DB file format (SQLite) for a frontend-centric, read-only container
Goals
- Provide a single file the frontend can query directly.
- Keep raw set data plus enough metadata to derive all widgets.
- Include convenient views for common aggregations.
- Be unit-agnostic but store canonical kg; allow UI conversion to lb.

Core tables
- meta
  - key TEXT PRIMARY KEY
  - value TEXT NOT NULL
  - Notes: includes last_ingested_at (ISO 8601), app_version, timezone if needed.
- sessions
  - id INTEGER PRIMARY KEY AUTOINCREMENT
  - date TEXT NOT NULL                         — “YYYY-MM-DD”
  - start_ts TEXT                              — ISO 8601 if available
  - end_ts TEXT                                — ISO 8601 if available
  - workout_name TEXT NOT NULL
  - duration_minutes INTEGER                   — nullable
  - notes TEXT                                 — nullable
  - UNIQUE(date, workout_name, start_ts)
  - Indexes: idx_sessions_date(date)
- exercises
  - id INTEGER PRIMARY KEY AUTOINCREMENT
  - name TEXT NOT NULL                         — canonical exercise name
  - category TEXT                              — e.g., “Push”, “Pull”, “Legs”, “Other”
  - is_bodyweight INTEGER DEFAULT 0            — 0/1
  - units TEXT DEFAULT 'kg'                    — base unit; keep ‘kg’ canonical
  - UNIQUE(name)
- exercise_aliases
  - alias TEXT PRIMARY KEY
  - exercise_id INTEGER NOT NULL               — FK → exercises(id)
- exercise_muscles
  - exercise_id INTEGER NOT NULL               — FK → exercises(id)
  - muscle_group TEXT NOT NULL                 — e.g., “Quads”, “Chest”
  - role TEXT NOT NULL                         — “primary” or “secondary”
  - PRIMARY KEY (exercise_id, muscle_group, role)
- sets
  - id INTEGER PRIMARY KEY AUTOINCREMENT
  - session_id INTEGER NOT NULL                — FK → sessions(id)
  - exercise_id INTEGER NOT NULL               — FK → exercises(id)
  - set_order INTEGER NOT NULL                 — 1-based within exercise
  - weight_kg REAL                             — nullable (e.g., bodyweight only)
  - reps INTEGER                               — nullable (e.g., timed sets)
  - rpe REAL                                   — nullable
  - rir REAL                                   — nullable
  - seconds REAL                               — nullable
  - distance_m REAL                            — nullable
  - notes TEXT                                 — nullable
  - volume REAL GENERATED ALWAYS AS (COALESCE(weight_kg,0)*COALESCE(reps,0)) STORED
  - Indexes: idx_sets_session(session_id), idx_sets_exercise(exercise_id), idx_sets_volume(volume)

Computed views (for convenient reads)
- v_sets_enriched
  - set_id, date, workout_name, duration_minutes, exercise, category, muscle_group_primary, weight_kg, reps, volume, rpe, seconds
- v_sessions
  - date, total_sets, total_volume, total_duration_minutes
- v_best_of_day
  - date, exercise, weight_kg, reps, e1rm (estimated_1rm = weight_kg*(1 + reps/30.0))
- v_prs
  - exercise, max_weight, max_e1rm, date_of_max_e1rm
- v_exercise_volume
  - exercise, volume, sets, sessions
- v_weekly_category
  - week (ISO week like “2025-W37”), category, sessions, volume
- v_muscle_balance
  - muscle_group, volume, sessions
- v_rep_distribution
  - reps, count

Data conventions and semantics
- Dates and times
  - date is always “YYYY-MM-DD” and represents local training date.
  - start_ts/end_ts are ISO strings if available; otherwise null.
- Units
  - weight_kg is canonical; convert to lb in UI when needed (1 kg = 2.2046226218 lb).
- Estimated 1RM
  - e1rm = weight_kg * (1 + reps/30.0) for reps >= 1; ignore rows with null weight or reps.
- Categories and muscles
  - category is free-form but intended for P/P/L split; fallback to “Other”.
  - exercise→muscle mapping supports primary/secondary for muscle balance.
- Uniqueness/deduplication
  - sessions uniqueness uses date/workout_name/start_ts; sets are ordered per exercise inside a session.
- Null handling
  - Timed or distance-based sets may lack reps/weight; their “volume” is 0 by definition for volume charts.

Example schema file (DDL)
```sql
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  date              TEXT NOT NULL,      -- "YYYY-MM-DD"
  start_ts          TEXT,               -- ISO 8601
  end_ts            TEXT,               -- ISO 8601
  workout_name      TEXT NOT NULL,
  duration_minutes  INTEGER,
  notes             TEXT,
  UNIQUE(date, workout_name, start_ts)
);
CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions(date);

CREATE TABLE IF NOT EXISTS exercises (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  name          TEXT NOT NULL,
  category      TEXT,
  is_bodyweight INTEGER DEFAULT 0,
  units         TEXT DEFAULT 'kg',
  UNIQUE(name)
);

CREATE TABLE IF NOT EXISTS exercise_aliases (
  alias       TEXT PRIMARY KEY,
  exercise_id INTEGER NOT NULL,
  FOREIGN KEY (exercise_id) REFERENCES exercises(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS exercise_muscles (
  exercise_id  INTEGER NOT NULL,
  muscle_group TEXT NOT NULL,
  role         TEXT NOT NULL CHECK (role IN ('primary','secondary')),
  PRIMARY KEY (exercise_id, muscle_group, role),
  FOREIGN KEY (exercise_id) REFERENCES exercises(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sets (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id  INTEGER NOT NULL,
  exercise_id INTEGER NOT NULL,
  set_order   INTEGER NOT NULL,
  weight_kg   REAL,
  reps        INTEGER,
  rpe         REAL,
  rir         REAL,
  seconds     REAL,
  distance_m  REAL,
  notes       TEXT,
  volume      REAL GENERATED ALWAYS AS (COALESCE(weight_kg,0) * COALESCE(reps,0)) STORED,
  FOREIGN KEY (session_id)  REFERENCES sessions(id)  ON DELETE CASCADE,
  FOREIGN KEY (exercise_id) REFERENCES exercises(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_sets_session  ON sets(session_id);
CREATE INDEX IF NOT EXISTS idx_sets_exercise ON sets(exercise_id);
CREATE INDEX IF NOT EXISTS idx_sets_volume   ON sets(volume);

-- Convenience views
CREATE VIEW IF NOT EXISTS v_sets_enriched AS
SELECT
  s.id AS set_id,
  ses.date,
  ses.workout_name,
  ses.duration_minutes,
  e.name AS exercise,
  e.category,
  (SELECT em.muscle_group FROM exercise_muscles em WHERE em.exercise_id=e.id AND em.role='primary' LIMIT 1) AS muscle_group_primary,
  s.weight_kg,
  s.reps,
  s.volume,
  s.rpe,
  s.seconds
FROM sets s
JOIN sessions ses ON ses.id=s.session_id
JOIN exercises e ON e.id=s.exercise_id;

CREATE VIEW IF NOT EXISTS v_sessions AS
SELECT
  date,
  COUNT(*) AS total_sets,
  SUM(volume) AS total_volume,
  SUM(duration_minutes) AS total_duration_minutes
FROM (
  SELECT ses.date, s.volume, ses.duration_minutes
  FROM sessions ses
  LEFT JOIN sets s ON s.session_id=ses.id
)
GROUP BY date
ORDER BY date;

CREATE VIEW IF NOT EXISTS v_best_of_day AS
WITH scored AS (
  SELECT
    ses.date,
    e.name AS exercise,
    s.weight_kg,
    s.reps,
    (s.weight_kg*(1 + s.reps/30.0)) AS e1rm,
    ROW_NUMBER() OVER (PARTITION BY ses.date, e.id ORDER BY (s.weight_kg*(1 + s.reps/30.0)) DESC) AS rn
  FROM sets s
  JOIN sessions ses ON ses.id=s.session_id
  JOIN exercises e ON e.id=s.exercise_id
  WHERE s.weight_kg IS NOT NULL AND s.reps IS NOT NULL
)
SELECT date, exercise, weight_kg, reps, e1rm
FROM scored WHERE rn=1;

CREATE VIEW IF NOT EXISTS v_prs AS
WITH scored AS (
  SELECT e.name AS exercise, ses.date,
         s.weight_kg, s.reps,
         (s.weight_kg*(1 + s.reps/30.0)) AS e1rm
  FROM sets s
  JOIN sessions ses ON ses.id=s.session_id
  JOIN exercises e ON e.id=s.exercise_id
  WHERE s.weight_kg IS NOT NULL AND s.reps IS NOT NULL
)
SELECT exercise,
       MAX(weight_kg) AS max_weight,
       MAX(e1rm)      AS max_e1rm,
       (SELECT date FROM scored s2 WHERE s2.exercise=scored.exercise ORDER BY e1rm DESC, date ASC LIMIT 1) AS date_of_max_e1rm
FROM scored
GROUP BY exercise;

CREATE VIEW IF NOT EXISTS v_exercise_volume AS
SELECT
  e.name AS exercise,
  SUM(s.volume) AS volume,
  COUNT(s.id) AS sets,
  COUNT(DISTINCT ses.id) AS sessions
FROM sets s
JOIN sessions ses ON ses.id=s.session_id
JOIN exercises e ON e.id=s.exercise_id
GROUP BY e.id;

CREATE VIEW IF NOT EXISTS v_weekly_category AS
WITH rows AS (
  SELECT
    strftime('%Y-W%W', ses.date) AS week,
    COALESCE(e.category,'Other') AS category,
    ses.id AS session_id,
    s.id AS set_id,
    s.volume AS volume
  FROM sets s
  JOIN sessions ses ON ses.id=s.session_id
  JOIN exercises e ON e.id=s.exercise_id
)
SELECT week, category,
       COUNT(DISTINCT session_id) AS sessions,
       SUM(volume) AS volume
FROM rows
GROUP BY week, category
ORDER BY week;

CREATE VIEW IF NOT EXISTS v_muscle_balance AS
SELECT
  em.muscle_group,
  COUNT(s.id) AS sets,
  SUM(s.volume) AS volume
FROM sets s
JOIN exercises e ON e.id=s.exercise_id
LEFT JOIN exercise_muscles em ON em.exercise_id=e.id AND em.role='primary'
GROUP BY em.muscle_group;

CREATE VIEW IF NOT EXISTS v_rep_distribution AS
SELECT reps, COUNT(*) AS count
FROM sets
WHERE reps IS NOT NULL
GROUP BY reps
ORDER BY reps;
```

How to use this spec
- Build the new frontend against the “Data slices each widget expects” section.
- Point your containerized app at a SQLite file with the schema above (or adapt to your existing ingestion output if it already matches).
- If your ingestion already produces a different schema, share its .schema and I will map it to these slices 1:1.