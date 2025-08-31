Project Brief: “Lifting Pipeline” — One-Tap CSV → Self-Hosted Dashboard
0) Executive Summary (Context)

Problem: I log workouts in the Strong app on iOS. Strong’s charts are proprietary. I want my own charts, fully self-hosted.

Reality: iOS “Export Data” opens the share sheet. I can do one tap (as I leave the gym) to send the CSV.

Goal: A full-stack app on Windows that:

exposes an /ingest endpoint to receive the Strong CSV (multipart upload from iOS Shortcut),

stores raw CSVs and upserts sets into a local DB (SQLite),

serves a dashboard at / with charts (volume, duration, PRs, progression, Tuesday vs Tuesday, exercise filters),

runs on localhost and is reachable at https://lifting.dakheera47.com via Cloudflare Tunnel,

starts automatically, keeps running, and logs plainly.

Key constraint: The CSV has the same structure as the file I already shared:

Date,Workout Name,Duration,Exercise Name,Set Order,Weight,Reps,Distance,Seconds,RPE
2024-11-28 17:36:08,B,35m,Bent Over Row (Barbell),1,10.0,13.0,0,0.0,
...


Timezone: I’m in Europe/London. Use that for display; store timestamps as ISO strings (no TZ offset) or normalize to UTC with a stored local TZ reference.

1) Architecture Overview

Components:

FastAPI web app (Python) with:

POST /ingest?token=... (multipart file param file) — saves CSV, parses + upserts into SQLite.

GET / — dashboard (Jinja2 template + Plotly.js via CDN).

JSON APIs for charts, e.g.:

GET /api/sessions — date, total volume, duration.

GET /api/top-exercises?limit=15

GET /api/prs?exercise=...

GET /api/tuesday-strength

GET /api/exercise-progress?exercise=...

GET /health — basic health probe.

SQLite DB (data/lifting.db), local to the server.

Static files: HTML template + client JS.

Windows host with Python venv + Task Scheduler (or NSSM/WinSW) to keep app running.

iOS Shortcut in Share Sheet to upload the Strong CSV via HTTPS (token protected).

Data flow:

iPhone: Strong → Export → Share Sheet → Shortcut → POST /ingest.

Server: store raw CSV in data/uploads/, parse & normalize, idempotent upsert into sets table.

Dashboard reads from DB, renders interactive charts client-side.

1) Data Model & Rules

Incoming CSV columns (Strong):

Date (string, e.g., 2024-11-28 17:36:08)

Workout Name (string)

Duration (string like 35m; may be blank)

Exercise Name (string)

Set Order (int)

Weight (float)

Reps (float)

Distance (int)

Seconds (float)

RPE (float; often empty)

Normalization:

Parse Date to a datetime. Store as ISO string (YYYY-MM-DDTHH:MM:SS) or as original string; treat as naive local time (Europe/London) for charts. (If you prefer UTC, convert on ingest and store tz='Europe/London' in settings.)

Duration_min = numeric minutes extracted from Duration by removing trailing m. May be NULL.

Standardize column names in DB: date, workout_name, duration_min, exercise, set_order, weight, reps, distance, seconds.

Volume is computed on the fly as weight * reps (treat nulls as 0).

SQLite schema:

CREATE TABLE IF NOT EXISTS sets (
  id INTEGER PRIMARY KEY,
  date TEXT NOT NULL,
  workout_name TEXT,
  duration_min REAL,
  exercise TEXT NOT NULL,
  set_order INTEGER,
  weight REAL,
  reps REAL,
  distance REAL,
  seconds REAL
);
-- Dedup key: same set seen in multiple exports shouldn’t double-insert:
CREATE UNIQUE INDEX IF NOT EXISTS ux_sets_row
  ON sets(date, workout_name, exercise, set_order, weight, reps, seconds);
CREATE INDEX IF NOT EXISTS idx_sets_date_exercise
  ON sets(date, exercise);


Idempotence: Use INSERT OR IGNORE with the unique index above.

Assumptions & edge cases:

Empty RPE ignored.

Duration might be missing; duration_min becomes NULL.

CSV may contain duplicates across exports; dedup via unique index.

CSV order not guaranteed; ingestion doesn’t assume ordering.

3) API Contract

POST /ingest?token=STRING
Content-Type: multipart/form-data
Field: file (the CSV).
Response 200 JSON: { "stored":"strong_YYYYMMDD_HHMMSS.csv", "rows": <int> }
Errors:

401 if token mismatch

400 if file empty or not CSV

500 on unexpected

GET /api/sessions
Returns array of { session_date: "YYYY-MM-DD", volume: <float>, duration: <float|null> }.

GET /api/top-exercises?limit=15
Returns array of { exercise: <string>, volume: <float> } sorted desc.

GET /api/prs?exercise=...
Returns { exercise: <string>, max_weight: <float>, best_set: {date, weight, reps} }. If none, empty.

GET /api/tuesday-strength
Returns array of { date: "YYYY-MM-DD", exercise: <string>, max_weight: <float> } for Tuesdays only.

GET /api/exercise-progress?exercise=...
Returns array of { date: "YYYY-MM-DD", max_weight: <float>, top_set_volume: <float> }.

GET /health
Returns { ok: true, time: <ISO> }.

4) Frontend (Dashboard)

Stack: Jinja2 template for /, vanilla JS + Plotly.js (CDN).

Charts:

Volume over time (line, markers): /api/sessions.

Duration trend (line): /api/sessions.

Top exercises by total volume (bar): /api/top-exercises?limit=15.

Tuesday strength — line per exercise of max weight on each Tuesday (/api/tuesday-strength), with legend + toggle.

Exercise picker to view PRs & progress (line): /api/exercise-progress?exercise=….

UI goals: fast load, simple layout, date axis labels rotate, responsive.

5) Security

Token in querystring: ?token=VERY_LONG_RANDOM (store in .env).

CORS: allow your own domain only (tighten after initial setup).

Size limits: reject files > e.g. 10 MB.

Content sniffing: accept text/csv or application/vnd.ms-excel; do basic header validation.

1) Windows Setup & Run

Prereqs:

Windows 10/11 or Server

Python 3.11+ (py -V)

git (optional)

Steps:

Create project folder, e.g., C:\lifting-pipeline.

Create venv:

cd C:\lifting-pipeline
py -m venv .venv
.\.venv\Scripts\activate


Create structure:

lifting-pipeline\
  app\ (code lives here)
  data\uploads\ (create empty)
  .env
  requirements.txt


requirements.txt:

fastapi
uvicorn[standard]
python-multipart
jinja2
pydantic
pandas
plotly
python-dotenv


Install deps: pip install -r requirements.txt

Generate .env:

INGEST_TOKEN=put_a_very_long_random_string_here
TZ=Europe/London


Run app:

uvicorn app.main:app --host 127.0.0.1 --port 8069 --reload


Visit http://localhost:8069.

Keep it running (pick one):

Task Scheduler: Create task “LiftingPipeline”, trigger At logon & At startup, action:

Program: powershell.exe
Args: -NoProfile -ExecutionPolicy Bypass -Command "cd C:\lifting-pipeline; .\.venv\Scripts\activate.ps1; uvicorn app.main:app --host 127.0.0.1 --port 8069"


Settings: “Run whether user is logged on or not”, “Run with highest privileges”, “Restart task every 1 minute if fails”.

NSSM (if no GUI needed): wrap the uvicorn command as a Windows Service (Session 0 safe).


8) iOS Shortcut (Share Sheet → POST /ingest)

Goal: From Strong → Export → Share → your Shortcut → upload the CSV to https://lifting.dakheera47.com/ingest?token=....

Steps:

Open Shortcuts → + → New Shortcut.

Tap i (settings) → Show in Share Sheet: ON.
Accepted Types: Files.

Add actions (in this order):

Get File from Shortcut Input

If [Get Details of Files → Name] Ends With .csv

Otherwise → Quick Look (or Show Alert “Not a CSV”) → Stop Shortcut

Get Contents of URL

URL: https://lifting.dakheera47.com/ingest?token=YOUR_LONG_TOKEN

Method: POST

Request Body: Form

Add field:

Key: file

Type: File

File: Provided Input (the CSV from Strong)

Headers: (optional) User-Agent: LiftingShortcut/1.0

Get Dictionary from Input (to parse JSON response)

Show Result (optional: show {stored})

Offline behavior: If you’re offline leaving the gym, the POST will fail—add a second branch: If upload fails, Save File to iCloud Shortcuts/Outbox. Create another automation (personal automation “When I arrive home Wi-Fi”) to Find Files in Outbox and POST each, then delete on success.

Manual test: Share any CSV to the Shortcut; expect 200 and {stored: "...", rows: N}.

9) Code Skeleton (files and key content)

The AI should generate these with clean, readable code and docstrings.

app/db.py

DB_PATH, init_db(), get_conn() context manager.

Create schema + indices exactly as in §2.

app/processing.py

normalize_df(df) — implement as in §2.

upsert_sets(df) — INSERT OR IGNORE.

process_csv_to_db(csv_path) — returns row count.

compute_session_agg() — grouped totals.

app/main.py

Load .env (INGEST_TOKEN, TZ).

FastAPI app + CORS (origins = your domain and localhost).

Mount /static.

Routes:

GET / — render templates/dashboard.html (Jinja2).

POST /ingest — token guard, store file to data/uploads/strong_YYYYMMDD_HHMMSS.csv, process to DB.

GET /api/sessions — session aggregates.

GET /api/top-exercises — top N.

GET /api/prs — per exercise PR (highest weight).

GET /api/tuesday-strength — Tuesday max weight per exercise per date.

GET /api/exercise-progress — timeseries for a chosen exercise.

GET /health.

app/templates/dashboard.html

Minimal HTML with 4–5 div containers for Plotly charts.

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

Fetch helper: fetch('/api/sessions').then(...) etc.

Render:

Volume over time

Duration trend

Top exercises

Tuesday strength (multi-line with legend)

Exercise dropdown (built from /api/top-exercises?limit=1000 or /api/exercises endpoint) to render /api/exercise-progress.

Optional app/static/main.js for cleaner JS.

10) Testing & Validation

Unit-ish tests (manual or simple pytest):

Feed a sample CSV (like the one I provided) to normalize_df → expected columns and parsed types.

Insert same CSV twice → row count doesn’t increase (dedup works).

/api/sessions returns date-sorted entries; volume equals sum(weight*reps) across that date.

/api/tuesday-strength only returns Tuesdays.

/api/prs?exercise=... returns the max weight and identifies the correct row.

HTTP tests:

curl -F "file=@strong.csv" "http://127.0.0.1:8069/ingest?token=TOKEN"

curl http://127.0.0.1:8069/api/sessions

Load http://127.0.0.1:8069/ — charts render.


11) Observability & Ops

Logging: Log each ingest with filename and row count; log exceptions with stack trace to logs/app.log (rotating).

Health: /health endpoint; add “last_ingested_at” to DB or a status.json.

Backups: Periodically copy data/lifting.db and data/uploads/ to another disk (Windows File History or a PowerShell script via Task Scheduler).

Auto-start: Task Scheduler task “At startup” runs uvicorn; enable restart on failure.

Security hardening (optional):

Replace token with HMAC signature header (e.g., X-Signature: hex(hmac_sha256(secret, body))).


12) Product Details (Charts & Features)

Volume over time: x = session date, y = Σ(weight*reps) per date.

Duration trend: x = date, y = mean duration_min per date.

Top exercises by total volume: descending bar.

Tuesday strength: For each Tuesday, max(weight) per exercise; rendered as multi-line chart with legend.

Exercise progress:

max(weight) per date for the chosen exercise (line with markers).

Optional secondary series: top set volume for that date.

PRs pagelet: For a chosen exercise, display:

1RM estimators (e1RM via Epley: weight * (1 + reps/30) for best set of the day).

Highest actual weight and e1RM to date.

Filters: quick filter to exclude November (like we did), or pick date range.

13) Acceptance Criteria (what “done” means)

I can visit http://localhost:8069/ and see charts populated after the first ingest.

From iOS, I can export Strong CSV via Share Sheet → Shortcut and receive HTTP 200 and a JSON body.

Re-uploading the same CSV does not duplicate sets.

/api/sessions, /api/top-exercises, /api/tuesday-strength, /api/exercise-progress?exercise=... all return sensible data and are used on the dashboard.

App auto-starts on reboot and keeps running. Logs are written to disk.

Basic security: token required for ingest; dashboard not world-writable; CORS restricted to site hosts.

14) Nice-to-Haves (future)

Pagination & CSV download of session data (raw or filtered).

e1RM trendlines per exercise.

Weekly tonnage view (group by ISO week).

Streaks and consistency metrics.

CSV schema validation with clear “what’s wrong” error message to the Shortcut.

Import from legacy files (batch drop into data/uploads/ folder plus a CLI re-ingest).

---

Implementation Status (Generated Scaffold)

The initial FastAPI scaffold has been generated with:

- `app/main.py` FastAPI app & routes (/ingest, /api/*, /health, dashboard template).
- `app/db.py` SQLite initialization + meta table (tracks last_ingested_at).
- `app/processing.py` CSV normalization, upsert & aggregation helpers.
- `app/templates/dashboard.html` Basic Plotly.js dashboard layout.
- `app/static/main.js` Frontend fetch + chart rendering logic.
- `requirements.txt` Dependency list.
- `.env.example` Environment variable template.
- `tests_sample.csv` Sample Strong export for manual testing.

Quick Start

```bash
python -m venv .venv
source .venv/Scripts/activate  # (On PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
copy .env.example .env  # then edit INGEST_TOKEN
uvicorn app.main:app --host 127.0.0.1 --port 8069 --reload
```

Test Ingest

```bash
curl -F "file=@tests_sample.csv" "http://127.0.0.1:8069/ingest?token=$INGEST_TOKEN"
curl http://127.0.0.1:8069/api/sessions
```

Open http://127.0.0.1:8069 in a browser to view charts.

Next Steps

- Add automated tests (pytest) for normalization & endpoints.
- Harden security (restrict origins, optional HMAC signature).
- Add weekly tonnage & e1RM calculations.
- Implement backup script (PowerShell) for `data/`.

Notes

The unique index prevents duplicate set insertion across repeated CSV uploads.
`last_ingested_at` stored in `meta` table and surfaced on the dashboard.

Rebuilding / Resetting the Database

Use the helper script to wipe and optionally restore data from existing CSV uploads:

```bash
python scripts/rebuild_db.py              # backup then delete + recreate empty schema
python scripts/rebuild_db.py --with-imports  # backup, recreate, re-ingest all CSVs in data/uploads/
python scripts/rebuild_db.py --no-backup --with-imports  # destructive quick rebuild
```

Backups are stored in `data/backups/` named `lifting_YYYYMMDD_HHMMSS.db`.