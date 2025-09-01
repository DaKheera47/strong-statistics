# .dockerignore

```
.venv
__pycache__
*.pyc
*.pyo
*.pyd
*.db
*.sqlite
logs
data/uploads/*
.env
.git
.gitignore
out
frontend
scripts
tests_sample.csv
debug.html
run.bat

```

# .gitignore

```
# Python bytecode / cache
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
.venv/
venv/
env/

# Distribution / packaging
build/
dist/
*.egg-info/
.eggs/

# Logs
logs/*.log
logs/*.log.*

# Environment & secrets
.env
.env.*
!.env.example

# Data artifacts (keep schema code, ignore db + uploaded raw CSVs)
data/lifting.db
data/uploads/

# Jupyter / notebooks
.ipynb_checkpoints/

# Coverage / testing
.coverage
coverage.xml
htmlcov/
.pytest_cache/

# OS files
.DS_Store
Thumbs.db

# Editor
.vscode/
.idea/

# Cache
*.cache

# Node (if frontend experiments come later)
node_modules/

# Backup / temp
*.swp
*.tmp
~$*

.venv/
out/
```

# app\__init__.py

```py

```

# app\db.py

```py
"""Database utilities for the Lifting Pipeline.

Responsible for initializing the SQLite schema and providing a context
manager for connections.
"""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path
import threading
import time
import logging

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "lifting.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# SQLite connections are NOT thread-safe by default when shared;
# we create short-lived connections via contextmanager.
_lock = threading.Lock()
_LOCK_MAX_WAIT_SEC = 5  # fail fast if something holds the lock too long
logger = logging.getLogger("lifting.db")

SCHEMA_SQL = """
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
CREATE UNIQUE INDEX IF NOT EXISTS ux_sets_row
  ON sets(date, workout_name, exercise, set_order, weight, reps, seconds);
CREATE INDEX IF NOT EXISTS idx_sets_date_exercise
  ON sets(date, exercise);

CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT
);
"""

def init_db() -> None:
    """Initialize database schema if not present."""
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)


def set_meta(key: str, value: str) -> None:
    with get_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (key, value))
        conn.commit()


def get_meta(key: str) -> str | None:
    with get_conn() as conn:
        cur = conn.execute("SELECT value FROM meta WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

@contextmanager
def get_conn():
  """Yield a SQLite connection with row factory as dict-like tuples.

  Adds defensive diagnostics so if the global lock is ever held
  unexpectedly long (e.g. a crashed thread), we surface an error
  instead of silently hanging the request.
  """
  start_wait = time.time()
  acquired = _lock.acquire(timeout=_LOCK_MAX_WAIT_SEC)
  if not acquired:  # pragma: no cover - only triggers on pathological hang
    waited = time.time() - start_wait
    logger.error("DB lock acquisition failed after %.2fs (possible deadlock)", waited)
    raise RuntimeError("database busy: internal lock timeout")
  try:
    waited = time.time() - start_wait
    if waited > 0.25:
      logger.warning("DB lock waited %.3fs (unexpected contention)", waited)
    conn = sqlite3.connect(DB_PATH, timeout=2)
    try:
      conn.row_factory = sqlite3.Row
      yield conn
    finally:
      conn.close()
  finally:
    _lock.release()

```

# app\main.py

```py
"""FastAPI application entrypoint."""
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.templating import Jinja2Templates
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import faulthandler
import sys
import yaml
import threading, traceback, time

from .db import init_db, get_meta
from .processing import (
    process_csv_to_db,
    progressive_overload_data,
    volume_progression,
    personal_records_timeline,
    training_consistency,
    strength_balance,
    exercise_analysis,
    list_exercises,
    get_exercise_records,
    get_body_measurements,
    get_workout_calendar,
    get_workout_detail,
    get_exercise_detail,
    get_muscle_group_balance,
    get_training_streak,
    get_weekly_volume_heatmap,
    get_rep_range_distribution,
    get_exercise_frequency,
    get_strength_ratios,
    get_recovery_tracking,
    get_progressive_overload_rate,
    get_workout_duration_trends,
    get_best_sets_analysis,
    get_plateau_detection,
    build_dashboard_data,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
LOG_DIR = BASE_DIR / "logs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost").split(",") if o.strip()]
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

# Logging
logger = logging.getLogger("lifting")
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
try:
    logger.setLevel(getattr(logging, _log_level, logging.INFO))
except Exception:  # pragma: no cover
    logger.setLevel(logging.INFO)
handler = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=2_000_000, backupCount=3)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
faulthandler.enable(file=sys.stderr)

app = FastAPI(title="Lifting Pipeline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ---------------- Layout Config (YAML) -----------------
DEFAULT_LAYOUT = {
    "layout": {
        "rows": [
            ["progressive_overload:12"],
            ["volume_trend:8", "exercise_volume:4"],
            ["rep_distribution:4", "weekly_ppl:7", "muscle_balance:5"],
            ["calendar:12"],
        ]
    }
}

_LAYOUT_PATH = BASE_DIR / "dashboard_layout.yaml"

def load_layout_config():
    try:
        if _LAYOUT_PATH.exists():
            with _LAYOUT_PATH.open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        else:
            raw = {}
    except Exception as e:  # pragma: no cover - config parse failure fallback
        logger.error("failed parsing dashboard_layout.yaml: %s", e)
        raw = {}
    # Merge defaults
    rows = (raw.get("layout", {}) or {}).get("rows") or DEFAULT_LAYOUT["layout"]["rows"]
    # Normalise rows -> list[list[str]] of tokens "widget[:width]"
    norm_rows: list[list[dict]] = []
    for row in rows:
        norm_row = []
        if not isinstance(row, list):
            continue
        for cell in row:
            # Accept either "name:width" or dict {name: width}
            name = None
            width = 12
            if isinstance(cell, str):
                if ":" in cell:
                    name_part, width_part = cell.split(":", 1)
                    name = name_part.strip()
                    try:
                        width = int(width_part)
                    except ValueError:
                        width = 12
                else:
                    name = cell.strip()
            elif isinstance(cell, dict):
                # first key
                if cell:
                    name, width_val = next(iter(cell.items()))
                    name = str(name)
                    try:
                        width = int(width_val)
                    except Exception:
                        width = 12
            if not name:
                continue
            width = max(1, min(12, width))
            norm_row.append({"widget": name, "width": width})
        if norm_row:
            norm_rows.append(norm_row)
    return {"rows": norm_rows}

LAYOUT_CONFIG = load_layout_config()

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# Favicon / logo routes serving the root-level icon.svg so we don't have to duplicate the asset.
ICON_PATH = BASE_DIR / "icon.svg"

# Derived icon (apple touch png) path (generated lazily on first request)
APPLE_TOUCH_PATH = BASE_DIR / "static" / "apple-touch-icon.png"

if ICON_PATH.exists():
    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon_svg():  # type: ignore[override]
        from fastapi.responses import FileResponse
        return FileResponse(str(ICON_PATH), media_type="image/svg+xml")

    # Provide /favicon.ico for user agents expecting .ico – browsers accept SVG served as ICO in practice; if needed convert later.
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon_ico():  # type: ignore[override]
        from fastapi.responses import FileResponse
        return FileResponse(str(ICON_PATH), media_type="image/svg+xml")

    @app.get("/apple-touch-icon.png", include_in_schema=False)
    async def apple_touch_icon():  # type: ignore[override]
        """Serve (lazily generate) a 180x180 PNG for iOS home screen.

        Generation only happens if file missing or older than svg mtime.
        """
        from fastapi.responses import FileResponse
        try:
            # Only import heavy deps when needed
            if (not APPLE_TOUCH_PATH.exists()) or (APPLE_TOUCH_PATH.stat().st_mtime < ICON_PATH.stat().st_mtime):
                from cairosvg import svg2png  # type: ignore
                APPLE_TOUCH_PATH.parent.mkdir(parents=True, exist_ok=True)
                png_bytes = svg2png(url=str(ICON_PATH), output_width=180, output_height=180)
                APPLE_TOUCH_PATH.write_bytes(png_bytes)
        except Exception:  # pragma: no cover - non critical
            # Fall back to serving svg directly if conversion failed
            return FileResponse(str(ICON_PATH), media_type="image/svg+xml")
        return FileResponse(str(APPLE_TOUCH_PATH), media_type="image/png")

init_db()

# Simple request/response timing middleware for diagnostics when requests hang
@app.middleware("http")
async def timing_middleware(request: Request, call_next):  # type: ignore[override]
    start = datetime.utcnow()
    path = request.url.path
    # Early diagnostic log to confirm request entered middleware
    logger.info("middleware_enter path=%s method=%s", path, request.method)
    try:
        response = await call_next(request)
        return response
    finally:
        dur_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info("request path=%s method=%s dur_ms=%.1f", path, request.method, dur_ms)

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "layout_config": LAYOUT_CONFIG})


@app.get("/workout/{workout_date}", response_class=HTMLResponse)
async def workout_detail_page(request: Request, workout_date: str):
    """Shareable workout detail page - loads dashboard with specific workout opened"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "workout_date": workout_date,
        "layout_config": LAYOUT_CONFIG,
    })


@app.get("/debug", response_class=HTMLResponse)
async def debug_page(request: Request):
    """Debug page to test API endpoints"""
    debug_html = """<!DOCTYPE html>
<html>
<head>
    <title>API Debug Test</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .success { color: green; }
        .error { color: red; }
        pre { background: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto; }
    </style>
</head>
<body>
    <h1>API Debug Test</h1>
    <div id="status"></div>
    <div id="results"></div>
    
    <script>
        const statusDiv = document.getElementById('status');
        const resultsDiv = document.getElementById('results');
        
        async function testAPI(endpoint, name) {
            try {
                statusDiv.innerHTML += `<p>Testing ${name}...</p>`;
                const response = await fetch(`/api/${endpoint}`);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                statusDiv.innerHTML += `<p class="success">✅ ${name} - Success (${JSON.stringify(data).length} chars)</p>`;
                
                resultsDiv.innerHTML += `<h3>${name}</h3><pre>${JSON.stringify(data, null, 2).substring(0, 500)}${JSON.stringify(data).length > 500 ? '...' : ''}</pre>`;
                return data;
            } catch (error) {
                statusDiv.innerHTML += `<p class="error">❌ ${name} - Error: ${error.message}</p>`;
                console.error(`Error testing ${name}:`, error);
                return null;
            }
        }
        
        async function runTests() {
            statusDiv.innerHTML = '<h2>Running API Tests...</h2>';
            
            await testAPI('progressive-overload', 'Progressive Overload');
            await testAPI('volume-progression', 'Volume Progression');
            await testAPI('personal-records', 'Personal Records');
            await testAPI('training-consistency', 'Training Consistency');
            await testAPI('strength-balance', 'Strength Balance');
            await testAPI('exercises', 'Exercises List');
            
            // Strong-inspired endpoints
            await testAPI('records', 'Personal Records Table');
            await testAPI('calendar', 'Training Calendar');
            await testAPI('muscle-balance', 'Muscle Balance');
            await testAPI('measurements', 'Body Measurements');
            await testAPI('training-streak', 'Training Streak');
            
            statusDiv.innerHTML += '<h2>Tests Complete!</h2>';
        }
        
        // Run tests when page loads
        document.addEventListener('DOMContentLoaded', runTests);
    </script>
</body>
</html>"""
    return HTMLResponse(content=debug_html)


@app.post("/ingest")
async def ingest(request: Request, token: str = Query(""), file: UploadFile | None = File(None)):
    """Ingest a Strong export CSV.

    Auth: token provided either as query param (?token=...) or X-Token header.
    Returns 401 on mismatch, 500 if server misconfigured without token.
    """
    header_token = request.headers.get("X-Token", "")
    provided = token or header_token
    if not INGEST_TOKEN:
        raise HTTPException(status_code=500, detail="server token not configured")
    if not provided:
        raise HTTPException(status_code=401, detail="missing token")
    if provided != INGEST_TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")

    # Determine content type & obtain raw bytes (multipart or raw body)
    if file is not None:
        content_type = file.content_type or ""
        raw = await file.read()
        mode = "multipart"
    else:
        content_type = request.headers.get("content-type", "")
        raw = await request.body()
        mode = "raw"

    if content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(status_code=400, detail=f"unsupported content type: {content_type} (mode={mode})")

    if not raw:
        raise HTTPException(status_code=400, detail="empty body")

    # Size guard
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(status_code=400, detail="file too large")

    # Basic header validation
    header_line = raw.splitlines()[0].decode(errors="ignore") if raw else ""
    if not header_line.startswith("Date,Workout Name,Duration,Exercise Name"):
        raise HTTPException(status_code=400, detail="invalid csv header")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    stored_name = f"strong_{ts}.csv"
    dest = UPLOAD_DIR / stored_name
    dest.write_bytes(raw)

    try:
        inserted = process_csv_to_db(dest)
    except Exception as e:
        logger.exception("failed ingest for %s", stored_name)
        raise HTTPException(status_code=500, detail=str(e))

    logger.info("ingested file=%s inserted_rows=%s mode=%s size_bytes=%s", stored_name, inserted, mode, len(raw))
    return {"stored": stored_name, "rows": inserted}


@app.get("/api/progressive-overload")
async def api_progressive_overload():
    """Get strength progression data for top exercises."""
    return progressive_overload_data()


@app.get("/api/volume-progression") 
async def api_volume_progression():
    """Get training volume progression over time."""
    return volume_progression()


@app.get("/api/personal-records")
async def api_personal_records():
    """Get timeline of personal records."""
    return personal_records_timeline()


@app.get("/api/training-consistency")
async def api_training_consistency():
    """Get training frequency and consistency metrics."""
    return training_consistency()


@app.get("/api/strength-balance")
async def api_strength_balance():
    """Get strength balance across movement patterns."""
    return strength_balance()


@app.get("/api/exercise-analysis")
async def api_exercise_analysis(exercise: str):
    """Get detailed analysis for a specific exercise."""
    return exercise_analysis(exercise)


@app.get("/api/exercises")
async def api_exercises():
    return list_exercises()


# Strong App inspired endpoints

@app.get("/api/records")
async def get_records():
    """Get personal records for all exercises - like Strong's Records screen"""
    return get_exercise_records()

@app.get("/api/measurements")
async def get_measurements():
    """Get body measurements data - like Strong's measurements feature"""
    return get_body_measurements()

@app.get("/api/calendar")
async def get_calendar():
    """Get workout calendar data - like Strong's workout frequency tracking"""
    return get_workout_calendar()

@app.get("/api/workout/{workout_date}")
async def get_workout_detail_endpoint(workout_date: str):
    """Get detailed workout information for a specific date - like Strong's workout view"""
    result = get_workout_detail(workout_date)
    if result is None:
        raise HTTPException(status_code=404, detail="Workout not found for the specified date")
    return result

@app.get("/api/exercise-detail/{exercise}")
async def get_exercise_detail_endpoint(exercise: str):
    """Get detailed statistics for a specific exercise - like Strong's Exercise Detail Screen"""
    result = get_exercise_detail(exercise)
    if result is None:
        raise HTTPException(status_code=404, detail="Exercise not found or no data available")
    return result

@app.get("/api/muscle-balance")
async def get_muscle_balance():
    """Get muscle group balance analysis - like Strong's muscle balance feature"""
    return get_muscle_group_balance()

@app.get("/api/training-streak")
async def get_training_streak_endpoint():
    """Get training streak data - like Strong's consistency tracking"""
    return get_training_streak()


# Additional hardcore analytics endpoints

@app.get("/api/volume-heatmap")
async def get_volume_heatmap():
    """Get weekly volume heatmap data - like GitHub contributions"""
    return get_weekly_volume_heatmap()

@app.get("/api/rep-distribution")
async def get_rep_distribution():
    """Get rep range distribution analysis"""
    return get_rep_range_distribution()

@app.get("/api/exercise-frequency")
async def get_exercise_frequency_endpoint():
    """Get exercise frequency analysis"""
    return get_exercise_frequency()

@app.get("/api/strength-ratios")
async def get_strength_ratios_endpoint():
    """Get strength ratios between major lifts"""
    return get_strength_ratios()

@app.get("/api/recovery-tracking")
async def get_recovery_tracking_endpoint():
    """Get recovery time analysis between muscle group sessions"""
    return get_recovery_tracking()

@app.get("/api/progression-rate")
async def get_progression_rate():
    """Get progressive overload rate analysis"""
    return get_progressive_overload_rate()

@app.get("/api/workout-duration")
async def get_workout_duration():
    """Get workout duration trends and efficiency metrics"""
    return get_workout_duration_trends()

@app.get("/api/best-sets")
async def get_best_sets():
    """Get best single set performance for each exercise"""
    return get_best_sets_analysis()

@app.get("/api/plateau-detection")
async def get_plateau_detection_endpoint():
    """Detect potential plateaus in strength progression"""
    return get_plateau_detection()


# New consolidated dashboard endpoint (refactored progression-focused dashboard)
@app.get("/api/dashboard")
async def get_dashboard_endpoint(
    start: str | None = Query(None),
    end: str | None = Query(None),
    exercises: str | None = Query(None, description="(deprecated – ignored)")
):
    # exercises param kept for backward compat but ignored (frontend handles filtering)
    return build_dashboard_data(start, end, None)


@app.get("/health")
async def health():
    payload = {"ok": True, "time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "last_ingested_at": get_meta("last_ingested_at")}
    # Add diagnostic header so we can tell if Cloudflare is reaching origin.
    return JSONResponse(payload, headers={"X-Lifting-Origin": "fastapi"})


@app.post("/debug/inspect")
async def debug_inspect(request: Request, token: str = Query("")):
    """Return detailed info about an incoming upload (diagnostic only).

    Protected by the same token as ingest. DO NOT leave enabled in production long-term.
    Reports headers, query params, limited body sample, and multipart form structure.
    """
    provided = token or request.headers.get("X-Token", "")
    if not INGEST_TOKEN:
        raise HTTPException(status_code=500, detail="server token not configured")
    if provided != INGEST_TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")

    # Raw body (may be large) – cap sample size
    body_bytes = await request.body()
    body_sample_max = 1024
    body_sample = body_bytes[:body_sample_max]
    body_truncated = len(body_bytes) > body_sample_max

    info = {
        "method": request.method,
        "url": str(request.url),
        "query_params": dict(request.query_params),
        "headers": {k: v for k, v in request.headers.items() if k.lower() not in {"authorization"}},
        "content_type": request.headers.get("content-type"),
        "raw_body_length": len(body_bytes),
        "raw_body_sample_hex": body_sample.hex(),
        "raw_body_sample_ascii": body_sample.decode(errors="replace"),
        "raw_body_truncated": body_truncated,
        "parsed_form": None,
    }

    # Try parse multipart/form-data
    try:
        form = await request.form()
        form_dict = {}
        for key, val in form.multi_items():  # preserves duplicates
            if hasattr(val, "filename") and val.filename:
                file_bytes = await val.read()
                form_dict.setdefault(key, []).append({
                    "filename": val.filename,
                    "content_type": val.content_type,
                    "size": len(file_bytes),
                    "head_ascii": file_bytes[:200].decode(errors="replace"),
                })
            else:
                form_dict.setdefault(key, []).append(str(val))
        info["parsed_form"] = form_dict
    except Exception as e:  # noqa: BLE001
        info["form_error"] = str(e)

    return JSONResponse(info)


@app.get("/debug/threads")
async def debug_threads(token: str = Query("")):
    """Return a snapshot of all thread stack traces (diagnostic).

    Protect with ingest token to avoid exposing internals.
    Useful if requests start hanging: call /debug/threads to see where code is stuck.
    """
    if INGEST_TOKEN and token != INGEST_TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")
    frames = sys._current_frames()  # type: ignore[attr-defined]
    out = []
    for thread in threading.enumerate():
        fid = getattr(thread, 'ident', None)
        frame = frames.get(fid)
        out.append({
            'thread_name': thread.name,
            'ident': fid,
            'daemon': thread.daemon,
            'stack': traceback.format_stack(frame) if frame else []
        })
    return {'threads': out, 'count': len(out), 'time': datetime.utcnow().isoformat()+ 'Z'}

```

# app\processing.py

```py
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
    "Date",
    "Workout Name",
    "Duration",
    "Exercise Name",
    "Set Order",
    "Weight",
    "Reps",
    "Distance",
    "Seconds",
    "RPE",
]

NORMALIZED_COLUMNS = [
    "date",
    "workout_name",
    "duration_min",
    "exercise",
    "set_order",
    "weight",
    "reps",
    "distance",
    "seconds",
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

    df["workout_name"] = (
        df["Workout Name"].astype(str).where(~df["Workout Name"].isna(), None)
    )
    df["exercise"] = df["Exercise Name"].astype(str)

    numeric_map = {
        "Set Order": "set_order",
        "Weight": "weight",
        "Reps": "reps",
        "Distance": "distance",
        "Seconds": "seconds",
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
        return (
            cur.rowcount
        )  # number attempted; not exact inserted (SQLite sets to # of executions). We'll compute inserted via changes().


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
                (exercise,),
            )

            data = []
            for row in cur.fetchall():
                data.append(
                    {
                        "date": row[0],
                        "max_weight": row[1],
                        "best_set_volume": row[2],
                        "estimated_1rm": round(row[3], 1),
                    }
                )
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

            data.append(
                {
                    "date": row[0],
                    "volume": volume,
                    "volume_7day_avg": round(rolling_avg, 1),
                    "exercises_count": row[2],
                    "duration": row[3],
                }
            )

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

        return [
            {
                "exercise": row[0],
                "date": row[1],
                "weight": row[2],
                "reps": row[3],
                "estimated_1rm": round(row[4], 1),
            }
            for row in cur.fetchall()
        ]


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
            weekly_data.append(
                {
                    "week": row[0],
                    "workouts": row[1],
                    "volume": row[2],
                    "avg_duration": row[3],
                }
            )

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
            monthly_data.append(
                {
                    "month": row[0],
                    "workouts": row[1],
                    "volume": row[2],
                    "unique_exercises": row[3],
                }
            )

        return {"weekly": weekly_data, "monthly": monthly_data}


def strength_balance():
    """Analyze strength balance across muscle groups/movement patterns."""
    with get_conn() as conn:
        # Categorize exercises by movement pattern
        movement_patterns = {
            "Push": ["Chest Press", "Shoulder Press", "Push"],
            "Pull": ["Seated Row", "Lat Pulldown", "Bent Over Row", "Pull"],
            "Squat": ["Squat", "Leg Press"],
            "Deadlift": ["Deadlift", "Romanian Deadlift"],
            "Accessory": ["Bicep Curl", "Pec Deck", "Leg Extension", "Glute Kickback"],
        }

        result = {}
        for pattern, keywords in movement_patterns.items():
            like_conditions = " OR ".join(
                [f"exercise LIKE '%{keyword}%'" for keyword in keywords]
            )

            cur = conn.execute(
                f"""
                SELECT date(substr(date,1,10)) as workout_date,
                       MAX(weight) as max_weight,
                       SUM(COALESCE(weight,0) * COALESCE(reps,0)) as total_volume,
                       MAX(COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0)) as best_estimated_1rm
                FROM sets
                WHERE ({like_conditions}) AND weight IS NOT NULL
                GROUP BY workout_date
                HAVING COUNT(*) > 0
                ORDER BY workout_date
            """
            )

            pattern_data = []
            for row in cur.fetchall():
                pattern_data.append(
                    {
                        "date": row[0],
                        "max_weight": row[1],
                        "volume": row[2],
                        "estimated_1rm": round(row[3], 1),
                    }
                )

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
            (exercise,),
        )

        all_sets = []
        for row in cur.fetchall():
            all_sets.append(
                {
                    "date": row[0],
                    "weight": row[1],
                    "reps": row[2],
                    "volume": row[3],
                    "estimated_1rm": round(row[4], 1),
                }
            )

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
            (exercise,),
        )

        session_summary = []
        for row in cur.fetchall():
            session_summary.append(
                {
                    "date": row[0],
                    "max_weight": row[1],
                    "session_volume": row[2],
                    "sets_count": row[3],
                    "avg_reps": round(row[4], 1),
                    "estimated_1rm": round(row[5], 1),
                }
            )

        return {"all_sets": all_sets, "session_summary": session_summary}


def list_exercises():
    with get_conn() as conn:
        cur = conn.execute("SELECT DISTINCT exercise FROM sets ORDER BY exercise")
        return [r[0] for r in cur.fetchall()]


# Strong App inspired analytics functions


def get_exercise_records():
    """Get personal records for all exercises - like Strong's Records screen"""
    with get_conn() as conn:
        cur = conn.execute(
            """
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
        """
        )

        records = []
        for row in cur.fetchall():
            records.append(
                {
                    "exercise": row[0],
                    "max_weight": row[1],
                    "max_reps": row[2],
                    "max_volume_set": row[3],
                    "estimated_1rm": round(row[7], 1),
                    "total_sets": row[4],
                    "first_performed": row[5],
                    "last_performed": row[6],
                }
            )

        return records


def get_body_measurements():
    """Get body measurements data - like Strong's measurements feature"""
    # For now, we'll simulate this since we don't have a measurements table
    # In a real app, you'd have a separate measurements table
    with get_conn() as conn:
        # Get workout dates to simulate body weight tracking
        cur = conn.execute(
            """
            SELECT date(substr(date,1,10)) as workout_date, COUNT(*) as workout_count
            FROM sets
            GROUP BY workout_date
            ORDER BY workout_date
        """
        )

        result = cur.fetchall()

        # Simulate body weight data (in a real app, this would come from a measurements table)
        measurements = []
        base_weight = 80  # kg

        for i, row in enumerate(result):
            # Simulate gradual weight changes over time
            weight_change = (i * 0.1) - 2  # Slight weight loss trend
            measurements.append(
                {
                    "date": row[0],
                    "weight": round(base_weight + weight_change, 1),
                    "type": "weight",
                }
            )

        return measurements


def get_workout_calendar():
    """Get workout calendar data - like Strong's workout frequency tracking"""
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT 
                date(substr(date,1,10)) as workout_date,
                MIN(workout_name) as workout_name,
                COUNT(DISTINCT exercise) as exercises_performed,
                SUM(COALESCE(weight,0) * COALESCE(reps,0)) as total_volume,
                COUNT(*) as total_sets,
                AVG(duration_min) as avg_duration_minutes
            FROM sets
            WHERE weight IS NOT NULL AND reps > 0
            GROUP BY workout_date
            ORDER BY workout_date
        """
        )

        calendar_data = []
        for row in cur.fetchall():
            calendar_data.append(
                {
                    "date": row[0],
                    "workout_name": row[1] or "Workout",
                    "exercises_performed": row[2],
                    "total_volume": round(row[3], 1),
                    "total_sets": row[4],
                    "duration_minutes": round(row[5], 1) if row[5] else 0,
                }
            )

        return calendar_data


def get_workout_detail(workout_date):
    """Get detailed workout information for a specific date - like Strong's workout view"""
    with get_conn() as conn:
        # First, get basic workout info
        workout_info_cur = conn.execute(
            """
            SELECT 
                date(substr(date,1,10)) as workout_date,
                workout_name,
                AVG(duration_min) as duration_minutes,
                MIN(substr(date,12,8)) as start_time,
                MAX(substr(date,12,8)) as end_time,
                COUNT(DISTINCT exercise) as total_exercises,
                COUNT(*) as total_sets,
                SUM(COALESCE(weight,0) * COALESCE(reps,0)) as total_volume
            FROM sets
            WHERE date(substr(date,1,10)) = ?
            GROUP BY workout_date, workout_name
        """,
            (workout_date,),
        )

        workout_info = workout_info_cur.fetchone()
        if not workout_info:
            return None

        # Get all exercises and their sets for this workout
        exercises_cur = conn.execute(
            """
            SELECT 
                exercise,
                set_order,
                weight,
                reps,
                COALESCE(weight,0) * COALESCE(reps,0) as volume,
                -- Estimate 1RM using Epley formula
                COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0) as estimated_1rm,
                substr(date,12,8) as set_time
            FROM sets
            WHERE date(substr(date,1,10)) = ?
            ORDER BY exercise, set_order
        """,
            (workout_date,),
        )

        exercises_data = {}
        total_prs = 0

        # Group sets by exercise
        for row in exercises_cur.fetchall():
            exercise = row[0]
            if exercise not in exercises_data:
                exercises_data[exercise] = {
                    "exercise_name": exercise,
                    "sets": [],
                    "best_set": {"weight": 0, "reps": 0, "volume": 0, "1rm": 0},
                    "total_volume": 0,
                    "total_sets": 0,
                    "personal_records": 0,
                }

            set_data = {
                "set_number": row[1],
                "weight": row[2],
                "reps": row[3],
                "volume": round(row[4], 1),
                "estimated_1rm": round(row[5], 1),
                "time": row[6],
            }

            exercises_data[exercise]["sets"].append(set_data)
            exercises_data[exercise]["total_volume"] += set_data["volume"]
            exercises_data[exercise]["total_sets"] += 1

            # Track best set for this exercise in this workout
            best = exercises_data[exercise]["best_set"]
            if set_data["volume"] > best["volume"]:
                best.update(
                    {
                        "weight": set_data["weight"],
                        "reps": set_data["reps"],
                        "volume": set_data["volume"],
                        "1rm": set_data["estimated_1rm"],
                    }
                )

        # Check for personal records by comparing with historical data
        for exercise in exercises_data.keys():
            best_set = exercises_data[exercise]["best_set"]

            # Check if this was a weight PR
            weight_pr_cur = conn.execute(
                """
                SELECT MAX(weight) as prev_max_weight
                FROM sets
                WHERE exercise = ? AND date(substr(date,1,10)) < ? AND weight IS NOT NULL
            """,
                (exercise, workout_date),
            )

            prev_max = weight_pr_cur.fetchone()[0] or 0
            if best_set["weight"] > prev_max:
                exercises_data[exercise]["personal_records"] += 1
                total_prs += 1

        # Round total volumes
        for exercise in exercises_data.values():
            exercise["total_volume"] = round(exercise["total_volume"], 1)

        return {
            "date": workout_info[0],
            "workout_name": workout_info[1],
            "duration_minutes": round(workout_info[2], 1) if workout_info[2] else 0,
            "start_time": workout_info[3],
            "end_time": workout_info[4],
            "total_exercises": workout_info[5],
            "total_sets": workout_info[6],
            "total_volume": round(workout_info[7], 1),
            "total_prs": total_prs,
            "exercises": list(exercises_data.values()),
        }


def get_exercise_detail(exercise_name):
    """Get detailed statistics for a specific exercise - like Strong's Exercise Detail Screen"""
    with get_conn() as conn:
        # Get all sets for this exercise
        cur = conn.execute(
            """
            SELECT 
                date(substr(date,1,10)) as workout_date,
                weight,
                reps,
                COALESCE(weight,0) * COALESCE(reps,0) as volume,
                COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0) as estimated_1rm
            FROM sets
            WHERE exercise = ? AND weight IS NOT NULL AND reps > 0
            ORDER BY date, set_order
        """,
            (exercise_name,),
        )

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
                "date": row[0],
                "weight": row[1],
                "reps": row[2],
                "volume": row[3],
                "estimated_1rm": row[4],
            }
            sets_data.append(set_data)

            max_weight = max(max_weight, row[1])
            max_reps = max(max_reps, row[2])
            max_volume = max(max_volume, row[3])
            max_1rm = max(max_1rm, row[4])

        # Get workout frequency
        frequency_cur = conn.execute(
            """
            SELECT COUNT(DISTINCT date(substr(date,1,10))) as workout_days
            FROM sets
            WHERE exercise = ?
        """,
            (exercise_name,),
        )

        workout_days = frequency_cur.fetchone()[0]

        # Calculate days since first and last workout
        first_date = result[0][0]
        last_date = result[-1][0]

        return {
            "exercise_name": exercise_name,
            "total_sets": len(sets_data),
            "workout_days": workout_days,
            "max_weight": max_weight,
            "max_reps": max_reps,
            "max_volume": max_volume,
            "max_1rm": round(max_1rm, 1),
            "first_workout": first_date,
            "last_workout": last_date,
            "sets_history": sets_data,
        }


def get_muscle_group_balance():
    """Analyze strength balance across muscle groups - Strong's muscle balance feature"""
    with get_conn() as conn:
        # Categorize exercises by muscle groups (more detailed than movement patterns)
        muscle_groups = {
            "Chest": ["Chest Press", "Bench Press", "Pec Deck", "Incline"],
            "Back": ["Seated Row", "Lat Pulldown", "Bent Over Row", "Pull-up", "Chin"],
            "Shoulders": [
                "Shoulder Press",
                "Lateral Raise",
                "Military Press",
                "Overhead Press",
            ],
            "Legs": ["Squat", "Leg Press", "Leg Extension", "Leg Curl", "Calf"],
            "Arms": ["Bicep Curl", "Tricep", "Hammer Curl"],
            "Core": ["Plank", "Crunch", "Russian Twist", "Dead Bug"],
        }

        balance_data = {}

        for muscle_group, keywords in muscle_groups.items():
            like_conditions = " OR ".join(
                [f"exercise LIKE '%{keyword}%'" for keyword in keywords]
            )

            cur = conn.execute(
                f"""
                SELECT 
                    MAX(COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0)) as max_estimated_1rm,
                    SUM(COALESCE(weight,0) * COALESCE(reps,0)) as total_volume,
                    COUNT(*) as total_sets,
                    COUNT(DISTINCT date(substr(date,1,10))) as workout_days,
                    MAX(weight) as max_weight
                FROM sets
                WHERE ({like_conditions}) AND weight IS NOT NULL
            """
            )

            result = cur.fetchone()

            if result and result[0]:  # Only include if we have data
                balance_data[muscle_group] = {
                    "max_estimated_1rm": round(result[0], 1),
                    "total_volume": round(result[1], 1),
                    "total_sets": result[2],
                    "workout_days": result[3],
                    "max_weight": result[4],
                }

        return balance_data


def get_training_streak():
    """Calculate current and longest training streaks - like Strong's consistency tracking"""
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT DISTINCT date(substr(date,1,10)) as workout_date
            FROM sets
            WHERE weight IS NOT NULL
            ORDER BY workout_date DESC
        """
        )

        workout_dates = [row[0] for row in cur.fetchall()]

        if not workout_dates:
            return {"current_streak": 0, "longest_streak": 0, "last_workout": None}

        # Calculate current streak
        current_streak = 0
        today = datetime.now().date()

        for date_str in workout_dates:
            workout_date = datetime.strptime(date_str, "%Y-%m-%d").date()
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
            current_date = datetime.strptime(workout_dates[i - 1], "%Y-%m-%d").date()
            next_date = datetime.strptime(workout_dates[i], "%Y-%m-%d").date()

            # If within 7 days, continue streak
            if (current_date - next_date).days <= 7:
                temp_streak += 1
                longest_streak = max(longest_streak, temp_streak)
            else:
                temp_streak = 1

        return {
            "current_streak": current_streak,
            "longest_streak": longest_streak,
            "last_workout": workout_dates[0],
            "total_workout_days": len(workout_dates),
        }


# Additional Strong-inspired analytics for maximum data visualization


def get_weekly_volume_heatmap():
    """Generate heatmap data for weekly training volume - like GitHub contribution graph"""
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT 
                date(substr(date,1,10)) as workout_date,
                strftime('%w', date(substr(date,1,10))) as day_of_week,
                strftime('%W', date(substr(date,1,10))) as week_number,
                SUM(COALESCE(weight,0) * COALESCE(reps,0)) as daily_volume
            FROM sets
            WHERE weight IS NOT NULL AND reps > 0
            GROUP BY workout_date
            ORDER BY workout_date
        """
        )

        heatmap_data = []
        for row in cur.fetchall():
            heatmap_data.append(
                {
                    "date": row[0],
                    "day_of_week": int(row[1]),  # 0=Sunday, 1=Monday, etc.
                    "week_number": row[2],
                    "volume": round(row[3], 1),
                    "intensity": min(
                        5, max(1, int(row[3] / 500))
                    ),  # Scale 1-5 for color intensity
                }
            )

        return heatmap_data


def get_rep_range_distribution():
    """Analyze rep range distribution to see training focus"""
    with get_conn() as conn:
        cur = conn.execute(
            """
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
        """
        )

        return [
            {
                "rep_range": row[0],
                "set_count": row[1],
                "total_volume": round(row[2], 1),
                "percentage": 0,  # Will calculate in frontend
            }
            for row in cur.fetchall()
        ]


def get_exercise_frequency():
    """Track how frequently each exercise is performed"""
    with get_conn() as conn:
        cur = conn.execute(
            """
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
        """
        )

        return [
            {
                "exercise": row[0],
                "workout_days": row[1],
                "total_sets": row[2],
                "avg_weight": round(row[3], 1),
                "last_performed": row[4],
                "first_performed": row[5],
                "frequency_score": row[1] * row[2],  # Days * Sets for ranking
            }
            for row in cur.fetchall()
        ]


def get_strength_ratios():
    """Calculate strength ratios between major lifts"""
    with get_conn() as conn:
        # Define major lift patterns
        lift_patterns = {
            "Squat": ["Squat", "Leg Press"],
            "Bench": ["Chest Press", "Bench Press"],
            "Deadlift": ["Deadlift"],
            "Row": ["Seated Row", "Bent Over Row"],
            "Press": ["Shoulder Press", "Overhead Press"],
        }

        ratios = {}
        max_weights = {}

        for lift_name, patterns in lift_patterns.items():
            like_conditions = " OR ".join(
                [f"exercise LIKE '%{pattern}%'" for pattern in patterns]
            )

            cur = conn.execute(
                f"""
                SELECT MAX(COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0)) as max_1rm
                FROM sets
                WHERE ({like_conditions}) AND weight IS NOT NULL
            """
            )

            result = cur.fetchone()
            if result and result[0]:
                max_weights[lift_name] = round(result[0], 1)

        # Calculate ratios (using typical strength standards)
        if "Squat" in max_weights and "Bench" in max_weights:
            ratios["Squat_to_Bench"] = round(
                max_weights["Squat"] / max_weights["Bench"], 2
            )

        if "Deadlift" in max_weights and "Squat" in max_weights:
            ratios["Deadlift_to_Squat"] = round(
                max_weights["Deadlift"] / max_weights["Squat"], 2
            )

        if "Bench" in max_weights and "Row" in max_weights:
            ratios["Bench_to_Row"] = round(max_weights["Bench"] / max_weights["Row"], 2)

        return {
            "max_lifts": max_weights,
            "ratios": ratios,
            "ideal_ratios": {
                "Squat_to_Bench": 1.3,  # Squat should be ~30% higher than bench
                "Deadlift_to_Squat": 1.2,  # Deadlift should be ~20% higher than squat
                "Bench_to_Row": 1.0,  # Bench and row should be roughly equal
            },
        }


def get_recovery_tracking():
    """Track recovery time between sessions for each muscle group"""
    with get_conn() as conn:
        # Group exercises by muscle groups
        muscle_groups = {
            "Chest": ["Chest Press", "Bench Press", "Pec Deck"],
            "Back": ["Seated Row", "Lat Pulldown", "Bent Over Row", "Pull"],
            "Shoulders": ["Shoulder Press", "Military Press", "Lateral Raise"],
            "Legs": ["Squat", "Leg Press", "Leg Extension", "Leg Curl"],
            "Arms": ["Bicep Curl", "Tricep", "Hammer Curl"],
        }

        recovery_data = {}

        for muscle_group, exercises in muscle_groups.items():
            like_conditions = " OR ".join(
                [f"exercise LIKE '%{ex}%'" for ex in exercises]
            )

            cur = conn.execute(
                f"""
                SELECT 
                    date(substr(date,1,10)) as workout_date,
                    LAG(date(substr(date,1,10))) OVER (ORDER BY date) as prev_date
                FROM sets
                WHERE ({like_conditions}) AND weight IS NOT NULL
                GROUP BY workout_date
                ORDER BY workout_date
            """
            )

            recovery_times = []
            for row in cur.fetchall():
                if row[1]:  # If there's a previous date
                    current = datetime.strptime(row[0], "%Y-%m-%d")
                    previous = datetime.strptime(row[1], "%Y-%m-%d")
                    days_between = (current - previous).days
                    recovery_times.append(days_between)

            if recovery_times:
                recovery_data[muscle_group] = {
                    "avg_recovery": round(sum(recovery_times) / len(recovery_times), 1),
                    "min_recovery": min(recovery_times),
                    "max_recovery": max(recovery_times),
                    "total_sessions": len(recovery_times) + 1,
                }

        return recovery_data


def get_progressive_overload_rate():
    """Calculate the rate of strength progression per exercise"""
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT DISTINCT exercise FROM sets WHERE weight IS NOT NULL
            ORDER BY exercise
        """
        )

        exercises = [row[0] for row in cur.fetchall()]
        progression_rates = []

        for exercise in exercises:
            cur = conn.execute(
                """
                SELECT 
                    date(substr(date,1,10)) as workout_date,
                    MAX(COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0)) as estimated_1rm
                FROM sets
                WHERE exercise = ? AND weight IS NOT NULL
                GROUP BY workout_date
                ORDER BY workout_date
            """,
                (exercise,),
            )

            sessions = cur.fetchall()

            if len(sessions) >= 2:
                first_1rm = sessions[0][1]
                last_1rm = sessions[-1][1]
                first_date = datetime.strptime(sessions[0][0], "%Y-%m-%d")
                last_date = datetime.strptime(sessions[-1][0], "%Y-%m-%d")

                days_elapsed = (last_date - first_date).days
                if days_elapsed > 0 and first_1rm > 0:
                    # Calculate weekly progression rate
                    total_gain = last_1rm - first_1rm
                    weekly_rate = (total_gain / days_elapsed) * 7
                    percentage_gain = (total_gain / first_1rm) * 100

                    progression_rates.append(
                        {
                            "exercise": exercise,
                            "first_1rm": round(first_1rm, 1),
                            "last_1rm": round(last_1rm, 1),
                            "total_gain": round(total_gain, 1),
                            "weekly_rate": round(weekly_rate, 2),
                            "percentage_gain": round(percentage_gain, 1),
                            "days_tracked": days_elapsed,
                            "sessions": len(sessions),
                        }
                    )

        # Sort by percentage gain
        return sorted(
            progression_rates, key=lambda x: x["percentage_gain"], reverse=True
        )


def get_workout_duration_trends():
    """Analyze workout duration patterns over time"""
    with get_conn() as conn:
        cur = conn.execute(
            """
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
        """
        )

        duration_data = []
        for row in cur.fetchall():
            # Calculate efficiency metrics
            volume_per_minute = row[4] / row[1] if row[1] > 0 else 0
            sets_per_minute = row[3] / row[1] if row[1] > 0 else 0

            duration_data.append(
                {
                    "date": row[0],
                    "duration": round(row[1], 1),
                    "exercises": row[2],
                    "total_sets": row[3],
                    "volume": round(row[4], 1),
                    "volume_per_minute": round(volume_per_minute, 1),
                    "sets_per_minute": round(sets_per_minute, 2),
                    "efficiency_score": round(
                        volume_per_minute / 10, 1
                    ),  # Arbitrary scale
                }
            )

        return duration_data


def get_best_sets_analysis():
    """Find the single best set for each exercise"""
    with get_conn() as conn:
        cur = conn.execute(
            """
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
        """
        )

        return [
            {
                "exercise": row[0],
                "date": row[1],
                "weight": row[2],
                "reps": row[3],
                "volume": row[4],
                "estimated_1rm": round(row[5], 1),
            }
            for row in cur.fetchall()
        ]


def get_plateau_detection():
    """Detect potential plateaus in strength progression"""
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT DISTINCT exercise FROM sets 
            WHERE weight IS NOT NULL 
            ORDER BY exercise
        """
        )

        exercises = [row[0] for row in cur.fetchall()]
        plateau_analysis = []

        for exercise in exercises:
            cur = conn.execute(
                """
                SELECT 
                    date(substr(date,1,10)) as workout_date,
                    MAX(COALESCE(weight,0) * (1 + COALESCE(reps,0)/30.0)) as estimated_1rm
                FROM sets
                WHERE exercise = ? AND weight IS NOT NULL
                GROUP BY workout_date
                ORDER BY workout_date DESC
                LIMIT 5
            """,
                (exercise,),
            )

            recent_sessions = cur.fetchall()

            if len(recent_sessions) >= 3:
                # Check if last 3 sessions show no improvement
                last_3_1rms = [session[1] for session in recent_sessions[:3]]

                if len(set(last_3_1rms)) == 1:  # All the same
                    status = "Plateau"
                elif max(last_3_1rms) == last_3_1rms[0]:  # Best is most recent
                    status = "Progressing"
                elif last_3_1rms[0] < max(last_3_1rms):  # Recent performance declined
                    status = "Declining"
                else:
                    status = "Variable"

                plateau_analysis.append(
                    {
                        "exercise": exercise,
                        "status": status,
                        "current_1rm": round(last_3_1rms[0], 1),
                        "best_recent_1rm": round(max(last_3_1rms), 1),
                        "sessions_analyzed": len(recent_sessions),
                        "last_session": (
                            recent_sessions[0][0] if recent_sessions else None
                        ),
                    }
                )

        return sorted(plateau_analysis, key=lambda x: x["current_1rm"], reverse=True)


# ---------------------------------------------------------------------------
# New consolidated dashboard data builder for progression-focused refactor
# ---------------------------------------------------------------------------
from datetime import date as _date, timedelta as _timedelta
from collections import defaultdict

PUSH_KEYWORDS = [
    "Bench",
    "Press",
    "Dip",
    "Push-Up",
    "Push Up",
    "Dip",
    "Overhead Press",
    "Incline",
]
PULL_KEYWORDS = [
    "Row",
    "Pulldown",
    "Pull-Down",
    "Pullup",
    "Pull-Up",
    "Chin",
    "Deadlift",
    "Curl",
]
LEGS_KEYWORDS = [
    "Squat",
    "Lunge",
    "Leg Press",
    "Leg Extension",
    "Leg Curl",
    "Calf",
    "RDL",
    "Romanian",
]


def _classify_group(ex_name: str) -> str:
    n = ex_name.lower()
    for k in PUSH_KEYWORDS:
        if k.lower() in n:
            return "Push"
    for k in PULL_KEYWORDS:
        if k.lower() in n:
            return "Pull"
    for k in LEGS_KEYWORDS:
        if k.lower() in n:
            return "Legs"
    # Fallback guess by heuristic
    if any(x in n for x in ("press", "push")):
        return "Push"
    if any(x in n for x in ("row", "pull", "curl", "dead")):
        return "Pull"
    if any(x in n for x in ("squat", "leg", "lunge", "rdl", "romanian", "calf")):
        return "Legs"
    return "Other"


def build_dashboard_data(
    start: str | None, end: str | None, exercises: list[str] | None
):
    """Aggregate and return data required for the dashboard (weight only).

    Changes (Aug 2025):
    - Removed e1RM mode (weight only baseline)
    - Always compute daily max for ALL exercises in range (sparklines choose subset client-side)
    - Progressive overload defaults to top 5 exercises by volume (returned as top_exercises)
    - Removed recovery time aggregation (muscle_recovery)
    - Added ISO week start date (Monday) as 'week_start' to weekly structures for clearer labels
    """
    with get_conn() as conn:
        # Determine overall date range if not supplied
        cur = conn.execute(
            "SELECT MIN(date(substr(date,1,10))), MAX(date(substr(date,1,10))) FROM sets WHERE weight IS NOT NULL AND reps > 0"
        )
        row = cur.fetchone()
        if not row or row[0] is None:
            return {
                "filters": {
                    "start": start,
                    "end": end,
                    "exercises": [],
                    "data_start": None,
                    "data_end": None,
                },
                "exercises_daily_max": [],
                "top_exercises": [],
                "sessions": [],
                "weekly_ppl": [],
                "rep_bins_weekly": [],
                "rep_bins_total": {},
                "muscle_28d": [],
            }
        data_min, data_max = row
        if start is None:
            # Default: last 84 days (12w) or from data_min if more recent
            end_date = _date.fromisoformat(end if end else data_max)
            start_date_default = end_date - _timedelta(days=83)
            data_min_date = _date.fromisoformat(data_min)
            start = max(start_date_default, data_min_date).isoformat()
        if end is None:
            end = data_max

        # Fetch all relevant sets in date range
        cur = conn.execute(
            """
            SELECT date(substr(date,1,10)) as d, exercise, COALESCE(weight,0) as w, COALESCE(reps,0) as r
            FROM sets
            WHERE weight IS NOT NULL AND reps > 0
              AND date(substr(date,1,10)) BETWEEN ? AND ?
            ORDER BY d
            """,
            (start, end),
        )
        rows = cur.fetchall()
        if not rows:
            return {
                "filters": {
                    "start": start,
                    "end": end,
                    "exercises": [],
                    "data_start": data_min,
                    "data_end": data_max,
                },
                "exercises_daily_max": [],
                "top_exercises": [],
                "sessions": [],
                "weekly_ppl": [],
                "rep_bins_weekly": [],
                "rep_bins_total": {},
                "muscle_28d": [],
            }

        # Pre-aggregate total volume per exercise for selection purposes
        exercise_total_volume: dict[str, float] = defaultdict(float)
        for d, ex, w, r in rows:
            exercise_total_volume[ex] += w * r

        # Determine top 5 exercises for progressive overload chart (weight volume)
        top_exercises = [
            ex
            for ex, _ in sorted(
                exercise_total_volume.items(), key=lambda x: x[1], reverse=True
            )[:5]
        ]
        selected_set = set(
            top_exercises
        )  # used only for sessions & weekly splits volume classification

        # Daily maxima per exercise (for all exercises, not just top)
        per_day_ex: dict[tuple[str, str], dict] = {}
        per_day_volume: dict[tuple[str, str], float] = {}
        for d, ex, w, r in rows:
            key = (ex, d)
            rec = per_day_ex.setdefault(
                key, {"exercise": ex, "date": d, "max_weight": 0.0, "e1rm": 0.0}
            )
            if w > rec["max_weight"]:
                rec["max_weight"] = w
            e1rm_val = w * (1 + r / 30.0)
            if e1rm_val > rec["e1rm"]:
                rec["e1rm"] = e1rm_val
            if w and r:
                per_day_volume[key] = per_day_volume.get(key, 0.0) + w * r

        # Convert to list & compute PR flags per exercise (weight only)
        exercises_daily_max: list[dict] = []
        by_ex: dict[str, list[dict]] = defaultdict(list)
        for rec in per_day_ex.values():
            rec["e1rm"] = round(rec["e1rm"], 1)
            by_ex[rec["exercise"]].append(rec)
        for ex, lst in by_ex.items():
            lst.sort(key=lambda x: x["date"])
            best = -1.0
            metric_key = "max_weight"
            for item in lst:
                val = item[metric_key]
                if val > best:
                    item["is_pr"] = True
                    best = val
                else:
                    item["is_pr"] = False
            exercises_daily_max.extend(lst)

        # Build per-exercise progression metrics (delta weight)
        progression_list: list[dict] = []
        for ex, lst in by_ex.items():
            if len(lst) < 2:
                continue
            first = lst[0]["max_weight"]
            last = lst[-1]["max_weight"]
            delta = last - first
            days = (
                _date.fromisoformat(lst[-1]["date"])
                - _date.fromisoformat(lst[0]["date"])
            ).days or 1
            slope_per_day = delta / days
            progression_list.append(
                {
                    "exercise": ex,
                    "first": first,
                    "last": last,
                    "delta": round(delta, 2),
                    "slope_per_day": round(slope_per_day, 4),
                }
            )
        progression_list.sort(key=lambda x: x["delta"], reverse=True)

        # Daily volume list
        exercises_daily_volume = [
            {"exercise": ex, "date": d, "volume": round(vol, 1)}
            for (ex, d), vol in per_day_volume.items()
        ]
        exercises_daily_volume.sort(key=lambda x: (x["exercise"], x["date"]))

        # Session (day) total volume & PPL split
        sessions_map: dict[str, dict] = {}
        # rep bin weekly & weekly PPL structures
        weekly_ppl_acc = defaultdict(lambda: {"push": 0.0, "pull": 0.0, "legs": 0.0})
        rep_weekly_acc = defaultdict(
            lambda: {"bin_1_5": 0.0, "bin_6_12": 0.0, "bin_13_20": 0.0, "total": 0.0}
        )

        def iso_week(dstr: str) -> str:
            dt = _date.fromisoformat(dstr)
            iso = dt.isocalendar()  # (year, week, weekday)
            return f"{iso[0]}-W{iso[1]:02d}"

        for d, ex, w, r in rows:
            vol = w * r
            group = _classify_group(ex)
            sess = sessions_map.setdefault(
                d,
                {
                    "date": d,
                    "total_volume": 0.0,
                    "push_volume": 0.0,
                    "pull_volume": 0.0,
                    "legs_volume": 0.0,
                },
            )
            sess["total_volume"] += vol
            if group == "Push":
                sess["push_volume"] += vol
            elif group == "Pull":
                sess["pull_volume"] += vol
            elif group == "Legs":
                sess["legs_volume"] += vol

            week = iso_week(d)
            if group == "Push":
                weekly_ppl_acc[week]["push"] += vol
            elif group == "Pull":
                weekly_ppl_acc[week]["pull"] += vol
            elif group == "Legs":
                weekly_ppl_acc[week]["legs"] += vol

            # Rep bins
            if 1 <= r <= 5:
                bin_key = "bin_1_5"
            elif 6 <= r <= 12:
                bin_key = "bin_6_12"
            else:
                bin_key = "bin_13_20"
            rep_weekly_acc[week][bin_key] += vol
            rep_weekly_acc[week]["total"] += vol

        sessions = sorted(sessions_map.values(), key=lambda x: x["date"])
        # Rolling 4-week avg added client side (lighter)

        weekly_ppl = []

        def week_start_date(iso_week: str) -> str:
            # iso_week format YYYY-Www; compute Monday date
            y, w = iso_week.split("-W")
            y = int(y)
            w = int(w)
            # ISO week 1 may start in previous year; use datetime ISO helpers
            from datetime import date as _d

            # Find the Monday of the ISO week
            # From Python 3.8+, date.fromisocalendar exists
            return _d.fromisocalendar(y, w, 1).isoformat()

        for wk, vals in sorted(weekly_ppl_acc.items()):
            weekly_ppl.append(
                {
                    "iso_week": wk,
                    "week_start": week_start_date(wk),
                    "push": round(vals["push"], 1),
                    "pull": round(vals["pull"], 1),
                    "legs": round(vals["legs"], 1),
                }
            )

        rep_bins_weekly = []
        total_bins = {"bin_1_5": 0.0, "bin_6_12": 0.0, "bin_13_20": 0.0, "total": 0.0}
        for wk, vals in sorted(rep_weekly_acc.items()):
            rep_bins_weekly.append(
                {
                    "iso_week": wk,
                    "week_start": week_start_date(wk),
                    "bin_1_5": round(vals["bin_1_5"], 1),
                    "bin_6_12": round(vals["bin_6_12"], 1),
                    "bin_13_20": round(vals["bin_13_20"], 1),
                    "total": round(vals["total"], 1),
                }
            )
            for k in total_bins:
                total_bins[k] += vals[k]
        rep_bins_total = {k: round(v, 1) for k, v in total_bins.items()}

        # Muscle recovery (P/P/L)
        def recovery(group_key: str):
            # gather days where group volume >0
            ds = [s["date"] for s in sessions if s[f"{group_key.lower()}_volume"] > 0]
            ds_sorted = sorted(set(ds))
            if len(ds_sorted) < 2:
                return None
            intervals = []
            prev = _date.fromisoformat(ds_sorted[0])
            for dstr in ds_sorted[1:]:
                curd = _date.fromisoformat(dstr)
                intervals.append((curd - prev).days)
                prev = curd
            return {
                "muscle": group_key,
                "mean_days": round(sum(intervals) / len(intervals), 1),
                "min_days": min(intervals),
                "max_days": max(intervals),
                "n_intervals": len(intervals),
            }

        # (Recovery removed from payload)

        # Muscle 28d volume (P/P/L proxy)
        end_dt = _date.fromisoformat(end)
        start_28 = end_dt - _timedelta(days=27)
        volume_28 = {"Push": 0.0, "Pull": 0.0, "Legs": 0.0}
        for d, ex, w, r in rows:
            dt = _date.fromisoformat(d)
            if dt < start_28 or dt > end_dt:
                continue
            vol = w * r
            group = _classify_group(ex)
            if group in volume_28:
                volume_28[group] += vol
        muscle_28d = [{"group": g, "volume": round(v, 1)} for g, v in volume_28.items()]

        return {
            "filters": {
                "start": start,
                "end": end,
                "exercises": top_exercises,
                "data_start": data_min,
                "data_end": data_max,
            },
            "exercises_daily_max": exercises_daily_max,
            "exercises_daily_volume": exercises_daily_volume,
            "exercise_progression": progression_list,
            "top_exercises": top_exercises,
            "sessions": sessions,
            "weekly_ppl": weekly_ppl,
            "rep_bins_weekly": rep_bins_weekly,
            "rep_bins_total": rep_bins_total,
            "muscle_28d": muscle_28d,
        }

```

# app\static\main.charts.js

```js
// Chart rendering functions extracted from main.js (reuse global window.charts)
window.charts = window.charts || {};
function _clearLoading(el){ if(!el) return; const pulse=el.querySelector('.animate-pulse'); if(pulse) pulse.remove(); }
function getChart(id){
  const el=document.getElementById(id);
  if(!el) return null;
  _clearLoading(el);
  if(!el.dataset.fixedHeight){
    if((!el.style.height || el.clientHeight<120)){
      el.style.height = (id==='progressiveOverloadChart'?'300px':'230px');
    }
  }
  if(!window.charts[id]) window.charts[id]=echarts.init(el);
  setTimeout(()=>{ try { window.charts[id].resize(); } catch(_){} }, 40);
  return window.charts[id];
}

function baseTimeAxis(){ return { type:'time', axisLine:{lineStyle:{color:'#3f3f46'}}, axisLabel:{color:'#a1a1aa', formatter: v=> new Date(v).toISOString().slice(5,10)}, splitLine:{show:false} }; }
function baseValueAxis(name){ return { type:'value', name, nameTextStyle:{color:'#a1a1aa'}, axisLine:{lineStyle:{color:'#3f3f46'}}, axisLabel:{color:'#a1a1aa'}, splitLine:{lineStyle:{color:'#27272a'}}, scale:true }; }

function renderSparklines(){
  const container=document.getElementById('sparklineContainer');
    if(!state.data || !state.data.exercises_daily_max || !state.data.exercises_daily_max.length){
      if(container) container.innerHTML='<div class="text-sm text-zinc-500 italic">No exercise data in range</div>';
      return;
    }
  console.log('[dashboard] renderSparklines count', state.data.exercises_daily_max.length);
    container.innerHTML='';
  const metricKey= 'max_weight';
    const byEx={};
    state.data.exercises_daily_max.forEach(r=>{ (byEx[r.exercise] ||= []).push(r); });
    Object.entries(byEx).forEach(([ex, arr], idx)=>{
      const card=document.createElement('div');
      card.className='bg-zinc-900 rounded-2xl ring-1 ring-zinc-800 shadow-sm p-4 flex flex-col';
      const last = arr[arr.length-1];
      card.innerHTML=`<div class='flex items-center justify-between mb-2'><span class='text-sm font-medium text-zinc-300 truncate'>${ex}</span><span class='text-xs text-zinc-500'>${fmt1(last[metricKey])}</span></div><div class='flex-1' id='spark_${idx}' style='height:60px;'></div>`;
      container.appendChild(card);
      const chart=echarts.init(card.querySelector('#spark_'+idx));
      const prPoints = arr.filter(a=>a.is_pr).map(a=> [a.date, a[metricKey]]);
      chart.setOption({ animation:false, grid:{left:2,right:2,top:0,bottom:0}, xAxis:{type:'time',show:false}, yAxis:{type:'value',show:false}, tooltip:{trigger:'axis', formatter: params=>{
        const p=params[0]; return `${ex}<br>${p.axisValueLabel}: ${fmt1(p.data[1])}`;}}, series:[{type:'line',data:arr.map(a=>[a.date,a[metricKey]]), showSymbol:false, smooth:true, lineStyle:{width:1.2,color:SERIES_COLORS[idx%SERIES_COLORS.length]}, areaStyle:{color:SERIES_COLORS[idx%SERIES_COLORS.length]+'33'}},{type:'scatter', data: prPoints.slice(-8), symbolSize:6, itemStyle:{color:'#fde047'}}] });
    });
}

function renderProgressiveOverload(){
  if(!state.data || !state.data.exercises_daily_max || !state.data.exercises_daily_max.length){ const el=document.getElementById('progressiveOverloadChart'); if(el) el.innerHTML='<div class="flex items-center justify-center h-full text-sm text-zinc-500 italic">No data</div>'; return; }
  const metricKey= 'max_weight';
  const byEx={}; state.data.exercises_daily_max.forEach(r=>{ (byEx[r.exercise] ||= []).push(r); });
  // Sort exercises by number of datapoints (desc) then alphabetically
  const names = Object.keys(byEx).sort((a,b)=> byEx[b].length - byEx[a].length || a.localeCompare(b));
  // Helper to derive base name (strip cable/machine words)
  const baseName = (n)=> n.replace(/\b(cable|machine)\b/ig,'').replace(/\s+/g,' ').replace(/\(.*?\)/g,'').trim() || n;
  // Color map per base name so cable/machine variants share color
  const baseColorMap={}; let colorIdx=0;
  names.forEach(n=>{ const b=baseName(n); if(!baseColorMap[b]){ baseColorMap[b]= SERIES_COLORS[colorIdx%SERIES_COLORS.length]; colorIdx++; } });
  const colorMap={}; names.forEach(n=>{ colorMap[n]= baseColorMap[baseName(n)]; });
  // Preselect top 5 (most datapoints)
  const activeSet = new Set(names.slice(0,2));
  const legendRoot=document.getElementById('poLegend'); if(legendRoot) legendRoot.innerHTML='';
  names.forEach(name=>{
    const pill=document.createElement('button');
    pill.className='px-2 py-0.5 rounded-full border text-xs flex items-center gap-1 transition-colors';
    pill.dataset.fullName = name;
    const lname=name.toLowerCase(); const isCable=lname.includes('cable'); const isMachine=lname.includes('machine');
    const labelBase = baseName(name);
    pill.textContent = labelBase + (isCable?' (Cable)': isMachine? ' (Machine)':'' );
    const applyStyles = ()=>{
      const on = activeSet.has(name);
      pill.style.borderColor = on? colorMap[name]: '#3f3f46';
      pill.style.background = on? colorMap[name]+'22':'#18181b';
      pill.style.color = on? '#e4e4e7':'#a1a1aa';
      pill.title = (on? 'Hide ':'Show ')+ name;
    };
    applyStyles();
    pill.onclick = ()=>{ if(activeSet.has(name)) activeSet.delete(name); else activeSet.add(name); applyStyles(); renderChart(); };
    legendRoot.appendChild(pill);
  });
  function buildSeries(){
    const series=[];
    names.forEach(name=>{
      if(!activeSet.has(name)) return;
      const arr = byEx[name].slice().sort((a,b)=> a.date.localeCompare(b.date));
      const lname = name.toLowerCase();
      const isCable = lname.includes('cable');
      const isMachine = lname.includes('machine');
      if(arr.length === 1){
        // Single point: use scatter so it actually shows
        series.push({ name, type:'scatter', data: arr.map(a=> [a.date, a[metricKey]]), symbol:'circle', symbolSize:8, itemStyle:{color: colorMap[name]}, emphasis:{focus:'none'} });
        return; // no MA for single-point series
      }
      series.push({ name, type:'line', showSymbol: arr.length<=3, smooth:true, data: arr.map(a=> [a.date, a[metricKey]]), lineStyle:{width:2, color: colorMap[name], type: isCable? 'dashed':'solid'}, areaStyle:{color: colorMap[name] + (isMachine? '35':'25')}, symbol: isCable? 'circle':'none', symbolSize: isCable? 5:4 });
      if(arr.length>2){
        const ma=[]; const vals=[]; arr.forEach(a=>{ vals.push(a[metricKey]); if(vals.length>7) vals.shift(); ma.push([a.date, vals.reduce((s,v)=>s+v,0)/vals.length]); });
        series.push({ name: name+' 7MA', type:'line', showSymbol:false, smooth:true, data: ma, lineStyle:{width:1, type:'dashed', color: colorMap[name]}, emphasis:{disabled:true}, tooltip:{show:false} });
      }
    });
    return series;
  }
  function renderChart(){
    const chart=getChart('progressiveOverloadChart');
    chart.setOption({animationDuration:250, grid:{left:42,right:12,top:10,bottom:55}, legend:{show:false}, dataZoom:[{type:'inside'},{type:'slider',height:16,bottom:18}], xAxis: baseTimeAxis(), yAxis: baseValueAxis('Max Weight (kg)'), tooltip:{trigger:'axis', valueFormatter:v=>fmt1(v)}, series: buildSeries().map(s=> ({...s, emphasis:{focus:'none'}})) }, true);
    chart.off('dataZoom'); chart.on('dataZoom', ()=> updateSlopes(chart, byEx, metricKey));
    updateSlopes(chart, byEx, metricKey);
  }
  document.getElementById('resetOverloadZoom').onclick=()=>{ const chart=getChart('progressiveOverloadChart'); chart.dispatchAction({type:'dataZoom', start:0, end:100}); };
  renderChart();
}

function updateSlopes(chart, byEx, metricKey){
  const opt=chart.getOption(); const [min,max]= opt.xAxis[0].range || [opt.xAxis[0].min, opt.xAxis[0].max];
  const start = min? new Date(min): null; const end = max? new Date(max): null;
  const container=document.getElementById('overloadSlopes'); if(container) container.innerHTML='';
  Object.entries(byEx).forEach(([ex, arr], idx)=>{
  const pts= arr.filter(a=> (!start || window.parseISO(a.date)>=start) && (!end || window.parseISO(a.date)<=end));
  if(pts.length<2) return;
  const t0 = window.parseISO(pts[0].date).getTime();
  const xs = pts.map(p=> (window.parseISO(p.date).getTime()-t0)/ (86400000*7));
    const ys = pts.map(p=> p[metricKey]);
    const mean = xs.reduce((s,v)=>s+v,0)/xs.length; const meanY= ys.reduce((s,v)=>s+v,0)/ys.length;
    let num=0, den=0; for(let i=0;i<xs.length;i++){ const dx=xs[i]-mean; num+= dx*(ys[i]-meanY); den+= dx*dx; }
    const slope = den? num/den:0; // units per week
    const pill=document.createElement('span'); pill.className='px-2 py-1 rounded bg-zinc-800 text-zinc-300'; pill.textContent=`${ex}: ${slope>=0?'+':''}${fmt1(slope)} /wk`; container.appendChild(pill);
  });
}

function renderVolumeTrend(){
  if(!state.data || !state.data.sessions || !state.data.sessions.length){ const el=document.getElementById('volumeTrendChart'); if(el) el.innerHTML='<div class="flex items-center justify-center h-full text-sm text-zinc-500 italic">No sessions</div>'; return; }
  const sessions = state.data.sessions.slice(); sessions.sort((a,b)=> a.date.localeCompare(b.date));
  const seriesBar = sessions.map(s=> [s.date, s.total_volume]);
  const rolling=[]; for(let i=0;i<sessions.length;i++){ const di=window.parseISO(sessions[i].date); const since = di.getTime()-27*86400000; const subset=sessions.filter(s=> window.parseISO(s.date).getTime()>=since && window.parseISO(s.date)<=di); const avg=subset.reduce((s,v)=>s+v.total_volume,0)/subset.length; rolling.push([sessions[i].date, avg]); }
  const chart=getChart('volumeTrendChart');
  chart.setOption({ grid:{left:50,right:16,top:20,bottom:55}, xAxis: baseTimeAxis(), yAxis: baseValueAxis('Volume (kg)'), dataZoom:[{type:'inside'},{type:'slider',height:18,bottom:20}], tooltip:{trigger:'axis'}, series:[{type:'bar', name:'Session Volume', data:seriesBar, itemStyle:{color:COLORS.primary}, emphasis:{focus:'none'}},{type:'line', name:'4W Avg', data:rolling, smooth:true, showSymbol:false, lineStyle:{width:2,color:COLORS.secondary}, emphasis:{focus:'none'}}], markLine:{symbol:'none', silent:true, lineStyle:{color:'#3f3f46', width:1}, data: sessions.map(s=> ({xAxis:s.date}))} });
}

function renderWeeklyPPL(){
  if(!state.data || !state.data.weekly_ppl || !state.data.weekly_ppl.length){ const el=document.getElementById('weeklyPPLChart'); if(el) el.innerHTML='<div class="flex items-center justify-center h-full text-sm text-zinc-500 italic">No weekly data</div>'; return; } const mode = document.getElementById('pplModeToggle').dataset.mode; 
  const weeks = state.data.weekly_ppl.map(w=> w.week_start);
  const push = state.data.weekly_ppl.map(w=> w.push);
  const pull = state.data.weekly_ppl.map(w=> w.pull);
  const legs = state.data.weekly_ppl.map(w=> w.legs);
  let pushD=push, pullD=pull, legsD=legs; let yAxis={type:'value', name: mode==='absolute'? 'Weekly Volume (kg)':'% Volume', nameTextStyle:{color:'#a1a1aa'}, axisLine:{lineStyle:{color:'#3f3f46'}}, axisLabel:{color:'#a1a1aa'}};
  if(mode!=='absolute'){
    pushD=[]; pullD=[]; legsD=[];
    for(let i=0;i<weeks.length;i++){ const tot=push[i]+pull[i]+legs[i]; if(tot===0){ pushD.push(0); pullD.push(0); legsD.push(0);} else { pushD.push(push[i]/tot*100); pullD.push(pull[i]/tot*100); legsD.push(legs[i]/tot*100);} }
    yAxis.max=100;
  }
  const chart=getChart('weeklyPPLChart');
  // Legend orientation: horizontal for absolute, vertical (right side) for percent to better use space
  const legendOpt = mode==='absolute'
    ? {top:0, left:0, orient:'horizontal', textStyle:{color:'#d4d4d8'}, padding:0}
    : {top:'middle', right:0, orient:'vertical', textStyle:{color:'#d4d4d8'}, itemGap:8, padding:0};
  chart.setOption({
    grid:{left:50,right: mode==='absolute'? 16:90, top: mode==='absolute'? 70:28,bottom:40}, // generous first pass for absolute mode
    legend: legendOpt,
    tooltip:{trigger:'axis', axisPointer:{type:'shadow'}},
    xAxis:{type:'category', data:weeks, axisLine:{lineStyle:{color:'#3f3f46'}}, axisLabel:{color:'#a1a1aa'}},
    yAxis,
    series:[
      {name:'Push', type:'bar', stack:'ppl', data:pushD, itemStyle:{color:COLORS.primary}, emphasis:{focus:'none'}},
      {name:'Pull', type:'bar', stack:'ppl', data:pullD, itemStyle:{color:COLORS.tertiary}, emphasis:{focus:'none'}},
      {name:'Legs', type:'bar', stack:'ppl', data:legsD, itemStyle:{color:COLORS.quaternary}, emphasis:{focus:'none'}}
    ]
  }, true); // notMerge=true ensures legend layout updates when toggling
  setTimeout(()=>{
    try {
      const dom = chart.getDom();
      const legend = dom.querySelector('.echarts-legend');
      if(!legend) return;
      if(mode==='absolute'){
        const h = legend.getBoundingClientRect().height;
        const newTop = Math.min(Math.max(h + 24, 60), 140);
        const opt = chart.getOption(); if(opt.grid[0].top !== newTop){ chart.setOption({grid:{left:50,right:16,top:newTop,bottom:40}}); chart.resize(); }
      } else {
        // percent mode: ensure right padding fits legend width
        const w = legend.getBoundingClientRect().width;
        const neededRight = Math.min(Math.max(w + 16, 70), 160);
        const opt = chart.getOption(); if(opt.grid[0].right !== neededRight){ chart.setOption({grid:{left:50,right:neededRight,top:28,bottom:40}}); chart.resize(); }
      }
    } catch(e){ console.warn('weeklyPPL legend adjust fail', e); }
  }, 50);
}

function renderMuscleBalance(){
  if(!state.data || !state.data.muscle_28d){ const el=document.getElementById('muscleBalanceChart'); if(el) el.innerHTML='<div class="flex items-center justify-center h-full text-sm text-zinc-500 italic">No data</div>'; return; } const data = state.data.muscle_28d;
  const names=data.map(d=> d.group); const vals=data.map(d=> d.volume);
  getChart('muscleBalanceChart').setOption({ grid:{left:110,right:30,top:10,bottom:25}, xAxis:{type:'value', axisLine:{lineStyle:{color:'#3f3f46'}}, axisLabel:{color:'#a1a1aa'}, splitLine:{lineStyle:{color:'#27272a'}}}, yAxis:{type:'category', data:names, axisLine:{show:false}, axisLabel:{color:'#d4d4d8'}}, tooltip:{trigger:'item', formatter: p=> `${p.name}: ${fmtInt(p.value)} kg`}, series:[{type:'bar', data:vals, barWidth:'40%', itemStyle:{color:(p)=> SERIES_COLORS[p.dataIndex]}, emphasis:{focus:'none'}, label:{show:true, position:'right', color:'#a1a1aa', formatter: p=> fmtInt(p.value)}}] });
}

function renderRepDistribution(){
  const mode = document.getElementById('repModeToggle').dataset.mode; 
  if(!state.data) return;
  if(mode==='weekly'){
    const weeks = state.data.rep_bins_weekly.map(r=> r.week_start);
    const b1 = state.data.rep_bins_weekly.map(r=> r.bin_1_5);
    const b2 = state.data.rep_bins_weekly.map(r=> r.bin_6_12);
    const b3 = state.data.rep_bins_weekly.map(r=> r.bin_13_20);
    const chart = getChart('repDistributionChart');
    chart.setOption({ grid:{left:55,right:16,top:70,bottom:40}, legend:{top:0,textStyle:{color:'#d4d4d8'}, padding:0}, tooltip:{trigger:'axis', axisPointer:{type:'shadow'}}, xAxis:{type:'category', data:weeks, axisLabel:{color:'#a1a1aa'}, axisLine:{lineStyle:{color:'#3f3f46'}}}, yAxis:{type:'value', name:'Volume (kg)', nameTextStyle:{color:'#a1a1aa'}, axisLabel:{color:'#a1a1aa'}, splitLine:{lineStyle:{color:'#27272a'}}}, series:[{name:'1–5', type:'bar', stack:'reps', data:b1, itemStyle:{color:COLORS.secondary}, emphasis:{focus:'none'}},{name:'6–12', type:'bar', stack:'reps', data:b2, itemStyle:{color:COLORS.primary}, emphasis:{focus:'none'}},{name:'13–20', type:'bar', stack:'reps', data:b3, itemStyle:{color:COLORS.tertiary}, emphasis:{focus:'none'}}] });
    setTimeout(()=>{
      try {
        const legend = chart.getDom().querySelector('.echarts-legend');
        if(!legend) return; const h=legend.getBoundingClientRect().height; const newTop = Math.min(Math.max(h+24,60),140);
        const opt=chart.getOption(); if(opt.grid[0].top!==newTop){ chart.setOption({grid:{left:55,right:16,top:newTop,bottom:40}}); chart.resize(); }
      } catch(e){ console.warn('repDistribution legend adjust failed', e); }
    },50);
  } else {
    const t = state.data.rep_bins_total; const total=t.total||1; const bars=[{name:'1–5', val:t.bin_1_5},{name:'6–12', val:t.bin_6_12},{name:'13–20', val:t.bin_13_20}];
    getChart('repDistributionChart').setOption({ grid:{left:110,right:30,top:10,bottom:25}, xAxis:{type:'value', axisLabel:{color:'#a1a1aa'}, axisLine:{lineStyle:{color:'#3f3f46'}}, splitLine:{lineStyle:{color:'#27272a'}}}, yAxis:{type:'category', data:bars.map(b=> b.name), axisLabel:{color:'#d4d4d8'}}, tooltip:{trigger:'item', formatter: p=>{ const v=bars[p.dataIndex].val; return `${p.name}: ${fmtInt(v)} kg (${(v/total*100).toFixed(1)}%)`; }}, series:[{type:'bar', data:bars.map(b=> b.val), itemStyle:{color:(p)=> [COLORS.secondary,COLORS.primary,COLORS.tertiary][p.dataIndex]}, emphasis:{focus:'none'}, barWidth:'45%', label:{show:true, position:'right', formatter: p=> (bars[p.dataIndex].val/total*100).toFixed(1)+'%', color:'#a1a1aa'}}] });
  }
}

// Exported single function to render all echarts-based charts
// Exercise Volume (stacked / grouped) moved from main.js so all ECharts live together
function renderExerciseVolume(){
  if(!window.state || !window.state.data){ return; }
  const volumeArr = state.data.exercises_daily_volume;
  if(!Array.isArray(volumeArr)){
    const el=document.getElementById('exerciseVolumeChart');
    if(el) el.innerHTML='<div class="flex items-center justify-center h-full text-xs text-zinc-500 italic">No volume field</div>';
    console.warn('[exerciseVolume] missing exercises_daily_volume in payload keys=', Object.keys(state.data||{}));
    return;
  }
  if(volumeArr.length===0){
    const el=document.getElementById('exerciseVolumeChart');
    if(el) el.innerHTML='<div class="flex items-center justify-center h-full text-xs text-zinc-500 italic">No volume entries</div>';
    console.info('[exerciseVolume] empty array received');
    return;
  }
  const toggleEl = document.getElementById('volumeModeToggle');
  const mode = toggleEl? toggleEl.dataset.mode : 'grouped'; // stacked|grouped
  const volsByEx={};
  volumeArr.forEach(r=> { volsByEx[r.exercise] = (volsByEx[r.exercise]||0)+ r.volume; });
  const top = Object.entries(volsByEx).sort((a,b)=> b[1]-a[1]).slice(0,6).map(e=> e[0]);
  const filtered = volumeArr.filter(r=> top.includes(r.exercise));
  const dates = Array.from(new Set(filtered.map(r=> r.date))).sort();
  const series = top.map((ex,i)=>{
    const data = dates.map(d=> { const rec = filtered.find(r=> r.exercise===ex && r.date===d); return rec? rec.volume:0; });
    return { name: ex.split('(')[0].trim(), type:'bar', stack: mode==='stacked'? 'vol': undefined, data, itemStyle:{color: SERIES_COLORS[i%SERIES_COLORS.length]}, emphasis:{focus:'none'} };
  });
  const chart=getChart('exerciseVolumeChart');
  // First pass: render with generous top padding; then measure legend height and adjust grid
  chart.setOption({ grid:{left:50,right:12,top:84,bottom:55}, legend:{top:0,textStyle:{color:'#d4d4d8'}, padding:0}, tooltip:{trigger:'axis', axisPointer:{type:'shadow'}}, xAxis:{type:'category', data:dates, axisLabel:{color:'#a1a1aa', formatter:v=> v.slice(5)}, axisLine:{lineStyle:{color:'#3f3f46'}}}, yAxis:{type:'value', name:'Volume (kg)', nameTextStyle:{color:'#a1a1aa'}, axisLabel:{color:'#a1a1aa'}, splitLine:{lineStyle:{color:'#27272a'}}}, series });
  setTimeout(()=>{ // allow DOM layout
    try {
      const legendEl = chart.getDom().querySelector('.echarts-legend');
      if(legendEl){
        const h = legendEl.getBoundingClientRect().height; // actual legend height
  const newTop = Math.min(Math.max(h + 24, 60), 140); // extra spacing below legend
        const opt = chart.getOption();
        if(opt.grid[0].top !== newTop){
          chart.setOption({grid:{left:50,right:12,top:newTop,bottom:55}});
          chart.resize();
        }
      }
    } catch(e){ console.warn('legend measure failed', e); }
  }, 50);
}

function renderAll(){
  renderSparklines();
  renderProgressiveOverload();
  renderVolumeTrend();
  renderExerciseVolume();
  renderWeeklyPPL();
  renderMuscleBalance();
  renderRepDistribution();
}

// Expose
window.getChart = getChart;
window.renderAllCharts = renderAll;

// Attach volume mode toggle handler (was missing so chart never appeared)
document.addEventListener('DOMContentLoaded', ()=>{
  // Volume mode toggle
  const volBtn = document.getElementById('volumeModeToggle');
  if(volBtn && !volBtn.dataset._bound){
    volBtn.dataset._bound = '1';
    volBtn.textContent = volBtn.dataset.mode === 'stacked' ? 'Stacked' : 'Grouped';
    volBtn.addEventListener('click', function(){
      this.dataset.mode = this.dataset.mode === 'grouped' ? 'stacked' : 'grouped';
      this.textContent = this.dataset.mode === 'stacked' ? 'Stacked' : 'Grouped';
      renderExerciseVolume();
    });
  }
  // PPL mode toggle
  const pplBtn = document.getElementById('pplModeToggle');
  if(pplBtn && !pplBtn.dataset._bound){
    pplBtn.dataset._bound='1';
    pplBtn.addEventListener('click', function(){
      this.dataset.mode = this.dataset.mode==='absolute' ? 'percent':'absolute';
      this.textContent= this.dataset.mode==='absolute'?'Absolute':'Percent';
      renderWeeklyPPL();
    });
  }
  // Rep distribution mode toggle
  const repBtn = document.getElementById('repModeToggle');
  if(repBtn && !repBtn.dataset._bound){
    repBtn.dataset._bound='1';
    repBtn.addEventListener('click', function(){
      this.dataset.mode = this.dataset.mode==='weekly' ? 'summary':'weekly';
      this.textContent= this.dataset.mode==='weekly'?'Weekly':'Summary';
      renderRepDistribution();
    });
  }
});

```

# app\static\main.core.js

```js
// Core shared state, helpers, and data fetching for dashboard
// (extracted from original main.js)

// ------------------------------ State ----------------------------------
const state = {
  start: null, // date filters removed (backend auto range) but keep keys for cache key stability
  end: null,
  exercises: [], // applies ONLY to sparklines
  data: null,
  cache: new Map()
};

// Expose reference for modules that (incorrectly) check window.state
if(!window.state) window.state = state;

const COLORS = {
  primary: '#6366F1',
  secondary: '#EC4899',
  tertiary: '#10B981',
  quaternary: '#F59E0B',
  quinary: '#8B5CF6'
};
const SERIES_COLORS = [COLORS.primary, COLORS.secondary, COLORS.tertiary, COLORS.quaternary, COLORS.quinary];

function limitLegendSelection(series, maxVisible){
  const sel={};
  let count=0;
  series.forEach(s=>{
    if(!s.name.endsWith(' 7MA') && count<maxVisible){ sel[s.name]=true; count++; } else { sel[s.name]=false; }
  });
  return sel;
}

// Helpers (attach to window to avoid accidental redeclaration in other modules)
window.fetchJSON = window.fetchJSON || function(url) { return fetch(url).then(r => { if(!r.ok) throw new Error(r.statusText); return r.json(); }); };
window.fmtInt = window.fmtInt || function(x){ return x == null ? '-' : x.toLocaleString(); };
window.fmt1 = window.fmt1 || function(x){ return x==null?'-': (Math.round(x*10)/10).toString(); };
window.parseISO = window.parseISO || function(d){ return new Date(d+ (d.length===10?'T00:00:00Z':'')); };

// --------------------------- Filters UI --------------------------------
function initFilters(){
  // Only metric placeholder remains (weight-only)
  const metricWrap=document.getElementById('metricToggle');
  if(metricWrap) metricWrap.innerHTML='<span class="text-xs text-zinc-500">Weight mode</span>';
}

function updateMetricButtons(){}


// ------------------------- Data Fetch & Cache ---------------------------
async function fetchDashboard(){
  const key = JSON.stringify({start:state.start,end:state.end});
  if(state.cache.has(key)) return state.cache.get(key);
  const params = new URLSearchParams();
  if(state.start) params.set('start', state.start);
  if(state.end) params.set('end', state.end);
  // exercises & metric no longer passed to backend
  const data = await fetchJSON('/api/dashboard?'+params.toString());
  state.cache.set(key,data); return data;
}

async function refreshData(){
  try {
  console.log('[dashboard] refreshData start', {exercises:state.exercises});
  const loadingTargets=['sparklineContainer','progressiveOverloadChart','volumeTrendChart','exerciseVolumeChart','weeklyPPLChart','muscleBalanceChart','repDistributionChart','recoveryChart','calendarChart'];
    loadingTargets.forEach(id=>{ const el=document.getElementById(id); if(el) el.innerHTML='<div class="flex items-center justify-center h-full text-sm text-zinc-500 animate-pulse">Loading...</div>'; });
    const data = await fetchDashboard();
  window.__dashboardDebug = { phase:'afterFetch', fetchedAt: Date.now(), filters: data?.filters, params:{start:state.start,end:state.end,exercises:[...state.exercises]}, keys: data? Object.keys(data):[] };
  console.log('[dashboard] data fetched', window.__dashboardDebug);
  // No date preset logic
  state.data=data;
    const lastIngestedEl = document.getElementById('lastIngested');
    if(lastIngestedEl) lastIngestedEl.textContent = data.filters.end || '-';
    if(!state.exercises.length){
      // Preselect by most improvement (delta) (kept internally; UI removed)
      const prog = (data.exercise_progression || []).map(p=> p.exercise);
      state.exercises = prog.slice(0,12);
      if(state.exercises.length===0) state.exercises = data.filters.exercises || (data.top_exercises || []);
    }
  // Call whichever aggregate render function is available (charts split across modules)
  try { (window.renderAllCharts || window.renderAll || (()=>{}))(); } catch(e){ console.error('Error during renderAll:', e); }
  // Plotly calendar (and other plotly sections) lives in main.plotly.js
  if(window.loadTrainingCalendar){
    try { window.loadTrainingCalendar(); } catch(e){ console.error('calendar load failed', e); }
  }
  window.__dashboardDebug.phase='renderComplete';
  console.log('[dashboard] render complete');
  } catch(e){
    console.error(e);
    const msg='<div class="flex items-center justify-center h-full text-sm text-rose-400">Error loading data</div>';
    ['sparklineContainer','progressiveOverloadChart','volumeTrendChart','weeklyPPLChart','muscleBalanceChart','repDistributionChart','recoveryChart','calendarChart'].forEach(id=>{ const el=document.getElementById(id); if(el) el.innerHTML=msg; });
  window.__dashboardDebug = { phase:'error', error: e?.message || String(e) };
  }
}

// Range helper removed

function unique(arr){ return [...new Set(arr)]; }

// Load training streak and metadata helpers
async function loadTrainingStreak() {
  try {
    const data = await fetchJSON('/api/training-streak');
    
    const current = document.getElementById('currentStreak');
    const longest = document.getElementById('longestStreak');
    const total = document.getElementById('totalWorkouts');
    if(current) current.textContent = data.current_streak || '0';
    if(longest) longest.textContent = data.longest_streak || '0';
    if(total) total.textContent = data.total_workout_days || '0';
    
    // Add some animation
    const elements = ['currentStreak', 'longestStreak', 'totalWorkouts'];
    elements.forEach((id, index) => {
      const element = document.getElementById(id);
      if(!element) return;
      setTimeout(() => {
        element.style.transform = 'scale(1.1)';
        setTimeout(() => {
          element.style.transform = 'scale(1)';
        }, 200);
      }, index * 100);
    });

  } catch (error) {
    console.error('Error loading training streak:', error);
  }
}

async function loadLastIngested() {
  try {
    const health = await fetchJSON('/health');
    const lastIngested = health.last_ingested_at;
    const element = document.getElementById('lastIngested');
    
    if (element) {
      if (lastIngested) {
        const date = new Date(lastIngested);
        element.textContent = `Last data update: ${date.toLocaleString()}`;
        element.style.color = '#27ae60';
      } else {
        element.textContent = 'No data ingested yet';
        element.style.color = '#e74c3c';
      }
    }

  } catch (error) {
    console.error('Error loading last ingested:', error);
    const element = document.getElementById('lastIngested');
    if(element) element.textContent = 'Status unknown';
  }
}

// Add some global styles for better UX (only once)
if(!document.getElementById('dashboard-shared-style')){
  const style = document.createElement('style');
  style.id='dashboard-shared-style';
  style.textContent = `
  .streak-item {
    transition: all 0.3s ease;
  }
  .streak-item:hover {
    transform: translateY(-5px);
  }
  table tbody tr {
    transition: background-color 0.2s ease;
  }
  .chart-container {
    transition: box-shadow 0.3s ease;
  }
  .chart-container:hover {
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
  }
`;
  document.head.appendChild(style);
}

// Bootstrap
// Bootstrap is now centralized in main.js; expose helpers only if needed.
window.loadLastIngested = loadLastIngested;
window.loadTrainingStreak = loadTrainingStreak;
// Expose refresh so bootstrap in main.js can invoke
window.refreshData = window.refreshData || refreshData;
window.fetchDashboard = window.fetchDashboard || fetchDashboard;

```

# app\static\main.js

```js
// Slim main.js: only bootstrap + workout modal + share functionality (charts in core/charts files)
// NOTE: File replaced to remove duplicated logic.

(function(){
  function bootstrapDashboard(){
    if(window.__dashboardBooted) return; window.__dashboardBooted = true;
    if(typeof window.refreshData === 'function') window.refreshData(); else setTimeout(()=> window.refreshData && window.refreshData(), 150);
  }
  if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bootstrapDashboard); else bootstrapDashboard();

  window.addEventListener('resize', ()=>{ if(window.charts) Object.values(window.charts).forEach(c=> { try { c.resize(); } catch(_){ } }); });

  function fmt1(x){ return window.fmt1? window.fmt1(x): (x==null?'-': (Math.round(x*10)/10).toString()); }
  function fmtInt(x){ return window.fmtInt? window.fmtInt(x): (x==null?'-': x.toLocaleString()); }

  async function showWorkoutDetail(workoutDate) {
    try {
      const modal = document.getElementById('workoutModal');
      const content = document.getElementById('workoutModalContent');
      const title = document.getElementById('workoutModalTitle');
      if(!modal) return;
      modal.classList.remove('hidden');
      if(content) content.innerHTML = '<div class="flex items-center justify-center h-32 text-zinc-500"><div class="animate-pulse">Loading workout details...</div></div>';
      const resp = await fetch(`/api/workout/${workoutDate}`);
      if(!resp.ok) throw new Error('Failed to fetch workout');
      const workout = await resp.json();
      if(title) title.textContent = workout.workout_name || `Workout - ${workout.date}`;
      const badges = document.getElementById('workoutModalBadges');
      if(badges) badges.innerHTML = `
        <span class=\"px-2 py-1 bg-indigo-600/20 text-indigo-300 rounded text-xs font-medium\">${workout.total_exercises} exercises</span>
        <span class=\"px-2 py-1 bg-green-600/20 text-green-300 rounded text-xs font-medium\">${workout.duration_minutes}min</span>
        ${workout.total_prs > 0 ? `<span class=\"px-2 py-1 bg-yellow-600/20 text-yellow-300 rounded text-xs font-medium\">🏆 ${workout.total_prs} PR${workout.total_prs>1?'s':''}</span>`:''}
      `;
      if(content) content.innerHTML = generateWorkoutHTML(workout);
      const shareBtn=document.getElementById('shareWorkoutBtn');
      if(shareBtn) shareBtn.onclick=()=> shareWorkout(workoutDate);
    } catch(e){
      console.error('Error loading workout detail', e);
      const content = document.getElementById('workoutModalContent');
      if(content) content.innerHTML = '<div class="flex items-center justify-center h-32 text-red-400">Error loading workout details</div>';
    }
  }

  function generateWorkoutHTML(workout){
    let html = '<div class="space-y-6">';
    html += `<div class=\"grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-zinc-800/50 rounded-xl\">`+
      `<div class=\"text-center\"><div class=\"text-2xl font-bold text-indigo-400\">${workout.total_sets}</div><div class=\"text-sm text-zinc-400\">Total Sets</div></div>`+
      `<div class=\"text-center\"><div class=\"text-2xl font-bold text-green-400\">${fmtInt(workout.total_volume)}</div><div class=\"text-sm text-zinc-400\">Volume (kg)</div></div>`+
      `<div class=\"text-center\"><div class=\"text-2xl font-bold text-blue-400\">${workout.duration_minutes}</div><div class=\"text-sm text-zinc-400\">Duration (min)</div></div>`+
      `<div class=\"text-center\"><div class=\"text-2xl font-bold text-yellow-400\">${workout.total_prs}</div><div class=\"text-sm text-zinc-400\">Personal Records</div></div>`+
    `</div>`;
    html += '<div class="space-y-4">';
    workout.exercises.forEach((ex,i)=>{
      const prBadge = ex.personal_records>0? '<span class="text-xs bg-yellow-600/20 text-yellow-300 px-2 py-0.5 rounded font-medium ml-2">🏆 PR</span>':'';
      html += `<div class=\"border ${i===0?'bg-indigo-600/10 border-indigo-600/30':'bg-zinc-800/30 border-zinc-700'} rounded-xl p-4\">`+
        `<div class=\"flex items-center justify-between mb-3\"><h3 class=\"text-lg font-semibold text-zinc-100\">${ex.exercise_name}${prBadge}</h3>`+
        `<div class=\"text-sm text-zinc-400\">${ex.total_sets} sets • ${fmt1(ex.total_volume)} kg total</div></div>`+
        `<div class=\"overflow-x-auto\"><table class=\"w-full text-sm\"><thead><tr class=\"border-b border-zinc-700\"><th class=\"text-left py-2 text-zinc-400 font-medium\">Set</th><th class=\"text-right py-2 text-zinc-400 font-medium\">Weight</th><th class=\"text-right py-2 text-zinc-400 font-medium\">Reps</th><th class=\"text-right py-2 text-zinc-400 font-medium\">Volume</th><th class=\"text-right py-2 text-zinc-400 font-medium\">Est. 1RM</th></tr></thead><tbody>`;
      ex.sets.forEach(set=>{
        const isBest = set.volume === ex.best_set.volume;
        html += `<tr class=\"${isBest?'bg-green-600/10 text-green-300':'text-zinc-300'} border-b border-zinc-800/50\"><td class=\"py-2\">${set.set_number}${isBest?' <span class=\\"text-green-400\\">★</span>':''}</td><td class=\"text-right py-2\">${set.weight? fmt1(set.weight)+' kg':'-'}</td><td class=\"text-right py-2\">${set.reps||'-'}</td><td class=\"text-right py-2\">${set.volume? fmt1(set.volume)+' kg':'-'}</td><td class=\"text-right py-2\">${set.estimated_1rm? fmt1(set.estimated_1rm)+' kg':'-'}</td></tr>`;
      });
      html += '</tbody></table></div></div>';
    });
    html += '</div></div>';
    return html;
  }

  function shareWorkout(workoutDate){
    const url = `${window.location.origin}/workout/${workoutDate}`;
    const btn = document.getElementById('shareWorkoutBtn');
    if(navigator.share){
      navigator.share({title:'My Workout', text:`Check out my workout from ${workoutDate}`, url}).catch(()=>{});
    } else if(navigator.clipboard){
      navigator.clipboard.writeText(url).then(()=>{
        if(!btn) return; const orig=btn.textContent; btn.textContent='Copied!'; btn.classList.add('bg-green-600'); setTimeout(()=>{ btn.textContent=orig; btn.classList.remove('bg-green-600'); },1500);
      });
    }
  }

  document.addEventListener('DOMContentLoaded', ()=>{
    const modal=document.getElementById('workoutModal');
    const closeBtn=document.getElementById('closeWorkoutModal');
    closeBtn?.addEventListener('click', ()=>{ modal.classList.add('hidden'); if(window.location.pathname!=='/') history.pushState(null,'','/'); });
    modal?.addEventListener('click', e=>{ if(e.target===modal){ modal.classList.add('hidden'); if(window.location.pathname!=='/') history.pushState(null,'','/'); } });
    document.addEventListener('keydown', e=>{ if(e.key==='Escape' && !modal.classList.contains('hidden')){ modal.classList.add('hidden'); if(window.location.pathname!=='/') history.pushState(null,'','/'); } });
  });

    window.showWorkoutDetail = showWorkoutDetail;
    window.shareWorkout = shareWorkout;
  })();

// Prefer helpers from main.core.js if present
const fetchJSON = window.fetchJSON || (url => fetch(url).then(r => { if(!r.ok) throw new Error(r.statusText); return r.json(); }));
const fmtInt = window.fmtInt || (x => x == null ? '-' : x.toLocaleString());
const fmt1 = window.fmt1 || (x => x==null?'-': (Math.round(x*10)/10).toString());
// Use window.parseISO provided by core; do not declare a global named parseISO here to avoid collisions

// --------------------------- Filters UI --------------------------------
// (legacy code removed)

// Strength Balance Chart
async function loadStrengthBalance() {
  try {
    const data = await fetchJSON('/api/strength-balance');
    
    if (!data || Object.keys(data).length === 0) return;

    const traces = [];
    let colorIndex = 0;

    // Show estimated 1RM progression for each movement pattern
    Object.entries(data).forEach(([pattern, sessions]) => {
      if (sessions.length > 0) {
        traces.push({
          x: sessions.map(s => s.date),
          y: sessions.map(s => s.estimated_1rm),
          type: 'scatter',
          mode: 'lines+markers',
          name: pattern,
          line: { width: 2 },
          marker: { size: 6 },
          hovertemplate: `<b>${pattern}</b><br>` +
                        `Date: %{x}<br>` +
                        `Est. 1RM: %{y:.1f} kg<br>` +
                        `<extra></extra>`
        });
      }
      colorIndex++;
    });

    const layout = {
      title: 'Movement Pattern Balance',
      xaxis: { 
        title: 'Date',
        tickangle: -45,
        type: 'date'
      },
      yaxis: { 
        title: 'Estimated 1RM (kg)'
      },
      hovermode: 'x unified',
      legend: { orientation: 'h', y: -0.2 }
    };

    Plotly.newPlot('strengthBalanceChart', traces, layout, { 
      responsive: true,
      displayModeBar: false 
    });

  } catch (error) {
    console.error('Error loading strength balance:', error);
  }
}

// Exercise Analysis (detailed view for selected exercise)
async function loadExerciseAnalysis(exercise) {
  try {
    if (!exercise) {
      Plotly.newPlot('exerciseAnalysisChart', [], {
        title: 'Select an exercise above for detailed analysis',
        font: { size: 14 }
      });
      return;
    }

    const data = await fetchJSON(`/api/exercise-analysis?exercise=${encodeURIComponent(exercise)}`);
    
    if (!data || !data.session_summary || data.session_summary.length === 0) {
      Plotly.newPlot('exerciseAnalysisChart', [], {
        title: `No data available for ${exercise}`,
        font: { size: 14 }
      });
      return;
    }

    const traces = [
      // Max weight progression
      {
        x: data.session_summary.map(s => s.date),
        y: data.session_summary.map(s => s.max_weight),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Max Weight',
        yaxis: 'y',
        line: { width: 3, color: colors.primary },
        marker: { size: 8 },
        hovertemplate: 'Date: %{x}<br>Max Weight: %{y} kg<extra></extra>'
      },
      // Session volume bars
      {
        x: data.session_summary.map(s => s.date),
        y: data.session_summary.map(s => s.session_volume),
        type: 'bar',
        name: 'Session Volume',
        yaxis: 'y2',
        opacity: 0.6,
        marker: { color: colors.info },
        hovertemplate: 'Date: %{x}<br>Volume: %{y} kg<extra></extra>'
      },
      // Estimated 1RM
      {
        x: data.session_summary.map(s => s.date),
        y: data.session_summary.map(s => s.estimated_1rm),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Est. 1RM',
        yaxis: 'y',
        line: { width: 2, color: colors.success, dash: 'dot' },
        marker: { size: 6 },
        hovertemplate: 'Date: %{x}<br>Est. 1RM: %{y:.1f} kg<extra></extra>'
      }
    ];

    const layout = {
      title: `${exercise} - Detailed Progress Analysis`,
      xaxis: { 
        title: 'Date',
        tickangle: -45,
        type: 'date'
      },
      yaxis: { 
        title: 'Weight (kg)',
        side: 'left'
      },
      yaxis2: { 
        title: 'Volume (kg)',
        side: 'right',
        overlaying: 'y'
      },
      hovermode: 'x unified',
      legend: { orientation: 'h', y: -0.15 }
    };

    Plotly.newPlot('exerciseAnalysisChart', traces, layout, { 
      responsive: true,
      displayModeBar: false 
    });

  } catch (error) {
    console.error('Error loading exercise analysis:', error);
  }
}

// Load exercises list for dropdown
async function loadExerciseOptions() {
  try {
    const exercises = await fetchJSON('/api/exercises');
    const select = document.getElementById('exerciseSelect');
    
    select.innerHTML = '<option value="">-- Select for detailed analysis --</option>' + 
      exercises.map(e => `<option value="${e}">${e}</option>`).join('');
    
    select.addEventListener('change', () => {
      if (select.value) {
        loadExerciseAnalysis(select.value);
      } else {
        loadExerciseAnalysis(null);
      }
    });

  } catch (error) {
    console.error('Error loading exercises list:', error);
  }
}

// Strong-inspired analytics functions

async function loadPersonalRecordsTable() {
  try {
    const data = await fetchJSON('/api/records');
    
    const container = document.getElementById('recordsTable');
    
    if (!data || data.length === 0) {
      container.innerHTML = '<p style="text-align: center; padding: 20px;">No records found</p>';
      return;
    }
    
    let html = `
      <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
        <thead>
          <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
            <th style="padding: 12px 8px; text-align: left; font-weight: bold;">Exercise</th>
            <th style="padding: 12px 8px; text-align: center; font-weight: bold;">Max Weight</th>
            <th style="padding: 12px 8px; text-align: center; font-weight: bold;">Max Reps</th>
            <th style="padding: 12px 8px; text-align: center; font-weight: bold;">Est. 1RM</th>
            <th style="padding: 12px 8px; text-align: center; font-weight: bold;">Total Sets</th>
          </tr>
        </thead>
        <tbody>
    `;
    
    data.slice(0, 15).forEach((record, index) => {
      html += `
        <tr style="border-bottom: 1px solid #dee2e6; ${index % 2 === 0 ? 'background-color: #f8f9fa;' : ''} transition: background-color 0.2s;">
          <td style="padding: 10px 8px; font-weight: 600; color: #2c3e50;">${record.exercise}</td>
          <td style="padding: 10px 8px; text-align: center; color: #27ae60; font-weight: 500;">${record.max_weight} kg</td>
          <td style="padding: 10px 8px; text-align: center; color: #8e44ad; font-weight: 500;">${record.max_reps}</td>
          <td style="padding: 10px 8px; text-align: center; font-weight: bold; color: #e74c3c; font-size: 1.1rem;">${record.estimated_1rm} kg</td>
          <td style="padding: 10px 8px; text-align: center; color: #7f8c8d;">${record.total_sets}</td>
        </tr>
      `;
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;

  } catch (error) {
    console.error('Error loading personal records table:', error);
  }
}

async function loadTrainingCalendar() {
  try {
    const data = await fetchJSON('/api/calendar');
    
    if (!data || data.length === 0) {
      Plotly.newPlot('calendarChart', [], {
        title: 'No training calendar data available'
      });
      return;
    }

    const trace = {
      x: data.map(d => d.date),
      y: data.map(d => d.exercises_performed),
      name: 'Exercises per Day',
      type: 'bar',
      marker: { 
        color: data.map(d => d.exercises_performed),
        colorscale: 'Viridis',
        showscale: true
      },
      hovertemplate: 'Date: %{x}<br>Exercises: %{y}<br>Volume: %{customdata.volume} kg<br>Sets: %{customdata.sets}<br><i>Click to view workout details</i><extra></extra>',
      customdata: data.map(d => ({ volume: d.total_volume, sets: d.total_sets }))
    };

    const layout = {
      title: '📅 Training Calendar - Workout Intensity',
      xaxis: { title: 'Date', type: 'date' },
      yaxis: { title: 'Exercises Performed' },
      showlegend: false
    };

    const plotDiv = document.getElementById('calendarChart');
    Plotly.newPlot(plotDiv, [trace], layout, { responsive: true, displayModeBar: false });
    
    // Add click event listener for workout details
    plotDiv.on('plotly_click', function(data) {
      if (data.points && data.points.length > 0) {
        const point = data.points[0];
        const workoutDate = point.x; // Date from the clicked point
        showWorkoutDetail(workoutDate);
      }
    });

  populateRecentWorkouts(data);

  } catch (error) {
    console.error('Error loading training calendar:', error);
  }
}

function populateRecentWorkouts(calendarData){
  if(!Array.isArray(calendarData)) return;
  const recentContainer = document.getElementById('recentWorkouts');
  if(!recentContainer) return; // widget not present
  const countEl = document.getElementById('recentWorkoutsCount');
  const recent = calendarData.slice().reverse();
  const filtered = recent.filter(d=> d.total_sets>0).slice(0,50); // latest 50
  if(countEl) countEl.textContent = filtered.length ? `${filtered.length} shown` : 'None';
  // Build table
  let html = `
    <div class="overflow-x-auto">
    <table class="w-full text-sm border-separate border-spacing-0">
      <thead class="text-[11px] uppercase tracking-wide text-zinc-400 bg-zinc-800/60">
        <tr>
          <th class="py-2 pl-3 pr-2 text-left font-medium">Date</th>
          <th class="py-2 px-2 text-left font-medium">Workout</th>
          <th class="py-2 px-2 text-right font-medium">Exercises</th>
          <th class="py-2 px-2 text-right font-medium">Sets</th>
          <th class="py-2 px-2 text-right font-medium">Volume</th>
          <th class="py-2 px-2 text-right font-medium">Duration</th>
        </tr>
      </thead>
      <tbody>
  `;
  filtered.forEach(d=>{
  const volStr = (d.total_volume != null ? Math.round(d.total_volume).toLocaleString() + ' kg' : '-');
  const durStr = d.duration_minutes ? (d.duration_minutes >= 60 ? (Math.floor(d.duration_minutes/60)+'h '+Math.round(d.duration_minutes%60)+'m') : Math.round(d.duration_minutes)+'m') : '-';
    html += `
      <tr class="group cursor-pointer transition-colors odd:bg-zinc-800/20 hover:bg-zinc-800/60" data-date="${d.date}">
        <td class="py-2 pl-3 pr-2 font-medium text-zinc-200 whitespace-nowrap">${d.date}</td>
        <td class="py-2 px-2 text-indigo-300 group-hover:underline whitespace-nowrap">${(d.workout_name||'Workout')}</td>
        <td class="py-2 px-2 text-right text-zinc-300">${d.exercises_performed}</td>
        <td class="py-2 px-2 text-right text-zinc-300">${d.total_sets}</td>
    <td class="py-2 px-2 text-right text-zinc-300">${volStr}</td>
    <td class="py-2 px-2 text-right text-zinc-300">${durStr}</td>
      </tr>`;
  });
  html += `</tbody></table></div>`;
  recentContainer.innerHTML = html;
  // Row click handlers
  recentContainer.querySelectorAll('tr[data-date]').forEach(row=>{
    row.addEventListener('click', ()=>{
      const date = row.getAttribute('data-date');
      if (window.location.pathname !== `/workout/${date}`) {
        window.history.pushState(null, '', `/workout/${date}`);
      }
      showWorkoutDetail(date);
    });
  });
}

// Refresh button for recent workouts
document.addEventListener('DOMContentLoaded', ()=>{
  const btn = document.getElementById('recentRefreshBtn');
  if(btn && !btn.dataset._bound){
    btn.dataset._bound='1';
    btn.addEventListener('click', async ()=>{
      try { const data = await fetchJSON('/api/calendar'); populateRecentWorkouts(data); } catch(e){ console.error('recent refresh failed', e); }
    });
  }
});

// Expose for other script files (plotly loader)
window.populateRecentWorkouts = window.populateRecentWorkouts || populateRecentWorkouts;

async function loadMuscleGroupBalance() {
  try {
    const data = await fetchJSON('/api/muscle-balance');
    
    if (!data || Object.keys(data).length === 0) {
      Plotly.newPlot('muscleBalanceChart', [], {
        title: 'No muscle balance data available'
      });
      return;
    }

    const muscleGroups = Object.keys(data);
    const values = muscleGroups.map(group => data[group].max_estimated_1rm);
    
    const trace = {
      labels: muscleGroups,
      values: values,
      type: 'pie',
      hovertemplate: '<b>%{label}</b><br>Max Est. 1RM: %{value} kg<br>%{percent}<extra></extra>',
      textinfo: 'label+percent',
      marker: {
        colors: exerciseColors,
        line: { color: '#FFFFFF', width: 2 }
      }
    };

    const layout = {
      title: '💪 Muscle Group Strength Distribution',
      showlegend: false,
      margin: { l: 20, r: 20, t: 50, b: 20 }
    };

    Plotly.newPlot('muscleBalanceChart', [trace], layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading muscle balance:', error);
  }
}

async function loadBodyMeasurements() {
  try {
    const data = await fetchJSON('/api/measurements');
    
    if (!data || data.length === 0) {
      Plotly.newPlot('measurementsChart', [], {
        title: 'No body measurements data available'
      });
      return;
    }

    const trace = {
      x: data.map(m => m.date),
      y: data.map(m => m.weight),
      name: 'Body Weight',
      type: 'scatter',
      mode: 'lines+markers',
      fill: 'tonexty',
      fillcolor: 'rgba(52, 152, 219, 0.2)',
      line: { color: colors.secondary, width: 3 },
      marker: { size: 6 },
      hovertemplate: 'Date: %{x}<br>Weight: %{y} kg<extra></extra>'
    };

    const layout = {
      title: '📊 Body Weight Progression',
      xaxis: { title: 'Date', type: 'date' },
      yaxis: { title: 'Weight (kg)' },
      showlegend: false
    };

    Plotly.newPlot('measurementsChart', [trace], layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading body measurements:', error);
  }
}

// Training streak & last ingested now exposed by core (avoid dup)

// (duplicate style block removed; core injects once)

// Ensure workout modal functions are exposed early if calendar loaded first
if(typeof window.showWorkoutDetail !== 'function'){
  window.showWorkoutDetail = (...args)=>{
    // replaced later when real function defined
    console.warn('showWorkoutDetail placeholder invoked before definition');
  };
}

// ADVANCED ANALYTICS FUNCTIONS (Plotly extras)

async function loadVolumeHeatmap() {
  try {
    const data = await fetchJSON('/api/volume-heatmap');
    
    if (!data || data.length === 0) {
      Plotly.newPlot('volumeHeatmapChart', [], {
        title: 'No volume heatmap data available'
      });
      return;
    }

    // Create calendar heatmap similar to GitHub contributions
    const trace = {
      x: data.map(d => d.date),
      y: data.map(d => ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][d.day_of_week]),
      z: data.map(d => d.intensity),
      type: 'scatter',
      mode: 'markers',
      marker: {
        size: 15,
        color: data.map(d => d.intensity),
        colorscale: 'Viridis',
        showscale: true,
        colorbar: { title: 'Training Intensity' }
      },
      hovertemplate: 'Date: %{x}<br>Day: %{y}<br>Volume: %{customdata} kg<extra></extra>',
      customdata: data.map(d => d.volume)
    };

    const layout = {
      title: '🔥 Training Volume Heatmap (GitHub Style)',
      xaxis: { title: 'Date', type: 'date' },
      yaxis: { title: 'Day of Week' },
      showlegend: false
    };

    Plotly.newPlot('volumeHeatmapChart', [trace], layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading volume heatmap:', error);
  }
}

async function loadRepDistribution() {
  try {
    const data = await fetchJSON('/api/rep-distribution');
    
    if (!data || data.length === 0) return;

    // Calculate percentages
    const totalSets = data.reduce((sum, item) => sum + item.set_count, 0);
    data.forEach(item => {
      item.percentage = ((item.set_count / totalSets) * 100).toFixed(1);
    });

    const trace = {
      labels: data.map(d => d.rep_range),
      values: data.map(d => d.set_count),
      type: 'pie',
      hovertemplate: '<b>%{label}</b><br>Sets: %{value}<br>%{percent}<br>Volume: %{customdata} kg<extra></extra>',
      customdata: data.map(d => d.total_volume),
      marker: {
        colors: ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7'],
        line: { color: '#FFFFFF', width: 2 }
      }
    };

    const layout = {
      title: '📊 Rep Range Distribution - Training Focus',
      showlegend: true,
      legend: { orientation: 'h', y: -0.1 }
    };

    Plotly.newPlot('repDistributionChart', [trace], layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading rep distribution:', error);
  }
}

async function loadExerciseFrequency() {
  try {
    const data = await fetchJSON('/api/exercise-frequency');
    
    if (!data || data.length === 0) return;

    const trace = {
      x: data.slice(0, 10).map(d => d.exercise),
      y: data.slice(0, 10).map(d => d.workout_days),
      type: 'bar',
      marker: {
        color: data.slice(0, 10).map(d => d.workout_days),
        colorscale: 'Viridis',
        showscale: true
      },
      hovertemplate: '<b>%{x}</b><br>Workout Days: %{y}<br>Total Sets: %{customdata.sets}<br>Avg Weight: %{customdata.weight} kg<extra></extra>',
      customdata: data.slice(0, 10).map(d => ({
        sets: d.total_sets,
        weight: d.avg_weight
      }))
    };

    const layout = {
      title: '🎯 Exercise Frequency - Top 10 Most Trained',
      xaxis: { title: 'Exercise', tickangle: -45 },
      yaxis: { title: 'Workout Days' },
      showlegend: false
    };

    Plotly.newPlot('exerciseFrequencyChart', [trace], layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading exercise frequency:', error);
  }
}

async function loadStrengthRatios() {
  try {
    const data = await fetchJSON('/api/strength-ratios');
    
    if (!data || !data.ratios || Object.keys(data.ratios).length === 0) {
      Plotly.newPlot('strengthRatiosChart', [], {
        title: 'No strength ratio data available'
      });
      return;
    }

    const ratioNames = Object.keys(data.ratios);
    const actualRatios = Object.values(data.ratios);
    const idealRatios = ratioNames.map(name => data.ideal_ratios[name] || 1);

    const traces = [
      {
        x: ratioNames,
        y: actualRatios,
        name: 'Your Ratios',
        type: 'bar',
        marker: { color: '#3498db' }
      },
      {
        x: ratioNames,
        y: idealRatios,
        name: 'Ideal Ratios',
        type: 'scatter',
        mode: 'markers',
        marker: { size: 12, color: '#e74c3c', symbol: 'diamond' }
      }
    ];

    const layout = {
      title: '⚡ Strength Ratios vs Ideal Standards',
      xaxis: { title: 'Lift Ratios' },
      yaxis: { title: 'Ratio Value' },
      showlegend: true
    };

    Plotly.newPlot('strengthRatiosChart', traces, layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading strength ratios:', error);
  }
}

async function loadRecoveryTracking() {
  try {
    const data = await fetchJSON('/api/recovery-tracking');
    
    if (!data || Object.keys(data).length === 0) return;

    const muscleGroups = Object.keys(data);
    const avgRecovery = muscleGroups.map(group => data[group].avg_recovery);
    const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7'];

    const trace = {
      x: muscleGroups,
      y: avgRecovery,
      type: 'bar',
      marker: {
        color: colors.slice(0, muscleGroups.length)
      },
      hovertemplate: '<b>%{x}</b><br>Avg Recovery: %{y} days<br>Sessions: %{customdata.sessions}<br>Range: %{customdata.min}-%{customdata.max} days<extra></extra>',
      customdata: muscleGroups.map(group => ({
        sessions: data[group].total_sessions,
        min: data[group].min_recovery,
        max: data[group].max_recovery
      }))
    };

    const layout = {
      title: '🔄 Recovery Time Between Sessions',
      xaxis: { title: 'Muscle Group' },
      yaxis: { title: 'Average Days Between Sessions' },
      showlegend: false
    };

    Plotly.newPlot('recoveryChart', [trace], layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading recovery tracking:', error);
  }
}

async function loadProgressionRate() {
  try {
    const data = await fetchJSON('/api/progression-rate');
    
    if (!data || data.length === 0) return;

    const topProgressors = data.slice(0, 8);

    const trace = {
      x: topProgressors.map(d => d.exercise),
      y: topProgressors.map(d => d.percentage_gain),
      type: 'bar',
      marker: {
        color: topProgressors.map(d => d.percentage_gain),
        colorscale: 'RdYlGn',
        showscale: true,
        colorbar: { title: '% Gain' }
      },
      hovertemplate: '<b>%{x}</b><br>Total Gain: %{y}%<br>Weekly Rate: %{customdata.weekly} kg/week<br>Days Tracked: %{customdata.days}<extra></extra>',
      customdata: topProgressors.map(d => ({
        weekly: d.weekly_rate,
        days: d.days_tracked
      }))
    };

    const layout = {
      title: '📈 Progressive Overload Rate - Top Performers',
      xaxis: { title: 'Exercise', tickangle: -45 },
      yaxis: { title: 'Percentage Gain (%)' },
      showlegend: false
    };

    Plotly.newPlot('progressionRateChart', [trace], layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading progression rate:', error);
  }
}

async function loadDurationTrends() {
  try {
    const data = await fetchJSON('/api/workout-duration');
    
    if (!data || data.length === 0) return;

    const traces = [
      {
        x: data.map(d => d.date),
        y: data.map(d => d.duration),
        name: 'Duration (min)',
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#3498db' },
        yaxis: 'y'
      },
      {
        x: data.map(d => d.date),
        y: data.map(d => d.efficiency_score),
        name: 'Efficiency Score',
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#e74c3c', dash: 'dash' },
        yaxis: 'y2'
      }
    ];

    const layout = {
      title: '⏱️ Workout Duration & Efficiency Trends',
      xaxis: { title: 'Date', type: 'date' },
      yaxis: { title: 'Duration (minutes)', side: 'left' },
      yaxis2: { title: 'Efficiency Score', side: 'right', overlaying: 'y' },
      hovermode: 'x unified',
      showlegend: true
    };

    Plotly.newPlot('durationTrendsChart', traces, layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading duration trends:', error);
  }
}

async function loadBestSets() {
  try {
    const data = await fetchJSON('/api/best-sets');
    
    if (!data || data.length === 0) return;

    const topSets = data.slice(0, 12);

    const trace = {
      x: topSets.map(d => d.exercise),
      y: topSets.map(d => d.estimated_1rm),
      type: 'bar',
      marker: {
        color: topSets.map(d => d.estimated_1rm),
        colorscale: 'Viridis',
        showscale: true,
        colorbar: { title: 'Est. 1RM (kg)' }
      },
      hovertemplate: '<b>%{x}</b><br>Est. 1RM: %{y} kg<br>Weight: %{customdata.weight} kg<br>Reps: %{customdata.reps}<br>Date: %{customdata.date}<extra></extra>',
      customdata: topSets.map(d => ({
        weight: d.weight,
        reps: d.reps,
        date: d.date
      }))
    };

    const layout = {
      title: '🏆 Best Set Performance - Personal Records',
      xaxis: { title: 'Exercise', tickangle: -45 },
      yaxis: { title: 'Estimated 1RM (kg)' },
      showlegend: false
    };

    Plotly.newPlot('bestSetsChart', [trace], layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading best sets:', error);
  }
}

async function loadPlateauDetection() {
  try {
    const data = await fetchJSON('/api/plateau-detection');
    
    if (!data || data.length === 0) return;

    // Group by status for better visualization
    const statusColors = {
      'Progressing': '#27ae60',
      'Plateau': '#f39c12', 
      'Declining': '#e74c3c',
      'Variable': '#8e44ad'
    };

    const traces = [];
    const statuses = ['Progressing', 'Plateau', 'Declining', 'Variable'];
    
    statuses.forEach(status => {
      const statusData = data.filter(d => d.status === status);
      if (statusData.length > 0) {
        traces.push({
          x: statusData.map(d => d.exercise),
          y: statusData.map(d => d.current_1rm),
          name: status,
          type: 'scatter',
          mode: 'markers',
          marker: {
            size: 12,
            color: statusColors[status]
          },
          hovertemplate: '<b>%{x}</b><br>Status: ' + status + '<br>Current 1RM: %{y} kg<br>Best Recent: %{customdata} kg<extra></extra>',
          customdata: statusData.map(d => d.best_recent_1rm)
        });
      }
    });

    const layout = {
      title: '🚨 Plateau Detection - Exercise Performance Status',
      xaxis: { title: 'Exercise', tickangle: -45 },
      yaxis: { title: 'Current Est. 1RM (kg)' },
      hovermode: 'closest',
      showlegend: true,
      legend: { orientation: 'h', y: -0.2 }
    };

    Plotly.newPlot('plateauDetectionChart', traces, layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading plateau detection:', error);
  }
}

// ----------------------- Workout Detail Modal Functions -----------------------

async function showWorkoutDetail(workoutDate) {
  try {
    const modal = document.getElementById('workoutModal');
    const content = document.getElementById('workoutModalContent');
    const title = document.getElementById('workoutModalTitle');
    
    // Show modal with loading state
    modal.classList.remove('hidden');
    content.innerHTML = '<div class="flex items-center justify-center h-32 text-zinc-500"><div class="animate-pulse">Loading workout details...</div></div>';
    
    // Fetch workout data
    const response = await fetch(`/api/workout/${workoutDate}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch workout: ${response.statusText}`);
    }
    
    const workout = await response.json();
    
    // Update modal title and badges
    title.textContent = workout.workout_name || `Workout - ${workout.date}`;
    
    const badges = document.getElementById('workoutModalBadges');
    badges.innerHTML = `
      <span class="px-2 py-1 bg-indigo-600/20 text-indigo-300 rounded text-xs font-medium">
        ${workout.total_exercises} exercises
      </span>
      <span class="px-2 py-1 bg-green-600/20 text-green-300 rounded text-xs font-medium">
        ${workout.duration_minutes}min
      </span>
      ${workout.total_prs > 0 ? `<span class="px-2 py-1 bg-yellow-600/20 text-yellow-300 rounded text-xs font-medium">🏆 ${workout.total_prs} PR${workout.total_prs > 1 ? 's' : ''}</span>` : ''}
    `;
    
    // Generate workout content
    content.innerHTML = generateWorkoutHTML(workout);
    
    // Set up share functionality
    const shareBtn = document.getElementById('shareWorkoutBtn');
    shareBtn.onclick = () => shareWorkout(workoutDate);
    
  } catch (error) {
    console.error('Error loading workout detail:', error);
    document.getElementById('workoutModalContent').innerHTML = `
      <div class="flex items-center justify-center h-32 text-red-400">
        <div>Error loading workout details</div>
      </div>
    `;
  }
}

function generateWorkoutHTML(workout) {
  let html = `
    <div class="space-y-6">
      <!-- Workout Summary -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-zinc-800/50 rounded-xl">
        <div class="text-center">
          <div class="text-2xl font-bold text-indigo-400">${workout.total_sets}</div>
          <div class="text-sm text-zinc-400">Total Sets</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold text-green-400">${fmtInt(workout.total_volume)}</div>
          <div class="text-sm text-zinc-400">Volume (kg)</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold text-blue-400">${workout.duration_minutes}</div>
          <div class="text-sm text-zinc-400">Duration (min)</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold text-yellow-400">${workout.total_prs}</div>
          <div class="text-sm text-zinc-400">Personal Records</div>
        </div>
      </div>
      
      <!-- Exercise Details -->
      <div class="space-y-4">
  `;
  
  workout.exercises.forEach((exercise, index) => {
    const isFirstExercise = index === 0;
    const bgColor = isFirstExercise ? 'bg-indigo-600/10 border-indigo-600/30' : 'bg-zinc-800/30 border-zinc-700';
    const prBadge = exercise.personal_records > 0 ? '<span class="text-xs bg-yellow-600/20 text-yellow-300 px-2 py-0.5 rounded font-medium ml-2">🏆 PR</span>' : '';
    
    html += `
      <div class="border ${bgColor} rounded-xl p-4">
        <div class="flex items-center justify-between mb-3">
          <h3 class="text-lg font-semibold text-zinc-100">${exercise.exercise_name}${prBadge}</h3>
          <div class="text-sm text-zinc-400">
            ${exercise.total_sets} sets • ${fmt1(exercise.total_volume)} kg total
          </div>
        </div>
        
        <!-- Sets Table -->
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="border-b border-zinc-700">
                <th class="text-left py-2 text-zinc-400 font-medium">Set</th>
                <th class="text-right py-2 text-zinc-400 font-medium">Weight</th>
                <th class="text-right py-2 text-zinc-400 font-medium">Reps</th>
                <th class="text-right py-2 text-zinc-400 font-medium">Volume</th>
                <th class="text-right py-2 text-zinc-400 font-medium">Est. 1RM</th>
              </tr>
            </thead>
            <tbody>
    `;
    
    exercise.sets.forEach(set => {
      const isBestSet = set.volume === exercise.best_set.volume;
      const rowClass = isBestSet ? 'bg-green-600/10 text-green-300' : 'text-zinc-300';
      const bestSetIcon = isBestSet ? ' <span class="text-green-400">★</span>' : '';
      
      html += `
        <tr class="${rowClass} border-b border-zinc-800/50">
          <td class="py-2">${set.set_number}${bestSetIcon}</td>
          <td class="text-right py-2">${set.weight ? fmt1(set.weight) + ' kg' : '-'}</td>
          <td class="text-right py-2">${set.reps || '-'}</td>
          <td class="text-right py-2">${set.volume ? fmt1(set.volume) + ' kg' : '-'}</td>
          <td class="text-right py-2">${set.estimated_1rm ? fmt1(set.estimated_1rm) + ' kg' : '-'}</td>
        </tr>
      `;
    });
    
    html += `
            </tbody>
          </table>
        </div>
      </div>
    `;
  });
  
  html += `
      </div>
    </div>
  `;
  
  return html;
}

function shareWorkout(workoutDate) {
  const url = `${window.location.origin}/workout/${workoutDate}`;
  
  if (navigator.share) {
    // Use native share API if available (mobile)
    navigator.share({
      title: 'My Workout',
      text: `Check out my workout from ${workoutDate}`,
      url: url
    }).catch(console.error);
  } else {
    // Fallback to clipboard
    navigator.clipboard.writeText(url).then(() => {
      // Show success feedback
      const btn = document.getElementById('shareWorkoutBtn');
      const originalText = btn.textContent;
      btn.textContent = 'Copied!';
      btn.className = btn.className.replace('bg-indigo-600 hover:bg-indigo-700', 'bg-green-600 hover:bg-green-700');
      
      setTimeout(() => {
        btn.textContent = originalText;
        btn.className = btn.className.replace('bg-green-600 hover:bg-green-700', 'bg-indigo-600 hover:bg-indigo-700');
      }, 2000);
    }).catch(console.error);
  }
}

// Set up modal event listeners
document.addEventListener('DOMContentLoaded', () => {
  const modal = document.getElementById('workoutModal');
  const closeBtn = document.getElementById('closeWorkoutModal');
  
  // Close modal handlers
  closeBtn?.addEventListener('click', () => {
    modal.classList.add('hidden');
    // Update URL to remove workout parameter
    if (window.location.pathname !== '/') {
      window.history.pushState(null, '', '/');
    }
  });
  
  // Close on backdrop click
  modal?.addEventListener('click', (e) => {
    if (e.target === modal) {
      modal.classList.add('hidden');
      if (window.location.pathname !== '/') {
        window.history.pushState(null, '', '/');
      }
    }
  });
  
  // Close on escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
      modal.classList.add('hidden');
      if (window.location.pathname !== '/') {
        window.history.pushState(null, '', '/');
      }
    }
  });
});

// Make showWorkoutDetail globally available
window.showWorkoutDetail = showWorkoutDetail;

// Expose other utilities if needed
window.shareWorkout = window.shareWorkout || shareWorkout;

```

# app\static\main.plotly.js

```js
// Plotly-driven sections of the dashboard (personal records, calendar, advanced analytics)

async function loadPersonalRecords() {
  try {
    const data = await fetchJSON('/api/personal-records');
    if (!data || data.length === 0) return;
    const exerciseGroups = {};
    data.forEach(pr => { (exerciseGroups[pr.exercise] ||= []).push(pr); });
    const traces = [];
    let colorIndex = 0;
    Object.entries(exerciseGroups).forEach(([exercise, prs]) => {
      traces.push({ x: prs.map(pr => pr.date), y: prs.map(pr => pr.weight), type: 'scatter', mode: 'markers+text', name: exercise, marker: { size: 12, color: '#f59e0b', symbol: 'star' }, text: prs.map(pr => `${pr.weight}kg`), textposition: 'top center', hovertemplate: `<b>${exercise}</b><br>Date: %{x}<br>Weight: %{y} kg<extra></extra>` });
      colorIndex++;
    });
    const layout = { title: 'Personal Records Achievement', xaxis:{title:'Date', type:'date'}, yaxis:{title:'Weight (kg)'}, hovermode:'closest', legend:{orientation:'h', y:-0.2} };
    Plotly.newPlot('personalRecordsChart', traces, layout, { responsive: true, displayModeBar: false });
  } catch (error) { console.error('Error loading personal records:', error); }
}

async function loadTrainingCalendar() {
  try {
    const data = await fetchJSON('/api/calendar');
    const plotDiv = document.getElementById('calendarChart');
    if(!plotDiv) return;
    // Clear any loading placeholder
    plotDiv.innerHTML='';
    if (!data || data.length === 0) {
      Plotly.newPlot(plotDiv, [], { title: 'No training calendar data available', paper_bgcolor:'#18181b', plot_bgcolor:'#18181b', font:{color:'#e4e4e7'} });
      return;
    }
  // Intensity metric based on relative session volume (weight normalized 0-1 -> scaled)
  const volumes = data.map(d=> d.total_volume || 0);
  const vMin = Math.min(...volumes);
  const vMax = Math.max(...volumes);
  const intensity = volumes.map(v=> vMax===vMin? 1 : ( (v - vMin) / (vMax - vMin) )*0.7 + 0.3 ); // keep within 0.3-1 range for color separation
    const trace = {
      x: data.map(d => d.date),
  y: data.map(d => d.exercises_performed),
  name: 'Session Intensity',
      type: 'bar',
      marker: {
        color: intensity,
        colorscale: 'Inferno',
        showscale: true,
        colorbar: {
          title: 'Intensity',
          tickcolor:'#a1a1aa',
          tickfont:{color:'#a1a1aa'},
          titlefont:{color:'#d4d4d8'}
        }
      },
      hovertemplate: 'Date: %{x}<br>Exercises: %{customdata.ex}<br>Volume: %{customdata.volume} kg<br>Sets: %{customdata.sets}<br>Intensity: %{marker.color:.2f}<br><i>Click to view workout details</i><extra></extra>',
      customdata: data.map(d => ({ volume: d.total_volume, sets: d.total_sets, ex: d.exercises_performed }))
    };
    const axisStyle = { tickcolor:'#3f3f46', color:'#a1a1aa', gridcolor:'#27272a', linecolor:'#3f3f46' };
    const layout = {
      title: { text:'📅 Training Calendar - Workout Intensity', font:{color:'#e4e4e7'} },
      paper_bgcolor:'#18181b',
      plot_bgcolor:'#18181b',
      xaxis:{ title:'Date', type:'date', tickfont:{color:'#a1a1aa'}, titlefont:{color:'#a1a1aa'}, gridcolor:'#27272a', linecolor:axisStyle.linecolor },
      yaxis:{ title:'Exercises Performed', tickfont:{color:'#a1a1aa'}, titlefont:{color:'#a1a1aa'}, gridcolor:'#27272a', linecolor:axisStyle.linecolor },
      font:{color:'#d4d4d8'},
      showlegend:false,
      margin:{l:60,r:40,t:60,b:60}
    };
    Plotly.newPlot(plotDiv, [trace], layout, { responsive: true, displayModeBar: false });
    plotDiv.on('plotly_click', function(ev) {
      if (ev.points && ev.points.length > 0) {
        let raw = ev.points[0].x;
        // Normalize to YYYY-MM-DD
        const workoutDate = raw instanceof Date ? raw.toISOString().slice(0,10) : String(raw).slice(0,10);
        openWorkout(workoutDate);
      }
    });

    // Populate recent workouts table (new view)
    if (typeof window.populateRecentWorkouts === 'function') {
      try { window.populateRecentWorkouts(data); } catch(e){ console.warn('populateRecentWorkouts failed', e); }
    }
  } catch (error) { console.error('Error loading training calendar:', error); }
}

function openWorkout(date){
  if(!date) return;
  // Update URL (pushState) then show modal
  try {
    if(window.location.pathname !== `/workout/${date}`){ window.history.pushState(null,'',`/workout/${date}`); }
    if(typeof window.showWorkoutDetail === 'function'){
      window.showWorkoutDetail(date);
    } else {
      console.warn('showWorkoutDetail not ready, retrying...');
      setTimeout(()=> window.showWorkoutDetail && window.showWorkoutDetail(date), 300);
    }
  } catch(e){ console.error('openWorkout failed', e); }
}

// Other Plotly-based loaders are left in this file for clarity (muscle balance, measurements, etc.)
async function loadMuscleGroupBalance() {
  try { const data = await fetchJSON('/api/muscle-balance'); if (!data || Object.keys(data).length === 0) { Plotly.newPlot('muscleBalanceChart', [], { title: 'No muscle balance data available' }); return; } const muscleGroups = Object.keys(data); const values = muscleGroups.map(group => data[group].max_estimated_1rm); const trace = { labels: muscleGroups, values: values, type: 'pie', hovertemplate: '<b>%{label}</b><br>Max Est. 1RM: %{value} kg<br>%{percent}<extra></extra>', textinfo: 'label+percent', marker: { colors: ['#8B5CF6','#10B981','#6366F1','#EC4899','#F59E0B'], line: { color: '#FFFFFF', width: 2 } } }; const layout = { title: '💪 Muscle Group Strength Distribution', showlegend: false, margin: { l: 20, r: 20, t: 50, b: 20 } }; Plotly.newPlot('muscleBalanceChart', [trace], layout, { responsive: true, displayModeBar: false }); } catch (error) { console.error('Error loading muscle balance:', error); } }

// Expose plotly loaders
window.loadPersonalRecords = loadPersonalRecords;
window.loadTrainingCalendar = loadTrainingCalendar;
window.loadMuscleGroupBalance = loadMuscleGroupBalance;

```

# app\templates\dashboard.html

```html
<!DOCTYPE html>
<html
    lang="en"
    class="dark"
>
    <head>
        <meta charset="UTF-8" />
        <meta
            name="viewport"
            content="width=device-width,initial-scale=1"
        />
        <title>Lifting Progression Dashboard</title>
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
    <link rel="alternate icon" href="/favicon.ico" />
    <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
        <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
        <!-- tailwind -->
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {
                background: #09090b;
            }
            .card {
                transition: background-color 0.2s ease, box-shadow 0.2s ease;
            }
            .card:hover {
                box-shadow: 0 4px 18px -2px rgba(0, 0, 0, 0.55);
            }
            .chart {
                width: 100%;
            }
        </style>
    </head>
    <body class="text-zinc-200 antialiased">
        <script id="workoutDateData" type="application/json">{{ workout_date|tojson if workout_date is defined else 'null' }}</script>
        <main class="mx-auto max-w-[1700px] px-6 py-6 space-y-8">
            <header class="flex flex-col gap-2">
                <div class="flex items-center gap-3">
                    <img src="/favicon.svg" alt="Logo" class="w-10 h-10 select-none" />
                    <h1 class="text-3xl font-bold tracking-tight">Progression • Overload • Balance</h1>
                </div>
                <p class="text-sm text-zinc-500">
                    Last updated: <span id="lastIngested" class="text-zinc-300">...</span>
                </p>
            </header>

            <!-- PR Sparklines -->
            <section>
                <h2 class="text-xl font-semibold mb-3">
                    Personal Record Sparklines
                </h2>
                <div
                    id="sparklineContainer"
                    class="grid gap-4 md:grid-cols-3 lg:grid-cols-5 auto-rows-fr"
                ></div>
            </section>

            {% for row in layout_config.rows %}
            <section class="grid gap-6 grid-cols-1 md:grid-cols-12">
                {% for cell in row %}
                    {% set w = cell.width %}
                    {% if cell.widget == 'progressive_overload' %}
                        <div class="card col-span-12 md:col-span-{{w}} bg-zinc-900 rounded-2xl ring-1 ring-zinc-800 shadow-sm p-5 flex flex-col">
                            <div class="flex items-center justify-between mb-2">
                                <h2 class="text-xl font-semibold">Progressive Overload</h2>
                                <button id="resetOverloadZoom" class="text-xs px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700">Reset Zoom</button>
                            </div>
                            <div id="progressiveOverloadChart" class="chart" style="height:300px" data-fixed-height="true"></div>
                            <div class="mt-3 flex flex-col gap-2">
                                <div id="poLegend" class="flex flex-wrap gap-2 text-xs"></div>
                                <div id="overloadSlopes" class="flex flex-wrap gap-2 text-xs"></div>
                            </div>
                        </div>
                    {% elif cell.widget == 'volume_trend' %}
                        <div class="card col-span-12 md:col-span-{{w}} bg-zinc-900 rounded-2xl ring-1 ring-zinc-800 shadow-sm p-5 flex flex-col">
                            <h2 class="text-xl font-semibold mb-2">Session Volume Trend</h2>
                            <div id="volumeTrendChart" class="chart" style="height:240px" data-fixed-height="true"></div>
                        </div>
                    {% elif cell.widget == 'exercise_volume' %}
                        <div class="card col-span-12 md:col-span-{{w}} bg-zinc-900 rounded-2xl ring-1 ring-zinc-800 shadow-sm p-5 flex flex-col">
                            <div class="flex items-center justify-between mb-2">
                                <h2 class="text-xl font-semibold">Volume / Exercise</h2>
                                <button id="volumeModeToggle" class="text-xs px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700" data-mode="grouped">Grouped</button>
                            </div>
                            <div id="exerciseVolumeChart" class="chart" style="height:240px" data-fixed-height="true"></div>
                        </div>
                    {% elif cell.widget == 'rep_distribution' %}
                        <div class="card col-span-12 md:col-span-{{w}} bg-zinc-900 rounded-2xl ring-1 ring-zinc-800 shadow-sm p-5 flex flex-col">
                            <div class="flex items-center justify-between mb-2">
                                <h2 class="text-xl font-semibold">Rep Range Distribution</h2>
                                <button id="repModeToggle" class="text-xs px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700" data-mode="weekly">Weekly</button>
                            </div>
                            <div id="repDistributionChart" class="chart" style="height:240px" data-fixed-height="true"></div>
                        </div>
                    {% elif cell.widget == 'weekly_ppl' %}
                        <div class="card col-span-12 md:col-span-{{w}} bg-zinc-900 rounded-2xl ring-1 ring-zinc-800 shadow-sm p-5 flex flex-col">
                            <div class="flex items-center justify-between mb-2">
                                <h2 class="text-xl font-semibold">Push / Pull / Legs Weekly</h2>
                                <button id="pplModeToggle" class="text-xs px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700" data-mode="absolute">Absolute</button>
                            </div>
                            <div id="weeklyPPLChart" class="chart" style="height:240px" data-fixed-height="true"></div>
                        </div>
                    {% elif cell.widget == 'muscle_balance' %}
                        <div class="card col-span-12 md:col-span-{{w}} bg-zinc-900 rounded-2xl ring-1 ring-zinc-800 shadow-sm p-5 flex flex-col">
                            <h2 class="text-xl font-semibold mb-2">Muscle Group Balance (28d)</h2>
                            <div id="muscleBalanceChart" class="chart" style="height:240px" data-fixed-height="true"></div>
                        </div>
                    {% elif cell.widget == 'calendar' %}
                        <div class="card col-span-12 md:col-span-{{w}} bg-zinc-900 rounded-2xl ring-1 ring-zinc-800 shadow-sm p-5 flex flex-col">
                            <h2 class="text-xl font-semibold mb-2">Training Calendar</h2>
                            <p class="text-sm text-zinc-400 mb-4">Click on any workout day to view detailed information</p>
                            <div id="calendarChart" class="chart" style="height:300px" data-fixed-height="true"></div>
                        </div>
                    {% elif cell.widget == 'recent_workouts' %}
                        <div class="card col-span-12 md:col-span-{{w}} bg-zinc-900 rounded-2xl ring-1 ring-zinc-800 shadow-sm p-5 flex flex-col">
                            <div class="flex items-center justify-between mb-4">
                                <h2 class="text-xl font-semibold flex items-center gap-2">Recent Workouts <span class="text-xs font-normal text-zinc-500" id="recentWorkoutsCount"></span></h2>
                                <button id="recentRefreshBtn" class="text-xs px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700">Refresh</button>
                            </div>
                            <div id="recentWorkouts" class="overflow-x-auto -mx-5 px-5"></div>
                        </div>
                    {% endif %}
                {% endfor %}
            </section>
            {% endfor %}
        </main>

        <!-- Workout Detail Modal -->
        <div
            id="workoutModal"
            class="fixed inset-0 bg-black/50 backdrop-blur-sm hidden z-50 flex items-center justify-center p-4"
        >
            <div
                class="bg-zinc-900 rounded-2xl ring-1 ring-zinc-800 shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col"
            >
                <!-- Modal Header -->
                <div
                    class="flex items-center justify-between p-6 border-b border-zinc-800"
                >
                    <div class="flex items-center gap-4">
                        <h2
                            id="workoutModalTitle"
                            class="text-2xl font-bold"
                        >
                            Workout Details
                        </h2>
                        <div
                            id="workoutModalBadges"
                            class="flex gap-2"
                        ></div>
                    </div>
                    <div class="flex items-center gap-2">
                        <button
                            id="shareWorkoutBtn"
                            class="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-sm font-medium transition-colors"
                            title="Copy shareable link"
                        >
                            Share
                        </button>
                        <button
                            id="closeWorkoutModal"
                            class="p-2 hover:bg-zinc-800 rounded-lg transition-colors"
                        >
                            <svg
                                class="w-5 h-5"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path
                                    stroke-linecap="round"
                                    stroke-linejoin="round"
                                    stroke-width="2"
                                    d="M6 18L18 6M6 6l12 12"
                                />
                            </svg>
                        </button>
                    </div>
                </div>

                <!-- Modal Content -->
                <div class="flex-1 overflow-auto p-6">
                    <div id="workoutModalContent">
                        <div
                            class="flex items-center justify-center h-32 text-zinc-500"
                        >
                            <div class="animate-pulse">
                                Loading workout details...
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

                {% set v = (layout_config.rows|length) %}
                <script src="/static/main.core.js?v={{v}}"></script>
                <script src="/static/main.charts.js?v={{v}}"></script>
                <script src="/static/main.js?v={{v}}"></script>
                <script src="/static/main.plotly.js?v={{v}}"></script>
                        <script>
                                // Wait until all modular scripts loaded, then bootstrap dashboard and handle shareable URL
                                document.addEventListener('DOMContentLoaded', () => {
                                    // Boot once
                                    try { if(window.bootstrapDashboard) window.bootstrapDashboard(); else console.warn('bootstrapDashboard not available'); } catch(e){ console.error(e); }
                                    const workoutDate = JSON.parse(document.getElementById('workoutDateData').textContent);
                                    if (workoutDate) {
                                        // Auto-open workout detail for shareable links
                                        setTimeout(() => {
                                            try { if(window.showWorkoutDetail) showWorkoutDetail(workoutDate); else console.warn('showWorkoutDetail not defined yet'); } catch(e){ console.error(e); }
                                        }, 1000);
                                    }
                                });
                        </script>
    </body>
</html>

```

# dashboard_layout.yaml

```yaml
# Dashboard layout configuration
# Define rows of widgets and their widths (Tailwind md:col-span-*)
# Each row is a list of entries in the form widget:width
# Valid widgets:
#   progressive_overload
#   volume_trend
#   exercise_volume
#   rep_distribution
#   weekly_ppl
#   muscle_balance
#   calendar
#   recent_workouts
# Width must be 1-12 (12 = full width). If omitted, defaults to 12.

layout:
  rows:
    - [progressive_overload:12]
    - [volume_trend:8, exercise_volume:4]
    - [rep_distribution:4, weekly_ppl:4, muscle_balance:4]
    - [calendar:12]
    - [recent_workouts:12]

```

# data\lifting.db

This is a binary file of the type: Binary

# data\uploads\strong_20250826_233708.csv

```csv
Date,Workout Name,Duration,Exercise Name,Set Order,Weight,Reps,Distance,Seconds,RPE
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",2,10.0,16.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",3,10.0,17.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",2,10.0,10.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",3,10.0,8.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",4,15.0,10.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",3,15.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",1,10.0,20.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",2,15.0,15.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",0,0,0.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",2,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",1,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",1,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",3,11.0,8.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",2,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",2,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",1,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",2,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",1,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",3,11.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",1,10.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",2,15.0,7.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",0,0,0.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",3,15.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",2,9.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",1,9.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",3,9.0,6.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",3,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",1,11.0,12.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",2,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",3,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",2,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",3,8.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",2,8.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Bicep Curl (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",3,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",2,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",1,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",2,18.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",3,18.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",1,11.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",1,17.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",2,23.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",3,23.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Glute Kickback (Machine)",1,5.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Glute Kickback (Machine)",2,5.0,7.0,0,0.0,
```

# data\uploads\strong_20250826_235603.csv

```csv
Date,Workout Name,Duration,Exercise Name,Set Order,Weight,Reps,Distance,Seconds,RPE
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",2,10.0,16.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",3,10.0,17.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",3,10.0,8.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",2,10.0,10.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",3,15.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",2,15.0,15.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",1,10.0,20.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",4,15.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",1,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",0,0,0.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",2,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",3,11.0,8.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",2,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",1,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",2,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",1,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",3,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",2,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",1,11.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",3,15.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",2,15.0,7.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",1,10.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",0,0,0.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",2,9.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",3,9.0,6.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",1,9.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",3,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",2,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",1,11.0,12.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",3,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",2,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",3,8.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",2,8.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Bicep Curl (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",2,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",1,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",3,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",2,18.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",3,18.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",1,11.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",2,23.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",1,17.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",3,23.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Glute Kickback (Machine)",1,5.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Glute Kickback (Machine)",2,5.0,7.0,0,0.0,
```

# data\uploads\strong_20250829_134256.csv

```csv
Date,Workout Name,Duration,Exercise Name,Set Order,Weight,Reps,Distance,Seconds,RPE
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",3,10.0,17.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",2,10.0,16.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",2,10.0,10.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",3,10.0,8.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",1,10.0,20.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",4,15.0,10.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",2,15.0,15.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",3,15.0,13.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",0,0,0.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",1,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",2,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",1,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",3,11.0,8.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",2,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",1,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",2,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",2,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",1,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",3,11.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",3,15.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",1,10.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",2,15.0,7.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",0,0,0.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",3,9.0,6.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",2,9.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",1,9.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",2,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",1,11.0,12.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",3,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",2,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",3,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",2,8.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",3,8.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Bicep Curl (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",2,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",1,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",3,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",3,18.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",1,11.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",2,18.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",1,17.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",2,23.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",3,23.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Glute Kickback (Machine)",2,5.0,7.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Glute Kickback (Machine)",1,5.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",2,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",3,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",1,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",2,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",3,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",2,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",3,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",1,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",3,4.0,7.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",2,4.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",3,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",2,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",1,23.0,10.0,0,0.0,
```

# debug.html

```html
<!DOCTYPE html>
<html>
<head>
    <title>API Debug Test</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
</head>
<body>
    <h1>API Debug Test</h1>
    <div id="status"></div>
    <div id="results"></div>
    
    <script>
        const statusDiv = document.getElementById('status');
        const resultsDiv = document.getElementById('results');
        
        async function testAPI(endpoint, name) {
            try {
                statusDiv.innerHTML += `<p>Testing ${name}...</p>`;
                const response = await fetch(`/api/${endpoint}`);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                statusDiv.innerHTML += `<p>✅ ${name} - Success (${JSON.stringify(data).length} chars)</p>`;
                
                resultsDiv.innerHTML += `<h3>${name}</h3><pre>${JSON.stringify(data, null, 2)}</pre>`;
                return data;
            } catch (error) {
                statusDiv.innerHTML += `<p>❌ ${name} - Error: ${error.message}</p>`;
                console.error(`Error testing ${name}:`, error);
                return null;
            }
        }
        
        async function runTests() {
            statusDiv.innerHTML = '<h2>Running API Tests...</h2>';
            
            await testAPI('progressive-overload', 'Progressive Overload');
            await testAPI('volume-progression', 'Volume Progression');
            await testAPI('personal-records', 'Personal Records');
            await testAPI('training-consistency', 'Training Consistency');
            await testAPI('strength-balance', 'Strength Balance');
            await testAPI('exercises', 'Exercises List');
            
            // Strong-inspired endpoints
            await testAPI('records', 'Personal Records Table');
            await testAPI('calendar', 'Training Calendar');
            await testAPI('muscle-balance', 'Muscle Balance');
            await testAPI('measurements', 'Body Measurements');
            await testAPI('training-streak', 'Training Streak');
            
            statusDiv.innerHTML += '<h2>Tests Complete!</h2>';
        }
        
        // Run tests when page loads
        document.addEventListener('DOMContentLoaded', runTests);
    </script>
</body>
</html>

```

# docker-compose.yml

```yml
services:
  lifting:
    build: .
    container_name: lifting
    restart: unless-stopped
    ports:
      - "8069:8069"
    env_file:
      - .env
    environment:
      # Fallbacks if not in .env
      - HOST=0.0.0.0
      - PORT=8069
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./dashboard_layout.yaml:/app/dashboard_layout.yaml:ro
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://127.0.0.1:8069/health"]
      interval: 30s
      timeout: 5s
      retries: 3

```

# Dockerfile

```
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# System deps (add build tools only if needed for future libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app ./app
COPY dashboard_layout.yaml ./
COPY icon.svg ./
COPY readme.md ./

# Create data & logs volume mount points
RUN mkdir -p data/uploads logs
VOLUME ["/app/data", "/app/logs"]

# Expose default port
EXPOSE 8069

# Provide a non-root user for better security
RUN useradd -u 10001 -m appuser
USER appuser

# Default envs (can be overridden at runtime)
ENV HOST=0.0.0.0 \
    PORT=8069 \
    LOG_LEVEL=INFO

# Healthcheck: hit /health
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://127.0.0.1:8069/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8069"]

```

# icon.svg

This is a file of the type: SVG Image

# logs\app.log

```log
2025-08-26 23:56:29,170 INFO ingested file=strong_20250826_225629.csv inserted_rows=6
2025-08-27 00:37:09,021 INFO ingested file=strong_20250826_233708.csv inserted_rows=50 mode=raw size_bytes=4032
2025-08-27 00:56:03,670 INFO ingested file=strong_20250826_235603.csv inserted_rows=0 mode=raw size_bytes=4032
2025-08-29 14:41:15,592 INFO request path=/ method=GET dur_ms=7.0
2025-08-29 14:41:15,593 INFO request path=/ method=GET dur_ms=7.0
2025-08-29 14:41:15,595 INFO request path=/ method=GET dur_ms=9.0
2025-08-29 14:41:15,741 INFO request path=/static/main.js method=GET dur_ms=103.8
2025-08-29 14:41:15,747 INFO request path=/static/main.js method=GET dur_ms=4.8
2025-08-29 14:41:15,806 INFO request path=/static/main.js method=GET dur_ms=2.0
2025-08-29 14:41:16,633 INFO request path=/api/dashboard method=GET dur_ms=7.0
2025-08-29 14:41:16,649 INFO request path=/api/dashboard method=GET dur_ms=3.8
2025-08-29 14:41:16,792 INFO request path=/favicon.ico method=GET dur_ms=0.0
2025-08-29 14:41:16,804 INFO request path=/api/dashboard method=GET dur_ms=4.7
2025-08-29 14:41:17,508 INFO request path=/favicon.ico method=GET dur_ms=0.0
2025-08-29 14:41:36,955 INFO request path=/health method=GET dur_ms=1.0
2025-08-29 14:42:56,409 INFO ingested file=strong_20250829_134256.csv inserted_rows=15 mode=raw size_bytes=5147
2025-08-29 14:42:56,410 INFO request path=/ingest method=POST dur_ms=55.3
2025-08-29 14:45:58,845 INFO request path=/health method=GET dur_ms=2.1
2025-08-29 14:45:59,033 INFO request path=/favicon.ico method=GET dur_ms=1.1
2025-08-29 14:46:08,899 INFO request path=/ method=GET dur_ms=0.5
2025-08-29 14:46:09,075 INFO request path=/static/main.js method=GET dur_ms=1.6
2025-08-29 14:46:13,344 INFO request path=/ method=GET dur_ms=0.5
2025-08-29 14:46:13,454 INFO request path=/apple-touch-icon.png method=GET dur_ms=0.0
2025-08-29 14:46:13,456 INFO request path=/apple-touch-icon-precomposed.png method=GET dur_ms=0.5
2025-08-29 14:46:14,271 INFO request path=/api/dashboard method=GET dur_ms=2.5
2025-08-29 14:46:20,442 INFO request path=/apple-touch-icon-120x120-precomposed.png method=GET dur_ms=0.0
2025-08-29 14:46:20,622 INFO request path=/apple-touch-icon-120x120.png method=GET dur_ms=0.0
2025-08-29 14:46:46,462 INFO request path=/ method=GET dur_ms=0.0
2025-08-29 14:46:47,862 INFO request path=/api/dashboard method=GET dur_ms=3.0
2025-08-29 14:47:29,339 INFO request path=/api/dashboard method=GET dur_ms=2.1
2025-08-29 14:47:47,172 INFO request path=/api/dashboard method=GET dur_ms=1.5
2025-08-29 14:52:17,611 INFO request path=/ method=GET dur_ms=0.0
2025-08-29 14:52:17,854 INFO request path=/api/dashboard method=GET dur_ms=5.0
2025-08-29 15:49:51,884 INFO request path=/ method=GET dur_ms=0.5
2025-08-29 15:49:51,983 INFO request path=/favicon.ico method=GET dur_ms=0.0
2025-08-29 16:15:00,698 INFO request path=/api/dashboard method=GET dur_ms=5.0
2025-08-29 16:15:00,713 INFO request path=/api/dashboard method=GET dur_ms=4.0
2025-08-29 16:15:02,526 INFO request path=/api/dashboard method=GET dur_ms=3.0
2025-08-29 16:15:26,927 INFO request path=/api/dashboard method=GET dur_ms=3.0
2025-08-29 16:15:27,840 INFO request path=/api/dashboard method=GET dur_ms=2.1
2025-08-29 16:15:28,338 INFO request path=/api/dashboard method=GET dur_ms=3.0
2025-08-29 16:15:28,798 INFO request path=/api/dashboard method=GET dur_ms=2.7
2025-08-29 16:15:29,455 INFO request path=/api/dashboard method=GET dur_ms=2.6
2025-08-29 16:15:31,841 INFO request path=/api/dashboard method=GET dur_ms=4.1
2025-08-29 16:15:32,309 INFO request path=/api/dashboard method=GET dur_ms=4.0
2025-08-29 16:15:32,808 INFO request path=/api/dashboard method=GET dur_ms=3.1
2025-08-29 16:15:35,665 INFO request path=/api/dashboard method=GET dur_ms=4.5
2025-08-29 16:15:39,475 INFO request path=/api/dashboard method=GET dur_ms=7.0
2025-08-29 16:15:40,378 INFO request path=/api/dashboard method=GET dur_ms=2.7
2025-08-29 16:15:41,335 INFO request path=/api/dashboard method=GET dur_ms=3.0
2025-08-29 16:15:41,857 INFO request path=/api/dashboard method=GET dur_ms=4.0
2025-08-29 16:42:18,705 INFO request path=/ method=GET dur_ms=20.1
2025-08-29 16:42:18,965 INFO request path=/api/dashboard method=GET dur_ms=5.5
2025-08-29 16:55:26,029 INFO request path=/ method=GET dur_ms=18.5
2025-08-29 16:55:26,190 INFO request path=/api/dashboard method=GET dur_ms=4.3
2025-08-29 16:55:30,188 INFO request path=/ method=GET dur_ms=0.0
2025-08-29 16:55:30,284 INFO request path=/api/dashboard method=GET dur_ms=4.0
2025-08-29 16:55:32,718 INFO request path=/ method=GET dur_ms=0.5
2025-08-29 16:55:32,780 INFO request path=/static/main.js method=GET dur_ms=36.0
2025-08-29 16:55:33,117 INFO request path=/favicon.ico method=GET dur_ms=0.0
2025-08-29 16:55:33,122 INFO request path=/api/dashboard method=GET dur_ms=3.0
2025-08-29 16:56:25,744 INFO request path=/api/dashboard method=GET dur_ms=2.0
2025-08-29 16:56:25,751 INFO request path=/api/dashboard method=GET dur_ms=2.5
2025-08-29 16:59:00,638 INFO request path=/ method=GET dur_ms=22.6
2025-08-29 16:59:00,705 INFO request path=/static/main.js method=GET dur_ms=53.7
2025-08-29 16:59:01,240 INFO request path=/favicon.ico method=GET dur_ms=0.0
2025-08-29 16:59:01,245 INFO request path=/api/dashboard method=GET dur_ms=2.1
2025-08-29 17:04:09,783 INFO request path=/ method=GET dur_ms=24.4
2025-08-29 17:04:09,849 INFO request path=/static/main.js method=GET dur_ms=46.1
2025-08-29 17:04:10,128 INFO request path=/favicon.ico method=GET dur_ms=1.0
2025-08-29 17:04:10,133 INFO request path=/api/dashboard method=GET dur_ms=3.0
2025-08-29 17:06:58,922 INFO request path=/ method=GET dur_ms=1.0
2025-08-29 17:06:58,964 INFO request path=/static/main.js method=GET dur_ms=5.0
2025-08-29 17:06:59,193 INFO request path=/api/dashboard method=GET dur_ms=2.1
2025-08-29 17:55:20,984 INFO request path=/ method=GET dur_ms=0.0
2025-08-29 17:55:21,077 INFO request path=/static/main.js method=GET dur_ms=0.6
2025-08-29 17:55:21,704 INFO request path=/api/dashboard method=GET dur_ms=3.7
2025-08-29 17:55:21,741 INFO request path=/favicon.ico method=GET dur_ms=0.0
2025-08-29 18:41:06,216 INFO request path=/ method=GET dur_ms=0.5
2025-08-29 18:41:06,388 INFO request path=/api/dashboard method=GET dur_ms=2.0
2025-08-30 09:12:07,150 INFO request path=/ method=GET dur_ms=32.2
2025-08-30 09:12:07,261 INFO request path=/static/main.js method=GET dur_ms=33.9
2025-08-30 09:12:08,660 INFO request path=/api/dashboard method=GET dur_ms=4.0
2025-08-30 09:12:08,896 INFO request path=/api/calendar method=GET dur_ms=1.5
2025-08-30 14:54:59,730 INFO request path=/ method=GET dur_ms=0.0
2025-08-30 14:54:59,821 INFO request path=/static/main.js method=GET dur_ms=1.0
2025-08-30 14:54:59,997 INFO request path=/api/dashboard method=GET dur_ms=3.1
2025-08-30 14:55:00,082 INFO request path=/api/calendar method=GET dur_ms=2.2
2025-08-30 15:29:34,195 INFO request path=/api/calendar method=GET dur_ms=1.0
2025-08-30 15:29:34,826 INFO request path=/api/calendar method=GET dur_ms=1.5
2025-08-30 15:29:35,480 INFO request path=/api/calendar method=GET dur_ms=1.0
2025-08-30 15:29:36,121 INFO request path=/api/calendar method=GET dur_ms=1.0
2025-08-30 15:29:36,414 INFO request path=/api/calendar method=GET dur_ms=1.5
2025-08-30 15:29:37,158 INFO request path=/api/calendar method=GET dur_ms=1.0
2025-08-30 15:29:37,730 INFO request path=/api/calendar method=GET dur_ms=0.5
2025-08-30 15:29:38,218 INFO request path=/api/calendar method=GET dur_ms=0.6
2025-08-30 15:29:43,775 INFO request path=/api/calendar method=GET dur_ms=1.0
2025-08-30 15:29:44,120 INFO request path=/api/calendar method=GET dur_ms=0.5
2025-08-30 15:29:44,509 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-08-30 15:29:45,078 INFO request path=/api/calendar method=GET dur_ms=1.0
2025-08-30 15:29:48,171 INFO request path=/ method=GET dur_ms=0.0
2025-08-30 15:29:48,318 INFO request path=/api/dashboard method=GET dur_ms=2.0
2025-08-30 15:29:48,430 INFO request path=/api/calendar method=GET dur_ms=1.0
2025-08-31 14:34:10,335 INFO request path=/favicon.ico method=GET dur_ms=0.0
2025-08-31 18:05:09,751 INFO request path=/ method=GET dur_ms=0.5
2025-08-31 18:05:09,792 INFO request path=/static/main.js method=GET dur_ms=3.5
2025-08-31 18:05:10,465 INFO request path=/api/dashboard method=GET dur_ms=6.4
2025-08-31 18:05:10,714 INFO request path=/api/calendar method=GET dur_ms=3.1
2025-08-31 18:05:57,771 INFO request path=/workout/2025-08-29 method=GET dur_ms=2.5
2025-08-31 18:05:58,107 INFO request path=/api/dashboard method=GET dur_ms=3.5
2025-08-31 18:05:58,325 INFO request path=/api/calendar method=GET dur_ms=1.0
2025-08-31 18:05:59,123 INFO request path=/api/workout/2025-08-29 method=GET dur_ms=2.0
2025-08-31 18:11:18,923 INFO request path=/ method=GET dur_ms=6.7
2025-08-31 18:11:20,618 INFO request path=/api/dashboard method=GET dur_ms=4.0
2025-08-31 18:11:20,787 INFO request path=/api/calendar method=GET dur_ms=3.0
2025-08-31 18:12:40,808 INFO request path=/api/calendar method=GET dur_ms=1.0
2025-08-31 18:13:09,143 INFO request path=/ method=GET dur_ms=6.0
2025-08-31 18:13:09,471 INFO request path=/static/main.js method=GET dur_ms=306.7
2025-08-31 18:13:10,118 INFO request path=/favicon.ico method=GET dur_ms=1.0
2025-08-31 18:13:10,123 INFO request path=/api/dashboard method=GET dur_ms=4.0
2025-08-31 18:13:10,320 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-08-31 18:13:14,701 INFO request path=/api/workout/2024-11-28 method=GET dur_ms=2.0
2025-08-31 18:13:16,975 INFO request path=/api/workout/2025-08-22 method=GET dur_ms=3.6
2025-08-31 18:13:20,237 INFO request path=/api/workout/2025-08-19 method=GET dur_ms=2.1
2025-08-31 18:13:28,874 INFO request path=/api/workout/2025-08-26 method=GET dur_ms=2.2
2025-08-31 18:14:01,110 INFO request path=/api/workout/2025-08-22 method=GET dur_ms=4.0
2025-08-31 18:14:27,874 INFO request path=/ method=GET dur_ms=29.3
2025-08-31 18:14:27,899 INFO request path=/static/main.js method=GET dur_ms=4.0
2025-08-31 18:14:28,679 INFO request path=/favicon.ico method=GET dur_ms=1.0
2025-08-31 18:14:28,684 INFO request path=/api/dashboard method=GET dur_ms=4.5
2025-08-31 18:14:28,867 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-08-31 18:14:32,546 INFO request path=/ method=GET dur_ms=1.9
2025-08-31 18:14:32,576 INFO request path=/static/main.js method=GET dur_ms=2.0
2025-08-31 18:14:33,367 INFO request path=/favicon.ico method=GET dur_ms=0.7
2025-08-31 18:14:33,375 INFO request path=/api/dashboard method=GET dur_ms=4.1
2025-08-31 18:14:33,628 INFO request path=/api/calendar method=GET dur_ms=1.0
2025-08-31 18:14:42,889 INFO request path=/ method=GET dur_ms=38.6
2025-08-31 18:14:42,917 INFO request path=/static/main.js method=GET dur_ms=3.4
2025-08-31 18:14:43,796 INFO request path=/favicon.ico method=GET dur_ms=0.0
2025-08-31 18:14:43,802 INFO request path=/api/dashboard method=GET dur_ms=3.0
2025-08-31 18:14:43,991 INFO request path=/api/calendar method=GET dur_ms=1.0
2025-08-31 18:26:08,165 INFO request path=/ method=GET dur_ms=34.5
2025-08-31 18:26:08,195 INFO request path=/static/main.core.js method=GET dur_ms=3.0
2025-08-31 18:26:08,209 INFO request path=/static/main.charts.js method=GET dur_ms=13.7
2025-08-31 18:26:08,210 INFO request path=/static/main.js method=GET dur_ms=8.2
2025-08-31 18:26:08,212 INFO request path=/static/main.plotly.js method=GET dur_ms=9.2
2025-08-31 18:26:08,920 INFO request path=/api/dashboard method=GET dur_ms=3.0
2025-08-31 18:26:08,925 INFO request path=/health method=GET dur_ms=5.3
2025-08-31 18:26:08,926 INFO request path=/api/training-streak method=GET dur_ms=5.9
2025-08-31 18:26:08,931 INFO request path=/favicon.ico method=GET dur_ms=0.6
2025-08-31 18:26:20,258 INFO request path=/ method=GET dur_ms=0.0
2025-08-31 18:26:20,275 INFO request path=/static/main.core.js method=GET dur_ms=4.0
2025-08-31 18:26:20,277 INFO request path=/static/main.charts.js method=GET dur_ms=4.5
2025-08-31 18:26:20,282 INFO request path=/static/main.plotly.js method=GET dur_ms=2.5
2025-08-31 18:26:20,284 INFO request path=/static/main.js method=GET dur_ms=4.5
2025-08-31 18:26:20,972 INFO request path=/api/dashboard method=GET dur_ms=5.0
2025-08-31 18:26:20,976 INFO request path=/health method=GET dur_ms=6.0
2025-08-31 18:26:20,977 INFO request path=/api/training-streak method=GET dur_ms=5.0
2025-08-31 18:26:20,980 INFO request path=/favicon.ico method=GET dur_ms=1.0
2025-08-31 18:27:54,834 INFO request path=/ method=GET dur_ms=1.0
2025-08-31 18:27:54,868 INFO request path=/static/main.core.js method=GET dur_ms=2.5
2025-08-31 18:27:54,872 INFO request path=/static/main.charts.js method=GET dur_ms=3.0
2025-08-31 18:27:54,875 INFO request path=/static/main.js method=GET dur_ms=6.0
2025-08-31 18:27:54,875 INFO request path=/static/main.plotly.js method=GET dur_ms=6.0
2025-08-31 18:27:55,313 INFO request path=/api/dashboard method=GET dur_ms=3.0
2025-08-31 18:27:55,318 INFO request path=/health method=GET dur_ms=4.5
2025-08-31 18:27:55,319 INFO request path=/api/training-streak method=GET dur_ms=3.5
2025-08-31 18:32:24,721 INFO request path=/ method=GET dur_ms=1.5
2025-08-31 18:32:24,754 INFO request path=/static/main.core.js method=GET dur_ms=7.5
2025-08-31 18:32:24,759 INFO request path=/static/main.js method=GET dur_ms=11.5
2025-08-31 18:32:24,771 INFO request path=/static/main.charts.js method=GET dur_ms=22.6
2025-08-31 18:32:24,772 INFO request path=/static/main.plotly.js method=GET dur_ms=22.9
2025-08-31 18:32:25,346 INFO request path=/api/dashboard method=GET dur_ms=6.0
2025-08-31 18:32:25,347 INFO request path=/health method=GET dur_ms=3.0
2025-08-31 18:32:25,351 INFO request path=/api/training-streak method=GET dur_ms=3.6
2025-08-31 18:33:05,042 INFO request path=/static/main.js method=GET dur_ms=2.0
2025-08-31 18:38:11,056 INFO request path=/ method=GET dur_ms=1.0
2025-08-31 18:38:11,125 INFO request path=/static/main.core.js method=GET dur_ms=13.1
2025-08-31 18:38:11,127 INFO request path=/static/main.plotly.js method=GET dur_ms=6.0
2025-08-31 18:38:11,128 INFO request path=/static/main.js method=GET dur_ms=7.0
2025-08-31 18:38:11,128 INFO request path=/static/main.charts.js method=GET dur_ms=6.0
2025-08-31 18:38:11,970 INFO request path=/favicon.ico method=GET dur_ms=0.0
2025-08-31 18:38:11,995 INFO request path=/api/dashboard method=GET dur_ms=5.9
2025-08-31 18:38:12,003 INFO request path=/api/training-streak method=GET dur_ms=6.0
2025-08-31 18:38:12,006 INFO request path=/health method=GET dur_ms=9.2
2025-08-31 18:38:53,381 INFO request path=/ method=GET dur_ms=1.0
2025-08-31 18:38:53,416 INFO request path=/static/main.core.js method=GET dur_ms=1.5
2025-08-31 18:38:53,426 INFO request path=/static/main.charts.js method=GET dur_ms=7.6
2025-08-31 18:38:53,428 INFO request path=/static/main.plotly.js method=GET dur_ms=3.1
2025-08-31 18:38:53,429 INFO request path=/static/main.js method=GET dur_ms=7.1
2025-08-31 18:38:54,337 INFO request path=/favicon.ico method=GET dur_ms=1.0
2025-08-31 18:38:54,349 INFO request path=/api/dashboard method=GET dur_ms=3.0
2025-08-31 18:38:54,354 INFO request path=/health method=GET dur_ms=4.0
2025-08-31 18:38:54,357 INFO request path=/api/training-streak method=GET dur_ms=6.5
2025-08-31 18:42:53,477 INFO request path=/ method=GET dur_ms=1.0
2025-08-31 18:42:53,523 INFO request path=/static/main.core.js method=GET dur_ms=2.0
2025-08-31 18:42:53,529 INFO request path=/static/main.charts.js method=GET dur_ms=6.0
2025-08-31 18:42:53,534 INFO request path=/static/main.js method=GET dur_ms=5.1
2025-08-31 18:42:53,540 INFO request path=/static/main.plotly.js method=GET dur_ms=12.2
2025-08-31 18:42:54,451 INFO request path=/favicon.ico method=GET dur_ms=1.3
2025-08-31 18:42:54,469 INFO request path=/api/dashboard method=GET dur_ms=5.6
2025-08-31 18:42:54,476 INFO request path=/api/training-streak method=GET dur_ms=4.0
2025-08-31 18:42:54,477 INFO request path=/health method=GET dur_ms=5.0
2025-08-31 18:42:57,708 INFO request path=/ method=GET dur_ms=0.5
2025-08-31 18:42:57,757 INFO request path=/static/main.js method=GET dur_ms=2.2
2025-08-31 18:42:57,761 INFO request path=/static/main.core.js method=GET dur_ms=11.9
2025-08-31 18:42:57,763 INFO request path=/static/main.charts.js method=GET dur_ms=10.2
2025-08-31 18:42:57,765 INFO request path=/static/main.plotly.js method=GET dur_ms=8.6
2025-08-31 18:42:58,596 INFO request path=/favicon.ico method=GET dur_ms=0.9
2025-08-31 18:42:58,610 INFO request path=/api/dashboard method=GET dur_ms=2.5
2025-08-31 18:42:58,616 INFO request path=/api/training-streak method=GET dur_ms=3.0
2025-08-31 18:42:58,617 INFO request path=/health method=GET dur_ms=4.0
2025-08-31 18:43:01,713 INFO request path=/api/workout/2025-08-29 method=GET dur_ms=2.5
2025-08-31 18:43:08,512 INFO request path=/ method=GET dur_ms=1.0
2025-08-31 18:43:08,571 INFO request path=/static/main.charts.js method=GET dur_ms=4.2
2025-08-31 18:43:08,572 INFO request path=/static/main.js method=GET dur_ms=4.8
2025-08-31 18:43:09,155 INFO request path=/api/dashboard method=GET dur_ms=4.1
2025-08-31 18:43:09,157 INFO request path=/health method=GET dur_ms=1.0
2025-08-31 18:43:09,161 INFO request path=/api/training-streak method=GET dur_ms=4.0
2025-08-31 18:44:51,229 INFO request path=/ method=GET dur_ms=0.9
2025-08-31 18:44:51,286 INFO request path=/static/main.core.js method=GET dur_ms=2.0
2025-08-31 18:44:51,298 INFO request path=/static/main.charts.js method=GET dur_ms=10.5
2025-08-31 18:44:51,299 INFO request path=/static/main.plotly.js method=GET dur_ms=4.0
2025-08-31 18:44:51,301 INFO request path=/static/main.js method=GET dur_ms=6.5
2025-08-31 18:44:52,020 INFO request path=/favicon.ico method=GET dur_ms=0.0
2025-08-31 18:44:52,038 INFO request path=/api/dashboard method=GET dur_ms=3.0
2025-08-31 18:44:52,044 INFO request path=/api/training-streak method=GET dur_ms=4.5
2025-08-31 18:44:52,047 INFO request path=/health method=GET dur_ms=7.6
2025-09-01 09:28:30,163 INFO middleware_enter path=/ method=GET
2025-09-01 09:28:30,163 INFO middleware_enter path=/health method=GET
2025-09-01 09:28:30,196 INFO request path=/ method=GET dur_ms=33.8
2025-09-01 09:28:30,196 INFO request path=/health method=GET dur_ms=32.7
2025-09-01 09:28:30,268 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 09:28:30,271 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 09:28:30,275 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 09:28:30,321 INFO request path=/static/main.core.js method=GET dur_ms=52.0
2025-09-01 09:28:30,325 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 09:28:30,326 INFO request path=/static/main.charts.js method=GET dur_ms=53.8
2025-09-01 09:28:30,328 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 09:28:30,329 INFO request path=/static/main.js method=GET dur_ms=53.9
2025-09-01 09:28:30,331 INFO request path=/favicon.ico method=GET dur_ms=3.4
2025-09-01 09:28:30,332 INFO request path=/static/main.plotly.js method=GET dur_ms=7.4
2025-09-01 09:28:31,310 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 09:28:31,311 INFO request path=/favicon.ico method=GET dur_ms=1.0
2025-09-01 09:28:31,314 INFO middleware_enter path=/health method=GET
2025-09-01 09:28:31,316 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:28:31,317 INFO request path=/health method=GET dur_ms=2.6
2025-09-01 09:28:31,321 INFO middleware_enter path=/api/training-streak method=GET
2025-09-01 09:28:31,321 INFO request path=/api/dashboard method=GET dur_ms=5.0
2025-09-01 09:28:31,324 INFO request path=/api/training-streak method=GET dur_ms=3.4
2025-09-01 09:28:33,072 INFO middleware_enter path=/health method=GET
2025-09-01 09:28:33,073 INFO request path=/health method=GET dur_ms=1.6
2025-09-01 09:29:03,722 INFO middleware_enter path=/ method=GET
2025-09-01 09:29:03,723 INFO request path=/ method=GET dur_ms=1.0
2025-09-01 09:29:04,211 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:29:04,215 INFO request path=/api/dashboard method=GET dur_ms=4.0
2025-09-01 09:29:04,216 INFO middleware_enter path=/health method=GET
2025-09-01 09:29:04,216 INFO middleware_enter path=/api/training-streak method=GET
2025-09-01 09:29:04,219 INFO request path=/health method=GET dur_ms=3.0
2025-09-01 09:29:04,220 INFO request path=/api/training-streak method=GET dur_ms=4.0
2025-09-01 09:29:08,805 INFO middleware_enter path=/workout/2025-08-29 method=GET
2025-09-01 09:29:08,806 INFO request path=/workout/2025-08-29 method=GET dur_ms=1.0
2025-09-01 09:29:09,253 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:29:09,256 INFO request path=/api/dashboard method=GET dur_ms=3.1
2025-09-01 09:29:09,259 INFO middleware_enter path=/health method=GET
2025-09-01 09:29:09,260 INFO middleware_enter path=/api/training-streak method=GET
2025-09-01 09:29:09,262 INFO request path=/health method=GET dur_ms=3.0
2025-09-01 09:29:09,263 INFO request path=/api/training-streak method=GET dur_ms=3.0
2025-09-01 09:30:27,981 INFO middleware_enter path=/ method=GET
2025-09-01 09:30:27,988 INFO request path=/ method=GET dur_ms=7.0
2025-09-01 09:30:28,601 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:30:28,605 INFO request path=/api/dashboard method=GET dur_ms=4.0
2025-09-01 09:30:28,607 INFO middleware_enter path=/health method=GET
2025-09-01 09:30:28,610 INFO middleware_enter path=/api/training-streak method=GET
2025-09-01 09:30:28,611 INFO request path=/health method=GET dur_ms=4.0
2025-09-01 09:30:28,617 INFO request path=/api/training-streak method=GET dur_ms=6.6
2025-09-01 09:32:06,557 INFO middleware_enter path=/ method=GET
2025-09-01 09:32:06,559 INFO request path=/ method=GET dur_ms=2.0
2025-09-01 09:32:06,599 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 09:32:06,608 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 09:32:06,609 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 09:32:06,609 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 09:32:06,650 INFO request path=/static/main.core.js method=GET dur_ms=50.5
2025-09-01 09:32:06,655 INFO request path=/static/main.js method=GET dur_ms=45.5
2025-09-01 09:32:06,656 INFO request path=/static/main.plotly.js method=GET dur_ms=47.5
2025-09-01 09:32:06,657 INFO request path=/static/main.charts.js method=GET dur_ms=47.5
2025-09-01 09:32:08,288 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 09:32:08,289 INFO request path=/favicon.ico method=GET dur_ms=1.0
2025-09-01 09:32:08,298 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:32:08,303 INFO middleware_enter path=/health method=GET
2025-09-01 09:32:08,304 INFO request path=/api/dashboard method=GET dur_ms=6.6
2025-09-01 09:32:08,306 INFO middleware_enter path=/api/training-streak method=GET
2025-09-01 09:32:08,310 INFO request path=/health method=GET dur_ms=7.0
2025-09-01 09:32:08,312 INFO request path=/api/training-streak method=GET dur_ms=6.0
2025-09-01 09:32:08,551 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:32:08,554 INFO request path=/api/calendar method=GET dur_ms=3.0
2025-09-01 09:35:22,404 INFO middleware_enter path=/workout/2024-11-28 method=GET
2025-09-01 09:35:22,405 INFO request path=/workout/2024-11-28 method=GET dur_ms=1.0
2025-09-01 09:35:22,445 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 09:35:22,454 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 09:35:22,454 INFO request path=/static/main.core.js method=GET dur_ms=8.4
2025-09-01 09:35:22,457 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 09:35:22,457 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 09:35:22,461 INFO request path=/static/main.charts.js method=GET dur_ms=7.5
2025-09-01 09:35:22,469 INFO request path=/static/main.js method=GET dur_ms=12.0
2025-09-01 09:35:22,471 INFO request path=/static/main.plotly.js method=GET dur_ms=14.0
2025-09-01 09:35:23,228 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 09:35:23,230 INFO request path=/favicon.ico method=GET dur_ms=2.0
2025-09-01 09:35:23,240 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:35:23,242 INFO request path=/api/dashboard method=GET dur_ms=2.0
2025-09-01 09:35:23,244 INFO middleware_enter path=/health method=GET
2025-09-01 09:35:23,246 INFO request path=/health method=GET dur_ms=2.0
2025-09-01 09:35:23,247 INFO middleware_enter path=/api/training-streak method=GET
2025-09-01 09:35:23,253 INFO request path=/api/training-streak method=GET dur_ms=6.0
2025-09-01 09:35:23,579 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:35:23,582 INFO request path=/api/calendar method=GET dur_ms=3.6
2025-09-01 09:35:38,744 INFO middleware_enter path=/workout/2025-08-29 method=GET
2025-09-01 09:35:38,745 INFO request path=/workout/2025-08-29 method=GET dur_ms=1.0
2025-09-01 09:35:38,776 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 09:35:38,777 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 09:35:38,781 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 09:35:38,782 INFO request path=/static/main.core.js method=GET dur_ms=6.0
2025-09-01 09:35:38,786 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 09:35:38,788 INFO request path=/static/main.charts.js method=GET dur_ms=11.0
2025-09-01 09:35:38,790 INFO request path=/static/main.js method=GET dur_ms=9.0
2025-09-01 09:35:38,792 INFO request path=/static/main.plotly.js method=GET dur_ms=6.0
2025-09-01 09:35:39,544 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:35:39,548 INFO request path=/api/dashboard method=GET dur_ms=4.5
2025-09-01 09:35:39,551 INFO middleware_enter path=/health method=GET
2025-09-01 09:35:39,552 INFO middleware_enter path=/api/training-streak method=GET
2025-09-01 09:35:39,556 INFO request path=/health method=GET dur_ms=5.0
2025-09-01 09:35:39,557 INFO request path=/api/training-streak method=GET dur_ms=5.0
2025-09-01 09:35:39,564 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 09:35:39,566 INFO request path=/favicon.ico method=GET dur_ms=2.0
2025-09-01 09:35:39,739 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:35:39,741 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-09-01 09:38:26,550 INFO middleware_enter path=/workout/2025-08-29 method=GET
2025-09-01 09:38:26,550 INFO request path=/workout/2025-08-29 method=GET dur_ms=0.5
2025-09-01 09:38:26,613 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 09:38:26,615 INFO request path=/static/main.core.js method=GET dur_ms=2.5
2025-09-01 09:38:26,617 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 09:38:26,619 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 09:38:26,621 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 09:38:26,622 INFO request path=/static/main.charts.js method=GET dur_ms=4.9
2025-09-01 09:38:26,631 INFO request path=/static/main.js method=GET dur_ms=11.7
2025-09-01 09:38:26,633 INFO request path=/static/main.plotly.js method=GET dur_ms=11.7
2025-09-01 09:38:27,836 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 09:38:27,837 INFO request path=/favicon.ico method=GET dur_ms=1.0
2025-09-01 09:38:27,839 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:38:27,843 INFO request path=/api/dashboard method=GET dur_ms=4.0
2025-09-01 09:38:27,860 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:38:27,864 INFO request path=/api/dashboard method=GET dur_ms=4.0
2025-09-01 09:38:28,052 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:38:28,055 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-09-01 09:38:28,210 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:38:28,211 INFO request path=/api/calendar method=GET dur_ms=1.0
2025-09-01 09:38:28,873 INFO middleware_enter path=/api/workout/2025-08-29 method=GET
2025-09-01 09:38:28,876 INFO request path=/api/workout/2025-08-29 method=GET dur_ms=3.0
2025-09-01 09:38:43,063 INFO middleware_enter path=/api/workout/2025-08-29 method=GET
2025-09-01 09:38:43,068 INFO request path=/api/workout/2025-08-29 method=GET dur_ms=4.9
2025-09-01 09:44:18,643 INFO middleware_enter path=/ method=GET
2025-09-01 09:44:18,644 INFO request path=/ method=GET dur_ms=1.0
2025-09-01 09:44:18,691 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 09:44:18,694 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 09:44:18,700 INFO request path=/static/main.core.js method=GET dur_ms=9.4
2025-09-01 09:44:18,703 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 09:44:18,704 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 09:44:18,705 INFO request path=/static/main.charts.js method=GET dur_ms=11.0
2025-09-01 09:44:18,708 INFO request path=/static/main.plotly.js method=GET dur_ms=5.6
2025-09-01 09:44:18,710 INFO request path=/static/main.js method=GET dur_ms=6.6
2025-09-01 09:44:19,424 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 09:44:19,425 INFO request path=/favicon.ico method=GET dur_ms=1.0
2025-09-01 09:44:19,426 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:44:19,432 INFO request path=/api/dashboard method=GET dur_ms=6.0
2025-09-01 09:44:19,461 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:44:19,466 INFO request path=/api/dashboard method=GET dur_ms=5.0
2025-09-01 09:44:19,671 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:44:19,672 INFO request path=/api/calendar method=GET dur_ms=1.0
2025-09-01 09:44:19,803 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:44:19,805 INFO request path=/api/calendar method=GET dur_ms=1.7
2025-09-01 09:48:43,561 INFO middleware_enter path=/ method=GET
2025-09-01 09:48:43,562 INFO request path=/ method=GET dur_ms=1.0
2025-09-01 09:48:43,613 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 09:48:43,618 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 09:48:43,637 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 09:48:43,639 INFO request path=/static/main.core.js method=GET dur_ms=26.2
2025-09-01 09:48:43,642 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 09:48:43,643 INFO request path=/static/main.charts.js method=GET dur_ms=24.6
2025-09-01 09:48:43,654 INFO request path=/static/main.js method=GET dur_ms=16.5
2025-09-01 09:48:43,655 INFO request path=/static/main.plotly.js method=GET dur_ms=13.5
2025-09-01 09:48:44,545 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 09:48:44,546 INFO request path=/favicon.ico method=GET dur_ms=1.0
2025-09-01 09:48:44,549 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:48:44,554 INFO request path=/api/dashboard method=GET dur_ms=4.7
2025-09-01 09:48:44,573 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:48:44,578 INFO request path=/api/dashboard method=GET dur_ms=4.0
2025-09-01 09:48:44,819 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:48:44,821 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-09-01 09:48:44,986 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:48:44,988 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-09-01 09:49:45,240 INFO middleware_enter path=/ method=GET
2025-09-01 09:49:45,241 INFO request path=/ method=GET dur_ms=1.0
2025-09-01 09:49:45,277 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 09:49:45,279 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 09:49:45,289 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 09:49:45,290 INFO request path=/static/main.core.js method=GET dur_ms=13.5
2025-09-01 09:49:45,293 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 09:49:45,294 INFO request path=/static/main.charts.js method=GET dur_ms=14.9
2025-09-01 09:49:45,295 INFO request path=/static/main.js method=GET dur_ms=6.0
2025-09-01 09:49:45,297 INFO request path=/static/main.plotly.js method=GET dur_ms=4.5
2025-09-01 09:49:46,120 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 09:49:46,122 INFO request path=/favicon.ico method=GET dur_ms=1.0
2025-09-01 09:49:46,124 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:49:46,127 INFO request path=/api/dashboard method=GET dur_ms=3.0
2025-09-01 09:49:46,146 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:49:46,152 INFO request path=/api/dashboard method=GET dur_ms=5.5
2025-09-01 09:49:46,370 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:49:46,373 INFO request path=/api/calendar method=GET dur_ms=3.0
2025-09-01 09:49:46,522 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:49:46,525 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-09-01 09:53:19,522 INFO middleware_enter path=/ method=GET
2025-09-01 09:53:19,523 INFO request path=/ method=GET dur_ms=1.0
2025-09-01 09:53:19,569 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 09:53:19,572 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 09:53:19,575 INFO request path=/static/main.core.js method=GET dur_ms=5.1
2025-09-01 09:53:19,576 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 09:53:19,577 INFO request path=/static/main.charts.js method=GET dur_ms=5.0
2025-09-01 09:53:19,578 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 09:53:19,582 INFO request path=/static/main.js method=GET dur_ms=6.6
2025-09-01 09:53:19,586 INFO request path=/static/main.plotly.js method=GET dur_ms=8.6
2025-09-01 09:53:20,328 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:53:20,333 INFO request path=/api/dashboard method=GET dur_ms=5.5
2025-09-01 09:53:20,338 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 09:53:20,339 INFO request path=/favicon.ico method=GET dur_ms=1.0
2025-09-01 09:53:20,349 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:53:20,354 INFO request path=/api/dashboard method=GET dur_ms=5.0
2025-09-01 09:53:20,561 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:53:20,563 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-09-01 09:53:20,784 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:53:20,786 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-09-01 09:55:32,888 INFO middleware_enter path=/ method=GET
2025-09-01 09:55:32,888 INFO request path=/ method=GET dur_ms=0.0
2025-09-01 09:55:32,961 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 09:55:32,963 INFO request path=/static/main.core.js method=GET dur_ms=2.1
2025-09-01 09:55:32,967 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 09:55:32,968 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 09:55:32,969 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 09:55:32,970 INFO request path=/static/main.charts.js method=GET dur_ms=2.9
2025-09-01 09:55:32,972 INFO request path=/static/main.js method=GET dur_ms=4.3
2025-09-01 09:55:32,973 INFO request path=/static/main.plotly.js method=GET dur_ms=4.4
2025-09-01 09:55:33,565 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:55:33,567 INFO request path=/api/dashboard method=GET dur_ms=2.0
2025-09-01 09:55:33,573 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:55:33,575 INFO request path=/api/dashboard method=GET dur_ms=1.9
2025-09-01 09:55:33,678 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:55:33,680 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-09-01 09:55:33,711 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:55:33,712 INFO request path=/api/calendar method=GET dur_ms=1.1
2025-09-01 09:55:33,853 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 09:55:33,854 INFO request path=/favicon.ico method=GET dur_ms=1.0
2025-09-01 09:57:03,118 INFO middleware_enter path=/ method=GET
2025-09-01 09:57:03,147 INFO request path=/ method=GET dur_ms=28.1
2025-09-01 09:57:03,184 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 09:57:03,186 INFO request path=/favicon.svg method=GET dur_ms=1.9
2025-09-01 09:57:03,195 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 09:57:03,226 INFO request path=/static/main.core.js method=GET dur_ms=30.6
2025-09-01 09:57:03,327 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 09:57:03,327 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 09:57:03,329 INFO request path=/static/main.charts.js method=GET dur_ms=3.5
2025-09-01 09:57:03,330 INFO request path=/static/main.js method=GET dur_ms=3.5
2025-09-01 09:57:03,370 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 09:57:03,372 INFO request path=/static/main.plotly.js method=GET dur_ms=1.6
2025-09-01 09:57:03,508 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:57:03,511 INFO request path=/api/dashboard method=GET dur_ms=3.1
2025-09-01 09:57:03,514 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:57:03,516 INFO request path=/api/dashboard method=GET dur_ms=2.0
2025-09-01 09:57:03,558 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:57:03,560 INFO request path=/api/calendar method=GET dur_ms=2.4
2025-09-01 09:57:03,566 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 09:57:03,567 INFO request path=/favicon.ico method=GET dur_ms=1.5
2025-09-01 09:57:03,641 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:57:03,648 INFO request path=/api/calendar method=GET dur_ms=7.5
2025-09-01 09:59:21,562 INFO middleware_enter path=/ method=GET
2025-09-01 09:59:21,605 INFO request path=/ method=GET dur_ms=43.4
2025-09-01 09:59:21,635 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 09:59:21,638 INFO request path=/favicon.svg method=GET dur_ms=4.4
2025-09-01 09:59:21,641 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 09:59:21,645 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 09:59:21,647 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 09:59:21,648 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 09:59:21,688 INFO request path=/static/main.core.js method=GET dur_ms=46.9
2025-09-01 09:59:21,690 INFO request path=/static/main.plotly.js method=GET dur_ms=43.1
2025-09-01 09:59:21,691 INFO request path=/static/main.js method=GET dur_ms=43.0
2025-09-01 09:59:21,691 INFO request path=/static/main.charts.js method=GET dur_ms=46.7
2025-09-01 09:59:22,392 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 09:59:22,393 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 09:59:22,394 INFO request path=/apple-touch-icon.png method=GET dur_ms=2.0
2025-09-01 09:59:22,396 INFO request path=/favicon.svg method=GET dur_ms=3.0
2025-09-01 09:59:22,397 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:59:22,399 INFO request path=/api/dashboard method=GET dur_ms=2.6
2025-09-01 09:59:22,418 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 09:59:22,425 INFO request path=/api/dashboard method=GET dur_ms=6.6
2025-09-01 09:59:22,741 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:59:22,744 INFO request path=/api/calendar method=GET dur_ms=2.2
2025-09-01 09:59:22,911 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:59:22,913 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-09-01 09:59:30,170 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:59:30,172 INFO request path=/api/calendar method=GET dur_ms=2.7
2025-09-01 09:59:30,757 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 09:59:30,759 INFO request path=/api/calendar method=GET dur_ms=1.9
2025-09-01 10:02:49,773 INFO middleware_enter path=/ method=GET
2025-09-01 10:02:49,819 INFO request path=/ method=GET dur_ms=45.7
2025-09-01 10:02:49,859 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 10:02:49,862 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 10:02:49,863 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 10:02:49,868 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 10:02:49,919 INFO request path=/static/main.core.js method=GET dur_ms=59.6
2025-09-01 10:02:49,922 INFO request path=/static/main.js method=GET dur_ms=58.6
2025-09-01 10:02:49,923 INFO request path=/static/main.charts.js method=GET dur_ms=60.7
2025-09-01 10:02:49,924 INFO request path=/static/main.plotly.js method=GET dur_ms=55.5
2025-09-01 10:02:50,234 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 10:02:50,236 INFO request path=/favicon.svg method=GET dur_ms=1.0
2025-09-01 10:02:50,338 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 10:02:50,341 INFO request path=/api/dashboard method=GET dur_ms=3.0
2025-09-01 10:02:50,357 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 10:02:50,362 INFO request path=/api/dashboard method=GET dur_ms=5.0
2025-09-01 10:02:50,529 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 10:02:50,530 INFO request path=/api/calendar method=GET dur_ms=1.0
2025-09-01 10:02:50,688 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 10:02:50,690 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-09-01 10:05:19,338 INFO middleware_enter path=/ method=GET
2025-09-01 10:05:19,369 INFO request path=/ method=GET dur_ms=31.7
2025-09-01 10:05:19,427 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 10:05:19,429 INFO request path=/favicon.svg method=GET dur_ms=2.0
2025-09-01 10:05:19,431 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 10:05:19,432 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 10:05:19,435 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 10:05:19,439 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 10:05:19,440 INFO request path=/static/main.charts.js method=GET dur_ms=7.5
2025-09-01 10:05:19,442 INFO request path=/static/main.core.js method=GET dur_ms=11.1
2025-09-01 10:05:19,446 INFO request path=/static/main.js method=GET dur_ms=11.6
2025-09-01 10:05:19,450 INFO request path=/static/main.plotly.js method=GET dur_ms=11.6
2025-09-01 10:05:20,188 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 10:05:20,191 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 10:05:20,193 INFO request path=/apple-touch-icon.png method=GET dur_ms=5.5
2025-09-01 10:05:20,195 INFO request path=/favicon.svg method=GET dur_ms=4.5
2025-09-01 10:05:26,091 INFO middleware_enter path=/ method=GET
2025-09-01 10:05:26,093 INFO request path=/ method=GET dur_ms=2.5
2025-09-01 10:05:26,121 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 10:05:26,123 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 10:05:26,126 INFO request path=/favicon.svg method=GET dur_ms=5.0
2025-09-01 10:05:26,128 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 10:05:26,129 INFO request path=/static/main.core.js method=GET dur_ms=6.5
2025-09-01 10:05:26,131 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 10:05:26,132 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 10:05:26,133 INFO request path=/static/main.charts.js method=GET dur_ms=4.5
2025-09-01 10:05:26,136 INFO request path=/static/main.js method=GET dur_ms=4.5
2025-09-01 10:05:26,138 INFO request path=/static/main.plotly.js method=GET dur_ms=5.5
2025-09-01 10:05:26,826 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 10:05:26,828 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 10:05:26,829 INFO request path=/apple-touch-icon.png method=GET dur_ms=3.0
2025-09-01 10:05:26,830 INFO request path=/favicon.svg method=GET dur_ms=2.5
2025-09-01 10:07:32,129 INFO middleware_enter path=/ method=GET
2025-09-01 10:07:32,160 INFO request path=/ method=GET dur_ms=31.4
2025-09-01 10:07:32,205 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 10:07:32,207 INFO request path=/favicon.svg method=GET dur_ms=2.0
2025-09-01 10:07:32,209 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 10:07:32,209 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 10:07:32,211 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 10:07:32,211 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 10:07:32,220 INFO request path=/static/main.core.js method=GET dur_ms=12.0
2025-09-01 10:07:32,221 INFO request path=/static/main.charts.js method=GET dur_ms=12.1
2025-09-01 10:07:32,222 INFO request path=/static/main.js method=GET dur_ms=10.6
2025-09-01 10:07:32,223 INFO request path=/static/main.plotly.js method=GET dur_ms=12.1
2025-09-01 10:07:32,938 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 10:07:32,940 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 10:07:32,941 INFO request path=/apple-touch-icon.png method=GET dur_ms=3.6
2025-09-01 10:07:32,943 INFO request path=/favicon.svg method=GET dur_ms=2.5
2025-09-01 10:07:32,944 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 10:07:32,949 INFO request path=/api/dashboard method=GET dur_ms=5.0
2025-09-01 10:07:33,161 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 10:07:33,163 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-09-01 10:08:28,826 INFO middleware_enter path=/api/workout/2025-08-26 method=GET
2025-09-01 10:08:28,828 INFO request path=/api/workout/2025-08-26 method=GET dur_ms=2.1
2025-09-01 10:10:16,513 INFO middleware_enter path=/ method=GET
2025-09-01 10:10:16,514 INFO request path=/ method=GET dur_ms=0.6
2025-09-01 10:10:16,644 INFO middleware_enter path=/ method=GET
2025-09-01 10:10:16,644 INFO request path=/ method=GET dur_ms=0.0
2025-09-01 10:10:19,793 INFO middleware_enter path=/ method=GET
2025-09-01 10:10:19,793 INFO request path=/ method=GET dur_ms=0.0
2025-09-01 10:10:19,845 INFO middleware_enter path=/ method=GET
2025-09-01 10:10:19,845 INFO request path=/ method=GET dur_ms=0.0
2025-09-01 10:10:19,939 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 10:10:19,940 INFO request path=/favicon.svg method=GET dur_ms=1.0
2025-09-01 10:10:19,945 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 10:10:19,946 INFO request path=/static/main.core.js method=GET dur_ms=1.3
2025-09-01 10:10:20,082 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 10:10:20,083 INFO request path=/static/main.charts.js method=GET dur_ms=1.0
2025-09-01 10:10:20,120 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 10:10:20,121 INFO request path=/static/main.js method=GET dur_ms=1.1
2025-09-01 10:10:20,176 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 10:10:20,179 INFO request path=/static/main.plotly.js method=GET dur_ms=2.1
2025-09-01 10:10:20,681 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 10:10:20,683 INFO request path=/api/dashboard method=GET dur_ms=2.0
2025-09-01 10:10:20,693 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 10:10:20,694 INFO request path=/api/dashboard method=GET dur_ms=1.7
2025-09-01 10:10:21,065 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 10:10:21,066 INFO request path=/api/calendar method=GET dur_ms=1.0
2025-09-01 10:10:21,072 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 10:10:21,073 INFO request path=/api/calendar method=GET dur_ms=1.1
2025-09-01 10:10:25,619 INFO middleware_enter path=/ method=GET
2025-09-01 10:10:25,619 INFO request path=/ method=GET dur_ms=0.0
2025-09-01 10:10:25,868 INFO middleware_enter path=/robots.txt method=GET
2025-09-01 10:10:25,868 INFO request path=/robots.txt method=GET dur_ms=0.0
2025-09-01 10:10:26,649 INFO middleware_enter path=/ method=GET
2025-09-01 10:10:26,649 INFO request path=/ method=GET dur_ms=0.0
2025-09-01 10:13:32,233 INFO middleware_enter path=/ method=GET
2025-09-01 10:13:32,279 INFO request path=/ method=GET dur_ms=45.8
2025-09-01 10:13:32,312 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 10:13:32,314 INFO request path=/favicon.svg method=GET dur_ms=2.0
2025-09-01 10:13:32,315 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 10:13:32,316 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 10:13:32,319 INFO request path=/static/main.core.js method=GET dur_ms=4.2
2025-09-01 10:13:32,321 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 10:13:32,325 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 10:13:32,327 INFO request path=/static/main.charts.js method=GET dur_ms=10.8
2025-09-01 10:13:32,329 INFO request path=/static/main.js method=GET dur_ms=8.6
2025-09-01 10:13:32,332 INFO request path=/static/main.plotly.js method=GET dur_ms=7.7
2025-09-01 10:13:33,113 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 10:13:33,115 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 10:13:33,116 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 10:13:33,120 INFO request path=/api/dashboard method=GET dur_ms=4.0
2025-09-01 10:13:33,121 INFO request path=/apple-touch-icon.png method=GET dur_ms=7.5
2025-09-01 10:13:33,122 INFO request path=/favicon.svg method=GET dur_ms=7.0
2025-09-01 10:13:33,359 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 10:13:33,360 INFO request path=/api/calendar method=GET dur_ms=1.0
2025-09-01 10:54:48,388 INFO middleware_enter path=/ method=GET
2025-09-01 10:54:48,434 INFO request path=/ method=GET dur_ms=46.1
2025-09-01 10:54:48,482 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 10:54:48,485 INFO request path=/favicon.svg method=GET dur_ms=3.0
2025-09-01 10:54:48,487 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 10:54:48,488 INFO middleware_enter path=/static/main.charts.echarts.js method=GET
2025-09-01 10:54:48,488 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 10:54:48,495 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 10:54:48,497 INFO request path=/static/main.core.js method=GET dur_ms=10.0
2025-09-01 10:54:48,498 INFO request path=/static/main.charts.echarts.js method=GET dur_ms=10.3
2025-09-01 10:54:48,500 INFO request path=/static/main.charts.js method=GET dur_ms=12.3
2025-09-01 10:54:48,501 INFO request path=/static/main.js method=GET dur_ms=6.3
2025-09-01 10:54:48,849 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 10:54:48,851 INFO request path=/apple-touch-icon.png method=GET dur_ms=2.0
2025-09-01 10:54:48,853 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 10:54:48,854 INFO request path=/favicon.svg method=GET dur_ms=1.0
2025-09-01 10:54:48,857 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 10:54:48,863 INFO request path=/api/dashboard method=GET dur_ms=6.1
2025-09-01 10:54:49,101 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 10:54:49,102 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-09-01 11:03:04,478 INFO middleware_enter path=/ method=GET
2025-09-01 11:03:04,484 INFO request path=/ method=GET dur_ms=6.5
2025-09-01 11:03:04,563 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 11:03:04,564 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 11:03:04,567 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 11:03:04,582 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 11:03:04,584 INFO request path=/static/main.core.js method=GET dur_ms=20.5
2025-09-01 11:03:04,595 INFO request path=/static/main.charts.js method=GET dur_ms=30.6
2025-09-01 11:03:04,599 INFO request path=/static/main.js method=GET dur_ms=31.7
2025-09-01 11:03:04,609 INFO request path=/static/main.plotly.js method=GET dur_ms=26.6
2025-09-01 11:03:05,231 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 11:03:05,233 INFO request path=/favicon.svg method=GET dur_ms=2.0
2025-09-01 11:03:05,405 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 11:03:05,412 INFO request path=/api/dashboard method=GET dur_ms=7.1
2025-09-01 11:03:05,704 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 11:03:05,708 INFO request path=/api/calendar method=GET dur_ms=5.0
2025-09-01 11:04:41,520 INFO middleware_enter path=/ method=GET
2025-09-01 11:04:41,521 INFO request path=/ method=GET dur_ms=1.0
2025-09-01 11:04:41,575 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 11:04:41,577 INFO request path=/favicon.svg method=GET dur_ms=2.0
2025-09-01 11:04:41,586 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 11:04:41,587 INFO request path=/static/main.core.js method=GET dur_ms=1.0
2025-09-01 11:04:41,606 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 11:04:41,608 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 11:04:41,608 INFO request path=/static/main.charts.js method=GET dur_ms=2.0
2025-09-01 11:04:41,613 INFO request path=/static/main.js method=GET dur_ms=4.5
2025-09-01 11:04:41,628 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 11:04:41,630 INFO request path=/static/main.plotly.js method=GET dur_ms=2.0
2025-09-01 11:04:41,730 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 11:04:41,735 INFO request path=/api/dashboard method=GET dur_ms=4.5
2025-09-01 11:04:41,819 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 11:04:41,819 INFO request path=/api/calendar method=GET dur_ms=0.0
2025-09-01 11:04:43,617 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 11:04:43,618 INFO request path=/favicon.ico method=GET dur_ms=1.0
2025-09-01 11:04:44,831 INFO middleware_enter path=/ method=GET
2025-09-01 11:04:44,832 INFO request path=/ method=GET dur_ms=1.1
2025-09-01 11:04:44,919 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 11:04:44,920 INFO request path=/favicon.svg method=GET dur_ms=0.6
2025-09-01 11:04:44,929 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 11:04:44,930 INFO request path=/static/main.core.js method=GET dur_ms=1.0
2025-09-01 11:04:45,042 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 11:04:45,044 INFO request path=/static/main.charts.js method=GET dur_ms=1.8
2025-09-01 11:04:45,175 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 11:04:45,176 INFO request path=/static/main.plotly.js method=GET dur_ms=1.0
2025-09-01 11:04:45,197 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 11:04:45,198 INFO request path=/static/main.js method=GET dur_ms=1.0
2025-09-01 11:04:45,449 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 11:04:45,451 INFO request path=/api/dashboard method=GET dur_ms=1.6
2025-09-01 11:04:45,572 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 11:04:45,573 INFO request path=/api/calendar method=GET dur_ms=1.1
2025-09-01 11:05:19,229 INFO middleware_enter path=/ method=GET
2025-09-01 11:05:19,230 INFO request path=/ method=GET dur_ms=1.5
2025-09-01 11:05:19,260 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 11:05:19,261 INFO request path=/favicon.svg method=GET dur_ms=0.6
2025-09-01 11:05:19,265 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 11:05:19,267 INFO request path=/static/main.core.js method=GET dur_ms=2.5
2025-09-01 11:05:19,282 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 11:05:19,283 INFO request path=/static/main.charts.js method=GET dur_ms=1.0
2025-09-01 11:05:19,288 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 11:05:19,290 INFO request path=/static/main.js method=GET dur_ms=1.9
2025-09-01 11:05:19,306 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 11:05:19,308 INFO request path=/static/main.plotly.js method=GET dur_ms=2.1
2025-09-01 11:05:19,709 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 11:05:19,712 INFO request path=/api/dashboard method=GET dur_ms=2.9
2025-09-01 11:05:19,801 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 11:05:19,802 INFO request path=/api/calendar method=GET dur_ms=1.1
2025-09-01 11:05:19,873 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 11:05:19,874 INFO request path=/favicon.ico method=GET dur_ms=0.6
2025-09-01 11:05:32,414 INFO middleware_enter path=/ method=GET
2025-09-01 11:05:32,415 INFO request path=/ method=GET dur_ms=1.1
2025-09-01 11:05:32,443 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 11:05:32,445 INFO request path=/favicon.svg method=GET dur_ms=1.9
2025-09-01 11:05:32,449 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 11:05:32,450 INFO request path=/static/main.core.js method=GET dur_ms=2.0
2025-09-01 11:05:32,475 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 11:05:32,477 INFO request path=/static/main.charts.js method=GET dur_ms=2.0
2025-09-01 11:05:32,506 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 11:05:32,507 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 11:05:32,509 INFO request path=/static/main.js method=GET dur_ms=3.0
2025-09-01 11:05:32,509 INFO request path=/static/main.plotly.js method=GET dur_ms=2.0
2025-09-01 11:05:32,789 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 11:05:32,791 INFO request path=/api/dashboard method=GET dur_ms=2.6
2025-09-01 11:05:32,807 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 11:05:32,808 INFO request path=/favicon.ico method=GET dur_ms=1.0
2025-09-01 11:05:32,838 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 11:05:32,840 INFO request path=/api/calendar method=GET dur_ms=1.1
2025-09-01 11:06:38,121 INFO middleware_enter path=/ method=GET
2025-09-01 11:06:38,122 INFO request path=/ method=GET dur_ms=1.0
2025-09-01 11:06:38,163 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 11:06:38,165 INFO request path=/favicon.svg method=GET dur_ms=2.0
2025-09-01 11:06:38,166 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 11:06:38,176 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 11:06:38,176 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 11:06:38,177 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 11:06:38,179 INFO request path=/static/main.core.js method=GET dur_ms=13.6
2025-09-01 11:06:38,182 INFO request path=/static/main.charts.js method=GET dur_ms=6.4
2025-09-01 11:06:38,183 INFO request path=/static/main.js method=GET dur_ms=5.4
2025-09-01 11:06:38,184 INFO request path=/static/main.plotly.js method=GET dur_ms=7.4
2025-09-01 11:06:38,928 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 11:06:38,929 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 11:06:38,932 INFO request path=/apple-touch-icon.png method=GET dur_ms=4.3
2025-09-01 11:06:38,932 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 11:06:38,938 INFO request path=/api/dashboard method=GET dur_ms=5.9
2025-09-01 11:06:38,939 INFO request path=/favicon.svg method=GET dur_ms=10.3
2025-09-01 11:06:39,146 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 11:06:39,149 INFO request path=/api/calendar method=GET dur_ms=3.6
2025-09-01 11:13:20,586 INFO middleware_enter path=/ method=GET
2025-09-01 11:13:20,587 INFO request path=/ method=GET dur_ms=1.0
2025-09-01 11:13:20,658 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 11:13:20,661 INFO request path=/favicon.svg method=GET dur_ms=2.0
2025-09-01 11:13:20,668 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 11:13:20,670 INFO request path=/static/main.core.js method=GET dur_ms=2.0
2025-09-01 11:13:20,672 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 11:13:20,673 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 11:13:20,674 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 11:13:20,675 INFO request path=/static/main.charts.js method=GET dur_ms=3.4
2025-09-01 11:13:20,679 INFO request path=/static/main.js method=GET dur_ms=6.4
2025-09-01 11:13:20,681 INFO request path=/static/main.plotly.js method=GET dur_ms=7.0
2025-09-01 11:13:21,429 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 11:13:21,431 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 11:13:21,432 INFO request path=/apple-touch-icon.png method=GET dur_ms=3.5
2025-09-01 11:13:21,433 INFO request path=/favicon.svg method=GET dur_ms=2.5
2025-09-01 11:13:21,436 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 11:13:21,439 INFO request path=/api/dashboard method=GET dur_ms=3.0
2025-09-01 11:13:21,761 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 11:13:21,764 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-09-01 11:15:38,600 INFO middleware_enter path=/ method=GET
2025-09-01 11:15:38,601 INFO request path=/ method=GET dur_ms=1.0
2025-09-01 11:15:38,657 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 11:15:38,658 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 11:15:38,659 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 11:15:38,661 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 11:15:38,662 INFO request path=/static/main.core.js method=GET dur_ms=4.6
2025-09-01 11:15:38,664 INFO request path=/static/main.charts.js method=GET dur_ms=6.1
2025-09-01 11:15:38,666 INFO request path=/static/main.js method=GET dur_ms=7.1
2025-09-01 11:15:38,667 INFO request path=/static/main.plotly.js method=GET dur_ms=6.5
2025-09-01 11:15:39,211 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 11:15:39,214 INFO request path=/api/dashboard method=GET dur_ms=2.5
2025-09-01 11:15:39,398 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 11:15:39,400 INFO request path=/api/calendar method=GET dur_ms=1.6
2025-09-01 11:18:41,959 INFO middleware_enter path=/ method=GET
2025-09-01 11:18:41,960 INFO request path=/ method=GET dur_ms=1.0
2025-09-01 11:18:42,011 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 11:18:42,019 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 11:18:42,020 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 11:18:42,021 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 11:18:42,023 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 11:18:42,024 INFO request path=/favicon.svg method=GET dur_ms=13.1
2025-09-01 11:18:42,026 INFO request path=/static/main.core.js method=GET dur_ms=7.5
2025-09-01 11:18:42,033 INFO request path=/static/main.js method=GET dur_ms=13.1
2025-09-01 11:18:42,037 INFO request path=/static/main.charts.js method=GET dur_ms=16.6
2025-09-01 11:18:42,038 INFO request path=/static/main.plotly.js method=GET dur_ms=15.7
2025-09-01 11:18:42,775 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 11:18:42,776 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 11:18:42,778 INFO request path=/favicon.svg method=GET dur_ms=2.0
2025-09-01 11:18:42,779 INFO request path=/apple-touch-icon.png method=GET dur_ms=5.1
2025-09-01 11:18:42,781 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 11:18:42,784 INFO request path=/api/dashboard method=GET dur_ms=3.4
2025-09-01 11:18:43,011 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 11:18:43,015 INFO request path=/api/calendar method=GET dur_ms=3.5
2025-09-01 11:20:24,469 INFO middleware_enter path=/ method=GET
2025-09-01 11:20:24,472 INFO request path=/ method=GET dur_ms=3.0
2025-09-01 11:20:24,531 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 11:20:24,533 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 11:20:24,535 INFO request path=/favicon.svg method=GET dur_ms=3.7
2025-09-01 11:20:24,542 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 11:20:24,543 INFO request path=/static/main.core.js method=GET dur_ms=10.5
2025-09-01 11:20:24,545 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 11:20:24,546 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 11:20:24,548 INFO request path=/static/main.charts.js method=GET dur_ms=5.5
2025-09-01 11:20:24,558 INFO request path=/static/main.plotly.js method=GET dur_ms=13.4
2025-09-01 11:20:24,560 INFO request path=/static/main.js method=GET dur_ms=14.3
2025-09-01 11:20:25,310 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 11:20:25,312 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 11:20:25,313 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 11:20:25,316 INFO request path=/api/dashboard method=GET dur_ms=2.5
2025-09-01 11:20:25,317 INFO request path=/apple-touch-icon.png method=GET dur_ms=6.5
2025-09-01 11:20:25,318 INFO request path=/favicon.svg method=GET dur_ms=5.5
2025-09-01 11:20:25,643 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 11:20:25,647 INFO request path=/api/calendar method=GET dur_ms=3.6
2025-09-01 11:21:34,018 INFO middleware_enter path=/ method=GET
2025-09-01 11:21:34,019 INFO request path=/ method=GET dur_ms=1.0
2025-09-01 11:21:34,064 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 11:21:34,066 INFO request path=/favicon.svg method=GET dur_ms=2.5
2025-09-01 11:21:34,067 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 11:21:34,078 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 11:21:34,081 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 11:21:34,082 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 11:21:34,085 INFO request path=/static/main.core.js method=GET dur_ms=18.0
2025-09-01 11:21:34,087 INFO request path=/static/main.charts.js method=GET dur_ms=9.5
2025-09-01 11:21:34,088 INFO request path=/static/main.plotly.js method=GET dur_ms=7.5
2025-09-01 11:21:34,090 INFO request path=/static/main.js method=GET dur_ms=8.5
2025-09-01 11:21:34,814 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 11:21:34,817 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 11:21:34,818 INFO request path=/apple-touch-icon.png method=GET dur_ms=3.1
2025-09-01 11:21:34,820 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 11:21:34,822 INFO request path=/favicon.svg method=GET dur_ms=5.0
2025-09-01 11:21:34,823 INFO request path=/api/dashboard method=GET dur_ms=3.0
2025-09-01 11:21:35,078 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 11:21:35,082 INFO request path=/api/calendar method=GET dur_ms=4.0
2025-09-01 11:22:10,167 INFO middleware_enter path=/api/workout/2025-08-26 method=GET
2025-09-01 11:22:10,169 INFO request path=/api/workout/2025-08-26 method=GET dur_ms=2.0
2025-09-01 11:23:55,758 INFO middleware_enter path=/ method=GET
2025-09-01 11:23:55,759 INFO request path=/ method=GET dur_ms=1.0
2025-09-01 11:23:55,810 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 11:23:55,813 INFO request path=/favicon.svg method=GET dur_ms=3.0
2025-09-01 11:23:55,814 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 11:23:55,823 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 11:23:55,824 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 11:23:55,824 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 11:23:55,826 INFO request path=/static/main.core.js method=GET dur_ms=11.5
2025-09-01 11:23:55,828 INFO request path=/static/main.js method=GET dur_ms=4.4
2025-09-01 11:23:55,829 INFO request path=/static/main.charts.js method=GET dur_ms=4.9
2025-09-01 11:23:55,830 INFO request path=/static/main.plotly.js method=GET dur_ms=7.0
2025-09-01 11:23:57,112 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 11:23:57,114 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 11:23:57,116 INFO request path=/favicon.svg method=GET dur_ms=2.5
2025-09-01 11:23:57,117 INFO request path=/apple-touch-icon.png method=GET dur_ms=5.5
2025-09-01 11:23:57,119 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 11:23:57,124 INFO request path=/api/dashboard method=GET dur_ms=5.0
2025-09-01 11:23:57,377 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 11:23:57,379 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-09-01 11:24:25,343 INFO middleware_enter path=/ method=GET
2025-09-01 11:24:25,344 INFO request path=/ method=GET dur_ms=1.0
2025-09-01 11:24:25,372 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 11:24:25,374 INFO request path=/favicon.svg method=GET dur_ms=2.0
2025-09-01 11:24:25,376 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 11:24:25,379 INFO request path=/static/main.core.js method=GET dur_ms=3.0
2025-09-01 11:24:25,400 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 11:24:25,401 INFO request path=/static/main.charts.js method=GET dur_ms=1.1
2025-09-01 11:24:25,463 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 11:24:25,464 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 11:24:25,465 INFO request path=/static/main.js method=GET dur_ms=2.0
2025-09-01 11:24:25,466 INFO request path=/static/main.plotly.js method=GET dur_ms=2.1
2025-09-01 11:24:25,864 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 11:24:25,867 INFO request path=/api/dashboard method=GET dur_ms=2.3
2025-09-01 11:24:25,966 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 11:24:25,967 INFO request path=/api/calendar method=GET dur_ms=1.3
2025-09-01 11:24:26,054 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 11:24:26,056 INFO request path=/favicon.ico method=GET dur_ms=1.7
2025-09-01 11:37:26,227 INFO middleware_enter path=/ method=GET
2025-09-01 11:37:26,228 INFO request path=/ method=GET dur_ms=1.1
2025-09-01 11:37:26,266 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 11:37:26,269 INFO request path=/favicon.svg method=GET dur_ms=2.2
2025-09-01 11:37:26,273 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 11:37:26,277 INFO request path=/static/main.core.js method=GET dur_ms=4.8
2025-09-01 11:37:26,291 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 11:37:26,292 INFO request path=/static/main.charts.js method=GET dur_ms=1.6
2025-09-01 11:37:26,305 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 11:37:26,305 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 11:37:26,307 INFO request path=/static/main.js method=GET dur_ms=2.2
2025-09-01 11:37:26,308 INFO request path=/static/main.plotly.js method=GET dur_ms=2.7
2025-09-01 11:37:26,416 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 11:37:26,420 INFO request path=/api/dashboard method=GET dur_ms=4.5
2025-09-01 11:37:26,502 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 11:37:26,504 INFO request path=/api/calendar method=GET dur_ms=2.1
2025-09-01 11:37:26,587 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 11:37:26,589 INFO request path=/favicon.ico method=GET dur_ms=1.5
2025-09-01 16:13:49,862 INFO middleware_enter path=/ method=GET
2025-09-01 16:13:49,925 INFO request path=/ method=GET dur_ms=62.7
2025-09-01 16:13:49,971 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 16:13:49,974 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 16:13:49,974 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 16:13:49,976 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 16:13:50,036 INFO request path=/static/main.core.js method=GET dur_ms=65.6
2025-09-01 16:13:50,042 INFO request path=/static/main.js method=GET dur_ms=68.3
2025-09-01 16:13:50,045 INFO request path=/static/main.charts.js method=GET dur_ms=72.3
2025-09-01 16:13:50,055 INFO request path=/static/main.plotly.js method=GET dur_ms=79.3
2025-09-01 16:13:50,602 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:13:50,604 INFO request path=/favicon.svg method=GET dur_ms=2.0
2025-09-01 16:13:50,675 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 16:13:50,680 INFO request path=/api/dashboard method=GET dur_ms=4.4
2025-09-01 16:13:50,856 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:13:50,858 INFO request path=/api/calendar method=GET dur_ms=3.0
2025-09-01 16:16:10,511 INFO middleware_enter path=/ method=GET
2025-09-01 16:16:10,512 INFO request path=/ method=GET dur_ms=1.1
2025-09-01 16:16:10,533 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:16:10,534 INFO request path=/favicon.svg method=GET dur_ms=1.1
2025-09-01 16:16:10,540 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 16:16:10,542 INFO request path=/static/main.core.js method=GET dur_ms=2.6
2025-09-01 16:16:10,567 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 16:16:10,568 INFO request path=/static/main.charts.js method=GET dur_ms=0.9
2025-09-01 16:16:10,569 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 16:16:10,570 INFO request path=/static/main.js method=GET dur_ms=1.0
2025-09-01 16:16:10,579 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 16:16:10,580 INFO request path=/static/main.plotly.js method=GET dur_ms=1.0
2025-09-01 16:16:10,686 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 16:16:10,688 INFO request path=/api/dashboard method=GET dur_ms=2.1
2025-09-01 16:16:10,786 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:16:10,788 INFO request path=/api/calendar method=GET dur_ms=1.9
2025-09-01 16:16:10,869 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 16:16:10,870 INFO request path=/favicon.ico method=GET dur_ms=2.0
2025-09-01 16:16:46,316 INFO middleware_enter path=/ method=GET
2025-09-01 16:16:46,317 INFO request path=/ method=GET dur_ms=1.0
2025-09-01 16:16:46,361 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:16:46,363 INFO request path=/favicon.svg method=GET dur_ms=1.1
2025-09-01 16:16:46,369 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 16:16:46,371 INFO request path=/static/main.core.js method=GET dur_ms=2.0
2025-09-01 16:16:46,383 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 16:16:46,386 INFO request path=/static/main.charts.js method=GET dur_ms=2.8
2025-09-01 16:16:46,398 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 16:16:46,400 INFO request path=/static/main.js method=GET dur_ms=2.0
2025-09-01 16:16:46,401 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 16:16:46,403 INFO request path=/static/main.plotly.js method=GET dur_ms=2.0
2025-09-01 16:16:46,826 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 16:16:46,828 INFO request path=/api/dashboard method=GET dur_ms=2.1
2025-09-01 16:16:46,911 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:16:46,912 INFO request path=/api/calendar method=GET dur_ms=1.1
2025-09-01 16:16:46,999 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 16:16:47,000 INFO request path=/favicon.ico method=GET dur_ms=1.1
2025-09-01 16:17:51,296 INFO middleware_enter path=/ method=GET
2025-09-01 16:17:51,304 INFO request path=/ method=GET dur_ms=7.1
2025-09-01 16:17:51,446 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 16:17:51,450 INFO request path=/api/dashboard method=GET dur_ms=3.9
2025-09-01 16:17:51,514 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:17:51,516 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-09-01 16:20:46,660 INFO middleware_enter path=/ method=GET
2025-09-01 16:20:46,714 INFO request path=/ method=GET dur_ms=54.8
2025-09-01 16:20:46,755 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:20:46,760 INFO request path=/favicon.svg method=GET dur_ms=5.0
2025-09-01 16:20:46,764 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 16:20:46,767 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 16:20:46,768 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 16:20:46,779 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 16:20:46,828 INFO request path=/static/main.core.js method=GET dur_ms=64.1
2025-09-01 16:20:46,830 INFO request path=/static/main.plotly.js method=GET dur_ms=50.1
2025-09-01 16:20:46,830 INFO request path=/static/main.charts.js method=GET dur_ms=61.1
2025-09-01 16:20:46,831 INFO request path=/static/main.js method=GET dur_ms=63.1
2025-09-01 16:20:47,867 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 16:20:47,870 INFO request path=/api/dashboard method=GET dur_ms=4.0
2025-09-01 16:20:47,883 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 16:20:47,885 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:20:47,886 INFO request path=/apple-touch-icon.png method=GET dur_ms=3.6
2025-09-01 16:20:47,887 INFO request path=/favicon.svg method=GET dur_ms=2.7
2025-09-01 16:20:48,098 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:20:48,101 INFO request path=/api/calendar method=GET dur_ms=3.0
2025-09-01 16:24:20,919 ERROR failed parsing dashboard_layout.yaml: while parsing a block mapping
  in "C:\Users\shahe\coding\lifting\dashboard_layout.yaml", line 16, column 3
expected <block end>, but found '-'
  in "C:\Users\shahe\coding\lifting\dashboard_layout.yaml", line 19, column 3
2025-09-01 16:24:21,537 INFO middleware_enter path=/ method=GET
2025-09-01 16:24:21,549 INFO request path=/ method=GET dur_ms=12.0
2025-09-01 16:24:21,592 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:24:21,596 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 16:24:21,597 INFO request path=/favicon.svg method=GET dur_ms=5.0
2025-09-01 16:24:21,602 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 16:24:21,610 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 16:24:21,611 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 16:24:21,677 INFO request path=/static/main.core.js method=GET dur_ms=80.9
2025-09-01 16:24:21,677 INFO request path=/static/main.js method=GET dur_ms=66.9
2025-09-01 16:24:21,678 INFO request path=/static/main.charts.js method=GET dur_ms=77.4
2025-09-01 16:24:21,680 INFO request path=/static/main.plotly.js method=GET dur_ms=68.9
2025-09-01 16:24:22,430 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 16:24:22,434 INFO request path=/api/dashboard method=GET dur_ms=4.0
2025-09-01 16:24:22,444 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 16:24:22,445 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:24:22,446 INFO request path=/apple-touch-icon.png method=GET dur_ms=2.0
2025-09-01 16:24:22,447 INFO request path=/favicon.svg method=GET dur_ms=2.0
2025-09-01 16:24:22,646 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:24:22,648 INFO request path=/api/calendar method=GET dur_ms=2.1
2025-09-01 16:24:51,216 INFO middleware_enter path=/ method=GET
2025-09-01 16:24:51,229 INFO request path=/ method=GET dur_ms=11.4
2025-09-01 16:24:51,264 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:24:51,269 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 16:24:51,270 INFO request path=/favicon.svg method=GET dur_ms=6.0
2025-09-01 16:24:51,276 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 16:24:51,277 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 16:24:51,281 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 16:24:51,327 INFO request path=/static/main.core.js method=GET dur_ms=58.8
2025-09-01 16:24:51,328 INFO request path=/static/main.charts.js method=GET dur_ms=51.8
2025-09-01 16:24:51,330 INFO request path=/static/main.js method=GET dur_ms=52.8
2025-09-01 16:24:51,331 INFO request path=/static/main.plotly.js method=GET dur_ms=49.8
2025-09-01 16:24:52,191 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 16:24:52,194 INFO request path=/api/dashboard method=GET dur_ms=3.0
2025-09-01 16:24:52,198 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 16:24:52,200 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:24:52,202 INFO request path=/apple-touch-icon.png method=GET dur_ms=4.0
2025-09-01 16:24:52,203 INFO request path=/favicon.svg method=GET dur_ms=3.0
2025-09-01 16:24:52,429 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:24:52,431 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-09-01 16:32:33,598 INFO middleware_enter path=/ method=GET
2025-09-01 16:32:33,610 INFO request path=/ method=GET dur_ms=12.3
2025-09-01 16:32:33,670 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:32:33,676 INFO request path=/favicon.svg method=GET dur_ms=6.0
2025-09-01 16:32:33,680 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 16:32:33,680 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 16:32:33,681 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 16:32:33,691 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 16:32:33,737 INFO request path=/static/main.core.js method=GET dur_ms=57.1
2025-09-01 16:32:33,738 INFO request path=/static/main.charts.js method=GET dur_ms=58.1
2025-09-01 16:32:33,739 INFO request path=/static/main.js method=GET dur_ms=58.1
2025-09-01 16:32:33,741 INFO request path=/static/main.plotly.js method=GET dur_ms=50.1
2025-09-01 16:32:34,665 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 16:32:34,673 INFO request path=/api/dashboard method=GET dur_ms=7.7
2025-09-01 16:32:34,677 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 16:32:34,679 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:32:34,680 INFO request path=/apple-touch-icon.png method=GET dur_ms=3.0
2025-09-01 16:32:34,685 INFO request path=/favicon.svg method=GET dur_ms=5.9
2025-09-01 16:32:34,938 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:32:34,941 INFO request path=/api/calendar method=GET dur_ms=3.0
2025-09-01 16:32:41,432 INFO middleware_enter path=/api/workout/2025-08-29 method=GET
2025-09-01 16:32:41,436 INFO request path=/api/workout/2025-08-29 method=GET dur_ms=4.0
2025-09-01 16:32:56,712 INFO middleware_enter path=/ method=GET
2025-09-01 16:32:56,713 INFO request path=/ method=GET dur_ms=1.2
2025-09-01 16:32:56,747 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:32:56,749 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 16:32:56,755 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 16:32:56,756 INFO request path=/favicon.svg method=GET dur_ms=9.0
2025-09-01 16:32:56,758 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 16:32:56,760 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 16:32:56,760 INFO request path=/static/main.core.js method=GET dur_ms=11.0
2025-09-01 16:32:56,763 INFO request path=/static/main.charts.js method=GET dur_ms=7.5
2025-09-01 16:32:56,764 INFO request path=/static/main.js method=GET dur_ms=5.6
2025-09-01 16:32:56,766 INFO request path=/static/main.plotly.js method=GET dur_ms=5.5
2025-09-01 16:32:57,669 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 16:32:57,675 INFO request path=/api/dashboard method=GET dur_ms=5.6
2025-09-01 16:32:57,680 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 16:32:57,681 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:32:57,683 INFO request path=/apple-touch-icon.png method=GET dur_ms=3.4
2025-09-01 16:32:57,685 INFO request path=/favicon.svg method=GET dur_ms=3.3
2025-09-01 16:32:58,092 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:32:58,095 INFO request path=/api/calendar method=GET dur_ms=3.3
2025-09-01 16:33:32,143 INFO middleware_enter path=/ method=GET
2025-09-01 16:33:32,144 INFO request path=/ method=GET dur_ms=1.0
2025-09-01 16:33:32,212 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:33:32,214 INFO request path=/favicon.svg method=GET dur_ms=2.0
2025-09-01 16:33:32,215 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 16:33:32,216 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 16:33:32,221 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 16:33:32,223 INFO request path=/static/main.core.js method=GET dur_ms=8.0
2025-09-01 16:33:32,225 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 16:33:32,227 INFO request path=/static/main.charts.js method=GET dur_ms=11.0
2025-09-01 16:33:32,228 INFO request path=/static/main.js method=GET dur_ms=7.0
2025-09-01 16:33:32,230 INFO request path=/static/main.plotly.js method=GET dur_ms=5.0
2025-09-01 16:33:33,167 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 16:33:33,172 INFO request path=/api/dashboard method=GET dur_ms=5.0
2025-09-01 16:33:33,181 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 16:33:33,183 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:33:33,184 INFO request path=/apple-touch-icon.png method=GET dur_ms=3.7
2025-09-01 16:33:33,189 INFO request path=/favicon.svg method=GET dur_ms=6.5
2025-09-01 16:33:33,452 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:33:33,454 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-09-01 16:34:08,334 INFO middleware_enter path=/ method=GET
2025-09-01 16:34:08,347 INFO request path=/ method=GET dur_ms=14.0
2025-09-01 16:34:08,544 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 16:34:08,545 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 16:34:08,547 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 16:34:08,547 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 16:34:08,598 INFO request path=/static/main.js method=GET dur_ms=53.0
2025-09-01 16:34:08,599 INFO request path=/static/main.plotly.js method=GET dur_ms=53.0
2025-09-01 16:34:08,600 INFO request path=/static/main.core.js method=GET dur_ms=54.0
2025-09-01 16:34:08,601 INFO request path=/static/main.charts.js method=GET dur_ms=54.0
2025-09-01 16:34:09,127 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:34:09,128 INFO request path=/favicon.svg method=GET dur_ms=1.0
2025-09-01 16:34:09,512 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 16:34:09,514 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:34:09,515 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 16:34:09,516 INFO request path=/favicon.svg method=GET dur_ms=1.9
2025-09-01 16:34:09,517 INFO request path=/apple-touch-icon.png method=GET dur_ms=5.0
2025-09-01 16:34:09,521 INFO request path=/api/dashboard method=GET dur_ms=5.6
2025-09-01 16:34:09,785 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:34:09,788 INFO request path=/api/calendar method=GET dur_ms=2.3
2025-09-01 16:34:14,804 INFO middleware_enter path=/ method=GET
2025-09-01 16:34:14,805 INFO request path=/ method=GET dur_ms=1.0
2025-09-01 16:34:14,850 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:34:14,852 INFO request path=/favicon.svg method=GET dur_ms=2.0
2025-09-01 16:34:14,854 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 16:34:14,855 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 16:34:14,861 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 16:34:14,862 INFO request path=/static/main.core.js method=GET dur_ms=8.0
2025-09-01 16:34:14,864 INFO request path=/static/main.charts.js method=GET dur_ms=9.0
2025-09-01 16:34:14,865 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 16:34:14,871 INFO request path=/static/main.plotly.js method=GET dur_ms=5.6
2025-09-01 16:34:14,877 INFO request path=/static/main.js method=GET dur_ms=15.6
2025-09-01 16:34:16,016 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 16:34:16,018 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:34:16,018 INFO request path=/apple-touch-icon.png method=GET dur_ms=2.5
2025-09-01 16:34:16,019 INFO request path=/favicon.svg method=GET dur_ms=1.5
2025-09-01 16:34:16,171 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 16:34:16,177 INFO request path=/api/dashboard method=GET dur_ms=6.0
2025-09-01 16:34:16,514 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:34:16,516 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-09-01 16:34:42,315 INFO middleware_enter path=/ method=GET
2025-09-01 16:34:42,330 INFO request path=/ method=GET dur_ms=15.0
2025-09-01 16:34:42,370 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:34:42,374 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 16:34:42,374 INFO request path=/favicon.svg method=GET dur_ms=5.0
2025-09-01 16:34:42,385 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 16:34:42,386 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 16:34:42,387 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 16:34:42,432 INFO request path=/static/main.core.js method=GET dur_ms=58.5
2025-09-01 16:34:42,433 INFO request path=/static/main.plotly.js method=GET dur_ms=46.5
2025-09-01 16:34:42,434 INFO request path=/static/main.js method=GET dur_ms=46.5
2025-09-01 16:34:42,435 INFO request path=/static/main.charts.js method=GET dur_ms=49.5
2025-09-01 16:34:43,256 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 16:34:43,260 INFO request path=/api/dashboard method=GET dur_ms=4.0
2025-09-01 16:34:43,266 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 16:34:43,268 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:34:43,270 INFO request path=/apple-touch-icon.png method=GET dur_ms=3.8
2025-09-01 16:34:43,271 INFO request path=/favicon.svg method=GET dur_ms=3.0
2025-09-01 16:34:43,502 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:34:43,504 INFO request path=/api/calendar method=GET dur_ms=3.0
2025-09-01 16:35:05,011 INFO middleware_enter path=/ method=GET
2025-09-01 16:35:05,020 INFO request path=/ method=GET dur_ms=9.0
2025-09-01 16:35:05,066 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:35:05,068 INFO request path=/favicon.svg method=GET dur_ms=2.0
2025-09-01 16:35:05,069 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 16:35:05,071 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 16:35:05,073 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 16:35:05,125 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 16:35:05,127 INFO request path=/static/main.core.js method=GET dur_ms=58.1
2025-09-01 16:35:05,131 INFO request path=/static/main.charts.js method=GET dur_ms=60.1
2025-09-01 16:35:05,133 INFO request path=/static/main.js method=GET dur_ms=60.1
2025-09-01 16:35:05,135 INFO request path=/static/main.plotly.js method=GET dur_ms=10.0
2025-09-01 16:35:06,190 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 16:35:06,197 INFO request path=/api/dashboard method=GET dur_ms=7.1
2025-09-01 16:35:06,204 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 16:35:06,208 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:35:06,213 INFO request path=/favicon.svg method=GET dur_ms=5.6
2025-09-01 16:35:06,215 INFO request path=/apple-touch-icon.png method=GET dur_ms=10.9
2025-09-01 16:35:06,419 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:35:06,422 INFO request path=/api/calendar method=GET dur_ms=3.0
2025-09-01 16:37:32,698 INFO middleware_enter path=/ method=GET
2025-09-01 16:37:32,756 INFO request path=/ method=GET dur_ms=58.2
2025-09-01 16:37:32,784 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 16:37:32,828 INFO request path=/static/main.core.js method=GET dur_ms=44.2
2025-09-01 16:37:32,830 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 16:37:32,830 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 16:37:32,830 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 16:37:32,837 INFO request path=/static/main.charts.js method=GET dur_ms=7.0
2025-09-01 16:37:32,843 INFO request path=/static/main.plotly.js method=GET dur_ms=13.0
2025-09-01 16:37:32,845 INFO request path=/static/main.js method=GET dur_ms=15.0
2025-09-01 16:37:33,454 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 16:37:33,458 INFO request path=/api/dashboard method=GET dur_ms=4.5
2025-09-01 16:37:33,690 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:37:33,693 INFO request path=/api/calendar method=GET dur_ms=3.0
2025-09-01 16:37:40,230 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:37:40,231 INFO request path=/api/calendar method=GET dur_ms=1.0
2025-09-01 16:37:45,208 INFO middleware_enter path=/api/workout/2025-08-29 method=GET
2025-09-01 16:37:45,211 INFO request path=/api/workout/2025-08-29 method=GET dur_ms=3.4
2025-09-01 16:39:44,139 INFO middleware_enter path=/ method=GET
2025-09-01 16:39:44,195 INFO request path=/ method=GET dur_ms=55.9
2025-09-01 16:39:44,224 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:39:44,228 INFO request path=/favicon.svg method=GET dur_ms=3.7
2025-09-01 16:39:44,232 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 16:39:44,234 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 16:39:44,236 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 16:39:44,237 INFO request path=/static/main.core.js method=GET dur_ms=5.1
2025-09-01 16:39:44,239 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 16:39:44,240 INFO request path=/static/main.charts.js method=GET dur_ms=5.6
2025-09-01 16:39:44,248 INFO request path=/static/main.js method=GET dur_ms=12.3
2025-09-01 16:39:44,252 INFO request path=/static/main.plotly.js method=GET dur_ms=13.5
2025-09-01 16:39:45,090 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 16:39:45,095 INFO request path=/api/dashboard method=GET dur_ms=5.0
2025-09-01 16:39:45,107 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 16:39:45,109 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:39:45,113 INFO request path=/favicon.svg method=GET dur_ms=4.8
2025-09-01 16:39:45,118 INFO request path=/apple-touch-icon.png method=GET dur_ms=11.8
2025-09-01 16:39:45,312 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:39:45,315 INFO request path=/api/calendar method=GET dur_ms=3.5
2025-09-01 16:40:11,053 INFO middleware_enter path=/api/workout/2025-08-29 method=GET
2025-09-01 16:40:11,057 INFO request path=/api/workout/2025-08-29 method=GET dur_ms=4.1
2025-09-01 16:42:33,463 INFO middleware_enter path=/ method=GET
2025-09-01 16:42:33,464 INFO request path=/ method=GET dur_ms=1.0
2025-09-01 16:42:33,521 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:42:33,523 INFO request path=/favicon.svg method=GET dur_ms=2.0
2025-09-01 16:42:33,525 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 16:42:33,527 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 16:42:33,527 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 16:42:33,529 INFO request path=/static/main.core.js method=GET dur_ms=4.0
2025-09-01 16:42:33,532 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 16:42:33,533 INFO request path=/static/main.js method=GET dur_ms=6.0
2025-09-01 16:42:33,534 INFO request path=/static/main.charts.js method=GET dur_ms=8.0
2025-09-01 16:42:33,538 INFO request path=/static/main.plotly.js method=GET dur_ms=6.0
2025-09-01 16:42:35,352 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 16:42:35,356 INFO request path=/api/dashboard method=GET dur_ms=4.0
2025-09-01 16:42:35,362 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 16:42:35,363 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 16:42:35,365 INFO request path=/favicon.svg method=GET dur_ms=2.0
2025-09-01 16:42:35,365 INFO request path=/apple-touch-icon.png method=GET dur_ms=3.0
2025-09-01 16:42:35,567 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:42:35,570 INFO request path=/api/calendar method=GET dur_ms=3.0
2025-09-01 16:48:00,576 INFO middleware_enter path=/docs method=GET
2025-09-01 16:48:00,578 INFO request path=/docs method=GET dur_ms=1.0
2025-09-01 16:48:00,981 INFO middleware_enter path=/openapi.json method=GET
2025-09-01 16:48:00,993 INFO request path=/openapi.json method=GET dur_ms=11.5
2025-09-01 16:48:28,716 INFO middleware_enter path=/ method=GET
2025-09-01 16:48:28,767 INFO request path=/ method=GET dur_ms=51.1
2025-09-01 16:48:28,815 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 16:48:28,818 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 16:48:28,867 INFO request path=/static/main.js method=GET dur_ms=52.5
2025-09-01 16:48:28,870 INFO request path=/static/main.plotly.js method=GET dur_ms=52.8
2025-09-01 16:48:29,283 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 16:48:29,285 INFO request path=/api/dashboard method=GET dur_ms=2.0
2025-09-01 16:48:29,506 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:48:29,509 INFO request path=/api/calendar method=GET dur_ms=3.5
2025-09-01 16:51:05,704 INFO middleware_enter path=/ method=GET
2025-09-01 16:51:05,764 INFO request path=/ method=GET dur_ms=60.0
2025-09-01 16:51:05,815 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 16:51:05,815 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 16:51:05,862 INFO request path=/static/main.js method=GET dur_ms=47.3
2025-09-01 16:51:05,863 INFO request path=/static/main.plotly.js method=GET dur_ms=47.8
2025-09-01 16:51:06,293 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 16:51:06,296 INFO request path=/api/dashboard method=GET dur_ms=3.0
2025-09-01 16:51:06,545 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 16:51:06,547 INFO request path=/api/calendar method=GET dur_ms=2.0
2025-09-01 16:51:06,965 INFO middleware_enter path=/docs method=GET
2025-09-01 16:51:06,966 INFO request path=/docs method=GET dur_ms=1.0
2025-09-01 16:51:07,245 INFO middleware_enter path=/openapi.json method=GET
2025-09-01 16:51:07,257 INFO request path=/openapi.json method=GET dur_ms=13.0
2025-09-01 16:51:13,189 INFO middleware_enter path=/docs/layout method=GET
2025-09-01 16:51:13,195 INFO request path=/docs/layout method=GET dur_ms=5.9
2025-09-01 16:51:15,540 INFO middleware_enter path=/docs/readme method=GET
2025-09-01 16:51:15,541 INFO request path=/docs/readme method=GET dur_ms=1.9
2025-09-01 17:00:43,791 INFO middleware_enter path=/health method=GET
2025-09-01 17:00:43,792 INFO request path=/health method=GET dur_ms=1.0
2025-09-01 17:10:20,509 INFO middleware_enter path=/health method=GET
2025-09-01 17:10:20,522 INFO request path=/health method=GET dur_ms=13.7
2025-09-01 17:10:50,580 INFO middleware_enter path=/health method=GET
2025-09-01 17:10:50,588 INFO request path=/health method=GET dur_ms=8.0
2025-09-01 17:11:08,610 INFO middleware_enter path=/workout/2025-08-29 method=GET
2025-09-01 17:11:08,630 INFO request path=/workout/2025-08-29 method=GET dur_ms=20.2
2025-09-01 17:11:08,652 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 17:11:08,667 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 17:11:08,667 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 17:11:08,668 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 17:11:08,668 INFO request path=/favicon.svg method=GET dur_ms=16.0
2025-09-01 17:11:08,670 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 17:11:08,679 INFO request path=/static/main.js method=GET dur_ms=11.4
2025-09-01 17:11:08,680 INFO request path=/static/main.plotly.js method=GET dur_ms=10.6
2025-09-01 17:11:08,681 INFO request path=/static/main.core.js method=GET dur_ms=14.3
2025-09-01 17:11:08,681 INFO request path=/static/main.charts.js method=GET dur_ms=14.0
2025-09-01 17:11:09,544 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 17:11:09,836 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 17:11:09,836 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 17:11:09,851 INFO request path=/api/dashboard method=GET dur_ms=15.0
2025-09-01 17:11:09,853 INFO request path=/favicon.svg method=GET dur_ms=17.1
2025-09-01 17:11:09,853 INFO request path=/apple-touch-icon.png method=GET dur_ms=309.0
2025-09-01 17:11:10,039 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 17:11:10,049 INFO request path=/api/calendar method=GET dur_ms=10.5
2025-09-01 17:11:10,570 INFO middleware_enter path=/api/workout/2025-08-29 method=GET
2025-09-01 17:11:10,593 INFO request path=/api/workout/2025-08-29 method=GET dur_ms=22.6
2025-09-01 17:11:20,635 INFO middleware_enter path=/health method=GET
2025-09-01 17:11:20,644 INFO request path=/health method=GET dur_ms=8.3
2025-09-01 17:11:50,706 INFO middleware_enter path=/health method=GET
2025-09-01 17:11:50,713 INFO request path=/health method=GET dur_ms=7.1
2025-09-01 17:12:20,774 INFO middleware_enter path=/health method=GET
2025-09-01 17:12:20,784 INFO request path=/health method=GET dur_ms=9.6
2025-09-01 17:12:35,478 INFO middleware_enter path=/ method=GET
2025-09-01 17:12:35,479 INFO request path=/ method=GET dur_ms=1.4
2025-09-01 17:12:35,508 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 17:12:35,511 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 17:12:35,517 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 17:12:35,519 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 17:12:35,520 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 17:12:35,521 INFO request path=/favicon.svg method=GET dur_ms=12.5
2025-09-01 17:12:35,524 INFO request path=/static/main.charts.js method=GET dur_ms=12.7
2025-09-01 17:12:35,525 INFO request path=/static/main.core.js method=GET dur_ms=7.8
2025-09-01 17:12:35,526 INFO request path=/static/main.plotly.js method=GET dur_ms=7.3
2025-09-01 17:12:35,528 INFO request path=/static/main.js method=GET dur_ms=7.6
2025-09-01 17:12:36,304 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 17:12:36,316 INFO request path=/api/dashboard method=GET dur_ms=11.8
2025-09-01 17:12:36,323 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 17:12:36,323 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 17:12:36,421 INFO request path=/favicon.svg method=GET dur_ms=98.6
2025-09-01 17:12:36,423 INFO request path=/apple-touch-icon.png method=GET dur_ms=99.5
2025-09-01 17:12:36,490 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 17:12:36,505 INFO request path=/api/calendar method=GET dur_ms=14.3
2025-09-01 17:12:50,840 INFO middleware_enter path=/health method=GET
2025-09-01 17:12:50,853 INFO request path=/health method=GET dur_ms=12.7
2025-09-01 17:13:20,897 INFO middleware_enter path=/health method=GET
2025-09-01 17:13:20,902 INFO request path=/health method=GET dur_ms=5.3
2025-09-01 17:13:32,300 INFO middleware_enter path=/api/workout/2025-08-29 method=GET
2025-09-01 17:13:32,353 INFO request path=/api/workout/2025-08-29 method=GET dur_ms=52.6
2025-09-01 17:13:50,969 INFO middleware_enter path=/health method=GET
2025-09-01 17:13:50,982 INFO request path=/health method=GET dur_ms=12.6
2025-09-01 17:13:54,868 INFO middleware_enter path=/ method=GET
2025-09-01 17:13:54,870 INFO request path=/ method=GET dur_ms=1.8
2025-09-01 17:13:54,896 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 17:13:54,898 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 17:13:54,909 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 17:13:54,911 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 17:13:54,912 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 17:13:54,913 INFO request path=/favicon.svg method=GET dur_ms=16.6
2025-09-01 17:13:54,914 INFO request path=/static/main.core.js method=GET dur_ms=15.6
2025-09-01 17:13:54,916 INFO request path=/static/main.js method=GET dur_ms=7.5
2025-09-01 17:13:54,917 INFO request path=/static/main.charts.js method=GET dur_ms=6.0
2025-09-01 17:13:54,923 INFO request path=/static/main.plotly.js method=GET dur_ms=10.7
2025-09-01 17:13:55,678 INFO middleware_enter path=/apple-touch-icon.png method=GET
2025-09-01 17:13:55,763 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 17:13:55,777 INFO request path=/api/dashboard method=GET dur_ms=14.3
2025-09-01 17:13:55,779 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 17:13:55,779 INFO request path=/apple-touch-icon.png method=GET dur_ms=101.5
2025-09-01 17:13:55,782 INFO request path=/favicon.svg method=GET dur_ms=3.3
2025-09-01 17:13:55,977 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 17:13:55,987 INFO request path=/api/calendar method=GET dur_ms=9.7
2025-09-01 17:14:21,045 INFO middleware_enter path=/health method=GET
2025-09-01 17:14:21,054 INFO request path=/health method=GET dur_ms=8.6
2025-09-01 17:14:41,067 INFO middleware_enter path=/ method=GET
2025-09-01 17:14:41,068 INFO request path=/ method=GET dur_ms=1.1
2025-09-01 17:14:41,171 INFO middleware_enter path=/static/main.core.js method=GET
2025-09-01 17:14:41,173 INFO request path=/static/main.core.js method=GET dur_ms=2.1
2025-09-01 17:14:41,176 INFO middleware_enter path=/favicon.svg method=GET
2025-09-01 17:14:41,177 INFO request path=/favicon.svg method=GET dur_ms=0.9
2025-09-01 17:14:41,285 INFO middleware_enter path=/static/main.charts.js method=GET
2025-09-01 17:14:41,287 INFO request path=/static/main.charts.js method=GET dur_ms=1.8
2025-09-01 17:14:41,399 INFO middleware_enter path=/static/main.js method=GET
2025-09-01 17:14:41,401 INFO request path=/static/main.js method=GET dur_ms=1.5
2025-09-01 17:14:41,412 INFO middleware_enter path=/static/main.plotly.js method=GET
2025-09-01 17:14:41,414 INFO request path=/static/main.plotly.js method=GET dur_ms=1.5
2025-09-01 17:14:41,752 INFO middleware_enter path=/api/dashboard method=GET
2025-09-01 17:14:41,767 INFO request path=/api/dashboard method=GET dur_ms=14.7
2025-09-01 17:14:41,791 INFO middleware_enter path=/favicon.ico method=GET
2025-09-01 17:14:41,792 INFO request path=/favicon.ico method=GET dur_ms=1.1
2025-09-01 17:14:41,892 INFO middleware_enter path=/api/calendar method=GET
2025-09-01 17:14:41,900 INFO request path=/api/calendar method=GET dur_ms=8.7
2025-09-01 17:14:51,110 INFO middleware_enter path=/health method=GET
2025-09-01 17:14:51,117 INFO request path=/health method=GET dur_ms=7.2

```

# out\linecounter.json

```json
{
    "extension": "linecounter",
    "version": "0.2.7",
    "workspace": "c:\\Users\\shahe\\coding\\lifting",
    "linecount": [
        {
            "version": "0.2.7",
            "counttime": "2025-08-26 23:37:30",
            "filesum": 10,
            "codesum": 624,
            "commentsum": 55,
            "blanksum": 310,
            "statistics": {
                ".txt": {
                    "code": 8,
                    "comment": 0,
                    "blank": 0
                },
                ".md": {
                    "code": 255,
                    "comment": 45,
                    "blank": 220
                },
                ".py": {
                    "code": 257,
                    "comment": 9,
                    "blank": 81
                },
                ".js": {
                    "code": 57,
                    "comment": 0,
                    "blank": 9
                },
                ".html": {
                    "code": 35,
                    "comment": 0,
                    "blank": 0
                },
                ".example": {
                    "code": 5,
                    "comment": 1,
                    "blank": 0
                },
                ".csv": {
                    "code": 7,
                    "comment": 0,
                    "blank": 0
                }
            },
            "filelist": [
                {
                    "blank": 0,
                    "code": 5,
                    "comment": 1,
                    "filename": ".env.example"
                },
                {
                    "blank": 0,
                    "code": 0,
                    "comment": 0,
                    "filename": "app\\__init__.py"
                },
                {
                    "blank": 12,
                    "code": 30,
                    "comment": 2,
                    "filename": "app\\db.py"
                },
                {
                    "blank": 31,
                    "code": 99,
                    "comment": 3,
                    "filename": "app\\main.py"
                },
                {
                    "blank": 38,
                    "code": 128,
                    "comment": 4,
                    "filename": "app\\processing.py"
                },
                {
                    "blank": 9,
                    "code": 57,
                    "comment": 0,
                    "filename": "app\\static\\main.js"
                },
                {
                    "blank": 0,
                    "code": 35,
                    "comment": 0,
                    "filename": "app\\templates\\dashboard.html"
                },
                {
                    "blank": 220,
                    "code": 255,
                    "comment": 45,
                    "filename": "readme.md"
                },
                {
                    "blank": 0,
                    "code": 8,
                    "comment": 0,
                    "filename": "requirements.txt"
                },
                {
                    "blank": 0,
                    "code": 7,
                    "comment": 0,
                    "filename": "tests_sample.csv"
                }
            ]
        }
    ]
}
```

# out\linecounter.txt

```txt
===============================================================================
EXTENSION NAME : linecounter
EXTENSION VERSION : 0.2.7
-------------------------------------------------------------------------------
count time : 2025-08-26 23:37:30
count workspace : c:\Users\shahe\coding\lifting
total files : 10
total code lines : 624
total comment lines : 55
total blank lines : 310

    statistics
   |      extension|     total code|  total comment|    total blank|percent|
   -------------------------------------------------------------------------
   |           .txt|              8|              0|              0|    1.3|
   |            .md|            255|             45|            220|     41|
   |            .py|            257|              9|             81|     41|
   |            .js|             57|              0|              9|    9.1|
   |          .html|             35|              0|              0|    5.6|
   |       .example|              5|              1|              0|   0.80|
   |           .csv|              7|              0|              0|    1.1|
   -------------------------------------------------------------------------
.env.example, code is 5, comment is 1, blank is 0.
app\__init__.py, code is 0, comment is 0, blank is 0.
app\db.py, code is 30, comment is 2, blank is 12.
app\main.py, code is 99, comment is 3, blank is 31.
app\processing.py, code is 128, comment is 4, blank is 38.
app\static\main.js, code is 57, comment is 0, blank is 9.
app\templates\dashboard.html, code is 35, comment is 0, blank is 0.
readme.md, code is 255, comment is 45, blank is 220.
requirements.txt, code is 8, comment is 0, blank is 0.
tests_sample.csv, code is 7, comment is 0, blank is 0.
===============================================================================

```

# readme.md

```md
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

---

## Self‑Hosting (Docker) Quick Start

For a minimal reproducible deployment using Docker:

1. Clone repo & copy env
  cp .env.example .env
  (Edit .env to set a strong INGEST_TOKEN and adjust ALLOWED_ORIGINS.)

2. Build & run (foreground)
  docker compose up --build

3. Visit the dashboard:
  http://localhost:8069

4. Ingest a Strong export:
  curl -F "file=@strong_export.csv" "http://localhost:8069/ingest?token=$INGEST_TOKEN"

Data & logs persist in the host `data/` and `logs/` folders via volume mounts.

Upgrade:
  git pull
  docker compose build
  docker compose up -d

Stop:
  docker compose down

### Run in Background (Detached) With Auto-Restart

If you just want it to stay up ("run unless stopped") run detached with the existing `restart: unless-stopped` policy in `docker-compose.yml`:

\`\`\`
docker compose up -d --build
\`\`\`


Check status:
\`\`\`
docker compose ps
\`\`\`

Follow logs live:
\`\`\`
docker compose logs -f
\`\`\`

Stop (this also disables the restart policy until you start again):
\`\`\`
docker compose down
\`\`\`

After a host reboot the container will auto-start because of `restart: unless-stopped` (unless you explicitly brought it down with `docker compose down`).

## Production Hardening Checklist

- Use a reverse proxy (Caddy / Nginx / Traefik) terminating HTTPS in front of the container.
- Restrict /debug/* endpoints by disabling them (comment routes) or ensuring token set.
- Rotate `INGEST_TOKEN` periodically; never share it publicly.
- Backup `data/lifting.db` (simple copy) on a schedule.
- Enable metrics (optional): sidecar (e.g., cAdvisor) + reverse proxy logs.
- Consider read-only FS except for `/app/data` and `/app/logs`.
- Run container as non-root (already default `appuser` in Dockerfile).

## Deploy via Docker + Caddy (Example Snippet)

Example Caddyfile mapping a domain with automatic HTTPS:
\`\`\`
lifting.example.com {
  reverse_proxy 127.0.0.1:8069
  encode zstd gzip
  header Content-Security-Policy "default-src 'self'; script-src 'self' https://cdn.plot.ly; style-src 'self' 'unsafe-inline'"
  header Referrer-Policy no-referrer
  header X-Content-Type-Options nosniff
  header X-Frame-Options DENY
}
\`\`\`

Run Caddy beside the app container, or use Traefik labels if preferred.

## Environment Variables Summary

| Var | Purpose |
|-----|---------|
| INGEST_TOKEN | Required token for uploads & debug endpoints |
| ALLOWED_ORIGINS | CORS allow-list for browser clients |
| MAX_UPLOAD_MB | File size guard for ingestion |
| LOG_LEVEL | Logging verbosity (INFO default) |
| TZ | Human display timezone (client side) |

---

## License

This project is released under the MIT License (see `LICENSE`).


## Dashboard Layout Configuration (YAML)

You can control the order & widths of dashboard components via `dashboard_layout.yaml` at the project root. Example (default):

\`\`\`
layout:
  rows:
    - [progressive_overload:12]
    - [volume_trend:8, exercise_volume:4]
    - [rep_distribution:4, weekly_ppl:7, muscle_balance:5]
    - [calendar:12]
\`\`\`

Rules:
* Each row is a list. Items use `widget:width` (1–12). Missing width defaults to 12.
* Supported widget keys:
  * progressive_overload
  * volume_trend
  * exercise_volume
  * rep_distribution
  * weekly_ppl
  * muscle_balance
  * calendar
* Unknown widgets are ignored silently.
* File reload requires an application restart (config is read at startup).

Widths map to Tailwind `md:col-span-{width}` (grid is 12 columns). On small screens all widgets stack full-width.

To hide a widget, remove it from the YAML. To move it, change its row order or width.


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

\`\`\`bash
python -m venv .venv
source .venv/Scripts/activate  # (On PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
copy .env.example .env  # then edit INGEST_TOKEN
uvicorn app.main:app --host 127.0.0.1 --port 8069 --reload
\`\`\`

Test Ingest

\`\`\`bash
curl -F "file=@tests_sample.csv" "http://127.0.0.1:8069/ingest?token=$INGEST_TOKEN"
curl http://127.0.0.1:8069/api/sessions
\`\`\`

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

\`\`\`bash
python scripts/rebuild_db.py              # backup then delete + recreate empty schema
python scripts/rebuild_db.py --with-imports  # backup, recreate, re-ingest all CSVs in data/uploads/
python scripts/rebuild_db.py --no-backup --with-imports  # destructive quick rebuild
\`\`\`

Backups are stored in `data/backups/` named `lifting_YYYYMMDD_HHMMSS.db`.
```

# requirements.txt

```txt
fastapi
uvicorn[standard]
python-multipart
jinja2
pydantic
pandas
plotly
python-dotenv
cairosvg
pillow
PyYAML

```

# run.bat

```bat
@echo off
REM Lifting Pipeline launcher (development convenience)
REM Usage: run.bat [host] [port]
REM Examples:
REM   run.bat            -> 0.0.0.0:8069 (LAN accessible)
REM   run.bat 127.0.0.1  -> 127.0.0.1:8069 (loopback only)
REM   run.bat 0.0.0.0 9000 -> 0.0.0.0:9000

setlocal ENABLEDELAYEDEXPANSION
set BASE_DIR=%~dp0
cd /d %BASE_DIR%

REM Ensure virtual environment exists
if not exist "%BASE_DIR%\.venv\Scripts\python.exe" (
  echo [INFO] Creating virtual environment .venv ...
  py -m venv .venv || goto :fail
)

REM Activate venv
call "%BASE_DIR%\.venv\Scripts\activate.bat" || goto :fail

REM Install deps if fastapi missing
python -c "import fastapi" 2>NUL 1>NUL
if errorlevel 1 (
  echo [INFO] Installing requirements...
  pip install -q -r requirements.txt || goto :fail
)

REM Determine host / port (allow env override, then args)
set HOST=%HOST%
if "%HOST%"=="" set HOST=0.0.0.0
set PORT=8069
if not "%1"=="" (
  REM If first arg contains a dot assume it's a host, else treat as port
  echo %1 | findstr /R /C:"[0-9][0-9]*\.[0-9]" >NUL && (set HOST=%1) || (set PORT=%1)
)
if not "%2"=="" set PORT=%2

if exist .env (
  echo [INFO] Using .env file.
) else (
  echo [WARN] .env not found. Copy .env.example to .env and set INGEST_TOKEN.
)

echo [START] Lifting Pipeline binding %HOST%:%PORT% (Ctrl+C to stop)
uvicorn app.main:app --host %HOST% --port %PORT% --reload
set CODE=%ERRORLEVEL%

echo [EXIT] Uvicorn exited with code %CODE%
endlocal & exit /b %CODE%

:fail
echo [ERROR] Startup failed.
endlocal & exit /b 1

```

# scripts\kill_8069.bat

```bat
@echo off
REM Kill all processes using TCP/UDP port 8069
set PORT=8069

echo Searching for processes using port %PORT% ...
netstat -ano | findstr :%PORT% >nul
if errorlevel 1 (
  echo No processes found using port %PORT%.
  goto :eof
)

setlocal enabledelayedexpansion
for /f "tokens=5" %%P in ('netstat -ano ^| findstr :%PORT%') do (
  if not defined SEEN_%%P (
    set SEEN_%%P=1
    echo Killing PID %%P
    taskkill /F /PID %%P >nul 2>&1
    if errorlevel 1 (
      echo   Failed to kill PID %%P (may require Administrator privileges)
    ) else (
      echo   PID %%P terminated.
    )
  )
)
endlocal

echo Done.

```

# scripts\rebuild_db.py

```py
"""Utility script to rebuild (reset) the lifting SQLite database.

Actions:
1. Optional backup current DB to data/backups/lifting_YYYYMMDD_HHMMSS.db
2. Delete existing data/lifting.db
3. Recreate schema
4. Optionally re-import all CSV files in data/uploads/ (idempotent)

Usage:
  python scripts/rebuild_db.py --with-imports
  python scripts/rebuild_db.py --no-backup

Safe: uses INSERT OR IGNORE so repeated CSVs won't duplicate sets.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import shutil
import sys

# Local imports after adjusting sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import DB_PATH, init_db  # noqa: E402
from app.processing import process_csv_to_db  # noqa: E402

UPLOAD_DIR = ROOT / "data" / "uploads"
BACKUP_DIR = ROOT / "data" / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def backup_db():
    if not DB_PATH.exists():
        print("[INFO] No DB to backup.")
        return None
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"lifting_{ts}.db"
    shutil.copy2(DB_PATH, dest)
    print(f"[OK] Backup created: {dest}")
    return dest


def delete_db():
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"[OK] Deleted existing DB: {DB_PATH}")
    else:
        print("[INFO] DB file not present.")


def recreate_schema():
    init_db()
    print("[OK] Schema recreated.")


def reimport_csvs():
    if not UPLOAD_DIR.exists():
        print("[INFO] Upload dir missing; skipping re-import.")
        return
    csv_files = sorted(UPLOAD_DIR.glob("*.csv"))
    if not csv_files:
        print("[INFO] No CSV files to import.")
        return
    total_inserted = 0
    for p in csv_files:
        try:
            inserted = process_csv_to_db(p)
            total_inserted += inserted
            print(f"[OK] {p.name}: +{inserted} rows")
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {p.name}: {e}")
    print(f"[SUMMARY] Total inserted: {total_inserted}")


def main():
    ap = argparse.ArgumentParser(description="Rebuild lifting DB")
    ap.add_argument("--with-imports", action="store_true", help="Re-import all CSVs in data/uploads/")
    ap.add_argument("--no-backup", action="store_true", help="Skip creating a backup before deletion")
    args = ap.parse_args()

    if not args.no_backup:
        backup_db()
    delete_db()
    recreate_schema()
    if args.with_imports:
        reimport_csvs()


if __name__ == "__main__":
    main()

```

# tests_sample.csv

```csv
Date,Workout Name,Duration,Exercise Name,Set Order,Weight,Reps,Distance,Seconds,RPE
2024-11-28 17:36:08,B,35m,Bent Over Row (Barbell),1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,B,35m,Bent Over Row (Barbell),2,20.0,5.0,0,0.0,
2024-11-29 10:15:00,A,40m,Squat (Barbell),1,60.0,5.0,0,0.0,
2024-11-29 10:15:00,A,40m,Squat (Barbell),2,60.0,5.0,0,0.0,
2024-12-03 18:00:00,C,30m,Bench Press (Barbell),1,80.0,3.0,0,0.0,
2024-12-03 18:00:00,C,30m,Bench Press (Barbell),2,85.0,2.0,0,0.0,

```

