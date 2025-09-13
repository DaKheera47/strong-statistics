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
# Legacy DB filename (pre-rename) & current DB
data/lifting.db
data/strong.db
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
"""Database utilities for strong-statistics.

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

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# New canonical DB filename after project rename
DB_PATH = DATA_DIR / "strong.db"
LEGACY_DB_PATH = DATA_DIR / "lifting.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# SQLite connections are NOT thread-safe by default when shared;
# we create short-lived connections via contextmanager.
_lock = threading.Lock()
_LOCK_MAX_WAIT_SEC = 5  # fail fast if something holds the lock too long
logger = logging.getLogger("strong.db")

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

def _migrate_legacy_db_if_needed() -> None:
  """If the old lifting.db exists and strong.db does not, rename it.

  This provides a seamless upgrade path for existing deployments.
  Safe to call multiple times (idempotent).
  """
  try:
    if LEGACY_DB_PATH.exists() and not DB_PATH.exists():
      LEGACY_DB_PATH.rename(DB_PATH)
      logger.info("Migrated legacy database %s -> %s", LEGACY_DB_PATH.name, DB_PATH.name)
  except Exception as e:  # pragma: no cover - best effort
    logger.error("Legacy DB migration failed: %s", e)


def init_db() -> None:
  """Initialize database schema if not present (and migrate legacy file)."""
  _migrate_legacy_db_if_needed()
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

import faulthandler
import logging
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.templating import Jinja2Templates

from .db import get_meta, init_db
from .processing import (
    build_dashboard_data,
    exercise_analysis,
    get_best_sets_analysis,
    get_body_measurements,
    get_exercise_detail,
    get_exercise_frequency,
    get_exercise_records,
    get_muscle_group_balance,
    get_plateau_detection,
    get_progressive_overload_rate,
    get_recovery_tracking,
    get_rep_range_distribution,
    get_strength_ratios,
    get_training_streak,
    get_weekly_volume_heatmap,
    get_workout_calendar,
    get_workout_detail,
    get_workout_duration_trends,
    list_exercises,
    personal_records_timeline,
    process_csv_to_db,
    progressive_overload_data,
    strength_balance,
    training_consistency,
    volume_progression,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
LOG_DIR = BASE_DIR / "logs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")
INGEST_TOKEN = os.getenv("INGEST_TOKEN", "")
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost").split(",")
    if o.strip()
]
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))

# Logging
logger = logging.getLogger("strong-statistics")
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

app = FastAPI(title="strong-statistics")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

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
            if (not APPLE_TOUCH_PATH.exists()) or (
                APPLE_TOUCH_PATH.stat().st_mtime < ICON_PATH.stat().st_mtime
            ):
                from cairosvg import svg2png  # type: ignore

                APPLE_TOUCH_PATH.parent.mkdir(parents=True, exist_ok=True)
                png_bytes = svg2png(
                    url=str(ICON_PATH), output_width=180, output_height=180
                )
                APPLE_TOUCH_PATH.write_bytes(png_bytes)
        except Exception:  # pragma: no cover - non critical
            # Fall back to serving svg directly if conversion failed
            return FileResponse(str(ICON_PATH), media_type="image/svg+xml")
        return FileResponse(str(APPLE_TOUCH_PATH), media_type="image/png")


init_db()


@app.post("/ingest")
async def ingest(
    request: Request, token: str = Query(""), file: UploadFile | None = File(None)
):
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

    if content_type not in (
        "text/csv",
        "application/vnd.ms-excel",
        "application/octet-stream",
    ):
        raise HTTPException(
            status_code=400,
            detail=f"unsupported content type: {content_type} (mode={mode})",
        )

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

    logger.info(
        "ingested file=%s inserted_rows=%s mode=%s size_bytes=%s",
        stored_name,
        inserted,
        mode,
        len(raw),
    )
    return {"stored": stored_name, "rows": inserted}


@app.get("/health")
async def health():
    payload = {
        "ok": True,
        "time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_ingested_at": get_meta("last_ingested_at"),
    }
    # Add diagnostic header so we can tell if Cloudflare is reaching origin.
    return JSONResponse(payload, headers={"X-Strong-Origin": "fastapi"})

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
    import pandas as pd  # if not already imported

    df = pd.read_csv(csv_path)

    # Convert "Duration" like "35m" or "1h 3m" -> integer minutes
    if "Duration" in df.columns:
        s = df["Duration"].astype("string").str.lower().str.strip()
        hrs = pd.to_numeric(s.str.extract(r"(\d+)\s*h", expand=False), errors="coerce")
        mins = pd.to_numeric(s.str.extract(r"(\d+)\s*m", expand=False), errors="coerce")

        total_min = (hrs.fillna(0) * 60 + mins.fillna(0)).astype("Int64")
        # if neither h nor m was present, keep it null
        total_min[hrs.isna() & mins.isna()] = pd.NA

        df["Duration"] = total_min

    normalized = normalize_df(df)
    before = count_sets()
    upsert_sets(normalized)  # your existing upsert; no extra dedupe added
    after = count_sets()
    inserted = after - before
    set_meta("last_ingested_at", datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
    return inserted

```

# data\strong.db

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

# data\uploads\strong_20250902_143652.csv

```csv
Date,Workout Name,Duration,Exercise Name,Set Order,Weight,Reps,Distance,Seconds,RPE
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",3,10.0,17.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",2,10.0,16.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",3,10.0,8.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",2,10.0,10.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",2,15.0,15.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",1,10.0,20.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",3,15.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",4,15.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",2,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",0,0,0.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",1,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",3,11.0,8.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",2,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",1,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",2,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",1,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",1,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",3,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",2,11.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",2,15.0,7.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",0,0,0.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",1,10.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",3,15.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",1,9.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",3,9.0,6.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",2,9.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",2,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",3,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",1,11.0,12.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",3,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",2,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",3,8.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",2,8.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Bicep Curl (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",2,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",1,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",3,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",2,18.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",1,11.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",3,18.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",3,23.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",1,17.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",2,23.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Glute Kickback (Machine)",1,5.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Glute Kickback (Machine)",2,5.0,7.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",3,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",1,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",2,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",3,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",2,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",3,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",1,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",2,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",3,4.0,7.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",2,4.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",1,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",2,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",3,23.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",3,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",1,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",2,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",2,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",3,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",1,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",2,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",1,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",3,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",2,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",3,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",1,18.0,10.0,0,0.0,
```

# data\uploads\strong_20250905_132939.csv

```csv
Date,Workout Name,Duration,Exercise Name,Set Order,Weight,Reps,Distance,Seconds,RPE
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",3,10.0,17.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",2,10.0,16.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",3,10.0,8.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",2,10.0,10.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",3,15.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",1,10.0,20.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",2,15.0,15.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",4,15.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",1,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",0,0,0.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",2,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",2,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",1,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",3,11.0,8.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",1,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",2,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",3,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",1,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",2,11.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",3,15.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",2,15.0,7.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",1,10.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",0,0,0.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",1,9.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",2,9.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",3,9.0,6.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",2,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",3,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",1,11.0,12.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",3,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",2,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",2,8.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",3,8.0,8.0,0,0.0,
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
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",2,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",3,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",2,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",3,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",1,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",2,4.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",3,4.0,7.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",3,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",2,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",1,23.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",2,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",3,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",1,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",2,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",3,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",1,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",1,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",3,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",2,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",2,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",3,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",1,18.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pull Up (Assisted)",2,77.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pull Up (Assisted)",1,77.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",3,25.0,14.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",2,25.0,14.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",1,25.0,14.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",0,0,0.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Seated Row (Cable)",1,29.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Seated Row (Cable)",3,29.0,9.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Seated Row (Cable)",2,29.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pec Deck (Machine)",3,25.0,7.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pec Deck (Machine)",1,25.0,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pec Deck (Machine)",2,25.0,9.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Triceps Extension (Cable)",3,5.7,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Triceps Extension (Cable)",2,5.7,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Triceps Extension (Cable)",1,5.7,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Dumbbell)",1,12.0,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Dumbbell)",2,12.0,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Dumbbell)",0,0,0.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Bicep Curl (Dumbbell)",1,5.0,6.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Bicep Curl (Dumbbell)",2,5.0,5.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Barbell)",2,10.0,6.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Barbell)",3,10.0,7.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Barbell)",1,0.0,10.0,0,0.0,
```

# data\uploads\strong_20250907_213225.csv

```csv
Date,Workout Name,Duration,Exercise Name,Set Order,Weight,Reps,Distance,Seconds,RPE
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",2,10.0,16.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",3,10.0,17.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",3,10.0,8.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",2,10.0,10.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",4,15.0,10.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",2,15.0,15.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",1,10.0,20.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",3,15.0,13.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",0,0,0.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",1,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",2,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",2,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",1,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",3,11.0,8.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",1,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",2,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",3,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",1,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",2,11.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",0,0,0.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",1,10.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",3,15.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",2,15.0,7.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",3,9.0,6.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",2,9.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",1,9.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",2,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",3,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",1,11.0,12.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",2,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",3,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",2,8.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",3,8.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Bicep Curl (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",3,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",2,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",1,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",3,18.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",1,11.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",2,18.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",3,23.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",2,23.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",1,17.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Glute Kickback (Machine)",1,5.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Glute Kickback (Machine)",2,5.0,7.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",2,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",3,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",1,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",3,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",2,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",2,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",1,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",3,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",3,4.0,7.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",2,4.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",1,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",3,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",2,23.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",1,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",2,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",3,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",1,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",3,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",2,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",1,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",2,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",3,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",1,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",3,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",2,18.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pull Up (Assisted)",1,77.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pull Up (Assisted)",2,77.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",3,25.0,14.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",1,25.0,14.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",0,0,0.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",2,25.0,14.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Seated Row (Cable)",3,29.0,9.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Seated Row (Cable)",1,29.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Seated Row (Cable)",2,29.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pec Deck (Machine)",1,25.0,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pec Deck (Machine)",2,25.0,9.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pec Deck (Machine)",3,25.0,7.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Triceps Extension (Cable)",2,5.7,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Triceps Extension (Cable)",1,5.7,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Triceps Extension (Cable)",3,5.7,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Dumbbell)",2,12.0,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Dumbbell)",0,0,0.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Dumbbell)",1,12.0,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Bicep Curl (Dumbbell)",2,5.0,5.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Bicep Curl (Dumbbell)",1,5.0,6.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Barbell)",2,30.0,6.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Barbell)",3,30.0,7.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Barbell)",1,20.0,10.0,0,0.0,
```

# data\uploads\strong_20250909_132006.csv

```csv
Date,Workout Name,Duration,Exercise Name,Set Order,Weight,Reps,Distance,Seconds,RPE
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",2,10.0,16.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",3,10.0,17.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",2,10.0,10.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",3,10.0,8.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",4,15.0,10.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",2,15.0,15.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",3,15.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",1,10.0,20.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",2,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",0,0,0.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",1,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",2,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",1,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",3,11.0,8.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",1,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",2,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",1,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",3,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",2,11.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",0,0,0.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",1,10.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",3,15.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",2,15.0,7.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",2,9.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",3,9.0,6.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",1,9.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",3,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",1,11.0,12.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",2,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",3,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",2,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",2,8.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",3,8.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Bicep Curl (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",1,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",3,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",2,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",1,11.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",3,18.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",2,18.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",1,17.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",3,23.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",2,23.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Glute Kickback (Machine)",2,5.0,7.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Glute Kickback (Machine)",1,5.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",3,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",2,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",1,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",3,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",2,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",3,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",1,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",2,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",2,4.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",3,4.0,7.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",1,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",3,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",2,23.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",1,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",2,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",3,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",2,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",1,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",3,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",2,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",3,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",1,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",2,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",1,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",3,18.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pull Up (Assisted)",1,77.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pull Up (Assisted)",2,77.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",0,0,0.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",1,25.0,14.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",3,25.0,14.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",2,25.0,14.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Seated Row (Cable)",1,29.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Seated Row (Cable)",3,29.0,9.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Seated Row (Cable)",2,29.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pec Deck (Machine)",1,25.0,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pec Deck (Machine)",3,25.0,7.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pec Deck (Machine)",2,25.0,9.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Triceps Extension (Cable)",1,5.7,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Triceps Extension (Cable)",3,5.7,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Triceps Extension (Cable)",2,5.7,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Dumbbell)",2,12.0,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Dumbbell)",1,12.0,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Dumbbell)",0,0,0.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Bicep Curl (Dumbbell)",1,5.0,6.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Bicep Curl (Dumbbell)",2,5.0,5.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Barbell)",3,30.0,7.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Barbell)",1,20.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Barbell)",2,30.0,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Chest Press (Machine)",2,25.0,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Chest Press (Machine)",1,18.0,12.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Chest Press (Machine)",3,25.0,4.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Shoulder Press (Machine)",3,9.0,7.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Shoulder Press (Machine)",2,9.0,7.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Shoulder Press (Machine)",1,9.0,8.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pec Deck (Machine)",2,25.0,8.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pec Deck (Machine)",3,25.0,7.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Seated Row (Cable)",3,29.0,10.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Seated Row (Cable)",2,29.0,12.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Seated Row (Cable)",1,29.0,12.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Triceps Extension (Cable)",3,5.7,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Triceps Extension (Cable)",2,5.7,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Triceps Extension (Cable)",1,5.7,12.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pull Up (Assisted)",2,73.0,4.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pull Up (Assisted)",1,73.0,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pull Up (Assisted)",3,77.0,5.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Dumbbell)",1,4.0,8.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Dumbbell)",2,4.0,5.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Dumbbell)",0,0,0.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Barbell)",1,10.0,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Barbell)",0,0,0.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Barbell)",0,0,0.0,0,0.0,
```

# data\uploads\strong_20250911_181842.csv

```csv
Date,Workout Name,Duration,Exercise Name,Set Order,Weight,Reps,Distance,Seconds,RPE
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",2,10.0,16.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",3,10.0,17.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",3,10.0,8.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",2,10.0,10.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",1,10.0,20.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",4,15.0,10.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",2,15.0,15.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",3,15.0,13.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",0,0,0.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",2,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",1,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",2,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",1,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",3,11.0,8.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",1,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",2,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",1,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",2,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",3,11.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",2,15.0,7.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",3,15.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",0,0,0.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",1,10.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",2,9.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",3,9.0,6.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",1,9.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",1,11.0,12.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",3,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",2,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",3,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",2,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",3,8.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",2,8.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Bicep Curl (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",1,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",3,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",2,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",2,18.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",1,11.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",3,18.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",3,23.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",1,17.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",2,23.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Glute Kickback (Machine)",1,5.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Glute Kickback (Machine)",2,5.0,7.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",2,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",3,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",1,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",2,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",3,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",3,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",2,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",1,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",3,4.0,7.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",2,4.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",1,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",3,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",2,23.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",1,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",3,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",2,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",2,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",3,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",1,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",1,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",3,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",2,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",3,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",1,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",2,18.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pull Up (Assisted)",2,77.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pull Up (Assisted)",1,77.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",1,25.0,14.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",0,0,0.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",3,25.0,14.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",2,25.0,14.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Seated Row (Cable)",2,29.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Seated Row (Cable)",1,29.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Seated Row (Cable)",3,29.0,9.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pec Deck (Machine)",1,25.0,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pec Deck (Machine)",2,25.0,9.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pec Deck (Machine)",3,25.0,7.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Triceps Extension (Cable)",2,5.7,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Triceps Extension (Cable)",3,5.7,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Triceps Extension (Cable)",1,5.7,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Dumbbell)",0,0,0.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Dumbbell)",2,12.0,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Dumbbell)",1,12.0,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Bicep Curl (Dumbbell)",2,5.0,5.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Bicep Curl (Dumbbell)",1,5.0,6.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Barbell)",2,30.0,6.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Barbell)",3,30.0,7.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Barbell)",1,20.0,10.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Chest Press (Machine)",1,18.0,12.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Chest Press (Machine)",2,25.0,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Chest Press (Machine)",3,25.0,4.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Shoulder Press (Machine)",2,9.0,7.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Shoulder Press (Machine)",1,9.0,8.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Shoulder Press (Machine)",3,9.0,7.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pec Deck (Machine)",2,25.0,8.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pec Deck (Machine)",3,25.0,7.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Seated Row (Cable)",2,29.0,12.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Seated Row (Cable)",3,29.0,10.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Seated Row (Cable)",1,29.0,12.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Triceps Extension (Cable)",1,5.7,12.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Triceps Extension (Cable)",3,5.7,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Triceps Extension (Cable)",2,5.7,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pull Up (Assisted)",1,73.0,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pull Up (Assisted)",2,73.0,4.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pull Up (Assisted)",3,77.0,5.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Dumbbell)",0,0,0.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Dumbbell)",2,4.0,5.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Dumbbell)",1,4.0,8.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Barbell)",1,10.0,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Barbell)",0,0,0.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Barbell)",0,0,0.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Squat (Barbell)",3,30.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Squat (Barbell)",2,30.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Squat (Barbell)",1,20.0,5.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Squat (Barbell)",4,30.0,7.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Leg Press",2,10.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Leg Press",1,0.0,5.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Leg Press",3,10.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Leg Extension (Machine)",2,5.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Leg Extension (Machine)",1,5.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Leg Extension (Machine)",3,5.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Seated Leg Curl (Machine)",3,25.0,12.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Seated Leg Curl (Machine)",2,25.0,12.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Seated Leg Curl (Machine)",1,18.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Calf Press on Seated Leg Press",3,39.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Calf Press on Seated Leg Press",1,39.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Calf Press on Seated Leg Press",2,39.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Kettlebell Carry",2,12.0,60.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Kettlebell Carry",1,12.0,60.0,0,0.0,
```

# data\uploads\strong_20250913_133242.csv

```csv
Date,Workout Name,Duration,Exercise Name,Set Order,Weight,Reps,Distance,Seconds,RPE
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",3,10.0,17.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",2,10.0,16.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",2,10.0,10.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",3,10.0,8.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",3,15.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",4,15.0,10.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",2,15.0,15.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",1,10.0,20.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",2,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",1,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",0,0,0.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",3,11.0,8.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",1,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",2,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",1,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",2,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",2,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",1,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",3,11.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",3,15.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",2,15.0,7.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",1,10.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",0,0,0.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",3,9.0,6.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",1,9.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",2,9.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",3,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",1,11.0,12.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",2,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",2,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",3,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",3,8.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",2,8.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Bicep Curl (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",1,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",2,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",3,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",3,18.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",2,18.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",1,11.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",2,23.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",3,23.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",1,17.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Glute Kickback (Machine)",2,5.0,7.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Glute Kickback (Machine)",1,5.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",1,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",2,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",3,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",3,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",2,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",1,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",3,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",2,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",3,4.0,7.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",2,4.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",1,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",3,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",2,23.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",2,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",1,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",3,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",3,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",1,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",2,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",2,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",3,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",1,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",2,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",3,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",1,18.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pull Up (Assisted)",2,77.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pull Up (Assisted)",1,77.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",1,25.0,14.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",2,25.0,14.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",3,25.0,14.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",0,0,0.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Seated Row (Cable)",1,29.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Seated Row (Cable)",2,29.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Seated Row (Cable)",3,29.0,9.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pec Deck (Machine)",2,25.0,9.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pec Deck (Machine)",1,25.0,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pec Deck (Machine)",3,25.0,7.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Triceps Extension (Cable)",1,5.7,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Triceps Extension (Cable)",2,5.7,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Triceps Extension (Cable)",3,5.7,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Dumbbell)",2,12.0,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Dumbbell)",1,12.0,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Dumbbell)",0,0,0.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Bicep Curl (Dumbbell)",2,5.0,5.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Bicep Curl (Dumbbell)",1,5.0,6.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Barbell)",2,30.0,6.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Barbell)",1,20.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Barbell)",3,30.0,7.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Chest Press (Machine)",3,25.0,4.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Chest Press (Machine)",2,25.0,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Chest Press (Machine)",1,18.0,12.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Shoulder Press (Machine)",1,9.0,8.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Shoulder Press (Machine)",2,9.0,7.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Shoulder Press (Machine)",3,9.0,7.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pec Deck (Machine)",2,25.0,8.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pec Deck (Machine)",3,25.0,7.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Seated Row (Cable)",3,29.0,10.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Seated Row (Cable)",2,29.0,12.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Seated Row (Cable)",1,29.0,12.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Triceps Extension (Cable)",3,5.7,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Triceps Extension (Cable)",1,5.7,12.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Triceps Extension (Cable)",2,5.7,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pull Up (Assisted)",2,73.0,4.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pull Up (Assisted)",3,77.0,5.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pull Up (Assisted)",1,73.0,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Dumbbell)",1,4.0,8.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Dumbbell)",2,4.0,5.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Dumbbell)",0,0,0.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Barbell)",0,0,0.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Barbell)",1,10.0,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Barbell)",0,0,0.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Squat (Barbell)",4,30.0,7.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Squat (Barbell)",2,30.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Squat (Barbell)",3,30.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Squat (Barbell)",1,20.0,5.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Leg Press",2,10.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Leg Press",1,0.0,5.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Leg Press",3,10.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Leg Extension (Machine)",2,5.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Leg Extension (Machine)",1,5.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Leg Extension (Machine)",3,5.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Seated Leg Curl (Machine)",2,25.0,12.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Seated Leg Curl (Machine)",3,25.0,12.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Seated Leg Curl (Machine)",1,18.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Calf Press on Seated Leg Press",1,39.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Calf Press on Seated Leg Press",3,39.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Calf Press on Seated Leg Press",2,39.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Kettlebell Carry",2,12.0,60.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Kettlebell Carry",1,12.0,60.0,0,0.0,
```

# data\uploads\strong_20250913_133313.csv

```csv
Date,Workout Name,Duration,Exercise Name,Set Order,Weight,Reps,Distance,Seconds,RPE
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",3,10.0,17.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bent Over Row (Barbell)",2,10.0,16.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",2,10.0,10.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",1,10.0,13.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Bicep Curl (Barbell)",3,10.0,8.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",2,15.0,15.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",4,15.0,10.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",1,10.0,20.0,0,0.0,
2024-11-28 17:36:08,"B",35m,"Lat Pulldown (Machine)",3,15.0,13.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",1,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",0,0,0.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Press",2,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",2,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",3,11.0,8.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Chest Press (Machine)",1,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",3,0.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",1,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Leg Extension (Machine)",2,5.0,5.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",1,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",2,11.0,10.0,0,0.0,
2025-08-19 13:47:15,"Midday Workout",30m,"Seated Row (Cable)",3,11.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",3,15.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",1,10.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",2,15.0,7.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Lat Pulldown (Machine)",0,0,0.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",2,9.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",1,9.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Shoulder Press (Machine)",3,9.0,6.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",1,11.0,12.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",3,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Seated Row (Cable)",2,17.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",2,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",3,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",3,8.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",2,8.0,8.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Squat (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-22 12:02:37,"Midday Workout",41m,"Bicep Curl (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",2,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",3,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Leg Press",1,0.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",2,18.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",1,11.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Chest Press (Machine)",3,18.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",3,23.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",2,23.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Seated Row (Cable)",1,17.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Glute Kickback (Machine)",1,5.0,10.0,0,0.0,
2025-08-26 13:27:10,"Tuesdays",35m,"Glute Kickback (Machine)",2,5.0,7.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",2,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",3,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Seated Row (Cable)",1,23.0,12.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",3,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Pec Deck (Machine)",2,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",3,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",2,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Squat (Dumbbell)",1,10.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",3,4.0,7.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",1,4.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Bicep Curl (Dumbbell)",2,4.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",1,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",3,23.0,10.0,0,0.0,
2025-08-29 13:15:19,"Fridays",40m,"Lat Pulldown (Cable)",2,23.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",1,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",2,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Chest Press (Machine)",3,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",1,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",3,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Row (Cable)",2,29.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",2,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",1,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Leg Extension (Machine)",3,5.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",1,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",2,18.0,10.0,0,0.0,
2025-09-02 13:19:12,"Tuesdays",42m,"Seated Leg Curl (Machine)",3,18.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pull Up (Assisted)",1,77.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pull Up (Assisted)",2,77.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",0,0,0.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",3,25.0,14.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",2,25.0,14.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Lat Pulldown (Cable)",1,25.0,14.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Seated Row (Cable)",1,29.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Seated Row (Cable)",2,29.0,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Seated Row (Cable)",3,29.0,9.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pec Deck (Machine)",3,25.0,7.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pec Deck (Machine)",2,25.0,9.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Pec Deck (Machine)",1,25.0,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Triceps Extension (Cable)",1,5.7,10.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Triceps Extension (Cable)",2,5.7,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Triceps Extension (Cable)",3,5.7,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Dumbbell)",0,0,0.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Dumbbell)",1,12.0,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Dumbbell)",2,12.0,12.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Bicep Curl (Dumbbell)",2,5.0,5.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Bicep Curl (Dumbbell)",1,5.0,6.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Barbell)",2,30.0,6.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Barbell)",3,30.0,7.0,0,0.0,
2025-09-05 12:37:47,"Fridays",1h 19m,"Squat (Barbell)",1,20.0,10.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Chest Press (Machine)",3,25.0,4.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Chest Press (Machine)",2,25.0,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Chest Press (Machine)",1,18.0,12.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Shoulder Press (Machine)",1,9.0,8.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Shoulder Press (Machine)",2,9.0,7.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Shoulder Press (Machine)",3,9.0,7.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pec Deck (Machine)",3,25.0,7.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pec Deck (Machine)",2,25.0,8.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pec Deck (Machine)",1,18.0,10.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Seated Row (Cable)",3,29.0,10.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Seated Row (Cable)",1,29.0,12.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Seated Row (Cable)",2,29.0,12.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Triceps Extension (Cable)",3,5.7,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Triceps Extension (Cable)",1,5.7,12.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Triceps Extension (Cable)",2,5.7,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pull Up (Assisted)",2,73.0,4.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pull Up (Assisted)",1,73.0,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Pull Up (Assisted)",3,77.0,5.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Dumbbell)",0,0,0.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Dumbbell)",2,4.0,5.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Dumbbell)",1,4.0,8.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Barbell)",0,0,0.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Barbell)",1,10.0,6.0,0,0.0,
2025-09-09 13:12:12,"Tuesdays",1h 5m,"Bicep Curl (Barbell)",0,0,0.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Squat (Barbell)",4,30.0,7.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Squat (Barbell)",2,30.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Squat (Barbell)",1,20.0,5.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Squat (Barbell)",3,30.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Leg Press",3,10.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Leg Press",1,0.0,5.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Leg Press",2,10.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Leg Extension (Machine)",2,5.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Leg Extension (Machine)",1,5.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Leg Extension (Machine)",3,5.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Seated Leg Curl (Machine)",1,18.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Seated Leg Curl (Machine)",3,25.0,12.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Seated Leg Curl (Machine)",2,25.0,12.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Calf Press on Seated Leg Press",1,39.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Calf Press on Seated Leg Press",2,39.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Calf Press on Seated Leg Press",3,39.0,10.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Kettlebell Carry",2,12.0,60.0,0,0.0,
2025-09-11 18:16:11,"Fridays",1h 1m,"Kettlebell Carry",1,12.0,60.0,0,0.0,
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
  strong-statistics:
    build: .
    container_name: strong-statistics
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

# readme.md

```md
# strong-statistics

Self‑hosted strength‑training analytics for **Strong** app exports. Import your CSV, see PRs, volume trends, rep ranges, and workout history — all stored locally in SQLite.

![Dashboard overview](screenshots/full%20page%20desktop.png)

---

## 🚀 TL;DR (self‑host)

\`\`\`bash
git clone https://github.com/DaKheera47/strong-statistics.git
cd strong-statistics
cp .env.example .env   # set INGEST_TOKEN to a long random string
docker compose up -d
\`\`\`

Then open:

* Dashboard → [http://localhost:8000/](http://localhost:8000/)

---

## ⚙️ Configuration (minimal)

Edit `.env` before first run:

| Variable       | Required | Default | What it does                                  |
| -------------- | -------- | ------- | --------------------------------------------- |
| `INGEST_TOKEN` | ✅        | —       | Secret required to upload CSVs via `/ingest`. |
| `APP_PORT`     | ❌        | `8000`  | Web port inside the container.                |
| `DATA_DIR`     | ❌        | `/data` | Where the SQLite DB (`strong.db`) lives.      |
| `LOG_DIR`      | ❌        | `/logs` | Where app logs are written.                   |

Data & logs are bind‑mounted to `./data` and `./logs` by the included `docker-compose.yml`.

---

## 📥 Import your Strong data

1. **Export from Strong** (iOS/Android): Settings → **Export Data** → **CSV**.
2. **Upload to strong-statistics** using your token.

**cURL**

\`\`\`bash
curl -X POST "http://localhost:8000/ingest?token=$INGEST_TOKEN" \
  -F "file=@/path/to/strong-export.csv"
\`\`\`

**HTTPie**

\`\`\`bash
http -f POST :8000/ingest?token=$INGEST_TOKEN file@/path/to/strong-export.csv
\`\`\`

**Expected response**

\`\`\`json
{
  "status": "ok",
  "rows_received": 1234,
  "rows_inserted": 1230,
  "duplicates_skipped": 4,
  "workouts_detected": 87
}
\`\`\`

> Safe to re‑upload newer exports — duplicates are ignored.

---

## 📱 iOS Shortcut: one‑tap export → ingest

**Goal:** export from the Strong app, the iOS share sheet pops up, you tap a shortcut, and it POSTs the CSV straight to your server.

![iOS Shortcut share sheet](screenshots/shortcut.jpg)

### A) Create the shortcut (one‑time)

1. Open **Shortcuts** on iOS → tap **+** to create a new shortcut.
2. Name it **“Send to strong‑statistics”**.
3. Tap the **info (ⓘ)** button → enable **Show in Share Sheet** → under **Accepts**, select **Files** (CSV).
4. Add action **Get Contents of URL**:

   * **URL:** `https://YOUR_DOMAIN/ingest?token=<TOKEN>`

     > Replace `YOUR_DOMAIN` and `<TOKEN>` with your real domain and **INGEST\_TOKEN**.
   * **Method:** `POST`
   * **Request Body:** `Form`
   * Add form field: **Name** `file` → **Type** `File` → **Value** **Shortcut Input** (a.k.a. Provided Input)
   * (Optional) If you prefer header auth instead of query: set **Headers** → `X-Token: <your INGEST_TOKEN>` and remove `?token=...` from the URL.
5. (Optional) Add **Show Result** to see the JSON response after upload.

> If you don’t see the shortcut in the share sheet later, scroll to the bottom → **Edit Actions** → enable it.

### B) Use it every time

1. In **Strong**: **Settings → Export Data**.
2. The **share sheet** opens automatically → select **Send to strong‑statistics**.
3. Wait a moment; you’ll get a success response. Open your dashboard to see new data.

**Tip:** Large exports can take a few seconds; you can re‑run later — duplicates are skipped.

---

---

## 📊 Using the dashboard

* Visit `/` for the main dashboard.
* Click a date on the calendar to see that workout.
* Share a workout page at `/workout/YYYY-MM-DD`.

### Workout detail example

![Workout detail view](screenshots/one%20workout.png)

---

## 🔌 Handy API endpoints

(Full list with schemas at `/docs`.)

* `GET /health` → `{ "status": "ok" }`
* `POST /ingest?token=<TOKEN>` → upload CSV (needs `<TOKEN>`)
* `GET /api/personal-records`
* `GET /api/calendar?year=2025&month=8`
* `GET /api/workout/2025-08-14`
* `GET /api/volume?group=week`

---

## 🔒 Quick security note

* Keep `INGEST_TOKEN` secret. Don’t post it in screenshots.

---

## ♻️ Update the app

From the repo root:

\`\`\`bash
git pull
docker compose up -d --build
\`\`\`

---

## 🧪 Troubleshooting

* **401 on `/ingest`** → missing/incorrect `?token=`.
* **400 on `/ingest`** → wrong form field (must be `file`) or not a CSV.
* **`database is locked`** → try again; avoid concurrent imports; SQLite is single‑writer.
* **CORS errors** → if you changed origins, set `ALLOWED_ORIGINS` in `.env`.

---

## 📝 License

MIT.

---

## 📫 Contact

- **Discord:** `dakheera47`
- **Email:** [shaheer30sarfaraz@gmail.com](mailto\:shaheer30sarfaraz@gmail.com)
- **Website:** [https://dakheera47.com](https://dakheera47.com)

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
REM strong-statistics launcher (development convenience)
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

echo [START] strong-statistics binding %HOST%:%PORT% (Ctrl+C to stop)
uvicorn app.main:app --host %HOST% --port %PORT% --reload
set CODE=%ERRORLEVEL%

echo [EXIT] Uvicorn exited with code %CODE%
endlocal & exit /b %CODE%

:fail
echo [ERROR] Startup failed.
endlocal & exit /b 1

```

# screenshots\full page desktop.png

This is a binary file of the type: Image

# screenshots\one workout.png

This is a binary file of the type: Image

# screenshots\shortcut.jpg

This is a binary file of the type: Image

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

