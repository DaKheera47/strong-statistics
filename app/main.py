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
