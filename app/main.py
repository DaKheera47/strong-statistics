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
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=2_000_000, backupCount=3)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI(title="Lifting Pipeline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

init_db()

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


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
