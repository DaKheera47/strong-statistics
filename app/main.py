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
    compute_session_agg,
    top_exercises,
    prs,
    tuesday_strength,
    exercise_progress,
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
async def ingest(request: Request, token: str = Query(""), file: UploadFile = File(...)):
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

    content_type = file.content_type or ""
    if content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(status_code=400, detail=f"unsupported content type: {content_type}")

    # Size guard (stream to temp file while counting)
    raw = await file.read()
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

    logger.info("ingested file=%s inserted_rows=%s", stored_name, inserted)
    return {"stored": stored_name, "rows": inserted}


@app.get("/api/sessions")
async def api_sessions():
    return compute_session_agg()


@app.get("/api/top-exercises")
async def api_top_exercises(limit: int = 15):
    return top_exercises(limit=limit)


@app.get("/api/prs")
async def api_prs(exercise: str):
    return prs(exercise)


@app.get("/api/tuesday-strength")
async def api_tuesday_strength():
    return tuesday_strength()


@app.get("/api/exercise-progress")
async def api_exercise_progress(exercise: str):
    return exercise_progress(exercise)


@app.get("/api/exercises")
async def api_exercises():
    return list_exercises()


@app.get("/health")
async def health():
    return {"ok": True, "time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), "last_ingested_at": get_meta("last_ingested_at")}
