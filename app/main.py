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
