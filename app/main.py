"""FastAPI application entrypoint."""

from __future__ import annotations

import faulthandler
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request

from .db import get_meta, init_db
from .config import get_enabled_import_type, detect_format_mismatch, get_processor_function, validate_file_constraints

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

    # Validate file constraints (size, content type)
    try:
        validate_file_constraints(raw, content_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Get configured import type
    import_type = get_enabled_import_type()
    if not import_type:
        raise HTTPException(
            status_code=500,
            detail="No import type is enabled in configuration"
        )

    # Check for format mismatch and provide helpful error message
    header_line = raw.splitlines()[0].decode(errors="ignore") if raw else ""
    detected_format = detect_format_mismatch(header_line, import_type)

    if detected_format:
        raise HTTPException(
            status_code=400,
            detail=f"CSV header looks like {detected_format} format, but configuration is set to {import_type}. "
                   f"Please check your config.yml file and enable the correct import type."
        )

    # Optional: Validate that the header actually matches the expected format
    # This provides better error messages for completely unknown formats

    # Store file with import type prefix
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    stored_name = f"{import_type}_{ts}.csv"
    dest = UPLOAD_DIR / stored_name
    dest.write_bytes(raw)

    # Route to appropriate processor based on import type
    try:
        processor_function = get_processor_function(import_type)
        inserted = processor_function(dest)
    except Exception as e:
        logger.exception("failed ingest for %s (type: %s)", stored_name, import_type)
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        "ingested file=%s inserted_rows=%s mode=%s size_bytes=%s type=%s",
        stored_name,
        inserted,
        mode,
        len(raw),
        import_type,
    )
    return {"stored": stored_name, "rows": inserted, "type": import_type}


@app.get("/health")
async def health():
    payload = {
        "ok": True,
        "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_ingested_at": get_meta("last_ingested_at"),
    }
    # Add diagnostic header so we can tell if Cloudflare is reaching origin.
    return JSONResponse(payload, headers={"X-Strong-Origin": "fastapi"})
