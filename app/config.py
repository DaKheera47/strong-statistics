"""Configuration management for import types and validation."""

from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import importlib

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yml"

_config_cache: Optional[Dict[str, Any]] = None


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yml with caching."""
    global _config_cache
    if _config_cache is None:
        with open(CONFIG_PATH, "r") as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


def get_enabled_import_type() -> Optional[str]:
    """Get the single enabled import type from configuration."""
    config = load_config()

    enabled_types = [
        import_type
        for import_type, settings in config["import_types"].items()
        if settings.get("enabled", False)
    ]

    if len(enabled_types) == 1:
        return enabled_types[0]
    elif len(enabled_types) == 0:
        return None
    else:
        # Multiple types enabled - return first one or raise error
        return enabled_types[0]


def detect_format_mismatch(header_line: str, expected_type: str) -> Optional[str]:
    """Detect if header matches a different format than expected and suggest correction."""
    config = load_config()

    for import_type, settings in config["import_types"].items():
        if import_type == expected_type:
            continue

        expected_header = settings.get("header_validation", "")
        if header_line.startswith(expected_header):
            return import_type

    return None


def get_processor_info(import_type: str) -> tuple[str, str]:
    """Get processor module and function name for given import type."""
    config = load_config()

    if import_type not in config["import_types"]:
        raise ValueError(f"Unknown import type: {import_type}")

    settings = config["import_types"][import_type]
    module_name = settings["processor_module"]
    function_name = settings["processor_function"]

    return module_name, function_name


def get_processor_function(import_type: str):
    """Dynamically import and return the processor function for given type."""
    module_name, function_name = get_processor_info(import_type)

    try:
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import {function_name} from {module_name}: {e}")


def validate_file_constraints(raw_bytes: bytes, content_type: str) -> None:
    """Validate file size and content type constraints."""
    config = load_config()

    # Check file size
    max_size_mb = config["validation"]["max_file_size_mb"]
    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)")

    # Check content type
    supported_types = config["validation"]["supported_content_types"]
    if content_type not in supported_types:
        raise ValueError(f"Unsupported content type: {content_type}")
