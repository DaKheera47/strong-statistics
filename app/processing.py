"""Legacy CSV processing module - redirects to new modular processors."""

import warnings
from pathlib import Path

def process_csv_to_db(csv_path: Path) -> int:
    """Legacy function that redirects to the new modular processing system.

    This function is deprecated and maintained for backward compatibility.
    New code should use the routing system in main.py instead.
    """
    warnings.warn(
        "process_csv_to_db is deprecated. Use the new routing system via /ingest endpoint",
        DeprecationWarning,
        stacklevel=2
    )

    # For backward compatibility, assume it's a Strong CSV
    from .processors.strong_processor import process_strong_csv
    return process_strong_csv(csv_path)
