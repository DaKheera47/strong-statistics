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
