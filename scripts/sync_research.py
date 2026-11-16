"""Run with python -m scripts.sync_research; scheduling is external.

Requires the same ODDS_API_KEY and RESEARCH_DATABASE_URL as the hosted app.
No database URL means no hosted worker: never fall back to an ephemeral file.
"""
import json
import os

from dotenv import load_dotenv

from src.service import poll
from src.tracking import Store


def main():
    load_dotenv()
    url = os.environ.get("RESEARCH_DATABASE_URL", "")
    if not url.startswith(("postgresql://", "postgres://")):
        print("Worker skipped: external PostgreSQL storage is not configured.")
        return
    try:
        store = Store(url)
    except Exception:
        print("Worker could not open external storage; check private configuration.")
        raise SystemExit(1) from None
    try:
        report = poll(store)
        print(json.dumps(report))
        if report["errors"]:
            raise SystemExit(1)
    finally:
        store.close()


if __name__ == "__main__":
    main()
