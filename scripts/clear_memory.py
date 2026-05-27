"""Clear all persistent game memory from PostgreSQL.

Usage:
    python scripts/clear_memory.py
    python scripts/clear_memory.py --yes  # skip confirmation

Connects to the same database as run_real_game.py and truncates all
game, RAG, and memory snapshot tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DSN = "postgresql://wofkill:wofkill-dev@localhost:5432/wofkill"


def main() -> None:
    if "--yes" not in sys.argv:
        ans = input("Clear ALL game, RAG, and memory data from PostgreSQL? [y/N] ")
        if ans.strip().lower() != "y":
            print("Cancelled.")
            return

    # Ensure PostgreSQL is running via Docker
    import subprocess
    r = subprocess.run(
        ["docker", "compose", "ps", "--services", "--filter", "status=running"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if "postgres" not in r.stdout:
        print("PostgreSQL not running. Start it with:")
        print("  docker compose up -d postgres")
        print("Then re-run this script.")
        sys.exit(1)

    try:
        from werewolf_agent.storage.postgres_store import PostgresGameRepository
    except ImportError:
        print("ERROR: PostgresGameRepository not available.")
        sys.exit(1)

    repo = PostgresGameRepository(DSN)
    conn = repo._ensure_connection()

    tables = [
        "games",
        "rag_entries",
        "memory_snapshots",
        "custom_configs",
    ]
    total = 0
    for table in tables:
        cur = conn.execute(f"SELECT count(*) FROM {table}")
        count = cur.fetchone()[0]
        if count > 0:
            conn.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")
            total += count
            print(f"  Cleared {table}: {count} rows")
        else:
            print(f"  {table}: already empty")

    if total == 0:
        print("No data to clear — database is clean.")
    else:
        print(f"Done. {total} total rows cleared.")


if __name__ == "__main__":
    main()
