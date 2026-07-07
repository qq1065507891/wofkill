# -*- coding: utf-8 -*-
"""
迁移并清理 Reflection Memory V2 的历史快照边界数据。

作者: Project contributors
修改日期: 2026-07-07

使用示例:
    python scripts/migrate_reflection_memory_v2.py --backend postgres
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from werewolf_agent.memory.migration import (
    clean_snapshot_reflection_boundary,
    dry_run_legacy_reflection,
)


def _repo_from_args(args: argparse.Namespace) -> Any:
    if args.sqlite:
        from werewolf_agent.storage.sqlite_store import SqliteGameRepository

        return SqliteGameRepository(args.sqlite)

    from werewolf_agent.storage.production import (
        ProductionStorageConfig,
        create_game_repository,
    )

    return create_game_repository(ProductionStorageConfig(backend=args.backend))


def _backup(repo: Any, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = backup_dir / f"reflection_memory_v2_backup_{stamp}.json"
    snapshots: list[dict[str, Any]] = []
    for meta in repo.list_memory_snapshots():
        snapshot_id = meta.get("snapshot_id", "")
        snapshots.append({
            "snapshot_id": snapshot_id,
            "snapshot_json": repo.load_memory_snapshot(snapshot_id),
        })
    payload = {
        "created_at": stamp,
        "reflections": repo.load_all_reflections(),
        "memory_snapshots": snapshots,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reflection Memory V2 dry-run report and snapshot-boundary cleanup helper"
        )
    )
    parser.add_argument("--sqlite", default="", help="SQLite database path")
    parser.add_argument("--backend", default="postgres", choices=["postgres", "sqlite", "memory"])
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Apply snapshot cleanup after backup. Reflection rows are reported "
            "but not mutated by this helper."
        ),
    )
    parser.add_argument(
        "--backup-dir",
        default="tmp_analysis/reflection_memory_v2",
        help="Directory for backup JSON files",
    )
    args = parser.parse_args()

    repo = _repo_from_args(args)
    backup_path = _backup(repo, Path(args.backup_dir))

    reflection_report = [
        dry_run_legacy_reflection(row)
        for row in repo.load_all_reflections()
    ]
    snapshot_report: list[dict[str, Any]] = []
    for meta in repo.list_memory_snapshots():
        snapshot_id = meta.get("snapshot_id", "")
        original = repo.load_memory_snapshot(snapshot_id) or {}
        cleaned = clean_snapshot_reflection_boundary(original)
        changed = cleaned != original
        snapshot_report.append({
            "snapshot_id": snapshot_id,
            "changed": changed,
            "old_reflection_count": len(original.get("reflections", []) or []),
            "new_reflection_count": len(cleaned.get("reflections", []) or []),
        })
        if args.apply and changed:
            repo.save_memory_snapshot(snapshot_id, cleaned)

    print(json.dumps({
        "backup_path": str(backup_path),
        "apply": args.apply,
        "reflection_rows": "report_only",
        "reflections": reflection_report,
        "snapshots": snapshot_report,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
