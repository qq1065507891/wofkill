"""Analyze recent saved game logs with balance guardrails."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from werewolf_agent.evaluation.balance_audit import (
    compute_balance_audit,
    load_game_logs,
)


def build_recent_balance_report(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Load saved game logs and return the balance audit report."""
    games = load_game_logs(paths)
    return compute_balance_audit(games)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze saved werewolf game logs for recent balance skew.",
    )
    parser.add_argument("game_json", nargs="+", type=Path)
    args = parser.parse_args(argv)

    report = build_recent_balance_report(args.game_json)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
