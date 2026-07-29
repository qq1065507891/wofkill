# -*- coding: utf-8 -*-
"""
导出仓库审查用的自主玩家提案 JSON Schema 快照。

作者: Project contributors
创建日期: 2026-07-29

使用示例:
    conda run -n wofkill python -m scripts.export_player_agent_schemas
"""

from __future__ import annotations

import json
from pathlib import Path

from werewolf_agent.player_agents.contracts.schema_catalog import (
    speech_proposal_schema,
)

OUTPUT = (
    Path(__file__).parents[1]
    / "tests"
    / "fixtures"
    / "player_agents"
    / "speech_proposal_schema_v1.json"
)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            speech_proposal_schema(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
