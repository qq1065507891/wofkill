# -*- coding: utf-8 -*-
"""
验证 prompt 文本清洗和 JSON 压缩工具的行为。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.agents.prompt_formatting import clean_prompt_text
    >>> clean_prompt_text("p01 发言")
"""

import json

from werewolf_agent.agents.prompt_formatting import (
    MAX_JSON_CONTEXT_CHARS,
    clean_prompt_text,
    compact_json,
)


def test_clean_prompt_text_scrubs_player_ids_and_flattens_whitespace():
    text = clean_prompt_text(" p12\n声称 player_7 支持 agent-3 ")

    assert text == "历史玩家 声称 历史玩家 支持 历史玩家"


def test_compact_json_keeps_large_payload_under_context_budget():
    payload = {
        f"key_{index}": {
            "speaker": f"p{index}",
            "text": "很长的学习记录" * 80,
        }
        for index in range(30)
    }

    rendered = compact_json(payload)
    parsed = json.loads(rendered)

    assert len(rendered) <= MAX_JSON_CONTEXT_CHARS
    assert parsed["truncated"] is True
    assert parsed["original_type"] == "dict"
