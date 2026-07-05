# -*- coding: utf-8 -*-
"""
验证 LLM JSON 文本修复和动作候选提取工具。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.agents.json_repair import repair_json_text
    >>> repair_json_text("{action_type:'vote',}")
"""

import json

import pytest

from werewolf_agent.agents.json_repair import (
    extract_json_object_candidates,
    repair_json_text,
)


def test_repair_json_text_handles_comments_unquoted_keys_and_trailing_commas():
    repaired = repair_json_text(
        """
        {
            // LLM 注释
            action_type: 'vote',
            target_id: 'p01',
            confidence: NaN,
        }
        """
    )

    assert json.loads(repaired) == {
        "action_type": "vote",
        "target_id": "p01",
        "confidence": None,
    }


def test_extract_json_object_candidates_rejects_non_action_thinking_object():
    with pytest.raises(ValueError, match="no_action_type_found"):
        extract_json_object_candidates('前置思考 {"thinking": "先分析局势"} 后续文本')
