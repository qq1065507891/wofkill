"""Contract tests for werewolf_agent.runtime.agent_adapter.

These tests guard architectural invariants of the agent adapter module
(static analysis via ``inspect.getsource`` for contract / API-shape
properties that are not easily testable through normal invocation).

Post-review architecture fixes (A5, A6, ...) attach new test classes here.
"""

from __future__ import annotations

import inspect

import pytest

from werewolf_agent.runtime import agent_adapter


class TestSheriffPickSpeechOrderContract:
    """审查 A5: agent_sheriff_pick_speech_order 不应 model_copy 改 legal_actions 改 task_type。"""

    def test_pick_speech_order_no_model_copy_legal_actions_mutation(self):
        """检查 agent_adapter 中 agent_sheriff_pick_speech_order 函数体不含 model_copy 改 legal_actions。"""
        if not hasattr(agent_adapter, "agent_sheriff_pick_speech_order"):
            pytest.skip("function not found (may have been moved/renamed)")
        fn_src = inspect.getsource(agent_adapter.agent_sheriff_pick_speech_order)
        # 旧实现含 model_copy(update={"legal_actions": [...]}) 或类似
        # 排除注释中提及（去掉所有以 # 开头的行）
        code_lines = [
            line for line in fn_src.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        code_only = "\n".join(code_lines)
        assert "model_copy" not in code_only or "legal_actions" not in code_only, (
            f"agent_sheriff_pick_speech_order still mutates legal_actions via model_copy:\n{fn_src}"
        )
