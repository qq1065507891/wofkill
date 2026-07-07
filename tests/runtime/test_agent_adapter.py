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
            line
            for line in fn_src.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        code_only = "\n".join(code_lines)
        assert "model_copy" not in code_only or "legal_actions" not in code_only, (
            f"agent_sheriff_pick_speech_order still mutates legal_actions via model_copy:\n{fn_src}"
        )


class TestAgentActionPipelineSplit:
    """Task 15: agent_adapter 应退化为 action pipeline 的兼容 facade。"""

    def test_action_pipeline_exports_are_compatibility_imports(self) -> None:
        from werewolf_agent.runtime import agent_action_pipeline

        assert agent_adapter.agent_day_vote is agent_action_pipeline.agent_day_vote
        assert agent_adapter.agent_day_speech is agent_action_pipeline.agent_day_speech
        assert (
            agent_adapter.agent_night_witch is agent_action_pipeline.agent_night_witch
        )
        assert agent_adapter.agent_sheriff_election_speech is (
            agent_action_pipeline.agent_sheriff_election_speech
        )

    def test_sheriff_action_exports_are_compatibility_imports(self) -> None:
        from werewolf_agent.runtime import agent_action_pipeline, agent_sheriff_actions

        sheriff_exports = (
            "agent_sheriff_pick_speech_order",
            "agent_sheriff_endorse",
            "agent_sheriff_vote",
            "agent_sheriff_register",
            "agent_sheriff_withdraw",
            "agent_sheriff_election_speech",
        )

        for export_name in sheriff_exports:
            assert getattr(agent_action_pipeline, export_name) is getattr(
                agent_sheriff_actions, export_name
            )
            assert getattr(agent_adapter, export_name) is getattr(
                agent_sheriff_actions, export_name
            )

    def test_wolf_action_exports_are_compatibility_imports(self) -> None:
        from werewolf_agent.runtime import agent_action_pipeline, agent_wolf_actions

        wolf_exports = (
            "agent_wolf_team_plan",
            "agent_wolf_consensus",
            "agent_wolf_discussion",
        )

        for export_name in wolf_exports:
            assert getattr(agent_action_pipeline, export_name) is getattr(
                agent_wolf_actions, export_name
            )
            assert getattr(agent_adapter, export_name) is getattr(
                agent_wolf_actions, export_name
            )

    def test_special_action_exports_are_compatibility_imports(self) -> None:
        from werewolf_agent.runtime import agent_action_pipeline, agent_special_actions

        special_exports = (
            "agent_night_witch",
            "agent_night_seer",
            "agent_hybrid_choose_master",
            "agent_badge_decision",
            "agent_hunter_shot",
        )

        for export_name in special_exports:
            assert getattr(agent_action_pipeline, export_name) is getattr(
                agent_special_actions, export_name
            )
            assert getattr(agent_adapter, export_name) is getattr(
                agent_special_actions, export_name
            )

    def test_facade_patch_propagates_to_action_pipeline(self, monkeypatch) -> None:
        from werewolf_agent.runtime import (
            agent_action_pipeline,
            agent_sheriff_actions,
            agent_wolf_actions,
        )

        def fake_build_context(*args, **kwargs):  # noqa: ANN002, ANN003
            return None

        monkeypatch.setattr(agent_adapter, "build_agent_context", fake_build_context)

        assert agent_action_pipeline.build_agent_context is fake_build_context
        assert agent_sheriff_actions.build_agent_context is fake_build_context
        assert agent_wolf_actions.build_agent_context is fake_build_context


class TestKillValueAssessmentAdapterContract:
    """P-v3: kill_value_assessment 在 agent_adapter 的两个 call site
    （_build_wolf_kill_directive + _single_wolf_vote）必须共享同一份 cache。

    这两个 call site 历史上各自独立调 evaluate_wolf_kill_target, 每夜
    4 狼 × 2 次 = 8 次 O(N) 重算. P-v3 之后应只剩 4 次（1 次/狼/夜）.
    """

    def test_adapter_and_strategy_share_same_cached_entry(self) -> None:
        """agent_adapter._evaluate_wolf_kill_target 与
        strategy.wolf.evaluate_wolf_kill_target 必须指向同一份 cache.

        防御性: 防止有人未来误改成局部 cache / lru_cache 副本, 重新
        引入 double-call.
        """
        # 清空 cache 拿到基线
        from werewolf_agent.runtime.strategy import wolf as wolf_strategy

        wolf_strategy.clear_kill_value_cache()

        from werewolf_agent.core.models import GameState, PlayerState

        players = {
            f"p{i:02d}": PlayerState(
                id=f"p{i:02d}",
                role="werewolf" if i <= 4 else "villager",
                alive=True,
            )
            for i in range(1, 10)
        }
        gs = GameState(
            game_id="g_dup_test",
            night_number=1,
            players=players,
        )
        legal = [
            pid for pid, p in gs.players.items() if p.alive and p.role != "werewolf"
        ]

        # 经由 agent_adapter 的 re-export 调一次
        agent_adapter._evaluate_wolf_kill_target(gs, "p01", legal)
        # 经由 strategy.wolf 调一次 (相同 key)
        wolf_strategy.evaluate_wolf_kill_target(gs, "p01", legal)
        # 第三次: 再次经 agent_adapter (相同 key)
        agent_adapter._evaluate_wolf_kill_target(gs, "p01", legal)

        # impl 实际被调的次数 - 1 (第一次的真正计算)
        # 通过再次以新 key 调用, 触发 impl 一次, 然后检查 cache size 增量
        # 来推断历史 call 数
        # 简化做法: 直接看 _evaluate_wolf_kill_target_impl 的真实调用次数
        # 这里通过 patch 验证
        from unittest.mock import patch

        wolf_strategy.clear_kill_value_cache()
        with patch.object(
            wolf_strategy,
            "_evaluate_wolf_kill_target_impl",
            wraps=wolf_strategy._evaluate_wolf_kill_target_impl,
        ) as mock_impl:
            # 模拟两个 call site 的 3 次调用 (相同 key)
            agent_adapter._evaluate_wolf_kill_target(gs, "p01", legal)
            wolf_strategy.evaluate_wolf_kill_target(gs, "p01", legal)
            agent_adapter._evaluate_wolf_kill_target(gs, "p01", legal)
            assert mock_impl.call_count == 1, (
                f"shared cache: expected 1 impl call across adapter + strategy, "
                f"got {mock_impl.call_count}"
            )

    def test_wolf_kill_directive_and_single_wolf_vote_use_same_module(self) -> None:
        """_build_wolf_kill_directive 与 _single_wolf_vote 调用的必须是
        ``werewolf_agent.runtime.strategy.wolf.evaluate_wolf_kill_target``
        (或经由 _evaluate_wolf_kill_target 的 re-export) — 不能用本地副本
        或 inline 实现绕开 cache.
        """
        from werewolf_agent.runtime import wolf_kill_support

        assert agent_adapter._build_wolf_kill_directive is (
            wolf_kill_support._build_wolf_kill_directive
        )
        assert agent_adapter._single_wolf_vote is wolf_kill_support._single_wolf_vote

        support_src = inspect.getsource(wolf_kill_support)
        # 必须至少有 2 处引用 evaluate_wolf_kill_target (导入 + 实际调用)
        refs = support_src.count("evaluate_wolf_kill_target")
        assert refs >= 2, (
            f"wolf_kill_support should import + call evaluate_wolf_kill_target "
            f"in at least 2 places (directive + single_wolf_vote), got {refs}"
        )
