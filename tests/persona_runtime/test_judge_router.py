"""Tests for JudgeProfileRouter — YAML load, profile resolution, task_styles."""

from __future__ import annotations

import pytest

from werewolf_agent.persona_runtime.judge_router import (
    JudgePersonaSnapshot,
    JudgeProfileRouter,
)


PROFILES_YAML = "config/personas/judge_profiles.yaml"


# ---------------------------------------------------------------------------
# P5 (post-review-v2): judge_router task_styles 应被拼进 system_prompt。
# ---------------------------------------------------------------------------


class TestJudgeRouterTaskStylesUsed:
    """P5: persona.task_styles[task_type] 必须拼进 snapshot.system_prompt。"""

    def test_task_styles_in_system_prompt(self) -> None:
        """当 profile 含 task_styles[task_type] 时，system_prompt 应包含该文本。"""
        router = JudgeProfileRouter(
            profiles={
                "p5_demo": {
                    "display_name": "P5 Demo",
                    "tone_variant": "neutral",
                    "base": {},
                    "task_styles": {
                        "judge_phase": "P5_TASK_STYLE_TOKEN",
                        "judge_vote_calling": "P5_VOTE_TOKEN",
                    },
                    "broadcast_patterns": {},
                    "system_prompt": "BASE_SYSTEM_PROMPT",
                },
            }
        )
        snap = router.resolve("p5_demo", "judge_phase")
        # 必须包含 base system_prompt 原文
        assert "BASE_SYSTEM_PROMPT" in snap.system_prompt, (
            f"P5: base system_prompt lost; got {snap.system_prompt!r}"
        )
        # 必须拼入 task_styles[task_type] 文本
        assert "P5_TASK_STYLE_TOKEN" in snap.system_prompt, (
            f"P5: task_styles[task_type] not injected into system_prompt; "
            f"got {snap.system_prompt!r}"
        )
        # 不应注入无关 task_type 的内容
        assert "P5_VOTE_TOKEN" not in snap.system_prompt, (
            f"P5: wrong task_type injected; got {snap.system_prompt!r}"
        )

    def test_task_styles_missing_is_no_op(self) -> None:
        """当 profile 缺 task_styles 或当前 task_type 未定义时，
        system_prompt 不应被破坏（base 仍保留）。"""
        router = JudgeProfileRouter(
            profiles={
                "p5_empty": {
                    "display_name": "Empty",
                    "tone_variant": "neutral",
                    "base": {},
                    "task_styles": {},
                    "broadcast_patterns": {},
                    "system_prompt": "BASE_ONLY",
                },
            }
        )
        snap = router.resolve("p5_empty", "judge_phase")
        assert snap.system_prompt == "BASE_ONLY", (
            f"P5: empty task_styles must not modify system_prompt; "
            f"got {snap.system_prompt!r}"
        )

    def test_task_styles_used_for_real_profiles(self) -> None:
        """真实 judge_profiles.yaml 里所有 profile 的 task_styles 都应被
        注入到 system_prompt。"""
        router = JudgeProfileRouter.from_yaml(PROFILES_YAML)
        for pid in router.list_profiles():
            snap = router.resolve(pid, "judge_phase")
            ts = snap.task_styles.get("judge_phase", "")
            if ts:
                assert ts in snap.system_prompt, (
                    f"P5: profile {pid!r} task_styles['judge_phase'] not "
                    f"in system_prompt; got {snap.system_prompt!r}"
                )
