# -*- coding: utf-8 -*-
"""
测试狼队计划适配器的纯数据处理辅助函数。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.wolf_team_plan_support import build_prior_plan_summary
    >>> build_prior_plan_summary({})
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.runtime.wolf_team_plan_support import (
    build_prior_plan_summary,
    build_wolf_role_definitions,
    build_wolf_team_plan_evidence,
    collect_current_wolf_discussion_text,
    validate_wolf_team_plan_membership,
)


@dataclass
class _Plan:
    fake_seer: str | None = None
    pusher: str | None = None
    hooker: str | None = None
    deep_cover: str | None = None
    night_kill_primary: str | None = None
    night_kill_backup: str | None = None


def _make_game_state() -> GameState:
    players = {
        "w1": PlayerState(id="w1", role="werewolf", alive=True),
        "p1": PlayerState(id="p1", role="villager", alive=True),
    }
    return GameState(
        game_id="wolf_plan_support",
        phase="night",
        night_number=2,
        players=players,
        events=[
            GameEvent(
                type="wolf_discussion",
                payload={
                    "wolf_id": "w1",
                    "round": 1,
                    "night_number": 1,
                    "text": "旧夜聊不应出现",
                },
            ),
            GameEvent(
                type="wolf_discussion",
                payload={
                    "wolf_id": "w1",
                    "round": 2,
                    "night_number": 2,
                    "text": "今晚优先刀 p1",
                },
            ),
        ],
    )


def test_collect_current_wolf_discussion_text_uses_only_current_night() -> None:
    """只收集当前夜且非空的狼队夜聊文本。"""
    assert collect_current_wolf_discussion_text(_make_game_state()) == "[第2轮 w1]: 今晚优先刀 p1"


def test_collect_current_wolf_discussion_text_uses_empty_fallback() -> None:
    """当前夜没有夜聊时返回稳定占位文本。"""
    gs = replace(_make_game_state(), events=[])

    assert collect_current_wolf_discussion_text(gs) == "(本夜无夜聊文本)"


def test_build_prior_plan_summary_formats_first_night_and_existing_plan() -> None:
    """上局摘要在首夜和延续计划两种场景下都保持旧格式。"""
    assert build_prior_plan_summary({}) == "无上局计划 (首夜)"
    assert build_prior_plan_summary({
        "fake_seer": "w1",
        "pusher": "w2",
        "hooker": "w3",
        "deep_cover": "w4",
        "night_kill_primary": "p1",
    }) == "上夜计划: fake_seer=w1, pusher=w2, hooker=w3, deep_cover=w4, primary=p1"


def test_build_wolf_role_definitions_keeps_first_line_only() -> None:
    """角色定义只取每个策略说明的首行，避免提示词过长。"""
    assert build_wolf_role_definitions({"fake_seer": "第一行\n第二行"}) == "- fake_seer: 第一行"


def test_validate_wolf_team_plan_membership_returns_first_error() -> None:
    """成员校验返回第一个越界字段，供重试提示复用。"""
    err = validate_wolf_team_plan_membership(
        _Plan(pusher="p1"),
        alive_wolves=["w1", "w2"],
        alive_non_wolves=["p1"],
    )

    assert err == "pusher=p1 not in alive_wolves=['w1', 'w2']"


def test_validate_wolf_team_plan_membership_accepts_valid_plan() -> None:
    """所有成员都在合法候选集内时不返回错误。"""
    err = validate_wolf_team_plan_membership(
        _Plan(fake_seer="w1", night_kill_primary="p1"),
        alive_wolves=["w1", "w2"],
        alive_non_wolves=["p1"],
    )

    assert err is None


def test_build_wolf_team_plan_evidence_uses_llm_reason_tags() -> None:
    """为主刀和备刀目标生成下游审计需要的合成证据。"""
    assert build_wolf_team_plan_evidence(
        {
            "night_kill_primary": "p1",
            "night_kill_backup": "p2",
        },
        captain_id="w1",
    ) == [
        {"target": "p1", "wolf_id": "w1", "reason": "llm_captain_decision"},
        {"target": "p2", "wolf_id": "w1", "reason": "llm_captain_backup"},
    ]
