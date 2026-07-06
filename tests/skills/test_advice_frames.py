# -*- coding: utf-8 -*-
"""
验证技能建议帧 helper 拆分后的兼容导入。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> python -m pytest tests/skills/test_advice_frames.py -q
"""

from __future__ import annotations


def test_advice_frame_helpers_remain_compatibly_importable() -> None:
    from werewolf_agent.skills import advice_frames
    from werewolf_agent.skills import werewolf_skills

    assert werewolf_skills.PROMPT_INJECTABLE_CAP == advice_frames.PROMPT_INJECTABLE_CAP
    assert werewolf_skills.PROMPT_INJECTABLE_MARKER_TAIL == advice_frames.PROMPT_INJECTABLE_MARKER_TAIL
    assert werewolf_skills._advice_frame is advice_frames._advice_frame
    assert werewolf_skills._cap_prompt_injectable is advice_frames._cap_prompt_injectable
    assert werewolf_skills._push_vote_advice_frame is advice_frames._push_vote_advice_frame
    assert werewolf_skills._counter_claim_advice_frame is advice_frames._counter_claim_advice_frame
    assert werewolf_skills._hide_identity_advice_frame is advice_frames._hide_identity_advice_frame
    assert werewolf_skills._ensure_skill_advice_frame is advice_frames._ensure_skill_advice_frame
    assert werewolf_skills._generic_skill_advice_frame is advice_frames._generic_skill_advice_frame


def test_cap_prompt_injectable_preserves_boundary_contract() -> None:
    from werewolf_agent.skills.advice_frames import (
        PROMPT_INJECTABLE_CAP,
        PROMPT_INJECTABLE_MARKER_TAIL,
        _cap_prompt_injectable,
    )

    exact = "x" * PROMPT_INJECTABLE_CAP
    over = "x" * (PROMPT_INJECTABLE_CAP + 1)

    assert _cap_prompt_injectable("") == ""
    assert _cap_prompt_injectable(exact) == exact
    capped = _cap_prompt_injectable(over)
    assert len(capped) == PROMPT_INJECTABLE_CAP
    assert capped.endswith(PROMPT_INJECTABLE_MARKER_TAIL)
