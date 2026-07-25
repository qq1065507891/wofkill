# -*- coding: utf-8 -*-
"""
验证讨论摘要 V2 Schema、旧 checkpoint 迁移和文本兼容投影。

作者: Project contributors
创建日期: 2026-07-25
修改日期: 2026-07-25

使用示例:
    >>> python -m pytest tests/agents/test_discussion_summary.py -q
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from werewolf_agent.agents.discussion_summary import (
    DiscussionSummary,
    discussion_summary_for_player,
    discussion_summary_text,
)
from werewolf_agent.agents.player import PlayerAgent
from werewolf_agent.agents.schemas import AgentContext, TaskType
from werewolf_agent.model_gateway.router import ModelRouter
from werewolf_agent.model_gateway.usage_records import GenerateResult


def _v2_payload(summary: str = "我怀疑p03") -> dict[str, object]:
    return {
        "summary": summary,
        "suspected_players": ["p03"],
        "trusted_players": ["p02"],
        "vote_target": "p03",
        "evidence_refs": ["event-7"],
    }


def test_discussion_summary_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        DiscussionSummary.model_validate({
            **_v2_payload(),
            "private_reason": "我是狼人，所以要推动p03。",
        })


@pytest.mark.parametrize(
    ("field", "coercible_value"),
    [
        ("suspected_players", ("p03",)),
        ("trusted_players", {"p02"}),
        ("evidence_refs", ("event-7",)),
    ],
)
def test_discussion_summary_rejects_coercible_container_types(
    field: str,
    coercible_value: object,
) -> None:
    with pytest.raises(ValidationError):
        DiscussionSummary.model_validate({
            **_v2_payload(),
            field: coercible_value,
        })


def test_unversioned_coercible_mapping_does_not_upgrade() -> None:
    payload = _v2_payload()
    payload["suspected_players"] = ("p03",)
    state = {"discussion_positions": {"p01": payload}}

    assert discussion_summary_for_player(state, "p01") is None
    assert "discussion_positions_version" not in state
    assert state["discussion_positions"]["p01"]["suspected_players"] == (
        "p03",
    )


def test_legacy_summary_string_upgrades_to_v2() -> None:
    state = {
        "discussion_positions": {
            "p01": "我怀疑p03",
            "p02": "我信任p04",
        },
    }

    summary = discussion_summary_for_player(state, "p01")

    assert summary == DiscussionSummary(summary="我怀疑p03")
    assert state["discussion_positions_version"] == 2
    assert state["discussion_positions"] == {
        "p01": DiscussionSummary(summary="我怀疑p03").model_dump(),
        "p02": DiscussionSummary(summary="我信任p04").model_dump(),
    }


def test_unversioned_v2_mapping_upgrades_in_memory() -> None:
    state = {
        "discussion_positions": {
            "p01": _v2_payload(),
            "p02": _v2_payload("我信任p03"),
        },
    }

    summary = discussion_summary_for_player(state, "p02")

    assert summary == DiscussionSummary.model_validate(_v2_payload("我信任p03"))
    assert state["discussion_positions_version"] == 2
    assert state["discussion_positions"]["p02"] == _v2_payload("我信任p03")


def test_explicit_v2_payload_is_read_without_coercion() -> None:
    state = {
        "discussion_positions_version": 2,
        "discussion_positions": {"p01": _v2_payload()},
    }

    assert discussion_summary_for_player(
        state,
        "p01",
    ) == DiscussionSummary.model_validate(_v2_payload())


def test_v2_version_schema_conflict_fails_closed() -> None:
    state = {
        "discussion_positions_version": 2,
        "discussion_positions": {"p01": "legacy string"},
    }

    assert discussion_summary_for_player(state, "p01") is None
    assert state["discussion_positions"]["p01"] == "legacy string"


def test_explicit_v2_global_mapping_conflict_fails_closed() -> None:
    state = {
        "discussion_positions_version": 2,
        "discussion_positions": {
            "p01": _v2_payload(),
            "p02": "legacy string conflicts with explicit V2",
        },
    }

    assert discussion_summary_for_player(state, "p01") is None
    assert state["discussion_positions"]["p01"] == _v2_payload()
    assert state["discussion_positions"]["p02"] == (
        "legacy string conflicts with explicit V2"
    )


def test_unknown_explicit_version_fails_closed() -> None:
    state = {
        "discussion_positions_version": 3,
        "discussion_positions": {"p01": _v2_payload()},
    }

    assert discussion_summary_for_player(state, "p01") is None


def test_mixed_unversioned_entries_validate_per_player_without_upgrade() -> None:
    state = {
        "discussion_positions": {
            "p01": _v2_payload(),
            "p02": {"summary": "字段冲突", "unexpected": True},
        },
    }

    assert discussion_summary_for_player(
        state,
        "p01",
    ) == DiscussionSummary.model_validate(_v2_payload())
    assert discussion_summary_for_player(state, "p02") is None
    assert "discussion_positions_version" not in state


def test_missing_player_returns_none() -> None:
    assert discussion_summary_for_player(
        {"discussion_positions": {}},
        "p09",
    ) is None


def test_text_projection_is_deterministic_and_excludes_private_fields() -> None:
    summary = DiscussionSummary.model_validate(_v2_payload())

    first = discussion_summary_text(summary)
    second = discussion_summary_text(summary)

    assert first == second
    assert first == (
        "我怀疑p03\n"
        "怀疑玩家: p03\n"
        "信任玩家: p02\n"
        "投票目标: p03\n"
        "证据引用: event-7"
    )
    assert "private" not in first.lower()
    assert "我是狼人" not in first


def test_text_json_provider_request_contains_exact_narrow_schema() -> None:
    class _CapturingProvider:
        name = "capture"

        def __init__(self) -> None:
            self.requests: list[dict[str, object]] = []

        def generate(
            self,
            prompt,
            config,
            system_prompt=None,
            tools=None,
            tool_choice=None,
            final_prompt_observer=None,
        ):
            self.requests.append({
                "prompt": prompt,
                "mode": config.structured_output_mode,
                "tools": tools,
                "tool_choice": tool_choice,
            })
            return GenerateResult(
                text=DiscussionSummary(
                    summary="p02发言前后矛盾。",
                    suspected_players=["p02"],
                    vote_target="p02",
                    evidence_refs=["speech-1"],
                ).model_dump_json(),
                provider=self.name,
                model=config.model,
            )

    provider = _CapturingProvider()
    router = ModelRouter(
        model_profiles={
            "summary_model": {
                "model": "summary-model",
                "structured_output": {"mode": "text_json"},
                "reasoning": {"level": "medium"},
                "retry_count": 0,
            },
        },
        llm_profiles={
            "player": {
                "tasks": {
                    "discussion_summary": {
                        "provider": "capture",
                        "model_profile": "summary_model",
                    },
                },
            },
        },
        player_assignments={"p01": "player"},
        providers={"capture": provider},
    )
    agent = PlayerAgent(agent_id="p01", model_router=router)

    summary = agent.summarize_discussion(AgentContext(
        agent_id="p01",
        task_type=TaskType.DISCUSSION_SUMMARY,
        strategy_directive={"transcript_text": "[p02]: 我怀疑p03。"},
    ))

    request = provider.requests[0]
    prompt = str(request["prompt"])
    assert summary.vote_target == "p02"
    assert request["mode"] == "text_json"
    assert request["tool_choice"] is None
    assert '"additionalProperties": false' in prompt
    assert '"required": ["summary"]' in prompt
    for field in DiscussionSummary.model_fields:
        assert f'"{field}"' in prompt
    for generic_field in ("action_type", "speech", "reason", "private_intent"):
        assert f'"{generic_field}"' not in prompt
