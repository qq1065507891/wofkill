# -*- coding: utf-8 -*-
"""
验证赛后反思提示与真实 Agent 链的隐私和核验契约。

作者: Project contributors
创建日期: 2026-07-13
"""

from __future__ import annotations

import json

from werewolf_agent.agents.schemas import ActionType, AgentContext, PlayerAction
from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState
from werewolf_agent.memory.reflection_synthesis import (
    ReflectionClaim,
    ReflectionDraft,
    verify_reflection_draft,
)
from werewolf_agent.runtime import agent_adapter
from werewolf_agent.runtime.reflection_prompt import build_reflection_prompt
from werewolf_agent.runtime.nodes import summary


def _state() -> GameState:
    return GameState(
        game_id="g-live", phase="finished", winning_faction="good",
        players={
            "p01": PlayerState(id="p01", role="seer"),
            "p02": PlayerState(id="p02", role="werewolf", alive=False),
            "p03": PlayerState(id="p03", role="witch"),
        },
        events=[
            GameEvent(type="roles_assigned", payload={}),
            GameEvent(type="vote", payload={"voter": "p01", "target": "p02"}),
            GameEvent(type="player_died", payload={"player_id": "p02", "reason": "exile"}),
            GameEvent(type="witch_poison_used", payload={"target_id": "p02", "visibility": "witch_private"}),
        ],
        deaths=[Death(
            player_id="p02", reason="exile", timing="day_vote",
            resolution_batch="day_1_vote",
        )],
    )


def test_reflection_prompt_event_refs_match_verifier_contract() -> None:
    state = _state()
    prompt = build_reflection_prompt(
        player=state.players["p01"], winner="good",
        hybrid_master_faction=None, state=state,
    )
    refs_json = prompt.split("VERIFIABLE_EVENT_REFS_JSON=", 1)[1].splitlines()[0]
    refs = json.loads(refs_json)

    assert {item["claim_type"] for item in refs} >= {"role", "vote", "death", "potion"}
    assert all(item["event_ref"].startswith("g-live:") for item in refs)
    assert all(item["visibility"] == "moderator_postgame" for item in refs)
    assert "hidden_thinking" not in prompt
    assert "provider_response" not in prompt
    for claim_type in ("role", "vote", "death", "potion"):
        item = next(candidate for candidate in refs if candidate["claim_type"] == claim_type)
        claim = ReflectionClaim(
            claim_id=f"claim-{claim_type}",
            event_ref=item["event_ref"],
            claim_type=claim_type,
            subject_id=item["subject_id"],
            target_id=item.get("target_id", ""),
            value=item.get("value", ""),
        )
        verified = verify_reflection_draft(ReflectionDraft(claims=[claim]), state)
        assert verified.verified_claims == [claim]


def test_agent_reflection_verifies_real_ids_before_anonymization(monkeypatch) -> None:
    draft = json.dumps({
        "claims": [{
            "claim_id": "c1", "event_ref": "g-live:1", "claim_type": "vote",
            "subject_id": "p01", "target_id": "p02", "value": "",
        }],
        "lessons": [{
            "lesson_id": "l1", "abstraction": "p01 投 p02 前应复核公开票型",
            "claim_dependencies": ["c1"],
        }, {
            "lesson_id": "l2", "abstraction": "p02 质疑 p01 时应先比较替代解释",
            "claim_dependencies": ["c1"],
        }],
    }, ensure_ascii=False)

    def fake_context(engine, gs, player_id, task_type, **kwargs):
        return AgentContext(agent_id=player_id, task_type=task_type)

    class Agent:
        def act(self, context):
            return PlayerAction(action_type=ActionType.SPEECH, speech=draft), None

    class Registry:
        def get_agent(self, player_id):
            return Agent()

    monkeypatch.setattr(agent_adapter, "build_agent_context", fake_context)
    result = agent_adapter._agent_reflection(
        {"game_state": _state()}, engine=None, registry=Registry(), player_id="p01",
    )

    verification = result["reflection_verification"]
    assert verification["verified_fact_count"] == 1
    assert verification["rejected_fact_count"] == 0
    abstractions = [lesson["abstraction"] for lesson in verification["verified_lessons"]]
    assert abstractions == [
        "历史玩家A 投 历史玩家B 前应复核公开票型",
        "历史玩家B 质疑 历史玩家A 时应先比较替代解释",
    ]
    assert draft not in json.dumps(result, ensure_ascii=False)


def test_reflection_complete_contains_only_moderator_safe_verification(monkeypatch) -> None:
    safe = {
        "status": "verified", "verified_fact_count": 1,
        "verified_claim_ids": [], "rejected_claim_ids": [],
        "verified_lessons": [{"lesson_id": "l1", "abstraction": "先复核公开票型"}],
        "rejected_fact_count": 0, "rejected_lesson_count": 0,
    }
    monkeypatch.setattr(
        summary,
        "_dispatch_agent",
        lambda *args, **kwargs: {"reflection_verification": safe},
    )
    result = summary.reflection({"game_state": _state(), "engine": None})
    event = result["game_state"].events[-1]
    serialized = json.dumps(event.payload, ensure_ascii=False)

    assert event.type == "reflection_complete"
    assert event.payload["visibility"] == "moderator_only"
    assert event.payload["entries"][0]["verification"] == {
        **safe,
        "decision_id": "reflection:g-live:p01",
    }
    assert "reflection_text" not in serialized
    assert "provider_response" not in serialized


def test_prompt_refs_and_verifier_share_supported_fact_registry() -> None:
    state = GameState(
        game_id="g-registry", phase="finished", winning_faction="good",
        players={
            "p01": PlayerState(id="p01", role="seer"),
            "p02": PlayerState(id="p02", role="werewolf", alive=False),
            "p03": PlayerState(id="p03", role="hunter", alive=False),
        },
        events=[
            GameEvent(type="seer_check", payload={"target_id": "p02", "alignment": "werewolf"}),
            GameEvent(type="sheriff_vote_record", payload={"votes": [{"voter": "p01", "target": "p03"}]}),
            GameEvent(type="hunter_shot_public", payload={"hunter_id": "p03", "target_id": "p02"}),
            GameEvent(type="victory", payload={"winner": "good", "winning_faction": "good", "reason": "all_werewolves_out"}),
        ],
    )
    prompt = build_reflection_prompt(state.players["p01"], "good", None, state)
    refs = json.loads(prompt.split("VERIFIABLE_EVENT_REFS_JSON=", 1)[1].splitlines()[0])

    assert {ref["claim_type"] for ref in refs} >= {"seer_check", "vote", "skill", "victory"}
    for index, ref in enumerate(refs):
        claim = ReflectionClaim(
            claim_id=f"c{index}", event_ref=ref["event_ref"],
            claim_type=ref["claim_type"], subject_id=ref["subject_id"],
            target_id=ref.get("target_id", ""), value=ref.get("value", ""),
        )
        assert verify_reflection_draft(ReflectionDraft(claims=[claim]), state).verified_claims == [claim]


def test_reflection_complete_rebuilds_strict_allowlist_from_untrusted_adapter_result(monkeypatch) -> None:
    poisoned = {
        "status": "verified", "decision_id": "reflection:p01:1",
        "verified_fact_count": 1,
        "verified_lessons": [{
            "lesson_id": "l1", "abstraction": "p01 应先复核 p02 的票型",
            "provider_response": "SECRET_IN_LESSON",
        }],
        "rejected_fact_count": 0, "rejected_lesson_count": 0,
        "provider_response": "SECRET_PROVIDER", "raw": "SECRET_RAW", "prompt": "SECRET_PROMPT",
    }
    monkeypatch.setattr(
        summary, "_dispatch_agent",
        lambda *args, **kwargs: {"reflection_verification": poisoned},
    )

    event = summary.reflection({"game_state": _state(), "engine": None})["game_state"].events[-1]
    serialized = json.dumps(event.payload, ensure_ascii=False)
    verification = event.payload["entries"][0]["verification"]

    assert set(verification) == {
        "status", "decision_id", "verified_fact_count", "verified_lessons",
        "verified_claim_ids", "rejected_claim_ids",
        "rejected_fact_count", "rejected_lesson_count",
    }
    assert set(verification["verified_lessons"][0]) == {"lesson_id", "abstraction"}
    assert verification["verified_lessons"][0]["abstraction"] == "历史玩家A 应先复核 历史玩家B 的票型"
    assert "SECRET" not in serialized


def test_safe_reflection_verification_never_accepts_pre_persistence_count() -> None:
    from werewolf_agent.runtime.reflection_events import safe_reflection_verification

    safe = safe_reflection_verification(
        {"status": "verified", "persisted_rejected_fact_count": 0},
        decision_id="reflection:g1:p01",
    )
    legacy = safe_reflection_verification(
        {"status": "verified"},
        decision_id="reflection:g1:p01",
    )

    assert "persisted_rejected_fact_count" not in safe
    assert "persisted_rejected_fact_count" not in legacy
