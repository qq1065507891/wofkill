# -*- coding: utf-8 -*-
"""
验证死亡结算批次 V2 的解析、校验和序列化契约。

作者: Project contributors
创建日期: 2026-07-15

使用示例:
    >>> python -m pytest tests/runtime/test_resolution_batches.py -q
"""

from __future__ import annotations

import pytest

from werewolf_agent.core.models import Death, GameEvent, GameState, PlayerState
from werewolf_agent.core.resolution_batches import (
    ResolutionBatchV2,
    parse_resolution_batch,
    serialize_resolution_batch,
)


@pytest.mark.parametrize(
    ("raw", "phase", "number", "cause"),
    [
        ("day_3", "day", 3, "unknown"),
        ("day_3_vote", "day", 3, "vote"),
        ("day_3_self_destruct", "day", 3, "self_destruct"),
        ("night_3", "night", 3, "unknown"),
        ("night_3_wolf_kill", "night", 3, "wolf_kill"),
        ("night_3_witch_poison", "night", 3, "witch_poison"),
        ("day_3_hunter_shot", "day", 3, "hunter_shot"),
        ("night_3_rule_effect", "night", 3, "rule_effect"),
    ],
)
def test_parse_legacy_resolution_batch(
    raw: str,
    phase: str,
    number: int,
    cause: str,
) -> None:
    result = parse_resolution_batch(raw)

    assert result.batch == ResolutionBatchV2(phase=phase, number=number, cause=cause)
    assert result.raw_value == raw
    assert result.batch_parse_failed is False


@pytest.mark.parametrize(
    "raw",
    [
        "day_BAD",
        "day_3_exile",
        "day_3_vote_extra",
        "d3",
        "unknown",
        "",
    ],
)
def test_parse_unknown_legacy_resolution_batch_fails_closed(raw: str) -> None:
    result = parse_resolution_batch(raw)

    assert result.batch is None
    assert result.raw_value == raw
    assert result.batch_parse_failed is True


def test_parse_resolution_batch_v2_instance_is_already_normalized() -> None:
    batch = ResolutionBatchV2(phase="night", number=2, cause="wolf_kill")

    result = parse_resolution_batch(batch)

    assert result.batch is batch
    assert result.raw_value is None
    assert result.batch_parse_failed is False


def test_parse_resolution_batch_mapping_is_validated() -> None:
    result = parse_resolution_batch(
        {"phase": "day", "number": 4, "cause": "self_destruct"}
    )

    assert result.batch == ResolutionBatchV2(
        phase="day", number=4, cause="self_destruct"
    )
    assert result.raw_value is None
    assert result.batch_parse_failed is False


@pytest.mark.parametrize(
    "value",
    [
        {"phase": "dusk", "number": 1, "cause": "vote"},
        {"phase": "day", "number": -1, "cause": "vote"},
        {"phase": "day", "number": True, "cause": "vote"},
        {"phase": "day", "number": 1, "cause": "exile"},
        {"phase": "day", "number": 1},
    ],
)
def test_parse_invalid_mapping_fails_closed_with_stable_json(value: dict[str, object]) -> None:
    result = parse_resolution_batch(value)

    assert result.batch is None
    assert result.raw_value is not None
    assert result.raw_value.startswith("{")
    assert result.batch_parse_failed is True


class _OpaqueValue:
    def __init__(self, secret: str) -> None:
        self.secret = secret


def test_invalid_mapping_sanitizer_is_stable_and_hides_object_details() -> None:
    first = parse_resolution_batch({"phase": "bad", "payload": _OpaqueValue("one")})
    second = parse_resolution_batch({"phase": "bad", "payload": _OpaqueValue("two")})

    assert first.raw_value == second.raw_value
    assert "0x" not in first.raw_value
    assert "one" not in first.raw_value
    assert "two" not in first.raw_value
    assert "_OpaqueValue" in first.raw_value


def test_invalid_mapping_sanitizer_preserves_distinct_typed_keys() -> None:
    result = parse_resolution_batch({1: "integer", "1": "string"})

    assert result.batch_parse_failed is True
    assert result.raw_value is not None
    sanitized = __import__("json").loads(result.raw_value)
    assert sanitized["$mapping"] == [[1, "integer"], ["1", "string"]]


def test_invalid_mapping_sanitizer_fails_closed_on_canonical_key_collision() -> None:
    result = parse_resolution_batch({_OpaqueValue("one"): 1, _OpaqueValue("two"): 2})

    assert result.raw_value is not None
    sanitized = __import__("json").loads(result.raw_value)
    assert "$mapping_key_collision" in sanitized
    assert "one" not in result.raw_value
    assert "two" not in result.raw_value


def test_invalid_mapping_sanitizer_marks_cycles_and_depth_limit() -> None:
    cycle: dict[str, object] = {"phase": "bad"}
    cycle["payload"] = cycle
    deep: object = "leaf"
    for _ in range(20):
        deep = [deep]

    cycle_result = parse_resolution_batch(cycle)
    deep_result = parse_resolution_batch({"phase": "bad", "payload": deep})

    assert cycle_result.raw_value is not None
    assert "$cycle" in cycle_result.raw_value
    assert deep_result.raw_value is not None
    assert "$max_depth" in deep_result.raw_value


def test_invalid_mapping_sanitizer_sorts_sets_deterministically() -> None:
    first = parse_resolution_batch({"phase": "bad", "payload": {3, 1, 2}})
    second = parse_resolution_batch({"payload": {2, 3, 1}, "phase": "bad"})

    assert first.raw_value == second.raw_value
    assert first.raw_value is not None
    assert __import__("json").loads(first.raw_value)["payload"] == {
        "$set": [1, 2, 3]
    }


def test_resolution_batch_v2_rejects_invalid_constructor_values() -> None:
    with pytest.raises(ValueError):
        ResolutionBatchV2(phase="day", number=True, cause="vote")
    with pytest.raises(ValueError):
        ResolutionBatchV2(phase="day", number=-1, cause="vote")
    with pytest.raises(ValueError):
        ResolutionBatchV2(phase="dusk", number=1, cause="vote")
    with pytest.raises(ValueError):
        ResolutionBatchV2(phase="day", number=1, cause="exile")


def test_serialize_resolution_batch_is_the_only_json_safe_shape() -> None:
    batch = ResolutionBatchV2(phase="day", number=3, cause="vote")

    serialized, parse_failed = serialize_resolution_batch(batch)

    assert serialized == {"phase": "day", "number": 3, "cause": "vote"}
    assert parse_failed is False


def test_serialize_unknown_legacy_batch_preserves_raw_and_failure_marker() -> None:
    serialized, parse_failed = serialize_resolution_batch("day_BAD")

    assert serialized == "day_BAD"
    assert parse_failed is True


def test_exile_producer_writes_v2_death_and_json_safe_event() -> None:
    from werewolf_agent.engine.rule_exile import resolve_exile

    state = GameState(
        game_id="exile-v2",
        day_number=3,
        players={"p01": PlayerState(id="p01", role="villager")},
    )
    captured = []

    _, events = resolve_exile(
        state,
        target_id="p01",
        apply_death_fn=lambda current, death: captured.append(death) or current,
        apply_idiot_reveal_fn=lambda current, _player_id: current,
    )

    assert captured[0].resolution_batch == ResolutionBatchV2("day", 3, "vote")
    assert captured[0].resolution_batch_parse_failed is False
    assert events[-1].payload["resolution_batch"] == {
        "phase": "day",
        "number": 3,
        "cause": "vote",
    }


def test_event_reducer_normalizes_legacy_batch_and_preserves_bad_raw() -> None:
    from werewolf_agent.engine.event_reducer import EventReducer

    players = {
        "p01": PlayerState(id="p01", role="villager"),
        "p02": PlayerState(id="p02", role="villager"),
    }
    reducer = EventReducer(raw={})
    state = GameState(game_id="replay-v2", players=players)

    state = reducer.reduce_event(
        state,
        GameEvent(
            type="player_died",
            payload={
                "player_id": "p01",
                "reason": "wolf_kill",
                "timing": "night",
                "resolution_batch": "night_2",
            },
        ),
    )
    state = reducer.reduce_event(
        state,
        GameEvent(
            type="player_died",
            payload={
                "player_id": "p02",
                "reason": "rule_effect",
                "timing": "day",
                "resolution_batch": "day_BAD",
            },
        ),
    )

    assert state.deaths[0].resolution_batch == ResolutionBatchV2(
        "night", 2, "unknown"
    )
    assert state.deaths[0].resolution_batch_parse_failed is False
    assert state.deaths[1].resolution_batch == "day_BAD"
    assert state.deaths[1].resolution_batch_parse_failed is True


def test_event_reducer_missing_self_destruct_day_fails_closed_without_crash() -> None:
    from werewolf_agent.engine.event_reducer import EventReducer

    state = GameState(
        game_id="legacy-self-destruct",
        players={"wolf": PlayerState(id="wolf", role="werewolf")},
    )

    reduced = EventReducer(raw={}).reduce_event(
        state,
        GameEvent(type="werewolf_self_destructed", payload={"player_id": "wolf"}),
    )

    assert reduced.deaths[0].resolution_batch == "day_?_self_destruct"
    assert reduced.deaths[0].resolution_batch_parse_failed is True


@pytest.mark.parametrize(
    ("batch", "input_marker"),
    [
        ("day_BAD", False),
        (ResolutionBatchV2("day", 2, "rule_effect"), True),
    ],
)
def test_apply_death_ors_input_and_parser_failure_markers(
    batch: str | ResolutionBatchV2,
    input_marker: bool,
) -> None:
    from werewolf_agent.engine.rule_death import apply_death

    state = GameState(
        game_id="marker-or",
        players={"p01": PlayerState(id="p01", role="villager")},
    )
    result = apply_death(
        state,
        Death(
            "p01",
            "rule_effect",
            "day",
            batch,
            resolution_batch_parse_failed=input_marker,
        ),
        can_leave_last_words_fn=lambda **_kwargs: False,
        can_hunter_shoot_fn=lambda *_args, **_kwargs: False,
    )

    assert result.deaths[0].resolution_batch == batch
    assert result.deaths[0].resolution_batch_parse_failed is True
    event_payload = result.events[-1].payload
    assert event_payload["resolution_batch_parse_failed"] is True
    if batch == "day_BAD":
        assert event_payload["resolution_batch"] == "day_BAD"
