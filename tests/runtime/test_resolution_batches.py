# -*- coding: utf-8 -*-
"""
验证死亡结算批次 V2 的解析、校验和序列化契约。

作者: Project contributors
创建日期: 2026-07-15
修改日期: 2026-07-16

使用示例:
    >>> python -m pytest tests/runtime/test_resolution_batches.py -q
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

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


def test_apply_death_does_not_trust_marked_night_batch_for_last_words() -> None:
    from werewolf_agent.engine.rule_death import apply_death

    observed_night_numbers: list[int] = []
    state = GameState(
        game_id="marked-night-number",
        night_number=0,
        players={"p01": PlayerState(id="p01", role="villager")},
    )

    apply_death(
        state,
        Death(
            "p01",
            "wolf_kill",
            "night",
            ResolutionBatchV2("night", 9, "wolf_kill"),
            resolution_batch_parse_failed=True,
        ),
        can_leave_last_words_fn=lambda **kwargs: (
            observed_night_numbers.append(kwargs["night_number"]) or False
        ),
        can_hunter_shoot_fn=lambda *_args, **_kwargs: False,
    )

    assert observed_night_numbers == [0]


def test_apply_death_day_batch_keeps_state_night_number_for_last_words() -> None:
    from werewolf_agent.engine.rule_death import apply_death

    observed_night_numbers: list[int] = []
    state = GameState(
        game_id="day-batch-night-number",
        night_number=3,
        players={"p01": PlayerState(id="p01", role="villager")},
    )

    apply_death(
        state,
        Death("p01", "exile", "day_vote", ResolutionBatchV2("day", 9, "vote")),
        can_leave_last_words_fn=lambda **kwargs: (
            observed_night_numbers.append(kwargs["night_number"]) or False
        ),
        can_hunter_shoot_fn=lambda *_args, **_kwargs: False,
    )

    assert observed_night_numbers == [3]


class _CountingMapping(Mapping[str, object]):
    def __init__(
        self,
        *,
        actual_size: int,
        reported_size: int,
        items_error: bool = False,
        malformed_items: bool = False,
        len_error: bool = False,
        getitem_error: bool = False,
    ) -> None:
        self.actual_size = actual_size
        self.reported_size = reported_size
        self.items_error = items_error
        self.malformed_items = malformed_items
        self.len_error = len_error
        self.getitem_error = getitem_error
        self.item_reads = 0

    def __len__(self) -> int:
        if self.len_error:
            raise RuntimeError("len failed")
        return self.reported_size

    def __iter__(self) -> Iterator[str]:
        yield from ("phase", "number", "cause")

    def __getitem__(self, key: str) -> object:
        if self.getitem_error:
            raise RuntimeError("getitem failed")
        values: dict[str, object] = {
            "phase": "night",
            "number": 1,
            "cause": "wolf_kill",
        }
        return values[key]

    def items(self) -> Iterator[tuple[str, object]]:
        if self.items_error:
            raise RuntimeError("items failed")
        if self.malformed_items:
            yield ("missing-value",)  # type: ignore[misc]
            return
        for index in range(self.actual_size):
            self.item_reads += 1
            yield f"key-{index}", index


def test_mapping_sanitizer_caps_reads_when_len_lies() -> None:
    mapping = _CountingMapping(actual_size=10_000, reported_size=3)

    first = parse_resolution_batch(mapping)
    first_reads = mapping.item_reads
    mapping.item_reads = 0
    second = parse_resolution_batch(mapping)

    assert first.batch is None
    assert first.batch_parse_failed is True
    assert first.raw_value == second.raw_value
    assert first_reads <= 65
    assert mapping.item_reads <= 65


@pytest.mark.parametrize(
    "mapping",
    [
        _CountingMapping(actual_size=3, reported_size=3, items_error=True),
        _CountingMapping(actual_size=1, reported_size=1, malformed_items=True),
        _CountingMapping(actual_size=3, reported_size=3, len_error=True),
        _CountingMapping(actual_size=3, reported_size=3, getitem_error=True),
    ],
)
def test_mapping_parser_fails_closed_on_mapping_protocol_errors(
    mapping: _CountingMapping,
) -> None:
    first = parse_resolution_batch(mapping)
    second = parse_resolution_batch(mapping)

    assert first.batch is None
    assert first.batch_parse_failed is True
    assert first.raw_value == second.raw_value
    assert first.raw_value is not None
    assert "failed" not in first.raw_value


class _LyingSet(set[int]):
    def __init__(self) -> None:
        super().__init__()
        self.item_reads = 0

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator[int]:
        for value in range(10_000):
            self.item_reads += 1
            yield value


def test_set_subclass_is_summarized_without_unbounded_iteration() -> None:
    values = _LyingSet()

    result = parse_resolution_batch({"phase": "bad", "payload": values})

    assert result.batch_parse_failed is True
    assert values.item_reads == 0
    assert result.raw_value is not None
    assert "$set_summary" in result.raw_value


class _SnapshotMapping(Mapping[str, object]):
    def __init__(
        self,
        *,
        iter_keys: tuple[str, ...] = ("phase", "number", "cause"),
        item_pairs: tuple[tuple[str, object], ...] = (
            ("phase", "night"),
            ("number", 1),
            ("cause", "wolf_kill"),
        ),
        getitem_values: Mapping[str, object] | None = None,
        iter_error: bool = False,
    ) -> None:
        self.iter_keys = iter_keys
        self.item_pairs = item_pairs
        self.getitem_values = dict(
            getitem_values
            or {"phase": "night", "number": 1, "cause": "wolf_kill"}
        )
        self.iter_error = iter_error

    def __len__(self) -> int:
        return len(self.item_pairs)

    def __iter__(self) -> Iterator[str]:
        if self.iter_error:
            raise RuntimeError("iter failed")
        yield from self.iter_keys

    def __getitem__(self, key: str) -> object:
        return self.getitem_values[key]

    def items(self) -> Iterator[tuple[str, object]]:
        yield from self.item_pairs


class _ListPairSnapshotMapping(_SnapshotMapping):
    def items(self) -> Iterator[tuple[str, object]]:
        yield ["phase", "night"]  # type: ignore[misc]
        yield ["number", 1]  # type: ignore[misc]
        yield ["cause", "wolf_kill"]  # type: ignore[misc]


def test_consistent_custom_mapping_snapshot_is_supported() -> None:
    result = parse_resolution_batch(_SnapshotMapping())

    assert result.batch == ResolutionBatchV2("night", 1, "wolf_kill")
    assert result.batch_parse_failed is False


@pytest.mark.parametrize(
    "mapping",
    [
        _SnapshotMapping(iter_error=True),
        _ListPairSnapshotMapping(),
        _SnapshotMapping(iter_keys=("phase", "number", "unexpected")),
        _SnapshotMapping(
            item_pairs=(
                ("phase", "day"),
                ("number", 1),
                ("cause", "vote"),
            ),
        ),
        _SnapshotMapping(
            item_pairs=(
                ("phase", "night"),
                ("phase", "day"),
                ("number", 1),
            ),
        ),
    ],
)
def test_inconsistent_mapping_snapshot_fails_closed(
    mapping: _SnapshotMapping,
) -> None:
    result = parse_resolution_batch(mapping)

    assert result.batch is None
    assert result.batch_parse_failed is True
    assert result.raw_value is not None
