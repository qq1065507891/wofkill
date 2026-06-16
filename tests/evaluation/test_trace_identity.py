from werewolf_agent.evaluation.trace_identity import (
    ActionIndexAllocator,
    DecisionIdentity,
    make_trace_id,
)


def test_make_trace_id_uses_approved_public_format() -> None:
    assert make_trace_id(
        game_id="g1",
        player_id="p01",
        phase="vote",
        day_number=2,
        night_number=1,
        task_type="vote",
        action_index=4,
    ) == "g1:p01:vote:D2:N1:vote:4"


def test_decision_identity_returns_trace_id() -> None:
    identity = DecisionIdentity("g1", "p01", "vote", 2, 1, "vote", 4)
    assert identity.trace_id() == "g1:p01:vote:D2:N1:vote:4"


def test_action_index_allocator_is_monotonic_per_game() -> None:
    allocator = ActionIndexAllocator()
    assert allocator.next("g1") == 0
    assert allocator.next("g1") == 1
    assert allocator.next("g2") == 0
