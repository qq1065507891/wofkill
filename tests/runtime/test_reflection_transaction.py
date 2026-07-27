# -*- coding: utf-8 -*-
"""
验证赛后反思事务的状态推进、非真空成功和验收闭环。

作者: Project contributors
创建日期: 2026-07-17
修改日期: 2026-07-27

使用示例:
    python -m pytest tests/runtime/test_reflection_transaction.py -q
"""

from __future__ import annotations

import copy
import pickle
from dataclasses import replace

import pytest

from werewolf_agent.runtime import reflection_transaction as transaction_module
from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.evaluation.acceptance_reflection_metrics import (
    compute_reflection_acceptance_metrics,
)
from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig
from werewolf_agent.runtime.nodes import summary
from werewolf_agent.runtime.reflection_events import safe_reflection_verification
from werewolf_agent.runtime.reflection_transaction import (
    PlayerReflectionTransaction,
    ReflectionStage,
    ReflectionTransitionError,
    summarize_reflection_transaction,
)
from werewolf_agent.storage.memory_store import InMemoryGameRepository
from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator


def _advance_valid_entry(
    player_id: str = "p01",
    *,
    persisted: bool = False,
) -> PlayerReflectionTransaction:
    entry = PlayerReflectionTransaction(
        player_id=player_id,
        decision_id=f"reflection:g1:{player_id}",
    )
    entry = entry.advance(ReflectionStage.GENERATED)
    entry = entry.advance(ReflectionStage.SCHEMA_VALIDATED)
    entry = entry.advance(
        ReflectionStage.FACTS_VERIFIED,
        verified_claim_ids=(f"claim-{player_id}",),
    )
    entry = entry.advance(
        ReflectionStage.LESSONS_VERIFIED,
        verified_lesson_ids=(f"lesson-{player_id}",),
    )
    if persisted:
        entry = entry.advance(
            ReflectionStage.PERSISTED,
            entry_id=f"reflection_g1_{player_id}",
        )
    return entry


def test_player_reflection_transaction_accepts_only_adjacent_transitions() -> None:
    entry = _advance_valid_entry(persisted=True)

    assert entry.stage is ReflectionStage.PERSISTED
    assert entry.verified_claim_ids == ("claim-p01",)
    assert entry.verified_lesson_ids == ("lesson-p01",)
    assert entry.entry_id == "reflection_g1_p01"

    with pytest.raises(ReflectionTransitionError, match="illegal reflection transition"):
        PlayerReflectionTransaction(
            player_id="p01", decision_id="reflection:g1:p01",
        ).advance(ReflectionStage.FACTS_VERIFIED)


def test_direct_constructor_and_dataclass_replace_cannot_forge_persisted_state() -> None:
    with pytest.raises(TypeError):
        PlayerReflectionTransaction(  # type: ignore[call-arg]
            player_id="p01",
            decision_id="reflection:g1:p01",
            stage=ReflectionStage.PERSISTED,
            verified_claim_ids=("claim-p01",),
            verified_lesson_ids=("lesson-p01",),
            entry_id="reflection_g1_p01",
        )

    initial = PlayerReflectionTransaction("p01", "reflection:g1:p01")
    with pytest.raises((TypeError, ValueError)):
        replace(
            initial,
            stage=ReflectionStage.PERSISTED,
            verified_claim_ids=("claim-p01",),
            verified_lesson_ids=("lesson-p01",),
            entry_id="reflection_g1_p01",
        )


def test_summarizer_rejects_object_new_bypass_without_transition_provenance() -> None:
    forged = object.__new__(PlayerReflectionTransaction)
    object.__setattr__(forged, "player_id", "p01")
    object.__setattr__(forged, "decision_id", "reflection:g1:p01")
    object.__setattr__(forged, "stage", ReflectionStage.PERSISTED)
    object.__setattr__(forged, "failure_stage", None)
    object.__setattr__(forged, "failure_code", None)
    object.__setattr__(forged, "verified_claim_ids", ("claim-p01",))
    object.__setattr__(forged, "verified_lesson_ids", ("lesson-p01",))
    object.__setattr__(forged, "entry_id", "reflection_g1_p01")

    result = summarize_reflection_transaction(
        [forged], persistence_attempted=True,
    )

    assert result.status == "no_valid_entries"
    assert result.persistence_complete is False


def test_summarizer_rejects_forged_complete_object_with_importable_seal() -> None:
    assert "_TRANSACTION_SEAL" not in vars(transaction_module)
    forged = object.__new__(PlayerReflectionTransaction)
    values = {
        "player_id": "p01",
        "decision_id": "reflection:g1:p01",
        "stage": ReflectionStage.PERSISTED,
        "failure_stage": None,
        "failure_code": None,
        "verified_claim_ids": ("claim-p01",),
        "verified_lesson_ids": ("lesson-p01",),
        "entry_id": "reflection_g1_p01",
        "_stage_path": tuple(ReflectionStage),
        "_seal": object(),
    }
    for field, value in values.items():
        object.__setattr__(forged, field, value)

    result = summarize_reflection_transaction(
        [forged], persistence_attempted=True,
    )

    assert result.status == "no_valid_entries"
    assert result.persistence_complete is False


def test_factory_controlled_transaction_survives_pickle_checkpoint() -> None:
    entry = _advance_valid_entry(persisted=True)

    restored = pickle.loads(pickle.dumps(entry))
    result = summarize_reflection_transaction(
        [restored], persistence_attempted=True,
    )

    assert restored.to_payload() == entry.to_payload()
    assert result.status == "complete"
    assert result.persistence_complete is True


def test_snapshot_type_constructor_cannot_clone_registered_provenance() -> None:
    legitimate = _advance_valid_entry(persisted=True)
    forged = type(legitimate)(tuple(legitimate))

    result = summarize_reflection_transaction(
        [forged], persistence_attempted=True,
    )

    assert result.status == "no_valid_entries"
    assert result.persistence_complete is False


def test_snapshot_builtin_new_cannot_clone_registered_provenance() -> None:
    legitimate = _advance_valid_entry(persisted=True)
    try:
        forged = tuple.__new__(type(legitimate), tuple(legitimate))
    except TypeError:
        return

    result = summarize_reflection_transaction(
        [forged], persistence_attempted=True,
    )

    assert result.status == "no_valid_entries"
    assert result.persistence_complete is False


def test_snapshot_new_immutable_base_cannot_clone_registered_provenance() -> None:
    legitimate = _advance_valid_entry(persisted=True)
    forged = frozenset.__new__(type(legitimate), legitimate)

    result = summarize_reflection_transaction(
        [forged], persistence_attempted=True,
    )

    assert result.status == "no_valid_entries"
    assert result.persistence_complete is False


def test_snapshot_object_new_is_rejected_by_immutable_builtin() -> None:
    legitimate = _advance_valid_entry(persisted=True)

    with pytest.raises(TypeError):
        object.__new__(type(legitimate))


def test_registered_snapshot_cannot_be_mutated_with_object_setattr() -> None:
    legitimate = _advance_valid_entry(persisted=True)

    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(legitimate, "stage", ReflectionStage.NOT_REQUESTED)


def test_legitimate_snapshot_supports_copy_deepcopy_and_pickle() -> None:
    legitimate = _advance_valid_entry(persisted=True)
    shallow = copy.copy(legitimate)
    deep = copy.deepcopy(legitimate)
    candidates = (
        shallow,
        deep,
        pickle.loads(pickle.dumps(legitimate)),
    )

    assert shallow is legitimate
    assert deep is legitimate
    for candidate in candidates:
        result = summarize_reflection_transaction(
            [candidate], persistence_attempted=True,
        )
        assert candidate.to_payload() == legitimate.to_payload()
        assert result.status == "complete"
        assert result.persistence_complete is True


def test_copy_and_pickle_cannot_register_forged_snapshot() -> None:
    legitimate = _advance_valid_entry(persisted=True)
    forged = type(legitimate)(tuple(legitimate))

    for candidate in (forged, copy.copy(forged), copy.deepcopy(forged)):
        result = summarize_reflection_transaction(
            [candidate], persistence_attempted=True,
        )
        assert result.status == "no_valid_entries"
        assert result.persistence_complete is False
    with pytest.raises(ValueError, match="provenance"):
        pickle.dumps(forged)


def test_object_new_cannot_call_instance_assign_to_mint_valid_provenance() -> None:
    forged = object.__new__(PlayerReflectionTransaction)

    with pytest.raises(AttributeError):
        forged._assign(  # type: ignore[attr-defined]
            player_id="p01",
            decision_id="reflection:g1:p01",
            stage=ReflectionStage.PERSISTED,
            failure_stage=None,
            failure_code=None,
            verified_claim_ids=("claim-p01",),
            verified_lesson_ids=("lesson-p01",),
            entry_id="reflection_g1_p01",
            stage_path=tuple(ReflectionStage),
        )


@pytest.mark.parametrize("duplicate_kind", ("object", "identity", "decision", "entry"))
def test_summarizer_rejects_duplicate_transaction_identities(
    duplicate_kind: str,
) -> None:
    first = _advance_valid_entry(persisted=True)
    if duplicate_kind == "object":
        second = first
    elif duplicate_kind == "identity":
        second = _advance_valid_entry(persisted=True)
    else:
        decision_id = (
            "reflection:g1:p01" if duplicate_kind == "decision"
            else "reflection:g1:p02"
        )
        second = PlayerReflectionTransaction("p02", decision_id)
        second = second.advance(ReflectionStage.GENERATED)
        second = second.advance(ReflectionStage.SCHEMA_VALIDATED)
        second = second.advance(
            ReflectionStage.FACTS_VERIFIED,
            verified_claim_ids=("claim-p02",),
        )
        second = second.advance(
            ReflectionStage.LESSONS_VERIFIED,
            verified_lesson_ids=("lesson-p02",),
        )
        second = second.advance(
            ReflectionStage.PERSISTED,
            entry_id=(
                "reflection_g1_p01" if duplicate_kind == "entry"
                else "reflection_g1_p02"
            ),
        )

    result = summarize_reflection_transaction(
        [first, second], persistence_attempted=True,
    )

    assert result.status == "persistence_failed"
    assert result.persistence_complete is False


def test_failed_player_requires_explicit_stage_and_code() -> None:
    entry = PlayerReflectionTransaction(
        player_id="p02", decision_id="reflection:g1:p02",
    ).advance(ReflectionStage.GENERATED)

    with pytest.raises(ValueError, match="failure_stage and failure_code"):
        entry.fail(failure_stage="schema_validated", failure_code="")

    failed = entry.fail(
        failure_stage="schema_validated",
        failure_code="invalid_structured_draft",
    )
    assert failed.failure_stage == "schema_validated"
    assert failed.failure_code == "invalid_structured_draft"


def test_reflection_transaction_game_statuses_are_non_vacuous() -> None:
    assert summarize_reflection_transaction(
        entries=[], transaction_run=False,
    ).status == "not_run"

    empty = summarize_reflection_transaction(entries=[])
    assert empty.status == "no_valid_entries"
    assert empty.persistence_complete is False

    valid = _advance_valid_entry()
    failed = PlayerReflectionTransaction(
        player_id="p02", decision_id="reflection:g1:p02",
    ).fail(
        failure_stage="generated", failure_code="reflection_not_generated",
    )
    pending = summarize_reflection_transaction(entries=[valid, failed])
    assert pending.status == "partial"
    assert pending.persistence_complete is False

    persisted = valid.advance(
        ReflectionStage.PERSISTED, entry_id="reflection_g1_p01",
    )
    partial = summarize_reflection_transaction(
        entries=[persisted, failed], persistence_attempted=True,
    )
    assert partial.status == "partial"
    assert partial.persistence_complete is True

    complete = summarize_reflection_transaction(
        entries=[persisted], persistence_attempted=True,
    )
    assert complete.status == "complete"
    assert complete.persistence_complete is True

    persistence_failed = summarize_reflection_transaction(
        entries=[valid], persistence_attempted=True,
    )
    assert persistence_failed.status == "persistence_failed"
    assert persistence_failed.persistence_complete is False


def test_zero_expected_entries_emits_no_valid_entries_and_never_succeeds() -> None:
    repo = InMemoryGameRepository()
    runner = GameRunner(GameRunnerConfig(
        seed=1200,
        repository=repo,
        memory_coordinator=PersistentMemoryCoordinator(repo),
    ))
    runner._state = GameState(
        game_id=runner.game_id,
        phase="finished",
        status="finished",
        winning_faction="good",
        players={"p01": PlayerState(id="p01", role="seer")},
        events=[GameEvent(type="reflection_complete", payload={
            "status": "no_valid_entries",
            "player_count": 1,
            "entries": [{
                "player_id": "p01",
                "decision_id": f"reflection:{runner.game_id}:p01",
                "transaction_state": "generated",
                "failure_stage": "schema_validated",
                "failure_code": "invalid_structured_draft",
                "verification": {
                    "status": "invalid_structured_draft",
                    "decision_id": f"reflection:{runner.game_id}:p01",
                    "verified_lessons": [],
                },
            }],
        })],
    )

    runner._save_memory_snapshot()

    audit = next(
        event for event in runner.state.events
        if event.type == "reflection_persistence_audit"
    )
    assert audit.payload["status"] == "no_valid_entries"
    assert audit.payload["expected_entry_count"] == 0
    assert audit.payload["persistence_complete"] is False
    assert any(
        event.type == "reflection_no_valid_entries"
        for event in runner.state.events
    )
    assert repo.load_reflections_by_game(runner.game_id) == []


def test_reflection_generation_exception_has_explicit_game_and_player_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_dispatch(*_args, **_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(summary, "_dispatch_agent", fail_dispatch)
    state = GameState(
        game_id="g1",
        phase="finished",
        status="finished",
        winning_faction="good",
        players={"p01": PlayerState(id="p01", role="seer")},
    )

    result = summary.reflection({"game_state": state, "engine": None})

    reflection_event, no_valid_event = result["game_state"].events[-2:]
    entry = reflection_event.payload["entries"][0]
    assert reflection_event.payload["status"] == "no_valid_entries"
    assert reflection_event.payload["persistence_complete"] is False
    assert entry["transaction_state"] == "not_requested"
    assert entry["failure_stage"] == "generated"
    assert entry["failure_code"] == "agent_error"
    assert entry["decision_id"] == "reflection:g1:p01"
    assert no_valid_event.type == "reflection_no_valid_entries"


def test_safe_verification_normalizes_impossible_agent_error_stage() -> None:
    safe = safe_reflection_verification(
        {
            "status": "agent_error",
            "failure_stage": "facts_verified",
            "failure_code": "provider_error",
        },
        decision_id="reflection:g1:p01",
    )

    assert safe["status"] == "agent_error"
    assert safe["decision_id"] == "reflection:g1:p01"
    assert safe["failure_stage"] == "generated"
    assert safe["failure_code"] == "invalid_reflection_failure_stage"


def test_safe_verification_normalizes_unhashable_status_without_crashing() -> None:
    safe = safe_reflection_verification(
        {"status": {"unexpected": "mapping"}},
        decision_id="reflection:g1:p01",
    )

    assert safe["status"] == "agent_error"
    assert safe["decision_id"] == "reflection:g1:p01"
    assert safe["failure_stage"] == "generated"
    assert safe["failure_code"] == "invalid_reflection_status"


def test_impossible_adapter_failure_metadata_records_failure_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        summary,
        "_dispatch_agent",
        lambda *_args, **_kwargs: {"reflection_verification": {
            "status": "agent_error",
            "failure_stage": "facts_verified",
            "failure_code": "provider_error",
        }},
    )
    state = GameState(
        game_id="g1", phase="finished", status="finished",
        winning_faction="good",
        players={"p01": PlayerState(id="p01", role="seer")},
    )

    result = summary.reflection({"game_state": state, "engine": None})

    complete = next(
        event for event in result["game_state"].events
        if event.type == "reflection_complete"
    )
    entry = complete.payload["entries"][0]
    assert complete.payload["status"] == "no_valid_entries"
    assert entry["failure_stage"] == "generated"
    assert entry["failure_code"] == "invalid_reflection_failure_stage"


def test_adapter_decision_mismatch_becomes_canonical_attributed_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        summary,
        "_dispatch_agent",
        lambda *_args, **_kwargs: {"reflection_verification": {
            "status": "verified",
            "decision_id": "reflection:stale:p99",
            "verified_claim_ids": ["claim-p01"],
            "verified_lessons": [{
                "lesson_id": "lesson-p01", "abstraction": "先核验公开票型",
            }],
        }},
    )
    state = GameState(
        game_id="g1", phase="finished", status="finished",
        winning_faction="good",
        players={"p01": PlayerState(id="p01", role="seer")},
    )

    result = summary.reflection({"game_state": state, "engine": None})

    complete = next(
        event for event in result["game_state"].events
        if event.type == "reflection_complete"
    )
    entry = complete.payload["entries"][0]
    assert entry["decision_id"] == "reflection:g1:p01"
    assert entry["verification"]["decision_id"] == "reflection:g1:p01"
    assert entry["failure_stage"] == "generated"
    assert entry["failure_code"] == "reflection_decision_id_mismatch"


def test_snapshot_preflight_failure_with_valid_lesson_is_persistence_failed() -> None:
    class SnapshotReadFailingRepository(InMemoryGameRepository):
        def load_memory_snapshot(self, snapshot_id: str):
            raise RuntimeError(f"snapshot unavailable: {snapshot_id}")

    repo = SnapshotReadFailingRepository()
    runner = GameRunner(GameRunnerConfig(
        seed=1201,
        repository=repo,
        memory_coordinator=PersistentMemoryCoordinator(repo),
    ))
    decision_id = f"reflection:{runner.game_id}:p01"
    runner._state = GameState(
        game_id=runner.game_id,
        phase="finished",
        status="finished",
        winning_faction="good",
        players={"p01": PlayerState(id="p01", role="seer")},
        events=[GameEvent(type="reflection_complete", payload={
            "status": "partial",
            "player_count": 1,
            "entries": [{
                "player_id": "p01",
                "decision_id": decision_id,
                "transaction_state": "lessons_verified",
                "failure_stage": None,
                "failure_code": None,
                "verification": {
                    "status": "verified",
                    "decision_id": decision_id,
                    "verified_claim_ids": ["claim-p01"],
                    "rejected_claim_ids": [],
                    "verified_lessons": [{
                        "lesson_id": "lesson-p01",
                        "abstraction": "先核验公开票型",
                    }],
                    "rejected_fact_count": 0,
                    "rejected_lesson_count": 0,
                },
            }],
        })],
    )

    runner._save_memory_snapshot()

    audit = next(
        event for event in runner.state.events
        if event.type == "reflection_persistence_audit"
    )
    assert audit.payload["status"] == "persistence_failed"
    assert audit.payload["persisted_entry_count"] == 0
    assert audit.payload["repository_read_complete"] is True
    assert audit.payload["snapshot_read_complete"] is False
    assert audit.payload["persistence_complete"] is False


@pytest.mark.parametrize(
    "mismatched_field",
    ("entry", "verification", "earlier_event"),
)
def test_persistence_rejects_stale_decision_before_memory_or_repository_write(
    mismatched_field: str,
) -> None:
    class WriteSpyRepository(InMemoryGameRepository):
        def __init__(self) -> None:
            super().__init__()
            self.reflection_memory_writes: list[tuple[str, str]] = []

        def save_reflection(self, entry: dict) -> None:
            self.reflection_memory_writes.append(("save_reflection", entry.get("entry_id", "")))
            super().save_reflection(entry)

        def delete_reflection(self, entry_id: str) -> None:
            self.reflection_memory_writes.append(("delete_reflection", entry_id))
            super().delete_reflection(entry_id)

        def save_memory_snapshot(self, snapshot_id: str, data: dict) -> None:
            self.reflection_memory_writes.append(("save_memory_snapshot", snapshot_id))
            super().save_memory_snapshot(snapshot_id, data)

        def delete_memory_snapshot(self, snapshot_id: str) -> None:
            self.reflection_memory_writes.append(("delete_memory_snapshot", snapshot_id))
            super().delete_memory_snapshot(snapshot_id)

    repo = WriteSpyRepository()
    runner = GameRunner(GameRunnerConfig(
        seed=1202,
        repository=repo,
        memory_coordinator=PersistentMemoryCoordinator(repo),
    ))
    canonical_decision_id = f"reflection:{runner.game_id}:p01"
    entry_decision_id = canonical_decision_id
    verification_decision_id = canonical_decision_id
    if mismatched_field == "entry":
        entry_decision_id = "reflection:stale:p01"
    elif mismatched_field == "verification":
        verification_decision_id = "reflection:stale:p01"
    runner._state = GameState(
        game_id=runner.game_id,
        phase="finished",
        status="finished",
        winning_faction="good",
        players={"p01": PlayerState(id="p01", role="seer")},
        events=[GameEvent(type="reflection_complete", payload={
            "status": "complete",
            "player_count": 1,
            "entries": [{
                "player_id": "p01",
                "decision_id": entry_decision_id,
                "transaction_state": "lessons_verified",
                "failure_stage": None,
                "failure_code": None,
                "verification": {
                    "status": "verified",
                    "decision_id": verification_decision_id,
                    "verified_claim_ids": ["claim-p01"],
                    "rejected_claim_ids": [],
                    "verified_lessons": [{
                        "lesson_id": "lesson-p01",
                        "abstraction": "先核验公开票型",
                    }],
                    "rejected_fact_count": 0,
                    "rejected_lesson_count": 0,
                },
            }],
        })],
    )
    if mismatched_field == "earlier_event":
        stale_event = GameEvent(type="reflection_complete", payload={
            "entries": [{
                "player_id": "p01",
                "decision_id": "reflection:stale:p01",
                "verification": {
                    "status": "verified",
                    "decision_id": "reflection:stale:p01",
                },
            }],
        })
        runner._state = replace(
            runner._state,
            events=[stale_event, *runner._state.events],
        )
    mem_store = runner._cognition_state_manager.memory_store
    relation_graph_before = mem_store.relation_graph
    repo.reflection_memory_writes.clear()

    runner._save_memory_snapshot()

    audit = next(
        event for event in runner.state.events
        if event.type == "reflection_persistence_audit"
    )
    assert audit.payload["status"] == "persistence_failed"
    assert audit.payload["persisted_entry_count"] == 0
    assert audit.payload["repository_read_complete"] is False
    assert audit.payload["snapshot_read_complete"] is False
    assert audit.payload["persistence_complete"] is False
    assert audit.payload["entries"] == [{
        "player_id": "p01",
        "decision_id": canonical_decision_id,
        "failure_stage": "persisted",
        "failure_code": "reflection_decision_id_mismatch",
        "persistence_complete": False,
    }]
    assert repo.reflection_memory_writes == []
    assert repo.load_reflections_by_game(runner.game_id) == []
    assert repo.load_memory_snapshot(runner.game_id) is None
    assert repo.load_memory_snapshot("latest") is None
    assert mem_store.relation_graph is relation_graph_before
    assert mem_store.cognition_matrices == {}
    assert mem_store.reflections.all_v2_entries() == []


def _reflection_game(
    *,
    status: str,
    entries: list[dict],
    persistence_status: str,
    persistence_entries: list[dict],
) -> dict:
    return {
        "game_id": "g1",
        "status": "finished",
        "winning_faction": "good",
        "players": {
            "p01": {"role": "seer"},
            "p02": {"role": "werewolf"},
        },
        "events": [
            {"type": "reflection_complete", "payload": {
                "status": status,
                "player_count": 2,
                "valid_entry_count": sum(
                    1 for entry in entries
                    if entry.get("transaction_state") in {
                        "lessons_verified", "persisted",
                    }
                ),
                "failure_count": sum(
                    1 for entry in entries if entry.get("failure_code")
                ),
                "entries": entries,
            }},
            {"type": "reflection_persistence_audit", "payload": {
                "status": persistence_status,
                "expected_entry_count": len(persistence_entries),
                "persistence_complete": persistence_status in {"complete", "partial"},
                "rollback_complete": True,
                "entries": persistence_entries,
            }},
        ],
    }


def _verified_event_entry(player_id: str = "p01") -> dict:
    return {
        "player_id": player_id,
        "decision_id": f"reflection:g1:{player_id}",
        "transaction_state": "persisted",
        "entry_id": f"reflection_g1_{player_id}",
        "verification": {
            "status": "verified",
            "decision_id": f"reflection:g1:{player_id}",
            "verified_claim_ids": [f"claim-{player_id}"],
            "rejected_claim_ids": [],
            "verified_lessons": [{
                "lesson_id": f"lesson-{player_id}",
                "abstraction": "先核验公开票型",
            }],
            "rejected_fact_count": 0,
            "rejected_lesson_count": 0,
        },
    }


def _persisted_audit_entry(player_id: str = "p01") -> dict:
    return {
        "player_id": player_id,
        "decision_id": f"reflection:g1:{player_id}",
        "verified_claim_ids": [f"claim-{player_id}"],
        "entry_id": f"reflection_g1_{player_id}",
        "row_found": True,
        "persistence_complete": True,
        "persisted_rejected_fact_count": 0,
    }


def test_acceptance_allows_attributed_partial_transaction() -> None:
    failed = {
        "player_id": "p02",
        "decision_id": "reflection:g1:p02",
        "transaction_state": "generated",
        "failure_stage": "schema_validated",
        "failure_code": "invalid_structured_draft",
        "verification": {
            "status": "invalid_structured_draft",
            "decision_id": "reflection:g1:p02",
            "verified_claim_ids": [],
            "rejected_claim_ids": [],
            "verified_lessons": [],
            "rejected_fact_count": 0,
            "rejected_lesson_count": 0,
        },
    }
    metrics = compute_reflection_acceptance_metrics([_reflection_game(
        status="partial",
        entries=[_verified_event_entry(), failed],
        persistence_status="partial",
        persistence_entries=[_persisted_audit_entry()],
    )])

    assert metrics["reflection_audited_game_count"] == 1
    assert metrics["reflection_contamination_metrics_supported"] is True
    assert metrics["reflection_persisted_rejected_fact_count"] == 0


def test_acceptance_allows_exact_complete_transaction_summary() -> None:
    second_event = _verified_event_entry("p02")
    second_audit = _persisted_audit_entry("p02")
    metrics = compute_reflection_acceptance_metrics([_reflection_game(
        status="complete",
        entries=[_verified_event_entry(), second_event],
        persistence_status="complete",
        persistence_entries=[_persisted_audit_entry(), second_audit],
    )])

    assert metrics["reflection_audited_game_count"] == 1
    assert metrics["reflection_contamination_metrics_supported"] is True


@pytest.mark.parametrize(
    "reflection_status",
    (None, "not_run", "no_valid_entries", "persistence_failed"),
)
def test_acceptance_rejects_unsuccessful_or_missing_game_summary_status(
    reflection_status: str | None,
) -> None:
    game = _reflection_game(
        status="complete",
        entries=[_verified_event_entry(), _verified_event_entry("p02")],
        persistence_status="complete",
        persistence_entries=[
            _persisted_audit_entry(), _persisted_audit_entry("p02"),
        ],
    )
    payload = game["events"][0]["payload"]
    if reflection_status is None:
        del payload["status"]
    else:
        payload["status"] = reflection_status

    metrics = compute_reflection_acceptance_metrics([game])

    assert metrics["reflection_audited_game_count"] == 0
    assert metrics["reflection_contamination_metrics_supported"] is False


def test_acceptance_rejects_conflicting_summary_status_and_counts() -> None:
    failed = {
        "player_id": "p02",
        "decision_id": "reflection:g1:p02",
        "transaction_state": "generated",
        "failure_stage": "schema_validated",
        "failure_code": "invalid_structured_draft",
        "verification": {
            "status": "invalid_structured_draft",
            "decision_id": "reflection:g1:p02",
            "verified_claim_ids": [],
            "verified_lessons": [],
        },
    }
    game = _reflection_game(
        status="complete",
        entries=[_verified_event_entry(), failed],
        persistence_status="partial",
        persistence_entries=[_persisted_audit_entry()],
    )
    game["events"][0]["payload"]["valid_entry_count"] = 2

    metrics = compute_reflection_acceptance_metrics([game])

    assert metrics["reflection_audited_game_count"] == 0
    assert metrics["reflection_contamination_metrics_supported"] is False


def test_acceptance_rejects_boolean_summary_counts() -> None:
    failed = {
        "player_id": "p02",
        "decision_id": "reflection:g1:p02",
        "transaction_state": "generated",
        "failure_stage": "schema_validated",
        "failure_code": "invalid_structured_draft",
        "verification": {
            "status": "invalid_structured_draft",
            "decision_id": "reflection:g1:p02",
            "verified_claim_ids": [],
            "verified_lessons": [],
        },
    }
    game = _reflection_game(
        status="partial",
        entries=[_verified_event_entry(), failed],
        persistence_status="partial",
        persistence_entries=[_persisted_audit_entry()],
    )
    game["events"][0]["payload"]["valid_entry_count"] = True

    metrics = compute_reflection_acceptance_metrics([game])

    assert metrics["reflection_audited_game_count"] == 0
    assert metrics["reflection_contamination_metrics_supported"] is False


@pytest.mark.parametrize("missing_count", ("valid_entry_count", "failure_count"))
def test_acceptance_requires_game_summary_transaction_counts(
    missing_count: str,
) -> None:
    game = _reflection_game(
        status="complete",
        entries=[_verified_event_entry(), _verified_event_entry("p02")],
        persistence_status="complete",
        persistence_entries=[
            _persisted_audit_entry(), _persisted_audit_entry("p02"),
        ],
    )
    del game["events"][0]["payload"][missing_count]

    metrics = compute_reflection_acceptance_metrics([game])

    assert metrics["reflection_audited_game_count"] == 0
    assert metrics["reflection_contamination_metrics_supported"] is False


def test_acceptance_rejects_failed_player_decision_mismatch() -> None:
    failed = {
        "player_id": "p02",
        "decision_id": "reflection:g1:p02",
        "transaction_state": "generated",
        "failure_stage": "schema_validated",
        "failure_code": "invalid_structured_draft",
        "verification": {
            "status": "invalid_structured_draft",
            "decision_id": "reflection:g1:other",
            "verified_claim_ids": [],
            "verified_lessons": [],
        },
    }
    metrics = compute_reflection_acceptance_metrics([_reflection_game(
        status="partial",
        entries=[_verified_event_entry(), failed],
        persistence_status="partial",
        persistence_entries=[_persisted_audit_entry()],
    )])

    assert metrics["reflection_audited_game_count"] == 0
    assert metrics["reflection_contamination_metrics_supported"] is False


@pytest.mark.parametrize(
    "wrong_decision_id",
    ("reflection:stale:p01", "reflection:g1:p02"),
)
def test_acceptance_binds_success_decision_to_current_game_and_player(
    wrong_decision_id: str,
) -> None:
    event_entry = _verified_event_entry()
    event_entry["decision_id"] = wrong_decision_id
    event_entry["verification"]["decision_id"] = wrong_decision_id
    audit_entry = _persisted_audit_entry()
    audit_entry["decision_id"] = wrong_decision_id
    metrics = compute_reflection_acceptance_metrics([_reflection_game(
        status="complete",
        entries=[event_entry, _verified_event_entry("p02")],
        persistence_status="complete",
        persistence_entries=[audit_entry, _persisted_audit_entry("p02")],
    )])

    assert metrics["reflection_audited_game_count"] == 0
    assert metrics["reflection_contamination_metrics_supported"] is False


def test_acceptance_binds_failed_decision_to_current_game_and_player() -> None:
    failed = {
        "player_id": "p02",
        "decision_id": "reflection:stale:p02",
        "transaction_state": "generated",
        "failure_stage": "schema_validated",
        "failure_code": "invalid_structured_draft",
        "verification": {
            "status": "invalid_structured_draft",
            "decision_id": "reflection:stale:p02",
            "verified_claim_ids": [],
            "verified_lessons": [],
        },
    }
    metrics = compute_reflection_acceptance_metrics([_reflection_game(
        status="partial",
        entries=[_verified_event_entry(), failed],
        persistence_status="partial",
        persistence_entries=[_persisted_audit_entry()],
    )])

    assert metrics["reflection_audited_game_count"] == 0
    assert metrics["reflection_contamination_metrics_supported"] is False


def test_acceptance_rejects_conflicting_persisted_event_entry_id() -> None:
    first = _verified_event_entry()
    first["entry_id"] = "reflection_g1_other"
    metrics = compute_reflection_acceptance_metrics([_reflection_game(
        status="complete",
        entries=[first, _verified_event_entry("p02")],
        persistence_status="complete",
        persistence_entries=[
            _persisted_audit_entry(), _persisted_audit_entry("p02"),
        ],
    )])

    assert metrics["reflection_audited_game_count"] == 0
    assert metrics["reflection_contamination_metrics_supported"] is False


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("player_count", True),
        ("player_count", False),
        ("player_count", 2.0),
        ("player_count", 0.0),
        ("player_count", "2"),
        ("player_count", -1),
        ("expected_entry_count", True),
        ("expected_entry_count", False),
        ("expected_entry_count", 2.0),
        ("expected_entry_count", 0.0),
        ("expected_entry_count", "2"),
        ("expected_entry_count", -1),
        ("persisted_rejected_fact_count", True),
        ("persisted_rejected_fact_count", False),
        ("persisted_rejected_fact_count", 1.0),
        ("persisted_rejected_fact_count", 0.0),
        ("persisted_rejected_fact_count", "0"),
        ("persisted_rejected_fact_count", -1),
    ),
)
def test_acceptance_rejects_non_integer_transaction_counts(
    field: str,
    invalid_value: object,
) -> None:
    game = _reflection_game(
        status="complete",
        entries=[_verified_event_entry(), _verified_event_entry("p02")],
        persistence_status="complete",
        persistence_entries=[
            _persisted_audit_entry(), _persisted_audit_entry("p02"),
        ],
    )
    if field == "player_count":
        game["events"][0]["payload"][field] = invalid_value
    elif field == "expected_entry_count":
        game["events"][1]["payload"][field] = invalid_value
    else:
        game["events"][1]["payload"]["entries"][0][field] = invalid_value

    metrics = compute_reflection_acceptance_metrics([game])

    assert metrics["reflection_audited_game_count"] == 0
    assert metrics["reflection_contamination_metrics_supported"] is False


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("decision_id", "reflection:g1:other"),
        ("verified_claim_ids", ["claim-other"]),
        ("entry_id", "reflection_g1_other"),
    ),
)
def test_acceptance_rejects_broken_decision_claim_entry_chain(
    field: str,
    wrong_value: object,
) -> None:
    audit_entry = _persisted_audit_entry()
    audit_entry[field] = wrong_value
    failed = {
        "player_id": "p02",
        "decision_id": "reflection:g1:p02",
        "transaction_state": "not_requested",
        "failure_stage": "generated",
        "failure_code": "reflection_not_generated",
        "verification": {
            "status": "not_generated",
            "decision_id": "reflection:g1:p02",
            "verified_lessons": [],
        },
    }
    metrics = compute_reflection_acceptance_metrics([_reflection_game(
        status="partial",
        entries=[_verified_event_entry(), failed],
        persistence_status="partial",
        persistence_entries=[audit_entry],
    )])

    assert metrics["reflection_audited_game_count"] == 0
    assert metrics["reflection_contamination_metrics_supported"] is False


@pytest.mark.parametrize("missing_field", ("event_decision_id", "audit_claim_ids"))
def test_acceptance_rejects_missing_decision_or_claim_chain_field(
    missing_field: str,
) -> None:
    event_entry = _verified_event_entry()
    audit_entry = _persisted_audit_entry()
    if missing_field == "event_decision_id":
        del event_entry["decision_id"]
    else:
        del audit_entry["verified_claim_ids"]
    failed = {
        "player_id": "p02",
        "decision_id": "reflection:g1:p02",
        "transaction_state": "not_requested",
        "failure_stage": "generated",
        "failure_code": "reflection_not_generated",
        "verification": {
            "status": "not_generated",
            "decision_id": "reflection:g1:p02",
            "verified_lessons": [],
        },
    }
    metrics = compute_reflection_acceptance_metrics([_reflection_game(
        status="partial",
        entries=[event_entry, failed],
        persistence_status="partial",
        persistence_entries=[audit_entry],
    )])

    assert metrics["reflection_audited_game_count"] == 0
    assert metrics["reflection_contamination_metrics_supported"] is False


@pytest.mark.parametrize("status", ("no_valid_entries", "persistence_failed"))
def test_acceptance_never_supports_unsuccessful_transaction_status(status: str) -> None:
    failed = {
        "player_id": "p01",
        "decision_id": "reflection:g1:p01",
        "transaction_state": "generated",
        "failure_stage": "lessons_verified",
        "failure_code": "reflection_no_verified_lessons",
        "verification": {
            "status": "verified",
            "decision_id": "reflection:g1:p01",
            "verified_lessons": [],
        },
    }
    second = {
        **failed,
        "player_id": "p02",
        "decision_id": "reflection:g1:p02",
        "verification": {
            **failed["verification"],
            "decision_id": "reflection:g1:p02",
        },
    }
    metrics = compute_reflection_acceptance_metrics([_reflection_game(
        status=status,
        entries=[failed, second],
        persistence_status=status,
        persistence_entries=[],
    )])

    assert metrics["reflection_audited_game_count"] == 0
    assert metrics["reflection_contamination_metrics_supported"] is False
    assert metrics["reflection_persisted_rejected_fact_count"] is None
