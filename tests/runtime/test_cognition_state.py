from __future__ import annotations

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.memory.store import MemoryStore
from werewolf_agent.runtime.cognition_state import CognitionStateManager


def _game_with_events(events: list[GameEvent] | None = None) -> GameState:
    return GameState(
        game_id="cognition_state_test",
        phase="day",
        day_number=1,
        night_number=1,
        players={
            "p01": PlayerState(id="p01", role="seer", alive=True),
            "p02": PlayerState(id="p02", role="villager", alive=True),
            "p03": PlayerState(id="p03", role="werewolf", alive=True),
            "p04": PlayerState(id="p04", role="witch", alive=True),
        },
        events=events or [],
    )


def test_initializes_one_matrix_per_player() -> None:
    store = MemoryStore()
    manager = CognitionStateManager(store)

    manager.initialize(_game_with_events())

    assert sorted(store.cognition_matrices) == ["p01", "p02", "p03", "p04"]
    for viewer_id, matrix in store.cognition_matrices.items():
        assert viewer_id not in matrix.player_ids()
        assert sorted(matrix.player_ids()) == sorted(
            pid for pid in ("p01", "p02", "p03", "p04") if pid != viewer_id
        )
        first_entry = matrix.all_entries()[0]
        assert set(first_entry.role_probabilities) == {
            "seer",
            "villager",
            "werewolf",
            "witch",
        }

    assert manager.processed_event_count() == 0


def test_updates_are_incremental_and_idempotent() -> None:
    store = MemoryStore()
    manager = CognitionStateManager(store)
    gs = _game_with_events([
        GameEvent(
            type="speech",
            payload={
                "speaker": "p01",
                "text": "我是预言家，查验 p03 是狼人",
                "day_number": 1,
            },
        )
    ])
    manager.initialize(gs)

    first_records = manager.update_from_events(gs)
    first_evidence_count = sum(
        len(entry.key_evidence)
        for matrix in store.cognition_matrices.values()
        for entry in matrix.all_entries()
    )
    second_records = manager.update_from_events(gs)
    second_evidence_count = sum(
        len(entry.key_evidence)
        for matrix in store.cognition_matrices.values()
        for entry in matrix.all_entries()
    )

    assert first_records
    assert second_records == []
    assert first_evidence_count > 0
    assert second_evidence_count == first_evidence_count
    assert manager.processed_event_count() == 1


def test_updates_respect_viewer_visibility() -> None:
    store = MemoryStore()
    manager = CognitionStateManager(store)
    gs = _game_with_events([
        GameEvent(
            type="seer_check",
            payload={"target_id": "p03", "alignment": "werewolf", "night_number": 1},
        ),
        GameEvent(
            type="wolf_kill_selected",
            payload={"target_id": "p02", "night_number": 1},
        ),
    ])
    manager.initialize(gs)

    manager.update_from_events(gs)

    villager_matrix = store.get_matrix("p02")
    seer_matrix = store.get_matrix("p01")
    wolf_matrix = store.get_matrix("p03")
    assert villager_matrix is not None
    assert seer_matrix is not None
    assert wolf_matrix is not None

    villager_evidence = str([
        evidence.to_dict() if hasattr(evidence, "to_dict") else evidence
        for entry in villager_matrix.all_entries()
        for evidence in entry.key_evidence
    ])
    seer_evidence = str([
        evidence.to_dict() if hasattr(evidence, "to_dict") else evidence
        for entry in seer_matrix.all_entries()
        for evidence in entry.key_evidence
    ])
    wolf_evidence = str([
        evidence.to_dict() if hasattr(evidence, "to_dict") else evidence
        for entry in wolf_matrix.all_entries()
        for evidence in entry.key_evidence
    ])

    assert "seer_check" not in villager_evidence
    assert "wolf_kill_selected" not in villager_evidence
    assert "witch_kill_target" not in villager_evidence
    assert "seer_check" in seer_evidence
    assert "wolf_kill_selected" in wolf_evidence


def test_prompt_belief_summary_uses_live_matrix() -> None:
    store = MemoryStore()
    manager = CognitionStateManager(store)
    gs = _game_with_events([
        GameEvent(
            type="speech",
            payload={
                "speaker": "p01",
                "text": "我是预言家，查验 p03 是狼人",
                "day_number": 1,
            },
        )
    ])
    manager.initialize(gs)
    manager.update_from_events(gs)

    summary = manager.prompt_belief_summary("p02", gs)

    assert summary["my_suspects"]
    assert summary["my_suspects"][0]["player"] == "p03"
    assert summary["my_suspects"][0]["top_role_guess"] == "werewolf"


def test_incremental_remembers_prior_seer_claim_for_later_vote() -> None:
    """acceptance #7: 增量更新记住先前 seer 线，后来 vote 用先前 claim 判断。"""
    store = MemoryStore()
    manager = CognitionStateManager(store)
    speech = GameEvent(
        type="speech",
        payload={"speaker": "p01", "text": "我是预言家 查验p03是狼人", "day_number": 1},
    )
    gs1 = _game_with_events([speech])
    manager.initialize(gs1)
    manager.update_from_events(gs1)
    gs2 = _game_with_events([
        speech,
        GameEvent(type="vote", payload={"voter": "p01", "target": "p03", "day_number": 1}),
    ])
    manager.update_from_events(gs2)
    eng = manager._credibility_engines["p02"]
    cred = eng.score_for("p01")
    assert "vote_follows_black" in cred.evidence


def test_incremental_vote_uses_prior_public_evidence_for_trust() -> None:
    """Later incremental votes should use public anchors from previous batches."""
    store = MemoryStore()
    manager = CognitionStateManager(store)
    speech = GameEvent(
        type="speech",
        payload={"speaker": "p01", "text": "我是预言家 查验p03是狼人", "day_number": 1},
    )
    gs1 = _game_with_events([speech])
    manager.initialize(gs1)
    manager.update_from_events(gs1)

    matrix = store.get_matrix("p02")
    assert matrix is not None
    before = matrix.get("p04").trust

    gs2 = _game_with_events([
        speech,
        GameEvent(type="vote", payload={"voter": "p04", "target": "p03", "day_number": 1}),
    ])
    manager.update_from_events(gs2)

    after = matrix.get("p04").trust
    assert after > before
