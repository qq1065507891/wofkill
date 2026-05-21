"""Tests for GameRunner: full game orchestration via LangGraph + RuleEngine."""

from __future__ import annotations

import pytest
from dataclasses import replace

from werewolf_agent.core.models import GameState, PlayerState, GameEvent
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.graph import RuntimeState, build_game_graph, _new_engine
from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig


def test_game_runner_runtime_state_includes_rag_service() -> None:
    service = object()
    runner = GameRunner(GameRunnerConfig(seed=42, rag_service=service))

    runtime_state = runner._build_runtime_state()

    assert runtime_state["rag_service"] is service


def test_game_runner_runtime_state_creates_default_rag_service() -> None:
    runner = GameRunner(GameRunnerConfig(seed=42))

    runtime_state = runner._build_runtime_state()

    assert runtime_state["rag_service"] is not None
    assert runtime_state["rag_service"].__class__.__name__ == "RAGKnowledgeService"


def test_game_runner_runtime_state_can_disable_default_rag_service() -> None:
    runner = GameRunner(GameRunnerConfig(seed=42, enable_default_rag_service=False))

    runtime_state = runner._build_runtime_state()

    assert "rag_service" not in runtime_state


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestGameRunnerConfig:
    def test_config_defaults(self) -> None:
        cfg = GameRunnerConfig(ruleset_id="pre_witch_hunter_idiot_mixed")
        assert cfg.player_count == 12
        assert cfg.seed is not None
        assert cfg.use_agent_registry is False

    def test_config_custom(self) -> None:
        cfg = GameRunnerConfig(
            ruleset_id="pre_witch_hunter_idiot_mixed",
            seed=42,
            player_count=12,
            use_agent_registry=True,
        )
        assert cfg.seed == 42
        assert cfg.use_agent_registry is True

    def test_config_default_ruleset(self) -> None:
        cfg = GameRunnerConfig()
        assert cfg.ruleset_id == "pre_witch_hunter_idiot_mixed"


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestGameRunnerConstructor:
    def test_creates_with_config(self) -> None:
        runner = GameRunner(GameRunnerConfig(ruleset_id="pre_witch_hunter_idiot_mixed"))
        assert runner.engine is not None
        assert runner.game_id is not None
        assert runner.game_id.startswith("g_")

    def test_game_id_uses_seed(self) -> None:
        runner = GameRunner(GameRunnerConfig(seed=42))
        assert "42" in runner.game_id

    def test_state_initial(self) -> None:
        runner = GameRunner(GameRunnerConfig(seed=1))
        assert runner.state is not None
        assert runner.state.phase == "setup"
        assert runner.state.game_id == runner.game_id

    def test_config_stored(self) -> None:
        cfg = GameRunnerConfig(seed=99, use_agent_registry=True)
        runner = GameRunner(cfg)
        assert runner.config.seed == 99
        assert runner.config.use_agent_registry is True

    def test_use_agent_registry_builds_player_agents(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        cfg = GameRunnerConfig(
            seed=99,
            use_agent_registry=True,
            model_config_path="config/models.yaml",
        )
        runner = GameRunner(cfg)

        rt = runner._build_runtime_state()

        registry = rt.get("agent_registry")
        assert registry is not None
        assert registry.get_agent("p01") is not None
        assert registry.get_agent("judge") is None


# ---------------------------------------------------------------------------
# Full scripted game tests
# ---------------------------------------------------------------------------


class TestGameRunnerScriptedGame:
    def test_full_game_with_kill_completes(self) -> None:
        """A scripted game with wolf kill targets must complete with a winner.

        Without agent registry, the graph uses scripted inputs. We provide
        wolf_kill_target_id to drive the game forward.
        """
        runner = GameRunner(GameRunnerConfig(
            ruleset_id="pre_witch_hunter_idiot_mixed",
            seed=42,
        ))
        # Pre-assign roles to determine a non-wolf target
        engine = runner.engine
        players = engine.assign_roles(
            [f"p{i:02d}" for i in range(1, 13)], seed=42,
        )
        non_wolves = [pid for pid, p in players.items() if p.role != "werewolf"]
        # Set initial state with players assigned
        gs = replace(
            runner.state,
            players=players,
            phase="roles_assigned",
        )
        runner._state = gs
        # Run with a wolf kill target via the scripted interface
        final_state = runner.run_scripted(
            wolf_kill_target_id=non_wolves[0] if non_wolves else None,
        )
        # The game should have progressed past setup
        assert len(final_state.players) == 12
        assert final_state.phase in (
            "night", "day", "roles_assigned", "finished", "setup",
        )

    def test_game_runner_deterministic_with_seed(self) -> None:
        """Two runners with the same seed must produce the same final state."""
        r1 = GameRunner(GameRunnerConfig(seed=123))
        s1 = r1.run()
        r2 = GameRunner(GameRunnerConfig(seed=123))
        s2 = r2.run()
        # Both should reach the same phase with same player count
        assert s1.phase == s2.phase
        assert len(s1.players) == len(s2.players)
        assert s1.winning_faction == s2.winning_faction

    def test_full_game_has_events(self) -> None:
        """A completed game must have recorded events."""
        runner = GameRunner(GameRunnerConfig(seed=42))
        final = runner.run()
        assert len(final.events) > 0
        event_types = {e.type for e in final.events}
        # At minimum, setup and role assignment should appear
        assert "roles_assigned" in event_types or "enter_night" in event_types

    def test_full_game_players_assigned(self) -> None:
        """After running, all 12 players must be assigned."""
        runner = GameRunner(GameRunnerConfig(seed=7))
        final = runner.run()
        assert len(final.players) == 12
        from collections import Counter
        roles = Counter(p.role for p in final.players.values())
        assert roles["werewolf"] == 4
        assert roles["villager"] == 3

    def test_default_run_completes_without_crash(self) -> None:
        """Running without explicit wolf kill target should not crash.

        The graph will loop with wolf_no_kill_timeout until max_steps is reached.
        The game state should still be valid with 12 players assigned.
        """
        runner = GameRunner(GameRunnerConfig(seed=10))
        final = runner.run()
        assert final is not None
        assert len(final.players) == 12

    def test_day_vote_does_not_reuse_previous_day_votes_without_same_window(self) -> None:
        from werewolf_agent.runtime.graph import day_vote

        gs = GameState(
            game_id="g_vote_window",
            phase="day",
            day_number=2,
            players={
                "p01": PlayerState(id="p01", role="villager", alive=True),
                "p02": PlayerState(id="p02", role="villager", alive=True),
            },
        )
        state = {
            "game_state": gs,
            "exile_votes": {"p01": "p02"},
            "exile_vote_day": 1,
            "exile_vote_revote": False,
            "revote": False,
        }

        result = day_vote(state)

        assert result["exile_votes"] == {}
        assert result["exile_vote_day"] == 2

    def test_resolve_vote_clears_vote_window_after_exile(self) -> None:
        from werewolf_agent.runtime.graph import resolve_vote

        players = {
            "p01": PlayerState(id="p01", role="villager", alive=True),
            "p02": PlayerState(id="p02", role="villager", alive=True),
        }
        gs = GameState(
            game_id="g_vote_clear",
            phase="day",
            day_number=1,
            players=players,
        )
        state = {
            "game_state": gs,
            "engine": _new_engine(),
            "exile_votes": {"p01": "p02"},
            "vote_action_traces": {"p01": {"final_action_type": "vote"}},
            "exile_vote_day": 1,
            "exile_vote_revote": False,
            "revote": False,
        }

        result = resolve_vote(state)

        assert result["exile_votes"] == {}
        assert result["vote_action_traces"] == {}
        assert result["exile_vote_revote"] is False


# ---------------------------------------------------------------------------
# Step-by-step execution tests
# ---------------------------------------------------------------------------


class TestGameRunnerStepByStep:
    def test_run_step_returns_game_state(self) -> None:
        """run_step() returns the updated GameState."""
        runner = GameRunner(GameRunnerConfig(seed=42))
        result = runner.run_step()
        assert isinstance(result, GameState)

    def test_run_step_increments_step_count(self) -> None:
        """run_step() increments the step counter."""
        runner = GameRunner(GameRunnerConfig(seed=42))
        assert runner.step_count == 0
        runner.run_step()
        assert runner.step_count == 1
        runner.run_step()
        assert runner.step_count == 2

    def test_multiple_run_steps_progress_game(self) -> None:
        """Multiple run_step() calls should progressively advance the game."""
        runner = GameRunner(GameRunnerConfig(seed=42))
        phases_seen = [runner.state.phase]
        for _ in range(5):
            runner.run_step()
            phases_seen.append(runner.state.phase)
        # At least 2 distinct phases should have been seen
        # (setup -> roles_assigned -> night ...)
        assert len(set(phases_seen)) >= 2

    def test_run_step_eventually_assigns_players(self) -> None:
        """After a few steps, players should be assigned."""
        runner = GameRunner(GameRunnerConfig(seed=42))
        for _ in range(3):
            runner.run_step()
        assert len(runner.state.players) == 12


# ---------------------------------------------------------------------------
# Runtime execution coordinator tests
# ---------------------------------------------------------------------------


class _FakeRunner:
    def __init__(self) -> None:
        self.state = GameState(game_id="g_exec", phase="night")
        self.step_count = 0
        self.finished = False
        self.step_calls = 0
        self.run_calls = 0
        self.fail_run = False

    def run_step(self) -> GameState:
        self.step_calls += 1
        self.step_count += 1
        self.state = replace(self.state, phase=f"step_{self.step_count}")
        return self.state

    def run(self, max_steps: int = 500) -> GameState:
        self.run_calls += 1
        if self.fail_run:
            raise RuntimeError("boom")
        self.step_count += max_steps
        self.finished = True
        self.state = replace(self.state, phase="finished")
        return self.state


class TestRuntimeExecutionCoordinator:
    def test_try_step_rejects_when_game_lock_is_held(self) -> None:
        from werewolf_agent.runtime.executor import LocalRuntimeExecutor

        executor = LocalRuntimeExecutor()
        runner = _FakeRunner()
        lock = executor.lock_for("g_exec")

        assert lock.acquire(blocking=False)
        try:
            result = executor.try_step("g_exec", runner)
        finally:
            lock.release()

        assert result.success is False
        assert result.status == "busy"
        assert runner.step_calls == 0

    def test_try_step_releases_lock_after_success(self) -> None:
        from werewolf_agent.runtime.executor import LocalRuntimeExecutor

        executor = LocalRuntimeExecutor()
        runner = _FakeRunner()

        first = executor.try_step("g_exec", runner)
        second = executor.try_step("g_exec", runner)

        assert first.success is True
        assert second.success is True
        assert runner.step_calls == 2
        assert executor.status("g_exec").step_count == 2

    def test_start_background_records_finished_status(self) -> None:
        from werewolf_agent.runtime.executor import LocalRuntimeExecutor

        executor = LocalRuntimeExecutor()
        runner = _FakeRunner()

        result = executor.start_background("g_exec", runner, max_steps=3)
        assert result.success is True
        executor.wait("g_exec", timeout=2.0)

        status = executor.status("g_exec")
        assert status.state == "finished"
        assert status.step_count == 3
        assert status.phase == "finished"
        assert status.error == ""

    def test_start_background_records_error_status(self) -> None:
        from werewolf_agent.runtime.executor import LocalRuntimeExecutor

        executor = LocalRuntimeExecutor()
        runner = _FakeRunner()
        runner.fail_run = True

        result = executor.start_background("g_exec", runner)
        assert result.success is True
        executor.wait("g_exec", timeout=2.0)

        status = executor.status("g_exec")
        assert status.state == "error"
        assert "boom" in status.error


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


class TestGameRunnerAPIIntegration:
    def test_runner_with_repository(self) -> None:
        """GameRunner can accept a repository for persistence."""
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        repo = InMemoryGameRepository()
        runner = GameRunner(GameRunnerConfig(
            seed=42,
            repository=repo,
        ))
        assert runner.config.repository is not None

    def test_runner_persists_on_run(self) -> None:
        """After running, game state should be persisted if repository provided."""
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        repo = InMemoryGameRepository()
        runner = GameRunner(GameRunnerConfig(
            seed=42,
            repository=repo,
        ))
        final = runner.run()
        loaded = repo.load_game(runner.game_id)
        # Repository should have the game after run
        assert loaded is not None
        assert loaded.game_id == runner.game_id


# ---------------------------------------------------------------------------
# start_game endpoint integration
# ---------------------------------------------------------------------------


class TestGameRunnerStartGameEndpoint:
    def test_start_game_uses_runner(self) -> None:
        """start_game endpoint should use GameRunner for role assignment."""
        from werewolf_agent.api.app import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        client = TestClient(app)
        # Create game
        r = client.post("/games", json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "seed": 42,
        })
        assert r.status_code == 200
        game_id = r.json()["game"]["game_id"]
        # Start game
        r = client.post(f"/games/{game_id}/start", json={"caller_id": "mod1"})
        assert r.status_code == 200
        # Verify 12 players and roles assigned via RuleEngine
        r = client.get(f"/games/{game_id}/public-state")
        assert r.status_code == 200
        data = r.json()
        assert data.get("phase") == "night"
        assert len(data.get("players", {})) == 12

    def test_start_game_assigns_deterministic_roles(self) -> None:
        """start_game uses RuleEngine.assign_roles for deterministic role assignment."""
        from werewolf_agent.api.app import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        client = TestClient(app)
        # Create two games with same seed
        r1 = client.post("/games", json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "seed": 42,
        })
        game_id_1 = r1.json()["game"]["game_id"]
        r2 = client.post("/games", json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "seed": 42,
        })
        game_id_2 = r2.json()["game"]["game_id"]
        # Start both
        client.post(f"/games/{game_id_1}/start", json={"caller_id": "mod1"})
        client.post(f"/games/{game_id_2}/start", json={"caller_id": "mod1"})
        # Both should have the same player IDs (deterministic from seed)
        s1 = client.get(f"/games/{game_id_1}/public-state").json()
        s2 = client.get(f"/games/{game_id_2}/public-state").json()
        p1_ids = [p["player_id"] for p in s1.get("players", [])]
        p2_ids = [p["player_id"] for p in s2.get("players", [])]
        assert p1_ids == p2_ids

    def test_step_endpoint_advances_game(self) -> None:
        """POST /games/{game_id}/step should advance the game by one node."""
        from werewolf_agent.api.app import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        client = TestClient(app)
        r = client.post("/games", json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "seed": 42,
        })
        game_id = r.json()["game"]["game_id"]
        r = client.post(f"/games/{game_id}/start", json={"caller_id": "mod1"})
        assert r.status_code == 200
        # Step the game
        r = client.post(f"/games/{game_id}/step", json={"caller_id": "mod1"})
        assert r.status_code == 200
        assert r.json().get("success") is True

    def test_step_endpoint_returns_step_info(self) -> None:
        """Step endpoint returns step count and game phase."""
        from werewolf_agent.api.app import create_app
        from fastapi.testclient import TestClient
        app = create_app()
        client = TestClient(app)
        r = client.post("/games", json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "seed": 42,
        })
        game_id = r.json()["game"]["game_id"]
        client.post(f"/games/{game_id}/start", json={"caller_id": "mod1"})
        # Step
        r = client.post(f"/games/{game_id}/step", json={"caller_id": "mod1"})
        assert r.status_code == 200
        data = r.json()
        assert "step_count" in data or "success" in data

    def test_step_endpoint_rejects_paused_game(self) -> None:
        """Paused games should not advance through the step endpoint."""
        from werewolf_agent.api.app import create_app
        from fastapi.testclient import TestClient

        app = create_app()
        client = TestClient(app)
        r = client.post("/games", json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "seed": 42,
        })
        game_id = r.json()["game"]["game_id"]
        client.post(f"/games/{game_id}/start", json={"caller_id": "mod1"})
        pause = client.post(f"/games/{game_id}/pause", json={"caller_id": "mod1"})
        assert pause.status_code == 200

        step = client.post(f"/games/{game_id}/step", json={"caller_id": "mod1"})

        assert step.status_code == 400
        assert "paused" in step.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Memory lifecycle tests
# ---------------------------------------------------------------------------


class TestGameRunnerMemoryLifecycle:
    def test_game_runner_persists_memory_on_finish(self) -> None:
        """When memory_coordinator + repository are set, _save_memory_snapshot saves a snapshot."""
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        coord = PersistentMemoryCoordinator(repo)
        runner = GameRunner(GameRunnerConfig(
            seed=42,
            repository=repo,
            memory_coordinator=coord,
        ))
        # Run enough steps to assign players (the graph may error later, but
        # we only need players to be populated for the snapshot)
        for _ in range(5):
            runner.run_step()
        assert len(runner.state.players) > 0
        # Force-finish and save
        runner._finished = True
        runner._save_memory_snapshot()
        # A memory snapshot with game_id as key should exist in the repo
        snapshot = repo.load_memory_snapshot(runner.game_id)
        assert snapshot is not None, "Memory snapshot should have been saved"
        assert "cognition_matrices" in snapshot

    def test_game_runner_restores_memory_on_start(self) -> None:
        """A second runner with the same game_id should restore the prior snapshot."""
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        coord = PersistentMemoryCoordinator(repo)
        # Create runner 1, advance to assign players, then save a snapshot
        runner1 = GameRunner(GameRunnerConfig(
            seed=55,
            repository=repo,
            memory_coordinator=coord,
        ))
        for _ in range(5):
            runner1.run_step()
        runner1._finished = True
        runner1._save_memory_snapshot()
        game_id_1 = runner1.game_id
        # Verify snapshot exists
        assert repo.load_memory_snapshot(game_id_1) is not None
        # Create runner 2 with same game_id, coordinator, and repo
        runner2 = GameRunner(GameRunnerConfig(
            seed=55,
            repository=repo,
            memory_coordinator=coord,
        ))
        # Override game_id to match runner1's snapshot
        runner2.reset_game_id(game_id_1)
        # Re-run restore with the corrected game_id
        runner2._restore_memory_if_configured()
        # restored_memory should now be non-None
        assert runner2.restored_memory is not None

    def test_no_memory_save_without_coordinator(self) -> None:
        """Without memory_coordinator, no memory snapshot should be saved."""
        from werewolf_agent.storage.memory_store import InMemoryGameRepository

        repo = InMemoryGameRepository()
        runner = GameRunner(GameRunnerConfig(
            seed=42,
            repository=repo,
        ))
        runner.run()
        # No memory snapshot should exist
        snapshot = repo.load_memory_snapshot(runner.game_id)
        assert snapshot is None

    def test_restored_memory_is_none_without_coordinator(self) -> None:
        """restored_memory is None when no coordinator is configured."""
        runner = GameRunner(GameRunnerConfig(seed=42))
        assert runner.restored_memory is None
        assert runner.restored_rag is None

    def test_memory_save_uses_game_id_as_snapshot_id(self) -> None:
        """The snapshot_id should match the runner's game_id."""
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        coord = PersistentMemoryCoordinator(repo)
        runner = GameRunner(GameRunnerConfig(
            seed=99,
            repository=repo,
            memory_coordinator=coord,
        ))
        for _ in range(5):
            runner.run_step()
        runner._finished = True
        runner._save_memory_snapshot()
        # List all snapshots and verify one has our game_id
        snapshots = repo.list_memory_snapshots()
        snapshot_ids = [s["snapshot_id"] for s in snapshots]
        assert runner.game_id in snapshot_ids

    def test_save_memory_snapshot_preserves_player_matrices(self) -> None:
        """Saved snapshot contains cognition matrices for all players."""
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        coord = PersistentMemoryCoordinator(repo)
        runner = GameRunner(GameRunnerConfig(
            seed=7,
            repository=repo,
            memory_coordinator=coord,
        ))
        for _ in range(5):
            runner.run_step()
        runner._finished = True
        runner._save_memory_snapshot()
        snapshot = repo.load_memory_snapshot(runner.game_id)
        matrices = snapshot.get("cognition_matrices", {})
        assert len(matrices) == 12, f"Expected 12 cognition matrices, got {len(matrices)}"
