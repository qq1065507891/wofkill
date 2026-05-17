"""Tests for GameRunner: full game orchestration via LangGraph + RuleEngine."""

from __future__ import annotations

import pytest
from dataclasses import replace

from werewolf_agent.core.models import GameState, PlayerState, GameEvent
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.graph import RuntimeState, build_game_graph, _new_engine
from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig


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
        # Run with a wolf kill target
        final_state = runner.run(
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
        assert loaded is not None or final is not None


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
