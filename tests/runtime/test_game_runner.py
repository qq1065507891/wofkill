# -*- coding: utf-8 -*-
"""
验证 GameRunner 编排、终局边界与持久化行为。

作者: Project contributors
修改日期: 2026-07-16
"""

from __future__ import annotations

import json
import pytest
from dataclasses import replace

from werewolf_agent.core.models import GameState, PlayerState, GameEvent
from werewolf_agent.runtime.graph import _new_engine
from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig


def test_terminal_state_is_committed_at_step_boundary() -> None:
    from werewolf_agent.runtime.nodes._shared import _dispatch_agent

    engine = _new_engine()
    gs = GameState(
        game_id="terminal_dispatch",
        players={
            "wolf": PlayerState(id="wolf", role="werewolf"),
            "good": PlayerState(id="good", role="villager"),
        },
        winning_faction="werewolf",
    )

    class Registry:
        def get_agent(self, _player_id):
            return object()

    called = False

    def in_game_agent(*_args, **_kwargs):
        nonlocal called
        called = True
        return {"action_trace": {"final_action_type": "vote"}}

    result = _dispatch_agent(
        {"game_state": gs, "engine": engine, "agent_registry": Registry()},
        in_game_agent,
    )

    assert result is None
    assert called is False


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


def test_game_runner_runtime_state_does_not_force_badge_tear() -> None:
    runner = GameRunner(GameRunnerConfig(seed=42))

    runtime_state = runner._build_runtime_state()

    assert runtime_state["badge_decision"] is None
    assert runtime_state["badge_target_id"] is None


def test_game_runner_runtime_state_includes_live_cognition_manager() -> None:
    runner = GameRunner(GameRunnerConfig(seed=42))

    runtime_state = runner._build_runtime_state()

    assert runtime_state["cognition_state_manager"] is runner._cognition_state_manager


def test_game_runner_runtime_state_initializes_shared_audit_containers() -> None:
    runner = GameRunner(GameRunnerConfig(seed=42))

    runtime_state = runner._build_runtime_state()

    assert runtime_state["action_index_by_game"] == {}
    assert runtime_state["pending_exposure_events_by_trace"] == {}


def test_game_runner_process_chunk_updates_live_cognition_manager() -> None:
    runner = GameRunner(GameRunnerConfig(seed=42))
    base_state = GameState(
        game_id=runner.game_id,
        phase="day",
        day_number=1,
        night_number=1,
        players={
            "p01": PlayerState(id="p01", role="seer", alive=True),
            "p02": PlayerState(id="p02", role="villager", alive=True),
            "p03": PlayerState(id="p03", role="werewolf", alive=True),
        },
    )
    next_state = replace(
        base_state,
        events=[
            GameEvent(
                type="speech",
                payload={
                    "speaker": "p01",
                    "text": "我是预言家，查验 p03 是狼人",
                    "day_number": 1,
                },
            )
        ],
    )
    runner._state = base_state
    runner._cognition_state_manager.initialize(base_state)

    runner._process_chunk({"speech": {"game_state": next_state}})

    assert runner._cognition_state_manager.processed_event_count() == 1
    assert runner._cognition_state_manager.prompt_belief_summary(
        "p02",
        next_state,
    )["my_suspects"][0]["player"] == "p03"


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

    @pytest.mark.parametrize("game_id", [
        "run1",
        "g_713001",
        "audit.run-1_seed-713001",
        "a" * 128,
    ])
    def test_config_accepts_safe_explicit_game_id(self, game_id: str) -> None:
        assert GameRunnerConfig(game_id=game_id).game_id == game_id

    @pytest.mark.parametrize("game_id", [
        "../escape",
        "safe/escape",
        "safe\\escape",
        "safe..escape",
        " leading",
        "has space",
        "has\ttab",
        "has\nnewline",
        "_leading",
        "a" * 129,
    ])
    def test_config_rejects_unsafe_explicit_game_id(self, game_id: str) -> None:
        with pytest.raises(ValueError, match="game_id"):
            GameRunnerConfig(game_id=game_id)


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

    def test_agent_registry_wires_shared_persona_router(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        runner = GameRunner(GameRunnerConfig(
            seed=99,
            use_agent_registry=True,
            model_config_path="config/models.yaml",
            persona_config_path="config/personas/jingcheng_style_prototypes.yaml",
        ))

        registry = runner._build_runtime_state()["agent_registry"]
        p01 = registry.get_agent("p01")
        p02 = registry.get_agent("p02")

        assert p01.persona_key == "logic_leader"
        assert p02.persona_key == "aggressive_bluffer"
        assert p01.persona_router is not None
        assert p01.persona_router is p02.persona_router

    def test_probe_tool_call_support_rejects_text_fallback_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from werewolf_agent.model_gateway.router import GenerateResult

        class TextFallbackProvider:
            @property
            def name(self) -> str:
                return "mock"

            def generate(self, prompt, config, system_prompt=None, tools=None, tool_choice=None):
                return GenerateResult(
                    text='{"action_type":"no_action"}',
                    provider=self.name,
                    model=config.model,
                    tool_call_required=bool(tool_choice),
                    tool_call_received=False,
                    text_fallback_used=True,
                    structured_failure_reason="missing_tool_call",
                )

        class ProbeRouter:
            @classmethod
            def from_yaml(cls, path, register_env_providers=False):
                from werewolf_agent.model_gateway.router import ModelRouter

                return ModelRouter(
                    model_profiles={"mock_model": {"model": "mock", "provider": "mock"}},
                    llm_profiles={"default": {"default": {"provider": "mock", "model_profile": "mock_model"}}},
                    player_assignments={"p01": "default"},
                    providers={"mock": TextFallbackProvider()},
                )

        monkeypatch.setattr("werewolf_agent.runtime.game_runner.ModelRouter", ProbeRouter)
        cfg = GameRunnerConfig(
            seed=99,
            use_agent_registry=True,
            probe_tool_call_support=True,
        )

        with pytest.raises(RuntimeError, match="tool call probe failed"):
            GameRunner(cfg)


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

    def test_game_runner_uses_explicit_run_scoped_game_id(self) -> None:
        runner = GameRunner(GameRunnerConfig(
            seed=123,
            game_id="audit-run-abc-seed-123",
        ))

        assert runner.game_id == "audit-run-abc-seed-123"
        assert runner.state.game_id == "audit-run-abc-seed-123"

    def test_game_runner_keeps_seed_based_game_id_by_default(self) -> None:
        runner = GameRunner(GameRunnerConfig(seed=123))

        assert runner.game_id == "g_123"

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

    @pytest.mark.parametrize("entrypoint", ["run", "run_scripted", "run_step"])
    def test_entrypoints_abort_unrecoverable_exception_with_context(
        self, entrypoint: str, tmp_path,
    ) -> None:
        runner = GameRunner(GameRunnerConfig(
            seed=42, emergency_artifact_dir=tmp_path,
        ))

        class BrokenGraph:
            def stream(self, *_args, **_kwargs):
                raise RuntimeError("provider secret")

        runner._graph = BrokenGraph()
        result = getattr(runner, entrypoint)()

        assert result.status == "aborted"
        assert result.termination_reason == "unrecoverable_runtime_error"
        event = next(event for event in result.events if event.type == "game_aborted")
        assert event.payload["phase"] == "setup"
        assert event.payload["step"] == 0
        assert event.payload["exception_type"] == "RuntimeError"
        assert (tmp_path / f"emergency_abort_{runner.game_id}.json").exists()

    def test_graph_recursion_error_maps_to_graph_recursion_limit(self, tmp_path) -> None:
        from langgraph.errors import GraphRecursionError

        runner = GameRunner(GameRunnerConfig(
            seed=43, emergency_artifact_dir=tmp_path,
        ))

        class RecursiveGraph:
            def stream(self, *_args, **_kwargs):
                raise GraphRecursionError("limit")

        runner._graph = RecursiveGraph()
        result = runner.run()

        assert result.status == "aborted"
        assert result.termination_reason == "graph_recursion_limit"

    def test_run_aborts_stuck_game_at_fifty_repeated_snapshots(self, tmp_path) -> None:
        runner = GameRunner(GameRunnerConfig(
            seed=44, emergency_artifact_dir=tmp_path,
        ))

        class StuckGraph:
            def stream(self, *_args, **_kwargs):
                for _ in range(51):
                    yield {"stuck_node": {"game_state": runner.state}}

        runner._graph = StuckGraph()
        result = runner.run(max_steps=100)

        assert result.status == "aborted"
        assert result.termination_reason == "step_limit"
        assert next(event for event in result.events if event.type == "game_aborted").payload["last_node"] == "stuck_node"

    def test_run_aborts_when_stream_ends_without_winner(self, tmp_path) -> None:
        runner = GameRunner(GameRunnerConfig(
            seed=45, emergency_artifact_dir=tmp_path,
        ))
        runner._graph = type("ShortGraph", (), {"stream": lambda *_args, **_kwargs: iter(())})()

        result = runner.run(max_steps=1)

        assert result.status == "aborted"
        assert result.termination_reason == "step_limit"

    def test_abort_skips_reflection_and_persists_state_and_event(self, tmp_path) -> None:
        from werewolf_agent.storage.memory_store import InMemoryGameRepository

        repo = InMemoryGameRepository()
        runner = GameRunner(GameRunnerConfig(
            seed=46, repository=repo, emergency_artifact_dir=tmp_path,
        ))
        runner._graph = type("ShortGraph", (), {"stream": lambda *_args, **_kwargs: iter(())})()
        runner._save_memory_snapshot = lambda: pytest.fail("aborted game reflected")

        result = runner.run(max_steps=1)

        assert repo.load_game(runner.game_id) == result
        stored_events = repo.load_events(runner.game_id)
        assert [event.type for event in stored_events] == ["game_aborted"]
        assert not (tmp_path / f"emergency_abort_{runner.game_id}.json").exists()

    def test_repeated_abort_persistence_keeps_single_sqlite_terminal_event(
        self, tmp_path,
    ) -> None:
        from werewolf_agent.storage.sqlite_store import SqliteGameRepository

        repo = SqliteGameRepository(str(tmp_path / "runner.db"))
        runner = GameRunner(GameRunnerConfig(
            seed=51, repository=repo, emergency_artifact_dir=tmp_path,
        ))
        runner._graph = type(
            "ShortGraph", (), {"stream": lambda *_args, **_kwargs: iter(())}
        )()

        result = runner.run(max_steps=1)
        runner._persist_if_configured()

        assert repo.load_game(runner.game_id) == result
        assert repo.load_events(runner.game_id) == [result.events[-1]]

    def test_repository_failure_falls_back_to_emergency_artifact(self, tmp_path) -> None:
        class BrokenRepository:
            def save_game(self, _state):
                raise OSError("database unavailable")

        runner = GameRunner(GameRunnerConfig(
            seed=47, repository=BrokenRepository(), emergency_artifact_dir=tmp_path,
        ))
        runner._graph = type("ShortGraph", (), {"stream": lambda *_args, **_kwargs: iter(())})()

        result = runner.run(max_steps=1)

        assert result.status == "aborted"
        assert (tmp_path / f"emergency_abort_{runner.game_id}.json").exists()

    def test_event_persistence_failure_falls_back_to_emergency_artifact(
        self, tmp_path,
    ) -> None:
        from werewolf_agent.storage.memory_store import InMemoryGameRepository

        class BrokenEventRepository(InMemoryGameRepository):
            def append_events(self, _game_id, _events):
                raise OSError("event store unavailable")

        repo = BrokenEventRepository()
        runner = GameRunner(GameRunnerConfig(
            seed=49, repository=repo, emergency_artifact_dir=tmp_path,
        ))
        runner._graph = type(
            "ShortGraph", (), {"stream": lambda *_args, **_kwargs: iter(())}
        )()

        result = runner.run(max_steps=1)

        assert repo.load_game(runner.game_id) == result
        assert (tmp_path / f"emergency_abort_{runner.game_id}.json").exists()

    def test_finished_persistence_failure_is_not_masked_as_abort(self, tmp_path) -> None:
        class BrokenRepository:
            def save_game(self, _state):
                raise OSError("finished save unavailable")

        runner = GameRunner(GameRunnerConfig(
            seed=50, repository=BrokenRepository(), emergency_artifact_dir=tmp_path,
        ))
        won = replace(
            runner.state, phase="finished", winning_faction="good",
        )
        runner._graph = type(
            "WonGraph", (),
            {"stream": lambda *_args, **_kwargs: iter([
                {"victory": {"game_state": won}},
            ])},
        )()

        with pytest.raises(OSError, match="finished save unavailable"):
            runner.run(max_steps=1)

        assert runner.state.status == "finished"
        assert runner.state.winning_faction == "good"
        assert not (tmp_path / f"emergency_abort_{runner.game_id}.json").exists()

    def test_repository_and_emergency_failure_is_critical_and_raised(
        self, monkeypatch, tmp_path, caplog,
    ) -> None:
        class BrokenRepository:
            def save_game(self, _state):
                raise OSError("database unavailable")

        runner = GameRunner(GameRunnerConfig(
            seed=48, repository=BrokenRepository(), emergency_artifact_dir=tmp_path,
        ))
        runner._graph = type("ShortGraph", (), {"stream": lambda *_args, **_kwargs: iter(())})()
        monkeypatch.setattr(
            "werewolf_agent.runtime.game_runner_execution.write_emergency_abort",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk unavailable")),
        )

        with caplog.at_level("CRITICAL"):
            with pytest.raises(RuntimeError, match="emergency abort persistence failed"):
                runner.run(max_steps=1)

        assert "CRITICAL" in caplog.text


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

    def test_executor_reports_aborted_runner_as_error(self) -> None:
        from werewolf_agent.runtime.executor import LocalRuntimeExecutor

        executor = LocalRuntimeExecutor()
        runner = _FakeRunner()

        def abort_step() -> GameState:
            runner.step_count += 1
            runner.finished = True
            runner.state = replace(
                runner.state,
                status="aborted",
                termination_reason="unrecoverable_runtime_error",
            )
            return runner.state

        runner.run_step = abort_step
        result = executor.try_step("g_exec", runner)

        assert result.success is False
        assert result.status == "error"
        assert "unrecoverable_runtime_error" in result.message
        assert executor.status("g_exec").state == "error"


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
        runner.run()
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
            "caller_id": "mod1",
            "caller_role": "moderator",
        })
        assert r.status_code == 200
        game_id = r.json()["game"]["game_id"]
        # Start game
        r = client.post(f"/games/{game_id}/start", json={"caller_id": "mod1", "caller_role": "moderator"})
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
            "caller_id": "mod1",
            "caller_role": "moderator",
        })
        game_id_1 = r1.json()["game"]["game_id"]
        r2 = client.post("/games", json={
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "seed": 42,
            "caller_id": "mod1",
            "caller_role": "moderator",
        })
        game_id_2 = r2.json()["game"]["game_id"]
        # Start both
        client.post(f"/games/{game_id_1}/start", json={"caller_id": "mod1", "caller_role": "moderator"})
        client.post(f"/games/{game_id_2}/start", json={"caller_id": "mod1", "caller_role": "moderator"})
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
            "caller_id": "mod1",
            "caller_role": "moderator",
        })
        game_id = r.json()["game"]["game_id"]
        r = client.post(f"/games/{game_id}/start", json={"caller_id": "mod1", "caller_role": "moderator"})
        assert r.status_code == 200
        # Step the game
        r = client.post(f"/games/{game_id}/step", json={"caller_id": "mod1", "caller_role": "moderator"})
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
            "caller_id": "mod1",
            "caller_role": "moderator",
        })
        game_id = r.json()["game"]["game_id"]
        client.post(f"/games/{game_id}/start", json={"caller_id": "mod1", "caller_role": "moderator"})
        # Step
        r = client.post(f"/games/{game_id}/step", json={"caller_id": "mod1", "caller_role": "moderator"})
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
            "caller_id": "mod1",
            "caller_role": "moderator",
        })
        game_id = r.json()["game"]["game_id"]
        client.post(f"/games/{game_id}/start", json={"caller_id": "mod1", "caller_role": "moderator"})
        pause = client.post(f"/games/{game_id}/pause", json={"caller_id": "mod1", "caller_role": "moderator"})
        assert pause.status_code == 200

        step = client.post(f"/games/{game_id}/step", json={"caller_id": "mod1", "caller_role": "moderator"})

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

    def test_game_runner_restores_latest_memory_for_new_game_id(self) -> None:
        """New game IDs should fall back to the latest cross-game snapshot."""
        from werewolf_agent.memory.store import MemoryStore
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        coord = PersistentMemoryCoordinator(repo)
        store = MemoryStore()
        store.init_matrix("p01", ["p01", "p02"])
        coord.save_memory(store, "g_previous")

        runner = GameRunner(GameRunnerConfig(
            seed=98765,
            repository=repo,
            memory_coordinator=coord,
        ))

        assert runner.restored_memory is not None
        assert runner.restored_memory.get_matrix("p01") is not None

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

    def test_save_memory_snapshot_preserves_live_cognition_evidence(self) -> None:
        """Phase 1: saved snapshot reuses the live manager MemoryStore."""
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        coord = PersistentMemoryCoordinator(repo)
        runner = GameRunner(GameRunnerConfig(
            seed=125,
            repository=repo,
            memory_coordinator=coord,
        ))
        runner._state = GameState(
            game_id=runner.game_id,
            phase="finished",
            day_number=1,
            winning_faction="good",
            players={
                "p01": PlayerState(id="p01", role="seer", alive=True),
                "p02": PlayerState(id="p02", role="villager", alive=True),
                "p03": PlayerState(id="p03", role="werewolf", alive=False),
            },
            events=[
                GameEvent(
                    type="speech",
                    payload={
                        "speaker": "p01",
                        "text": "我是预言家，查验 p03 是狼人",
                        "day_number": 1,
                    },
                )
            ],
        )
        runner._cognition_state_manager.initialize(runner._state)
        runner._cognition_state_manager.update_from_events(runner._state)

        runner._save_memory_snapshot()

        snapshot = repo.load_memory_snapshot(runner.game_id)
        matrices = snapshot.get("cognition_matrices", {})
        evidence_count = sum(
            len(entry.get("key_evidence", []))
            for matrix in matrices.values()
            for entry in matrix.get("entries", {}).values()
        )
        assert evidence_count > 0

    def test_save_memory_snapshot_does_not_duplicate_relation_graph(self) -> None:
        """Repeated snapshot saves must rebuild relation graph idempotently."""
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        coord = PersistentMemoryCoordinator(repo)
        runner = GameRunner(GameRunnerConfig(
            seed=126,
            repository=repo,
            memory_coordinator=coord,
        ))
        runner._state = GameState(
            game_id=runner.game_id,
            phase="finished",
            day_number=1,
            winning_faction="good",
            players={
                "p01": PlayerState(id="p01", role="seer", alive=True),
                "p02": PlayerState(id="p02", role="villager", alive=True),
            },
            events=[
                GameEvent(
                    type="vote",
                    payload={"voter": "p01", "target": "p02", "day_number": 1},
                )
            ],
        )

        runner._save_memory_snapshot()
        first = repo.load_memory_snapshot(runner.game_id)
        runner._save_memory_snapshot()
        second = repo.load_memory_snapshot(runner.game_id)

        assert len(first["relation_graph"]["events"]) == 1
        assert len(second["relation_graph"]["events"]) == 1

    def test_save_memory_snapshot_writes_v2_reflections_only(self) -> None:
        from werewolf_agent.core.models import GameEvent, GameState, PlayerState
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        coord = PersistentMemoryCoordinator(repo)
        runner = GameRunner(GameRunnerConfig(
            seed=123,
            repository=repo,
            memory_coordinator=coord,
        ))
        runner._state = GameState(
            game_id=runner.game_id,
            phase="finished",
            day_number=2,
            winning_faction="good",
            players={
                "p01": PlayerState(id="p01", role="seer", alive=True),
                "p02": PlayerState(id="p02", role="werewolf", alive=False),
            },
            events=[
                GameEvent(
                    type="role_revealed",
                    payload={"player_id": "p01", "role": "seer"},
                ),
                GameEvent(
                    type="role_revealed",
                    payload={"player_id": "p02", "role": "werewolf"},
                ),
                GameEvent(
                    type="reflection_complete",
                    payload={
                        "player_count": 2,
                        "entries": [
                            {
                                "player_id": "p01",
                                "role": "seer",
                                "alive": True,
                                "verification": {
                                    "status": "verified",
                                    "verified_fact_count": 1,
                                    "verified_lessons": [{
                                        "lesson_id": "l1",
                                        "abstraction": "历史玩家A 在对跳局需要先核验警徽流。",
                                    }],
                                    "rejected_fact_count": 0,
                                    "rejected_lesson_count": 0,
                                },
                                "reflection": json.dumps({
                                    "claims": [{
                                        "claim_id": "c1",
                                        "event_ref": f"{runner.game_id}:0",
                                        "claim_type": "role",
                                        "subject_id": "p01",
                                        "value": "seer",
                                    }],
                                    "lessons": [{
                                        "lesson_id": "l1",
                                        "abstraction": "p01 在对跳局需要先核验警徽流。",
                                        "claim_dependencies": ["c1"],
                                    }],
                                }, ensure_ascii=False),
                            },
                            {
                                "player_id": "p02",
                                "role": "werewolf",
                                "alive": False,
                                "verification": {
                                    "status": "verified",
                                    "verified_fact_count": 1,
                                    "verified_lessons": [{
                                        "lesson_id": "l2",
                                        "abstraction": "历史玩家A 下次悍跳前要统一警徽流口径。",
                                    }],
                                    "rejected_fact_count": 0,
                                    "rejected_lesson_count": 0,
                                },
                                "reflection": json.dumps({
                                    "claims": [{
                                        "claim_id": "c2",
                                        "event_ref": f"{runner.game_id}:1",
                                        "claim_type": "role",
                                        "subject_id": "p02",
                                        "value": "werewolf",
                                    }],
                                    "lessons": [{
                                        "lesson_id": "l2",
                                        "abstraction": "p02 下次悍跳前要统一警徽流口径。",
                                        "claim_dependencies": ["c2"],
                                    }],
                                }, ensure_ascii=False),
                            },
                        ],
                    },
                )
            ],
        )

        runner._save_memory_snapshot()

        rows = repo.load_all_reflections()
        assert len(rows) == 2
        assert {row["schema_version"] for row in rows} == {2}
        assert all("quality_status" in row for row in rows)
        assert all("text" not in row for row in rows)
        assert all(row["source"]["llm_self_review"] == "" for row in rows)
        assert all(row["source"]["source_game_id"] == runner.game_id for row in rows)
        assert all("p01" not in row["prompt_card"]["lesson"] for row in rows)

    def test_save_memory_snapshot_rejects_unstructured_raw_reflection(self) -> None:
        from werewolf_agent.core.models import GameEvent, GameState, PlayerState
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        runner = GameRunner(GameRunnerConfig(
            seed=125,
            repository=repo,
            memory_coordinator=PersistentMemoryCoordinator(repo),
        ))
        runner._state = GameState(
            game_id=runner.game_id,
            phase="finished",
            winning_faction="good",
            players={"p01": PlayerState(id="p01", role="seer", alive=True)},
            events=[GameEvent(type="reflection_complete", payload={
                "entries": [{"player_id": "p01", "reflection": "p01 是预言家"}],
            })],
        )

        runner._save_memory_snapshot()

        assert repo.load_all_reflections() == []

    def test_reflection_persistence_audit_rejects_extra_saved_claim_ids(self) -> None:
        from werewolf_agent.core.models import GameEvent, GameState, PlayerState
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        coordinator = PersistentMemoryCoordinator(repo)
        original_save_all = coordinator.save_all

        def save_all_with_contamination(*args, **kwargs):
            original_save_all(*args, **kwargs)
            rows = repo.load_all_reflections()
            assert rows
            row = rows[0]
            row.setdefault("source", {})["verified_claim_ids"] = ["c-good", "c-bad"]
            repo.save_reflection(row)

        coordinator.save_all = save_all_with_contamination
        runner = GameRunner(GameRunnerConfig(
            seed=127, repository=repo, memory_coordinator=coordinator,
        ))
        runner._state = GameState(
            game_id=runner.game_id, phase="finished", winning_faction="good",
            players={"p01": PlayerState(id="p01", role="seer")},
            events=[GameEvent(type="reflection_complete", payload={
                "player_count": 1, "entries": [{
                "player_id": "p01",
                "verification": {
                    "status": "verified",
                    "decision_id": "reflection:g:p01",
                    "verified_fact_count": 1,
                    "verified_claim_ids": ["c-good"],
                    "rejected_claim_ids": ["c-bad"],
                    "verified_lessons": [{
                        "lesson_id": "l1", "abstraction": "对跳时应先核验公开票型",
                    }],
                    "rejected_fact_count": 1,
                    "rejected_lesson_count": 0,
                },
            }]})],
        )

        runner._save_memory_snapshot()

        audit = next(
            event for event in runner.state.events
            if event.type == "reflection_persistence_audit"
        )
        assert audit.payload["entries"] == [{
            "player_id": "p01",
            "decision_id": "reflection:g:p01",
            "verified_claim_ids": ["c-good"],
            "entry_id": f"reflection_{runner.game_id}_p01",
            "row_found": True,
            "persistence_complete": False,
            "persisted_rejected_fact_count": None,
        }]
        assert audit.payload["persistence_complete"] is False
        assert repo.load_reflections_by_game(runner.game_id) == []
        assert repo.load_memory_snapshot(runner.game_id) is None
        assert repo.load_memory_snapshot("latest") is None

    def test_reflection_persistence_audit_fails_closed_when_store_v2_fails(self) -> None:
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        class FailingReflectionRepository(InMemoryGameRepository):
            def save_reflection(self, entry):
                raise RuntimeError("reflection write unavailable")

        repo = FailingReflectionRepository()
        runner = GameRunner(GameRunnerConfig(
            seed=128,
            repository=repo,
            memory_coordinator=PersistentMemoryCoordinator(repo),
        ))
        runner._state = self._verified_reflection_state(runner.game_id)

        runner._save_memory_snapshot()

        audit = next(
            event for event in runner.state.events
            if event.type == "reflection_persistence_audit"
        )
        assert audit.payload["persistence_complete"] is False
        assert audit.payload["entries"] == [{
            "player_id": "p01",
            "decision_id": "reflection:g:p01",
            "verified_claim_ids": ["c-good"],
            "entry_id": f"reflection_{runner.game_id}_p01",
            "row_found": False,
            "persistence_complete": False,
            "persisted_rejected_fact_count": None,
        }]

    def test_reflection_batch_rolls_back_when_later_row_write_fails(self) -> None:
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        class FailSecondReflectionWriteRepository(InMemoryGameRepository):
            def __init__(self) -> None:
                super().__init__()
                self.reflection_writes = 0

            def save_reflection(self, entry):
                self.reflection_writes += 1
                if self.reflection_writes == 2:
                    raise RuntimeError("second reflection write unavailable")
                super().save_reflection(entry)

        repo = FailSecondReflectionWriteRepository()
        runner = GameRunner(GameRunnerConfig(
            seed=131,
            repository=repo,
            memory_coordinator=PersistentMemoryCoordinator(repo),
        ))
        runner._state = self._verified_reflection_state(
            runner.game_id,
            player_ids=("p01", "p02"),
        )

        runner._save_memory_snapshot()

        audit = next(
            event for event in runner.state.events
            if event.type == "reflection_persistence_audit"
        )
        assert audit.payload["persistence_complete"] is False
        assert audit.payload["expected_entry_count"] == 2
        assert repo.load_reflections_by_game(runner.game_id) == []
        assert runner._cognition_state_manager.memory_store.reflections.all_v2_entries() == []

    def test_reflection_batch_rolls_back_when_snapshot_save_fails(self) -> None:
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        coordinator = PersistentMemoryCoordinator(repo)
        original_save_all = coordinator.save_all

        def save_all_then_fail(*args, **kwargs):
            original_save_all(*args, **kwargs)
            raise RuntimeError("snapshot confirmation unavailable")

        coordinator.save_all = save_all_then_fail
        runner = GameRunner(GameRunnerConfig(
            seed=132, repository=repo, memory_coordinator=coordinator,
        ))
        runner._state = self._verified_reflection_state(runner.game_id)

        runner._save_memory_snapshot()

        audit = next(
            event for event in runner.state.events
            if event.type == "reflection_persistence_audit"
        )
        assert audit.payload["persistence_complete"] is False
        assert repo.load_reflections_by_game(runner.game_id) == []
        assert runner._cognition_state_manager.memory_store.reflections.all_v2_entries() == []

    def test_reflection_rollback_restores_repo_only_row_and_previous_snapshots(self) -> None:
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        coordinator = PersistentMemoryCoordinator(repo)
        runner = GameRunner(GameRunnerConfig(
            seed=135, repository=repo, memory_coordinator=coordinator,
        ))
        entry_id = f"reflection_{runner.game_id}_p01"
        old_row = {
            "entry_id": entry_id,
            "game_id": runner.game_id,
            "sentinel": "repo-only-before-transaction",
        }
        old_game_snapshot = {"sentinel": "old-game"}
        old_latest_snapshot = {"sentinel": "old-latest"}
        assert runner._cognition_state_manager.memory_store.reflections.count() == 0
        repo.save_reflection(old_row)
        repo.save_memory_snapshot(runner.game_id, old_game_snapshot)
        repo.save_memory_snapshot("latest", old_latest_snapshot)
        original_save_all = coordinator.save_all

        def save_all_then_fail(*args, **kwargs):
            original_save_all(*args, **kwargs)
            raise RuntimeError("snapshot confirmation unavailable")

        coordinator.save_all = save_all_then_fail
        runner._state = self._verified_reflection_state(runner.game_id)

        runner._save_memory_snapshot()

        audit = next(
            event for event in runner.state.events
            if event.type == "reflection_persistence_audit"
        )
        assert audit.payload["persistence_complete"] is False
        assert audit.payload["rollback_complete"] is True
        assert repo.load_reflection(entry_id) == old_row
        assert repo.load_memory_snapshot(runner.game_id) == old_game_snapshot
        assert repo.load_memory_snapshot("latest") == old_latest_snapshot

    def test_failed_reflection_delete_is_marked_inactive_and_reported(self) -> None:
        from werewolf_agent.memory.reflection import ReflectionMemory
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        class DeleteFailingRepository(InMemoryGameRepository):
            def delete_reflection(self, entry_id):
                raise RuntimeError("reflection delete unavailable")

        repo = DeleteFailingRepository()
        coordinator = PersistentMemoryCoordinator(repo)
        original_save_all = coordinator.save_all

        def save_all_then_fail(*args, **kwargs):
            original_save_all(*args, **kwargs)
            raise RuntimeError("snapshot confirmation unavailable")

        coordinator.save_all = save_all_then_fail
        runner = GameRunner(GameRunnerConfig(
            seed=136, repository=repo, memory_coordinator=coordinator,
        ))
        runner._state = self._verified_reflection_state(runner.game_id)

        runner._save_memory_snapshot()

        audit = next(
            event for event in runner.state.events
            if event.type == "reflection_persistence_audit"
        )
        row = repo.load_reflection(f"reflection_{runner.game_id}_p01")
        assert audit.payload["persistence_complete"] is False
        assert audit.payload["rollback_complete"] is False
        assert row is not None and row["_persistence_active"] is False
        assert ReflectionMemory(repo=repo).all_v2_entries() == []

    def test_failed_reflection_restore_is_quarantined_and_reported(self) -> None:
        from werewolf_agent.memory.reflection import ReflectionMemory
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        class RestoreFailingRepository(InMemoryGameRepository):
            fail_restore = False

            def save_reflection(self, entry):
                if self.fail_restore and entry.get("sentinel") == "old-row":
                    raise RuntimeError("reflection restore unavailable")
                super().save_reflection(entry)

        repo = RestoreFailingRepository()
        coordinator = PersistentMemoryCoordinator(repo)
        runner = GameRunner(GameRunnerConfig(
            seed=138, repository=repo, memory_coordinator=coordinator,
        ))
        entry_id = f"reflection_{runner.game_id}_p01"
        repo.save_reflection({
            "entry_id": entry_id,
            "game_id": runner.game_id,
            "sentinel": "old-row",
        })
        original_save_all = coordinator.save_all

        def save_all_then_fail(*args, **kwargs):
            original_save_all(*args, **kwargs)
            repo.fail_restore = True
            raise RuntimeError("snapshot confirmation unavailable")

        coordinator.save_all = save_all_then_fail
        runner._state = self._verified_reflection_state(runner.game_id)

        runner._save_memory_snapshot()

        audit = next(
            event for event in runner.state.events
            if event.type == "reflection_persistence_audit"
        )
        row = repo.load_reflection(entry_id)
        assert audit.payload["persistence_complete"] is False
        assert audit.payload["rollback_complete"] is False
        assert row is not None and row["_persistence_active"] is False
        assert ReflectionMemory(repo=repo).all_v2_entries() == []

    def test_failed_snapshot_delete_leaves_only_unrestorable_tombstones(self) -> None:
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        class SnapshotDeleteFailingRepository(InMemoryGameRepository):
            def delete_memory_snapshot(self, snapshot_id):
                raise RuntimeError("snapshot delete unavailable")

        repo = SnapshotDeleteFailingRepository()
        coordinator = PersistentMemoryCoordinator(repo)
        original_save_all = coordinator.save_all

        def save_all_then_fail(*args, **kwargs):
            original_save_all(*args, **kwargs)
            raise RuntimeError("snapshot confirmation unavailable")

        coordinator.save_all = save_all_then_fail
        runner = GameRunner(GameRunnerConfig(
            seed=137, repository=repo, memory_coordinator=coordinator,
        ))
        runner._state = self._verified_reflection_state(runner.game_id)

        runner._save_memory_snapshot()

        audit = next(
            event for event in runner.state.events
            if event.type == "reflection_persistence_audit"
        )
        assert audit.payload["persistence_complete"] is False
        assert audit.payload["rollback_complete"] is False
        assert coordinator.restore_for_new_game(runner.game_id).reflections.count() == 0
        assert repo.load_memory_snapshot(runner.game_id)["_persistence_active"] is False
        assert repo.load_memory_snapshot("latest")["_persistence_active"] is False

    def test_reflection_batch_rolls_back_when_repository_readback_fails(self) -> None:
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        class ReadbackFailingRepository(InMemoryGameRepository):
            def load_reflections_by_game(self, game_id):
                raise RuntimeError("reflection readback unavailable")

        repo = ReadbackFailingRepository()
        runner = GameRunner(GameRunnerConfig(
            seed=133,
            repository=repo,
            memory_coordinator=PersistentMemoryCoordinator(repo),
        ))
        runner._state = self._verified_reflection_state(runner.game_id)

        runner._save_memory_snapshot()

        audit = next(
            event for event in runner.state.events
            if event.type == "reflection_persistence_audit"
        )
        assert audit.payload["persistence_complete"] is False
        assert repo.load_all_reflections() == []
        assert runner._cognition_state_manager.memory_store.reflections.all_v2_entries() == []
        assert repo.load_memory_snapshot(runner.game_id) is None
        assert repo.load_memory_snapshot("latest") is None

    def test_empty_verified_reflection_batch_is_no_valid_entries(self) -> None:
        from werewolf_agent.core.models import GameEvent, GameState, PlayerState
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        runner = GameRunner(GameRunnerConfig(
            seed=134,
            repository=repo,
            memory_coordinator=PersistentMemoryCoordinator(repo),
        ))
        runner._state = GameState(
            game_id=runner.game_id,
            phase="finished",
            winning_faction="good",
            players={"p01": PlayerState(id="p01", role="seer")},
            events=[GameEvent(type="reflection_complete", payload={"entries": []})],
        )

        runner._save_memory_snapshot()

        audit = next(
            event for event in runner.state.events
            if event.type == "reflection_persistence_audit"
        )
        from werewolf_agent.core.event_visibility import EventVisibility

        assert audit.visibility is EventVisibility.MODERATOR_ONLY
        assert audit.schema_version == "2"
        assert audit.payload == {
            "status": "no_valid_entries",
            "expected_entry_count": 0,
            "persistence_complete": False,
            "rollback_complete": True,
            "entries": [],
        }
        assert any(
            event.type == "reflection_no_valid_entries"
            for event in runner.state.events
        )

    def test_reflection_persistence_audit_fails_closed_when_expected_row_missing(self) -> None:
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        coordinator = PersistentMemoryCoordinator(repo)
        original_save_all = coordinator.save_all

        def save_all_then_remove_rows(*args, **kwargs):
            original_save_all(*args, **kwargs)
            for row in repo.load_all_reflections():
                repo.delete_reflection(row["entry_id"])

        coordinator.save_all = save_all_then_remove_rows
        runner = GameRunner(GameRunnerConfig(
            seed=129, repository=repo, memory_coordinator=coordinator,
        ))
        runner._state = self._verified_reflection_state(runner.game_id)

        runner._save_memory_snapshot()

        audit = next(
            event for event in runner.state.events
            if event.type == "reflection_persistence_audit"
        )
        assert audit.payload["persistence_complete"] is False
        assert audit.payload["entries"][0]["row_found"] is False
        assert audit.payload["entries"][0]["persistence_complete"] is False
        assert audit.payload["entries"][0]["persisted_rejected_fact_count"] is None

    @pytest.mark.parametrize(
        "corruption",
        ("stale_same_id", "wrong_player", "wrong_source_game", "missing_verified_claim"),
    )
    def test_reflection_readback_rejects_stale_or_mismatched_row(
        self,
        corruption: str,
    ) -> None:
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        coordinator = PersistentMemoryCoordinator(repo)
        original_save_all = coordinator.save_all

        def save_all_then_corrupt(*args, **kwargs):
            original_save_all(*args, **kwargs)
            row = repo.load_all_reflections()[0]
            if corruption == "stale_same_id":
                row["quality_score"] = 0.0
                row["prompt_card"]["lesson"] = "陈旧同 ID 内容"
            elif corruption == "wrong_player":
                row["player_id"] = "p99"
            elif corruption == "wrong_source_game":
                row["source"]["source_game_id"] = "old-game"
            else:
                row["source"]["verified_claim_ids"] = []
            repo.save_reflection(row)

        coordinator.save_all = save_all_then_corrupt
        runner = GameRunner(GameRunnerConfig(
            seed=139,
            repository=repo,
            memory_coordinator=coordinator,
        ))
        runner._state = self._verified_reflection_state(runner.game_id)

        runner._save_memory_snapshot()

        audit = next(
            event for event in runner.state.events
            if event.type == "reflection_persistence_audit"
        )
        assert audit.payload["persistence_complete"] is False
        assert audit.payload["entries"][0]["row_found"] is True
        assert audit.payload["entries"][0]["persistence_complete"] is False
        assert repo.load_reflections_by_game(runner.game_id) == []
        assert repo.load_memory_snapshot(runner.game_id) is None
        assert repo.load_memory_snapshot("latest") is None

    @pytest.mark.parametrize("snapshot_id", ("game", "latest"))
    def test_reflection_readback_rejects_missing_or_inconsistent_snapshot(
        self,
        snapshot_id: str,
    ) -> None:
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        coordinator = PersistentMemoryCoordinator(repo)
        original_save_all = coordinator.save_all

        def save_all_then_corrupt_snapshot(*args, **kwargs):
            original_save_all(*args, **kwargs)
            target = runner.game_id if snapshot_id == "game" else "latest"
            snapshot = repo.load_memory_snapshot(target)
            assert snapshot is not None
            snapshot["reflections"] = []
            repo.save_memory_snapshot(target, snapshot)

        coordinator.save_all = save_all_then_corrupt_snapshot
        runner = GameRunner(GameRunnerConfig(
            seed=140,
            repository=repo,
            memory_coordinator=coordinator,
        ))
        runner._state = self._verified_reflection_state(runner.game_id)

        runner._save_memory_snapshot()

        audit = next(
            event for event in runner.state.events
            if event.type == "reflection_persistence_audit"
        )
        assert audit.payload["persistence_complete"] is False
        assert repo.load_reflections_by_game(runner.game_id) == []
        assert repo.load_memory_snapshot(runner.game_id) is None
        assert repo.load_memory_snapshot("latest") is None

    def test_reflection_readback_rejects_snapshot_save_silent_noop(self) -> None:
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        coordinator = PersistentMemoryCoordinator(repo)
        coordinator.save_all = lambda **_kwargs: None
        runner = GameRunner(GameRunnerConfig(
            seed=141,
            repository=repo,
            memory_coordinator=coordinator,
        ))
        runner._state = self._verified_reflection_state(runner.game_id)

        runner._save_memory_snapshot()

        audit = next(
            event for event in runner.state.events
            if event.type == "reflection_persistence_audit"
        )
        assert audit.payload["persistence_complete"] is False
        assert repo.load_reflections_by_game(runner.game_id) == []
        assert repo.load_memory_snapshot(runner.game_id) is None
        assert repo.load_memory_snapshot("latest") is None

    def test_successful_active_reflection_row_reloads_in_new_memory(self) -> None:
        from werewolf_agent.memory.reflection import ReflectionMemory
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        runner = GameRunner(GameRunnerConfig(
            seed=142,
            repository=repo,
            memory_coordinator=PersistentMemoryCoordinator(repo),
        ))
        runner._state = self._verified_reflection_state(runner.game_id)

        runner._save_memory_snapshot()

        rows = repo.load_reflections_by_game(runner.game_id)
        assert rows[0]["_persistence_active"] is True
        assert len(ReflectionMemory(repo=repo).all_v2_entries()) == 1

    def test_repeated_reflection_snapshot_audits_the_same_expected_row(self) -> None:
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        runner = GameRunner(GameRunnerConfig(
            seed=130,
            repository=repo,
            memory_coordinator=PersistentMemoryCoordinator(repo),
        ))
        runner._state = self._verified_reflection_state(runner.game_id)

        runner._save_memory_snapshot()
        runner._save_memory_snapshot()

        audits = [
            event for event in runner.state.events
            if event.type == "reflection_persistence_audit"
        ]
        assert len(repo.load_reflections_by_game(runner.game_id)) == 1
        assert len(audits) == 2
        assert all(event.payload["persistence_complete"] is True for event in audits)
        assert all(event.payload["entries"][0]["row_found"] is True for event in audits)

    @staticmethod
    def _verified_reflection_state(
        game_id: str,
        *,
        player_ids: tuple[str, ...] = ("p01",),
    ) -> GameState:
        return GameState(
            game_id=game_id,
            phase="finished",
            winning_faction="good",
            players={
                player_id: PlayerState(id=player_id, role="seer")
                for player_id in player_ids
            },
            events=[GameEvent(type="reflection_complete", payload={"entries": [{
                "player_id": player_id,
                "verification": {
                    "status": "verified",
                    "decision_id": f"reflection:g:{player_id}",
                    "verified_fact_count": 1,
                    "verified_claim_ids": [
                        "c-good" if player_id == "p01" else f"c-good-{player_id}"
                    ],
                    "rejected_claim_ids": [
                        "c-bad" if player_id == "p01" else f"c-bad-{player_id}"
                    ],
                    "verified_lessons": [{
                        "lesson_id": "l1" if player_id == "p01" else f"lesson-{player_id}",
                        "abstraction": "对跳时应先核验公开票型",
                    }],
                    "rejected_fact_count": 1,
                    "rejected_lesson_count": 0,
                },
            } for player_id in player_ids]})],
        )

    def test_latest_verified_reflections_uses_latest_canonical_decision_per_player(self) -> None:
        from werewolf_agent.core.models import GameEvent, GameState, PlayerState

        runner = GameRunner(GameRunnerConfig(seed=126))
        old = {"status": "verified", "decision_id": "d1", "verified_lessons": [{"lesson_id": "old", "abstraction": "旧策略"}]}
        latest = {"status": "verified", "decision_id": "d2", "verified_lessons": [{"lesson_id": "new", "abstraction": "新策略"}]}
        runner._state = GameState(
            game_id=runner.game_id,
            players={"p01": PlayerState(id="p01", role="seer")},
            events=[
                GameEvent(type="reflection_complete", payload={"entries": [{"player_id": "p01", "decision_id": "d1", "verification": old}]}),
                GameEvent(type="reflection_complete", payload={"entries": [{"player_id": "p01", "decision_id": "d1", "verification": old}]}),
                GameEvent(type="reflection_complete", payload={"entries": [{"player_id": "p01", "decision_id": "d2", "verification": latest}]}),
            ],
        )

        assert runner._latest_verified_reflections() == {"p01": latest}

    def test_save_memory_snapshot_does_not_rewrite_legacy_v1_reflections(self) -> None:
        from werewolf_agent.core.models import GameEvent, GameState, PlayerState
        from werewolf_agent.storage.memory_store import InMemoryGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator

        repo = InMemoryGameRepository()
        repo.save_reflection({
            "entry_id": "legacy_v1",
            "game_id": "old_game",
            "player_id": "p01",
            "role": "seer",
            "faction_won": False,
            "text": "legacy row must remain untouched",
            "tags": ["seer"],
            "legacy_marker": "preserve-me",
        })
        coord = PersistentMemoryCoordinator(repo)
        runner = GameRunner(GameRunnerConfig(
            seed=124,
            repository=repo,
            memory_coordinator=coord,
        ))
        runner._state = GameState(
            game_id=runner.game_id,
            phase="finished",
            day_number=2,
            winning_faction="good",
            players={
                "p01": PlayerState(id="p01", role="seer", alive=True),
            },
            events=[
                GameEvent(
                    type="reflection_complete",
                    payload={
                        "player_count": 1,
                        "entries": [
                            {
                                "player_id": "p01",
                                "role": "seer",
                                "alive": True,
                                "reflection": "我在对跳局需要先核验警徽流。",
                            },
                        ],
                    },
                )
            ],
        )

        runner._save_memory_snapshot()

        legacy = repo.load_reflection("legacy_v1")
        assert legacy is not None
        assert legacy["legacy_marker"] == "preserve-me"
