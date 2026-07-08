"""Layer 4: Judge HITL interface tests.

Coverage: pause/resume lifecycle, command parsing, protected-field enforcement,
event sourcing, integration with GameRunner.
"""

from __future__ import annotations

import threading
import time

from werewolf_agent.core.models import GameState, GameEvent, PlayerState
from werewolf_agent.agents.judge_hitl import (
    HITLState,
    HITLCommand,
    HITLRole,
    JudgeHITLInterface,
    _PROTECTED_TOP_KEYS,
    _PROTECTED_PLAYER_KEYS,
)


def test_hitl_command_is_split_from_judge_hitl_facade() -> None:
    from werewolf_agent.agents import judge_hitl, judge_hitl_commands

    assert judge_hitl.HITLCommand is judge_hitl_commands.HITLCommand


def test_hitl_guard_constants_are_split_from_judge_hitl_facade() -> None:
    from werewolf_agent.agents import judge_hitl, judge_hitl_guards

    assert judge_hitl._PROTECTED_TOP_KEYS is judge_hitl_guards.PROTECTED_TOP_KEYS
    assert judge_hitl._PROTECTED_PLAYER_KEYS is judge_hitl_guards.PROTECTED_PLAYER_KEYS


def _make_gs(**kwargs) -> GameState:
    return GameState(
        game_id="test_hitl",
        phase=kwargs.get("phase", "day"),
        day_number=kwargs.get("day_number", 2),
        night_number=kwargs.get("night_number", 1),
        players=kwargs.get("players", {
            f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
            for i in range(1, 13)
        }),
        sheriff_id=kwargs.get("sheriff_id"),
        events=kwargs.get("events", []),
    )


class TestHITLCommand:
    def test_parse_pause(self):
        cmd = HITLCommand.parse("pause")
        assert cmd.command == "pause"
        assert cmd.args == []

    def test_parse_resume_with_steps(self):
        cmd = HITLCommand.parse("resume 5")
        assert cmd.command == "resume"
        assert cmd.args == ["5"]

    def test_parse_inspect_target(self):
        cmd = HITLCommand.parse("inspect p03")
        assert cmd.command == "inspect"
        assert cmd.args == ["p03"]

    def test_parse_empty(self):
        cmd = HITLCommand.parse("")
        assert cmd.command == ""

    def test_judge_agent_param_removed(self):
        """J-10: JudgeHITLInterface no longer takes/keeps a judge_agent reference."""
        import inspect
        sig = inspect.signature(JudgeHITLInterface.__init__)
        assert "judge_agent" not in sig.parameters, (
            "J-10: judge_agent must be removed (unused storage)"
        )


class TestShowVotesResolved:
    """J-12: `_cmd_show_votes` must surface `vote_resolved` and
    `sheriff_vote_resolved` events, not just the bare `vote` /
    `sheriff_vote` types. The runtime emits `vote_resolved` for
    day votes and (per the design) `sheriff_vote_resolved` for
    sheriff elections — both are the actual public record the
    human asks to inspect with `show_votes`.
    """

    def test_show_votes_returns_resolved_votes(self) -> None:
        """J-12: a GameState carrying a `vote_resolved` event must
        be reflected in `show_votes` output."""
        hitl = JudgeHITLInterface()
        gs = _make_gs(
            events=[
                GameEvent(type="vote_resolved", payload={
                    "tally": {"p03": 5.0, "p07": 3.0},
                    "exiled": "p03",
                }),
            ],
        )
        result = hitl.handle_command(HITLCommand.parse("show_votes"), gs)
        assert "暂无投票记录" not in result["response"], (
            f"J-12: vote_resolved event must surface in show_votes; "
            f"got: {result['response']!r}"
        )
        assert "vote_resolved" in result["response"]
        assert "p03" in result["response"]

    def test_show_votes_returns_sheriff_vote_resolved(self) -> None:
        """J-12: a `sheriff_vote_resolved` event must also surface."""
        hitl = JudgeHITLInterface()
        gs = _make_gs(
            events=[
                GameEvent(type="sheriff_vote_resolved", payload={
                    "tally": {"p05": 4.0, "p08": 2.0},
                    "elected": "p05",
                }),
            ],
        )
        result = hitl.handle_command(HITLCommand.parse("show_votes"), gs)
        assert "sheriff_vote_resolved" in result["response"], (
            f"J-12: sheriff_vote_resolved must surface; got: {result['response']!r}"
        )
        assert "p05" in result["response"]

    def test_show_votes_returns_legacy_vote_type(self) -> None:
        """J-12 regression: legacy `vote` / `sheriff_vote` events
        must still surface — the fix is additive."""
        hitl = JudgeHITLInterface()
        gs = _make_gs(
            events=[
                GameEvent(type="vote", payload={"voter": "p01", "target": "p07"}),
            ],
        )
        result = hitl.handle_command(HITLCommand.parse("show_votes"), gs)
        assert "vote" in result["response"]
        assert "p07" in result["response"]


class TestShouldPauseDirectionConsistency:
    """J-14: lock in the `should_pause` `direction` parameter contract.

    The runtime call sites in ``runtime/nodes/{day,night}.py`` pass
    ``direction="after"`` explicitly, and the default value of
    ``should_pause.direction`` is also ``"after"``. This is the
    uniform contract — the test pins it so future drift gets caught
    at unit-test time rather than at the integration level.
    """

    def test_should_pause_direction_default_is_after(self) -> None:
        """J-14: default ``direction`` argument is the string ``"after"``."""
        import inspect
        sig = inspect.signature(JudgeHITLInterface.should_pause)
        assert "direction" in sig.parameters
        default = sig.parameters["direction"].default
        assert default == "after", (
            f"J-14: should_pause.direction default must be 'after' to "
            f"match the runtime call-site convention; got {default!r}"
        )

    def test_should_pause_direction_keyword_callable(self) -> None:
        """J-14: callers can pass direction as a keyword and the
        auto-pause branch honours it."""
        hitl = JudgeHITLInterface(auto_pause_phases={"announce_deaths"})
        assert hitl.should_pause("announce_deaths", direction="after") is True
        # Non-trigger phase: no pause
        assert hitl.should_pause("free_discussion", direction="after") is False

    def test_should_pause_direction_uniform_across_callers(self) -> None:
        """J-14: scan the codebase for ``should_pause`` / ``_hitl_checkpoint``
        calls and assert they all use ``direction="after"`` (the documented
        value). This is the cross-module consistency lock.

        Implementation note: we only flag calls where a *string literal*
        is passed as the direction value. Internal forwarding calls
        (e.g. ``_hitl_checkpoint`` -> ``should_pause(phase, direction)``)
        are allowed to pass the local variable as-is — that's the
        intended pass-through. Function definitions are also skipped.
        """
        import re
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        # Acceptable:  direction="after"   or  ,"after"  as positional literal.
        ok_positional = re.compile(r""",\s*['"]after['"]""")
        ok_keyword = re.compile(r"""direction\s*=\s*['"]after['"]""")
        offenders: list[str] = []
        for py in (root / "werewolf_agent" / "runtime" / "nodes").rglob("*.py"):
            src = py.read_text(encoding="utf-8")
            lines = src.splitlines()
            for m in re.finditer(r"(should_pause|_hitl_checkpoint)\([^)]*\)", src):
                snippet = m.group(0)
                line_no = src.count("\n", 0, m.start()) + 1
                line = lines[line_no - 1]
                # Skip function definitions.
                if line.lstrip().startswith(("def ", "async def ")):
                    continue
                # Match either a positional "after" or a keyword "after".
                if ok_positional.search(snippet) or ok_keyword.search(snippet):
                    continue
                # Plain forwarding (no string literal at all) is allowed
                # when the call site just forwards the parameter.
                if '"' not in snippet and "'" not in snippet:
                    continue
                offenders.append(f"{py}:{line_no}: {snippet}")
        assert not offenders, (
            "J-14: every should_pause / _hitl_checkpoint call site must "
            f"use direction='after'; offenders: {offenders}"
        )


class TestJudgeHITLInterface:
    def test_initial_state_running(self):
        hitl = JudgeHITLInterface()
        assert hitl.state == HITLState.RUNNING
        assert not hitl.is_paused

    def test_pause_resume_cycle(self):
        hitl = JudgeHITLInterface()
        hitl.pause()
        assert hitl.is_paused
        assert hitl.state == HITLState.PAUSED_USER
        hitl.resume()
        assert hitl.is_running

    def test_stop(self):
        hitl = JudgeHITLInterface()
        hitl.stop()
        assert hitl.is_stopped
        assert hitl.state == HITLState.STOPPED

    def test_auto_pause_on_trigger_phase(self):
        hitl = JudgeHITLInterface(auto_pause_phases={"death_announce"})
        assert hitl.state == HITLState.RUNNING
        should = hitl.should_pause("free_discussion", "after")
        assert not should
        should = hitl.should_pause("death_announce", "after")
        assert should
        assert hitl.state == HITLState.WAITING_AFTER

    def test_resume_with_steps(self):
        hitl = JudgeHITLInterface()
        hitl.resume(steps=3)
        assert hitl.is_running
        # First 2 steps should not pause
        assert not hitl.should_pause("any", "after")
        assert not hitl.should_pause("any", "after")
        # 3rd step should trigger pause
        assert hitl.should_pause("any", "after")
        assert hitl.is_paused

    def test_send_command_during_pause(self):
        hitl = JudgeHITLInterface()
        hitl.pause()
        hitl.send_command("show_phase")
        gs = _make_gs()
        cmd = HITLCommand.parse("show_phase")
        result = hitl.handle_command(cmd, gs)
        assert "D2" in result["response"] or "2" in result["response"]

    def test_hitl_interactions_are_event_sourced(self):
        hitl = JudgeHITLInterface()
        hitl.pause()
        hitl.resume()
        hitl.stop()
        events = hitl.flush_events()
        assert len(events) >= 3
        types = [e.type for e in events]
        assert all(t == "judge_hitl_interaction" for t in types)
        actions = [e.payload["action"] for e in events]
        assert "paused" in actions
        assert "resumed" in actions
        assert "stopped" in actions

    def test_handle_unknown_command(self):
        hitl = JudgeHITLInterface()
        gs = _make_gs()
        result = hitl.handle_command(HITLCommand.parse("foobar"), gs)
        assert "未知命令" in result["response"]

    def test_inject_event_disallowed_for_protected_keys(self):
        hitl = JudgeHITLInterface()
        gs = _make_gs()
        for key in _PROTECTED_TOP_KEYS:
            cmd_raw = f"inject_event test {key}=fake_value"
            result = hitl.handle_command(HITLCommand.parse(cmd_raw), gs)
            assert "拒绝" in result["response"] or "受保护" in result["response"], \
                f"Protected key '{key}' should be rejected"

    def test_inject_event_allowed_for_safe_keys(self):
        hitl = JudgeHITLInterface()
        gs = _make_gs()
        result = hitl.handle_command(
            HITLCommand.parse("inject_event custom_event safe_key=hello"),
            gs,
        )
        assert "已注入" in result["response"]
        assert "game_state" in result

    def test_show_alive_lists_players(self):
        hitl = JudgeHITLInterface()
        gs = _make_gs()
        result = hitl.handle_command(HITLCommand.parse("show_alive"), gs)
        assert "p01" in result["response"]

    def test_show_roles_debug_view(self):
        hitl = JudgeHITLInterface()
        gs = _make_gs()
        result = hitl.handle_command(HITLCommand.parse("show_roles"), gs)
        assert "villager" in result["response"]

    def test_status_summary(self):
        hitl = JudgeHITLInterface()
        gs = _make_gs()
        result = hitl.handle_command(HITLCommand.parse("status"), gs)
        assert "test_hitl" in result["response"]
        assert "D2" in result["response"] or "2" in result["response"]

    def test_help_has_all_commands(self):
        hitl = JudgeHITLInterface()
        gs = _make_gs()
        result = hitl.handle_command(HITLCommand.parse("help"), gs)
        for cmd in ("pause", "resume", "inspect", "show_phase", "show_alive",
                     "show_roles", "status"):
            assert cmd in result["response"], f"'{cmd}' missing from help"

    # ------------------------------------------------------------------
    # J-6: send_command must signal a threading.Event
    # ------------------------------------------------------------------

    def test_send_command_signals_event(self):
        """J-6: send_command must set the threading.Event so any thread
        blocked in wait_for_human can be woken up.

        The old design used a bare ``self._pending_command`` flag, which
        had no way to wake a thread that was already blocked in
        ``wait_for_human``. Using ``threading.Event`` provides a proper
        signal primitive: the waiter calls ``event.wait()`` and the
        setter calls ``event.set()``.
        """
        hitl = JudgeHITLInterface()
        # New field must exist
        assert hasattr(hitl, "_command_event"), \
            "JudgeHITLInterface must expose a _command_event threading.Event"
        assert isinstance(hitl._command_event, threading.Event), \
            f"_command_event must be a threading.Event, got {type(hitl._command_event)}"
        # Initially the event is clear
        assert not hitl._command_event.is_set()
        # send_command must set the event
        hitl.send_command("show_phase")
        assert hitl._command_event.is_set(), \
            "send_command must set the threading.Event"

    # ------------------------------------------------------------------
    # J-1: wait_for_human actually blocks until send_command fires
    # ------------------------------------------------------------------

    def test_wait_for_human_blocks_until_command(self):
        """J-1: wait_for_human() must block (not return None immediately).

        Spawn a thread that calls wait_for_human with a long timeout. After
        ~100ms, the main thread invokes send_command. The wait must return
        shortly after that (well under 200ms), proving the event was
        signalled and the waiter woke up.
        """
        hitl = JudgeHITLInterface()
        hitl.pause()  # ensure state is PAUSED_USER

        result_box: dict[str, object] = {}

        def waiter() -> None:
            result_box["cmd"] = hitl.wait_for_human(timeout=5.0)

        t = threading.Thread(target=waiter)
        t.start()
        # Sleep enough for the waiter to be inside wait_for_human
        time.sleep(0.1)
        sent_at = time.time()
        hitl.send_command("show_phase")
        t.join(timeout=2.0)
        elapsed = time.time() - sent_at

        assert t.is_alive() is False, "wait_for_human did not return"
        assert result_box.get("cmd") is not None
        assert result_box["cmd"].command == "show_phase"  # type: ignore[union-attr]
        assert elapsed < 0.2, f"wait_for_human returned too late: {elapsed}s"

    # ------------------------------------------------------------------
    # J-2: _state resets to RUNNING after a queued command is consumed
    # ------------------------------------------------------------------

    def test_state_resets_to_running_after_command(self):
        """J-2: After wait_for_human() consumes a queued command, the state
        should drop back to RUNNING — unless the user explicitly paused.

        Pattern: pause() -> send_command() -> wait_for_human() returns ->
        state must be RUNNING.
        """
        hitl = JudgeHITLInterface()
        hitl.pause()
        assert hitl.state == HITLState.PAUSED_USER

        hitl.send_command("show_phase")
        cmd = hitl.wait_for_human(timeout=1.0)
        assert cmd is not None
        # Consuming the command should reset state back to RUNNING
        assert hitl.state == HITLState.RUNNING
        assert hitl.is_running is True
        assert hitl.is_paused is False

    # ------------------------------------------------------------------
    # J-4: caller_role gates sensitive commands (show_roles/show_votes/inject_event)
    # ------------------------------------------------------------------

    def test_show_roles_requires_moderator_role(self):
        """J-4: show_roles must be denied for non-moderator callers.

        Privileged commands leak hidden identities or vote records, so they
        must be gated to MODERATOR/JUDGE roles. SPECTATOR (e.g. an audience
        dashboard) must NOT be allowed to invoke them.
        """
        hitl = JudgeHITLInterface()
        gs = _make_gs()
        # SPECTATOR is denied
        result = hitl.handle_command(
            HITLCommand.parse("show_roles"), gs, caller_role=HITLRole.SPECTATOR
        )
        assert "拒绝" in result["response"] or "denied" in result["response"].lower(), \
            f"SPECTATOR should be denied show_roles, got: {result['response']!r}"
        # No identity data should leak
        assert "villager" not in result["response"], \
            f"Identity leaked to SPECTATOR: {result['response']!r}"
        # MODERATOR still works
        result = hitl.handle_command(
            HITLCommand.parse("show_roles"), gs, caller_role=HITLRole.MODERATOR
        )
        assert "villager" in result["response"]

    # ------------------------------------------------------------------
    # J-5: inject_event must reject nested sensitive keys + non-custom types + oversize payloads
    # ------------------------------------------------------------------

    def test_inject_event_rejects_nested_sensitive_keys(self):
        """J-5: inject_event validation must recurse into nested values.

        A protected top-level key (e.g. 'players') hidden inside a nested
        dict must be rejected — otherwise an attacker can smuggle a
        protected-field mutation past the top-level check. The fix walks
        dicts and lists recursively.
        """
        import json
        hitl = JudgeHITLInterface()
        gs = _make_gs()
        # Nested payload: protected 'players' key is hidden one level deep.
        # JSON must be compact (no spaces) since the HITL parser splits on
        # whitespace — using commas/colons only keeps it a single token.
        nested_value = json.dumps({"players": "fake", "innocent": "ok"}, separators=(",", ":"))
        cmd_raw = f'inject_event custom_x data={nested_value}'
        result = hitl.handle_command(HITLCommand.parse(cmd_raw), gs)
        assert "拒绝" in result["response"], \
            f"Nested protected key should be rejected, got: {result['response']!r}"
        # The protected key should be named in the rejection so the user
        # understands what triggered it.
        assert "players" in result["response"], \
            f"Rejection should name 'players', got: {result['response']!r}"
        # State must NOT have been mutated
        assert "game_state" not in result, \
            f"game_state must not be returned on rejection, got keys: {list(result.keys())}"
        # Sanity: a flat safe payload still works
        result_ok = hitl.handle_command(
            HITLCommand.parse("inject_event custom_x benign=ok"),
            gs,
        )
        assert "已注入" in result_ok["response"]


class TestHITLGameRunnerIntegration:
    def test_game_runner_creates_hitl_when_enabled(self):
        from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig
        config = GameRunnerConfig(
            seed=42,
            use_agent_registry=True,
            model_config_path="config/models.yaml",
            judge_hitl_enabled=True,
            judge_hitl_auto_pause_triggers=["death_announce"],
        )
        runner = GameRunner(config)
        assert runner.hitl_interface is not None
        assert runner.hitl_interface.state == HITLState.RUNNING

    def test_game_runner_no_hitl_by_default(self):
        from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig
        config = GameRunnerConfig(seed=42, use_agent_registry=False)
        runner = GameRunner(config)
        assert runner.hitl_interface is None
        assert runner.pause() is None
        assert runner.resume() is None
        assert runner.send_command("status") is None

    def test_game_runner_pause_resume(self):
        from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig
        config = GameRunnerConfig(
            seed=42,
            use_agent_registry=True,
            model_config_path="config/models.yaml",
            judge_hitl_enabled=True,
        )
        runner = GameRunner(config)
        resp = runner.pause()
        assert resp is not None
        assert "暂停" in resp
        assert runner.hitl_interface.is_paused
        resp = runner.resume()
        assert resp is not None
        assert not runner.hitl_interface.is_paused

    def test_game_runner_send_status_command(self):
        from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig
        config = GameRunnerConfig(
            seed=42,
            use_agent_registry=True,
            model_config_path="config/models.yaml",
            judge_hitl_enabled=True,
        )
        runner = GameRunner(config)
        runner.pause()
        resp = runner.send_command("status")
        assert resp is not None
        assert "test" not in resp or "D" in resp or "N" in resp or "游戏" in resp
