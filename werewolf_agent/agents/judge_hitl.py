# -*- coding: utf-8 -*-
"""
功能描述：**：管理游戏暂停-恢复生命周期，解析人工命令，委托安全边界校验，记录所有交互为审计事件。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-08
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from enum import Enum
from typing import Any, Callable

from werewolf_agent.agents.judge_hitl_commands import HITLCommand
from werewolf_agent.agents import judge_hitl_guards
from werewolf_agent.core.models import GameState, GameEvent

_PROTECTED_TOP_KEYS = judge_hitl_guards.PROTECTED_TOP_KEYS
_PROTECTED_PLAYER_KEYS = judge_hitl_guards.PROTECTED_PLAYER_KEYS


# ---------------------------------------------------------------------------
# HITL State Machine
# ---------------------------------------------------------------------------

class HITLState(str, Enum):
    RUNNING = "running"
    WAITING_BEFORE = "waiting_before"
    WAITING_AFTER = "waiting_after"
    PAUSED_USER = "paused_user"
    STOPPED = "stopped"


class HITLRole(str, Enum):
    """Caller role for HITL command authorization (J-4).

    Privileged commands (show_roles / show_votes / inject_event) leak
    hidden identities or mutate game state, so they are gated to
    MODERATOR / JUDGE roles. SPECTATOR (e.g. an audience dashboard) is
    denied.
    """

    MODERATOR = "moderator"
    JUDGE = "judge"
    SPECTATOR = "spectator"


# J-4: commands gated to MODERATOR / JUDGE (anything else, including
# SPECTATOR, is denied with a refused response).
_PRIVILEGED_COMMANDS = frozenset({
    "show_roles",
    "show_votes",
    "inject_event",
})


# ---------------------------------------------------------------------------
# Judge HITL Interface
# ---------------------------------------------------------------------------

class JudgeHITLInterface:
    """Judge as the sole human-system bridge.

    Manages pause/resume lifecycle, parses HITL commands, enforces
    protected-field boundaries, and records all interactions as
    auditable GameEvents.

    Usage::

        hitl = JudgeHITLInterface()
        runner = GameRunner(config, hitl_interface=hitl)

        # During game execution:
        if hitl.should_pause("death_announce", "after"):
            cmd = hitl.wait_for_human(timeout=300)
            if cmd:
                gs = hitl.handle_command(cmd, gs)
    """

    def __init__(
        self,
        event_callback: Callable[[HITLCommand, str], Any] | None = None,
        auto_pause_phases: set[str] | None = None,
        pause_timeout: float = 300.0,
    ) -> None:
        self._event_callback = event_callback
        self._state: HITLState = HITLState.RUNNING
        self._auto_pause_after: set[str] = auto_pause_phases or set()
        self._pause_timeout = pause_timeout
        self._pending_command: HITLCommand | None = None
        self._pause_started: float = 0.0
        # J-6: threading.Event used as a signal primitive so any thread
        # blocked in wait_for_human() can be woken up by send_command().
        self._command_event: threading.Event = threading.Event()
        # J-2: track whether user explicitly paused (so we don't auto-resume
        # on every consumed command).
        self._user_paused: bool = False
        # Step counter for "resume N" command
        self._steps_to_run: int = 0
        self._events: list[GameEvent] = []

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def state(self) -> HITLState:
        return self._state

    @property
    def is_paused(self) -> bool:
        return self._state in (
            HITLState.WAITING_BEFORE,
            HITLState.WAITING_AFTER,
            HITLState.PAUSED_USER,
        )

    @property
    def is_running(self) -> bool:
        return self._state == HITLState.RUNNING

    @property
    def is_stopped(self) -> bool:
        return self._state == HITLState.STOPPED

    # ------------------------------------------------------------------
    # Pause / Resume lifecycle
    # ------------------------------------------------------------------

    def should_pause(self, phase: str, direction: str = "after") -> bool:
        """Check whether execution should pause at this phase transition."""
        if self._state == HITLState.STOPPED:
            return True
        # User-paused state takes priority over step counter
        if self._state == HITLState.PAUSED_USER:
            return True
        if self._steps_to_run > 0:
            self._steps_to_run -= 1
            if self._steps_to_run == 0:
                self._state = HITLState.PAUSED_USER
                return True
            return False
        if phase in self._auto_pause_after and direction == "after":
            self._state = HITLState.WAITING_AFTER
            return True
        return False

    def wait_for_human(self, timeout: float | None = None) -> HITLCommand | None:
        """Block waiting for a human command.

        In an interactive setting this would read from stdin or an API queue.
        For the simulation harness, it waits on a threading.Event that is set
        by ``send_command()``. Returns the queued command or ``None`` on
        timeout.
        """
        effective_timeout = timeout if timeout is not None else self._pause_timeout
        self._pause_started = time.time()

        # If a command is already queued, consume it immediately.
        if self._pending_command is not None:
            return self._consume_pending_command()

        # J-1: Block on the event until send_command() signals or timeout
        # elapses. Previously this returned None immediately, so the
        # runner could not actually pause for human input.
        got_signal = self._command_event.wait(timeout=effective_timeout)
        if not got_signal:
            # Timeout — auto-resume back to running (unless user explicitly paused)
            if not self._user_paused:
                self._state = HITLState.RUNNING
            elapsed = time.time() - self._pause_started
            self._log_event("auto_resume", {"reason": "timeout", "elapsed": elapsed})
            return None

        # Signalled — return the queued command (must be present since
        # send_command is the only setter).
        return self._consume_pending_command()

    def _consume_pending_command(self) -> HITLCommand | None:
        """Pop the pending command and reset state appropriately.

        J-2: After a command is consumed (a human actively interacted),
        drop back to RUNNING so the runner advances. The "pause was
        explicitly called" caveat only applies when no command is in the
        queue — i.e. the user paused to inspect, and there is nothing to
        consume.
        """
        cmd = self._pending_command
        self._pending_command = None
        # J-6: clear the event so a future wait_for_human can block again
        self._command_event.clear()
        # J-2: Reset to RUNNING — a consumed command means the user wants
        # the game to continue.
        if self._state != HITLState.STOPPED:
            self._state = HITLState.RUNNING
        return cmd

    def send_command(self, raw: str) -> None:
        """Queue a command from an external source (API, CLI, dashboard).

        J-6: also signals the threading.Event so any thread blocked in
        wait_for_human() wakes immediately.
        """
        self._pending_command = HITLCommand.parse(raw)
        self._command_event.set()

    def pause(self, by_user: bool = True) -> None:
        """Pause execution at the next checkpoint."""
        self._state = HITLState.PAUSED_USER if by_user else HITLState.WAITING_AFTER
        if by_user:
            self._user_paused = True
        self._log_event("paused", {"by_user": by_user})

    def resume(self, steps: int = 0) -> None:
        """Resume execution. If steps > 0, auto-pause after N steps."""
        if self._state == HITLState.STOPPED:
            return
        self._state = HITLState.RUNNING
        self._user_paused = False
        self._steps_to_run = max(0, steps)
        self._log_event("resumed", {"steps": steps})

    def stop(self) -> None:
        """Stop the game entirely."""
        self._state = HITLState.STOPPED
        self._log_event("stopped", {})

    # ------------------------------------------------------------------
    # Command handling
    # ------------------------------------------------------------------

    def handle_command(
        self,
        cmd: HITLCommand,
        game_state: GameState,
        *,
        caller_role: HITLRole | str = HITLRole.MODERATOR,
    ) -> dict[str, Any]:
        """Parse and execute a HITL command. Returns a result dict with at
        least a ``response`` key. May include ``game_state`` if mutated.

        J-4: ``caller_role`` gates privileged commands. The default of
        ``MODERATOR`` preserves backward compatibility for in-process
        callers; external callers (API, CLI, dashboard) must pass
        ``HITLRole.SPECTATOR`` (or any other non-privileged role) to
        actually be denied.
        """
        command = cmd.command
        args = cmd.args

        # J-4: gate privileged commands to MODERATOR / JUDGE.
        if command in _PRIVILEGED_COMMANDS:
            role_value = caller_role.value if isinstance(caller_role, HITLRole) else str(caller_role)
            if role_value not in (HITLRole.MODERATOR.value, HITLRole.JUDGE.value):
                response = (
                    f"拒绝: 命令 '{command}' 需要 MODERATOR 或 JUDGE 角色，"
                    f"当前角色 '{role_value}' 无权访问。"
                )
                self._log_event("command_denied", {
                    "command": command, "caller_role": role_value,
                })
                return {"response": response}

        handlers: dict[str, Callable[[list[str], GameState], dict[str, Any]]] = {
            "pause": self._cmd_pause,
            "resume": self._cmd_resume,
            "inspect": self._cmd_inspect,
            "inject_event": self._cmd_inject_event,
            "show_phase": self._cmd_show_phase,
            "show_alive": self._cmd_show_alive,
            "show_roles": self._cmd_show_roles,
            "show_votes": self._cmd_show_votes,
            "status": self._cmd_status,
            "stop": self._cmd_stop,
            "help": self._cmd_help,
        }

        handler = handlers.get(command)
        if handler is None:
            response = f"未知命令: {command}。输入 /help 查看可用命令。"
            self._log_event("unknown_command", {"command": command})
        else:
            try:
                result = handler(args, game_state)
                response = result.get("response", "命令已执行。")
                self._log_event("command_executed", {
                    "command": command, "args": args, "response": response[:200],
                })
                if "game_state" in result:
                    return result
            except Exception as exc:
                response = f"命令执行失败: {exc}"
                self._log_event("command_failed", {"command": command, "error": str(exc)})

        return {"response": response}

    # ------------------------------------------------------------------
    # Command implementations
    # ------------------------------------------------------------------

    def _cmd_pause(self, args: list[str], gs: GameState) -> dict[str, Any]:
        self.pause()
        return {"response": f"游戏已暂停。当前阶段: {gs.phase}，第{gs.day_number}天/第{gs.night_number}夜。"}

    def _cmd_resume(self, args: list[str], gs: GameState) -> dict[str, Any]:
        steps = int(args[0]) if args else 0
        self.resume(steps)
        if steps:
            return {"response": f"恢复执行，将在{steps}步后再次暂停。"}
        return {"response": "游戏已恢复运行。"}

    def _cmd_stop(self, args: list[str], gs: GameState) -> dict[str, Any]:
        self.stop()
        return {"response": "游戏已终止。"}

    def _cmd_show_phase(self, args: list[str], gs: GameState) -> dict[str, Any]:
        alive_count = sum(1 for p in gs.players.values() if p.alive)
        dead_count = len(gs.players) - alive_count
        sheriff = gs.sheriff_id or "无"
        return {"response": (
            f"阶段: {gs.phase} | 第{gs.day_number}天 / 第{gs.night_number}夜 | "
            f"存活: {alive_count}/{len(gs.players)} | 死亡: {dead_count} | "
            f"警长: {sheriff} | 警徽: {gs.sheriff_badge_state}"
        )}

    def _cmd_show_alive(self, args: list[str], gs: GameState) -> dict[str, Any]:
        alive = [pid for pid, p in gs.players.items() if p.alive]
        lines = [f"存活玩家 ({len(alive)}人):"]
        for pid in alive:
            lines.append(f"  {pid} - 存活" + (" (警长)" if pid == gs.sheriff_id else ""))
        return {"response": "\n".join(lines)}

    def _cmd_show_roles(self, args: list[str], gs: GameState) -> dict[str, Any]:
        """Show role assignments — DEBUG/MODERATOR only."""
        lines = ["身份分配 (调试视图):"]
        for pid in sorted(gs.players.keys()):
            p = gs.players[pid]
            status = "存活" if p.alive else "死亡"
            lines.append(f"  {pid}: {p.role} ({status})")
        return {"response": "\n".join(lines)}

    def _cmd_show_votes(self, args: list[str], gs: GameState) -> dict[str, Any]:
        """Show recent vote events from the event log.

        J-12: filter covers the resolved variants emitted by the runtime
        (``vote_resolved`` for day votes, ``sheriff_vote_resolved`` for
        sheriff elections) plus the legacy bare ``vote`` / ``sheriff_vote``
        types so older traces keep working.
        """
        vote_types = ("vote_resolved", "sheriff_vote_resolved", "vote", "sheriff_vote")
        vote_events = [e for e in gs.events if e.type in vote_types]
        if not vote_events:
            return {"response": "暂无投票记录。"}
        lines = [f"投票记录 ({len(vote_events)}条):"]
        for e in vote_events[-10:]:  # Last 10
            lines.append(f"  [{e.type}] {e.payload}")
        return {"response": "\n".join(lines)}

    def _cmd_inspect(self, args: list[str], gs: GameState) -> dict[str, Any]:
        """Inspect public game state. Target can be phase-level or player-level."""
        target = args[0] if args else "game"
        # Check reserved keywords first (players cannot have these IDs)
        reserved = {"game", "deaths", "events"}
        if target not in reserved and target in gs.players:
            p = gs.players[target]
            return {"response": (
                f"玩家 {target}: "
                f"存活={p.alive}, 有投票权={p.vote_enabled}"
                + (f", 警长" if target == gs.sheriff_id else "")
                + (f", 翻牌白痴" if p.revealed_idiot else "")
            )}
        if target == "game":
            return self._cmd_status(args, gs)
        if target == "deaths":
            deaths = [f"{d.player_id}({d.reason})" for d in gs.deaths]
            return {"response": f"死亡记录: {', '.join(deaths) if deaths else '无'}"}
        if target == "events":
            recent = gs.events[-20:]
            lines = [f"最近事件 ({len(recent)}条):"]
            for e in recent:
                phase = e.payload.get("phase", e.type)
                msg = str(e.payload.get("message", ""))[:80]
                lines.append(f"  [{phase}] {msg}")
            return {"response": "\n".join(lines)}
        return {"response": f"未知检查目标: {target}。可用: game, events, deaths, <player_id>"}

    def _cmd_inject_event(self, args: list[str], gs: GameState) -> dict[str, Any]:
        """Inject a custom GameEvent. Protected fields are validated and rejected.

        Usage: inject_event <event_type> key1=value1 key2=value2 ...

        J-5 hardening:
          1. Whitelist event types — only ``custom_*`` is allowed. This
             blocks injection of reserved/semantic event types such as
             ``vote_resolved`` or ``phase_changed`` that the rule engine
             uses to mutate state.
          2. Recursive nested key check — protected top-level keys
             (``players``, ``deaths``, ``votes``, ``phase``,
             ``winning_faction``, ``hybrid_result``) AND protected
             per-player keys (``role``, ``alive``, ``faction``,
             ``vote_enabled``, ``revealed_idiot``, ``badge_eligible``)
             are rejected anywhere in the value tree (dicts, lists).
          3. Size limit — the serialized payload must not exceed 4KB.
        """
        if len(args) < 1:
            return {"response": "用法: inject_event <type> [key=value ...]"}
        event_type = args[0]

        event_type_error = judge_hitl_guards.validate_event_type(event_type)
        if event_type_error is not None:
            return {"response": event_type_error}

        payload = judge_hitl_guards.parse_payload_tokens(args[1:])
        payload_error = judge_hitl_guards.validate_payload(payload)
        if payload_error is not None:
            return {"response": payload_error}

        # Log the custom event
        event = GameEvent(type=event_type, payload=payload)
        gs = replace(gs, events=gs.events + [event])
        self._log_event("event_injected", {"event_type": event_type, "payload": payload})
        return {
            "response": f"事件 '{event_type}' 已注入。",
            "game_state": gs,
        }

    def _cmd_status(self, args: list[str], gs: GameState) -> dict[str, Any]:
        """Full game status summary."""
        alive = [pid for pid, p in gs.players.items() if p.alive]
        dead = [pid for pid, p in gs.players.items() if not p.alive]
        lines = [
            f"=== 游戏状态 ===",
            f"Game ID: {gs.game_id}",
            f"规则包: {gs.ruleset_id}",
            f"阶段: {gs.phase} | D{gs.day_number} / N{gs.night_number}",
            f"存活 ({len(alive)}): {', '.join(alive)}",
            f"死亡 ({len(dead)}): {', '.join(dead)}",
            f"警长: {gs.sheriff_id or '无'} | 警徽: {gs.sheriff_badge_state}",
            f"HITL状态: {self._state.value}",
            f"暂停超时: {self._pause_timeout}s",
        ]
        return {"response": "\n".join(lines)}

    def _cmd_help(self, args: list[str], gs: GameState) -> dict[str, Any]:
        return {"response": (
            "可用命令:\n"
            "  pause        - 暂停游戏\n"
            "  resume [N]   - 恢复运行（可选N步后再次暂停）\n"
            "  stop         - 终止游戏\n"
            "  show_phase   - 显示当前阶段\n"
            "  show_alive   - 列出存活玩家\n"
            "  show_roles   - 显示身份分配（调试）\n"
            "  show_votes   - 显示投票记录\n"
            "  inspect [target] - 检查游戏/玩家/死亡/事件\n"
            "  inject_event <type> [k=v ...] - 注入自定义事件\n"
            "  status       - 完整状态摘要\n"
            "  help         - 显示此帮助"
        )}

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------

    def _log_event(self, action: str, detail: dict[str, Any]) -> None:
        """Record a HITL interaction event."""
        event = GameEvent(
            type="judge_hitl_interaction",
            payload={
                "action": action,
                "timestamp": time.time(),
                "state": self._state.value,
                **detail,
            },
        )
        self._events.append(event)

    def flush_events(self) -> list[GameEvent]:
        """Consume and return all accumulated HITL audit events."""
        evts = list(self._events)
        self._events.clear()
        return evts

    def get_events(self) -> list[GameEvent]:
        """Return HITL events without consuming them."""
        return list(self._events)
