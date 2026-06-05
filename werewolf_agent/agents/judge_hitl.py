"""Judge HITL (Human-in-the-Loop) Interface — Layer 4.

The judge becomes the sole entry point for human-system dialogue:
pause/resume/inspect/inject — all through the judge.

All HITL interactions are recorded as GameEvent(type="judge_hitl_interaction")
for full audit trail and replay compatibility.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable

from werewolf_agent.core.models import GameState, GameEvent


# ---------------------------------------------------------------------------
# HITL State Machine
# ---------------------------------------------------------------------------

class HITLState(str, Enum):
    RUNNING = "running"
    WAITING_BEFORE = "waiting_before"
    WAITING_AFTER = "waiting_after"
    PAUSED_USER = "paused_user"
    STOPPED = "stopped"


@dataclass
class HITLCommand:
    command: str
    args: list[str] = field(default_factory=list)
    timestamp: float = 0.0

    @classmethod
    def parse(cls, raw: str) -> HITLCommand:
        parts = raw.strip().split()
        if not parts:
            return cls(command="")
        return cls(
            command=parts[0].lower(),
            args=parts[1:] if len(parts) > 1 else [],
            timestamp=time.time(),
        )


# ---------------------------------------------------------------------------
# Protected fields — HITL commands must never mutate these
# ---------------------------------------------------------------------------

_PROTECTED_TOP_KEYS = frozenset({
    "players",   # player state (roles, alive, etc.)
    "deaths",    # death records
    "votes",     # resolved votes
    "phase",     # game phase (must follow normal flow)
    "winning_faction",  # victory result
    "hybrid_result",    # hybrid binding result
})

# Individual player protected fields — HITL inject must not touch
_PROTECTED_PLAYER_KEYS = frozenset({
    "role", "alive", "faction", "vote_enabled",
    "revealed_idiot", "badge_eligible",
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

        hitl = JudgeHITLInterface(judge_agent)
        runner = GameRunner(config, hitl_interface=hitl)

        # During game execution:
        if hitl.should_pause("death_announce", "after"):
            cmd = hitl.wait_for_human(timeout=300)
            if cmd:
                gs = hitl.handle_command(cmd, gs)
    """

    def __init__(
        self,
        judge_agent: Any = None,
        event_callback: Callable[[HITLCommand, str], Any] | None = None,
        auto_pause_phases: set[str] | None = None,
        pause_timeout: float = 300.0,
    ) -> None:
        self._judge_agent = judge_agent
        self._event_callback = event_callback
        self._state: HITLState = HITLState.RUNNING
        self._auto_pause_after: set[str] = auto_pause_phases or set()
        self._pause_timeout = pause_timeout
        self._pending_command: HITLCommand | None = None
        self._pause_started: float = 0.0
        # J-6: threading.Event used as a signal primitive so any thread
        # blocked in wait_for_human() can be woken up by send_command().
        self._command_event: threading.Event = threading.Event()
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
        For the simulation harness, it checks the pending command set via
        ``send_command()`` and returns immediately.
        """
        effective_timeout = timeout or self._pause_timeout
        self._pause_started = time.time()

        if self._pending_command is not None:
            cmd = self._pending_command
            self._pending_command = None
            return cmd

        # Non-blocking for simulation: return None to indicate timeout/no-input
        elapsed = time.time() - self._pause_started
        if elapsed >= effective_timeout:
            self._state = HITLState.RUNNING
            self._log_event("auto_resume", {"reason": "timeout", "elapsed": elapsed})
            return None
        return None  # Would block in interactive mode

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
        self._log_event("paused", {"by_user": by_user})

    def resume(self, steps: int = 0) -> None:
        """Resume execution. If steps > 0, auto-pause after N steps."""
        if self._state == HITLState.STOPPED:
            return
        self._state = HITLState.RUNNING
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
    ) -> dict[str, Any]:
        """Parse and execute a HITL command. Returns a result dict with at
        least a ``response`` key. May include ``game_state`` if mutated.
        """
        command = cmd.command
        args = cmd.args

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
            p = gs.players[pid]
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
        """Show recent vote events from the event log."""
        vote_events = [e for e in gs.events if e.type in ("vote", "sheriff_vote")]
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
        """
        if len(args) < 1:
            return {"response": "用法: inject_event <type> [key=value ...]"}
        event_type = args[0]
        # Validate event_type — reject empty, overlong, or system-reserved patterns
        if not event_type or len(event_type) > 64:
            return {"response": f"拒绝: 事件类型无效（空或过长: {len(event_type)}字符）"}
        if event_type.startswith("_"):
            return {"response": "拒绝: 事件类型不能以下划线开头（保留给内部事件）"}
        if event_type in ("judge_hitl_interaction", "judge_broadcast"):
            return {"response": f"拒绝: '{event_type}' 是系统保留事件类型"}
        # Parse key=value pairs
        payload: dict[str, Any] = {}
        for kv in args[1:]:
            if "=" in kv:
                k, v = kv.split("=", 1)
                payload[k] = v
        # Validate — reject protected field mutations (case-insensitive)
        protected_lower = {k.lower() for k in _PROTECTED_TOP_KEYS}
        for key in list(payload.keys()):
            if key.lower() in protected_lower:
                return {"response": f"拒绝: '{key}' 是受保护字段，不能通过 inject_event 修改。"}
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
