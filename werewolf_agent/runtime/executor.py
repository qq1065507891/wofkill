# -*- coding: utf-8 -*-
"""Local runtime execution coordination.
    作者: Mike
    创建日期: 2025-01-15
    修改日期: 2026-07-05
    使用示例: 内部模块，无对外接口
This module owns process-local scheduling concerns: per-game locks,
background thread state, and execution status. It does not adjudicate rules
or mutate game truth directly; all game changes still flow through GameRunner.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import threading
from typing import Any


@dataclass
class RuntimeExecutionResult:
    game_id: str
    success: bool
    status: str
    message: str = ""


@dataclass
class RuntimeExecutionStatus:
    game_id: str
    state: str = "idle"
    step_count: int = 0
    phase: str = ""
    error: str = ""
    started_at: str = ""
    finished_at: str = ""


class LocalRuntimeExecutor:
    """Process-local coordinator for GameRunner execution."""

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._statuses: dict[str, RuntimeExecutionStatus] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._guard = threading.Lock()

    def lock_for(self, game_id: str) -> threading.Lock:
        with self._guard:
            if game_id not in self._locks:
                self._locks[game_id] = threading.Lock()
            return self._locks[game_id]

    def status(self, game_id: str) -> RuntimeExecutionStatus:
        with self._guard:
            status = self._statuses.get(game_id)
            if status is None:
                return RuntimeExecutionStatus(game_id=game_id)
            return RuntimeExecutionStatus(**status.__dict__)

    def try_step(self, game_id: str, runner: Any) -> RuntimeExecutionResult:
        lock = self.lock_for(game_id)
        if not lock.acquire(blocking=False):
            return RuntimeExecutionResult(
                game_id=game_id,
                success=False,
                status="busy",
                message="Game execution is already running",
            )
        try:
            self._set_status(game_id, runner, "running")
            runner.run_step()
            state = "finished" if getattr(runner, "finished", False) else "idle"
            self._set_status(game_id, runner, state, finished=state == "finished")
            return RuntimeExecutionResult(game_id=game_id, success=True, status=state)
        except Exception as exc:
            self._set_status(game_id, runner, "error", error=str(exc), finished=True)
            return RuntimeExecutionResult(
                game_id=game_id,
                success=False,
                status="error",
                message=str(exc),
            )
        finally:
            lock.release()

    def start_background(
        self,
        game_id: str,
        runner: Any,
        *,
        max_steps: int = 500,
    ) -> RuntimeExecutionResult:
        lock = self.lock_for(game_id)
        if not lock.acquire(blocking=False):
            return RuntimeExecutionResult(
                game_id=game_id,
                success=False,
                status="busy",
                message="Game execution is already running",
            )

        self._set_status(game_id, runner, "running")

        def _target() -> None:
            try:
                runner.run(max_steps=max_steps)
                self._set_status(game_id, runner, "finished", finished=True)
            except Exception as exc:
                self._set_status(game_id, runner, "error", error=str(exc), finished=True)
            finally:
                lock.release()

        thread = threading.Thread(target=_target, name=f"werewolf-runtime-{game_id}", daemon=True)
        with self._guard:
            self._threads[game_id] = thread
        thread.start()
        return RuntimeExecutionResult(game_id=game_id, success=True, status="running")

    def wait(self, game_id: str, timeout: float | None = None) -> None:
        thread = self._threads.get(game_id)
        if thread is not None:
            thread.join(timeout=timeout)

    def _set_status(
        self,
        game_id: str,
        runner: Any,
        state: str,
        *,
        error: str = "",
        finished: bool = False,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        existing = self._statuses.get(game_id)
        started_at = existing.started_at if existing and existing.started_at else now
        runner_state = getattr(runner, "state", None)
        phase = getattr(runner_state, "phase", "") if runner_state is not None else ""
        status = RuntimeExecutionStatus(
            game_id=game_id,
            state=state,
            step_count=int(getattr(runner, "step_count", 0)),
            phase=phase,
            error=error,
            started_at=started_at,
            finished_at=now if finished else "",
        )
        with self._guard:
            self._statuses[game_id] = status

    def _cleanup_game(self, game_id: str) -> None:
        """清理已完成的游戏条目，释放内存。"""
        with self._guard:
            status = self._statuses.get(game_id)
            if status and status.state in ("finished", "error"):
                self._statuses.pop(game_id, None)
                self._locks.pop(game_id, None)
                self._threads.pop(game_id, None)
