# -*- coding: utf-8 -*-
"""
协调 GameRunner 的图执行、事件元数据盖章、单步推进和持久化。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-16

使用示例:
    >>> from werewolf_agent.runtime.game_runner_execution import GameRunnerExecutionMixin
"""

from __future__ import annotations

import logging
from dataclasses import replace

from langgraph.errors import GraphRecursionError

from werewolf_agent.core.models import GameState
from werewolf_agent.runtime.event_metadata import stamp_new_events
from werewolf_agent.runtime.game_termination import (
    abort_game,
    finish_game,
    validate_aborted_game,
    write_emergency_abort,
)


logger = logging.getLogger("werewolf_agent.runtime.game_runner")


class GameRunnerExecutionMixin:
    def _process_chunk(self, chunk: dict) -> str | None:
        """处理一个 LangGraph stream chunk，并更新内部状态。"""
        node_name = None
        for name, output in chunk.items():
            node_name = name
            if output is not None and "game_state" in output:
                next_state = output["game_state"]
                stamped_events = stamp_new_events(
                    next_state.game_id,
                    self._state.events,
                    next_state.events,
                )
                self._state = replace(next_state, events=stamped_events)
                self._last_node = node_name
                try:
                    self._cognition_state_manager.update_from_events(self._state)
                except Exception:
                    logger.warning(
                        "Live cognition update failed after node %s",
                        node_name,
                        exc_info=True,
                    )
        return node_name

    def run(self, max_steps: int = 1000) -> GameState:
        """执行完整游戏图直到 END 或达到 max_steps。"""
        if self._finished:
            return self._state

        if self._hitl_interface is not None and self._hitl_interface.is_paused:
            return self._state

        initial = self._build_runtime_state()
        last_phase_snapshot: tuple[str, int, int] | None = None
        stuck_count = 0

        try:
            for chunk in self._graph.stream(
                initial, {"recursion_limit": max_steps}
            ):
                self._step_count += 1
                self._process_chunk(chunk)
                if self._state.phase == "finished":
                    return self._finish_terminal_game()
                snapshot = (self._state.phase, self._state.day_number, self._state.night_number)
                if snapshot == last_phase_snapshot:
                    stuck_count += 1
                    if stuck_count >= 50:
                        logger.warning(
                            "Game stuck detected at step %d: phase=%s day=%d night=%d - forcing finish",
                            self._step_count, self._state.phase,
                            self._state.day_number, self._state.night_number,
                        )
                        return self._abort_terminal_game("step_limit")
                else:
                    stuck_count = 0
                    last_phase_snapshot = snapshot
        except GraphRecursionError as exc:
            logger.warning(
                "Graph recursion limit in run() at step %d (phase=%s): %s",
                self._step_count, self._state.phase,
                exc,
            )
            return self._abort_terminal_game("graph_recursion_limit", exc)
        except Exception as exc:
            if self._state.status in {"finished", "aborted"}:
                raise
            logger.warning(
                "Graph execution error in run() at step %d (phase=%s): %s",
                self._step_count, self._state.phase, exc, exc_info=True,
            )
            return self._abort_terminal_game("unrecoverable_runtime_error", exc)

        if self._state.phase == "finished" or self._state.winning_faction is not None:
            return self._finish_terminal_game()
        return self._abort_terminal_game("step_limit")

    def run_scripted(
        self,
        max_steps: int = 1000,
        *,
        wolf_kill_target_id: str | None = None,
        use_antidote: bool = False,
        poison_target_id: str | None = None,
        seer_target_id: str | None = None,
    ) -> GameState:
        """使用脚本化夜间输入执行完整游戏。"""
        if self._finished:
            return self._state

        initial = self._build_runtime_state(
            wolf_kill_target_id=wolf_kill_target_id,
            use_antidote=use_antidote,
            poison_target_id=poison_target_id,
            seer_target_id=seer_target_id,
        )

        try:
            for chunk in self._graph.stream(
                initial, {"recursion_limit": max_steps}
            ):
                self._step_count += 1
                self._process_chunk(chunk)
                if self._state.phase == "finished":
                    return self._finish_terminal_game()
        except GraphRecursionError as exc:
            logger.warning(
                "Graph recursion limit in run_scripted() at step %d: %s",
                self._step_count, exc,
            )
            return self._abort_terminal_game("graph_recursion_limit", exc)
        except Exception as exc:
            if self._state.status in {"finished", "aborted"}:
                raise
            logger.warning(
                "Graph execution error in run_scripted() at step %d: %s",
                self._step_count, exc, exc_info=True,
            )
            return self._abort_terminal_game("unrecoverable_runtime_error", exc)

        if self._state.phase == "finished" or self._state.winning_faction is not None:
            return self._finish_terminal_game()
        return self._abort_terminal_game("step_limit")

    def run_step(self, max_steps: int = 1000) -> GameState:
        """推进一个图节点。"""
        if self._finished:
            return self._state

        if self._hitl_interface is not None and self._hitl_interface.is_paused:
            return self._state

        if self._stream_gen is None:
            initial = self._build_runtime_state()
            try:
                self._stream_gen = self._graph.stream(
                    initial, {"recursion_limit": max_steps}
                )
            except GraphRecursionError as exc:
                logger.error("Failed to initialize stream generator: %s", exc)
                return self._abort_terminal_game("graph_recursion_limit", exc)
            except Exception as exc:
                logger.error(
                    "Failed to initialize stream generator: %s", exc,
                    exc_info=True,
                )
                return self._abort_terminal_game("unrecoverable_runtime_error", exc)

        try:
            chunk = next(self._stream_gen)
            self._step_count += 1
            self._process_chunk(chunk)
            if self._state.phase == "finished":
                return self._finish_terminal_game()
        except StopIteration:
            if self._state.phase == "finished" or self._state.winning_faction is not None:
                return self._finish_terminal_game()
            return self._abort_terminal_game("step_limit")
        except GraphRecursionError as exc:
            logger.error("Step recursion limit at step %d: %s", self._step_count, exc)
            return self._abort_terminal_game("graph_recursion_limit", exc)
        except Exception as exc:
            if self._state.status in {"finished", "aborted"}:
                raise
            logger.error(
                "Step execution error at step %d: %s",
                self._step_count, exc, exc_info=True,
            )
            return self._abort_terminal_game("unrecoverable_runtime_error", exc)

        self._persist_if_configured()
        return self._state

    def _finish_terminal_game(self) -> GameState:
        """仅对有胜者的正常终局执行赛后反思与持久化。"""
        self._state = finish_game(self._state)
        self._finished = True
        self._save_memory_snapshot()
        self._persist_if_configured()
        return self._state

    def _abort_terminal_game(
        self,
        reason: str,
        exception: BaseException | None = None,
    ) -> GameState:
        """保留最后节点与异常类型，不运行赛后反思。"""
        self._state = abort_game(
            self._state,
            reason=reason,
            last_node=getattr(self, "_last_node", None),
            step=self._step_count,
            exception=exception,
        )
        self._finished = True
        self._stream_gen = None
        self._persist_if_configured()
        return self._state

    def _persist_if_configured(self) -> None:
        """配置 repository 时保存当前游戏状态。"""
        repository = self._config.repository
        if self._state.status == "aborted":
            validate_aborted_game(self._state)
        if repository is None:
            if self._state.status == "aborted":
                write_emergency_abort(
                    self._state, self._config.emergency_artifact_dir,
                )
            return
        try:
            repository.save_game(self._state)
            if self._state.status == "aborted":
                event = next(
                    event for event in reversed(self._state.events)
                    if event.type == "game_aborted"
                )
                event_id = event.event_id
                if getattr(self, "_persisted_abort_event_id", None) != event_id:
                    repository.append_events(self._state.game_id, [event])
                    self._persisted_abort_event_id = event_id
        except Exception as persistence_error:
            if self._state.status != "aborted":
                raise
            try:
                path = write_emergency_abort(
                    self._state, self._config.emergency_artifact_dir,
                )
            except Exception as emergency_error:
                logger.critical(
                    "CRITICAL: emergency abort persistence failed after repository error: %s",
                    persistence_error,
                    exc_info=True,
                )
                raise RuntimeError(
                    "emergency abort persistence failed"
                ) from emergency_error
            logger.error(
                "Repository abort persistence failed; emergency artifact written to %s: %s",
                path, persistence_error,
            )


__all__ = ["GameRunnerExecutionMixin"]
