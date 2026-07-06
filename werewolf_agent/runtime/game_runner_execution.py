# -*- coding: utf-8 -*-
"""
GameRunner 的图执行、单步推进和持久化协调逻辑。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.runtime.game_runner_execution import GameRunnerExecutionMixin
"""

from __future__ import annotations

import logging

from werewolf_agent.core.models import GameState


logger = logging.getLogger("werewolf_agent.runtime.game_runner")


class GameRunnerExecutionMixin:
    def _process_chunk(self, chunk: dict) -> str | None:
        """处理一个 LangGraph stream chunk，并更新内部状态。"""
        node_name = None
        for name, output in chunk.items():
            node_name = name
            if output is not None and "game_state" in output:
                self._state = output["game_state"]
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
                    self._finished = True
                    self._save_memory_snapshot()
                    self._persist_if_configured()
                    return self._state
                snapshot = (self._state.phase, self._state.day_number, self._state.night_number)
                if snapshot == last_phase_snapshot:
                    stuck_count += 1
                    if stuck_count >= 50:
                        logger.warning(
                            "Game stuck detected at step %d: phase=%s day=%d night=%d - forcing finish",
                            self._step_count, self._state.phase,
                            self._state.day_number, self._state.night_number,
                        )
                        break
                else:
                    stuck_count = 0
                    last_phase_snapshot = snapshot
        except Exception as exc:
            import traceback

            logger.warning(
                "Graph execution error in run() at step %d (phase=%s, day=%d, night=%d): %s\n%s",
                self._step_count, self._state.phase,
                self._state.day_number, self._state.night_number,
                exc, traceback.format_exc(),
            )

        self._finished = self._state.phase == "finished" or self._state.winning_faction is not None
        if self._finished:
            self._save_memory_snapshot()
        self._persist_if_configured()
        return self._state

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
                    self._finished = True
                    self._save_memory_snapshot()
                    self._persist_if_configured()
                    return self._state
        except Exception as exc:
            logger.warning("Graph execution error in run_scripted() at step %d: %s", self._step_count, exc)

        self._finished = self._state.phase == "finished"
        if self._finished:
            self._save_memory_snapshot()
        self._persist_if_configured()
        return self._state

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
            except Exception as exc:
                logger.error("Failed to initialize stream generator: %s", exc)
                return self._state

        try:
            chunk = next(self._stream_gen)
            self._step_count += 1
            self._process_chunk(chunk)
            if self._state.phase == "finished":
                self._finished = True
        except StopIteration:
            self._finished = True
        except Exception as exc:
            logger.error("Step execution error at step %d: %s", self._step_count, exc)
            return self._state

        if self._finished:
            self._save_memory_snapshot()
        self._persist_if_configured()
        return self._state

    def _persist_if_configured(self) -> None:
        """配置 repository 时保存当前游戏状态。"""
        if self._config.repository is not None:
            try:
                self._config.repository.save_game(self._state)
            except Exception as exc:
                logger.warning("Persistence error: %s", exc)


__all__ = ["GameRunnerExecutionMixin"]
