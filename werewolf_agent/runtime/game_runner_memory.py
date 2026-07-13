# -*- coding: utf-8 -*-
"""
GameRunner 的跨局记忆恢复和终局快照保存逻辑。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-13

使用示例:
    >>> from werewolf_agent.runtime.game_runner_memory import GameRunnerMemoryMixin
"""

from __future__ import annotations

import logging

from werewolf_agent.memory.store import MemoryStore
from werewolf_agent.runtime.reflection_events import canonical_verified_reflections


logger = logging.getLogger("werewolf_agent.runtime.game_runner")


class GameRunnerMemoryMixin:
    def _restore_memory_if_configured(self) -> None:
        """初始化时尝试恢复上一局记忆快照。"""
        coordinator = self._config.memory_coordinator
        if coordinator is None or self._config.repository is None:
            return
        try:
            if hasattr(coordinator, "restore_for_new_game"):
                mem = coordinator.restore_for_new_game(self._game_id)
                rag = coordinator.restore_rag()
            else:
                mem, rag = coordinator.restore_all(snapshot_id=self._game_id)
            self._restored_memory = mem
            self._restored_rag = rag
            if mem is not None:
                logger.info("Restored memory snapshot for game %s", self._game_id)
        except Exception as exc:
            logger.warning("Memory restore error for game %s: %s", self._game_id, exc)

    def _save_memory_snapshot(self) -> None:
        """终局时持久化完整记忆快照。"""
        coordinator = self._config.memory_coordinator
        if coordinator is None or self._config.repository is None:
            return
        try:
            from werewolf_agent.cognition.world_state import build_world_state
            from werewolf_agent.memory.relation_graph import RelationGraph
            from werewolf_agent.memory.reflection import (
                ReflectionQualityGate,
                ReflectionSynthesizer,
            )
            from werewolf_agent.memory.reflection_sanitization import anonymize_player_ids

            mem_store = self._cognition_state_manager.memory_store
            player_ids = list(self._state.players.keys())
            role_names = list({p.role for p in self._state.players.values()})

            ws = build_world_state(self._state)
            mem_store.relation_graph = RelationGraph()
            mem_store.import_world_state(ws)

            for pid in player_ids:
                if mem_store.get_matrix(pid) is None:
                    mem_store.init_matrix(pid, player_ids, role_names)

            winning_faction = self._state.winning_faction or "good"
            ground_truth = {pid: p.role for pid, p in self._state.players.items()}
            reports = mem_store.generate_reviews_for_game(
                game_id=self._game_id,
                player_ids=player_ids,
                roles=ground_truth,
                winning_faction=winning_faction,
                ground_truth=ground_truth,
                hybrid_master_factions={
                    pid: self._state.hybrid_master_faction
                    for pid, role in ground_truth.items()
                    if role == "hybrid" and self._state.hybrid_master_faction
                },
                generate_reflection=False,
            )
            verified_reflections = self._latest_verified_reflections()
            synthesizer = ReflectionSynthesizer()
            for report in reports:
                verification = verified_reflections.get(report.player_id, {})
                lessons = verification.get("verified_lessons", [])
                if not isinstance(lessons, list) or not lessons:
                    continue
                verified_text = "\n".join(
                    anonymize_player_ids(str(lesson.get("abstraction") or ""))
                    for lesson in lessons if isinstance(lesson, dict)
                )
                if not verified_text.strip():
                    continue
                role = ground_truth.get(report.player_id, report.role)
                master_faction = (
                    self._state.hybrid_master_faction
                    if role == "hybrid"
                    else None
                )
                faction = MemoryStore._player_faction(
                    role,
                    master_faction=master_faction,
                )
                candidate = synthesizer.synthesize(
                    llm_self_review="",
                    review_report=report,
                    faction=faction,
                )
                candidate = candidate.model_copy(update={
                    "prompt_card": candidate.prompt_card.model_copy(update={
                        "lesson": verified_text,
                        "recommended_action": verified_text,
                        "fact_basis": "verified_event_claims",
                        "auto_verified": True,
                    }),
                    "source": candidate.source.model_copy(update={
                        "llm_self_review": "",
                        "auto_review_summary": "",
                        "source_game_id": self._game_id,
                    }),
                })
                gate = ReflectionQualityGate(
                    existing_entries=mem_store.reflections.all_v2_entries()
                )
                mem_store.reflections.store_v2(gate.evaluate(candidate))

            coordinator.save_all(
                memory_store=mem_store,
                retriever=None,
                snapshot_id=self._game_id,
            )
            logger.info(
                "Saved memory snapshot for game %s (%d players, %d reviews)",
                self._game_id, len(player_ids), len(ground_truth),
            )
        except Exception:
            logger.warning(
                "Failed to save memory snapshot for game %s", self._game_id,
                exc_info=True,
            )

    def _latest_verified_reflections(self) -> dict[str, dict]:
        return canonical_verified_reflections(self._state.events)

    def _latest_self_reviews(self) -> dict[str, dict]:
        """兼容旧调用名；只返回已核验的安全摘要，不返回 provider 草稿。"""
        return self._latest_verified_reflections()


__all__ = ["GameRunnerMemoryMixin"]
