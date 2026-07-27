# -*- coding: utf-8 -*-
"""
GameRunner 的跨局记忆恢复、终局快照和 V2 持久化审计事件逻辑。

作者: Project contributors
创建日期: 2026-07-06
修改日期: 2026-07-27

使用示例:
    >>> from werewolf_agent.runtime.game_runner_memory import GameRunnerMemoryMixin
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import replace

from werewolf_agent.core.event_visibility import EventVisibility
from werewolf_agent.memory.store import MemoryStore
from werewolf_agent.runtime.event_metadata import new_game_event
from werewolf_agent.runtime.reflection_events import canonical_verified_reflections


logger = logging.getLogger("werewolf_agent.runtime.game_runner")


def _persisted_reflection_matches(row: object, expected: dict) -> bool:
    """严格比较仓储回读行，拒绝陈旧同 ID、错误归属和 claim 集漂移。"""
    if not isinstance(row, dict) or row.get("_persistence_active") is not True:
        return False
    entry_id = expected.get("entry_id")
    player_id = expected.get("player_id")
    expected_row = expected.get("expected_row")
    if not isinstance(expected_row, dict):
        return False
    source = row.get("source")
    verified_claim_ids = source.get("verified_claim_ids") if isinstance(source, dict) else None
    expected_claim_ids = expected.get("verified_claim_ids")
    if not isinstance(verified_claim_ids, list) or not isinstance(expected_claim_ids, list):
        return False
    persisted_claim_set = {
        item for item in verified_claim_ids if isinstance(item, str) and item
    }
    expected_claim_set = {
        item for item in expected_claim_ids if isinstance(item, str) and item
    }
    if (
        len(persisted_claim_set) != len(verified_claim_ids)
        or len(expected_claim_set) != len(expected_claim_ids)
        or persisted_claim_set != expected_claim_set
    ):
        return False
    persisted_payload = {
        key: value for key, value in row.items() if key != "_persistence_active"
    }
    return (
        row.get("entry_id") == entry_id
        and row.get("player_id") == player_id
        and row.get("game_id") == expected_row.get("game_id")
        and isinstance(source, dict)
        and source.get("source_game_id") == expected_row.get("game_id")
        and persisted_payload == expected_row
    )


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
        expected_entries: list[dict] = []
        persistence_complete = True
        audit_appended = False
        mem_store = None
        reflection_backups: dict[str, dict | None] = {}
        local_v2_backups: dict[str, object | None] = {}
        current_rows: dict[str, dict] = {}
        snapshot_backups: dict[str, dict | None] = {}
        try:
            from werewolf_agent.cognition.world_state import build_world_state
            from werewolf_agent.memory.relation_graph import RelationGraph
            from werewolf_agent.memory.reflection import (
                ReflectionQualityGate,
                ReflectionSynthesizer,
            )
            from werewolf_agent.memory.reflection_sanitization import anonymize_player_ids

            mem_store = self._cognition_state_manager.memory_store
            decision_failures = self._reflection_decision_preflight_failures()
            if decision_failures:
                self._append_reflection_decision_preflight_failure(
                    decision_failures,
                )
                return
            local_v2_entries = {
                entry.entry_id: entry
                for entry in mem_store.reflections.all_v2_entries()
            }
            for snapshot_id in dict.fromkeys((self._game_id, "latest")):
                snapshot_backups[snapshot_id] = deepcopy(
                    self._config.repository.load_memory_snapshot(snapshot_id)
                )
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
                        "verified_claim_ids": list(
                            verification.get("verified_claim_ids", [])
                        ),
                    }),
                })
                gate = ReflectionQualityGate(
                    existing_entries=mem_store.reflections.all_v2_entries()
                )
                evaluated = gate.evaluate(candidate)
                expected_row = evaluated.to_dict()
                expected_entries.append({
                    "player_id": report.player_id,
                    "decision_id": verification.get("decision_id"),
                    "entry_id": evaluated.entry_id,
                    "verified_claim_ids": list(
                        verification.get("verified_claim_ids", [])
                    ),
                    "rejected_claim_ids": list(
                        verification.get("rejected_claim_ids", [])
                    ),
                    "expected_row": expected_row,
                })
                if evaluated.entry_id not in reflection_backups:
                    reflection_backups[evaluated.entry_id] = deepcopy(
                        self._config.repository.load_reflection(evaluated.entry_id)
                    )
                    local_v2_backups[evaluated.entry_id] = local_v2_entries.get(
                        evaluated.entry_id
                    )
                current_rows[evaluated.entry_id] = expected_row
                try:
                    # 终局验收路径必须暴露写入失败，不能让内存状态伪装成持久化成功。
                    mem_store.reflections.store_v2(
                        evaluated,
                        raise_on_failure=True,
                    )
                except Exception:
                    persistence_complete = False
                    logger.warning(
                        "Failed to persist reflection %s for game %s",
                        evaluated.entry_id,
                        self._game_id,
                        exc_info=True,
                    )

            if not expected_entries and self._has_reflection_complete_event():
                self._ensure_reflection_no_valid_entries_event()
                self._append_reflection_persistence_audit(
                    [],
                    upstream_complete=False,
                    transaction_status="no_valid_entries",
                )
                audit_appended = True
                return

            if not expected_entries:
                coordinator.save_all(
                    memory_store=mem_store,
                    retriever=None,
                    snapshot_id=self._game_id,
                )
                logger.info(
                    "Saved memory snapshot for game %s without reflection transaction",
                    self._game_id,
                )
                return

            if persistence_complete:
                try:
                    coordinator.save_all(
                        memory_store=mem_store,
                        retriever=None,
                        snapshot_id=self._game_id,
                    )
                except Exception:
                    persistence_complete = False
                    logger.warning(
                        "Failed to save memory snapshot for game %s",
                        self._game_id,
                        exc_info=True,
                    )
            persistence_complete = self._append_reflection_persistence_audit(
                expected_entries,
                upstream_complete=persistence_complete,
            )
            audit_appended = True
            if not persistence_complete:
                rollback_complete = self._rollback_reflection_transaction(
                    mem_store,
                    reflection_backups=reflection_backups,
                    local_v2_backups=local_v2_backups,
                    current_rows=current_rows,
                    snapshot_backups=snapshot_backups,
                )
                self._set_latest_reflection_rollback_status(rollback_complete)
            if persistence_complete:
                logger.info(
                    "Saved memory snapshot for game %s (%d players, %d reviews)",
                    self._game_id, len(player_ids), len(ground_truth),
                )
        except Exception:
            logger.warning(
                "Failed to save memory snapshot for game %s", self._game_id,
                exc_info=True,
            )
            rollback_complete = True
            if mem_store is not None:
                rollback_complete = self._rollback_reflection_transaction(
                    mem_store,
                    reflection_backups=reflection_backups,
                    local_v2_backups=local_v2_backups,
                    current_rows=current_rows,
                    snapshot_backups=snapshot_backups,
                )
            if self._has_reflection_complete_event() and not audit_appended:
                self._append_reflection_persistence_audit(
                    expected_entries,
                    upstream_complete=False,
                    rollback_complete=rollback_complete,
                    transaction_status=(
                        "persistence_failed" if (
                            expected_entries
                            or self._has_valid_reflection_candidates()
                        )
                        else "no_valid_entries"
                    ),
                )

    def _append_reflection_persistence_audit(
        self,
        expected_entries: list[dict],
        *,
        upstream_complete: bool,
        rollback_complete: bool = True,
        transaction_status: str | None = None,
    ) -> bool:
        """逐条回读预期反思行；写入或回读不完整时显式失败关闭。"""
        repository_read_complete = True
        try:
            rows = self._config.repository.load_reflections_by_game(self._game_id)
        except Exception:
            rows = []
            repository_read_complete = False
            logger.warning(
                "Failed to read persisted reflections for game %s",
                self._game_id,
                exc_info=True,
            )
        rows_by_entry_id: dict[str, dict] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            entry_id = row.get("entry_id")
            if isinstance(entry_id, str) and entry_id:
                rows_by_entry_id[entry_id] = row
        expected_entry_ids = {
            entry_id for entry_id in (
                expected.get("entry_id") for expected in expected_entries
            )
            if isinstance(entry_id, str) and entry_id
        }
        snapshot_read_complete = self._reflection_snapshots_match(
            expected_entry_ids
        )
        entries = []
        for expected in expected_entries:
            entry_id = expected.get("entry_id")
            row = rows_by_entry_id.get(entry_id)
            row_found = row is not None
            row_matches = _persisted_reflection_matches(row, expected)
            entry_complete = (
                upstream_complete
                and repository_read_complete
                and snapshot_read_complete
                and row_matches
            )
            source = row.get("source") if isinstance(row, dict) else None
            claim_ids = (
                source.get("verified_claim_ids")
                if isinstance(source, dict)
                else []
            )
            persisted = {
                claim_id for claim_id in claim_ids
                if isinstance(claim_id, str)
            } if isinstance(claim_ids, list) else set()
            rejected = {
                claim_id for claim_id in expected.get("rejected_claim_ids", [])
                if isinstance(claim_id, str)
            }
            entries.append({
                "player_id": expected.get("player_id"),
                "decision_id": expected.get("decision_id"),
                "verified_claim_ids": list(
                    expected.get("verified_claim_ids", [])
                ),
                "entry_id": entry_id,
                "row_found": row_found,
                "persistence_complete": entry_complete,
                "persisted_rejected_fact_count": (
                    len(rejected & persisted) if entry_complete else None
                ),
            })
        persisted_entry_count = sum(
            entry["persistence_complete"] is True for entry in entries
        )
        complete = (
            bool(expected_entries)
            and
            upstream_complete
            and rollback_complete
            and repository_read_complete
            and snapshot_read_complete
            and persisted_entry_count == len(expected_entries)
        )
        status = transaction_status or self._reflection_persistence_status(
            expected_entries,
            persistence_complete=complete,
        )
        if status not in {
            "complete", "partial", "no_valid_entries", "persistence_failed",
        }:
            status = "persistence_failed"
        if status not in {"complete", "partial"} or not rollback_complete:
            complete = False
        event = new_game_event(
            self._state,
            "reflection_persistence_audit",
            payload={
                "status": status,
                "expected_entry_count": len(expected_entries),
                "persisted_entry_count": persisted_entry_count,
                "repository_read_complete": repository_read_complete,
                "snapshot_read_complete": snapshot_read_complete,
                "persistence_complete": complete,
                "rollback_complete": rollback_complete,
                "entries": entries,
            },
            visibility=EventVisibility.MODERATOR_ONLY,
        )
        self._state = replace(self._state, events=[*self._state.events, event])
        return complete

    def _reflection_persistence_status(
        self,
        expected_entries: list[dict],
        *,
        persistence_complete: bool,
    ) -> str:
        """根据玩家全集和明确失败归因判定局级事务终态。"""
        if not expected_entries:
            return "no_valid_entries"
        if not persistence_complete:
            return "persistence_failed"

        expected_players = {
            item.get("player_id") for item in expected_entries
            if isinstance(item.get("player_id"), str)
        }
        reflection_entries = self._latest_reflection_event_entries()
        if not reflection_entries:
            return "persistence_failed"
        event_players = {
            item.get("player_id") for item in reflection_entries
            if isinstance(item.get("player_id"), str)
        }
        if event_players == expected_players == set(self._state.players):
            return "complete"
        failed_players = {
            item.get("player_id") for item in reflection_entries
            if isinstance(item.get("player_id"), str)
            and isinstance(item.get("failure_stage"), str)
            and bool(item.get("failure_stage"))
            and isinstance(item.get("failure_code"), str)
            and bool(item.get("failure_code"))
        }
        if (
            expected_players
            and event_players == set(self._state.players)
            and expected_players.isdisjoint(failed_players)
            and expected_players | failed_players == event_players
        ):
            return "partial"
        return "persistence_failed"

    def _latest_reflection_event_entries(self) -> list[dict]:
        """返回最近一次局级反思事件中结构有效的玩家条目。"""
        for event in reversed(self._state.events):
            if event.type != "reflection_complete":
                continue
            entries = event.payload.get("entries")
            if not isinstance(entries, list):
                return []
            return [item for item in entries if isinstance(item, dict)]
        return []

    def _reflection_decision_preflight_failures(self) -> list[dict]:
        """在合成或写入前拒绝不属于本局当前玩家的反思决策。"""
        failures: list[dict] = []
        entries = [
            item
            for event in self._state.events
            if event.type == "reflection_complete"
            for item in event.payload.get("entries", [])
            if isinstance(item, dict)
        ]
        for item in entries:
            player_id = item.get("player_id")
            if not isinstance(player_id, str) or player_id not in self._state.players:
                failures.append({
                    "player_id": player_id if isinstance(player_id, str) else "unknown",
                    "decision_id": None,
                    "failure_stage": "persisted",
                    "failure_code": "reflection_decision_id_mismatch",
                    "persistence_complete": False,
                })
                continue
            expected_decision_id = f"reflection:{self._game_id}:{player_id}"
            verification = item.get("verification")
            verification_decision_id = (
                verification.get("decision_id")
                if isinstance(verification, dict)
                else None
            )
            if (
                item.get("decision_id") != expected_decision_id
                or verification_decision_id != expected_decision_id
            ):
                failures.append({
                    "player_id": player_id,
                    "decision_id": expected_decision_id,
                    "failure_stage": "persisted",
                    "failure_code": "reflection_decision_id_mismatch",
                    "persistence_complete": False,
                })
        return failures

    def _append_reflection_decision_preflight_failure(
        self,
        failures: list[dict],
    ) -> None:
        """记录可归属的持久化前置失败，且不触碰仓储或跨局内存。"""
        event = new_game_event(
            self._state,
            "reflection_persistence_audit",
            payload={
                "status": "persistence_failed",
                "expected_entry_count": len(failures),
                "persistence_complete": False,
                "rollback_complete": True,
                "entries": failures,
            },
            visibility=EventVisibility.MODERATOR_ONLY,
        )
        self._state = replace(
            self._state,
            events=[*self._state.events, event],
        )

    def _has_reflection_complete_event(self) -> bool:
        return any(event.type == "reflection_complete" for event in self._state.events)

    def _has_valid_reflection_candidates(self) -> bool:
        """异常发生在合成前时，仍根据 verification 区分持久化失败与空事务。"""
        for item in self._latest_reflection_event_entries():
            verification = item.get("verification")
            lessons = verification.get("verified_lessons") if isinstance(
                verification, dict
            ) else None
            if isinstance(lessons, list) and any(
                isinstance(lesson, dict)
                and isinstance(lesson.get("abstraction"), str)
                and bool(lesson.get("abstraction", "").strip())
                for lesson in lessons
            ):
                return True
        return False

    def _ensure_reflection_no_valid_entries_event(self) -> None:
        """缺少有效 lesson 时写唯一的 moderator-only 诊断事件。"""
        latest_reflection_index = max(
            (
                index for index, event in enumerate(self._state.events)
                if event.type == "reflection_complete"
            ),
            default=-1,
        )
        if latest_reflection_index < 0 or any(
            event.type == "reflection_no_valid_entries"
            for event in self._state.events[latest_reflection_index + 1:]
        ):
            return
        failures = [
            {
                "player_id": item.get("player_id"),
                "decision_id": item.get("decision_id"),
                "failure_stage": item.get("failure_stage"),
                "failure_code": item.get("failure_code"),
            }
            for item in self._latest_reflection_event_entries()
        ]
        event = new_game_event(
            self._state,
            "reflection_no_valid_entries",
            payload={
                "status": "no_valid_entries",
                "player_count": len(failures),
                "failures": failures,
            },
            visibility=EventVisibility.MODERATOR_ONLY,
        )
        self._state = replace(
            self._state,
            events=[*self._state.events, event],
        )

    def _reflection_snapshots_match(self, expected_entry_ids: set[str]) -> bool:
        """回读本局与 latest 快照，要求元数据一致并包含本批反思 ID。"""
        for snapshot_id in (self._game_id, "latest"):
            try:
                snapshot = self._config.repository.load_memory_snapshot(snapshot_id)
            except Exception:
                logger.warning(
                    "Failed to read memory snapshot %s for game %s",
                    snapshot_id,
                    self._game_id,
                    exc_info=True,
                )
                return False
            if not isinstance(snapshot, dict):
                return False
            reflection_ids = snapshot.get("reflections")
            if (
                snapshot.get("_persistence_active") is not True
                or snapshot.get("_source_game_id") != self._game_id
                or snapshot.get("_snapshot_id") != snapshot_id
                or not isinstance(reflection_ids, list)
                or not expected_entry_ids.issubset({
                    item for item in reflection_ids if isinstance(item, str)
                })
            ):
                return False
        return True

    def _rollback_reflection_transaction(
        self,
        mem_store: MemoryStore,
        *,
        reflection_backups: dict[str, dict | None],
        local_v2_backups: dict[str, object | None],
        current_rows: dict[str, dict],
        snapshot_backups: dict[str, dict | None],
    ) -> bool:
        """补偿删除本批新反思，避免失败批次成为跨局可读状态。"""
        complete = True
        for entry_id, previous_row in reflection_backups.items():
            mem_store.reflections.restore_v2_local(
                entry_id,
                local_v2_backups.get(entry_id),
            )
            try:
                if previous_row is None:
                    self._config.repository.delete_reflection(entry_id)
                else:
                    self._config.repository.save_reflection(previous_row)
            except Exception:
                complete = False
                logger.warning(
                    "Failed to roll back reflection %s for game %s",
                    entry_id,
                    self._game_id,
                    exc_info=True,
                )
                quarantined = deepcopy(current_rows.get(entry_id, {}))
                quarantined.update({
                    "entry_id": entry_id,
                    "_persistence_active": False,
                    "_rollback_game_id": self._game_id,
                })
                try:
                    self._config.repository.save_reflection(quarantined)
                except Exception:
                    logger.warning(
                        "Failed to quarantine reflection %s for game %s",
                        entry_id,
                        self._game_id,
                        exc_info=True,
                    )

        for snapshot_id, previous_snapshot in snapshot_backups.items():
            try:
                if previous_snapshot is None:
                    self._config.repository.delete_memory_snapshot(snapshot_id)
                else:
                    self._config.repository.save_memory_snapshot(
                        snapshot_id,
                        previous_snapshot,
                    )
            except Exception:
                complete = False
                logger.warning(
                    "Failed to roll back memory snapshot %s for game %s",
                    snapshot_id,
                    self._game_id,
                    exc_info=True,
                )
                try:
                    self._config.repository.save_memory_snapshot(snapshot_id, {
                        "_persistence_active": False,
                        "_rollback_game_id": self._game_id,
                    })
                except Exception:
                    logger.warning(
                        "Failed to quarantine memory snapshot %s for game %s",
                        snapshot_id,
                        self._game_id,
                        exc_info=True,
                    )
        return complete

    def _set_latest_reflection_rollback_status(self, complete: bool) -> None:
        """回填最近一次持久化审计的最终补偿状态。"""
        events = list(self._state.events)
        for index in range(len(events) - 1, -1, -1):
            event = events[index]
            if event.type != "reflection_persistence_audit":
                continue
            events[index] = replace(
                event,
                payload={**event.payload, "rollback_complete": complete},
            )
            self._state = replace(self._state, events=events)
            return

    def _latest_verified_reflections(self) -> dict[str, dict]:
        return canonical_verified_reflections(self._state.events)

    def _latest_self_reviews(self) -> dict[str, dict]:
        """兼容旧调用名；只返回已核验的安全摘要，不返回 provider 草稿。"""
        return self._latest_verified_reflections()


__all__ = ["GameRunnerMemoryMixin"]
