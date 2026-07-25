# -*- coding: utf-8 -*-
"""
运行真实游戏，并生成带显式支持状态的在线质量指标。

作者: Project contributors
修改日期: 2026-07-23

使用示例:
    python scripts/run_real_game.py --seed 42 --max-steps 500
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import is_dataclass, replace
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from werewolf_agent.model_gateway.providers import load_local_dotenv  # noqa: E402
from werewolf_agent.evaluation.balance_audit import (  # noqa: E402
    compute_acceptance_audit_metrics,
    compute_wolf_plan_outcome_metrics,
)
from werewolf_agent.core.models import GameEvent  # noqa: E402
from werewolf_agent.evaluation.game_projection import (  # noqa: E402
    AcceptanceGameProjection,
    normalize_quality_score,
    project_acceptance_game,
)
from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig  # noqa: E402
from werewolf_agent.runtime.event_metadata import serialize_game_event  # noqa: E402
from werewolf_agent.runtime.exposure_audit import (  # noqa: E402
    summarize_persona_prompt_confirmation,
)
from scripts.run_real_game_reports import (  # noqa: E402
    _sep,
    check_leakage,
    print_game_summary,
    print_pace_report,
    print_quality_audit,
    print_usage_stats,
    reflection_verification_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
# 打开游戏步骤细节，便于观察身份分配、夜晚行动、发言和投票；压低 httpx 噪声。
logging.getLogger("werewolf_agent.runtime.nodes").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("real_game")

_SPEECH_GENERATED_BY_VALUES = frozenset({
    "model", "repair", "provider_fallback", "terminal_fallback",
})
_SPEECH_MODEL_SUCCESS_VALUES = frozenset({
    "model", "repair", "provider_fallback",
})
_REFLECTION_TRANSACTION_STATES = frozenset({
    "not_requested", "generated", "schema_validated", "facts_verified",
    "lessons_verified", "persisted",
})
_REFLECTION_STATUSES = frozenset({
    "complete", "partial", "no_valid_entries", "persistence_failed",
})
_REFLECTION_ENTRY_ID = re.compile(
    r"reflection_[A-Za-z0-9._-]{1,128}_[A-Za-z0-9._-]{1,128}\Z"
)
_SAFE_REFLECTION_PLAYER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_REFLECTION_ROLES = frozenset({
    "villager", "seer", "witch", "hunter", "idiot", "werewolf", "hybrid",
})
_REFLECTION_FAILURE_STAGES = frozenset({
    "generated", "schema_validated", "facts_verified", "lessons_verified",
})
_REFLECTION_VERIFICATION_STATUSES = frozenset({
    "not_generated", "invalid_structured_draft", "verified", "agent_error",
})
_REFLECTION_SENSITIVE_MARKERS = frozenset({
    "raw_prompt", "provider_response", "private_prompt", "original_text",
})
_REDACTED_LESSON_ABSTRACTION = "[REDACTED_VERIFIED_LESSON]"
_SPEECH_DECISION_OUTCOME_VALUES = frozenset({
    "direct_success", "retry_success", "repaired_success",
    "provider_fallback_success", "terminal_fallback",
})
_SPEECH_EVENT_TASKS = {
    "speech": "speech",
    "sheriff_speech": "sheriff_speech",
    "sheriff_pk_speech": "sheriff_speech",
    "tie_pk_speech": "pk_speech",
}
_SAFE_GAME_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def _configure_file_logging() -> Path:
    """在实际启动时创建日志文件，导入报告 helper 不产生文件副作用。"""
    path = Path(os.environ.get("WEREWOLF_GAME_LOG_PATH", ROOT / "game_stdout.log"))
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))
    logging.getLogger().addHandler(file_handler)
    return path


# ── 辅助函数 ─────────────────────────────────────────────────────────────

def _format_api_key_status(api_key: str) -> str:
    return "configured" if api_key else "missing"



# 保存游戏日志和质量指标。
def compute_game_quality_score(
    source: GameRunner | AcceptanceGameProjection,
) -> dict[str, Any]:
    """从固定游戏投影计算可在线保存、离线复算的结构化质量指标。"""
    projection = project_acceptance_game(
        source.state if hasattr(source, "state") else source,
        steps=getattr(source, "step_count", None),
    )
    projected = projection.to_mapping()
    fallback_metrics_supported = projection.events_supported
    fallback_unsupported_reason = (
        None if fallback_metrics_supported else projection.events_unsupported_reason
    )
    events = [
        GameEvent(type=str(event.get("type") or ""), payload=dict(event.get("payload") or {}))
        for event in projected["events"]
    ]
    reflection_metrics = reflection_verification_metrics(
        type("QualityState", (), {"events": events})()
    )
    traces = [e for e in events if e.type == "action_trace_audit"]
    action_fallback_traces = [
        e.payload.get("action_trace", {}) for e in traces
        if e.payload.get("action_trace", {}).get("fallback_reason")
    ]
    wolf_plan_fallbacks = [
        e for e in events if e.type == "wolf_team_plan_fallback"
    ]
    action_fallback_count = sum(
        1 for e in traces
        if e.payload.get("action_trace", {}).get("fallback_reason")
    )
    wolf_plan_outcomes = compute_wolf_plan_outcome_metrics([{
        "game_id": projection.game_id,
        "events": [
            {"type": event.type, "payload": event.payload}
            for event in events
            if event.type in {"wolf_team_plan", "wolf_team_plan_fallback"}
        ]
    }])
    acceptance_metrics = compute_acceptance_audit_metrics([projection])
    wolf_plan_fallback_count = wolf_plan_outcomes[
        "wolf_team_plan_terminal_fallback_count"
    ]
    wolf_plan_attempts = wolf_plan_outcomes["wolf_team_plan_total_count"]
    structured_fail = sum(
        1 for e in traces
        if e.payload.get("action_trace", {}).get("structured_failure_reason")
    )
    total = len(traces)
    total_quality_events = total + wolf_plan_attempts
    fallback_count = action_fallback_count + wolf_plan_fallback_count
    fallback_rate = fallback_count / total_quality_events if total_quality_events > 0 else 0.0
    speeches = [e for e in events if e.type == "speech"]
    speech_opportunities = _project_speech_opportunities(events)
    observed_speeches = [
        row for row in speech_opportunities if isinstance(row.get("text"), str)
    ]
    non_empty_speeches = sum(
        bool(row["text"].strip()) for row in observed_speeches
    )
    speech_rate = (
        non_empty_speeches / len(speech_opportunities)
        if speech_opportunities and len(observed_speeches) == len(speech_opportunities)
        else None
    )
    speech_traces = [dict(row.get("action_trace") or {}) for row in speech_opportunities]
    model_observation = _categorical_speech_observation(
        speech_traces,
        field="generated_by",
        allowed=_SPEECH_GENERATED_BY_VALUES,
        positive=_SPEECH_MODEL_SUCCESS_VALUES,
        missing_reason="missing_generated_by",
        invalid_reason="invalid_generated_by",
    )
    terminal_observation = _categorical_speech_observation(
        speech_traces,
        field="decision_outcome",
        allowed=_SPEECH_DECISION_OUTCOME_VALUES,
        positive=frozenset({"terminal_fallback"}),
        missing_reason="missing_decision_outcome",
        invalid_reason="invalid_decision_outcome",
    )
    speech_semantic_traces = [
        trace for trace in speech_traces
        if isinstance((trace.get("semantic_repair_audit") or {}).get("success"), bool)
    ]
    speech_semantic_observed_count = len(speech_semantic_traces)
    speech_semantic_acceptance_count = sum(
        (trace.get("semantic_repair_audit") or {}).get("success") is True
        for trace in speech_semantic_traces
    )
    phases_seen = {e.payload.get("phase") for e in events if e.type == "judge_broadcast"}
    has_winner = bool(projection.winning_faction)
    action_fallback_by_error_code = Counter(
        trace.get("retry", {}).get("error_code") or "unknown"
        for trace in action_fallback_traces
    )
    retry_error_counts = Counter(
        retry_error for retry_error in (
            e.payload.get("action_trace", {}).get("retry", {}).get("error_code")
            for e in traces
        )
        if retry_error
    )
    wolf_team_plan_fallback_by_reason = Counter(
        e.payload.get("reason") or "unknown" for e in wolf_plan_fallbacks
    )
    fallback_by_reason = Counter(
        trace.get("structured_failure_reason")
        or trace.get("retry", {}).get("error_code")
        or "unknown"
        for trace in action_fallback_traces
    )
    fallback_by_reason.update(wolf_team_plan_fallback_by_reason)
    fallback_by_stage = Counter(
        trace.get("structured_failure_stage") or "unknown"
        for trace in action_fallback_traces
    )
    fallback_by_stage.update(
        e.payload.get("stage") or "unknown" for e in wolf_plan_fallbacks
    )

    return {
        **reflection_metrics,
        "persona_prompt_confirmation": summarize_persona_prompt_confirmation(
            events
        ),
        "fallback_metrics_supported": fallback_metrics_supported,
        "fallback_metrics_unsupported_reason": fallback_unsupported_reason,
        "fallback_rate": round(fallback_rate, 3) if fallback_metrics_supported else None,
        "fallback_count": fallback_count if fallback_metrics_supported else None,
        "action_fallback_count": (
            action_fallback_count if fallback_metrics_supported else None
        ),
        "wolf_team_plan_fallback_count": (
            wolf_plan_fallback_count if fallback_metrics_supported else None
        ),
        "fallback_by_reason": (
            dict(sorted(fallback_by_reason.items())) if fallback_metrics_supported else None
        ),
        "fallback_by_stage": (
            dict(sorted(fallback_by_stage.items())) if fallback_metrics_supported else None
        ),
        "action_fallback_by_error_code": (
            dict(sorted(action_fallback_by_error_code.items()))
            if fallback_metrics_supported else None
        ),
        "retry_error_counts": (
            dict(sorted(retry_error_counts.items())) if fallback_metrics_supported else None
        ),
        "wolf_team_plan_fallback_by_reason": (
            dict(sorted(wolf_team_plan_fallback_by_reason.items()))
            if fallback_metrics_supported else None
        ),
        "structured_fail_count": structured_fail if fallback_metrics_supported else None,
        "total_action_traces": total if fallback_metrics_supported else None,
        "total_wolf_team_plans": wolf_plan_attempts if fallback_metrics_supported else None,
        **wolf_plan_outcomes,
        **acceptance_metrics,
        "total_quality_events": (
            total_quality_events if fallback_metrics_supported else None
        ),
        "speech_count": len(speeches),
        "speech_opportunity_count": len(speech_opportunities),
        "non_empty_speech_count": non_empty_speeches,
        "speech_non_empty_metrics_supported": bool(speech_opportunities) and speech_rate is not None,
        "speech_non_empty_unsupported_reason": (
            None if speech_opportunities and speech_rate is not None
            else "missing_speech_text_fields"
        ),
        "speech_non_empty_observed_count": len(observed_speeches),
        "speech_non_empty_rate": (
            round(speech_rate, 3) if speech_rate is not None else None
        ),
        "speech_model_success_metrics_supported": model_observation["supported"],
        "speech_model_success_unsupported_reason": model_observation["reason"],
        "speech_model_success_observed_count": model_observation["observed_count"],
        "speech_model_success_rate": model_observation["rate"],
        "speech_terminal_fallback_metrics_supported": terminal_observation["supported"],
        "speech_terminal_fallback_unsupported_reason": terminal_observation["reason"],
        "speech_terminal_fallback_observed_count": terminal_observation["observed_count"],
        "speech_terminal_fallback_rate": terminal_observation["rate"],
        "speech_semantic_acceptance_metrics_supported": (
            bool(speech_opportunities)
            and len(speech_semantic_traces) == len(speech_opportunities)
        ),
        "speech_semantic_acceptance_unsupported_reason": (
            None
            if speech_opportunities and len(speech_semantic_traces) == len(speech_opportunities)
            else "missing_speech_semantic_fields"
        ),
        "speech_semantic_acceptance_observed_count": speech_semantic_observed_count,
        "speech_semantic_acceptance_rate": (
            round(speech_semantic_acceptance_count / speech_semantic_observed_count, 3)
            if speech_opportunities
            and speech_semantic_observed_count == len(speech_opportunities)
            else None
        ),
        "phases_seen": len(phases_seen),
        "has_winner": has_winner,
        "steps": projection.steps,
    }


def _project_speech_opportunities(events: list[GameEvent]) -> list[dict[str, Any]]:
    """把公开节点与私有决策审计合并为一份真实发言机会投影。"""
    rows: list[dict[str, Any]] = []
    for event in events:
        task_type = _SPEECH_EVENT_TASKS.get(event.type)
        if task_type is not None:
            speaker = event.payload.get("speaker")
            if speaker is not None or "text" in event.payload:
                match = next((
                    row for row in reversed(rows)
                    if row["task_type"] == task_type
                    and row["player_id"] == speaker
                    and not row["public_observed"]
                ), None)
                if match is None:
                    rows.append({
                        "task_type": task_type,
                        "player_id": speaker,
                        "text": event.payload.get("text"),
                        "action_trace": None,
                        "public_observed": True,
                    })
                else:
                    match["text"] = event.payload.get("text")
                    match["public_observed"] = True
            continue
        if event.type != "action_trace_audit":
            continue
        task_type = event.payload.get("task_type") or event.payload.get("phase")
        if task_type not in {"speech", "sheriff_speech", "pk_speech"}:
            continue
        player_id = event.payload.get("player_id")
        match = next((
            row for row in rows
            if row["task_type"] == task_type
            and row["action_trace"] is None
            and (player_id is None or row["player_id"] == player_id)
        ), None)
        if match is None:
            match = {
                "task_type": task_type,
                "player_id": player_id,
                "text": "",
                "action_trace": None,
                "public_observed": False,
            }
            rows.append(match)
        trace = event.payload.get("action_trace")
        match["action_trace"] = trace if isinstance(trace, Mapping) else {}
    return rows


def _faction_for_player(gs, player_id: str | None) -> str | None:
    if not player_id:
        return None
    player = gs.players.get(player_id)
    if player is None:
        return None
    if player.faction in ("good", "werewolf"):
        return player.faction
    if player.role == "werewolf":
        return "werewolf"
    if player.role in {"villager", "seer", "witch", "hunter", "idiot"}:
        return "good"
    return None


def _final_hybrid_fields(gs) -> dict[str, str | None]:
    """Export final hybrid fields even when the runner state is stale."""
    fields = {
        "hybrid_master_id": gs.hybrid_master_id,
        "hybrid_master_faction": gs.hybrid_master_faction,
        "hybrid_result": gs.hybrid_result,
    }
    winner = gs.winning_faction

    for event in reversed(gs.events):
        if event.type != "victory":
            continue
        payload = event.payload or {}
        winner = winner or payload.get("winner") or payload.get("winning_faction")
        for key in fields:
            if fields[key] is None:
                fields[key] = payload.get(key)
        break

    if fields["hybrid_master_id"] is None:
        for event in reversed(gs.events):
            if event.type != "hybrid_master_chosen":
                continue
            payload = event.payload or {}
            fields["hybrid_master_id"] = payload.get("master_id")
            break

    if fields["hybrid_master_faction"] is None:
        fields["hybrid_master_faction"] = _faction_for_player(gs, fields["hybrid_master_id"])
    if fields["hybrid_result"] is None and fields["hybrid_master_faction"] and winner:
        fields["hybrid_result"] = "win" if fields["hybrid_master_faction"] == winner else "lose"

    return fields


def _safe_non_negative_int(value: Any) -> int | None:
    """只接受原生非负整数，拒绝 bool 和隐式字符串转换。"""
    return value if type(value) is int and value >= 0 else None


def _safe_enum(value: Any, allowed: frozenset[str]) -> str | None:
    """枚举边界先验证字符串类型，再执行集合 membership。"""
    return value if isinstance(value, str) and value in allowed else None


def _safe_reflection_component(value: Any) -> str | None:
    """验证可进入 canonical reflection 身份的单个安全组件。"""
    if not isinstance(value, str) or not _SAFE_REFLECTION_PLAYER_ID.fullmatch(value):
        return None
    lowered = value.lower()
    if any(marker in lowered for marker in _REFLECTION_SENSITIVE_MARKERS):
        return None
    return value


def _authoritative_reflection_player(
    players: Mapping[str, Any] | None,
    player_id: Any,
) -> tuple[str, str, bool] | None:
    """仅从投影玩家表重建玩家 ID、固定角色和存活状态。"""
    safe_player_id = _safe_reflection_component(player_id)
    if safe_player_id is None or not isinstance(players, Mapping):
        return None
    player = players.get(safe_player_id)
    if not isinstance(player, Mapping):
        return None
    authoritative_id = player.get("id", safe_player_id)
    role = player.get("role")
    alive = player.get("alive")
    if (
        authoritative_id != safe_player_id
        or not isinstance(role, str)
        or role not in _REFLECTION_ROLES
        or not isinstance(alive, bool)
    ):
        return None
    return safe_player_id, role, alive


def _redacted_reflection_identifier(
    raw_identifier: Any,
    *,
    game_id: str,
    player_id: str,
    identifier_kind: str,
) -> str | None:
    """把不可信标识映射为同局同玩家可复算的无原文摘要。"""
    if not isinstance(raw_identifier, str) or not raw_identifier:
        return None
    digest = hashlib.sha256(
        b"\x00".join((
            game_id.encode("utf-8"),
            player_id.encode("utf-8"),
            identifier_kind.encode("ascii"),
            raw_identifier.encode("utf-8"),
        ))
    ).hexdigest()[:20]
    return f"redacted_{identifier_kind}_{digest}"


def _redacted_reflection_identifiers(
    value: Any,
    *,
    game_id: str,
    player_id: str,
    identifier_kind: str,
) -> list[str]:
    """按原顺序去重并脱敏一组字符串标识。"""
    if not isinstance(value, (list, tuple)):
        return []
    redacted: list[str] = []
    seen: set[str] = set()
    for item in value:
        safe = _redacted_reflection_identifier(
            item,
            game_id=game_id,
            player_id=player_id,
            identifier_kind=identifier_kind,
        )
        if safe is not None and safe not in seen:
            seen.add(safe)
            redacted.append(safe)
    return redacted


def _safe_reflection_verification_payload(
    candidate: Any,
    *,
    game_id: str,
    player_id: str,
    decision_id: str,
    entry_decision_matches: bool,
) -> dict[str, Any]:
    """重建核验摘要，彻底移除 claim/lesson 原始标识与 lesson 文本。"""
    from werewolf_agent.runtime.reflection_events import safe_reflection_verification

    if entry_decision_matches:
        source = candidate if isinstance(candidate, Mapping) else {}
        raw_lessons = source.get("verified_lessons")
        normalized_lessons: list[dict[str, str]] = []
        if isinstance(raw_lessons, (list, tuple)):
            for lesson in raw_lessons:
                if not isinstance(lesson, Mapping):
                    continue
                lesson_id = lesson.get("lesson_id")
                abstraction = lesson.get("abstraction")
                if isinstance(lesson_id, str) and isinstance(abstraction, str):
                    normalized_lessons.append({
                        "lesson_id": lesson_id,
                        "abstraction": abstraction,
                    })
        normalized_candidate = {
            "status": _safe_enum(
                source.get("status"), _REFLECTION_VERIFICATION_STATUSES
            ) or "agent_error",
            "decision_id": (
                source.get("decision_id")
                if isinstance(source.get("decision_id"), str)
                else None
            ),
            "verified_fact_count": source.get("verified_fact_count"),
            "verified_claim_ids": (
                source.get("verified_claim_ids")
                if isinstance(source.get("verified_claim_ids"), (list, tuple))
                else []
            ),
            "rejected_claim_ids": (
                source.get("rejected_claim_ids")
                if isinstance(source.get("rejected_claim_ids"), (list, tuple))
                else []
            ),
            "verified_lessons": normalized_lessons,
            "rejected_fact_count": source.get("rejected_fact_count"),
            "rejected_lesson_count": source.get("rejected_lesson_count"),
            "failure_stage": _safe_enum(
                source.get("failure_stage"), _REFLECTION_FAILURE_STAGES
            ),
            "failure_code": (
                source.get("failure_code")
                if isinstance(source.get("failure_code"), str)
                else None
            ),
        }
        safe = safe_reflection_verification(
            normalized_candidate,
            decision_id=decision_id,
        )
    else:
        safe = safe_reflection_verification(
            {
                "status": "agent_error",
                "decision_id": decision_id,
                "failure_stage": "generated",
                "failure_code": "reflection_decision_id_mismatch",
            },
            decision_id=decision_id,
        )
    lessons: list[dict[str, str]] = []
    raw_lessons = safe.get("verified_lessons")
    if isinstance(raw_lessons, list):
        for lesson in raw_lessons:
            if not isinstance(lesson, Mapping):
                continue
            lesson_id = _redacted_reflection_identifier(
                lesson.get("lesson_id"),
                game_id=game_id,
                player_id=player_id,
                identifier_kind="lesson",
            )
            abstraction = lesson.get("abstraction")
            if lesson_id is None or not isinstance(abstraction, str) or not abstraction.strip():
                continue
            lessons.append({
                "lesson_id": lesson_id,
                "abstraction": _REDACTED_LESSON_ABSTRACTION,
            })
    result: dict[str, Any] = {
        "status": safe.get("status"),
        "decision_id": decision_id,
        "verified_fact_count": _safe_non_negative_int(
            safe.get("verified_fact_count")
        ) or 0,
        "verified_claim_ids": _redacted_reflection_identifiers(
            safe.get("verified_claim_ids"),
            game_id=game_id,
            player_id=player_id,
            identifier_kind="claim",
        ),
        "rejected_claim_ids": _redacted_reflection_identifiers(
            safe.get("rejected_claim_ids"),
            game_id=game_id,
            player_id=player_id,
            identifier_kind="claim",
        ),
        "verified_lessons": lessons,
        "rejected_fact_count": _safe_non_negative_int(
            safe.get("rejected_fact_count")
        ) or 0,
        "rejected_lesson_count": _safe_non_negative_int(
            safe.get("rejected_lesson_count")
        ) or 0,
    }
    failure_stage = safe.get("failure_stage")
    failure_code = safe.get("failure_code")
    if (
        isinstance(failure_stage, str)
        and failure_stage in _REFLECTION_FAILURE_STAGES
        and isinstance(failure_code, str)
        and failure_code
    ):
        result["failure_stage"] = failure_stage
        result["failure_code"] = "reflection_failure"
    return result


def _safe_reflection_complete_payload(
    payload: Mapping[str, Any],
    *,
    game_id: str,
    players: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """从权威上下文重建 reflection_complete。"""
    safe_entries: list[dict] = []
    raw_entries = payload.get("entries")
    for entry in raw_entries if isinstance(raw_entries, (list, tuple)) else ():
        if not isinstance(entry, Mapping):
            continue
        authoritative = _authoritative_reflection_player(
            players, entry.get("player_id")
        )
        if authoritative is None:
            continue
        player_id, role, alive = authoritative
        decision_id = f"reflection:{game_id}:{player_id}"
        raw_decision_id = entry.get("decision_id")
        safe_verification = _safe_reflection_verification_payload(
            entry.get("verification"),
            game_id=game_id,
            player_id=player_id,
            decision_id=decision_id,
            entry_decision_matches=raw_decision_id == decision_id,
        )
        transaction_state = _safe_enum(
            entry.get("transaction_state"), _REFLECTION_TRANSACTION_STATES
        )
        entry_id = entry.get("entry_id")
        canonical_entry_id = f"reflection_{game_id}_{player_id}"
        if (
            entry_id != canonical_entry_id
            or not _REFLECTION_ENTRY_ID.fullmatch(canonical_entry_id)
        ):
            entry_id = None
        safe_entries.append({
            "player_id": player_id,
            "role": role,
            "alive": alive,
            "decision_id": decision_id,
            "transaction_state": transaction_state,
            "failure_stage": safe_verification.get("failure_stage"),
            "failure_code": safe_verification.get("failure_code"),
            "entry_id": entry_id,
            "verification": safe_verification,
        })
    status = _safe_enum(payload.get("status"), _REFLECTION_STATUSES)
    persistence_complete = payload.get("persistence_complete")
    if not isinstance(persistence_complete, bool):
        persistence_complete = None
    return {
        "visibility": "moderator_only",
        "status": status,
        "persistence_complete": persistence_complete,
        "player_count": _safe_non_negative_int(payload.get("player_count")),
        "valid_entry_count": _safe_non_negative_int(
            payload.get("valid_entry_count")
        ),
        "failure_count": _safe_non_negative_int(payload.get("failure_count")),
        "entries": safe_entries,
    }


def _safe_reflection_persistence_payload(
    payload: Mapping[str, Any],
    *,
    game_id: str,
    players: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """从权威上下文重建 reflection_persistence_audit。"""
    safe_entries: list[dict[str, Any]] = []
    raw_entries = payload.get("entries")
    for entry in raw_entries if isinstance(raw_entries, (list, tuple)) else ():
        if not isinstance(entry, Mapping):
            continue
        authoritative = _authoritative_reflection_player(
            players, entry.get("player_id")
        )
        if authoritative is None:
            continue
        player_id, _role, _alive = authoritative
        decision_id = f"reflection:{game_id}:{player_id}"
        canonical_entry_id = f"reflection_{game_id}_{player_id}"
        safe_entries.append({
            "player_id": player_id,
            "decision_id": (
                decision_id if entry.get("decision_id") == decision_id else None
            ),
            "verified_claim_ids": _redacted_reflection_identifiers(
                entry.get("verified_claim_ids"),
                game_id=game_id,
                player_id=player_id,
                identifier_kind="claim",
            ),
            "entry_id": (
                canonical_entry_id
                if entry.get("entry_id") == canonical_entry_id
                and _REFLECTION_ENTRY_ID.fullmatch(canonical_entry_id)
                else None
            ),
            "row_found": (
                entry.get("row_found")
                if isinstance(entry.get("row_found"), bool)
                else None
            ),
            "persistence_complete": (
                entry.get("persistence_complete")
                if isinstance(entry.get("persistence_complete"), bool)
                else None
            ),
            "persisted_rejected_fact_count": _safe_non_negative_int(
                entry.get("persisted_rejected_fact_count")
            ),
        })
    status = _safe_enum(payload.get("status"), _REFLECTION_STATUSES)
    return {
        "visibility": "moderator_only",
        "status": status,
        "expected_entry_count": _safe_non_negative_int(
            payload.get("expected_entry_count")
        ),
        "persistence_complete": (
            payload.get("persistence_complete")
            if isinstance(payload.get("persistence_complete"), bool)
            else None
        ),
        "rollback_complete": (
            payload.get("rollback_complete")
            if isinstance(payload.get("rollback_complete"), bool)
            else None
        ),
        "entries": safe_entries,
    }


def _safe_event_payload(
    event_type: str,
    payload: dict,
    *,
    game_id: str | None = None,
    players: Mapping[str, Any] | None = None,
) -> dict:
    """按事件类型和权威局上下文重建可持久化摘要。"""
    if not isinstance(event_type, str) or event_type not in {
        "reflection_complete", "reflection_persistence_audit",
    }:
        return payload if isinstance(event_type, str) else {}
    safe_game_id = _safe_reflection_component(game_id)
    if safe_game_id is None:
        safe_game_id = ""
        players = None
    if event_type == "reflection_complete":
        return _safe_reflection_complete_payload(
            payload,
            game_id=safe_game_id,
            players=players,
        )
    return _safe_reflection_persistence_payload(
        payload,
        game_id=safe_game_id,
        players=players,
    )


def _serialize_event_for_log(
    event: GameEvent,
    *,
    players: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """复用规范 serializer，并在脱敏后维持 V2 顶层 visibility 权威。"""
    serialized = serialize_game_event(event)
    # 先使用规范 serializer 深拷贝并递归转换批次，再对副本做脱敏。
    safe_payload = dict(_safe_event_payload(
        event.type,
        serialized["payload"],
        game_id=event.game_id,
        players=players,
    ))
    if event.schema_version == "2" or event.visibility is not None:
        safe_payload.pop("visibility", None)
    serialized["payload"] = safe_payload
    return serialized


def _serialize_projected_event_for_log(
    event: dict[str, Any],
    *,
    game_id: str,
    players: Mapping[str, Any],
) -> dict[str, Any]:
    """仅从固定投影生成脱敏日志事件，不回读可变 runner 状态。"""
    serialized = dict(event)
    event_game_id = serialized.get("game_id")
    context_game_id = (
        game_id
        if event_game_id is None
        or (isinstance(event_game_id, str) and event_game_id == game_id)
        else ""
    )
    event_type = serialized.get("type")
    safe_payload = dict(_safe_event_payload(
        event_type if isinstance(event_type, str) else "",
        dict(serialized.get("payload") or {}),
        game_id=context_game_id,
        players=players,
    ))
    if serialized.get("schema_version") == "2" or serialized.get("visibility") is not None:
        safe_payload.pop("visibility", None)
    serialized["payload"] = safe_payload
    return serialized


def _source_events_for_log_sanitization(
    source: AcceptanceGameProjection | Mapping[str, Any] | Any,
    projected_events: list[dict[str, Any]],
) -> tuple[list[Any], bool]:
    """优先读取原始事件，使首次投影丢弃的敏感反思字段仍可安全重建。"""
    if isinstance(source, AcceptanceGameProjection):
        return projected_events, False
    raw_events = (
        source.get("events", ())
        if isinstance(source, Mapping)
        else getattr(source, "events", ())
    )
    if not isinstance(raw_events, (list, tuple)):
        return projected_events, False
    serialized: list[Any] = []
    try:
        for event in raw_events:
            serialized.append(
                serialize_game_event(event)
                if isinstance(event, GameEvent)
                else dict(event) if isinstance(event, Mapping)
                else event
            )
    except (AttributeError, TypeError, ValueError):
        return projected_events, False
    return serialized, True


def _source_without_events_is_supported(
    source: Mapping[str, Any] | Any,
    *,
    steps: int,
) -> bool:
    """确认 invalid_event_payload 没有遮蔽玩家、死亡等第二结构错误。"""
    try:
        if isinstance(source, Mapping):
            eventless_source = dict(source)
            eventless_source["events"] = []
            eventless_source.pop("_acceptance_projection_supported", None)
            eventless_source.pop("_acceptance_projection_unsupported_reason", None)
        elif is_dataclass(source):
            eventless_source = replace(source, events=[])
        else:
            return False
    except (AttributeError, TypeError, ValueError):
        return False
    return project_acceptance_game(eventless_source, steps=steps).supported


def sanitize_projected_game_for_log(
    source: AcceptanceGameProjection | Mapping[str, Any] | Any,
    *,
    steps: int | None = None,
) -> AcceptanceGameProjection:
    """生成评分与落盘共用的不可变、脱敏验收投影。"""
    baseline = project_acceptance_game(source, steps=steps)
    projected = baseline.to_mapping()
    raw_events, read_raw_events = _source_events_for_log_sanitization(
        source,
        projected["events"],
    )
    safe_events: list[Any] = []
    for event in raw_events:
        if not isinstance(event, Mapping):
            safe_events.append(event)
            continue
        event_mapping = dict(event)
        try:
            safe_events.append(_serialize_projected_event_for_log(
                event_mapping,
                game_id=baseline.game_id,
                players=projected["players"],
            ))
        except (AttributeError, TypeError, ValueError):
            # 非反思事件仍交回统一投影验证器 fail closed，不能静默删除。
            safe_events.append(event_mapping)
    projected["events"] = safe_events
    if (
        read_raw_events
        and baseline.unsupported_reason == "invalid_event_payload"
        and _source_without_events_is_supported(source, steps=baseline.steps)
    ):
        projected.pop("_acceptance_projection_supported", None)
        projected.pop("_acceptance_projection_unsupported_reason", None)
    sanitized = project_acceptance_game(projected, steps=baseline.steps)
    if (
        isinstance(source, AcceptanceGameProjection)
        and sanitized.to_mapping() == baseline.to_mapping()
    ):
        return source
    return sanitized


def _categorical_speech_observation(
    traces: list[dict[str, Any]],
    *,
    field: str,
    allowed: frozenset[str],
    positive: frozenset[str],
    missing_reason: str,
    invalid_reason: str,
) -> dict[str, Any]:
    """按单一封闭枚举字段计算指标，缺失和非法值分别 fail closed。"""
    values = [trace.get(field) for trace in traces]
    observed = [value for value in values if isinstance(value, str) and value in allowed]
    if any(
        value is not None
        and (not isinstance(value, str) or value not in allowed)
        for value in values
    ):
        return {"supported": False, "reason": invalid_reason,
                "observed_count": len(observed), "rate": None}
    if not traces or len(observed) != len(traces):
        return {"supported": False, "reason": missing_reason,
                "observed_count": len(observed), "rate": None}
    return {
        "supported": True,
        "reason": None,
        "observed_count": len(observed),
        "rate": round(sum(value in positive for value in observed) / len(observed), 3),
    }


def save_game_log(
    _runner: GameRunner,
    elapsed: float,
    *,
    projection: AcceptanceGameProjection,
    quality_score: dict[str, Any],
    output_dir: str | Path | None = None,
) -> Path:
    """保存单局 JSON；显式目录用于隔离批量验收产物。"""
    sanitized_projection = sanitize_projected_game_for_log(projection)
    if sanitized_projection is not projection:
        quality_score = compute_game_quality_score(sanitized_projection)
    projection = sanitized_projection
    if not _SAFE_GAME_ID.fullmatch(projection.game_id) or projection.game_id in {".", ".."}:
        raise ValueError(f"invalid game_id: {projection.game_id!r}")
    quality = normalize_quality_score(quality_score)
    quality.pop("speech_fill_rate", None)
    is_low_quality = (
        quality.get("fallback_metrics_supported") is not False
        and isinstance(quality.get("fallback_rate"), (int, float))
        and isinstance(quality.get("total_quality_events"), int)
        and quality["fallback_rate"] > 0.7
        and quality["total_quality_events"] > 5
    )
    artifact_root = (Path(output_dir) if output_dir is not None else ROOT).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    if is_low_quality:
        low_q_dir = artifact_root / "low_quality_games"
        low_q_dir.mkdir(parents=True, exist_ok=True)
        log_path = _contained_game_log_path(
            artifact_root, low_q_dir, projection.game_id,
        )
        logger.warning(
            "Low quality game (fallback_rate=%.1f%%, %d quality events) — saved to %s",
            quality["fallback_rate"] * 100, quality["total_quality_events"], log_path,
        )
    else:
        log_path = _contained_game_log_path(
            artifact_root, artifact_root, projection.game_id,
        )

    projected = projection.to_mapping()
    log_data = {
        "game_id": projection.game_id,
        "winning_faction": projection.winning_faction,
        "status": projection.status,
        "termination_reason": projection.termination_reason,
        "phase": projected.get("phase"),
        "day_number": projected.get("day_number", 0),
        "night_number": projected.get("night_number", 0),
        "hybrid_master_id": projected.get("hybrid_master_id"),
        "hybrid_master_faction": projected.get("hybrid_master_faction"),
        "hybrid_result": projected.get("hybrid_result"),
        "players": projected["players"],
        "deaths": projected["deaths"],
        "events": projected["events"],
        "_acceptance_projection_supported": projection.supported,
        "_acceptance_projection_unsupported_reason": projection.unsupported_reason,
        "_acceptance_events_supported": projection.events_supported,
        "_acceptance_events_unsupported_reason": projection.events_unsupported_reason,
        "elapsed_seconds": round(elapsed, 1),
        "steps": projection.steps,
        "quality_score": quality,
    }
    _atomic_write_json(log_path, log_data, trusted_root=artifact_root)
    return log_path


def _resolve_within_output_root(
    trusted_root: Path, candidate: Path,
) -> Path:
    """解析 junction/symlink 后仍要求候选路径位于原始信任根。"""
    root = trusted_root
    resolved = candidate.resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise ValueError("artifact path is outside output_dir")
    return resolved


def _contained_game_log_path(
    trusted_root: Path,
    directory: Path,
    game_id: str,
) -> Path:
    """相对原始 output_dir 信任根构造并复核日志路径。"""
    parent = _resolve_within_output_root(trusted_root, directory)
    return _resolve_within_output_root(
        trusted_root, parent / f"game_{game_id}.json",
    )


def _atomic_write_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    trusted_root: Path,
) -> None:
    """在已验证父目录写临时文件并原子替换，失败时保留旧文件。"""
    encoded = json.dumps(value, ensure_ascii=False, indent=2)
    target = _resolve_within_output_root(trusted_root, path)
    parent = _resolve_within_output_root(trusted_root, target.parent)
    temporary = _resolve_within_output_root(
        trusted_root,
        parent / f"{target.name}.{uuid.uuid4().hex}.tmp",
    )
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def finalize_game_log(
    runner: GameRunner,
    elapsed: float,
    *,
    output_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """在赛后反思与持久化审计完成后固定投影、评分一次并保存。"""
    projection = sanitize_projected_game_for_log(
        runner.state,
        steps=runner.step_count,
    )
    quality = compute_game_quality_score(projection)
    path = save_game_log(
        runner,
        elapsed,
        projection=projection,
        quality_score=quality,
        output_dir=output_dir,
    )
    return path, quality


# ── main ─────────────────────────────────────────────────────────────────

def _build_argument_parser() -> argparse.ArgumentParser:
    """构建真实游戏 CLI 参数，供入口和无副作用测试复用。"""
    parser = argparse.ArgumentParser(description="Run a real 12-player werewolf game")
    parser.add_argument("--seed", type=int, default=None, help="Game seed (default: auto)")
    parser.add_argument(
        "--game-id",
        default="",
        help="Explicit run-scoped game ID (default: g_<seed>)",
    )
    parser.add_argument("--max-steps", type=int, default=500, help="Max graph steps")
    parser.add_argument("--delay", type=int, default=0, help="Inter-call delay ms (0=random 3-6s, >0=fixed, <0=none)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for this game's JSON artifact (default: repository root)",
    )
    return parser


def _build_runner_config(
    args: argparse.Namespace,
    *,
    game_repo,
    memory_coordinator,
) -> GameRunnerConfig:
    """将 CLI 参数显式映射到 GameRunner 配置。"""
    return GameRunnerConfig(
        ruleset_id="pre_witch_hunter_idiot_mixed",
        player_count=12,
        seed=args.seed,
        game_id=args.game_id,
        use_agent_registry=True,
        model_config_path=str(ROOT / "config" / "models.yaml"),
        persona_config_path=str(
            ROOT / "config" / "personas" / "jingcheng_style_prototypes.yaml"
        ),
        repository=game_repo,
        memory_coordinator=memory_coordinator,
        agent_call_delay_ms=args.delay,
        emergency_artifact_dir=(
            args.output_dir
            if args.output_dir is not None
            else ROOT / "artifacts" / "emergency_game_aborts"
        ),
    )


def log_terminal_outcome(
    runner: GameRunner,
    elapsed: float,
    quality: dict[str, Any],
) -> int:
    """记录可机读终态，中止局返回非零退出码。"""
    gs = runner.state
    if gs.status == "finished":
        fallback_rate = quality.get("fallback_rate")
        fallback_display = (
            f"{fallback_rate:.3f}"
            if isinstance(fallback_rate, (int, float))
            else "unsupported"
        )
        logger.info(
            "GAME_COMPLETE winner=%s day=%d night=%d steps=%d elapsed=%.1f fallback_rate=%s",
            gs.winning_faction, gs.day_number, gs.night_number,
            runner.step_count, elapsed, fallback_display,
        )
        return 0
    logger.error(
        "GAME_ABORTED reason=%s phase=%s steps=%d elapsed=%.1f",
        gs.termination_reason, gs.phase, runner.step_count, elapsed,
    )
    return 1


def main() -> None:
    _configure_file_logging()
    args = _build_argument_parser().parse_args()

    load_local_dotenv(ROOT / ".env")

    from werewolf_agent.model_gateway.providers import get_env
    api_key = get_env("ANTHROPIC_API_KEY") or get_env("GLM_API_KEY") or ""
    base_url = get_env("ANTHROPIC_BASE_URL") or get_env("GLM_BASE_URL") or ""
    if not api_key:
        print("ERROR: No API key found. Set ANTHROPIC_API_KEY or GLM_API_KEY in .env")
        sys.exit(1)

    _sep("WEREWOLF AGENT - REAL LLM GAME")
    print(f"  API endpoint: {base_url}")
    print(f"  API key:      {_format_api_key_status(api_key)}")
    print(f"  Seed:         {args.seed or 'auto'}")
    print(f"  Max steps:    {args.max_steps}")

    # Connectivity test
    print("\n  Testing API connectivity...")
    from werewolf_agent.model_gateway.router import ModelRouter
    router = ModelRouter.from_yaml(ROOT / "config" / "models.yaml", register_env_providers=True)
    providers = router.provider_names()
    print(f"  Providers: {providers}")

    if not providers:
        print("ERROR: No providers registered. Check .env API keys.")
        sys.exit(1)

    _test_agent = next(
        (pid for pid in router._player_assignments if pid != "judge"), "p01"
    )
    _test_profile = router._player_assignments.get(_test_agent, "?")
    _test_llm = router._llm_profiles.get(_test_profile, {})
    _test_default = _test_llm.get("default", {})
    _test_model_profile = _test_default.get("model_profile", "?")
    _test_model_cfg = router._model_profiles.get(_test_model_profile, {})
    print(f"  Test agent:  {_test_agent} (profile={_test_profile})")
    print(f"  Test model:  provider={_test_default.get('provider','?')} model={_test_model_cfg.get('model','?')} timeout={_test_model_cfg.get('timeout','?')}s")
    print("  Calling API...", flush=True)

    test_result = router.generate(
        agent_id=_test_agent,
        task_type="speech",
        prompt='Visible state: {"phase": "night", "alive_players": ["p01", "p02"]}\nRespond with valid JSON:',
        system_prompt="You are a player in a Werewolf game. Output ONLY valid JSON.",
    )
    if test_result.text:
        print(f"  API OK ({test_result.provider}/{test_result.model}): {test_result.text[:100]}...")
    else:
        print(f"  WARNING: API returned empty text (provider={test_result.provider}, model={test_result.model})")

    _sep("STARTING GAME")
    start = time.monotonic()

    # Persistent memory: Docker PostgreSQL + coordinator for cross-game learning
    memory_coordinator = None
    game_repo = None
    try:
        import subprocess
        print("  Starting Docker PostgreSQL (30s timeout)...", flush=True)
        subprocess.run(
            ["docker", "compose", "up", "-d", "postgres"],
            cwd=ROOT, capture_output=True, check=False,
            timeout=30,
        )
        from werewolf_agent.storage.postgres_store import PostgresGameRepository
        from werewolf_agent.storage.persistent_memory import PersistentMemoryCoordinator
        db_dsn = os.getenv("WOFKILL_PG_DSN", "postgresql://wofkill:wofkill-dev@localhost:5432/wofkill")
        game_repo = PostgresGameRepository(db_dsn)
        memory_coordinator = PersistentMemoryCoordinator(game_repo)
        print("  Memory DB: PostgreSQL (via Docker)")
        if game_repo.load_rag_entries():
            print("  RAG entries: restored from previous session")
    except Exception:
        print("  Memory DB: disabled (Docker PostgreSQL unavailable)")

    config = _build_runner_config(
        args,
        game_repo=game_repo,
        memory_coordinator=memory_coordinator,
    )

    runner = GameRunner(config)
    print(f"  Game ID: {runner.game_id}")
    n_agents = len(runner._agent_registry._agents) if runner._agent_registry else 0
    print(f"  Agents:  {n_agents}")

    runner.run(max_steps=args.max_steps)
    elapsed = time.monotonic() - start

    outcome_label = "Finished" if runner.state.status == "finished" else "Aborted"
    print(f"\n  {outcome_label} in {elapsed:.1f}s ({runner.step_count} steps)")

    print_game_summary(runner)
    print_usage_stats(runner)
    print_pace_report(runner)
    print_quality_audit(runner)
    check_leakage(runner)

    log_path, quality = finalize_game_log(
        runner,
        elapsed,
        output_dir=args.output_dir,
    )
    exit_code = log_terminal_outcome(runner, elapsed, quality)

    print(f"\n  Game log: {log_path}")

    # Also generate audit markdown if script exists
    audit_script = ROOT / "scripts" / "print_game_audit.py"
    if audit_script.exists():
        print(f"  Audit:    python {audit_script} {log_path}")
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
