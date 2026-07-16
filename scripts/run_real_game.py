# -*- coding: utf-8 -*-
"""
运行一局由 LLM 智能体参与的 12 人狼人杀真实游戏。

作者: Project contributors
修改日期: 2026-07-16

使用示例:
    python scripts/run_real_game.py --seed 42 --max-steps 500
"""

from __future__ import annotations

import argparse
from collections import Counter
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
        "fallback_rate": round(fallback_rate, 3),
        "fallback_count": fallback_count,
        "action_fallback_count": action_fallback_count,
        "wolf_team_plan_fallback_count": wolf_plan_fallback_count,
        "fallback_by_reason": dict(sorted(fallback_by_reason.items())),
        "fallback_by_stage": dict(sorted(fallback_by_stage.items())),
        "action_fallback_by_error_code": dict(
            sorted(action_fallback_by_error_code.items())
        ),
        "retry_error_counts": dict(sorted(retry_error_counts.items())),
        "wolf_team_plan_fallback_by_reason": dict(
            sorted(wolf_team_plan_fallback_by_reason.items())
        ),
        "structured_fail_count": structured_fail,
        "total_action_traces": total,
        "total_wolf_team_plans": wolf_plan_attempts,
        **wolf_plan_outcomes,
        **acceptance_metrics,
        "total_quality_events": total_quality_events,
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
                rows.append({
                    "task_type": task_type,
                    "player_id": speaker,
                    "text": event.payload.get("text"),
                    "action_trace": None,
                })
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


def _safe_event_payload(event_type: str, payload: dict) -> dict:
    """过滤日志中的反思原始草稿和 provider 响应，仅保留核验摘要。"""
    if event_type != "reflection_complete":
        return payload
    from werewolf_agent.runtime.reflection_events import safe_reflection_verification

    safe_entries: list[dict] = []
    for entry in payload.get("entries", []):
        if not isinstance(entry, dict):
            continue
        verification = entry.get("verification", {})
        player_id = str(entry.get("player_id") or "")
        decision_id = entry.get("decision_id")
        if not isinstance(decision_id, str) or not decision_id:
            decision_id = f"legacy-reflection:{player_id}"
        safe_verification = safe_reflection_verification(
            verification,
            decision_id=decision_id,
        )
        safe_entries.append({
            "player_id": player_id,
            "role": str(entry.get("role") or ""),
            "alive": bool(entry.get("alive", False)),
            "decision_id": safe_verification["decision_id"],
            "verification": safe_verification,
        })
    safe_payload = {
        "visibility": "moderator_only",
        "player_count": int(payload.get("player_count") or len(safe_entries)),
        "entries": safe_entries,
    }
    return safe_payload


def _serialize_event_for_log(event: GameEvent) -> dict[str, Any]:
    """复用规范 serializer，并在脱敏后维持 V2 顶层 visibility 权威。"""
    serialized = serialize_game_event(event)
    # 先使用规范 serializer 深拷贝并递归转换批次，再对副本做脱敏。
    safe_payload = dict(_safe_event_payload(event.type, serialized["payload"]))
    if event.schema_version == "2" or event.visibility is not None:
        safe_payload.pop("visibility", None)
    serialized["payload"] = safe_payload
    return serialized


def _serialize_projected_event_for_log(event: dict[str, Any]) -> dict[str, Any]:
    """仅从固定投影生成脱敏日志事件，不回读可变 runner 状态。"""
    serialized = dict(event)
    safe_payload = dict(_safe_event_payload(
        str(serialized.get("type") or ""),
        dict(serialized.get("payload") or {}),
    ))
    if serialized.get("schema_version") == "2" or serialized.get("visibility") is not None:
        safe_payload.pop("visibility", None)
    serialized["payload"] = safe_payload
    return serialized


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
    if not _SAFE_GAME_ID.fullmatch(projection.game_id) or projection.game_id in {".", ".."}:
        raise ValueError(f"invalid game_id: {projection.game_id!r}")
    quality = normalize_quality_score(quality_score)
    quality.pop("speech_fill_rate", None)
    is_low_quality = quality["fallback_rate"] > 0.7 and quality["total_quality_events"] > 5
    artifact_root = (Path(output_dir) if output_dir is not None else ROOT).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    if is_low_quality:
        low_q_dir = artifact_root / "low_quality_games"
        low_q_dir.mkdir(parents=True, exist_ok=True)
        log_path = _contained_game_log_path(low_q_dir, projection.game_id)
        logger.warning(
            "Low quality game (fallback_rate=%.1f%%, %d quality events) — saved to %s",
            quality["fallback_rate"] * 100, quality["total_quality_events"], log_path,
        )
    else:
        log_path = _contained_game_log_path(artifact_root, projection.game_id)

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
        "events": [
            _serialize_projected_event_for_log(event)
            for event in projected["events"]
        ],
        "elapsed_seconds": round(elapsed, 1),
        "steps": projection.steps,
        "quality_score": quality,
    }
    _atomic_write_json(log_path, log_data)
    return log_path


def _contained_game_log_path(directory: Path, game_id: str) -> Path:
    """构造并复核日志路径始终位于指定目录。"""
    root = directory.resolve()
    target = (root / f"game_{game_id}.json").resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"invalid game_id: {game_id!r}")
    return target


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """同目录写临时文件并原子替换，失败时保留旧文件。"""
    encoded = json.dumps(value, ensure_ascii=False, indent=2)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def finalize_game_log(
    runner: GameRunner,
    elapsed: float,
    *,
    output_dir: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """在赛后反思与持久化审计完成后固定投影、评分一次并保存。"""
    projection = project_acceptance_game(runner.state, steps=runner.step_count)
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
    parser.add_argument("--timeout", type=float, default=120.0, help="Agent timeout (seconds)")
    parser.add_argument("--no-timeout", action="store_true", help="Disable agent timeout")
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
        agent_call_timeout=0 if args.no_timeout else args.timeout,
        repository=game_repo,
        memory_coordinator=memory_coordinator,
        agent_call_delay_ms=args.delay,
    )


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
    print(f"  Timeout:      {'disabled' if args.no_timeout else f'{args.timeout}s'}")

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
    print(f"  Calling API (this may take up to {int((_test_model_cfg.get('timeout') or 60) * 3)}s with retries)...", flush=True)

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

    print(f"\n  Finished in {elapsed:.1f}s ({runner.step_count} steps)")

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
    gs = runner.state
    logger.info(
        "GAME_COMPLETE winner=%s day=%d night=%d steps=%d elapsed=%.1f fallback_rate=%.3f",
        gs.winning_faction, gs.day_number, gs.night_number,
        runner.step_count, elapsed, quality["fallback_rate"],
    )

    print(f"\n  Game log: {log_path}")

    # Also generate audit markdown if script exists
    audit_script = ROOT / "scripts" / "print_game_audit.py"
    if audit_script.exists():
        print(f"  Audit:    python {audit_script} {log_path}")


if __name__ == "__main__":
    main()
