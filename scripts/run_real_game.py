# -*- coding: utf-8 -*-
"""
运行一局由 LLM 智能体参与的 12 人狼人杀真实游戏。

作者: Project contributors
修改日期: 2026-07-13

使用示例:
    python scripts/run_real_game.py --seed 42 --max-steps 500
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from werewolf_agent.model_gateway.providers import load_local_dotenv
from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig
from scripts.run_real_game_reports import (
    _sep,
    check_leakage,
    print_game_summary,
    print_pace_report,
    print_quality_audit,
    print_usage_stats,
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
def compute_game_quality_score(runner: GameRunner) -> dict[str, Any]:
    """Compute structured quality metrics for a completed game."""
    gs = runner.state
    traces = [e for e in gs.events if e.type == "action_trace_audit"]
    action_fallback_traces = [
        e.payload.get("action_trace", {}) for e in traces
        if e.payload.get("action_trace", {}).get("fallback_reason")
    ]
    wolf_plan_fallbacks = [
        e for e in gs.events if e.type == "wolf_team_plan_fallback"
    ]
    action_fallback_count = sum(
        1 for e in traces
        if e.payload.get("action_trace", {}).get("fallback_reason")
    )
    wolf_plan_fallback_count = len(wolf_plan_fallbacks)
    wolf_plan_attempts = max(
        sum(1 for e in gs.events if e.type == "wolf_team_plan"),
        wolf_plan_fallback_count,
    )
    structured_fail = sum(
        1 for e in traces
        if e.payload.get("action_trace", {}).get("structured_failure_reason")
    )
    total = len(traces)
    total_quality_events = total + wolf_plan_attempts
    fallback_count = action_fallback_count + wolf_plan_fallback_count
    fallback_rate = fallback_count / total_quality_events if total_quality_events > 0 else 0.0
    speeches = [e for e in gs.events if e.type == "speech"]
    non_empty_speeches = sum(1 for e in speeches if e.payload.get("text", "").strip())
    speech_rate = non_empty_speeches / len(speeches) if speeches else 0.0
    phases_seen = {e.payload.get("phase") for e in gs.events if e.type == "judge_broadcast"}
    has_winner = bool(gs.winning_faction)
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
        "total_quality_events": total_quality_events,
        "speech_count": len(speeches),
        "non_empty_speech_count": non_empty_speeches,
        "speech_fill_rate": round(speech_rate, 3),
        "phases_seen": len(phases_seen),
        "has_winner": has_winner,
        "steps": runner.step_count,
    }


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


def save_game_log(runner: GameRunner, elapsed: float) -> Path:
    gs = runner.state
    quality = compute_game_quality_score(runner)
    is_low_quality = quality["fallback_rate"] > 0.7 and quality["total_quality_events"] > 5

    if is_low_quality:
        low_q_dir = ROOT / "low_quality_games"
        low_q_dir.mkdir(exist_ok=True)
        log_path = low_q_dir / f"game_{runner.game_id}.json"
        logger.warning(
            "Low quality game (fallback_rate=%.1f%%, %d quality events) — saved to %s",
            quality["fallback_rate"] * 100, quality["total_quality_events"], log_path,
        )
    else:
        log_path = ROOT / f"game_{runner.game_id}.json"

    log_data = {
        "game_id": gs.game_id,
        "winning_faction": gs.winning_faction,
        "phase": gs.phase,
        "day_number": gs.day_number,
        "night_number": gs.night_number,
        **_final_hybrid_fields(gs),
        "players": {pid: {"role": p.role, "alive": p.alive} for pid, p in gs.players.items()},
        "deaths": [
            {
                "player_id": d.player_id,
                "reason": d.reason,
                "timing": d.timing,
                "resolution_batch": d.resolution_batch,
                "source_player_id": d.source_player_id,
                "can_leave_last_words": d.can_leave_last_words,
                "triggered_skills": list(d.triggered_skills),
            }
            for d in gs.deaths
        ],
        "events": [{"type": e.type, "payload": e.payload} for e in gs.events],
        "elapsed_seconds": round(elapsed, 1),
        "steps": runner.step_count,
        "quality_score": quality,
    }
    log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return log_path


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    _configure_file_logging()
    parser = argparse.ArgumentParser(description="Run a real 12-player werewolf game")
    parser.add_argument("--seed", type=int, default=None, help="Game seed (default: auto)")
    parser.add_argument("--max-steps", type=int, default=500, help="Max graph steps")
    parser.add_argument("--timeout", type=float, default=120.0, help="Agent timeout (seconds)")
    parser.add_argument("--no-timeout", action="store_true", help="Disable agent timeout")
    parser.add_argument("--delay", type=int, default=0, help="Inter-call delay ms (0=random 3-6s, >0=fixed, <0=none)")
    args = parser.parse_args()

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

    config = GameRunnerConfig(
        ruleset_id="pre_witch_hunter_idiot_mixed",
        player_count=12,
        seed=args.seed,
        use_agent_registry=True,
        model_config_path=str(ROOT / "config" / "models.yaml"),
        persona_config_path=str(ROOT / "config" / "personas" / "jingcheng_style_prototypes.yaml"),
        agent_call_timeout=0 if args.no_timeout else args.timeout,
        repository=game_repo,
        memory_coordinator=memory_coordinator,
        agent_call_delay_ms=args.delay,
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

    quality = compute_game_quality_score(runner)
    gs = runner.state
    logger.info(
        "GAME_COMPLETE winner=%s day=%d night=%d steps=%d elapsed=%.1f fallback_rate=%.3f",
        gs.winning_faction, gs.day_number, gs.night_number,
        runner.step_count, elapsed, quality["fallback_rate"],
    )

    log_path = save_game_log(runner, elapsed)
    print(f"\n  Game log: {log_path}")

    # Also generate audit markdown if script exists
    audit_script = ROOT / "scripts" / "print_game_audit.py"
    if audit_script.exists():
        print(f"  Audit:    python {audit_script} {log_path}")


if __name__ == "__main__":
    main()
