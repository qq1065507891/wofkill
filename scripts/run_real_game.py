"""Run a real 12-player werewolf game with LLM agents.

Usage:
    python scripts/run_real_game.py [--seed 42] [--max-steps 500] [--timeout 120]

Requires .env with ANTHROPIC_API_KEY (or GLM_API_KEY) configured.
"""

from __future__ import annotations

import argparse
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "game_stdout.log", encoding="utf-8"),
    ],
)
# Game-step detail: graph module at DEBUG so role assignments, night actions,
# speeches, votes, etc. are all visible. Suppress noisy httpx.
logging.getLogger("werewolf_agent.runtime.nodes").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("real_game")


# ── helpers ──────────────────────────────────────────────────────────────

def _format_api_key_status(api_key: str) -> str:
    return "configured" if api_key else "missing"


def _sep(title: str = "") -> None:
    line = "=" * 60
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def _role_of(runner: GameRunner, pid: str) -> str:
    p = runner.state.players.get(pid)
    return p.role if p else "?"


# ── summary ──────────────────────────────────────────────────────────────

def print_game_summary(runner: GameRunner) -> None:
    gs = runner.state
    _sep("GAME SUMMARY")
    print(f"  Game ID:      {gs.game_id}")
    print(f"  Ruleset:      {gs.ruleset_id}")
    print(f"  Phase:        {gs.phase}")
    print(f"  Winner:       {gs.winning_faction or 'N/A'}")
    print(f"  Day / Night:  {gs.day_number} / {gs.night_number}")
    print(f"  Steps:        {runner.step_count}")
    print(f"  Deaths:       {len(gs.deaths)}")

    print("\n  Players:")
    for pid, p in sorted(gs.players.items()):
        status = "ALIVE" if p.alive else "DEAD "
        print(f"    {pid}: {p.role:12s} [{status}]")

    print("\n  Death Log:")
    for d in gs.deaths:
        print(f"    {d.player_id} ({_role_of(runner, d.player_id)}): {d.reason}")

    if gs.sheriff_id:
        print(f"\n  Sheriff:        {gs.sheriff_id}")
    print(f"  Badge state:    {gs.sheriff_badge_state}")
    print(f"  Antidote used:  {gs.antidote_used}")
    print(f"  Poison used:    {gs.poison_used}")

    # Last 30 events
    print(f"\n  Events ({len(gs.events)} total, last 30):")
    for ev in gs.events[-30:]:
        vis = ev.payload.get("visibility", "public") if ev.payload else ""
        tag = ev.type
        if ev.payload:
            if "speech" in ev.type:
                tag = f"[{ev.payload.get('speaker', '?')}] {ev.payload.get('text', '')[:80]}"
            elif ev.type == "vote_resolved":
                tag = f"exiled={ev.payload.get('exiled')} reason={ev.payload.get('reason')}"
            elif "wolf_kill" in ev.type:
                tag = f"target={ev.payload.get('target_id', '?')}"
            elif ev.type == "seer_check":
                tag = f"{ev.payload.get('seer_id')} -> {ev.payload.get('target_id')}: {ev.payload.get('alignment')}"
            elif "witch" in ev.type or "badge" in ev.type or "hybrid" in ev.type:
                tag = str(ev.payload)[:100]
        if vis and vis != "public":
            tag += f" [{vis}]"
        print(f"    {ev.type:30s} {tag}")
    _sep()


# ── usage stats ──────────────────────────────────────────────────────────

def print_usage_stats(runner: GameRunner) -> None:
    registry = runner._agent_registry
    if registry is None:
        return
    sample_agent = next(iter(registry._agents.values()), None)
    if sample_agent is None:
        return
    usage_log = sample_agent.model_router.get_usage_log()
    if not usage_log:
        return

    _sep("API USAGE")
    total = len(usage_log)
    ok = sum(1 for u in usage_log if u.success)
    fail = total - ok
    p_tok = sum(u.prompt_tokens for u in usage_log)
    c_tok = sum(u.completion_tokens for u in usage_log)
    lat = sum(u.latency_ms for u in usage_log)

    print(f"  Calls: {total}  (ok {ok}, fail {fail})")
    print(f"  Tokens: {p_tok:,} prompt + {c_tok:,} completion = {p_tok + c_tok:,}")
    print(f"  Latency: {lat / 1000:.1f}s total")

    # Per-agent
    stats: dict[str, dict] = {}
    for u in usage_log:
        s = stats.setdefault(u.agent_id, {"calls": 0, "tokens": 0, "lat": 0})
        s["calls"] += 1
        s["tokens"] += u.prompt_tokens + u.completion_tokens
        s["lat"] += u.latency_ms
    print("\n  Per-agent:")
    for aid, s in sorted(stats.items()):
        print(f"    {aid}: {s['calls']} calls, {s['tokens']:,} tok, {s['lat'] / 1000:.1f}s")

    failures = {}
    for u in usage_log:
        if not u.success:
            r = u.fallback_reason or "unknown"
            failures[r] = failures.get(r, 0) + 1
    if failures:
        print("\n  Failures:")
        for r, c in sorted(failures.items(), key=lambda x: -x[1]):
            print(f"    {c}x {r}")
    _sep()


# ── pace report ──────────────────────────────────────────────────────────

def print_pace_report(runner: GameRunner) -> None:
    from werewolf_agent.evaluation.metrics import compute_pace_metrics

    gs = runner.state
    _sep("PACE REPORT")
    ev = [{"type": e.type, "payload": e.payload} for e in gs.events]
    dd = [{"player_id": d.player_id, "reason": d.reason} for d in gs.deaths]
    metrics = compute_pace_metrics(ev, deaths=dd, finish_night=gs.night_number)

    wolf_kills = sum(1 for d in gs.deaths if d.reason == "wolf_kill")
    exiles = sum(1 for d in gs.deaths if d.reason == "exile")
    shots = sum(1 for d in gs.deaths if d.reason == "hunter_shot")

    print(f"  Winner:          {gs.winning_faction or 'N/A'}")
    print(f"  Finish:          day {gs.day_number} / night {gs.night_number}")
    print(f"  Deaths:          {len(gs.deaths)} (wolf {wolf_kills}, exile {exiles}, shot {shots})")
    print(f"  Day exile rate:  {metrics['day_exile_rate']:.1%}")
    print(f"  2nd-tie count:   {metrics['second_tie_count']}")
    print(f"  Max no-exile:    {metrics['max_consecutive_no_exile_days']}")
    print(f"  Stale votes:     {metrics['stale_vote_reuse_count']}")
    print(f"  Pace OK:         {'YES' if metrics['pace_target_met'] else 'NO'}")
    _sep()


# ── quality audit (new modules) ──────────────────────────────────────────

def print_quality_audit(runner: GameRunner) -> None:
    """Audit game output against the new quality modules."""
    from werewolf_agent.runtime.speech_quality import validate_public_speech
    from werewolf_agent.runtime.vote_quality import validate_vote_reason
    gs = runner.state
    _sep("QUALITY AUDIT")

    # Speech quality
    speeches = [e for e in gs.events if e.type == "speech"]
    filler_count = 0
    short_count = 0
    for e in speeches:
        text = e.payload.get("text", "")
        if not text or len(text) < 10:
            short_count += 1
            continue
        r = validate_public_speech(text, phase="day_discussion")
        if not r["valid"]:
            filler_count += 1

    total_speeches = len(speeches)
    ok_speeches = total_speeches - filler_count - short_count
    print(f"  Speeches: {total_speeches} total, {ok_speeches} ok, {filler_count} filler, {short_count} empty")

    # Vote quality
    vote_events = [e for e in gs.events if e.type == "vote_resolved"]
    votes_with_basis = 0
    votes_without_basis = 0
    # Check vote action traces for basis
    for e in gs.events:
        if e.type == "action_trace_audit" and e.payload.get("phase") == "vote":
            trace = e.payload.get("action_trace", {})
            parsed_action = trace.get("parsed_action") or {}
            reason = parsed_action.get("reason", "")
            if reason:
                from werewolf_agent.runtime.vote_quality import extract_vote_basis
                bases = extract_vote_basis(reason)
                if bases:
                    votes_with_basis += 1
                else:
                    votes_without_basis += 1

    print(f"  Votes with basis:    {votes_with_basis}")
    print(f"  Votes without basis: {votes_without_basis}")

    # Judge broadcasts
    broadcasts = [e for e in gs.events if e.type == "judge_broadcast"]
    phases = {e.payload.get("phase") for e in broadcasts}
    expected = {
        "enter_night", "day_announce", "wolf_discussion_start",
        "wolf_kill_choice", "seer_wake", "witch_wake", "vote_start",
    }
    missing_broadcasts = expected - phases
    print(f"  Judge broadcasts:    {len(broadcasts)} ({len(phases)} unique phases)")
    if missing_broadcasts:
        print(f"  MISSING broadcasts:  {missing_broadcasts}")

    # Wolf discussion quality
    wolf_disc = [e for e in gs.events if e.type == "wolf_discussion"]
    silent = sum(1 for e in wolf_disc if not e.payload.get("text", "").strip())
    print(f"  Wolf discussion:     {len(wolf_disc)} rounds, {silent} silent")

    # Structured output metadata
    traces = [e for e in gs.events if e.type == "action_trace_audit"]
    fallback_count = sum(
        1 for e in traces
        if e.payload.get("action_trace", {}).get("fallback_reason")
    )
    structured_fail = sum(
        1 for e in traces
        if e.payload.get("action_trace", {}).get("structured_failure_reason")
    )
    print(f"  Action traces:       {len(traces)}")
    print(f"  Fallbacks:           {fallback_count}")
    print(f"  Structured failures: {structured_fail}")

    # Contradiction alerts
    try:
        from werewolf_agent.cognition.world_state import build_world_state
        from werewolf_agent.cognition.contradiction import ContradictionEngine
        ws = build_world_state(gs)
        engine = ContradictionEngine()
        alerts = engine.detect(ws.facts, gs.day_number)
        high = [a for a in alerts if a.priority == "high"]
        print(f"  Contradiction alerts: {len(alerts)} ({len(high)} high)")
        for a in high[:5]:
            print(f"    HIGH: {a.player_id} {a.alert_type}: {a.description}")
    except Exception as exc:
        print(f"  Contradiction check: skipped ({exc})")

    _sep()


# ── leakage check ────────────────────────────────────────────────────────

def check_leakage(runner: GameRunner) -> None:
    gs = runner.state
    _sep("LEAKAGE CHECK")
    leaks = []
    for ev in gs.events:
        vis = ev.payload.get("visibility", "public") if ev.payload else ""
        if vis in ("moderator_only", "seer_only", "witch_private", "werewolf_team_only"):
            continue
        if "wolf_kill" in ev.type and vis == "public":
            if "wolf_kill_target_id" in ev.payload or "target_id" in ev.payload:
                if "wolf_kill" in ev.type:
                    pass  # wolf_kill_selected is expected as internal routing
        if ev.type == "seer_check" and vis not in ("seer_only", "moderator_only"):
            leaks.append(f"Seer check leaked: vis={vis}")

    if leaks:
        print("  POTENTIAL LEAKS:")
        for l in leaks:
            print(f"    - {l}")
    else:
        print("  No public-state information leaks detected.")
    _sep()


# ── save game log ────────────────────────────────────────────────────────

def compute_game_quality_score(runner: GameRunner) -> dict[str, Any]:
    """Compute structured quality metrics for a completed game."""
    gs = runner.state
    traces = [e for e in gs.events if e.type == "action_trace_audit"]
    fallback_count = sum(
        1 for e in traces
        if e.payload.get("action_trace", {}).get("fallback_reason")
    )
    structured_fail = sum(
        1 for e in traces
        if e.payload.get("action_trace", {}).get("structured_failure_reason")
    )
    total = len(traces)
    fallback_rate = fallback_count / total if total > 0 else 0.0
    speeches = [e for e in gs.events if e.type == "speech"]
    non_empty_speeches = sum(1 for e in speeches if e.payload.get("text", "").strip())
    speech_rate = non_empty_speeches / len(speeches) if speeches else 0.0
    phases_seen = {e.payload.get("phase") for e in gs.events if e.type == "judge_broadcast"}
    has_winner = bool(gs.winning_faction)

    return {
        "fallback_rate": round(fallback_rate, 3),
        "fallback_count": fallback_count,
        "structured_fail_count": structured_fail,
        "total_action_traces": total,
        "speech_count": len(speeches),
        "non_empty_speech_count": non_empty_speeches,
        "speech_fill_rate": round(speech_rate, 3),
        "phases_seen": len(phases_seen),
        "has_winner": has_winner,
        "steps": runner.step_count,
    }


def save_game_log(runner: GameRunner, elapsed: float) -> Path:
    gs = runner.state
    quality = compute_game_quality_score(runner)
    is_low_quality = quality["fallback_rate"] > 0.7 and quality["total_action_traces"] > 5

    if is_low_quality:
        low_q_dir = ROOT / "low_quality_games"
        low_q_dir.mkdir(exist_ok=True)
        log_path = low_q_dir / f"game_{runner.game_id}.json"
        logger.warning(
            "Low quality game (fallback_rate=%.1f%%, %d traces) — saved to %s",
            quality["fallback_rate"] * 100, quality["total_action_traces"], log_path,
        )
    else:
        log_path = ROOT / f"game_{runner.game_id}.json"

    log_data = {
        "game_id": gs.game_id,
        "winning_faction": gs.winning_faction,
        "phase": gs.phase,
        "day_number": gs.day_number,
        "night_number": gs.night_number,
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
    parser = argparse.ArgumentParser(description="Run a real 12-player werewolf game")
    parser.add_argument("--seed", type=int, default=None, help="Game seed (default: auto)")
    parser.add_argument("--max-steps", type=int, default=500, help="Max graph steps")
    parser.add_argument("--timeout", type=float, default=120.0, help="Agent timeout (seconds)")
    parser.add_argument("--no-timeout", action="store_true", help="Disable agent timeout")
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

    # Use first configured player for connectivity test so router resolves a real provider
    _test_agent = next(
        (pid for pid in router._player_assignments if pid != "judge"), "p01"
    )
    test_result = router.generate(
        agent_id=_test_agent,
        task_type="speech",
        prompt='Visible state: {"phase": "night", "alive_players": ["p01", "p02"]}\nRespond with valid JSON:',
        system_prompt="You are a player in a Werewolf game. Output ONLY valid JSON.",
    )
    if test_result.text:
        print(f"  API OK: {test_result.text[:100]}...")
    else:
        print("  WARNING: API returned empty text")

    _sep("STARTING GAME")
    start = time.monotonic()

    # Persistent memory: Docker PostgreSQL + coordinator for cross-game learning
    memory_coordinator = None
    game_repo = None
    try:
        import subprocess
        subprocess.run(
            ["docker", "compose", "up", "-d", "postgres"],
            cwd=ROOT, capture_output=True, check=False,
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
    )

    runner = GameRunner(config)
    print(f"  Game ID: {runner.game_id}")
    n_agents = len(runner._agent_registry._agents) if runner._agent_registry else 0
    print(f"  Agents:  {n_agents}")

    final_state = runner.run(max_steps=args.max_steps)
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
