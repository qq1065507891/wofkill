"""Run a real 12-player werewolf game with MiniMax LLM agents.

Usage:
    python scripts/run_real_game.py [--seed 42] [--max-steps 500] [--timeout 120]

Requires .env with ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL configured.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from werewolf_agent.model_gateway.providers import load_local_dotenv
from werewolf_agent.runtime.game_runner import GameRunner, GameRunnerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "game_output.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("real_game")


def print_separator(title: str = "") -> None:
    line = "=" * 60
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def print_game_summary(runner: GameRunner) -> None:
    gs = runner.state
    print_separator("GAME SUMMARY")
    print(f"  Game ID:     {gs.game_id}")
    print(f"  Ruleset:     {gs.ruleset_id}")
    print(f"  Phase:       {gs.phase}")
    print(f"  Winner:      {gs.winning_faction or 'N/A'}")
    print(f"  Day:         {gs.day_number}")
    print(f"  Night:       {gs.night_number}")
    print(f"  Steps:       {runner.step_count}")
    print(f"  Deaths:      {len(gs.deaths)}")

    print("\n  Players:")
    role_map = {}
    for pid, p in gs.players.items():
        status = "ALIVE" if p.alive else "DEAD"
        role_map[pid] = p.role
        print(f"    {pid}: {p.role:12s} [{status}]")

    print("\n  Death Log:")
    for d in gs.deaths:
        print(f"    {d.player_id} ({role_map.get(d.player_id, '?')}): {d.reason}")

    if gs.sheriff_id:
        print(f"\n  Sheriff: {gs.sheriff_id}")
    print(f"  Badge state: {gs.sheriff_badge_state}")
    print(f"  Antidote used: {gs.antidote_used}")
    print(f"  Poison used: {gs.poison_used}")

    # Print recent events (last 30)
    print(f"\n  Events ({len(gs.events)} total, showing last 30):")
    for ev in gs.events[-30:]:
        vis = ev.payload.get("visibility", "public") if ev.payload else ""
        text = ev.type
        if ev.payload:
            if "speech" in ev.type:
                speaker = ev.payload.get("speaker", "?")
                text_content = ev.payload.get("text", "")
                text = f"{ev.type}: [{speaker}] {text_content}"
            elif "vote" in ev.type:
                voter = ev.payload.get("voter", ev.payload.get("player_id", "?"))
                target = ev.payload.get("target_id", "?")
                text = f"{ev.type}: {voter} -> {target}"
            elif "wolf_kill" in ev.type:
                target = ev.payload.get("wolf_kill_target_id", "?")
                text = f"{ev.type}: target={target}"
            elif "seer_check" in ev.type:
                seer = ev.payload.get("seer_id", "?")
                target = ev.payload.get("target_id", "?")
                alignment = ev.payload.get("alignment", "?")
                text = f"{ev.type}: {seer} checks {target} -> {alignment}"
            elif "witch" in ev.type:
                text = f"{ev.type}: {ev.payload}"
            elif "badge" in ev.type:
                text = f"{ev.type}: {ev.payload}"
            elif "hybrid" in ev.type:
                text = f"{ev.type}: {ev.payload}"
        if vis and vis != "public":
            text += f" [{vis}]"
        print(f"    [{ev.type}] {text}")

    print_separator()


def print_usage_stats(runner: GameRunner) -> None:
    registry = runner._agent_registry
    if registry is None:
        return
    # Get any agent's model router
    sample_agent = next(iter(registry._agents.values()), None)
    if sample_agent is None:
        return
    router = sample_agent.model_router
    usage_log = router.get_usage_log()
    if not usage_log:
        return

    print_separator("API USAGE STATS")
    total_calls = len(usage_log)
    success_calls = sum(1 for u in usage_log if u.success)
    failed_calls = total_calls - success_calls
    total_prompt = sum(u.prompt_tokens for u in usage_log)
    total_completion = sum(u.completion_tokens for u in usage_log)
    total_latency = sum(u.latency_ms for u in usage_log)

    print(f"  Total API calls:   {total_calls}")
    print(f"  Successful:        {success_calls}")
    print(f"  Failed:            {failed_calls}")
    print(f"  Prompt tokens:     {total_prompt:,}")
    print(f"  Completion tokens: {total_completion:,}")
    print(f"  Total tokens:      {total_prompt + total_completion:,}")
    print(f"  Total latency:     {total_latency / 1000:.1f}s")

    # Per-agent breakdown
    agent_stats: dict[str, dict] = {}
    for u in usage_log:
        if u.agent_id not in agent_stats:
            agent_stats[u.agent_id] = {"calls": 0, "tokens": 0, "latency": 0}
        agent_stats[u.agent_id]["calls"] += 1
        agent_stats[u.agent_id]["tokens"] += u.prompt_tokens + u.completion_tokens
        agent_stats[u.agent_id]["latency"] += u.latency_ms

    print("\n  Per-agent breakdown:")
    for aid, stats in sorted(agent_stats.items()):
        print(f"    {aid}: {stats['calls']} calls, {stats['tokens']:,} tokens, {stats['latency'] / 1000:.1f}s")

    failure_reasons: dict[str, int] = {}
    for u in usage_log:
        if not u.success:
            reason = u.fallback_reason or "unknown"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    if failure_reasons:
        print("\n  Failure reasons:")
        for reason, count in sorted(failure_reasons.items(), key=lambda item: item[1], reverse=True):
            print(f"    {count}x {reason}")

    print_separator()


def check_leakage(runner: GameRunner) -> None:
    """Basic information leakage check on public events."""
    gs = runner.state
    print_separator("LEAKAGE CHECK")
    leaks = []

    for ev in gs.events:
        vis = ev.payload.get("visibility", "public") if ev.payload else ""
        if vis in ("moderator_only", "seer_only", "witch_private", "werewolf_team_only"):
            continue  # private events are fine

        # Check for wolf_kill_target_id in public events
        if "wolf_kill" in ev.type and vis == "public":
            if "wolf_kill_target_id" in ev.payload:
                leaks.append(f"Wolf kill target leaked in public event: {ev.type}")

        # Check for seer check results in public events
        if ev.type == "seer_check" and vis not in ("seer_only", "moderator_only"):
            leaks.append(f"Seer check leaked: {ev.type} vis={vis}")

    if leaks:
        print("  POTENTIAL LEAKS DETECTED:")
        for leak in leaks:
            print(f"    - {leak}")
    else:
        print("  No obvious information leaks detected in public events.")

    print_separator()


def print_pace_report(runner: GameRunner) -> None:
    """Print game pace report with exile rate, no-exile streaks, etc."""
    from werewolf_agent.evaluation.metrics import compute_pace_metrics

    gs = runner.state
    print_separator("GAME PACE REPORT")

    events_dicts = [{"type": e.type, "payload": e.payload} for e in gs.events]
    deaths_dicts = [{"player_id": d.player_id, "reason": d.reason} for d in gs.deaths]

    metrics = compute_pace_metrics(
        events_dicts,
        deaths=deaths_dicts,
        finish_night=gs.night_number,
    )

    wolf_kills = sum(1 for d in gs.deaths if d.reason == "wolf_kill")
    exile_deaths = sum(1 for d in gs.deaths if d.reason == "exile")
    hunter_shots = sum(1 for d in gs.deaths if d.reason == "hunter_shot")

    print(f"  Winner:              {gs.winning_faction or 'N/A'}")
    print(f"  Finish night:        {gs.night_number}")
    print(f"  Finish day:          {gs.day_number}")
    print(f"  Total deaths:        {len(gs.deaths)}")
    print(f"  Wolf kills:          {wolf_kills}")
    print(f"  Exile deaths:        {exile_deaths}")
    print(f"  Hunter shots:        {hunter_shots}")
    print(f"  Day exile rate:      {metrics['day_exile_rate']:.1%}")
    print(f"  Second tie count:    {metrics['second_tie_count']}")
    print(f"  Max consecutive no-exile: {metrics['max_consecutive_no_exile_days']}")
    print(f"  Stale vote reuse:    {metrics['stale_vote_reuse_count']}")
    print(f"  Pace target met:     {'YES' if metrics['pace_target_met'] else 'NO'}")
    print_separator()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real 12-player werewolf game")
    parser.add_argument("--seed", type=int, default=None, help="Game seed (default: auto)")
    parser.add_argument("--max-steps", type=int, default=500, help="Max graph steps")
    parser.add_argument("--timeout", type=float, default=120.0, help="Agent act timeout (seconds)")
    parser.add_argument("--no-timeout", action="store_true", help="Disable agent call timeout")
    args = parser.parse_args()

    # Load .env
    load_local_dotenv(ROOT / ".env")

    import os
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    base_url = os.getenv("ANTHROPIC_BASE_URL", "")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Check .env file.")
        sys.exit(1)

    print_separator("WEREWOLF AGENT - REAL LLM GAME")
    print(f"  API endpoint: {base_url}")
    print(f"  API key:      {api_key[:20]}...")
    print(f"  Seed:         {args.seed or 'auto'}")
    print(f"  Max steps:    {args.max_steps}")
    print(f"  Timeout:      {'disabled' if args.no_timeout else f'{args.timeout}s'}")

    # Quick connectivity test
    print("\n  Testing API connectivity...")
    from werewolf_agent.model_gateway.router import ModelRouter
    router = ModelRouter.from_yaml(ROOT / "config" / "models.yaml", register_env_providers=True)
    provider_names = router.provider_names()
    print(f"  Registered providers: {provider_names}")

    if "anthropic" not in provider_names:
        print("ERROR: Anthropic provider not registered. Check API key.")
        sys.exit(1)

    # Quick test call
    from werewolf_agent.agents.schemas import AgentContext, TaskType, ActionType
    test_result = router.generate(
        agent_id="p01",
        task_type="speech",
        prompt='Visible state: {"phase": "night", "alive_players": ["p01", "p02"]}\nProvide your action as JSON:',
        system_prompt="You are a player in a Werewolf game. Output ONLY valid JSON.",
    )
    if test_result.text:
        print(f"  API test OK: {test_result.text[:100]}...")
    else:
        print("  WARNING: API test returned empty text")

    print_separator("STARTING GAME")
    start_time = time.monotonic()

    config = GameRunnerConfig(
        ruleset_id="pre_witch_hunter_idiot_mixed",
        player_count=12,
        seed=args.seed,
        use_agent_registry=True,
        model_config_path=str(ROOT / "config" / "models.yaml"),
        persona_config_path=str(ROOT / "config" / "personas" / "jingcheng_style_prototypes.yaml"),
        agent_call_timeout=0 if args.no_timeout else args.timeout,
    )

    runner = GameRunner(config)
    print(f"  Game ID: {runner.game_id}")
    print(f"  Agent registry: {len(runner._agent_registry._agents) if runner._agent_registry else 0} agents")

    final_state = runner.run(max_steps=args.max_steps)
    elapsed = time.monotonic() - start_time

    print(f"\n  Game finished in {elapsed:.1f}s ({runner.step_count} steps)")

    print_game_summary(runner)
    print_usage_stats(runner)
    print_pace_report(runner)
    check_leakage(runner)

    # Save game log
    log_path = ROOT / f"game_{runner.game_id}.json"
    try:
        import json
        log_data = {
            "game_id": final_state.game_id,
            "winning_faction": final_state.winning_faction,
            "phase": final_state.phase,
            "day_number": final_state.day_number,
            "night_number": final_state.night_number,
            "players": {
                pid: {"role": p.role, "alive": p.alive}
                for pid, p in final_state.players.items()
            },
            "deaths": [
                {"player_id": d.player_id, "reason": d.reason}
                for d in final_state.deaths
            ],
            "events": [
                {"type": e.type, "payload": e.payload}
                for e in final_state.events
            ],
            "elapsed_seconds": round(elapsed, 1),
            "steps": runner.step_count,
        }
        log_path.write_text(json.dumps(log_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  Game log saved to: {log_path}")
    except Exception as exc:
        print(f"\n  Warning: could not save game log: {exc}")


if __name__ == "__main__":
    main()
