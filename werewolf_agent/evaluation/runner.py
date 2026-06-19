"""Batch game runner: runs N games with fixed seed sets for reproducible evaluation.

Design doc §14: batch games, fixed seed sets, replay from initial_seed + ruleset_snapshot + event_log.
Evaluation never mutates rule truth — reads RuleEngine state, never writes to it.
"""

from __future__ import annotations

import copy
import hashlib
import random
import time
from dataclasses import replace
from typing import Any, Callable

from werewolf_agent.agents.schemas import AgentContext
from werewolf_agent.core.models import GameEvent, GameState
from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset
from werewolf_agent.evaluation.ablation import (
    AblationReport,
    AblationToggleSet,
    LiveAblationReport,
    LiveContextAblationHarness,
    OfflineTraceAblationRunner,
)
from werewolf_agent.evaluation.feedback_schemas import EvaluationTrace
from werewolf_agent.evaluation.schemas import (
    ActionRecord,
    ActionVerdict,
    BatchConfig,
    CostRecord,
    GameResult,
    LeakageRecord,
    ReplayRecord,
)


def run_offline_trace_ablation(
    traces: list[EvaluationTrace],
    *,
    removed_modules: list[str],
) -> AblationReport:
    """Run offline feedback ablation without executing live agents."""
    return OfflineTraceAblationRunner().run(
        traces,
        removed_modules=removed_modules,
    )


def run_live_context_ablation(
    contexts: list[AgentContext],
    *,
    removed_modules: list[str],
    runner: Callable[[AgentContext], Any],
) -> LiveAblationReport:
    """Run context-level live ablation through an injected agent runner."""
    return LiveContextAblationHarness(runner=runner).run(
        contexts,
        toggles=AblationToggleSet(removed_modules),
    )


class BatchRunner:
    """Runs batch games with fixed seed sets and collects GameResult objects.

    Evaluation is strictly read-only with respect to rule truth:
    - Reads RuleEngine for role assignment, night resolution, etc.
    - Never modifies ruleset or rule engine state.
    - All results are derived from RuleEngine outputs, not from agent claims.
    """

    def __init__(
        self,
        rule_engine: RuleEngine,
        batch_config: BatchConfig,
    ) -> None:
        self._engine = rule_engine
        self._config = batch_config
        self._results: list[GameResult] = []

    @property
    def config(self) -> BatchConfig:
        return self._config

    @property
    def results(self) -> list[GameResult]:
        return list(self._results)

    def generate_seed_set(self) -> list[int]:
        """Generate a deterministic seed set from batch config.

        If seed_set is provided in config, use it. Otherwise generate from
        batch_id hash for reproducibility.
        """
        if self._config.seed_set:
            return list(self._config.seed_set)

        base_hash = int(hashlib.sha256(self._config.batch_id.encode()).hexdigest(), 16)
        rng = random.Random(base_hash)
        return [rng.randint(0, 2**31 - 1) for _ in range(self._config.num_games)]

    def run_game(self, seed: int, game_index: int = 0) -> GameResult:
        """Run a single game deterministically from the given seed.

        This uses the RuleEngine for all rule decisions. Agent actions are
        simulated with deterministic defaults (mock actions) for evaluation
        purposes — real LLM agent integration is orthogonal to the evaluation
        framework.
        """
        game_id = f"{self._config.batch_id}_game_{game_index:04d}"
        player_ids = [f"player_{i:02d}" for i in range(1, self._config.player_count + 1)]

        # Assign roles deterministically from seed
        players = self._engine.assign_roles(player_ids, seed=seed)

        # Build player_roles and player_factions from RuleEngine output
        player_roles = {pid: p.role for pid, p in players.items()}
        player_factions = {}
        for pid, p in players.items():
            role_cfg = self._engine.ruleset.raw["roles"].get(p.role, {})
            faction = role_cfg.get("faction", "unknown")
            if faction == "special_bound_to_master":
                faction = "good"  # hybrid defaults, actual faction set after master chosen
            player_factions[pid] = faction

        # Build initial state
        state = GameState(
            game_id=game_id,
            ruleset_id=self._config.ruleset_id,
            players=dict(players),
        )

        # Track events through the game
        events: list[dict[str, Any]] = []
        action_records: list[ActionRecord] = []

        def append_new_events(start_index: int) -> None:
            for event in state.events[start_index:]:
                events.append({"type": event.type, "payload": dict(event.payload)})

        def hybrid_result_for(winner: str | None) -> str | None:
            if winner is None or state.hybrid_master_faction is None:
                return None
            return "win" if winner == state.hybrid_master_faction else "lose"

        # Simulate a minimal game flow for evaluation metrics
        # Phase 1: Hybrid chooses master (first night)
        hybrid_id = next((pid for pid, p in players.items() if p.role == "hybrid"), None)
        if hybrid_id:
            master_rng = random.Random(seed + 1)
            non_hybrid = [pid for pid in player_ids if pid != hybrid_id]
            master_id = master_rng.choice(non_hybrid)
            state, event = self._engine.choose_master(state, hybrid_id=hybrid_id, master_id=master_id)
            state = replace(state, events=state.events + [event])
            append_new_events(0)
            # Update hybrid faction after master chosen
            player_factions[hybrid_id] = state.hybrid_master_faction or "good"

        # Phase 2: Night resolution loop
        game_rng = random.Random(seed + 2)
        night_number = 1
        day_number = 0
        max_rounds = 20  # safety bound

        while night_number <= max_rounds:
            # Wolf kill target
            alive_non_wolf = [
                pid for pid, p in state.players.items()
                if p.alive and p.role != "werewolf"
            ]
            if not alive_non_wolf:
                break

            wolf_target = game_rng.choice(alive_non_wolf)

            # Witch decisions (simplified mock for evaluation)
            use_antidote = False
            poison_target = None

            # Simulate witch using antidote on first night kill (if available)
            witch_id = next(
                (pid for pid, p in state.players.items() if p.role == "witch" and p.alive),
                None,
            )
            if witch_id and not state.antidote_used and wolf_target != witch_id and night_number == 1:
                use_antidote = True

            # Simulate witch using poison on a random wolf-lean target later
            if witch_id and not state.poison_used and night_number > 1 and not use_antidote:
                alive_non_witch = [
                    pid for pid, p in state.players.items()
                    if p.alive and pid != witch_id and pid != wolf_target
                ]
                if alive_non_witch:
                    poison_target = game_rng.choice(alive_non_witch)

            # Resolve night
            before_events = len(state.events)
            state, night_events = self._engine.resolve_night(
                state,
                night_number=night_number,
                wolf_kill_target_id=wolf_target,
                use_antidote=use_antidote,
                poison_target_id=poison_target,
            )
            if night_events:
                state = replace(state, events=state.events + night_events)
            append_new_events(before_events)

            # Record actions
            action_records.append(ActionRecord(
                player_id=next((pid for pid, p in state.players.items() if p.role == "werewolf" and p.alive), "unknown"),
                action_type="wolf_kill",
                target_id=wolf_target,
                verdict=ActionVerdict.LEGAL,
                phase="night",
                night_number=night_number,
            ))

            # Check victory after night
            victory = self._engine.check_victory(state)
            events.append({"type": "victory_checked", "payload": {"winner": victory.winner, "reason": victory.reason}})
            if victory.winner is not None:
                hr = hybrid_result_for(victory.winner)
                victory_event = GameEvent(
                    type="victory",
                    payload={
                        "winner": victory.winner,
                        "winning_faction": victory.winner,
                        "reason": victory.reason,
                        "hybrid_master_id": state.hybrid_master_id,
                        "hybrid_master_faction": state.hybrid_master_faction,
                        "hybrid_result": hr,
                    },
                )
                state = replace(
                    state,
                    winning_faction=victory.winner,
                    hybrid_result=hr,
                    events=state.events + [victory_event],
                )
                events.append({"type": victory_event.type, "payload": dict(victory_event.payload)})
                break

            # Day phase: simple exile
            day_number += 1
            alive_players = [pid for pid, p in state.players.items() if p.alive]
            exile_targets = self._engine.legal_exile_targets(state)
            if exile_targets:
                # Use deterministic evidence-based fallback for reproducible metrics
                try:
                    from werewolf_agent.runtime.vote_quality import choose_vote_fallback_target
                    exiled = choose_vote_fallback_target(state, exile_targets, rng_seed=state.game_id)
                except Exception:
                    exiled = game_rng.choice(exile_targets)
                before_events = len(state.events)
                state, exile_events = self._engine.resolve_exile(state, target_id=exiled)
                if exile_events:
                    state = replace(state, events=state.events + exile_events)
                append_new_events(before_events)

                action_records.append(ActionRecord(
                    player_id="vote_majority",
                    action_type="vote",
                    target_id=exiled,
                    verdict=ActionVerdict.LEGAL,
                    phase="day",
                    day_number=day_number,
                ))

            # Check victory after exile
            victory = self._engine.check_victory(state)
            events.append({"type": "victory_checked", "payload": {"winner": victory.winner, "reason": victory.reason}})
            if victory.winner is not None:
                hr = hybrid_result_for(victory.winner)
                victory_event = GameEvent(
                    type="victory",
                    payload={
                        "winner": victory.winner,
                        "winning_faction": victory.winner,
                        "reason": victory.reason,
                        "hybrid_master_id": state.hybrid_master_id,
                        "hybrid_master_faction": state.hybrid_master_faction,
                        "hybrid_result": hr,
                    },
                )
                state = replace(
                    state,
                    winning_faction=victory.winner,
                    hybrid_result=hr,
                    events=state.events + [victory_event],
                )
                events.append({"type": victory_event.type, "payload": dict(victory_event.payload)})
                break

            night_number += 1

        # Set final winning faction
        final_victory = self._engine.check_victory(state)
        winning_faction = final_victory.winner
        hybrid_result = hybrid_result_for(winning_faction)

        result = GameResult(
            game_id=game_id,
            initial_seed=seed,
            ruleset_id=self._config.ruleset_id,
            ruleset_snapshot=copy.deepcopy(dict(self._engine.ruleset.raw)),
            winning_faction=winning_faction,
            hybrid_master_id=state.hybrid_master_id,
            hybrid_master_faction=state.hybrid_master_faction,
            hybrid_result=hybrid_result,
            victory_reason=final_victory.reason,
            total_days=day_number,
            total_nights=night_number,
            player_roles=player_roles,
            player_factions=player_factions,
            deaths=[
                {"player_id": d.player_id, "reason": d.reason, "timing": d.timing, "resolution_batch": d.resolution_batch}
                for d in state.deaths
            ],
            event_log=events,
            action_records=action_records,
            leakage_records=[],
            cost_records=[],
            persona_config_snapshot={},
            model_config_snapshot={},
            rag_config_snapshot={},
            strategy_config_snapshot={},
        )

        self._results.append(result)
        return result

    def run_batch(self) -> list[GameResult]:
        """Run the full batch of games from the seed set."""
        seeds = self.generate_seed_set()
        results = []
        for i, seed in enumerate(seeds):
            result = self.run_game(seed, game_index=i)
            results.append(result)
        return results

    def add_leakage_record(
        self,
        game_id: str,
        player_id: str,
        leaked_info_type: str,
        phase: str = "",
        day_number: int = 0,
        detail: str = "",
    ) -> None:
        """Add a leakage record to the matching game result."""
        for result in self._results:
            if result.game_id == game_id:
                result.leakage_records.append(LeakageRecord(
                    game_id=game_id,
                    player_id=player_id,
                    leaked_info_type=leaked_info_type,
                    phase=phase,
                    day_number=day_number,
                    detail=detail,
                ))
                break

    def add_action_record(
        self,
        game_id: str,
        record: ActionRecord,
    ) -> None:
        """Add an action record to the matching game result."""
        for result in self._results:
            if result.game_id == game_id:
                result.action_records.append(record)
                break

    def add_cost_record(
        self,
        game_id: str,
        record: CostRecord,
    ) -> None:
        """Add a cost record to the matching game result."""
        for result in self._results:
            if result.game_id == game_id:
                result.cost_records.append(record)
                break

    @staticmethod
    def verify_replay(replay: ReplayRecord, engine: RuleEngine) -> GameState:
        """Replay a game from initial_seed + ruleset_snapshot + event_log.

        Verification that replay produces the same final state.
        Does NOT mutate rule truth — creates a fresh GameState.
        """
        replay_engine = engine
        if replay.ruleset_snapshot:
            replay_engine = RuleEngine(Ruleset(raw=copy.deepcopy(replay.ruleset_snapshot)))

        # Reconstruct initial state from the captured ruleset snapshot.
        player_count = int(replay_engine.ruleset.raw.get("player_count", 12))
        player_ids = [f"player_{i:02d}" for i in range(1, player_count + 1)]
        players = replay_engine.assign_roles(player_ids, seed=replay.initial_seed)

        state = GameState(
            game_id=replay.game_id,
            ruleset_id=(
                replay_engine.ruleset.raw.get("id")
                or replay_engine.ruleset.raw.get("ruleset_id")
                or "pre_witch_hunter_idiot_mixed"
            ),
            players=dict(players),
        )

        # Replay all events through the reducer
        for event_data in replay.event_log:
            event = GameEvent(
                type=event_data["type"],
                payload=event_data.get("payload", {}),
            )
            state = replay_engine.reduce_event(state, event)

        return state
