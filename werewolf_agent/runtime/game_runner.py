"""GameRunner: orchestrates a full game by wiring LangGraph + RuleEngine + agents.

GameRunner is the top-level orchestrator that connects the LangGraph runtime graph,
RuleEngine, AgentRegistry, and persistence into a runnable game flow. It supports
both full-game execution (run()) and step-by-step execution (run_step()).

Step-by-step execution keeps the LangGraph stream generator alive between calls.
Each run_step() reads one node output from the stream, then returns.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from typing import Any, Iterator

from werewolf_agent.core.models import GameState, GameEvent
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.graph import RuntimeState, build_game_graph

RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"


@dataclass
class GameRunnerConfig:
    """Configuration for a GameRunner instance."""

    ruleset_id: str = "pre_witch_hunter_idiot_mixed"
    player_count: int = 12
    seed: int | None = None
    use_agent_registry: bool = False
    model_config_path: str = ""
    persona_config_path: str = ""
    repository: Any = None  # GameRepository, optional
    memory_coordinator: Any = None  # PersistentMemoryCoordinator, optional

    def __post_init__(self) -> None:
        if self.seed is None:
            # Generate a stable default seed from a hash of the ruleset_id
            import hashlib
            raw = self.ruleset_id.encode("utf-8")
            self.seed = int.from_bytes(hashlib.sha256(raw).digest()[:4], "big") & 0xFFFFFFFF


class GameRunner:
    """Orchestrates a complete game from setup to finish.

    Usage:
        runner = GameRunner(GameRunnerConfig(seed=42))
        final_state = runner.run()          # Run entire game
        # Or step-by-step:
        runner = GameRunner(GameRunnerConfig(seed=42))
        runner.run_step()                   # Advance one node
        runner.run_step()                   # Advance another node
    """

    def __init__(self, config: GameRunnerConfig) -> None:
        self._config = config
        self._game_id = f"g_{config.seed or uuid.uuid4().hex[:8]}"
        self._engine = RuleEngine.from_yaml(RULESET_PATH)
        self._state = GameState(
            game_id=self._game_id,
            ruleset_id=config.ruleset_id,
        )
        self._graph = build_game_graph()
        self._step_count: int = 0
        self._finished: bool = False
        # Lazy-initialized stream generator for step-by-step execution
        self._stream_gen: Iterator | None = None

    @property
    def game_id(self) -> str:
        return self._game_id

    @property
    def engine(self) -> RuleEngine:
        return self._engine

    @property
    def state(self) -> GameState:
        return self._state

    @property
    def config(self) -> GameRunnerConfig:
        return self._config

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def finished(self) -> bool:
        return self._finished

    def _build_runtime_state(
        self,
        *,
        wolf_kill_target_id: str | None = None,
        use_antidote: bool = False,
        poison_target_id: str | None = None,
        seer_target_id: str | None = None,
    ) -> RuntimeState:
        """Build a RuntimeState dict from current game state and config."""
        rt: RuntimeState = {
            "game_state": self._state,
            "engine": self._engine,
            "wolf_kill_target_id": wolf_kill_target_id,
            "wolf_action": "kill" if wolf_kill_target_id else None,
            "wolf_action_reason": "",
            "use_antidote": use_antidote,
            "poison_target_id": poison_target_id,
            "seer_target_id": seer_target_id,
            "hybrid_master_target_id": None,
            "self_destruct_wolf_id": None,
            "exile_votes": {},
            "revote": False,
            "sheriff_candidates": [],
            "sheriff_votes": {},
            "sheriff_withdrawing": [],
            "badge_decision": "tear",
            "badge_target_id": None,
            "hunter_shot_target_id": None,
        }
        return rt

    def _process_chunk(self, chunk: dict) -> str | None:
        """Process a single stream chunk, updating internal state.

        Returns the node name that was processed, or None if the chunk
        didn't contain valid output.
        """
        node_name = None
        for name, output in chunk.items():
            node_name = name
            if "game_state" in output:
                self._state = output["game_state"]
        return node_name

    def run(
        self,
        max_steps: int = 500,
        *,
        wolf_kill_target_id: str | None = None,
        use_antidote: bool = False,
        poison_target_id: str | None = None,
        seer_target_id: str | None = None,
    ) -> GameState:
        """Execute the full game graph until END or max_steps reached.

        Uses LangGraph stream mode to process nodes one at a time,
        updating internal state at each step.

        Args:
            max_steps: Maximum number of graph nodes to process.
            wolf_kill_target_id: Optional scripted wolf kill target (first night).
            use_antidote: Whether witch uses antidote (scripted).
            poison_target_id: Optional scripted witch poison target.
            seer_target_id: Optional scripted seer check target.

        Returns:
            The final GameState after the game completes or max_steps reached.
        """
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
                # Check if game ended
                if self._state.phase == "finished" or self._state.winning_faction is not None:
                    self._finished = True
                    self._persist_if_configured()
                    return self._state
        except Exception:
            # Graph hit recursion limit — keep accumulated state
            pass

        self._finished = self._state.phase == "finished" or self._state.winning_faction is not None
        self._persist_if_configured()
        return self._state

    def run_step(self, max_steps: int = 500) -> GameState:
        """Advance the game by one graph node.

        Uses a persistent stream generator that is kept alive between calls.
        The first call creates the generator; subsequent calls read the next
        node output from it.

        Args:
            max_steps: Recursion limit for the graph execution.

        Returns:
            The updated GameState after the step.
        """
        if self._finished:
            return self._state

        # Initialize stream generator on first call
        if self._stream_gen is None:
            initial = self._build_runtime_state()
            try:
                self._stream_gen = self._graph.stream(
                    initial, {"recursion_limit": max_steps}
                )
            except Exception:
                self._finished = True
                return self._state

        # Read one chunk from the stream
        try:
            chunk = next(self._stream_gen)
            self._step_count += 1
            self._process_chunk(chunk)
            # Check if game ended
            if self._state.phase == "finished" or self._state.winning_faction is not None:
                self._finished = True
        except StopIteration:
            # Stream exhausted — game has ended
            self._finished = True
        except Exception:
            # Graph error — mark as finished
            self._finished = True

        self._persist_if_configured()
        return self._state

    def _persist_if_configured(self) -> None:
        """Save game state to repository if one is configured."""
        if self._config.repository is not None:
            try:
                self._config.repository.save_game(self._state)
            except Exception:
                pass  # Persistence failure should not crash the game
