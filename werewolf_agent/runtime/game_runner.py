"""GameRunner: orchestrates a full game by wiring LangGraph + RuleEngine + agents.

GameRunner is the top-level orchestrator that connects the LangGraph runtime graph,
RuleEngine, AgentRegistry, and persistence into a runnable game flow. It supports
both full-game execution (run()) and step-by-step execution (run_step()).

Step-by-step execution keeps the LangGraph stream generator alive between calls.
Each run_step() reads one node output from the stream, then returns.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, replace
from typing import Any, Iterator

from werewolf_agent.core.models import GameState, GameEvent
from werewolf_agent.customization.ruleset_registry import RulesetRegistry
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.graph import RuntimeState, build_game_graph
from werewolf_agent.runtime.agent_adapter import SimpleAgentRegistry
from werewolf_agent.agents.player import PlayerAgent
from werewolf_agent.model_gateway.router import ModelRouter

RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"

logger = logging.getLogger(__name__)


@dataclass
class GameRunnerConfig:
    """Configuration for a GameRunner instance."""

    ruleset_id: str = "pre_witch_hunter_idiot_mixed"
    player_count: int = 12
    seed: int | None = None
    use_agent_registry: bool = False
    model_config_path: str = ""
    persona_config_path: str = ""
    agent_call_timeout: float = 0  # seconds; 0 = no timeout wrapper
    ruleset_registry: Any = None  # RulesetRegistry, optional
    repository: Any = None  # GameRepository, optional
    memory_coordinator: Any = None  # PersistentMemoryCoordinator, optional

    def __post_init__(self) -> None:
        if self.seed is None:
            import random
            import time
            self.seed = random.Random(time.time()).randrange(0, 2**32)


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
        self._ruleset_registry: RulesetRegistry = config.ruleset_registry or RulesetRegistry()
        self._ruleset_entry = self._ruleset_registry.require_playable(config.ruleset_id)
        self._engine = RuleEngine.from_yaml(self._ruleset_entry.path)
        self._state = GameState(
            game_id=self._game_id,
            ruleset_id=config.ruleset_id,
        )
        self._graph = build_game_graph()
        self._step_count: int = 0
        self._finished: bool = False
        # Lazy-initialized stream generator for step-by-step execution
        self._stream_gen: Iterator | None = None
        # Memory restored from previous game (None if no coordinator/repository)
        self._restored_memory: Any = None
        self._restored_rag: list[Any] | None = None
        self._agent_registry: SimpleAgentRegistry | None = self._build_agent_registry()
        # Attempt to restore memory from a previous snapshot at init
        self._restore_memory_if_configured()

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

    @property
    def restored_memory(self) -> Any:
        """MemoryStore restored from a previous game snapshot, or None."""
        return self._restored_memory

    @property
    def restored_rag(self) -> list[Any] | None:
        """RAG entries restored from a previous snapshot, or None."""
        return self._restored_rag

    def reset_game_id(self, game_id: str) -> None:
        """Update game_id for API-level game tracking. Updates both runner and state."""
        self._game_id = game_id
        self._state = replace(self._state, game_id=game_id)

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
        if self._agent_registry is not None:
            rt["agent_registry"] = self._agent_registry
        if self._config.agent_call_timeout > 0:
            rt["agent_call_timeout"] = self._config.agent_call_timeout
        return rt

    def _build_agent_registry(self) -> SimpleAgentRegistry | None:
        """Build PlayerAgent registry when real agent mode is enabled."""
        if not self._config.use_agent_registry:
            return None
        model_config_path = self._config.model_config_path or "config/models.yaml"
        router = ModelRouter.from_yaml(model_config_path, register_env_providers=True)
        # Load persona config for player names
        persona_map = self._load_persona_names()
        registry = SimpleAgentRegistry()
        for i in range(1, self._config.player_count + 1):
            player_id = f"p{i:02d}"
            name, pkey = persona_map.get(player_id, (player_id, None))
            registry.register(player_id, PlayerAgent(
                agent_id=player_id, model_router=router,
                player_name=name, persona_key=pkey,
            ))
        return registry

    def _load_persona_names(self) -> dict[str, tuple[str, str | None]]:
        """Load player_name from persona config, round-robin assignment."""
        persona_path = self._config.persona_config_path
        if not persona_path:
            return {}
        from pathlib import Path
        p = Path(persona_path)
        if not p.exists():
            return {}
        import yaml
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        profiles = data.get("persona_profiles", {})
        # Collect (key, player_name) pairs in definition order
        names = []
        for key, prof in profiles.items():
            pname = prof.get("player_name", key)
            names.append((key, pname))
        if not names:
            return {}
        result: dict[str, tuple[str, str | None]] = {}
        for i in range(1, self._config.player_count + 1):
            pid = f"p{i:02d}"
            pkey, pname = names[(i - 1) % len(names)]
            result[pid] = (pname, pkey)
        return result

    def _process_chunk(self, chunk: dict) -> str | None:
        """Process a single stream chunk, updating internal state.

        Returns the node name that was processed, or None if the chunk
        didn't contain valid output.
        """
        node_name = None
        for name, output in chunk.items():
            node_name = name
            if output is not None and "game_state" in output:
                self._state = output["game_state"]
        return node_name

    def run(self, max_steps: int = 500) -> GameState:
        """Execute the full game graph until END or max_steps reached.

        Uses LangGraph stream mode to process nodes one at a time,
        updating internal state at each step.

        Args:
            max_steps: Maximum number of graph nodes to process.

        Returns:
            The final GameState after the game completes or max_steps reached.
        """
        if self._finished:
            return self._state

        initial = self._build_runtime_state()

        try:
            for chunk in self._graph.stream(
                initial, {"recursion_limit": max_steps}
            ):
                self._step_count += 1
                self._process_chunk(chunk)
                # Check if game ended
                if self._state.phase == "finished" or self._state.winning_faction is not None:
                    self._finished = True
                    self._save_memory_snapshot()
                    self._persist_if_configured()
                    return self._state
        except Exception as exc:
            import traceback
            logger.warning("Graph execution error in run() at step %d: %s\n%s", self._step_count, exc, traceback.format_exc())

        self._finished = self._state.phase == "finished" or self._state.winning_faction is not None
        if self._finished:
            self._save_memory_snapshot()
        self._persist_if_configured()
        return self._state

    def run_scripted(
        self,
        max_steps: int = 500,
        *,
        wolf_kill_target_id: str | None = None,
        use_antidote: bool = False,
        poison_target_id: str | None = None,
        seer_target_id: str | None = None,
    ) -> GameState:
        """Execute the full game with scripted night-action inputs.

        This variant is intended for testing and replay scenarios where
        caller-controlled inputs replace LLM agent decisions for wolf kills,
        witch actions, and seer checks.

        Args:
            max_steps: Maximum number of graph nodes to process.
            wolf_kill_target_id: Optional scripted wolf kill target.
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
                if self._state.phase == "finished" or self._state.winning_faction is not None:
                    self._finished = True
                    self._save_memory_snapshot()
                    self._persist_if_configured()
                    return self._state
        except Exception as exc:
            logger.warning("Graph execution error in run_scripted() at step %d: %s", self._step_count, exc)

        self._finished = self._state.phase == "finished" or self._state.winning_faction is not None
        if self._finished:
            self._save_memory_snapshot()
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
            except Exception as exc:
                logger.error("Failed to initialize stream generator: %s", exc)
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
        except Exception as exc:
            # Transient error — do NOT mark finished
            logger.error("Step execution error at step %d: %s", self._step_count, exc)
            return self._state

        if self._finished:
            self._save_memory_snapshot()
        self._persist_if_configured()
        return self._state

    def _persist_if_configured(self) -> None:
        """Save game state to repository if one is configured."""
        if self._config.repository is not None:
            try:
                self._config.repository.save_game(self._state)
            except Exception as exc:
                logger.warning("Persistence error: %s", exc)

    def _restore_memory_if_configured(self) -> None:
        """Attempt to restore memory from a previous snapshot at init.

        Only acts when both memory_coordinator and repository are configured.
        On failure, logs a warning and leaves restored_memory as None.
        """
        coordinator = self._config.memory_coordinator
        if coordinator is None or self._config.repository is None:
            return
        try:
            mem, rag = coordinator.restore_all(snapshot_id=self._game_id)
            self._restored_memory = mem
            self._restored_rag = rag
            if mem is not None:
                logger.info("Restored memory snapshot for game %s", self._game_id)
        except Exception as exc:
            logger.warning("Memory restore error for game %s: %s", self._game_id, exc)

    def _save_memory_snapshot(self) -> None:
        """Persist a memory snapshot when a memory_coordinator is configured.

        Creates a minimal MemoryStore, inits cognition matrices for all
        players in the final game state, and saves via the coordinator.
        Called at game end, before _persist_if_configured().
        """
        coordinator = self._config.memory_coordinator
        if coordinator is None or self._config.repository is None:
            return
        try:
            from werewolf_agent.memory.store import MemoryStore

            mem_store = MemoryStore()
            player_ids = list(self._state.players.keys())
            if player_ids:
                # Collect unique role names for matrix initialization
                role_names = [
                    self._state.players[pid].role
                    for pid in player_ids
                    if pid in self._state.players
                ]
                for pid in player_ids:
                    mem_store.init_matrix(pid, player_ids, role_names)

            coordinator.save_all(
                memory_store=mem_store,
                retriever=None,
                snapshot_id=self._game_id,
            )
            logger.info("Saved memory snapshot for game %s", self._game_id)
        except Exception as exc:
            logger.warning("Memory snapshot save error for game %s: %s", self._game_id, exc)
