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
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Any, Iterator

from werewolf_agent.core.models import GameState
from werewolf_agent.customization.ruleset_registry import RulesetRegistry
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.graph import RuntimeState, build_game_graph
from werewolf_agent.runtime.agent_adapter import SimpleAgentRegistry
from werewolf_agent.agents.player import PlayerAgent
from werewolf_agent.agents.judge import JudgeAgent
from werewolf_agent.agents.judge_hitl import JudgeHITLInterface
from werewolf_agent.model_gateway.router import ModelRouter
from werewolf_agent.memory.store import MemoryStore
from werewolf_agent.persona_runtime.judge_router import JudgeProfileRouter
from werewolf_agent.persona_runtime.router import PersonaRouter
from werewolf_agent.runtime.cognition_state import CognitionStateManager

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
    rag_service: Any = None  # RAGKnowledgeService, optional
    enable_default_rag_service: bool = True
    probe_tool_call_support: bool = False
    judge_llm_enabled: bool = False
    judge_persona_profile_id: str = "tournament_referee"
    judge_persona_config_path: str = ""
    judge_hitl_enabled: bool = False
    judge_hitl_auto_pause_triggers: list[str] | None = None  # e.g. ["death_announce", "exile"]
    agent_call_delay_ms: int = 0  # -1=no delay, 0=10s, >0=fixed ms

    def __post_init__(self) -> None:
        if self.seed is None:
            import secrets
            self.seed = secrets.randbits(32)


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
        self._game_id = f"g_{config.seed if config.seed is not None else uuid.uuid4().hex[:8]}"
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
        self._cognition_state_manager = CognitionStateManager(
            MemoryStore(repo=config.repository)
        )
        self._model_router: ModelRouter | None = None
        self._persona_router: PersonaRouter | None = None
        self._rag_service: Any = config.rag_service
        if self._rag_service is None and config.enable_default_rag_service:
            self._rag_service = self._build_default_rag_service()
        self._agent_registry: SimpleAgentRegistry | None = self._build_agent_registry()
        self._judge_agent: JudgeAgent | None = None
        if self._model_router is not None:
            profile_router = self._load_judge_profile_router()
            self._judge_agent = JudgeAgent(
                model_router=self._model_router,
                profile_router=profile_router,
                profile_id=self._config.judge_persona_profile_id,
            )
        elif self._config.judge_llm_enabled:
            logger.warning(
                "judge_llm_enabled=True but use_agent_registry=False: "
                "JudgeAgent requires an agent registry. Set use_agent_registry=True "
                "to enable LLM-powered judge broadcasts."
            )
        # HITL interface (Layer 4)
        self._hitl_interface: JudgeHITLInterface | None = None
        if self._config.judge_hitl_enabled:
            auto_pause = set(self._config.judge_hitl_auto_pause_triggers or [])
            self._hitl_interface = JudgeHITLInterface(
                auto_pause_phases=auto_pause,
            )
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

    @property
    def hitl_interface(self) -> JudgeHITLInterface | None:
        """Layer 4: Judge HITL interface, if enabled."""
        return self._hitl_interface

    def pause(self) -> str | None:
        """Pause game execution at the next checkpoint. Returns response string."""
        if self._hitl_interface is None:
            return None
        self._hitl_interface.pause()
        return f"游戏已暂停。步骤: {self._step_count}"

    def resume(self, steps: int = 0) -> str | None:
        """Resume game execution. Optionally auto-pause after N steps."""
        if self._hitl_interface is None:
            return None
        self._hitl_interface.resume(steps)
        return f"游戏已恢复。{'将执行' + str(steps) + '步后暂停' if steps else ''}"

    def send_command(self, raw: str) -> str | None:
        """Send a HITL command to the running game. Returns response string."""
        if self._hitl_interface is None:
            return None
        self._hitl_interface.send_command(raw)
        if self._hitl_interface.is_paused:
            from werewolf_agent.agents.judge_hitl import HITLCommand
            cmd = self._hitl_interface._pending_command
            if cmd is None:
                cmd = HITLCommand.parse(raw)
            result = self._hitl_interface.handle_command(cmd, self._state)
            if "game_state" in result:
                self._state = result["game_state"]
            hitl_events = self._hitl_interface.flush_events()
            if hitl_events:
                self._state = replace(self._state, events=self._state.events + hitl_events)
            return result.get("response", "OK")
        return f"命令已排队: {raw}"

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
            "badge_decision": None,
            "badge_target_id": None,
            "hunter_shot_target_id": None,
            "action_index_by_game": {},
            "pending_exposure_events_by_trace": {},
        }
        if self._agent_registry is not None:
            rt["agent_registry"] = self._agent_registry
        if self._judge_agent is not None:
            rt["judge_agent"] = self._judge_agent
            rt["judge_llm_enabled"] = self._config.judge_llm_enabled
        if self._hitl_interface is not None:
            rt["judge_hitl"] = self._hitl_interface
            rt["judge_hitl_enabled"] = True
            rt["hitl_auto_pause_after"] = self._config.judge_hitl_auto_pause_triggers or []
        rt["agent_call_delay_ms"] = self._config.agent_call_delay_ms
        if self._rag_service is not None:
            rt["rag_service"] = self._rag_service
        if self._config.agent_call_timeout > 0:
            rt["agent_call_timeout"] = self._config.agent_call_timeout
        rt["cognition_state_manager"] = self._cognition_state_manager
        if self._restored_memory is not None:
            rt["restored_memory"] = self._restored_memory
        if self._config.repository is not None:
            rt["repository"] = self._config.repository
        return rt

    def _build_default_rag_service(self) -> Any | None:
        """Build no-Docker seed RAG service for runtime wiring."""
        try:
            from werewolf_agent.rag.knowledge_service import RAGKnowledgeService

            return RAGKnowledgeService()
        except Exception:
            logger.warning("Default RAG service initialization failed", exc_info=True)
            return None

    def _build_agent_registry(self) -> SimpleAgentRegistry | None:
        """Build PlayerAgent registry when real agent mode is enabled."""
        if not self._config.use_agent_registry:
            return None
        model_config_path = self._config.model_config_path or str(
            Path(__file__).resolve().parent.parent.parent / "config" / "models.yaml"
        )
        router = ModelRouter.from_yaml(model_config_path, register_env_providers=True)
        self._model_router = router
        if self._config.probe_tool_call_support:
            probe = router.probe_tool_call_support("p01", "speech")
            if not probe.get("supported"):
                raise RuntimeError(f"tool call probe failed: {probe}")
        # Load persona config for player names
        persona_map = self._load_persona_names()
        persona_path = self._player_persona_path()
        if persona_path is not None and persona_map:
            self._persona_router = PersonaRouter.from_yaml(persona_path)
            self._persona_router.load_assignments({
                player_id: persona_key
                for player_id, (_, persona_key) in persona_map.items()
                if persona_key
            })
        registry = SimpleAgentRegistry()
        for i in range(1, self._config.player_count + 1):
            player_id = f"p{i:02d}"
            name, pkey = persona_map.get(player_id, (player_id, None))
            registry.register(player_id, PlayerAgent(
                agent_id=player_id, model_router=router,
                player_name=name, persona_key=pkey,
                persona_router=self._persona_router,
            ))
        return registry

    def _player_persona_path(self) -> Path | None:
        configured = self._config.persona_config_path
        path = Path(configured) if configured else (
            Path(__file__).resolve().parent.parent.parent
            / "config" / "personas" / "jingcheng_style_prototypes.yaml"
        )
        return path if path.exists() else None

    def _load_persona_names(self) -> dict[str, tuple[str, str | None]]:
        """Load player_name from persona config, round-robin assignment."""
        p = self._player_persona_path()
        if p is None:
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

    def _load_judge_profile_router(self) -> JudgeProfileRouter | None:
        """Load judge persona profiles from the configured YAML path."""
        profile_path = self._config.judge_persona_config_path
        if not profile_path:
            default_path = Path(__file__).resolve().parent.parent.parent / "config" / "personas" / "judge_profiles.yaml"
            if default_path.exists():
                profile_path = str(default_path)
            else:
                return None
        try:
            return JudgeProfileRouter.from_yaml(profile_path)
        except Exception:
            logger.warning("Failed to load judge profile router", exc_info=True)
            return None

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
                try:
                    self._cognition_state_manager.update_from_events(self._state)
                except Exception:
                    logger.warning(
                        "Live cognition update failed after node %s",
                        node_name,
                        exc_info=True,
                    )
        return node_name

    def run(self, max_steps: int = 1000) -> GameState:
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

        # J-3: HITL pause must be enforced. If the interface is paused
        # (user or auto-trigger), do not iterate the stream — return the
        # current state unchanged so the runner honors the pause.
        if self._hitl_interface is not None and self._hitl_interface.is_paused:
            return self._state

        initial = self._build_runtime_state()
        last_phase_snapshot: tuple[str, int, int] | None = None
        stuck_count = 0

        try:
            for chunk in self._graph.stream(
                initial, {"recursion_limit": max_steps}
            ):
                self._step_count += 1
                self._process_chunk(chunk)
                # Check if game ended — use phase=="finished" only.
                # winning_faction is set by check_victory before the reflection
                # node runs; using it as an exit condition would skip reflection.
                if self._state.phase == "finished":
                    self._finished = True
                    self._save_memory_snapshot()
                    self._persist_if_configured()
                    return self._state
                # Detect stuck state: same phase+day+night for many steps
                snapshot = (self._state.phase, self._state.day_number, self._state.night_number)
                if snapshot == last_phase_snapshot:
                    stuck_count += 1
                    if stuck_count >= 50:
                        logger.warning(
                            "Game stuck detected at step %d: phase=%s day=%d night=%d — forcing finish",
                            self._step_count, self._state.phase,
                            self._state.day_number, self._state.night_number,
                        )
                        break
                else:
                    stuck_count = 0
                    last_phase_snapshot = snapshot
        except Exception as exc:
            import traceback
            logger.warning(
                "Graph execution error in run() at step %d (phase=%s, day=%d, night=%d): %s\n%s",
                self._step_count, self._state.phase,
                self._state.day_number, self._state.night_number,
                exc, traceback.format_exc(),
            )

        self._finished = self._state.phase == "finished" or self._state.winning_faction is not None
        if self._finished:
            self._save_memory_snapshot()
        self._persist_if_configured()
        return self._state

    def run_scripted(
        self,
        max_steps: int = 1000,
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
                if self._state.phase == "finished":
                    self._finished = True
                    self._save_memory_snapshot()
                    self._persist_if_configured()
                    return self._state
        except Exception as exc:
            logger.warning("Graph execution error in run_scripted() at step %d: %s", self._step_count, exc)

        self._finished = self._state.phase == "finished"
        if self._finished:
            self._save_memory_snapshot()
        self._persist_if_configured()
        return self._state

    def run_step(self, max_steps: int = 1000) -> GameState:
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

        # J-3: HITL pause must be enforced. If the interface is paused
        # (user or auto-trigger), do not initialize the stream generator
        # or read from it — return the current state unchanged.
        if self._hitl_interface is not None and self._hitl_interface.is_paused:
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
            # Check if game ended — phase=="finished" only (not winning_faction;
            # the reflection node must run after check_victory sets winning_faction)
            if self._state.phase == "finished":
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
        """Persist full memory snapshot at game end.

        Builds structured world state, imports relations, syncs cognition
        matrices, generates reviews, and saves via the coordinator for
        cross-game retrieval.
        """
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
            self_reviews = self._latest_self_reviews()
            synthesizer = ReflectionSynthesizer()
            for report in reports:
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
                    llm_self_review=self_reviews.get(report.player_id, ""),
                    review_report=report,
                    faction=faction,
                )
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

    def _latest_self_reviews(self) -> dict[str, str]:
        for event in reversed(self._state.events):
            if event.type != "reflection_complete":
                continue
            entries = event.payload.get("entries", [])
            result: dict[str, str] = {}
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                pid = str(entry.get("player_id", ""))
                if pid:
                    result[pid] = str(entry.get("reflection", "") or "")
            return result
        return {}
