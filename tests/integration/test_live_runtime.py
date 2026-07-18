"""Live runtime orchestration: complete 12-player agent-driven game through LangGraph.

Uses DeterministicMockProvider to drive all agent decisions.
Verifies the game completes from setup to finish with a valid winner
and no information leakage in public events.
"""

from __future__ import annotations

import json

from werewolf_agent.agents.player import PlayerAgent, DefaultActionValidator
from werewolf_agent.core.models import GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.model_gateway.router import (
    GenerateResult,
    ModelConfig,
    ModelRouter,
    UsageRecord,
)
from werewolf_agent.model_gateway.final_prompt_observer import (
    FinalPromptAssembly,
    notify_final_prompt_observer,
)
from werewolf_agent.runtime.agent_adapter import SimpleAgentRegistry
from werewolf_agent.runtime.graph import (
    RuntimeState,
    build_game_graph,
    _new_engine,
)

RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"


def _project_tool_payload(action: dict, tool: dict) -> dict:
    """Project a full mock action onto the active structured-output schema."""
    properties = tool["input_schema"]["properties"]
    action_values = properties.get("action_type", {}).get("enum", [])
    force_seer_check = (
        "check_alignment" in action_values
        and action.get("action_type") == "no_action"
    )
    projected: dict = {}
    for field, schema in properties.items():
        schema_type = schema.get("type")
        schema_types = (
            [schema_type] if isinstance(schema_type, str) else schema_type or []
        )
        if field == "speech":
            projected[field] = (
                "我是好人阵营。根据公开发言与投票变化，p01的站边有连续性，"
                "p02的票型却缺少解释；我当前更怀疑p03前后逻辑不一致。"
                "今天我会继续核对p03的发言并给出明确票型，若其无法回应矛盾，"
                "我倾向投票p03，同时保留复查p02的可能。"
            )
        elif field == "target_stance":
            target_schema = schema["properties"]["target_id"]
            target_id = next(
                (value for value in target_schema.get("enum", []) if value is not None),
                None,
            )
            projected[field] = (
                {"target_id": target_id, "stance": "propose", "priority": "primary"}
                if target_id is not None else None
            )
        elif field == "action_type" and force_seer_check:
            projected[field] = "check_alignment"
        elif field == "target_id" and force_seer_check:
            projected[field] = next(
                (value for value in schema.get("enum", []) if value is not None),
                None,
            )
        elif field in action:
            projected[field] = action[field]
        elif field in {"reason", "suspect_reason", "not_voting_reason", "candidate_comparison"}:
            projected[field] = "基于公开发言、站边和票型进行对比判断"
        elif schema.get("enum"):
            projected[field] = next(
                (value for value in schema["enum"] if value is not None),
                None,
            )
        elif "null" in schema_types:
            projected[field] = None
        elif "array" in schema_types:
            projected[field] = []
        elif "object" in schema_types:
            projected[field] = {}
        elif "integer" in schema_types:
            projected[field] = 1
        elif "number" in schema_types:
            projected[field] = 0.7
        else:
            projected[field] = "test"
    return projected


class _DeterministicMockProvider:
    """Returns deterministic JSON actions for all action types."""

    def __init__(self, name: str = "live_mock") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate(
        self,
        prompt: str,
        config: ModelConfig,
        system_prompt: str | None = None,
        tools=None,
        tool_choice=None,
        final_prompt_observer=None,
    ) -> GenerateResult:
        if final_prompt_observer is not None and system_prompt:
            notify_final_prompt_observer(
                final_prompt_observer,
                FinalPromptAssembly(
                    system_bytes=system_prompt.encode("utf-8"),
                    final_system_location="messages",
                    final_system_message_index=0,
                    provider=self.name,
                    model=config.model,
                ),
            )
        action = self._decide(system_prompt or "", prompt)
        if tools:
            action = _project_tool_payload(action, tools[0])
        text = json.dumps(action, ensure_ascii=False)
        return GenerateResult(
            text=text,
            provider=self._name,
            model=config.model,
            tool_call_required=bool(tool_choice),
            tool_call_received=bool(tool_choice),
            tool_call_name=(tool_choice or {}).get("name", ""),
            usage=UsageRecord(
                agent_id="", task_type="",
                provider=self._name, model=config.model,
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(text.split()),
            ),
        )

    def _decide(self, system_prompt: str, prompt: str) -> dict:
        import re
        combined = system_prompt + "\n" + prompt

        # Extract role
        role = None
        for prefix in ("Your role: ", "own_role: "):
            idx = combined.find(prefix)
            if idx >= 0:
                rest = combined[idx + len(prefix):]
                end = rest.find("\n")
                role = rest[:end].strip() if end >= 0 else rest.strip()
                break

        # Extract legal actions
        legal_actions = []
        for pattern in (r"legal_actions.*?\[([^\]]*)\]", r"Legal actions: \[([^\]]*)\]"):
            m = re.search(pattern, combined)
            if m:
                legal_actions = [a.strip().strip("'\"") for a in m.group(1).split(",")]
                break

        # Extract legal targets
        legal_targets = []
        for pattern in (r"legal_targets.*?\[([^\]]*)\]", r"Legal targets: \[([^\]]*)\]"):
            m = re.search(pattern, combined)
            if m:
                legal_targets = [t.strip().strip("'\"") for t in m.group(1).split(",")]
                break

        # Night actions
        if "check_alignment" in str(legal_actions):
            target = legal_targets[0] if legal_targets else None
            return {
                "action_type": "check_alignment", "target_id": target,
                "speech": "", "reason": "seer check", "confidence": 0.9,
                "private_intent": {"true_role": role or "seer", "faction_goal": "find_wolves"},
            }

        if "use_antidote" in str(legal_actions):
            return {
                "action_type": "use_antidote",
                "target_id": legal_targets[0] if legal_targets else None,
                "speech": "", "reason": "save player", "confidence": 0.8,
                "private_intent": {"true_role": "witch", "faction_goal": "protect_team"},
            }

        if "use_poison" in str(legal_actions) and "antidote" not in str(legal_actions):
            return {
                "action_type": "no_action", "target_id": None,
                "speech": "", "reason": "wait for info", "confidence": 0.5,
                "private_intent": None,
            }

        if "wolf_kill" in str(legal_actions):
            target = legal_targets[0] if legal_targets else None
            return {
                "action_type": "wolf_kill", "target_id": target,
                "speech": "", "reason": "wolf consensus", "confidence": 0.7,
                "private_intent": {"true_role": "werewolf", "faction_goal": "eliminate_good"},
            }

        if "wolf_no_kill" in str(legal_actions):
            return {
                "action_type": "wolf_kill",
                "target_id": legal_targets[0] if legal_targets else None,
                "speech": "", "reason": "kill target", "confidence": 0.7,
                "private_intent": {"true_role": "werewolf"},
            }

        # Day speech
        if "speech" in str(legal_actions):
            return {
                "action_type": "speech", "target_id": None,
                "speech": "我观察到一些可疑行为，需要仔细分析。" if role != "werewolf" else "我认为应该冷静分析，不要冲动投票。",
                "reason": "express stance", "confidence": 0.6,
                "private_intent": {"true_role": role or "villager", "faction_goal": "survive"},
            }

        # Vote
        if "vote" in str(legal_actions):
            # Good players try to vote wolves, wolves vote good
            target = legal_targets[0] if legal_targets else None
            return {
                "action_type": "vote", "target_id": target,
                "speech": "", "reason": "suspicious behavior", "confidence": 0.6,
                "private_intent": {"true_role": role or "villager", "faction_goal": "find_wolves"},
            }

        return {
            "action_type": "no_action", "target_id": None,
            "speech": "", "reason": "no action", "confidence": 0.3,
            "private_intent": None,
        }


class _NoJitterModelRouter(ModelRouter):
    """Keep deterministic integration games fast and reproducible."""

    def generate(self, *args, **kwargs) -> GenerateResult:
        kwargs["jitter_seconds"] = (0.0, 0.0)
        return super().generate(*args, **kwargs)


def _build_registry(engine: RuleEngine) -> SimpleAgentRegistry:
    provider = _DeterministicMockProvider()
    router = _NoJitterModelRouter(
        model_profiles={}, llm_profiles={}, player_assignments={},
        providers={"mock": provider},
    )
    registry = SimpleAgentRegistry()
    for i in range(1, 13):
        pid = f"p{i:02d}"
        agent = PlayerAgent(
            agent_id=pid, model_router=router,
            validator=DefaultActionValidator(), max_retries=1,
        )
        registry.register(pid, agent)
    return registry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLiveRuntimeOrchestration:
    """Complete agent-driven 12-player game through LangGraph."""

    def test_complete_game_with_agent_registry(self) -> None:
        """Game runs from setup to finish with agent-driven decisions."""
        graph = build_game_graph()
        engine = _new_engine()
        registry = _build_registry(engine)

        state: RuntimeState = {
            "game_state": GameState(game_id="live_complete"),
            "engine": engine,
            "agent_registry": registry,
            "wolf_kill_target_id": None,
            "use_antidote": False,
            "poison_target_id": None,
            "seer_target_id": None,
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

        nodes_seen: list[str] = []
        final_gs: GameState | None = None

        for chunk in graph.stream(state, {"recursion_limit": 500}):
            for node_name, output in chunk.items():
                nodes_seen.append(node_name)
                if output and "game_state" in output:
                    final_gs = output["game_state"]
                # Safety: stop after finish_game
                if node_name == "finish_game":
                    break
            else:
                continue
            break

        # Must reach finish_game
        assert "finish_game" in nodes_seen, f"Game did not finish. Nodes seen: {nodes_seen[-20:]}"
        assert final_gs is not None

        # Must have a winner
        assert final_gs.winning_faction in ("good", "werewolf"), (
            f"No winner. Phase: {final_gs.phase}, Events: {len(final_gs.events)}"
        )

        # Must have 12 players
        assert len(final_gs.players) == 12

        # Must have deaths
        assert len(final_gs.deaths) >= 1, "Game should have at least one death"

        # Must have all expected event types
        event_types = {e.type for e in final_gs.events}
        assert "roles_assigned" in event_types
        assert "enter_night" in event_types
        assert "victory" in event_types

    def test_agent_game_produces_speech_events(self) -> None:
        """Agent-driven game produces speech events during day phase."""
        graph = build_game_graph()
        engine = _new_engine()
        registry = _build_registry(engine)

        state: RuntimeState = {
            "game_state": GameState(game_id="live_speech"),
            "engine": engine,
            "agent_registry": registry,
            "wolf_kill_target_id": None,
            "use_antidote": False,
            "poison_target_id": None,
            "seer_target_id": None,
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

        final_gs: GameState | None = None
        for chunk in graph.stream(state, {"recursion_limit": 500}):
            for node_name, output in chunk.items():
                if output and "game_state" in output:
                    final_gs = output["game_state"]
                if node_name == "finish_game":
                    break
            else:
                continue
            break

        assert final_gs is not None
        speech_events = [e for e in final_gs.events if e.type == "speech"]
        assert len(speech_events) >= 1, (
            f"Expected speech events. Event types: { {e.type for e in final_gs.events} }"
        )

    def test_agent_game_produces_vote_events(self) -> None:
        """Agent-driven game collects votes from all alive players."""
        graph = build_game_graph()
        engine = _new_engine()
        registry = _build_registry(engine)

        state: RuntimeState = {
            "game_state": GameState(game_id="live_votes"),
            "engine": engine,
            "agent_registry": registry,
            "wolf_kill_target_id": None,
            "use_antidote": False,
            "poison_target_id": None,
            "seer_target_id": None,
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

        final_gs: GameState | None = None
        for chunk in graph.stream(state, {"recursion_limit": 500}):
            for node_name, output in chunk.items():
                if output and "game_state" in output:
                    final_gs = output["game_state"]
                if node_name == "finish_game":
                    break
            else:
                continue
            break

        assert final_gs is not None
        vote_events = [e for e in final_gs.events if e.type == "vote_resolved"]
        assert len(vote_events) >= 1, "Expected at least one vote_resolved event"

    def test_agent_game_no_private_leaks_in_public_events(self) -> None:
        """Agent-driven game: public events have no private information leaks."""
        graph = build_game_graph()
        engine = _new_engine()
        registry = _build_registry(engine)

        state: RuntimeState = {
            "game_state": GameState(game_id="live_leaks"),
            "engine": engine,
            "agent_registry": registry,
            "wolf_kill_target_id": None,
            "use_antidote": False,
            "poison_target_id": None,
            "seer_target_id": None,
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

        final_gs: GameState | None = None
        for chunk in graph.stream(state, {"recursion_limit": 500}):
            for node_name, output in chunk.items():
                if output and "game_state" in output:
                    final_gs = output["game_state"]
                if node_name == "finish_game":
                    break
            else:
                continue
            break

        assert final_gs is not None

        # Events with visibility markers must not leak to public
        private_types = {"seer_check", "wolf_discussion", "wolf_kill_selected",
                         "wolf_no_kill_declared", "wolf_no_kill_timeout"}
        for event in final_gs.events:
            if event.type in private_types:
                vis = event.payload.get("visibility", "")
                assert vis in ("seer_only", "moderator_only", "private", "werewolf_team_only", ""), (
                    f"Event {event.type} has suspicious visibility: {vis}"
                )

        # Check for role leaks in speech text
        for event in final_gs.events:
            if event.type == "speech":
                text = event.payload.get("text", "")
                # Speech should not contain raw role names in a way that leaks
                # (this is a heuristic check)
                assert "private_intent" not in text.lower()

    def test_agent_game_seer_check_events_exist(self) -> None:
        """Agent-driven game produces seer_check events with proper visibility."""
        graph = build_game_graph()
        engine = _new_engine()
        registry = _build_registry(engine)

        state: RuntimeState = {
            "game_state": GameState(game_id="live_seer"),
            "engine": engine,
            "agent_registry": registry,
            "wolf_kill_target_id": None,
            "use_antidote": False,
            "poison_target_id": None,
            "seer_target_id": None,
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

        final_gs: GameState | None = None
        for chunk in graph.stream(state, {"recursion_limit": 500}):
            for node_name, output in chunk.items():
                if output and "game_state" in output:
                    final_gs = output["game_state"]
                if node_name == "finish_game":
                    break
            else:
                continue
            break

        assert final_gs is not None
        seer_checks = [e for e in final_gs.events if e.type == "seer_check"]
        assert len(seer_checks) >= 1, "Seer should have at least one check event"
        for sc in seer_checks:
            assert sc.payload.get("visibility") in ("seer_only", "moderator_only", "private"), (
                f"seer_check missing visibility: {sc.payload}"
            )

    def test_multiple_games_deterministic(self) -> None:
        """Same seed produces same game outcome with agent registry."""
        engine = _new_engine()
        registry = _build_registry(engine)
        graph = build_game_graph()

        state: RuntimeState = {
            "game_state": GameState(game_id="det_001"),
            "engine": engine,
            "agent_registry": registry,
            "wolf_kill_target_id": None,
            "use_antidote": False,
            "poison_target_id": None,
            "seer_target_id": None,
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

        results: list[GameState] = []
        for _ in range(2):
            final_gs: GameState | None = None
            for chunk in graph.stream(state, {"recursion_limit": 500}):
                for node_name, output in chunk.items():
                    if output and "game_state" in output:
                        final_gs = output["game_state"]
                    if node_name == "finish_game":
                        break
                else:
                    continue
                break
            assert final_gs is not None
            results.append(final_gs)

        # Same game_id should produce same results
        assert results[0].winning_faction == results[1].winning_faction
        assert len(results[0].deaths) == len(results[1].deaths)
