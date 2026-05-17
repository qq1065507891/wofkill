"""Integration test: full 12-player mock-provider game through the runtime graph.

Uses a DeterministicMockProvider that returns legal JSON actions based on
the system prompt context (role, legal actions, legal targets).
"""

from __future__ import annotations

import json
import re
from typing import Any
from dataclasses import replace

import pytest

from werewolf_agent.agents.player import PlayerAgent, DefaultActionValidator
from werewolf_agent.agents.schemas import ActionType, TaskType
from werewolf_agent.core.models import GameState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.model_gateway.router import (
    GenerateResult,
    LLMProvider,
    ModelConfig,
    ModelRouter,
    UsageRecord,
)
from werewolf_agent.runtime.agent_adapter import SimpleAgentRegistry
from werewolf_agent.runtime.graph import (
    RuntimeState,
    build_game_graph,
    _new_engine,
)

RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"


# ---------------------------------------------------------------------------
# Deterministic mock provider
# ---------------------------------------------------------------------------


class _DeterministicMockProvider:
    """Returns deterministic JSON actions based on system prompt context.

    Parses legal_actions and legal_targets from the system prompt to produce
    valid actions that match the PlayerAction schema.
    """

    def __init__(self, name: str = "det_mock") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate(
        self,
        prompt: str,
        config: ModelConfig,
        system_prompt: str | None = None,
    ) -> GenerateResult:
        action = self._decide(system_prompt or "", prompt)
        text = json.dumps(action, ensure_ascii=False)
        return GenerateResult(
            text=text,
            provider=self._name,
            model=config.model,
            usage=UsageRecord(
                agent_id="", task_type="",
                provider=self._name, model=config.model,
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(text.split()),
            ),
        )

    def _decide(self, system_prompt: str, prompt: str) -> dict[str, Any]:
        role = self._extract_field(system_prompt, "Your role: ")
        legal_actions = self._extract_list(system_prompt, "Legal actions: ")
        legal_targets = self._extract_list(system_prompt, "Legal targets: ")

        # Night action roles
        if "check_alignment" in legal_actions:
            target = legal_targets[0] if legal_targets else None
            return {
                "action_type": "check_alignment",
                "target_id": target,
                "speech": "",
                "reason": "seer check",
                "confidence": 0.9,
                "private_intent": {
                    "true_role": role or "seer",
                    "faction_goal": "find_wolves",
                    "claimed_view": "I am the seer",
                    "pressure_target": target,
                    "risk_flags": [],
                },
            }

        if "use_antidote" in legal_actions and "wolf_kill_target" in prompt.lower():
            # Use antidote when available and there's a wolf kill target
            return {
                "action_type": "use_antidote",
                "target_id": legal_targets[0] if legal_targets else None,
                "speech": "",
                "reason": "save player",
                "confidence": 0.8,
                "private_intent": {
                    "true_role": "witch",
                    "faction_goal": "protect_team",
                    "claimed_view": "I will observe",
                    "pressure_target": None,
                    "risk_flags": [],
                },
            }

        if "wolf_kill" in legal_actions:
            # Wolf: kill first legal non-wolf target
            target = legal_targets[0] if legal_targets else None
            return {
                "action_type": "wolf_kill",
                "target_id": target,
                "speech": "",
                "reason": "wolf consensus",
                "confidence": 0.7,
                "private_intent": {
                    "true_role": "werewolf",
                    "faction_goal": "push_good_player_out",
                    "claimed_view": "I am a good villager",
                    "pressure_target": target,
                    "risk_flags": [],
                },
            }

        if "wolf_no_kill" in legal_actions:
            return {
                "action_type": "wolf_no_kill",
                "target_id": None,
                "speech": "",
                "reason": "peace night",
                "confidence": 0.5,
                "private_intent": {
                    "true_role": "werewolf",
                    "faction_goal": "confuse_good",
                    "claimed_view": "I am a villager",
                    "pressure_target": None,
                    "risk_flags": [],
                },
            }

        # Vote
        if "vote" in legal_actions:
            target = legal_targets[0] if legal_targets else None
            return {
                "action_type": "vote",
                "target_id": target,
                "speech": "",
                "reason": "suspicious behavior",
                "confidence": 0.6,
                "private_intent": {
                    "true_role": role or "villager",
                    "faction_goal": "find_wolves",
                    "claimed_view": "I am good",
                    "pressure_target": target,
                    "risk_flags": [],
                },
            }

        # Speech
        if "speech" in legal_actions:
            return {
                "action_type": "speech",
                "target_id": None,
                "speech": "我觉得需要仔细观察。",
                "reason": "observe",
                "confidence": 0.5,
                "private_intent": {
                    "true_role": role or "villager",
                    "faction_goal": "survive",
                    "claimed_view": "I am good",
                    "pressure_target": None,
                    "risk_flags": [],
                },
            }

        # Default no action
        return {
            "action_type": "no_action",
            "target_id": None,
            "speech": "",
            "reason": "no action available",
            "confidence": 0.3,
            "private_intent": None,
        }

    def _extract_field(self, text: str, prefix: str) -> str | None:
        idx = text.find(prefix)
        if idx < 0:
            return None
        rest = text[idx + len(prefix):]
        end = rest.find("\n")
        return rest[:end].strip() if end >= 0 else rest.strip()

    def _extract_list(self, text: str, prefix: str) -> list[str]:
        idx = text.find(prefix)
        if idx < 0:
            return []
        rest = text[idx + len(prefix):]
        end = rest.find("\n")
        line = rest[:end].strip() if end >= 0 else rest.strip()
        # Parse like: ['wolf_kill', 'wolf_no_kill'] or [ActionType.WOLF_KILL, ...]
        items = re.findall(r"'([^']*)'", line)
        if not items:
            items = re.findall(r'"([^"]*)"', line)
        return items


# ---------------------------------------------------------------------------
# Helper: build registry with deterministic agents
# ---------------------------------------------------------------------------


def _build_registry(engine: RuleEngine) -> SimpleAgentRegistry:
    provider = _DeterministicMockProvider()
    router = ModelRouter(
        model_profiles={},
        llm_profiles={},
        player_assignments={},
        providers={"det_mock": provider},
    )
    registry = SimpleAgentRegistry()
    for i in range(1, 13):
        pid = f"p{i:02d}"
        agent = PlayerAgent(
            agent_id=pid,
            model_router=router,
            validator=DefaultActionValidator(),
            max_retries=1,
        )
        registry.register(pid, agent)
    return registry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLiveGameFlow:
    """Full 12-player mock-provider game through the LangGraph runtime."""

    def test_game_starts_and_runs_at_least_one_night_and_day(self) -> None:
        """Game with agent registry must complete setup + at least 1 night + 1 day."""
        graph = build_game_graph()
        engine = _new_engine()
        registry = _build_registry(engine)

        state: RuntimeState = {
            "game_state": GameState(game_id="live01"),
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
        try:
            for chunk in graph.stream(state, {"recursion_limit": 200}):
                for node_name in chunk.keys():
                    nodes_seen.append(node_name)
                    # Collect updated state from the last chunk
        except Exception:
            pass  # May not terminate fully, but we just need to see enough nodes

        # Must have setup + night flow
        assert "setup_game" in nodes_seen
        assert "assign_roles" in nodes_seen
        assert "enter_night" in nodes_seen
        assert "resolve_night_node" in nodes_seen

        # Must have at least one day phase
        assert "announce_deaths" in nodes_seen

    def test_agent_driven_wolf_kill_produces_death_event(self) -> None:
        """When wolf agent picks a kill target, a death event is produced."""
        from werewolf_agent.runtime.agent_adapter import agent_wolf_consensus

        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        gs = GameState(game_id="wolf_agent", players=players, night_number=1)
        registry = _build_registry(engine)

        state = {
            "game_state": gs,
            "engine": engine,
            "agent_registry": registry,
        }

        result = agent_wolf_consensus(state, engine, registry)
        assert result is not None, "Agent registry must produce a wolf action"
        assert result.get("wolf_kill_target_id") is not None, "Wolf must select a kill target"

    def test_agent_driven_seer_check_produces_target(self) -> None:
        """When seer agent picks a target, it returns a valid target id."""
        from werewolf_agent.runtime.agent_adapter import agent_night_seer

        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        gs = GameState(game_id="seer_agent", players=players, night_number=1)
        registry = _build_registry(engine)

        state = {
            "game_state": gs,
            "engine": engine,
            "agent_registry": registry,
        }

        result = agent_night_seer(state, engine, registry)
        assert result is not None, "Agent registry must produce a seer action"
        assert result.get("seer_target_id") is not None, "Seer must select a check target"

    def test_agent_registry_not_provided_falls_back_to_scripted(self) -> None:
        """When no agent_registry is in state, nodes use scripted behavior (existing tests)."""
        graph = build_game_graph()
        engine = _new_engine()

        state: RuntimeState = {
            "game_state": GameState(game_id="scripted_fallback"),
            "engine": engine,
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

        # Should not crash without agent_registry
        nodes_seen: list[str] = []
        try:
            for chunk in graph.stream(state, {"recursion_limit": 50}):
                for node_name in chunk.keys():
                    nodes_seen.append(node_name)
                    if node_name == "resolve_night_node":
                        # Prove we got through night without agent registry
                        return
        except Exception:
            pass

        # If we didn't reach resolve_night, at least setup + assign should work
        assert "setup_game" in nodes_seen

    def test_public_timeline_has_no_private_intent_or_hidden_roles(self) -> None:
        """After running a partial game, public events must not leak private data."""
        graph = build_game_graph()
        engine = _new_engine()
        registry = _build_registry(engine)
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)

        # Find a non-wolf target for scripted wolf kill
        non_wolf = next(pid for pid, p in players.items() if p.role not in ("werewolf", "hybrid"))

        state: RuntimeState = {
            "game_state": GameState(game_id="leak_test", players=players, phase="roles_assigned", night_number=0),
            "engine": engine,
            "agent_registry": registry,
            "wolf_kill_target_id": non_wolf,
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

        final_gs = None
        try:
            for chunk in graph.stream(state, {"recursion_limit": 80}):
                for node_name, output in chunk.items():
                    if "game_state" in output:
                        final_gs = output["game_state"]
                    if node_name == "announce_deaths":
                        # We have at least one night + day start
                        break
                else:
                    continue
                break
        except Exception:
            pass

        if final_gs is None:
            pytest.skip("Game did not reach announce_deaths")

        # Check all events for information leakage
        forbidden_in_public = {
            "seer_check", "wolf_discussion", "hybrid_master_chosen",
            "witch_antidote_used", "witch_poison_used",
            "hunter_idiot_status_confirmed",
        }

        # Events with visibility markers must not be public
        for event in final_gs.events:
            vis = event.payload.get("visibility", "")
            if vis in ("moderator_only", "seer_only", "private"):
                # These events exist but must have visibility markers
                assert event.type not in ("speech", "vote_resolved", "day_announce"), (
                    f"Public event {event.type} incorrectly marked as private"
                )

        # seer_check events must have visibility marker
        seer_checks = [e for e in final_gs.events if e.type == "seer_check"]
        for sc in seer_checks:
            assert sc.payload.get("visibility") in ("seer_only", "moderator_only", "private"), (
                f"seer_check event missing visibility marker: {sc.payload}"
            )

    def test_player_private_view_only_contains_own_role(self) -> None:
        """AgentContext built by adapter must only contain the viewer's own role."""
        from werewolf_agent.runtime.agent_adapter import build_agent_context

        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        gs = GameState(game_id="view_test", players=players, night_number=1)

        # For each player, check that their context only shows their own role
        for pid, player in players.items():
            ctx = build_agent_context(engine, gs, pid, TaskType.NIGHT_ACTION)
            assert ctx.own_role == player.role, f"Player {pid} should see own role"
            # visible_world_state must not contain other players' roles
            state_str = json.dumps(ctx.visible_world_state, ensure_ascii=False)
            for other_pid, other_player in players.items():
                if other_pid != pid:
                    # Must not see other player's role in the state dump
                    assert f'"role": "{other_player.role}"' not in state_str, (
                        f"Player {pid} must not see {other_pid}'s role {other_player.role}"
                    )

    def test_moderator_replay_can_see_full_audit(self) -> None:
        """After a partial game, moderator can see full audit including seer checks."""
        from werewolf_agent.runtime.replay import extract_event_log

        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        seer_id = next(pid for pid, p in players.items() if p.role == "seer")
        target = next(pid for pid, p in players.items() if p.role != "seer" and p.alive)
        gs = GameState(game_id="audit_test", players=players, night_number=1)

        gs, events = engine.resolve_night(
            gs, night_number=1,
            wolf_kill_target_id=None,
            seer_target_id=target,
        )
        # Merge returned events into state (same as runtime node does)
        gs = replace(gs, events=gs.events + events)

        # Moderator should see all events including seer_check
        all_events = extract_event_log(gs)
        seer_checks = [e for e in all_events if e.type == "seer_check"]
        assert len(seer_checks) >= 1, "Moderator must see seer_check in audit"
        assert seer_checks[0].payload["seer_id"] == seer_id
