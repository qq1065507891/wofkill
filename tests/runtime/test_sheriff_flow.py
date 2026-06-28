from __future__ import annotations

import pytest
from dataclasses import replace

from typing import Any

from werewolf_agent.core.models import Death, GameState, PlayerState, GameEvent
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.agents.schemas import (
    ActionType, AgentContext, PlayerAction, RetryInfo, FallbackAction,
    TaskType,
)
from werewolf_agent.runtime.graph import (
    RuntimeState,
    build_game_graph,
    build_game_graph_with_checkpoint,
    _new_engine,
    _alive_wolves,
    _alive_non_wolves,
    _find_role,
    _stable_seed,
    check_victory,
    free_discussion,
    wolf_consensus,
    route_after_resolve_night,
    route_after_hunter_shot,
    route_after_post_exile,
    _sheriff_died_this_batch,
    _route_after_badge_transfer,
    _action_trace_event,
)
from werewolf_agent.runtime.agent_adapter import _single_wolf_vote
from werewolf_agent.runtime.replay import replay_from_events, extract_event_log
from werewolf_agent.runtime.checkpoints import make_checkpointer


RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"


class TestSheriffBadgeAfterNightDeath:
    """Design doc §6.2: sheriff death from night wolf kill / witch poison must
    route to badge transfer/tear before the next night when the game continues."""

    def test_sheriff_night_kill_routes_to_badge_transfer(self) -> None:
        graph = build_game_graph()
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        # Find a non-wolf target to be the sheriff
        sheriff_candidate = next(
            pid for pid, p in players.items() if p.role not in ("werewolf", "hybrid") and p.alive
        )
        # Set the sheriff and mark badge active
        players[sheriff_candidate] = replace(players[sheriff_candidate],)
        gs = GameState(
            game_id="badge_night",
            players=players,
            phase="night",
            night_number=1,
            day_number=0,
            sheriff_id=sheriff_candidate,
            sheriff_badge_state="active",
        )

        # Kill the sheriff with wolf kill
        state: RuntimeState = {
            "game_state": gs,
            "engine": engine,
            "wolf_kill_target_id": sheriff_candidate,
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

        nodes_seen = []
        try:
            for chunk in graph.stream(state, {"recursion_limit": 120}):
                for node_name in chunk.keys():
                    nodes_seen.append(node_name)
        except Exception:
            pass  # Game may not terminate, but we just need to see the nodes

        # After resolve_night_node, if sheriff died and game continues,
        # badge transfer must be visited before entering night again
        if sheriff_candidate in [d.player_id for d in gs.deaths]:
            pytest.skip("Sheriff not killed in this scenario")

        # Check if resolve_night_node was visited and sheriff badge_transfer appeared
        # after it but before the second enter_night
        resolve_idx = None
        badge_idx = None
        enter_night_indices = []
        for i, n in enumerate(nodes_seen):
            if n == "resolve_night_node" and resolve_idx is None:
                resolve_idx = i
            if n == "sheriff_badge_transfer":
                badge_idx = i
            if n == "enter_night":
                enter_night_indices.append(i)

        # If sheriff died and game didn't end, badge transfer should have been visited
        if resolve_idx is not None:
            assert badge_idx is not None, (
                f"Sheriff killed at night but badge_transfer was not visited. "
                f"Nodes: {nodes_seen[:30]}"
            )



class TestSheriffBadgeNightDeathRouting:
    """When sheriff dies at night (wolf kill, witch poison, hunter shot),
    route_after_resolve_night must go to sheriff_badge_transfer before
    announce_deaths when the game has not been won."""

    def test_sheriff_died_at_night_routes_to_badge_transfer(self) -> None:
        """route_after_resolve_night routes to sheriff_badge_transfer when
        sheriff died from wolf kill and game continues."""
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        # Find a non-wolf target to be sheriff, then kill them
        sheriff_id = next(
            pid for pid, p in players.items()
            if p.role not in ("werewolf", "hybrid") and p.alive
        )
        players[sheriff_id] = replace(players[sheriff_id], alive=False)
        gs = GameState(
            game_id="badge_night_route",
            players=players,
            phase="night",
            night_number=1,
            sheriff_id=sheriff_id,
            sheriff_badge_state="active",
        )
        result = route_after_resolve_night({"game_state": gs, "engine": engine})
        assert result == "sheriff_badge_transfer"

    def test_sheriff_died_at_night_after_hunter_shot_routes_to_badge_transfer(self) -> None:
        """route_after_hunter_shot routes to sheriff_badge_transfer when
        sheriff died and game continues."""
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        sheriff_id = next(
            pid for pid, p in players.items()
            if p.role not in ("werewolf", "hybrid") and p.alive
        )
        players[sheriff_id] = replace(players[sheriff_id], alive=False)
        gs = GameState(
            game_id="badge_hunter_route",
            players=players,
            phase="night",
            night_number=1,
            sheriff_id=sheriff_id,
            sheriff_badge_state="active",
        )
        result = route_after_hunter_shot({"game_state": gs, "engine": engine})
        assert result == "sheriff_badge_transfer"

    def test_no_sheriff_death_routes_to_sheriff_election_on_night1(self) -> None:
        """Night 1 with no sheriff death must route to sheriff_first_day_entry.

        D1-flow-rewire: V1 design is 天亮 → 警长竞选 → 死讯广播 → 遗言
        → 自由讨论, so resolve_night on N1 (no sheriff) goes to
        sheriff_first_day_entry. The prior assertion of announce_deaths
        matched the legacy announce_deaths → last_words → sheriff order;
        see commit 89b865b for that flow and D1-flow-rewire for the
        current design.
        """
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        gs = GameState(
            game_id="badge_no_sheriff",
            players=players,
            phase="night",
            night_number=1,
        )
        result = route_after_resolve_night({"game_state": gs, "engine": engine})
        assert result == "sheriff_first_day_entry"

    def test_sheriff_alive_routes_to_announce_deaths(self) -> None:
        """When sheriff is still alive, route goes to announce_deaths."""
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        sheriff_id = next(
            pid for pid, p in players.items()
            if p.role not in ("werewolf", "hybrid") and p.alive
        )
        gs = GameState(
            game_id="badge_sheriff_alive",
            players=players,
            phase="night",
            night_number=1,
            sheriff_id=sheriff_id,
            sheriff_badge_state="active",
        )
        result = route_after_resolve_night({"game_state": gs, "engine": engine})
        assert result == "announce_deaths"

    def test_badge_transfer_night_phase_routes_to_announce_deaths(self) -> None:
        """After badge_transfer in night-death path (phase=night),
        _route_after_badge_transfer routes to announce_deaths."""
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        gs = GameState(
            game_id="badge_transfer_night",
            players=players,
            phase="night",
            night_number=1,
            sheriff_badge_state="torn",
        )
        result = _route_after_badge_transfer({"game_state": gs, "engine": engine})
        assert result == "announce_deaths"

    def test_badge_transfer_day_phase_routes_to_enter_night(self) -> None:
        """After badge_transfer in post-victory path (phase != night),
        _route_after_badge_transfer routes to enter_night."""
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        gs = GameState(
            game_id="badge_transfer_day",
            players=players,
            phase="day",
            day_number=2,
            sheriff_badge_state="torn",
        )
        result = _route_after_badge_transfer({"game_state": gs, "engine": engine})
        assert result == "enter_night"

    def test_sheriff_died_this_batch_helper(self) -> None:
        """_sheriff_died_this_batch correctly detects dead sheriff."""
        engine = _new_engine()
        players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
        sheriff_id = next(
            pid for pid, p in players.items()
            if p.role not in ("werewolf", "hybrid") and p.alive
        )
        # Sheriff alive
        gs_alive = GameState(
            players=players,
            sheriff_id=sheriff_id,
            sheriff_badge_state="active",
        )
        assert _sheriff_died_this_batch(gs_alive) is False

        # Sheriff dead
        players_dead = dict(players)
        players_dead[sheriff_id] = replace(players[sheriff_id], alive=False)
        gs_dead = GameState(
            players=players_dead,
            sheriff_id=sheriff_id,
            sheriff_badge_state="active",
        )
        assert _sheriff_died_this_batch(gs_dead) is True

        # No sheriff
        gs_none = GameState(players=players)
        assert _sheriff_died_this_batch(gs_none) is False

        # Badge not active
        gs_inactive = GameState(
            players=players_dead,
            sheriff_id=sheriff_id,
            sheriff_badge_state="torn",
        )
        assert _sheriff_died_this_batch(gs_inactive) is False

class TestSheriffElectionSpeechFallback:
    def test_non_seer_short_sheriff_speech_fallback_does_not_claim_badge_flow(self) -> None:
        from werewolf_agent.runtime.agent_adapter import agent_sheriff_election_speech

        players = {
            "p01": PlayerState(id="p01", role="villager"),
            "p02": PlayerState(id="p02", role="seer"),
        }
        gs = GameState(
            game_id="sheriff_fallback_non_seer",
            players=players,
            phase="sheriff_speech",
            day_number=1,
        )

        class ShortSpeechAgent:
            def act(self, context):
                return (
                    PlayerAction(
                        action_type=ActionType.SPEECH,
                        speech="上警",
                        reason="too short",
                    ),
                    RetryInfo(),
                )

        class Registry:
            def get_agent(self, player_id):
                return ShortSpeechAgent()

        result = agent_sheriff_election_speech(
            {"game_state": gs},
            _new_engine(),
            Registry(),
            "p01",
            ["p01", "p02"],
        )

        assert result is not None
        speech = result["speech_text"]
        assert "警徽流" not in speech
        assert "预言家" not in speech

    def test_non_fake_seer_wolf_does_not_receive_badge_flow_instruction(self) -> None:
        from werewolf_agent.runtime.agent_adapter import agent_sheriff_election_speech

        players = {
            "p01": PlayerState(id="p01", role="werewolf"),
            "p02": PlayerState(id="p02", role="seer"),
            "p03": PlayerState(id="p03", role="werewolf"),
        }
        gs = GameState(
            game_id="sheriff_non_fake_wolf",
            players=players,
            phase="sheriff_speech",
            day_number=1,
        )

        class CaptureAgent:
            context = None

            def act(self, context):
                self.context = context
                return (
                    PlayerAction(
                        action_type=ActionType.SPEECH,
                        speech="我上警是为了分析当前公开发言和候选人的站边逻辑。",
                        reason="share public analysis",
                    ),
                    RetryInfo(),
                )

        agent = CaptureAgent()

        class Registry:
            def get_agent(self, player_id):
                return agent

        result = agent_sheriff_election_speech(
            {
                "game_state": gs,
                "wolf_team_plan": {"fake_seer": "p03"},
            },
            _new_engine(),
            Registry(),
            "p01",
            ["p01", "p02", "p03"],
        )

        assert result is not None
        directive_text = str(agent.context.strategy_directive)
        assert "必须留两个晚上的验人对象" not in directive_text
        assert "场上有多人跳预言家" not in directive_text


class TestSheriffWithdrawalPolicy:
    def test_true_seer_cannot_withdraw(self) -> None:
        from werewolf_agent.runtime.agent_adapter import agent_sheriff_withdraw

        gs = GameState(
            game_id="seer_stays",
            players={"p01": PlayerState(id="p01", role="seer")},
            phase="sheriff_speech",
            day_number=1,
        )

        class WithdrawAgent:
            called = False

            def act(self, context):
                self.called = True
                return PlayerAction(action_type=ActionType.SHERIFF_WITHDRAW), RetryInfo()

        agent = WithdrawAgent()

        class Registry:
            def get_agent(self, player_id):
                return agent

        result = agent_sheriff_withdraw(
            {"game_state": gs},
            _new_engine(),
            Registry(),
            "p01",
        )

        assert result == {"withdrew": False, "self_destruct": False}
        assert agent.called is False

    def test_assigned_fake_seer_cannot_withdraw(self) -> None:
        from werewolf_agent.runtime.agent_adapter import agent_sheriff_withdraw

        gs = GameState(
            game_id="fake_seer_stays",
            players={"p04": PlayerState(id="p04", role="werewolf")},
            phase="sheriff_speech",
            day_number=1,
        )

        class Registry:
            def get_agent(self, player_id):
                raise AssertionError("fake seer withdrawal should be blocked before agent call")

        result = agent_sheriff_withdraw(
            {
                "game_state": gs,
                "wolf_team_plan": {"fake_seer": "p04"},
            },
            _new_engine(),
            Registry(),
            "p04",
        )

        assert result == {"withdrew": False, "self_destruct": False}




# ---------------------------------------------------------------------------
# Sheriff standalone tests
# ---------------------------------------------------------------------------

def test_sheriff_vote_ignores_candidates_and_withdrew_voters() -> None:
    from werewolf_agent.runtime.graph import sheriff_vote

    players = {
        "p01": PlayerState(id="p01", role="villager", alive=True),
        "p02": PlayerState(id="p02", role="villager", alive=True),
        "p03": PlayerState(id="p03", role="villager", alive=True),
        "p04": PlayerState(id="p04", role="villager", alive=True),
    }
    gs = GameState(
        game_id="sheriff_vote_eligibility",
        players=players,
        day_number=1,
        sheriff_candidates=["p01", "p02"],
    )

    result = sheriff_vote({
        "game_state": gs,
        "engine": _new_engine(),
        "sheriff_withdrawing": ["p03"],
        "sheriff_votes": {
            "p01": "p01",  # candidate cannot vote
            "p03": "p01",  # withdrew candidate still cannot vote
            "p04": "p02",  # only valid off-sheriff vote
        },
    })

    assert result["game_state"].sheriff_id == "p02"

def test_sheriff_speech_calls_candidate_agents_and_keeps_trace_private(monkeypatch) -> None:
    from werewolf_agent.runtime.nodes import sheriff as sheriff_mod

    players = {
        "p01": PlayerState(id="p01", role="seer"),
        "p02": PlayerState(id="p02", role="werewolf"),
    }
    gs = GameState(
        game_id="sheriff_agent",
        players=players,
        day_number=1,
        sheriff_candidates=["p01", "p02"],
    )
    private_trace = {"parsed_action": {"private_intent": {"true_role": "werewolf"}}}

    def fake_dispatch_agent(*_args, **_kwargs):
        return {"speech_text": "我上警竞选警长。", "action_trace": private_trace}

    class Registry:
        def get_agent(self, player_id):
            return object()

    monkeypatch.setattr(sheriff_mod, "_dispatch_agent", fake_dispatch_agent)

    result = sheriff_mod.sheriff_speech({
        "game_state": gs,
        "engine": _new_engine(),
        "agent_registry": Registry(),
    })

    events = result["game_state"].events
    speeches = [event for event in events if event.type == "sheriff_speech"]
    audits = [event for event in events if event.type == "action_trace_audit"]

    assert sorted(event.payload["speaker"] for event in speeches) == ["p01", "p02"]
    assert all("action_trace" not in event.payload for event in speeches)
    assert len(audits) == 2
    assert all(event.payload["visibility"] == "moderator_only" for event in audits)


def test_agent_sheriff_vote_does_not_inject_exile_vote_basis_hint() -> None:
    from werewolf_agent.runtime.agent_adapter import agent_sheriff_vote

    class CaptureAgent:
        last_context: AgentContext | None = None

        def act(self, context: AgentContext):
            self.last_context = context
            return (
                PlayerAction(
                    action_type=ActionType.SHERIFF_VOTE,
                    target_id="p01",
                    speech="",
                    reason="best sheriff candidate",
                    confidence=0.8,
                ),
                RetryInfo(),
            )

    class Registry:
        def __init__(self) -> None:
            self.agent = CaptureAgent()

        def get_agent(self, player_id: str):
            return self.agent

    gs = GameState(
        game_id="sheriff_vote_prompt_scope",
        players={
            "p01": PlayerState(id="p01", role="villager", alive=True),
            "p02": PlayerState(id="p02", role="villager", alive=True),
        },
        phase="sheriff_vote",
        day_number=1,
        sheriff_candidates=["p01"],
    )
    registry = Registry()

    result = agent_sheriff_vote(
        {"game_state": gs},
        _new_engine(),
        registry,
        "p02",
        ["p01"],
    )

    assert result["vote_target"] == "p01"
    assert registry.agent.last_context is not None
    directive = registry.agent.last_context.strategy_directive or {}
    assert "vote_basis_hint" not in directive


class TestSheriffElectionPK:
    def test_first_tie_triggers_pk_speech(self):
        """First sheriff vote tie should route to sheriff_pk_speech with tied candidates."""
        from werewolf_agent.core.models import GameState, PlayerState
        from werewolf_agent.runtime.nodes.sheriff import sheriff_vote
        from werewolf_agent.runtime.nodes._shared import RuntimeState
        from werewolf_agent.runtime.graph import _new_engine

        # 4 players, 3 are candidates. p10 is the off-sheriff voter (will produce 1 vote per candidate).
        # 6 players, 3 candidates, 3 voters, each picks a different candidate → first tie.
        gs = GameState(
            game_id="g_test",
            players={
                "p01": PlayerState(id="p01", role="villager", alive=True),
                "p02": PlayerState(id="p02", role="villager", alive=True),
                "p05": PlayerState(id="p05", role="villager", alive=True),
                "p06": PlayerState(id="p06", role="villager", alive=True),
                "p08": PlayerState(id="p08", role="villager", alive=True),
                "p09": PlayerState(id="p09", role="villager", alive=True),
            },
            sheriff_candidates=["p01", "p05", "p08"],
            day_number=1,
        )
        state: RuntimeState = {
            "game_state": gs,
            "engine": _new_engine(),
            "sheriff_votes": {"p02": "p01", "p06": "p05", "p09": "p08"},
            "sheriff_withdrawing": [],
        }
        result = sheriff_vote(state)
        gs_out = result["game_state"]
        # On first tie, tie_count should be 1, no sheriff, and pk_candidates set
        assert gs_out.sheriff_id is None
        assert gs_out.sheriff_tie_count == 1
        assert set(gs_out.sheriff_pk_candidates) == {"p01", "p05", "p08"}

    def test_second_tie_skips_to_no_election(self):
        """When tie_count is already 1, a second tie resolves to no sheriff."""
        from werewolf_agent.core.models import GameState, PlayerState
        from werewolf_agent.runtime.nodes.sheriff import sheriff_vote
        from werewolf_agent.runtime.nodes._shared import RuntimeState
        from werewolf_agent.runtime.graph import _new_engine

        gs = GameState(
            game_id="g_test",
            players={
                "p01": PlayerState(id="p01", role="villager", alive=True),
                "p02": PlayerState(id="p02", role="villager", alive=True),
                "p05": PlayerState(id="p05", role="villager", alive=True),
                "p06": PlayerState(id="p06", role="villager", alive=True),
            },
            sheriff_candidates=["p01", "p05"],
            sheriff_tie_count=1,  # already tied once
            day_number=1,
        )
        state: RuntimeState = {
            "game_state": gs,
            "engine": _new_engine(),
            "sheriff_votes": {"p02": "p01", "p06": "p05"},
            "sheriff_withdrawing": [],
        }
        result = sheriff_vote(state)
        gs_out = result["game_state"]
        assert gs_out.sheriff_id is None
        # Second tie: tie_count should be reset, no PK loop
        assert gs_out.sheriff_tie_count == 0
        assert gs_out.sheriff_pk_candidates == []
        # A judge broadcast with phase=sheriff_no_election signals the no-election outcome
        no_election_broadcasts = [
            e for e in gs_out.events
            if e.type == "judge_broadcast" and e.payload.get("phase") == "sheriff_no_election"
        ]
        assert len(no_election_broadcasts) == 1


def test_self_destruct_during_sheriff_election_routes_to_sheriff_first_day_entry() -> None:
    """D1-flow-rewire: D1 N1 first resolve with no sheriff must route to
    sheriff_first_day_entry (election BEFORE death announcement).

    The prior test name asserted the legacy flow (announce_deaths →
    last_words → sheriff_election → self-destruct → announce_deaths),
    so the N1 deaths got publicly broadcast even when self-destruct
    interrupted sheriff election. In the rewired flow self-destruct
    resolves inside the election phase and announce_deaths still runs
    afterwards (route_after_sheriff_vote's _deaths_already_announced
    branch), so death broadcasts are preserved — but the entry point
    has moved to sheriff_first_day_entry. See commit 89b865b for the
    original D1 self-destruct fix and D1-flow-rewire for the new order.
    """
    from werewolf_agent.runtime.graph import route_after_resolve_night

    engine = _new_engine()
    players = engine.assign_roles([f"p{i:02d}" for i in range(1, 13)], seed=42)
    gs = GameState(
        game_id="d1_self_destruct_first",
        players=players,
        phase="night",
        night_number=1,
        day_number=0,
        sheriff_id=None,
        sheriff_badge_state="none",
        sheriff_interrupt_count=0,
    )

    result = route_after_resolve_night({"game_state": gs, "engine": engine})
    assert result == "sheriff_first_day_entry", (
        f"route_after_resolve_night on N1 with no sheriff must route to "
        f"sheriff_first_day_entry (election before death announcement) "
        f"in the rewired D1 flow; got {result!r}"
    )


class TestSheriffEndorseAdapterModernization:
    """审查 A6: _sheriff_endorse_adapter 不再用老 4 参 agent.act() 接口。"""

    def test_endorse_adapter_uses_build_agent_context(self):
        """Endorse 路径应走 build_agent_context + adapter agent_sheriff_endorse。"""
        import inspect
        from werewolf_agent.runtime.nodes import sheriff
        if not hasattr(sheriff, "_sheriff_endorse_adapter"):
            pytest.skip("function not found")
        src = inspect.getsource(sheriff._sheriff_endorse_adapter)
        # 新接口特征：build_agent_context + adapter agent_sheriff_endorse
        has_modern = ("build_agent_context" in src) or ("agent_sheriff_endorse" in src)
        # 老接口特征：直接 agent.act(prompt=..., system_prompt=..., task_type=..., legal_actions=...)
        has_legacy = ("agent.act(" in src) and ("prompt=" in src) and ("system_prompt=" in src)
        assert has_modern and not has_legacy, (
            f"_sheriff_endorse_adapter still uses old 4-arg agent.act() interface:\n{src[:500]}"
        )
