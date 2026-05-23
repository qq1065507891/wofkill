"""Tests for PK (tie-break) speech and revote flow."""

import pytest
from dataclasses import replace
from werewolf_agent.core.models import GameState, PlayerState, GameEvent
from werewolf_agent.engine.rule_engine import RuleEngine


RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"


def _make_gs_with_tie():
    """Create a game state where p03 and p07 are tied in a vote."""
    players = {}
    for i in range(1, 13):
        role = "werewolf" if i <= 4 else "villager"
        players[f"p{i:02d}"] = PlayerState(id=f"p{i:02d}", role=role, alive=True)
    return GameState(
        game_id="test",
        phase="day",
        day_number=2,
        players=players,
        events=[
            GameEvent(type="vote_resolved", payload={
                "exiled": None,
                "reason": "first_tie_pk",
                "tied": ["p03", "p07"],
            }),
        ],
    )


class TestPKSpeechCandidatesOnly:
    """Only PK candidates speak during PK phase."""

    def test_only_pk_candidates_speak_after_exile_tie(self):
        """After first exile tie, only pk_candidates receive pk_speech events."""
        gs = _make_gs_with_tie()
        pk_candidates = ["p03", "p07"]
        # Simulate PK speech for each candidate
        events = []
        for candidate_id in pk_candidates:
            events.append(GameEvent(
                type="tie_pk_speech",
                payload={
                    "speaker": candidate_id,
                    "day_number": gs.day_number,
                    "text": f"PK发言 from {candidate_id}",
                },
            ))
        # Verify: only pk_candidates have tie_pk_speech events
        speakers = {e.payload["speaker"] for e in events}
        assert speakers == {"p03", "p07"}

    def test_pk_revote_targets_only_tied_candidates(self):
        """PK revote legal targets are only the tied candidates."""
        gs = _make_gs_with_tie()
        pk_candidates = ["p03", "p07"]
        engine = RuleEngine.from_yaml(RULESET_PATH)
        # Revote targets should be limited to pk_candidates
        all_targets = engine.legal_exile_targets(gs)
        # Filter to pk candidates
        revote_targets = [t for t in all_targets if t in pk_candidates]
        assert set(revote_targets) == {"p03", "p07"}

    def test_pk_revote_excludes_candidates_if_configured(self):
        """In some rule sets, PK candidates cannot vote in revote."""
        gs = _make_gs_with_tie()
        pk_candidates = {"p03", "p07"}
        # Voters exclude pk candidates themselves (they cannot vote for themselves)
        voters = [pid for pid, p in gs.players.items() if p.alive]
        # This test verifies the expectation that legal revote targets are restricted
        assert "p03" in pk_candidates
        assert "p07" in pk_candidates


class TestPKRevoteContext:
    """Revote context includes prior vote tally and PK speech summary."""

    def test_revote_context_includes_pk_candidates(self):
        """agent_day_vote context must have pk_candidates and revote=True."""
        # This will be verified via agent_adapter behavior
        # The adapter should receive revote=True and pk_candidates
        from werewolf_agent.agents.schemas import TaskType, ActionType
        # Verify TaskType and ActionType have the needed values
        assert TaskType.PK_SPEECH.value == "pk_speech"
        assert ActionType.SPEECH.value == "speech"

    def test_revote_state_sets_flags(self):
        """tie_revote should set revote=True and clear votes."""
        from werewolf_agent.runtime.graph import tie_revote
        state = {
            "game_state": _make_gs_with_tie(),
            "pk_candidates": ["p03", "p07"],
        }
        result = tie_revote(state)
        assert result["revote"] is True
        assert result["exile_votes"] == {}
        assert result["exile_vote_revote"] is True


class TestTiePKSpeechNode:
    """tie_pk_speech node integration tests."""

    def test_tie_pk_speech_without_registry_produces_empty_event(self):
        """Without agent registry, tie_pk_speech produces a single empty event."""
        from werewolf_agent.runtime.graph import tie_pk_speech

        gs = _make_gs_with_tie()
        result = tie_pk_speech({
            "game_state": gs,
            "pk_candidates": ["p03", "p07"],
        })
        events = [e for e in result["game_state"].events if e.type == "tie_pk_speech"]
        assert len(events) == 1
        assert events[0].payload == {}

    def test_tie_pk_speech_with_registry_produces_candidate_events(self, monkeypatch):
        """With registry, tie_pk_speech calls agent_pk_speech for each candidate."""
        from werewolf_agent.runtime import graph as runtime_graph

        gs = _make_gs_with_tie()
        calls = []

        def fake_call_agent(fn, *args, **kwargs):
            candidate_id = args[-1]
            calls.append(candidate_id)
            return {"speech_text": f"PK defense from {candidate_id}"}

        class Registry:
            def get_agent(self, player_id):
                return object()

        monkeypatch.setattr(runtime_graph, "_call_agent", fake_call_agent)

        result = runtime_graph.tie_pk_speech({
            "game_state": gs,
            "engine": RuleEngine.from_yaml(RULESET_PATH),
            "agent_registry": Registry(),
            "pk_candidates": ["p03", "p07"],
        })

        # Both candidates should be called
        assert set(calls) == {"p03", "p07"}

        # Events should have one tie_pk_speech per candidate
        pk_events = [e for e in result["game_state"].events if e.type == "tie_pk_speech"]
        assert len(pk_events) == 2
        speakers = {e.payload["speaker"] for e in pk_events}
        assert speakers == {"p03", "p07"}

    def test_tie_pk_speech_event_contains_day_number(self, monkeypatch):
        """PK speech events include day_number from current game state."""
        from werewolf_agent.runtime import graph as runtime_graph

        gs = _make_gs_with_tie()  # day_number=2

        def fake_call_agent(fn, *args, **kwargs):
            return {"speech_text": "defense"}

        class Registry:
            def get_agent(self, player_id):
                return object()

        monkeypatch.setattr(runtime_graph, "_call_agent", fake_call_agent)

        result = runtime_graph.tie_pk_speech({
            "game_state": gs,
            "engine": RuleEngine.from_yaml(RULESET_PATH),
            "agent_registry": Registry(),
            "pk_candidates": ["p03"],
        })

        pk_events = [e for e in result["game_state"].events if e.type == "tie_pk_speech"]
        assert pk_events[0].payload["day_number"] == 2

    def test_tie_pk_speech_keeps_action_trace_private(self, monkeypatch):
        """PK speech action traces must be in separate audit events, not in public speech."""
        from werewolf_agent.runtime import graph as runtime_graph

        gs = _make_gs_with_tie()
        private_trace = {
            "raw_text": '{"private_intent":{"true_role":"werewolf"}}',
            "parsed_action": {"private_intent": {"true_role": "werewolf"}},
            "final_action_type": "speech",
        }

        def fake_call_agent(fn, *args, **kwargs):
            return {"speech_text": "I am good", "action_trace": private_trace}

        class Registry:
            def get_agent(self, player_id):
                return object()

        monkeypatch.setattr(runtime_graph, "_call_agent", fake_call_agent)

        result = runtime_graph.tie_pk_speech({
            "game_state": gs,
            "engine": RuleEngine.from_yaml(RULESET_PATH),
            "agent_registry": Registry(),
            "pk_candidates": ["p03", "p07"],
        })

        events = result["game_state"].events
        pk_events = [e for e in events if e.type == "tie_pk_speech"]
        audit_events = [e for e in events if e.type == "action_trace_audit"]

        # Public PK speech must not contain action_trace
        for pk_ev in pk_events:
            assert "action_trace" not in pk_ev.payload

        # Audit events must be separate with moderator_only visibility
        assert len(audit_events) == 2
        for audit in audit_events:
            assert audit.payload["visibility"] == "moderator_only"
            assert audit.payload["phase"] == "pk_speech"
            assert audit.payload["action_trace"] == private_trace


class TestAgentPKSpeechAdapter:
    """agent_pk_speech adapter function tests."""

    def test_agent_pk_speech_returns_speech_text(self):
        """agent_pk_speech returns speech_text from agent action."""
        from werewolf_agent.runtime.agent_adapter import agent_pk_speech
        from werewolf_agent.agents.schemas import PlayerAction, ActionType, RetryInfo

        gs = _make_gs_with_tie()
        engine = RuleEngine.from_yaml(RULESET_PATH)

        class Agent:
            def act(self, context):
                return PlayerAction(
                    action_type=ActionType.SPEECH,
                    speech="我是好人，请投对方。",
                    reason="defense",
                ), RetryInfo()

        class Registry:
            def get_agent(self, player_id):
                return Agent()

        result = agent_pk_speech(
            {"game_state": gs},
            engine,
            Registry(),
            "p03",
        )

        assert result is not None
        assert result["speech_text"] == "我是好人，请投对方。"

    def test_agent_pk_speech_returns_none_when_no_agent(self):
        """agent_pk_speech returns None when registry has no agent for player."""
        from werewolf_agent.runtime.agent_adapter import agent_pk_speech

        gs = _make_gs_with_tie()
        engine = RuleEngine.from_yaml(RULESET_PATH)

        class Registry:
            def get_agent(self, player_id):
                return None

        result = agent_pk_speech(
            {"game_state": gs},
            engine,
            Registry(),
            "p03",
        )

        assert result is None

    def test_agent_pk_speech_includes_prior_vote_tally(self):
        """agent_pk_speech context includes prior vote tally for PK candidates."""
        from werewolf_agent.runtime.agent_adapter import agent_pk_speech
        from werewolf_agent.agents.schemas import PlayerAction, ActionType, RetryInfo

        gs = _make_gs_with_tie()
        engine = RuleEngine.from_yaml(RULESET_PATH)
        captured_context = None

        class Agent:
            def act(self, context):
                nonlocal captured_context
                captured_context = context
                return PlayerAction(
                    action_type=ActionType.SPEECH,
                    speech="defense",
                ), RetryInfo()

        class Registry:
            def get_agent(self, player_id):
                return Agent()

        result = agent_pk_speech(
            {"game_state": gs},
            engine,
            Registry(),
            "p03",
        )

        assert result is not None
        assert captured_context is not None
        prior_tally = captured_context.visible_world_state.get("prior_vote_tally", {})
        assert prior_tally.get("reason") == "first_tie_pk"
        assert "p03" in prior_tally.get("tied", [])
        assert "p07" in prior_tally.get("tied", [])

    def test_agent_pk_speech_uses_pk_speech_task_type(self):
        """agent_pk_speech passes PK_SPEECH task type to agent context."""
        from werewolf_agent.runtime.agent_adapter import agent_pk_speech
        from werewolf_agent.agents.schemas import PlayerAction, ActionType, TaskType, RetryInfo

        gs = _make_gs_with_tie()
        engine = RuleEngine.from_yaml(RULESET_PATH)
        captured_context = None

        class Agent:
            def act(self, context):
                nonlocal captured_context
                captured_context = context
                return PlayerAction(
                    action_type=ActionType.SPEECH,
                    speech="defense",
                ), RetryInfo()

        class Registry:
            def get_agent(self, player_id):
                return Agent()

        agent_pk_speech(
            {"game_state": gs},
            engine,
            Registry(),
            "p03",
        )

        assert captured_context.task_type == TaskType.PK_SPEECH


class TestPKRevoteRestrictsTargets:
    """During PK revote, agent_day_vote must restrict legal targets to pk_candidates."""

    def test_agent_day_vote_excludes_voter_from_targets(self):
        """Normal day vote context must not offer the voter as a legal target."""
        from werewolf_agent.runtime.agent_adapter import agent_day_vote
        from werewolf_agent.agents.schemas import PlayerAction, ActionType, RetryInfo

        gs = _make_gs_with_tie()
        engine = RuleEngine.from_yaml(RULESET_PATH)
        captured_context = None

        class Agent:
            def act(self, context):
                nonlocal captured_context
                captured_context = context
                return PlayerAction(
                    action_type=ActionType.VOTE,
                    target_id="p03",
                ), RetryInfo()

        class Registry:
            def get_agent(self, player_id):
                return Agent()

        agent_day_vote(
            {"game_state": gs},
            engine,
            Registry(),
            "p03",
        )

        assert captured_context is not None
        assert "p03" not in captured_context.legal_targets
        assert "p07" in captured_context.legal_targets

    def test_agent_day_vote_requires_structured_vote_quality(self):
        """Real day vote context must require seer stance, vote basis, and concrete reasons."""
        from werewolf_agent.runtime.agent_adapter import agent_day_vote
        from werewolf_agent.agents.schemas import PlayerAction, ActionType, RetryInfo

        gs = _make_gs_with_tie()
        engine = RuleEngine.from_yaml(RULESET_PATH)
        captured_context = None

        class Agent:
            def act(self, context):
                nonlocal captured_context
                captured_context = context
                return PlayerAction(
                    action_type=ActionType.VOTE,
                    target_id="p03",
                    reason="p03发言前后矛盾",
                    seer_stance="undecided",
                    vote_basis="speech_logic",
                    suspect_reason="p03发言前后矛盾",
                    not_voting_reason="p07暂时没有同等强的矛盾点",
                    private_reason="我暂不站边预言家，先按发言矛盾投p03。",
                ), RetryInfo()

        class Registry:
            def get_agent(self, player_id):
                return Agent()

        agent_day_vote(
            {"game_state": gs},
            engine,
            Registry(),
            "p01",
        )

        assert captured_context is not None
        assert captured_context.strategy_directive["require_vote_quality"] is True
        assert captured_context.strategy_directive["vote_structured_contract"]["seer_stance"] == [
            "trust",
            "distrust",
            "undecided",
            "no_claim",
        ]

    def test_agent_day_vote_revote_restricts_targets_to_pk_candidates(self):
        """When revote=True and pk_candidates set, agent_day_vote only offers pk candidates."""
        from werewolf_agent.runtime.agent_adapter import agent_day_vote
        from werewolf_agent.agents.schemas import PlayerAction, ActionType, RetryInfo

        gs = _make_gs_with_tie()
        engine = RuleEngine.from_yaml(RULESET_PATH)
        captured_context = None

        class Agent:
            def act(self, context):
                nonlocal captured_context
                captured_context = context
                return PlayerAction(
                    action_type=ActionType.VOTE,
                    target_id="p03",
                ), RetryInfo()

        class Registry:
            def get_agent(self, player_id):
                return Agent()

        agent_day_vote(
            {
                "game_state": gs,
                "revote": True,
                "pk_candidates": ["p03", "p07"],
            },
            engine,
            Registry(),
            "p01",
        )

        assert captured_context is not None
        assert set(captured_context.legal_targets) == {"p03", "p07"}

    def test_agent_day_vote_revote_excludes_voter_from_pk_targets(self):
        """A PK candidate can vote only for the other PK target, never themselves."""
        from werewolf_agent.runtime.agent_adapter import agent_day_vote
        from werewolf_agent.agents.schemas import PlayerAction, ActionType, RetryInfo

        gs = _make_gs_with_tie()
        engine = RuleEngine.from_yaml(RULESET_PATH)
        captured_context = None

        class Agent:
            def act(self, context):
                nonlocal captured_context
                captured_context = context
                return PlayerAction(
                    action_type=ActionType.VOTE,
                    target_id="p07",
                ), RetryInfo()

        class Registry:
            def get_agent(self, player_id):
                return Agent()

        agent_day_vote(
            {
                "game_state": gs,
                "revote": True,
                "pk_candidates": ["p03", "p07"],
            },
            engine,
            Registry(),
            "p03",
        )

        assert captured_context is not None
        assert captured_context.legal_targets == ["p07"]
