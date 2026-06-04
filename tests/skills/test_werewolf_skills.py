"""P1-K5: handler outputs adapt to task_type (vote vs speech).

Audit finding: each skill handler returns a single `prompt_injectable`
string regardless of whether the LLM is about to SPEAK or VOTE. The
advice is generic — e.g., push_vote always reads as "rally the room to
vote for X" even when the LLM is about to VOTE, not speak.

Fix: add `task_type: str = ""` to `SkillInput`. Handlers branch on
`task_type` where it materially changes advice:
- push_vote:
    - task_type=vote    → "vote for X" (target pick, evidence, no rhetoric)
    - task_type=speech  → "rally others to vote for X" (rhetoric, herd)
    - default           → keep original generic phrasing
- bold_claim: pre-existing static fallback is speech-specific; not
  branched in P1-K5 (other handlers can ignore per plan).

`_inject_skill_output` already accepts a `task_type` parameter (P0-K2)
and now forwards it into SkillInput.
"""

from __future__ import annotations

from werewolf_agent.core.models import GameState, PlayerState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill_gs(day: int = 1) -> GameState:
    """Minimal 6-player GameState for skill handler tests."""
    return GameState(
        ruleset_id="test",
        game_id="test_game",
        phase="speech",
        day_number=day,
        night_number=day,
        players={
            "p01": PlayerState(id="p01", role="werewolf", alive=True),
            "p02": PlayerState(id="p02", role="werewolf", alive=True),
            "p03": PlayerState(id="p03", role="seer", alive=True),
            "p04": PlayerState(id="p04", role="villager", alive=True),
            "p05": PlayerState(id="p05", role="villager", alive=True),
            "p06": PlayerState(id="p06", role="witch", alive=True),
        },
        events=[],
    )


def _build_cognition(gs: GameState, player_id: str):
    from werewolf_agent.cognition.world_state import build_world_state
    from werewolf_agent.cognition.belief import BeliefUpdater
    from werewolf_agent.cognition.contradiction import ContradictionEngine

    world_state = build_world_state(gs)
    updater = BeliefUpdater()
    belief_state = updater.initialize(list(gs.players.keys()), player_id)
    belief_state = updater.update(belief_state, world_state.facts, gs.day_number)
    alerts = ContradictionEngine().detect(world_state.facts, gs.day_number)
    return world_state, belief_state, alerts


# ---------------------------------------------------------------------------
# K5.1: SkillInput exposes task_type
# ---------------------------------------------------------------------------

class TestSkillInputTaskType:
    """P1-K5: SkillInput carries a `task_type` field for handlers."""

    def test_task_type_field_default_empty(self):
        from werewolf_agent.skills.schemas import SkillInput

        inp = SkillInput(role="werewolf", phase="speech")
        assert hasattr(inp, "task_type"), (
            "SkillInput must expose a `task_type` field for P1-K5 branching."
        )
        assert inp.task_type == ""

    def test_task_type_assignable(self):
        from werewolf_agent.skills.schemas import SkillInput

        inp = SkillInput(role="seer", phase="day", task_type="vote")
        assert inp.task_type == "vote"


# ---------------------------------------------------------------------------
# K5.2: push_vote branches on task_type
# ---------------------------------------------------------------------------

class TestPushVoteHandlerBranchesOnTaskType:
    """The push_vote handler must produce different advice for vote vs
    speech task types. Vote phase = "who to actually vote for".
    Speech phase = "how to rally the room to vote for X"."""

    def test_push_vote_handler_branches_on_task_type(self):
        """Static fallback (no game_state) branches on task_type."""
        from werewolf_agent.skills.schemas import SkillInput, SkillName
        from werewolf_agent.skills.werewolf_skills import apply_skill

        # VOTE task type — advice should mention voting-target / 投票目标
        # rather than "号召全场跟随" (rhetoric).
        vote_inp = SkillInput(role="villager", phase="day", task_type="vote")
        vote_out = apply_skill(SkillName.PUSH_VOTE, vote_inp)
        assert vote_out.prompt_injectable != "", "vote-task prompt must be non-empty"
        assert "投票" in vote_out.prompt_injectable, (
            f"vote-task advice should reference voting; got: {vote_out.prompt_injectable!r}"
        )
        # The vote-task fallback should NOT lead with rhetoric ("号召全场跟随").
        assert "号召全场跟随" not in vote_out.prompt_injectable, (
            f"vote-task advice should not be rhetoric-focused; got: {vote_out.prompt_injectable!r}"
        )

        # SPEECH task type — advice should mention rally / 号召.
        speech_inp = SkillInput(role="villager", phase="speech", task_type="speech")
        speech_out = apply_skill(SkillName.PUSH_VOTE, speech_inp)
        assert speech_out.prompt_injectable != "", "speech-task prompt must be non-empty"
        assert "号召" in speech_out.prompt_injectable, (
            f"speech-task advice should mention rallying; got: {speech_out.prompt_injectable!r}"
        )
        # The two outputs should differ for the same handler.
        assert vote_out.prompt_injectable != speech_out.prompt_injectable, (
            "push_vote must produce task_type-specific advice, not a single "
            "generic string for both vote and speech phases."
        )

    def test_push_vote_dynamic_branches_on_task_type(self):
        """With game_state, push_vote still branches on task_type."""
        from werewolf_agent.skills.schemas import SkillInput, SkillName
        from werewolf_agent.skills.werewolf_skills import apply_skill

        gs = _make_skill_gs(day=1)
        ws, bs, alerts = _build_cognition(gs, "p04")  # villager

        vote_inp = SkillInput(
            role="villager", phase="day", day=1,
            game_state=gs, world_state=ws, belief_state=bs,
            contradiction_alerts=alerts, player_id="p04",
            task_type="vote",
        )
        vote_out = apply_skill(SkillName.PUSH_VOTE, vote_inp)
        # Vote-task advice should mark itself as 投票阶段.
        assert "投票阶段" in vote_out.prompt_injectable, (
            f"vote-task dynamic advice should mark itself as 投票阶段; "
            f"got: {vote_out.prompt_injectable!r}"
        )

        speech_inp = SkillInput(
            role="villager", phase="speech", day=1,
            game_state=gs, world_state=ws, belief_state=bs,
            contradiction_alerts=alerts, player_id="p04",
            task_type="speech",
        )
        speech_out = apply_skill(SkillName.PUSH_VOTE, speech_inp)
        # Speech-task advice should mark itself as 发言阶段.
        assert "发言阶段" in speech_out.prompt_injectable, (
            f"speech-task dynamic advice should mark itself as 发言阶段; "
            f"got: {speech_out.prompt_injectable!r}"
        )

    def test_inject_skill_output_forwards_task_type_to_skill_input(self):
        """_inject_skill_output must forward task_type into SkillInput.

        This pins the wiring contract: context.py → SkillInput.task_type
        must not be dropped on the way in. We assert by building a
        scenario where push_vote would emit different prompt strings
        for vote vs speech, then verifying that the corresponding
        output (rendered `skill_tactical_advice`) matches the
        expected task_type-specific phrasing.

        Note: `_inject_skill_output` mutates and returns the input
        `strategy_directive` dict, so we must pass two SEPARATE dicts
        for the two calls (otherwise the second call's
        `skill_tactical_advice` overwrites the first and `result_vote
        is result_speech`).
        """
        from werewolf_agent.runtime.context import _inject_skill_output

        gs = _make_skill_gs(day=1)
        ws, bs, alerts = _build_cognition(gs, "p04")  # villager

        # Vote task: rendered advice should mention 投票阶段.
        # S-05: the 7th positional param is `task_type` (renamed from
        # the misnamed `phase`). Pass the task_type value directly.
        result_vote, _ = _inject_skill_output(
            {}, gs, "p04", ws, bs, alerts, "vote",
        )
        # Speech task: rendered advice should mention 发言阶段.
        result_speech, _ = _inject_skill_output(
            {}, gs, "p04", ws, bs, alerts, "speech",
        )

        vote_advice = result_vote.get("skill_tactical_advice", "")
        speech_advice = result_speech.get("skill_tactical_advice", "")

        # push_vote is in the applicable set for villager / day / speech
        # (it's `applicable_phases=[]` so all phases). The advice chunks
        # for the two task types must differ.
        if vote_advice and speech_advice:
            assert vote_advice != speech_advice, (
                f"vote and speech task_types should produce different rendered "
                f"advice; got the same: {vote_advice!r}"
            )
        # Stronger check: vote-task advice must mention 投票阶段,
        # speech-task advice must mention 发言阶段.
        assert "投票阶段" in vote_advice, (
            f"vote-task rendered advice should mention 投票阶段; "
            f"got: {vote_advice!r}"
        )
        assert "发言阶段" in speech_advice, (
            f"speech-task rendered advice should mention 发言阶段; "
            f"got: {speech_advice!r}"
        )


# ---------------------------------------------------------------------------
# S-02: hybrid faction dispatches wolf-only skills when master is wolf.
# ---------------------------------------------------------------------------

def test_hybrid_with_wolf_master_dispatches_wolf_skills():
    """S-02: when hybrid's master is a werewolf, the hybrid must be able
    to dispatch WOLF-faction skills (e.g. bold_claim, deep_hook, swing_vote).

    Pre-fix: `faction_for_role("hybrid")` unconditionally returns GOOD,
    so a hybrid-with-wolf-master can't see wolf-only advice and ends up
    with the GOOD skills (wolf_pit, protect_power) — wrong for a
    wolf-aligned hybrid.

    Post-fix: `faction_for_role("hybrid", gs=gs)` checks
    `gs.hybrid_master_faction` and returns WOLF when master is werewolf.
    Callers must thread `gs` through to the faction lookup.
    """
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.skills.registry import (
        SkillRegistry, faction_for_role,
    )
    from werewolf_agent.skills.schemas import SkillFaction, SkillInput

    # 6-player game with hybrid whose master is a werewolf.
    gs = GameState(
        ruleset_id="test",
        game_id="test_game",
        phase="speech",
        day_number=1,
        night_number=1,
        players={
            "p01": PlayerState(id="p01", role="werewolf", alive=True),
            "p02": PlayerState(id="p02", role="werewolf", alive=True),
            "p03": PlayerState(id="p03", role="hybrid", alive=True),
            "p04": PlayerState(id="p04", role="villager", alive=True),
            "p05": PlayerState(id="p05", role="villager", alive=True),
            "p06": PlayerState(id="p06", role="seer", alive=True),
        },
        # Hybrid's master is a werewolf → hybrid fights on the wolf side.
        hybrid_master_id="p01",
        hybrid_master_faction="werewolf",
    )

    # The faction lookup itself must surface WOLF for hybrid-with-wolf-master.
    assert faction_for_role("hybrid", gs=gs) == SkillFaction.WOLF, (
        "S-02: hybrid with wolf master must map to WOLF faction; got "
        f"{faction_for_role('hybrid', gs=gs)!r}"
    )

    # The dispatch path: dispatch_for_role must include bold_claim
    # (a WOLF-faction skill) for a hybrid-with-wolf-master in a speech phase.
    reg = SkillRegistry()
    skill_input = SkillInput(
        role="hybrid", phase="speech", day=1,
        game_state=gs, player_id="p03",
    )
    outputs = reg.dispatch_for_role("hybrid", "speech", skill_input, gs=gs)
    skill_names = {o.skill_name for o in outputs}
    assert "bold_claim" in skill_names, (
        f"S-02: hybrid-with-wolf-master should dispatch bold_claim in speech; "
        f"got {skill_names!r}"
    )


# ---------------------------------------------------------------------------
# S-06: prompt_injectable length cap.
# ---------------------------------------------------------------------------

def test_prompt_injectable_length_cap():
    """S-06: late-game review (last_words, review_correction, wolf_pit)
    can produce >1KB prompt_injectable. Cap to 800 chars with truncation
    marker.
    """
    from werewolf_agent.cognition.world_state import (
        StructuredFact, StructuredWorldState,
    )
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.skills.schemas import SkillInput, SkillName
    from werewolf_agent.skills.werewolf_skills import apply_skill

    # 12 players, day=4. Inject player_died + claimed_role + speech
    # facts so the last_words handler produces a multi-KB prompt.
    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 13)
    }
    gs = GameState(
        ruleset_id="test",
        game_id="g",
        phase="speech",
        day_number=4,
        night_number=3,
        players=players,
    )
    ws = StructuredWorldState()
    for i in range(1, 21):
        pid = f"p{(i % 12) + 1:02d}"
        ws.append(StructuredFact(
            fact_type="player_died",
            target_player=pid,
            value="wolf_kill",
            day=i % 4,
        ))
        ws.append(StructuredFact(
            fact_type="claimed_role",
            source_player=pid,
            value="seer",
            day=i % 4,
        ))

    inp = SkillInput(
        role="villager", phase="day", day=4,
        game_state=gs, world_state=ws, belief_state=None,
        contradiction_alerts=[], player_id="p01",
        task_type="speech",
    )
    out = apply_skill(SkillName.LAST_WORDS_ANALYSIS, inp)
    # The cap: prompt_injectable must be <= 800 chars (+ marker).
    assert len(out.prompt_injectable) <= 900, (
        f"S-06: prompt_injectable should be capped near 800 chars; "
        f"got {len(out.prompt_injectable)}"
    )


def test_prompt_injectable_length_cap_forces_marker_on_long_input():
    """S-06: if prompt content is forced > 800 chars, the truncation
    marker must appear.
    """
    from werewolf_agent.cognition.world_state import (
        StructuredFact, StructuredWorldState,
    )
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.skills.schemas import SkillInput, SkillName
    from werewolf_agent.skills.werewolf_skills import apply_skill

    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 13)
    }
    gs = GameState(
        ruleset_id="test",
        game_id="g",
        phase="speech",
        day_number=4,
        night_number=3,
        players=players,
    )
    ws = StructuredWorldState()
    # 30 deaths × 1 claimed_role + 4 speeches → ~3KB prompt
    for i in range(1, 31):
        pid = f"p{(i % 12) + 1:02d}"
        ws.append(StructuredFact(
            fact_type="player_died",
            target_player=pid,
            value="wolf_kill",
            day=(i % 4) + 1,
        ))
        ws.append(StructuredFact(
            fact_type="claimed_role",
            source_player=pid,
            value="seer",
            day=(i % 4) + 1,
        ))
        for _ in range(4):
            ws.append(StructuredFact(
                fact_type="speech",
                source_player=pid,
                value="我是预言家" + "x" * 50,
                day=(i % 4) + 1,
            ))

    inp = SkillInput(
        role="villager", phase="day", day=4,
        game_state=gs, world_state=ws, belief_state=None,
        contradiction_alerts=[], player_id="p01",
        task_type="speech",
    )
    out = apply_skill(SkillName.LAST_WORDS_ANALYSIS, inp)
    assert len(out.prompt_injectable) <= 900, (
        f"S-06: prompt_injectable should be capped; "
        f"got len={len(out.prompt_injectable)}"
    )
    # The marker is '...（已省略）'.
    assert "..." in out.prompt_injectable, (
        f"S-06: long prompt must include truncation marker; "
        f"got tail: {out.prompt_injectable[-50:]!r}"
    )


# ---------------------------------------------------------------------------
# S-13: SkillOutput no longer has recommended_action / recommended_target.
# ---------------------------------------------------------------------------

def test_skill_output_no_recommended_fields():
    """S-13: remove dead `recommended_action` and `recommended_target`
    fields from SkillOutput (populated by all 12 handlers but never
    read by any consumer).
    """
    from dataclasses import fields
    from werewolf_agent.skills.schemas import SkillOutput

    field_names = {f.name for f in fields(SkillOutput)}
    assert "recommended_action" not in field_names, (
        f"S-13: SkillOutput should not have recommended_action; "
        f"got fields={field_names!r}"
    )
    assert "recommended_target" not in field_names, (
        f"S-13: SkillOutput should not have recommended_target; "
        f"got fields={field_names!r}"
    )


# ---------------------------------------------------------------------------
# S-03: swing_vote_handler recommends night_kill in wolf_discussion.
# ---------------------------------------------------------------------------

def test_swing_vote_handler_wolf_discussion_recommends_night_kill():
    """S-03: `swing_vote_handler` in `wolf_discussion` is a NIGHT phase:
    the wolves are picking a night-kill target, not a day-vote target.

    Pre-fix: the handler always returns `recommended_action="vote"` even
    when `task_type == "wolf_discussion"`, which the wolf-discussion
    prompt builder would interpret as "vote for this player during the
    day" — wrong; the wolf needs a night-kill recommendation.

    Post-fix: branch on `inp.task_type`; for `wolf_discussion`, set
    `recommended_action="night_kill"` and rephrase the prompt to make
    the night-kill semantics explicit.
    """
    from werewolf_agent.skills.schemas import SkillInput, SkillName
    from werewolf_agent.skills.werewolf_skills import apply_skill

    gs = _make_skill_gs(day=1)
    # Build a wolf-discussion skill input. Day=1, role=werewolf, and
    # task_type=wolf_discussion. The handler receives an empty/no
    # fact-filled world state, so pressure-based logic still works.
    inp = SkillInput(
        role="werewolf", phase="night", day=1,
        game_state=gs, world_state=None, belief_state=None,
        contradiction_alerts=[], player_id="p01",
        task_type="wolf_discussion",
    )
    out = apply_skill(SkillName.SWING_VOTE, inp)
    # S-13: recommended_action is removed. Verify the prompt
    # explicitly marks itself as a night-kill (狼队夜杀) so the
    # downstream prompt builder renders the right action.
    assert "夜杀" in out.prompt_injectable, (
        f"S-03: swing_vote in wolf_discussion should mark itself as "
        f"夜杀; got: {out.prompt_injectable!r}"
    )
