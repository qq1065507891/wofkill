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
            # S-07: skill_tactical_advice is a list of {skill, advice,
            # confidence} dicts. Compare the joined-advice strings.
            def _join_advice(lst):
                if isinstance(lst, list):
                    return "\n".join(
                        e.get("advice", "") for e in lst
                        if isinstance(e, dict)
                    )
                return lst
            vote_joined = _join_advice(vote_advice)
            speech_joined = _join_advice(speech_advice)
            assert vote_joined != speech_joined, (
                f"vote and speech task_types should produce different rendered "
                f"advice; got the same: {vote_joined!r}"
            )
        # Stronger check: vote-task advice must mention 投票阶段,
        # speech-task advice must mention 发言阶段.
        def _join_advice(lst):
            if isinstance(lst, list):
                return "\n".join(
                    e.get("advice", "") for e in lst
                    if isinstance(e, dict)
                )
            return lst
        assert "投票阶段" in _join_advice(vote_advice), (
            f"vote-task rendered advice should mention 投票阶段; "
            f"got: {vote_advice!r}"
        )
        assert "发言阶段" in _join_advice(speech_advice), (
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
# S-18: last_words_handler static/dynamic prompt parity.
# ---------------------------------------------------------------------------

def test_last_words_handler_static_dynamic_parity():
    """S-18: last_words has 3 branches:
    1. gs is None  → static fallback prompt
    2. gs given, ws is None → "no-ws" branch (currently duplicates fallback)
    3. gs and ws both given → dynamic analysis

    The first two MUST produce the same prompt. The third may differ
    (dynamic analysis is the value-add).
    """
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
        day_number=1,
        night_number=1,
        players=players,
    )

    # Branch 1: gs is None (static fallback)
    static_inp = SkillInput(
        role="villager", phase="day", day=1,
        game_state=None, world_state=None, belief_state=None,
        contradiction_alerts=[], player_id="p01",
        task_type="speech",
    )
    static_out = apply_skill(SkillName.LAST_WORDS_ANALYSIS, static_inp)

    # Branch 2: gs given, world_state is None (no-ws branch)
    no_ws_inp = SkillInput(
        role="villager", phase="day", day=1,
        game_state=gs, world_state=None, belief_state=None,
        contradiction_alerts=[], player_id="p01",
        task_type="speech",
    )
    no_ws_out = apply_skill(SkillName.LAST_WORDS_ANALYSIS, no_ws_inp)

    # Parity: same prompt
    assert static_out.prompt_injectable == no_ws_out.prompt_injectable, (
        f"S-18: last_words static-fallback and no-ws branch must match; "
        f"got static={static_out.prompt_injectable!r} vs "
        f"no_ws={no_ws_out.prompt_injectable!r}"
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
# S-10: counter_claim_handler branches on inp.role.
# ---------------------------------------------------------------------------

def test_counter_claim_branch_by_role():
    """S-10: counter_claim must give different advice for seer vs
    werewolf. Seer countering wolf = "defend my real check result".
    Wolf countering real seer = "fabricate timeline to match my
    fake-seer story".
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
    # Make p01 a seer, p05 a werewolf, p08 a "claimant" who is a wolf
    players["p01"] = PlayerState(id="p01", role="seer", alive=True)
    players["p05"] = PlayerState(id="p05", role="werewolf", alive=True)
    gs = GameState(
        ruleset_id="test",
        game_id="g",
        phase="speech",
        day_number=1,
        night_number=1,
        players=players,
    )
    ws = StructuredWorldState()
    ws.append(StructuredFact(
        fact_type="claimed_role", source_player="p05", value="seer",
        day=1,
    ))

    seer_inp = SkillInput(
        role="seer", phase="speech", day=1,
        game_state=gs, world_state=ws, belief_state=None,
        contradiction_alerts=[], player_id="p01",
        task_type="speech",
    )
    seer_out = apply_skill(SkillName.COUNTER_CLAIM, seer_inp)
    wolf_inp = SkillInput(
        role="werewolf", phase="speech", day=1,
        game_state=gs, world_state=ws, belief_state=None,
        contradiction_alerts=[], player_id="p05",
        task_type="speech",
    )
    wolf_out = apply_skill(SkillName.COUNTER_CLAIM, wolf_inp)

    assert seer_out.prompt_injectable != "", "seer advice must be non-empty"
    assert wolf_out.prompt_injectable != "", "wolf advice must be non-empty"
    # The advice must differ by role — seer defends real seer, wolf fabricates.
    assert seer_out.prompt_injectable != wolf_out.prompt_injectable, (
        "S-10: counter_claim must give seer-specific and wolf-specific "
        "advice; got identical strings."
    )
    # seer-side markers: "真预言家", "金水", "查验" (defending real result)
    seer_text = seer_out.prompt_injectable
    assert any(k in seer_text for k in ("真预言家", "金水", "查验", "我查", "我验")), (
        f"S-10: seer counter advice should defend real check result; "
        f"got: {seer_text!r}"
    )
    # wolf-side markers: "假", "时间线", "悍跳" (fabricate timeline)
    wolf_text = wolf_out.prompt_injectable
    assert any(k in wolf_text for k in ("假", "时间线", "悍跳", "站边", "排坑")), (
        f"S-10: wolf counter advice should reference fake / timeline / "
        f"bold-claim framing; got: {wolf_text!r}"
    )


# ---------------------------------------------------------------------------
# S-14: bold_claim_handler does NOT name the fake_seer teammate.
# ---------------------------------------------------------------------------

def test_bold_claim_no_teammate_name():
    """S-14: bold_claim's teammate-skip path must use role-neutral
    phrasing. Naming the fake_seer leaks teammate identity into the
    prompt (a wolf team secret).
    """
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.skills.schemas import SkillInput, SkillName
    from werewolf_agent.skills.werewolf_skills import apply_skill

    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="werewolf", alive=True)
        for i in range(1, 13)
    }
    gs = GameState(
        ruleset_id="test",
        game_id="g",
        phase="speech",
        day_number=1,
        night_number=1,
        players=players,
    )
    # Player p01 is a non-fake_seer wolf (deep cover), teammate p05 is fake_seer.
    inp = SkillInput(
        role="werewolf", phase="speech", day=1,
        game_state=gs, world_state=None, belief_state=None,
        contradiction_alerts=[], player_id="p01",
        task_type="speech",
        extra={"wolf_team_plan": {"fake_seer": "p05"}},
    )
    out = apply_skill(SkillName.BOLD_CLAIM, inp)
    # prompt_injectable must NOT contain p05 (teammate's player_id).
    assert "p05" not in out.prompt_injectable, (
        f"S-14: bold_claim must not name the fake_seer teammate; "
        f"got: {out.prompt_injectable!r}"
    )


# ---------------------------------------------------------------------------
# S-11: protect_power includes idiot in power_roles.
# ---------------------------------------------------------------------------

def test_protect_power_includes_idiot():
    """S-11: protect_power's power_roles set must include 'idiot'.
    Post-reveal idiot is a confirmed good player who needs protection.
    """
    from werewolf_agent.cognition.belief import BeliefUpdater
    from werewolf_agent.cognition.world_state import build_world_state
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.skills.schemas import SkillInput, SkillName
    from werewolf_agent.skills.werewolf_skills import apply_skill

    # 12 players, p01 = seer, p08 = revealed idiot.
    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 13)
    }
    players["p01"] = PlayerState(id="p01", role="seer", alive=True)
    players["p08"] = PlayerState(id="p08", role="idiot", alive=True)
    gs = GameState(
        ruleset_id="test",
        game_id="g",
        phase="speech",
        day_number=2,
        night_number=2,
        players=players,
    )
    ws = build_world_state(gs)
    # Inject an idiot_revealed fact for p08 (post-exile state) +
    # 3 votes on p08.
    from werewolf_agent.cognition.world_state import StructuredFact
    ws.append(StructuredFact(
        fact_type="idiot_revealed", target_player="p08", value="revealed_idiot",
    ))
    for voter in ("p02", "p03", "p04"):
        ws.append(StructuredFact(
            fact_type="vote", source_player=voter,
            target_player="p08", day=2, value="voted_for",
        ))
    bs = BeliefUpdater().initialize(list(gs.players.keys()), "p05")
    bs = BeliefUpdater().update(bs, ws.facts, gs.day_number)

    # Sanity: p08's top_role_guess should be (idiot, 1.0)
    p08_top = bs.beliefs["p08"].top_role_guess()
    assert p08_top[0] == "idiot", (
        f"Test setup: p08 should have top_role_guess='idiot'; got {p08_top!r}"
    )

    inp = SkillInput(
        role="villager", phase="speech", day=2,
        game_state=gs, world_state=ws, belief_state=bs,
        contradiction_alerts=[], player_id="p05",
        task_type="speech",
    )
    out = apply_skill(SkillName.PROTECT_POWER, inp)
    # After S-11: idiot is in power_roles, so p08 is added to at_risk.
    # The prompt must mention p08 explicitly.
    assert "p08" in out.prompt_injectable, (
        f"S-11: protect_power should treat confirmed idiot p08 as a power "
        f"role to protect; got prompt={out.prompt_injectable!r}"
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


# ---------------------------------------------------------------------------
# NEW-MANIFEST-A: review_correction manifest name matches the enum value.
# ---------------------------------------------------------------------------


def test_manifest_name_matches_enum() -> None:
    """NEW-MANIFEST-A: the legacy review_correction.yaml manifest's
    `name:` field used the value `review_correction`, while the
    SkillName enum's value is `review_correct` (REVIEW_CORRECTION =
    'review_correct' in schemas.py:31). The two drifted — the manifest
    used a name the enum never produced. The fix is to align the
    manifest's `name:` with the enum.

    Assert: every manifest YAML under `manifests/` (legacy) has a
    `name:` that is a valid SkillName enum value.
    """
    import yaml as _yaml
    from pathlib import Path as _Path
    from werewolf_agent.skills import werewolf_skills as _ws
    from werewolf_agent.skills.schemas import SkillName

    manifest_dir = _Path(_ws.__file__).parent / "manifests"
    valid_names = {n.value for n in SkillName}

    mismatches: list[tuple[str, str, str]] = []
    for yf in sorted(manifest_dir.glob("*.yaml")):
        data = _yaml.safe_load(yf.read_text(encoding="utf-8")) or {}
        manifest_name = data.get("name")
        if manifest_name is None:
            continue
        if manifest_name not in valid_names:
            mismatches.append(
                (yf.name, manifest_name, f"valid={sorted(valid_names)}")
            )
    assert not mismatches, (
        f"NEW-MANIFEST-A: manifest `name:` must match a SkillName enum "
        f"value. Mismatches: {mismatches!r}"
    )


# ---------------------------------------------------------------------------
# NEW-S19-B: wolf_pit seer_check_claim branch skips dead players.
# ---------------------------------------------------------------------------


def test_wolf_pit_seer_check_skips_dead_players() -> None:
    """NEW-S19-B: wolf_pit's seer_check_claim branch must NOT add a
    dead target to `suspects` / `excluded`. Pre-fix, the branch only
    checked `if target and ...` but not `gs.players[target].alive`.
    A dead player previously tagged as "wolf" by a seer_check_claim
    would be added to suspects; the S-19 filter in context.py would
    then either drop the whole advice (because the dead player is
    outside legal_targets) or — worse — surface the dead player to
    the LLM as a vote target.
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
    # p05 is the dead target. p07 is the seer who claimed the check.
    players["p05"] = PlayerState(id="p05", role="villager", alive=False)
    players["p07"] = PlayerState(id="p07", role="seer", alive=True)
    gs = GameState(
        ruleset_id="test",
        game_id="g",
        phase="speech",
        day_number=2,
        night_number=2,
        players=players,
    )
    ws = StructuredWorldState()
    # Seer p07 claims p05 is wolf (BUT p05 is dead).
    ws.append(StructuredFact(
        fact_type="seer_check_claim",
        source_player="p07", target_player="p05",
        value="wolf", day=1,
    ))
    # Seer p07 also claims p09 is good (p09 alive).
    ws.append(StructuredFact(
        fact_type="seer_check_claim",
        source_player="p07", target_player="p09",
        value="good", day=1,
    ))

    inp = SkillInput(
        role="werewolf", phase="speech", day=2,
        game_state=gs, world_state=ws, belief_state=None,
        contradiction_alerts=[], player_id="p01",
        task_type="speech",
    )
    out = apply_skill(SkillName.WOLF_PIT_ANALYSIS, inp)
    text = out.prompt_injectable
    # NEW-S19-B: dead p05 must NOT appear in the suspect list (or
    # anywhere as a recommendation).
    # The suspect line uses `(p05...)` style with parentheses.
    # Allow for plain mention (e.g. "嫌疑区(0人)") but flag `p05(...)`
    # or `p05(被...)` patterns as suspect/exclude entries.
    assert "p05(被p07" not in text, (
        f"NEW-S19-B: dead p05 must NOT appear in wolf_pit suspect or "
        f"exclude list. Got prompt: {text!r}"
    )
    # Sanity: alive p09 IS mentioned in the exclude (good check).
    # The exclude line uses `(p09...)` style.
    assert "p09(被p07发金水)" in text, (
        f"Sanity: alive p09 (good check) must be in exclude list. "
        f"Got prompt: {text!r}"
    )


# ---------------------------------------------------------------------------
# NEW-S19-D: find_power skips dead players.
# ---------------------------------------------------------------------------


def test_find_power_skips_dead_players() -> None:
    """NEW-S19-D: find_power iterates bs.beliefs.items() without
    checking `gs.players[pid].alive`. A dead player with high
    role probability (e.g. a real seer who was just wolf-killed) would
    be added to candidates. Their prompt then names a dead player as
    a power-role holder, which:
    (a) is misleading (they can't act on the info),
    (b) triggers the S-19 illegal-target filter downstream.
    Add an alive check that mirrors the wolf_pit belief-state loop.
    """
    from werewolf_agent.cognition.belief import BeliefUpdater
    from werewolf_agent.cognition.world_state import build_world_state
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.skills.schemas import SkillInput, SkillName
    from werewolf_agent.skills.werewolf_skills import apply_skill

    # 12 players, p03 = seer (dead), p07 = seer (alive).
    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 13)
    }
    players["p03"] = PlayerState(id="p03", role="seer", alive=False)  # dead seer
    players["p07"] = PlayerState(id="p07", role="seer", alive=True)   # alive seer
    gs = GameState(
        ruleset_id="test",
        game_id="g",
        phase="speech",
        day_number=2,
        night_number=2,
        players=players,
    )
    ws = build_world_state(gs)
    bs = BeliefUpdater().initialize(list(gs.players.keys()), "p05")
    bs = BeliefUpdater().update(bs, ws.facts, gs.day_number)
    # Manually set high seer probability for the dead p03.
    bs.beliefs["p03"].role_probabilities = {"seer": 0.9, "werewolf": 0.05, "villager": 0.05}
    # And high seer probability for the alive p07.
    bs.beliefs["p07"].role_probabilities = {"seer": 0.9, "werewolf": 0.05, "villager": 0.05}

    inp = SkillInput(
        role="villager", phase="speech", day=2,
        game_state=gs, world_state=ws, belief_state=bs,
        contradiction_alerts=[], player_id="p05",
        task_type="speech",
    )
    out = apply_skill(SkillName.FIND_POWER, inp)
    text = out.prompt_injectable
    # NEW-S19-D: dead p03 must NOT appear in the candidate list.
    # The candidate line uses "p03 大概率是 seer" style.
    assert "p03" not in text, (
        f"NEW-S19-D: dead p03 must NOT appear in find_power "
        f"candidates. Got prompt: {text!r}"
    )
    # Sanity: alive p07 IS in the candidates.
    assert "p07" in text, (
        f"Sanity: alive p07 (high seer prob) must be in candidates. "
        f"Got prompt: {text!r}"
    )
