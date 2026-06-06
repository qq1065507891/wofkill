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
# S-15: legacy `skills/manifests/` folder is gone; SKILL.md is the
# single source of truth for skill metadata.
# ---------------------------------------------------------------------------


def test_legacy_manifests_folder_removed() -> None:
    """S-15: the legacy `werewolf_agent/skills/manifests/` folder is
    deleted. SKILL.md frontmatter in the per-skill subdirs is the single
    source of truth for skill metadata. Regression guard against
    re-introducing the legacy folder.
    """
    from pathlib import Path as _Path
    from werewolf_agent.skills import werewolf_skills as _ws
    manifest_dir = _Path(_ws.__file__).parent / "manifests"
    assert not manifest_dir.exists(), (
        f"S-15: legacy manifests/ folder must be deleted; "
        f"found at {manifest_dir}"
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


# ---------------------------------------------------------------------------
# NEW-R4-P1-2: review_correction_handler must branch on role.
# ---------------------------------------------------------------------------


def test_review_correction_handler_wolf_receives_wolf_advice():
    """NEW-R4-P1-2: review_correction_handler must NOT give wolves the
    good-side vote-accuracy review ("未命中狼人，需要反思站边").

    For a werewolf player, ``my_votes`` is a list of day-vote targets.
    Cross-referencing those targets against ``wolf_ids`` is almost
    always 0 — wolves are SUPPOSED to not vote out other wolves, that's
    their team-coordination goal. Calling that a "miss" inverts the
    wolf team's actual objective and confuses the LLM with
    goal-inverted feedback.

    Post-fix: branch on ``inp.role == "werewolf"`` and emit wolf-side
    advice focused on 悍跳 / night kill / teammate vote patterns
    instead. The good-side vote-accuracy branch remains for
    non-werewolf roles.
    """
    from werewolf_agent.cognition.world_state import (
        StructuredFact, StructuredWorldState,
    )
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.skills.schemas import SkillInput, SkillName
    from werewolf_agent.skills.werewolf_skills import apply_skill

    # 12-player game. p01 = werewolf (subject of review).
    # p02, p03, p04 = werewolf teammates. p05..p12 = good-side.
    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 13)
    }
    players["p01"] = PlayerState(id="p01", role="werewolf", alive=True)
    players["p02"] = PlayerState(id="p02", role="werewolf", alive=True)
    players["p03"] = PlayerState(id="p03", role="werewolf", alive=True)
    players["p04"] = PlayerState(id="p04", role="werewolf", alive=True)
    # Day 3 — well into the game, plenty of vote history to review.
    gs = GameState(
        ruleset_id="test",
        game_id="g",
        phase="day",
        day_number=3,
        night_number=2,
        players=players,
    )
    ws = StructuredWorldState()
    # Exile a good player (p05) on Day 1 — wolf team won the vote.
    ws.append(StructuredFact(
        fact_type="vote", source_player="p01", target_player="p05",
        day=1, value="voted_for",
    ))
    ws.append(StructuredFact(
        fact_type="vote", source_player="p02", target_player="p05",
        day=1, value="voted_for",
    ))
    ws.append(StructuredFact(
        fact_type="vote", source_player="p06", target_player="p05",
        day=1, value="voted_for",
    ))
    # Day 1 exile of p05
    ws.append(StructuredFact(
        fact_type="player_died", target_player="p05",
        value="exile", day=1,
    ))
    # Day 2 exile of another good player (p07) — p01 voted with team.
    ws.append(StructuredFact(
        fact_type="vote", source_player="p01", target_player="p07",
        day=2, value="voted_for",
    ))
    ws.append(StructuredFact(
        fact_type="vote", source_player="p03", target_player="p07",
        day=2, value="voted_for",
    ))
    ws.append(StructuredFact(
        fact_type="player_died", target_player="p07",
        value="exile", day=2,
    ))
    # Day 2 night kill of p08 — wolf team's first night kill
    ws.append(StructuredFact(
        fact_type="player_died", target_player="p08",
        value="wolf_kill", day=2,
    ))

    inp = SkillInput(
        role="werewolf", phase="day", day=3,
        game_state=gs, world_state=ws, belief_state=None,
        contradiction_alerts=[], player_id="p01",
        task_type="review",
    )
    out = apply_skill(SkillName.REVIEW_CORRECTION, inp)
    text = out.prompt_injectable

    # NEW-R4-P1-2: wolf-side advice must NOT tell the wolf that its
    # votes "missed" wolves — voting OUT wolves would be betraying
    # the team.
    assert "未命中狼人" not in text, (
        f"NEW-R4-P1-2: review_correction must NOT tell a werewolf "
        f"that their votes 'missed' wolves; got: {text!r}"
    )
    # The advice should reference wolf-specific concerns: 悍跳
    # (bold-claim coordination), night kill chain, or teammate
    # vote patterns. Any of these markers means the role branch fired.
    wolf_markers = ("悍跳", "夜杀", "夜刀", "狼刀", "狼队", "队友",
                    "归票", "放逐链", "队内", "狼同伴")
    assert any(m in text for m in wolf_markers), (
        f"NEW-R4-P1-2: review_correction for a werewolf should "
        f"reference wolf-specific advice (悍跳 / 夜杀 / 狼队 / 队友); "
        f"got: {text!r}"
    )


# ---------------------------------------------------------------------------
# NEW-R4-P2-1: counter_claim hybrid wolf-master receives 悍跳-specific advice.
# ---------------------------------------------------------------------------


def test_counter_claim_hybrid_wolf_master_receives_悍跳_advice() -> None:
    """NEW-R4-P2-1: when a HYBRID with `hybrid_master_faction='werewolf'`
    is faking-seer (counter_claiming a real seer), the counter_claim
    handler must give 悍跳-specific advice (wolf team plan), NOT the
    neutral "对跳建议" villagers receive.

    Pre-fix: `is_wolf = inp.role == 'werewolf'` is False for a hybrid,
    so the static-fallback and dynamic branches both fell through to
    the neutral `else` branch. A hybrid-wolf-master faking-seer then
    got "指出对方漏洞" (villager-style) advice instead of "假验人
    时间线" (wolf-style). That defeats the entire purpose of the
    hybrid-wolf-master mechanic.

    Post-fix: compute `effective_faction = WOLF` when
    `(role == 'werewolf') or (role == 'hybrid' and
    gs.hybrid_master_faction == 'werewolf')`; the handler branches
    on effective_faction so the hybrid gets 悍跳 framing.
    """
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.skills.schemas import SkillInput, SkillName
    from werewolf_agent.skills.werewolf_skills import apply_skill

    # 12 players; p03 is a hybrid whose master is a werewolf.
    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 13)
    }
    players["p03"] = PlayerState(id="p03", role="hybrid", alive=True)
    # p05 is the real seer (claimant); p03 is the hybrid faking seer.
    players["p05"] = PlayerState(id="p05", role="seer", alive=True)
    gs = GameState(
        ruleset_id="test",
        game_id="g",
        phase="speech",
        day_number=1,
        night_number=1,
        players=players,
        hybrid_master_id="p01",  # master is a werewolf (p01)
        hybrid_master_faction="werewolf",
    )

    # No world_state — exercise the static-fallback branch (gs is None
    # is the typical P2 test path).  But here gs is NOT None — we want
    # the dynamic branch (with `world_state=None` is also valid in
    # some handlers but counter_claim's dynamic branch assumes ws).
    from werewolf_agent.cognition.world_state import (
        StructuredFact, StructuredWorldState,
    )
    ws = StructuredWorldState()
    ws.append(StructuredFact(
        fact_type="claimed_role", source_player="p05", value="seer",
        day=1,
    ))

    # Sanity: a VILLAGER role with no wolf-master hybrid context must
    # NOT get the 悍跳 framing.
    villager_inp = SkillInput(
        role="villager", phase="speech", day=1,
        game_state=gs, world_state=ws, belief_state=None,
        contradiction_alerts=[], player_id="p04",
        task_type="speech",
    )
    villager_out = apply_skill(SkillName.COUNTER_CLAIM, villager_inp)
    villager_text = villager_out.prompt_injectable
    assert "悍跳" not in villager_text, (
        f"NEW-R4-P2-1: villager counter_claim must not use 悍跳 framing; "
        f"got: {villager_text!r}"
    )

    # Hybrid with wolf master counter-claiming real seer MUST get
    # 悍跳-specific advice (same as werewolf would).
    hybrid_inp = SkillInput(
        role="hybrid", phase="speech", day=1,
        game_state=gs, world_state=ws, belief_state=None,
        contradiction_alerts=[], player_id="p03",
        task_type="speech",
    )
    hybrid_out = apply_skill(SkillName.COUNTER_CLAIM, hybrid_inp)
    hybrid_text = hybrid_out.prompt_injectable
    assert any(k in hybrid_text for k in ("悍跳", "假", "时间线", "排坑")), (
        f"NEW-R4-P2-1: hybrid-with-wolf-master counter_claim must use "
        f"悍跳 framing; got: {hybrid_text!r}"
    )

    # The hybrid advice must DIFFER from the villager advice (proves
    # the role branch fired).
    assert hybrid_text != villager_text, (
        f"NEW-R4-P2-1: hybrid and villager should get different "
        f"counter_claim advice; both got: {hybrid_text!r}"
    )

    # A hybrid-with-GOOD-master must NOT get the 悍跳 framing.
    gs_good_master = GameState(
        ruleset_id="test",
        game_id="g",
        phase="speech",
        day_number=1,
        night_number=1,
        players=players,
        hybrid_master_id="p04",  # master is a villager
        hybrid_master_faction="good",
    )
    good_inp = SkillInput(
        role="hybrid", phase="speech", day=1,
        game_state=gs_good_master, world_state=ws, belief_state=None,
        contradiction_alerts=[], player_id="p03",
        task_type="speech",
    )
    good_out = apply_skill(SkillName.COUNTER_CLAIM, good_inp)
    good_text = good_out.prompt_injectable
    assert "悍跳" not in good_text, (
        f"NEW-R4-P2-1: hybrid-with-GOOD-master counter_claim must NOT "
        f"use 悍跳 framing; got: {good_text!r}"
    )


# ---------------------------------------------------------------------------
# NEW-R4-P2-2: dead `action = "..."` locals in handlers.
# ---------------------------------------------------------------------------


def test_no_dead_action_locals_in_handlers() -> None:
    """NEW-R4-P2-2: handlers must not assign to a local named `action`
    that is never read. S-19 removed `recommended_action` from the
    `SkillOutput` schema, so any `action = "..."` in handler bodies
    is dead — the local is assigned but never returned. We scan the
    module with `ast` to find every `ast.Assign` whose target name is
    `action` inside a function (handler) body and assert there are
    none.
    """
    import ast
    from pathlib import Path as _Path
    from werewolf_agent.skills import werewolf_skills as _ws

    src = _Path(_ws.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)

    offenders: list[tuple[str, int]] = []
    # Walk only the top-level function defs (handlers, helpers).
    # We allow `action` locals in tests or in module-level constants.
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Assign):
                continue
            for tgt in sub.targets:
                if (
                    isinstance(tgt, ast.Name)
                    and tgt.id == "action"
                ):
                    offenders.append((node.name, sub.lineno))

    assert not offenders, (
        f"NEW-R4-P2-2: dead `action = ...` assignments in handlers; "
        f"S-19 removed `recommended_action` from the schema so these "
        f"locals are written and never read. Offending sites: "
        f"{offenders!r}"
    )


# ---------------------------------------------------------------------------
# NEW-R4-P2-3: last_words handles empty per-death parts gracefully.
# ---------------------------------------------------------------------------


def test_last_words_handles_empty_input() -> None:
    """NEW-R4-P2-3: when a dead player has NO claims, NO contradictions,
    and NO speech, the per-death entry should NOT be just an empty
    `p05的遗言：` label (a useless artifact that wastes prompt
    budget). Either skip the entry entirely or fall back to a
    placeholder like `无具体遗言内容可分析`.

    Pre-fix: `parts = [f"{dead_player}的遗言："]` followed by
    conditional appends. With no claims/alerts/speech, the entry is
    just the bare label. The LLM gets `p05的遗言：` followed by a
    newline and no body — it has to invent content.

    Post-fix: when no content is available for a dead player, either
    skip the entry or substitute `无具体遗言内容可分析`. The
    rendered prompt must NOT contain a bare `p05的遗言：` label with
    no following content.
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
    # p05 just died and has NO claims, NO speeches, NO alerts.
    players["p05"] = PlayerState(id="p05", role="villager", alive=False)
    gs = GameState(
        ruleset_id="test",
        game_id="g",
        phase="speech",
        day_number=2,
        night_number=2,
        players=players,
    )
    ws = StructuredWorldState()
    ws.append(StructuredFact(
        fact_type="player_died", target_player="p05",
        value="wolf_kill", day=2,
    ))
    # Sanity: p05 has no claims/speeches. No contradiction_alerts either.

    inp = SkillInput(
        role="villager", phase="day", day=2,
        game_state=gs, world_state=ws, belief_state=None,
        contradiction_alerts=[], player_id="p01",
        task_type="speech",
    )
    out = apply_skill(SkillName.LAST_WORDS_ANALYSIS, inp)
    text = out.prompt_injectable
    # The bare `p05的遗言：` label with no body must NOT appear.
    # Acceptable forms: skipped entry, or `p05的遗言： 无具体遗言内容可分析`
    # (with placeholder body) — but NOT a bare label.
    assert "p05的遗言：" not in text or "无具体遗言内容可分析" in text, (
        f"NEW-R4-P2-3: last_words must not emit a bare `p05的遗言：` "
        f"label with no content; got: {text!r}"
    )
    # A p05 that died AND has no data should still be mentioned
    # somehow (the dead player matters) — but as a placeholder.
    if "p05" in text:
        assert "无具体" in text or "无内容" in text, (
            f"NEW-R4-P2-3: empty-death entry must include a placeholder; "
            f"got: {text!r}"
        )


# ---------------------------------------------------------------------------
# NEW-R4-P2-4: wolf_pit shows total count when truncated.
# ---------------------------------------------------------------------------


def test_wolf_pit_shows_total_count() -> None:
    """NEW-R4-P2-4: wolf_pit slices `unique_suspects[:5]` for the
    prompt body, but the header `嫌疑区({len(unique_suspects)}人)`
    shows the *untruncated* total. The LLM sees `嫌疑区(8人)` followed
    by 5 entries and a separate `...（已省略）` from the
    `_cap_prompt_injectable` 800-char ceiling — two different
    truncation signals, no clarity on what was cut.

    Post-fix: when the visible lines are a slice, the count in the
    header must read `(shown/total)`, e.g. `嫌疑区(5/8人)`. The
    `(shown/total)` form tells the LLM exactly how many entries
    were dropped and how many remain.
    """
    from werewolf_agent.cognition.belief import BeliefUpdater
    from werewolf_agent.cognition.world_state import (
        StructuredFact, StructuredWorldState, build_world_state,
    )
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.skills.schemas import SkillInput, SkillName
    from werewolf_agent.skills.werewolf_skills import apply_skill

    # 12 players, 8 with seer_check_claim "wolf" — should land in
    # unique_suspects after the wolf_pit handler dedupes.
    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 13)
    }
    gs = GameState(
        ruleset_id="test",
        game_id="g",
        phase="speech",
        day_number=2,
        night_number=2,
        players=players,
    )
    ws = build_world_state(gs)
    # Add 8 seer_check_claim "wolf" facts.
    for i in range(1, 9):
        ws.append(StructuredFact(
            fact_type="seer_check_claim",
            source_player="p07", target_player=f"p{i:02d}",
            value="wolf", day=1,
        ))

    bs = BeliefUpdater().initialize(list(gs.players.keys()), "p01")
    bs = BeliefUpdater().update(bs, ws.facts, gs.day_number)

    inp = SkillInput(
        role="werewolf", phase="speech", day=2,
        game_state=gs, world_state=ws, belief_state=bs,
        contradiction_alerts=[], player_id="p01",
        task_type="speech",
    )
    out = apply_skill(SkillName.WOLF_PIT_ANALYSIS, inp)
    text = out.prompt_injectable
    # The header must show the (shown/total) form when the list was
    # truncated. With 8 suspects and a [:5] cap, the header should
    # read `嫌疑区(5/8人)` (or similar slash form).
    assert "5/8" in text, (
        f"NEW-R4-P2-4: wolf_pit must show (5/8) shown/total when "
        f"truncating 8 suspects to 5; got: {text!r}"
    )
    # And the bare `嫌疑区(8人)` form (only the total, no shown count)
    # must NOT appear.
    assert "嫌疑区(8人)" not in text, (
        f"NEW-R4-P2-4: wolf_pit must NOT show bare total `嫌疑区(8人)` "
        f"when truncated; got: {text!r}"
    )


# ---------------------------------------------------------------------------
# NEW-R4-P2-5: hide_identity static fallback branches on role.
# ---------------------------------------------------------------------------


def test_hide_identity_role_conditional() -> None:
    """NEW-R4-P2-5: the static-fallback path (no game_state) of
    `hide_identity_handler` returned a one-size-fits-all string
    regardless of role. But seer/witch/wolf each have different
    "what to hide" priorities:
    - seer hides 查验信息 (check info) and 警徽流
    - witch hides 药剂 (antidote/poison availability) and 救人时机
    - wolf hides 夜间会议 (night meeting) and teammate coordination
    - villager hides almost nothing (no night info to leak)

    Post-fix: branch on `inp.role` so the fallback advice is
    role-tailored. The seer fallback must mention 查验 / 警徽;
    the witch fallback must mention 药剂 / 解药 / 毒药; the wolf
    fallback must mention 夜杀 / 队友 / 夜间信息. A villager
    fallback stays generic.
    """
    from werewolf_agent.skills.schemas import SkillInput, SkillName
    from werewolf_agent.skills.werewolf_skills import apply_skill

    seer_inp = SkillInput(
        role="seer", phase="day", day=1,
        game_state=None, world_state=None, belief_state=None,
        contradiction_alerts=[], player_id="p01",
        task_type="speech",
    )
    seer_text = apply_skill(SkillName.HIDE_IDENTITY, seer_inp).prompt_injectable
    assert any(k in seer_text for k in ("查验", "警徽", "金水")), (
        f"NEW-R4-P2-5: seer hide_identity fallback must mention "
        f"查验 / 警徽 / 金水; got: {seer_text!r}"
    )

    witch_inp = SkillInput(
        role="witch", phase="day", day=1,
        game_state=None, world_state=None, belief_state=None,
        contradiction_alerts=[], player_id="p01",
        task_type="speech",
    )
    witch_text = apply_skill(SkillName.HIDE_IDENTITY, witch_inp).prompt_injectable
    assert any(k in witch_text for k in ("药剂", "解药", "毒药", "救人", "女巫")), (
        f"NEW-R4-P2-5: witch hide_identity fallback must mention "
        f"药剂 / 解药 / 毒药 / 救人 / 女巫; got: {witch_text!r}"
    )

    wolf_inp = SkillInput(
        role="werewolf", phase="day", day=1,
        game_state=None, world_state=None, belief_state=None,
        contradiction_alerts=[], player_id="p01",
        task_type="speech",
    )
    wolf_text = apply_skill(SkillName.HIDE_IDENTITY, wolf_inp).prompt_injectable
    assert any(k in wolf_text for k in ("夜杀", "夜刀", "狼队", "队友", "夜间")), (
        f"NEW-R4-P2-5: wolf hide_identity fallback must mention "
        f"夜杀 / 狼队 / 队友 / 夜间; got: {wolf_text!r}"
    )

    # Sanity: the three role-specific fallbacks are NOT all the same.
    assert seer_text != witch_text, (
        f"NEW-R4-P2-5: seer and witch hide_identity fallbacks must "
        f"differ; both got: {seer_text!r}"
    )
    assert seer_text != wolf_text, (
        f"NEW-R4-P2-5: seer and wolf hide_identity fallbacks must "
        f"differ; both got: {seer_text!r}"
    )
    assert witch_text != wolf_text, (
        f"NEW-R4-P2-5: witch and wolf hide_identity fallbacks must "
        f"differ; both got: {witch_text!r}"
    )


# ---------------------------------------------------------------------------
# NEW-R4-P2-6: bold_claim static fallback branches by day.
# ---------------------------------------------------------------------------


def test_bold_claim_advice_varies_by_day() -> None:
    """NEW-R4-P2-6: the static-fallback path of `bold_claim_handler`
    binarizes day into `day <= 1` (conf=0.6) vs `day > 1` (conf=0.3).
    Day 2 (still early) gets the same "晚期悍跳风险极高" advice as
    day 3+ (genuinely late). The LLM has no way to distinguish
    "this is day 2, the seer-claim window is still open" from
    "this is day 3, the window is closed".

    Post-fix: branch on `inp.day` with a 3-way split —
    day=1 (window open, full 悍跳), day=2 (transitional, conditionally
    recommended), day>=3 (window closed, deprioritize).
    """
    from werewolf_agent.skills.schemas import SkillInput, SkillName
    from werewolf_agent.skills.werewolf_skills import apply_skill

    def _fallback(day: int) -> str:
        inp = SkillInput(
            role="werewolf", phase="speech", day=day,
            game_state=None, world_state=None, belief_state=None,
            contradiction_alerts=[], player_id="p01",
            task_type="speech",
        )
        return apply_skill(SkillName.BOLD_CLAIM, inp).prompt_injectable

    day1 = _fallback(1)
    day2 = _fallback(2)
    day3 = _fallback(3)

    # day 1 advice must NOT be the day-3 "don't 悍跳" advice.
    assert day1 != day3, (
        f"NEW-R4-P2-6: bold_claim fallback must distinguish day 1 "
        f"from day 3; both got: {day1!r}"
    )
    # day 2 (transitional) must NOT share day 3's late-window advice.
    # If day 2 == day 3, the LLM has no signal for the day 2
    # "window still open" case.
    assert day2 != day3, (
        f"NEW-R4-P2-6: bold_claim fallback must distinguish day 2 "
        f"from day 3+; both got: {day2!r}"
    )
    # day 1 should be most encouraging ("尽早", "窗口", or high
    # confidence marker). day 3 should be least encouraging.
    day1_encouraging = any(k in day1 for k in ("尽早", "立即", "建议跳", "窗口", "最佳"))
    day3_discouraging = any(k in day3 for k in ("不建议", "风险极高", "放弃", "已过"))
    assert day1_encouraging, (
        f"NEW-R4-P2-6: day 1 bold_claim must be encouraging; got: {day1!r}"
    )
    assert day3_discouraging, (
        f"NEW-R4-P2-6: day 3 bold_claim must be discouraging; got: {day3!r}"
    )


# ---------------------------------------------------------------------------
# NEW-R4-P2-7: wolf_pit / find_power static fallback says "wait".
# ---------------------------------------------------------------------------


def test_wolf_pit_static_fallback_waits_for_signal() -> None:
    """NEW-R4-P2-7: when no specific signal is available (gs is
    None, or ws is empty), the wolf_pit and find_power static
    fallbacks used abstract "系统性分析..." advice that gives the
    LLM nothing actionable. Replace with explicit "wait for
    critical speech" — the skill value is the dynamic analysis, the
    fallback is just a placeholder.
    """
    from werewolf_agent.skills.schemas import SkillInput, SkillName
    from werewolf_agent.skills.werewolf_skills import apply_skill

    wolf_pit_inp = SkillInput(
        role="werewolf", phase="speech", day=1,
        game_state=None, world_state=None, belief_state=None,
        contradiction_alerts=[], player_id="p01",
        task_type="speech",
    )
    wolf_pit_text = apply_skill(
        SkillName.WOLF_PIT_ANALYSIS, wolf_pit_inp,
    ).prompt_injectable
    # The abstract "系统性分析" advice must NOT appear — it gives
    # the LLM no concrete next step.
    assert "系统性分析" not in wolf_pit_text, (
        f"NEW-R4-P2-7: wolf_pit static fallback must not use abstract "
        f"'系统性分析' advice; got: {wolf_pit_text!r}"
    )
    # The fallback should suggest waiting for a specific signal.
    assert any(k in wolf_pit_text for k in (
        "等待", "关键", "出现", "再下判断", "观察",
    )), (
        f"NEW-R4-P2-7: wolf_pit static fallback should say to wait "
        f"for critical signal; got: {wolf_pit_text!r}"
    )

    # find_power fallback has the same problem and same fix.
    find_power_inp = SkillInput(
        role="villager", phase="speech", day=1,
        game_state=None, world_state=None, belief_state=None,
        contradiction_alerts=[], player_id="p01",
        task_type="speech",
    )
    find_power_text = apply_skill(
        SkillName.FIND_POWER, find_power_inp,
    ).prompt_injectable
    assert "系统性分析" not in find_power_text, (
        f"NEW-R4-P2-7: find_power static fallback must not use abstract "
        f"'系统性分析' advice; got: {find_power_text!r}"
    )


# ---------------------------------------------------------------------------
# NEW-R4-P2-9: protect_power empty at_risk lists concrete candidates.
# ---------------------------------------------------------------------------


def test_protect_power_empty_at_risk_specific() -> None:
    """NEW-R4-P2-9: when no power role is currently under pressure
    (`at_risk` empty), the previous fallback said only "继续观察"
    — circular, gave the LLM no concrete next step. Post-fix: list
    the identified power candidates (with role + confidence) and
    suggest a concrete protective action.
    """
    from werewolf_agent.cognition.belief import BeliefUpdater
    from werewolf_agent.cognition.world_state import build_world_state
    from werewolf_agent.core.models import GameState, PlayerState
    from werewolf_agent.skills.schemas import SkillInput, SkillName
    from werewolf_agent.skills.werewolf_skills import apply_skill

    # 12 players. p03 is a suspected seer (high seer prob) but NOT
    # under vote pressure — so at_risk stays empty.
    players = {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager", alive=True)
        for i in range(1, 13)
    }
    players["p03"] = PlayerState(id="p03", role="seer", alive=True)
    gs = GameState(
        ruleset_id="test",
        game_id="g",
        phase="speech",
        day_number=2,
        night_number=2,
        players=players,
    )
    ws = build_world_state(gs)
    bs = BeliefUpdater().initialize(list(gs.players.keys()), "p01")
    bs = BeliefUpdater().update(bs, ws.facts, gs.day_number)
    # Force p03 to look like a seer (top role seer, high prob).
    bs.beliefs["p03"].role_probabilities = {
        "seer": 0.9, "werewolf": 0.05, "villager": 0.05,
    }
    # Make sure p03 has no vote/pressure on them.
    assert not _vote_targets_for_player(ws, "p03"), (
        "Test setup: p03 must have no votes for at_risk to be empty"
    )

    inp = SkillInput(
        role="villager", phase="speech", day=2,
        game_state=gs, world_state=ws, belief_state=bs,
        contradiction_alerts=[], player_id="p01",
        task_type="speech",
    )
    out = apply_skill(SkillName.PROTECT_POWER, inp)
    text = out.prompt_injectable
    # The circular "继续观察" advice (no concrete candidate) must
    # NOT be the only thing the LLM gets.
    assert text != (
        "保护强神建议：场上疑似神职暂时安全，无被推票压力。"
        "继续观察，注意保护已识别的疑似神职不被狼队发现。"
    ), (
        f"NEW-R4-P2-9: protect_power empty-at_risk fallback must not "
        f"be the circular '继续观察' advice; got: {text!r}"
    )
    # The concrete p03 candidate must appear so the LLM has a name
    # to work with.
    assert "p03" in text, (
        f"NEW-R4-P2-9: protect_power empty-at_risk fallback must "
        f"list the identified candidate; got: {text!r}"
    )
    # And the role label (seer).
    assert "seer" in text, (
        f"NEW-R4-P2-9: protect_power empty-at_risk fallback must "
        f"include the candidate's likely role; got: {text!r}"
    )


def _vote_targets_for_player(ws, player_id):
    """Local helper — mirror of the production helper for use in
    this test module without depending on the production symbol."""
    return [
        {"source": f.source_player, "day": f.day, "value": f.value}
        for f in ws.facts_of_type("vote")
        if f.target_player == player_id
    ]


# ---------------------------------------------------------------------------
# NEW-R4-P2-10: last_words has no dead `night` field reference.
# ---------------------------------------------------------------------------


def test_last_words_no_dead_night_field() -> None:
    """NEW-R4-P2-10: last_words's `f.day or f.night if hasattr(f,
    "night") else ""` style reference is dead code. Scan the
    module to confirm there's no `f.night` or
    `hasattr(f, "night")` access — the `night` field does not
    exist on `StructuredFact` and the reference is leftover from
    a refactor.
    """
    import ast
    from pathlib import Path as _Path
    from werewolf_agent.skills import werewolf_skills as _ws

    src = _Path(_ws.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)

    offenders: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        # `f.night` access (ast.Attribute).
        if isinstance(node, ast.Attribute) and node.attr == "night":
            offenders.append(("attribute", getattr(node, "lineno", 0)))
        # `hasattr(f, "night")` — match a Call to `hasattr` whose
        # second positional arg is a Constant string "night".
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "hasattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == "night"
        ):
            offenders.append(("hasattr", node.lineno))

    assert not offenders, (
        f"NEW-R4-P2-10: dead `f.night` / `hasattr(f, 'night')` "
        f"references in last_words or anywhere in "
        f"werewolf_skills.py; offenders: {offenders!r}"
    )
