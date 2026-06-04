"""P1-K3: low-confidence skill output should not be dropped when it carries
actionable negative-signal advice.

Audit finding: `_inject_skill_output` (in `werewolf_agent/runtime/context.py`)
drops any `SkillOutput` with `confidence < 0.4`. But low-confidence output is
often negative-signal advice ("don't do X", "avoid Y") that is still useful.

Example from the bold_claim handler (werewolf_skills.py):
    if wolf_plan and wolf_plan.get("fake_seer") and wolf_plan["fake_seer"] != inp.player_id:
        return SkillOutput(
            skill_name="bold_claim",
            confidence=0.3,                                    # < 0.4
            reasoning=f"队友 {wolf_plan['fake_seer']} 负责悍跳，你不需要",
            prompt_injectable="...你不需要悍跳...",
        )

Under the old code, that whole output is dropped on the `< 0.4` check, and
the wolf never receives the "you don't need to claim seer" reminder — which
is exactly when it would prevent identity leakage from non-fake_seer wolves
trying to ride the fake_seer.

Fix: do NOT drop on `confidence < 0.4`; instead, sort by confidence
(descending) so higher-confidence advice appears first in the rendered
`skill_tactical_advice` string. The prompt still has a finite budget, so
we sort to keep the best advice visible; the actionable low-confidence
advice remains reachable.
"""

from __future__ import annotations

from werewolf_agent.core.models import GameState, PlayerState


# ---------------------------------------------------------------------------
# Helpers (shared with test_skills.py but duplicated to keep this file
# self-contained for review).
# ---------------------------------------------------------------------------

def _make_skill_gs_with_fake_seer_plan(
    day: int = 1,
) -> tuple[GameState, dict]:
    """Build a GameState where p01 is a wolf but p02 is the team's fake_seer.

    This is the canonical setup that triggers the bold_claim handler's
    negative-signal branch: `confidence=0.3` + "你不需要悍跳" advice.
    """
    players = {
        "p01": PlayerState(id="p01", role="werewolf", alive=True),
        "p02": PlayerState(id="p02", role="werewolf", alive=True),
        "p03": PlayerState(id="p03", role="seer", alive=True),
        "p04": PlayerState(id="p04", role="villager", alive=True),
        "p05": PlayerState(id="p05", role="villager", alive=True),
        "p06": PlayerState(id="p06", role="witch", alive=True),
    }
    gs = GameState(
        ruleset_id="test",
        game_id="test_game",
        phase="speech",
        day_number=day,
        night_number=day,
        players=players,
        events=[],
    )
    wolf_plan = {"fake_seer": "p02", "pusher": None, "hooker": None, "deep_cover": None}
    return gs, wolf_plan


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
# K3.1: low-confidence negative-signal advice is kept in the rendered
# skill_tactical_advice (not silently dropped on the < 0.4 filter).
# ---------------------------------------------------------------------------

class TestLowConfidenceSkillNotDropped:
    """P1-K3: low-confidence skill output with negative-signal advice
    must not be dropped. It should appear in the rendered prompt."""

    def test_low_confidence_skill_not_dropped(self):
        """The bold_claim handler emits confidence=0.3 + '你不需要悍跳' advice
        when a teammate is already assigned as fake_seer. That advice is
        critical for non-fake_seer wolves — it tells them to NOT claim seer
        and avoid identity leakage. Pre-fix, this output is dropped on
        `confidence < 0.4` and the advice never reaches the prompt.
        """
        from werewolf_agent.runtime.context import _inject_skill_output

        gs, wolf_plan = _make_skill_gs_with_fake_seer_plan(day=1)
        ws, bs, alerts = _build_cognition(gs, "p01")
        directive: dict = {}
        result, _ = _inject_skill_output(
            directive, gs, "p01", ws, bs, alerts, "speech",
            wolf_team_plan=wolf_plan,
        )

        # The bold_claim advice from p01 (a non-fake_seer wolf) should be
        # present in skill_tactical_advice. Pre-fix this assertion fails
        # because confidence=0.3 is < 0.4 and the entire output is dropped.
        assert "skill_tactical_advice" in result, (
            "skill_tactical_advice should be populated even when handler "
            "returns low confidence."
        )
        rendered = result["skill_tactical_advice"]
        # S-07: skill_tactical_advice is now a list of dicts.
        # Walk the list to find the bold_claim entry.
        rendered_str = " ".join(
            e["advice"] for e in rendered
            if isinstance(e, dict) and e.get("skill") == "bold_claim"
        )
        assert "你不需要悍跳" in rendered_str or "p02" in rendered_str, (
            f"Negative-signal advice from low-confidence handler was dropped. "
            f"Rendered advice was: {rendered!r}"
        )

    def test_high_confidence_advice_appears_before_low_confidence(self):
        """Sort by confidence (descending): high-confidence advice first."""
        from werewolf_agent.skills.schemas import (
            SkillInput, SkillOutput, SkillName,
        )
        from werewolf_agent.skills.registry import SkillRegistry

        # Build a fake registry state by monkeypatching the handler for one
        # skill. We override bold_claim to produce a HIGH-confidence output,
        # and rely on a separate low-confidence skill (push_vote with no
        # top suspects → conf=0.4) as a contrast.
        # Easier approach: call dispatch_for_role with a constructed input
        # and then verify the rendered ordering on the result.
        reg = SkillRegistry()
        gs, _ = _make_skill_gs_with_fake_seer_plan(day=1)
        ws, bs, alerts = _build_cognition(gs, "p01")

        # Build a SkillInput that triggers multiple skills with different
        # confidences. Push_vote with no belief suspects → conf=0.4.
        # Bold_claim with a fake_seer plan → conf=0.3.
        skill_input = SkillInput(
            role="werewolf",
            phase="speech",
            day=1,
            game_state=gs,
            world_state=ws,
            belief_state=bs,
            contradiction_alerts=alerts,
            player_id="p01",
            extra={"wolf_team_plan": {"fake_seer": "p02", "pusher": None,
                                       "hooker": None, "deep_cover": None}},
        )
        outputs = reg.dispatch_for_role("werewolf", "speech", skill_input)
        # We expect at least one low-confidence and one high-confidence output.
        confidences = [o.confidence for o in outputs]
        assert len(confidences) >= 2, (
            f"Need at least 2 skill outputs to verify sort; got {len(confidences)}"
        )
        # Sanity: not all are the same confidence.
        assert min(confidences) != max(confidences), (
            f"All outputs have the same confidence ({confidences}); "
            "cannot verify ordering."
        )

        # Now verify the actual injection sorts them. We assert by
        # checking that the FIRST chunk of skill_tactical_advice comes
        # from the highest-confidence output, and the last chunk from
        # the lowest-confidence one.
        from werewolf_agent.runtime.context import _inject_skill_output
        directive: dict = {}
        result, _ = _inject_skill_output(
            directive, gs, "p01", ws, bs, alerts, "speech",
            wolf_team_plan={"fake_seer": "p02", "pusher": None,
                            "hooker": None, "deep_cover": None},
        )
        rendered = result.get("skill_tactical_advice", "")
        # We don't pin exact chunk boundaries (handlers may emit multi-line
        # blocks), but we can verify that the lowest-confidence output's
        # prompt_injectable appears AFTER the highest-confidence one in
        # the joined string. The '你不需要悍跳' snippet is the
        # lowest-confidence one (0.3) and appears later.
        max_conf_output = max(outputs, key=lambda o: o.confidence)
        min_conf_output = min(outputs, key=lambda o: o.confidence)
        # S-07: rendered is a list of {skill, advice, confidence} dicts.
        # Verify the max-confidence entry appears before the
        # min-confidence entry in the list.
        if (max_conf_output.prompt_injectable
                and min_conf_output.prompt_injectable
                and max_conf_output.prompt_injectable != min_conf_output.prompt_injectable):
            # Build a joined string from the structured list, preserving
            # order — the prompt builder does the same when it renders.
            rendered_str = "\n".join(
                e.get("advice", "") for e in rendered
                if isinstance(e, dict)
            )
            idx_max = rendered_str.find(max_conf_output.prompt_injectable[:30])
            idx_min = rendered_str.find(min_conf_output.prompt_injectable[:30])
            assert idx_max >= 0, "highest-confidence chunk not in rendered advice"
            assert idx_min >= 0, "lowest-confidence chunk not in rendered advice"
            assert idx_max < idx_min, (
                f"Expected highest-confidence chunk (idx={idx_max}) to "
                f"appear before lowest-confidence chunk (idx={idx_min}). "
                f"Rendered: {rendered!r}"
            )
