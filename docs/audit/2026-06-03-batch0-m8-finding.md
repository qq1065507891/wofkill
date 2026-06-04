# P0-M8 Finding: BeliefUpdater Algorithm is Private-Info Safe; Risk is at Rendering

**Date:** 2026-06-03
**Audit target:** `werewolf_agent/cognition/belief.py` — `BeliefUpdater._apply_fact` and friends
**Method:** Code review of all `_apply_*` methods + call site in `runtime/context.py:686-695`

## Findings

`BeliefUpdater.update()` (line 82) is fed only `visible_facts` (filtered by `VisibilityPolicy.filter_visible_facts`, line 692). It processes:
- `player_died` (public)
- `self_destruct` (public, becomes `confirmed_wolves`)
- `idiot_revealed` (public)
- `claimed_role` (public claim boosts role probability)
- `vote` (public)
- `speech` (signal: long speech → +0.02 trust)
- `seer_check_claim` (public claim marks target as wolf/good)

**None of these methods consume wolf-team-private events, seer check results, or witch potion events.** The `role_probabilities` distribution for any player is derived from public behavior only.

## Implication

P0-M8 is **not an algorithm bug**; the BeliefUpdater is private-info safe. The remaining risk is the **rendering layer** (`_cognition_matrix_hint` in `context.py:273-301`), which exposes `key_evidence` text and `open_questions` text in the prompt.

## Fix Direction

Consolidate P0-M8 into P0-M9 (cognition_matrix_hint rendering). The fix:
1. Render `key_evidence` and `open_questions` as **ID references** (`salience_items#abc123`), not full text.
2. Trust/faction_lean/top_role_guess remain as summary statistics (already public-derived).
3. Add regression test that a wolf player does NOT see more in their `cognition_matrix_hint` than what they could derive from public facts.

## Test

```python
# tests/memory/test_belief_visibility.py
def test_belief_state_uses_only_public_signals():
    """Wolf player's BeliefUpdater gets same signals as villager's for a player they've never seen in private channels."""
    updater = BeliefUpdater()
    wolf_belief = updater.initialize(["p01", "p02", "p03"], viewer_id="p01")
    villager_belief = updater.initialize(["p01", "p02", "p03"], viewer_id="p02")
    # Both players see same public facts; belief states should match
    public_facts = [
        StructuredFact(fact_type="speech", source_player="p03", value="long speech..."),
        StructuredFact(fact_type="claimed_role", source_player="p03", value="seer"),
    ]
    wolf_belief = updater.update(wolf_belief, public_facts, 1)
    villager_belief = updater.update(villager_belief, public_facts, 1)
    assert wolf_belief.beliefs["p03"].role_probabilities == villager_belief.beliefs["p03"].role_probabilities
```
