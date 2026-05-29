# RuleEngine Decomposition Plan

> **For agentic workers:** Use Task tool to track progress.

**Goal:** Split rule_engine.py (848 lines) into 3 focused modules. `RuleEngine` stays as a backward-compatible facade.

**Architecture:** Extract `EventReducer` (event-sourced state transitions) and `SheriffRules` (badge/election logic) from the monolithic class. `RuleEngine` delegates to them internally, preserving `engine.method()` API for all callers.

**Tech Stack:** Python 3.12, pytest, RuleEngine facade pattern

---

### Task 1: Extract Sheriff/Badge Rules

**Files:**
- Create: `werewolf_agent/engine/sheriff.py`
- Modify: `werewolf_agent/engine/rule_engine.py`

Extract these 6 methods:
- `badge_options_after_sheriff_death` (L440-456)
- `resolve_badge_decision` (L458-465)
- `sheriff_register` (L467-471)
- `sheriff_withdraw` (L473-481)
- `resolve_sheriff_vote` (L483-498)
- `speech_order_policy` (L500-506)

Into a new `SheriffRules` class that takes `ruleset: Ruleset` in `__init__`. `RuleEngine` creates a `self._sheriff = SheriffRules(ruleset)` and delegates.

Run: `python -m pytest tests/rules/ tests/runtime/ -q --tb=short -p no:cacheprovider`

---

### Task 2: Extract Event Reducer

**Files:**
- Create: `werewolf_agent/engine/event_reducer.py`
- Modify: `werewolf_agent/engine/rule_engine.py`

Extract:
- `reduce_event` (L664-826) — 12 case switch
- `reduce_events` (L828-831)
- `_apply_idiot_reveal` (L90-104) — used only by reduce_event and resolve_exile
- `_faction_for_player` (L833-835)
- `_hybrid_result` (L837-840)
- `_validate_alive_target` (L842-847)

Into a new `EventReducer` class. `resolve_exile` (L207-230) calls `self.apply_death()` and `self._apply_idiot_reveal()` — these need to be passed from `RuleEngine` to `EventReducer` or `resolve_exile` stays in `RuleEngine` with the private helpers moving.

**Design decision**: `resolve_exile` stays in `RuleEngine` (it calls `apply_death` which is the canonical state mutator). Only `reduce_event/reduce_events` and helpers move.

Run: `python -m pytest tests/rules/ tests/runtime/ -q --tb=short -p no:cacheprovider`

---

### Task 3: Final Verification

**Files:**
- Modify: `werewolf_agent/engine/rule_engine.py` (remove moved code, add delegation)
- All callers should be unchanged (facade pattern)

Run full test suite, verify `rule_engine.py` is ~400-450 lines.

Run: `python -m pytest tests/ -q --tb=short -p no:cacheprovider`
