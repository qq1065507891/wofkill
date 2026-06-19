# Deadcode Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove confirmed dead code (no production callers, replaced by newer paths) from cognition/, runtime/context.py, and agents/prompt_builder.py, along with the tests that lock the dead behaviour. Reduce surface area and confusion for future work.

**Architecture:** Five delete-only tasks + one regression/docs task. Each task: delete the dead module/function + its dedicated tests + any now-dangling imports, then verify `compileall` (catches dangling imports) + the affected test files pass. No new code. No behaviour change (the deleted code was never called in production).

**Tech Stack:** Python 3.11, pytest, conda env `wofkill`, PowerShell. Tests run with `-o addopts=""` (no `--basetemp`, to avoid the local pytest-xdist `.pytest_tmp` permission issue).

---

## File Structure

- **Delete** `werewolf_agent/cognition/attention.py`, `werewolf_agent/cognition/salience.py`, `werewolf_agent/cognition/strategy.py` — entire modules, no production callers (replaced by `build_agent_context`).
- **Modify** `tests/cognition/test_cognition.py` — remove the 3 imports + 3 test classes (TestAttentionFilter/TestSalienceEngine/TestStrategySelector).
- **Modify** `werewolf_agent/runtime/context.py` — delete `_REFLECTION_HEADER_CATEGORIES` / `_categorize_reflection_text` / `_compute_error_pattern` (replaced by `live_error_pattern` in reflection V2).
- **Modify** `tests/runtime/test_strategy_directives.py` — remove the 5 tests covering the deleted context helpers.
- **Modify** `werewolf_agent/agents/prompt_builder.py` — delete `_GROUNDING_SECTIONS` (unused frozenset) + `_build_skill_analysis_hints` (returns "").
- **Modify** `PROGRESS.md`.

**Out of scope (preserved as unfinished infrastructure, NOT dead-by-replacement):**
- `memory/reflection_effectiveness.py::evaluate_reflection_effectiveness` — spec-planned evaluation capability, no pipeline caller yet (PROGRESS tracks wiring it); deleting loses future infrastructure.
- `cognition/claim_credibility.py` + `cognition/public_evidence.py` `snapshot`/`from_snapshot` — spec-designed cross-process resume persistence, no production wiring yet; deleting loses the persistence mechanism.
- `prompt_builder._LOW_VALUE_SECTIONS` / `_uses_choice_pipeline` / `_uses_speech_intent_pipeline` — have test assertions locking behaviour; removing needs test rewrite, deferred to a separate cleanup pass.

---

## Task 1: Delete cognition/attention.py + TestAttentionFilter

**Why:** `AttentionFilter` has zero production callers (grep: only `tests/cognition/test_cognition.py` imports it, plus PROGRESS/docs references). It was part of the old CognitivePipeline (`cognition/pipeline.py` already deleted in cognition-deadcode-cleanup), replaced by `build_agent_context`. PROGRESS line 72 explicitly deferred its evaluation to "later" — this is that later.

**Files:**
- Delete: `werewolf_agent/cognition/attention.py`
- Modify: `tests/cognition/test_cognition.py` (remove import line 17 + TestAttentionFilter class lines 472-498)

- [ ] **Step 1: Read the current state**

Read `werewolf_agent/cognition/attention.py` (confirm it's self-contained — only imports from `werewolf_agent.cognition.world_state`/`visibility`, nothing imports IT in production). Read `tests/cognition/test_cognition.py` lines 1-25 (imports) and 472-498 (TestAttentionFilter + its `# TestAttentionFilter` comment at 472).

- [ ] **Step 2: Delete the module**

```bash
git rm werewolf_agent/cognition/attention.py
```

- [ ] **Step 3: Remove the test import + test class**

In `tests/cognition/test_cognition.py`:
- Delete line 17: `from werewolf_agent.cognition.attention import AttentionFilter`
- Delete lines 472-498 (the `# TestAttentionFilter` comment at 472 through the end of `class TestAttentionFilter` just before `# TestSalienceEngine` at line 499). Read the exact end line first — the class ends right before the `# TestSalienceEngine` comment.

- [ ] **Step 4: Verify cognition tests + compileall**

Run:
```
python -m compileall -q werewolf_agent
python -m pytest tests/cognition -q -o addopts=""
```
Expected: compileall clean (no module imports `attention`); cognition tests PASS (TestStructuredWorldState/TestVisibilityPolicy/TestBeliefUpdater/TestContradictionEngine/TestStrategySelector + the seer-claim tests all remain green). If compileall flags a dangling import of `attention`, grep for it and remove that import too.

- [ ] **Step 5: Commit**

```bash
git add tests/cognition/test_cognition.py
git commit -m "refactor: remove dead cognition/attention.py and its tests"
```

---

## Task 2: Delete cognition/salience.py + TestSalienceEngine

**Why:** `SalienceEngine` has zero production callers (grep: only `tests/cognition/test_cognition.py` + design-doc references). Same replacement story as attention — old CognitivePipeline component, superseded by `build_agent_context`.

**Files:**
- Delete: `werewolf_agent/cognition/salience.py`
- Modify: `tests/cognition/test_cognition.py` (remove import line 20 + TestSalienceEngine class)

- [ ] **Step 1: Read current state**

Read `werewolf_agent/cognition/salience.py` (confirm self-contained). Read `tests/cognition/test_cognition.py` line 20 (import) and the TestSalienceEngine block (`# TestSalienceEngine` comment + `class TestSalienceEngine` through just before `# TestBeliefUpdater`).

- [ ] **Step 2: Delete the module**

```bash
git rm werewolf_agent/cognition/salience.py
```

- [ ] **Step 3: Remove import + test class**

In `tests/cognition/test_cognition.py`:
- Delete line 20: `from werewolf_agent.cognition.salience import SalienceEngine`
- Delete the `# TestSalienceEngine` comment + `class TestSalienceEngine` block (through just before `# TestBeliefUpdater`). Note: after Task 1, line numbers shifted — re-locate by the `# TestSalienceEngine` / `# TestBeliefUpdater` comments, not by absolute line number.

- [ ] **Step 4: Verify**

```
python -m compileall -q werewolf_agent
python -m pytest tests/cognition -q -o addopts=""
```
Expected: clean + PASS. Check `AgentContext.salience_items` isn't populated by anything that imports SalienceEngine (it's a separate deferred field — Task 5 of the original audit noted it's never filled; SalienceEngine itself isn't the populator).

- [ ] **Step 5: Commit**

```bash
git add tests/cognition/test_cognition.py
git commit -m "refactor: remove dead cognition/salience.py and its tests"
```

---

## Task 3: Delete cognition/strategy.py + TestStrategySelector

**Why:** `StrategySelector` / `STRATEGIES` have zero production callers (grep: only `tests/cognition/test_cognition.py`). The skill system (`skills/`) replaced the old strategy packages.

**Files:**
- Delete: `werewolf_agent/cognition/strategy.py`
- Modify: `tests/cognition/test_cognition.py` (remove import line 21 + TestStrategySelector class)

- [ ] **Step 1: Read current state**

Read `werewolf_agent/cognition/strategy.py` (confirm self-contained, no production importer). Read `tests/cognition/test_cognition.py` line 21 (import) + the TestStrategySelector block (`# TestStrategySelector` through just before the next class `TestSeerClaimContractExtraction`).

- [ ] **Step 2: Delete the module**

```bash
git rm werewolf_agent/cognition/strategy.py
```

- [ ] **Step 3: Remove import + test class**

In `tests/cognition/test_cognition.py`:
- Delete line 21: `from werewolf_agent.cognition.strategy import StrategySelector, STRATEGIES`
- Delete the `# TestStrategySelector` comment + `class TestStrategySelector` block (through just before `class TestSeerClaimContractExtraction`). Re-locate by comments (line numbers shifted after Tasks 1-2).

- [ ] **Step 4: Verify**

```
python -m compileall -q werewolf_agent
python -m pytest tests/cognition -q -o addopts=""
```
Expected: clean + PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/cognition/test_cognition.py
git commit -m "refactor: remove dead cognition/strategy.py and its tests"
```

---

## Task 4: Delete context.py dead reflection-header helpers

**Why:** `_REFLECTION_HEADER_CATEGORIES` / `_categorize_reflection_text` / `_compute_error_pattern` (runtime/context.py ~589-680) have zero production callers. They were the V1 error-pattern aggregation; reflection-synthesis-upgrade replaced the live injection path with `live_error_pattern` (from `ReflectionEntryV2.mistake_patterns` via `reflection.py`). Only `tests/runtime/test_strategy_directives.py` still calls them (5 tests).

**Files:**
- Modify: `werewolf_agent/runtime/context.py` (delete the 3 symbols)
- Modify: `tests/runtime/test_strategy_directives.py` (delete the 5 tests)

- [ ] **Step 1: Read current state**

Read `werewolf_agent/runtime/context.py` lines 585-685 to capture the exact span of `_REFLECTION_HEADER_CATEGORIES` + `_categorize_reflection_text` + `_compute_error_pattern` (all three are contiguous; the function ends where the next unrelated definition begins — read past `_compute_error_pattern`'s `return` to find the boundary). Read `tests/runtime/test_strategy_directives.py` around the 5 tests (search `_categorize_reflection_text` / `_compute_error_pattern` — they're in a `TestReflectionCrossGameLearning` class or similar; capture the class boundary).

Also grep `Counter` usage in context.py — `_compute_error_pattern` uses `Counter`; if nothing else in the file uses it, remove the `from collections import Counter` import too.

- [ ] **Step 2: Delete the helpers**

In `werewolf_agent/runtime/context.py`, delete the contiguous block: `_REFLECTION_HEADER_CATEGORIES` dict + `_categorize_reflection_text` function + `_compute_error_pattern` function (and the `# these headers...` comment above the dict, lines ~587-597). Remove the `Counter` import if now unused.

- [ ] **Step 3: Delete the tests**

In `tests/runtime/test_strategy_directives.py`, delete the 5 tests: `test_categorize_reflection_text_parses_section_headers`, `test_categorize_reflection_text_dedupes_repeats`, `test_categorize_reflection_text_empty_or_no_header`, `test_compute_error_pattern_aggregates_top_mistakes`, `test_compute_error_pattern_empty_reflections`. If they're the only members of a test class, delete the class too.

- [ ] **Step 4: Verify**

```
python -m compileall -q werewolf_agent
python -m pytest tests/runtime -q -o addopts=""
```
Expected: clean + PASS. compileall catches any dangling `_compute_error_pattern`/`_categorize_reflection_text` reference.

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/runtime/context.py tests/runtime/test_strategy_directives.py
git commit -m "refactor: remove dead reflection-header helpers from context.py"
```

---

## Task 5: Delete prompt_builder dead helpers (_GROUNDING_SECTIONS + _build_skill_analysis_hints)

**Why:** `_GROUNDING_SECTIONS` (prompt_builder.py ~423) is a frozenset with zero references (no production use, no test). `_build_skill_analysis_hints` (~1410) is a method that returns `""` and has no caller (the comment at ~487 documents it's dead). Both are inert weight.

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py`

- [ ] **Step 1: Read current state**

Read `werewolf_agent/agents/prompt_builder.py` around line 423 (`_GROUNDING_SECTIONS` frozenset — capture its exact span + the surrounding class context) and around line 1410 (`_build_skill_analysis_hints` method — capture its full body). Grep `_GROUNDING_SECTIONS` and `_build_skill_analysis_hints` across the repo to confirm zero references outside their definitions (the earlier audit grep showed `_LOW_VALUE_SECTIONS` has test refs but `_GROUNDING_SECTIONS` does not).

- [ ] **Step 2: Delete _GROUNDING_SECTIONS**

Delete the `_GROUNDING_SECTIONS: frozenset[str] = frozenset(...)` block (and any comment directly above it that documents only this constant).

- [ ] **Step 3: Delete _build_skill_analysis_hints**

Delete the `def _build_skill_analysis_hints(self) -> str:` method (full body). If a comment block above it (e.g. the `# The opaque-dict dual render is gone...` comment at ~487) refers only to this method, delete that too.

- [ ] **Step 4: Verify**

```
python -m compileall -q werewolf_agent
python -m pytest tests/agents -q -o addopts=""
```
Expected: clean + PASS. (If `_LOW_VALUE_SECTIONS` is adjacent and shares a comment, do NOT delete it — it has test assertions; leave it.)

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/agents/prompt_builder.py
git commit -m "refactor: remove dead _GROUNDING_SECTIONS and _build_skill_analysis_hints"
```

---

## Task 6: Whole-suite regression + PROGRESS

**Files:**
- Modify: `PROGRESS.md`

- [ ] **Step 1: Run cognition + runtime + agents + memory suites**

Run: `python -m pytest tests/cognition tests/runtime tests/agents tests/memory -q -o addopts="" 2>&1 | tail -5`
Expected: all PASS.

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest -q -o addopts="" 2>&1 | tail -5`
Expected: all PASS, FEWER tests than before (the deleted tests are gone). No regressions in the remaining suite.

- [ ] **Step 3: compile-check**

Run: `python -m compileall -q werewolf_agent`
Expected: no output.

- [ ] **Step 4: Update PROGRESS.md**

Add a new section at the top of `PROGRESS.md` (above `attribution-engine`). Update `Current Status` (phase → `deadcode-cleanup`, last updated → `2026-06-19`, active task summarizing the deletions). Record: the 5 deletions (3 cognition modules + context.py helpers + prompt_builder helpers), files deleted/modified, verification commands, and the deliberately-preserved items (evaluate_reflection_effectiveness, snapshot/from_snapshot, _LOW_VALUE_SECTIONS/_uses_*_pipeline) with their follow-up rationale.

- [ ] **Step 5: Commit**

```bash
git add PROGRESS.md
git commit -m "docs: log deadcode cleanup in PROGRESS"
```

---

## Open Risks / Out of Scope

- **Deliberately preserved (unfinished infrastructure, NOT dead-by-replacement):**
  - `memory/reflection_effectiveness.py::evaluate_reflection_effectiveness` + `ReflectionEffectivenessReport` — spec-planned evaluation capability awaiting a pipeline caller. Tracked for wiring, NOT deletion.
  - `cognition/claim_credibility.py` + `cognition/public_evidence.py` `snapshot`/`from_snapshot` — spec-designed cross-process resume persistence awaiting production wiring. Tracked, NOT deleted.
  - `prompt_builder._LOW_VALUE_SECTIONS` / `_uses_choice_pipeline` / `_uses_speech_intent_pipeline` — have test assertions locking behaviour; removing needs coordinated test rewrite, deferred to a separate pass.
- **Cognition module internal deps:** attention/salience/strategy only import from `world_state`/`visibility` (lower layers); no other cognition module imports them, so deletion is self-contained. Confirmed by grep (zero production importers).
- **`Counter` import in context.py:** if `_compute_error_pattern` was the only `Counter` user, the `from collections import Counter` becomes dangling — compileall won't catch unused imports, so grep + remove manually in Task 4.
- **Test count drops:** the full suite will have ~13+ fewer tests (3 TestAttentionFilter/TestSalienceEngine/TestStrategySelector cases + 5 context-helper tests + any prompt_builder ones). This is expected (dead tests removed with dead code), not a regression.

## Self-Review

- **Coverage:** the audit's "deadcode-cleanup" bucket maps to Tasks 1-5 (cognition 3 modules + context.py helpers + prompt_builder helpers). The preserved items (evaluate_reflection_effectiveness, snapshots, _LOW_VALUE_SECTIONS/_uses_*) are explicitly out of scope with rationale.
- **Placeholder scan:** no TBD/TODO. Each task says exactly what to delete (file/class/function) + which tests + verify commands. Line numbers are approximate (shift after earlier tasks) — each task re-locates by content/comment markers.
- **Type consistency:** deletions are pure removals; no new symbols introduced, so no cross-task type dependencies.
- **Safety:** every deletion is of code with verified zero production callers (grep-confirmed). compileall + full-suite green after each task proves no dangling reference.
