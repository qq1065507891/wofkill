# Prompt Sanitizer Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the in-game cognition sections (`possible_worlds` / `simulation_predictions`) from scrubbing current-game player IDs into "历史玩家", fix the hardcoded `p05` in the speech example, cap/clean `seer_credibility.evidence`, and resolve the persona never-drop vs design-§3 conflict by recording the decision in the design doc (no behaviour change).

**Architecture:** Four focused changes in `werewolf_agent/agents/prompt_builder.py` + one design-doc note. The root cause is a sanitizer mis-match: `_clean_prompt_text` / `_clean_list_items` run `_PLAYER_ID_RE.sub("历史玩家")` — correct for *cross-game* sections (RAG/reflection/profile) but wrong for *current-game* cognition sections, whose `why`/`watch_for`/`rationale`/`event` legitimately reference this game's `p03` etc. The current-game sanitizer `_clean_current_game_token` (strips whitespace/newlines, truncates, NO id substitution) is already used for `key_assignments`/`affected_players` and is the right tool. This plan switches the mis-used call sites to current-game cleaning, fixes the speech example's hardcoded `p05`, adds a cap+clean to `seer_credibility.evidence`, and documents the persona never-drop decision (keeping the existing behaviour + test, updating design §3 to match).

**Tech Stack:** Python 3.11, pytest, conda env `wofkill`, PowerShell. Tests run with `-o addopts=""` (no `--basetemp`, to avoid the local pytest-xdist `.pytest_tmp` reuse permission issue — see `memory/project-pytest-addopts-bypass.md`).

---

## File Structure

- **Modify** `werewolf_agent/agents/prompt_builder.py` — `_build_possible_worlds`, `_build_simulation_predictions`, `_build_seer_credibility`, `_format_examples` (speech example).
- **Modify** `docs/design/werewolf-agent-v1-design.md` §3 (prompt-balance) — record the persona never-drop decision.
- **Test** `tests/agents/test_prompt_builder.py` (extend existing tests + new ones).

**Design boundaries respected:** No change to game rules, role abilities, rule engine, or information-visibility. The fix only changes which sanitizer is applied to current-game cognition text — it preserves MORE information (the real current-game IDs), never less, and never exposes anything that wasn't already public-record in-game (these sections are derived from visible facts). Cross-game sections (RAG/reflection/profile/error-pattern) keep using `_clean_prompt_text` (id → 历史玩家) unchanged.

---

## Task 1: possible_worlds keeps current-game IDs

**Why:** `_build_possible_worlds` (`prompt_builder.py:720-770`) renders `label`, `why`, `watch_for` through `_clean_prompt_text` / `_clean_list_items`, which run `_PLAYER_ID_RE.sub("历史玩家")`. These fields come from the current game's cognition (e.g. `why: ["p03 vote pattern conflicts"]`) and the real `p03` is meaningful in-game signal. After scrubbing it becomes "历史玩家 vote pattern conflicts", which damages the reasoning the prompt is supposed to convey. `key_assignments` (line 750-751) already correctly uses `_clean_current_game_token` (no id substitution). The `warning` header is a static template string (no real id), so it can stay on `_clean_prompt_text` — only the id-bearing fields need switching.

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py` (`_build_possible_worlds`)
- Test: `tests/agents/test_prompt_builder.py`

- [ ] **Step 1: Write the failing test**

The existing `test_possible_worlds_render_prompt_safe_hypotheses` (around `tests/agents/test_prompt_builder.py:5047`) asserts `p03=werewolf` (from key_assignments) but does NOT check `why`/`watch_for`. Add a new test right after it that pins the current-game-id behaviour:

```python
    def test_possible_worlds_keeps_current_game_ids_in_why_and_watch_for(self):
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            possible_worlds={
                "type": "possible_worlds",
                "top_worlds": [
                    {
                        "label": "World A",
                        "probability": 0.5,
                        "key_assignments": {"p03": "werewolf"},
                        "why": ["p03 vote pattern conflicts"],
                        "watch_for": ["p03 defense target"],
                    }
                ],
                "warning": "These are hypotheses from visible evidence, not ground truth.",
            },
        )
        prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
        # Current-game IDs in why/watch_for must be PRESERVED (not scrubbed to 历史玩家).
        assert "p03 vote pattern conflicts" in prompt
        assert "p03 defense target" in prompt
        assert "历史玩家" not in prompt.split("These are hypotheses")[1] if "These are hypotheses" in prompt else True
```

(The last assert is defensive — it only checks the worlds block region; the simpler `assert "历史玩家 vote pattern" not in prompt` is also acceptable. Use whichever passes type-check; the core assertions are the first two.)

- [ ] **Step 2: Run to verify it fails**

Run: `conda activate wofkill; python -m pytest "tests/agents/test_prompt_builder.py::TestPromptBuilderSectionRendering::test_possible_worlds_keeps_current_game_ids_in_why_and_watch_for" -q -o addopts=""` 
(If the test is not inside a class named `TestPromptBuilderSectionRendering`, run it by function name: `python -m pytest tests/agents/test_prompt_builder.py -q -o addopts="" -k test_possible_worlds_keeps_current_game_ids_in_why_and_watch_for`)
Expected: FAIL — `why`/`watch_for` currently scrub `p03` → `历史玩家`, so `"p03 vote pattern conflicts"` is absent.

- [ ] **Step 3: Switch the id-bearing fields to current-game cleaning**

In `werewolf_agent/agents/prompt_builder.py` `_build_possible_worlds`:

- `label` (line ~742): change `self._clean_prompt_text(...)` → `_clean_current_game_token(world.get("label") or f"World {idx}", max_chars=40)`.
- `why` (line ~756): replace `self._clean_list_items(world.get("why"), limit=2, max_chars=80)` with a current-game list clean. Since there's no `_clean_current_game_list_items`, build it inline:
```python
            why = [
                _clean_current_game_token(w, max_chars=80)
                for w in (world.get("why") or [])[:2]
                if str(w or "").strip()
            ]
```
- `watch_for` (line ~757): same pattern:
```python
            watch_for = [
                _clean_current_game_token(w, max_chars=80)
                for w in (world.get("watch_for") or [])[:2]
                if str(w or "").strip()
            ]
```
- Leave `warning` (line ~727) on `_clean_prompt_text` (static template string, no real id) and `key_assignments` already on `_clean_current_game_token`.

`_clean_current_game_token` is a module-level function (line 143) already imported/used in this file — confirm it's in scope at the call site (it's used at line 750 already, so no new import needed).

- [ ] **Step 4: Run the possible_worlds tests to verify pass + no regressions**

Run: `python -m pytest tests/agents/test_prompt_builder.py -q -o addopts="" -k possible_worlds`
Expected: PASS (both the existing `test_possible_worlds_render_prompt_safe_hypotheses` and the new test). The existing test still passes because `key_assignments` was already on `_clean_current_game_token` and the `roles`/`supporting_evidence` scrub assertions are unaffected.

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/agents/prompt_builder.py tests/agents/test_prompt_builder.py
git commit -m "fix: possible_worlds preserves current-game player ids in why/watch_for"
```

---

## Task 2: simulation_predictions keeps current-game IDs

**Why:** Same root cause in `_build_simulation_predictions` (`prompt_builder.py:772-825`). `event` (796) and `rationale` (808) go through `_clean_prompt_text` → id scrubbed to 历史玩家. `affected_players` (802) already correctly uses `_clean_current_game_token`. `warning`/`horizon` are static template strings (safe on either sanitizer, but switch for consistency since they're current-game context). `world_ids` are current-game hypothesis labels (e.g. "World A"), so they should also keep real text.

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py` (`_build_simulation_predictions`)
- Test: `tests/agents/test_prompt_builder.py`

- [ ] **Step 1: Write the failing test**

Add after the existing `test_simulation_predictions_render_prompt_safe_cards` (around `tests/agents/test_prompt_builder.py:5080`):

```python
    def test_simulation_keeps_current_game_ids_in_rationale_and_event(self):
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.SPEECH,
            phase="day",
            own_role="villager",
            legal_actions=[ActionType.SPEECH],
            simulation_predictions={
                "type": "simulation",
                "horizon": "next_turn",
                "predictions": [
                    {
                        "event": "p03 exile pressure",
                        "probability": 0.7,
                        "affected_players": ["p03"],
                        "rationale": "p03 vote pressure concentrated",
                        "world_ids": ["World A"],
                    }
                ],
                "warning": "Prediction, not fact.",
            },
        )
        prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
        # Current-game IDs in event/rationale must be PRESERVED.
        assert "p03 exile pressure" in prompt
        assert "p03 vote pressure concentrated" in prompt
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/agents/test_prompt_builder.py -q -o addopts="" -k test_simulation_keeps_current_game_ids`
Expected: FAIL — `event`/`rationale` scrub `p03` → `历史玩家`.

- [ ] **Step 3: Switch fields to current-game cleaning**

In `_build_simulation_predictions`:
- `event` (line ~796): `event = _clean_current_game_token(item.get("event"), max_chars=48)`
- `rationale` (line ~808): `rationale = _clean_current_game_token(item.get("rationale"), max_chars=100)`
- `world_ids` (line ~812): replace `self._clean_list_items(item.get("world_ids"), limit=3, max_chars=32)` with:
```python
            world_ids = [
                _clean_current_game_token(w, max_chars=32)
                for w in (item.get("world_ids") or [])[:3]
                if str(w or "").strip()
            ]
```
- Leave `warning`/`horizon` as-is (static template strings; switching is optional and not required for correctness — keep the diff minimal, do NOT change them).

- [ ] **Step 4: Run simulation tests**

Run: `python -m pytest tests/agents/test_prompt_builder.py -q -o addopts="" -k simulation`
Expected: PASS (existing `test_simulation_predictions_render_prompt_safe_cards` + new test). Existing test still passes: it asserts `next_day_vote_pressure` (no id) in event, `p03` in affected_players (already current-game), `Prediction, not fact.` in warning, and `roles`/`hidden truth` absent — none affected by the event/rationale switch.

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/agents/prompt_builder.py tests/agents/test_prompt_builder.py
git commit -m "fix: simulation predictions preserve current-game ids in event/rationale"
```

---

## Task 3: seer_credibility.evidence capped + cleaned

**Why:** `_build_seer_credibility` (`prompt_builder.py:694-718`) cleans `claimant`/`status`/`checks` via `_clean_current_game_token` but `evidence` (line 711) is joined raw with no length cap and no cleaning: `evidence = ", ".join(item.get("evidence", [])[:3])`. For consistency with the sibling fields (and to bound prompt length / strip newlines), evidence items should also go through `_clean_current_game_token` (this is a current-game section — credibility evidence references this game's seer lines, so keep ids, just clean/truncate).

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py` (`_build_seer_credibility`)
- Test: `tests/agents/test_prompt_builder.py`

- [ ] **Step 1: Write the failing test**

There's an existing seer-credibility test file `tests/agents/test_seer_credibility_section.py`. Read it to copy the AgentContext construction pattern, then add a test there (or in test_prompt_builder.py if that's where credibility tests live — check first):

```python
def test_seer_credibility_evidence_is_cleaned_and_capped():
    # Read the existing test file to get the exact AgentContext + possible
    # seer_credibility construction helper; mirror it. The evidence items
    # contain newlines/long text that must be cleaned and capped.
    ctx = _make_seer_credibility_context(  # mirror existing helper name
        evidence=["p03 black-check line one\nline two", "x" * 200]
    )
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    # evidence is cleaned (no newline leak into the prompt line) and capped.
    assert "line one" in prompt
    assert "\nline two" not in prompt  # newline collapsed
    # each evidence item capped (the 200-char 'x' string is truncated)
    assert "x" * 200 not in prompt
```

(Adjust the helper name/context construction to match `tests/agents/test_seer_credibility_section.py`'s actual pattern. The key assertions: newline collapse + length cap on evidence items.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/agents/test_seer_credibility_section.py -q -o addopts="" -k evidence`
Expected: FAIL — evidence currently joined raw, so the newline leaks and the 200-char string is untruncated.

- [ ] **Step 3: Clean + cap evidence**

In `_build_seer_credibility`, replace line 711:
```python
            evidence = ", ".join(item.get("evidence", [])[:3])
```
with:
```python
            evidence = ", ".join(
                _clean_current_game_token(e, max_chars=40)
                for e in (item.get("evidence") or [])[:3]
                if str(e or "").strip()
            )
```

(`_clean_current_game_token` collapses newlines/whitespace and truncates to max_chars, matching the sibling `checks` field's `max_chars=20` style — use 40 for evidence to allow a bit more context.)

- [ ] **Step 4: Run seer credibility tests**

Run: `python -m pytest tests/agents/test_seer_credibility_section.py -q -o addopts=""`
Expected: PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/agents/prompt_builder.py tests/agents/test_seer_credibility_section.py
git commit -m "fix: cap and clean seer_credibility evidence in prompt"
```

---

## Task 4: Speech example uses pXX placeholder (not hardcoded p05)

**Why:** `_format_examples` (`prompt_builder.py:1688-1693`) — the vote example was already converted to `pXX` placeholders (line 1720-1739 comment + code), but the SPEECH example still hardcodes `p05`: `"speech": "我觉得p05很可疑..."` and `"pressure_target": "p05"`. The LLM may copy the example's `p05` into its current-game output, defeating the placeholder hardening. Per the P1 post-review-v2 comment (line 1720-1725), ALL example player IDs should be `pXX`.

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py` (`_format_examples` speech example, lines ~1688-1693)
- Test: `tests/agents/test_prompt_builder.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/agents/test_prompt_builder.py` (near the existing `test_format_examples_non_seer_vote_keeps_pXX_example` at line 3728):

```python
def test_format_examples_speech_uses_pXX_not_hardcoded_p05():
    ctx = _make_speech_example_context()  # mirror existing example-test context helper
    prompt = PlayerPromptBuilder(ctx).build_user_prompt(RetryInfo())
    speech_section = prompt[prompt.find("示例输出（发言场景）"):] if "示例输出（发言场景）" in prompt else ""
    assert speech_section, "expected a speech example section"
    # The speech example must NOT hardcode p05; use pXX placeholder like the vote example.
    assert "p05" not in speech_section
    assert "pXX" in speech_section
```

(Read the existing `test_format_examples_non_seer_vote_keeps_pXX_example` and reuse its context-construction helper — if the helper builds a speech-triggering context, reuse it; otherwise construct an AgentContext with `task_type=TaskType.SPEECH` and the same legal_actions the existing example tests use. The key is that `_format_examples` emits the SPEECH example branch.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/agents/test_prompt_builder.py -q -o addopts="" -k test_format_examples_speech_uses_pXX`
Expected: FAIL — speech example contains hardcoded `p05`.

- [ ] **Step 3: Replace p05 with pXX in the speech example**

In `_format_examples`, the speech example block (lines ~1688-1693). Change:
```python
            parts.append('{"action_type": "speech", "target_id": null, '
                         '"speech": "我觉得p05很可疑，昨晚他的发言前后矛盾。", '
                         '"reason": "根据发言分析", "confidence": 0.7, '
                         f'"private_intent": {{"true_role": "{example_role}", '
                         f'"faction_goal": "{example_goal}", "claimed_view": "{example_view}", '
                         '"pressure_target": "p05", "risk_flags": []}}')
```
to:
```python
            parts.append('{"action_type": "speech", "target_id": null, '
                         '"speech": "我觉得pXX很可疑，昨晚他的发言前后矛盾。", '
                         '"reason": "根据发言分析", "confidence": 0.7, '
                         f'"private_intent": {{"true_role": "{example_role}", '
                         f'"faction_goal": "{example_goal}", "claimed_view": "{example_view}", '
                         '"pressure_target": "pXX", "risk_flags": []}}')
```

(Two substitutions: the speech text `p05` → `pXX`, and `pressure_target` `p05` → `pXX`. Do NOT touch the `example_target` variable at line 1571 — that's the `legal_targets[0]` fallback which is a different concern; leave it.)

- [ ] **Step 4: Run example tests**

Run: `python -m pytest tests/agents/test_prompt_builder.py -q -o addopts="" -k format_examples`
Expected: PASS (new + existing `test_format_examples_non_seer_vote_keeps_pXX_example`).

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/agents/prompt_builder.py tests/agents/test_prompt_builder.py
git commit -m "fix: speech example uses pXX placeholder not hardcoded p05"
```

---

## Task 5: Document persona never-drop decision in design §3 + whole-suite regression + PROGRESS

**Why:** `docs/superpowers/specs/2026-06-12-prompt-balance-hardening-design.md:93` says "Persona is therefore no longer a never-drop section", but the implementation (`_SectionSpec("_build_persona", ..., _NEVER_DROP_TIER)` at `prompt_builder.py:383`) and the test `test_persona_core_survives_under_budget_pressure` (`test_prompt_builder.py:4457`, asserts `"_build_persona" in PlayerPromptBuilder._NEVER_DROP`) both keep persona as never-drop. This is a real conflict. Decision: KEEP persona as never-drop (the behaviour is already shipped, tested, and persona is now behaviourally compact via `_clean_prompt_text` + `_MAX_PERSONA_LINE_CHARS`, so budget pressure rarely bites), and update the design doc to record this decision rather than changing code/tests (changing it would risk dropping persona under budget pressure — a behaviour change with no clear benefit). This is a doc-only reconciliation, no code change.

**Files:**
- Modify: `docs/design/werewolf-agent-v1-design.md` §3 (or the prompt-balance section)
- Modify: `PROGRESS.md`

- [ ] **Step 1: Run the affected suites**

Run: `python -m pytest tests/agents -q -o addopts="" 2>&1 | tail -5`
Expected: all PASS.

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest -q -o addopts="" 2>&1 | tail -5`
Expected: all PASS (no regressions; full suite was 3382 passed, 1 skipped on master before this branch; expect +N new tests).

- [ ] **Step 3: compile-check**

Run: `python -m compileall -q werewolf_agent`
Expected: no output.

- [ ] **Step 4: Update design doc**

In `docs/design/werewolf-agent-v1-design.md`, find the prompt-balance / persona section (§3). Add a clarifying note that persona remains never-drop in the implementation. Locate the section that mirrors the spec's "Persona is therefore no longer a never-drop section" intent (or the persona styling paragraph) and append:

```markdown

**Persona never-drop 决议**（2026-06-18，`prompt-sanitizer-fix`）：尽管 prompt-balance spec 曾提议将 persona 改为可裁剪 section，实现中 persona 仍是 never-drop（`_SectionSpec("_build_persona", ..., _NEVER_DROP_TIER)`，测试 `test_persona_core_survives_under_budget_pressure` 锁定）。原因：persona 已行为化为短行（`_clean_prompt_text` + `_MAX_PERSONA_LINE_CHARS` 限长），预算压力下很少触发裁剪；保持 persona 稳定可避免 LLM 在预算紧张时丢失人格一致性。若未来需改为可裁剪，需同步更新该测试并评估对发言风格一致性的影响。
```

(If the exact anchor paragraph isn't found, place it at the end of the §3 persona subsection. The note is doc-only — no rule change.)

- [ ] **Step 5: Update PROGRESS.md**

Add a new section at the top of `PROGRESS.md` (above `reflection-synthesis-upgrade`). Update `Current Status` (phase → `prompt-sanitizer-fix`, last updated → `2026-06-18`, active task summarizing the four fixes). Record: the four fixes (possible_worlds/simulation current-game ids, seer_credibility evidence cap, speech example pXX, persona never-drop doc decision), files changed, verification commands, and open risks (the `example_target` fallback `legal_targets[0] or "p05"` at line 1571 is a separate concern — it only fires when legal_targets is empty, rare; tracked but not fixed here).

- [ ] **Step 6: Commit**

```bash
git add PROGRESS.md docs/design/werewolf-agent-v1-design.md
git commit -m "docs: record persona never-drop decision and log prompt sanitizer fix"
```

---

## Open Risks / Out of Scope

- **`example_target` fallback** (`prompt_builder.py:1571`): `ctx.legal_targets[0] if ctx.legal_targets else "p05"` — still hardcodes `p05` as a fallback when `legal_targets` is empty. This is rare (most tasks have legal targets) and is a different code path from the speech example. Tracked, not fixed here. A future fix should make the fallback a `pXX` placeholder too.
- **`warning`/`horizon` static strings**: left on `_clean_prompt_text` (Task 1/2) since they contain no real ids — switching them is cosmetic, not required for correctness, and kept the diff minimal.
- **Persona never-drop**: resolved by doc reconciliation (keep behaviour, update design §3). If a future requirement wants persona droppable under budget pressure, that's a separate code+test change.
- **`_clean_list_items` still uses `_clean_prompt_text`** (id → 历史玩家) for genuinely cross-game callers — unchanged. Only the current-game cognition call sites switched. No new `_clean_current_game_list_items` helper added (inline comprehensions are minimal and clear); if a third current-game list caller appears, extract a helper then.

## Self-Review

- **Spec/coverage:** The four审查 findings (possible_worlds/simulation sanitizer mis-use, seer_credibility evidence, speech p05, persona conflict) each map to a task. The `example_target` fallback is explicitly out of scope (tracked).
- **Placeholder scan:** No TBD/TODO. Each code step shows actual code. Tasks 3 & 4 Step 1 include "mirror existing helper" reads because the test fixtures live in existing test files — this is verification, not a placeholder.
- **Type consistency:** `_clean_current_game_token(value, *, max_chars)` signature used consistently (module-level, line 143). Inline comprehensions in Tasks 1/2 mirror the existing `key_assignments`/`affected_players` patterns (lines 750, 802). `pXX` placeholder matches the existing vote-example convention (line 1727+).
- **Safety:** The fix preserves MORE current-game information (real ids), never less. These sections are derived from visible facts (architecture-boundaries: public_record / private_reasoning tiers), so no new leak — the ids were always meant to be in-game. Cross-game sections keep the id→历史玩家 scrub unchanged.
