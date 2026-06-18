# Reflection Synthesis Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ReflectionSynthesizer` actually absorb the LLM self-review (currently discarded except as raw `source` text) and close the `wrong_action` prompt-safety gap, so V2 reflections carry the player's own 【保留的优点】 structured strengths and the quality gate's truth-claim scan can no longer be bypassed by `auto_verified`.

**Architecture:** Three focused, backward-compatible changes inside `werewolf_agent/memory/`. (1) `ReflectionEntryV2.prompt_visible_texts()` stops including `mistake_patterns[].wrong_action` — that field carries deterministic ground-truth (`review.py:108` writes `实际 {actual}`), is not a `prompt_card` field, and is already excluded from the slim render whitelist (`_slim_reflection_hints`), so removing it from the visible-scan set closes the defence-in-depth hole where `_has_unsafe_truth_claim` was bypassed by `auto_verified=True`. (2) `ReflectionSynthesizer` gains `_extract_llm_strengths(llm_self_review)` that parses the 【保留的优点】 section the role-family templates already mandate, producing fact-free `preserved_strengths` (`fact_basis="llm_transferable"`) merged alongside the deterministic ones — implementing spec Synthesis rule 1 for strengths. (3) `corrected_from_llm` denial detection covers more LLM phrasings. No DB schema change; no rule-engine change; hybrid master-unknown → generic template is an intentional design (spec §10.2 line 1112), explicitly NOT changed.

**Tech Stack:** Python 3.11, pytest, pydantic, dataclasses, conda env `wofkill`, PowerShell. Tests run with `-o addopts=""` to dodge the local pytest-xdist `.pytest_tmp` permission issue (see `memory/project-pytest-addopts-bypass.md`).

---

## File Structure

- **Modify** `werewolf_agent/memory/schemas.py` — `ReflectionEntryV2.prompt_visible_texts()` drops `pattern.wrong_action` from the scanned set (field stays in the schema for audit).
- **Modify** `werewolf_agent/memory/reflection.py` — `ReflectionSynthesizer.synthesize` calls a new `_extract_llm_strengths` and merges results into `preserved_strengths`; `_preserved_strengths` enriched; `corrected` denial regex widened; new helper for section parsing.
- **Modify** `docs/design/werewolf-agent-v1-design.md` §10.2 — clarifying note (no rule change): LLM 【保留的优点】 is now structured into `preserved_strengths`; `wrong_action` is audit-only, not prompt-visible.
- **Test** `tests/memory/test_reflection_v2.py` (extend), plus a new focused test module if needed.

**Design boundaries respected:** No change to game rules, role abilities, or the rule engine. No change to the `reflections` DB table `(entry_id, game_id, player_id, entry_json)`. Reflections remain LLM-prompt-layer owned. `wrong_action` stays in `entry_json` for audit — it just stops being treated as prompt-visible. Hybrid master-unknown → generic template is intentional (spec §10.2), untouched.

---

## Task 1: Exclude wrong_action from prompt-visible safety scope

**Why:** `prompt_visible_texts()` (`schemas.py:333-339`) includes `pattern.wrong_action`. `wrong_action` is populated from `review_report.error_analysis` (`reflection.py:338`), whose text is `误判 {pid} 为 {guessed}（实际 {actual}）…` (`review.py:108`) — i.e. it carries deterministic ground-truth role. The quality gate's `_has_unsafe_truth_claim` (`reflection.py:220-224`) is supposed to reject entries whose prompt-visible text contains truth tokens (`实际`/`真实身份`/`底牌`/…), BUT it short-circuits to `return False` whenever `prompt_card.auto_verified` is True (`reflection.py:221`), and every deterministic mistake pattern sets `auto_verified=True` (`reflection.py:341`). So a `wrong_action` containing `实际预言家` is never caught. Today this is latent (the slim render whitelist `_slim_reflection_hints` at `prompt_builder.py:1961` does not include `wrong_action`, so it never reaches a live prompt), but it is a closed-by-accident safety path. Per spec line 284 ("`prompt_card` is the only player-prompt-facing summary") `wrong_action` is not a prompt-facing field, so it should not be in the prompt-visible scan set.

**Files:**
- Modify: `werewolf_agent/memory/schemas.py` (`prompt_visible_texts`)
- Test: `tests/memory/test_reflection_v2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/memory/test_reflection_v2.py` (reuse the file's existing imports; `_v2_entry` helper already constructs entries):

```python
def test_prompt_visible_texts_excludes_wrong_action_with_truth_token() -> None:
    # wrong_action carries deterministic ground-truth ("实际预言家"); it must
    # NOT be in prompt_visible_texts, so the quality gate's truth-claim scan
    # is not bypassed by auto_verified.
    entry = _v2_entry(
        mistake_patterns=[
            {
                "category": "vote_mistake",
                "trigger": "双预言家对跳",
                "wrong_action": "误判 某玩家 为 狼人（实际 预言家）",
                "better_action": "先核验警徽流",
                "fact_basis": "auto_review",
                "auto_verified": True,
                "corrected_from_llm": False,
            }
        ],
    )
    visible = entry.prompt_visible_texts()
    joined = "\n".join(visible)
    # The truth-bearing wrong_action text must not appear in the prompt-visible set.
    assert "实际 预言家" not in joined
    assert "误判 某玩家 为 狼人" not in joined
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda activate wofkill; python -m pytest tests/memory/test_reflection_v2.py::test_prompt_visible_texts_excludes_wrong_action_with_truth_token -q -o addopts="" --basetemp=.pytest_tmp/rs01`
Expected: FAIL — `wrong_action` is currently included, so `"实际 预言家"` is present in `joined`.

- [ ] **Step 3: Drop wrong_action from the visible set**

In `werewolf_agent/memory/schemas.py`, edit `prompt_visible_texts()` so the per-pattern loop no longer appends `pattern.wrong_action`. Keep `category`, `trigger`, `better_action` (these are fact-free / transferable). The loop currently (`schemas.py:333-339`) is:

```python
        for pattern in self.mistake_patterns:
            texts.extend([
                pattern.category,
                pattern.trigger,
                pattern.wrong_action,
                pattern.better_action,
            ])
```

Change it to:

```python
        for pattern in self.mistake_patterns:
            # wrong_action carries deterministic ground-truth (review.py:108
            # writes "实际 {actual}"); it is audit-only, not prompt-visible
            # (spec: prompt_card is the only player-prompt-facing summary).
            texts.extend([
                pattern.category,
                pattern.trigger,
                pattern.better_action,
            ])
```

- [ ] **Step 4: Run the reflection_v2 test file to verify pass + no regressions**

Run: `python -m pytest tests/memory/test_reflection_v2.py -q -o addopts="" --basetemp=.pytest_tmp/rs01`
Expected: PASS (all existing tests stay green; the quality gate still rejects `player_id_leak` via other visible fields).

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/memory/schemas.py tests/memory/test_reflection_v2.py
git commit -m "fix: exclude wrong_action from reflection prompt-visible safety scan"
```

---

## Task 2: Structure LLM 【保留的优点】 into preserved_strengths

**Why:** spec Synthesis rule 1 says "Convert subjective review into structured mistake and strength patterns." Today `ReflectionSynthesizer.synthesize` (`reflection.py:256-319`) builds `mistake_patterns` only from deterministic `review_report.error_analysis` and `preserved_strengths` only from `review_report.successful_strategies` (`reflection.py:347-356`). The LLM self-review text — which the role-family templates (`agent_adapter.py:2700-2702`, `2725-2727`) explicitly instruct the player to fill with 1-2 concrete reusable strengths under a 【保留的优点】 header — is only stashed into `source.llm_self_review` and otherwise discarded. So the player's own articulated strengths never become structured learning. This task extracts the fact-free 【保留的优点】 bullets into `preserved_strengths` (provenance `llm_transferable`), merged alongside the deterministic ones. Per spec rule 7, LLM-only fact-bound claims (votes/roles/deaths/checks) must not be promoted; the 【保留的优点】 section is tactical/reusable by design, and the extractor additionally drops any bullet containing truth tokens, keeping it prompt-safe.

**Files:**
- Modify: `werewolf_agent/memory/reflection.py` (`ReflectionSynthesizer.synthesize`, new `_extract_llm_strengths`, `_preserved_strengths`)
- Test: `tests/memory/test_reflection_v2.py`

- [ ] **Step 1: Confirm the ReflectionPreservedStrength schema**

Read `werewolf_agent/memory/schemas.py` around `ReflectionPreservedStrength` to confirm its fields. The deterministic builder (`reflection.py:351-355`) sets `category`, `behavior`, `reuse_condition`. Check whether a `fact_basis` (or similar provenance) field exists on `ReflectionPreservedStrength`. If it does, LLM-extracted strengths set `fact_basis="llm_transferable"`; if it does not, proceed without it (the `behavior` text is the learning payload) and note it in self-review. Mirror the exact field names the deterministic builder uses.

- [ ] **Step 2: Write the failing tests**

Append to `tests/memory/test_reflection_v2.py`. Build a `ReviewReport` the way the file's existing synthesizer tests do (read them first to copy the constructor pattern):

```python
def test_synthesize_structures_llm_preserved_strengths() -> None:
    from werewolf_agent.memory.reflection import ReflectionSynthesizer
    from werewolf_agent.memory.schemas import ReviewReport

    report = ReviewReport(
        game_id="g1",
        player_id="p01",
        role="seer",
        faction_won=True,
    )
    llm_review = (
        "【投票错误】我 D2 站错边。\n"
        "【保留的优点】本局做对的:\n"
        "- N2 用解药救了警长,后续归票翻盘\n"
        "- D3 提前质疑悍跳狼警徽流时间线,被采信\n"
    )
    entry = ReflectionSynthesizer().synthesize(
        llm_self_review=llm_review,
        review_report=report,
        faction="good",
    )
    behaviors = [s.behavior for s in entry.preserved_strengths]
    # The LLM-articulated strengths are now structured (not just stashed in source).
    assert any("解药救了警长" in b for b in behaviors)
    assert any("警徽流" in b for b in behaviors)


def test_synthesize_llm_strength_drops_truth_tokens() -> None:
    # A bullet leaking truth tokens (实际/真实身份) must NOT be structured.
    from werewolf_agent.memory.reflection import ReflectionSynthesizer
    from werewolf_agent.memory.schemas import ReviewReport

    report = ReviewReport(game_id="g1", player_id="p01", role="villager", faction_won=True)
    llm_review = (
        "【保留的优点】\n"
        "- 某玩家实际是预言家我保住了他\n"
        "- 我坚持证据优先的站边\n"
    )
    entry = ReflectionSynthesizer().synthesize(
        llm_self_review=llm_review,
        review_report=report,
        faction="good",
    )
    behaviors = [s.behavior for s in entry.preserved_strengths]
    assert not any("实际" in b for b in behaviors)
    assert any("证据优先" in b for b in behaviors)


def test_synthesize_keeps_deterministic_strengths_when_no_llm_section() -> None:
    from werewolf_agent.memory.reflection import ReflectionSynthesizer
    from werewolf_agent.memory.schemas import ReviewReport

    report = ReviewReport(game_id="g1", player_id="p01", role="seer", faction_won=True)
    report.successful_strategies.append("角色判断准确率高，继续保持基于证据的推理方式")
    entry = ReflectionSynthesizer().synthesize(
        llm_self_review="没有段落头的纯文本反思。",
        review_report=report,
        faction="good",
    )
    behaviors = [s.behavior for s in entry.preserved_strengths]
    assert any("证据" in b for b in behaviors)
```

- [ ] **Step 3: Run to verify they fail**

Run: `python -m pytest tests/memory/test_reflection_v2.py::test_synthesize_structures_llm_preserved_strengths tests/memory/test_reflection_v2.py::test_synthesize_llm_strength_drops_truth_tokens tests/memory/test_reflection_v2.py::test_synthesize_keeps_deterministic_strengths_when_no_llm_section -q -o addopts="" --basetemp=.pytest_tmp/rs02`
Expected: first test FAILS (LLM strengths currently discarded); the others may pass trivially or fail depending on current behaviour.

- [ ] **Step 4: Implement _extract_llm_strengths and merge**

In `werewolf_agent/memory/reflection.py`:

Add a module-level regex near the top (after `_PLAYER_ID_RE`):

```python
# Captures the 【保留的优点】 section body up to the next 【...】 header or end.
_LLM_STRENGTH_SECTION_RE = re.compile(
    r"【保留的优点】[】:：\s]*(.*?)(?=【[^】]+】|$)",
    re.DOTALL,
)
_LLM_TRUTH_TOKENS = ("实际", "真实身份", "底牌", "查验结果", "死亡原因")
```

Add the extractor method to `ReflectionSynthesizer`:

```python
    @staticmethod
    def _extract_llm_strengths(llm_self_review: str) -> list[ReflectionPreservedStrength]:
        """Parse the 【保留的优点】 section into fact-free preserved strengths.

        Spec Synthesis rule 1: subjective review must become structured
        strengths. Spec rule 7: LLM-only fact-bound claims (votes/roles/
        deaths/checks) must not be promoted — so any bullet containing a
        truth token is dropped. Returns at most 2 strengths.
        """
        match = _LLM_STRENGTH_SECTION_RE.search(str(llm_self_review or ""))
        if not match:
            return []
        body = match.group(1)
        strengths: list[ReflectionPreservedStrength] = []
        for raw_line in body.splitlines():
            line = _scrub_ids(raw_line).strip().lstrip("-•").strip()
            if len(line) < 6:
                continue
            if any(token in line for token in _LLM_TRUTH_TOKENS):
                continue
            strengths.append(ReflectionPreservedStrength(
                category="speech_quality" if any(k in line for k in ("发言", "质疑", "表达")) else "strategy",
                behavior=line,
                reuse_condition="本局公开事实支持同类判断时",
            ))
            if len(strengths) >= 2:
                break
        return strengths
```

Wire it into `synthesize` — replace the line `strengths = self._preserved_strengths(review_report)` (currently `reflection.py:268`) with a merge that keeps deterministic strengths first, then appends LLM strengths up to a total cap of 3:

```python
        strengths = self._preserved_strengths(review_report)
        for llm_strength in self._extract_llm_strengths(llm_self_review):
            if any(_jaccard(llm_strength.behavior, s.behavior) >= 0.6 for s in strengths):
                continue
            strengths.append(llm_strength)
            if len(strengths) >= 3:
                break
```

(`_jaccard` already exists at `reflection.py:63` for de-dup. If the `ReflectionPreservedStrength` schema (Step 1) has a `fact_basis` field, set `fact_basis="llm_transferable"` on the LLM-extracted strengths; otherwise omit.)

- [ ] **Step 5: Run the reflection_v2 test file to verify pass**

Run: `python -m pytest tests/memory/test_reflection_v2.py -q -o addopts="" --basetemp=.pytest_tmp/rs02`
Expected: PASS (new + existing).

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/memory/reflection.py tests/memory/test_reflection_v2.py
git commit -m "feat: structure LLM 保留的优点 into reflection preserved_strengths"
```

---

## Task 3: Widen corrected_from_llm denial detection

**Why:** `corrected_from_llm` (`reflection.py:263-266`) is set True only when the deterministic review found errors AND the LLM self-review literally contains one of `没有犯错|没犯错|无错误|都正确`. Players phrase denial many other ways ("我这局没什么问题", "我的判断都挺准", "没有明显失误", "都还好"), so real contradictions between an optimistic LLM self-review and a deterministic error analysis are missed, understating the `corrected_from_llm` provenance signal that downstream audit relies on.

**Files:**
- Modify: `werewolf_agent/memory/reflection.py` (`synthesize` corrected line)
- Test: `tests/memory/test_reflection_v2.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/memory/test_reflection_v2.py`:

```python
def test_synthesize_marks_corrected_for_varied_denial_phrasings() -> None:
    from werewolf_agent.memory.reflection import ReflectionSynthesizer
    from werewolf_agent.memory.schemas import ReviewReport

    report = ReviewReport(game_id="g1", player_id="p01", role="villager", faction_won=False)
    report.error_analysis.append("误判 某玩家 为 狼人（实际 预言家），最佳角色概率 0.80")
    for denial in (
        "我这局没什么问题",
        "我的判断都挺准的",
        "没有明显失误",
        "都还好",
    ):
        entry = ReflectionSynthesizer().synthesize(
            llm_self_review=denial,
            review_report=report,
            faction="good",
        )
        assert entry.mistake_patterns, f"expected patterns for denial={denial!r}"
        assert entry.mistake_patterns[0].corrected_from_llm is True, (
            f"expected corrected_from_llm=True for denial={denial!r}"
        )
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/memory/test_reflection_v2.py::test_synthesize_marks_corrected_for_varied_denial_phrasings -q -o addopts="" --basetemp=.pytest_tmp/rs03`
Expected: FAIL — the extra phrasings are not matched by the current narrow regex.

- [ ] **Step 3: Widen the regex**

In `werewolf_agent/memory/reflection.py`, replace the `corrected` assignment in `synthesize` (currently `reflection.py:263-266`):

```python
        corrected = bool(
            review_report.error_analysis
            and re.search(
                r"没有犯错|没犯错|无错误|都正确|没什么问题|没有明显失误|都还好|"
                r"判断都挺准|判断都挺准的|没失误|没有失误|都对|我的判断都",
                llm_self_review or "",
            )
        )
```

- [ ] **Step 4: Run the test file to verify pass**

Run: `python -m pytest tests/memory/test_reflection_v2.py -q -o addopts="" --basetemp=.pytest_tmp/rs03`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/memory/reflection.py tests/memory/test_reflection_v2.py
git commit -m "fix: widen reflection corrected_from_llm denial phrasing detection"
```

---

## Task 4: Whole-suite regression + PROGRESS + design §10.2 note

**Files:**
- Modify: `PROGRESS.md`
- Modify: `docs/design/werewolf-agent-v1-design.md` §10.2

- [ ] **Step 1: Run memory + cognition + runtime + agents suites**

Run: `python -m pytest tests/memory tests/cognition tests/runtime tests/agents -q -o addopts="" --basetemp=.pytest_tmp/rs04`
Expected: all PASS.

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest -q -o addopts="" 2>&1 | tail -5`
Expected: all PASS (no regressions; full suite was 3376 passed, 1 skipped on `master` before this branch).

- [ ] **Step 3: compile-check**

Run: `python -m compileall -q werewolf_agent`
Expected: no output.

- [ ] **Step 4: Update design §10.2**

In `docs/design/werewolf-agent-v1-design.md` §10.2 (复盘机制), append a clarifying paragraph (no rule change). After the existing bullet list (around line 1115), add:

```markdown

**结构化与 prompt 边界补充**（2026-06-18）：

- LLM 自评中【保留的优点】段由 `ReflectionSynthesizer._extract_llm_strengths` 解析为 `preserved_strengths`（fact-free，含真相 token 的条目丢弃），与确定性 `successful_strategies` 合并；这是 spec Synthesis rule 1 的 strengths 部分。
- `mistake_patterns[].wrong_action` 承载确定性真相（`实际 {actual}`），是审计字段，**不**属 prompt-visible；`prompt_card` 仍是唯一面向玩家 prompt 的摘要（spec line 284）。
- 混血儿 master 阵营未知时走通用模板仍是设计意图（见上），本次不动。
```

- [ ] **Step 5: Update PROGRESS.md**

Add a new section at the top of `PROGRESS.md` (following the existing entry style), above the `monitoring-closure-fix` section. Update the `Current Status` block (phase → `reflection-synthesis-upgrade`, last updated → `2026-06-18`, active task summarizing the three changes). Record: the three fixes, files changed, verification commands, and open risks (LLM mistake-pattern extraction with deterministic correction is deferred — spec rule 7 requires fact-bound LLM claims be corrected, which needs more than this plan; `evaluate_reflection_effectiveness` still has no production caller, tracked separately).

- [ ] **Step 6: Commit**

```bash
git add PROGRESS.md docs/design/werewolf-agent-v1-design.md
git commit -m "docs: log reflection synthesis upgrade in PROGRESS and design 10.2"
```

---

## Open Risks / Out of Scope

- **LLM mistake-pattern extraction is deferred.** spec Synthesis rule 1 also wants subjective *mistakes* structured, but spec rule 7 forbids promoting LLM-only fact-bound claims (votes/roles/deaths/checks) — correctly structuring them needs deterministic correction/reconciliation logic beyond this plan. This plan delivers the safe, high-value half: 【保留的优点】 → `preserved_strengths`. The mistake half is a follow-up plan.
- **`evaluate_reflection_effectiveness` still has no production caller.** The module + its tests exist, but no offline evaluation pipeline invokes it. Wiring a caller is a separate concern (depends on the evaluation pipeline) and is tracked, not done here.
- **Hybrid master-unknown → generic template is intentional** (spec §10.2 line 1112, design §10.2). Do NOT "fix" it to a hybrid-specific template — that would contradict the spec.
- No `reflections` DB schema change; `wrong_action` stays in `entry_json` for audit.

## Self-Review

- **Spec coverage:** spec Synthesis rule 1 (strengths half) → Task 2; spec prompt-safety / line 284 (`prompt_card` only prompt-facing) + the `unsafe_truth_claim` hard-reject intent → Task 1; provenance `corrected_from_llm` → Task 3. The mistake-extraction half of rule 1 is explicitly deferred (rule 7 reconciliation needs more). Hybrid template untouched per spec §10.2.
- **Placeholder scan:** No TBD/TODO. Every code step shows actual code. Task 2 Step 1 includes a schema confirmation read for `ReflectionPreservedStrength` fields (provenance field presence) — verification, not a placeholder.
- **Type consistency:** `_extract_llm_strengths` returns `list[ReflectionPreservedStrength]` (same type `_preserved_strengths` returns); merged in `synthesize` with `_jaccard` de-dup (existing helper). `_LLM_STRENGTH_SECTION_RE` / `_LLM_TRUTH_TOKENS` defined in Task 2 Step 4 and used there. `corrected` regex widened in Task 3 references the same `review_report.error_analysis` guard.
- **Safety:** Task 1 narrows prompt-visible scope (less truth surface). Task 2 drops truth-token bullets and reuses `_scrub_ids`. No change to slim render whitelist or to what reaches a live prompt beyond the intended strengths merge (strengths are rendered via existing `_slim_reflection_hints` allowed field `actionable_advice`/lesson paths, which are already prompt-safe).
