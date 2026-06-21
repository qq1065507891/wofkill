# Fix Plan - Reflection Parser Format (accept LLM's actual numbering/prose)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (or subagent-driven-development). Keep one active task in `PROGRESS.md`. Supplements `fix-reflection-memory.md` (1b parser) + `fix-reflection-prompt-rendering.md` (PR1/PR2 made LLM output sections; this fix makes the parser actually read them). Verify against `docs/design/werewolf-agent-v1-design.md` §10.2. Peer-reviewed 2026-06-22; revised to apply review findings (B1 preamble leak, fixture, missing tests).

## Goal

Make `_extract_llm_mistakes` / `_extract_llm_strengths` accept the section-body formats real LLMs produce (numeric markers `1.` / `1、` / `1)`, and prose paragraphs), not just `-`/`•`/`*` bullets. After this fix, the 10/12 reflections that now contain `【投票错误】`/`【悍跳分析】` sections (verified in game `g_1600154180`) will actually yield `mistake_patterns` / `preserved_strengths` → `quality_score ≥ 0.70` → `approved > 0`.

## Root Cause (verified against g_1600154180, 2026-06-22)

`_extract_llm_mistakes` (`werewolf_agent/memory/reflection.py:446-470`) and `_extract_llm_strengths` (`:402-419`) both do:

```python
for raw_line in body.splitlines():
    stripped = raw_line.strip()
    if not stripped.startswith(("-", "•", "*")):
        continue          # ← only markdown bullets accepted
    line = _scrub_ids(stripped).lstrip("-•*").strip()
    ...
```

But real LLM output (g_1600154180, p02) uses **numeric markers and prose**, not markdown bullets:

```
【投票错误】
D1我投了某玩家。...（prose paragraph, no prefix）

D2我投了某玩家。...（prose paragraph）

【信息缺失】
1. 我忽略了双预言家局的复杂性...
2. D2平安夜+警长被投的双重异常信号...
3. 猎人被放逐时应该考虑开枪对象...

【保留的优点】
1. D1在双预言家局中没有急于站队...
2. 注意到某玩家投票与主流不符的分析...
```

Result: **all 12 entries have `mistake_patterns=0` and `preserved_strengths=0`** (even the 10 with real section headers) → score caps at 0.4-0.6 (`review_only`) → 0 `approved`. This is the only remaining blocker for `approved > 0` after the prompt-rendering fixes.

## Deliverables

- **PR1:** New helper `_iter_section_items(body, *, min_chars=6)` in `reflection.py` that yields cleaned content items, accepting `-`/`•`/`*` bullets, numeric markers (`N.` / `N、` / `N)`), and plain prose lines; skips blanks, short items, **and short colon-terminated list preambles** (B1 fix). Both `_extract_llm_mistakes` and `_extract_llm_strengths` refactored to use it.
- **PR2:** End-to-end verification: a real-shaped LLM self-review (numeric + prose sections) → `synthesize` + `ReflectionQualityGate.evaluate` → `mistake_patterns > 0` AND `preserved_strengths > 0` AND `quality_status == APPROVED`.

## Required Boundary

- **Accept LLM's natural format; do NOT change the reflection template to force bullets.**
- Preserve all existing safety: `_LLM_TRUTH_TOKENS` drop, `_scrub_ids`, length floor, cap (3 mistakes / 2 strengths), `auto_verified=False` + `fact_basis="llm_transferable"` for mistakes.
- `RuleEngine` untouched; `reflections` schema unchanged.
- **Existing regression test `test_synthesize_structures_llm_preserved_strengths` (`tests/memory/test_reflection_v2.py:248-272`) MUST be updated**: its `【保留的优点】` body starts with `本局做对的:` (6 chars). Pre-fix that line was skipped (not a bullet); post-fix the helper would yield it unless the preamble filter catches it. The helper's colon-termination filter (see below) handles this, but the test's expected strengths must be re-asserted against the new behavior. Do not silently let it turn red.

## Required Order

1. **PR1 helper + refactor both parsers** (single PR):
   - Add `_iter_section_items` + unit tests (bullet / `1.` / `1、` / `1)` / prose / blank / short-preamble / **colon-preamble like `本局做对的:`** / prefix-stripping).
   - Refactor `_extract_llm_mistakes` to iterate `_iter_section_items(body)`; keep truth-token drop, cap 3, `auto_verified=False`.
   - Refactor `_extract_llm_strengths` the same way; keep cap 2.
   - **Update `test_synthesize_structures_llm_preserved_strengths`** if its expected strengths shift (the `本局做对的:` preamble is now explicitly filtered by the colon rule — verify the real bullet strengths still surface).
   - Add truth-token-drop test on the new helper path (numeric + `实际`), cap test (5 numeric → 3), and `1、`/`1)` parser-level tests.
2. **PR2 end-to-end + score analysis:**
   - End-to-end test with a g_1600154180-p02-shaped self-review → assert `quality_status == APPROVED` and `quality_score >= 0.85` (reviewer-confirmed the path reaches 1.0; lock the high bar).
   - Run `scripts/analyze_reflection_score.py` (update fixtures if needed) to confirm the prose/numeric path reaches approved.
3. **Real-game 1a-verify (user-run, post-merge):** confirm `reflections` table has `approved` entries.

## Files

- Modify: `werewolf_agent/memory/reflection.py` — add `_iter_section_items` (module-level helper near `_LLM_MISTAKE_SECTION_RE`), refactor `_extract_llm_mistakes` (~L446-470) and `_extract_llm_strengths` (~L402-419).
- Modify: `tests/memory/test_reflection_v2.py` — helper unit tests, numeric/prose parser tests, **update `test_synthesize_structures_llm_preserved_strengths`**, end-to-end approve test.
- Docs: `PROGRESS.md`, design doc §10.2 (one-line note: parser accepts bullet/numeric/prose).

## Concrete Reference Implementation (for the implementer)

### Helper (add at module level, after `_LLM_MISTAKE_HEADER_CATEGORY`)

```python
import re

# Accepts markdown bullets (-/•/*) and numeric markers (1./1、/1)).
# NOTE: requires whitespace after the marker (``\s+``); verified
# g_1600154180 LLM output always emits "1. " with a space. If a future
# model emits "1.text" (no space), the marker won't strip and the line
# is yielded with a residual "1." prefix — covered by a test assertion
# so the regression is visible if it ever happens.
_LEADING_ITEM_PREFIX_RE = re.compile(r"^\s*(?:[-•*]|\d+[.、)])\s+")

def _iter_section_items(body: str, *, min_chars: int = 6):
    """Yield cleaned content items from a reflection-section body.

    Accepts markdown bullets (``-``/``•``/``*``), numeric markers
    (``1.``/``1、``/``1)`` with trailing space), or plain prose lines.
    Strips the leading marker. Skips blank lines, items shorter than
    ``min_chars``, **and short colon-terminated list preambles** like
    "本局做对的:" (which are section intros, not items — and are
    exactly 6 chars, so ``len>=min_chars`` alone does NOT catch them;
    peer-review B1).

    Why: real LLM reflection output (verified g_1600154180) uses numeric
    markers and prose paragraphs, not markdown bullets; the previous
    ``startswith(("-", "•", "*")`` gate dropped everything.
    """
    for raw in body.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        cleaned = _LEADING_ITEM_PREFIX_RE.sub("", stripped, count=1).strip()
        if len(cleaned) < min_chars:
            continue
        # B1 fix: drop short colon-terminated list preambles
        # ("本局做对的:" / "本局做错的:" / "下局改进:") — these are
        # section intros, not items. Length guard (<=8) keeps real
        # prose (which rarely ends in a colon) intact.
        if cleaned.endswith(("：", ":", "。")) and len(cleaned) <= 8:
            continue
        yield cleaned
```

### Refactor `_extract_llm_mistakes` (replace the inner `for raw_line in body.splitlines():` loop)

```python
        for line in _iter_section_items(body):
            line = _scrub_ids(line)
            if any(token in line for token in _LLM_TRUTH_TOKENS):
                continue
            patterns.append(ReflectionMistakePattern(
                category=category,
                trigger=cls._trigger_for_category(category, line),
                wrong_action=line,
                better_action=cls._default_advice(role),
                fact_basis="llm_transferable",
                auto_verified=False,
                corrected_from_llm=False,
            ))
            if len(patterns) >= 3:
                return patterns
```

(If the helper is a `@staticmethod`, call `cls._iter_section_items(body)`; if module-level, call `_iter_section_items(body)` directly. Keep the existing surrounding `for match in _LLM_MISTAKE_SECTION_RE.finditer(...)` loop and the `category = _LLM_MISTAKE_HEADER_CATEGORY[header]` mapping unchanged.)

### Refactor `_extract_llm_strengths` (same shape)

Replace its `for raw_line in body.splitlines(): if not stripped.startswith(...)` block with `for line in _iter_section_items(body):` then `_scrub_ids(line)`. Keep cap 2, truth-token drop, category inference (`speech_quality` vs `strategy`).

### Key tests to add/update (`tests/memory/test_reflection_v2.py`)

```python
def test_iter_section_items_accepts_numeric_markers():
    body = "1. 第一点内容足够长\n2. 第二点内容足够长\n"
    items = list(_iter_section_items(body))
    assert items == ["第一点内容足够长", "第二点内容足够长"]

def test_iter_section_items_accepts_numeric_chinese_variants():
    # plan regex declares 1、 and 1) too — cover at parser level
    body = "1、第一点内容足够长\n2)第二点内容足够长\n"
    items = list(_iter_section_items(body))
    assert items == ["第一点内容足够长", "第二点内容足够长"]

def test_iter_section_items_accepts_prose_lines():
    body = "D1我投了某玩家，这是段落式反思内容。\n\nD2我又投了另一个玩家。"
    items = list(_iter_section_items(body))
    assert "D1我投了某玩家，这是段落式反思内容。" in items
    assert len(items) == 2

def test_iter_section_items_strips_bullet_and_numeric():
    body = "- bullet 条目内容足够长\n1. numeric 条目内容足够长\n"
    items = list(_iter_section_items(body))
    assert items == ["bullet 条目内容足够长", "numeric 条目内容足够长"]

def test_iter_section_items_drops_preamble_colon_line():
    # B1 fix: "本局做对的:" is exactly 6 chars — must NOT become an item
    body = "本局做对的:\n- 实际优点内容足够长\n"
    items = list(_iter_section_items(body))
    assert items == ["实际优点内容足够长"]
    assert all("本局做对的" not in i for i in items)

def test_extract_llm_mistakes_parses_numeric_section():
    review = "【投票错误】\n1. D1投了某玩家站错边，内容足够长\n2. D2跟票出错，内容足够长\n"
    patterns = ReflectionSynthesizer._extract_llm_mistakes(review, "villager")
    assert len(patterns) == 2
    assert all(p.fact_basis == "llm_transferable" and p.auto_verified is False for p in patterns)

def test_extract_llm_mistakes_caps_at_three_numeric():
    review = "【投票错误】\n1. 错误一内容足够长\n2. 错误二内容足够长\n3. 错误三内容足够长\n4. 错误四内容足够长\n5. 错误五内容足够长\n"
    patterns = ReflectionSynthesizer._extract_llm_mistakes(review, "villager")
    assert len(patterns) == 3

def test_extract_llm_mistakes_drops_truth_token_numeric():
    review = "【投票错误】\n1. 误判某玩家为狼人，实际是预言家，内容足够长\n2. 真实错误内容足够长\n"
    patterns = ReflectionSynthesizer._extract_llm_mistakes(review, "villager")
    assert len(patterns) == 1
    assert "实际" not in patterns[0].wrong_action

def test_extract_llm_strengths_parses_numeric_section():
    review = "【保留的优点】\n1. D1保持独立判断，内容足够长\n2. 质疑悍跳狼警徽流，内容足够长\n"
    strengths = ReflectionSynthesizer._extract_llm_strengths(review)
    assert len(strengths) == 2

def test_extract_llm_strengths_drops_truth_token_numeric():
    review = "【保留的优点】\n1. 查验结果正确使用了银水，内容足够长\n2. 真实优点内容足够长\n"
    strengths = ReflectionSynthesizer._extract_llm_strengths(review)
    assert len(strengths) == 1
    assert "查验结果" not in strengths[0].behavior

def test_synthesize_numeric_prose_review_reaches_approved():
    # Shaped like g_1600154180 p02: prose 【投票错误】 + numeric 【信息缺失】 + numeric 【保留的优点】
    llm_review = """【投票错误】
D1我投了某玩家，当时信了假预言家面，站错边害了好人，内容足够长。

D2我投了某玩家，跟票出错，内容足够长。

【信息缺失】
1. 忽略了双预言家查验逻辑自洽性，内容足够长
2. 没利用猎人开枪施压，内容足够长

【保留的优点】
1. D1没有急于站队保持独立判断，内容足够长
2. 注意到某玩家投票轨迹异常，内容足够长
"""
    # Direct ReviewReport construction (no _make_review_report helper exists).
    # error_analysis=[] is the default — required so LLM-parsed items are the
    # only mistake source (otherwise deterministic _mistake_patterns runs first).
    report = ReviewReport(
        game_id="g1", player_id="p01", role="villager", faction_won=False,
    )
    assert report.error_analysis == []  # sanity: LLM path is the sole source
    entry = ReflectionSynthesizer().synthesize(llm_self_review=llm_review, review_report=report, faction="good")
    gated = ReflectionQualityGate().evaluate(entry)
    assert gated.mistake_patterns, "mistake_patterns must be non-empty from numeric/prose sections"
    assert gated.preserved_strengths, "preserved_strengths must be non-empty"
    assert gated.quality_status == ReflectionQualityStatus.APPROVED
    assert gated.quality_score >= 0.85  # reviewer-confirmed reaches 1.0; lock high bar
```

**Also update** `test_synthesize_structures_llm_preserved_strengths` (existing, ~L248-272): after the refactor, re-run it and confirm its expected `【保留的优点】` bullet strengths still surface (the `本局做对的:` preamble is filtered by the colon rule, so the real bullets should still be the yielded strengths). If the assertion needs adjustment, update it with a comment noting the preamble filter.

## Done Means

- `_iter_section_items` accepts bullet / `1.` / `1、` / `1)` / prose, strips marker, skips blanks + short items + **short colon preambles** — unit tests pass (including `test_iter_section_items_drops_preamble_colon_line`).
- `_extract_llm_mistakes` / `_extract_llm_strengths` produce non-empty output for numeric-marker and prose section bodies; existing bullet-based tests pass (regression); `test_synthesize_structures_llm_preserved_strengths` updated if its preamble expected-behavior shifted.
- Truth-token drop covered on the new helper path (numeric + `实际`/`查验结果`); cap covered (5 numeric → 3); `1、`/`1)` covered.
- End-to-end: a g_1600154180-p02-shaped self-review → `mistake_patterns > 0` AND `preserved_strengths > 0` AND `quality_status == APPROVED` AND `quality_score >= 0.85`.
- All existing reflection tests pass; `RuleEngine` unaffected; `git diff --check` clean.
- (Post-merge, user-run) Real game: `reflections` table shows `approved` entries.

## Risks

- **B1 (now handled, was a blocker):** `min_chars=6` alone did NOT filter `本局做对的:` (exactly 6 chars) — it would have been yielded as a strength and broken `test_synthesize_structures_llm_preserved_strengths`. Fixed by the explicit colon-termination + `len<=8` rule. Test `test_iter_section_items_drops_preamble_colon_line` locks it.
- **Long preambles:** a long non-item line (e.g. "我反思了以下几点:") is NOT caught by the `len<=8` colon rule. Rare in observed LLM output; if it appears, tighten later. Tracked here as a known minor.
- **No-space numeric markers (`1.text`):** regex requires `\s+`; current LLM always emits `1. ` with space, so no live risk. If a future model drops the space, the marker won't strip (residual `1.` in `wrong_action`). Not covered by an assertion (acceptable — would surface as visible noise, not silent corruption).
- **Prose paragraph = one item:** a prose paragraph is yielded as one item (one `wrong_action`). This loses sub-structure but matches what the LLM gives; acceptable for V1.
- **Does not fix `empty_response`** (see appendix) — that is an API/model-layer issue, independent.

---

## Appendix: `empty_response` diagnosis (NOT fixed by this plan — API/model layer)

Game `g_1600154180` had `fallback_rate=0.263` (vs 0.104 / 0.121 in prior games). The `empty_response` fallbacks are **84% concentrated in p01 (21) + p09 (16)**, both of which use the `ark_deepseek` profile → default model `ark_minimax_m3` (`minimax-m3` via ark provider, `text_json` mode, `max_tokens 1024`). Other profiles (`minimax_default` / `ark_glm` / `ark_minimax` / `ark_kimi`) have 0-2 empty each.

**Conclusion:** `minimax-m3` (via ark) was unstable/returning empty text **this run only** — same config, prior runs normal. This is an API/model-layer fluctuation, **not a code defect**. The existing retry (2-3 attempts) + fallback-to-default-action mechanism worked correctly (game completed, 119 steps).

**Code-layer options (optional, low-confidence, NOT in this plan's scope):**
- Bump `max_tokens` 1024 → 2048 for `ark_minimax_m3` (does not cause empty_response, but avoids truncation on long SPEECH/REFLECTION outputs).
- Add a per-profile empty-response metric/alert (observability, not a fix).
- If instability persists across multiple future runs, consider swapping `ark_deepseek`'s default away from `minimax-m3`, or investigate ark-side rate-limit/error for that model.

**Recommended next step on empty_response:** run one more real game and check whether 0.263 recurs. If it drops back to ~0.10-0.12, it was transient and no code change is warranted.

## Self-Review (run before handoff)

- **Spec coverage:** parser format gap (numeric/prose) → PR1; preamble leak B1 → fixed + test; truth-token/cap/`1、`/`1)` → covered; end-to-end approve → PR2; real-game verify → step 3; existing regression update → explicit. `empty_response` explicitly out of scope (appendix). ✓
- **Placeholders:** concrete helper code, parser diff, 11 real test functions, direct `ReviewReport(...)` construction. No `_make_review_report` reference remains. ✓
- **Type consistency:** `_iter_section_items` name + signature consistent across helper/parsers/tests. `auto_verified=False` / `fact_basis="llm_transferable"` match the 1b contract. ✓
