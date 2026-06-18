# Attribution Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a post-game `AttributionEngine` that annotates cognition-module exposures with `cited_by_decision` / `aligned_with_decision` / `harmful_transfer`, runs the consistency judge per trace with rebuilt `public_facts`, and produces `harmful_transfer_rate` + `judge_consistency_rate` that feed the regression gate.

**Architecture:** New `werewolf_agent/evaluation/attribution.py` (pure post-game pass) + `evaluation/text_similarity.py` (shared token/Jaccard helpers). `AttributionEngine.annotate(traces, result)` runs after `EvaluationTraceBuilder.build`, resolves RAG/reflection card text via an injected `AttributionTextResolver`, marks unresolved exposures `UNSUPPORTED`, and rebuilds the frozen traces with `dataclasses.replace`. `FullGameAblationRunner` gains an `attribution_text_resolver` constructor param and an `_enriched_metrics` helper that emits the two new metric keys plus unsupported reasons. Zero runtime/schema/rule-engine change.

**Tech Stack:** Python 3.11, pytest, dataclasses (`replace` on frozen dataclasses), conda env `wofkill`, PowerShell. Tests run with `-o addopts=""` (no `--basetemp`, to avoid the local pytest-xdist `.pytest_tmp` permission issue).

---

## File Structure

- **Create** `werewolf_agent/evaluation/text_similarity.py` — shared `tokenize` + `jaccard` (same regex as `memory.reflection._token_set`). Single responsibility: pure text-similarity primitives.
- **Create** `werewolf_agent/evaluation/attribution.py` — `AttributionTextResolver`, `AttributionEngine`, and module-level helpers (`_speech_from_decision`, `_exposure_representative_text`, `_cited`, `_aligned`, `_trace_outcome_is_bad`, `_rebuild_visible_facts`). Single responsibility: post-game decision-vs-exposure attribution.
- **Modify** `werewolf_agent/memory/reflection.py` — `_token_set` / `_jaccard` delegate to `evaluation.text_similarity` (keeps the private names as thin aliases so all in-file call sites are unchanged).
- **Modify** `werewolf_agent/evaluation/full_game_ablation.py` — `FullGameAblationRunner.__init__` gains `attribution_text_resolver`; new `_enriched_metrics(result, resolver)` emits the two metric keys and unsupported reasons; `run`/`_run_replay` use it; replay path omits both attribution keys.
- **Test** `tests/evaluation/test_attribution.py` (new), `tests/evaluation/test_full_game_ablation.py` (extend), `tests/evaluation/test_regression_gate.py` (extend), `tests/evaluation/test_text_similarity.py` (new, small).

**Design boundaries respected:** No change to game rules, role abilities, rule engine, runtime decision path, `action_trace_audit` payload, or the `reflections` DB schema. Attribution reads `GameResult` (a post-game artifact already allowed to hold ground truth, same boundary as `world_model_eval`). `public_facts` are rebuilt in-memory post-game, never persisted to an audit payload.

---

## Task 1: Shared text-similarity helpers

**Why:** `memory.reflection` has private `_token_set` / `_jaccard`; the attribution engine needs the same logic. Importing private memory helpers into evaluation is brittle. spec Open Decision 1 recommends a shared `evaluation/text_similarity.py` with `memory.reflection` re-importing. Verified no import cycle: `evaluation/` imports nothing from `memory.reflection`.

**Files:**
- Create: `werewolf_agent/evaluation/text_similarity.py`
- Modify: `werewolf_agent/memory/reflection.py` (`_token_set`, `_jaccard`)
- Test: `tests/evaluation/test_text_similarity.py`

- [ ] **Step 1: Write the failing test**

Create `tests/evaluation/test_text_similarity.py`:

```python
from werewolf_agent.evaluation.text_similarity import tokenize, jaccard


def test_tokenize_splits_ascii_and_cjk():
    assert tokenize("p03 vote 警徽流") == {"p03", "vote", "警", "徽", "流"}


def test_jaccard_identical_is_one():
    assert jaccard("警徽流 对跳", "警徽流 对跳") == 1.0


def test_jaccard_disjoint_is_zero():
    assert jaccard("abc", "xyz") == 0.0


def test_jaccard_partial():
    # {p03, vote} vs {p03, vote, extra} → 2/3
    assert round(jaccard("p03 vote", "p03 vote extra"), 4) == round(2 / 3, 4)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/evaluation/test_text_similarity.py -q -o addopts=""`
Expected: FAIL — `ModuleNotFoundError: werewolf_agent.evaluation.text_similarity`.

- [ ] **Step 3: Create text_similarity.py**

Create `werewolf_agent/evaluation/text_similarity.py`:

```python
"""Shared text-similarity helpers for evaluation and memory.

Same token regex as the original ``memory.reflection._token_set``:
ASCII word chars OR single CJK characters. Kept here (not in memory) so
``evaluation.attribution`` and ``memory.reflection`` share one source without
a brittle private-helper import. Verified import-cycle-free: ``evaluation``
imports nothing from ``memory.reflection``.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9_]+|[一-鿿]")


def tokenize(text: str) -> set[str]:
    """Token set: lowercase ASCII runs + individual CJK characters."""
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def jaccard(left: str, right: str) -> float:
    """Jaccard similarity over token sets; 0.0 if either side is empty."""
    a = tokenize(left)
    b = tokenize(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
```

- [ ] **Step 4: Re-point reflection at the shared helpers**

In `werewolf_agent/memory/reflection.py`, replace the local `_token_set` (lines ~63-66) and `_jaccard` (lines ~69-72) definitions with thin aliases so all existing in-file call sites (`_jaccard` is used by `_find_duplicate`) stay unchanged:

```python
from werewolf_agent.evaluation.text_similarity import jaccard as _jaccard, tokenize as _token_set
```

Delete the bodies of the old `_token_set` / `_jaccard` functions (the import now provides them). Keep the `_PLAYER_ID_RE` / `_scrub_ids` / `_cap_source_text` helpers above untouched.

- [ ] **Step 5: Run text_similarity + reflection tests + import-cycle check**

Run:
```
python -m pytest tests/evaluation/test_text_similarity.py tests/memory/test_reflection_v2.py -q -o addopts=""
python -c "import werewolf_agent.memory.reflection, werewolf_agent.evaluation.text_similarity; print('no cycle')"
```
Expected: tests PASS; the import smoke print outputs `no cycle` (confirms `memory.reflection` importing `evaluation.text_similarity` does not cycle).

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/evaluation/text_similarity.py werewolf_agent/memory/reflection.py tests/evaluation/test_text_similarity.py
git commit -m "refactor: share text-similarity helpers between evaluation and memory"
```

---

## Task 2: AttributionTextResolver

**Why:** spec: RAG/reflection attribution must resolve the prompt-safe card text post-game by `item_id`. The compact exposure record (entry_id/rank/score/prompt_visible) does NOT carry the card text the player saw. The resolver bridges exposure → full prompt-safe text. It is an interface (test fixture map in tests; real store-backed in production), injected into `AttributionEngine`.

**Files:**
- Create: `werewolf_agent/evaluation/attribution.py` (start the module with the resolver)
- Test: `tests/evaluation/test_attribution.py` (start the file)

- [ ] **Step 1: Write the failing test**

Create `tests/evaluation/test_attribution.py`:

```python
from werewolf_agent.evaluation.attribution import AttributionTextResolver
from werewolf_agent.evaluation.feedback_schemas import ModuleExposure


def test_resolver_returns_rag_prompt_safe_text():
    resolver = AttributionTextResolver(
        rag_entries={"r1": {
            "title": "对跳局核验警徽流",
            "situation_signature": "seer counterclaim",
            "transferable_lesson": "先比较验人时间线",
            "recommended_action": "列对比表再站边",
            "misuse_risk": "不要套用历史身份",
        }},
    )
    exposure = ModuleExposure(module="rag", item_id="r1")
    text = resolver.rag_text(exposure)
    assert "对跳局核验警徽流" in text
    assert "列对比表再站边" in text


def test_resolver_returns_reflection_prompt_card_text():
    resolver = AttributionTextResolver(
        reflection_entries={"ref1": {
            "theme": "投票前核验",
            "lesson": "过早站边会误判",
            "recommended_action": "核验警徽流",
            "misuse_risk": "不映射本局玩家",
        }},
    )
    exposure = ModuleExposure(module="reflection", item_id="ref1")
    text = resolver.reflection_text(exposure)
    assert "投票前核验" in text
    assert "核验警徽流" in text


def test_resolver_returns_none_when_item_missing():
    resolver = AttributionTextResolver()
    exposure = ModuleExposure(module="rag", item_id="unknown")
    assert resolver.rag_text(exposure) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/evaluation/test_attribution.py -q -o addopts="" -k resolver`
Expected: FAIL — `ModuleNotFoundError: werewolf_agent.evaluation.attribution`.

- [ ] **Step 3: Create attribution.py with the resolver**

Create `werewolf_agent/evaluation/attribution.py`:

```python
"""Post-game attribution engine for evaluation feedback traces.

Annotates cognition-module exposures (rag / reflection / possible_worlds /
simulator) with cited_by_decision / aligned_with_decision / harmful_transfer,
and runs the consistency judge per trace with rebuilt public_facts. Pure
post-game: no runtime change, no audit payload growth.
"""

from __future__ import annotations

from typing import Any, Mapping

from werewolf_agent.evaluation.feedback_schemas import ModuleExposure


_RAG_TEXT_FIELDS = (
    "title",
    "situation_signature",
    "transferable_lesson",
    "recommended_action",
    "misuse_risk",
)
_REFLECTION_TEXT_FIELDS = (
    "theme",
    "lesson",
    "recommended_action",
    "misuse_risk",
)


class AttributionTextResolver:
    """Resolve compact exposure records to the prompt-safe card text.

    Production wiring wraps ``RAGRepository.get(entry_id)`` /
    ``ReflectionMemory.all_v2_entries()`` (or a service-level cache). Tests
    pass fixture dicts. Returns ``None`` when the entry cannot be resolved —
    the engine then marks that exposure ``MetricSupport.UNSUPPORTED``.
    """

    def __init__(
        self,
        *,
        rag_entries: Mapping[str, Mapping[str, Any]] | None = None,
        reflection_entries: Mapping[str, Mapping[str, Any]] | None = None,
        rag_provider=None,
        reflection_provider=None,
    ) -> None:
        self._rag_entries = rag_entries
        self._reflection_entries = reflection_entries
        self._rag_provider = rag_provider
        self._reflection_provider = reflection_provider

    def rag_text(self, exposure: ModuleExposure) -> str | None:
        data = self._resolve("rag", exposure.item_id)
        if not data:
            return None
        return " ".join(str(data.get(f, "") or "") for f in _RAG_TEXT_FIELDS).strip()

    def reflection_text(self, exposure: ModuleExposure) -> str | None:
        data = self._resolve("reflection", exposure.item_id)
        if not data:
            return None
        card = data.get("prompt_card", data)
        return " ".join(str(card.get(f, "") or "") for f in _REFLECTION_TEXT_FIELDS).strip()

    def _resolve(self, module: str, item_id: str) -> Mapping[str, Any] | None:
        if module == "rag":
            if self._rag_provider is not None:
                return self._rag_provider(item_id)
            if self._rag_entries is not None:
                return self._rag_entries.get(item_id)
        elif module == "reflection":
            if self._reflection_provider is not None:
                return self._reflection_provider(item_id)
            if self._reflection_entries is not None:
                return self._reflection_entries.get(item_id)
        return None
```

- [ ] **Step 4: Run the resolver tests**

Run: `python -m pytest tests/evaluation/test_attribution.py -q -o addopts="" -k resolver`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/evaluation/attribution.py tests/evaluation/test_attribution.py
git commit -m "feat: add AttributionTextResolver for post-game card text"
```

---

## Task 3: cited — text match with resolver + UNSUPPORTED handling

**Why:** spec: `cited_by_decision` = Jaccard(`decision.reason + speech`, `exposure_representative_text`) ≥ 0.15. RAG/reflection representative text comes from the resolver; unresolved → `MetricSupport.UNSUPPORTED`. `speech` is read from `DecisionSnapshot.raw` (trace does not retain `parsed_action`). simulator representative text is `exposure.item_id + affected_players` (trace_builder stores the prediction event in `item_id`).

**Files:**
- Modify: `werewolf_agent/evaluation/attribution.py` (add helpers + `_cited`)
- Test: `tests/evaluation/test_attribution.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/evaluation/test_attribution.py`:

```python
from werewolf_agent.evaluation.attribution import (
    speech_from_decision,
    exposure_representative_text,
    cited,
)
from werewolf_agent.evaluation.feedback_schemas import DecisionSnapshot


def test_speech_from_decision_reads_raw_speech():
    d = DecisionSnapshot(action_type="speech", raw={"speech": "我怀疑p03", "reason": "x"})
    assert speech_from_decision(d) == "我怀疑p03"


def test_speech_from_decision_falls_back_to_public_story():
    d = DecisionSnapshot(action_type="speech", raw={"public_story": "归票p07"})
    assert speech_from_decision(d) == "归票p07"


def test_exposure_representative_text_rag_uses_resolver():
    resolver = AttributionTextResolver(rag_entries={"r1": {
        "title": "对跳局核验警徽流", "recommended_action": "列对比表",
    }})
    text = exposure_representative_text(
        ModuleExposure(module="rag", item_id="r1"), resolver,
    )
    assert "对跳局核验警徽流" in text


def test_exposure_representative_text_possible_worlds_uses_assignments():
    exposure = ModuleExposure(
        module="possible_worlds", item_id="World A",
        metadata={"key_assignments": {"p03": "werewolf", "p07": "seer"}},
    )
    text = exposure_representative_text(exposure, AttributionTextResolver())
    assert "p03=werewolf" in text
    assert "p07=seer" in text


def test_exposure_representative_text_simulator_uses_item_id():
    exposure = ModuleExposure(
        module="simulator", item_id="p03 exile pressure",
        metadata={"affected_players": ["p03"]},
    )
    text = exposure_representative_text(exposure, AttributionTextResolver())
    assert "p03 exile pressure" in text
    assert "p03" in text


def test_cited_true_when_jaccard_above_threshold():
    decision = DecisionSnapshot(
        action_type="vote", reason="p03发言可疑，需核验警徽流", raw={},
    )
    resolver = AttributionTextResolver(rag_entries={"r1": {
        "title": "核验警徽流", "recommended_action": "核验",
    }})
    exposure = ModuleExposure(module="rag", item_id="r1")
    assert cited(decision, exposure, resolver) is True


def test_cited_false_when_no_overlap():
    decision = DecisionSnapshot(action_type="vote", reason="abcdefg", raw={})
    resolver = AttributionTextResolver(rag_entries={"r1": {"title": "完全不同的内容"}})
    exposure = ModuleExposure(module="rag", item_id="r1")
    assert cited(decision, exposure, resolver) is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/evaluation/test_attribution.py -q -o addopts="" -k "speech_from_decision or exposure_representative_text or cited"`
Expected: FAIL — the names are not exported yet.

- [ ] **Step 3: Implement the helpers**

Append to `werewolf_agent/evaluation/attribution.py` (after the resolver):

```python
from werewolf_agent.evaluation.text_similarity import jaccard, tokenize  # noqa: E402 (added import)

_CITED_THRESHOLD = 0.15
_ACTION_VERBS = ("先", "不要", "避免", "必须", "优先", "核验", "比较", "列")


def speech_from_decision(decision) -> str:
    """Read the public speech text from DecisionSnapshot.raw.

    EvaluationTrace does not retain a standalone parsed_action; speech lives
    in decision.raw (set by EvaluationTraceBuilder._decision_snapshot).
    """
    if decision is None:
        return ""
    raw = decision.raw or {}
    return str(raw.get("speech") or raw.get("public_story") or "")


def exposure_representative_text(
    exposure: ModuleExposure,
    resolver: AttributionTextResolver,
) -> str | None:
    """Prompt-safe representative text for an exposure, or None if unresolved.

    None signals the engine to mark the exposure MetricSupport.UNSUPPORTED
    (RAG/reflection whose card text cannot be resolved post-game).
    possible_worlds/simulator always resolve from their structured metadata.
    """
    module = exposure.module
    meta = exposure.metadata
    if module == "rag":
        return resolver.rag_text(exposure)
    if module == "reflection":
        return resolver.reflection_text(exposure)
    if module == "possible_worlds":
        assignments = meta.get("key_assignments") or {}
        return " ".join(f"{pid}={role}" for pid, role in assignments.items())
    if module == "simulator":
        affected = meta.get("affected_players") or []
        return f"{exposure.item_id} {' '.join(str(p) for p in affected)}".strip()
    return None


def cited(decision, exposure: ModuleExposure, resolver: AttributionTextResolver) -> bool:
    exp_text = exposure_representative_text(exposure, resolver)
    if not exp_text:
        return False  # unresolved → engine marks UNSUPPORTED; not cited
    decision_text = f"{decision.reason or ''} {speech_from_decision(decision)}"
    return jaccard(decision_text, exp_text) >= _CITED_THRESHOLD
```

(If `decision.reason` is not an attribute — confirm `DecisionSnapshot` has `reason` at `feedback_schemas.py:34`; it does. `_ACTION_VERBS` is defined here for Task 4 use.)

- [ ] **Step 4: Run the cited tests**

Run: `python -m pytest tests/evaluation/test_attribution.py -q -o addopts="" -k "speech_from_decision or exposure_representative_text or cited"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/evaluation/attribution.py tests/evaluation/test_attribution.py
git commit -m "feat: attribution cited match with resolver and unsupported handling"
```

---

## Task 4: aligned — per-module direction rules

**Why:** spec: `aligned_with_decision` is a per-module direction check, independent of outcome. possible_worlds/simulator use `target_id` (or reason-mentioned players) vs assignments/affected_players; rag/reflection use action-verb adoption in the resolver-provided text.

**Files:**
- Modify: `werewolf_agent/evaluation/attribution.py` (add `aligned`)
- Test: `tests/evaluation/test_attribution.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/evaluation/test_attribution.py`:

```python
from werewolf_agent.evaluation.attribution import aligned


def _player_ids_from_reason(text: str) -> set[str]:
    import re
    return set(re.findall(r"p\d{1,3}", text or ""))


def test_aligned_possible_worlds_target_in_wolf_assignments():
    decision = DecisionSnapshot(action_type="vote", target_id="p03", reason="", raw={})
    exposure = ModuleExposure(
        module="possible_worlds", item_id="W",
        metadata={"key_assignments": {"p03": "werewolf", "p07": "seer"}},
    )
    assert aligned(decision, exposure, "good") is True


def test_aligned_possible_worlds_target_not_wolf_is_false():
    decision = DecisionSnapshot(action_type="vote", target_id="p07", reason="", raw={})
    exposure = ModuleExposure(
        module="possible_worlds", item_id="W",
        metadata={"key_assignments": {"p03": "werewolf", "p07": "seer"}},
    )
    assert aligned(decision, exposure, "good") is False


def test_aligned_simulator_target_in_affected_players():
    decision = DecisionSnapshot(action_type="vote", target_id="p03", reason="", raw={})
    exposure = ModuleExposure(
        module="simulator", item_id="evt",
        metadata={"affected_players": ["p03", "p05"]},
    )
    assert aligned(decision, exposure, "good") is True


def test_aligned_rag_adopts_recommended_action_verb():
    decision = DecisionSnapshot(action_type="vote", reason="我先核验了警徽流再投", raw={})
    resolver = AttributionTextResolver(rag_entries={"r1": {
        "title": "x", "recommended_action": "核验警徽流再决定",
    }})
    exposure = ModuleExposure(module="rag", item_id="r1")
    assert aligned(decision, exposure, "good", resolver) is True


def test_aligned_rag_missing_resolver_text_is_false():
    decision = DecisionSnapshot(action_type="vote", reason="核验", raw={})
    resolver = AttributionTextResolver()  # no entries
    exposure = ModuleExposure(module="rag", item_id="missing")
    assert aligned(decision, exposure, "good", resolver) is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/evaluation/test_attribution.py -q -o addopts="" -k aligned`
Expected: FAIL — `aligned` not exported.

- [ ] **Step 3: Implement aligned**

Append to `werewolf_agent/evaluation/attribution.py`:

```python
import re as _re

_PLAYER_ID_RE = _re.compile(r"p\d{1,3}")
_WOLF_ROLES = frozenset({"werewolf", "wolf"})


def _reason_players(decision) -> set[str]:
    if decision is None or not decision.reason:
        return set()
    return set(_PLAYER_ID_RE.findall(decision.reason))


def aligned(
    decision,
    exposure: ModuleExposure,
    faction: str,
    resolver: AttributionTextResolver | None = None,
) -> bool:
    """Per-module direction rule: did the decision follow the exposure?"""
    module = exposure.module
    meta = exposure.metadata
    target = decision.target_id if decision else None
    mentioned = _reason_players(decision)
    relevant_players = ({target} if target else set()) | mentioned

    if module == "possible_worlds":
        assignments = meta.get("key_assignments") or {}
        wolves = {pid for pid, role in assignments.items() if role in _WOLF_ROLES}
        return bool(relevant_players & wolves)
    if module == "simulator":
        affected = set(meta.get("affected_players") or [])
        return bool(relevant_players & affected)
    if module in ("rag", "reflection"):
        if resolver is None:
            return False
        exp_text = exposure_representative_text(exposure, resolver)
        if not exp_text:
            return False
        reason = decision.reason or "" if decision else ""
        # the decision adopted a recommended action verb that the card also mentions
        return any(verb in reason and verb in exp_text for verb in _ACTION_VERBS)
    return False
```

- [ ] **Step 4: Run aligned tests**

Run: `python -m pytest tests/evaluation/test_attribution.py -q -o addopts="" -k aligned`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/evaluation/attribution.py tests/evaluation/test_attribution.py
git commit -m "feat: attribution aligned per-module direction rules"
```

---

## Task 5: harmful + outcome_is_bad + beneficial byproduct

**Why:** spec: `harmful_transfer = cited ∧ aligned ∧ outcome_is_bad`. `outcome_is_bad` reuses monitoring-closure-fix signals (`legal=False`, `leaked_hidden_info`) plus trace-level `good voter ∧ vote_hit_wolf=False` and wrong_target. `beneficial` (`cited ∧ aligned ∧ ¬bad`) is a stored byproduct (spec Open Decision 2: yes).

**Files:**
- Modify: `werewolf_agent/evaluation/attribution.py` (add `trace_outcome_is_bad`, `is_harmful`)
- Test: `tests/evaluation/test_attribution.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/evaluation/test_attribution.py`:

```python
from werewolf_agent.evaluation.attribution import trace_outcome_is_bad, is_harmful
from werewolf_agent.evaluation.feedback_schemas import DecisionOutcome, EvaluationTrace


def _trace(action_type="vote", target_id="p03", faction="good", outcome=None, decision=None):
    return EvaluationTrace(
        trace_id="t1", game_id="g1", player_id="p01", role="villager",
        faction=faction, phase="day_vote", day_number=1,
        decision=decision or DecisionSnapshot(action_type=action_type, target_id=target_id),
        outcome=outcome,
    )


def test_outcome_is_bad_legal_false():
    t = _trace(outcome=DecisionOutcome(legal=False))
    assert trace_outcome_is_bad(t) is True


def test_outcome_is_bad_leaked():
    t = _trace(outcome=DecisionOutcome(leaked_hidden_info=True))
    assert trace_outcome_is_bad(t) is True


def test_outcome_is_bad_good_voter_missed_wolf():
    t = _trace(
        action_type="vote", target_id="p07", faction="good",
        outcome=DecisionOutcome(target_faction="good", vote_hit_wolf=False),
    )
    assert trace_outcome_is_bad(t) is True


def test_outcome_is_bad_wrong_target_good_hits_good():
    t = _trace(
        action_type="use_poison", target_id="p07", faction="good",
        outcome=DecisionOutcome(target_faction="good", vote_hit_wolf=None),
    )
    assert trace_outcome_is_bad(t) is True


def test_outcome_is_bad_correct_good_vote_is_false():
    t = _trace(
        action_type="vote", target_id="p03", faction="good",
        outcome=DecisionOutcome(target_faction="werewolf", vote_hit_wolf=True),
    )
    assert trace_outcome_is_bad(t) is False


def test_is_harmful_requires_cited_aligned_bad():
    exposure = ModuleExposure(module="possible_worlds", item_id="W",
                              metadata={"key_assignments": {"p03": "werewolf"}},
                              cited_by_decision=True, aligned_with_decision=True)
    t = _trace(action_type="vote", target_id="p03", faction="good",
              outcome=DecisionOutcome(target_faction="good", vote_hit_wolf=False))
    # p03 is wolf in world but actual good → bad outcome; cited+aligned → harmful
    assert is_harmful(exposure, t) is True


def test_is_harmful_false_when_not_cited():
    exposure = ModuleExposure(module="possible_worlds", item_id="W",
                              metadata={"key_assignments": {"p03": "werewolf"}},
                              cited_by_decision=False, aligned_with_decision=True)
    t = _trace(outcome=DecisionOutcome(legal=False))
    assert is_harmful(exposure, t) is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/evaluation/test_attribution.py -q -o addopts="" -k "outcome_is_bad or is_harmful"`
Expected: FAIL — names not exported.

- [ ] **Step 3: Implement**

Append to `werewolf_agent/evaluation/attribution.py`:

```python
_HARMFUL_ACTION_TYPES = frozenset({"vote", "use_poison", "hunter_shot", "sheriff_vote"})


def trace_outcome_is_bad(trace) -> bool:
    """Did this trace's decision produce a bad outcome? Uses the signals
    revived by monitoring-closure-fix (legal/leak) plus trace-level vote/
    wrong-target checks that need faction + action_type."""
    outcome = trace.outcome
    if outcome is None:
        return False
    if outcome.legal is False:
        return True
    if outcome.leaked_hidden_info:
        return True
    decision = trace.decision
    action_type = decision.action_type if decision else ""
    if trace.faction == "good" and action_type == "vote":
        if outcome.vote_hit_wolf is False:
            return True
    if action_type in _HARMFUL_ACTION_TYPES and trace.faction == "good":
        if outcome.target_faction == "good":
            return True
    return False


def is_harmful(exposure: ModuleExposure, trace) -> bool:
    return bool(
        exposure.cited_by_decision
        and exposure.aligned_with_decision
        and trace_outcome_is_bad(trace)
    )


def is_beneficial(exposure: ModuleExposure, trace) -> bool:
    return bool(
        exposure.cited_by_decision
        and exposure.aligned_with_decision
        and not trace_outcome_is_bad(trace)
    )
```

- [ ] **Step 4: Run the harmful tests**

Run: `python -m pytest tests/evaluation/test_attribution.py -q -o addopts="" -k "outcome_is_bad or is_harmful"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/evaluation/attribution.py tests/evaluation/test_attribution.py
git commit -m "feat: attribution harmful and beneficial outcome signals"
```

---

## Task 6: judge producer — rebuild public_facts + public_claim + scored sentinel

**Why:** spec: per trace, rebuild `public_facts` from the `result.event_log` prefix (filter by payload/top-level `day_number` and same-day phase scope, convert dicts to `GameEvent`, build a temporary `GameState`, run `build_world_state`, then `VisibilityPolicy.filter_visible_facts`), derive `public_claim` from the player's prior public role claims, run `judge_speech_consistency`, and store `consistency_score`. Because `local_quality_score` defaults to 0.0 and a judged trace can legitimately score 0.0, store an explicit sentinel (`"judge_consistency_scored"` appended to `outcome.outcome_refs`) — do NOT filter by `> 0`.

**Files:**
- Modify: `werewolf_agent/evaluation/attribution.py` (add `rebuild_visible_facts`, `derive_public_claim`, `judge_trace`)
- Test: `tests/evaluation/test_attribution.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/evaluation/test_attribution.py`:

```python
from werewolf_agent.evaluation.attribution import (
    derive_public_claim,
    rebuild_visible_facts,
    judge_trace,
)
from werewolf_agent.evaluation.schemas import GameResult


def _result_with_claim(event_log, player_roles=None, player_factions=None):
    return GameResult(
        game_id="g1", initial_seed=0, ruleset_id="pre_witch_hunter_idiot_mixed",
        event_log=event_log,
        player_roles=player_roles or {"p01": "villager", "p03": "werewolf"},
        player_factions=player_factions or {"p01": "good", "p03": "werewolf"},
    )


def test_rebuild_visible_facts_uses_payload_day_and_excludes_future_events():
    event_log = [
        {"type": "speech", "payload": {
            "speaker": "p03", "text": "我是预言家", "day_number": 1, "phase": "speech",
        }},
        {"type": "speech", "payload": {
            "speaker": "p04", "text": "我是预言家", "day_number": 2, "phase": "speech",
        }},
    ]
    result = _result_with_claim(
        event_log,
        player_roles={"p01": "villager", "p03": "werewolf", "p04": "seer"},
        player_factions={"p01": "good", "p03": "werewolf", "p04": "good"},
    )
    trace = EvaluationTrace(
        trace_id="t", game_id="g1", player_id="p01", role="villager",
        faction="good", phase="speech", day_number=1,
        decision=DecisionSnapshot(action_type="speech", reason="x"),
        outcome=DecisionOutcome(),
    )
    facts = rebuild_visible_facts(result, trace)
    assert any(f.fact_type == "speech" and f.source_player == "p03" and f.day == 1 for f in facts)
    assert not any(f.source_player == "p04" for f in facts)


def test_derive_public_claim_reads_prior_self_role_claim():
    event_log = [
        {"type": "speech", "payload": {
            "speaker": "p01", "text": "我是预言家", "day_number": 1, "phase": "speech",
        }},
    ]
    result = _result_with_claim(event_log)
    trace = EvaluationTrace(
        trace_id="t", game_id="g1", player_id="p01", role="villager",
        faction="good", phase="speech", day_number=2,
        decision=DecisionSnapshot(action_type="speech", reason="继续警徽流"),
        outcome=DecisionOutcome(),
    )
    assert derive_public_claim(result, trace) == "seer"


def test_judge_trace_sets_score_and_sentinel():
    event_log = [
        {"type": "speech", "payload": {
            "speaker": "p01", "text": "我是狼人", "day_number": 1, "phase": "speech",
        }},
    ]
    result = _result_with_claim(
        event_log,
        player_roles={"p01": "werewolf", "p03": "villager"},
        player_factions={"p01": "werewolf", "p03": "good"},
    )
    trace = EvaluationTrace(
        trace_id="t", game_id="g1", player_id="p01", role="werewolf",
        faction="werewolf", phase="speech", day_number=2,
        decision=DecisionSnapshot(action_type="speech", reason="继续公开认狼", raw={"speech": "我是狼人"}),
        outcome=DecisionOutcome(),
    )
    judged = judge_trace(trace, result)
    assert "judge_consistency_scored" in judged.outcome.outcome_refs
    assert judged.outcome.local_quality_score == 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/evaluation/test_attribution.py -q -o addopts="" -k "rebuild_visible_facts or derive_public_claim or judge_trace"`
Expected: FAIL — names not exported.

- [ ] **Step 3: Implement rebuild_visible_facts + derive_public_claim + judge_trace**

Append to `werewolf_agent/evaluation/attribution.py`:

```python
import dataclasses

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.cognition.world_state import build_world_state
from werewolf_agent.cognition.visibility import VisibilityPolicy
from werewolf_agent.evaluation.feedback_schemas import DecisionOutcome
from werewolf_agent.evaluation.llm_judge import judge_speech_consistency

_JUDGE_SENTINEL = "judge_consistency_scored"
_PHASE_ORDER = {
    "setup": 0,
    "night": 10,
    "wolf": 11,
    "witch": 12,
    "seer": 13,
    "sheriff": 20,
    "sheriff_speech": 21,
    "speech": 30,
    "day": 35,
    "day_vote": 40,
    "vote": 40,
}
_ROLE_CLAIMS = {
    "werewolf": ("我是狼人", "我是狼", "我们狼队", "狼队视角"),
    "seer": ("我是预言家", "我跳预言家", "认预言家"),
    "witch": ("我是女巫", "我认女巫"),
    "hunter": ("我是猎人", "我认猎人"),
    "villager": ("我是村民", "我是民", "我认民"),
}


def _event_payload(entry: dict) -> dict:
    payload = entry.get("payload")
    return dict(payload) if isinstance(payload, dict) else {}


def _event_day(entry: dict) -> int:
    payload = _event_payload(entry)
    value = entry.get("day_number", payload.get("day_number", 0))
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _event_phase(entry: dict) -> str:
    payload = _event_payload(entry)
    return str(entry.get("phase") or payload.get("phase") or "")


def _phase_rank(phase: str) -> int:
    return _PHASE_ORDER.get(str(phase or ""), 999)


def _entry_is_in_trace_prefix(entry: dict, trace) -> bool:
    day = _event_day(entry)
    trace_day = trace.day_number or 0
    if day > trace_day:
        return False
    if day == trace_day and _phase_rank(_event_phase(entry)) > _phase_rank(trace.phase):
        return False
    return True


def _game_event_from_entry(entry: dict) -> GameEvent:
    payload = _event_payload(entry)
    if "day_number" not in payload and entry.get("day_number") is not None:
        payload["day_number"] = entry.get("day_number")
    if "phase" not in payload and entry.get("phase"):
        payload["phase"] = entry.get("phase")
    if "speaker" not in payload and payload.get("player_id"):
        payload["speaker"] = payload["player_id"]
    return GameEvent(type=str(entry.get("type") or ""), payload=payload)


def rebuild_visible_facts(result, trace):
    """Rebuild the public facts visible to ``trace``'s player at decision time.

    Filter result.event_log to the decision prefix using payload/top-level
    day_number and same-day phase rank, convert dicts to GameEvent, build a
    temporary GameState with concrete PlayerState objects, run
    build_world_state, then filter to what this player could see.
    """
    events = []
    for entry in result.event_log:
        if not isinstance(entry, dict):
            continue
        if not _entry_is_in_trace_prefix(entry, trace):
            continue
        events.append(_game_event_from_entry(entry))
    players = {
        pid: PlayerState(id=pid, role=role, faction=result.player_factions.get(pid))
        for pid, role in result.player_roles.items()
    }
    state = GameState(
        game_id=result.game_id,
        ruleset_id=result.ruleset_id,
        players=players,
        phase=trace.phase,
        day_number=trace.day_number,
        night_number=trace.night_number,
        events=events,
    )
    world_state = build_world_state(state)
    return VisibilityPolicy().filter_visible_facts(world_state, trace.player_id, trace.role)


def derive_public_claim(result, trace) -> str:
    """Return the player's latest prior public role claim, if one exists."""
    latest = ""
    for entry in result.event_log:
        if not isinstance(entry, dict):
            continue
        if not _entry_is_in_trace_prefix(entry, trace):
            continue
        if str(entry.get("type") or "") not in {"speech", "sheriff_speech"}:
            continue
        payload = _event_payload(entry)
        speaker = str(payload.get("speaker") or payload.get("player_id") or "")
        if speaker != trace.player_id:
            continue
        for claim in payload.get("claims", []) or []:
            if isinstance(claim, dict) and claim.get("type") == "role" and claim.get("value"):
                latest = str(claim["value"]).lower()
        text = str(payload.get("text") or payload.get("speech") or "")
        for role, markers in _ROLE_CLAIMS.items():
            if any(marker in text for marker in markers):
                latest = role
    return latest


def judge_trace(trace, result):
    """Run the consistency judge on a trace with rebuilt public_facts.

    Returns a NEW EvaluationTrace whose outcome carries the consistency score
    and the ``judge_consistency_scored`` sentinel in outcome_refs. Traces with
    no speech/reason are returned unchanged (not judged).
    """
    speech = speech_from_decision(trace.decision)
    reason = trace.decision.reason if trace.decision else ""
    if not (speech.strip() or (reason or "").strip()):
        return trace
    visible_facts = rebuild_visible_facts(result, trace)
    context = {
        "role": trace.role,
        "faction": trace.faction,
        "public_claim": derive_public_claim(result, trace),
        "public_facts": visible_facts,
        "visible_facts": visible_facts,
    }
    action = {"speech": speech, "reason": reason}
    judgment = judge_speech_consistency(context, action)
    old_outcome = trace.outcome or DecisionOutcome()
    new_refs = list((old_outcome.outcome_refs if old_outcome else []) or [])
    if _JUDGE_SENTINEL not in new_refs:
        new_refs.append(_JUDGE_SENTINEL)
    new_outcome = dataclasses.replace(
        old_outcome,
        local_quality_score=judgment.consistency_score,
        outcome_refs=new_refs,
    )
    return dataclasses.replace(trace, outcome=new_outcome)
```

- [ ] **Step 4: Run the judge tests**

Run: `python -m pytest tests/evaluation/test_attribution.py -q -o addopts="" -k "rebuild_visible_facts or derive_public_claim or judge_trace"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/evaluation/attribution.py tests/evaluation/test_attribution.py
git commit -m "feat: attribution judge producer with rebuilt public_facts and scored sentinel"
```

---

## Task 7: AttributionEngine.annotate — integrate with frozen replace

**Why:** spec: `annotate(traces, result)` iterates traces, for each cognition exposure rebuilds it with `cited_by_decision` / `aligned_with_decision` / metadata flags (harmful/beneficial/attribution_missing_text/support=UNSUPPORTED), rebuilds the trace with the new exposures + judged outcome. Returns the new list (frozen dataclasses — callers must use the return value).

**Files:**
- Modify: `werewolf_agent/evaluation/attribution.py` (add `AttributionEngine`)
- Test: `tests/evaluation/test_attribution.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/evaluation/test_attribution.py`:

```python
from werewolf_agent.evaluation.attribution import AttributionEngine
from werewolf_agent.evaluation.feedback_schemas import MetricSupport


def test_annotate_marks_unresolved_rag_exposure_unsupported():
    resolver = AttributionTextResolver()  # no entries → r1 unresolved
    exposure = ModuleExposure(module="rag", item_id="r1", prompt_visible=True)
    trace = EvaluationTrace(
        trace_id="t", game_id="g1", player_id="p01", role="villager",
        faction="good", phase="speech", day_number=1,
        decision=DecisionSnapshot(action_type="speech", reason="核验警徽流", raw={"speech": "核验"}),
        outcome=DecisionOutcome(),
        module_exposures=[exposure],
    )
    result = _result_with_claim([])
    out = AttributionEngine(resolver).annotate([trace], result)
    exp = out[0].module_exposures[0]
    assert exp.support == MetricSupport.UNSUPPORTED
    assert exp.metadata.get("attribution_missing_text") is True


def test_annotate_sets_cited_aligned_harmful_on_matching_exposure():
    resolver = AttributionTextResolver(rag_entries={"r1": {
        "title": "核验警徽流", "recommended_action": "核验",
    }})
    exposure = ModuleExposure(module="rag", item_id="r1", prompt_visible=True)
    # bad outcome + cited + aligned → harmful
    trace = EvaluationTrace(
        trace_id="t", game_id="g1", player_id="p01", role="villager",
        faction="good", phase="day_vote", day_number=1,
        decision=DecisionSnapshot(action_type="vote", target_id="p07", reason="核验后投p07"),
        outcome=DecisionOutcome(target_faction="good", vote_hit_wolf=False),
        module_exposures=[exposure],
    )
    result = _result_with_claim(
        [],
        player_roles={"p01": "villager", "p07": "villager"},
        player_factions={"p01": "good", "p07": "good"},
    )
    out = AttributionEngine(resolver).annotate([trace], result)
    exp = out[0].module_exposures[0]
    assert exp.cited_by_decision is True
    assert exp.aligned_with_decision is True
    assert exp.metadata.get("harmful_transfer") is True
    assert "judge_consistency_scored" in out[0].outcome.outcome_refs


def test_annotate_returns_new_traces_does_not_mutate_input():
    resolver = AttributionTextResolver()
    trace = EvaluationTrace(
        trace_id="t", game_id="g1", player_id="p01", role="villager",
        faction="good", phase="speech", day_number=1,
        decision=DecisionSnapshot(action_type="speech", reason="x", raw={}),
        outcome=DecisionOutcome(),
        module_exposures=[],
    )
    original_exposures = list(trace.module_exposures)
    AttributionEngine(resolver).annotate([trace], _result_with_claim([]))
    # input trace is frozen/unchanged
    assert trace.module_exposures == original_exposures
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/evaluation/test_attribution.py -q -o addopts="" -k annotate`
Expected: FAIL — `AttributionEngine` not defined (only `AttributionTextResolver` exists so far).

- [ ] **Step 3: Implement AttributionEngine**

Append to `werewolf_agent/evaluation/attribution.py`:

```python
_COGNITION_MODULES = frozenset({"rag", "reflection", "possible_worlds", "simulator"})


class AttributionEngine:
    """Post-game attribution pass. Rebuilds frozen traces with cited/aligned/
    harmful annotations and judged outcome scores."""

    def __init__(self, text_resolver: AttributionTextResolver | None = None) -> None:
        self._resolver = text_resolver or AttributionTextResolver()

    def annotate(self, traces, result):
        out = []
        for trace in traces:
            trace = judge_trace(trace, result)
            new_exposures = []
            for exposure in trace.module_exposures:
                if exposure.module not in _COGNITION_MODULES:
                    new_exposures.append(exposure)
                    continue
                new_exposures.append(self._annotate_exposure(exposure, trace))
            out.append(dataclasses.replace(trace, module_exposures=new_exposures))
        return out

    def _annotate_exposure(self, exposure, trace):
        exp_text = exposure_representative_text(exposure, self._resolver)
        is_rag_reflection = exposure.module in ("rag", "reflection")
        if is_rag_reflection and not exp_text:
            # unresolved → UNSUPPORTED, excluded from harmful denominator
            return dataclasses.replace(
                exposure,
                support=MetricSupport.UNSUPPORTED,
                metadata={**exposure.metadata, "attribution_missing_text": True},
            )
        cited_flag = cited(trace.decision, exposure, self._resolver)
        aligned_flag = aligned(trace.decision, exposure, trace.faction, self._resolver)
        meta = dict(exposure.metadata)
        if cited_flag and aligned_flag:
            if trace_outcome_is_bad(trace):
                meta["harmful_transfer"] = True
            else:
                meta["beneficial"] = True
        return dataclasses.replace(
            exposure,
            cited_by_decision=cited_flag,
            aligned_with_decision=aligned_flag,
            metadata=meta,
        )
```

(Add `from werewolf_agent.evaluation.feedback_schemas import MetricSupport` to the imports at the top of the file alongside the existing `ModuleExposure` import.)

- [ ] **Step 4: Run annotate tests**

Run: `python -m pytest tests/evaluation/test_attribution.py -q -o addopts=""`
Expected: PASS (all attribution tests).

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/evaluation/attribution.py tests/evaluation/test_attribution.py
git commit -m "feat: AttributionEngine.annotate integrates cited/aligned/harmful/judge"
```

---

## Task 8: harmful_rate + mean_consistency metric helpers

**Why:** spec: `_harmful_rate` counts harmful over supported cognition exposures (UNSUPPORTED excluded from denominator); `_mean_consistency` averages `local_quality_score` over traces carrying the `judge_consistency_scored` sentinel (NOT filtering by `> 0`).

**Files:**
- Modify: `werewolf_agent/evaluation/attribution.py` (add `harmful_rate`, `mean_consistency`)
- Test: `tests/evaluation/test_attribution.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/evaluation/test_attribution.py`:

```python
from werewolf_agent.evaluation.attribution import harmful_rate, mean_consistency


def test_harmful_rate_excludes_unsupported_from_denominator():
    traces = [
        EvaluationTrace(
            trace_id="t1", game_id="g", player_id="p", role="villager",
            faction="good", phase="x",
            decision=DecisionSnapshot(action_type="vote"),
            outcome=DecisionOutcome(),
            module_exposures=[
                ModuleExposure(module="rag", item_id="a", support=MetricSupport.SUPPORTED,
                               metadata={"harmful_transfer": True}),
                ModuleExposure(module="rag", item_id="b", support=MetricSupport.UNSUPPORTED,
                               metadata={"harmful_transfer": True}),
            ],
        ),
    ]
    # supported denominator = 1, harmful = 1 → 1.0
    assert harmful_rate(traces) == 1.0


def test_harmful_rate_zero_when_no_supported():
    assert harmful_rate([]) == 0.0


def test_mean_consistency_uses_sentinel_not_score_filter():
    traces = [
        EvaluationTrace(
            trace_id="t1", game_id="g", player_id="p", role="villager",
            faction="good", phase="speech",
            decision=DecisionSnapshot(action_type="speech", reason="x"),
            outcome=DecisionOutcome(local_quality_score=0.0,
                                    outcome_refs=["judge_consistency_scored"]),
        ),
        EvaluationTrace(
            trace_id="t2", game_id="g", player_id="p", role="villager",
            faction="good", phase="speech",
            decision=DecisionSnapshot(action_type="speech", reason="y"),
            outcome=DecisionOutcome(local_quality_score=0.8,
                                    outcome_refs=["judge_consistency_scored"]),
        ),
        # not judged — no sentinel, excluded
        EvaluationTrace(
            trace_id="t3", game_id="g", player_id="p", role="villager",
            faction="good", phase="speech",
            decision=DecisionSnapshot(action_type="speech", reason="z"),
            outcome=DecisionOutcome(local_quality_score=0.0, outcome_refs=[]),
        ),
    ]
    # (0.0 + 0.8) / 2 = 0.4
    assert mean_consistency(traces) == 0.4


def test_mean_consistency_none_when_no_judged_traces():
    traces = [EvaluationTrace(
        trace_id="t", game_id="g", player_id="p", role="villager",
        faction="good", phase="x",
        decision=DecisionSnapshot(action_type="vote"),
        outcome=DecisionOutcome(),
    )]
    assert mean_consistency(traces) is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/evaluation/test_attribution.py -q -o addopts="" -k "harmful_rate or mean_consistency"`
Expected: FAIL — names not exported.

- [ ] **Step 3: Implement**

Append to `werewolf_agent/evaluation/attribution.py`:

```python
def harmful_rate(traces) -> float:
    supported = 0
    harmful = 0
    for trace in traces:
        for exposure in trace.module_exposures:
            if exposure.module not in _COGNITION_MODULES:
                continue
            if exposure.support != MetricSupport.SUPPORTED:
                continue
            supported += 1
            if exposure.metadata.get("harmful_transfer") is True:
                harmful += 1
    return harmful / supported if supported else 0.0


def mean_consistency(traces) -> float | None:
    scores = []
    for trace in traces:
        outcome = trace.outcome
        if outcome is None or trace.decision is None:
            continue
        if _JUDGE_SENTINEL not in (outcome.outcome_refs or []):
            continue
        if not (speech_from_decision(trace.decision).strip() or (trace.decision.reason or "").strip()):
            continue
        scores.append(outcome.local_quality_score)
    return sum(scores) / len(scores) if scores else None
```

- [ ] **Step 4: Run the metric tests**

Run: `python -m pytest tests/evaluation/test_attribution.py -q -o addopts="" -k "harmful_rate or mean_consistency"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/evaluation/attribution.py tests/evaluation/test_attribution.py
git commit -m "feat: attribution harmful_rate and mean_consistency metric helpers"
```

---

## Task 9: FullGameAblationRunner — resolver param + _enriched_metrics

**Why:** spec: `FullGameAblationRunner.__init__` gains `attribution_text_resolver`. `_enriched_metrics(result, resolver)` emits `judge_consistency_rate` always (when traces can be judged) and `harmful_transfer_rate` only when a resolver is present (else records `unsupported_metrics["attribution"]`). `run` must merge per-result unsupported reasons into `FullGameAblationReport.unsupported_metrics`; otherwise the resolver-missing signal is silently lost. Replay path (sparse `GameResult`) omits both.

**Files:**
- Modify: `werewolf_agent/evaluation/full_game_ablation.py`
- Test: `tests/evaluation/test_full_game_ablation.py`

- [ ] **Step 1: Read the current runner**

Read `werewolf_agent/evaluation/full_game_ablation.py` `__init__` (lines 67-75), `run` (98-129), `_run_game` (131-149). The existing `_game_metrics(result)` (242-249) stays; you add `_enriched_metrics` that calls it then layers the two new keys.

- [ ] **Step 2: Write the failing test**

Append to `tests/evaluation/test_full_game_ablation.py` (reuse the file's existing `ActionRecord`/`GameResult` import pattern and the `_make_result_with_votes` helper from monitoring-closure-fix Task 2; if a simpler fixture is easier, construct a `GameResult` directly):

```python
def test_enriched_metrics_emits_both_keys_with_resolver():
    from werewolf_agent.evaluation.attribution import AttributionTextResolver
    from werewolf_agent.evaluation.full_game_ablation import _enriched_metrics
    result = _make_result_with_votes(
        player_factions={"p01": "good", "p02": "werewolf"},
        votes=[("p01", "p02")],
        winning_faction="good",
    )
    # add an action_trace_audit so traces are non-empty
    result.event_log.append({
        "type": "action_trace_audit",
        "day_number": 1,
        "payload": {
            "player_id": "p01", "phase": "day_vote", "day_number": 1,
            "action_trace": {
                "final_action_type": "vote", "legal_actions": ["vote"],
                "legal_targets": ["p02"],
                "parsed_action": {"target_id": "p02", "reason": "投p02", "speech": ""},
            },
        },
    })
    metrics, unsupported = _enriched_metrics(result, AttributionTextResolver())
    assert "judge_consistency_rate" in metrics
    assert "harmful_transfer_rate" in metrics


def test_enriched_metrics_omits_harmful_without_resolver():
    from werewolf_agent.evaluation.full_game_ablation import _enriched_metrics
    result = _make_result_with_votes(
        player_factions={"p01": "good", "p02": "werewolf"},
        votes=[("p01", "p02")],
        winning_faction="good",
    )
    result.event_log.append({
        "type": "action_trace_audit",
        "payload": {
            "player_id": "p01", "phase": "day_vote", "day_number": 1,
            "action_trace": {
                "final_action_type": "vote",
                "parsed_action": {"target_id": "p02", "reason": "投p02"},
            },
        },
    })
    metrics, unsupported = _enriched_metrics(result, None)
    assert "harmful_transfer_rate" not in metrics
    assert unsupported.get("attribution") == "text_resolver_required"
    # judge_consistency_rate may or may not be present depending on traces; not asserted here


def test_enriched_metrics_replay_omits_both():
    from werewolf_agent.evaluation.full_game_ablation import _enriched_metrics
    from werewolf_agent.evaluation.schemas import GameResult
    # sparse replay GameResult: no action_records, empty-ish event_log
    result = GameResult(
        game_id="g", initial_seed=0, ruleset_id="replay",
        event_log=[], winning_faction="good",
    )
    metrics, unsupported = _enriched_metrics(result, None)
    # replay-path sparse GameResult → no action_trace_audit → no traces → both keys omitted
    assert "harmful_transfer_rate" not in metrics
    assert "judge_consistency_rate" not in metrics


def test_runner_reports_attribution_unsupported_when_resolver_missing():
    from werewolf_agent.evaluation.full_game_ablation import FullGameAblationRunner

    def fake_runner(**kwargs):
        result = _make_result_with_votes(
            player_factions={"p01": "good", "p02": "werewolf"},
            votes=[("p01", "p02")],
            winning_faction="good",
        )
        result.event_log.append({
            "type": "action_trace_audit",
            "payload": {
                "player_id": "p01", "phase": "day_vote", "day_number": 1,
                "action_trace": {
                    "final_action_type": "vote",
                    "parsed_action": {"target_id": "p02", "reason": "投p02"},
                },
            },
        })
        return result

    report = FullGameAblationRunner(game_runner_factory=fake_runner).run(_config(seed_set=[1]))
    assert report.unsupported_metrics["attribution"] == "text_resolver_required"
```

(Adjust `_make_result_with_votes` usage to match the helper that monitoring-closure-fix Task 2 added to this file. If the helper isn't there, construct `GameResult` with `action_records=[ActionRecord(...)]` directly per the existing file pattern.)

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest tests/evaluation/test_full_game_ablation.py -q -o addopts="" -k enriched_metrics`
Expected: FAIL — `_enriched_metrics` not defined; after `_enriched_metrics` exists but before report merging, the runner unsupported test still fails.

- [ ] **Step 4: Implement _enriched_metrics + resolver param**

In `werewolf_agent/evaluation/full_game_ablation.py`:

Add import at top:
```python
from werewolf_agent.evaluation.attribution import (
    AttributionEngine,
    AttributionTextResolver,
    harmful_rate,
    mean_consistency,
)
from werewolf_agent.evaluation.trace_builder import EvaluationTraceBuilder
```

Add the module-level helper (near `_game_metrics`):
```python
def _enriched_metrics(result: GameResult, text_resolver):
    """Existing _game_metrics + harmful_transfer_rate + judge_consistency_rate.

    Returns (metrics, unsupported). harmful_transfer_rate requires a resolver;
    without one it is omitted and unsupported['attribution'] is set so the
    gate's required_metrics fail-closed governs. Replay-path sparse GameResult
    (no action_trace_audit) yields no traces → both keys omitted.
    """
    metrics = _game_metrics(result)
    unsupported: dict[str, str] = {}
    traces = EvaluationTraceBuilder().build(result)
    if not traces:
        return metrics, unsupported
    traces = AttributionEngine(text_resolver).annotate(traces, result)
    consistency = mean_consistency(traces)
    if consistency is not None:
        metrics["judge_consistency_rate"] = consistency
    if text_resolver is not None:
        metrics["harmful_transfer_rate"] = harmful_rate(traces)
    else:
        unsupported["attribution"] = "text_resolver_required"
    return metrics, unsupported


def _merge_unsupported_metrics(target: dict[str, str], *sources: dict[str, str]) -> None:
    """Merge unsupported reasons from baseline/ablated result enrichment."""
    for source in sources:
        for key, reason in source.items():
            target.setdefault(key, reason)
```

Update `FullGameAblationRunner.__init__` to accept and store the resolver:
```python
    def __init__(
        self,
        game_runner_factory: Callable[..., GameResult] | None = None,
        *,
        replay_artifact: ReplayArtifact | None = None,
        attribution_text_resolver: AttributionTextResolver | None = None,
    ) -> None:
        self._game_runner_factory = game_runner_factory
        self._replay_artifact = replay_artifact
        self._attribution_text_resolver = attribution_text_resolver
```

Update the pair construction in `run` (lines 112-118) and `_run_replay` (184-190) to use `_enriched_metrics`, and merge unsupported reasons into the report.

Immediately before the seed loop, add:

```python
        unsupported_metrics: dict[str, str] = {}
```

After each baseline and ablated `GameResult` pair is available, replace direct `_game_metrics(...)` calls with:

```python
            baseline_metrics, baseline_unsupported = _enriched_metrics(
                baseline,
                self._attribution_text_resolver,
            )
            ablated_metrics, ablated_unsupported = _enriched_metrics(
                ablated,
                self._attribution_text_resolver,
            )
            _merge_unsupported_metrics(
                unsupported_metrics,
                baseline_unsupported,
                ablated_unsupported,
            )
```

Then pass `unsupported_metrics=unsupported_metrics` into the returned `FullGameAblationReport`.

(For `_run_replay`, the replay path's sparse GameResult yields no traces → both keys omitted and no attribution unsupported reason, which the test asserts. Keep `_metric_deltas` unchanged — it already aggregates whatever keys are present.)

- [ ] **Step 5: Run the ablation test file**

Run: `python -m pytest tests/evaluation/test_full_game_ablation.py -q -o addopts=""`
Expected: PASS (new + existing).

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/evaluation/full_game_ablation.py tests/evaluation/test_full_game_ablation.py
git commit -m "feat: full-game ablation emits harmful_transfer_rate and judge_consistency_rate"
```

---

## Task 10: Regression gate required-metrics hardening + producer wiring test

**Why:** spec: prove the monitoring-closure-fix `required_metrics` fail-closed now has real producers behind it. An ablation pair carrying the two metrics, with `required_metrics=("judge_consistency_rate","harmful_transfer_rate")`, must pass when present and fail-closed when absent. The current gate only fails when both baseline and candidate lack a required metric; this task tightens it so one-sided producer loss also fails closed.

**Files:**
- Modify: `werewolf_agent/evaluation/regression_gate.py`
- Test: `tests/evaluation/test_regression_gate.py`

- [ ] **Step 1: Write the producer-wiring and fail-closed tests**

Append to `tests/evaluation/test_regression_gate.py`:

```python
def test_gate_accepts_metrics_emitted_by_enriched_producer():
    from werewolf_agent.evaluation.attribution import AttributionTextResolver
    from werewolf_agent.evaluation.full_game_ablation import _enriched_metrics
    from werewolf_agent.evaluation.regression_gate import (
        CandidateRegressionConfig, RegressionGate,
    )
    from werewolf_agent.evaluation.schemas import ActionRecord, GameResult

    result = GameResult(
        game_id="g", initial_seed=0, ruleset_id="pre_witch_hunter_idiot_mixed",
        player_roles={"p01": "villager", "p02": "werewolf"},
        player_factions={"p01": "good", "p02": "werewolf"},
        winning_faction="good",
        action_records=[ActionRecord(player_id="p01", action_type="vote", target_id="p02")],
        event_log=[{
            "type": "action_trace_audit",
            "payload": {
                "player_id": "p01", "phase": "day_vote", "day_number": 1,
                "action_trace": {
                    "final_action_type": "vote",
                    "parsed_action": {"target_id": "p02", "reason": "投p02", "speech": ""},
                },
            },
        }],
    )
    metrics, unsupported = _enriched_metrics(result, AttributionTextResolver())
    assert unsupported == {}
    assert {"judge_consistency_rate", "harmful_transfer_rate"} <= set(metrics)

    config = CandidateRegressionConfig(
        candidate_id="c1",
        required_metrics=("judge_consistency_rate", "harmful_transfer_rate"),
    )
    report = RegressionGate().evaluate(
        config,
        baseline_metrics=metrics,
        candidate_metrics=metrics,
        prompt_safe=True,
    )
    assert report.passed is True


def test_gate_fails_closed_when_producer_silently_absent():
    from werewolf_agent.evaluation.regression_gate import (
        CandidateRegressionConfig, RegressionGate,
    )
    config = CandidateRegressionConfig(
        candidate_id="c1",
        required_metrics=("judge_consistency_rate", "harmful_transfer_rate"),
    )
    report = RegressionGate().evaluate(
        config,
        baseline_metrics={"good_win_rate": 0.5},
        candidate_metrics={"good_win_rate": 0.5},
        prompt_safe=True,
    )
    assert report.passed is False
    reasons = " ".join(report.blocked_reasons)
    assert "judge_consistency_rate" in reasons
    assert "harmful_transfer_rate" in reasons


def test_gate_fails_closed_when_required_metric_missing_on_one_side():
    from werewolf_agent.evaluation.regression_gate import (
        CandidateRegressionConfig, RegressionGate,
    )
    config = CandidateRegressionConfig(
        candidate_id="c1",
        required_metrics=("judge_consistency_rate", "harmful_transfer_rate"),
    )
    report = RegressionGate().evaluate(
        config,
        baseline_metrics={
            "good_win_rate": 0.5,
            "judge_consistency_rate": 0.8,
            "harmful_transfer_rate": 0.1,
        },
        candidate_metrics={
            "good_win_rate": 0.5,
            "judge_consistency_rate": 0.8,
        },
        prompt_safe=True,
    )
    assert report.passed is False
    assert "required_metric_missing:harmful_transfer_rate:candidate" in report.blocked_reasons
```

- [ ] **Step 2: Run to verify the one-sided missing case fails**

Run: `python -m pytest tests/evaluation/test_regression_gate.py -q -o addopts="" -k "silently_absent or one_side or enriched_producer"`
Expected: FAIL before the gate hardening — the one-sided missing test passes incorrectly today because baseline has the key.

- [ ] **Step 3: Harden required_metrics**

In `werewolf_agent/evaluation/regression_gate.py`, replace the current `for required in config.required_metrics` block with:

```python
        for required in config.required_metrics:
            missing_sides = []
            if required not in baseline_metrics:
                missing_sides.append("baseline")
            if required not in candidate_metrics:
                missing_sides.append("candidate")
            if not missing_sides:
                continue
            checks.append(GateCheck(
                name=f"required_{required}",
                passed=False,
                reason=f"required_metric_missing:{required}:{','.join(missing_sides)}",
                metric=required,
            ))
```

- [ ] **Step 4: Run the gate tests**

Run: `python -m pytest tests/evaluation/test_regression_gate.py -q -o addopts="" -k "silently_absent or one_side or enriched_producer"`
Expected: PASS (exercises producer wiring, both-sides missing fail-closed, and one-sided producer loss fail-closed).

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/evaluation/regression_gate.py tests/evaluation/test_regression_gate.py
git commit -m "fix: fail closed when required metrics are missing on one side"
```

---

## Task 11: Whole-suite regression + PROGRESS

**Files:**
- Modify: `PROGRESS.md`

- [ ] **Step 1: Run evaluation + cognition + memory suites**

Run: `python -m pytest tests/evaluation tests/cognition tests/memory -q -o addopts="" 2>&1 | tail -5`
Expected: all PASS.

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest -q -o addopts="" 2>&1 | tail -5`
Expected: all PASS (no regressions; master was 3388 passed, 1 skipped before this branch; expect +N new tests).

- [ ] **Step 3: compile-check**

Run: `python -m compileall -q werewolf_agent`
Expected: no output.

- [ ] **Step 4: Update PROGRESS.md**

Add a new section at the top of `PROGRESS.md` (above `rag-role-hardening`). Update `Current Status` (phase → `attribution-engine`, last updated → `2026-06-19`, active task summarizing the engine). Record: the AttributionEngine (post-game cited/aligned/harmful + judge), AttributionTextResolver (RAG/reflection card resolution + UNSUPPORTED), the two metric producers wired into FullGameAblationRunner, the shared text_similarity helper, files changed, verification commands, and open risks (attribution is correlation not causation; `skill`/`persona` excluded; `beneficial` byproduct stored but not yet fed to reflection downrank; resolver needs store-backed wiring for production runs — tests use fixtures).

- [ ] **Step 5: Commit**

```bash
git add PROGRESS.md
git commit -m "docs: log attribution engine in PROGRESS"
```

---

## Open Risks / Out of Scope

- **Attribution is correlation, not causation** (spec `evaluation-feedback-loop-design.md:842`). `harmful = cited ∧ aligned ∧ bad` is high-precision but may miss half-followed cards; strong causal claims need the live-agent ablation harness (out of scope).
- **`skill` / `persona` excluded** (behavioural priors; ill-defined citation semantics).
- **`beneficial` byproduct stored** (`metadata["beneficial"]`) but not fed to `reflection.query_live` re-ranking (source-design future downrank). Tracked.
- **Resolver production wiring:** tests use fixture dicts; production needs a store-backed resolver (`RAGRepository.get` / `ReflectionMemory.all_v2_entries`). A follow-up wires the real resolver into `game_runner` / the evaluation pipeline.
- **`memory.reflection` now imports `evaluation.text_similarity`** (memory → evaluation edge). Verified non-cyclic (evaluation imports nothing from memory.reflection). If a future change makes evaluation import memory.reflection, this edge cycles — move `text_similarity` to a neutral `core/` module then.
- **Judge `public_claim` derived as `""`** when no public claim event is found pre-trace; the identity dimension then only fires on wolf self-identification. Acceptable; documented in spec.
- No runtime / schema / DB / rule-engine change.

## Self-Review

- **Spec coverage:** Goal 1 (cited/aligned/harmful + resolver + UNSUPPORTED) → Tasks 2-5, 7. Goal 2 (judge_consistency) → Task 6. Goal 3 (harmful_transfer_rate) → Tasks 8-9. Goal 4 (wire into runner) → Task 9. Goal 5 (zero runtime change) → respected throughout. Goal 6 (deterministic/unit-tested) → every task is TDD. spec Components (Resolver, Engine, cited, aligned, harmful, judge, metric producers) each map to a task. spec Testing bullets each map to test steps.
- **Completeness scan:** no unresolved marker text. Each code step shows actual code. Task 9 Step 1 references the `_make_result_with_votes` helper from monitoring-closure-fix Task 2 with a fallback instruction — verification, not a placeholder.
- **Type consistency:** `AttributionTextResolver` constructor used identically in Tasks 2/3/7/9. `AttributionEngine(text_resolver).annotate(traces, result)` signature consistent across Tasks 7/9. `_JUDGE_SENTINEL = "judge_consistency_scored"` defined in Task 6, used in Tasks 6/8. `_COGNITION_MODULES` defined in Task 7, used in Task 8. `MetricSupport.UNSUPPORTED` import added in Task 7 Step 3. `harmful_rate`/`mean_consistency` defined Task 8, imported Task 9.
