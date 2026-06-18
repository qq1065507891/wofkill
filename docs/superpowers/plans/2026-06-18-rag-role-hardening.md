# RAG Role Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the role/phase isolation gap on the default runtime RAG retrieval path (`vector_store=None`): today a villager live query can receive a werewolf-perspective case because `StrategyRetriever._filter_candidates` has NO role/phase hard filter (only the vector path `_passes_live_metadata_filter` does). Extract one shared `role_phase_matches` predicate so all three call sites (vector hard gate, vector fallback inline re-check, retriever hard filter) use the same wildcard convention, and add the missing hard filter to `_filter_candidates`.

**Architecture:** Two-file change in `werewolf_agent/rag/`. (1) Add a module-level `role_phase_matches(query, meta) -> bool` predicate in `retriever.py` encoding the role wildcard (`query.role`, `general`, `any`, `""`) and phase wildcard (`query.phase`, `general`, `""`) — the single source of truth currently duplicated across `_passes_live_metadata_filter` (knowledge_service.py:249-256) and `_vector_candidates` inline (knowledge_service.py:343-350). (2) `StrategyRetriever._filter_candidates` calls `role_phase_matches` as a HARD filter (currently it has none — only visibility/ruleset/quality/source/case_type), so the default `vector_store=None` runtime path gets role isolation. (3) `knowledge_service._passes_live_metadata_filter` and the `_vector_candidates` inline re-check both delegate to `role_phase_matches`, eliminating the three-way drift. No schema change; no rule-engine change; visibility boundary (GOD_VIEW exclusion) unchanged.

**Tech Stack:** Python 3.11, pytest, pydantic, conda env `wofkill`, PowerShell. Tests run with `-o addopts=""` (no `--basetemp`, to avoid the local pytest-xdist `.pytest_tmp` reuse permission issue — see `memory/project-pytest-addopts-bypass.md`).

---

## File Structure

- **Modify** `werewolf_agent/rag/retriever.py` — add module-level `role_phase_matches`; call it in `_filter_candidates` as a hard filter.
- **Modify** `werewolf_agent/rag/knowledge_service.py` — `_passes_live_metadata_filter` and the `_vector_candidates` inline role/phase re-check delegate to `role_phase_matches` (remove the duplicated logic).
- **Test** `tests/rag/test_knowledge_service.py` (extend) + `tests/rag/test_rag.py` or a new retriever test for the default-path isolation.

**Design boundaries respected:** No change to game rules, role abilities, rule engine, or RAG schema. RAG remains strategy-reference only (never answers base rules / adjudicates state). The fix STRENGTHENS role isolation (a villager no longer receives werewolf-only tactical frames on the default path) — it removes information, never adds. Visibility boundary (GOD_VIEW) handling unchanged. The `general`/`any` wildcard convention (基础常识 seed family) is preserved exactly — it becomes the single encoded truth rather than three copies.

---

## Task 1: Add shared role_phase_matches predicate + harden _filter_candidates

**Why:** `StrategyRetriever._filter_candidates` (`retriever.py:386-435`) filters on visibility/ruleset/quality/source/case_type but NOT on role/phase. On the default runtime path (`vector_store=None`, which is what `game_runner._build_default_rag_service` produces), `retrieve_live_hints` → `injector.inject` → `retriever.retrieve` → `_filter_candidates` runs with no role gate, then `_score` only gives a soft ±0.15 role bonus. So a `role_perspective="werewolf"` case can outrank role-correct entries and slip into a villager's live top-3. The vector path avoids this via `_passes_live_metadata_filter` (knowledge_service.py:249-257), but that gate is never reached when `vector_store=None`. The wildcard convention (`general`/`any`/`""`) is currently duplicated in three places (the two in knowledge_service + `_score` in retriever), which drifts easily.

**Files:**
- Modify: `werewolf_agent/rag/retriever.py` (new `role_phase_matches` + `_filter_candidates`)
- Test: `tests/rag/test_knowledge_service.py` (new default-path isolation test)

- [ ] **Step 1: Add the predicate**

In `werewolf_agent/rag/retriever.py`, add a module-level function near the top (after the imports/logger, before the class or near other module helpers). It encodes the role wildcard (`query.role`, `"general"`, `"any"`, `""`) and phase wildcard (`query.phase`, `"general"`, `""`):

```python
def role_phase_matches(query: RAGQuery, meta: Any) -> bool:
    """Hard role/phase gate shared by all RAG retrieval paths.

    Single source of truth for the wildcard convention:
    - role matches when ``meta.role_perspective`` equals ``query.role``,
      or is a universal marker (``"general"`` / ``"any"`` / empty), or
      when the query carries no role.
    - phase matches when ``meta.phase`` equals ``query.phase``, or is
      ``"general"`` / empty, or when the query carries no phase.

    Both must hold (AND semantics) so a cross-role case cannot leak in
    just because the phase happens to match. ``meta`` is a RAGMetadata
    (duck-typed: needs ``role_perspective`` and ``phase`` attrs).
    """
    role_ok = (
        not query.role
        or meta.role_perspective in (query.role, "general", "any", "")
    )
    phase_ok = (
        not query.phase
        or meta.phase in (query.phase, "general", "")
    )
    return role_ok and phase_ok
```

- [ ] **Step 2: Write the failing test (default-path isolation)**

The existing `test_role_phase_fallback_is_and_not_or` (`tests/rag/test_knowledge_service.py:314`) tests the VECTOR fallback path (`_vector_candidates` with an empty `LocalVectorStore`). It does NOT cover `vector_store=None`. Add a new test that constructs a `RAGKnowledgeService` with NO vector store and asserts the default path rejects cross-role cases. Append to `tests/rag/test_knowledge_service.py` (reuse the file's `_make_rag_entry` helper — read it first to confirm signature):

```python
def test_default_runtime_path_rejects_cross_role_case() -> None:
    """rag-role-hardening: with NO vector store (the default runtime path
    from game_runner._build_default_rag_service), retrieve_live_hints must
    still hard-filter by role/phase. Pre-fix, _filter_candidates had no
    role gate, so a werewolf-perspective case could leak into a villager
    query's live hits.
    """
    from werewolf_agent.rag.knowledge_service import RAGKnowledgeService

    wolf_speech = _make_rag_entry(
        entry_id="wolf_speech",
        role_perspective="werewolf",
        phase="speech",
    )
    villager_speech = _make_rag_entry(
        entry_id="villager_speech",
        role_perspective="villager",
        phase="speech",
    )
    # No vector_store → default runtime path (_filter_candidates direct).
    service = RAGKnowledgeService(
        seed_provider=lambda: [wolf_speech, villager_speech],
    )
    hits = service.retrieve_live_hints(
        RAGQuery(role="villager", phase="speech", max_results=5),
    )
    hit_ids = {h.entry_id for h in hits}
    assert "villager_speech" in hit_ids
    assert "wolf_speech" not in hit_ids, (
        f"rag-role-hardening: werewolf-perspective case leaked into a "
        f"villager live query on the default (no-vector) path; hits={hit_ids!r}"
    )
```

(Confirm `_make_rag_entry` accepts `entry_id`/`role_perspective`/`phase` kwargs by reading it. If its signature differs, adjust. The key: two entries, one werewolf-perspective + one villager-perspective, both phase=speech; a villager/speech query must return ONLY the villager one.)

- [ ] **Step 3: Run to verify it fails**

Run: `conda activate wofkill; python -m pytest tests/rag/test_knowledge_service.py::test_default_runtime_path_rejects_cross_role_case -q -o addopts=""`
Expected: FAIL — `wolf_speech` leaks into the villager query because `_filter_candidates` has no role hard filter (the wolf case may rank via case_type/quality soft score).

- [ ] **Step 4: Wire the hard filter into _filter_candidates**

In `werewolf_agent/rag/retriever.py` `_filter_candidates`, add the role/phase hard gate. After the existing `case_types` filter block (around line 432) and before `results.append(entry)` (line 434), insert:

```python
            # rag-role-hardening: role/phase hard gate so the default
            # runtime path (vector_store=None) keeps role isolation.
            # Without this, a werewolf-perspective case could leak into
            # a villager live query via soft score alone.
            if not role_phase_matches(query, meta):
                continue

            results.append(entry)
```

(`meta` is `entry.metadata`, already bound at the top of the loop body — line 390. `role_phase_matches` is the module-level function from Step 1.)

- [ ] **Step 5: Run the new test + the role/phase suite to verify pass + no regressions**

Run: `python -m pytest tests/rag/test_knowledge_service.py -q -o addopts="" -k "role or phase or default_runtime or cross_role or any_included or general_wildcard"`
Expected: PASS (the new test + existing `test_role_phase_fallback_is_and_not_or`, `test_role_phase_fallback_admits_general_wildcard`, `test_role_perspective_any_included_in_metadata_fallback`). The existing tests use `LocalVectorStore` (vector path) and are unaffected by the retriever change; the new test covers the default path.

Then run the full knowledge_service file: `python -m pytest tests/rag/test_knowledge_service.py -q -o addopts=""` — all PASS.

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/rag/retriever.py tests/rag/test_knowledge_service.py
git commit -m "fix: harden default-runtime rag path with role/phase hard filter"
```

---

## Task 2: Deduplicate — delegate knowledge_service filters to role_phase_matches

**Why:** The role/phase wildcard logic now lives in `retriever.role_phase_matches` (Task 1). `knowledge_service.py` still has TWO copies: `_passes_live_metadata_filter` (lines 249-257) and the inline re-check inside `_vector_candidates` (lines 343-350). They must stay in sync with the retriever predicate or the three paths drift. This task makes both delegate to `role_phase_matches`, so there is exactly one encoded convention.

**Files:**
- Modify: `werewolf_agent/rag/knowledge_service.py` (`_passes_live_metadata_filter`, `_vector_candidates` inline re-check)
- Test: `tests/rag/test_knowledge_service.py` (no new test — existing role/phase tests guard the delegation; optionally a regression assertion)

- [ ] **Step 1: Confirm the import path**

`knowledge_service.py` already imports from `werewolf_agent.rag.retriever` (read its imports — `StrategyRetriever` is imported there). Add `role_phase_matches` to that import. Read the top of `knowledge_service.py` to get the exact existing import line.

- [ ] **Step 2: Delegate _passes_live_metadata_filter**

In `werewolf_agent/rag/knowledge_service.py`, `_passes_live_metadata_filter` (lines 239-257) currently inlines the role/phase logic. Replace the inlined role_ok/phase_ok block with a call to the shared predicate. The method becomes:

```python
    def _passes_live_metadata_filter(self, query: RAGQuery, entry: RAGEntry) -> bool:
        """Hard metadata gate shared by vector hits and metadata fallback."""
        meta = entry.metadata
        if meta.visibility_boundary not in (
            VisibilityBoundary.PUBLIC_ONLY,
            VisibilityBoundary.PLAYER_PERSPECTIVE,
        ):
            return False
        if query.ruleset_id and meta.ruleset_id and meta.ruleset_id != query.ruleset_id:
            return False
        return role_phase_matches(query, meta)
```

(Add `role_phase_matches` to the `from werewolf_agent.rag.retriever import ...` line at the top of the file.)

- [ ] **Step 3: Delegate the _vector_candidates inline re-check**

In `_vector_candidates` (lines 322-352), the fallback loop has an inline `role_ok`/`phase_ok` block (343-351) duplicating the predicate. Replace the inline block with the shared predicate. The loop tail becomes:

```python
        for entry in entries:
            if not self._passes_live_metadata_filter(query, entry):
                continue
            meta = entry.metadata
            if query.ruleset_id and meta.ruleset_id and meta.ruleset_id != query.ruleset_id:
                continue
            # rag-role-hardening: role/phase gate delegated to the shared
            # predicate (single source of truth with _filter_candidates).
            if role_phase_matches(query, meta):
                selected.setdefault(entry.entry_id, (0.0, entry))
```

(Remove the now-redundant `role_ok`/`phase_ok` inline definitions. Note `_passes_live_metadata_filter` is already called at the top of this loop body — line 323 — which now itself calls `role_phase_matches`; the second explicit `role_phase_matches` call here is the original "R9 AND semantics" re-check preserved verbatim through the shared predicate. Keep both calls: the first is the full metadata gate [visibility+ruleset+role+phase], the second is the historical explicit role/phase assertion that the R9 comment documents. If you find the double-call redundant after reading, you may collapse to just `_passes_live_metadata_filter` — but only if the existing `test_role_phase_fallback_is_and_not_or` still passes, since it asserts the AND semantics on this exact path.)

- [ ] **Step 4: Run the role/phase + vector suites**

Run: `python -m pytest tests/rag/test_knowledge_service.py -q -o addopts=""`
Expected: all PASS (the delegation must not change behaviour — `test_role_phase_fallback_is_and_not_or`, `test_role_phase_fallback_admits_general_wildcard`, `test_role_perspective_any_included_in_metadata_fallback`, `test_vector_final_fallback_preserves_live_metadata_filter`, and Task 1's `test_default_runtime_path_rejects_cross_role_case` all green).

- [ ] **Step 5: Run the broader rag suite**

Run: `python -m pytest tests/rag -q -o addopts=""`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add werewolf_agent/rag/knowledge_service.py
git commit -m "refactor: delegate rag role/phase filters to shared predicate"
```

---

## Task 3: Whole-suite regression + PROGRESS

**Files:**
- Modify: `PROGRESS.md`

- [ ] **Step 1: Run rag + agents suites**

Run: `python -m pytest tests/rag tests/agents -q -o addopts="" 2>&1 | tail -5`
Expected: all PASS.

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest -q -o addopts="" 2>&1 | tail -5`
Expected: all PASS (no regressions; full suite was 3387 passed, 1 skipped on master before this branch; expect +1 new test).

- [ ] **Step 3: compile-check**

Run: `python -m compileall -q werewolf_agent`
Expected: no output.

- [ ] **Step 4: Update PROGRESS.md**

Add a new section at the top of `PROGRESS.md` (above `prompt-sanitizer-fix`). Update `Current Status` (phase → `rag-role-hardening`, last updated → `2026-06-18`, active task summarizing the fix). Record: the role/phase hard filter added to `_filter_candidates` (closes the default-runtime isolation gap), the shared `role_phase_matches` predicate (eliminates three-way drift), files changed, verification commands, and open risks (the fix is conservative — it only REMOVES cross-role cases from live queries, never adds content; no schema change; `general`/`any` wildcard preserved).

- [ ] **Step 5: Commit**

```bash
git add PROGRESS.md
git commit -m "docs: log rag role hardening in PROGRESS"
```

---

## Open Risks / Out of Scope

- **Behaviour change is conservative:** the fix only REMOVES cross-role cases from live queries (stricter). No live query gains new content. The only theoretical regression: a role-specific query that previously relied on a cross-role case slipping in (incorrectly) would now return fewer/different hits — but that was always a bug, not intended behaviour.
- **`_score` soft role bonus (retriever.py:471-475) is unchanged.** It still gives +0.15 for exact role / +0.05 for `general`/`any`. After the hard filter, only role-matching entries reach `_score`, so the soft bonus now ranks WITHIN the admitted set (correct). No change needed.
- **Wildcard convention frozen:** `general`/`any`/`""` for role, `general`/`""` for phase — now encoded once in `role_phase_matches`. Adding a new wildcard (e.g. `all`) is a one-line change in the predicate, propagating to all three paths automatically.
- **Vector path unaffected:** `_passes_live_metadata_filter` already had the gate; Task 2 only refactors it to delegate (no behaviour change). Existing vector-path tests guard this.
- No RAG schema change; no DB change; no rule-engine change.

## Self-Review

- **Spec/coverage:** The审查 P1-1 finding (default-runtime cross-role leak) → Task 1 hard filter + test. The三处漂移 finding → Task 2 delegation. Both map to tasks.
- **Placeholder scan:** No TBD/TODO. Each code step shows actual code. Task 2 Step 1 includes a "read the import line" verification — that's confirmation, not a placeholder.
- **Type consistency:** `role_phase_matches(query: RAGQuery, meta: Any) -> bool` — `meta` duck-typed as RAGMetadata (has `role_perspective`/`phase`). Used identically in `_filter_candidates` (Task 1 Step 4), `_passes_live_metadata_filter` (Task 2 Step 2), and `_vector_candidates` (Task 2 Step 3). Wildcard sets match exactly across all three original sites (`general`/`any`/`""` role; `general`/`""` phase).
- **Safety:** The fix only tightens role isolation (removes cross-role content from live prompts). RAG never held base rules or hidden truth (architecture-boundaries); visibility boundary (GOD_VIEW) unchanged. No leak introduced — the opposite.
