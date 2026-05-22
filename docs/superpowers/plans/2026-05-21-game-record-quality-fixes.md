# Game Record Quality Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the latest live-game defects found in `game_g_4056969886.json`: delayed hunter shot resolution, false public-memory claims, and excessive structured-output fallback.

**Architecture:** Keep deterministic rules inside `RuleEngine`, orchestration order inside `runtime.graph`, and prompt/context hardening inside `runtime.agent_adapter` plus `agents.player`. Add regression tests around exact event order and public fact grounding before changing implementation.

**Tech Stack:** Python 3.12, pytest, Pydantic, LangGraph runtime, existing `GameState`/`GameEvent` dataclasses.

---

## File Structure

- Modify `werewolf_agent/runtime/graph.py`: route and resolve post-exile hunter shots before entering the next night; ensure victory checks happen after the full death/skill chain.
- Modify `werewolf_agent/engine/rule_engine.py`: only if the delayed hunter-shot bug is caused by event reduction rather than graph routing.
- Modify `werewolf_agent/runtime/agent_adapter.py`: add public-fact grounding constraints and richer vote/speech context for claims like "X said Y".
- Modify `werewolf_agent/agents/player.py`: tighten retry hints and examples so malformed XML/tool wrappers and missing `action_type` recover faster.
- Modify `werewolf_agent/runtime/speech_quality.py` or create `werewolf_agent/runtime/fact_grounding.py`: validate quoted public claims against actual public transcript.
- Test `tests/runtime/test_game_runner.py` or `tests/runtime/test_runtime.py`: hunter-shot ordering regression.
- Test `tests/runtime/test_agent_timeline_context.py`: public transcript grounding and no fabricated "player claimed role" facts.
- Test `tests/agents/test_agents.py`: structured retry behavior and fallback trace accounting.

---

### Task 1: Reproduce The Delayed Hunter Shot Bug

**Files:**
- Test: `tests/runtime/test_runtime.py`
- Reference: `game_g_4056969886.json`

- [ ] **Step 1: Write a failing regression test**

Create a scenario where a hunter is exiled on day vote and chooses an alive werewolf target. Assert that the `hunter_shot` `player_died` event appears before any later `enter_night` or `wolf_kill_selected` event.

- [ ] **Step 2: Run the targeted test**

Run: `python -m pytest tests\runtime\test_runtime.py::TestHunterShotOrdering -q --basetemp=.pytest-tmp`

Expected before fix: FAIL because current record shows `enter_night` and `wolf_kill_selected` before hunter-shot death.

- [ ] **Step 3: Inspect current graph routing**

Review `resolve_exile`, `post_exile_skills`, `resolve_hunter_shot`, `route_after_post_exile`, `route_after_hunter_shot`, and `check_victory` in `werewolf_agent/runtime/graph.py`.

Expected finding: post-exile routing allows night transition before the hunter-shot chain is fully resolved or persisted.

---

### Task 2: Fix Hunter Shot Resolution Ordering

**Files:**
- Modify: `werewolf_agent/runtime/graph.py`
- Test: `tests/runtime/test_runtime.py`

- [ ] **Step 1: Update post-exile routing**

Make `route_after_post_exile` route to `resolve_hunter_shot` when the latest death has `triggered_skills` containing `hunter_shot`, before `check_victory` or `enter_night`.

- [ ] **Step 2: Ensure shot death is applied immediately**

In `resolve_hunter_shot`, apply the target death and append the `player_died` event in the same day resolution batch, then clear `hunter_shot_target_id`.

- [ ] **Step 3: Run focused tests**

Run: `python -m pytest tests\runtime\test_runtime.py::TestHunterShotOrdering tests\rules\test_rule_engine_v1.py -q --basetemp=.pytest-tmp`

Expected: hunter-shot ordering test passes; rule-engine tests remain green.

---

### Task 3: Add Public Fact Grounding For Agent Claims

**Files:**
- Create: `werewolf_agent/runtime/fact_grounding.py`
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Test: `tests/runtime/test_agent_timeline_context.py`

- [ ] **Step 1: Write failing tests for fabricated public claims**

Use a `GameState` transcript where p02 never says "我是狼人". Assert a helper rejects or flags the sentence `p02声称自己是狼人（Day 1公开记录）`.

- [ ] **Step 2: Implement a small grounding helper**

Add helpers that extract public speech text by day/player and check whether a quoted claim is supported by actual transcript text.

- [ ] **Step 3: Wire grounding into speech/vote context**

When building `strategy_directive`, add a hard instruction: claims about public speech must cite exact speaker/day evidence from `recent_transcript` or be phrased as suspicion, not fact.

- [ ] **Step 4: Run grounding tests**

Run: `python -m pytest tests\runtime\test_agent_timeline_context.py -q --basetemp=.pytest-tmp`

Expected: fabricated hard claims are flagged; legitimate quotes pass.

---

### Task 4: Reduce Repetitive Fallback Speech

**Files:**
- Modify: `werewolf_agent/agents/player.py`
- Modify: `werewolf_agent/runtime/speech_quality.py`
- Test: `tests/agents/test_agents.py`

- [ ] **Step 1: Add tests for fallback diversity**

Create cases where multiple players hit fallback in the same phase. Assert fallback speech includes different evidence anchors or player-specific context, not the repeated template `我先明确给出一个判断...`.

- [ ] **Step 2: Improve fallback speech generation**

Use `context.visible_world_state`, `recent_transcript`, and `legal_targets` to produce varied fallback speeches. Keep deterministic behavior for testability.

- [ ] **Step 3: Run focused tests**

Run: `python -m pytest tests\agents\test_agents.py::TestPlayerAgentFallback -q --basetemp=.pytest-tmp`

Expected: fallback remains legal and deterministic but no longer collapses into identical text across players.

---

### Task 5: Harden Structured Output Parsing And Retry Hints

**Files:**
- Modify: `werewolf_agent/agents/player.py`
- Test: `tests/agents/test_agents.py`

- [ ] **Step 1: Add parser tests for MiniMax-style wrappers**

Cover `<minimax:tool_call>`, `<invoke>`, JSON nested inside XML-ish parameters, missing `action_type`, and enum values in Chinese.

- [ ] **Step 2: Improve parsing conservatively**

Before JSON extraction, detect known tool wrapper formats and extract the `submit_player_action` parameters into a dict. Do not add broad free-text guessing.

- [ ] **Step 3: Improve retry hints**

When schema failure is missing `action_type` or invalid enum, set a correction hint that names the exact allowed values from `context.legal_actions`.

- [ ] **Step 4: Run parser tests**

Run: `python -m pytest tests\agents\test_agents.py::TestPlayerAgentParsing -q --basetemp=.pytest-tmp`

Expected: wrapper outputs parse; invalid outputs still fail safely.

---

### Task 6: Add A Replay Audit Test For The Latest Failure Pattern

**Files:**
- Modify: `tests/test_game_audit.py`
- Optional Modify: `scripts/print_game_audit.py`

- [ ] **Step 1: Write audit assertions**

Given a completed game event list, assert:
- No `wolf_discussion` or `wolf_kill_selected` by a player who has already died.
- Any hunter-shot death occurs before the next night's first wolf action.
- Public speeches do not contain unsupported hard claims marked as "公开记录".

- [ ] **Step 2: Add audit output**

If `scripts/print_game_audit.py` is used, add a "Rule-order anomalies" section showing event indices and offending players.

- [ ] **Step 3: Run audit tests**

Run: `python -m pytest tests\test_game_audit.py -q --basetemp=.pytest-tmp`

Expected: new audit checks pass for generated scenarios.

---

### Task 7: Full Verification

**Files:**
- No code changes unless previous tasks require cleanup.

- [ ] **Step 1: Run focused quality suite**

Run: `python -m pytest tests\runtime tests\agents\test_agents.py tests\test_game_audit.py -q --basetemp=.pytest-tmp`

Expected: all selected tests pass.

- [ ] **Step 2: Run full suite**

Run: `python -m pytest -q --basetemp=.pytest-tmp`

Expected: full suite passes with the existing expected skip count.

- [ ] **Step 3: Run a fresh real game smoke**

Run: `python scripts\run_real_game.py --max-steps 180`

Expected: generated game has no delayed hunter-shot ordering issue and fallback count is lower or at least auditable.

- [ ] **Step 4: Update progress ledger**

Update `PROGRESS.md` with changed files, verification commands, residual risks, and next recommended task.

---

## Success Criteria

- Hunter shot death is resolved in the same day chain before any later night action.
- Public-memory hard claims are grounded in actual transcript or downgraded to suspicion.
- Fallback speeches are deterministic but not repeated across most players.
- Structured output retry/fallback rate decreases on MiniMax-style outputs.
- `python -m pytest -q --basetemp=.pytest-tmp` passes.
