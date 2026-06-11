# Agent Balance Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove implementation-layer biases that corrupt public information and balance metrics without changing game rules or model routing.

**Architecture:** Make surgical changes at existing parser, cognition, runtime
directive, speech-quality, summary, and evaluation boundaries. Preserve current
APIs where possible and add optional parameters only when role capacities must
be injected.

**Tech Stack:** Python, Pydantic, pytest

---

### Task 1: Speech semantic preservation

**Files:**
- Modify: `werewolf_agent/agents/output_parser.py`
- Test: `tests/agents/test_output_parser.py`

- [ ] Add failing tests for seer gold-result preservation and vote reason mismatch.
- [ ] Run the tests and confirm the current behavior fails.
- [ ] Stop adding suspicion/vote text and normalize mismatched vote explanations.
- [ ] Run the parser tests.

### Task 2: Capacity-aware role claims

**Files:**
- Modify: `werewolf_agent/cognition/contradiction.py`
- Modify: `werewolf_agent/runtime/context.py`
- Test: `tests/cognition/test_cognition.py`

- [ ] Add failing tests for two legal villager claims and two seer claims.
- [ ] Inject role capacities from the ruleset.
- [ ] Run cognition and context tests.

### Task 3: Sheriff public-information policy

**Files:**
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Test: `tests/runtime/test_sheriff_flow.py`

- [ ] Add failing tests for public-claim counting, fake-seer gating, and withdrawal.
- [ ] Build sheriff guidance from public events and wolf assignment only.
- [ ] Run sheriff tests.

### Task 4: Hybrid and summary visibility

**Files:**
- Modify: `werewolf_agent/runtime/directives/hybrid.py`
- Modify: `werewolf_agent/runtime/context.py`
- Modify: `werewolf_agent/runtime/nodes/summary.py`
- Test: `tests/runtime/test_strategy_directives.py`
- Test: `tests/runtime/test_context.py`

- [ ] Add failing tests proving master faction is absent from hybrid prompts.
- [ ] Remove faction-specific hybrid directives and dead-master disclosure.
- [ ] Keep player-private summaries out of public events.
- [ ] Run visibility and directive tests.

### Task 5: Intent-aware speech quality

**Files:**
- Modify: `werewolf_agent/runtime/speech_quality.py`
- Modify: relevant caller to pass speech intent
- Test: `tests/runtime/test_speech_quality.py`

- [ ] Add failing tests for stand-with-seer, question, and defensive speech.
- [ ] Select required fields by intent while retaining filler rejection.
- [ ] Run speech-quality tests.

### Task 6: Evaluation event contract

**Files:**
- Modify: `werewolf_agent/evaluation/balance_audit.py`
- Modify: `werewolf_agent/evaluation/metrics.py`
- Test: `tests/evaluation/test_game_balance_batch.py`
- Test: `tests/evaluation/test_evaluation.py`

- [ ] Add failing tests using real runtime event names and vote payload shape.
- [ ] Update event readers and schema-failure detection.
- [ ] Run evaluation tests.

### Task 7: Final verification

- [ ] Run all targeted suites.
- [ ] Run the complete pytest suite.
- [ ] Review `git diff --check` and `git diff --stat`.
