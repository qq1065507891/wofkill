# Prompt Injection Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split prompt inputs into explicit system and user sections so current facts, private information, RAG knowledge, cross-game memory, cognition, strategy directives, and skill analysis cannot be confused.

**Architecture:** Keep stable behavioral rules in the system prompt and dynamic per-turn evidence in the user prompt. Add explicit `AgentContext` fields for RAG, memory, cognition, and skill analysis hints; keep legacy fields readable but prefer the new sections.

**Tech Stack:** Python, Pydantic models, pytest, existing `PlayerPromptBuilder`, `build_agent_context`, `RAGKnowledgeService`.

---

### Task 1: Prompt Section Tests

**Files:**
- Modify: `tests/agents/test_agents.py`
- Modify: `tests/runtime/test_runtime.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert:
- system prompt contains stable information-boundary, reasoning, and skill-use sections;
- user prompt renders `知识库提示`, `跨局反思记忆`, `长期能力画像`, `我的当前局记忆`, `我的认知矩阵`, and `技能分析结果` as separate sections;
- RAG injection writes to `context.rag_hints`, not `context.salience_items`;
- restored memory writes profile/reflections to explicit memory fields, not only `strategy_directive`.

- [ ] **Step 2: Run tests and verify red**

Run:

```powershell
python -m pytest tests\agents\test_agents.py::TestPromptInjectionBoundaries tests\runtime\test_runtime.py::TestPromptInjectionBoundaries -q
```

Expected: failures for missing fields/sections.

### Task 2: Context Fields and Prompt Builder

**Files:**
- Modify: `werewolf_agent/agents/schemas.py`
- Modify: `werewolf_agent/agents/prompt_builder.py`

- [ ] **Step 3: Add explicit fields**

Add `rag_hints`, `private_memory_hints`, `reflection_memory_hints`, `profile_memory_hint`, `cognition_matrix_hint`, and `skill_analysis_hints` with defaults.

- [ ] **Step 4: Add system prompt boundary sections**

Add stable sections: `信息边界`, `推理方法`, `工具与技能使用规范`.

- [ ] **Step 5: Add user prompt dynamic sections**

Render sections in this order: phase, belief, public summary, visible state, private memory, key events, RAG hints, reflection memory, profile memory, cognition matrix, strategy directive, skill analysis, transcript, retry, task, contract.

### Task 3: Runtime Injection Paths

**Files:**
- Modify: `werewolf_agent/runtime/agent_adapter.py`

- [ ] **Step 6: Move RAG hints**

Change `_inject_seed_rag_hints` to write retrieved `rag_hit` items into `AgentContext.rag_hints`; leave `salience_items` for real current-game events.

- [ ] **Step 7: Move memory hints**

Move `private_memory`, cross-game profile, and cross-game reflections into the new fields. Keep strategy directives focused on current-turn instructions.

- [ ] **Step 8: Summarize cognition matrix only when meaningful**

If restored memory has a matrix for the current player, inject a compact self-view summary only for that player.

### Task 4: Verification

- [ ] **Step 9: Run targeted tests**

Run:

```powershell
python -m pytest tests\agents\test_agents.py::TestPromptInjectionBoundaries tests\runtime\test_runtime.py::TestPromptInjectionBoundaries -q
python -m pytest tests\agents\test_agents.py::TestSkillSkipRetry tests\runtime\test_runtime.py::TestSeedRAGContext tests\runtime\test_runtime.py::TestWitchPoisonPressureContext -q
```

- [ ] **Step 10: Report evidence**

Report exact commands and pass/fail output. Do not claim completion without fresh verification.
