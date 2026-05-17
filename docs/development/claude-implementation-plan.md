# Design-Complete V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Werewolf Agent V1 project according to `docs/design/werewolf-agent-v1-design.md` and the current gap ledger in `PROGRESS.md`.

**Architecture:** Finish the deterministic rule/runtime core first, then wire real player/judge agents into the runtime, then add persistence, observer UI, and expanded evaluation. RuleEngine remains the only rule authority; agents, RAG, memory, tools, and UI must never adjudicate hidden truth or mutate `GameState` directly.

**Tech Stack:** Python, pytest, LangGraph, FastAPI, Pydantic, httpx, YAML config, optional SQLite/local persistence first, later PostgreSQL/Redis/vector store adapters, and a frontend stack to be chosen when the UI milestone begins.

---

## Source Of Truth

Read these before every implementation session:

- `PROGRESS.md`: current completion audit and known gaps.
- `docs/design/werewolf-agent-v1-design.md`: final design requirements.
- `config/rulesets/pre_witch_hunter_idiot_mixed.yaml`: deterministic V1 ruleset.
- `tests/rules/test_rule_engine_v1.py`: rule acceptance baseline.
- `tests/runtime/test_runtime.py`: runtime graph baseline.
- `README.md`: current startup and API key instructions.

Do not start development from older claims that "all phases are complete." The current true status is: backend prototype and module tests exist; design-complete V1 is not done.

## Global Claude Rules

- Work one task at a time. Do not combine runtime, persistence, UI, and evaluation in one change.
- Use TDD: write or update failing tests first, run the targeted test, implement the smallest change, then run targeted and full tests.
- Never let an agent, RAG, memory, MCP, API, or UI decide rules. RuleEngine is the only rule authority.
- Never inject moderator-only state, other players' private state, wolf chat, seer result, witch result, hybrid master binding, or `private_intent` into a player context unless the design explicitly allows that viewer to see it.
- Never use RAG for base rules. RAG is only for strategy, cases, speech patterns, and review experience.
- Keep existing public APIs backward compatible unless the task explicitly updates the API contract and tests.
- If a task touches runtime visibility, add leak-prevention tests.
- If a task touches event flow, add replay or reducer tests.
- After every task, update `PROGRESS.md` with exact completed behavior, remaining gaps, and verification output.

## Verification Commands

Use these commands unless a task specifies a narrower command:

```powershell
D:/Miniforge3/envs/wofkill/python.exe -m pytest -q
```

Targeted examples:

```powershell
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/runtime/test_runtime.py -q
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/rules/test_rule_engine_v1.py -q
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/api/test_api.py -q
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/evaluation/test_evaluation.py -q
```

## Milestone Order

1. Runtime rule-flow correctness.
2. Agent-driven live game orchestration.
3. End-to-end visibility and replay safety.
4. Persistence and startup/deployment foundation.
5. Observer dashboard.
6. RAG, memory, MCP production hardening.
7. Evaluation expansion.
8. Final documentation and delivery checklist.

Do not build the observer UI before the runtime can produce trustworthy public, player, and moderator views. Do not expand evaluation before live game records contain the data needed for those metrics.

---

### Task 1: Runtime Gap Tests

**Files:**
- Modify: `tests/runtime/test_runtime.py`
- Modify if needed: `tests/rules/test_rule_engine_v1.py`

- [ ] **Step 1: Add failing tests for missing runtime design nodes**

Add tests that assert `build_game_graph()` contains the design node `night_hunter_idiot_status`, and that the night edge order is:

```text
night_witch -> night_seer -> night_hunter_idiot_status -> first_night_hybrid_master -> resolve_night_node
```

- [ ] **Step 2: Add failing tests for seer night resolution**

Create a state with a seer and target. Invoke runtime night flow or `resolve_night` with `seer_target_id`. Assert a private/auditable `seer_check` event exists with `seer_id`, `target_id`, and alignment, and does not appear in public timeline.

- [ ] **Step 3: Add failing tests for sheriff badge after night death**

Create a sheriff killed at night by wolf kill or witch poison. Assert the runtime routes to badge transfer/tear before the next night when victory has not ended the game.

- [ ] **Step 4: Add failing tests for hunter shot timing**

Cover hunter killed by wolf kill and exile. Assert legal shot target selection creates a `hunter_shot` death in the correct resolution batch and that replay reconstructs it.

- [ ] **Step 5: Run targeted tests and confirm failures**

Run:

```powershell
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/runtime/test_runtime.py -q
```

Expected: failures prove missing runtime behavior.

- [ ] **Step 6: Commit tests only**

Commit message:

```text
test: add runtime design gap coverage
```

---

### Task 2: Complete Night Rule Events

**Files:**
- Modify: `werewolf_agent/engine/rule_engine.py`
- Modify: `werewolf_agent/core/models.py` if new structured fields are needed
- Modify: `tests/rules/test_rule_engine_v1.py`
- Modify: `tests/runtime/test_runtime.py`

- [ ] **Step 1: Implement seer check event generation**

`RuleEngine.resolve_night(...)` must use `seer_target_id` when an alive seer exists. It should call `check_alignment` and emit `GameEvent(type="seer_check", payload={...})`.

Acceptance:

- Event payload includes `seer_id`, `target_id`, `alignment`, `night_number`.
- Hybrid returns good alignment.
- The event is private to seer/moderator in view layers.

- [ ] **Step 2: Add reducer support if needed**

If replay needs seer checks, `reduce_event` should append `seer_check` without changing public state.

- [ ] **Step 3: Implement or formalize hunter shot events**

Ensure hunter shot creates deterministic events and deaths for both exile and night death paths. Avoid applying duplicate deaths if target is already dead.

- [ ] **Step 4: Run rule and runtime tests**

```powershell
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/rules/test_rule_engine_v1.py tests/runtime/test_runtime.py -q
```

- [ ] **Step 5: Commit**

```text
feat: complete core night rule events
```

---

### Task 3: Complete Runtime Night Routing

**Files:**
- Modify: `werewolf_agent/runtime/graph.py`
- Modify: `werewolf_agent/runtime/replay.py` if replay gaps appear
- Modify: `tests/runtime/test_runtime.py`

- [ ] **Step 1: Add `night_hunter_idiot_status` node**

The node should record first-night hunter/idiot status confirmation as moderator/private audit only. It must not leak to public or player contexts.

- [ ] **Step 2: Rewire night edges**

Required order:

```text
wolf_discussion -> wolf_consensus -> night_witch -> night_seer -> night_hunter_idiot_status -> first_night_hybrid_master -> resolve_night_node
```

- [ ] **Step 3: Pass `seer_target_id` into `resolve_night`**

Runtime `resolve_night` must forward `state.get("seer_target_id")`.

- [ ] **Step 4: Fix sheriff badge routing after night deaths**

After night settlement and victory check, if sheriff died and the game continues, route to `sheriff_badge_transfer` before regular day progression where required by design.

- [ ] **Step 5: Run runtime tests**

```powershell
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/runtime/test_runtime.py -q
```

- [ ] **Step 6: Commit**

```text
feat: align runtime night routing with design
```

---

### Task 4: Wire Agent Decisions Into Runtime

**Files:**
- Modify: `werewolf_agent/runtime/graph.py`
- Modify: `werewolf_agent/agents/player.py`
- Modify: `werewolf_agent/agents/schemas.py` if contexts need small additions
- Modify: `werewolf_agent/cognition/pipeline.py`
- Modify: `werewolf_agent/cognition/context.py`
- Modify: `tests/agents/test_agents.py`
- Modify: `tests/runtime/test_runtime.py`
- Create: `tests/integration/test_live_game_flow.py`

- [ ] **Step 1: Define an agent runtime adapter**

Create a small adapter boundary inside runtime or a focused helper module. It should convert `GameState + phase + actor_id` into `AgentContext` using existing cognition, RAG, memory, and legal-action tools.

- [ ] **Step 2: Replace scripted `night_witch` and `night_seer` behavior**

Use PlayerAgent only when an agent registry/router is provided. Keep deterministic test fallback for existing tests.

- [ ] **Step 3: Replace scripted day speech and vote behavior**

Day speech must produce public speech and private intent separately. Vote must validate against legal actions and targets.

- [ ] **Step 4: Keep all illegal action handling schema-constrained**

Illegal model output must retry and then fallback. Fallback must be logged as an evaluation `ActionRecord`.

- [ ] **Step 5: Add one full 12-player mock-provider game test**

Use deterministic mock providers. Assert:

- Game starts.
- At least one night and one day run.
- Public timeline does not contain private intent or hidden roles.
- Player private views only contain allowed private information.
- Moderator replay can see full audit after game or in debugger mode.

- [ ] **Step 6: Run integration and agent tests**

```powershell
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/agents/test_agents.py tests/runtime/test_runtime.py tests/integration/test_live_game_flow.py -q
```

- [ ] **Step 7: Commit**

```text
feat: drive runtime actions through agents
```

---

### Task 5: End-To-End Visibility And Replay Safety

**Files:**
- Modify: `werewolf_agent/api/views.py`
- Modify: `werewolf_agent/api/permissions.py`
- Modify: `werewolf_agent/runtime/replay.py`
- Modify: `tests/api/test_api.py`
- Create: `tests/integration/test_visibility_replay.py`

- [ ] **Step 1: Add leak tests across API views**

For public, player, spectator, moderator, and debugger views, test role reveal, wolf chat, seer result, witch result, hybrid master, private intent, cognition matrix, and RAG hits.

- [ ] **Step 2: Make intervention and pause/resume event-sourced**

Pause, resume, and moderator intervention must create audit events and be replayable. Avoid direct state-only mutation without event log.

- [ ] **Step 3: Verify replay source of truth**

For a complete mock game, rebuild from `initial_seed + ruleset_snapshot + event_log` and compare public state, deaths, sheriff state, victory, and hybrid result.

- [ ] **Step 4: Run API and integration tests**

```powershell
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/api/test_api.py tests/integration/test_visibility_replay.py -q
```

- [ ] **Step 5: Commit**

```text
feat: enforce end-to-end visibility and replay safety
```

---

### Task 6: Persistence Foundation

**Files:**
- Create: `werewolf_agent/storage/__init__.py`
- Create: `werewolf_agent/storage/schemas.py`
- Create: `werewolf_agent/storage/repository.py`
- Create: `werewolf_agent/storage/sqlite_store.py`
- Modify: `werewolf_agent/api/app.py`
- Modify: `werewolf_agent/runtime/checkpoints.py`
- Create: `tests/storage/test_storage.py`
- Modify: `README.md`

- [ ] **Step 1: Start with SQLite/local file persistence**

Implement a repository interface before adding production database dependencies. The interface must support games, players, events, deaths, sheriff records, model usage records, evaluation results, and config snapshots.

- [ ] **Step 2: Wire API to repository interface**

`create_app()` should accept an optional repository. Default local development can still use memory, but persistent mode must be testable.

- [ ] **Step 3: Persist event log and replay data**

Every committed event should be saved with ordering, game id, phase, and visibility metadata when available.

- [ ] **Step 4: Add storage tests**

Tests must cover round-trip create game, append events, load replay, store model usage, store evaluation, and restart-like reload.

- [ ] **Step 5: Update README startup docs**

Document memory mode and persistent mode separately.

- [ ] **Step 6: Run storage/API tests**

```powershell
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/storage/test_storage.py tests/api/test_api.py -q
```

- [ ] **Step 7: Commit**

```text
feat: add local persistence foundation
```

---

### Task 7: Observer Dashboard

**Files:**
- Create: `werewolf_agent/ui/` or `frontend/` after choosing stack
- Modify: `werewolf_agent/api/app.py` if serving static assets
- Modify: `README.md`
- Create: UI tests appropriate to chosen stack

- [ ] **Step 1: Choose UI stack and document it**

Prefer the simplest stack that can be run locally by Claude and the user. Do not introduce a large framework unless needed.

- [ ] **Step 2: Build the first usable observer screen**

It must show:

- Game list.
- Public state.
- Timeline.
- Deaths and votes.
- Pause/resume controls.

- [ ] **Step 3: Add moderator/debug views**

It must show:

- Private audit.
- RAG hits.
- Memory calls.
- Cognitive pipeline snapshots.
- Cognitive diff graph.
- Model/persona routing and cost/latency/token usage.

- [ ] **Step 4: Enforce permissions in UI**

The UI must not display moderator-only content in public/player modes.

- [ ] **Step 5: Verify in browser**

Use a local dev server and browser screenshot checks after implementation.

- [ ] **Step 6: Commit**

```text
feat: add observer dashboard
```

---

### Task 8: RAG, Memory, And MCP Hardening

**Files:**
- Modify: `werewolf_agent/rag/*`
- Modify: `werewolf_agent/memory/*`
- Modify: `werewolf_agent/tools/mcp_registry.py`
- Modify: `tests/rag/test_rag.py`
- Modify: `tests/memory/test_memory.py`
- Modify: `tests/tools/test_tools.py`

- [ ] **Step 1: Persist RAG and memory through storage interface**

RAG entries, reflection memory, profile updates, relation graph facts, and cognition snapshots must survive process restart in persistent mode.

- [ ] **Step 2: Add vector-store abstraction**

Add an interface first. Implement a local lightweight option before adding Qdrant/pgvector.

- [ ] **Step 3: Add RAG hit audit**

Every injected RAG hit must be traceable by game id, player id, phase, source, quality, and visibility boundary.

- [ ] **Step 4: Clarify MCP integration boundary**

If real MCP transport is added, keep MCP as external suggestion-only capability. It must not own game state truth.

- [ ] **Step 5: Run RAG/memory/tool tests**

```powershell
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/rag/test_rag.py tests/memory/test_memory.py tests/tools/test_tools.py -q
```

- [ ] **Step 6: Commit**

```text
feat: harden rag memory and mcp layers
```

---

### Task 9: Evaluation Expansion

**Files:**
- Modify: `werewolf_agent/evaluation/schemas.py`
- Modify: `werewolf_agent/evaluation/metrics.py`
- Modify: `werewolf_agent/evaluation/reports.py`
- Modify: `werewolf_agent/evaluation/runner.py`
- Modify: `tests/evaluation/test_evaluation.py`
- Create: `tests/integration/test_evaluation_live_game.py`

- [ ] **Step 1: Replace placeholder metrics with data-backed metrics**

Implement meaningful inputs and aggregation for:

- Lie detection.
- Stance accuracy.
- Speech influence.
- Bold-claim success.
- Hybrid master choice value.
- Witch potion yield.
- Seer badge-flow quality.
- Contradiction adoption.
- Cognitive pipeline compression and comparison.
- RAG strategy and memory strategy comparisons.

- [ ] **Step 2: Add metric provenance**

Every advanced metric should explain which events/reviews/snapshots produced it.

- [ ] **Step 3: Export reports**

Add JSON report output that can be loaded by the observer UI.

- [ ] **Step 4: Run evaluation tests**

```powershell
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/evaluation/test_evaluation.py tests/integration/test_evaluation_live_game.py -q
```

- [ ] **Step 5: Commit**

```text
feat: complete evaluation metrics
```

---

### Task 10: Final Delivery Hardening

**Files:**
- Modify: `README.md`
- Modify: `PROGRESS.md`
- Create or modify: `docker-compose.yml` if production-like local services are added
- Modify: `docs/development/conda-environment.md`
- Add any deployment docs needed

- [ ] **Step 1: Final full test run**

```powershell
D:/Miniforge3/envs/wofkill/python.exe -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run one complete local game**

Use mock providers first, then configured real providers if API keys exist.

- [ ] **Step 3: Verify API startup**

```powershell
D:/Miniforge3/envs/wofkill/python.exe -m uvicorn werewolf_agent.api.app:app --reload
```

Document the actual working command if the app factory requires a different invocation.

- [ ] **Step 4: Verify observer dashboard startup**

Document the exact command and URL.

- [ ] **Step 5: Verify API key setup**

Document player/judge model assignment through `config/models.yaml` and `.env`.

- [ ] **Step 6: Update `PROGRESS.md` honestly**

Only mark design-complete when all gaps are closed and verified. Keep any remaining limitations explicit.

- [ ] **Step 7: Commit**

```text
docs: finalize design-complete delivery guide
```

---

## Final Acceptance Checklist

- [ ] Full tests pass.
- [ ] One complete 12-player mock-provider game runs end to end.
- [ ] Real provider configuration is documented and can be tested when keys are present.
- [ ] Public/player/moderator views have end-to-end leak tests.
- [ ] Replay reconstructs deaths, votes, sheriff state, victory, and hybrid result.
- [ ] Runtime graph includes `night_hunter_idiot_status`.
- [ ] Seer, witch, hunter, wolf discussion, sheriff badge transfer, speech, and vote flows are connected to live runtime.
- [ ] Persistence can reload a game and event log after process restart.
- [ ] Observer dashboard exists and respects permissions.
- [ ] RAG and memory are auditable and do not decide rules.
- [ ] Evaluation metrics no longer claim placeholder values as meaningful.
- [ ] README contains exact startup, API key, model assignment, test, API, UI, and persistence instructions.

## Handoff To Claude

Give Claude this prompt before implementation:

```text
You are implementing docs/development/claude-implementation-plan.md. Start at Task 1. Use TDD. Do not skip tasks. Do not combine tasks. After each task, run the specified tests, update PROGRESS.md, and stop for review. RuleEngine is the only rule authority. Do not leak private information into player contexts. Do not claim completion without test evidence.
```
