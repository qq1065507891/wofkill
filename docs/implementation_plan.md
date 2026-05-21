# Refactoring Plan: Cleanup, Fail-Fast Vector Store, and Graph Boilerplate Reduction

This implementation plan outlines three major cleanup and refactoring goals to align the codebase with Andrej Karpathy's guidelines (specifically Simplicity First and Think Before Coding).

---

## User Review Required

> [!WARNING]
> Removing `wolf_king_guard_demo` from the ruleset registry's hardcoded check will change the error thrown when loading that ruleset from compatibility rejection (`display_only`) to registry lookup failure (`ValueError: Unknown ruleset_id`). We will update the corresponding tests to verify this cleaner lookup error or use dynamic rulesets.

---

## Open Questions

> [!IMPORTANT]
> **No open questions currently.** We will directly proceed with the execution once this plan is approved.

---

## Proposed Changes

### Customization Registry & Ruleset Templates
Remove speculative/placeholder ruleset configurations that are not backed by real RuleEngine implementations.

#### [MODIFY] [ruleset_registry.py](file:///e:/NLP/agent/wofkill/werewolf_agent/customization/ruleset_registry.py)
* Remove the hardcoded `wolf_king_guard_demo` check from the `get()` method.
* Maintain a clean lookup path that throws `ValueError` for unregistered rulesets.

#### [MODIFY] [test_ruleset_registry.py](file:///e:/NLP/agent/wofkill/tests/customization/test_ruleset_registry.py)
* Update `test_game_runner_rejects_display_only_ruleset` to assert `ValueError` with `Unknown ruleset_id` when requesting `wolf_king_guard_demo`.
* Alternatively, test compatibility rejection by generating a custom display-only entry dynamically via `RulesetRegistryEntry` and `GameRunner`.

---

### Vector Store & RAG Fail-Fast Logging
Make automatic backend selection transparent by adding warnings, and ensure explicit configurations fail fast when dependencies are missing.

#### [MODIFY] [vector_store.py](file:///e:/NLP/agent/wofkill/werewolf_agent/rag/vector_store.py)
* Refactor `AutoVectorStore.__init__` to log a warning when falling back to lesser backends (e.g. falling back from `siliconflow` to `embedding` or `local`).
* Ensure explicit configuration strings like `"siliconflow"` raise an error immediately if the environment variables or dependencies are not met (this is already handled by `SiliconFlowEmbeddingClient.__init__`, but we will make it explicit).

---

### Graph Orchestration Boilerplate Reduction
Reduce duplicate `agent_registry` checks and `_call_agent` code blocks in runtime orchestration node functions.

#### [MODIFY] [graph.py](file:///e:/NLP/agent/wofkill/werewolf_agent/runtime/graph.py)
* Introduce `_dispatch_agent(state, agent_fn, player_id=None, timeout_override=None, fallback_value=None)` helper function.
* Refactor all action nodes in `graph.py` (such as `night_seer`, `night_witch`, `day_vote`, `sheriff_register`, etc.) to use the helper. This reduces repeating the registry-check, timeout invocation, and fallback default checks in every node.

---

## Verification Plan

### Automated Tests
Run the existing pytest suites before and after modifications:
* Customization tests: `pytest tests/customization/test_ruleset_registry.py`
* RAG and vector store tests: `pytest tests/rag/test_rag_hardening.py`
* Runtime and graph tests: `pytest tests/runtime/`
* Full suite regression run: `pytest`
