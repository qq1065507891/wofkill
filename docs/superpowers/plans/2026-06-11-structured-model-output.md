# Structured Model Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify player action contracts, make structured-output protocols explicit, and classify failures without adding model failover or rotation.

**Architecture:** `ActionContract` owns mode-specific schemas and field lists. `StructuredOutputPolicy` resolves configured protocol order, providers render the selected protocol, and `PlayerAgent` only advances protocol after protocol/schema failures. Traces record the actual protocol and failure stage.

**Tech Stack:** Python 3.11+, Pydantic v2, PyYAML, pytest

---

### Task 1: Canonical Action Contract

**Files:**
- Create: `werewolf_agent/agents/action_contract.py`
- Modify: `werewolf_agent/agents/tool_schema.py`
- Modify: `werewolf_agent/agents/player.py`
- Modify: `werewolf_agent/agents/prompt_builder.py`
- Test: `tests/agents/test_action_contract.py`
- Test: `tests/agents/test_player_agent.py`

- [ ] Write failing tests for full-action, target-choice vote, and speech-intent schemas.
- [ ] Run the focused tests and verify the missing contract fails.
- [ ] Implement `ActionContract` and delegate the existing tool helper to it.
- [ ] Make `PlayerAgent` and strict prompt rendering use the same contract.
- [ ] Run the focused tests and existing prompt/tool-schema tests.

### Task 2: Structured Output Policy

**Files:**
- Create: `werewolf_agent/model_gateway/structured_output.py`
- Modify: `werewolf_agent/model_gateway/router.py`
- Modify: `config/models.yaml`
- Test: `tests/model_gateway/test_structured_output.py`
- Test: `tests/agents/test_model_router.py`

- [ ] Write failing tests for explicit modes, legacy mapping, fallback order, and YAML resolution.
- [ ] Run the focused tests and verify the policy is absent.
- [ ] Implement mode and policy parsing with backward compatibility.
- [ ] Add explicit protocol declarations to current model profiles.
- [ ] Run the policy and router tests.

### Task 3: Provider Protocol Rendering

**Files:**
- Modify: `werewolf_agent/model_gateway/providers/openai.py`
- Modify: `werewolf_agent/model_gateway/providers/anthropic.py`
- Modify: `werewolf_agent/model_gateway/providers/minimax.py`
- Test: `tests/model_gateway/test_structured_output.py`
- Test: `tests/model_gateway/test_anthropic_provider.py`

- [ ] Write failing payload tests for `native_tool`, `json_schema`, `json_object`, and `text_json`.
- [ ] Run tests and verify mode-specific payload assertions fail.
- [ ] Implement provider request rendering from `ModelConfig.structured_output_mode`.
- [ ] Preserve legacy direct-provider behavior for `allow_text_tool_fallback`.
- [ ] Run provider tests.

### Task 4: Retry Classification And Trace Telemetry

**Files:**
- Modify: `werewolf_agent/agents/player.py`
- Modify: `werewolf_agent/agents/schemas.py`
- Modify: `werewolf_agent/agents/trace_builder.py`
- Modify: `werewolf_agent/model_gateway/router.py`
- Test: `tests/agents/test_player_agent.py`
- Test: `tests/agents/test_trace_builder.py`

- [ ] Write failing tests for protocol fallback, semantic retry stability, and trace metadata.
- [ ] Run tests and verify the new behavior fails.
- [ ] Route protocol/schema failures to the next configured mode.
- [ ] Keep semantic retries on the current mode.
- [ ] Record output mode and failure stage in result, usage, and trace objects.
- [ ] Run focused agent and trace tests.

### Task 5: Verification

**Files:**
- Verify only

- [ ] Run agents and model-gateway test suites with a unique workspace-local `--basetemp`.
- [ ] Run runtime and integration test shards with separate `--basetemp` directories.
- [ ] Inspect `git diff --check` and the final diff for unintended changes.
- [ ] Remove only temporary test directories created by this task after validating their resolved paths.
