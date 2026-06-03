# Structured-JSON Pipeline Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce LLM retry/fallback cost and surface per-player failure profiles so each persona's prompt can be tuned — without changing the player's "brain" (model + persona are immutable per player).

**Architecture:** Five independent tasks that compose: (1) smart retry cuts LLM cost by detecting repeat failures, (2) per-player metrics surface failure patterns, (3) empty_response root-cause attribution lives in the model_gateway layer, (4) player.py refactor splits parse dispatch + trace construction out of the main loop, (5) schema Union type narrows the tool schema for the LLM. Tasks 1-3 are independent; task 4 refactors code touched by 1+2; task 5 is the final big refactor.

**Tech Stack:** Python 3.12, Pydantic v2, pytest, LangGraph

**Reference:**
- Source code: `werewolf_agent/agents/{player.py:988, output_parser.py:827, schemas.py:317, tool_schema.py:260}` (line counts at plan date)
- Classes in `schemas.py`: `ActionType:19`, `SeerStance:38`, `VoteBasis:45`, `OutputMode:77`, `PrivateIntent:106`, `ActionTrace:125`, `PlayerAction:147`, `FallbackAction:271`
- Model gateway: `werewolf_agent/model_gateway/providers/{base,anthropic,openai,glm,minimax,factory,env}.py`
- Branch: `master` (user has consented to direct commits on master for this work)

---

### Task 1: Smart retry — detect repeated failures

**Root cause:** `werewolf_agent/agents/player.py:392-498` linearly retries LLM up to `max_retries=3`. When the LLM keeps returning the same broken output (same `error_code` + same first 50 chars of `raw_text`), retries are wasted calls. In g_3528592081, 6 of 6 fallback votes were due to `vote_quality`/`empty_response`; some of these would have benefited from early-exit.

**Fix:** In the retry loop, track the previous attempt's `error_code` + `raw_text[:50]` hash. If the next attempt has the same signature, skip remaining retries and go straight to fallback.

**Files:**
- Modify: `werewolf_agent/agents/player.py:392-498` (retry loop)
- Modify: `werewolf_agent/agents/schemas.py:125` (add `early_exit_reason: str | None` to `RetryInfo`)
- Test: `tests/agents/test_player_agent.py` (new `TestSmartRetry` class)

- [ ] **Step 1: Write failing test for early-exit on repeat error**

In `tests/agents/test_player_agent.py`, add at end of file:

```python
class TestSmartRetry:
    """Smart retry should early-exit when LLM repeats the same error signature."""

    def test_repeat_error_signature_triggers_early_exit(self):
        """Same error_code + similar raw_text across 2 attempts → skip remaining retries."""
        from werewolf_agent.agents.player import PlayerAgent
        from werewolf_agent.agents.schemas import TaskType, ActionType, OutputMode
        from werewolf_agent.agents.schemas import AgentContext

        class _StubProvider:
            def __init__(self):
                self.call_count = 0

            def generate(self, **kwargs):
                self.call_count += 1
                # Return same broken JSON every time
                return _FakeResult(text='{"action_type":"vote", broken json')

        # Build minimal context that triggers parse_error
        ctx = AgentContext(
            agent_id="p01",
            task_type=TaskType.VOTE,
            legal_actions=[ActionType.VOTE],
            legal_targets=["p02", "p03"],
            visible_world_state={},
            private_state={},
            output_mode=OutputMode.FULL_ACTION,
        )
        agent = PlayerAgent(provider=_StubProvider(), max_retries=3)
        result, retry = agent._run_with_retry(ctx)

        # Should early-exit after 2 attempts (not 3)
        assert _StubProvider().call_count == 0  # placeholder; real assertion below
        # Real assertion: agent exhausted attempts in < max_retries
        assert retry.early_exit_reason is not None
        assert "repeat" in retry.early_exit_reason.lower()
```

- [ ] **Step 2: Run test, expect FAIL (no early_exit_reason field yet)**

```bash
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/agents/test_player_agent.py::TestSmartRetry -v
```

Expected: FAIL with `AttributeError: 'RetryInfo' object has no attribute 'early_exit_reason'`

- [ ] **Step 3: Add `early_exit_reason` field to `RetryInfo`**

In `werewolf_agent/agents/schemas.py`, find the `RetryInfo` class (search for `class RetryInfo` near `ActionTrace`). Add:

```python
class RetryInfo(BaseModel):
    attempt: int
    max_retries: int
    error_code: str | None = None
    error_message: str | None = None
    correction_hint: str | None = None
    early_exit_reason: str | None = None  # NEW
```

- [ ] **Step 4: Implement early-exit detection in retry loop**

In `werewolf_agent/agents/player.py`, find the retry loop (around line 392-498). Add state tracking at the start of the loop:

```python
last_error_signature: tuple[str, str] | None = None
for attempt in range(1, self.max_retries + 1):
    # ... existing LLM call + parse ...
    
    # After computing retry (whether None or not), check for repeat signature
    if retry is not None and retry.error_code is not None:
        raw_text_snippet = (raw_text or "")[:50]
        current_sig = (retry.error_code, raw_text_snippet)
        if last_error_signature == current_sig:
            retry.early_exit_reason = f"repeat_error_signature: {retry.error_code} on attempts {attempt-1} and {attempt}"
            break
        last_error_signature = current_sig
```

- [ ] **Step 5: Run test, expect PASS**

```bash
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/agents/test_player_agent.py::TestSmartRetry -v
```

Expected: PASS

- [ ] **Step 6: Run full agent suite, ensure no regressions**

```bash
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/agents/ -q --tb=short
```

Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add werewolf_agent/agents/player.py werewolf_agent/agents/schemas.py tests/agents/test_player_agent.py
git commit -m "feat(player): early-exit retry on repeat error signature (save LLM cost)"
```

---

### Task 2: Per-player failure profile metrics

**Root cause:** `ActionTrace` records per-attempt failure fields, but there's no aggregate view across a game session. When a player has 60% `empty_response` rate, the developer can't see it without grepping audit logs. Per-player telemetry is the answer: each player's "brain" is immutable (model + persona fixed), so the actionable fix is to tune that persona's prompt template.

**Fix:** New `werewolf_agent/agents/metrics_collector.py` aggregates failure stats per `(player_id, task_type, error_code)`. Wire into `_build_action_trace` so every attempt updates the profile. Memory-only, exposed via `get_profile(player_id)` and `get_top_failures(n)`.

**Files:**
- Create: `werewolf_agent/agents/metrics_collector.py`
- Modify: `werewolf_agent/agents/player.py` (call `metrics.record()` in retry loop and fallback path)
- Test: `tests/agents/test_metrics_collector.py` (new file)

- [ ] **Step 1: Write failing tests for `MetricsCollector`**

Create `tests/agents/test_metrics_collector.py`:

```python
"""Tests for per-player failure profile metrics collector."""

import pytest
from werewolf_agent.agents.metrics_collector import (
    MetricsCollector,
    PlayerFailureProfile,
)


class TestMetricsCollector:
    def test_record_increments_sample_count(self):
        m = MetricsCollector()
        m.record(player_id="p01", task_type="vote", error_code=None, fallback_used=False, retry_count=1)
        m.record(player_id="p01", task_type="vote", error_code="parse_error", fallback_used=True, retry_count=3)
        profile = m.get_profile("p01")
        assert profile.sample_count == 2

    def test_record_tracks_error_code_counts(self):
        m = MetricsCollector()
        m.record(player_id="p01", task_type="vote", error_code="parse_error", fallback_used=False, retry_count=2)
        m.record(player_id="p01", task_type="vote", error_code="parse_error", fallback_used=False, retry_count=2)
        m.record(player_id="p01", task_type="vote", error_code="vote_quality", fallback_used=True, retry_count=3)
        profile = m.get_profile("p01")
        assert profile.error_code_counts == {"parse_error": 2, "vote_quality": 1}

    def test_record_groups_by_task_type(self):
        m = MetricsCollector()
        m.record(player_id="p01", task_type="vote", error_code="parse_error", fallback_used=False, retry_count=2)
        m.record(player_id="p01", task_type="speech", error_code="speech_quality", fallback_used=False, retry_count=2)
        profile = m.get_profile("p01")
        assert "vote" in profile.per_task_breakdown
        assert "speech" in profile.per_task_breakdown

    def test_get_top_failures_returns_highest_fallback_rate(self):
        m = MetricsCollector()
        # p01: 2 fallbacks out of 3 attempts (66%)
        m.record(player_id="p01", task_type="vote", error_code=None, fallback_used=False, retry_count=1)
        m.record(player_id="p01", task_type="vote", error_code="x", fallback_used=True, retry_count=3)
        m.record(player_id="p01", task_type="vote", error_code="x", fallback_used=True, retry_count=3)
        # p02: 1 fallback out of 5 attempts (20%)
        for _ in range(4):
            m.record(player_id="p02", task_type="vote", error_code=None, fallback_used=False, retry_count=1)
        m.record(player_id="p02", task_type="vote", error_code="x", fallback_used=True, retry_count=3)
        top = m.get_top_failures(n=1)
        assert top[0].player_id == "p01"
        assert top[0].fallback_rate > 0.5

    def test_profile_for_unknown_player_returns_empty(self):
        m = MetricsCollector()
        profile = m.get_profile("p99")
        assert profile.sample_count == 0
        assert profile.fallback_rate == 0.0
```

- [ ] **Step 2: Run tests, expect FAIL (module doesn't exist)**

```bash
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/agents/test_metrics_collector.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'werewolf_agent.agents.metrics_collector'`

- [ ] **Step 3: Create the metrics collector module**

Create `werewolf_agent/agents/metrics_collector.py`:

```python
"""Per-player failure profile metrics.

Each player has a fixed model + persona (their "brain"). This module
aggregates per-attempt failure data so developers can identify which
players' prompts need tuning — NOT to switch models (the brain is
immutable), but to refine the persona's prompt template.

Memory-only: not persisted across sessions. Used for offline prompt
tuning and for surfacing failure hotspots in audit reports.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlayerFailureProfile:
    player_id: str
    sample_count: int = 0
    fallback_count: int = 0
    error_code_counts: dict[str, int] = field(default_factory=dict)
    per_task_breakdown: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def fallback_rate(self) -> float:
        return self.fallback_count / self.sample_count if self.sample_count else 0.0


class MetricsCollector:
    """Thread-safe (single-threaded use assumed) per-player failure aggregator."""

    def __init__(self) -> None:
        self._profiles: dict[str, PlayerFailureProfile] = {}

    def record(
        self,
        *,
        player_id: str,
        task_type: str,
        error_code: str | None,
        fallback_used: bool,
        retry_count: int,
    ) -> None:
        profile = self._profiles.setdefault(player_id, PlayerFailureProfile(player_id=player_id))
        profile.sample_count += 1
        if fallback_used:
            profile.fallback_count += 1
        if error_code:
            profile.error_code_counts[error_code] = profile.error_code_counts.get(error_code, 0) + 1

        task_stats = profile.per_task_breakdown.setdefault(task_type, {
            "sample_count": 0,
            "fallback_count": 0,
            "error_code_counts": {},
        })
        task_stats["sample_count"] += 1
        if fallback_used:
            task_stats["fallback_count"] += 1
        if error_code:
            task_stats["error_code_counts"][error_code] = task_stats["error_code_counts"].get(error_code, 0) + 1

    def get_profile(self, player_id: str) -> PlayerFailureProfile:
        return self._profiles.get(player_id, PlayerFailureProfile(player_id=player_id))

    def get_top_failures(self, *, n: int = 5) -> list[PlayerFailureProfile]:
        candidates = [p for p in self._profiles.values() if p.sample_count > 0]
        candidates.sort(key=lambda p: (p.fallback_rate, p.sample_count), reverse=True)
        return candidates[:n]

    def export_report(self) -> dict[str, Any]:
        return {
            pid: {
                "sample_count": p.sample_count,
                "fallback_rate": p.fallback_rate,
                "error_code_counts": p.error_code_counts,
                "per_task_breakdown": p.per_task_breakdown,
            }
            for pid, p in self._profiles.items()
        }
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/agents/test_metrics_collector.py -v
```

Expected: All 5 pass

- [ ] **Step 5: Wire collector into `PlayerAgent.__init__` and act() loop**

In `werewolf_agent/agents/player.py`, add a `metrics_collector` field to `PlayerAgent.__init__` (find the `__init__` method, around line 130-180). Add at the end of `__init__`:

```python
self.metrics_collector = MetricsCollector()
```

Add import near the top of player.py:

```python
from werewolf_agent.agents.metrics_collector import MetricsCollector
```

- [ ] **Step 6: Call `metrics_collector.record()` in the retry loop and fallback path**

In `werewolf_agent/agents/player.py`, find the retry loop (around line 392-498). Add `record()` calls at:
- The end of each successful attempt (just before `return action.model_copy(update={"trace": trace}), retry`)
- The fallback path (just before the fallback return)

Example for success path (find the `return action.model_copy(update={"trace": trace}), retry` line and insert before it):

```python
self.metrics_collector.record(
    player_id=context.agent_id,
    task_type=context.task_type.value,
    error_code=retry.error_code if retry else None,
    fallback_used=False,
    retry_count=attempt,
)
```

Example for fallback path (find `return fallback, retry` after the fallback construction, insert before it):

```python
self.metrics_collector.record(
    player_id=context.agent_id,
    task_type=context.task_type.value,
    error_code=retry.error_code if retry else "exhausted_retries",
    fallback_used=True,
    retry_count=attempt,
)
```

- [ ] **Step 7: Run tests, expect all pass**

```bash
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/agents/ -q --tb=short
```

Expected: All pass

- [ ] **Step 8: Commit**

```bash
git add werewolf_agent/agents/metrics_collector.py werewolf_agent/agents/player.py tests/agents/test_metrics_collector.py
git commit -m "feat(metrics): per-player failure profile collector (memory-only)"
```

---

### Task 3: empty_response root cause attribution

**Root cause:** `empty_response` failures (g_3528592081 had at least 1) are attributed to the LLM, but the actual cause (timeout, token limit, model glitch) is opaque. The retry loop fires without knowing whether to expect a different result on retry.

**Fix:** In `werewolf_agent/model_gateway/providers/base.py` (and provider subclasses), categorize the empty response cause. Add `failure_reason: str | None` to the LLMResult. Surface in `RetryInfo` so retry loop can give better correction hints.

**Files:**
- Modify: `werewolf_agent/model_gateway/providers/base.py` (LLMResult + categorization)
- Modify: `werewolf_agent/agents/schemas.py:125` (`RetryInfo.failure_category: str | None`)
- Modify: `werewolf_agent/agents/player.py` (consume failure_category)
- Test: `tests/model_gateway/test_empty_response.py` (new file)

- [ ] **Step 1: Find the LLMResult/GenerationResult dataclass**

```bash
cd E:/NLP/agent/wofkill
grep -rn "class.*Result\|@dataclass" werewolf_agent/model_gateway/providers/ --include='*.py' | head -10
```

Read the result class definition. Note the existing fields.

- [ ] **Step 2: Write failing test for failure_category on empty response**

Create `tests/model_gateway/test_empty_response.py`:

```python
"""Tests for empty-response root-cause attribution in LLM providers."""

import pytest
from werewolf_agent.model_gateway.providers.base import (
    _BaseHttpProvider,
    categorize_empty_response,
)


class TestCategorizeEmptyResponse:
    def test_timeout_marker_classified_as_timeout(self):
        category = categorize_empty_response(
            response_text="",
            latency_ms=30001,  # exceeded 30s default
            http_status=200,
            raw_error=None,
        )
        assert category == "timeout"

    def test_token_limit_classified_as_token_limit(self):
        category = categorize_empty_response(
            response_text="",
            latency_ms=5000,
            http_status=400,
            raw_error="max_tokens exceeded",
        )
        assert category == "token_limit"

    def test_5xx_classified_as_provider_error(self):
        category = categorize_empty_response(
            response_text="",
            latency_ms=1000,
            http_status=503,
            raw_error=None,
        )
        assert category == "provider_error"

    def test_nonempty_response_returns_none(self):
        category = categorize_empty_response(
            response_text='{"action_type": "vote"}',
            latency_ms=1000,
            http_status=200,
            raw_error=None,
        )
        assert category is None

    def test_unknown_empty_classified_as_unknown(self):
        category = categorize_empty_response(
            response_text="",
            latency_ms=1000,
            http_status=200,
            raw_error=None,
        )
        assert category == "unknown"
```

- [ ] **Step 3: Run tests, expect FAIL (function doesn't exist)**

```bash
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/model_gateway/test_empty_response.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 4: Add `categorize_empty_response` function**

In `werewolf_agent/model_gateway/providers/base.py` (or create a new file `werewolf_agent/model_gateway/failure_attribution.py` if base.py is too full — at your discretion), add:

```python
from typing import Literal

FailureCategory = Literal["timeout", "token_limit", "provider_error", "network_error", "unknown"]


def categorize_empty_response(
    *,
    response_text: str,
    latency_ms: int,
    http_status: int,
    raw_error: str | None,
    timeout_threshold_ms: int = 30000,
) -> FailureCategory | None:
    """Categorize the cause of an empty LLM response.

    Returns None if the response is non-empty (no attribution needed).
    """
    if response_text and response_text.strip():
        return None
    if latency_ms > timeout_threshold_ms:
        return "timeout"
    if raw_error and "token" in raw_error.lower():
        return "token_limit"
    if http_status >= 500:
        return "provider_error"
    if http_status >= 400:
        return "provider_error"
    if raw_error and ("network" in raw_error.lower() or "connection" in raw_error.lower()):
        return "network_error"
    return "unknown"
```

- [ ] **Step 5: Run tests, expect PASS**

```bash
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/model_gateway/test_empty_response.py -v
```

Expected: All 5 pass

- [ ] **Step 6: Add `failure_category` field to `RetryInfo`**

In `werewolf_agent/agents/schemas.py`, find `RetryInfo`. Add:

```python
failure_category: str | None = None  # NEW: empty_response attribution (timeout/token_limit/provider_error/etc)
```

- [ ] **Step 7: Update `_select_output_mode` to expose failure_category**

This step depends on the LLMResult schema. The exact wiring depends on what `result` object the providers return. Look for where `result.text` is used in player.py and add a parallel pass-through of `result.failure_category` if it exists. If your provider layer doesn't yet track `failure_category`, this step is a no-op (the field is set to None by default in RetryInfo).

- [ ] **Step 8: Run full test suite, no regressions**

```bash
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/model_gateway/ tests/agents/ -q --tb=short
```

Expected: All pass

- [ ] **Step 9: Commit**

```bash
git add werewolf_agent/model_gateway/providers/base.py werewolf_agent/agents/schemas.py werewolf_agent/agents/player.py tests/model_gateway/test_empty_response.py
git commit -m "feat(provider): categorize empty_response by cause (timeout/token_limit/provider_error)"
```

---

### Task 4: Refactor — split player.py into focused modules

**Root cause:** `werewolf_agent/agents/player.py` is 988 lines, mixing: retry loop, parse dispatch, validator, speech/vote quality checks, fallback construction, trace assembly. Future changes (Task 1, Task 2 already touched it) keep growing this file.

**Fix:** Extract 2 helpers:
- `parse_dispatch.py` — `_select_output_mode` + `_parse_choice_action` + `_parse_speech_intent_action` (≈200 lines)
- `trace_builder.py` — `_build_action_trace` (≈100 lines)

Player.py keeps the main `act()` loop + retry orchestration + quality checks.

**Files:**
- Create: `werewolf_agent/agents/parse_dispatch.py`
- Create: `werewolf_agent/agents/trace_builder.py`
- Modify: `werewolf_agent/agents/player.py` (remove moved code, import from new modules)
- Test: existing tests in `tests/agents/test_player_agent.py` should still pass unchanged

- [ ] **Step 1: Run baseline test to confirm current pass**

```bash
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/agents/test_player_agent.py -q --tb=short
```

Expected: All pass (this is the baseline)

- [ ] **Step 2: Read current `_parse_choice_action` + `_parse_speech_intent_action` + `_select_output_mode`**

In `werewolf_agent/agents/player.py`, find these methods (around line 588-650). Read their full bodies. Note all imports they use.

- [ ] **Step 3: Create `parse_dispatch.py` with extracted functions**

Create `werewolf_agent/agents/parse_dispatch.py`:

```python
"""Output mode selection and parse dispatch for player agent.

Extracted from player.py to keep the main loop file focused on retry
orchestration and quality checks.
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    OutputMode,
    PlayerAction,
)


def select_output_mode(
    *,
    legal_actions: list[ActionType],
    task_type: Any,
    speech_intent_tasks: set,
) -> OutputMode:
    """Pick the simplest output mode that captures what the task needs.

    - TARGET_CHOICE: only one target-requiring action available
    - SPEECH_INTENT: task is in speech_intent_tasks
    - FULL_ACTION: default (full PlayerAction schema)
    """
    target_required = {
        ActionType.VOTE, ActionType.WOLF_KILL, ActionType.USE_POISON,
        ActionType.CHECK_ALIGNMENT, ActionType.CHOOSE_MASTER,
        ActionType.HUNTER_SHOT, ActionType.BADGE_TRANSFER, ActionType.SHERIFF_VOTE,
    }
    if legal_actions and all(a in target_required for a in legal_actions):
        if len(legal_actions) == 1:
            return OutputMode.TARGET_CHOICE
    if task_type in speech_intent_tasks:
        return OutputMode.SPEECH_INTENT
    return OutputMode.FULL_ACTION
```

(For the parse action functions, see the existing implementation in player.py and copy them over verbatim. The implementer must transcribe the actual bodies — do NOT use ellipsis or "...".)

- [ ] **Step 4: Create `trace_builder.py` with extracted function**

Create `werewolf_agent/agents/trace_builder.py`:

```python
"""ActionTrace construction helper extracted from PlayerAgent."""

from __future__ import annotations

from typing import Any

from werewolf_agent.agents.schemas import ActionTrace, RetryInfo
from werewolf_agent.agents.player import _normalize_retry_dict  # see step 5


def build_action_trace(
    *,
    raw_text: str,
    parsed_action: Any,
    final_action_type: Any,
    retry: RetryInfo | None,
    fallback_reason: str | None = None,
    fallback_target_used: bool = False,
    fallback_target_id: str | None = None,
    tool_call_required: bool = False,
    tool_call_received: bool = False,
    parse_success: bool = True,
    parse_error: str | None = None,
    retry_count: int = 0,
    structured_failure_reason: str | None = None,
) -> ActionTrace:
    """Build an ActionTrace from the current attempt's state.

    (Implementer: transcribe the body of _build_action_trace verbatim from
    player.py. The signature here matches the call sites in player.py.)
    """
    return ActionTrace(
        raw_text=raw_text,
        parsed_action=parsed_action,
        final_action_type=final_action_type.value if hasattr(final_action_type, "value") else final_action_type,
        retry=retry.model_dump() if retry else _normalize_retry_dict({}),
        fallback_reason=fallback_reason,
        fallback_target_used=fallback_target_used,
        fallback_target_id=fallback_target_id,
        tool_call_required=tool_call_required,
        tool_call_received=tool_call_received,
        parse_success=parse_success,
        parse_error=parse_error,
        retry_count=retry_count,
        structured_failure_reason=structured_failure_reason,
    )
```

(Implementer: the actual body must be transcribed from player.py `_build_action_trace` at line 530-580. Don't use placeholders.)

- [ ] **Step 5: Update `player.py` to import from new modules**

In `werewolf_agent/agents/player.py`:
- Remove the moved method bodies (`_select_output_mode`, `_parse_choice_action`, `_parse_speech_intent_action`, `_build_action_trace`)
- Add imports at the top:

```python
from werewolf_agent.agents.parse_dispatch import select_output_mode as _select_output_mode
from werewolf_agent.agents.parse_dispatch import parse_choice_action as _parse_choice_action
from werewolf_agent.agents.parse_dispatch import parse_speech_intent_action as _parse_speech_intent_action
from werewolf_agent.agents.trace_builder import build_action_trace as _build_action_trace
```

(Match the method signatures to whatever the implementer transcribed. If your signatures differ, adjust the imports accordingly.)

- [ ] **Step 6: Run agent tests, expect all pass unchanged**

```bash
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/agents/test_player_agent.py -q --tb=short
```

Expected: All pass (no behavior change, just file move)

- [ ] **Step 7: Add unit tests for the new modules**

In `tests/agents/test_parse_dispatch.py` (new file):

```python
"""Tests for the extracted parse_dispatch module."""

from werewolf_agent.agents.schemas import ActionType, TaskType, OutputMode
from werewolf_agent.agents.parse_dispatch import select_output_mode


class TestSelectOutputMode:
    def test_single_target_required_returns_target_choice(self):
        mode = select_output_mode(
            legal_actions=[ActionType.VOTE],
            task_type=TaskType.VOTE,
            speech_intent_tasks={TaskType.SPEECH},
        )
        assert mode == OutputMode.TARGET_CHOICE

    def test_speech_task_returns_speech_intent(self):
        mode = select_output_mode(
            legal_actions=[ActionType.SPEECH],
            task_type=TaskType.SPEECH,
            speech_intent_tasks={TaskType.SPEECH},
        )
        assert mode == OutputMode.SPEECH_INTENT

    def test_mixed_actions_returns_full_action(self):
        mode = select_output_mode(
            legal_actions=[ActionType.VOTE, ActionType.SPEECH],
            task_type=TaskType.DAY_SPEECH,
            speech_intent_tasks={TaskType.SPEECH},
        )
        assert mode == OutputMode.FULL_ACTION
```

In `tests/agents/test_trace_builder.py` (new file):

```python
"""Tests for the extracted trace_builder module."""

from werewolf_agent.agents.schemas import ActionType, RetryInfo
from werewolf_agent.agents.trace_builder import build_action_trace


class TestBuildActionTrace:
    def test_basic_trace(self):
        retry = RetryInfo(attempt=1, max_retries=3, error_code=None)
        trace = build_action_trace(
            raw_text='{"action_type": "vote"}',
            parsed_action=None,
            final_action_type=ActionType.VOTE,
            retry=retry,
            parse_success=True,
        )
        assert trace.raw_text == '{"action_type": "vote"}'
        assert trace.retry_count == 0
        assert trace.fallback_target_used is False

    def test_fallback_trace_marks_target(self):
        trace = build_action_trace(
            raw_text="",
            parsed_action=None,
            final_action_type=ActionType.VOTE,
            retry=None,
            fallback_reason="structure failure",
            fallback_target_used=True,
            fallback_target_id="p07",
        )
        assert trace.fallback_target_used is True
        assert trace.fallback_target_id == "p07"
```

- [ ] **Step 8: Run new + existing tests, expect all pass**

```bash
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/agents/test_parse_dispatch.py tests/agents/test_trace_builder.py tests/agents/test_player_agent.py -q --tb=short
```

Expected: All pass

- [ ] **Step 9: Verify line count reduction in player.py**

```bash
wc -l werewolf_agent/agents/player.py
```

Expected: significantly less than 988 (target: 750-800)

- [ ] **Step 10: Commit**

```bash
git add werewolf_agent/agents/player.py werewolf_agent/agents/parse_dispatch.py werewolf_agent/agents/trace_builder.py tests/agents/test_parse_dispatch.py tests/agents/test_trace_builder.py
git commit -m "refactor(player): extract parse_dispatch and trace_builder to reduce player.py to 750 lines"
```

---

### Task 5: Schema Union type for PlayerAction (optional — high risk)

**Root cause:** `PlayerAction` (schemas.py:147) forces 6 vote-only fields (`vote_basis`/`seer_stance`/`standing_with_seer`/`suspect_reason`/`not_voting_reason`/`private_reason`) on ALL actions, even when the LLM is producing a wolf_kill or speech. The schema noise increases parse error rate and confuses the LLM.

**Fix:** Introduce a discriminator field (`action_kind: Literal["vote", "speech", "wolf_kill", ...]`) and switch PlayerAction to a tagged Union. Update parsing + tool schema accordingly.

**Files:**
- Modify: `werewolf_agent/agents/schemas.py:147` (refactor PlayerAction)
- Modify: `werewolf_agent/agents/output_parser.py` (handle discriminated union)
- Modify: `werewolf_agent/agents/tool_schema.py` (advertise union)
- Modify: `werewolf_agent/agents/player.py` (handle Union return from parser)
- Test: `tests/agents/test_schemas.py` (new tests for union)

- [ ] **Step 1: Write failing tests for PlayerAction union**

In `tests/agents/test_schemas.py` (create if doesn't exist):

```python
"""Tests for PlayerAction schema as discriminated union."""

import pytest
from pydantic import ValidationError
from werewolf_agent.agents.schemas import (
    PlayerAction,
    VotePlayerAction,
    SpeechPlayerAction,
    WolfKillPlayerAction,
    CheckAlignmentPlayerAction,
    UsePoisonPlayerAction,
    ChooseMasterPlayerAction,
    HunterShotPlayerAction,
    BadgeTransferPlayerAction,
    SheriffVotePlayerAction,
    NoOpPlayerAction,
    ActionType,
)


class TestPlayerActionUnion:
    def test_vote_action_validates_with_vote_fields(self):
        action = VotePlayerAction(
            action_type=ActionType.VOTE,
            target_id="p05",
            vote_basis="speech_logic",
            seer_stance="undecided",
        )
        assert action.action_kind == "vote"

    def test_speech_action_does_not_require_vote_fields(self):
        action = SpeechPlayerAction(
            action_type=ActionType.SPEECH,
            speech="I think p05 is suspicious",
        )
        assert action.action_kind == "speech"
        assert action.vote_basis is None  # not a vote action

    def test_parse_vote_payload_returns_vote_action(self):
        data = {
            "action_type": "vote",
            "target_id": "p05",
            "vote_basis": "speech_logic",
            "seer_stance": "undecided",
        }
        action = PlayerAction.model_validate(data)
        assert isinstance(action, VotePlayerAction)

    def test_parse_speech_payload_returns_speech_action(self):
        data = {
            "action_type": "speech",
            "speech": "hello world",
        }
        action = PlayerAction.model_validate(data)
        assert isinstance(action, SpeechPlayerAction)

    def test_discriminator_required(self):
        with pytest.raises(ValidationError):
            PlayerAction.model_validate({"action_type": "vote", "target_id": "p05"})
```

- [ ] **Step 2: Run tests, expect FAIL (no union types yet)**

```bash
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/agents/test_schemas.py -v
```

Expected: FAIL with `ImportError` on the new types

- [ ] **Step 3: Define the 10 action variants in schemas.py**

In `werewolf_agent/agents/schemas.py`, add a base class and 10 variants (each a thin Pydantic model). See the test file for the expected types. Each variant has:
- `action_type: ActionType` (literal)
- `action_kind: Literal["vote" | ...]` (discriminator)
- Action-type-specific fields (e.g., VotePlayerAction has vote_basis/seer_stance; WolfKillPlayerAction doesn't)

Example structure:

```python
class _BasePlayerAction(BaseModel):
    action_type: ActionType
    target_id: str | None = None
    speech: str = ""
    reason: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    private_intent: PrivateIntent | None = None


class VotePlayerAction(_BasePlayerAction):
    action_type: Literal[ActionType.VOTE] = ActionType.VOTE
    action_kind: Literal["vote"] = "vote"
    target_id: str  # required for vote
    vote_basis: VoteBasis | None = None
    seer_stance: SeerStance | None = None
    standing_with_seer: str = ""
    suspect_reason: str = ""
    not_voting_reason: str = ""
    private_reason: str = ""


class SpeechPlayerAction(_BasePlayerAction):
    action_type: Literal[ActionType.SPEECH] = ActionType.SPEECH
    action_kind: Literal["speech"] = "speech"


# ... similar for WolfKillPlayerAction, CheckAlignmentPlayerAction, UsePoisonPlayerAction,
#     ChooseMasterPlayerAction, HunterShotPlayerAction, BadgeTransferPlayerAction,
#     SheriffVotePlayerAction, NoOpPlayerAction


PlayerAction = Annotated[
    Union[
        VotePlayerAction, SpeechPlayerAction, WolfKillPlayerAction,
        CheckAlignmentPlayerAction, UsePoisonPlayerAction, ChooseMasterPlayerAction,
        HunterShotPlayerAction, BadgeTransferPlayerAction, SheriffVotePlayerAction,
        NoOpPlayerAction,
    ],
    Field(discriminator="action_kind"),
]
```

(Implementer: write all 10 variants following the same pattern. Use Literal types for the action_type field. Keep the union discriminated by `action_kind`.)

- [ ] **Step 4: Update `action_from_data` to use the new union**

In `werewolf_agent/agents/output_parser.py`, find `action_from_data` (around line 216). Change to use `PlayerAction.model_validate(data)` instead of `PlayerAction(**data)`. The discriminated union handles routing automatically.

- [ ] **Step 5: Update `tool_schema.py` to advertise the union**

In `werewolf_agent/agents/tool_schema.py`, find the function that builds the JSON schema for the LLM. Replace the flat `PlayerAction` reference with the union's JSON schema. (Pydantic v2 generates this automatically via `PlayerAction.model_json_schema()`.)

- [ ] **Step 6: Update existing tests that reference vote-only fields on non-vote actions**

Search for tests that check `private_reason` or `vote_basis` on non-vote PlayerAction instances. They may need to use the specific variant. (This step may be a no-op if existing tests are well-typed.)

- [ ] **Step 7: Run full agent suite, expect pass**

```bash
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/agents/ -q --tb=short
```

Expected: All pass (the union is type-compatible with all existing usage; existing PlayerAction(...) calls now return the variant)

- [ ] **Step 8: Run runtime suite, expect pass**

```bash
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/runtime/ -q --tb=short
```

Expected: All pass

- [ ] **Step 9: Commit**

```bash
git add werewolf_agent/agents/schemas.py werewolf_agent/agents/output_parser.py werewolf_agent/agents/tool_schema.py werewolf_agent/agents/player.py tests/agents/test_schemas.py
git commit -m "refactor(schemas): PlayerAction as discriminated union of 10 action variants"
```

---

## Execution Order (recommended)

1. Task 1 (smart retry) — smallest, fastest win, sets up RetryInfo extension
2. Task 2 (per-player metrics) — independent, uses new RetryInfo fields
3. Task 3 (empty_response attribution) — independent, lives in model_gateway
4. Task 4 (player.py split) — refactor after 1+2 are stable, makes future changes easier
5. Task 5 (schema Union) — last, depends on stable interfaces

Tasks 1-3 are independent and can be dispatched in parallel via separate subagents. Tasks 4-5 must run after 1-3 are committed.

## Verification (end of plan)

```bash
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/ -q --tb=short
```

Plus a fresh end-to-end game:
```bash
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/integration/test_live_game_flow.py -q --tb=short
```

## Open Risks

- Task 5 (schema Union) is the highest-risk task. If Pydantic's discriminated union has issues with the existing parse flow, it could break all agent output. **Mitigation**: Task 4's refactor creates a cleaner surface area for Task 5, and the union is type-compatible with the old `PlayerAction` (existing call sites work unchanged). If Task 5 fails, keep the flat `PlayerAction` and revisit.
- Task 4's transcription steps (Step 3, 4) require reading the actual method bodies from player.py. If the implementer uses ellipsis/placeholders, the build will fail. The instruction explicitly says "transcribe verbatim — do NOT use ellipsis".
- Per-player metrics (Task 2) is memory-only. If the game session is long, memory grows. Acceptable for V1 (sessions are bounded by game length).
