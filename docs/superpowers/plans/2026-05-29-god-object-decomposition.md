# God Object Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the 4 most responsibility-bloated modules into focused, single-concern files without breaking any existing imports or tests.

**Architecture:** Each decomposition follows the same pattern: (1) create new module with extracted code, (2) add backward-compatible re-exports in the original file, (3) verify all existing tests pass, (4) remove re-exports only after all callers are migrated. No functional behavior changes — pure structural refactoring.

**Tech Stack:** Python 3.12, pytest, existing codebase patterns

---

## File Structure After Decomposition

### `werewolf_agent/runtime/` (agent_adapter.py → 5 files)

```
runtime/
  agent_adapter.py          # AgentRegistry + agent_* entry points + re-exports (slim ~400 lines)
  directives/
    __init__.py              # Public API: build_day_speech_directive(role, ...) → str
    seer.py                  # _build_seer_day_speech_directive → build_seer_directive
    wolf.py                  # _build_wolf_day_speech_directive, _build_wolf_vote_strategy
    hunter.py                # _build_hunter_day_speech_directive
    hybrid.py                # _build_hybrid_day_speech_directive
    villager.py              # _build_villager_day_speech_directive
    idiot.py                 # _build_idiot_day_speech_directive
    _shared.py               # _collect_public_vote_history, _collect_death_order, shared helpers
  strategy/
    __init__.py              # Public API: evaluate_* functions
    witch.py                 # _estimate_witch_save_value, _build_witch_pressure_targets
    seer.py                  # _evaluate_seer_check_value, _public_seer_claimants
    wolf.py                  # _evaluate_wolf_kill_target, _get_wolf_role_assignment, _has_publicly_claimed_seer
    hunter.py                # _evaluate_hunter_shot_target
    hybrid.py                # _evaluate_hybrid_master_candidates
    death.py                 # _evaluate_death_cause_claims
  context.py                 # build_agent_context + persona/RAG/memory injection helpers
```

### `werewolf_agent/agents/` (player.py → 3 files)

```
agents/
  player.py                  # PlayerAgent class (slim ~500 lines, delegates parsing)
  output_parser.py           # JSON repair, choice/speech-intent parsing, enum cleaning
  tool_schema.py             # _player_action_tool, _vote_audit_tool_properties
```

### `werewolf_agent/skills/` (werewolf_skills.py → table-driven refactor)

```
skills/
  werewolf_skills.py         # Refactored: table-driven handlers, ~400 lines
```

### `werewolf_agent/rag/` (ingestion.py → 2 files + data)

```
rag/
  ingestion.py               # CaseIngester + IngestionError only (~160 lines)
  seed_data.py               # create_seed_entries() loader from YAML
config/rag_seeds/
  seed_entries.yaml           # All seed entry data
```

---

## Task 1: Extract Role Speech Directives from agent_adapter.py

**Files:**
- Create: `werewolf_agent/runtime/directives/__init__.py`
- Create: `werewolf_agent/runtime/directives/_shared.py`
- Create: `werewolf_agent/runtime/directives/seer.py`
- Create: `werewolf_agent/runtime/directives/wolf.py`
- Create: `werewolf_agent/runtime/directives/hunter.py`
- Create: `werewolf_agent/runtime/directives/hybrid.py`
- Create: `werewolf_agent/runtime/directives/villager.py`
- Create: `werewolf_agent/runtime/directives/idiot.py`
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Test: `pytest tests/runtime/ -q`

- [ ] **Step 1: Create `runtime/directives/__init__.py` with public API**

```python
"""Role-specific speech directive builders.

Each builder produces a Chinese-language strategy directive string
for a given role's day speech, vote, or night action context.
"""

from werewolf_agent.runtime.directives.seer import build_seer_directive
from werewolf_agent.runtime.directives.wolf import (
    build_wolf_directive,
    build_wolf_vote_directive,
)
from werewolf_agent.runtime.directives.hunter import build_hunter_directive
from werewolf_agent.runtime.directives.hybrid import build_hybrid_directive
from werewolf_agent.runtime.directives.villager import build_villager_directive
from werewolf_agent.runtime.directives.idiot import build_idiot_directive

__all__ = [
    "build_seer_directive",
    "build_wolf_directive",
    "build_wolf_vote_directive",
    "build_hunter_directive",
    "build_hybrid_directive",
    "build_villager_directive",
    "build_idiot_directive",
]
```

- [ ] **Step 2: Create `runtime/directives/_shared.py` with shared helpers**

Move `_collect_public_vote_history` (L787-809) and `_collect_death_order` (L811-828) from `agent_adapter.py`. These are used by multiple directive builders. Signature stays identical; only the import paths change inside.

```python
"""Shared helpers for speech directive builders."""

from __future__ import annotations
from typing import Any


def collect_public_vote_history(gs: Any) -> list[dict[str, Any]]:
    """Collect public vote records grouped by day."""
    # Copy body from agent_adapter._collect_public_vote_history exactly
    ...


def collect_death_order(gs: Any) -> list[dict[str, Any]]:
    """Collect death records in order."""
    # Copy body from agent_adapter._collect_death_order exactly
    ...
```

- [ ] **Step 3: Create `runtime/directives/seer.py`**

Move `_build_seer_day_speech_directive` (L534-632) from `agent_adapter.py`. Rename to `build_seer_directive`. Adjust internal references to use `_shared` helpers. The function signature stays the same — it takes `gs`, `player_id`, `visible`, etc. and returns `dict[str, Any]` (the strategy directive dict, not a string).

```python
"""Seer speech directive builder."""

from __future__ import annotations
from typing import Any

from werewolf_agent.runtime.directives._shared import (
    collect_public_vote_history,
    collect_death_order,
)


def build_seer_directive(gs: Any, player_id: str, visible: dict[str, Any]) -> dict[str, Any]:
    """Build seer day speech strategy directive."""
    # Copy body from agent_adapter._build_seer_day_speech_directive
    ...
```

- [ ] **Step 4: Create the remaining 5 directive files**

Follow the same pattern for each:
- `wolf.py` — move `_build_wolf_day_speech_directive` (L1247-1345) → `build_wolf_directive` and `_build_wolf_vote_strategy` (L1346-1395) → `build_wolf_vote_directive`
- `hunter.py` — move `_build_hunter_day_speech_directive` (L634-673) → `build_hunter_directive`
- `hybrid.py` — move `_build_hybrid_day_speech_directive` (L675-714) → `build_hybrid_directive`
- `villager.py` — move `_build_villager_day_speech_directive` (L716-785) → `build_villager_directive`
- `idiot.py` — move `_build_idiot_day_speech_directive` (L830-893) → `build_idiot_directive`

Each file: copy the function body verbatim, rename the public function (drop the leading underscore), update any references to `_shared` helpers.

- [ ] **Step 5: Add backward-compatible re-exports in agent_adapter.py**

In `agent_adapter.py`, replace the moved function bodies with imports from the new modules:

```python
# Backward-compatible re-exports — will be removed after caller migration
from werewolf_agent.runtime.directives import (
    build_seer_directive as _build_seer_day_speech_directive,
    build_wolf_directive as _build_wolf_day_speech_directive,
    build_wolf_vote_directive as _build_wolf_vote_strategy,
    build_hunter_directive as _build_hunter_day_speech_directive,
    build_hybrid_directive as _build_hybrid_day_speech_directive,
    build_villager_directive as _build_villager_day_speech_directive,
    build_idiot_directive as _build_idiot_day_speech_directive,
)
```

Delete the original function definitions. The re-exports preserve all existing callers that import from `agent_adapter`.

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -q`

Expected: All tests pass. No behavioral change — functions are just re-exported from new locations.

- [ ] **Step 7: Commit**

```bash
git add werewolf_agent/runtime/directives/ werewolf_agent/runtime/agent_adapter.py
git commit -m "refactor: extract role speech directives into runtime/directives/"
```

---

## Task 2: Extract Strategy/Evaluation Functions from agent_adapter.py

**Files:**
- Create: `werewolf_agent/runtime/strategy/__init__.py`
- Create: `werewolf_agent/runtime/strategy/witch.py`
- Create: `werewolf_agent/runtime/strategy/seer.py`
- Create: `werewolf_agent/runtime/strategy/wolf.py`
- Create: `werewolf_agent/runtime/strategy/hunter.py`
- Create: `werewolf_agent/runtime/strategy/hybrid.py`
- Create: `werewolf_agent/runtime/strategy/death.py`
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Test: `pytest tests/runtime/ -q`

- [ ] **Step 1: Create `runtime/strategy/__init__.py` with public API**

```python
"""Strategy evaluation functions for agent decision support.

These are deterministic analysis helpers — the LLM still makes the final call.
"""

from werewolf_agent.runtime.strategy.witch import (
    estimate_witch_save_value,
    build_witch_pressure_targets,
)
from werewolf_agent.runtime.strategy.seer import (
    evaluate_seer_check_value,
    public_seer_claimants,
)
from werewolf_agent.runtime.strategy.wolf import (
    evaluate_wolf_kill_target,
    get_wolf_role_assignment,
    has_publicly_claimed_seer,
)
from werewolf_agent.runtime.strategy.hunter import evaluate_hunter_shot_target
from werewolf_agent.runtime.strategy.hybrid import evaluate_hybrid_master_candidates
from werewolf_agent.runtime.strategy.death import evaluate_death_cause_claims
```

- [ ] **Step 2: Create each strategy module**

Move each function group from `agent_adapter.py`:

- `witch.py` — `_estimate_witch_save_value` (L266-369) → `estimate_witch_save_value`, `_build_witch_pressure_targets` (L371-403) → `build_witch_pressure_targets`
- `seer.py` — `_evaluate_seer_check_value` (L435-532) → `evaluate_seer_check_value`, `_public_seer_claimants` (L405-433) → `public_seer_claimants`
- `wolf.py` — `_evaluate_wolf_kill_target` (L1055-1172) → `evaluate_wolf_kill_target`, `_get_wolf_role_assignment` (L1174-1185) → `get_wolf_role_assignment`, `_has_publicly_claimed_seer` (L1187-1245) → `has_publicly_claimed_seer`
- `hunter.py` — `_evaluate_hunter_shot_target` (L3005-3130) → `evaluate_hunter_shot_target`
- `hybrid.py` — `_evaluate_hybrid_master_candidates` (L2776-2854) → `evaluate_hybrid_master_candidates`
- `death.py` — `_evaluate_death_cause_claims` (L895-1053) → `evaluate_death_cause_claims`

Each: copy function body verbatim, rename (drop underscore prefix), keep same signature and return type.

- [ ] **Step 3: Add backward-compatible re-exports in agent_adapter.py**

```python
from werewolf_agent.runtime.strategy import (
    estimate_witch_save_value as _estimate_witch_save_value,
    build_witch_pressure_targets as _build_witch_pressure_targets,
    evaluate_seer_check_value as _evaluate_seer_check_value,
    public_seer_claimants as _public_seer_claimants,
    evaluate_wolf_kill_target as _evaluate_wolf_kill_target,
    get_wolf_role_assignment as _get_wolf_role_assignment,
    has_publicly_claimed_seer as _has_publicly_claimed_seer,
    evaluate_hunter_shot_target as _evaluate_hunter_shot_target,
    evaluate_hybrid_master_candidates as _evaluate_hybrid_master_candidates,
    evaluate_death_cause_claims as _evaluate_death_cause_claims,
)
```

Delete original function bodies.

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -q`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/runtime/strategy/ werewolf_agent/runtime/agent_adapter.py
git commit -m "refactor: extract strategy evaluation into runtime/strategy/"
```

---

## Task 3: Extract build_agent_context into runtime/context.py

**Files:**
- Create: `werewolf_agent/runtime/context.py`
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Test: `pytest tests/runtime/ -q`

- [ ] **Step 1: Create `runtime/context.py`**

Move the following from `agent_adapter.py` into `runtime/context.py`:

1. `_SPEECH_STYLE_HINTS` dict (L16-24)
2. `_SHERIFF_SPEECH_STYLE_OVERRIDES` dict (L26-32)
3. `_TASK_STYLE_HINTS` dict (L34-41)
4. `_PERSONA_PROFILES_CACHE` / `_PERSONA_PROFILES_LOCK` (L43-44)
5. `_load_persona_profile` (L47-65)
6. `_get_persona_speech_style` (L68-73)
7. `_get_persona_task_style` (L75-80)
8. `_rag_phase_for_task` (L127-138)
9. `_inject_seed_rag_hints` (L141-185)
10. `_profile_memory_hint` (L188-205)
11. `_reflection_memory_hints` (L208-257)
12. `_cognition_matrix_hint` (L230-258)
13. `_action_trace_payload` (L261-263)
14. `_inject_skill_output` (L1397-1467)
15. `_resolve_tool_skills` (L1469-1480)
16. `_build_skill_tool_defs` (L1482-1491)
17. `_merge_strategy_directive` (L1494-1501)
18. `build_agent_context` (L1504-1893)

The new `runtime/context.py` imports from `runtime/directives` and `runtime/strategy` instead of using the underscore-prefixed local functions.

```python
"""Agent context builder: assembles AgentContext from GameState.

Orchestrates visibility, private memory, belief state, contradiction
detection, skill injection, role monitoring, death cause evaluation,
cross-game memory, and RAG hint injection.
"""

from __future__ import annotations
# ... imports from directives/, strategy/, core schemas ...
# ... copy all helper functions listed above ...

from werewolf_agent.runtime.directives import (
    build_seer_directive,
    build_wolf_directive,
    # ... etc
)
from werewolf_agent.runtime.strategy import (
    estimate_witch_save_value,
    build_witch_pressure_targets,
    # ... etc
)
```

- [ ] **Step 2: Add backward-compatible re-exports in agent_adapter.py**

```python
from werewolf_agent.runtime.context import (
    build_agent_context,
    _SPEECH_STYLE_HINTS,
    _SHERIFF_SPEECH_STYLE_OVERRIDES,
    _TASK_STYLE_HINTS,
    _get_persona_speech_style,
    _get_persona_task_style,
)
```

Delete the original function/class definitions from `agent_adapter.py`.

- [ ] **Step 3: Update direct callers of moved functions**

Search for callers of `_get_persona_speech_style`, `_get_persona_task_style`, `_SPEECH_STYLE_HINTS` etc. outside `agent_adapter.py`. If any exist (check `prompt_builder.py`, test files), update their imports to point to `runtime.context`.

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -q`

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/runtime/context.py werewolf_agent/runtime/agent_adapter.py
git commit -m "refactor: extract build_agent_context into runtime/context.py"
```

---

## Task 4: Slim down agent_adapter.py to entry points only

**Files:**
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Test: `pytest tests/runtime/ tests/integration/ -q`

- [ ] **Step 1: Verify agent_adapter.py is now slim**

After Tasks 1-3, `agent_adapter.py` should contain only:
- `AgentRegistry` protocol + `SimpleAgentRegistry` (L108-124)
- `_action_result_to_dict` (L1896-1906)
- All `agent_*` entry-point functions (L1908-3570+)
- Re-exports from directives, strategy, context

If any helper functions remain that are only used by `agent_*` entry points, keep them in `agent_adapter.py` — they are private to this module.

- [ ] **Step 2: Migrate callers from re-exports to direct imports**

Update callers in `runtime/nodes/day.py`, `runtime/nodes/night.py`, `runtime/nodes/sheriff.py`, `runtime/nodes/skills.py`, `runtime/nodes/summary.py`, `runtime/nodes/_shared.py`, `runtime/game_runner.py` to import from the new modules instead of from `agent_adapter.py` where appropriate.

Example for `day.py`:
```python
# Before:
from werewolf_agent.runtime.agent_adapter import agent_day_speech, agent_day_vote, agent_exile_last_words
# After: (no change needed — agent_* functions stay in agent_adapter)
```

The `agent_*` functions stay in `agent_adapter.py`. Only the helpers they call have moved.

- [ ] **Step 3: Remove backward-compatible re-exports**

Once all callers import from the new modules, remove the `as _old_name` re-exports from `agent_adapter.py`.

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/ -q`

Expected: All tests pass. `agent_adapter.py` should now be ~400-500 lines (AgentRegistry + agent_* entry points only).

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/runtime/ werewolf_agent/agents/prompt_builder.py
git commit -m "refactor: slim agent_adapter.py to entry points, migrate callers"
```

---

## Task 5: Extract JSON Parsing from player.py into agents/output_parser.py

**Files:**
- Create: `werewolf_agent/agents/output_parser.py`
- Modify: `werewolf_agent/agents/player.py`
- Test: `pytest tests/agents/ -q`

- [ ] **Step 1: Create `agents/output_parser.py`**

Move the following methods from `PlayerAgent` into standalone functions in `output_parser.py`:

1. `_repair_json_text` (L510-540) → `repair_json_text`
2. `_extract_json_object_candidates` (L1114-1150) → `extract_json_object_candidates`
3. `_extract_parameter_tag_action` (L632-654) → `extract_parameter_tag_action`
4. `_normalize_action_data` (L614-630) → `normalize_action_data`
5. `_clean_enum_value` (L1102-1106) → `clean_enum_value`
6. `_clean_reason` (L900-904) → `clean_reason`
7. `_sanitize_optional_private_fields` (L1152-1183) → `sanitize_optional_private_fields`

Each becomes a module-level function with the same signature minus `self`. The first parameter becomes the relevant input (e.g., `text: str` instead of `self` + `text`).

```python
"""LLM output parsing and repair for structured player actions.

Handles JSON extraction from markdown-wrapped responses, field
normalization, and fallback repair logic.
"""

from __future__ import annotations
import json
import re
from typing import Any


def repair_json_text(text: str) -> str:
    """Attempt to repair common JSON formatting issues."""
    # Copy body from PlayerAgent._repair_json_text
    ...


def extract_json_object_candidates(text: str) -> list[str]:
    """Extract all JSON object candidates from text."""
    # Copy body from PlayerAgent._extract_json_object_candidates
    ...

# ... etc
```

- [ ] **Step 2: Delegate from PlayerAgent to output_parser**

In `player.py`, replace method bodies with calls to the new module functions:

```python
from werewolf_agent.agents.output_parser import (
    repair_json_text as _repair_json_text_impl,
    extract_json_object_candidates as _extract_json_impl,
    # ...
)

class PlayerAgent:
    def _repair_json_text(self, text: str) -> str:
        return _repair_json_text_impl(text)

    def _extract_json_object_candidates(self, text: str) -> list[str]:
        return _extract_json_impl(text)
```

This preserves the existing API surface while delegating to the new module.

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/agents/ tests/runtime/ -q`

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add werewolf_agent/agents/output_parser.py werewolf_agent/agents/player.py
git commit -m "refactor: extract JSON parsing into agents/output_parser.py"
```

---

## Task 6: Extract Choice/Speech-Intent Parsing from player.py

**Files:**
- Modify: `werewolf_agent/agents/output_parser.py`
- Modify: `werewolf_agent/agents/player.py`
- Test: `pytest tests/agents/ -q`

- [ ] **Step 1: Move choice pipeline functions to output_parser.py**

Move these methods from `PlayerAgent` to `output_parser.py` as standalone functions:

1. `_parse_choice_action` (L676-707) → `parse_choice_action`
2. `_parse_speech_intent_action` (L709-728) → `parse_speech_intent_action`
3. `_extract_decision_data` (L730-754) → `extract_decision_data`
4. `_repair_vote_decision` (L756-807) → `repair_vote_decision`
5. `_repair_target_decision` (L809-835) → `repair_target_decision`
6. `_repair_speech_intent_decision` (L837-865) → `repair_speech_intent_decision`
7. `_vote_choice_map` (L867-872) → `vote_choice_map`
8. `_target_from_vote_decision` (L874-892) → `target_from_vote_decision`
9. `_choice_for_target` (L894-898) → `choice_for_target`
10. `_vote_candidate_summary` (L906-926) → `vote_candidate_summary`
11. `_target_candidate_summary` (L928-947) → `target_candidate_summary`
12. `_infer_speech_intent` (L949-963) → `infer_speech_intent`
13. `_speech_target_from_decision` (L965-977) → `speech_target_from_decision`
14. `_synthesize_intent_speech` (L979-1024) → `synthesize_intent_speech`
15. `_ensure_speech_quality_components` (L1026-1048) → `ensure_speech_quality_components`
16. `_speech_pressure_target` (L1050-1060) → `speech_pressure_target`
17. `_speech_intent_reason` (L1062-1066) → `speech_intent_reason`
18. `_infer_standing_with_seer` (L1068-1076) → `infer_standing_with_seer`
19. `_infer_seer_stance` (L1078-1087) → `infer_seer_stance`
20. `_infer_vote_basis` (L1089-1100) → `infer_vote_basis`
21. `_default_not_voting_reason` (L1108-1112) → `default_not_voting_reason`

Each becomes a standalone function. `self` references that accessed `legal_actions` / `legal_targets` become explicit parameters.

- [ ] **Step 2: Delegate from PlayerAgent**

Same pattern as Task 5 — thin delegation wrappers in `PlayerAgent` calling the standalone functions.

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/agents/ -q`

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add werewolf_agent/agents/output_parser.py werewolf_agent/agents/player.py
git commit -m "refactor: extract choice/speech-intent parsing into output_parser.py"
```

---

## Task 7: Extract Tool Schema Generation from player.py

**Files:**
- Create: `werewolf_agent/agents/tool_schema.py`
- Modify: `werewolf_agent/agents/player.py`
- Test: `pytest tests/agents/ -q`

- [ ] **Step 1: Create `agents/tool_schema.py`**

Move these methods from `PlayerAgent`:

1. `_player_action_tool` (L1185-1277) → `player_action_tool`
2. `_vote_audit_tool_properties` (L1279-1307) → `vote_audit_tool_properties`
3. `_speech_quality_error` (L1309-1331) → `speech_quality_error`
4. `_speech_quality_phase` (L1333-1340) → `speech_quality_phase`
5. `_vote_quality_error` (L1342-1365) → `vote_quality_error`
6. `_all_legal_actions_require_target` (L1367-1371) → `all_legal_actions_require_target`

These are pure functions that build tool schema dicts — they have no mutable state dependency on `PlayerAgent`.

- [ ] **Step 2: Delegate from PlayerAgent**

```python
from werewolf_agent.agents.tool_schema import (
    player_action_tool as _build_tool,
    vote_audit_tool_properties,
)
```

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/agents/ -q`

Expected: All tests pass. `player.py` should now be ~500-600 lines.

- [ ] **Step 4: Commit**

```bash
git add werewolf_agent/agents/tool_schema.py werewolf_agent/agents/player.py
git commit -m "refactor: extract tool schema generation into agents/tool_schema.py"
```

---

## Task 8: Refactor werewolf_skills.py to Table-Driven Pattern

**Files:**
- Modify: `werewolf_agent/skills/werewolf_skills.py`
- Test: `pytest tests/skills/ -q`

- [ ] **Step 1: Identify the repetitive pattern**

The current file has 12 skills, each with a `_X_handler → _X_static → _X_dynamic` triad (36 functions). The handler selects between static and dynamic based on whether `game_state` is provided. The static handler returns a fixed output; the dynamic handler inspects game state.

This can be replaced with a registration table:

```python
from typing import Callable

_SkillHandler = Callable[..., SkillOutput]

_SKILL_REGISTRY: dict[SkillName, _SkillHandler] = {}


def register_skill(name: SkillName) -> Callable[[_SkillHandler], _SkillHandler]:
    """Decorator to register a skill handler."""
    def decorator(fn: _SkillHandler) -> _SkillHandler:
        _SKILL_REGISTRY[name] = fn
        return fn
    return decorator


def get_skill_handler(name: SkillName) -> _SkillHandler | None:
    return _SKILL_REGISTRY.get(name)
```

- [ ] **Step 2: Convert each skill to a single registered handler**

Each handler merges the static/dynamic logic into one function that checks `game_state is None` internally:

```python
@register_skill(SkillName.BOLD_CLAIM)
def bold_claim(input: SkillInput) -> SkillOutput:
    if input.game_state is None:
        return SkillOutput(
            recommended_action="bold_claim_role",
            speech_structure="先报身份，再报信息，最后施压",
            risk_alerts=["悍跳风险高，必须有可靠信息支撑"],
            confidence=0.6,
        )
    # Dynamic logic (previously in _bold_claim_dynamic)
    ...
```

Remove all `_X_static` and `_X_dynamic` functions. Keep only the registered handlers. This cuts ~36 functions down to 12.

- [ ] **Step 3: Update dispatch_site_skill and SkillRegistry integration**

The `dispatch_site_skill` function currently calls `globals()[f"_{name.value}_handler"]`. Replace with `get_skill_handler(name)`.

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest tests/skills/ -q`

Expected: All tests pass. File should shrink from ~1068 to ~400-500 lines.

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/skills/werewolf_skills.py
git commit -m "refactor: table-driven skill handlers, eliminate repetitive static/dynamic triads"
```

---

## Task 9: Move Seed Data from ingestion.py to YAML

**Files:**
- Create: `config/rag_seeds/seed_entries.yaml`
- Create: `werewolf_agent/rag/seed_data.py`
- Modify: `werewolf_agent/rag/ingestion.py`
- Modify: `werewolf_agent/rag/knowledge_service.py` (update import)
- Test: `pytest tests/rag/ -q`

- [ ] **Step 1: Create YAML seed data file**

Convert all `create_seed_entries()` data (L158-1110) to YAML. Each entry becomes:

```yaml
- entry_id: seed_ext_seer_claim_01
  title: "预言家起跳声明警徽流标准打法"
  summary: "预言家在警上起跳，报查验结果和警徽流..."
  key_decisions:
    - "起跳时机：警上发言阶段"
    - "信息传递顺序：先报查验，再声明警徽流"
  # ... all other RAGEntry fields
```

This removes ~950 lines of Python data literals.

- [ ] **Step 2: Create `rag/seed_data.py` loader**

```python
"""Seed data loader: reads RAG seed entries from YAML."""

from __future__ import annotations
from pathlib import Path
from typing import Any

import yaml

from werewolf_agent.rag.ingestion import CaseIngester, IngestionError
from werewolf_agent.rag.schemas import RAGEntry


_SEED_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "rag_seeds" / "seed_entries.yaml"


def create_seed_entries() -> list[RAGEntry]:
    """Load and validate seed entries from YAML."""
    ingester = CaseIngester()
    entries: list[RAGEntry] = []

    if not _SEED_PATH.exists():
        return entries

    data = yaml.safe_load(_SEED_PATH.read_text(encoding="utf-8"))
    if not data:
        return entries

    for item in data:
        try:
            entry = RAGEntry(**item)
            ingester.ingest(entry)
            entries.append(entry)
        except (IngestionError, Exception) as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Seed entry %s failed validation: %s",
                item.get("entry_id", "?"), exc,
            )
    return entries
```

- [ ] **Step 3: Remove create_seed_entries from ingestion.py**

Delete `create_seed_entries()` (L158-1110) from `ingestion.py`. The file now contains only `IngestionError` and `CaseIngester` (~160 lines).

- [ ] **Step 4: Update knowledge_service.py import**

```python
# Before:
from werewolf_agent.rag.ingestion import create_seed_entries
# After:
from werewolf_agent.rag.seed_data import create_seed_entries
```

- [ ] **Step 5: Update test imports**

In `tests/rag/test_rag.py`, change:
```python
from werewolf_agent.rag.ingestion import CaseIngester, IngestionError, create_seed_entries
```
to:
```python
from werewolf_agent.rag.ingestion import CaseIngester, IngestionError
from werewolf_agent.rag.seed_data import create_seed_entries
```

Add a backward-compatible re-export in `ingestion.py` temporarily:
```python
from werewolf_agent.rag.seed_data import create_seed_entries  # backward compat
```

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/rag/ -q`

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add config/rag_seeds/ werewolf_agent/rag/seed_data.py werewolf_agent/rag/ingestion.py werewolf_agent/rag/knowledge_service.py tests/rag/test_rag.py
git commit -m "refactor: move RAG seed data to YAML, slim ingestion.py to validation only"
```

---

## Task 10: Final Verification and Cleanup

**Files:**
- All modified files
- Test: `pytest tests/ -q`

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -q`

Expected: All tests pass with 0 failures.

- [ ] **Step 2: Verify line counts**

Run:
```powershell
$files = @(
    "werewolf_agent/runtime/agent_adapter.py",
    "werewolf_agent/agents/player.py",
    "werewolf_agent/skills/werewolf_skills.py",
    "werewolf_agent/rag/ingestion.py"
)
foreach ($f in $files) { $c = (Get-Content $f).Count; Write-Output "$f : $c lines" }
```

Expected targets:
- `agent_adapter.py`: ~400-500 lines (was 3203)
- `player.py`: ~500-600 lines (was 1393)
- `werewolf_skills.py`: ~400-500 lines (was 1068)
- `ingestion.py`: ~160 lines (was 1110)

- [ ] **Step 3: Remove any remaining backward-compat re-exports that are no longer needed**

Search for `# backward compat` comments. If no external callers remain, remove the re-exports.

- [ ] **Step 4: Update PROGRESS.md**

Record the decomposition in `PROGRESS.md`:
- Changed files list
- Before/after line counts
- Verification command

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "refactor: God object decomposition — agent_adapter, player, skills, ingestion"
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Import breakage | Re-exports in original files until all callers migrated |
| Test regression | Full test suite run after every task |
| Circular imports | New modules only import from `core/`, `engine/`, `schemas`; never back into `agent_adapter` |
| Behavioral change | Pure structural refactoring — no logic changes, only moves |
| YAML seed data format errors | `CaseIngester` validation still runs on loaded entries |
