# Game Quality Fixes — Design Spec

**Date:** 2026-05-27
**Trigger:** g_1123488389 game log analysis — wolf victory (slaughter_gods), 6 defects found

---

## Issue #1: Dead Players Speaking in Free Discussion

### Root Cause
`free_discussion` in `day.py` auto-populates `speech_order` via `choose_sheriff_led_speech_order`/`choose_no_sheriff_speech_order` without filtering dead players. Dead players remain in `gs.players` with `alive=False` but are not excluded from the order.

### Fix
**File:** `werewolf_agent/runtime/sheriff_policy.py`

In both `choose_sheriff_led_speech_order` and `choose_no_sheriff_speech_order`, filter out dead players:

```python
alive = [pid for pid in players if gs.players[pid].alive]
```

**File:** `werewolf_agent/runtime/nodes/day.py`

In `free_discussion`, when auto-populating speech_order via agent call, also filter:

```python
speech_order = [pid for pid in (agent_order or choose_...) if gs.players[pid].alive]
```

### Verification
- Unit test: speech_order excludes dead players when populated via all three code paths
- Unit test: dead player does not appear in free_discussion speaker queue

---

## Issue #2: Hunter Shot Crash (AttributeError)

### Root Cause
`skills.py` `resolve_hunter_shot` (~line 75) calls `_dispatch_agent(state, agent_hunter_shot, death.player_id, ...)`. `_dispatch_agent` returns `dict[str, Any] | None`, but the caller assigns the result directly to `target` and later checks `target in gs.players`. When the agent adapter returns an unexpected type (e.g., `list`), the subsequent `.get()` call crashes.

The agent adapter `agent_hunter_shot` in `agent_adapter.py` was returning inconsistent types under error conditions.

### Fix
**File:** `werewolf_agent/runtime/nodes/skills.py`

Guard the `_dispatch_agent` return value:

```python
shot_result = _dispatch_agent(
    shot_state, agent_hunter_shot, death.player_id,
    timeout_override=AGENT_TIMEOUTS.hunter_shot,
)
if isinstance(shot_result, dict):
    target = shot_result.get("hunter_shot_target_id")
elif isinstance(shot_result, str):
    target = shot_result
else:
    target = None
```

**File:** `werewolf_agent/runtime/agent_adapter.py`

Ensure `agent_hunter_shot` always returns `dict[str, Any]` or `None` — never a list.

### Verification
- Unit test: `resolve_hunter_shot` handles `_dispatch_agent` returning `None`, `dict`, `str`, and `list` without crashing
- Unit test: `agent_hunter_shot` return type consistency

---

## Issue #3: Seer Speech Quality — "按顺序验"

### Root Cause
The seer's sheriff election speech lacks structured guidance on explaining verification rationale. The prompt says "必须解释上警原因" but does not enforce concrete reasoning for each night's check choice.

### Fix
**File:** `werewolf_agent/runtime/agent_adapter.py` `agent_sheriff_election_speech`

When the agent's role is seer, add to `strategy_directive`:

```
你的查验理由必须具体——不能说"按顺序验"。每一夜的查验目标必须有一个
明确的理由，例如："N1验p03是因为他在警下位置靠前、发言量预期较大，
我需要确认他的身份来建立信息基点"。如果没有特定理由，可以说"首夜查
验随机选择，但我选择了发声量可能较大的位置以尽早获取信息"。
```

### New RoleStateMonitor alert: `SEER_UNDER_PRESSURE`
See Issue #7 (RoleStateMonitor).

### Verification
- Agent test: seer sheriff speech contains non-trivial verification rationale
- Integration test: seer under pressure triggers RoleStateMonitor alert

---

## Issue #4: Witch Poison Unused

### Root Cause
Witch context building (`_build_witch_context`) already includes poison pressure targets, but the strategy directive does not escalate urgency when the witch is in danger of being exiled with unused poison.

### Fix
**File:** `werewolf_agent/runtime/agent_adapter.py` `agent_night_witch` / `_build_witch_context`

Add urgency directive when poison unused AND alive count ≤ 8:

```
你还有一瓶毒药未使用。场上形势严峻——存活玩家已不多。如果你今晚不撒毒而被
狼人刀杀或被放逐，毒药将彻底浪费。你必须今晚选择一个目标使用毒药。
```

### New RoleStateMonitor alert: `WITCH_POISON_UNUSED_AT_RISK`
See Issue #7 (RoleStateMonitor).

### Verification
- Agent test: witch with unused poison and ≤8 alive receives urgency directive
- Agent test: witch with used poison does not receive urgency directive

---

## Issue #5: Sheriff Election Chaos — 8 Candidates

### Root Cause
Two issues: (a) No filter on candidate quality — anyone can register regardless of speech quality, (b) Agent templates produce mechanically identical speeches ("我这轮先把视角压到XX身上").

### Fix
**File:** `werewolf_agent/runtime/sheriff_policy.py`

Add `filter_sheriff_candidates(state, candidates)`: strip candidates whose sheriff speech matches template patterns or is shorter than 30 characters. If filtering leaves zero candidates → badge is lost.

```python
_TEMPLATE_PATTERNS = [
    r"我这轮先把视角压到",
    r"依据是.*最近发言",
]

def filter_sheriff_candidates(state, candidates, speeches):
    kept = []
    for c in candidates:
        text = speeches.get(c, "")
        if len(text) < 30:
            continue
        if any(re.search(p, text) for p in _TEMPLATE_PATTERNS):
            continue
        kept.append(c)
    return kept
```

**File:** `werewolf_agent/runtime/agent_adapter.py` `agent_sheriff_election_speech`

Add anti-template constraint to `strategy_directive`:

```
禁止使用以下模板句式，这些会让你的发言看起来机械且不可信：
- "我这轮先把视角压到XX身上"
- "依据是XX最近发言：..."
你的发言必须有独特的角度和独立的分析。
```

### Verification
- Unit test: `filter_sheriff_candidates` removes template-matching speeches
- Unit test: `filter_sheriff_candidates` keeps valid speeches
- Agent test: sheriff election speech does not contain template patterns

---

## Issue #6: Hybrid Zero Impact

### Root Cause
The hybrid's `build_agent_context` does not provide post-master-death guidance. When the master dies, the hybrid is effectively a vanilla villager but receives no strategic framing.

### Fix
**File:** `werewolf_agent/runtime/agent_adapter.py` `build_agent_context` (hybrid path)

When master is dead, add to strategy directive:

```
你的主人 [master_name] 已经死亡。你以 [faction] 阵营的身份继续游戏。
你现在等同于一个村民——你没有特殊技能，但你的分析和投票对你的阵营
至关重要。回顾主人的行为模式：主人是 [master_role]，他的行为是否
与这个身份一致？从主人的遗言和投票记录中寻找线索。
```

### New RoleStateMonitor alert: `HYBRID_MASTER_DEAD`
See Issue #7 (RoleStateMonitor).

---

## Issue #7 (New): RoleStateMonitor Module

### Purpose
Proactive role-specific state detection and strategic intervention. Sits in the CognitivePipeline between SalienceEngine and ContradictionEngine.

### Location
`werewolf_agent/cognition/role_monitor.py`

### Interface

```python
@dataclass(frozen=True)
class RoleAlert:
    alert_type: str      # SEER_UNDER_PRESSURE | WITCH_POISON_UNUSED_AT_RISK | ...
    severity: str        # "critical" | "warning" | "info"
    message: str         # Human-readable directive injected into strategy_directive
    evidence: list[str]  # Specific events/speeches that triggered this alert


class RoleStateMonitor:
    def __init__(self, ruleset: Ruleset) -> None: ...

    def assess(
        self,
        gs: GameState,
        player_id: str,
        role: str,
        phase: str,
    ) -> list[RoleAlert]:
        """Return alerts for this player in current game state."""
```

### Alert Types

| Alert | Trigger | Severity |
|-------|---------|----------|
| `SEER_UNDER_PRESSURE` | Player is seer, ≥2 public speech events this day question/suspect this seer, seer has unreported check results | critical |
| `WITCH_POISON_UNUSED_AT_RISK` | Player is witch, poison unused, alive ≤ 8, ≥2 speakers this day targeting this witch | critical |
| `HUNTER_MAY_DIE_SOON` | Player is hunter, alive ≤ 6, hunter is mentioned in ≥1 speech this day as potential target or "high value" | warning |
| `HYBRID_MASTER_DEAD` | Player is hybrid, master is dead, alive ≤ 8 | info |

### Detection Logic (deterministic)

All detection is keyword/regex-based on `gs.events` — no LLM calls:

- "Questioning/suspecting": regex matches for `怀疑|标狼|狼面|投票.*出|投.*出局|有问题` against the target player ID or display name
- "Unreported check results": compare seer_check events vs speech/sheriff_speech events to see which checks have been publicly reported
- "Poison unused": `not gs.poison_used`
- "Targeting witch": same regex as questioning, against witch's ID/name

### Integration

**File:** `werewolf_agent/cognition/pipeline.py` `CognitivePipeline.run()`

```python
monitor = RoleStateMonitor(engine.ruleset)
alerts = monitor.assess(gs, player_id, role, phase)
context.alerts = context.alerts + alerts
```

**File:** `werewolf_agent/runtime/agent_adapter.py` `build_agent_context()`

Append alert messages to `strategy_directive` under a new key `role_alerts`.

### Verification
- Unit test per alert type: trigger condition produces alert, non-trigger produces empty list
- Unit test: multiple alerts can fire simultaneously
- Integration test: RoleStateMonitor in pipeline produces alerts for real game scenario

---

## File Change Summary

| File | Change |
|------|--------|
| `werewolf_agent/runtime/sheriff_policy.py` | `filter_sheriff_candidates` + alive filter in speech order functions |
| `werewolf_agent/runtime/nodes/day.py` | `free_discussion` alive filter in speech order |
| `werewolf_agent/runtime/nodes/skills.py` | `resolve_hunter_shot` return type guard |
| `werewolf_agent/runtime/agent_adapter.py` | Seer/witch/hybrid strategy directives; `agent_hunter_shot` return type fix; anti-template constraints |
| `werewolf_agent/cognition/role_monitor.py` | **New file** — RoleStateMonitor + RoleAlert |
| `werewolf_agent/cognition/pipeline.py` | Wire RoleStateMonitor into pipeline |
| `tests/runtime/test_sheriff_policy.py` | `filter_sheriff_candidates` + alive filter tests |
| `tests/runtime/test_runtime.py` | Dead player speech exclusion test |
| `tests/runtime/test_hunter_shot.py` | Hunter shot crash resilience test (new file or added to existing) |
| `tests/agents/test_agents.py` | Seer/witch/hybrid directive tests |
| `tests/cognition/test_role_monitor.py` | **New file** — RoleStateMonitor unit tests |

## Verification

```powershell
python -m pytest tests/cognition/test_role_monitor.py -q
python -m pytest tests/runtime/test_sheriff_policy.py -q
python -m pytest tests/runtime/test_runtime.py -q
python -m pytest tests/agents/test_agents.py -q
python -m pytest tests/ -q --tb=short
```
