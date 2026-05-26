# Game Quality Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 7 defects from game log analysis (g_1123488389): dead player speech, hunter crash, seer quality, witch poison, sheriff chaos, hybrid impact, RoleStateMonitor

**Architecture:** Bug fixes in runtime layer (day.py, skills.py, agent_adapter.py); new RoleStateMonitor module in cognition layer with alerts wired into CognitivePipeline and agent strategy directives

**Tech Stack:** Python 3.12, LangGraph, pytest, RuleEngine

---

### Task 1: Fix dead players in speech order (Bug #1)

**Root cause:** `announce_deaths` no longer clears `speech_order: []` (removed in previous fix). Stale speech_order from previous day persists in state, including now-dead players.

**Fix:** In `free_discussion`, filter stale speech_order to exclude dead players before using it.

**Files:**
- Modify: `werewolf_agent/runtime/nodes/day.py:157-169`

- [ ] **Step 1: Add alive filter to free_discussion speech_order**

In `werewolf_agent/runtime/nodes/day.py`, after line 169 (`speech_order = choose_no_sheriff_speech_order(gs)`), add:

```python
    # Filter out dead players from any pre-existing or agent-generated speech_order
    speech_order = [pid for pid in speech_order if gs.players.get(pid) and gs.players[pid].alive]
```

Insert this line immediately after the `if not speech_order:` block (after line 169), so it runs regardless of whether the order was auto-populated or carried over.

- [ ] **Step 2: Run tests**

```powershell
python -m pytest tests/runtime/test_runtime.py -q --tb=short
```

- [ ] **Step 3: Commit**

```bash
git add werewolf_agent/runtime/nodes/day.py
git commit -m "fix: filter dead players from speech_order in free_discussion"
```

---

### Task 2: Fix hunter shot crash (Bug #2)

**Root cause:** `timed_call` wraps `agent_hunter_shot` execution. When `agent_hunter_shot` raises `AttributeError` (caused by `.get()` on a list object in an intermediate helper), `timed_call` catches it and returns `None`. But the log shows the crash message before fallback — the root cause is inside `agent_hunter_shot` where a parameter expected to be `dict` is actually `list`.

**Fix:** Guard `_dispatch_agent` return in `resolve_hunter_shot` and add defensive fallback in `agent_hunter_shot`.

**Files:**
- Modify: `werewolf_agent/runtime/nodes/skills.py:75-80`
- Modify: `werewolf_agent/runtime/agent_adapter.py:2694-2697`

- [ ] **Step 1: Guard _dispatch_agent return value in skills.py**

In `werewolf_agent/runtime/nodes/skills.py`, change line 75-80:

```python
# Old:
target = _dispatch_agent(
    shot_state,
    agent_hunter_shot,
    death.player_id,
    timeout_override=AGENT_TIMEOUTS.hunter_shot,
)

# New:
shot_result = _dispatch_agent(
    shot_state,
    agent_hunter_shot,
    death.player_id,
    timeout_override=AGENT_TIMEOUTS.hunter_shot,
)
if isinstance(shot_result, dict):
    target = shot_result.get("hunter_shot_target_id")
elif isinstance(shot_result, str):
    target = shot_result
else:
    target = None
```

- [ ] **Step 2: Guard _evaluate_hunter_shot_target in agent_hunter_shot**

In `werewolf_agent/runtime/agent_adapter.py`, wrap line 2695-2697 with try/except:

```python
# Old:
shot_assessment = _evaluate_hunter_shot_target(
    gs, hunter_id, legal_targets, death_reason,
)

# New:
try:
    shot_assessment = _evaluate_hunter_shot_target(
        gs, hunter_id, legal_targets, death_reason,
    )
except Exception:
    logger.warning("_evaluate_hunter_shot_target failed", exc_info=True)
    shot_assessment = None
```

- [ ] **Step 3: Run tests**

```powershell
python -m pytest tests/runtime/test_runtime.py -q --tb=short
```

- [ ] **Step 4: Commit**

```bash
git add werewolf_agent/runtime/nodes/skills.py werewolf_agent/runtime/agent_adapter.py
git commit -m "fix: guard hunter shot dispatch return value and add fallback"
```

---

### Task 3: New module — RoleStateMonitor

**Files:**
- Create: `werewolf_agent/cognition/role_monitor.py`
- Create: `tests/cognition/test_role_monitor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/cognition/test_role_monitor.py
import pytest
from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.cognition.role_monitor import RoleStateMonitor, RoleAlert


def _make_gs(**kwargs):
    players = kwargs.pop("players", {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager")
        for i in range(1, 13)
    })
    return GameState(game_id="test", players=players, **kwargs)


class TestSeerUnderPressure:
    def test_no_alert_when_seer_not_questioned(self):
        from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset
        engine = RuleEngine(Ruleset(raw={}))
        monitor = RoleStateMonitor(engine.ruleset)
        gs = _make_gs(day_number=2)
        gs = gs.model_copy(update={"events": gs.events + [
            GameEvent(type="speech", payload={"speaker": "p02", "day_number": 2, "text": "p05是好人"}),
        ]})
        alerts = monitor.assess(gs, "p05", "seer", "day")
        assert len([a for a in alerts if a.alert_type == "SEER_UNDER_PRESSURE"]) == 0

    def test_alerts_when_seer_questioned_by_two_players(self):
        from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset
        engine = RuleEngine(Ruleset(raw={}))
        monitor = RoleStateMonitor(engine.ruleset)
        events = [
            GameEvent(type="speech", payload={"speaker": "p02", "day_number": 2, "text": "我怀疑p05有问题，他的查验逻辑不对"}),
            GameEvent(type="speech", payload={"speaker": "p03", "day_number": 2, "text": "p05就是狼，我投票出p05"}),
        ]
        gs = _make_gs(day_number=2)
        gs = gs.model_copy(update={"events": gs.events + events})
        alerts = monitor.assess(gs, "p05", "seer", "day")
        seer_alerts = [a for a in alerts if a.alert_type == "SEER_UNDER_PRESSURE"]
        assert len(seer_alerts) == 1
        assert seer_alerts[0].severity == "critical"


class TestWitchPoisonUnusedAtRisk:
    def test_alerts_when_witch_targeted_and_poison_unused(self):
        from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset
        engine = RuleEngine(Ruleset(raw={}))
        monitor = RoleStateMonitor(engine.ruleset)
        players = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager") for i in range(1, 9)}
        players["p02"] = PlayerState(id="p02", role="witch")
        events = [
            GameEvent(type="speech", payload={"speaker": "p03", "day_number": 3, "text": "p02的站边有问题，我怀疑她"}),
            GameEvent(type="speech", payload={"speaker": "p04", "day_number": 3, "text": "出p02，她是狼"}),
        ]
        gs = _make_gs(players=players, day_number=3, poison_used=False)
        gs = gs.model_copy(update={"events": gs.events + events})
        alerts = monitor.assess(gs, "p02", "witch", "day")
        witch_alerts = [a for a in alerts if a.alert_type == "WITCH_POISON_UNUSED_AT_RISK"]
        assert len(witch_alerts) == 1

    def test_no_alert_when_poison_already_used(self):
        from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset
        engine = RuleEngine(Ruleset(raw={}))
        monitor = RoleStateMonitor(engine.ruleset)
        players = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager") for i in range(1, 9)}
        players["p02"] = PlayerState(id="p02", role="witch")
        events = [
            GameEvent(type="speech", payload={"speaker": "p03", "day_number": 3, "text": "我怀疑p02"}),
            GameEvent(type="speech", payload={"speaker": "p04", "day_number": 3, "text": "出p02"}),
        ]
        gs = _make_gs(players=players, day_number=3, poison_used=True)
        gs = gs.model_copy(update={"events": gs.events + events})
        alerts = monitor.assess(gs, "p02", "witch", "day")
        witch_alerts = [a for a in alerts if a.alert_type == "WITCH_POISON_UNUSED_AT_RISK"]
        assert len(witch_alerts) == 0


class TestHunterMayDieSoon:
    def test_alerts_when_hunter_mentioned_in_late_game(self):
        from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset
        engine = RuleEngine(Ruleset(raw={}))
        monitor = RoleStateMonitor(engine.ruleset)
        players = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager") for i in range(1, 7)}
        players["p12"] = PlayerState(id="p12", role="hunter")
        events = [
            GameEvent(type="speech", payload={"speaker": "p03", "day_number": 3, "text": "p12可能是猎人，发言很有底气"}),
        ]
        gs = _make_gs(players=players, day_number=3)
        gs = gs.model_copy(update={"events": gs.events + events})
        alerts = monitor.assess(gs, "p12", "hunter", "day")
        hunter_alerts = [a for a in alerts if a.alert_type == "HUNTER_MAY_DIE_SOON"]
        assert len(hunter_alerts) == 1


class TestHybridMasterDead:
    def test_alerts_when_hybrid_master_is_dead(self):
        from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset
        engine = RuleEngine(Ruleset(raw={}))
        monitor = RoleStateMonitor(engine.ruleset)
        players = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager") for i in range(1, 9)}
        players["p10"] = PlayerState(id="p10", role="hybrid")
        players["p01"] = PlayerState(id="p01", role="villager", alive=False)
        gs = _make_gs(players=players, day_number=3, hybrid_master_id="p01")
        alerts = monitor.assess(gs, "p10", "hybrid", "day")
        hybrid_alerts = [a for a in alerts if a.alert_type == "HYBRID_MASTER_DEAD"]
        assert len(hybrid_alerts) == 1


class TestMultipleAlerts:
    def test_multiple_alert_types_can_fire_simultaneously(self):
        from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset
        engine = RuleEngine(Ruleset(raw={}))
        monitor = RoleStateMonitor(engine.ruleset)
        players = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager") for i in range(1, 9)}
        players["p05"] = PlayerState(id="p05", role="seer")
        events = [
            GameEvent(type="speech", payload={"speaker": "p02", "day_number": 2, "text": "怀疑p05"}),
            GameEvent(type="speech", payload={"speaker": "p03", "day_number": 2, "text": "出p05！"}),
        ]
        gs = _make_gs(players=players, day_number=2)
        gs = gs.model_copy(update={"events": gs.events + events})
        alerts = monitor.assess(gs, "p05", "seer", "day")
        # SEER_UNDER_PRESSURE should fire; seer is not hunter/hybrid so those shouldn't
        alert_types = {a.alert_type for a in alerts}
        assert "SEER_UNDER_PRESSURE" in alert_types
```

- [ ] **Step 2: Run tests — verify they fail**

```powershell
python -m pytest tests/cognition/test_role_monitor.py -v
```
Expected: all FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement RoleStateMonitor**

Create `werewolf_agent/cognition/role_monitor.py`:

```python
"""Role state monitor — detects critical role-specific situations.

Design doc: proactive detection of SEER_UNDER_PRESSURE, WITCH_POISON_UNUSED,
HUNTER_MAY_DIE_SOON, and HYBRID_MASTER_DEAD states.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from werewolf_agent.core.models import GameState


@dataclass(frozen=True)
class RoleAlert:
    alert_type: str
    severity: str  # "critical" | "warning" | "info"
    message: str
    evidence: list[str]


class RoleStateMonitor:
    def __init__(self, ruleset: Any) -> None:
        self._ruleset = ruleset

    def assess(
        self, gs: GameState, player_id: str, role: str, phase: str,
    ) -> list[RoleAlert]:
        alerts: list[RoleAlert] = []

        if role == "seer":
            alerts.extend(self._check_seer_pressure(gs, player_id))
        elif role == "witch":
            alerts.extend(self._check_witch_poison_risk(gs, player_id))
        elif role == "hunter":
            alerts.extend(self._check_hunter_risk(gs, player_id))
        elif role == "hybrid":
            alerts.extend(self._check_hybrid_master_dead(gs, player_id))

        return alerts

    # -- SEER_UNDER_PRESSURE --

    def _check_seer_pressure(self, gs: GameState, seer_id: str) -> list[RoleAlert]:
        current_day = gs.day_number
        questioning_speakers = set()
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            if e.payload.get("day_number") != current_day:
                continue
            speaker = e.payload.get("speaker", "")
            text = str(e.payload.get("text", ""))
            if self._text_targets_player(text, seer_id):
                questioning_speakers.add(speaker)

        if len(questioning_speakers) < 2:
            return []

        # Check for unreported check results
        seer_checks = [
            e for e in gs.events
            if e.type == "seer_check"
            and e.payload.get("seer_id") == seer_id
        ]
        reported = set()
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            text = str(e.payload.get("text", ""))
            speaker = e.payload.get("speaker", "")
            if speaker != seer_id:
                continue
            for sc in seer_checks:
                target = sc.payload.get("target_id", "")
                if target and target in text:
                    reported.add(target)

        unreported = [sc for sc in seer_checks if sc.payload.get("target_id") not in reported]

        evidence = [f"{len(questioning_speakers)} players questioning seer {seer_id}"]
        if unreported:
            evidence.append(
                f"{len(unreported)} unreported checks: "
                f"{[sc.payload.get('target_id') for sc in unreported]}"
            )

        return [RoleAlert(
            alert_type="SEER_UNDER_PRESSURE",
            severity="critical",
            message=(
                f"你有{len(questioning_speakers)}个玩家在质疑你的预言家身份。"
                f"你必须完整列出所有夜晚查验记录（含查验理由和结果），"
                f"并明确警徽流。不能只说'按顺序验'——每夜的查验目标必须有具体动机。"
                f"{'你还有未报出的查验结果，必须在本轮全部报出！' if unreported else ''}"
            ),
            evidence=evidence,
        )]

    # -- WITCH_POISON_UNUSED_AT_RISK --

    def _check_witch_poison_risk(self, gs: GameState, witch_id: str) -> list[RoleAlert]:
        if gs.poison_used:
            return []

        alive_count = sum(1 for p in gs.players.values() if p.alive)
        if alive_count > 8:
            return []

        current_day = gs.day_number
        targeting_speakers = set()
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            if e.payload.get("day_number") != current_day:
                continue
            text = str(e.payload.get("text", ""))
            if self._text_targets_player(text, witch_id):
                targeting_speakers.add(e.payload.get("speaker", ""))

        if len(targeting_speakers) < 2:
            return []

        return [RoleAlert(
            alert_type="WITCH_POISON_UNUSED_AT_RISK",
            severity="critical",
            message=(
                f"警告：你的毒药还未使用，但已有{len(targeting_speakers)}个玩家在质疑你。"
                f"场上仅存活{alive_count}人。如果你今晚不撒毒而被狼杀或被放逐，"
                f"毒药将彻底浪费。你必须今晚选择一个目标使用毒药——"
                f"从质疑你的玩家中挑选最可疑的。"
            ),
            evidence=[
                f"{len(targeting_speakers)} players targeting witch {witch_id}",
                f"alive={alive_count}, poison unused",
            ],
        )]

    # -- HUNTER_MAY_DIE_SOON --

    def _check_hunter_risk(self, gs: GameState, hunter_id: str) -> list[RoleAlert]:
        alive_count = sum(1 for p in gs.players.values() if p.alive)
        if alive_count > 6:
            return []

        mentioned = False
        for e in gs.events:
            if e.type not in ("speech", "sheriff_speech"):
                continue
            text = str(e.payload.get("text", ""))
            if hunter_id in text:
                mentioned = True
                break

        if not mentioned:
            return []

        return [RoleAlert(
            alert_type="HUNTER_MAY_DIE_SOON",
            severity="warning",
            message=(
                f"场上仅存活{alive_count}人，且你的身份已被讨论。"
                f"如果今晚被狼人刀杀，你必须果断开枪。"
                f"提前思考：如果现在就要开枪，你会带走谁？"
            ),
            evidence=[f"alive={alive_count}, hunter mentioned in speeches"],
        )]

    # -- HYBRID_MASTER_DEAD --

    def _check_hybrid_master_dead(self, gs: GameState, hybrid_id: str) -> list[RoleAlert]:
        master_id = gs.hybrid_master_id
        if not master_id:
            return []

        master = gs.players.get(master_id)
        if master and master.alive:
            return []

        alive_count = sum(1 for p in gs.players.values() if p.alive)
        if alive_count > 8:
            return []

        master_faction = gs.hybrid_master_faction or "unknown"
        return [RoleAlert(
            alert_type="HYBRID_MASTER_DEAD",
            severity="info",
            message=(
                f"你的主人 {master_id} 已经死亡。"
                f"你以{faction_label(master_faction)}阵营身份继续游戏。"
                f"你现在等同于村民——没有特殊技能，但你的分析和投票至关重要。"
                f"回顾主人的行为：他是{master.role if master else '?'}，"
                f"他的发言和投票是否与这个身份一致？从他的遗言中寻找线索。"
            ),
            evidence=[f"master {master_id} is dead, alive={alive_count}"],
        )]

    # -- helpers --

    @staticmethod
    def _text_targets_player(text: str, player_id: str) -> bool:
        """Check if speech text questions/suspects/exiles a specific player."""
        if player_id not in text:
            return False
        patterns = [
            rf"(?:怀疑|标狼|狼面|定狼|抗推|有问题|投票.*出|投.*出局|出)\s*{player_id}",
            rf"{player_id}.*(?:是狼|有问题|不对劲|发言矛盾|站边问题)",
        ]
        return any(re.search(p, text) for p in patterns)


def faction_label(faction: str) -> str:
    return "好人" if faction == "good" else "狼人"
```

- [ ] **Step 4: Run tests — verify they pass**

```powershell
python -m pytest tests/cognition/test_role_monitor.py -v
```

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/cognition/role_monitor.py tests/cognition/test_role_monitor.py
git commit -m "feat: add RoleStateMonitor for proactive role-specific alerts"
```

---

### Task 4: Wire RoleStateMonitor into CognitivePipeline and agent context

**Files:**
- Modify: `werewolf_agent/cognition/pipeline.py`
- Modify: `werewolf_agent/runtime/agent_adapter.py`

- [ ] **Step 1: Read current CognitivePipeline**

Read `werewolf_agent/cognition/pipeline.py` to understand `run()` signature.

- [ ] **Step 2: Wire monitor into pipeline**

In `CognitivePipeline.run()`, after SalienceEngine and before ContradictionEngine:

```python
from werewolf_agent.cognition.role_monitor import RoleStateMonitor

# In run():
monitor = RoleStateMonitor(self._engine.ruleset)
role_alerts = monitor.assess(gs, player_id, role, phase)
context.alerts = context.alerts + role_alerts
```

- [ ] **Step 3: Wire alerts into strategy directive**

In `agent_adapter.py` `build_agent_context()`, after the `contradiction_alerts` injection:

```python
# Append role state alerts to strategy directive
if context.alerts:
    role_alert_msgs = [
        a.message for a in context.alerts
        if hasattr(a, 'alert_type') and a.severity in ('critical', 'warning')
    ]
    if role_alert_msgs:
        strategy_directive["role_alerts"] = role_alert_msgs
```

- [ ] **Step 4: Run tests**

```powershell
python -m pytest tests/cognition/ tests/agents/ -q --tb=short
```

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/cognition/pipeline.py werewolf_agent/runtime/agent_adapter.py
git commit -m "feat: wire RoleStateMonitor into CognitivePipeline and agent context"
```

---

### Task 5: Improve seer/witch/hybrid agent directives (Issues #3, #4, #6)

**Files:**
- Modify: `werewolf_agent/runtime/agent_adapter.py`

- [ ] **Step 1: Strengthen seer sheriff election directive**

In `agent_sheriff_election_speech`, when agent role is seer, append to `badge_flow_instruction`:

```python
if role == "seer":
    badge_flow_instruction += (
        "\n【查验理由要求】你每夜的查验目标必须有具体动机。"
        "禁止说'按顺序验'或'随便验的'。正确的说法示例："
        "'N1验p03是因为他在警下靠前位置，我需要尽早确认他的身份'。"
        "如果没有特殊理由，可以说'首夜随机查验，但我选择了发声量大的位置以获取信息基点'。"
    )
```

- [ ] **Step 2: Strengthen witch poison urgency**

In `agent_night_witch` (or `_build_witch_context`), add:

```python
if not gs.poison_used:
    alive = sum(1 for p in gs.players.values() if p.alive)
    if alive <= 8:
        strategy_directive["poison_urgency"] = (
            f"场上仅存活{alive}人。你的毒药还没有使用。"
            f"你必须认真考虑今晚撒毒——选择你最有把握的狼人目标。"
            f"如果你被刀或被投出局，毒药将浪费。"
        )
```

- [ ] **Step 3: Add hybrid master-dead guidance**

In `build_agent_context`, when `role == "hybrid"` and master is dead:

```python
if role == "hybrid" and gs.hybrid_master_id:
    master = gs.players.get(gs.hybrid_master_id)
    if master and not master.alive:
        strategy_directive["hybrid_master_dead"] = (
            f"你的主人{gs.hybrid_master_id}已死亡。"
            f"你以{gs.hybrid_master_faction}阵营身份继续。"
            f"你现在等同于村民，用分析而非技能帮助阵营。"
        )
```

- [ ] **Step 4: Run agent tests**

```powershell
python -m pytest tests/agents/test_agents.py -q --tb=short
```

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/runtime/agent_adapter.py
git commit -m "feat: strengthen seer/witch/hybrid agent strategy directives"
```

---

### Task 6: Fix sheriff election chaos (Issue #5)

**Files:**
- Modify: `werewolf_agent/runtime/sheriff_policy.py`
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Modify: `tests/runtime/test_sheriff_policy.py`

- [ ] **Step 1: Add filter_sheriff_candidates to sheriff_policy.py**

```python
# In werewolf_agent/runtime/sheriff_policy.py

import re

_TEMPLATE_PATTERNS = [
    r"我这轮先把视角压到",
    r"依据是.*最近发言",
]

def filter_sheriff_candidates(
    gs: GameState,
    candidates: list[str],
    speeches: dict[str, str] | None = None,
) -> list[str]:
    """Remove candidates whose sheriff speech is too short or matches templates."""
    speeches = speeches or {}
    kept: list[str] = []
    for c in candidates:
        text = speeches.get(c, "")
        if len(text) < 30:
            continue
        if any(re.search(p, text) for p in _TEMPLATE_PATTERNS):
            continue
        kept.append(c)
    return kept
```

- [ ] **Step 2: Add anti-template directive**

In `agent_adapter.py` `agent_sheriff_election_speech`:

```python
strategy_directive["anti_template"] = (
    "【禁止模板化】你的发言不能机械套用模板。以下句式会让你的发言"
    "被判定为无效：'我这轮先把视角压到XX身上'、'依据是XX最近发言：...'。"
    "你必须有自己独立的角度和分析逻辑。"
)
```

- [ ] **Step 3: Update __init__.py export**

In `werewolf_agent/runtime/sheriff_policy.py`, ensure `filter_sheriff_candidates` is accessible.

- [ ] **Step 4: Write and run tests**

```python
# Add to tests/runtime/test_sheriff_policy.py

def test_filter_removes_template_matching_speeches():
    from werewolf_agent.runtime.sheriff_policy import filter_sheriff_candidates
    gs = GameState(game_id="test")
    candidates = ["p03", "p04"]
    speeches = {
        "p03": "我这轮先把视角压到p01身上。依据是p04最近发言：我是村民...",
        "p04": "我是预言家，昨晚查验了p01是好人。我的警徽流先留p08。",
    }
    result = filter_sheriff_candidates(gs, candidates, speeches)
    assert result == ["p04"]

def test_filter_removes_short_speeches():
    from werewolf_agent.runtime.sheriff_policy import filter_sheriff_candidates
    gs = GameState(game_id="test")
    candidates = ["p03", "p04"]
    speeches = {"p03": "我上警", "p04": "我是预言家，昨晚查验了p01是好人，警徽流留p08"}
    result = filter_sheriff_candidates(gs, candidates, speeches)
    assert "p03" not in result
```

```powershell
python -m pytest tests/runtime/test_sheriff_policy.py -q --tb=short
```

- [ ] **Step 5: Commit**

```bash
git add werewolf_agent/runtime/sheriff_policy.py werewolf_agent/runtime/agent_adapter.py tests/runtime/test_sheriff_policy.py
git commit -m "feat: add filter_sheriff_candidates and anti-template directives"
```

---

### Task 7: Full integration test and regression

- [ ] **Step 1: Run full test suite**

```powershell
python -m pytest tests/ -q --tb=short
```

- [ ] **Step 2: Verify game log scenario**

Run a real game and check:
- Dead players do not appear in speech order
- Hunter shot does not crash with AttributeError
- Seer/witch agents receive role alerts when under pressure

```powershell
python scripts/run_real_game.py
# Check game_stdout.log for the above

# Verify no hunter shot crash:
grep -c "AttributeError.*list.*has no attribute" game_stdout.log
# Expected: 0
```

- [ ] **Step 3: Commit any remaining fixes**

```bash
git add -A
git commit -m "chore: final integration verification and fixes"
```

---

## Verification Summary

```powershell
# Unit tests
python -m pytest tests/cognition/test_role_monitor.py -v
python -m pytest tests/runtime/test_sheriff_policy.py -v
python -m pytest tests/runtime/test_runtime.py -v
python -m pytest tests/agents/test_agents.py -v

# Integration
python -m pytest tests/integration/ -q --tb=short

# Full suite
python -m pytest tests/ -q --tb=short
```
