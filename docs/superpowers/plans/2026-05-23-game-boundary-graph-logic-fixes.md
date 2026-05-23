# Game Boundary Graph Logic Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make real-game runtime boundaries explicit so deaths, role skills, votes, and exported records are coherent and auditable.

**Architecture:** Keep deterministic rule decisions in `RuleEngine`; keep graph nodes responsible for phase routing, judge broadcasts, and agent interaction. Skill effects must follow a closed loop: trigger -> prompt -> choose/decline -> resolve -> record. Export code must preserve enough event/death metadata for replay and audit.

**Tech Stack:** Python 3, dataclasses, LangGraph runtime nodes, pytest, existing `GameState` / `Death` / `GameEvent` models.

---

## File Structure

- Modify `werewolf_agent/runtime/graph.py`: role phase guards, hunter-shot closed loop, broadcasts, route invariants.
- Modify `werewolf_agent/runtime/agent_adapter.py`: fallback vote reason payload consistency if needed.
- Modify `werewolf_agent/runtime/vote_quality.py`: helper for fallback public vote reason if centralizing reason repair.
- Modify `scripts/run_real_game.py`: complete `Death` export and audit-friendly game log.
- Modify `tests/runtime/test_runtime.py`: graph and skill regression tests.
- Modify `tests/runtime/test_pk_flow.py`: keep vote target/self-vote context tests.
- Modify `tests/runtime/test_vote_quality.py`: fallback reason and structured vote tests if helper changes.
- Modify `tests/scripts/test_run_real_game.py`: exported log schema regression.

---

### Task 1: Preserve Current In-Progress Fixes

**Files:**
- Inspect: `git status --short`
- Verify: `tests/agents/test_agents.py`, `tests/runtime/test_pk_flow.py`, `tests/runtime/test_runtime.py`, `tests/runtime/test_vote_quality.py`, `werewolf_agent/agents/player.py`, `werewolf_agent/agents/schemas.py`, `werewolf_agent/runtime/agent_adapter.py`, `werewolf_agent/runtime/graph.py`, `werewolf_agent/runtime/vote_quality.py`

- [ ] **Step 1: Review current diff**

Run:

```powershell
git diff --stat
git diff -- werewolf_agent/runtime/graph.py
```

Expected: existing uncommitted changes include structured vote quality and hunter-shot closed-loop work.

- [ ] **Step 2: Run current safety tests**

Run:

```powershell
python -m pytest tests\runtime\test_runtime.py tests\agents\test_agents.py tests\runtime\test_vote_quality.py tests\runtime\test_pk_flow.py -q --basetemp=.pytest-tmp
```

Expected: PASS before adding new boundary fixes.

- [ ] **Step 3: Commit current stable changes**

Run:

```powershell
git add tests\agents\test_agents.py tests\runtime\test_pk_flow.py tests\runtime\test_runtime.py tests\runtime\test_vote_quality.py werewolf_agent\agents\player.py werewolf_agent\agents\schemas.py werewolf_agent\runtime\agent_adapter.py werewolf_agent\runtime\graph.py werewolf_agent\runtime\vote_quality.py
git commit -m "Tighten vote quality and hunter shot boundaries"
```

Expected: clean checkpoint before continuing.

---

### Task 2: Skip Dead Role Night Phases

**Files:**
- Modify: `werewolf_agent/runtime/graph.py`
- Test: `tests/runtime/test_runtime.py`

- [ ] **Step 1: Write failing tests for dead witch and dead seer**

Add tests asserting:

```python
def test_night_witch_skips_when_witch_dead() -> None:
    from werewolf_agent.runtime.graph import night_witch
    engine = _new_engine()
    gs = GameState(
        game_id="dead_witch",
        night_number=3,
        players={
            "witch": PlayerState(id="witch", role="witch", alive=False),
            "wolf": PlayerState(id="wolf", role="werewolf", alive=True),
        },
    )

    result = night_witch({"game_state": gs, "engine": engine})

    phases = [e.payload.get("phase") for e in result["game_state"].events if e.type == "judge_broadcast"]
    assert "witch_wake" not in phases
    assert "witch_choose" not in phases
    assert result["use_antidote"] is False
    assert result["poison_target_id"] is None
```

Also add equivalent seer test:

```python
def test_night_seer_skips_when_seer_dead() -> None:
    from werewolf_agent.runtime.graph import night_seer
    engine = _new_engine()
    gs = GameState(
        game_id="dead_seer",
        night_number=3,
        players={
            "seer": PlayerState(id="seer", role="seer", alive=False),
            "wolf": PlayerState(id="wolf", role="werewolf", alive=True),
        },
    )

    result = night_seer({"game_state": gs, "engine": engine})

    phases = [e.payload.get("phase") for e in result["game_state"].events if e.type == "judge_broadcast"]
    assert "seer_wake" not in phases
    assert "seer_choose" not in phases
    assert result["seer_target_id"] is None
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```powershell
python -m pytest tests\runtime\test_runtime.py::test_night_witch_skips_when_witch_dead tests\runtime\test_runtime.py::test_night_seer_skips_when_seer_dead -q --basetemp=.pytest-tmp
```

Expected: FAIL because current graph broadcasts dead roles.

- [ ] **Step 3: Implement role-alive guard**

In `werewolf_agent/runtime/graph.py`, add helper near `_find_role` or local node helpers:

```python
def _find_alive_role(gs: GameState, role: str) -> str | None:
    return next((pid for pid, p in gs.players.items() if p.role == role and p.alive), None)
```

At the start of `night_witch`:

```python
if _find_alive_role(gs, "witch") is None:
    return {"game_state": gs, "use_antidote": False, "poison_target_id": None}
```

At the start of `night_seer`:

```python
if _find_alive_role(gs, "seer") is None:
    return {"game_state": gs, "seer_target_id": None}
```

- [ ] **Step 4: Run target tests**

Run same command as Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add werewolf_agent\runtime\graph.py tests\runtime\test_runtime.py
git commit -m "Skip dead role night phases"
```

---

### Task 3: Make Hunter Shot Closed Loop Non-Silent

**Files:**
- Modify: `werewolf_agent/runtime/graph.py`
- Test: `tests/runtime/test_runtime.py`

- [ ] **Step 1: Confirm existing hunter tests cover the desired behavior**

Ensure tests exist for:

- `post_exile_skills` does not resolve hunter shot silently.
- `resolve_hunter_shot` broadcasts prompt.
- `resolve_hunter_shot` records choice and applies death.
- `resolve_hunter_shot` records declined shot when no target.
- route from post-exile goes to `resolve_hunter_shot` before `check_victory`.

- [ ] **Step 2: Add missing test for invalid scripted target**

Add:

```python
def test_resolve_hunter_shot_declines_invalid_target() -> None:
    from werewolf_agent.runtime.graph import resolve_hunter_shot
    engine = _new_engine()
    hunter_death = Death(
        player_id="hunter",
        reason="exile",
        timing="day_vote",
        resolution_batch="day_3_vote",
        triggered_skills=["hunter_shot"],
    )
    gs = GameState(
        game_id="hunter_invalid_target",
        day_number=3,
        phase="day",
        players={
            "hunter": PlayerState(id="hunter", role="hunter", alive=False),
            "wolf": PlayerState(id="wolf", role="werewolf", alive=True),
            "dead": PlayerState(id="dead", role="villager", alive=False),
        },
        deaths=[hunter_death],
    )

    result = resolve_hunter_shot({
        "game_state": gs,
        "engine": engine,
        "hunter_shot_target_id": "dead",
    })

    new_state = result["game_state"]
    assert new_state.players["dead"].alive is False
    assert not any(d.reason == "hunter_shot" for d in new_state.deaths)
    assert any(e.type == "hunter_shot_declined" for e in new_state.events)
```

- [ ] **Step 3: Run hunter target tests**

Run:

```powershell
python -m pytest tests\runtime\test_runtime.py -q --basetemp=.pytest-tmp
```

Expected: PASS after implementation.

- [ ] **Step 4: Commit**

Run:

```powershell
git add werewolf_agent\runtime\graph.py tests\runtime\test_runtime.py
git commit -m "Record hunter shot choices and declines"
```

---

### Task 4: Make Fallback Vote Reasons Public and Non-Empty

**Files:**
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Possibly modify: `werewolf_agent/runtime/graph.py`
- Test: `tests/runtime/test_runtime.py` or `tests/agents/test_agents.py`

- [ ] **Step 1: Write failing test for fallback vote reason**

Add test around `agent_day_vote` with an agent returning a `FallbackAction`:

```python
def test_agent_day_vote_uses_fallback_reason_when_action_reason_empty() -> None:
    from werewolf_agent.runtime.agent_adapter import agent_day_vote
    from werewolf_agent.agents.schemas import ActionType, FallbackAction

    engine = _new_engine()
    gs = GameState(
        game_id="fallback_vote_reason",
        day_number=1,
        players={
            "p01": PlayerState(id="p01", role="villager", alive=True),
            "p02": PlayerState(id="p02", role="werewolf", alive=True),
        },
    )

    class Agent:
        def act(self, context):
            from werewolf_agent.agents.schemas import RetryInfo
            return FallbackAction(
                action_type=ActionType.VOTE,
                target_id="p02",
                reason="fallback: 结构化输出失败，按当前线索选择p02",
            ), RetryInfo()

    class Registry:
        def get_agent(self, player_id):
            return Agent()

    result = agent_day_vote({"game_state": gs}, engine, Registry(), "p01")

    assert result["vote_target"] == "p02"
    assert result["vote_reason"]
```

- [ ] **Step 2: Run RED**

Run:

```powershell
python -m pytest tests\runtime\test_runtime.py::test_agent_day_vote_uses_fallback_reason_when_action_reason_empty -q --basetemp=.pytest-tmp
```

Expected: FAIL if fallback reason is not propagated in the same shape used by runtime.

- [ ] **Step 3: Implement reason fallback**

In `agent_day_vote`, after `reason = getattr(action, "reason", "") or ""`, add:

```python
if not reason:
    trace_obj = getattr(action, "trace", None)
    if trace_obj and getattr(trace_obj, "fallback_reason", None):
        reason = trace_obj.fallback_reason
if not reason and target:
    reason = f"fallback: 模型未给出投票理由，按当前可见线索选择{target}"
```

If `resolve_vote` builds public reasons from action traces, also update `_public_vote_reason` to use `fallback_reason` and `target_id` before returning empty.

- [ ] **Step 4: Verify**

Run:

```powershell
python -m pytest tests\runtime\test_runtime.py tests\runtime\test_vote_quality.py -q --basetemp=.pytest-tmp
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add werewolf_agent\runtime\agent_adapter.py werewolf_agent\runtime\graph.py tests\runtime\test_runtime.py
git commit -m "Ensure fallback votes have public reasons"
```

---

### Task 5: Enforce Self-Vote Defense in Runtime Records

**Files:**
- Verify: `werewolf_agent/runtime/agent_adapter.py`
- Verify: `werewolf_agent/engine/rule_engine.py`
- Test: `tests/runtime/test_pk_flow.py`, `tests/rules/test_rule_engine_v1.py`

- [ ] **Step 1: Run current self-vote tests**

Run:

```powershell
python -m pytest tests\runtime\test_pk_flow.py::TestPKRevoteRestrictsTargets tests\rules\test_rule_engine_v1.py::test_self_votes_do_not_count_in_exile_vote -q --basetemp=.pytest-tmp
```

Expected: PASS.

- [ ] **Step 2: Add audit test for no self target in action trace legal_targets**

Assert `agent_day_vote` context and trace target lists exclude the voter.

- [ ] **Step 3: Verify no current code reintroduces self into `legal_targets`**

Run:

```powershell
rg -n "legal_exile_targets|legal_targets = .*exile|target_id == voter_id|pid != voter_id" werewolf_agent tests
```

Expected: `agent_day_vote` filters `pid != voter_id`; `RuleEngine.resolve_vote` ignores `target_id == voter_id`.

- [ ] **Step 4: Commit if changes are needed**

Run:

```powershell
git add werewolf_agent tests
git commit -m "Keep self votes out of runtime vote records"
```

---

### Task 6: Export Complete Death Records

**Files:**
- Modify: `scripts/run_real_game.py`
- Test: `tests/scripts/test_run_real_game.py`

- [ ] **Step 1: Write failing export test**

Add or extend a test for `save_game_log`:

```python
def test_save_game_log_exports_complete_death_fields(tmp_path, monkeypatch):
    from werewolf_agent.core.models import Death, GameState, PlayerState
    from scripts import run_real_game

    gs = GameState(
        game_id="g_export",
        players={"hunter": PlayerState(id="hunter", role="hunter", alive=False)},
        deaths=[
            Death(
                player_id="hunter",
                reason="exile",
                timing="day_vote",
                resolution_batch="day_3_vote",
                source_player_id=None,
                can_leave_last_words=True,
                triggered_skills=["hunter_shot"],
            )
        ],
    )

    class Runner:
        game_id = "g_export"
        state = gs
        step_count = 1

    monkeypatch.setattr(run_real_game, "ROOT", tmp_path)
    path = run_real_game.save_game_log(Runner(), elapsed=1.2)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["deaths"][0]["timing"] == "day_vote"
    assert data["deaths"][0]["resolution_batch"] == "day_3_vote"
    assert data["deaths"][0]["triggered_skills"] == ["hunter_shot"]
```

- [ ] **Step 2: Run RED**

Run:

```powershell
python -m pytest tests\scripts\test_run_real_game.py::test_save_game_log_exports_complete_death_fields -q --basetemp=.pytest-tmp
```

Expected: FAIL because only `player_id/reason` are exported.

- [ ] **Step 3: Implement complete export**

In `scripts/run_real_game.py::save_game_log`, replace:

```python
"deaths": [{"player_id": d.player_id, "reason": d.reason} for d in gs.deaths],
```

with:

```python
"deaths": [
    {
        "player_id": d.player_id,
        "reason": d.reason,
        "timing": d.timing,
        "resolution_batch": d.resolution_batch,
        "source_player_id": d.source_player_id,
        "can_leave_last_words": d.can_leave_last_words,
        "triggered_skills": list(d.triggered_skills),
    }
    for d in gs.deaths
],
```

- [ ] **Step 4: Verify**

Run:

```powershell
python -m pytest tests\scripts\test_run_real_game.py tests\test_game_audit.py -q --basetemp=.pytest-tmp
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add scripts\run_real_game.py tests\scripts\test_run_real_game.py
git commit -m "Export complete death records"
```

---

### Task 7: Add Game Record Boundary Audit

**Files:**
- Modify: `tests/test_game_audit.py`
- Possibly modify: `scripts/print_game_audit.py`

- [ ] **Step 1: Add audit assertions for latest game-style logs**

Add checks that fail when:

- A dead witch/seer receives wake/choose broadcasts after death.
- A hunter death with `triggered_skills=["hunter_shot"]` lacks `hunter_shot_prompt` plus choice/decline.
- `vote_resolved.votes[].reason` is empty.
- A vote has `voter == target`.
- exported deaths omit required death metadata.

- [ ] **Step 2: Run audit tests**

Run:

```powershell
python -m pytest tests\test_game_audit.py -q --basetemp=.pytest-tmp
```

Expected: FAIL until implementation tasks are complete, then PASS.

- [ ] **Step 3: Keep audit as regression net**

Do not make this audit depend on a specific `game_g_*.json`; use small synthetic game records or generated dicts.

- [ ] **Step 4: Commit**

Run:

```powershell
git add tests\test_game_audit.py scripts\print_game_audit.py
git commit -m "Audit game record boundary invariants"
```

---

### Task 8: Rate Limit / Timeout Follow-Up Guardrails

**Files:**
- Inspect: `werewolf_agent/model_gateway/router.py`
- Inspect: `config/models.yaml`
- Test: existing model gateway tests or new `tests/model_gateway/test_router.py`

- [ ] **Step 1: Do not block graph-boundary fixes on provider throttling**

Document that 429/timeout issues are provider reliability problems, but graph nodes must still record deterministic fallback events.

- [ ] **Step 2: Add follow-up tests if provider routing is in scope**

Desired behavior:

- Repeated 429 from Qianfan marks provider temporarily unavailable for that task window.
- Critical actions such as `hunter_shot`, `vote`, `badge_transfer` prefer a stable fallback provider.
- Timeouts produce explicit fallback audit events.

- [ ] **Step 3: Implement only after boundary fixes pass**

This can be a separate PR/commit to avoid mixing provider policy with graph correctness.

---

### Task 9: Final Verification and Real-Game Smoke

**Files:**
- Verify all touched files.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m pytest tests\runtime\test_runtime.py tests\runtime\test_pk_flow.py tests\runtime\test_vote_quality.py tests\scripts\test_run_real_game.py tests\test_game_audit.py -q --basetemp=.pytest-tmp
```

Expected: PASS.

- [ ] **Step 2: Run full tests**

Run:

```powershell
python -m pytest tests -q --basetemp=.pytest-tmp
```

Expected: PASS.

- [ ] **Step 3: Run one real-game smoke if API quota allows**

Run:

```powershell
python .\scripts\run_real_game.py --timeout 120 --max-steps 500
```

Expected:

- No self-votes in `vote_resolved`.
- No empty vote reasons.
- No dead role wake/choose broadcasts.
- Hunter death has prompt plus choice/decline.
- Exported `deaths` include full metadata.

- [ ] **Step 4: Commit final audit/docs if needed**

Run:

```powershell
git status --short
git add .
git commit -m "Clarify game graph boundary invariants"
```

