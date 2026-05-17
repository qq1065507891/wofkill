# Platform Customization And Growth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a platform layer for configurable AI Werewolf rooms: rule templates, 12-player persona packs, launch wizard, shareable spectating, replay-driven growth, and safe customization.

**Architecture:** Keep `RuleEngine` as the only adjudication authority. Add `RulesetRegistry` and `PersonaPackAdapter` before any upload UI becomes playable: validated IDs must resolve to actual RuleEngine and PersonaRouter runtime inputs. Implement growth features in small slices: capability registry and persistence first, validation APIs second, frontend launch wizard third, then marketplace/share/replay flows.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, PyYAML/safe YAML parsing, existing RuleEngine, existing PersonaRouter, SQLite repository, vanilla HTML/CSS/JS dashboard, pytest.

---

## File Structure Map

```text
config/
  rulesets/
    templates/
      custom_ruleset_template.yaml          # New: downloadable rule template
    marketplace.yaml                        # New: static marketplace index with playable/display-only status
  personas/
    templates/
      player_profile_pack_template.yaml     # New: downloadable 12-player persona template
    default_12_ai_players.yaml              # New: default user-facing persona pack
    marketplace.yaml                        # New: static persona marketplace index

werewolf_agent/
  customization/
    __init__.py                             # New package
    schemas.py                              # New: Pydantic schemas for rules/persona uploads
    ruleset_registry.py                     # New: RuleEngine capability and ruleset lookup
    compatibility.py                        # New: ruleset compatibility matrix and template diffs
    persona_adapter.py                      # New: user persona pack -> PersonaRouter runtime format
    validators.py                           # New: whitelist validation and normalization
    repository.py                           # New: local persistence helpers for validated configs
    preview.py                              # New: deterministic persona preview utterances
  api/
    app.py                                  # Modify: template download, upload validation, room creation params
    schemas.py                              # Modify: ruleset/profile/mode fields
    views.py                                # Modify: leak audit summary in replay/share views if applicable
  runtime/
    game_runner.py                          # Modify: load rules through RulesetRegistry
  ui/static/
    dashboard.html                          # Modify: launch wizard DOM shell
    dashboard.css                           # New: dashboard styles extracted from HTML
    dashboard.js                            # New: API/upload/render logic extracted from HTML

tests/
  customization/
    test_ruleset_registry.py                # New tests for playable vs display-only rules
    test_compatibility.py                   # New tests for compatibility matrix and template diffs
    test_ruleset_templates.py               # New tests for rule validation
    test_persona_templates.py               # New tests for persona validation and previews
    test_persona_adapter.py                 # New tests for PersonaRouter-compatible output
  api/
    test_customization_api.py               # New API tests
    test_share_audit.py                     # New public replay/share audit tests
  ui/
    test_dashboard_enhanced.py              # Extend static UI tests

docs/
  design/platform-customization-growth-design.md
  superpowers/plans/2026-05-17-platform-customization-growth.md
```

## Implementation Principle

Every upload feature follows this order:

1. Download a known-good template.
2. Upload YAML as plain data.
3. Parse with safe YAML only.
4. Validate against whitelist schemas.
5. Return structured errors and warnings.
6. Store raw YAML, normalized data, validation result, content hash, creator, status, and timestamps.
7. Use only `status=playable` IDs to create a room.

No uploaded content may execute code or override system prompts.
No display-only ruleset may be passed to `GameRunner`.

---

### Task 0: RuleEngine Capability And RulesetRegistry

**Files:**
- Create: `werewolf_agent/customization/__init__.py`
- Create: `werewolf_agent/customization/ruleset_registry.py`
- Create: `werewolf_agent/customization/compatibility.py`
- Modify: `werewolf_agent/runtime/game_runner.py`
- Test: `tests/customization/test_ruleset_registry.py`
- Test: `tests/customization/test_compatibility.py`
- Test: `tests/runtime/test_game_runner.py`

- [ ] **Step 1: Write failing tests for playable registry behavior**

```python
from werewolf_agent.customization.ruleset_registry import RulesetRegistry


def test_default_ruleset_is_playable():
    registry = RulesetRegistry()
    entry = registry.get("pre_witch_hunter_idiot_mixed")
    assert entry.status == "playable"
    assert "werewolf" in entry.capabilities.supported_roles
    assert "hybrid" in entry.capabilities.supported_roles


def test_unimplemented_roles_are_display_only():
    registry = RulesetRegistry()
    entry = registry.from_normalized({
        "ruleset_id": "wolf_king_guard_demo",
        "roles": {"wolf_king": {"count": 1}, "guard": {"count": 1}},
        "player_count": 2,
    })
    assert entry.status == "display_only"
    assert "wolf_king" in entry.unsupported_roles
    assert "guard" in entry.unsupported_roles


def test_ruleset_compatibility_matrix_reports_missing_abilities():
    registry = RulesetRegistry()
    entry = registry.from_normalized({
        "ruleset_id": "wolf_king_guard_demo",
        "roles": {"wolf_king": {"count": 1}, "guard": {"count": 1}},
        "abilities": ["wolf_king_shot", "guard_protect"],
        "player_count": 2,
    })
    assert entry.compatibility.status == "display_only"
    assert "wolf_king_shot" in entry.compatibility.missing_abilities
    assert "guard_protect" in entry.compatibility.missing_abilities
```

- [ ] **Step 2: Run tests and confirm they fail**

Run: `pytest tests/customization/test_ruleset_registry.py -q`

Expected: fail because registry does not exist.

- [ ] **Step 3: Implement registry and compatibility matrix from current RuleEngine capability**

First playable role set:

- `werewolf`
- `villager`
- `seer`
- `witch`
- `hunter`
- `idiot`
- `hybrid`

`guard` and `wolf_king` must be `display_only` until RuleEngine has real tests and implementation for their night/day effects.

Compatibility matrix fields:

- `status`
- `supported_roles`
- `unsupported_roles`
- `supported_abilities`
- `missing_abilities`
- `supported_victory_conditions`
- `unsupported_victory_conditions`
- `warnings`

- [ ] **Step 4: Make GameRunner use registry lookup**

`GameRunnerConfig.ruleset_id` must resolve through `RulesetRegistry`; if status is not `playable`, GameRunner raises a clear validation error.

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/customization/test_ruleset_registry.py tests/customization/test_compatibility.py tests/runtime/test_game_runner.py -q`

Expected: pass.

---

### Task 1: Rule Template Schema And Validator

**Files:**
- Create: `werewolf_agent/customization/schemas.py`
- Create: `werewolf_agent/customization/validators.py`
- Create: `config/rulesets/templates/custom_ruleset_template.yaml`
- Test: `tests/customization/test_ruleset_templates.py`

- [ ] **Step 1: Write failing tests for valid and invalid rule templates**

```python
from pathlib import Path

from werewolf_agent.customization.validators import validate_ruleset_yaml


def test_default_custom_ruleset_template_validates():
    text = Path("config/rulesets/templates/custom_ruleset_template.yaml").read_text(encoding="utf-8")
    result = validate_ruleset_yaml(text)
    assert result.valid is True
    assert result.summary["player_count"] == 12
    assert result.normalized["ruleset_id"]


def test_ruleset_rejects_role_count_mismatch():
    text = """
ruleset_id: bad_count
name: Bad Count
version: 1
player_count: 12
roles:
  werewolf: {count: 4, faction: werewolf}
  villager: {count: 7, faction: good}
night_order: [werewolf]
victory:
  good: [eliminate_all_wolves]
  werewolf: [slaughter_villagers]
constraints: {}
"""
    result = validate_ruleset_yaml(text)
    assert result.valid is False
    assert any("player_count" in err.message for err in result.errors)
```

- [ ] **Step 2: Run tests and confirm they fail**

Run: `pytest tests/customization/test_ruleset_templates.py -q`

Expected: fail because `werewolf_agent.customization` does not exist.

- [ ] **Step 3: Implement minimal schemas and validator**

Implement:

- `ValidationIssue`
- `ValidationResult`
- `validate_ruleset_yaml(text: str) -> ValidationResult`

Playable whitelist:

- roles: initially `werewolf`, `villager`, `seer`, `witch`, `hunter`, `idiot`, `hybrid`
- victory: `eliminate_all_wolves`, `slaughter_villagers`, `slaughter_gods`
- constraints: `witch_can_self_save`, `witch_can_use_both_potions_same_night`, `werewolf_can_no_kill`, `wolf_timeout_default`, `hybrid_enabled`

`guard` and `wolf_king` may be accepted only with `status=display_only`; they must not produce a playable config.

- [ ] **Step 3.5: Add normalized diff against default ruleset**

Return `diff_against_default` from validation using normalized whitelist fields only. Example diff entry:

```json
{"path": "constraints.witch_can_self_save", "default": false, "uploaded": true}
```

- [ ] **Step 4: Run tests and confirm pass**

Run: `pytest tests/customization/test_ruleset_templates.py -q`

Expected: all pass.

---

### Task 2: Persona Pack Schema, Validator, And Preview

**Files:**
- Modify: `werewolf_agent/customization/schemas.py`
- Modify: `werewolf_agent/customization/validators.py`
- Create: `werewolf_agent/customization/preview.py`
- Create: `werewolf_agent/customization/persona_adapter.py`
- Create: `config/personas/templates/player_profile_pack_template.yaml`
- Create: `config/personas/default_12_ai_players.yaml`
- Test: `tests/customization/test_persona_templates.py`
- Test: `tests/customization/test_persona_adapter.py`

- [ ] **Step 1: Write failing tests for persona pack validation**

```python
from pathlib import Path

from werewolf_agent.customization.preview import build_persona_preview
from werewolf_agent.customization.validators import validate_persona_pack_yaml


def test_persona_template_has_exactly_12_players():
    text = Path("config/personas/templates/player_profile_pack_template.yaml").read_text(encoding="utf-8")
    result = validate_persona_pack_yaml(text)
    assert result.valid is True
    assert result.summary["player_count"] == 12


def test_persona_pack_rejects_duplicate_seats():
    text = """
profile_pack_id: duplicate
name: Duplicate Seats
version: 1
players:
  - seat: 1
    name: A
    archetype: leader
    speech_style: calm
    risk_tolerance: medium
    deception: low
    cooperation: high
    aggression: low
    memory_focus: medium
    logic_focus: high
    emotionality: low
  - seat: 1
    name: B
    archetype: analyst
    speech_style: calm
    risk_tolerance: medium
    deception: low
    cooperation: high
    aggression: low
    memory_focus: medium
    logic_focus: high
    emotionality: low
"""
    result = validate_persona_pack_yaml(text)
    assert result.valid is False
    assert any("seat" in err.message for err in result.errors)


def test_persona_preview_generates_four_samples():
    persona = {
        "seat": 1,
        "name": "归票官",
        "archetype": "leader",
        "speech_style": "强势、简洁",
        "risk_tolerance": "medium",
        "deception": "low",
        "cooperation": "high",
        "aggression": "medium",
        "memory_focus": "high",
        "logic_focus": "high",
        "emotionality": "low",
    }
    preview = build_persona_preview(persona)
    assert set(preview) == {"villager_opening", "defense", "wolf_night", "seer_claim"}
```

- [ ] **Step 2: Run tests and confirm they fail**

Run: `pytest tests/customization/test_persona_templates.py -q`

Expected: fail because validator and preview are missing.

- [ ] **Step 3: Implement persona validation and deterministic preview**

Validation rules:

- exactly 12 players for a valid playable pack
- `seat` unique 1-12
- enum values: `low`, `medium`, `high`
- allowed roles in `preferred_roles`
- length limits on `name`, `speech_style`, `catchphrases`
- prompt-injection phrase checks for uploaded text

- [ ] **Step 4: Write failing tests for PersonaRouter adapter**

```python
from werewolf_agent.customization.persona_adapter import adapt_persona_pack


def test_persona_pack_adapter_outputs_router_profiles_and_assignments():
    pack = {
        "profile_pack_id": "custom_pack",
        "players": [
            {
                "seat": i,
                "name": f"P{i}",
                "archetype": "analyst",
                "speech_style": "calm",
                "risk_tolerance": "medium",
                "deception": "low",
                "cooperation": "high",
                "aggression": "low",
                "memory_focus": "medium",
                "logic_focus": "high",
                "emotionality": "low",
            }
            for i in range(1, 13)
        ],
    }
    adapted = adapt_persona_pack(pack)
    assert "persona_profiles" in adapted
    assert "player_assignments" in adapted
    assert adapted["player_assignments"]["p01"].startswith("custom_pack_seat_01")
```

- [ ] **Step 5: Implement adapter**

Adapter output must match existing `PersonaRouter` expectations:

- `persona_profiles`
- `player_assignments`
- profile `base`
- profile `task_styles`
- profile `dynamic_policy`

Also return `diff_against_default` for normalized persona packs, comparing only whitelisted player fields.

- [ ] **Step 6: Run tests and confirm pass**

Run: `pytest tests/customization/test_persona_templates.py tests/customization/test_persona_adapter.py -q`

Expected: all pass.

---

### Task 3: Customization Persistence And API Endpoints

**Files:**
- Create: `werewolf_agent/customization/repository.py`
- Modify: `werewolf_agent/storage/repository.py`
- Modify: `werewolf_agent/storage/sqlite_store.py`
- Modify: `werewolf_agent/storage/migrations.py`
- Modify: `werewolf_agent/api/app.py`
- Create: `tests/api/test_customization_api.py`
- Create: `tests/storage/test_customization_storage.py`

- [ ] **Step 1: Write failing API tests**

```python
from fastapi.testclient import TestClient

from werewolf_agent.api.app import create_app


def test_download_ruleset_template():
    client = TestClient(create_app())
    resp = client.get("/templates/ruleset")
    assert resp.status_code == 200
    assert "ruleset_id" in resp.text


def test_validate_ruleset_upload_returns_summary():
    client = TestClient(create_app())
    template = client.get("/templates/ruleset").text
    resp = client.post("/customization/rulesets/validate", content=template)
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert data["summary"]["player_count"] == 12


def test_download_persona_template():
    client = TestClient(create_app())
    resp = client.get("/templates/persona-pack")
    assert resp.status_code == 200
    assert "players:" in resp.text
```

- [ ] **Step 2: Run tests and confirm fail**

Run: `pytest tests/api/test_customization_api.py -q`

Expected: fail with 404 endpoints and missing persistence.

- [ ] **Step 3: Add persistence contract**

Store:

- `config_id`
- `config_type`: `ruleset` or `persona_pack`
- `raw_yaml`
- `normalized`
- `validation_result`
- `content_hash`
- `status`
- `version`
- `maturity`
- `compatibility_matrix`
- `diff_against_default`
- `creator_id`
- `created_at`
- `updated_at`

Game creation must create a locked config snapshot containing:

- `ruleset_id`
- `ruleset_version`
- `ruleset_hash`
- `profile_pack_id`
- `profile_pack_version`
- `profile_pack_hash`
- `model_config_hash`
- `persona_adapter_version`
- `rag_config_hash`
- `engine_version`
- `random_seed`
- `agent_behavior_seed`
- `speech_order_seed`

- [ ] **Step 4: Add endpoints**

Endpoints:

- `GET /templates/ruleset`
- `GET /templates/persona-pack`
- `POST /customization/rulesets/validate`
- `POST /customization/persona-packs/validate`
- `POST /customization/rulesets`
- `POST /customization/persona-packs`

Return JSON `ValidationResult` for validation endpoints.

- [ ] **Step 5: Add security tests**

Cover:

- malicious YAML tag
- oversized body
- unknown fields
- HTML/JS/shell/template expression text
- prompt-injection phrases
- missing/invalid caller
- audit metadata
- locked config snapshot created on game creation

- [ ] **Step 6: Run API and storage tests**

Run: `pytest tests/api/test_customization_api.py tests/storage/test_customization_storage.py -q`

Expected: pass.

---

### Task 4: Launch Wizard UI

**Files:**
- Modify: `werewolf_agent/ui/static/dashboard.html`
- Create: `werewolf_agent/ui/static/dashboard.css`
- Create: `werewolf_agent/ui/static/dashboard.js`
- Modify: `tests/ui/test_dashboard_enhanced.py`

- [ ] **Step 1: Write failing static UI test**

```python
def test_dashboard_has_launch_wizard_customization_controls(dashboard_html):
    assert "开局向导" in dashboard_html
    assert "下载规则模板" in dashboard_html
    assert "上传规则" in dashboard_html
    assert "下载玩家模板" in dashboard_html
    assert "上传玩家配置" in dashboard_html
    assert "规则校验结果" in dashboard_html
    assert "人格预览" in dashboard_html
```

- [ ] **Step 2: Run test and confirm fail**

Run: `pytest tests/ui/test_dashboard_enhanced.py -q`

Expected: fail until UI controls exist.

- [ ] **Step 3: Split dashboard files before adding controls**

Move CSS and JS out of `dashboard.html` while preserving all existing IDs and API hooks. Add tests that confirm the HTML references `dashboard.css` and `dashboard.js`.

- [ ] **Step 4: Add static wizard controls**

Add to left lobby or main room creation area:

- ruleset selector
- download/upload rule template buttons
- persona pack selector
- download/upload persona template buttons
- validation result panel
- persona preview panel
- mode selector: `公开观战`, `我参与一席`, `全 AI 自动跑`, `教学复盘`

- [ ] **Step 5: Run UI tests**

Run: `pytest tests/ui/test_dashboard_enhanced.py -q`

Expected: pass.

---

### Task 5: Wire UI Uploads To Validation APIs

**Files:**
- Modify: `werewolf_agent/ui/static/dashboard.html`
- Modify: `tests/ui/test_dashboard_enhanced.py`

- [ ] **Step 1: Add static test for JavaScript hooks**

Assert the HTML contains:

- `validateRulesetUpload`
- `validatePersonaUpload`
- `renderValidationResult`
- `renderPersonaPreview`
- `dashboard.js`

- [ ] **Step 2: Implement upload flow**

JavaScript flow:

1. User selects YAML file.
2. Read as text.
3. POST to validation endpoint.
4. Render errors/warnings/summary.
5. Keep normalized config in memory only if `valid`.

- [ ] **Step 3: Manual browser verification**

Run server:

`python -m uvicorn werewolf_agent.api.app:create_app --factory --host 127.0.0.1 --port 18000`

Verify:

- template download works
- upload valid template shows pass
- invalid YAML shows errors

---

### Task 6: Rules And Persona Marketplace Static Version

**Files:**
- Create: `config/rulesets/marketplace.yaml`
- Create: `config/personas/marketplace.yaml`
- Modify: `werewolf_agent/api/app.py`
- Modify: `werewolf_agent/ui/static/dashboard.html`
- Test: `tests/api/test_customization_api.py`

- [ ] **Step 1: Write tests for marketplace listing endpoints**

Endpoints:

- `GET /marketplace/rulesets`
- `GET /marketplace/persona-packs`

- [ ] **Step 2: Add static marketplace configs**

Include:

- 经典 12 人狼王守卫
- 新手友好局
- 高压悍跳局
- 全员逻辑局
- 综艺效果局
- 默认竞技型 12 人
- 新手陪练 12 人
- 高端局压力测试

- [ ] **Step 2.5: Mark every marketplace item as playable or display-only**

`guard` and `wolf_king` based rules must be `display_only` until RuleEngine support exists. UI must show display-only cards as previewable but not selectable for real room creation.

- [ ] **Step 3: Render marketplace cards in dashboard**

Cards should be selectable but not require account infrastructure yet.

---

### Task 7: Shareable Replay Skeleton

**Files:**
- Modify: `werewolf_agent/api/app.py`
- Modify: `werewolf_agent/ui/static/dashboard.html`
- Test: `tests/api/test_customization_api.py`

- [ ] **Step 1: Add failing API test for replay share summary**

Endpoint:

- `GET /games/{game_id}/share-summary`

Expected response fields:

- `game_id`
- `winning_faction`
- `highlight_events`
- `mvp_candidate`
- `share_title`
- `public_only`
- `leak_audit_summary`

- [ ] **Step 2: Implement summary from existing timeline/replay data**

Keep it deterministic. Do not call LLM yet.
Only public timeline, public final results, and sanitized highlights may be included.

- [ ] **Step 2.5: Add public-safe leak audit summary**

Public share summary may include:

```json
{
  "leak_check_status": "passed",
  "private_role_leaks": 0,
  "illegal_view_references": 0,
  "forbidden_event_exposures": 0
}
```

Detailed evidence remains moderator/debug only.

- [ ] **Step 3: Add dashboard share link**

Show a “生成复盘分享” button after a game has timeline data.

---

### Task 8: Human Seat Mode Planning Hooks

**Files:**
- Modify: `werewolf_agent/api/schemas.py`
- Modify: `werewolf_agent/api/app.py`
- Modify: `werewolf_agent/ui/static/dashboard.html`
- Test: `tests/api/test_customization_api.py`

- [ ] **Step 1: Add explicit storage/schema fields**

Room creation accepts optional:

- `experience_mode`
- `human_seat`
- `ruleset_id`
- `profile_pack_id`
- `share_code`

- [ ] **Step 2: Validate but do not fully implement human gameplay yet**

If `experience_mode="human_seat"`, require `human_seat` in 1-12. Store fields in an explicit game metadata/config snapshot structure, not just response payload.

- [ ] **Step 3: Display selected human seat in UI**

This prepares the product path without claiming live human turn handling is complete.

---

### Future Phase: RoleAbilityRegistry And Wolf King Guard Playable Rules

This phase is intentionally not part of the first executable slice. It records the path for turning `guard`/`wolf_king` from `display_only` into real `playable` rules.

**Files:**
- Create: `werewolf_agent/engine/abilities.py`
- Modify: `werewolf_agent/engine/rule_engine.py`
- Modify: `werewolf_agent/runtime/graph.py`
- Modify: `werewolf_agent/cognition/visibility.py`
- Modify: `werewolf_agent/customization/ruleset_registry.py`
- Test: `tests/rules/test_guard_ability.py`
- Test: `tests/rules/test_wolf_king_ability.py`
- Test: `tests/runtime/test_death_trigger_chain.py`
- Test: `tests/integration/test_wolf_king_guard_visibility.py`

- [ ] **Future Step 1: Add RoleAbilityRegistry**

Define built-in abilities only. User YAML can reference `ability_id`, but cannot provide executable code.

Initial ability metadata:

- `wolf_kill`
- `witch_potion`
- `seer_check`
- `hunter_shot`
- `idiot_reveal`
- `hybrid_bind`
- `guard_protect`
- `wolf_king_shot`

- [ ] **Future Step 2: Generalize death trigger chain**

Refactor current hunter-shot-specific handling into a reusable chain:

- collect triggered abilities from death reason and role
- resolve in deterministic order
- append auditable events
- prevent infinite chain loops
- run victory checks after chain completion

- [ ] **Future Step 3: Implement wolf_king_shot**

Configurable trigger:

```yaml
abilities:
  - id: wolf_king_shot
    triggers:
      death_reasons: [exile, hunter_shot]
      blocked_by: [witch_poison]
```

Required tests:

- exile triggers shot
- witch poison blocks shot
- shot creates death event
- shot can change victory
- public timeline does not reveal hidden role before reveal policy allows it

- [ ] **Future Step 4: Implement guard_protect**

Configurable guard:

```yaml
abilities:
  - id: guard_protect
    night_order: 15
    target_filter: alive_player
    can_self_target: true
    can_repeat_target: false
    same_guard_and_antidote_policy: target_dies
```

Required tests:

- guarded wolf-kill target survives
- repeat target blocked when configured
- self target allowed/blocked by config
- guard target remains private
- replay restores guard action and resolution

- [ ] **Future Step 5: Add night protection resolution**

Unify wolf kill, guard protection, witch antidote, witch poison, and same-guard-and-antidote policy into one auditable resolution batch.

- [ ] **Future Step 6: Upgrade marketplace status**

Only after all above tests pass may the wolf king guard marketplace ruleset move from `display_only` to `playable`.

---

### Future Phase: Platform Professionalization Roadmap

This phase records future platform capabilities that should not block the first customization slice.

**Future Capability Areas:**

- Room experience presets.
- AI player skill levels.
- Human-player training modes.
- Multi-perspective replay.
- Key moment timeline.
- Game quality scoring.
- Judge/commentator separation.
- Configuration sandbox simulation.
- Template health score.
- Agent behavior regression scenarios.
- Prompt/model A/B experiments.
- Operations console.
- Content moderation and copyright boundaries.
- AI player memory growth.
- Cost budget modes.

- [ ] **Future Step 1: Room experience presets**

Create presets such as beginner tutorial, standard competitive, high-pressure bluffing, wolf-favored, good-favored, variety-show, and replay training. Each preset binds ruleset, persona pack, AI skill level, commentary mode, game pace, and cost budget.

- [ ] **Future Step 2: AI player skill levels**

Separate persona style from skill level:

- `novice`
- `normal`
- `advanced`
- `competitive`
- `stress_test`

The same persona pack should be runnable at different skill levels.

- [ ] **Future Step 3: Human training modes**

Extend human-seat mode into targeted drills:

- seer claim
- witch potion use
- wolf fake-claim
- villager defense
- hunter reveal
- sheriff-style vote leadership

Each drill returns personal scoring and next-game recommendations.

- [ ] **Future Step 4: Multi-perspective replay**

Replay views:

- public
- player
- wolf team
- moderator
- god-view
- AI private strategy

Public sharing must default to public view only.

- [ ] **Future Step 5: Key moment timeline and quality score**

Automatically mark first seer claim, counterclaim, potion use, wolf vote push, vote split, and win/loss turning points. Add quality metrics for speech, vote, faction cooperation, role skill value, leak risk, and drama.

- [ ] **Future Step 6: Judge/commentator separation**

Keep judge deterministic and visibility-safe. Commentator sees only authorized view data and can switch between beginner, advanced, variety-show, and silent modes.

- [ ] **Future Step 7: Configuration sandbox validation**

Run uploaded rules/persona packs through simulated games before marketplace recommendation. Report completion rate, illegal action rate, faction balance, average length, leak risk, and estimated cost.

- [ ] **Future Step 8: Template health score**

Compute playability, balance, completion rate, average game length, illegal action rate, leak risk, reuse, and favorites.

- [ ] **Future Step 9: Agent behavior regression scenarios**

Create fixed scenarios for seer counterclaim, witch first-night self-save, wolf teammate under pressure, villager defense, and poisoned hunter. Run after prompt/model/persona/strategy changes.

- [ ] **Future Step 10: Prompt/model A/B experiments**

Compare model, persona pack, RAG strategy, commentary style, and ruleset balance using win rate, illegal action rate, game length, leak risk, cost, and replay quality.

- [ ] **Future Step 11: Operations, moderation, memory, and cost controls**

Add operations dashboard, upload moderation, attribution/copyright metadata, AI memory-growth modes, and cost budget modes.

---

## Verification Checklist

Run after each task:

```powershell
pytest tests/customization tests/api/test_customization_api.py tests/ui/test_dashboard_enhanced.py -q
```

Run before claiming completion:

```powershell
pytest tests/customization tests/api tests/ui tests/runtime tests/storage -q
pytest -q
```

Manual browser check after UI tasks:

```powershell
python -m uvicorn werewolf_agent.api.app:create_app --factory --host 127.0.0.1 --port 18000
```

Open `http://127.0.0.1:18000/` and verify:

- Launch wizard controls render.
- Rule template download works.
- Persona template download works.
- Valid uploads show summaries.
- Invalid uploads show field-level errors.
- Existing create/start/step controls still work.

## Documentation Updates

After each completed task:

- Update `PROGRESS.md`.
- If supported fields change, update `docs/design/platform-customization-growth-design.md`.
- If RuleEngine adjudication semantics change, update `docs/design/werewolf-agent-v1-design.md` and rule tests first.
