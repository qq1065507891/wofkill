# Public Fact and Timeline Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除公开发言中的归属、条件、死亡原因和夜数误提取，并保证 N1 查验理由不引用 D1 信息。

**Architecture:** 在 `core` 新增一个无运行时副作用的窄语义解析模块，统一向公开摘要、认知世界状态和预言家声明校验器提供分句后的结构化声明。真实查验理由随夜间选择事件保存；悍跳查验口径在首个白天发言前冻结，白天提示只消费快照。

**Tech Stack:** Python 3.12、dataclasses、Enum、Pydantic、pytest、现有 GameEvent/StructuredFact/AgentContext。

## Global Constraints

- 设计依据：`docs/superpowers/specs/2026-07-28-public-fact-timeline-correctness-design.md`。
- 所有 Python、pytest 和静态检查命令通过 `conda run -n wofkill` 执行。
- 新 Python 文件使用 UTF-8 声明、中文模块 docstring、`作者: Project contributors` 和日期 `2026-07-28`。
- 保持现有 `build_public_summary()`、`_infer_claims_from_text()`、`extract_seer_claims()` 和 `validate_seer_claim()` 调用方式兼容。
- 无法可靠分类的文本不产生执行事实；不得用更宽松的正则兜底。
- 每个任务先完成红测，再写最小实现，并独立提交。

---

## File Structure

- `werewolf_agent/core/public_speech_semantics.py`：分句、模态、归属、夜数及死亡/查验声明解析；不读取 GameState。
- `werewolf_agent/runtime/context_public_summary.py`：把结构化声明渲染为公开摘要。
- `werewolf_agent/cognition/world_state.py`：把结构化声明投影为 StructuredFact。
- `werewolf_agent/runtime/seer_claim_validator.py`：基于同一解析结果执行夜数和一夜一验约束。
- `werewolf_agent/runtime/nodes/night_specialists.py`：在私有选择事件中保存真实查验理由。
- `werewolf_agent/runtime/directives/seer.py`：向预言家白天提示提供冻结理由。
- `werewolf_agent/runtime/sheriff_election_directives.py`：移除时间穿越示例并渲染真实/悍跳快照。
- `werewolf_agent/runtime/agent_sheriff_speech_actions.py`：构建竞选发言时接入冻结口径。

### Task 1: 建立共享公开声明解析器

**Files:**
- Create: `werewolf_agent/core/public_speech_semantics.py`
- Create: `tests/core/test_public_speech_semantics.py`

**Interfaces:**
- Consumes: 原始公开发言字符串和说话人 ID。
- Produces: `SpeechClaimKind`、`SpeechClaimModality`、`ParsedSpeechClaim` 和 `parse_public_speech_claims(text: str, *, speaker_id: str) -> tuple[ParsedSpeechClaim, ...]`。

- [ ] **Step 1: 写真实日志回归红测**

```python
def test_reported_wolf_accusation_is_not_self_claim() -> None:
    claims = parse_public_speech_claims(
        "p03直接点名我是狼人。", speaker_id="p01",
    )
    assert not any(
        claim.kind is SpeechClaimKind.ROLE
        and claim.speaker_id == "p01"
        and claim.value == "werewolf"
        for claim in claims
    )


def test_conditional_badge_flow_is_not_completed_check() -> None:
    claims = parse_public_speech_claims(
        "首夜我验了p01，结果是好人。N2验p02若为好人，警徽给p02。",
        speaker_id="p08",
    )
    assert [(c.night_number, c.target_id, c.modality) for c in claims if c.kind is SpeechClaimKind.SEER_CHECK] == [
        (1, "p01", SpeechClaimModality.COMPLETED),
        (2, "p02", SpeechClaimModality.CONDITIONAL),
    ]


def test_plain_werewolf_word_is_not_wolf_kill() -> None:
    claims = parse_public_speech_claims(
        "p04查验p01也是狼人，并且验p03是金水。", speaker_id="p08",
    )
    assert not any(c.kind is SpeechClaimKind.DEATH for c in claims)
```

- [ ] **Step 2: 运行红测**

Run: `conda run -n wofkill python -m pytest -n 0 tests/core/test_public_speech_semantics.py -q`

Expected: FAIL，因为模块和接口尚不存在。

- [ ] **Step 3: 实现最小共享模型和分句解析**

```python
class SpeechClaimKind(str, Enum):
    ROLE = "role"
    SEER_CHECK = "seer_check"
    DEATH = "death"
    BADGE_FLOW = "badge_flow"


class SpeechClaimModality(str, Enum):
    COMPLETED = "completed"
    CONDITIONAL = "conditional"
    FUTURE = "future"
    REPORTED = "reported"
    INFERENCE = "inference"


@dataclass(frozen=True)
class ParsedSpeechClaim:
    kind: SpeechClaimKind
    modality: SpeechClaimModality
    speaker_id: str
    target_id: str | None = None
    value: str = ""
    night_number: int | None = None
    source_text: str = ""


def parse_public_speech_claims(
    text: str, *, speaker_id: str,
) -> tuple[ParsedSpeechClaim, ...]:
    clauses = tuple(part.strip() for part in re.split(r"[。！？；;\n]+", text) if part.strip())
    parsed: list[ParsedSpeechClaim] = []
    for clause in clauses:
        modality = _claim_modality(clause)
        parsed.extend(_parse_role_claims(clause, speaker_id, modality))
        parsed.extend(_parse_seer_claims(clause, speaker_id, modality))
        parsed.extend(_parse_death_claims(clause, speaker_id, modality))
    return tuple(parsed)
```

实现中 `_claim_modality()` 对“若/如果”返回 `CONDITIONAL`，对“今晚/下一夜/计划”返回 `FUTURE`，对“点名/说我/称我”返回 `REPORTED`。夜数解析覆盖 `首夜`、`第一夜`、`第1夜` 和 `N1`。死亡解析只接受明确的“狼刀/被刀/被狼人杀害/毒杀”，不能用 `狼[刀杀人]` 字符类。

- [ ] **Step 4: 运行共享解析器测试**

Run: `conda run -n wofkill python -m pytest -n 0 tests/core/test_public_speech_semantics.py -q`

Expected: PASS。

- [ ] **Step 5: 提交共享解析器**

```bash
git add werewolf_agent/core/public_speech_semantics.py tests/core/test_public_speech_semantics.py
git commit -m "fix: classify public speech claims safely"
```

### Task 2: 让公开摘要和认知事实消费共享解析结果

**Files:**
- Modify: `werewolf_agent/runtime/context_public_summary.py:169-207`
- Modify: `werewolf_agent/cognition/world_state.py:241-402`
- Modify: `tests/runtime/test_context_public_summary.py`
- Modify: `tests/cognition/test_cognition.py`

**Interfaces:**
- Consumes: Task 1 的 `parse_public_speech_claims()`。
- Produces: 只包含 `COMPLETED` 查验的 `[验人]` 摘要；正确归属的 `claimed_role`、`seer_check_claim` 和死亡声明事实。

- [ ] **Step 1: 写两个消费者的红测**

```python
def test_summary_keeps_first_night_check_and_excludes_conditional_n2() -> None:
    items: list[tuple[int, str]] = []
    _append_speech_claims(items, {
        "speaker": "p08",
        "text": "首夜我验了p01是好人；N2验p02若为好人，警徽给p02。",
    })
    assert (1, "[验人] p08 报 N1 p01=好人") in items
    assert all("N2 p02=好人" not in text for _, text in items)


def test_world_state_keeps_reported_wolf_accusation_out_of_role_claims() -> None:
    facts = _infer_claims_from_text(
        speaker="p01", text="p03直接点名我是狼人。", day=1,
    )
    assert not any(f.fact_type == "claimed_role" and f.value == "werewolf" for f in facts)
```

- [ ] **Step 2: 确认现有代码失败**

Run: `conda run -n wofkill python -m pytest -n 0 tests/runtime/test_context_public_summary.py tests/cognition/test_cognition.py -k "conditional_n2 or reported_wolf" -q`

Expected: FAIL，分别显示虚假 N2 查验和自认狼事实。

- [ ] **Step 3: 用结构化声明替换宽松正则**

```python
claims = parse_public_speech_claims(text, speaker_id=speaker)
for claim in claims:
    if claim.kind is SpeechClaimKind.SEER_CHECK and claim.modality is SpeechClaimModality.COMPLETED:
        result_cn = "狼人" if claim.value == "wolf" else "好人"
        summary_items.append((1, f"[验人] {speaker} 报 N{claim.night_number} {claim.target_id}={result_cn}"))
```

在 `world_state.py` 中只把直接 `ROLE` 声明转为 `claimed_role`；转述保留在普通 speech 文本，不生成说话人的身份声明。`CONDITIONAL/FUTURE` 查验只生成 `badge_flow_claim` 或不提取，不能成为 `seer_check_claim`。

- [ ] **Step 4: 运行消费者和既有账本测试**

Run: `conda run -n wofkill python -m pytest -n 0 tests/runtime/test_context_public_summary.py tests/cognition/test_cognition.py tests/runtime/test_public_ledger.py -q`

Expected: PASS。

- [ ] **Step 5: 提交消费者迁移**

```bash
git add werewolf_agent/runtime/context_public_summary.py werewolf_agent/cognition/world_state.py tests/runtime/test_context_public_summary.py tests/cognition/test_cognition.py
git commit -m "fix: share public claim semantics across ledgers"
```

### Task 3: 统一预言家夜数校验和公开死亡原因合同

**Files:**
- Modify: `werewolf_agent/runtime/seer_claim_validator.py:24-99`
- Modify: `werewolf_agent/runtime/speech_quality.py:101-159`
- Modify: `werewolf_agent/runtime/day_vote_directives.py`
- Modify: `tests/runtime/test_seer_claim_validator.py`
- Modify: `tests/runtime/test_speech_quality.py`
- Modify: `tests/runtime/test_judge_flow.py:228-258`
- Modify: `tests/runtime/test_visible_state.py`
- Modify: `tests/runtime/test_day_vote_directives.py`

**Interfaces:**
- Consumes: `parse_public_speech_claims()`、`day_number` 和公开 `judge_broadcast`/`dead_players`。
- Produces: `extract_seer_claims()` 兼容字典列表；`validate_seer_claim()` 的稳定错误字符串；一致的公开死亡原因判断。

- [ ] **Step 1: 写首夜、条件计划和公开死亡原因红测**

```python
def test_first_night_alias_is_valid_completed_check() -> None:
    assert extract_seer_claims("首夜我验了p04是好人") == [
        {"night": 1, "target_id": "p04"},
    ]


def test_conditional_future_check_does_not_count_as_completed() -> None:
    assert validate_seer_claim(
        "N2验p02若为好人，警徽给p02", day_number=1,
    ) is None


def test_public_judge_death_reason_is_valid_public_evidence() -> None:
    result = validate_public_speech(
        "法官已公告p01毒杀，我据此重新分析票型。",
        phase="day_discussion",
        context={"public_summary": "[死讯] p01(毒杀)"},
    )
    assert "public_record_grounding" not in result.get("missing_fields", [])
```

- [ ] **Step 2: 运行红测**

Run: `conda run -n wofkill python -m pytest -n 0 tests/runtime/test_seer_claim_validator.py tests/runtime/test_speech_quality.py tests/runtime/test_judge_flow.py tests/runtime/test_visible_state.py tests/runtime/test_day_vote_directives.py -q`

Expected: 至少首夜解析和当前互相矛盾的死亡公告合同失败。

- [ ] **Step 3: 基于共享解析器实现校验并统一测试合同**

```python
def extract_seer_claims(speech: str) -> list[dict[str, Any]]:
    return [
        {"night": claim.night_number, "target_id": claim.target_id}
        for claim in parse_public_speech_claims(speech, speaker_id="public_speaker")
        if claim.kind is SpeechClaimKind.SEER_CHECK
        and claim.modality is SpeechClaimModality.COMPLETED
        and claim.target_id is not None
    ]
```

保留一夜一验和未来夜校验。将 `test_public_death_announcement_does_not_reveal_death_reason` 改为断言公开消息、visible state 和投票提示都使用同一原因标签；同时新增“尚未公告前不可见”测试，避免夜间提前泄露。

- [ ] **Step 4: 运行时间线和公开可见性测试**

Run: `conda run -n wofkill python -m pytest -n 0 tests/runtime/test_seer_claim_validator.py tests/runtime/test_speech_quality.py tests/runtime/test_judge_flow.py tests/runtime/test_visible_state.py tests/runtime/test_day_vote_directives.py tests/runtime/test_day_discussion.py -q`

Expected: PASS。

- [ ] **Step 5: 提交规则合同统一**

```bash
git add werewolf_agent/runtime/seer_claim_validator.py werewolf_agent/runtime/speech_quality.py werewolf_agent/runtime/day_vote_directives.py tests/runtime/test_seer_claim_validator.py tests/runtime/test_speech_quality.py tests/runtime/test_judge_flow.py tests/runtime/test_visible_state.py tests/runtime/test_day_vote_directives.py
git commit -m "fix: enforce public timeline claim contract"
```

### Task 4: 冻结真实预言家的夜间查验理由

**Files:**
- Modify: `werewolf_agent/runtime/agent_special_actions.py:346-354`
- Modify: `werewolf_agent/runtime/nodes/night_specialists.py:190-223`
- Modify: `werewolf_agent/runtime/directives/seer.py:48-169`
- Modify: `werewolf_agent/runtime/sheriff_election_directives.py:171-183`
- Modify: `tests/runtime/test_seer_flow.py`
- Modify: `tests/runtime/test_seer_night_directives.py`
- Modify: `tests/runtime/test_sheriff_election_directives.py`

**Interfaces:**
- Consumes: 夜间 `PlayerAction.reason`、`seer_check_selected` 和 `seer_check` 事件。
- Produces: 私有事件字段 `selection_reason: str`；`my_check_history[*].selection_reason`；不引用未来发言的竞选提示。

- [ ] **Step 1: 写夜间理由冻结红测**

```python
def test_seer_choice_persists_private_selection_reason() -> None:
    result = night_seer(state_with_action_reason("首夜随机覆盖未知位置"))
    event = next(e for e in result["game_state"].events if e.type == "seer_check_selected")
    assert event.payload["selection_reason"] == "首夜随机覆盖未知位置"


def test_day_directive_uses_stored_reason_not_day_speech() -> None:
    parts = build_seer_directive(game_with_stored_reason("首夜随机覆盖未知位置"), "p08")
    assert parts["my_check_history"][0]["selection_reason"] == "首夜随机覆盖未知位置"
    assert "D1警上发言" not in repr(parts)
```

在 `tests/runtime/test_seer_flow.py` 中定义 `state_with_action_reason()`：复用现有 12 人夜间 fixture，并让 FakeAgent 返回带该 `reason` 的 `seer_action_trace.final_action`。在 `tests/runtime/test_seer_night_directives.py` 中定义 `game_with_stored_reason()`：构造含 `seer_check_selected` 与 `seer_check` 私有事件的 `GameState`。

- [ ] **Step 2: 运行红测**

Run: `conda run -n wofkill python -m pytest -n 0 tests/runtime/test_seer_flow.py tests/runtime/test_seer_night_directives.py tests/runtime/test_sheriff_election_directives.py -k "selection_reason or stored_reason" -q`

Expected: FAIL，因为选择事件和历史中没有 `selection_reason`。

- [ ] **Step 3: 从 action trace 提取并保存安全理由**

```python
action_payload = result.get("seer_action_trace", {}).get("final_action", {})
selection_reason = str(action_payload.get("reason") or "").strip()
choice_payload = {
    "target_id": target,
    "selection_reason": selection_reason[:240],
}
```

把同一字段附加到最终 `seer_check`/`seer_check_resolved` 私有事件，并在 `my_check_history` 投影。修改 `build_seer_verification_rationale()`：N1 只能使用已存理由；无理由时输出“首夜随机查验”，删除基于发言量和投票的示例。

- [ ] **Step 4: 运行预言家完整测试**

Run: `conda run -n wofkill python -m pytest -n 0 tests/runtime/test_seer_flow.py tests/runtime/test_seer_night_directives.py tests/runtime/test_sheriff_election_directives.py tests/runtime/test_directive_role_gating.py -q`

Expected: PASS。

- [ ] **Step 5: 提交真实查验理由快照**

```bash
git add werewolf_agent/runtime/agent_special_actions.py werewolf_agent/runtime/nodes/night_specialists.py werewolf_agent/runtime/directives/seer.py werewolf_agent/runtime/sheriff_election_directives.py tests/runtime/test_seer_flow.py tests/runtime/test_seer_night_directives.py tests/runtime/test_sheriff_election_directives.py
git commit -m "fix: freeze seer rationale at night"
```

### Task 5: 冻结悍跳查验口径并执行整域验收

**Files:**
- Modify: `werewolf_agent/runtime/agent_wolf_actions.py`
- Modify: `werewolf_agent/runtime/sheriff_election_directives.py:186-215`
- Modify: `werewolf_agent/runtime/agent_sheriff_speech_actions.py:100-135`
- Modify: `tests/runtime/test_agent_wolf_team_plan.py`
- Modify: `tests/runtime/test_sheriff_election_directives.py`
- Modify: `tests/integration/test_wolf_team_plan_e2e.py`

**Interfaces:**
- Consumes: 狼队计划、首个白天开始前的存活玩家和既定 `fake_seer`。
- Produces: `freeze_fake_seer_check(gs, wolf_plan) -> dict[str, Any]`；`wolf_team_plan.fake_check = {night, target_id, alignment, selection_reason}`；悍跳发言必须复用该结构。

- [ ] **Step 1: 写悍跳快照红测**

```python
def test_fake_seer_plan_freezes_n1_check_before_sheriff_speeches() -> None:
    plan = freeze_fake_seer_check(state["game_state"], {"fake_seer": "p03"})
    fake = plan["fake_check"]
    assert fake["night"] == 1
    assert fake["target_id"] in state["game_state"].players
    assert fake["alignment"] in {"good", "wolf"}
    assert "发言" not in fake["selection_reason"]
    assert "投票" not in fake["selection_reason"]
```

测试中的 `state` 使用现有 wolf-team-plan fixture，且 `p03` 为存活狼人；目标选择按存活非狼人 ID 排序，保证不调用模型也可重复。

- [ ] **Step 2: 运行红测**

Run: `conda run -n wofkill python -m pytest -n 0 tests/runtime/test_agent_wolf_team_plan.py tests/runtime/test_sheriff_election_directives.py tests/integration/test_wolf_team_plan_e2e.py -k "fake_seer_plan_freezes" -q`

Expected: FAIL，因为现有计划没有 `fake_check`。

- [ ] **Step 3: 生成并强制复用确定性假查验结构**

```python
def freeze_fake_seer_check(
    gs: GameState,
    wolf_plan: dict[str, Any],
) -> dict[str, Any]:
    target_id = sorted(
        player_id
        for player_id, player in gs.players.items()
        if player.alive and player.role != "werewolf"
    )[0]
    wolf_plan["fake_check"] = {
        "night": 1,
        "target_id": target_id,
        "alignment": "good",
        "selection_reason": "首夜在未知位置中预设查验口径",
    }
    return wolf_plan
```

`build_wolf_sheriff_election_directives()` 必须把四个字段作为强制口径渲染，不允许模型重新选择目标、结果或基于 D1 发言补理由。计划缺失时 fail closed 为固定“首夜随机预设”口径，不读取当前 transcript。

- [ ] **Step 4: 运行整域测试和静态检查**

Run: `conda run -n wofkill python -m pytest -n 0 tests/core/test_public_speech_semantics.py tests/runtime/test_context_public_summary.py tests/cognition/test_cognition.py tests/runtime/test_seer_claim_validator.py tests/runtime/test_seer_flow.py tests/runtime/test_sheriff_election_directives.py tests/integration/test_wolf_team_plan_e2e.py -q`

Run: `conda run -n wofkill python -m ruff check werewolf_agent/core/public_speech_semantics.py werewolf_agent/runtime/context_public_summary.py werewolf_agent/runtime/seer_claim_validator.py werewolf_agent/runtime/sheriff_election_directives.py`

Expected: 两条命令均退出 0。

- [ ] **Step 5: 运行全量测试**

Run: `conda run -n wofkill python -m pytest -n 0 -q`

Expected: PASS；若存在与本计划无关的既有失败，记录完整测试名和基线证据，不修改无关模块。

- [ ] **Step 6: 提交悍跳快照与整域验收**

```bash
git add werewolf_agent/runtime/agent_wolf_actions.py werewolf_agent/runtime/sheriff_election_directives.py werewolf_agent/runtime/agent_sheriff_speech_actions.py tests/runtime/test_agent_wolf_team_plan.py tests/runtime/test_sheriff_election_directives.py tests/integration/test_wolf_team_plan_e2e.py
git commit -m "fix: freeze fake seer public check story"
```
