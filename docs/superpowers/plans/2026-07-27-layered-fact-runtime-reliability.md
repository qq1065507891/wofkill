# Layered Fact Runtime Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复真实对局中的警徽流事实丢失、玩家声明与规则动作混淆、结构化摘要失败不可诊断、配置生效不可审计和反思持久化日志错位问题。

**Architecture:** 保留 GameEvent 事件流作为规则事实源，在现有 StructuredFact、公开账本和 AgentContext 上增加 authority/support_kind 分层投影。玩家声明、公开执行事实和 moderator-only 冲突审计分别进入不同投影；摘要、发言、投票和语义修复只消费当前玩家可见的结构化投影。

**Tech Stack:** Python 3.12、Pydantic、PyYAML、pytest、httpx fake client；所有 Python、pytest 和项目脚本命令通过 conda run -n wofkill 执行。

---

## 文件职责

- werewolf_agent/cognition/world_state.py：警徽流解析和事实来源元数据。
- werewolf_agent/runtime/public_ledger.py：公开声明、公开执行动作和可见冲突。
- werewolf_agent/runtime/context.py、werewolf_agent/agents/prompt_schemas.py：把结构化公开账本传入 AgentContext。
- werewolf_agent/agents/prompt_sections.py、prompt_composer.py、prompt_user_context.py：渲染不可丢弃的分层事实区段。
- werewolf_agent/agents/player_fallback_speech.py：fallback 只引用带来源标签的线索。
- werewolf_agent/evaluation/balance_public_claims.py、werewolf_agent/agents/semantic_repair_audit.py：区分行动声明和已执行动作。
- werewolf_agent/agents/discussion_summary.py、player.py、werewolf_agent/runtime/nodes/summary.py：摘要失败审计和一次有界修复。
- werewolf_agent/model_gateway/usage_records.py、router_errors.py、scripts/run_real_game_reports.py：记录最终有效采样参数。
- werewolf_agent/runtime/game_runner_memory.py、reflection_transaction.py：反思生成与持久化回读分阶段审计。

config/models.yaml 的温度值不在本轮修改。当前配置已经按摘要、投票、发言和欺骗任务拆分；本计划增加实际 YAML 路由、wire payload 和运行审计验证。

每次非平凡修改 Python 文件时，同步检查模块中文 docstring 是否仍准确，并把已有“修改日期”更新为 2026-07-27；新增注释使用简洁中文。

### Task 1: Lock the g_42 regressions

**Files:**
- Modify: tests/cognition/test_cognition.py
- Modify: tests/runtime/test_public_ledger.py
- Modify: tests/agents/test_discussion_summary.py
- Modify: tests/runtime/test_game_runner.py

- [ ] **Step 1: Add the exact badge-flow regression**

~~~python
def test_g42_real_log_badge_flow_keeps_night_prefix_and_order() -> None:
    from werewolf_agent.cognition.world_state import _infer_claims_from_text

    claims = _infer_claims_from_text(
        speaker="p03",
        text=(
            "接下来公布警徽流：\n"
            "- 第二夜N2，我计划查验p02（赵猛）。理由：覆盖警上候选人。\n"
            "- 第三夜N3，我计划查验p12（冯弈）。理由：形成对照。\n"
        ),
        day=1,
    )
    badge = [item for item in claims if item.fact_type == "badge_flow_claim"]
    assert len(badge) == 1
    assert badge[0].target_player == "p02"
    assert badge[0].metadata["badge_flow_order"] == ["p02", "p12"]
~~~

- [ ] **Step 2: Add the action-claim characterization**

~~~python
def test_last_words_action_is_not_an_executed_action() -> None:
    state = GameState(
        game_id="g42_action_claim",
        events=[GameEvent(
            type="night_death_last_words",
            visibility="public",
            payload={
                "day_number": 1,
                "speaker": "p07",
                "text": "我是猎人，现在开枪带走p01。",
            },
        )],
    )
    ledger = build_public_ledger(state)
    assert ledger["action_claims"][0]["authority"] == "player_claim"
    assert ledger["action_claims"][0]["target"] == "p01"
    assert ledger.get("confirmed_actions", []) == []
~~~

- [ ] **Step 3: Add summary and reflection audit characterizations**

Add a malformed summary provider test asserting failure_code=invalid_json, response_shape=text, json_candidate_count=0, and no raw provider text in the audit. Add a caplog test asserting the reflection summary node reports generated/verified counts but does not print “持久化完成”.

~~~python
with pytest.raises(DiscussionSummaryGenerationError) as exc_info:
    _summary_agent_for_text("not json").summarize_discussion(_summary_context())
assert exc_info.value.audit["response_shape"] == "text"
assert exc_info.value.audit["json_candidate_count"] == 0
assert "not json" not in repr(exc_info.value.audit)

caplog.set_level(logging.DEBUG, logger="werewolf_agent.runtime.nodes.summary")
summary.reflection(runtime_state)
assert "持久化完成" not in caplog.text
~~~

- [ ] **Step 4: Run the pre-fix subset**

~~~bash
conda run -n wofkill python -m pytest -q -o addopts='' \
  tests/cognition/test_cognition.py \
  tests/runtime/test_public_ledger.py \
  tests/agents/test_discussion_summary.py \
  tests/runtime/test_game_runner.py
~~~

Expected before implementation: the new badge, ledger and audit tests fail while existing tests pass.

- [ ] **Step 5: Commit the characterization tests**

~~~bash
git add tests/cognition/test_cognition.py tests/runtime/test_public_ledger.py \
  tests/agents/test_discussion_summary.py tests/runtime/test_game_runner.py
git commit -m "test: capture g42 fact and runtime regressions"
~~~

### Task 2: Fix badge parsing and fact provenance

**Files:**
- Modify: werewolf_agent/cognition/world_state.py
- Test: tests/cognition/test_cognition.py

- [ ] **Step 1: Add claim and engine metadata tests**

For a public speech claim assert authority=player_claim and support_kind=public_speech. For a seer_check engine event assert authority=engine and support_kind=executed_action. Also assert source_event and visibility remain present.

~~~python
speech_fact = next(
    fact for fact in extract_facts(speech_event, state)
    if fact.fact_type == "claimed_role"
)
assert speech_fact.metadata["authority"] == "player_claim"
assert speech_fact.metadata["support_kind"] == "public_speech"

engine_fact = extract_facts(seer_event, state)[0]
assert engine_fact.metadata["authority"] == "engine"
assert engine_fact.metadata["support_kind"] == "executed_action"
~~~

- [ ] **Step 2: Add the bounded Chinese-night plan pattern**

~~~python
_BADGE_FLOW_PLAN_LINE_RE = re.compile(
    r"^\s*[-*]?\s*"
    r"(?:第[一二三四五六七八九十0-9]+夜\s*)?N\d+\s*"
    r"[,:：，]?\s*(?:我\s*)?(?:计划\s*)?"
    r"(?:查验|验|查)\s*(p\d{2})(?![A-Za-z0-9_])"
)
~~~

Use this pattern after the existing compact and 先/后 formats. Read only consecutive matching plan lines; stop at a blank line, new section, or first ordinary sentence. Preserve one badge_flow_claim with first target in target_player and all targets in metadata["badge_flow_order"].

- [ ] **Step 3: Attach provenance at one boundary**

~~~python
_CLAIM_FACT_TYPES = frozenset({
    "claimed_role",
    "claimed_good",
    "claimed_suspect",
    "seer_check_claim",
    "badge_flow_claim",
})

def _fact_provenance(fact: StructuredFact, event: GameEvent) -> dict[str, str]:
    is_claim = fact.fact_type in _CLAIM_FACT_TYPES or fact.fact_type.startswith("claimed_")
    return {
        "authority": "player_claim" if is_claim else "engine",
        "support_kind": (
            "last_words"
            if event.type in {"exile_last_words", "night_death_last_words"}
            else "public_speech"
            if event.type in {"speech", "sheriff_speech", "sheriff_pk_speech"}
            else "executed_action"
        ),
    }
~~~

Merge this in _attach_event_metadata. Register last-words events with _extract_speech only when payload contains both speaker and text; the legacy “please speak” event must remain a generic event.

- [ ] **Step 4: Verify cognition and visibility**

~~~bash
conda run -n wofkill python -m pytest -q -o addopts='' \
  tests/cognition/test_cognition.py \
  tests/runtime/test_public_ledger.py \
  tests/runtime/test_visible_state.py \
  tests/runtime/test_context.py
~~~

- [ ] **Step 5: Commit**

~~~bash
git add werewolf_agent/cognition/world_state.py tests/cognition/test_cognition.py
git commit -m "fix: preserve badge flow and fact provenance"
~~~

### Task 3: Separate public action claims from engine facts

**Files:**
- Modify: werewolf_agent/runtime/public_ledger.py
- Modify: werewolf_agent/runtime/visible_state.py
- Test: tests/runtime/test_public_ledger.py
- Test: tests/runtime/test_visible_state.py

- [ ] **Step 1: Extend the stable ledger shape**

Keep all old keys and add action_claims, confirmed_actions and claim_conflicts as empty lists.

~~~python
ledger: dict[str, list[dict[str, Any]]] = {
    "role_claims": [],
    "seer_check_claims": [],
    "badge_flow_claims": [],
    "vote_records": [],
    "last_words": [],
    "badge_events": [],
    "action_claims": [],
    "confirmed_actions": [],
    "claim_conflicts": [],
}
~~~

- [ ] **Step 2: Parse a bounded action vocabulary**

~~~python
_ACTION_CLAIM_PATTERNS = (
    ("hunter_shot", re.compile(
        r"(?:开枪|带走)\s*(p\d{2})(?![A-Za-z0-9_])"
    )),
    ("witch_antidote", re.compile(
        r"(?:用解药救|解药救)了?\s*(p\d{2})(?![A-Za-z0-9_])"
    )),
)
~~~

Each item contains day, speaker, action, target, authority=player_claim, support_kind=last_words or public_speech, and source_event. Do not parse “守卫下轮守p04” into an executed action; the ruleset has no guard role and the sentence stays an unsupported player statement.

- [ ] **Step 3: Project public engine resolutions**

Add hunter_shot_resolved to PUBLIC_EVENT_TYPES. A public resolution produces:

~~~python
{
    "day": event.payload.get("day_number", 0),
    "actor": event.payload.get("actor_id"),
    "action": "hunter_shot",
    "target": event.payload.get("target_id"),
    "authority": "engine",
    "support_kind": "executed_action",
    "source_event": "hunter_shot_resolved",
}
~~~

Only an explicit public resolution may enter confirmed_actions. Mismatched actor/action/target creates claim_conflicts. Missing public resolution leaves the statement unconfirmed; absence is not converted into a public no-action fact.

- [ ] **Step 4: Add moderator-only full audit**

Add build_claim_action_audit(game_state), which may compare claims with private hunter/witch decision events. Return status confirmed, conflicts_with_engine, or unconfirmed. Never pass this result into build_visible_player_state or AgentContext.

~~~python
def build_claim_action_audit(game_state: GameState) -> list[dict[str, Any]]:
    public_claims = build_public_ledger(game_state)["action_claims"]
    engine_actions = _all_engine_action_evidence(game_state)
    return [
        {
            **claim,
            "status": _claim_action_status(claim, engine_actions),
            "visibility": "moderator_only",
        }
        for claim in public_claims
    ]
~~~

Tests must prove private witch_antidote_used and moderator hunter selections never appear in the player ledger.

- [ ] **Step 5: Verify and commit**

~~~bash
conda run -n wofkill python -m pytest -q -o addopts='' \
  tests/runtime/test_public_ledger.py \
  tests/runtime/test_visible_state.py \
  tests/runtime/test_vote_flow.py
git add werewolf_agent/runtime/public_ledger.py werewolf_agent/runtime/visible_state.py \
  tests/runtime/test_public_ledger.py tests/runtime/test_visible_state.py
git commit -m "feat: separate public action claims from engine facts"
~~~

### Task 4: Feed the layered ledger to prompts and fallbacks

**Files:**
- Modify: werewolf_agent/agents/prompt_schemas.py
- Modify: werewolf_agent/runtime/context.py
- Modify: werewolf_agent/agents/prompt_sections.py
- Modify: werewolf_agent/agents/prompt_composer.py
- Modify: werewolf_agent/agents/prompt_user_context.py
- Modify: werewolf_agent/agents/player_fallback_speech.py
- Test: tests/runtime/test_context.py
- Test: tests/agents/test_prompt_builder.py
- Test: tests/agents/test_player_retry.py

- [ ] **Step 1: Add the excluded context field**

~~~python
public_fact_ledger: dict[str, list[dict[str, Any]]] = Field(
    default_factory=dict,
    exclude=True,
    description="当前玩家可见的规则事实、声明和冲突结构化投影。",
)
~~~

Build the public ledger once in build_agent_context and pass it to this field. The moderator-only audit from Task 3 is never passed.

- [ ] **Step 2: Add a never-drop prompt section**

Register _build_public_fact_ledger in USER_SECTION_SPECS with drop_tier=None and public_record=True. Render bounded buckets:

~~~python
def _build_public_fact_ledger(self) -> str:
    ledger = self.context.public_fact_ledger
    if not ledger:
        return ""
    payload = {
        "confirmed_public_facts": ledger.get("confirmed_actions", [])[:6],
        "player_claims": (
            ledger.get("role_claims", [])[:8]
            + ledger.get("seer_check_claims", [])[:8]
            + ledger.get("badge_flow_claims", [])[:8]
            + ledger.get("action_claims", [])[:8]
        ),
        "claim_conflicts": ledger.get("claim_conflicts", [])[:6],
    }
    return "分层公开账本（声明不是规则执行事实）: " + self._compact_json(payload)
~~~

Compose it after public_summary and before visible_state.

- [ ] **Step 3: Preserve labeled clues in fallback**

Extend context_clues with at most one badge flow and one action claim. Use phrases “p03公开声明警徽流” and “仅为玩家声明”; never write “已执行” for action_claims. Append the clue only within the existing output budget.

~~~python
for item in context.public_fact_ledger.get("badge_flow_claims", [])[:1]:
    clues.append(
        f"{item.get('speaker')}公开声明警徽流："
        f"{'、'.join(item.get('targets', []))}"
    )
for item in context.public_fact_ledger.get("action_claims", [])[:1]:
    clues.append(
        f"{item.get('speaker')}公开声称{item.get('action')}目标"
        f"{item.get('target')}，仅为玩家声明"
    )
~~~

- [ ] **Step 4: Test prompt retention**

Build a context with the exact p03 speech. Assert p02, p12, 玩家声明 and 执行事实 remain in the prompt even with an oversized transcript. Assert fallback contains a labeled badge clue and does not upgrade claims to execution.

~~~python
prompt = PlayerPromptBuilder(context, "p08").build_user_prompt(RetryInfo())
assert all(marker in prompt for marker in ("p02", "p12", "玩家声明", "执行事实"))

fallback = build_fallback_speech(context)
assert "公开声明" in fallback
assert "已执行" not in fallback
~~~

- [ ] **Step 5: Verify and commit**

~~~bash
conda run -n wofkill python -m pytest -q -o addopts='' \
  tests/runtime/test_context.py \
  tests/runtime/test_visible_state.py \
  tests/agents/test_prompt_builder.py \
  tests/agents/test_player_retry.py
git add werewolf_agent/agents/prompt_schemas.py werewolf_agent/runtime/context.py \
  werewolf_agent/agents/prompt_sections.py werewolf_agent/agents/prompt_composer.py \
  werewolf_agent/agents/prompt_user_context.py werewolf_agent/agents/player_fallback_speech.py \
  tests/runtime/test_context.py tests/agents/test_prompt_builder.py \
  tests/agents/test_player_retry.py
git commit -m "feat: expose layered public facts to agents"
~~~

### Task 5: Gate action assertions on engine evidence

**Files:**
- Modify: werewolf_agent/evaluation/balance_public_claims.py
- Modify: werewolf_agent/agents/semantic_repair_audit.py
- Test: tests/agents/test_semantic_repair_invariants.py
- Test: tests/agents/test_player_retry.py

- [ ] **Step 1: Classify completed-action claims**

Add bounded action patterns for 开枪/带走 and 用解药救. Let classify_public_claims accept keyword-only speaker=None so “我” can be attributed to context.agent_id without breaking old callers.

~~~python
_PUBLIC_ACTION_CLAIM_RE = re.compile(
    r"(?P<actor>我|p\d{2})?[^，。；;]{0,12}"
    r"(?P<action>已经开枪|开枪带走|首夜用解药救了|用解药救了)"
    r"\s*(?P<target>p\d{2})(?![A-Za-z0-9_])"
)

def _extract_completed_action_claims(
    text: str,
    *,
    speaker: str | None = None,
) -> list[ClassifiedPublicClaim]:
    found: list[ClassifiedPublicClaim] = []
    for match in _PUBLIC_ACTION_CLAIM_RE.finditer(text):
        actor = match.group("actor")
        found.append(ClassifiedPublicClaim(
            claim_type=PublicClaimType.PLAYER_CLAIM,
            text=match.group(0),
            start=match.start(),
            end=match.end(),
            target=match.group("target"),
            support_kind=(
                "hunter_shot"
                if "开枪" in match.group("action")
                else "witch_antidote"
            ),
            speaker_attribution=speaker if actor == "我" else actor,
        ))
    return found
~~~

Call this helper from classify_public_claims and merge its results through the existing overlap resolver.

- [ ] **Step 2: Accept structured engine evidence**

Extend public_claim_audit_keys with public_evidence=None. Existing calls remain valid. For completed-action claims, verification requires a confirmed_actions item with identical actor, action and target; matching player speech alone does not prove execution.

~~~python
def public_claim_audit_keys(
    text: str,
    public_speeches: list[tuple[str, str]],
    *,
    speaker: str | None = None,
    public_evidence: Mapping[str, Any] | None = None,
) -> tuple[set[PublicClaimAuditKey], set[PublicClaimAuditKey]]:
    claims = classify_public_claims(text, speaker=speaker)
    verified = {
        public_claim_audit_key(claim)
        for claim in claims
        if _claim_is_supported(claim, public_speeches, public_evidence or {})
    }
    return {public_claim_audit_key(claim) for claim in claims}, verified
~~~

- [ ] **Step 3: Add a specific rejection reason**

~~~python
_REJECTION_REASON_ORDER = (
    "unsupported_public_claim",
    "executed_action_without_engine_evidence",
    "speaker_attribution_changed",
    "negation_changed",
)
~~~

Pass context.public_fact_ledger and context.agent_id from semantic_repair_audit. Use the new reason only for completed wording such as “已经开枪” or “首夜用解药救了”; “建议开枪” remains an inference.

- [ ] **Step 4: Add semantic tests**

~~~python
def test_action_claim_cannot_be_upgraded_to_execution() -> None:
    context = _context(
        public_claim_ledger=[{"speaker": "p07", "text": "p07声称要开枪带走p01"}],
        public_fact_ledger={"action_claims": [], "confirmed_actions": []},
    )
    source = PlayerAction(speech="p07声称要开枪带走p01")
    final = PlayerAction(speech="p07已经开枪带走p01")
    result = validate_semantic_repair(context, source, final)
    assert result.accepted is False
    assert "executed_action_without_engine_evidence" in result.reason_codes
~~~

Add the inverse test with a matching confirmed_actions engine item and assert it passes.

- [ ] **Step 5: Verify and commit**

~~~bash
conda run -n wofkill python -m pytest -q -o addopts='' \
  tests/agents/test_semantic_repair_invariants.py \
  tests/agents/test_player_retry.py \
  tests/agents/test_task_terminal_fallbacks.py
git add werewolf_agent/evaluation/balance_public_claims.py \
  werewolf_agent/agents/semantic_repair_audit.py \
  tests/agents/test_semantic_repair_invariants.py tests/agents/test_player_retry.py
git commit -m "fix: gate action claims on engine evidence"
~~~

### Task 6: Diagnose and repair discussion-summary failures

**Files:**
- Modify: werewolf_agent/agents/discussion_summary.py
- Modify: werewolf_agent/agents/player.py
- Modify: werewolf_agent/runtime/nodes/summary.py
- Test: tests/agents/test_discussion_summary.py
- Test: tests/runtime/test_day_discussion.py

- [ ] **Step 1: Add a sanitized audit to the error**

Extend DiscussionSummaryGenerationError with audit: dict[str, Any]. Retain only structured_output_mode, tool_call_required, tool_call_received, response_shape, json_candidate_count, failure_stage and failure_code. str(error) remains the safe failure code. Raw text, prompts, exceptions and provider payloads are forbidden.

~~~python
class DiscussionSummaryGenerationError(RuntimeError):
    def __init__(self, failure_code: str, *, audit: dict[str, Any] | None = None) -> None:
        safe_code = failure_code if failure_code in _SAFE_FAILURE_CODES else "model_failure"
        super().__init__(safe_code)
        self.failure_code = safe_code
        self.audit = {
            "failure_code": safe_code,
            **_sanitize_summary_audit(audit or {}),
        }
~~~

- [ ] **Step 2: Classify parser stages**

Map JSONDecodeError and ambiguous-object ValueError to failure_stage=protocol/failure_code=invalid_json. Map ValidationError to failure_stage=schema/failure_code=schema_validation_failed. Empty text remains empty_response and provider exceptions remain model_generation_failed.

~~~python
except json.JSONDecodeError as exc:
    raise _summary_error("invalid_json", result, "protocol") from exc
except ValidationError as exc:
    raise _summary_error("schema_validation_failed", result, "schema") from exc
except ValueError as exc:
    raise _summary_error("invalid_json", result, "protocol") from exc
~~~

- [ ] **Step 3: Add one bounded repair attempt**

Use one GenerationAttemptContext across two calls. After the first parse failure call reject_latest_output(), append this fixed instruction, and retry once:

~~~python
repair_prompt = (
    prompt
    + "\n只输出一个符合 submit_discussion_summary Schema 的 JSON 对象；"
    "不要输出解释、数组或多个对象。"
)
~~~

Do not include failed response text. Keep the resolved text_json mode when the route has no native tool fallback. Never add an unbounded loop.

- [ ] **Step 4: Persist audit fields in the node**

Change discussion_summary_audit_records to list[dict[str, Any]]. On known failure merge exc.audit. On generic failure write response_shape=unknown and failure_stage=provider. Deterministic fallback must use structured ledger evidence_refs when available, so badge/check facts survive summary failure.

~~~python
except DiscussionSummaryGenerationError as exc:
    failure_code = exc.failure_code
    failure_audit = exc.audit

audit_records.append({
    "player_id": pid,
    "task": TaskType.DISCUSSION_SUMMARY.value,
    "outcome": "deterministic_fallback",
    "failure_code": failure_code,
    **failure_audit,
})
~~~

- [ ] **Step 5: Verify and commit**

Cover native tool arguments, fenced JSON, empty response, zero JSON candidates, two valid objects and schema violations.

~~~bash
conda run -n wofkill python -m pytest -q -o addopts='' \
  tests/agents/test_discussion_summary.py \
  tests/agents/test_output_parser.py \
  tests/runtime/test_day_discussion.py \
  tests/runtime/test_summary_visibility.py
git add werewolf_agent/agents/discussion_summary.py werewolf_agent/agents/player.py \
  werewolf_agent/runtime/nodes/summary.py tests/agents/test_discussion_summary.py \
  tests/runtime/test_day_discussion.py
git commit -m "fix: audit and repair discussion summary protocols"
~~~

### Task 7: Audit effective configuration

**Files:**
- Modify: werewolf_agent/model_gateway/usage_records.py
- Modify: werewolf_agent/model_gateway/router_errors.py
- Modify: scripts/run_real_game_reports.py
- Test: tests/model_gateway/test_per_profile_url_and_extra_body.py
- Test: tests/model_gateway/test_execution_records.py
- Test: tests/scripts/test_run_real_game.py

- [ ] **Step 1: Persist effective sampling values**

Add defaulted effective_temperature and temperature_override_reason fields to UsageRecord. In _record_success_usage copy both from GenerateResult. Failed/skipped routes that never reached a provider keep None.

~~~python
effective_temperature: float | None = None
temperature_override_reason: str | None = None

usage = UsageRecord(
    agent_id=agent_id,
    task_type=task_type,
    provider=result.provider,
    model=result.model,
    effective_temperature=result.effective_temperature,
    temperature_override_reason=result.temperature_override_reason,
)
~~~

- [ ] **Step 2: Test the actual models.yaml routes**

Load config/models.yaml with register_env_providers=False. For p03 assert discussion_summary=0.2, vote/night_action=0.3, speech=0.4, deception/wolf_discussion=0.5 and top_p=0.9. Also check p02 reflection and judge_vote_tally profiles. Existing fake HTTP tests continue to prove the final payload and MiniMax override:

~~~python
assert result.effective_temperature == 1.0
assert result.temperature_override_reason == "thinking_requires_temperature_1"
~~~

- [ ] **Step 3: Extend the real-game report**

Add effective_temperature and temperature_override_reason to the sanitized usage projection in run_real_game_reports.py. Add a script test proving the report retains the reason but contains no prompt or raw response.

~~~python
row["effective_temperature"] = getattr(usage, "effective_temperature", None)
row["temperature_override_reason"] = getattr(
    usage,
    "temperature_override_reason",
    None,
)
~~~

- [ ] **Step 4: Verify and commit**

~~~bash
conda run -n wofkill python -m pytest -q -o addopts='' \
  tests/model_gateway/test_router.py \
  tests/model_gateway/test_per_profile_url_and_extra_body.py \
  tests/model_gateway/test_execution_records.py \
  tests/scripts/test_run_real_game.py
git add werewolf_agent/model_gateway/usage_records.py \
  werewolf_agent/model_gateway/router_errors.py scripts/run_real_game_reports.py \
  tests/model_gateway/test_per_profile_url_and_extra_body.py \
  tests/model_gateway/test_execution_records.py tests/scripts/test_run_real_game.py
git commit -m "feat: audit effective model configuration"
~~~

### Task 8: Separate reflection generation and persistence audit

**Files:**
- Modify: werewolf_agent/runtime/nodes/summary.py
- Modify: werewolf_agent/runtime/game_runner_memory.py
- Test: tests/runtime/test_reflection_transaction.py
- Test: tests/runtime/test_game_runner.py
- Test: tests/runtime/test_reflection_security_contract.py

- [ ] **Step 1: Fix the pre-persistence wording**

Replace the summary-node log with:

~~~python
logger.debug(
    "  [复盘] 处理%d位：成功%d，未生成%d",
    len(reflection_entries),
    transaction_result.valid_entry_count,
    transaction_result.failure_count,
)
~~~

- [ ] **Step 2: Add post-readback counts**

Extend reflection_persistence_audit with persisted_entry_count, repository_read_complete and snapshot_read_complete. Log persistence success only after _append_reflection_persistence_audit has reread repository and snapshot. Failure or rollback must never log success.

~~~python
payload = {
    "status": status,
    "expected_entry_count": len(expected_entries),
    "persisted_entry_count": sum(
        entry["persistence_complete"] is True for entry in entries
    ),
    "repository_read_complete": repository_read_complete,
    "snapshot_read_complete": snapshot_read_complete,
    "persistence_complete": complete,
    "rollback_complete": rollback_complete,
    "entries": entries,
}
~~~

- [ ] **Step 3: Add readback and cross-game tests**

Cover success, repository read failure, snapshot mismatch, rollback, idempotent double-save, and a second GameRunner retrieving the previous game’s Reflection V2 through existing cross-game memory hints.

~~~python
runner1._save_memory_snapshot()
audit = next(
    event for event in reversed(runner1.state.events)
    if event.type == "reflection_persistence_audit"
)
assert audit.payload["persistence_complete"] is True
assert audit.payload["persisted_entry_count"] == audit.payload["expected_entry_count"]

runner2._restore_memory_snapshot()
hints = build_cross_game_memory_hints(
    runner2._restored_memory,
    player_id="p01",
    current_role="seer",
)
assert hints.reflection_memory_hints
~~~

- [ ] **Step 4: Verify and commit**

~~~bash
conda run -n wofkill python -m pytest -q -o addopts='' \
  tests/runtime/test_reflection_transaction.py \
  tests/runtime/test_game_runner.py \
  tests/runtime/test_reflection_security_contract.py
git add werewolf_agent/runtime/nodes/summary.py \
  werewolf_agent/runtime/game_runner_memory.py \
  tests/runtime/test_reflection_transaction.py tests/runtime/test_game_runner.py \
  tests/runtime/test_reflection_security_contract.py
git commit -m "fix: separate reflection generation and persistence audits"
~~~

### Task 9: Complete integration and fixed-seed verification

**Files:**
- Verify: files and tests modified in Tasks 2–8; no new file change is expected.

- [ ] **Step 1: Run the focused cross-layer suite**

~~~bash
conda run -n wofkill python -m pytest -q -o addopts='' \
  tests/cognition/test_cognition.py \
  tests/runtime/test_public_ledger.py \
  tests/runtime/test_visible_state.py \
  tests/runtime/test_context.py \
  tests/agents/test_prompt_builder.py \
  tests/agents/test_player_retry.py \
  tests/agents/test_semantic_repair_invariants.py \
  tests/agents/test_discussion_summary.py \
  tests/model_gateway/test_router.py \
  tests/model_gateway/test_per_profile_url_and_extra_body.py \
  tests/runtime/test_reflection_transaction.py
~~~

- [ ] **Step 2: Run the full suite**

~~~bash
conda run -n wofkill python -m pytest -q -o addopts=''
~~~

Record exact pass/fail counts. Do not claim completion with unexplained failures.

- [ ] **Step 3: Run a distinct fixed-seed game**

~~~bash
WEREWOLF_GAME_LOG_PATH=artifacts/real-game/g_42_repair/game_stdout.log \
conda run -n wofkill python scripts/run_real_game.py \
  --seed 42 \
  --game-id g_42_repair \
  --max-steps 500 \
  --delay -1 \
  --output-dir artifacts/real-game/g_42_repair
~~~

The process must finish and create game_g_42_repair.json. Timeout, interruption or stdout-only output is not completion evidence.

- [ ] **Step 4: Inspect every phase and audit chain**

~~~bash
rg -n "N1|D1|N2|D2|N3|D3|N4|badge_flow|action_claim|claim_conflict|invalid_json|missing_tool_call|reflection_persistence_audit|effective_temperature" \
  artifacts/real-game/g_42_repair/game_stdout.log
conda run -n wofkill python scripts/print_game_audit.py \
  artifacts/real-game/g_42_repair/game_g_42_repair.json
~~~

Required evidence: p03’s original badge targets are p02 then p12; action claims are not engine actions; summary failures have stage/shape codes; private witch and wolf facts are absent from player projections; reflection persistence is reported only after readback; active route effective temperatures are present.

- [ ] **Step 5: Final repository check**

~~~bash
git status --short --branch
git log -12 --oneline --decorate
~~~

Do not remove or revert the user’s existing deleted audit files, .idea, .DS_Store or unrelated untracked plan.

## Completion checklist

- [ ] g_42 badge-flow text parses to p02, p12 in order.
- [ ] Structured facts distinguish engine and player_claim provenance.
- [ ] Player prompts label claims separately from executed actions.
- [ ] Moderator action audit never leaks private events.
- [ ] Semantic repair rejects completed actions without engine evidence.
- [ ] Summary failures distinguish protocol, schema, empty and provider stages without raw output.
- [ ] Fallback preserves structured badge/check facts.
- [ ] Actual models.yaml routes and final provider payloads match expected parameters.
- [ ] Provider overrides are present in usage/report audits.
- [ ] Reflection success requires repository and snapshot readback.
- [ ] Cross-game Reflection V2 retrieval succeeds.
- [ ] Focused tests, full pytest and a formal fixed-seed artifact complete.
