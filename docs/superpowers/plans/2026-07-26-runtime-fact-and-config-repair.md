# Runtime Fact And Configuration Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留已有摘要契约、fallback 推理门禁和 prompt 压缩修复的前提下，修复仍可复现的事实归因、多行警徽流、摘要 JSON 兼容性和配置可观测性问题。

**Architecture:** 事实解析和摘要解析分别在现有模块内做窄修复；模型网关不重写路由，而是补齐“声明值、任务解析值、最终 wire 值”的观测和校验。温度调优通过独立 model profile 路由实现，不在 task route 中加入会被忽略的字段。

**Tech Stack:** Python 3.12、Pydantic、PyYAML、pytest、httpx fake client；所有 Python 命令通过 `conda run -n wofkill` 执行。

---

## Recheck Result

- **已修复并有测试：** fallback 在真正调用 provider 前继承 primary 的任务级 `reasoning_level`。
- **已修复并有测试：** 简短第三方格式 `p02报p01金水` 不再变成转述者自己的预言家声明。
- **已修复并有测试：** DiscussionSummary V2 已使用严格 Pydantic Schema、独立 task contract 和确定性 fallback。
- **当前无需修改：** `_MAX_TRANSCRIPT_TEXT_CHARS=220` 能在 p03 长发言样本中保留 `p02/p11` 警徽流；先锁测试，不扩大上下文。
- **仍可复现：** 长前缀转述 `p01陈思远，你跳预言家说验了p02金水` 仍生成 `p06 -> p02 good` 的 `seer_check_claim`。
- **仍可复现：** 多行 `警徽流：\n- N2...p02\n- N3...p11` 不生成 `badge_flow_claim`。
- **仍可复现：** fenced JSON 摘要仍抛出 `DiscussionSummaryGenerationError("invalid_json")`。
- **部分生效：** MiniMax thinking 的 YAML temperature 会被 provider 强制为 `1`，但 attempt 审计看不出 override 原因。
- **未覆盖：** route provider 与 `model_profile.provider` 不一致时，配置校验不会拒绝。

## Files

- Modify: `werewolf_agent/cognition/world_state.py`
- Modify: `werewolf_agent/agents/discussion_summary.py`
- Modify: `werewolf_agent/agents/json_repair.py`
- Modify: `werewolf_agent/agents/player.py`
- Modify: `werewolf_agent/model_gateway/router_config.py`
- Modify: `werewolf_agent/model_gateway/usage_records.py`
- Modify: `werewolf_agent/model_gateway/providers/minimax.py`
- Modify: `werewolf_agent/model_gateway/providers/openai.py`
- Modify: `config/models.yaml`
- Modify: `tests/cognition/test_cognition.py`
- Modify: `tests/agents/test_discussion_summary.py`
- Modify: `tests/agents/test_json_repair.py`
- Modify: `tests/agents/test_prompt_builder.py`
- Modify: `tests/model_gateway/test_router.py`
- Modify: `tests/model_gateway/test_per_profile_url_and_extra_body.py`
- Modify: `tests/model_gateway/test_providers.py`

## Task 1: Add Missing Characterization Tests

**Files:**
- Modify: `tests/cognition/test_cognition.py`
- Modify: `tests/agents/test_discussion_summary.py`
- Modify: `tests/agents/test_prompt_builder.py`

- [ ] **Step 1: Add the exact long-prefix attribution regression**

```python
def test_long_third_party_seer_recap_is_not_attributed_to_speaker() -> None:
    from werewolf_agent.cognition.world_state import _infer_claims_from_text

    claims = _infer_claims_from_text(
        speaker="p06",
        text=(
            "p01陈思远，你跳预言家说验了p02金水，但你警徽流里含了我p06。"
            "我想问，你首夜验人的逻辑是什么？"
        ),
        day=1,
    )
    assert not [
        claim for claim in claims
        if claim.fact_type == "seer_check_claim"
        and claim.source_player == "p06"
        and claim.target_player == "p02"
    ]
```

- [ ] **Step 2: Add multiline badge-flow regression**

```python
def test_multiline_badge_flow_is_extracted_in_order() -> None:
    from werewolf_agent.cognition.world_state import _infer_claims_from_text

    claims = _infer_claims_from_text(
        speaker="p03",
        text="警徽流：\n- N2我计划验p02赵猛\n- N3我计划验p11郑铭",
        day=1,
    )
    badge = [claim for claim in claims if claim.fact_type == "badge_flow_claim"]
    assert len(badge) == 1
    assert badge[0].target_player == "p02"
    assert badge[0].metadata["badge_flow_order"] == ["p02", "p11"]
```

- [ ] **Step 3: Lock the already-working prompt compaction behavior**

Build an `AgentContext` with the exact p03 speech from the log and assert:

```python
rendered = PlayerPromptBuilder(context, "p08")._build_recent_transcript()
assert "p02" in rendered
assert "p11" in rendered
```

This test must pass before implementation. If it passes, do not modify `prompt_user_context.py` in this repair.

- [ ] **Step 4: Add fenced JSON characterization**

Use the existing capturing-provider pattern in `tests/agents/test_discussion_summary.py`, returning:

```python
GenerateResult(
    text='```json\n{"summary":"怀疑p03"}\n```',
    provider=self.name,
    model=config.model,
)
```

The post-fix assertion is:

```python
assert agent.summarize_discussion(context).summary == "怀疑p03"
```

- [ ] **Step 5: Run characterization tests**

```bash
conda run -n wofkill python -m pytest -q -o addopts='' \
  tests/cognition/test_cognition.py \
  tests/agents/test_discussion_summary.py \
  tests/agents/test_prompt_builder.py
```

Expected before implementation: the new attribution, badge, and fenced JSON tests fail; prompt compaction passes.

## Task 2: Fix Remaining Public-Fact Parser Gaps

**Files:**
- Modify: `werewolf_agent/cognition/world_state.py`
- Modify: `tests/cognition/test_cognition.py`

- [ ] **Step 1: Make third-party detection clause-local**

Replace the fixed 16-character prefix in `_is_third_party_seer_report` with bounded clause analysis. A claim governed by `pNN...说/报/称/表示` or `你...说/报` is third-party. Keep direct first-person claims working:

```python
_infer_claims_from_text(
    speaker="p03", text="我是预言家，昨晚验p01查杀", day=1
)
_infer_claims_from_text(
    speaker="p03", text="我验了p02是好人", day=1
)
```

- [ ] **Step 2: Extend badge parsing without unbounded DOTALL**

After `警徽流`, inspect a bounded block ending at the next blank-line section or a maximum character limit. Extract planned player IDs in appearance order. Preserve the existing representation: one `badge_flow_claim`, first target in `target_player`, full order in `metadata["badge_flow_order"]`.

- [ ] **Step 3: Verify parser behavior**

```bash
conda run -n wofkill python -m pytest -q -o addopts='' \
  tests/cognition/test_cognition.py \
  tests/runtime/test_context.py \
  tests/runtime/test_visible_state.py \
  tests/runtime/test_public_ledger.py
```

## Task 3: Reuse JSON Repair For Discussion Summary

**Files:**
- Modify: `werewolf_agent/agents/json_repair.py`
- Modify: `werewolf_agent/agents/discussion_summary.py`
- Modify: `werewolf_agent/agents/player.py`
- Modify: `tests/agents/test_json_repair.py`
- Modify: `tests/agents/test_discussion_summary.py`

- [ ] **Step 1: Extract a generic balanced-object scanner**

Move the brace scan currently inside `extract_json_object_candidates` into:

```python
def extract_balanced_json_objects(text: str) -> list[str]:
    """按出现顺序提取字符串外的平衡 JSON 对象，不判断业务字段。"""
```

Keep `extract_json_object_candidates` as the existing action-specific filter over this function.

- [ ] **Step 2: Add one strict summary parser**

Add `parse_discussion_summary_text(raw_text)` to `discussion_summary.py`. It must run `repair_json_text`, decode JSON, and validate with `DiscussionSummary.model_validate`. If multiple objects exist, accept only when exactly one validates as DiscussionSummary; otherwise reject.

- [ ] **Step 3: Route `PlayerAgent.summarize_discussion` through it**

Map extraction/JSON decoding failures to `invalid_json` and Pydantic failures to `schema_validation_failed`. Do not discard unknown fields, invent `summary`, or coerce strict containers.

- [ ] **Step 4: Verify JSON behavior**

```bash
conda run -n wofkill python -m pytest -q -o addopts='' \
  tests/agents/test_json_repair.py \
  tests/agents/test_output_parser.py \
  tests/agents/test_discussion_summary.py \
  tests/runtime/test_summary_visibility.py
```

Expected: fences/BOM/mixed prose pass; unknown fields, missing summary, wrong containers, and ambiguous multiple objects fail closed.

## Task 4: Make Effective Model Parameters Auditable

**Files:**
- Modify: `werewolf_agent/model_gateway/usage_records.py`
- Modify: `werewolf_agent/model_gateway/providers/minimax.py`
- Modify: `werewolf_agent/model_gateway/providers/openai.py`
- Modify: `tests/model_gateway/test_per_profile_url_and_extra_body.py`
- Modify: `tests/model_gateway/test_providers.py`

- [ ] **Step 1: Add provider-effective sampling metadata**

Extend `GenerateResult` with defaults:

```python
effective_temperature: float | None = None
temperature_override_reason: str | None = None
```

OpenAI-compatible routes set configured temperature as effective. MiniMax thinking sets `1.0` and reason `thinking_requires_temperature_1`; non-thinking routes record no override.

- [ ] **Step 2: Normalize native MiniMax reasoning output**

Normalize `message.reasoning_content`, `message.reasoning`, `message.reasoning_details`, and embedded `<think>` into `GenerateResult.thinking_text`. For structured `reasoning_details`, retain textual reasoning only.

- [ ] **Step 3: Add active payload matrix tests**

Cover p01 speech, p07 speech, p02 reflection, judge vote tally, and one actual fallback. Assert final temperature, top_p, reasoning level, timeout, extra_body and structured-output keys. Retain the existing assertion that speech fallback requests `medium`.

- [ ] **Step 4: Run gateway tests**

```bash
conda run -n wofkill python -m pytest -q -o addopts='' \
  tests/model_gateway/test_router.py \
  tests/model_gateway/test_reasoning_policy.py \
  tests/model_gateway/test_openai.py \
  tests/model_gateway/test_providers.py \
  tests/model_gateway/test_per_profile_url_and_extra_body.py \
  tests/model_gateway/test_minimax_provider_routing.py
```

## Task 5: Tighten Configuration And Temperature Profiles

**Files:**
- Modify: `werewolf_agent/model_gateway/router_config.py`
- Modify: `config/models.yaml`
- Modify: `tests/model_gateway/test_router.py`
- Modify: `tests/agents/test_model_router.py`

- [ ] **Step 1: Reject route/profile provider mismatches**

In `_validate_route_block`, require normalized `route.provider` to equal the referenced `model_profile.provider`. An intentional protocol adapter must use a dedicated matching model profile.

- [ ] **Step 2: Add mismatch and current-config tests**

An in-memory `route.provider=minimax` plus `model_profile.provider=openai` must raise `ProviderConfigError`; current `config/models.yaml` must still load.

- [ ] **Step 3: Split profiles by task**

Use model profiles, not ignored task-route fields:

```yaml
# summary / judge
temperature: 0.2
top_p: 0.9

# vote / night action
temperature: 0.3
top_p: 0.9

# normal speech
temperature: 0.4
top_p: 0.9

# deception / wolf discussion
temperature: 0.5
top_p: 0.9
```

- [ ] **Step 4: Align MiniMax declarations with provider policy**

Use a dedicated judge profile at `0.2` and a dedicated thinking/fallback profile declaring `1.0`. Keep reflection at wire-compatible `1.0` unless it moves away from thinking.

- [ ] **Step 5: Keep text_json until capability is proven**

Do not switch to `native_tool` or `json_schema` from provider name alone. Require a passing `probe_tool_call_support` or equivalent payload/response test for that exact route.

- [ ] **Step 6: Verify config resolution**

```bash
conda run -n wofkill python -m pytest -q -o addopts='' \
  tests/agents/test_model_router.py \
  tests/model_gateway/test_router.py \
  tests/model_gateway/test_structured_output.py \
  tests/model_gateway/test_per_profile_url_and_extra_body.py
```

## Task 6: Full Verification And Real-Game Acceptance

- [ ] **Step 1: Run full tests**

```bash
conda run -n wofkill python -m pytest -q -o addopts=''
```

- [ ] **Step 2: Run static checks**

```bash
conda run -n wofkill python -m ruff check werewolf_agent tests
conda run -n wofkill python -m mypy werewolf_agent
```

- [ ] **Step 3: Run one fixed-seed bounded game**

Acceptance metrics:

```text
exact third-party attribution reproductions = 0 failures
multiline badge-flow reproductions = 0 failures
summary invalid_json caused by fences/BOM = 0
fallback speech requested_reasoning_level = medium
unexpected temperature overrides = 0
expected MiniMax thinking overrides = fully recorded
```

Compare parser errors, summary failures, retries, fallback counts and effective payloads. Win/loss alone is not acceptance.

- [ ] **Step 4: Review intended diff only**

```bash
git diff --check
git status --short
```

Preserve unrelated `.superpowers` deletions and user-created files.

## Rollout

1. Commit Tasks 1–2: public-fact parser only.
2. Commit Task 3: summary decoder only.
3. Commit Task 4: observability only; no tuning yet.
4. Commit Task 5: validation and profiles.
5. Run Task 6 before further temperature changes.

Suggested commits:

```text
fix: preserve third-party seer attribution
fix: parse fenced discussion summaries
feat: audit effective model temperature
fix: validate and split model task profiles
```
