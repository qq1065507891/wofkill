# Audit Repair Guardrails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复狼人杀审计计划指出的公共事实污染、警长票异常、fallback 兜底、狼队计划 schema 和可审计性问题。

**Architecture:** 将易破坏流程公平性的规则放在运行时 adapter / visibility / consensus 层做确定性校验，提示词只作为前置约束。投票质量、预言家上警、狼队 fallback 都保留允许策略空间，但非法或弱理由路径必须被拦截、修复或记录为安全 no-kill。

**Tech Stack:** Python, Pydantic, pytest, runtime node/adapters, cognition visibility policy.

---

## 1. 问题提取与优先级归类

| 编号 | 问题描述 | 问题类型 | 影响范围 | 严重程度 | 修复优先级 | 是否立即修复 |
|---|---|---|---|---|---|---|
| P0-1 | 私有/主持人/审计事件可能进入公开预言家线或公共推理 | 状态同步问题 | belief、seer credibility、contradiction | 致命 | P0 | 是 |
| P0-2 | 狼队 fallback 主备目标均非法时可能被下游当普通 fallback 继续推进 | fallback 兜底策略不合理 | 夜晚击杀、公平性 | 严重 | P0 | 是 |
| P0-3 | 狼队计划缺 deterministic 字段 `night_number` 会触发 schema fallback | 输出格式约束不足 | wolf_team_plan_node | 严重 | P0 | 是 |
| P1-1 | 真预言家不上警允许但缺高玩低概率门槛 | 规则缺失 | sheriff_registration | 中等 | P1 | 是 |
| P1-2 | 警长票投给非预言家候选允许但缺强理由校验 | 规则缺失 | sheriff_vote | 严重 | P1 | 是 |
| P1-3 | 投票理由与公开事实、候选身份关系缺专门审计 | 模型推理质量问题 | sheriff_vote / vote_quality | 中等 | P1 | 是 |
| P2-1 | fallback 原因、私有可见性来源字段不足 | 日志字段缺失 | audit / replay | 轻微 | P2 | 部分修复 |
| P2-2 | skill/tool 调用可审计性仍依赖现有 exposure audit | 日志字段缺失 | skill/tool | 轻微 | P2 | 是 |
| P2-3 | 各模块 prompt/context 注入缺少统一监控账本 | 日志字段缺失 | context / prompt injection / evaluation | 轻微 | P2 | 是 |

## 2. 根因分析

| 编号 | 表层现象 | 根本原因 | 证据来源 | 是否共性问题 | 备注 |
|---|---|---|---|---|---|
| P0-1 | 非公开 sheriff_speech 被当成 claimed_role / seer_check_claim | StructuredFact 未携带来源 visibility，context 矛盾检测直接使用完整 world_state | 审计计划 + 新增回归测试 | 是 | 与“私有事实伪装公开事实”同根 |
| P0-2 | fallback 后无合法目标仍可能进入旧共识路径 | 缺少 fallback 计划非法目标的最终安全出口 | 审计计划 + wolf fallback 测试 | 否 | 建议阻断/重试优先，运行时最终 no-kill |
| P0-3 | LLM 漏填 `night_number` 造成不必要 fallback | 可由 GameState 决定的字段仍强依赖模型输出 | 审计计划 + schema 测试 | 是 | deterministic 字段应自动补齐 |
| P1-1 | 真预言家普通情况下不上警 | 提示词没有把“不上警”限定为高玩极低概率战术，adapter 无兜底 | 用户口径 + sheriff 测试 | 否 | 高玩例外保留 |
| P1-2 | 警下玩家把警长票投给非预言家候选但理由弱 | 没有独立的 sheriff vote strong-reason validator | 用户口径 + vote_quality 测试 | 是 | 与普通放逐投票质量不同 |
| P2-2 | skill/tool 调用缺少逐调用成功/失败监控 | 旧日志只记录 prompt-visible skill exposure 和 action trace 中的 tool-call 元数据，缺少统一调用明细事件 | 用户补充口径 + skill/tool audit 测试 | 是 | 采用监控事件，不阻断流程 |
| P2-3 | 各模块是否实际注入 prompt/context 难以审计 | 旧日志分散记录 RAG/reflection/skill/persona，缺少对 final `AgentContext` 各字段的统一摘要账本 | 用户补充口径 + prompt injection audit 测试 | 是 | 仅记录摘要、数量和 hash，不记录原文 |

## 3. 修复动作

| 编号 | 修改模块 | 具体动作 | 验证 |
|---|---|---|---|
| P0-1 | `cognition/world_state.py`, `cognition/visibility.py`, `runtime/context.py` | StructuredFact 附加 `source_event`/`visibility`；VisibilityPolicy 尊重事件 visibility；contradiction 使用 visible_facts | `test_build_agent_context_excludes_moderator_only_sheriff_claims` |
| P0-2 | `runtime/nodes/wolf_consensus.py` | fallback 狼队计划主备目标均非法时记录 `wolf_no_kill_timeout(reason=wolf_plan_invalid_no_kill)` | `test_fallback_plan_with_two_illegal_targets_records_no_kill` |
| P0-3 | `runtime/agent_wolf_actions.py` | JSON 解析后、Pydantic 校验前用 `gs.night_number` 补齐缺失字段 | `test_missing_night_number_is_autofilled_from_game_state` |
| P1-1 | `runtime/agent_sheriff_actions.py` | seer sheriff registration prompt 增加高玩极低概率说明；非高玩 NO_ACTION 确定性修复为报名 | `test_seer_sheriff_registration_no_action_requires_expert_tactic` |
| P1-2 | `runtime/vote_quality.py`, `runtime/agent_sheriff_actions.py` | 新增 `validate_sheriff_vote_choice`；弱理由非预言家票改投可见 seer claimant；非法警长票目标置空；返回 `sheriff_vote_validation` 结构化记录 | `TestSheriffVoteChoice`, `test_agent_sheriff_vote_injects_non_seer_candidate_strong_reason_rule`, `test_agent_sheriff_vote_clears_invalid_candidate_target` |
| P2-1 | `runtime/strategy/seer.py` | `public_seer_claimants` 忽略非 public 发言 | context / sheriff vote 回归 |
| P2-2 | `skills/registry.py`, `runtime/exposure_audit.py`, `runtime/context.py`, `runtime/nodes/action_audit.py`, `evaluation/trace_builder.py` | 新增 `skill_tool_call_audit` 监控事件；逐 skill 记录调用名称、输入摘要、成功/失败、错误、输出摘要和是否进入 prompt；从 action trace 记录模型 tool-call 是否 required/received、失败原因、fallback 关系和决策使用状态；评估 trace 聚合该事件的白名单摘要，便于后续质量分析 | `test_dispatch_for_role_records_each_skill_success_and_failure`, `test_collector_records_detailed_skill_tool_call_rows`, `test_action_audit_emits_model_tool_call_monitor_event`, `test_build_agent_context_records_skill_tool_call_audit`, `test_skill_tool_call_audit_joins_trace_with_safe_call_details` |
| P2-3 | `runtime/exposure_audit.py`, `runtime/context.py`, `evaluation/trace_builder.py` | 新增 `prompt_injection_audit` 监控事件；统一记录 final `AgentContext` 中各模块字段是否注入、字段路径、注入类型、可见性范围、条数、字符规模和内容 hash；评估 trace 聚合为 `prompt_injections` exposure | `test_collector_records_prompt_injection_rows_without_raw_content`, `test_build_agent_context_records_prompt_injection_audit`, `test_prompt_injection_audit_joins_trace_with_safe_metadata` |

## 4. 验收标准

- P0 问题必须有运行时硬拦截，不只依赖 prompt。
- 真预言家不上警仍允许，但只对高玩画像开放；普通模型输出不上警会被修复。
- 警长票投非预言家候选仍允许，但必须说明所有跳预言家的候选不可信，并说明目标好人依据。
- 警长票目标必须来自当前候选；非法目标不能进入计票，并必须留下结构化校验/修复记录。
- 狼 fallback 主备目标均非法时先视为计划不可执行，运行时最终记录 no-kill，不随机刀人。
- skill/tool 调用必须有 moderator-only 监控事件：skill 记录逐个调用的成功/失败，模型 tool-call 记录 required/received、结构化失败原因、fallback 关系和决策使用状态。
- 各模块 prompt/context 注入必须有 moderator-only 监控事件：记录模块名、字段路径、是否注入、规模摘要和安全 hash；禁止记录完整 prompt 原文、私有推理链或未白名单字段。
- deterministic 字段缺失不触发不必要 fallback。
- 新增测试与相邻回归测试通过；超时或未完成测试不作为通过证据。

## 5. 风险与回滚

| 风险 | 影响 | 预防措施 | 回滚方式 |
|---|---|---|---|
| 非预言家警长票校验过严 | 降低少数合理战术空间 | 强理由模板允许“全 seer claimant 不可信 + 目标好人依据” | 回滚 `validate_sheriff_vote_choice` adapter 修复分支 |
| seer 不上警门槛过严 | 高阶隐忍战术减少 | 保留 expert profile + logic/credibility 阈值出口 | 回滚 `_seer_skip_sheriff_tactic_allowed` 强制修复 |
| visibility 元数据改变下游事实可见性 | 某些旧测试依赖完整 world_state | 仅在 visibility policy 和 context visible_facts 处生效；world_state 仍保留完整事实 | 回滚 VisibilityPolicy event visibility 映射 |
| fallback no-kill 降低狼人击杀率 | 非法 fallback 局出现空刀 | 只在 fallback 计划且目标全非法时触发 | 回滚 `_fallback_plan_has_only_illegal_targets` 分支 |

## 6. 测试计划

| 测试类型 | 命令 | 覆盖 |
|---|---|---|
| 目标单测 | `python -m pytest -n 0 --basetemp C:\tmp\wofkill-audit-repair-pytest ...` | 新增六组回归 |
| 相邻回归 | `python -m pytest -n 0 --basetemp C:\tmp\wofkill-audit-repair-pytest tests/runtime/test_vote_quality.py tests/runtime/test_sheriff_flow.py tests/runtime/test_agent_wolf_team_plan.py tests/runtime/test_wolf_flow.py tests/runtime/test_context.py tests/cognition/test_cognition.py tests/integration/test_e2e_info_leak.py` | 投票、警长、狼队、可见性 |
| 静态校验 | `python -m compileall -q werewolf_agent tests` | 语法和导入 |
| diff 校验 | `git diff --check` | 空白与补丁格式 |
