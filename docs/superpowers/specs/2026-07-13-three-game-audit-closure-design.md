# 三局复审问题全量闭环设计

## 1. 背景与目标

2026-07-12 的三局真实对局暴露出九类残余问题：终局检查晚于遗言模型调用、狼队计划 schema fallback 偏高、有效发言被语义校验误杀、推理模式缺少运行时证据、神职友伤风险高、可能世界重复且无证据、错误复盘污染跨局记忆、Persona 注入无法证明，以及 fallback 分类失真。

本设计目标是按 P0→P1→P2→P3 全量闭环这些问题，使系统满足：

1. 胜负条件一旦成立，立即原子落盘并停止所有游戏内 Agent 调用。
2. 每次 LLM 决策都能证明实际模型与推理模式状态，不能仅依赖静态配置。
3. Fallback 按真实失败层级分类，关键策略不会因可修复格式问题整体降级。
4. 玩家只接收合法、去重、可追溯的上下文和跨局记忆。
5. 三局复审中确认的每个问题均有自动化回归测试和批量验收指标。

## 2. 非目标与规则边界

- 本轮不重写整个 LangGraph 状态机。
- 不调整角色数量、警长票权或基础胜负规则；先消除运行时和决策质量偏差，再进行更大样本的平衡评估。
- 不让 LLM 计算胜负、存活人数、合法目标、技能次数或票数。
- 不记录或暴露模型隐藏思维链；仅记录供应商返回的推理状态、推理 token 数和可公开的简短依据。
- 不以提高好人胜率为目的强制神职选择；只增加证据比较和友伤风险提示。

## 3. 方案选择

采用“分层闭环修复”：保留现有模块边界，在状态机、模型审计、输出质量、认知上下文和跨局记忆五个责任域分别建立窄接口。

未选择纯局部修补，因为终局、fallback 和推理证据跨越多个模块，继续增加条件分支会留下不可观测路径。未选择状态机重写，因为改动范围和回归风险超过本轮目标。

## 4. P0：终局原子化

### 4.1 目标流程

```text
resolve_exile / resolve_night / resolve_hunter_shot / resolve_poison
→ deterministic victory gate
  → 有胜者：写入 victory_checked + victory，设置 winning_faction
             → post-game reflection → finish_game
  → 无胜者：继续遗言、警徽或下一阶段
```

放逐死亡写入后，胜负检查必须先于 `exile_last_words`。若终局成立，被放逐玩家不再调用游戏内遗言 Agent。复盘属于 post-game 工作流，只能在 `winning_faction` 已落盘后运行，且不得修改游戏结果。

### 4.2 防御性门禁

除图路由外，所有玩家 Agent 调度入口增加只读终局守卫：`winning_faction` 非空时拒绝游戏内任务。该守卫用于防止未来新增边遗漏，不替代正确路由。

### 4.3 验收

- 放逐后形成 3 狼对 1 好人：同一步产生胜负，不出现遗言 action trace。
- 最后一狼被放逐：同一步产生好人胜负，不出现后续模型调用。
- 猎人死亡但仍需合法开枪时，先按规则解析触发技能，再在最终死亡批次后检查胜负。
- `max_steps` 恰好落在终局附近时，胜负仍已落盘。

## 5. P1：推理模式运行时证据

### 5.1 推理模式硬要求

所有玩家 LLM profile 必须显式配置 `reasoning.level`，关键任务使用以下最低等级：

| 任务 | 最低等级 |
|---|---|
| 普通发言、遗言 | medium |
| 放逐投票、警长投票 | medium |
| 预言家查验、女巫用药、猎人开枪 | high |
| 狼队夜间计划、悍跳、警徽流 | high |
| post-game reflection | high |
| 法官模板广播、规则计算 | 不调用 LLM 或 none |

配置加载时，若玩家决策 profile 的推理等级缺失或为 `none`，启动应失败并指出 profile，而不是静默运行。

### 5.2 请求与响应证据链

`UsageRecord` 作为模型执行事实源，记录：

```text
request_id
primary_provider / primary_model
fallback_provider / fallback_model
retry_count / failure_category
reasoning_requested / reasoning_level
reasoning_status / reasoning_tokens
```

每个 `action_trace_audit` 通过 request ID 关联一个脱敏 `model_execution` 摘要。游戏 JSON 不保存 prompt、隐藏推理或 API 密钥。

状态分类固定为：

- `confirmed`：请求了推理且供应商响应含推理字段或推理 token。
- `requested_unconfirmed`：请求已发送，但供应商无可验证响应字段。
- `not_requested`：该任务明确无需推理。
- `unsupported`：供应商或模型拒绝推理参数。
- `fallback_disabled`：fallback 模型不支持推理。

Fallback 后必须重新记录备用模型推理状态。关键玩家任务若从推理模型降级到非推理模型，不得标记成功无异常；必须产生可审计的 `fallback_disabled`。

### 5.3 导出与指标

`scripts/run_real_game.py` 额外导出聚合指标：

- `reasoning_requested_count`
- `reasoning_confirmed_count`
- `reasoning_unconfirmed_count`
- `reasoning_fallback_disabled_count`
- `reasoning_confirmation_rate`

真实多局验收要求关键任务 `reasoning_requested_count > 0`，且不能再出现“日志无模型/推理字段”的情况。若供应商不返回推理 token，可以保留 `requested_unconfirmed`，但必须如实报告，不能宣称已确认生效。

## 6. P1：狼队计划 schema 与 fallback

Prompt、tool schema 和 `WolfTeamPlan` 统一由同一个契约生成或共享字段定义，禁止维护三份手写字段列表。

解析前只允许执行确定性规范化：

- 解包唯一的 `night_plan` 或 `plan` 包装层。
- 对 `reasoning` 做 Unicode 安全截断至 schema 上限。
- 缺少 `public_story` 时，若已有结构化角色分工和目标证据，生成明确标记为 `normalized_default` 的中性 public story；不得伪造目标证据。
- 非法目标、重复角色、缺失核心目标证据仍必须失败。

目标是将 schema fallback 与策略 fallback 分离。多局验收中 `wolf_team_plan_fallback_rate < 10%`，且弱计划击杀和无证据 fallback 击杀继续为 0。

## 7. P1：发言语义校验与神职决策

### 7.1 公开事实校验

公开事实分类器区分：

- 系统确认事实：死亡、票型、公开技能结果。
- 玩家公开声明：某玩家说过的查验、身份、站边。
- 当前玩家推测：必须使用“我认为、我怀疑、我推测”等措辞。

引用可定位的玩家公开发言不应因为不是系统事实而失败。校验错误必须返回具体 claim 和证据匹配结果。重复的 semantic failure 不得笼统写成“结构化输出失败”。

### 7.2 Fallback 行为

解析成功但语义质量失败时，优先做最小公开文本修复；只有修复后仍违法才生成 fallback。fallback 文案必须保留原行动目标和已验证的有效论点，禁止退化为“信息不足，需要更多信息”的统一模板。

### 7.3 神职友伤风险

猎人、女巫决策 Prompt 增加：最终目标证据、至少一个替代目标比较、友伤风险和保留技能选项。规则引擎只校验合法性，不替 Agent 选择目标。

## 8. P2：可能世界与验人依据

- 使用规范化 assignment 哈希去重完全相同的世界。
- 无公开证据时不生成具体神职身份猜测；输出“证据不足”的均匀阵营假设。
- `why` 为空的世界不能进入 prompt 中的推荐依据。
- 预言家首夜验人理由不得引用无证据的具体角色猜测。
- 模拟器只能引用实际保留的 world ID。

## 9. P2：复盘记忆事实门禁

post-game reflection 分成原始模型输出和可持久化学习卡：

1. 模型生成复盘草稿。
2. 确定性校验玩家身份、投票、死亡、技能、阵营和胜负陈述。
3. 无法验证或与日志冲突的句子不进入跨局记忆。
4. 持久化内容优先保存抽象策略和经过验证的事件引用，避免把上一局玩家 ID 当作当前局事实。

失败时记录 `reflection_rejected_fact_count`，并允许保存不依赖具体事实的安全学习点。

## 10. P2：Persona 注入证据

Persona 仍在 system prompt 层生效，但审计必须记录最终组装后的脱敏证据：profile ID、system message 序号、内容哈希、字符数和 `injected=true`。不记录完整 persona 文本。

测试覆盖初次请求、重试、provider fallback 和不同任务，确保 Persona 不在重试时丢失。

## 11. P3：Fallback 分类与质量指标

固定失败类别：

```text
provider_timeout
provider_error
provider_fallback
structured_protocol
schema_validation
semantic_quality
rules_illegal_action
deterministic_normalization
```

`fallback_reason`、`failure_category`、`structured_failure_stage` 和质量报告必须一致。`total_retry_count_until_success` 在最终 fallback 时不得错误保留为 0。

批量报告增加：

- 终局后游戏内模型调用数
- 狼队 schema/策略 fallback 分项
- 语义校验拒绝及修复成功数
- Persona 注入确认率
- 复盘事实拒绝数
- 推理确认率及 fallback 保持率

## 12. 文件责任边界

预计修改域：

- `werewolf_agent/runtime/graph.py`、`graph_registration.py`：终局路由。
- `werewolf_agent/runtime/nodes/day_deaths.py`、`day_finish.py`：死亡后终局写入和遗言门禁。
- `werewolf_agent/runtime/agent_dispatch.py` 或现有统一调度入口：终局防御守卫。
- `werewolf_agent/model_gateway/*`：推理状态、fallback 模型执行事实。
- `werewolf_agent/runtime/nodes/action_audit.py`、`scripts/run_real_game.py`：模型执行摘要和聚合导出。
- `werewolf_agent/agents/wolf_team_plan*`：共享契约与规范化。
- `werewolf_agent/agents/*quality*`、`balance_public_claims.py`：公开事实与语义修复。
- `werewolf_agent/cognition/worlds.py`：世界去重和证据门禁。
- `werewolf_agent/reflection/*`、跨局记忆模块：事实校验与持久化门禁。
- Prompt/persona 审计模块：最终 system prompt 注入证据。

不进行与这些责任无关的文件拆分或格式化。

## 13. 测试与提交策略

严格执行 TDD：每个问题先添加能复现三局症状的失败测试，确认预期失败后再实现最小修复。

提交按以下顺序独立完成：

1. P0 终局原子化。
2. P1 推理执行证据。
3. P1 狼队计划规范化。
4. P1 发言质量和神职风险提示。
5. P2 可能世界与复盘事实门禁。
6. P2 Persona 注入证明。
7. P3 分类、指标和文档。

每批先跑目标测试，再跑相关模块测试。最终运行 Agent、runtime、model gateway、evaluation、integration、compileall、diff check 和全量 pytest。任何超时只报告为未完成，不视为通过。

## 14. 多局验收标准

至少运行 5 局，推荐 10 局。硬门槛：

- 完局率 100%。
- 满足胜负后游戏内模型调用数为 0。
- `weak_wolf_plan_kill_count = 0`。
- `fallback_plan_kill_without_target_evidence_count = 0`。
- 狼队计划 fallback 率低于 10%。
- Persona 注入审计确认率 100%。
- 关键玩家任务推理请求覆盖率 100%，每次都有明确 `reasoning_status`。
- `fallback_disabled = 0`；若供应商只能给 `requested_unconfirmed`，报告必须明确，不能冒充 confirmed。
- 不支持的公开事实声明为 0。
- 复盘中与结构化日志冲突的内容不得进入下一局记忆。

阵营胜率不作为 5–10 局硬通过条件，但若狼人或好人达到 80% 以上，必须继续扩大样本并做行为归因后才能调整规则。
