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
resolve_exile / resolve_night / resolve_poison
→ pending death-trigger gate
  → 存在规则强制死亡反应（如合法猎人开枪）：完成反应及其新增死亡
  → 不存在待结算反应：deterministic victory gate
  → 有胜者：写入 victory_checked + victory，设置 winning_faction
             → post-game reflection → finish_game
  → 无胜者：继续遗言、警徽或下一阶段
```

放逐死亡写入后，必须先结算规则强制的死亡反应，再执行胜负检查；胜负检查必须先于 `exile_last_words`。猎人开枪可能改变胜者，因此待开枪状态存在时禁止提前写入 `victory`。若最终终局成立，被放逐玩家不再调用游戏内遗言 Agent。复盘属于 post-game 工作流，只能在 `winning_faction` 已落盘后运行，且不得修改游戏结果。

### 4.2 防御性门禁

除图路由外，所有玩家 Agent 调度入口增加只读终局守卫：`winning_faction` 非空时拒绝游戏内任务。该守卫用于防止未来新增边遗漏，不替代正确路由。

### 4.3 验收

- 放逐后形成 3 狼对 1 好人：同一步产生胜负，不出现遗言 action trace。
- 最后一狼被放逐：同一步产生好人胜负，不出现后续模型调用。
- 猎人死亡但仍需合法开枪时，先按规则解析触发技能，再在最终死亡批次后检查胜负。
- 猎人开枪前后胜者不同：只允许提交开枪后的最终胜者，且只提交一次 `victory`。
- `max_steps` 恰好落在终局附近时，胜负仍已落盘。

## 5. P1：推理模式运行时证据

### 5.1 推理模式硬要求

所有玩家 LLM profile 必须显式配置 `reasoning.level`。中央模型网关在每次请求建立时根据任务策略再次校验最低等级，不能只信任 profile。检查范围包含首次调用、语义修复、结构化重试、provider fallback、狼队计划子调用、玩家使用的 LLM cognition helper 和 post-game reflection。关键任务使用以下最低等级：

| 任务 | 最低等级 |
|---|---|
| 普通发言、遗言 | medium |
| 放逐投票、警长投票 | medium |
| 预言家查验、女巫用药、猎人开枪 | high |
| 狼队夜间计划、悍跳、警徽流 | high |
| post-game reflection | high |
| 法官模板广播、规则计算 | 不调用 LLM 或 none |

配置加载时，若玩家决策 profile 的推理等级缺失或低于任务最低值，启动应失败并指出 profile。运行时每次 attempt 再执行相同门禁，防止动态路由、重试或 fallback 绕过。

### 5.2 请求与响应证据链

模型执行事实源按 attempt 建模，而不是只保存一个 primary/fallback 结果。每次 attempt 记录：

```text
opaque_request_id / attempt_ordinal
provider / model / route_kind
attempt_outcome / root_cause
reasoning_requested / reasoning_level
reasoning_status / reasoning_tokens / reasoning_evidence_kind
```

每个 `action_trace_audit` 关联一个脱敏 `model_execution.attempts[]` 序列，完整保存首次、重试和备用供应商尝试；最终执行摘要不得覆盖早期失败。游戏 JSON 不保存 prompt、隐藏推理、原始 provider payload、原始错误正文或 API 密钥。

状态分类固定为：

- `confirmed`：请求了推理，且 provider 适配器取得正数推理 token，或供应商权威返回“已执行推理”的状态。仅回显请求字段、空 thinking block 或字段存在均不能确认。
- `requested_unconfirmed`：请求已发送，但供应商无可验证响应字段。
- `not_requested`：该任务明确无需推理。
- `unsupported`：供应商或模型拒绝推理参数。
- `fallback_disabled`：fallback 模型不支持推理。

Fallback 后必须重新记录备用模型推理状态。所有玩家游戏内任务均只允许路由到满足最低推理等级的备用模型；备用模型不支持或拒绝推理参数时继续尝试下一个推理模型，耗尽后 fail closed 为显式安全 fallback，不允许把非推理模型输出当作正常决策。post-game reflection 可在耗尽后跳过，不影响已落盘胜负。

`reasoning_fallback_keep_rate` 定义为：发生 provider fallback 且最终产生 LLM 决策的玩家请求中，最终 attempt 满足任务最低推理等级的请求数 / 同类请求总数。无同类请求时输出 `null` 和 `supported=false`，不得输出虚假的 0 或 1。

### 5.3 导出与指标

`scripts/run_real_game.py` 额外导出聚合指标：

- `reasoning_requested_count`
- `reasoning_confirmed_count`
- `reasoning_unconfirmed_count`
- `reasoning_fallback_disabled_count`
- `reasoning_confirmation_rate`
- `reasoning_fallback_keep_rate`

真实多局验收要求关键任务 `reasoning_requested_count > 0`，且不能再出现“日志无模型/推理字段”的情况。若供应商不返回权威执行证据，可以保留 `requested_unconfirmed`，但必须如实报告，不能宣称已确认生效。

精确定义：

- `reasoning_confirmation_rate = confirmed attempts / reasoning_requested attempts`。
- `critical_task_reasoning_request_coverage = 已请求且等级不低于任务最低值的关键任务首次 attempt 数 / 关键任务首次 attempt 总数`。
- 两项均以 attempt 为聚合单位；分母为 0 时输出 `null` 和 `supported=false`。

### 5.4 审计隐私白名单

`model_execution` 只允许导出上述 attempt 字段以及数值 token/延迟。`opaque_request_id` 使用局内随机 ID 或带密钥的运行期摘要，不能直接导出供应商 request ID。错误仅导出固定枚举和经过长度限制的无内容摘要。数据只进入 moderator-only 游戏审计和脱敏批量指标，不进入 public/player view；默认保留周期沿用游戏审计文件，不额外发送第三方。Persona 使用局内不透明指纹或带密钥摘要，禁止使用可被字典枚举的普通内容哈希。

## 6. P1：狼队计划 schema 与 fallback

Prompt、tool schema 和 `WolfTeamPlan` 统一由同一个契约生成或共享字段定义，禁止维护三份手写字段列表。

解析前只允许执行确定性规范化：

- 解包唯一的 `night_plan` 或 `plan` 包装层。
- 对 `reasoning` 做 Unicode 安全截断至 schema 上限。
- 缺少 `public_story` 时，若已有结构化角色分工和目标证据，生成明确标记为 `normalized_default` 的中性 public story；不得伪造目标证据。
- 非法目标、重复角色、缺失核心目标证据仍必须失败。

指标分别定义：

- `wolf_plan_schema_fallback_rate = schema 终态 fallback 数 / 总计划数`
- `wolf_plan_strategy_fallback_rate = 策略证据不足终态 fallback 数 / 总计划数`
- `wolf_plan_normalization_success_rate = 规范化后成功计划数 / 触发规范化计划数`
- `wolf_team_plan_terminal_fallback_rate = 所有终态 fallback 数 / 总计划数`

分母为 0 时输出 `null` 和 `supported=false`。多局验收要求终态 fallback 率低于 10%，schema fallback 率低于 5%，且规范化绝不生成、替换或推断击杀目标证据；弱计划击杀和无证据 fallback 击杀继续为 0。

## 7. P1：发言语义校验与神职决策

### 7.1 公开事实校验

公开事实分类器区分：

- 系统确认事实：死亡、票型、公开技能结果。
- 玩家公开声明：某玩家说过的查验、身份、站边。
- 当前玩家推测：必须使用“我认为、我怀疑、我推测”等措辞。

引用可定位的玩家公开发言不应因为不是系统事实而失败。校验错误必须返回具体 claim 和证据匹配结果。重复的 semantic failure 不得笼统写成“结构化输出失败”。

### 7.2 Fallback 行为

解析成功但语义质量失败时，只允许以下最小变换：把无法证实的确定语气降为明确推测、删除单个无法验证的 claim、补充缺失的 attribution。禁止改变行动目标、否定关系、说话者归属或已验证 claim，禁止引入任何新事实。混合有效/无效 claim 仅删除或降级无效部分。只有修复后仍违法才生成 fallback。fallback 文案必须保留原行动目标和全部已验证论点，禁止退化为“信息不足，需要更多信息”的统一模板。

### 7.3 神职友伤风险

猎人、女巫决策 Prompt 增加：最终目标证据、至少一个替代目标比较、友伤风险和保留技能选项。存在合法替代目标时必须比较；只有一个合法目标时允许 `alternative_target=null`，但必须记录 `no_legal_alternative`；没有合法目标时只能选择保留技能或规则允许的 no-action。测试覆盖单目标、无目标、存在替代目标和保留技能。规则引擎只校验合法性，不替 Agent 选择目标。

## 8. P2：可能世界与验人依据

- 对 player ID 排序后，将 `player_id=role` 序列以 UTF-8 JSON canonical form 序列化，再计算 assignment 哈希并去重完全相同的世界。
- 无公开证据时不生成具体神职身份猜测；输出“证据不足”的均匀阵营假设。
- `why` 必须包含可回查的公开 event ID 或 claim ID；为空或引用未知 ID 的世界 fail closed，不进入 prompt。
- 预言家首夜验人理由不得引用无证据的具体角色猜测。
- 模拟器只能引用实际保留的 world ID；未知 world ID 使该预测被丢弃并产生审计计数。

## 9. P2：复盘记忆事实门禁

post-game reflection 分成原始模型输出和结构化可持久化学习卡。模型必须输出 `claims[]`（claim_type、subject、predicate、object、event_refs）与 `lessons[]`（文本及依赖 claim ID），不直接把自由文本句子写入记忆：

1. 模型生成复盘草稿。
2. 确定性校验结构化 claim 的玩家身份、投票、死亡、技能、阵营和胜负；未知 claim type 或 event ref 一律拒绝。
3. lesson 依赖的任一 claim 被拒绝时，该 lesson 不进入跨局记忆。
4. 持久化内容优先保存抽象策略和经过验证的事件引用，玩家 ID 必须带来源 game ID，渲染到新局时默认匿名化。

失败时记录 `reflection_rejected_fact_count`，并允许保存不依赖具体事实的安全学习点。

## 10. P2：Persona 注入证据

Persona 仍在 system prompt 层生效，但审计必须记录最终组装后的脱敏证据：profile ID、system message 序号、局内随机不透明指纹（或带运行期密钥的摘要）、字符数和 `injected=true`。不记录完整 persona 文本、普通内容哈希或可被已知 profile 字典反查的摘要。测试必须证明导出中不存在原文和普通 SHA/MD5 内容哈希。

测试覆盖初次请求、重试、provider fallback 和不同任务，确保 Persona 不在重试时丢失。

## 11. P3：Fallback 分类与质量指标

根因、尝试结果和最终决策结果分开建模，三者均为互斥枚举：

```text
root_cause: provider_timeout | provider_error | structured_protocol |
            schema_validation | semantic_quality | rules_illegal_action | none
attempt_outcome: success | repair_success | retry | route_fallback | failed
decision_outcome: llm_success | normalized_success | safe_fallback | rejected
```

确定性规范化是成功路径 `normalized_success`，不是失败根因；provider fallback 是 attempt outcome，不是根因。`attempt_count` 包含首次尝试，`retry_count = attempt_count - 1`，`total_retry_count_until_success` 仅在 LLM/规范化成功时有数值，终态 fallback 时为 `null`。`fallback_reason`、`root_cause`、stage 和质量报告必须由同一事件生成器导出。表驱动测试覆盖 repair success、provider fallback、终态 fallback、非法行动和规范化成功。

批量报告增加：

- 终局后游戏内模型调用数
- 狼队 schema/策略 fallback 分项
- 语义校验拒绝及修复成功数
- Persona 注入确认率
- 复盘事实拒绝数
- 推理确认率及 fallback 保持率

语义修复指标以“首次 semantic quality 校验失败且被分类为允许修复”的行动为单位：`semantic_repair_success_rate = 保持目标且通过二次校验的行动数 / 可修复行动总数`。分母为 0 时输出 `null` 和 `supported=false`。

Persona 指标以实际发起 LLM 请求的玩家行动为单位：`persona_injection_confirmation_rate = 同一行动存在 persona exposure 且最终 system-message 注入审计 confirmed 的行动数 / 有 persona 配置的玩家 LLM 行动总数`。分母为 0 时输出 `null` 和 `supported=false`。

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

自动化回归通过后，固定运行 10 局 observational soak test。硬门槛：

- 完局率 100%。
- 满足胜负后游戏内模型调用数为 0。
- `weak_wolf_plan_kill_count = 0`。
- `fallback_plan_kill_without_target_evidence_count = 0`。
- `wolf_team_plan_terminal_fallback_rate < 10%`。
- `wolf_plan_schema_fallback_rate < 5%`。
- Persona 注入审计确认率 100%。
- 关键玩家任务推理请求覆盖率 100%，每次都有明确 `reasoning_status`。
- `fallback_disabled = 0`；若供应商只能给 `requested_unconfirmed`，报告必须明确，不能冒充 confirmed。
- 不支持的公开事实声明为 0。
- 复盘中与结构化日志冲突的内容不得进入下一局记忆。
- 语义修复保持原目标率 100%，不得引入新 claim；可修复语义问题的成功率至少 95%。
- 输入中存在已验证 claim 时，fallback 发言至少保留一个已验证论点；输入不存在有效 claim 时，使用保留行动目标、无新增事实的任务特定安全表达。两种路径都禁止使用泛化“信息不足”模板，模板率为 0。
- 进入 Prompt 的可能世界唯一率和有效 evidence ref 覆盖率均为 100%。
- 猎人和女巫每次伤害决策都记录目标、友伤风险，以及合法替代候选；无替代候选时必须记录 `alternative_target=null` 和 `no_legal_alternative`。该指标验证决策过程，不强制规定友伤率。
- retry/attempt 聚合与逐 attempt 明细逐项相等，差异数为 0。
- Persona exposure 与最终 system-message 注入审计关联一致率 100%。

阵营胜率不作为 5–10 局硬通过条件，但若狼人或好人达到 80% 以上，必须继续扩大样本并做行为归因后才能调整规则。
