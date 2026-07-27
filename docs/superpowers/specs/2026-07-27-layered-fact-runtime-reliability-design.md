# 分层事实账本与运行时可靠性修复设计

**日期：** 2026-07-27
**状态：** 已确认
**依据：** 最新完整对局 g_42 的 game_stdout.log、当前 master 实现及已合并修复

## 1. 背景与目标

最新完整对局依次经历 N1、D1、N2、D2、N3、D3、N4，并以狼人阵营通过 slaughter_villagers 获胜结束。真实预言家 p03 在警上发言中公开了 N1 查杀及后续两夜警徽流，但多个玩家随后持续基于“p03 没有警徽流”的错误前提推理，最终将 p03 放逐。

当前 master 已包含此前的事实归因、警徽流和温度路由修复，不能重复实施旧结论。不过最新对局发生在这些提交合并之后，证明仍有以下缺口：

- 真实句式“第二夜N2，我计划查验p02”“第三夜N3，我计划查验p12”未被警徽流解析器覆盖。
- 玩家遗言中的行动声明与规则引擎动作混在同一语义空间。例如 p07 声称开枪，但引擎记录为不开枪；p06 对用药历史的描述也与实际事件不符。
- DiscussionSummary 仍出现三次 invalid_json，日志不能区分工具调用缺失、文本 JSON 损坏、结构校验失败或响应为空。
- 反思节点打印“持久化完成0”，但真正的 repository 写入和回读发生在之后，阶段名称会误导运行审计。
- 配置虽然已在路由层生效，但仍需证明声明值最终进入 provider wire payload，并记录 provider 强制覆盖。

本设计同时解决事实链、声明与执行动作混淆、运行时失败可观测性、配置实际生效验证和反思持久化审计。胜负不作为唯一验收条件；验收重点是事实正确、可见性正确、失败可定位、配置可证明、持久化可回读。

## 2. 方案选择

采用在现有事件流与 StructuredFact 上增量建设“分层事实账本”的方案。

不选择仅扩展若干正则和提示词的局部补丁，因为这不能阻止玩家声明覆盖规则事实。也不在本轮重写整个世界状态系统，因为公开账本、可见性策略、语义修复和反思系统已有兼容契约，整体替换的影响面与当前问题不成比例。

本方案保持事件流为事实源，在现有结构上显式标注事实权威性，并让下游组件消费统一的分层投影。

## 3. 架构与事实边界

系统中的信息分为四层：

1. **规则事实（authoritative facts）**
   只来自规则引擎事件，例如 seer_check、hunter_shot_selected、hunter_shot_resolved、hunter_shot_skipped、witch_antidote_used、witch_poison_used、player_died、vote_resolved 和警徽事件。它们描述已经由规则引擎执行或确认的状态。

2. **玩家声明（public claims）**
   来自警上发言、白天发言、PK 发言和遗言。它们只证明“某玩家公开说过某事”，不能自动成为已执行动作或真实身份。

3. **可验证公开证据（verified public evidence）**
   在当前玩家可见范围内，由公开规则事件、投票、死亡、警徽和带归因声明组合成的紧凑投影。该层供 prompt、摘要、投票和语义修复使用，但不能泄露私密事件。

4. **冲突审计事实（derived audit facts）**
   当玩家声明与可见规则事实冲突时生成。例如“p07 声称现在开枪”与“规则引擎记录 p07 选择不开枪”形成声明/执行冲突。冲突事实描述不一致，不替代任一来源。

StructuredFact.metadata 使用稳定字段表达来源：

~~~python
{
    "authority": "engine" | "player_claim" | "derived_audit",
    "support_kind": (
        "executed_action"
        | "public_speech"
        | "last_words"
        | "inferred_conflict"
    ),
    "source_event": "...",
}
~~~

这些字段必须由事实创建入口显式设置。旧事件兼容路径可根据事件类型推导默认值，但下游不能再通过自由文本猜测权威等级。

## 4. 组件职责

### 4.1 事实解析

werewolf_agent/cognition/world_state.py 继续负责把单个事件转换成结构化事实。文本解析生成的身份、查验和警徽流一律标记为 player_claim；规则事件提取器生成的事实标记为 engine。

_extract_badge_flow_targets() 扩展为有界的连续计划块解析器，至少支持：

~~~text
接下来公布警徽流：
- 第二夜N2，我计划查验p02（赵猛）。
- 第三夜N3，我计划查验p12（冯弈）。
~~~

解析结果保持现有表示：单个 badge_flow_claim，首个目标写入 target_player，完整顺序写入 metadata["badge_flow_order"]。解析器只读取“警徽流”标记后的连续计划行，并在空行、新段落或不符合计划语法的行停止，避免吞入后续普通发言中的玩家编号。

### 4.2 公开账本

werewolf_agent/runtime/public_ledger.py 保持公开可见性过滤，并扩展为三类输出：

- 已公开确认的规则事实。
- 带说话人归因的玩家声明。
- 声明和公开规则事实之间的冲突。

最低覆盖范围包括猎人开枪、女巫公开用药信息、死亡、放逐、投票、警徽、查验声明和警徽流声明。私密夜间信息只能在现有 event_visibility 允许时进入某个玩家的投影。

遗言中的“开枪”“用药”“查验”等文本保留为声明。只有对应规则事件才能产生 executed_action。不存在于本局角色配置中的角色或能力描述只能作为玩家声明或不支持声明，不能成为权威事实。

### 4.3 Agent 上下文与提示词

AgentContext 接收紧凑、稳定的三段式投影：

~~~text
已公开确认
玩家公开声明
声明冲突
~~~

查验声明和警徽流直接来自分层账本，不依赖 DiscussionSummary 成功。提示词明确区分“某玩家声称执行”与“规则已经执行”，并禁止把 player_claim 改写为确定事实。

DiscussionSummary 只负责压缩立场、怀疑目标和证据引用，不再承担保存关键事实的职责。确定性 fallback 也从结构化账本提取关键事实，不以泛化模板覆盖已有公开证据。

### 4.4 语义修复

werewolf_agent/agents/semantic_repair_audit.py 在判断声明是否有公开支持时，同时检查支持类型。

- “p07 声称要开枪”可由公开遗言支持。
- “p07 已经开枪”必须由公开规则事件支持。
- “p06 N1 救了 p09”若没有对应公开执行事实，不能作为确定动作通过语义门禁。

原因码至少区分：

- 没有任何公开来源支持。
- 有玩家声明，但被改写成已执行动作。
- 说话人归因丢失。
- 否定语义丢失。

现有 unsupported_public_claim 保持失败关闭，新增细分原因只用于定位账本或模型问题，不放宽安全边界。

## 5. DiscussionSummary 失败处理

最新日志的三次 invalid_json 不能直接归因于旧的 fenced JSON 问题，因为当前解析器已经支持修复 fenced JSON。实施阶段先用诊断信息确定响应在哪一层损坏。

每次摘要尝试记录以下不含原始内容的字段：

~~~text
structured_output_mode
tool_call_required
tool_call_received
response_shape
json_candidate_count
failure_stage
failure_code
~~~

解析顺序为：

1. 原生工具调用模式优先解析工具参数。
2. 只有当前路由策略允许时才解析文本 JSON。
3. 协议或 schema 失败时，按 StructuredOutputPolicy 进行一次相应模式的修复尝试。
4. 最终失败使用确定性摘要，同时继续提供分层账本中的关键事实。

失败码明确区分 missing_tool_call、invalid_json、schema_validation_failed、empty_response 和 model_generation_failed。不得将所有失败统一记录为 invalid_json。

## 6. 配置实际生效闭环

配置验收覆盖完整链路：

~~~text
config/models.yaml 声明值
    -> 路由解析后的 ModelConfig
    -> provider 最终 HTTP 请求体
    -> GenerateResult / attempt 审计值
~~~

逐项验证 temperature、top_p、max_tokens、timeout、retry_count、reasoning_level、structured_output_mode、base_url 和 extra_body。

已合并的任务温度配置先保持不变：

- 摘要和法官判断：0.2。
- 投票和夜间选择：0.3。
- 普通发言：0.4。
- 狼人密谈和欺骗性发言：0.5。

本轮不凭单局输赢再次调整温度。只有活跃路由的 wire payload 与声明不一致，或多局指标证明质量退化时，才修改数值。

Provider 因协议约束覆盖参数时必须显式记录：

~~~python
effective_temperature = 1.0
temperature_override_reason = "thinking_requires_temperature_1"
~~~

Fake HTTP client 测试必须断言最终请求体，而不只检查 ModelConfig。完整对局报告再核对活跃路由的审计值。

## 7. Fallback 与重试审计

Speech fallback 按根因分类：内容质量失败、语义门禁失败、结构化协议失败、provider 或传输失败。

确定性 fallback 必须保留当前任务目标和已验证结构化事实。泛化发言只能在不存在可用事实和任务目标时使用。

unsupported_public_claim 失败不得绕过。审计必须能说明是“公开来源缺失”还是“声明被提升成执行事实”，以便区分模型输出问题与事实投影问题。

## 8. 反思生成与持久化

反思处理拆分为两个明确阶段。

### 8.1 生成与验证阶段

reflection_complete 只报告处理玩家数、成功生成数、验证通过数、未生成或验证失败数及安全原因码。该阶段不再打印“持久化完成”，因为 repository 写入尚未发生。

### 8.2 持久化与回读阶段

_save_memory_snapshot() 完成 Reflection V2 写入、snapshot 保存和回读校验后产生 reflection_persistence_audit。该事件报告预期条目数、成功写入数、repository 回读匹配数、snapshot 回读结果、失败阶段和安全原因码，以及回滚是否完成。

只有 repository 与 snapshot 回读都完全匹配时，日志才能报告“持久化完成”。保存 snapshot 本身不能替代 Reflection V2 持久化成功。

增加跨局测试：上一局有效 Reflection V2 必须能在下一局按既有检索契约读取，并进入允许的 RAG 战术投影。

## 9. 错误处理原则

- 权威事实缺失时保持未知，不从玩家文本推断执行结果。
- 可见性不允许时不生成给该玩家的事实或冲突投影。
- 解析不确定时保留原始声明事件，但不生成高置信结构化动作。
- 摘要失败不影响关键事实链。
- 配置覆盖必须可观测，不能静默更改。
- 持久化写入或回读不完整时失败关闭并执行已有事务回滚。
- 日志和审计字段不得包含私密 prompt、原始 provider 响应或不可见角色信息。

## 10. 测试策略

### 10.1 真实日志回归

从 g_42 提取最小原始样本：p03 的 N1 查杀及两行警徽流；p07 的不开枪规则事件及“现在开枪”的遗言；p06 的实际用药事件与冲突遗言；摘要失败的响应形态和原因码；反思生成完成到实际持久化之间的阶段顺序。

测试必须先在修复前证明目标行为不成立，再实施最小修复。

### 10.2 单元测试

- 中文夜次前缀和连续多行警徽流。
- 规则事实、玩家声明和冲突事实的权威分类。
- 执行动作断言与带归因声明的语义差异。
- 摘要工具调用、文本 JSON、损坏 JSON、schema 错误和空响应。
- 路由配置解析及 provider override 元数据。
- 反思阶段状态和回读匹配。

### 10.3 集成测试

- 事件流到公开账本、AgentContext 和 prompt 的完整投影。
- 可见性策略阻止私密夜间信息泄漏。
- 摘要 fallback 时关键事实仍可用于发言和投票。
- 反思生成、repository 写入、snapshot 保存、回读和跨局读取。

### 10.4 网关与完整对局

- 使用 fake HTTP client 验证每种活跃任务的最终 wire payload。
- 运行相关测试集后运行全量 pytest。
- 使用固定 seed 完成一局真实对局，按 N1、D1、N2、D2 等阶段逐段检查日志。

所有项目 Python 命令使用 conda run -n wofkill。

## 11. 实施顺序

1. 锁定真实日志回归样本和当前失败。
2. 修复警徽流解析并补齐事实权威元数据。
3. 扩展公开账本和声明/执行冲突投影。
4. 接入 AgentContext、提示词和确定性 fallback。
5. 加强语义门禁及细分原因码。
6. 诊断并修复 DiscussionSummary 返回形态与协议降级。
7. 完成配置 wire payload 和 provider override 审计。
8. 拆分反思生成与持久化审计，增加跨局读取验证。
9. 运行相关测试、全量测试和固定 seed 完整对局。

每一步先增加失败测试，再做最小实现，并在通过局部测试后独立提交。不得在本轮顺带重写世界状态架构或调整无证据支持的温度数值。

## 12. 完成标准

- p03 的真实警徽流解析为 [p02, p12]，并进入世界状态与公开账本。
- 后续上下文不得把“p03 没有警徽流”作为系统事实；该句只能作为带说话人归因的错误观点出现。
- p07 的“现在开枪”保持玩家声明，规则事实保持“选择不开枪”。
- p06 的遗言不能覆盖 N1 未用药和 N2 救 p05 的实际动作。
- 不存在的守卫只能作为玩家错误声明或不支持声明，不能进入权威事实。
- 摘要失败产生准确阶段和原因码；fallback 后查验与警徽流仍保留。
- 每个活跃路由的配置都能追踪到最终请求体，provider 覆盖有明确原因。
- 反思只有在 repository 与 snapshot 回读一致后才报告持久化成功。
- 上一局有效 Reflection V2 能在下一局被读取。
- 相关测试和全量测试通过，固定 seed 完整对局正常结束且无新增未分类异常。

完整对局中好人阵营是否获胜不作为单独的通过条件。模型输出存在随机性；事实链、审计链和持久化链的确定性正确才是本轮验收核心。
