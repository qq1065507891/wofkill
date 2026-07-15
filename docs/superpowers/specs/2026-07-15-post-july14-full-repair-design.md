# 7 月 14 日后对局问题全量修复设计

## 1. 背景与目标

`docs/audits/2026-07-15-post-july14-game-log-audit.md` 对 2026-07-14 之后的 6 份结构化对局和 1 局共享日志进行了复核。审计确认：狼队计划与执行证据契约不一致，31/42 个计划夜在执行边界空刀；两局耗尽 200 步；反思虽进入节点但没有有效经验；质量评分、决策重试、死亡批次和 provider fallback 还存在被防御性兜底掩盖的隐藏缺陷。

本设计目标是全量修复审计中的 K1–K3 与 N1–N12，并满足以下结果：

1. 合法狼队共识能够确定性执行，主刀、备刀、单狼和平票语义唯一。
2. 所有空刀、回退和异常终止都经过统一策略并可追溯。
3. 夜聊历史、动态状态和提示词不再制造系统概念或失效角色信息。
4. 反思只有产生并持久化有效经验时才可记为成功。
5. 保存质量分与完整日志离线重算一致，不再出现假阴性或空集合假阳性。
6. 旧 JSON、事件和批次格式继续可读；新运行只写修正后的 V2 契约。
7. 修复运行正确性后，再以独立样本评估阵营平衡，不用胜率反向掩盖规则 BUG。

## 2. 方案选择与边界

采用“契约优先、分批闭环”路线：先建立明确数据契约和可信指标，再恢复核心规则执行，随后修复回退、反思和提示行为，最后执行真实对局稳定性与平衡验证。

未选择逐函数最小补丁，因为现有缺陷跨越计划生成、证据聚合、执行边界、结算和指标；只增加条件分支会继续留下旁路。未选择一次性重写状态机，因为迁移风险过高，也无法判断平衡变化来自哪个修复。

本轮边界：

- 兼容读取旧格式，但不继续产生新的旧格式业务事件。
- 不改变狼人杀基础角色数量、胜负规则、警长票权和技能规则。
- 不用降低 quorum、随机击杀、忽略 schema 等方式掩盖失败。
- 不把猎人开枪为零、预言家平均查验次数或狼人自爆为零直接认定为路由 BUG；只修复机会链观测和真实已证实的行为缺陷。
- 不记录隐藏思维链、原始敏感 prompt、身份真值或密钥；观测数据继续遵守可见性边界。
- 本设计允许在实际实施时按相同责任边界优化局部代码，但不做无关重构。

## 3. 总体架构

```text
模型夜聊/行动
→ 确定性事实提取
→ WolfConsensusEvidenceV2
→ WolfKillDecisionService
→ NoKillPolicy
→ 规则结算
→ GameEventV2
→ AcceptanceGameProjection
→ 质量与验收

旧日志/旧事件
→ 业务域兼容适配器
→ 统一的内部 V2 视图
```

系统按六个责任域拆分：

1. 狼队共识与执行：结构化立场、确定性证据、主备优先级和统一空刀策略。
2. 生命周期与结算：结构化死亡批次、终止状态和应急导出。
3. 决策与回退：attempt/retry 语义、provider 路由和任务专用 fallback。
4. 上下文与提示：分层历史、动态存活状态、备刀与自爆等游戏语义。
5. 反思事务：生成、验证、经验提取、持久化和回滚状态机。
6. 质量与观测：完整游戏投影、支持性声明、时间戳和一致性门禁。

## 4. 狼队共识与执行

### 4.1 结构化狼人立场

每次狼人夜聊除自然语言发言外，产生私有结构化立场：

```python
class WolfTargetStance:
    wolf_id: str
    target_id: str | None
    stance: Literal["propose", "support", "oppose", "abstain"]
    priority: Literal["primary", "backup"]
    source_event_id: str
    round_number: int
```

立场在写入事件前验证：`wolf_id` 必须是当前存活狼人；`propose/support/oppose` 的目标必须是当前存活非狼人，`abstain` 的目标必须为空；`source_event_id` 必须引用本夜对应的 `wolf_discussion` 事件。相同狼人、相同优先级多次表达时，按事件序号保留最后一个有效立场作为当前立场，同时保留变更历史供审计。新的 `oppose` 或 `abstain` 会替换该狼在同一优先级上的旧 `propose/support`，因此旧支持不再计票。

自由文本只用于策略交流和展示，不作为执行证据。新运行不得从发言正则猜测 quorum。旧日志的 `evidence_from_discussion` 仅通过兼容适配器用于历史展示和离线审计，不能被新运行重新当作权威证据执行。

### 4.2 共识证据

确定性聚合器产生：

```python
class WolfPriorityConsensus:
    priority: Literal["primary", "backup"]
    target_id: str | None
    status: Literal[
        "majority",
        "single_wolf",
        "tie",
        "insufficient",
        "all_abstain",
    ]
    supporters_by_target: dict[str, list[str]]

class WolfConsensusEvidenceV2:
    night_number: int
    alive_wolf_ids: list[str]
    stances: list[WolfTargetStance]
    quorum: int
    primary: WolfPriorityConsensus
    backup: WolfPriorityConsensus
```

LLM 队长可以推荐主刀、备刀和角色分工，但不能生成支持者集合、修改 quorum 或把自然语言“大家一致”升级为事实。`evidence_quality=strong/weak` 可保留为旧日志展示字段，但不再参与 V2 执行判定。

聚合算法固定如下：

1. 分别为 `primary` 和 `backup` 建立每名存活狼的当前立场槽位。
2. 只有 `propose` 与 `support` 是目标的正向票；`oppose` 与 `abstain` 均不增加任何目标的支持数。
3. 对每个优先级按目标聚合独立 wolf ID。`quorum = floor(alive_wolf_count / 2) + 1`。
4. 单狼存在一个合法正向票时，得到 `single_wolf`；没有正向票时为 `all_abstain`。
5. 多狼时，所有当前槽位均为 `oppose/abstain` 或不存在时为 `all_abstain`。存在正向票时，只有达到 quorum 的目标可成为该优先级结果；若没有目标达到 quorum，有两个或以上目标并列最高正向票则为 `tie`，只有一个未达 quorum 的领先目标则为 `insufficient`。三种失败状态都不会产生目标 ID。
6. 分别输出 `primary: WolfPriorityConsensus` 和 `backup: WolfPriorityConsensus`；两者各自保留目标、状态和支持明细。LLM 计划中的目标若与对应聚合结果不同，只保留为非权威推荐并记录 mismatch。

严格多数和“一狼一槽位”意味着同一优先级至多一个目标达到 quorum；实现仍需断言该不变量，异常时 fail closed 并记录 `consensus_invariant_violation`。

### 4.3 主刀与备刀

- 多狼时，`primary.status=majority` 且目标在执行时仍合法，主刀才成立。
- 单狼时，其合法主刀得到 `primary.status=single_wolf` 共识。
- 主刀合法时永不比较备刀票数。
- 主刀在执行前已死亡、成为非法目标或不可选时，才读取 `backup` 聚合结果。
- 备刀必须独立处于 `majority` 或 `single_wolf`，并在执行时合法；不能因为主刀失败而自动放行 `tie/insufficient/all_abstain` 的备刀。
- 主刀和备刀同时由单狼提出时仍优先主刀，不构成平票。
- 主刀不可用且备刀失败时，NoKillPolicy 使用备刀自己的 status 映射 `true_tie/insufficient_quorum/strategic_abstain`；目标非法使用 `invalid_backup`。这样主刀和备刀的失败原因不会互相覆盖。

### 4.4 统一空刀策略

所有 `wolf_kill_target_id=None` 路径必须进入 `NoKillPolicy.resolve()`，包括计划生成失败、证据不足、真平票、非法目标、provider 不可用和主动弃权。

```python
class NoKillDecision:
    reason_code: Literal[
        "strategic_abstain",
        "true_tie",
        "insufficient_quorum",
        "invalid_primary",
        "invalid_backup",
        "plan_generation_failed",
        "provider_unavailable",
    ]
    consecutive_pre_resolution_no_kill_count: int
    forced_recovery_applied: bool
    recovered_target_id: str | None
```

统计同时保留“策略性空刀”和“系统故障空刀”分类，但连续保护使用统一的 `consecutive_pre_resolution_no_kill_count`：凡本夜在狼刀结算前因上述任一 reason 没有产生 `wolf_kill_selected`，计数加一；产生合法 `wolf_kill_selected` 时立即清零。女巫救药、守卫保护或其他技能在选刀后的抵消不属于空刀，必须清零且不得重新加一。

默认规则 `max_consecutive_pre_resolution_no_kill=2`：连续两夜没有选出合法狼刀后，第三夜应用强制恢复。该值由规则集显式保存，但测试和默认正式规则固定为 2，不能因调用路径变化。

强制恢复目标按以下稳定顺序决定：

1. 仅考虑当前存活、非狼人且规则允许的目标。
2. 为每个目标计算 `(primary_positive_support, backup_positive_support, -seat_index)`。
3. 按元组降序选择；即优先尊重主刀支持，其次备刀支持，完全无证据或同分时选择座位号最小者。
4. 没有合法目标时不伪造击杀，写 `forced_recovery_no_legal_target`，并保持计数供终局/异常审计。

`wolf_kill_forced_recovery` 必须记录原始 no-kill reasons、计数、候选评分和最终目标，使恢复策略可完全复算。

## 5. 生命周期、事件与结算

### 5.1 GameEvent V2 元数据

新事件带有：

```python
event_id: str
sequence_number: int
occurred_at: datetime
game_id: str
trace_id: str | None
schema_version: Literal["2"]
visibility: Visibility
```

`event_id` 基于 game ID 与绝对事件序号稳定生成；`occurred_at` 使用带时区时间；`sequence_number` 是审计顺序的权威来源。旧事件缺字段时由适配器生成只读规范视图，不反写历史文件。

### 5.2 死亡批次

新写入采用：

```python
class ResolutionBatchV2:
    phase: Literal["day", "night"]
    number: int
    cause: Literal[
        "vote",
        "self_destruct",
        "wolf_kill",
        "witch_poison",
        "hunter_shot",
        "rule_effect",
        "unknown",
    ]
```

兼容解析 `day_N`、`day_N_vote`、`day_N_self_destruct` 和 `night_N`。所有内部调用只比较结构字段。未知旧格式保留死亡事实并标记 `batch_parse_failed`，但不得注入需要“当前日死亡”的行为指令；同一坏批次只记录一次聚合告警。

### 5.3 游戏终态

```python
status: Literal["running", "finished", "aborted"]
termination_reason: Literal[
    "good_victory",
    "werewolf_victory",
    "step_limit",
    "graph_recursion_limit",
    "unrecoverable_runtime_error",
] | None
```

`finished` 必须有胜方，`aborted` 必须有原因、最后节点、阶段、步数和异常类型。`winner=None` 不得输出 `GAME_COMPLETE`。中止局保存 JSON 并进入稳定性统计，但不进入胜率与平衡统计。正常事件写入失败时，runner 使用最小应急导出保存终止事实。

## 6. 决策执行与回退

### 6.1 统一计数语义

`DecisionExecutionTraceV2` 使用：

```python
attempt_count: int
retry_count: int
provider_fallback_count: int
decision_outcome: DecisionOutcome
terminal_failure_code: str | None
generated_by: Literal[
    "model", "repair", "provider_fallback", "terminal_fallback"
]
```

首次成功的 `attempt_count=1`、`retry_count=0`。`attempt_count` 等于 `execution_attempts` 长度，`retry_count` 只统计 retry route，`provider_fallback_count` 只统计 provider fallback route。旧字段 `total_retry_count_until_success` 不再由新运行写入；兼容适配器负责把旧 trace 归一化。

### 6.2 provider fallback

启动时构建并验证 fallback 路由图：相邻路由不得使用相同 `(provider, model)`，重复候选被去除，不满足任务最低 reasoning 能力的候选被拒绝。没有可用备用路由时记录 `fallback_route_unavailable`，不能等到结果翻译阶段才抛异常。运行时在调用前再次验证，防止动态配置绕过启动门禁。

网络和 provider 故障可切换路由；schema 与语义故障优先执行结构化修复或同模型定向重试，避免把内容问题错误归因到 provider。

### 6.3 任务专用 terminal fallback

- 发言：仅基于当前可见事实产生最小安全陈述。
- 夜间行动：选择规则允许的确定性动作或输出明确弃权原因。
- 反思：标记 `not_generated`，不伪造经验。
- 狼队计划：从结构化立场构造，不从自由文本猜测。
- 遗言：记录无模型输出，不阻塞游戏主流程。

终退保留 `original_failure_code`、`failure_stage`、attempt/retry 计数、route history、fallback kind 和 reason code。通用 fallback 文案不得覆盖真实失败分类。

## 7. 上下文、提示词与角色行为

### 7.1 分层夜聊上下文

夜聊输入由三部分组成：

1. 当前夜确定性状态：存活玩家、存活狼人、合法目标、上夜执行结果和本夜结构化立场。
2. 当前夜全部轮次的确定性摘要。
3. 最近原始发言；超出预算的更早发言保留摘要和事件 ID。

任何有效结构化立场都不能因自然语言 transcript 截断而丢失。每次模型调用记录被注入的事件 ID、原文条数、摘要条数和截断数量，并记录：

- `prompt_contract_id` 与 `prompt_contract_version`；
- 各必需动态区块的布尔注入确认，例如存活状态、备刀语义和当前夜立场；
- `rendered_prompt_length`；
- 对最终实际发送 prompt 计算的 HMAC-SHA256 指纹。

HMAC 使用部署侧审计密钥，密钥和原始 prompt 均不得写入游戏日志。测试使用固定测试密钥捕获最终请求并复算指纹，从而证明修复后的动态状态和规则语义确实进入模型输入，而不暴露原始私有内容。模板版本或任一必需区块缺失时，关键任务在调用前 fail closed。

### 7.2 提示语义

所有狼人提示明确：备刀仅在主刀结算前非法时替代，不是女巫救药后的同夜第二刀；已死亡玩家不能继续作为击杀、归票和角色分工目标；动态目标应描述为局内存活事实，不能出现“系统候选列表”；队长只能引用真实结构化支持事件。

每夜重新生成动态指令。旧计划只作为历史，不直接复制目标和已失效角色分工。不确定信息必须保留不确定性，不能被写成已知事实。

### 7.3 技能机会链

自爆增加 `self_destruct_opportunity → selected/declined → resolved`；猎人增加 `hunter_shot_opportunity → selected/declined/blocked → resolved`；预言家按“存活且具备查验资格的夜晚”记录 `seer_check_opportunity → selected/repaired/skipped → resolved`。

可见性与 payload 固定如下：

| 事件 | 运行期可见性 | 允许字段 | 禁止字段 |
|---|---|---|---|
| self-destruct opportunity/selected/declined | 仅行动狼人和 moderator | actor_id、机会/选择状态、公开证据引用、失败码 | 其他狼人私有计划、隐藏角色真值 |
| self-destruct resolved | public | actor_id、公开宣告、规则结算结果 | 狼队计划和私有理由 |
| hunter opportunity/selected/declined/blocked | 仅猎人本人和 moderator | actor_id、合法目标、选择/阻塞码 | 目标真实身份、其他私有信息 |
| hunter resolved | public | shooter_id、target_id、公开结算结果 | 私有候选和模型 intent |
| seer opportunity/selected/repaired/skipped/resolved | 仅预言家本人和 moderator | actor_id、target_id、查验结果、修复/跳过码 | 向任何公共事件复制查验结果或角色真值 |

“仅本人和 moderator”事件在导出时沿用现有私有可见性模型；若当前枚举不支持双主体访问，内部保存 moderator-only 权威事件，并通过角色私有视图投影给本人，不能扩大为 public。

指标据此区分没有机会、主动拒绝、非法动作、节点未达和成功结算。报告中未证实的猎人/预言家路由问题不通过修改规则来“修复”。

## 8. 反思事务

每个玩家的反思经过：

```text
not_requested
→ generated
→ schema_validated
→ facts_verified
→ lessons_verified
→ persisted
```

任一步失败保存 `failure_stage`、`failure_code` 和 `decision_id`。局级结果为 `complete`、`partial`、`no_valid_entries`、`persistence_failed` 或 `not_run`。

`complete` 要求至少一条已验证 lesson 且所有预期条目成功持久化。`expected_entry_count=0` 必须得到 `no_valid_entries`，不能利用 `all([])` 成为成功事务。每条经验通过 `decision_id/claim_id/entry_id` 串联；写入或回读失败触发事务回滚，并记录回滚是否完成。

质量分必须在反思与持久化审计事件写入后最终计算。过程中的预览分不得保存为最终 `quality_score`。

## 9. 质量、观测与统计

### 9.1 完整游戏投影

建立 `AcceptanceGameProjection`，至少包含 `game_id`、`events`、`players`、`winning_faction` 和 `status`。所有验收计算器只消费该类型，不接受任意缺字段字典。缺少历史输入时输出 `supported=false` 及原因，不输出误导性的 0。

### 9.2 指标拆分

- `speech_non_empty_rate`：只表示文本非空。
- `speech_model_success_rate`：最终内容由模型或有效修复产生。
- `speech_terminal_fallback_rate`：最终使用终退内容。
- `speech_semantic_acceptance_rate`：通过语义校验。
- possible-world 唯一性必须得到玩家角色集合，否则 unsupported。
- 平安夜只以已完成夜间结算为分母，并区分主动空刀、系统空刀、救药、守护和其他技能抵消。
- aborted 局不进入胜率，但进入稳定性分母。

保存质量分必须与读取完整 JSON 后的离线重算逐字段一致。阵营平衡指标同时纳入 `wolf_team_plan_fallback`、执行边界 no-kill 和技能抵消，不能只统计 `action_trace_audit`。

### 9.3 隐私边界

公共事件不得包含角色真值、狼队私有计划、私有 intent、隐藏目标或原始 provider 请求。新增事件元数据和失败码必须通过现有 exposure audit。私有证据通过 `source_event_id` 引用，不能把私有正文复制到公共 payload。

## 10. 模块责任与预计落点

| 模块 | 责任 |
|---|---|
| `runtime/wolf_consensus_evidence.py` | 构造和验证 V2 狼队共识证据 |
| `runtime/wolf_no_kill_policy.py` | 所有空刀原因、连续结算前空刀和恢复策略 |
| `core/resolution_batches.py` | V2 死亡批次及旧格式兼容解析 |
| `runtime/game_termination.py` | finished/aborted 终态和应急导出 |
| `agents/trace_schemas.py` | V2 决策 trace 与计数不变量 |
| `runtime/provider_fallback_policy.py` | fallback 路由构建、去重和验证 |
| `runtime/reflection_transaction.py` | 反思状态机、持久化和回滚 |
| `evaluation/game_projection.py` | 完整验收输入投影 |
| `evaluation/*_metrics.py` | 只消费完整 projection 的领域指标 |
| `runtime/event_metadata.py` | 事件 ID、序号、时间戳和 trace ID |

兼容逻辑放在相应业务模块附近，不建立承担所有格式迁移的万能模块。实施时若现有文件仍保持单一职责，可直接扩展；只有形成独立业务责任时才新增上述文件。

## 11. 分批迁移

### 第一批：测量基础与兼容契约

- 为 6 份冻结 JSON 建立只读特征测试。
- 引入事件 V2 元数据和业务域兼容读取。
- 修复完整游戏 projection、attempt/retry 语义和死亡批次解析。
- 增加保存指标与离线重算一致性门禁。

第一批不改变游戏策略，确保后续修复能被正确测量。

### 第二批：核心规则执行

- 引入结构化狼人立场和确定性共识证据。
- 修复多狼多数、单狼、主备优先级和非法目标降级。
- 统一所有空刀出口并恢复连续结算前空刀保护。
- 为步数、递归上限和运行异常写结构化 abort。

### 第三批：回退、反思与行为质量

- 启动期和运行期验证 provider fallback。
- 使用任务专用 terminal fallback 并保留原始失败码。
- 引入反思事务，禁止零条目成功。
- 修复分层上下文、备刀语义、动态状态和系统措辞。
- 补齐自爆、猎人和预言家机会事件。

### 第四批：集成与平衡回归

- 兼容回放历史 JSON。
- 执行固定场景和精确 10 局稳定性 soak。
- 结构门禁通过后，执行至少 30 个完整局的独立平衡批次。
- 只根据无运行污染的数据调整平衡参数。

每批独立提交和验证。前一批未通过，不进入下一批。

## 12. 测试与验收

### 12.1 单元与属性测试

- 三狼 2:1、真平票、全弃权、单狼主备和主刀非法。
- 系统故障与主动策略原因分别分类，同时统一更新连续结算前空刀计数。
- 所有旧死亡批次格式和未知格式。
- 首次成功、一次重试、provider fallback 和终退。
- 相同 provider/model fallback 在启动时被拒绝。
- 零反思、部分反思、持久化失败和回滚。
- finished 必有 winner，aborted 必有 reason。
- 任意共识输入最多选择一个合法、存活、非狼人目标。
- 当前日指令不能包含其他日期死亡。

### 12.2 集成测试

```text
夜聊 → 立场 → 共识 → 狼刀 → 技能抵消 → 死亡
计划失败 → NoKillPolicy → 连续故障恢复
递归上限 → aborted → JSON → 稳定性指标
终局 → reflection → persistence audit → final quality score
旧 JSON → compatibility adapter → 离线指标重算
```

### 12.3 硬性门禁

- `attempt_retry_consistency_error_count == 0`
- malformed death batch 为 0
- same-route provider fallback 为 0
- 完整局误记 completed=0 为 0
- 零有效反思判 complete 为 0
- 保存分与离线重算差异为 0
- `winner=None` 的 `GAME_COMPLETE` 为 0
- 非 finished 局结构化 abort 覆盖率 100%
- 当前日死亡跨日注入为 0
- 公共敏感字段泄漏为 0
- 单狼合法主刀执行率 100%
- 达到多数的合法主刀执行率 100%
- 主刀合法时备刀抢占为 0
- 连续结算前空刀超过配置上限为 0
- 未执行狼刀结构化原因覆盖率 100%
- terminal fallback 原始失败码覆盖率 100%
- 狼队结构化立场 `source_event_id` 可追溯率 100%
- 反思 `decision_id/claim_id/entry_id` 可追溯率 100%

## 13. 平衡验证

恢复狼刀后狼人胜率可能上升，这是修复规则执行后的预期变化，不能因此回滚正确逻辑。结构门禁全部通过后，30 局独立样本观察：阵营胜率及置信区间、平均天数、异常长局、狼刀到实际死亡的转化、白天与夜间死亡占比、预言家有效查验夜、猎人和自爆机会/选择率，以及 terminal fallback 与阵营胜率的关联。

10 局稳定性 soak 用于发现结构错误，不用于宣称阵营平衡。30 局仍只作为初步平衡信号；若置信区间过宽，应扩样而不是过拟合参数。

## 14. 问题覆盖矩阵

| 编号 | 修复设计 | 主要验收 |
|---|---|---|
| K1 | 结构化立场、确定性共识、主备优先级、统一空刀 | 合法多数和单狼主刀执行率 100% |
| K2 | 分层历史、事件 ID、执行证据不依赖 transcript | 结构化立场在截断后仍完整 |
| K3 | 备刀、存活状态、系统措辞和不确定性提示 | 提示契约测试与真实 trace 抽检 |
| N1 | 单狼分支和主刀优先 | 单狼主备不再平票 |
| N2 | 所有空刀进入 `NoKillPolicy` | 连续结算前空刀不超过阈值 |
| N3 | 结构化 aborted 终态 | 两类上限均保存 abort，不能伪完成 |
| N4 | 支持者只来自结构化 stance | reasoning 不能伪造支持者 |
| N5 | 明确定义备刀不是第二刀 | 女巫救药后不执行备刀 |
| N6 | 指标拆分和完整 decisive path 统计 | 白天流程不能掩盖狼刀失败 |
| N7 | 事件时间、序号、trace ID | 事件链可按 ID 和时间审计 |
| N8 | 反思状态机与失败分类 | 无有效 lesson 时为 no_valid_entries |
| N9 | `ResolutionBatchV2` 与兼容解析 | malformed 和跨日注入均为 0 |
| N10 | 任务专用 fallback 与发言指标拆分 | fallback 不再计作模型成功 |
| N11 | 完整 projection、空集合失败、最终时点重算 | 保存与离线评分一致 |
| N12 | fallback 路由启动/运行双重验证 | 相同 provider/model 路由为 0 |

外部报告中未成立的主张也有明确处理：预言家按可查验夜统计；猎人记录机会、拒绝和阻塞；自爆记录机会链。它们不通过修改基础规则制造虚假事件。

## 15. 发布与回滚

- 所有新游戏运行强制写 V2，不提供写回 legacy 业务事件的配置开关。
- 迁移采用兼容双读并直接切换 V2 新写；不双写两个权威事件，也不保留 legacy writer 的生产路径。
- 兼容读取长期保留，只用于历史 JSON、回放和离线审计。
- 回滚只能回滚某批业务逻辑，同时保持 V2 写契约；不得以重新启用 legacy writer 作为回滚手段。若旧版本不能读取 V2，则不能直接回滚到该版本，必须发布带 V2 reader 的修复版本。
- 每一批提供独立回滚点，禁止把四批压成一个不可审查提交。
- 真实 soak 若触发 P1，立即停止平衡批次并回到对应批次修复。
- 临时兼容告警必须有聚合计数和移除条件，不能永久淹没运行日志。

## 16. 完成定义

四批代码、测试和文档全部合并；硬性门禁全绿；精确 10 局稳定性 soak 无结构错误；至少 30 个完整局的平衡报告明确区分规则执行、技能抵消、模型终退和真实策略结果；审计报告中的 K1–K3、N1–N12 均能映射到自动化测试、运行指标和最终证据。
