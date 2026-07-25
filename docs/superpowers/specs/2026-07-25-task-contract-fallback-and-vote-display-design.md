# 任务契约、终退可观测性与投票显示完整修复设计

## 1. 背景

2026-07-25 实局 `g_3334463270` 暴露了几类相互关联但责任边界不同的问题：

- 三次公开发言耗尽三次行动重试并进入安全模板终退；
- 一次内部讨论摘要使用 `TaskType.SPEECH`，被错误地按公开发言质量校验；
- 一条公开发言修复链先后出现 `speech_quality` 和
  `unsupported_public_claim`，说明质量补全与公开证据安全约束未共同收敛；
- 一名玩家的赛后反思连续三次 `schema_validation` 后未生成；
- 结构化游戏导出因 `json_item_limit_exceeded` 丢失事件，却仍显示
  `fallback_rate=0.000`；
- 日间投票直接显示内部半票单位，普通玩家显示为 2 票，警长显示为 3 票。

本设计采用“共享执行骨架 + 任务专用契约”。模型路由、provider 重试、attempt
身份和审计继续共享；输出 Schema、任务校验、修复策略和终退结果由任务契约负责。

## 2. 目标

1. 区分公开发言、内部讨论摘要和赛后反思，不再共用错误的输出及质量契约。
2. 让公开发言修复同时满足内容完整性和公开证据安全，不在两类门禁之间反复摆动。
3. 让反思直接生成专用草稿，而不是先包装成通用 `PlayerAction`。
4. 在不保存被拒自由文本、私有信息或 correction hint 的前提下，提供字段级诊断。
5. 截断或缺失事件时，事件派生指标必须标记为不支持，不能用零伪装成功。
6. 保留整数半票单位的规则结算，同时向日志、事件、API、UI 和报告显示真实票数。
7. 保持既有公开 API 和事件消费者可迁移，避免一次性破坏性变更。

## 3. 非目标

- 不改变 provider 路由、模型分配或现有 300 秒 provider timeout。
- 不通过简单增加行动重试次数掩盖契约或修复算法问题。
- 不改变日间投票、平票、复投、警长加权或放逐结果的规则语义。
- 不保存被质量门或语义门拒绝的原始模型文本。
- 不在本次工作中构建通用工作流引擎或重写全部 `PlayerAgent`。
- 不根据这一个实局重新平衡角色数量或阵营胜率。

## 4. 总体架构

### 4.1 共享执行骨架

共享层只负责：

- provider/model 路由；
- provider 传输重试和 fallback route；
- structured-output 模式降级；
- request、decision 和 attempt 身份；
- attempt 计数与结果状态；
- 脱敏诊断投影；
- 将最终结果交回调用方。

共享层不判断发言是否充分、摘要是否正确或反思事实是否可信。

### 4.2 任务契约

每个任务契约显式声明：

- 输出 Schema；
- 允许的 structured-output 模式；
- parser；
- validator 和 quality gate；
- 内容是否会公开；
- repair 策略；
- terminal fallback；
- 可持久化的安全诊断字段。

首批契约为：

- `PublicSpeechContract`
- `DiscussionSummaryContract`
- `ReflectionContract`

契约可以实现共同协议，但不要求引入重量级注册框架。现有调用点可以通过窄入口函数
选择契约，以小步迁移方式替代一次性重写。

## 5. 公开发言契约

### 5.1 适用范围

覆盖普通发言、警长竞选发言、PK 发言和其他明确公开的 speech task。最后遗言保留
现有特殊动作语义，但可复用安全诊断数据结构。

### 5.2 联合修复约束

当前串行修补容易产生：

```text
内容不足 -> 补内容 -> 引入不支持事实 -> 删事实 -> 内容再次不足
```

新流程在首次可修复失败后生成结构化 `SpeechRepairRequirements`，至少包含：

```text
missing_requirements
forbidden_claim_keys
required_target
allowed_evidence_refs
preserve_negations
preserve_attributions
```

后续 attempt 必须同时满足累计约束。新失败只增加或收紧约束，不能覆盖先前约束。

### 5.3 校验顺序

每次生成后的校验顺序固定为：

1. Schema 和合法动作；
2. 公开证据及归因安全；
3. 发言内容质量；
4. 目标、否定和已验证 claim 保留；
5. 发布。

校验结果应一次返回所有当前可判定的稳定 reason codes，避免每次只暴露一个问题。

### 5.4 终端 fallback

公开发言终退继续 fail-closed：

- 只能使用公开可见事实；
- 优先保留经过验证的原目标；
- 不声称系统未确认的角色、查验、死亡原因或夜间事实；
- 明确标记 `generated_by=terminal_fallback`；
- 不把被拒草稿写入 trace。

## 6. 内部讨论摘要契约

### 6.1 新任务类型

增加 `TaskType.DISCUSSION_SUMMARY`。`summarize_positions()` 不再构造
`TaskType.SPEECH`。

### 6.2 输出结构

新增窄 Schema `DiscussionSummary`：

```text
summary: str
suspected_players: list[str]
trusted_players: list[str]
vote_target: str | None
evidence_refs: list[str]
```

字段只表达玩家对当天公开讨论的内部整理。它不是公开发言，不生成 speech 事件，
也不经过公开发言质量门。

### 6.3 兼容投影

`discussion_positions` 的现有字符串消费者继续可用。新结构化结果提供确定性文本投影；
读取旧 checkpoint 时仍接受字符串。新写入同时保留结构化对象或通过明确版本字段区分，
避免消费者猜测类型。

### 6.4 失败行为

模型或 Schema 失败后直接使用现有确定性摘要，不生成普通发言模板。审计记录：

```text
task=discussion_summary
outcome=deterministic_fallback
failure_code=<safe code>
```

## 7. Reflection 专用契约

### 7.1 专用输出

反思入口直接请求 `ReflectionDraft`，不再先生成通用 `PlayerAction`。草稿至少包括：

```text
claims
lessons
mistakes
preserved_strengths
```

具体字段复用现有 reflection synthesis 和 verification 领域模型，不重复创建平行概念。

### 7.2 数据流

```text
生成 ReflectionDraft
-> Schema 校验
-> 游戏事实和公开/私有边界验证
-> 玩家标识匿名化
-> 反思质量门
-> 持久化
```

任一安全或事实验证失败都不能通过自由文本 fallback 写入长期记忆。

### 7.3 失败事务

三次生成失败最终只形成一个 reflection terminal transaction：

- `status=not_generated`
- `failure_code` 保留最终原始安全代码；
- `failure_stage` 明确；
- verified facts/lessons 均为空；
- 不持久化模型草稿。

`reflection_complete` 的整体状态必须基于成功、失败和持久化计数，而非仅依据遍历了多少
玩家。日志改为：

```text
[复盘] 处理12位：成功11，未生成1，持久化完成11
```

## 8. 安全诊断

### 8.1 可记录字段

每个失败 attempt 可以记录：

- decision/request/attempt 身份；
- task contract；
- structured-output mode；
- failure stage 和 stable reason code；
- Schema 字段路径；
- Pydantic 错误类型；
- 缺失质量要求；
- unsupported claim 的规范化类别或安全键；
- 是否发生 mode downgrade；
- 是否进入 terminal fallback。

### 8.2 禁止记录

禁止持久化或输出：

- 被拒发言或反思原文；
- correction hint 原文；
- provider 原始响应正文；
- 私有角色、狼队信息或密钥；
- 能反推出私有内容的完整 validation input。

Schema 错误路径只保留字段名和安全错误类型，例如：

```text
field_paths=claims.0.source_event_id,lessons.1.abstraction
error_types=missing,string_type
```

## 9. 指标真实性

### 9.1 支持性判定

任何依赖事件全集的指标必须先判断输入是否完整。当出现
`json_item_limit_exceeded`、`events=[]` 或明确的部分导出标记时：

- 值设为 `null`；
- `<metric>_supported=false`；
- `<metric>_unsupported_reason=json_item_limit_exceeded`；
- 禁止用 `0` 或 `0.000` 表示无失败。

### 9.2 stdout 与正式指标

stdout 可以作为实局诊断证据，但不能静默补入结构化 acceptance 指标。报告可以分别展示：

- structured metric；
- stdout diagnostic count；
- 两者的数据来源和支持状态。

只有完整结构化事件才能生成正式 fallback rate。

## 10. 投票单位与显示契约

### 10.1 内部规则保持不变

继续以整数单位结算：

- 普通玩家一票为 `base_vote_weight=2`；
- 警长 1.5 票为内部 `3`；
- 最大票、平票、复投和放逐判断均使用整数单位。

### 10.2 统一换算

新增集中 helper，把内部单位转换为实际票数。转换不得使用二进制浮点做规则判断；
展示可使用 `Decimal` 或保证精确的整数/半整数格式化：

```text
2 units -> 1
3 units -> 1.5
21 units -> 10.5
13 units -> 6.5
```

日志、法官广播、事件、API、UI 和报告不得各自实现一套换算。

### 10.3 事件字段

新事件明确区分单位与实际票数：

```json
{
  "base_vote_weight": 2,
  "weighted_tally_units": {"p02": 21},
  "weighted_tally": {"p02": 10.5},
  "vote_weight_units": {"p04": 3},
  "vote_weights": {"p04": 1.5}
}
```

新写入遵循以上语义。迁移期读取端兼容旧 payload；旧字段若缺少版本或
`base_vote_weight`，按旧事件版本解释，不能对同一数据重复除以 2。

### 10.4 本局验收样例

`g_3334463270` 的同等票型应显示：

- D1：p02 为 10.5 票，p01 为 1 票；
- D3：p11 为 6.5 票，p07 为 1 票。

放逐结果必须与修复前一致。

## 11. 分批实施

### 批次一：票权显示

- 建立统一换算 helper；
- 修复 `day_vote.py` 日志和事件；
- 更新 API/UI/报告消费者；
- 验证规则结算不变。

### 批次二：摘要任务分离

- 增加 `DISCUSSION_SUMMARY`；
- 引入 `DiscussionSummary` Schema；
- 迁移 `summarize_positions()`；
- 保持字符串兼容投影。

### 批次三：公开发言联合修复

- 抽离 repair requirements 与状态；
- 聚合安全门和质量门结果；
- 保持脱敏终退。

### 批次四：Reflection 专用管线

- 直接生成 `ReflectionDraft`；
- 复用现有验证和匿名化；
- 修正整体完成状态和日志。

### 批次五：指标支持状态

- 统一截断检测；
- 把 unsupported 零值改为 `null`；
- 分离 stdout 诊断和正式指标。

每批独立提交、独立验证，可单独回滚。

## 12. 测试策略

### 12.1 票权

- 普通票、警长票和半票格式化；
- D1/D3 实局票型回归；
- 平票、复投、无效票和死人票；
- 旧事件兼容读取；
- API/UI 不重复换算。

### 12.2 摘要

- 摘要不调用公开发言质量门；
- 不产生公开 speech 事件；
- 结构化结果提供稳定文本投影；
- 模型失败使用确定性摘要；
- 日志任务名为 `discussion_summary`。

### 12.3 公开发言

- unsupported claim 修复不丢目标；
- 补充证据不引入新私有事实；
- 多个 reason code 同时返回并跨 attempt 累积；
- rejected text 和 correction hint 不进入 trace；
- terminal fallback 只使用公开事实。

### 12.4 Reflection

- 缺字段、错误类型和多余字段的安全诊断；
- terminal failure 只形成一个失败事务；
- 失败草稿不持久化；
- 成功草稿仍通过事实验证、匿名化和质量门；
- 单个玩家失败时整体状态和日志计数正确。

### 12.5 指标

- 完整事件且确实零 fallback；
- `events=[] + json_item_limit_exceeded`；
- 部分事件导出；
- stdout 有 fallback 但结构化事件被截断；
- unsupported 指标不被序列化为零。

## 13. 验证与生产验收

每批按以下顺序验证：

1. 先添加能够复现当前问题的失败测试；
2. 实现单一批次；
3. 运行定向测试；
4. 运行 agent/runtime/evaluation 交叉回归；
5. 执行静态检查、编译和 `git diff --check`；
6. 所有批次合并后运行新的真实游戏。

生产验收至少检查：

- 普通票和警长票显示为 1 和 1.5；
- 摘要不再报告为 speech fallback；
- 每个失败有字段级且脱敏的诊断；
- speech repair 不在安全门和质量门之间反复摆动；
- reflection 成功、失败和持久化数量一致；
- JSON 指标不与 stdout 事实矛盾。

只有新实局可以证明生产行为改善；单元测试通过不能替代实局复核。

## 14. 成功标准

- `TaskType.SPEECH` 终退只代表真实公开发言任务；
- discussion summary 有独立契约和独立指标；
- reflection 不再依赖通用 `PlayerAction` Schema；
- 失败诊断可定位但不泄露被拒内容；
- 截断导出不再显示虚假的零 fallback；
- 投票日志正确显示实际票数，规则结果保持一致；
- 新实局中报告、事件与 stdout 对同一失败给出一致结论。
