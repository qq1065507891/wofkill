# Fallback and Observability Design

## 目标

让普通发言、警长发言、讨论摘要和反思的 fallback 保持任务可用性，同时不泄露内部提示、不制造新事实，并提供足以定位真实失败原因的安全审计。

## 已确认问题

警长发言 fallback 直接截取 `public_summary`，输出了内部时间线说明并在标题中间截断。普通发言 fallback 可能继承由错误公开事实产生的投票目标。终止审计通常只剩 `speech_quality`，无法判断缺少立场、证据、目标或声明逻辑。`attempt_count` 包含 repair 和 safe fallback，而 `retry_count` 只统计 `RouteKind.RETRY`，容易被误解。

讨论摘要会进行一次修复，但日志只显示最终 `invalid_json` 或 `schema_validation_failed`，缺少按结构化模式和失败阶段的聚合。

## 设计

### fallback 输入边界

fallback 不再消费任意 `public_summary` 字符串。它只接收：

- 已分类的公开事实引用；
- 当前任务目标；
- 已验证且未被拒绝的立场或目标；
- 当前合法行动和合法目标。

内部时间线说明、Prompt 标题、未验证模型输出、私密推理和被语义门拒绝的声明均不得成为 fallback 文本。

### 任务化 fallback

- 普通发言：输出一个已验证公开依据和一个明确但可撤回的倾向。
- 警长发言：输出竞选目标、公开依据和后续承诺，不复述内部时间线。
- 讨论摘要：保留结构化账本引用，不用泛化模板覆盖关键事实。
- 反思：继续使用 `not_generated`，不得把模板文本写入长期记忆。

文本裁剪按完整句或结构化线索单元进行，禁止按字符硬截断造成残句。

### 目标继承

只有目标及其证据都通过语义门禁时，terminal fallback 才能继承目标。若目标来自被拒绝事实、条件计划或无法验证的摘要，fallback 应降级为不点名的观察陈述，不能继续推动该玩家。

### 审计模型

统一区分：

- `provider_attempt_count`
- `schema_repair_count`
- `semantic_repair_count`
- `route_retry_count`
- `terminal_fallback_count`

保留现有兼容字段，但报告层必须使用明确的新名称。每次失败记录安全枚举形式的 `missing_fields`、`reason_codes`、`failure_stage`、结构化模式和累计修复历史。不得保存原始失败文本。

讨论摘要审计还需记录 primary/repair 的响应形态、是否收到 tool call、JSON candidate 数以及最终 fallback 原因。

## 错误处理

无法安全构造任务化 fallback 时，返回显式 `NO_ACTION` 或无目标的最小公开陈述。安全性优先于维持攻击目标。fallback 自身发生异常时必须 fail closed，不能回退到原始模型文本。

## 测试与验收

- 使用真实 p09 警长 fallback 前缀构造回归测试，断言不出现内部时间线、标题或截断残句。
- 测试被拒绝目标不能进入 terminal fallback。
- 测试完整句裁剪和不同任务 fallback 合同。
- 测试三类 repair 与 terminal fallback 统计互不混淆。
- 测试 `speech_quality` 保存安全 `missing_fields`，不泄露原文。
- 完整局报告必须能够按玩家、任务和失败阶段解释每一个 terminal fallback。
