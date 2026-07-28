# Reflection Memory Reliability Design

## 目标

让完整游戏的反思链路对全部 12 名玩家达到生成、Schema 校验、事实与经验核验、持久化回读和下一局恢复的完整闭环。任何一名玩家失败都必须使反思验收失败，但不得阻止游戏终局状态保存。

## 已确认问题

`g_42` 和 `g42_acceptance` 均只有 9/12 个有效反思。六名失败玩家全部停在 `schema_validated` 阶段，原因均为 `invalid_structured_draft`。

当前反思使用 `text_json`、温度 `1.0` 和相同提示的有限重试。重试没有携带安全字段路径或错误类型，也没有结构化模式 fallback。字段级诊断在运行节点被丢弃。持久化实际只覆盖 9 个有效条目，日志却按玩家总数输出“12 reviews”。验收允许 `partial` 事务通过。

## 设计

### 五段事务

每名玩家的反思使用统一事务状态：

1. `generated`
2. `schema_validated`
3. `lessons_verified`
4. `persisted`
5. `restored`

游戏级反思状态只有全部 12 名玩家达到当前要求的最终阶段时才为 `complete`。`partial` 只用于诊断和恢复操作，不能满足完整局验收。

### 结构化生成

启动时根据 provider 能力选择最强受支持模式：原生 JSON Schema 或 tool call 优先，其次 `json_object`，最后才允许 `text_json + 本地严格校验`。不向不支持的 provider 强行发送模式。

反思使用独立低随机性配置。默认温度应为严格结构化生成可接受的低值；具体值通过配置测试和真实探针确定，不与普通发言或推理 profile 共用。

第一次 Schema 失败后，下一次请求必须携带脱敏后的 `field_paths` 和 `error_types`，并使用同一个有界 `GenerationAttemptContext` 记录 primary/repair。原始模型文本、游戏私密信息和 Pydantic 输入值不得进入审计或修复提示。

如果当前 provider 在限定次数内仍失败，可以进入一个显式配置的反思 provider fallback。provider fallback 和 Schema repair 分开计数，总调用次数必须有固定上限。

### 核验与持久化

只有同时存在至少一个已核验事实声明和至少一条绑定该声明的已核验经验，才允许写入 V2 反思。失败玩家不得生成模板经验、不得复制其他玩家经验、不得用旧同 ID 条目冒充本局成功。

保存后必须从 repository 回读并严格比较 `game_id/player_id/decision_id/entry_id/verified_claim_ids`。反思条目全部回读成功后才保存或更新 `latest` 快照。中途失败沿用现有事务回滚机制。

下一局恢复测试必须确认本局 12 个条目可检索，并且实际进入对应玩家的记忆提示；仅能读到快照文件不算恢复成功。

### 审计和日志

每名玩家记录以下安全字段：

- provider、model、structured output mode
- provider attempt、repair attempt 和 fallback attempt 数
- failure stage、failure code
- 脱敏字段路径和错误类型
- verified claim/lesson 数
- persisted/restored 布尔结果

日志分别输出 `generated=12/12`、`verified=12/12`、`persisted=12/12` 和 `restored=12/12`。禁止使用玩家总数或自动 ReviewReport 数替代反思条目数。

## 失败语义

- 反思失败不回滚已经完成的游戏终局。
- 反思失败使反思 acceptance 失败，并留下可重试事务。
- 不允许静默把 `partial` 视为成功。
- 重跑必须使用明确的反思事务 ID，不能覆盖不相干游戏的终局或记忆。

## 测试与验收

- 单元测试覆盖结构化模式选择、定向 Schema repair、provider fallback 和调用上限。
- 事务测试覆盖 11/12、持久化失败、回读不一致、回滚和重试。
- 日志测试确认 9 个条目只能报告 `persisted=9/12`。
- acceptance 测试确认 `partial` 必须失败，只有 12/12 可通过。
- 恢复端到端测试确认 12 名玩家均从新快照获得本局已核验经验。
- 至少两局真实模型完整局连续达到生成、核验和持久化 12/12；第二局启动后恢复验证也必须 12/12。
