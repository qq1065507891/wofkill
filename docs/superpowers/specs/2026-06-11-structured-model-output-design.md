# 模型结构化输出设计

## 目标

降低玩家行动的解析失败、协议冲突和无信息 fallback，并让日志能区分模型服务错误、输出协议错误、schema 错误和游戏语义错误。

本次不实现：

- GLM 熔断或新增备用模型
- 模型、人格、座位轮换
- 50～100 局成对实验和消融实验

## 现状问题

1. `allow_text_tool_fallback` 同时表达“是否强制工具调用”和“是否接受文本 JSON”，无法记录实际使用的协议。
2. `TARGET_CHOICE` 提示要求 `choice`，但工具 schema 仍要求 `action_type` 和 `target_id`，模型同时收到冲突合同。
3. 缺少工具调用、JSON 解析失败、Pydantic schema 失败和行动语义失败共用相近的重试路径。
4. `ActionTrace` 只能记录笼统的 `structured_failure_reason`，无法按协议和失败阶段统计。

## 方案

### 1. ActionContract

增加统一的 `ActionContract`，根据当前 `OutputMode` 生成工具 schema 和字段清单：

- `FULL_ACTION`：`action_type`、`target_id`、`speech`、`reason`、`confidence`
- `TARGET_CHOICE`：`choice`、`reason`、`confidence`；投票时追加六个审计字段
- `SPEECH_INTENT`：`intent`、`target_id`、`speech`、`reason`、`confidence`

`PlayerAgent` 的工具 schema 和 `PlayerPromptBuilder` 的严格输出合同都从该对象读取，避免两套字段定义漂移。

### 2. StructuredOutputPolicy

增加四种显式协议：

- `native_tool`：强制工具调用
- `json_schema`：使用供应商 JSON Schema 响应格式
- `json_object`：要求 JSON 对象，但不强制完整 schema
- `text_json`：仅依赖提示词生成 JSON 文本

模型配置声明首选协议和降级协议。旧配置保持兼容：

- `allow_text_tool_fallback: false` 映射为 `native_tool`
- OpenAI 兼容模型且允许文本 fallback 时映射为 `json_schema`
- Anthropic/MiniMax 兼容模型且允许文本 fallback 时映射为 `text_json`

协议降级只发生在结构化协议失败后，不切换模型。模型级 fallback 保留现状，不在本次扩展。

### 3. 失败分流

失败阶段分为：

- `provider`：超时、网络、HTTP 或模型服务失败
- `protocol`：要求工具调用但未收到、没有 JSON 对象
- `schema`：JSON 可解析，但字段不符合 ActionContract/Pydantic
- `semantic`：动作非法、发言质量或投票质量不足

`protocol` 和 `schema` 失败可切换到下一个结构化协议；`semantic` 失败保持当前协议，只注入针对性纠错提示。

### 4. 可观测性

在 `GenerateResult`、`UsageRecord` 和 `ActionTrace` 中记录：

- `structured_output_mode`
- `structured_failure_stage`
- 现有 `structured_failure_reason`

这样游戏日志可以回答“哪个模型在哪种协议上失败”“失败在解析还是游戏语义”“协议降级后是否恢复”等问题。

## 配置

OpenAI 兼容模型采用：

```yaml
structured_output:
  mode: json_schema
  fallback_modes: [json_object, text_json]
```

MiniMax Anthropic 兼容模型采用：

```yaml
structured_output:
  mode: text_json
  fallback_modes: []
```

继续保留 `allow_text_tool_fallback`，用于旧调用方和配置兼容。

## 测试

1. ActionContract 三种模式的字段、枚举和 required 列表。
2. OpenAI 四种协议对应的请求 payload。
3. 模型配置到 StructuredOutputPolicy 的解析与旧配置兼容。
4. `PlayerAgent` 在协议/schema 失败时降级协议，在语义失败时保持协议。
5. 成功和 fallback trace 都记录协议及失败阶段。
6. 运行 agents、model_gateway、runtime/integration 定向测试及完整测试分片。
