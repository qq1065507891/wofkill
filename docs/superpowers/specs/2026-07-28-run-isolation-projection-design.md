# Run Isolation and Projection Reliability Design

## 目标

保证相同 seed 的多次运行具有不同持久化身份和独立日志，并让正常完整局能够生成全部验收投影与质量指标。

## 已确认问题

默认 `game_id` 为 `g_<seed>`，重复 seed 会与已有终局冲突。多个进程写同一个 `game_stdout.log`，且日志行不包含 game ID、run ID 或 PID，导致不同对局内容交叉。数据库终局不可变保护正常生效，但同名后续运行只能留下 emergency artifact，数据库和 JSON 文件代表不同运行。

完整局包含约 1400 至 1600 个富审计事件。当前投影对整个游戏共享 10,000 JSON item 预算，两个完成局都因 `json_item_limit_exceeded` 失去 events 和 fallback 指标。

## 设计

### 身份模型

- `seed` 只控制确定性随机过程。
- `game_id` 表示对局业务身份。
- `run_id` 表示一次具体执行尝试。

默认 `game_id` 和 `run_id` 必须包含不可碰撞的运行成分；显式传入的 `game_id` 保持可读，但启动时必须检查 repository。已有 terminal `game_id` 默认拒绝启动，除非调用方显式选择新的 ID。不得提供隐式覆盖终局的开关。

每个事件、action trace、reflection decision 和 emergency artifact 都应能关联 `game_id` 与 `run_id`。

### 日志隔离

每次运行使用独立日志文件，文件名包含安全化的 `game_id` 与 `run_id`。日志格式至少包含时间、级别、PID、game ID、run ID、logger 和消息。

聚合 stdout 可以保留，但不能作为唯一证据源。所有终局报告必须引用本次运行的独立日志和 artifact 路径。

### 启动预检

Runner 在模型调用和角色分配之前完成：

- game/run ID 合法性校验；
- terminal game ID 冲突检测；
- artifact 路径固定；
- repository 可读性检查。

预检失败不进入图执行，也不写新的对局事件。

### 投影预算

安全限制改为分层预算：单值深度、单事件项目数、事件数量和总体序列化大小分别限制。一个超限事件应被明确标记或安全裁剪，不能让其他 1500 个合法事件全部变成 unsupported。

验收所需的关键事件类型必须完整保留，包括 action trace、terminal fallback、reflection complete 和 reflection persistence audit。所有裁剪都记录原因、事件序号和类型。

### 指标合同

完成局只有在 events 投影可用时才能声明质量验收完成。必须能计算：

- 公开事实正确性指标；
- provider/repair/fallback 指标；
- 反思生成、核验、持久化与恢复 12/12；
- 终局状态与 artifact 一致性。

## 错误处理和迁移

已有 `g_<seed>` 数据保持只读兼容，不进行自动重命名。新运行采用新身份合同。旧日志缺少 run ID 时报告为 legacy/ambiguous，不尝试自动猜测归属。

超限投影必须 fail closed 并保留结构化失败原因，但不能静默清空全部事件。

## 测试与验收

- 相同 seed 连续启动两局，断言 game/run ID 和日志文件不同，但确定性角色分配仍一致。
- 已有 terminal game ID 的启动预检必须在任何模型调用前失败。
- 并发运行测试断言日志和 artifacts 不交叉。
- 使用 1600 个富审计事件构造投影测试，确保关键指标可计算。
- 单个恶意深层事件只影响自身，不能使整局 events unsupported。
- 两局真实模型运行均使用独立日志、唯一 run ID，并成功生成完整验收报告。
