# Autonomous Player Commit Transaction Design

**Status:** Approved for implementation

**Goal:** 为自主玩家第一条 daytime-speech vertical slice 提供一个独立、可
比较版本且可回滚的 `CommitTurn` 持久化能力，同时不改变旧游戏运行路径。

## 1. 范围

本阶段只实现仓储能力，不接入 HostRuntime、scheduler、model dispatch、
workspace 或 feature gate。成功标准是三个受支持的测试仓储都能提供相同的
提交语义：内存仓储、SQLite 和 PostgreSQL。没有该能力的仓储显式报告不支持，
不能被未来的自主运行时启用。

本阶段包含：

- per-game `game_revision`，初始值为 0，每次成功提交严格加 1；
- `CommitTurn` compare-and-swap，串行窗口要求 `base_revision` 等于当前 head；
- 以 `(game_id, turn_id, idempotency_key)` 唯一识别成功提交，重复请求返回原结果；
- 幂等键复用但请求内容不同返回稳定的 `idempotency_conflict`；
- GameEvent、公共语义记录、关键审计载荷、幂等结果和 projection outbox 的原子提交；
- 提交失败时所有上述写入回滚；outbox 消费者不在本阶段实现，但 outbox ID 必须唯一。

不包含：审计加密、外部 anchor、dispatch attempt、scheduler、投影 worker、
真实 speech agent loop、旧 API 切换，以及 PostgreSQL 集成环境的端到端运行。

## 2. 设计

### 2.1 独立能力接口

新增 `werewolf_agent/storage/autonomous_commit.py`，定义只依赖核心
`GameEvent` 和新的 player-agent contract 的协议：

```python
class AutonomousCommitRepository(Protocol):
    def supports_autonomous_commit(self) -> bool: ...
    def commit_turn(self, request: CommitTurnRequest) -> CommitResult: ...
    def load_game_revision(self, game_id: str) -> int: ...
    def load_outbox(self, game_id: str) -> list[ProjectionOutboxRecord]: ...
```

`GameRepository` 的旧方法保持不变；新能力作为可选 capability 检查，不通过
鸭子类型推断。未来 runtime 先检查 capability，再调用 `commit_turn`。

### 2.2 事务输入与结果

`CommitTurnRequest` 是严格、不可变的模型，字段包括：

- `game_id`、`turn_id`、`idempotency_key`、`base_game_revision`；
- `read_set`（`ReadReference` 元组）；
- 已通过 host 绑定的 `SpeechProposalEnvelope`；
- 待追加的无身份 `GameEvent`；
- 可选的 `PublicSpeechRecord`；
- 关键审计项和 projection-outbox 项元组。

仓储负责为事件分配 `sequence_number = game_revision` 对应的下一序号，并
通过现有 V2 event metadata 规则补齐 event ID、game ID、时间和可见性。结果
`CommitResult` 返回提交 revision、event ID、outbox IDs 和 `replayed` 标记。

`rule_result` 在本阶段以不可变 JSON 对象随 request 保存并参与幂等指纹，
但 RuleEngine 校验仍由未来 HostRuntime 完成；仓储只负责事务边界和版本比较。

### 2.3 SQLite 结构

在已有 `_SCHEMA` 和 legacy 初始化路径中追加幂等 `CREATE TABLE IF NOT EXISTS`，
新增：

- `autonomous_game_streams(game_id PRIMARY KEY, game_revision)`；
- `autonomous_turn_commits(game_id, turn_id, idempotency_key, request_hash,
  result_json, committed_revision, UNIQUE(game_id, turn_id, idempotency_key))`；
- `autonomous_public_records(record_id PRIMARY KEY, game_id, turn_id,
  committed_revision, record_json)`；
- `autonomous_audit_records(game_id, committed_revision, audit_id PRIMARY KEY,
  record_json)`；
- `autonomous_projection_outbox(outbox_id PRIMARY KEY, game_id,
  committed_revision, request_json, delivered_at)`。

建立 `(game_id, committed_revision)` 索引，所有提交在同一 SQLite transaction
中完成。现有 MigrationManager 的版本号和旧迁移语义保持不变；仓储初始化
会为旧数据库补齐这些表。已有 `games/events` 数据不会自动填充自主 stream；
第一次自主提交以该游戏事件表的最大已提交 `seq`（无事件时为 0）作为 revision
基线并写入 stream，之后只允许通过新事务前进。

### 2.4 内存与 PostgreSQL

内存仓储使用同一组记录结构和 `RLock`，在锁内执行完整 compare-and-swap，
并在异常时通过临时副本保证无部分写入。PostgreSQL 使用现有 connection
和 game advisory transaction lock，表结构与 SQLite 语义相同；当 psycopg
不可用或未初始化时只暴露 capability 方法，不执行连接。

### 2.5 错误与回滚

定义存储层异常：

- `AutonomousCommitUnsupported`：仓储未实现 capability；
- `StaleCommitError`：`base_game_revision` 不是当前 head；
- `IdempotencyConflictError`：同一幂等键对应不同 request hash；
- `CommitTransactionError`：事务失败，原始异常通过 `__cause__` 保留。

重复提交只有在 request hash 完全一致时返回原 `CommitResult`；重复提交不会
再次追加事件、审计或 outbox。任何验证、序列化或唯一性错误都必须让事件、
stream revision、审计、幂等和 outbox 同时保持提交前状态。

## 3. 验证策略

- 先为协议、错误、哈希和回滚写失败测试；
- 使用同一参数化行为套件覆盖内存和 SQLite；
- PostgreSQL 增加无连接时的 capability/schema 单元测试，并保留可选集成测试入口；
- 注入事件、审计和 outbox 写入失败，验证无 truth change；
- 50 个并发重复提交只产生一个 revision、一个事件、一个审计批次和一组 outbox；
- 保留现有完整仓储、事件和 runtime 回归测试。

## 4. 后续边界

本阶段不创建任何 live runtime 调用点。下一阶段在此能力之上实现 durable
dispatch、`serial_public` scheduler 和 restart reconciliation；再下一阶段
才生成 player workspace projection。
