# Autonomous Player Durable Dispatch Design

Date: 2026-07-29  
Status: Design confirmed; written review pending  
Owner: Codex development session

## 1. Goal

为自主玩家第一阶段提供独立、可恢复的外部 dispatch 持久化能力。一次
model 或 tool 请求在网络 I/O 之前先写入 `DISPATCHING`，进程重启后能够
区分可安全复用的幂等请求和只能进入 `UNKNOWN_OUTCOME` 的未知请求。

本阶段只实现 durable dispatch contract、内存/SQLite/PostgreSQL 三种仓储
和 recovery reconciler。它不接入旧 `ModelRouter`、旧 `_dispatch_agent`、
`HostRuntime`、scheduler、AgentLoop 或 feature gate。

## 2. Design Choices

采用独立 capability，而不是改造旧模型路由器或把 dispatch 状态写进
`GameEvent`：

- `UsageRecord` 是完成后的观测，不能替代网络调用前的 durable intent；
- dispatch 状态不是游戏真相，不应推进 `game_revision` 或污染事件流；
- 新 capability 可以先在 fake provider 和仓储合同测试中证明语义，之后由
  scheduler/HostRuntime 显式启用；
- 未声明 capability 的仓储永远不能被新 runtime 猜测启用。

## 3. Dispatch Contract

新模块 `werewolf_agent/player_agents/contracts/dispatch.py` 提供严格、冻结
且可 JSON 往返的模型。

### 3.1 Attempt

`DispatchAttempt` 包含：

- `dispatch_id`、`game_id`、`turn_id`、`actor_id`；
- `operation_kind`：`model` 或 `tool`；
- `executor_id`：provider 或 tool 的稳定注册名；
- `provider_idempotency_key`：由 host 生成，恢复时禁止换新；
- `recovery_policy`：`idempotent_lookup_or_reissue` 或
  `at_most_once_unknown`；
- `request_hash`、`lease_hash`、`view_fingerprint`；
- timezone-aware `deadline`、`created_at`、`updated_at`；
- `status`、`state_version`（初始为 0）和可选 `reason_code`。

`request_hash` 只表示待发送的结构化请求，不保存认证头、完整 prompt、
未过滤 provider 输出或隐藏推理。`lease_hash` 和 `view_fingerprint` 用于
防止结果跨租约、跨视图写入。

### 3.2 Result

`DispatchResultRecord` 单独保存一次已脱敏、已结构化的结果：

- `result_id`、`dispatch_id`；
- `request_hash`、`lease_hash`、`result_hash`；
- `result_kind`、`outcome`（success/failure）和有限 JSON object `payload`；
- timezone-aware `recorded_at`。

结果载荷由 host 先完成 schema/隐私清理。该 contract 不允许把原始 provider
响应当作任意字符串写入 durable store。

`record_result` 返回三种处置：

- `RECORDED`：首次合法写入；
- `REPLAYED`：相同结果重复提交；
- `DISCARDED_LATE`：attempt 已取消或已知未知，结果被丢弃并由上层审计。

不同 `request_hash`、`lease_hash` 或 `result_hash` 的重复结果返回稳定冲突
错误，不覆盖已有结果。

## 4. State Machine

```text
PENDING -> DISPATCHING -> DISPATCHED -> RESULT_RECORDED
PENDING | DISPATCHING -> CANCELLED
DISPATCHING | DISPATCHED -> UNKNOWN_OUTCOME
```

所有成功迁移都将 `state_version` 加一，并要求调用方提供
`expected_version`。非法迁移、版本不匹配、未知 dispatch ID、租约不匹配
和重复结果冲突都是稳定的 storage errors。

允许的操作：

| 操作 | 前置状态 | 结果 |
| --- | --- | --- |
| `create_dispatch` | 不存在 | 创建 `PENDING` |
| `mark_dispatching` | `PENDING` | `DISPATCHING` |
| `mark_dispatched` | `DISPATCHING` | `DISPATCHED` |
| `record_result` | `DISPATCHED` | `RESULT_RECORDED` 并写 result |
| `cancel_dispatch` | `PENDING` 或 `DISPATCHING` | `CANCELLED` |
| `mark_unknown_outcome` | `DISPATCHING` 或 `DISPATCHED` | `UNKNOWN_OUTCOME` |

`RESULT_RECORDED`、`CANCELLED`、`UNKNOWN_OUTCOME` 都是终态。终态上的迟到
provider 结果不重新打开 attempt。

## 5. Repository Capability

新模块 `werewolf_agent/storage/durable_dispatch.py` 定义：

```python
class DurableDispatchRepository(Protocol):
    def supports_durable_dispatch(self) -> bool: ...
    def create_dispatch(self, attempt: DispatchAttempt) -> DispatchAttempt: ...
    def mark_dispatching(self, dispatch_id: str, expected_version: int) -> DispatchAttempt: ...
    def mark_dispatched(self, dispatch_id: str, expected_version: int) -> DispatchAttempt: ...
    def record_result(
        self,
        dispatch_id: str,
        expected_version: int,
        result: DispatchResultRecord,
    ) -> DispatchResultDisposition: ...
    def cancel_dispatch(self, dispatch_id: str, expected_version: int, reason_code: str) -> DispatchAttempt: ...
    def mark_unknown_outcome(self, dispatch_id: str, expected_version: int, reason_code: str) -> DispatchAttempt: ...
    def load_dispatch(self, dispatch_id: str) -> DispatchAttempt | None: ...
    def list_recoverable_dispatches(self, game_id: str) -> list[DispatchAttempt]: ...
    def assert_dispatch_allowed(self, game_id: str) -> None: ...
```

`create_dispatch` 和 `assert_dispatch_allowed` 对同一游戏执行 recovery
barrier 检查。`provider_idempotency_key` 在 executor scope 内唯一，避免
两个 dispatch ID 代表同一次可能计费的外部请求。

后端语义：

- 内存仓储使用 `RLock` 和整体快照回滚，作为确定性合同实现；
- SQLite 新增独立 attempt/result 表，使用事务和唯一索引；
- PostgreSQL 使用 JSONB、`SELECT ... FOR UPDATE` 和事务级状态 CAS；
- SQLite 的 `MigrationManager` 版本号和 legacy migration SQL 不变，仓储
  初始化独立补齐 dispatch 表；
- 未实现 capability 的仓储返回 `DurableDispatchUnsupported`，不得由方法
  存在性推断支持。

## 6. Restart Reconciliation

`DispatchReconciler.reconcile_game(game_id)` 查询该游戏中所有
`DISPATCHING`/`DISPATCHED` attempt，按 `(created_at, dispatch_id)` 稳定排序。

### 6.1 Idempotent Executor

恢复 resolver 只能接收持久化的 `dispatch_id`、`request_hash`、租约信息和
原 `provider_idempotency_key`，返回以下结果：

- `FOUND`：发现已完成结果，经过绑定检查后执行 `record_result`；
- `REISSUED`：使用同一个幂等键重新交付，保持 `DISPATCHED`；
- `PENDING`：provider 仍未确认，attempt 保持未决；
- `UNAVAILABLE`：暂时无法查询，attempt 保持未决；
- `UNSAFE`：结果无法安全绑定，转 `UNKNOWN_OUTCOME`。

resolver 不得创建新的 dispatch ID 或替换 request/lease hash。

### 6.2 Non-idempotent Executor

`AT_MOST_ONCE_UNKNOWN` 不发起恢复网络请求，直接转为
`UNKNOWN_OUTCOME(reason_code="provider_not_idempotent")`，并向报告返回
`budget_consumption_required=True`。上层未来 runtime 决定如何记录预算和
是否重开 turn；本阶段不自动 fallback 或重开窗口。

### 6.3 Recovery Barrier

同一游戏只要存在 `DISPATCHING` 或 `DISPATCHED` attempt，
`assert_dispatch_allowed` 就拒绝新的 dispatch。`PENDING` 不属于恢复未决，
但本阶段没有 scheduler，因此不会自动批量创建 PENDING work。

恢复报告包含 `resolved`、`unknown`、`pending`、`errors` 数量和
`barrier_open`；只有没有未决 attempt 时 barrier 才打开。

## 7. Persistence Schema

SQLite 和 PostgreSQL 都新增两张表，字段语义一致：

```text
autonomous_dispatch_attempts(
  dispatch_id PRIMARY KEY,
  game_id, turn_id, actor_id,
  operation_kind, executor_id,
  provider_idempotency_key, recovery_policy,
  request_hash, lease_hash, view_fingerprint,
  deadline, status, state_version,
  reason_code, created_at, updated_at
)

autonomous_dispatch_results(
  result_id PRIMARY KEY,
  dispatch_id UNIQUE REFERENCES autonomous_dispatch_attempts,
  request_hash, lease_hash, result_hash,
  result_kind, outcome, result_json, recorded_at
)
```

约束和索引：

- `UNIQUE(executor_id, provider_idempotency_key)`；
- `(game_id, status, created_at)` 查询索引；
- 每个 dispatch 只允许一个 result record（无论 success 或 failure）；
- PostgreSQL JSON 字段使用 JSONB，SQLite 使用 UTF-8 canonical JSON；
- foreign key 删除随 game 清理，dispatch 不会反向改变 game revision。

## 8. Verification Plan

### Contract and Backend Tests

- 严格字段、冻结/深度不可变 JSON、Python/JSON 输入和 enum round-trip；
- 每个合法迁移以及每个非法迁移；
- `state_version` CAS 竞争只允许一个写入者成功；
- 创建重复 dispatch ID 或 executor key 返回冲突；
- 首次结果、相同结果重放、不同结果冲突和 late discard；
- 内存故障注入后所有容器无部分写入；
- SQLite 关闭后重新打开能发现未决 attempt，且 legacy `game_revision` 不变；
- PostgreSQL mock 验证 DDL、锁、事务回滚和 capability guard。

### Recovery Fault Injection

fake resolver/provider 覆盖：

1. 写入 `DISPATCHING` 后进程崩溃；
2. provider 已返回但结果尚未记录；
3. 相同幂等键查询到结果；
4. 相同幂等键安全重发；
5. provider 仍 pending 或暂时不可达；
6. provider 不支持幂等；
7. 结果在取消或 `UNKNOWN_OUTCOME` 后迟到。

### Gates

```bash
conda run -n wofkill python -m pytest -q
conda run -n wofkill python -m ruff check --ignore UP009 <changed files>
conda run -n wofkill python -m mypy --follow-imports=skip <changed implementation files>
```

本阶段完成的定义：三种仓储和 reconciler 的合同测试通过；旧全量测试
通过；无 live runtime 调用点变更；不存在未记录的 TODO、placeholder 或
未定义的恢复终态。

## 9. Explicit Non-goals

本 spec 不包括：

- `serial_public` scheduler 和 `HostRuntime` turn 生命周期；
- 旧 `ModelRouter`/`_dispatch_agent` 接入或 AgentLoop；
- workspace/context projection、compaction、tool gateway；
- outbox worker、judge presenter、feature gate 和 runtime cutover；
- 审计链加密、完整 provider billing integration、自动 fallback 或 turn 重开。

上述能力在 durable dispatch 通过合同验证后进入后续独立设计/计划。
