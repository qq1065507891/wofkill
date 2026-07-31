# Autonomous Player Context Lifecycle Design

日期：2026-08-01

状态：设计已确认；等待实施计划

父级权威：
`docs/superpowers/specs/2026-07-28-autonomous-player-agent-runtime-design.md`
§7.1～7.5、§10、§11、§25～26、§28～30

下游消费者边界：
`docs/superpowers/specs/2026-07-31-deep-agent-player-executor-design.md`
§8、§16～17

## 1. 目的

本阶段为新 autonomous player runtime 增加 Host-owned、framework-neutral 的
context lifecycle。它负责：

- 模型请求前的 context token accounting；
- 80% inclusive automatic compaction trigger；
- 唯一可恢复的严格 `CompactionCheckpoint`；
- 可缺失、可丢弃且固定标为 `UNTRUSTED_DATA` 的
  `CompactionHandoff`；
- 不高于 55% 的 deterministic rehydration；
- checkpoint parent lineage validation；
- Memory、SQLite、PostgreSQL 等价持久化；
- 进程重启后的 checkpoint recovery。

这一能力只形成未来 `PlayerCognitionExecutor` 的 Host-facing 前置边界。本阶段
不调用模型、provider、工具或 Deep Agents，不接入 `HostRuntime` 编排、
RuleEngine、`CommitTurn`、live game path 或旧 `PlayerAgent`。

## 2. 当前基线与问题

当前代码已经具有：

- 严格 `AgentTurn` 状态机，包括
  `THINKING -> COMPACTING -> THINKING`；
- Memory、SQLite、PostgreSQL 的 durable schedule 与 managed turn CAS；
- durable active-turn dispatch fence；
- viewer-specific、revision-pinned `ObservationBundle`；
- deterministic player workspace projections。

当前仍没有 Host context budget、checkpoint authority、lineage、rehydration 或
restart recovery。Deep Agents/LangGraph 自带 summarization 与 checkpoint 不能
替代这些能力，因为 framework state 不能证明玩家视图、合法窗口、source hashes、
remaining budgets 或 durable dispatch identity 仍然有效。

## 3. 范围

### 3.1 本阶段实现

- `ContextBudgetPolicy` 与不可伪造的 `ContextTokenAccounting`；
- admission fit check、80% trigger 和 55% rehydration target；
- strict frozen checkpoint、handoff 与 rehydration contracts；
- checkpoint build、self-hash、authority、deadline 和 lineage validation；
- 独立 `ContextCheckpointRepository` capability；
- Memory、SQLite、PostgreSQL repository 实现；
- checkpoint insert 与 `COMPACTING -> THINKING` 的原子事务；
- SQLite reopen、PostgreSQL reconnect 与 process restart recovery；
- 显式 recovery observation rebuild API；
- 真实 PostgreSQL service integration tests。

### 3.2 明确不实现

- ToolGateway、ToolResult 或 Markdown presentation；
- model lease 创建、provider tokenizer 调用或真实 model request；
- AgentLoop、`PlayerCognitionExecutor` 或 Deep Agents adapter；
- model-generated handoff 调用；本阶段只定义 handoff contract 与失败语义；
- grants、beliefs、new-runtime memory、working reflection 或 tool-result stores；
- 完整 audit hash chain；
- proposal validation、RuleEngine、`CommitTurn`、feature gate 或 live path；
- 物理 player workspace 或 durable observation projection table。

不存在 authority store 的数据不会以空泛 `dict`、兼容层或伪造记录提前建模。

## 4. 方案选择

### 4.1 采用：独立 checkpoint capability 与 Host-facing service

新增独立 `ContextCheckpointRepository` protocol，三后端在同一 repository
对象上实现该 capability。`ContextLifecycleService` 负责协调 checkpoint commit
与 recovery，但不拥有模型、工具或游戏真理。

采用该方案的原因：

- 调度、context authority 与未来 cognition execution 各自保持单一职责；
- checkpoint history 与 parent lineage 不膨胀 `ManagedAgentTurn` JSON；
- 三后端可以共享纯事务准备逻辑和 conformance tests；
- 未来 executor 只消费 Host 已验证的 rehydration plan；
- 当前不需要修改 `HostRuntime` 编排或扩大其公开 API。

### 4.2 拒绝：扩展 `AutonomousTurnRepository`

该方案会把 schedule、managed-turn lifecycle、context persistence 和 recovery
合并为逐步膨胀的宽接口，降低内聚性，也使后续 model/tool gateway 更容易继续
向同一 protocol 堆叠职责。

### 4.3 拒绝：checkpoint 内嵌 `ManagedAgentTurn`

该方案需要每次 compaction 重写完整 turn JSON，并使多次 checkpoint、parent
lineage、唯一约束和独立恢复查询依附在 lifecycle snapshot 内，难以表达明确的
历史与并发语义。

## 5. 模块边界

```text
werewolf_agent/player_agents/context/
  __init__.py
  contracts.py      strict policy/checkpoint/handoff/rehydration contracts
  errors.py         stable sanitized failures
  accounting.py     pure token accounting and threshold decisions
  lifecycle.py      pure checkpoint build/authority/lineage/rehydration logic
  service.py        Host-facing commit and recovery coordination

werewolf_agent/storage/context_checkpoints.py
  ContextCheckpointRepository protocol
  repository capability guard
  backend-neutral transaction preparation
```

`player_agents/context` 可以依赖新运行时 contracts 与 observation contracts，
但不得导入 Deep Agents、旧 agents、旧 tools schemas、provider、RuleEngine、
`CommitTurn` 或 live runtime。

`ContextLifecycleService` 只接受同时具有 `AutonomousTurnRepository` 与
`ContextCheckpointRepository` capability 的同一仓储对象。它不接受拆分仓储，
避免 turn CAS 与 checkpoint persistence 落在不同事务边界。

## 6. Context budget contracts

### 6.1 `ContextBudgetPolicy`

固定字段：

```text
model_context_limit_tokens
reserved_output_tokens
reserved_tool_schema_tokens
estimator_version
tokenizer_mode
conservative_safety_margin = 0.10
auto_compact_threshold = 0.80
post_compact_target = 0.55
max_compactions_per_turn = 2
```

provider tokenizer 模式不增加 safety margin。conservative estimator 模式对
input components 的估算总和增加 10% margin。policy 必须证明 reserves 小于
model context limit，且 post-compaction target 小于 automatic threshold。

### 6.2 `ContextTokenAccounting`

一次 accounting 至少记录：

```text
immutable_prefix_tokens
active_history_tokens
exposed_tool_schema_tokens
reserved_output_tokens
safety_margin_tokens
total_active_tokens
model_context_limit_tokens
occupancy_ratio
estimator_version
tokenizer_mode
```

构造时复算 `safety_margin_tokens`、`total_active_tokens` 和
`occupancy_ratio`，拒绝调用方提供不一致派生值。

准入规则：immutable prefix、完整 tool-schema reserve 与 output reserve 在当前
tokenizer policy 下必须严格低于 55%。

模型请求规则：预测 occupancy 大于或等于 80% 时返回
`context_compaction_required`，不得开始下一次模型 dispatch。

## 7. Checkpoint 与 handoff contracts

### 7.1 `CompactionCheckpoint` v1

checkpoint 至少绑定：

- schema version、checkpoint ID、ordinal；
- parent checkpoint ID 与 parent checkpoint hash；
- game、player、schedule、turn 和 window identity；
- observation turn version、进入 `COMPACTING` 的 turn CAS version，以及
  commit 后恢复到 `THINKING` 的 expected version；
- task type、base game revision、view fingerprint、model lease hash；
- original workspace revision/hash、observation frame hash、source-history
  hash 与 event cursor；
- legal action/target snapshots；
- remaining `TurnBudget` 与 remaining compactions；
- deadline；
- confirmed `ReadReference` 与 observation
  `ProjectionSourceReference`；
- 当前已实现的 public-record/reference IDs；
- bounded `IntendedOperation`；
- pre-compaction accounting；
- checkpoint-only rehydration receipt/hash；
- created/persisted timestamps；
- checkpoint self-hash。

checkpoint 不保存 hidden chain-of-thought、reasoning tokens、完整 prompt、provider
原始响应、Markdown 事实副本、凭据或跨玩家私密内容。

checkpoint self-hash 覆盖除 self-hash 字段外的 canonical JSON。任何字段变化都
使校验失败。

### 7.2 `CompactionHandoff`

handoff 只允许：

```text
trust_class = UNTRUSTED_DATA
current_progress
decisions_made
important_constraints
remaining_steps
critical_reference_ids
estimated_tokens <= 2000
```

所有 reference ID 必须已经存在于 checkpoint 的 authorized reference set。
handoff 缺失、生成失败、schema 非法、引用越权或超过 rehydration target 时均等同
没有 handoff，不得阻止 checkpoint-only continuation。

handoff 文本永远不会被解析回 fact、evidence、grant、legal target、memory、tool
result 或 proposal。

## 8. Rehydration contract

`RehydrationPlan` 按固定顺序引用：

1. immutable system contract；
2. 当前 `AgentTurn`；
3. 最新最小 `ObservationFrame`；
4. 当前 document manifest；
5. Host `CompactionCheckpoint`；
6. 可选 `CompactionHandoff`。

plan 只保存 ordered component hashes、token accounting、handoff inclusion flag 与
final context hash，不保存模型 message 副本。

service 在 checkpoint commit 前构建 checkpoint-only rehydration candidate，保证
canonical authority 加 schema/output reserves 后不高于 55%。然后尝试加入合法
handoff；加入后超标则丢弃 handoff。

当前阶段没有 ToolGateway，因此不存在 adopted tool summary 的缩减步骤。未来
ToolGateway milestone 可以在新 schema version 中先把 adopted summaries 替换为
typed references，但不得改变 checkpoint authority。

如果 checkpoint-only candidate 仍高于 55%，返回
`rehydration_target_exceeded`，不提交 checkpoint，也不授权模型请求。

## 9. Compaction transaction

调用顺序：

```text
THINKING
  -> predict occupancy
  -> occupancy >= 80%
  -> managed-turn CAS: THINKING -> COMPACTING
  -> build exact checkpoint from current Host authority
  -> validate optional untrusted handoff
  -> build <= 55% rehydration candidate
  -> repository atomic checkpoint commit
       validate active schedule/turn/deadline/lineage
       insert checkpoint
       managed-turn CAS: COMPACTING -> THINKING
  -> expose deterministic RehydrationPlan
```

repository transaction 必须验证：

- schedule 存在、状态为 open，且仍是该 game 的活动 schedule；
- schedule version 精确匹配；
- active turn ID、schedule ID、game ID、player ID、window identity 匹配；
- managed turn 状态为 `COMPACTING` 且 version 精确匹配；
- revision、view fingerprint、lease、legal snapshots 和 deadline 匹配；
- deadline 尚未到期；
- ordinal 1 没有 parent；
- ordinal N 的 parent 是同一 turn 最新 checkpoint，ID/hash 匹配且 ordinal
  恰好加一；
- checkpoint ID 全局唯一，`(turn_id, ordinal)` 唯一。

checkpoint insert 与 `COMPACTING -> THINKING` CAS 必须全有或全无。

后端锁边界：

- Memory：repository 的同一 `RLock`；
- SQLite：`BEGIN IMMEDIATE`；
- PostgreSQL：先取得现有 game advisory transaction lock，再按固定顺序锁
  schedule、managed turn 和该 turn 最新 checkpoint lineage row。

不同 Host 操作不得采用相反锁序。不得使用进程内锁模拟跨进程 PostgreSQL/SQLite
事务语义。

## 10. Restart recovery

restart recovery 接受当前 active schedule/turn 和从当前 Host authority 重建的
`ObservationBundle`，然后：

1. 加载该 turn 的完整 checkpoint lineage；
2. 逐项验证 schema、自哈希、parent ID/hash 与连续 ordinal；
3. 验证当前 schedule、turn、player、window、revision、view、lease 和 deadline；
4. 要求 managed turn 处于 checkpoint commit 后的 `THINKING` version；
5. 拒绝任何增加 remaining budget 或 remaining compactions 的恢复；
6. 验证 legal action/target snapshots；
7. 验证当前 observation source IDs、revisions 和 content hashes；
8. 重新构建不高于 55% 的 deterministic rehydration plan。

`ObservationProjectionService` 增加显式 recovery rebuild API，允许为
`THINKING` turn 重建 observation。普通 initial-observation API 仍只接受原有状态，
不能借此放宽 admission。

managed-turn CAS 会改变 projection identity version，因此 recovery 不要求 original
与 rebuilt workspace hash 相等。checkpoint 保留 original workspace/frame hash 供
integrity 与审计使用；恢复授权比较稳定 authority lineage：viewer、window、revision、
view fingerprint、legal snapshots，以及全部 source ID/revision/content hash。

recovery 不重跑 handoff，不解释 handoff 文本，不重放 framework checkpoint，也不
自动开始 model/tool dispatch。

## 11. Stable failures

公开稳定错误码：

```text
context_lifecycle_unsupported
context_admission_rejected
context_compaction_required
checkpoint_integrity_failed
checkpoint_authority_conflict
checkpoint_lineage_conflict
checkpoint_deadline_expired
checkpoint_transaction_failed
rehydration_target_exceeded
checkpoint_recovery_rejected
```

错误文本不得携带 raw checkpoint JSON、SQL、private source text、handoff 文本、
credentials 或 provider output。后端异常在 repository boundary 映射为稳定错误，
同时保留 Python exception chaining 供受控日志诊断。

## 12. Persistence schemas

SQLite 与 PostgreSQL 新增 `autonomous_compaction_checkpoints`：

```text
checkpoint_id          primary key
game_id                not null
schedule_id            not null
turn_id                not null
player_id              not null
ordinal                not null
parent_checkpoint_id   nullable
checkpoint_hash        not null
checkpoint_json        not null
created_at              not null
persisted_at            not null
unique (turn_id, ordinal)
```

索引支持按 `(turn_id, ordinal desc)` 读取完整或最新 lineage。foreign keys 绑定
game、schedule 与 managed turn。schema 初始化必须先检测历史重复或不一致数据，
不能依赖裸 unique-index failure 作为迁移策略。

Memory 使用 checkpoint ID map、per-turn ordered lineage 和防御性深拷贝。

## 13. 测试策略

### 13.1 Contracts 与 accounting

- strict、frozen、extra-forbid 与 defensive collection freeze；
- 默认 80% / 55% / 2 次；
- 80% inclusive trigger；
- conservative estimator 10% margin；
- provider tokenizer 无额外 margin；
- derived accounting 防伪；
- admission 必须严格低于 55%；
- handoff 2,000-token bound 与 authorized-reference subset。

### 13.2 Lifecycle 与 recovery

- checkpoint self-hash tampering；
- stale game/player/turn/window/revision/view/lease/source/deadline；
- parent ID/hash/ordinal 连续性；
- handoff 缺失、非法与超预算时 checkpoint-only continuation；
- checkpoint-only context 超过 55% 时 fail closed；
- rebuilt observation source lineage validation；
- remaining budgets 不得增加；
- recovery 不触发任何 external dispatch。

### 13.3 Shared backend conformance

Memory、SQLite、PostgreSQL 对相同命令序列必须产生等价结果：

- successful atomic commit；
- stale schedule/turn CAS 零部分写；
- terminal/non-`COMPACTING` turn 拒绝；
- duplicate checkpoint ID；
- duplicate `(turn_id, ordinal)`；
- orphan、forked 和 skipped ordinal；
- 同一 ordinal 并发提交只有一个成功；
- defensive returned copies；
- SQLite close/reopen 与 PostgreSQL reconnect recovery。

### 13.4 PostgreSQL tests

fake-connection tests 覆盖精确 schema、lock order、CAS、rollback 与数据库异常映射。

真实 PostgreSQL integration tests 使用隔离测试数据库验证：

- schema 创建与约束；
- advisory transaction lock；
- checkpoint insert 与 turn CAS 的原子性；
- stale lineage rollback；
- 两连接并发竞争；
- reconnect 后 latest/full lineage recovery。

真实服务测试是本里程碑验收的一部分，不由 fake tests 替代。

### 13.5 Boundary 与回归

- context 包不得导入 Deep Agents、旧 agents、旧 tools、provider、RuleEngine、
  `CommitTurn` 或 live runtime；
- ordinary observation API 仍拒绝 `THINKING`；
- explicit recovery observation API 只允许恢复所需状态；
- existing active-turn fence、observation、autonomous-turn 和 CommitTurn tests
  保持绿色。

## 14. 验证命令

项目命令全部通过 `conda run -n wofkill` 执行。实施计划将把以下门槛拆成先 RED
后 GREEN 的窄测试：

```bash
conda run -n wofkill python -m pytest \
  tests/player_agents/test_context_contracts.py \
  tests/player_agents/test_context_lifecycle.py \
  tests/player_agents/test_context_checkpoint_conformance.py \
  tests/player_agents/test_context_import_boundary.py \
  tests/storage/test_context_checkpoints.py -o addopts='' -q
conda run -n wofkill python -m pytest \
  tests/storage/test_postgres_context_checkpoints.py -o addopts='' -q
conda run -n wofkill python -m ruff check --ignore UP009 \
  werewolf_agent/player_agents/context \
  werewolf_agent/player_agents/observation/service.py \
  werewolf_agent/storage/context_checkpoints.py \
  tests/player_agents/test_context_contracts.py \
  tests/player_agents/test_context_lifecycle.py \
  tests/player_agents/test_context_checkpoint_conformance.py \
  tests/player_agents/test_context_import_boundary.py \
  tests/storage/test_context_checkpoints.py \
  tests/storage/test_postgres_context_checkpoints.py
conda run -n wofkill python -m mypy --follow-imports=skip \
  werewolf_agent/player_agents/context \
  werewolf_agent/player_agents/observation/service.py \
  werewolf_agent/storage/context_checkpoints.py
git diff --check
conda run -n wofkill python -m pytest -q
```

full suite 通过不能证明 playable vertical slice；本阶段完成后仍没有模型、工具、
proposal validation、RuleEngine/`CommitTurn` 编排或 live-game integration。

## 15. 最终不变量

1. Host checkpoint 是唯一 resumable context authority。
2. Model handoff 始终是可缺失、可丢弃的 untrusted data。
3. Markdown、model prose 和 framework state 永远不能恢复事实或权限。
4. 80% trigger 在下一次模型 request 之前执行。
5. rehydrated context 不高于 55%。
6. checkpoint persistence 与 `COMPACTING -> THINKING` 在同一事务中完成。
7. stale、terminal、expired 或 lineage-invalid turn 不能发布 checkpoint。
8. recovery 不能增加预算、改变 viewer 或恢复不可见 source。
9. Memory、SQLite、PostgreSQL 对同一命令序列具有等价语义。
10. context lifecycle 不推进 `GameRevision`，不写 `GameState`，不替代
    `CommitTurn`。
11. 本阶段不安装、导入或调用 Deep Agents。
12. 本阶段不接入 ToolGateway、provider、AgentLoop、旧玩家或 live path。
