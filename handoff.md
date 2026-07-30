# Autonomous Player Runtime 跨会话接力手册

更新日期：2026-07-31

## 1. 这份文件怎么用

这是 autonomous player runtime 重写工作的唯一跨会话接力入口。新会话读取本文件后，应当能够直接回答：

1. 当前实现到了哪里；
2. 哪些功能只是合约或基础设施，尚未形成真实运行链路；
3. 下一项工作是什么，为什么必须先做它；
4. 开始工作前必须读取哪些设计、代码和测试；
5. 哪些架构边界、并发约束和验证要求不能破坏。

本文件是进度快照和执行导航，不替代权威设计。发生冲突时使用以下优先级：

1. `docs/superpowers/specs/2026-07-28-autonomous-player-agent-runtime-design.md` 是总体架构与最终约束的唯一权威；
2. 当前源码和测试决定“实际上已经实现了什么”；
3. 后续专题设计补充某一阶段的详细契约，但不能覆盖总体设计；
4. 本文件记录当前进度、阅读顺序和下一步；
5. implementation plan 是实施历史，不是新的架构权威。

如果本文件中的分支、HEAD、测试数量或远端领先数量与仓库不一致，先运行第 3 节的检查命令，再更新本文件。不要把快照当作永久事实。

## 2. 一句话进度结论

新运行时已经完成“严格合约、三后端原子提交、durable dispatch、`serial_public` 调度和 HostRuntime 生命周期”基础层，但尚未接入真实游戏、模型、工具、RuleEngine 提交流程或旧玩家运行入口。

按总体设计第 28 节的 13 步实施序列判断：

- 第 1 步已完成；
- 第 2 步已完成 stage-1 昼间发言所需的主要合约，但其他终端提案仍未实现；
- 第 3 步已完成 Memory、SQLite、PostgreSQL 的基础事务能力；
- 第 4 步部分完成：调度、Host 生命周期、取消、过期和重启协调已完成，生产 dispatcher 与 durable active-turn fence 未完成；
- 第 7 步中的 `SpeechProposal`、`PublicSpeechRecord` 和原子 commit 合约被提前实现，但完整纵向链路没有闭环；
- 第 5～13 步的主体尚未开始。

因此，当前不能称为“第一阶段完成”，也不能通过新运行时玩完一次真实昼间发言回合。

## 3. 当前仓库快照与首次检查

记录本文件时：

- 分支：`master`
- 本文件内容基线：`5d9b7ed`（`docs: design tool result markdown projections`）
- 提交本文件前相对 `origin/master`：ahead 100；实际值必须用下方命令刷新
- 提交本文件前除 `handoff.md` 外没有未提交变更
- 新运行时聚焦测试：306 passed
- 当前全量 pytest：退出码 0；只有既有 skips 和 10 条第三方弃用 warning

新会话不要直接相信以上动态值。先在仓库根目录执行：

```bash
git status --short --branch
git log --oneline -12
conda run -n wofkill python -m pytest tests/player_agents \
  tests/storage/test_autonomous_commit.py \
  tests/storage/test_autonomous_turns.py \
  tests/storage/test_durable_dispatch_protocol.py \
  tests/storage/test_postgres_autonomous_commit.py -q
```

仅在准备合并、交付或声称仓库级回归通过时，再运行：

```bash
conda run -n wofkill python -m pytest -q
```

项目所有 Python、pytest、ruff 和 mypy 命令都必须通过 `conda run -n wofkill` 执行。

## 4. 已经实现的功能

### 4.1 新运行时边界和严格合约

- `werewolf_agent/player_agents/` 已建立为独立的新运行时包。
- import-boundary 测试禁止新包导入旧 `PlayerAgent`、旧 prompt/model gateway、旧 strategy/directive 和被否决的 `SpeechAct` 实现。
- 已定义 `GameRevision` 上下文、read references、`LegalActionWindow`、`AgentTurn`、turn budget 和完整生命周期状态机。
- 已定义 `serial_public` schedule、slot、admission、managed turn、terminal disposition 和版本化 CAS 身份。
- 所有核心合约使用严格、冻结、禁止额外字段的 Pydantic 模型；copy/update 会重新执行完整验证。
- 已定义稳定 proposal error code、安全消息目录和严格的 repairable 范围。

主要文件：

- `werewolf_agent/player_agents/contracts/revisions.py`
- `werewolf_agent/player_agents/contracts/turns.py`
- `werewolf_agent/player_agents/contracts/scheduling.py`
- `werewolf_agent/player_agents/contracts/errors.py`
- `tests/player_agents/test_revision_contracts.py`
- `tests/player_agents/test_turn_contracts.py`
- `tests/player_agents/test_scheduling_contracts.py`
- `tests/player_agents/test_import_boundary.py`
- `tests/player_agents/test_runtime_import_boundary.py`

### 4.2 Stage-1 昼间发言数据契约

- 已实现独立于旧 `SpeechAct` 的 `SpeechProposalEnvelope` 和 `SpeechProposalBody`。
- 已实现 11 类 typed public moves、引用约束、循环检测、唯一 move ID、delivery plan 和 actor self-reference 约束。
- 已固定 schema version、生成 canonical JSON Schema，并用 checked-in fixture 与 hash 锁定。
- 已实现 `DisclosureGrant`、`PublicSpeechRecord`、`RenderedUtterance` 和 record origin。
- 已在合约层区分私有提案、公开语义记录和最终展示文本。

主要文件：

- `werewolf_agent/player_agents/contracts/speech.py`
- `werewolf_agent/player_agents/contracts/proposals.py`
- `werewolf_agent/player_agents/contracts/disclosure.py`
- `werewolf_agent/player_agents/contracts/records.py`
- `werewolf_agent/player_agents/contracts/schema_catalog.py`
- `tests/fixtures/player_agents/speech_proposal_schema_v1.json`
- `tests/player_agents/test_speech_contracts.py`
- `tests/player_agents/test_public_record_contracts.py`
- `tests/player_agents/test_schema_catalog.py`

### 4.3 原子 `CommitTurn`

- 已定义 `CommitTurnRequest`、`CommitResult`、event candidate、critical audit payload 和 projection outbox payload。
- Memory、SQLite、PostgreSQL 后端均提供显式 capability flag。
- 当前串行提交语义支持 canonical revision CAS、幂等重放、idempotency conflict、事件序列分配、公开记录 revision 绑定，以及 event/audit/idempotency/outbox 的全有或全无提交。
- 已覆盖重复提交、并发竞争、旧事件 revision 衔接、冲突回滚和跨后端一致性。

主要文件：

- `werewolf_agent/player_agents/contracts/transactions.py`
- `werewolf_agent/storage/autonomous_commit.py`
- `werewolf_agent/storage/memory_store.py`
- `werewolf_agent/storage/sqlite_store.py`
- `werewolf_agent/storage/postgres_store.py`
- `tests/storage/test_autonomous_commit.py`
- `tests/storage/test_postgres_autonomous_commit.py`

### 4.4 Durable dispatch

- 已定义 model/tool dispatch attempt、result record、recovery policy 和严格状态机。
- Memory、SQLite、PostgreSQL 均支持 create、CAS transition、result recording、late-result discard、unknown outcome、cancel、recovery barrier 和按 turn 查询。
- `DispatchReconciler` 能按创建顺序恢复 unresolved attempts；不支持幂等查询的 provider 会进入显式 `UNKNOWN_OUTCOME`，不会用新 ID 掩盖未知计费结果。
- 恢复报告会阻止新模型工作，直到 earlier attempts 都进入可接受的终态。

主要文件：

- `werewolf_agent/player_agents/contracts/dispatch.py`
- `werewolf_agent/storage/durable_dispatch.py`
- `tests/player_agents/test_dispatch_contracts.py`
- `tests/storage/test_durable_dispatch_protocol.py`

### 4.5 `serial_public` 调度与 HostRuntime 基础设施

- Memory、SQLite、PostgreSQL 都能持久化 schedule 和 managed turn。
- 支持 schedule 创建、当前 slot 准入、非终态迁移、advance、replace、close、活动回合读取和跨进程恢复。
- 新 schedule 必须从 `open / slot 0 / no active turn / version 0` 开始。
- 同一 schedule 的 replacement 必须使用新的 turn ID 和未使用过的 idempotency key。
- `HostRuntime` 覆盖 create、recover、admit、transition、complete、cancel、expire 和 load active turn。
- cancel/expire 会先处理可取消的 `PENDING`、`DISPATCHING` attempts，再原子终结 managed turn。

主要文件：

- `werewolf_agent/storage/autonomous_turns.py`
- `werewolf_agent/player_agents/runtime/serial_public.py`
- `werewolf_agent/player_agents/runtime/host.py`
- `tests/storage/test_autonomous_turns.py`
- `tests/player_agents/test_host_runtime.py`

## 5. 只有部分实现，不能误判为完成

### 5.1 HostRuntime 还不是完整 Host

当前 HostRuntime 只协调持久化调度、恢复门禁、回合生命周期、取消和过期。它还不会：

- 构建 observation；
- 建立 player workspace；
- 调用模型或工具；
- 管理真实 model lease 和 context budget；
- 验证 speech proposal 的可见性与语义；
- 调用 RuleEngine；
- 调用 `CommitTurn`；
- 运行 projection worker；
- 渲染玩家发言或裁判播报。

`complete_active_turn()` 只是“外部权威提交已经成功”之后的生命周期通知，绝不能替代 `CommitTurn`。

### 5.2 Speech 只有合约，没有生产链路

严格 proposal、schema fixture 和 public record 已存在，但尚无真正的：

- terminal submission gateway；
- bound-context 注入与验证；
- evidence/read-set/visibility/grant 解析；
- task-specific semantic validator；
- RuleEngine legality adapter；
- deterministic player renderer；
- commitment projection；
- personality-aware judge presenter。

### 5.3 Audit 只有事务载荷，不是完整审计子系统

当前 `CriticalAuditRecord` 只是随 truth 原子落盘的 `audit_id/kind/payload`。尚未实现总体设计要求的：

- per-game audit sequence；
- RFC 8785 canonicalization；
- previous hash / record hash 链；
- private payload AEAD envelope encryption；
- key provider、轮换和 crypto-shredding；
- access audit、retention、redaction manifest；
- external anchor sink 和 finalization；
- audit verifier、queries 和 evaluation projection。

### 5.4 PostgreSQL 还缺真实服务验证

当前 PostgreSQL 测试覆盖 schema、fake connection、锁、CAS、事务和异常映射，但没有在真实 PostgreSQL 服务上运行集成测试。接入生产前必须补齐。

## 6. 尚未实现的主体功能

- `ObservationFrame` 和 ObservationProjector；
- 每玩家隔离的 `PLAYER.md`、`ROLE.md`、`GAME.md`、`BELIEFS.md`、`COMMITMENTS.md`、`MEMORY.md`、`WORKING.md`、`INDEX.md` 投影；
- context token estimator、80% 自动压缩触发、55% rehydration target、checkpoint lineage；
- model lease、预算、failover 和真实 provider adapter；
- stage-1 ToolGateway、context/evidence 工具和 working reflection candidate；
- 已完成设计但尚未实现的专用 ToolResult Markdown 模型展示层，设计见
  `docs/superpowers/specs/2026-07-31-tool-result-markdown-projection-design.md`；
- 真正的 AgentLoop；
- 完整 proposal validator、visibility policy 和 disclosure grant 消费；
- deterministic player renderer、commitment/game projection worker；
- 新 `JudgePresenter`；
- `VoteProposal`、`NightActionProposal`、`RoleAbilityProposal`、`WolfTeamProposal`；
- serial private、commutative private 和 wolf coordinator 窗口；
- 新架构下的 world model、RAG、skills、post-game reflection 和 cross-game semantic memory；
- per-game `player_runtime_version` feature gate；
- stage-1 replay、并发、隐私、长上下文、性能和真人盲评门槛；
- 最终 cutover、历史 replay reader 和旧 `PlayerAgent` 删除。

仓库中已有的旧 world model、memory、RAG、skills、reflection、judge 和 `PlayerAgent` 不能计入上述新运行时完成度。总体设计明确拒绝把旧决策协议作为兼容合同。

## 7. 当前唯一下一里程碑

### Durable active-turn fence

在实现生产 dispatcher 或调用 `create_dispatch()` 之前，先设计并实现 durable active-turn fence。这是当前最高优先级，原因是没有它会存在多进程竞态：一个进程可能在另一个进程取消、过期、replace 或推进 turn 后仍创建新的外部请求。

该里程碑至少必须保证：

1. 创建 dispatch attempt 时，在同一后端事务中验证 schedule、managed turn 和 admission identity；
2. schedule 仍为 open，且 `active_turn_id` 精确匹配；
3. managed turn 尚未进入 terminal 状态；
4. turn、window、window version、base game revision、view fingerprint 和 model lease hash 与 attempt 精确绑定；
5. deadline 尚未失效，且 dispatch recovery barrier 开放；
6. fence 与 cancel、expire、replace、advance 和 terminal completion 使用相同的持久化竞争边界；
7. CAS 失败或任何中间异常不留下 partial attempt；
8. Memory、SQLite、PostgreSQL 行为等价；
9. 并发测试能够证明“dispatch 创建”和“turn 终态化”只能有一个合法赢家；
10. 本阶段仍不接入旧 PlayerAgent、旧 prompt、真实模型或 live game path。

不要在没有专题设计和实施计划的情况下直接修改生产 dispatcher。推荐先创建新的专题设计，再按测试驱动方式实施。

## 8. 下一会话必读文件

按以下顺序读取，不要一开始扫描整个仓库：

### 第一层：规则和总体目标

1. `AGENTS.md`
2. `handoff.md`
3. `docs/superpowers/specs/2026-07-28-autonomous-player-agent-runtime-design.md`
   - 优先阅读 §7.1～7.4、§11、§20、§25、§26、§27.7、§28～30。

### 第二层：刚完成阶段的专题设计和计划

4. `docs/superpowers/specs/2026-07-29-autonomous-player-durable-dispatch-design.md`
5. `docs/superpowers/plans/2026-07-29-autonomous-player-durable-dispatch.md`
6. `docs/superpowers/specs/2026-07-30-serial-public-scheduler-host-runtime-design.md`
7. `docs/superpowers/plans/2026-07-30-autonomous-player-serial-public-host-runtime.md`

### 第三层：下一任务直接相关代码

8. `werewolf_agent/player_agents/contracts/dispatch.py`
9. `werewolf_agent/player_agents/contracts/scheduling.py`
10. `werewolf_agent/player_agents/contracts/turns.py`
11. `werewolf_agent/storage/durable_dispatch.py`
12. `werewolf_agent/storage/autonomous_turns.py`
13. `werewolf_agent/player_agents/runtime/host.py`
14. `werewolf_agent/storage/memory_store.py`
15. `werewolf_agent/storage/sqlite_store.py`
16. `werewolf_agent/storage/postgres_store.py`

### 第四层：下一任务直接相关测试

17. `tests/storage/test_durable_dispatch_protocol.py`
18. `tests/storage/test_autonomous_turns.py`
19. `tests/player_agents/test_host_runtime.py`
20. `tests/storage/test_autonomous_commit.py`
21. `tests/storage/test_postgres_autonomous_commit.py`
22. `tests/player_agents/test_runtime_import_boundary.py`

仓库存在 `.codegraph/`。理解或定位代码时先运行 `codegraph explore "<问题或符号>"`，然后再做局部 `rg` 和文件读取。

## 9. 下一任务建议执行顺序

1. 刷新 Git 状态和聚焦测试基线。
2. 用 CodeGraph 查清 `create_dispatch`、`finish_active_turn`、cancel/expire/recovery 的所有后端路径。
3. 写 durable active-turn fence 专题设计，明确事务身份、锁顺序、CAS 和稳定错误。
4. 设计评审通过后写实施计划。
5. 从纯状态准备函数和失败测试开始，再实现 Memory 后端。
6. 实现 SQLite，并覆盖真实并发、rollback 和 migration 行为。
7. 实现 PostgreSQL schema/transaction，并补真实 PostgreSQL 集成测试。
8. 把 HostRuntime/dispatcher 收口到新 fence，禁止绕过仓储原子边界。
9. 运行跨后端聚焦测试、ruff、mypy 和 `git diff --check`。
10. 合并前运行全量 pytest，并更新本文件的进度、验证证据和唯一下一里程碑。

## 10. 必须保持的架构红线

- `RuleEngine` 是游戏机制、合法动作、阶段和胜负的唯一权威。
- `CommitTurn` 是 canonical game revision 前进的唯一新运行时事务；顺序调用 `save_game()` 和 `append_events()` 不能替代它。
- HostRuntime 管权限和副作用，模型、工具、RAG、memory、reflection、world model 和 presenter 都不能直接写 `GameState`。
- `PublicSpeechRecord` 是公开语义权威；`RenderedUtterance` 不能被重新解析成游戏事实。
- 新包不得导入或适配旧 `PlayerAgent`、旧 SpeechAct、旧 prompt/directive/strategy/retry/fallback 决策路径。
- 当前阶段不得把新 HostRuntime 接回 live game path；必须等第一条纵向链路和 feature gate 设计完成。
- 新建 schedule 必须从 fresh initial state 开始，不能伪造推进状态。
- replacement 必须同时使用新的 turn ID 和新的 idempotency key。
- cancel/expire 必须先处理可取消的 unresolved dispatch，再完成 terminal advancement。
- `DISPATCHED` 和 `UNKNOWN_OUTCOME` 不能被静默删除、重用或伪装成一次新请求。
- transient `assert_dispatch_allowed` 失败不能抹掉已经完成的 recovery qualification；真正的 recovery 失败仍必须阻止新工作。
- 不得用进程内锁、二次扫描或先写后检查模拟跨进程 durable fence。
- PostgreSQL 唯一冲突只能根据精确 constraint 和 SQLSTATE/PGCODE `23505` 映射。
- SQLite/PostgreSQL schema 变化必须先处理历史重复或不一致数据，不能泄露裸数据库异常。
- 不保存 hidden chain-of-thought、未过滤 provider output、凭据、完整系统 prompt 或跨玩家私有信息。
- 不为当前单一调用者提前建立不必要抽象；变更必须小、可回滚，并保持 Memory/SQLite/PostgreSQL 契约一致。

## 11. 明确不要读取或复用的旧设计

以下内容已被总体设计否决，不能用来指导新实现：

- `docs/superpowers/specs/2026-07-28-player-context-speech-act-design.md`
- `docs/superpowers/plans/2026-07-28-structured-speech-foundation.md`
- `werewolf_agent/agents/speech_act_schemas.py`
- `tests/agents/test_speech_act_schemas.py`
- 旧 `PlayerAgent` API、prompt、directive、strategy、quality、retry、parser 和 fallback 测试所表达的决策协议

旧规则、event、replay、visibility、persistence、API 和 runtime-flow 测试仍应保留，因为它们保护的是 Host/RuleEngine 行为，而不是旧玩家决策合同。

## 12. 测试与交付要求

开发中优先运行最窄的相关测试。当前新运行时基础回归命令：

```bash
conda run -n wofkill python -m pytest tests/player_agents \
  tests/storage/test_autonomous_commit.py \
  tests/storage/test_autonomous_turns.py \
  tests/storage/test_durable_dispatch_protocol.py \
  tests/storage/test_postgres_autonomous_commit.py -q
```

静态检查至少覆盖本次修改文件；示例：

```bash
conda run -n wofkill python -m ruff check --ignore UP009 \
  werewolf_agent/player_agents werewolf_agent/storage tests/player_agents tests/storage
conda run -n wofkill python -m mypy --follow-imports=skip \
  werewolf_agent/player_agents werewolf_agent/storage/autonomous_turns.py \
  werewolf_agent/storage/durable_dispatch.py
git diff --check
```

进入合并或声称仓库级完成前，必须重新运行：

```bash
conda run -n wofkill python -m pytest -q
```

测试通过只证明被覆盖的行为，不代表总体设计阶段完成。必须同时逐项核对该阶段的设计验收条件。

## 13. 已知外部边界

- 本地 `master` 尚未 push；是否 push 必须由用户明确授权。
- 尚未运行真实 PostgreSQL 服务集成测试。
- 当前没有真实 production dispatcher、AgentLoop、workspace、模型或工具调用。
- 当前新代码没有接入 live game runtime，这是有意的安全边界。
- stage-1 的 replay、隐私、并发、性能和真人质量门槛尚未执行。

## 14. 完成一次里程碑后如何更新本文件

每次完成并验证一个里程碑后，在同一变更中更新本文件：

1. 修改“更新日期”；
2. 刷新第 3 节的 branch、HEAD、ahead/behind、工作树和验证结果；
3. 把功能从“下一里程碑”移动到“已实现”或“部分实现”；
4. 明确哪些生产链路仍未接通，避免把 schema/文件存在误写成完成；
5. 将第 7 节改成一个且只有一个新的最高优先级里程碑；
6. 调整第 8 节必读文件，使其只覆盖下一任务需要的上下文；
7. 记录新增架构红线、迁移风险和已知外部边界；
8. 写出实际运行的验证命令和结果，不引用旧会话的“应该通过”；
9. 检查是否出现未填写的占位标记、互相矛盾的状态或已经不存在的路径；
10. 不删除历史设计和计划，也不把本文件变成逐提交 changelog。

## 15. 后续路线图

当前 fence 完成后，继续严格按以下顺序推进：

1. 真实 PostgreSQL 集成验证；
2. ObservationProjector 和隔离 player workspace；
3. context budget、compaction checkpoint 和 rehydration；
4. 最小 ToolGateway、working reflection，以及专用 ToolResult Markdown
   模型展示层；Markdown 只作为结构化结果的确定性只读投影，并按
   `result_kind` 通过 JSON/Markdown A/B 门槛后启用；
5. 第一版 daytime-speech AgentLoop；
6. 完整 Host validation、RuleEngine resolution、`CommitTurn` 编排；
7. commitment/game projections、deterministic player renderer 和 JudgePresenter；
8. stage-1 feature gate 与全部可执行验收门槛；
9. vote、private role windows 和 wolf coordinator；
10. world model、RAG、skills、reflection 和 cross-game memory 的独立 feature/ablation gates；
11. 默认启用评估、历史 replay 兼容和最终 legacy cutover。

在第一条真实昼间发言纵向链路通过 replay、隐私、并发和确定性渲染门槛之前，不要扩展到投票、夜间能力或 cognition extensions。
