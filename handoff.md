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

新运行时已经完成严格合约、三后端原子提交、durable dispatch、`serial_public` 调度、HostRuntime 生命周期、durable active-turn fence，以及隔离的只读 observation projection 边界；同时已形成每玩家逻辑隔离的 Deep Agents cognition adapter 专题设计，但尚未安装或实现 Deep Agents，也尚未接入真实游戏、模型、工具、RuleEngine 提交流程或旧玩家运行入口。

按总体设计第 28 节的 13 步实施序列判断：

- 第 1 步已完成；
- 第 2 步已完成 stage-1 昼间发言所需的主要合约，但其他终端提案仍未实现；
- 第 3 步已完成 Memory、SQLite、PostgreSQL 的基础事务能力；
- 第 4 步完成 durable active-turn fence：调度、Host 生命周期、取消、过期和重启协调均通过同一持久化围栏竞争边界；生产 dispatcher 仍未实现；
- 第 5 步的隔离 player documents 和 optimistic `ObservationFrame` 子阶段已完成；context budget、compaction checkpoint 和 restart rehydration 尚未实现，因此第 5 步整体仍未完成；
- 第 7 步中的 `SpeechProposal`、`PublicSpeechRecord` 和原子 commit 合约被提前实现，但完整纵向链路没有闭环；
- 第 6 步及第 8～13 步的主体尚未开始；第 7 步只有上述前置合约，不是纵向链路。

因此，当前不能称为“第一阶段完成”，也不能通过新运行时玩完一次真实昼间发言回合。

## 3. 当前仓库快照与首次检查

记录本文件时：

- 分支：`codex/deep-agent-player-executor-design`
- worktree：`/Users/zengyilin/NLP/wofkill/.worktrees/deep-agent-player-executor-design`
- Deep Agent 专题设计提交：`c1a295b`（`docs: design deep agent player executor`）；分支 base 是已合并 observation milestone `b4cb575`
- 设计提交后运行 focused 与 full pytest；更新本 handoff 前 tracked worktree 只有 handoff 修改
- 新运行时 focused pytest：fresh exit 0，`585 passed in 1.56s`
- 全量 pytest：fresh exit 0，明确 12 个 skip 和 10 条既有第三方 `StarletteDeprecationWarning`；quiet/xdist 未打印 passed 汇总，因此不把 collected 数写成 passed 数
- 全量 pytest：`pytest -q` 与独立的 `pytest -q -rs` 均 fresh exit 0；collection-only 的 308 个逐文件计数独立求和为 6365 collected；`-rs` 明确列出 12 个 skip（1 个 shared-negation、9 个 PowerShell soak、1 个 PowerShell AST、1 个 real-provider smoke），warning summary 是 10 条既有第三方 `StarletteDeprecationWarning`（`fastapi.testclient`）；quiet/xdist 输出没有打印 passed 汇总，因此 6365 只记为 collected，不记为 passed

新会话不要直接相信以上动态值。先在仓库根目录执行：

```bash
git status --short --branch
git log --oneline -12
conda run -n wofkill python -m pytest tests/player_agents \
  tests/storage/test_autonomous_commit.py \
  tests/storage/test_autonomous_turns.py \
  tests/storage/test_active_turn_fence.py \
  tests/storage/test_durable_dispatch_protocol.py \
  tests/storage/test_postgres_autonomous_commit.py -o addopts='' -q
```

仅在准备合并、交付或声称仓库级回归通过时，再运行：

```bash
conda run -n wofkill python -m pytest -q
```

项目所有 Python、pytest、ruff 和 mypy 命令都必须通过 `conda run -n wofkill` 执行。

本阶段的 fresh 验证证据：

```bash
conda run -n wofkill python -m pytest tests/player_agents \
  tests/storage/test_autonomous_commit.py \
  tests/storage/test_autonomous_turns.py \
  tests/storage/test_active_turn_fence.py \
  tests/storage/test_durable_dispatch_protocol.py \
  tests/storage/test_postgres_autonomous_commit.py -o addopts='' -q
conda run -n wofkill python -m ruff check --ignore UP009 \
  werewolf_agent/player_agents \
  werewolf_agent/storage/active_turn_fence.py \
  tests/player_agents \
  tests/storage/test_active_turn_fence.py
conda run -n wofkill python -m mypy --follow-imports=skip \
  werewolf_agent/player_agents \
  werewolf_agent/storage/active_turn_fence.py \
  werewolf_agent/storage/autonomous_turns.py
git diff --check
conda run -n wofkill python -m pytest -q
```

上述 focused pytest、Ruff、mypy 与 diff check 都以 exit 0 结束；focused 结果是 `585 passed in 1.21s`，Ruff 输出 `All checks passed!`，mypy 输出 `Success: no issues found in 27 source files`。full `pytest -q` 与 `pytest -q -rs` 均 fresh exit 0，独立 collection-only 的 308 个逐文件计数求和为 6365 collected；`-rs` 明确列出 12 个 skip，warning summary 是 10 条既有第三方 warning。pytest 在该 quiet/xdist 配置下没有打印 passed 数量汇总，因此这里不把 collected 数虚构成 passed 数。boundary scan 仅命中 deliberate boundary-test 和文档中的禁用边界字面量，生产 observation 代码零命中；没有运行真实 PostgreSQL service integration。

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
- cancel/expire/complete 均通过 active-turn fence 原子终结；它们不在 Host 内预扫描 dispatch。

主要文件：

- `werewolf_agent/storage/autonomous_turns.py`
- `werewolf_agent/player_agents/runtime/serial_public.py`
- `werewolf_agent/player_agents/runtime/host.py`
- `tests/storage/test_autonomous_turns.py`
- `tests/player_agents/test_host_runtime.py`

### 4.6 Durable active-turn fence

- `ActiveTurnDispatchFence` 将 dispatch 绑定到持久化 `schedule_id`、schedule CAS、预约后的 managed-turn CAS、window ID/version 和 base game revision；attempt 原有 game、turn、actor、lease、view 和 deadline 字段完成其余身份绑定。
- `create_active_turn_dispatch()` 在同一后端事务中校验精确身份、恢复门禁和截止时间，并只增加 managed-turn `state_version` 后写入仓储生成的 fence；历史 unfenced rows 仍可读取，但不能作为生产授权。
- `finish_active_turn_fenced()` 与预约共享 game → schedule → managed turn → dispatch attempts 的竞争边界：cancel/expire 原子取消 `PENDING`/`DISPATCHING`，complete 拒绝 unresolved work，再原子更新回合和调度。
- Memory、SQLite 与 PostgreSQL 都声明显式 capability。Memory/SQLite 的共享 conformance 覆盖预约、取消、稳定 CAS error code 和 defensive copies；PostgreSQL 仅有 schema/lock/CAS/rollback 的 fake-connection 合同验证。
- 本次新增 conformance（包括预约后 managed-turn 版本、fence version、advance 后 cursor/identity 及 terminal turn status）首次即绿，属于对已审查行为的 characterization，不代表重新实现生产逻辑。

主要文件：

- `werewolf_agent/storage/active_turn_fence.py`
- `werewolf_agent/storage/memory_store.py`
- `werewolf_agent/storage/sqlite_store.py`
- `werewolf_agent/storage/postgres_store.py`
- `tests/storage/test_active_turn_fence.py`
- `tests/player_agents/test_host_runtime.py`

### 4.7 隔离 observation projections 与 optimistic `ObservationFrame`

- `werewolf_agent/player_agents/observation/` 定义严格、冻结且禁止额外字段的 projection identity、source reference、document、manifest、workspace、`ObservationFrame` 和 `ObservationBundle` 合约，并提供稳定、安全的错误类型。
- `ObservationAuthoritySnapshot` 是 revision/visibility-pinned、单 viewer 的只读 authority 输入；每个可用文档都绑定 game、player、schedule、turn、window、base revision、view fingerprint、renderer version、source IDs 和 source hashes；进入 `GAME.md` 的 bounded summary 另外以规范 authority digest 参与 pre-render workspace revision 与 cache key。
- 专用 renderer 已实现 `PLAYER.md`、`ROLE.md`、`GAME.md` 和 `COMMITMENTS.md`；其中 `GAME.md` 因包含 viewer-authorized private fact references，document、manifest 与 `INDEX.md` 均标为 `MIXED_VIEWER_FILTERED`。`WorkspaceProjector` 在任何 cache lookup 前把 section requiredness、visibility、availability、renderer/estimator、source references 和 summary digest 纳入 revision，再生成确定性 manifest、workspace hash 与 `INDEX.md`，Markdown 永远不解析回 Host state。
- `ProjectedDocument` 强制 LF-only、恰一个末尾 LF，并复算 Markdown UTF-8 SHA-256；`PlayerWorkspaceSnapshot` 强制文档按 manifest available 顺序排列并复算完整 ordered workspace hash。contracts、renderer 与 projector 共用同一组 canonical hash helpers。
- required 且可用的是 `PLAYER.md`、`ROLE.md`、`GAME.md`、`INDEX.md`。只有 committed-record capability 存在时 `COMMITMENTS.md` 才可用；存在 capability 但记录为空时仍是 available-empty。`BELIEFS.md`、`MEMORY.md`、`WORKING.md` 明确 unavailable。
- 可选 `InMemoryProjectionCache` 只在进程内保存可重建 projection；cache fault 等同 miss，较窄且安全的 projection error 优先。没有创建物理 `players/{player_id}/*.md` 文件，也没有 Memory、SQLite 或 PostgreSQL projection table。
- `ObservationProjectionService` 捕获活动回合身份，读取单 viewer authority，装配 workspace/frame，再 optimistic recheck 活动回合；活动身份、source、visibility 或 deadline 改变时 fail closed，不返回部分 bundle。
- Memory 与 SQLite observation conformance 覆盖相同可观察行为；边界 scanner 检查具体生产 observation boundary 和显式 dispatch 形态，不宣称实现完整 Python 名称解析器。

主要文件：

- `werewolf_agent/player_agents/observation/contracts.py`
- `werewolf_agent/player_agents/observation/authority.py`
- `werewolf_agent/player_agents/observation/rendering.py`
- `werewolf_agent/player_agents/observation/workspace.py`
- `werewolf_agent/player_agents/observation/cache.py`
- `werewolf_agent/player_agents/observation/service.py`
- `tests/player_agents/test_observation_contracts.py`
- `tests/player_agents/test_observation_rendering.py`
- `tests/player_agents/test_workspace_projector.py`
- `tests/player_agents/test_observation_service.py`
- `tests/player_agents/test_observation_conformance.py`
- `tests/player_agents/test_observation_import_boundary.py`

## 5. 只有部分实现，不能误判为完成

### 5.1 HostRuntime 还不是完整 Host

当前 HostRuntime 只协调持久化调度、恢复门禁、围栏预约和回合生命周期。独立的 Host-facing `ObservationProjectionService` 已能构建 observation，但尚未接入 HostRuntime 的真实执行链路。HostRuntime 还不会：

- 调用模型或工具；
- 管理真实 model lease、`ContextBudgetPolicy`、compaction checkpoint 或 rehydration；
- 验证 speech proposal 的可见性与语义；
- 调用 RuleEngine；
- 调用 `CommitTurn`；
- 运行持久化 projection worker；
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

### 5.5 Deep Agents 只有专题设计，没有运行实现

`docs/superpowers/specs/2026-07-31-deep-agent-player-executor-design.md` 已确定：每个玩家具有独立的逻辑 deep-agent 身份、profile、viewer workspace 和 turn-scoped thread，但所有玩家共享同一个 framework-neutral `PlayerCognitionExecutor` 与 `DeepAgentPlayerExecutor` harness 实现。Deep Agents 只是 bounded `AgentLoop` 的可替换实现，不是新的 Host、scheduler、repository 或 game authority。

当前没有安装 `deepagents`，也没有实现 cognition contracts、adapter、virtual backend、`FencedChatModelAdapter`、terminal tool 或 feature gate。Stage 1 必须显式禁用 Deep Agents 自动 summarization、durable framework checkpoint、long-term memory、shell/execute 和全部 subagent；任何模型或外部工具调用仍须经过 durable active-turn fence。设计存在不等于 AgentLoop 已完成。

## 6. 尚未实现的主体功能

- `ContextBudgetPolicy`、80% 自动压缩触发、严格 `CompactionCheckpoint`、可选且不可信的 `CompactionHandoff`、55% rehydration target、lineage validation 和 restart recovery；
- `BELIEFS.md`、`MEMORY.md`、`WORKING.md` 的新运行时 authority stores 和 projectors；
- durable compaction/checkpoint persistence；当前 projection 本身仍有意不持久化；
- model lease、预算、failover 和真实 provider adapter；
- stage-1 ToolGateway、context/evidence 工具和 working reflection candidate；
- 已完成设计但尚未实现的专用 ToolResult Markdown 模型展示层，设计见
  `docs/superpowers/specs/2026-07-31-tool-result-markdown-projection-design.md`；
- 真正的 AgentLoop；
- framework-neutral `PlayerCognitionExecutor`、`DeepAgentPlayerExecutor`、每玩家 profile/namespace、turn-scoped Deep Agents thread 和受控 virtual filesystem；
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

### Context budget、compaction checkpoint 与 rehydration

下一阶段只实现 `ContextBudgetPolicy` 与 context accounting、80% 自动 compaction trigger、严格且可恢复的 `CompactionCheckpoint`、可选且标为 untrusted 的 `CompactionHandoff`、55% rehydration target、checkpoint lineage validation，以及进程重启后的 checkpoint recovery。Host checkpoint 是唯一 resumable authority；model handoff 只是可缺失的建议性数据，不能新增事实、证据、grant、合法目标或动作。

该阶段必须遵循新 Deep Agent 专题设计预留 framework-neutral `PlayerCognitionExecutor` 消费边界，但仍不安装或调用 Deep Agents。Deep Agents 内置 summarization 和 LangGraph checkpoint 不能替代本阶段的 Host checkpoint；未来 adapter 只能消费 Host 已验证的 rehydrated context。

该阶段仍不实现 ToolGateway、ToolResult Markdown projection、真实 provider/model 调用、AgentLoop、proposal validation、RuleEngine/`CommitTurn` 编排、live game path 或旧 `PlayerAgent` 接入，也不把 projection 变成物理文件或 durable authority；因此它仍不是可玩的纵向链路。

## 8. 下一会话必读文件

按以下顺序读取，不要一开始扫描整个仓库：

### 第一层：规则和总体目标

1. `AGENTS.md`
2. `handoff.md`
3. `docs/superpowers/specs/2026-07-28-autonomous-player-agent-runtime-design.md`
   - 优先阅读 §7.1～7.4、§11、§20、§25、§26、§27.7、§28～30。

### 第二层：刚完成阶段的专题设计和计划

4. `docs/superpowers/specs/2026-07-31-deep-agent-player-executor-design.md`
5. `docs/superpowers/specs/2026-07-31-autonomous-player-observation-projections-design.md`
6. `docs/superpowers/plans/2026-07-31-autonomous-player-observation-projections.md`

### 第三层：下一任务直接相关代码

7. `werewolf_agent/player_agents/observation/contracts.py`
8. `werewolf_agent/player_agents/observation/service.py`
9. `werewolf_agent/player_agents/observation/workspace.py`
10. `werewolf_agent/player_agents/contracts/turns.py`
11. `werewolf_agent/storage/autonomous_turns.py`

### 第四层：下一任务直接相关测试

12. `tests/player_agents/test_observation_contracts.py`
13. `tests/player_agents/test_observation_service.py`
14. `tests/player_agents/test_observation_conformance.py`
15. `tests/player_agents/test_observation_import_boundary.py`

仓库存在 `.codegraph/`。理解或定位代码时先运行 `codegraph explore "<问题或符号>"`，然后再做局部 `rg` 和文件读取。

## 9. 下一任务建议执行顺序

1. 刷新 Git 状态和 observation focused suite 基线。
2. 用 CodeGraph 查清 `AgentTurn` 状态机、active-turn repository、observation identity 和未来 checkpoint 持久化边界。
3. 按 Deep Agent 专题设计的 framework-neutral seam，为 `ContextBudgetPolicy`、checkpoint transaction、lineage、rehydration 和 restart recovery 写专题设计与实施计划；本阶段不得安装或调用 Deep Agents。
4. 先写 80% trigger、55% target、stale lineage、optional handoff failure 和 restart recovery 的失败测试，再实现最小 context lifecycle capability。
5. 保持 Host checkpoint 为唯一可恢复 authority，严格验证 player/turn/revision/view/source lineage；不得从 model handoff 恢复事实或权限。
6. 在不接入 ToolGateway、provider、AgentLoop、旧玩家或 live game path 的前提下完成聚焦测试、ruff、mypy、diff check 和全量 pytest。

## 10. 必须保持的架构红线

- `RuleEngine` 是游戏机制、合法动作、阶段和胜负的唯一权威。
- `CommitTurn` 是 canonical game revision 前进的唯一新运行时事务；顺序调用 `save_game()` 和 `append_events()` 不能替代它。
- HostRuntime 管权限和副作用，模型、工具、RAG、memory、reflection、world model 和 presenter 都不能直接写 `GameState`。
- `PublicSpeechRecord` 是公开语义权威；`RenderedUtterance` 不能被重新解析成游戏事实。
- 新包不得导入或适配旧 `PlayerAgent`、旧 SpeechAct、旧 prompt/directive/strategy/retry/fallback 决策路径。
- 当前阶段不得把新 HostRuntime 接回 live game path；必须等第一条纵向链路和 feature gate 设计完成。
- 新建 schedule 必须从 fresh initial state 开始，不能伪造推进状态。
- replacement 必须同时使用新的 turn ID 和新的 idempotency key。
- cancel/expire/complete 必须在同一 durable fence transaction 中处理精确 turn 的 unresolved dispatch 与 terminal advancement，不能由 Host 的二次扫描实现。
- `DISPATCHED` 和 `UNKNOWN_OUTCOME` 不能被静默删除、重用或伪装成一次新请求。
- transient `assert_dispatch_allowed` 失败不能抹掉已经完成的 recovery qualification；真正的 recovery 失败仍必须阻止新工作。
- 不得用进程内锁、二次扫描或先写后检查模拟跨进程 durable fence。
- PostgreSQL 唯一冲突只能根据精确 constraint 和 SQLSTATE/PGCODE `23505` 映射。
- SQLite/PostgreSQL schema 变化必须先处理历史重复或不一致数据，不能泄露裸数据库异常。
- 不保存 hidden chain-of-thought、未过滤 provider output、凭据、完整系统 prompt 或跨玩家私有信息。
- observation Markdown 只是 source-bound、viewer-specific 的只读展示，不能解析回 Host state 或作为 dispatch/commit 权限。
- `GAME.md` 含当前 viewer 授权的私有事实引用，必须在 document、manifest、INDEX 和 revision/cache 语义中保持 `MIXED_VIEWER_FILTERED`，不能降级标为 public。
- projection cache 是可删的进程内优化，cache failure 必须等同 miss；projection 不写物理 workspace 文件或 durable projection table。
- Deep Agents 只能作为可替换的 `PlayerCognitionExecutor` 实现；一名玩家对应一个逻辑 agent identity，一个 admitted turn 对应一个 bounded framework thread，不能用一个 supervisor agent 扮演或读取多个玩家。
- Deep Agents/LangGraph checkpoint、自动 summary、TODO、scratch file 和 final text 都是可丢弃的模型侧数据；Host `CompactionCheckpoint` 仍是唯一 resumable authority。
- Stage 1 必须显式禁用 Deep Agents 默认 general-purpose subagent、async subagent、自动 summarization、durable framework store、long-term memory 和 shell/execute；framework retry/failover 不能绕过 Host durable dispatch。
- Stage 1 昼间发言以及未来普通投票、简单夜间动作默认禁用 `write_todos`；TODO 不是提交动作的前置步骤。只有复杂 cognition task 在预声明消融证明净收益后才能按 `task_type` 开启，且其状态只属于当前 turn、计入预算、终态后丢弃并排除在 Host checkpoint authority 之外。
- Deep Agents adapter 不得直接构造 provider、MCP、repository、RuleEngine 或 `CommitTurn`；所有外部模型/工具调用必须穿过 durable active-turn fence，所有终态动作只能通过严格 terminal proposal gateway 返回 Host。
- required observation sections 必须 fail closed；optional unavailable section 不能泄露隐藏记录是否存在，`COMMITMENTS.md` 的 available-empty 必须与 capability-absent 区分。
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
  tests/storage/test_active_turn_fence.py \
  tests/storage/test_durable_dispatch_protocol.py \
  tests/storage/test_postgres_autonomous_commit.py -o addopts='' -q
```

静态检查至少覆盖本次修改文件；示例：

```bash
conda run -n wofkill python -m ruff check --ignore UP009 \
  werewolf_agent/player_agents \
  werewolf_agent/storage/active_turn_fence.py \
  tests/player_agents \
  tests/storage/test_active_turn_fence.py
conda run -n wofkill python -m mypy --follow-imports=skip \
  werewolf_agent/player_agents \
  werewolf_agent/storage/active_turn_fence.py \
  werewolf_agent/storage/autonomous_turns.py
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
- 当前已有隔离的内存中 observation workspace/bundle 构建能力，但没有物理 workspace 文件、durable projection table、真实 production dispatcher、context checkpoint、AgentLoop、模型或工具调用。
- 当前环境有 `langchain 1.3.14` 和 `langgraph 1.2.9`，但未安装 `deepagents`；专题设计记录时 PyPI 最新稳定版为 `0.6.12`，实现前必须做兼容性 spike 并精确 pin 稳定版本，不能直接采用 alpha。
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

当前 observation projection 完成后，继续严格按以下顺序推进：

1. `ContextBudgetPolicy`、80% trigger、`CompactionCheckpoint`、optional untrusted handoff、55% rehydration、lineage validation 和 restart recovery；
2. 最小 ToolGateway、working reflection，以及专用 ToolResult Markdown
   模型展示层；Markdown 只作为结构化结果的确定性只读投影，并按
   `result_kind` 通过 JSON/Markdown A/B 门槛后启用；
3. framework-neutral `PlayerCognitionExecutor` 与 `DeepAgentPlayerExecutor` PoC：共享 harness、每玩家逻辑身份、每 turn 独立 thread、只读 observation virtual backend、ephemeral scratch、fenced model/tool adapter、严格 `submit_speech`；第一版明确关闭 `write_todos`、自动 summarization、durable framework checkpoint、long-term memory、shell/execute 和全部 subagent；
4. 完整 Host validation、RuleEngine resolution、`CommitTurn` 编排；
5. post-commitment/game projection updates、deterministic player renderer 和 JudgePresenter；
6. stage-1 feature gate 与全部可执行验收门槛；
7. vote、private role windows 和 wolf coordinator；
8. world model、RAG、skills、reflection 和 cross-game memory 的独立 feature/ablation gates；
9. 默认启用评估、历史 replay 兼容和最终 legacy cutover。

真实 PostgreSQL service integration 是所有生产接入前必须通过的 repository gate，不是当前隔离 projection 里程碑之前的独立第一步。

在第一条真实昼间发言纵向链路通过 replay、隐私、并发和确定性渲染门槛之前，不要扩展到投票、夜间能力或 cognition extensions。
