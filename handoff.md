# Autonomous Player Serial Public Host Runtime Handoff

更新日期：2026-07-31

## 1. 现在做了什么

本阶段落地了 autonomous player 的 `serial_public` 调度与 HostRuntime 基础设施，目标是提供可恢复、可审计、可 CAS 更新的托管回合生命周期，但暂不接入真实游戏、模型、工具或旧玩家运行链路。

核心范围包括：

- 严格的 schedule、slot、admission、managed turn、terminal disposition 合约。
- Memory、SQLite、PostgreSQL 三套 `AutonomousTurnRepository` 能力及等价生命周期语义。
- durable dispatch 的按 turn 查询、恢复门禁和取消路径。
- `SerialPublicScheduler` 与 `HostRuntime`，覆盖创建、恢复、准入、状态迁移、完成、取消、过期和活动回合读取。
- CAS、事务回滚、schema migration 预检、import boundary 和跨后端原子性测试。

当前 `master` HEAD 为 `7757655`，功能分支已经快进合并并删除；功能 worktree 已清理，旧的两个 worktree 保留未动。

## 2. 已经完成了什么

- 实施计划 52/52 步完成。
- 终审修复 3/3 完成：
  - transient dispatch block 不再撤销当前进程的 recovery qualification；真正未恢复或 recovery 失败仍被阻止。
  - 新建 schedule 强制为 `open / slot 0 / no active turn / version 0`。
  - replacement 在同一 schedule 内不能复用已持久化的 idempotency key；Memory、SQLite、PostgreSQL 均有原子检查和持久约束。
- PostgreSQL 唯一约束异常只在精确 constraint 且 SQLSTATE/PGCODE `23505` 时映射为 `InvalidTurnAdmission`。
- SQLite/PostgreSQL 历史重复 key 会在 schema 初始化时被预检并抛稳定 migration error。
- 合并后的 `master` 验证结果：
  - `conda run -n wofkill python -m pytest -q`：通过，100% 完成，仅既有 warnings/skips。
  - scoped Ruff：通过。
  - scoped mypy：22 个源文件无错误。
  - live-runtime boundary `rg`：无匹配，符合本阶段不接入旧运行链路的要求。
  - `git diff --check`：通过。

设计与计划：

- `docs/superpowers/specs/2026-07-30-serial-public-scheduler-host-runtime-design.md`
- `docs/superpowers/plans/2026-07-30-autonomous-player-serial-public-host-runtime.md`

## 3. 卡在哪里

当前没有未解决的代码阻塞；本阶段已经合并到本地 `master`。仍有三个明确的外部/阶段边界：

- 本地 `master` 尚未 push，当前比 `origin/master` 超前 98 个提交。是否 push 需要后续明确执行。
- 没有运行真实 PostgreSQL 服务集成测试；当前是 PostgreSQL schema、fake connection、锁和异常路径测试。接入生产前应补真实数据库验证。
- 真实 dispatcher、AgentLoop、workspace、模型和工具调用尚未实现，这是设计上刻意保留的下一阶段边界，不是本阶段失败。

## 4. 下一该做什么

按以下顺序推进：

1. 先决定是否将本地 `master` push 到远端，并记录远端提交结果。
2. 为下一阶段创建新的 `.worktrees` 分支，不要在已合并的 `master` 上直接开发。
3. 在生产 dispatcher 调用 `create_dispatch` 之前，先设计并实现 durable active-turn fence：创建 attempt 必须与 schedule/turn 身份、lease、window、revision 在同一后端事务中校验，且与取消和 terminal advancement 共用该 fence。
4. 补真实 PostgreSQL 集成测试，再接入 observation projection、player workspace、context budgeting 和第一版 AgentLoop。
5. 让后续运行链路只调用 HostRuntime API，不绕过 scheduler ordering、recovery barrier、cancellation 或 `CommitTurn` 权威提交。

## 5. 那些坑不要踩

- 所有 Python 命令使用 `conda run -n wofkill`，不要依赖当前 shell 的 `conda activate` 状态。
- 仓库存在 `.codegraph/`；理解代码和定位符号前先尝试 CodeGraph，再做局部搜索。
- 新建 schedule 不能用 `next_slot_ordinal=1` 或非零 `state_version` 伪造 final-slot；测试必须从 slot 0 正常 admit/advance。
- replacement 必须同时使用新 turn ID 和新 idempotency key；旧 terminal turn 的 key 仍属于该 schedule 的历史，不能复用。
- 不要把 `SerialPublicSchedule` 合约模型收紧成只能表示初始状态；它还必须能加载合法的已推进持久状态。
- transient `assert_dispatch_allowed` 失败不能删除 `_recovered_games`；但真正失败的 `recover_game()` 和未恢复的重启实例仍必须阻止清理操作。
- 取消/过期要先取消 `PENDING`、`DISPATCHING` dispatch，再完成 terminal turn；已 `DISPATCHED` 或 `UNKNOWN_OUTCOME` 的 attempt 要保留审计，不能静默删除或复用。
- 不要用进程内锁、二次扫描或“先 INSERT 后补检查”代替 dispatcher 的 durable active-turn fence；多进程下仍会产生 terminalization race。
- PostgreSQL 不要按错误消息 substring 粗暴映射唯一冲突；必须检查精确 constraint 和 `23505`。
- SQLite/PostgreSQL schema 变更要考虑已有重复数据，先预检并抛稳定 migration error，不能暴露裸数据库异常或静默接受。
- 不要把 HostRuntime 接回旧的 model/tool/workspace、RuleEngine、legacy player 或 live game path；import boundary 测试是有意的护栏。
- `CommitTurn` 仍是 canonical game truth 的后续权威；`complete_active_turn()` 只是提交成功后的生命周期通知，不能代替游戏提交。
- 修改 Python 文件时同步检查中文模块说明、作者、创建日期和 `修改日期`，避免只改行为不更新头部。
- 提交前至少跑聚焦测试、scoped Ruff/mypy、`git diff --check`；进入合并前必须在合并后的结果上跑全量 pytest。
