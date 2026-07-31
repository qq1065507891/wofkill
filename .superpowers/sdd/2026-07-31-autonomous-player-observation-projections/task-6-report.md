# Task 6 报告：Observation Projection 里程碑收口

日期：2026-07-31

## 1. 范围与仓库快照

- 分支：`codex/autonomous-player-observation-projections`
- worktree：`/Users/zengyilin/NLP/wofkill/.worktrees/autonomous-player-observation-projections`
- Task 6 原实现 HEAD（文档提交前）：`6afff34b605b533a752869fd6b2e57d1441635b6`
- Task 6 文档 HEAD / final-fix base：`3c8781c`
- final-fix 最终实现 HEAD（证据文档提交前）：`7f3fc8739a9d6cf289e8c477b412c87aa482c3da`
- whole-branch review base：`83eb1be`
- Task 6 本身未修改生产代码；只更新 canonical `handoff.md` 并新增本报告。之后 final review 发现 merge-blocking gap，已由唯一一次 final-fix wave 修改 observation contracts/rendering/workspace 及其测试；本报告以下结果以修复后的 fresh gate 为准。

## 2. Fresh 验证证据

### Focused new-runtime suite

```bash
conda run -n wofkill python -m pytest \
  tests/player_agents \
  tests/storage/test_autonomous_commit.py \
  tests/storage/test_autonomous_turns.py \
  tests/storage/test_active_turn_fence.py \
  tests/storage/test_durable_dispatch_protocol.py \
  tests/storage/test_postgres_autonomous_commit.py \
  -o addopts='' -q
```

结果：exit 0，`585 passed in 1.21s`。

### Scoped static checks

```bash
conda run -n wofkill python -m ruff check --ignore UP009 \
  werewolf_agent/player_agents \
  werewolf_agent/storage/active_turn_fence.py \
  tests/player_agents \
  tests/storage/test_active_turn_fence.py
```

结果：exit 0，`All checks passed!`。

```bash
conda run -n wofkill python -m mypy --follow-imports=skip \
  werewolf_agent/player_agents \
  werewolf_agent/storage/active_turn_fence.py \
  werewolf_agent/storage/autonomous_turns.py
```

结果：exit 0，`Success: no issues found in 27 source files`。

`git diff --check`：exit 0，无输出。

### Full repository suite

```bash
conda run -n wofkill python -m pytest -q
```

结果：`pytest -q` 与独立的 `pytest -q -rs` 均 exit 0。quiet/xdist 执行输出没有打印 passed 数量汇总；另行执行 `conda run -n wofkill python -m pytest --collect-only -qq`，exit 0，将 308 个逐文件计数独立求和为 6365。因此 6365 仅表示 collected，不表示 passed。`-rs` 明确列出 12 个 skip：1 个已移除本地 negation 常量、9 个缺 PowerShell 的 soak orchestration、1 个缺 PowerShell 的 AST、1 个未启用 real-provider smoke。warning summary 为 10 条既有第三方 `StarletteDeprecationWarning`，来源是 `fastapi.testclient`。

## 3. 设计验收映射

1. 一个有效的 active `serial_public` daytime speech turn 能构建 required PLAYER/ROLE/GAME/INDEX、capability 存在时的 COMMITMENTS、显式 unavailable optional entries 和 immutable `ObservationFrame`；由 contracts、workspace、service 与 conformance tests 覆盖。
2. frame 和 documents 都绑定 game、player、schedule、turn、window、base revision 与 view fingerprint；authority source IDs/hashes 也参与校验。
3. deterministic rendering、canonical document hash、ordered workspace hash、包含 required/visibility/summary authority digest 的 workspace revision，以及 rebuild equality 由 contracts/rendering/workspace tests 覆盖。
4. cross-player、stale turn、changed source/visibility、expired/cancelled/replaced turn 均 fail closed，不返回 partial bundle。
5. adversarial fixtures 检查 hidden-role 与 other-player-private leakage；`GAME.md` 的 document/manifest/INDEX visibility 全链保持 `MIXED_VIEWER_FILTERED`，shared cache 按 viewer 隔离。
6. 生成内容是只读 presentation；Markdown 不解析回 authority，也不授权 dispatch 或 commit。
7. Memory 与 SQLite conformance 覆盖相同 observation service 行为。
8. boundary tests 和最终 scan 证明没有引入物理 workspace、projection persistence、provider/model、ToolResult Markdown、AgentLoop、live/legacy path。
9. focused/full pytest、scoped Ruff、mypy 和 diff check 都 fresh exit 0。

## 4. 可用性与明确排除项

- required/available：`PLAYER.md`、`ROLE.md`、`GAME.md`、`INDEX.md`。
- `COMMITMENTS.md`：只有 committed-record capability 存在时 available；空 tuple 是 available-empty，不是 unavailable。
- unavailable：`BELIEFS.md`、`MEMORY.md`、`WORKING.md`。
- 没有真实 `players/{player_id}/*.md` 文件，没有 Memory/SQLite/PostgreSQL projection table；process-local cache 是可删优化，fault 等同 miss。
- 没有 beliefs、cross-game memory、working reflection、context compaction/checkpoint、ToolGateway、ToolResult Markdown、provider/model、AgentLoop、live/legacy path、RuleEngine/`CommitTurn` 编排或完整可玩链路。
- PostgreSQL 仍只有 schema/fake-connection/contract 层覆盖；未运行真实 PostgreSQL service integration，仍是生产接入前 gate。

## 5. Final boundary scan

执行：

```bash
rg -n "PlayerAgent|GameRunner|ModelRouter|_dispatch_agent|ToolResultMarkdownProjection|players/" \
  werewolf_agent/player_agents/observation \
  tests/player_agents/test_observation_* \
  handoff.md
git status --short --branch
git diff --stat 83eb1be..HEAD
git diff --check
```

结果：scan 共 26 个 deliberate literal matches，其中 production observation package 为 0，boundary test fixtures 为 19，canonical handoff 的明确禁止/排除说明为 7；`git diff --check` exit 0。必须区分四个快照：Task 1～5 的 implementation-only snapshot `83eb1be..6afff34` 是 19 个文件、7236 insertions、3 deletions；Task 6 文档后的 reviewed branch `83eb1be..3c8781c` 是 21 个文件、7422 insertions、66 deletions；final-fix 最终实现 HEAD `83eb1be..7f3fc87` 是 21 个文件、7789 insertions、66 deletions；最终 evidence/docs commit 后的 final branch diff `83eb1be..HEAD` 是 22 个文件、7975 insertions、66 deletions。scan 未发现 production forbidden dependency 或物理 workspace path。

## 6. Final review 修复更正

- `bounded_public_summary` 不再是未绑定的 GAME 渲染旁路：其排序后的规范 authority digest 进入 GAME section plan 和 pre-render workspace revision，因此共享 cache 无法在摘要 authority 改变后复用旧 GAME bytes。
- `GAME.md` 不再误标 `PUBLIC`，renderer、manifest、INDEX、revision 与 cache 验证均使用 `MIXED_VIEWER_FILTERED`。
- `ProjectedDocument` 与 `PlayerWorkspaceSnapshot` 现在自行验证 canonical bytes/hash，contracts 与 projector 共享 canonical helpers，外部不能提交自洽 metadata 之外的伪造 hash。
- uncertainty alternatives 使用 `(value_id, confidence, sorted support_refs)` total stable key；available-empty COMMITMENTS、exact INDEX public fields/source-metadata exclusion、cache fault/error priority 均有直接回归。
- Final-fix TDD 定向 RED 为 12 selected 中 8 failed、4 passed；生产修复后同一选择集 12 passed，三份核心测试 79 passed，完整 observation/boundary 组合 173 passed。

## 7. 下一唯一里程碑

实现 `ContextBudgetPolicy`、80% 自动 trigger、严格 `CompactionCheckpoint`、可选且不可信的 `CompactionHandoff`、55% rehydration target、lineage validation 和 restart recovery。ToolGateway、ToolResult Markdown、provider/model dispatch 与 AgentLoop 继续后置。
