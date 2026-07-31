# Autonomous Player Observation Projections Final-Fix Report

日期：2026-07-31

## 1. 范围与快照

- 分支：`codex/autonomous-player-observation-projections`
- worktree：`/Users/zengyilin/NLP/wofkill/.worktrees/autonomous-player-observation-projections`
- whole-branch review base：`83eb1be`
- fix-wave base：`3c8781c`
- final-fix code/test commit：`7f3fc8739a9d6cf289e8c477b412c87aa482c3da`
- 生产范围严格限制在 `werewolf_agent/player_agents/observation/{contracts,rendering,workspace}.py`；测试只修改已有 observation fixtures/tests，没有 legacy/live/model/provider/tool/ToolResult/CommitTurn/later-milestone 代码。

## 2. Findings 到实现与证据映射

### Finding 1：bounded summary cache staleness

- `WorkspaceProjector._SectionPlan` 新增 `authority_dependency_digest`。
- GAME plan 对已经过严格 authority 校验的 `bounded_public_summary` 排序后生成 canonical JSON SHA-256，并在任何 cache lookup 前进入 `workspace_revision`。
- 保留原有 pre-render revision/cache-key 架构；没有添加事后 invalidation，也没有让 cache 成为 authority。
- `test_shared_cache_cannot_reuse_game_after_summary_authority_changes` 证明同 identity/source refs 的有效 summary authority 改变会产生不同 revision、不同 GAME hash 和新 GAME bytes。
- `test_workspace_revision_covers_required_visibility_and_summary_authority` 在测试侧独立构造 canonical JSON payload 和 digest，不调用生产 hash helper。

### Finding 2：GAME visibility

- `render_game_document()` 与 `WorkspaceProjector._visibility_for(GAME)` 统一为 `MIXED_VIEWER_FILTERED`。
- document、manifest、INDEX、workspace revision、cache completeness check 全链使用同一 visibility semantics。
- `test_game_visibility_is_mixed_and_shared_cache_isolates_viewers` 使用共享 `InMemoryProjectionCache` 验证两个 view fingerprint 的隔离与 metadata/INDEX 标签。

### Finding 3：canonical integrity contracts

- canonical JSON bytes/hash、Markdown content hash 和 ordered workspace hash 集中到 `contracts.py`；renderer/projector 直接复用，删除重复实现。
- `ProjectedDocument` 强制 LF-only、恰一个末尾 LF，并验证 `content_hash == sha256(content_markdown.encode("utf-8"))`。
- `PlayerWorkspaceSnapshot` 强制 documents 按 available manifest 顺序排列并复算完整 manifest + document bytes 的 `workspace_hash`。
- `workspace_revision` 在既有 identity/availability/sources/renderer/estimator 基础上加入每个 pre-render plan 的 `required`、`visibility_class` 和 authority dependency digest。
- 负向证据覆盖缺失尾 LF、多余尾 LF、CR、伪造 content hash 与伪造 workspace hash；测试侧 workspace hash 计算为独立 literal algorithm。

### Finding 4：uncertainty total ordering

- alternatives 使用 `(value_id, confidence.value, tuple(sorted(support_refs)))` total stable key。
- permutation regression 使用两个相同 `value_id`、不同 confidence/support refs 的 alternatives，修复前 bytes/hash 随输入排列变化，修复后完全相同。

### Finding 5：available-empty COMMITMENTS

- 直接 workspace 测试证明 `commitment_records=()` 产生 available、non-required、empty-source `COMMITMENTS.md`，并断言精确 Markdown bytes；`None` 仍表示 capability absent。

### Finding 6：exact INDEX public fields

- 测试逐字段独立构造完整 expected `INDEX.md`，覆盖 section ID、availability、required、renderer、content hash、token estimate、estimator、visibility、source IDs、unavailable reason。
- 明确断言 INDEX 不包含 `source_identity`、`record_kind:`、view fingerprint 或任何 source content hash。

### Finding 7：cache fault/error priority ruling A

- fault-injection cache 先发生 read miss 和 write fault，再触发 required ROLE renderer failure。
- 直接断言最终仍是 sanitized `ProjectionRenderFailed`，cache fault 不提升为 `ProjectionBuildFailed`，也不泄漏 private cause/context。

### Findings 8～9：文档与 fresh gates

- `task-6-report.md` 与 canonical `handoff.md` 已更正为 final-fix 后事实；明确区分 implementation-only snapshot、reviewed Task 6 branch 与 final-fix implementation diff。
- 未实现 final review 已接受 defer 的 ledger minors，也未扩展架构 scanner 到完整 Python name resolution。

## 3. TDD RED / GREEN 证据

RED 命令：

```bash
conda run -n wofkill python -m pytest \
  tests/player_agents/test_observation_contracts.py \
  tests/player_agents/test_observation_rendering.py \
  tests/player_agents/test_workspace_projector.py \
  -k 'noncanonical_markdown or hash_not_matching_utf8_bytes or forged_ordered_workspace_hash or uncertainty_alternatives_with_equal_value_ids or available_empty_commitments or index_exact_public_fields or workspace_revision_covers_required_visibility_and_summary_authority or shared_cache_cannot_reuse_game_after_summary_authority_changes or game_visibility_is_mixed_and_shared_cache_isolates_viewers or cache_read_write_faults_do_not_mask_required_renderer_failure' \
  -o addopts='' -vv
```

RED 结果：exit 1，79 collected / 67 deselected / 12 selected，`8 failed, 4 passed`。

- failed：缺/多 trailing LF、伪造 content hash、伪造 workspace hash、equal-value uncertainty permutation、revision semantics、summary shared-cache staleness、GAME PUBLIC label。
- already passed：CR rejection、available-empty COMMITMENTS、exact INDEX、cache fault/error priority；这些是 review 要求补齐的直接 characterization evidence。

同一选择集 GREEN：exit 0，`12 passed, 67 deselected in 0.07s`。

扩大 GREEN：

- contracts/rendering/workspace 三文件：`79 passed in 0.12s`。
- observation/service/conformance/import boundary 组合：`173 passed in 0.35s`。

## 4. Fresh final verification

### Exact Task 6 focused suite

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

### Full repository

`conda run -n wofkill python -m pytest -q`：exit 0，进度到 100%；quiet/xdist 未打印 passed 汇总。

独立证据：

- `pytest --collect-only -qq`：exit 0；308 个逐文件计数求和为 6365 collected，不能称为 passed。
- `pytest -q -rs`：exit 0；明确列出 12 skips（1 shared-negation、9 PowerShell soak、1 PowerShell AST、1 real-provider smoke）。
- 两次 full run 的 warning summary 均为 10 条既有 `fastapi.testclient` `StarletteDeprecationWarning`。

### Boundary / diff snapshots

- final boundary scan 命中 26 个 deliberate literals：production observation 0、boundary test fixture 19、canonical handoff 7。
- `83eb1be..6afff34` implementation-only snapshot：19 files，7236 insertions，3 deletions。
- `83eb1be..3c8781c` reviewed Task 6 branch：21 files，7422 insertions，66 deletions。
- `3c8781c..7f3fc87` final-fix code/tests：6 files，426 insertions，59 deletions。
- `83eb1be..7f3fc87` final-fix implementation HEAD：21 files，7789 insertions，66 deletions。
- evidence/docs commit 后的 final branch diff `83eb1be..HEAD`：22 files，7975 insertions，66 deletions；`git status --short --branch` 仅打印 branch header，tracked worktree clean。该统计与 implementation-only snapshot 明确分开。

## 5. Files changed

Production：

- `werewolf_agent/player_agents/observation/contracts.py`
- `werewolf_agent/player_agents/observation/rendering.py`
- `werewolf_agent/player_agents/observation/workspace.py`

Tests：

- `tests/player_agents/test_observation_contracts.py`
- `tests/player_agents/test_observation_rendering.py`
- `tests/player_agents/test_workspace_projector.py`

Evidence/docs：

- `.superpowers/sdd/2026-07-31-autonomous-player-observation-projections/final-fix-report.md`
- `.superpowers/sdd/2026-07-31-autonomous-player-observation-projections/task-6-report.md`
- `handoff.md`

## 6. Self-review 与 concerns

- cache：summary digest 在 lookup 前计算并进入 workspace revision；没有依赖偶然 cache invalidation。
- privacy：GAME mixed visibility 与 viewer fingerprint cache key/identity validation 一致；INDEX 只投影允许公开的 manifest fields。
- integrity：contract validation 与 projector calculation 共用 helpers；测试 expected hash 独立实现，避免复制生产 helper 后同错。
- determinism：summary 与 uncertainty 都按渲染语义的 total/canonical order 计算依赖或 bytes。
- scope：没有改 authority schema、active-turn service、persistence、legacy/live、model/provider/tool 或下一里程碑代码。
- 已知外部 gate 未改变：没有真实 PostgreSQL service integration；这不是本次修复失败，也不应被误报为已覆盖。
- 未发现未解决的 final-review finding 或需要隐藏的 concern。
