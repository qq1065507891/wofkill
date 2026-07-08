# SRP File Splitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按单一职责、高内聚低耦合、显式优于隐式的原则，把剩余大文件拆成聚焦模块，并保持现有公开 API 和运行行为不变。

**Architecture:** 每个阶段先保留原文件作为兼容外壳，把一类职责迁移到新模块，再从原文件显式导入并重新导出需要兼容的符号。运行时核心文件优先拆，因为它们阻塞后续维护；测试、schema 和低风险工具文件放在后面。

**Tech Stack:** Python 3, pytest, ruff, compileall, CodeGraph, PowerShell, repo-local `.tmp` pytest basetemp。

---

## 执行状态快照

更新日期: 2026-07-08

已完成并提交:

- [x] Task 1: `a559c01 refactor: split agent adapter support modules`
- [x] Task 2: `a81b690 refactor: split runtime context builders`
- [x] Task 3: `2fa1be7 refactor: split night runtime nodes`
- [x] Task 4: `1461247 refactor: split player prompt builder`
- [x] Task 5: `692fde1 refactor: split werewolf skill handlers`
- [x] Task 6: `894fbbe refactor: split agent output parser`
- [x] Task 7: `8d5d848 refactor: split player agent pipeline`
- [x] Task 8: `daabb0d refactor: split reflection memory modules`
- [x] Task 9: `5da36bf refactor: split day runtime nodes`
- [x] Task 10: `7ec7de7 refactor: split game route helpers`
- [x] Task 11.2: `74fb02a refactor: split evaluation metrics`
- [x] Task 11.3: `9dd826f refactor: split model gateway router`
- [x] Task 11.4: `6d7946f refactor: split shared runtime node helpers`
- [x] Task 11.5: `c833c19 refactor: split game runner responsibilities`
- [x] Task 11.6: `dcb6a5f` 至 `57ecb4b`，按职责拆分 `RuleEngine`
- [x] Task 11.7: `7c834c2 refactor: split sheriff runtime nodes`
- [x] Task 11.8 / Task 14: `2b22ac4`、`8bc75e1`、`d82aab9`、`84d5779`，拆分 RAG retrieval/ranking/vector storage facade
- [x] Task 13: 当前批次最终全量验证、清理 `.tmp`、同步 CodeGraph

仍未完成:

- 无。

不执行项:

- [x] Task 12: 不拆 `tests/` 目录。2026-07-07 用户决策：测试目录保持现状，不做按生产边界拆分；后续只在新增功能或修复时就地维护相关测试。

代码级复核:

- [x] 2026-07-07 已逐项检查 Task 1-11.8 / Task 14 的实际代码边界，不依赖提交记录；确认承诺拆出的模块存在、旧 facade 仍导入/委托到拆分模块、主要兼容导出可从旧路径导入。
- [x] 复核结论：未发现已标记完成 task 的代码缺口。`agent_adapter.py`、`prompt_builder.py`、`player.py`、`games.py`、`metrics.py`、`router.py` 仍偏大，但已列入 Task 15-21 二次候选，不作为前序 task 未完成项。

下一轮候选池:

- 当前计划内候选已处理完毕。后续新增候选需基于新的行数和职责扫描另起计划。

2026-07-07 非测试文件标准复查:

- 范围：仅检查 tracked Python 文件，排除 `tests/` 目录；共扫描 313 个非测试 Python 文件。
- 结论：原 SRP 拆分计划没有遗留未完成 task；文件头规范问题已在本批修复，剩余 500 行以上文件属于新的 SRP 拆分候选，需要另起小批次处理并分别验证。
- Python 文件头顺序不合规已修复：`werewolf_agent/runtime/context_public_summary.py`、`werewolf_agent/runtime/reflection_prompt.py`。
- Python 文件头缺失或不完整已修复：`scripts/analyze_recent_balance.py`、`scripts/analyze_reflection_score.py`、`scripts/clear_memory.py`、`scripts/migrate_reflection_memory_v2.py`、`scripts/offline_belief_worlds_compare.py`、`scripts/print_game_audit.py`、`scripts/probe_ark_tool_support.py`、`scripts/probe_real_game_baidu_action.py`、`scripts/run_real_game.py`、`scripts/verify_ark_text_json.py`。
- 验证记录：按语义口径接受 `作者:` / `作者：` 与 `创建日期:` / `修改日期:` 后，313 个非测试 tracked Python 文件头复查通过。
- 后续批次已处理：`werewolf_agent/agents/prompt_builder.py` 已继续拆出输出协议、示例、choice 候选摘要和最终提交约束到 `werewolf_agent/agents/prompt_output.py`，继续拆出稳定 system prompt 区段到 `werewolf_agent/agents/prompt_system.py`，并拆出动态 user context 到 `werewolf_agent/agents/prompt_user_context.py`，旧 `PlayerPromptBuilder` 方法名和 current-game 清理 helper 导入保持兼容；`werewolf_agent/runtime/context.py` 已继续拆出跨局记忆 orchestration 到 `werewolf_agent/runtime/context_cross_game_memory.py`，并将角色统计聚合收敛为专用 helper，旧 `build_agent_context` 和 memory hint helper 导入保持兼容；`werewolf_agent/runtime/private_memory.py` 已继续拆出角色信息脱敏和站边目标解析到 `werewolf_agent/runtime/private_memory_safety.py`，旧安全 helper 和 `_ROLE_LABEL_CN` 等兼容导入保持可用；`werewolf_agent/runtime/agent_action_pipeline.py` 已继续拆出女巫、预言家、混血儿选主、警徽处置、猎人开枪等特殊/夜间行动到 `werewolf_agent/runtime/agent_special_actions.py`，并继续拆出日间发言/投票/遗言行动到 `agent_day_actions.py` facade、`agent_day_speech_actions.py` 和 `agent_day_vote_actions.py`，旧 pipeline 和 `agent_adapter` 导入路径保持兼容；`werewolf_agent/evaluation/metric_aggregation.py` 已继续拆出阵营、玩家、角色基础 outcome 指标到 `werewolf_agent/evaluation/metric_outcomes.py`，并拆出质量指标到 `werewolf_agent/evaluation/metric_quality.py`，旧 `MetricsAggregator._compute_*` 私有方法名保持兼容。
- 2026-07-08 特殊行动拆分验证记录：新增兼容测试先因缺少 `agent_special_actions` 失败；拆分后 `tests/runtime/test_agent_adapter.py`、special/night action 相关 runtime 测试集、touched-file `ruff F401,F841`、`compileall`、`git diff --check` 均返回 exit code 0。
- 2026-07-08 system prompt 拆分验证记录：新增兼容测试先因缺少 `prompt_system` 失败；拆分后 `tests/agents/test_prompt_sections.py`、prompt builder 相关测试集、touched-file `ruff F401,F841`、`compileall`、`git diff --check` 均返回 exit code 0。
- 2026-07-08 outcome metrics 拆分验证记录：新增兼容测试先因缺少 `metric_outcomes` 失败；拆分后 `tests/evaluation/test_metrics_split_helpers.py`、evaluation metrics 相关测试集、touched-file `ruff F401,F841`、`compileall`、`git diff --check` 均返回 exit code 0。
- 2026-07-08 day actions 拆分验证记录：新增兼容测试先因缺少 `agent_day_actions` 失败；拆分后 `tests/runtime/test_agent_adapter.py`、日间发言/投票相关 runtime 测试集、touched-file `ruff F401,F841`、`compileall`、`git diff --check` 均返回 exit code 0。
- 2026-07-08 quality metrics 拆分验证记录：新增兼容测试先因缺少 `metric_quality` 失败；拆分后 `tests/evaluation/test_metrics_split_helpers.py`、evaluation metrics 相关测试集、touched-file `ruff F401,F841`、`compileall`、`git diff --check` 均返回 exit code 0。
- 2026-07-08 prompt user context 拆分验证记录：新增兼容测试先因缺少 `prompt_user_context` 失败；拆分后 `tests/agents/test_prompt_builder.py::TestPromptUserContextSplit` 和 prompt builder 相关测试集通过，production touched-file `ruff F401,F841`、`compileall`、`git diff --check` 均返回 exit code 0；完整 `test_prompt_builder.py` 文件仍有既有测试文件 F401/F841 债务，本批不作为 Ruff gate。
- 2026-07-08 context cross-game memory 拆分验证记录：新增兼容测试先因缺少 `context_cross_game_memory` 失败；拆分后 `tests/runtime/test_context_memory_hints.py` 和 context 相关 runtime 测试集通过，touched-file `ruff F401,F841`、`compileall`、`git diff --check` 均返回 exit code 0；本批顺手优化为 `_role_stats_for_refs` helper，减少 `build_agent_context` 内联循环。
- 2026-07-08 private memory safety 拆分验证记录：新增兼容测试先因缺少 `private_memory_safety` 失败；拆分后 `tests/runtime/test_private_memory_sanitize.py` 和 `tests/memory/test_reflection.py` 通过，touched-file `ruff F401,F841`、`compileall`、`git diff --check` 均返回 exit code 0；`private_memory.py` 从 544 行降至 418 行，旧 `_sanitize_role_claims`、`_resolve_stance_target`、`_ROLE_LABEL_CN` 等导入保持兼容。
- 2026-07-08 player generation request 拆分验证记录：新增兼容测试先因缺少 `player_generation_request` 失败，再因缺少 `call_player_generation_request` 失败；拆分后 `tests/agents/test_player_agent.py` 通过，production touched-file `ruff F401,F841`、`compileall`、`git diff --check` 均返回 exit code 0；完整 `test_player_agent.py` 文件仍有既有测试文件 F401/F841 债务，本批不作为 Ruff gate；`player_action_flow.py` 从 561 行降至 555 行，provider 请求构建和实际调用已迁移到 `player_generation_request.py`。
- 2026-07-08 player action result 拆分验证记录：新增兼容测试先因缺少 `player_action_result` 失败；拆分后 `tests/agents/test_player_agent.py` 通过，production touched-file `ruff F401,F841`、`compileall`、`git diff --check` 均返回 exit code 0；完整 `test_player_agent.py` 文件仍有既有测试文件 F401/F841 债务，本批不作为 Ruff gate；`player_action_flow.py` 从 555 行降至 496 行，成功 action 与 fallback action 的 trace/metrics 收尾已迁移到 `player_action_result.py`。
- 2026-07-08 graph registration 拆分验证记录：新增兼容测试先因缺少 `graph_registration` 失败；拆分后 `tests/runtime/test_graph_lifecycle.py` 通过，production touched-file `ruff F401,F841`、`compileall`、`git diff --check` 均返回 exit code 0；`graph.py` 从 542 行降至 340 行，节点/边注册清单已迁移到 `graph_registration.py`，旧 `_add_all_nodes` / `_add_all_edges` 兼容入口保持可导入。
- 2026-07-08 sheriff election speech 拆分验证记录：新增兼容测试先因缺少 `agent_sheriff_speech_actions` 失败；拆分后 `tests/runtime/test_agent_adapter.py`、`tests/runtime/test_sheriff_flow.py`、`tests/runtime/test_sheriff_election_directives.py` 通过，production touched-file `ruff F401,F841`、`compileall`、`git diff --check` 均返回 exit code 0；`agent_sheriff_actions.py` 从 535 行降至 381 行，警长竞选发言适配迁移到 `agent_sheriff_speech_actions.py`，旧 `agent_sheriff_election_speech` 兼容入口保持可导入。
- 仍超过 500 行的非测试文件：`scripts/run_real_game.py`(579)、`werewolf_agent/agents/judge_hitl.py`(576)、`werewolf_agent/agents/action_schemas.py`(562)、`werewolf_agent/agents/judge.py`(557)、`werewolf_agent/evaluation/balance_audit.py`(520)、`werewolf_agent/agents/player.py`(516)、`werewolf_agent/agents/player_action_flow.py`(515)、`werewolf_agent/runtime/nodes/night_specialists.py`(509)。
- 建议顺序：下一步处理 `night_specialists.py` 这类 runtime 节点责任；随后处理 `agents/player_action_flow.py` / `agents/player.py` facade 或 `evaluation/balance_audit.py`。`private_memory.py`、`graph.py` 和 `agent_sheriff_actions.py` 已低于 500 行，脚本类文件可放在生产核心之后。

已完成候选:

- `werewolf_agent/rag/retriever.py` / `werewolf_agent/rag/vector_store.py`，Task 11.8 / Task 14 已拆分为 `query_processing.py`、`retrieval_ranking.py`、`local_vector_store.py`、`embedding_vector_store.py`，旧模块保留兼容 facade。
- `werewolf_agent/runtime/agent_adapter.py`，Task 15 已拆分为 `agent_action_pipeline.py`，旧模块保留兼容 facade；`agent_dispatch.py` / `agent_decision_contract.py` 本轮未创建，因为未识别出足够独立的清晰边界。
- `werewolf_agent/runtime/agent_action_pipeline.py`，2026-07-08 后续批次已拆分特殊/夜间行动到 `agent_special_actions.py`，并拆分日间行动到 `agent_day_actions.py` facade、`agent_day_speech_actions.py` 和 `agent_day_vote_actions.py`；旧 pipeline 与 `agent_adapter` 继续作为兼容入口重新导出原函数名，facade monkeypatch 会传播到新实现模块。
- `werewolf_agent/runtime/agent_sheriff_actions.py`，2026-07-08 后续批次已拆分警长竞选发言到 `agent_sheriff_speech_actions.py`；旧模块继续重新导出 `agent_sheriff_election_speech`，保持兼容入口可导入。
- `werewolf_agent/runtime/graph.py`，2026-07-08 后续批次已拆分节点/边注册清单到 `graph_registration.py`；旧 graph 模块继续保留 graph factory、setup 节点、route 函数和兼容 re-export。
- `werewolf_agent/agents/prompt_builder.py` / `werewolf_agent/agents/player.py`，Task 16 已拆分为 `prompt_composer.py` 和 `player_action_flow.py`；2026-07-08 后续批次已从 `player_action_flow.py` 拆出 provider 请求构建和调用到 `player_generation_request.py`，并拆出成功/fallback trace 与 metrics 收尾到 `player_action_result.py`；旧 `player_action_flow.py` 继续作为兼容入口重新导出这些 helper。
- `werewolf_agent/agents/prompt_builder.py`，2026-07-08 后续批次已拆分稳定 system prompt 区段到 `prompt_system.py`，并拆分动态 user context 到 `prompt_user_context.py`；旧 `PlayerPromptBuilder._build_*` 方法名继续通过 mixin 暴露。
- `werewolf_agent/runtime/context.py`，2026-07-08 后续批次已拆分跨局记忆 orchestration 到 `context_cross_game_memory.py`；旧 `build_agent_context` 继续作为兼容入口组装 AgentContext。
- `werewolf_agent/skills/good_skill_handlers.py` / `werewolf_agent/runtime/nodes/wolf_night_nodes.py`，Task 17 已拆分为 `good_claim_handlers.py`、`good_vote_handlers.py`、`good_power_handlers.py`、`wolf_discussion.py`、`wolf_consensus.py`，旧模块保留兼容 facade。
- `werewolf_agent/agents/schemas.py`，Task 18 已拆分为 `action_schemas.py`、`prompt_schemas.py`、`trace_schemas.py`，旧模块保留兼容 facade。
- `werewolf_agent/evaluation/metrics.py`，Task 19 已拆分为 `metric_aggregation.py` 和 `metric_reporting.py`，旧模块保留兼容 facade。
- `werewolf_agent/evaluation/metric_aggregation.py`，2026-07-08 后续批次已拆分基础 outcome 指标到 `metric_outcomes.py`，并拆分质量指标到 `metric_quality.py`；旧 `MetricsAggregator` 私有方法名继续作为类属性兼容入口。
- `werewolf_agent/api/routes/games.py`，Task 20 已拆分为 `game_lifecycle.py`、`game_commands.py` 和 `game_snapshots.py`，旧模块保留 `create_game_router` 与 helper 兼容 facade。
- `werewolf_agent/model_gateway/router.py`，Task 21 已拆分为 `router_config.py`、`router_selection.py` 和 `router_errors.py`，旧模块保留 `ModelRouter` 与 helper 兼容 facade。
- `werewolf_agent/engine/rule_engine.py`，RuleEngine 收尾已拆分 `ruleset_loader.py`，旧模块保留 `Ruleset` 与 `RuleEngine.from_yaml` 兼容入口。

## 拆分原则

- 原文件先作为 facade 保留，公共导入路径不变，避免一次性破坏调用方。
- 每次只迁移一个职责组，测试通过后提交。
- 新 Python 文件必须包含 UTF-8 编码声明和中文模块 docstring。
- 注释默认中文，只有协议字段、API 名称、第三方名词和必要 literal 保持英文。
- 优先拆生产代码，再同步拆对应测试文件。
- 保留兼容导出，尤其不要误删 `AGENT_TIMEOUTS`、`_inject_skill_output` 等兼容入口。
- 每阶段结束都运行 targeted tests、`compileall`、针对 touched files 的 `ruff F401,F841`、`git diff --check`。

## 总体执行顺序

- [x] 稳定基线和工作区边界。
- [x] 继续拆 `werewolf_agent/runtime/agent_adapter.py`。
- [x] 拆 `werewolf_agent/runtime/context.py`。
- [x] 拆 `werewolf_agent/runtime/nodes/night.py`。
- [x] 拆 `werewolf_agent/agents/prompt_builder.py`。
- [x] 拆 `werewolf_agent/skills/werewolf_skills.py`。
- [x] 拆 `werewolf_agent/agents/output_parser.py`。
- [x] 拆 `werewolf_agent/agents/player.py`。
- [x] 拆 `werewolf_agent/memory/reflection.py`。
- [x] 拆 `werewolf_agent/runtime/nodes/day.py`。
- [x] 拆 `werewolf_agent/api/routes/games.py`。
- [x] 拆第二批 600 行以上但风险较低的模块。RAG retrieval/storage 已完成 facade 拆分。
- [x] 不拆对应超大测试文件（2026-07-07 用户决策：`tests/` 目录保持现状）。
- [x] 当前批次最终全量验证、提交、CodeGraph 同步。
- [x] 下一轮候选模块进入正式拆分计划。

---

### Task 0: Baseline And Safety Rails

**Files:**
- Inspect: `werewolf_agent/runtime/agent_adapter.py`
- Inspect: `werewolf_agent/runtime/context.py`
- Inspect: `werewolf_agent/runtime/nodes/night.py`
- Inspect: `werewolf_agent/agents/prompt_builder.py`
- Inspect: `werewolf_agent/skills/werewolf_skills.py`
- Inspect: `werewolf_agent/agents/output_parser.py`
- Inspect: `werewolf_agent/agents/player.py`
- Inspect: `werewolf_agent/memory/reflection.py`
- Inspect: `werewolf_agent/runtime/nodes/day.py`
- Inspect: `werewolf_agent/api/routes/games.py`

- [ ] **Step 1: Confirm CodeGraph is current**

Run:

```powershell
codegraph status .
```

Expected: `[OK] Index is up to date`.

- [ ] **Step 2: Capture dirty worktree before edits**

Run:

```powershell
git status --short
```

Expected: unrelated user changes may exist; do not revert them.

- [ ] **Step 3: Run characterization suite for current refactor surface**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/runtime tests/agents tests/skills tests/memory tests/api -q
```

Expected: current known baseline. If too broad or slow, record failures and use focused suites per task.

- [ ] **Step 4: Commit only if there are plan-only changes to save**

Run:

```powershell
git add docs/superpowers/plans/2026-07-05-srp-file-splitting.md
git commit -m "docs: plan srp file splitting"
```

Expected: optional docs commit; skip if the user wants implementation commits only.

---

### Task 1: Split Remaining Agent Adapter Runtime Responsibilities

**Files:**
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Create: `werewolf_agent/runtime/agent_registry.py`
- Create: `werewolf_agent/runtime/agent_action_audit.py`
- Create: `werewolf_agent/runtime/wolf_kill_support.py`
- Create: `werewolf_agent/runtime/agent_reflection_support.py`
- Test: `tests/runtime/test_agent_registry.py`
- Test: `tests/runtime/test_agent_action_audit.py`
- Test: `tests/runtime/test_wolf_kill_support.py`
- Test: `tests/runtime/test_agent_reflection_support.py`
- Existing tests: `tests/runtime/test_witch_flow.py`, `tests/runtime/test_night_flow.py`, `tests/runtime/test_day_discussion.py`, `tests/runtime/test_vote_flow.py`, `tests/runtime/test_pk_flow.py`, `tests/runtime/test_sheriff_flow.py`, `tests/runtime/test_hunter_flow.py`

- [x] **Step 1: Write characterization tests for registry behavior**

Test `AgentRegistry` and `SimpleAgentRegistry` import from the old module and new module. Verify the same agent lookup behavior.

- [x] **Step 2: Move registry types**

Move `AgentRegistry` and `SimpleAgentRegistry` into `agent_registry.py`. In `agent_adapter.py`, import and re-export them.

- [x] **Step 3: Move audit helper functions**

Move `_audit_context_kwargs`, `_seer_credibility_audit_payload`, `_is_sheriff_silenced`, and vote-basis hint helpers into `agent_action_audit.py`.

- [x] **Step 4: Move wolf kill support**

Move `_build_wolf_kill_directive` and `_single_wolf_vote` into `wolf_kill_support.py`. Keep public compatibility imports in `agent_adapter.py`.

- [x] **Step 5: Move reflection support**

Move `_strip_in_game_directives` and `_agent_reflection` into `agent_reflection_support.py`.

- [x] **Step 6: Run focused runtime adapter tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/runtime/test_agent_registry.py tests/runtime/test_agent_action_audit.py tests/runtime/test_wolf_kill_support.py tests/runtime/test_agent_reflection_support.py tests/runtime/test_witch_flow.py tests/runtime/test_night_flow.py tests/runtime/test_day_discussion.py tests/runtime/test_vote_flow.py tests/runtime/test_pk_flow.py tests/runtime/test_sheriff_flow.py tests/runtime/test_hunter_flow.py -q
```

Expected: all selected tests pass.

- [x] **Step 7: Verify and commit**

Run:

```powershell
python -m compileall -q werewolf_agent/runtime
python -m ruff check werewolf_agent/runtime/agent_adapter.py werewolf_agent/runtime/agent_registry.py werewolf_agent/runtime/agent_action_audit.py werewolf_agent/runtime/wolf_kill_support.py werewolf_agent/runtime/agent_reflection_support.py tests/runtime/test_agent_registry.py tests/runtime/test_agent_action_audit.py tests/runtime/test_wolf_kill_support.py tests/runtime/test_agent_reflection_support.py --select F401,F841
git diff --check
git add werewolf_agent/runtime/agent_adapter.py werewolf_agent/runtime/agent_registry.py werewolf_agent/runtime/agent_action_audit.py werewolf_agent/runtime/wolf_kill_support.py werewolf_agent/runtime/agent_reflection_support.py tests/runtime/test_agent_registry.py tests/runtime/test_agent_action_audit.py tests/runtime/test_wolf_kill_support.py tests/runtime/test_agent_reflection_support.py
git commit -m "refactor: split agent adapter support modules"
```

---

### Task 2: Split Agent Context Construction

**Files:**
- Modify: `werewolf_agent/runtime/context.py`
- Create: `werewolf_agent/runtime/context_persona.py`
- Create: `werewolf_agent/runtime/context_rag.py`
- Create: `werewolf_agent/runtime/context_memory_hints.py`
- Create: `werewolf_agent/runtime/context_skill_advice.py`
- Create: `werewolf_agent/runtime/context_strategy_directives.py`
- Create: `werewolf_agent/runtime/context_action_trace.py`
- Test: `tests/runtime/test_context_persona.py`
- Test: `tests/runtime/test_context_rag.py`
- Test: `tests/runtime/test_context_memory_hints.py`
- Test: `tests/runtime/test_context_skill_advice.py`
- Test: `tests/runtime/test_context_strategy_directives.py`
- Existing tests: `tests/runtime/test_context.py`, `tests/runtime/test_strategy_directives.py`

- [x] **Step 1: Write characterization tests for compatibility imports**

Verify `build_agent_context` and `_inject_skill_output` remain importable from `werewolf_agent.runtime.context`.

- [x] **Step 2: Move persona helpers**

Move `_load_persona_profile`, `_get_persona_speech_style`, and `_get_persona_task_style` into `context_persona.py`.

- [x] **Step 3: Move RAG helpers**

Move `_rag_phase_for_task`, `_normalize_legal_actions_to_tags`, `_inject_seed_rag_hints`, `_extract_suspects`, `_extract_trusts`, `_extract_role_claim`, `_extract_vote_intent`, and `_first_sentence` into `context_rag.py`.

- [x] **Step 4: Move memory and cognition hints**

Move `_profile_memory_hint`, `_reflection_memory_hints`, `_evidence_id_ref`, and `_cognition_matrix_hint` into `context_memory_hints.py`.

- [x] **Step 5: Move skill advice conversion**

Move `_skill_output_to_advice_frame`, `_skill_advice_frame_to_prompt_dict`, and `_inject_skill_output` into `context_skill_advice.py`; re-export `_inject_skill_output` from `context.py`.

- [x] **Step 6: Move strategy directive merging**

Move `_merge_strategy_directive`, `_directive_size`, and `_cap_strategy_directive` into `context_strategy_directives.py`.

- [x] **Step 7: Move action trace payload construction**

Move `_action_trace_payload` and direct payload helpers into `context_action_trace.py`.

- [x] **Step 8: Keep `context.py` as orchestration facade**

Leave `build_agent_context` in `context.py`, but make dependencies explicit through imports.

- [x] **Step 9: Run focused context tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/runtime/test_context.py tests/runtime/test_context_persona.py tests/runtime/test_context_rag.py tests/runtime/test_context_memory_hints.py tests/runtime/test_context_skill_advice.py tests/runtime/test_context_strategy_directives.py tests/runtime/test_strategy_directives.py -q
```

Expected: all selected tests pass.

- [x] **Step 10: Verify and commit**

Run compileall, touched-file ruff, `git diff --check`, then commit:

```powershell
git commit -m "refactor: split runtime context builders"
```

---

### Task 3: Split Night Runtime Nodes

**Files:**
- Modify: `werewolf_agent/runtime/nodes/night.py`
- Create: `werewolf_agent/runtime/nodes/night_entry.py`
- Create: `werewolf_agent/runtime/nodes/night_specialists.py`
- Create: `werewolf_agent/runtime/nodes/night_resolution.py`
- Create: `werewolf_agent/runtime/nodes/wolf_night_nodes.py`
- Test: `tests/runtime/test_night_entry.py`
- Test: `tests/runtime/test_night_specialists.py`
- Test: `tests/runtime/test_night_resolution.py`
- Test: `tests/runtime/test_wolf_night_nodes.py`
- Existing tests: `tests/runtime/test_night_flow.py`, `tests/runtime/test_witch_flow.py`, `tests/runtime/test_wolf_flow.py`

- [x] **Step 1: Characterize public imports from `night.py`**

Verify existing imports for `enter_night`, `night_witch`, `night_seer`, `resolve_night`, `wolf_discussion`, `wolf_team_plan_node`, and `wolf_consensus` still work.

- [x] **Step 2: Move night entry and simple status nodes**

Move `enter_night`, `night_hunter_idiot_status`, and shared entry helpers to `night_entry.py`.

- [x] **Step 3: Move specialist night actions**

Move `night_witch`, `night_seer`, and `first_night_hybrid_master` to `night_specialists.py`.

- [x] **Step 4: Move night resolution**

Move `resolve_night` and death/victory-related night resolution helpers to `night_resolution.py`.

- [x] **Step 5: Move wolf night planning and consensus**

Move `_legacy_wolf_consensus`, `wolf_discussion`, `_build_fallback_wolf_team_plan`, `wolf_team_plan_node`, and `wolf_consensus` to `wolf_night_nodes.py`.

- [x] **Step 6: Convert `night.py` into explicit facade**

Import and re-export all public node functions. Avoid wildcard imports.

- [x] **Step 7: Run focused night tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/runtime/test_night_entry.py tests/runtime/test_night_specialists.py tests/runtime/test_night_resolution.py tests/runtime/test_wolf_night_nodes.py tests/runtime/test_night_flow.py tests/runtime/test_witch_flow.py tests/runtime/test_wolf_flow.py tests/runtime/test_hybrid_master_directives.py -q
```

Expected: all selected tests pass.

- [x] **Step 8: Verify and commit**

Commit message:

```powershell
git commit -m "refactor: split night runtime nodes"
```

---

### Task 4: Split Prompt Builder

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py`
- Create: `werewolf_agent/agents/prompt_sections.py`
- Create: `werewolf_agent/agents/prompt_persona.py`
- Create: `werewolf_agent/agents/prompt_memory.py`
- Create: `werewolf_agent/agents/prompt_strategy.py`
- Create: `werewolf_agent/agents/prompt_salience.py`
- Test: `tests/agents/test_prompt_sections.py`
- Test: `tests/agents/test_prompt_persona.py`
- Test: `tests/agents/test_prompt_memory.py`
- Test: `tests/agents/test_prompt_strategy.py`
- Existing tests: `tests/agents/test_prompt_builder.py`, `tests/agents/test_prompt_formatting.py`

- [x] **Step 1: Characterize final prompt output**

Add focused snapshots or exact string assertions for system prompt, user prompt, persona hints, memory hints, and strategy directives.

- [x] **Step 2: Move section model and section rendering**

Move `_SectionSpec`, section ordering, section formatting, and common section assembly helpers into `prompt_sections.py`.

- [x] **Step 3: Move persona prompt rendering**

Move persona style and persona text rendering into `prompt_persona.py`.

- [x] **Step 4: Move memory prompt rendering**

Move profile/reflection/cognition memory prompt rendering into `prompt_memory.py`.

- [x] **Step 5: Move strategy and directive prompt rendering**

Move strategy directive rendering into `prompt_strategy.py`.

- [x] **Step 6: Move salience helpers**

Move `_slim_salience_item` and salience-specific formatting into `prompt_salience.py`.

- [x] **Step 7: Keep `PlayerPromptBuilder` as a small facade**

`PlayerPromptBuilder` should coordinate inputs and call focused renderers. It should not own low-level formatting logic.

- [x] **Step 8: Run focused prompt tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/agents/test_prompt_builder.py tests/agents/test_prompt_formatting.py tests/agents/test_prompt_sections.py tests/agents/test_prompt_persona.py tests/agents/test_prompt_memory.py tests/agents/test_prompt_strategy.py -q
```

Expected: all selected tests pass.

- [x] **Step 9: Verify and commit**

Commit message:

```powershell
git commit -m "refactor: split player prompt builder"
```

---

### Task 5: Split Werewolf Skill Handlers

**Files:**
- Modify: `werewolf_agent/skills/werewolf_skills.py`
- Create: `werewolf_agent/skills/advice_frames.py`
- Create: `werewolf_agent/skills/skill_context.py`
- Create: `werewolf_agent/skills/good_skill_handlers.py`
- Create: `werewolf_agent/skills/wolf_skill_handlers.py`
- Create: `werewolf_agent/skills/review_skill_handlers.py`
- Create: `werewolf_agent/skills/skill_handler_registry.py`
- Test: `tests/skills/test_advice_frames.py`
- Test: `tests/skills/test_skill_context.py`
- Test: `tests/skills/test_good_skill_handlers.py`
- Test: `tests/skills/test_wolf_skill_handlers.py`
- Existing tests: `tests/skills/test_werewolf_skills.py`, `tests/skills/test_skill_manifest_loader.py`, `tests/skills/test_registry.py`

- [x] **Step 1: Characterize `apply_skill` behavior**

Add tests for unchanged output for at least one good skill, one wolf skill, and one review correction skill.

- [x] **Step 2: Move advice frame helpers**

Move `_cap_prompt_injectable`, `_advice_frame`, `_push_vote_advice_frame`, `_counter_claim_advice_frame`, `_hide_identity_advice_frame`, `_generic_skill_advice_frame`, and `_ensure_skill_advice_frame` into `advice_frames.py`.

- [x] **Step 3: Move skill context extraction**

Move `_count_seer_claimants`, `_get_seer_claimants`, `_alive_wolves`, `_alive_non_wolves`, `_vote_targets_for_player`, `_seer_checks_on_target`, `_alerts_for_player`, `_belief_top_suspects`, and `_wolf_teammates_exposed` into `skill_context.py`.

- [x] **Step 4: Move good-side handlers**

Move `bold_claim_handler`, `counter_claim_handler`, `push_vote_handler`, `swing_vote_handler`, `find_power_handler`, `protect_power_handler`, `resist_push_handler`, and `last_words_handler` into `good_skill_handlers.py` where appropriate.

- [x] **Step 5: Move wolf-side handlers**

Move `deep_hook_handler`, `hide_identity_handler`, and `wolf_pit_handler` into `wolf_skill_handlers.py`.

- [x] **Step 6: Move review correction handlers**

Move `review_correction_handler`, `_review_correction_wolf`, and `_review_correction_good` into `review_skill_handlers.py`.

- [x] **Step 7: Move handler registry**

Move `register_handler`, `get_handler`, and default handler registration into `skill_handler_registry.py`. Keep `apply_skill` as the public facade in `werewolf_skills.py`.

- [x] **Step 8: Run focused skill tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/skills/test_werewolf_skills.py tests/skills/test_advice_frames.py tests/skills/test_skill_context.py tests/skills/test_good_skill_handlers.py tests/skills/test_wolf_skill_handlers.py tests/skills/test_skill_manifest_loader.py tests/skills/test_registry.py -q
```

Expected: all selected tests pass.

- [x] **Step 9: Verify and commit**

Commit message:

```powershell
git commit -m "refactor: split werewolf skill handlers"
```

---

### Task 6: Split Output Parser

**Files:**
- Modify: `werewolf_agent/agents/output_parser.py`
- Create: `werewolf_agent/agents/action_data_extraction.py`
- Create: `werewolf_agent/agents/action_normalization.py`
- Create: `werewolf_agent/agents/action_repair.py`
- Create: `werewolf_agent/agents/speech_intent_parser.py`
- Create: `werewolf_agent/agents/choice_pipeline.py`
- Test: `tests/agents/test_action_data_extraction.py`
- Test: `tests/agents/test_action_normalization.py`
- Test: `tests/agents/test_action_repair.py`
- Test: `tests/agents/test_speech_intent_parser.py`
- Existing tests: `tests/agents/test_output_parser.py`, `tests/agents/test_parse_dispatch.py`

- [x] **Step 1: Characterize `parse_action` and public helper imports**

Verify imports from `output_parser.py` continue to work.

- [x] **Step 2: Move raw extraction helpers**

Move `extract_parameter_tag_action`, `extract_partial_decision_data`, and `extract_decision_data` into `action_data_extraction.py`.

- [x] **Step 3: Move normalization helpers**

Move `normalize_action_data`, `_normalize_typos`, `clean_enum_value`, `clean_reason`, and optional-private-field sanitizers into `action_normalization.py`.

- [x] **Step 4: Move repair helpers**

Move vote, target, and speech repair helpers into `action_repair.py`.

- [x] **Step 5: Move speech intent helpers**

Move `infer_speech_intent`, seer stance inference, vote basis inference, and speech pressure helpers into `speech_intent_parser.py`.

- [x] **Step 6: Move choice pipeline helpers**

Move `parse_choice_action`, `parse_speech_intent_action`, `uses_choice_pipeline`, and `uses_speech_intent_pipeline` into `choice_pipeline.py`.

- [x] **Step 7: Keep `output_parser.py` as parser facade**

`parse_action` remains the main entry point and delegates to focused modules.

- [x] **Step 8: Run focused parser tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/agents/test_output_parser.py tests/agents/test_parse_dispatch.py tests/agents/test_action_data_extraction.py tests/agents/test_action_normalization.py tests/agents/test_action_repair.py tests/agents/test_speech_intent_parser.py -q
```

Expected: all selected tests pass.

- [x] **Step 9: Verify and commit**

Commit message:

```powershell
git commit -m "refactor: split agent output parser"
```

---

### Task 7: Split Player Agent Pipeline

**Files:**
- Modify: `werewolf_agent/agents/player.py`
- Create: `werewolf_agent/agents/player_failures.py`
- Create: `werewolf_agent/agents/player_latency.py`
- Create: `werewolf_agent/agents/player_generation.py`
- Create: `werewolf_agent/agents/player_retry.py`
- Test: `tests/agents/test_player_failures.py`
- Test: `tests/agents/test_player_latency.py`
- Test: `tests/agents/test_player_generation.py`
- Existing tests: `tests/agents/test_player_agent.py`, `tests/integration/test_live_runtime.py`, `tests/integration/test_live_game_flow.py`

- [x] **Step 1: Characterize `PlayerAgent.act` behavior**

Add tests for success path, provider failure fallback, invalid action retry, and latency recording if not already covered.

- [x] **Step 2: Move failure classification**

Move `_fallback_reason`, `_categorize_failure_category`, and related failure metadata into `player_failures.py`.

- [x] **Step 3: Move latency helpers**

Move `_latency_from_result` and timing extraction helpers into `player_latency.py`.

- [x] **Step 4: Move provider generation wrapper**

Move model call orchestration into `player_generation.py`, keeping dependencies explicit.

- [x] **Step 5: Move retry/fallback pipeline helpers**

Move retry decision and fallback construction logic into `player_retry.py`.

- [x] **Step 6: Keep `PlayerAgent` as coordinator**

`player.py` should own public class lifecycle and high-level `act`, not low-level failure parsing or prompt/generation plumbing.

- [x] **Step 7: Run focused player tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/agents/test_player_agent.py tests/agents/test_player_failures.py tests/agents/test_player_latency.py tests/agents/test_player_generation.py tests/integration/test_live_runtime.py tests/integration/test_live_game_flow.py -q
```

Expected: all selected tests pass.

- [x] **Step 8: Verify and commit**

Commit message:

```powershell
git commit -m "refactor: split player agent pipeline"
```

---

### Task 8: Split Reflection Memory Module

**Files:**
- Modify: `werewolf_agent/memory/reflection.py`
- Create: `werewolf_agent/memory/reflection_sanitization.py`
- Create: `werewolf_agent/memory/reflection_quality.py`
- Create: `werewolf_agent/memory/reflection_synthesis.py`
- Create: `werewolf_agent/memory/reflection_repository.py`
- Test: `tests/memory/test_reflection_sanitization.py`
- Test: `tests/memory/test_reflection_quality.py`
- Test: `tests/memory/test_reflection_synthesis.py`
- Existing tests: `tests/memory/test_reflection_v2.py`, `tests/memory/test_memory.py`

- [x] **Step 1: Characterize public reflection classes**

Verify `ReflectionQualityGate`, `ReflectionSynthesizer`, and `ReflectionMemory` are still importable from `reflection.py`.

- [x] **Step 2: Move text sanitization**

Move `_iter_section_items`, `_scrub_ids`, and `_cap_source_text` into `reflection_sanitization.py`.

- [x] **Step 3: Move quality gate**

Move `ReflectionQualityGate` and quality scoring helpers into `reflection_quality.py`.

- [x] **Step 4: Move synthesizer**

Move `ReflectionSynthesizer` into `reflection_synthesis.py`.

- [x] **Step 5: Move storage wrapper**

Move `ReflectionMemory` into `reflection_repository.py`.

- [x] **Step 6: Keep `reflection.py` as compatibility facade**

Explicitly import and re-export public classes and helpers still used by callers.

- [x] **Step 7: Run focused memory tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/memory/test_reflection_v2.py tests/memory/test_memory.py tests/memory/test_reflection_sanitization.py tests/memory/test_reflection_quality.py tests/memory/test_reflection_synthesis.py -q
```

Expected: all selected tests pass.

- [x] **Step 8: Verify and commit**

Commit message:

```powershell
git commit -m "refactor: split reflection memory modules"
```

---

### Task 9: Split Day Runtime Nodes

**Files:**
- Modify: `werewolf_agent/runtime/nodes/day.py`
- Create: `werewolf_agent/runtime/nodes/day_deaths.py`
- Create: `werewolf_agent/runtime/nodes/day_discussion.py`
- Create: `werewolf_agent/runtime/nodes/day_vote.py`
- Create: `werewolf_agent/runtime/nodes/day_finish.py`
- Test: `tests/runtime/test_day_deaths.py`
- Test: `tests/runtime/test_day_vote_nodes.py`
- Existing tests: `tests/runtime/test_day_discussion.py`, `tests/runtime/test_vote_flow.py`, `tests/runtime/test_pk_flow.py`

- [x] **Step 1: Characterize public imports from `day.py`**

Verify public node functions remain importable.

- [x] **Step 2: Move death announcement and last words**

Move `_death_reason_label`, `announce_deaths`, `announce_deaths_with_badge_loss`, `night_death_last_words`, and `exile_last_words` into `day_deaths.py`.

- [x] **Step 3: Move discussion node**

Move `free_discussion` into `day_discussion.py`.

- [x] **Step 4: Move vote and exile nodes**

Move `day_vote`, `_broadcast_vote_details`, `resolve_vote`, and `resolve_exile` into `day_vote.py`.

- [x] **Step 5: Move game finish**

Move `check_victory` and `finish_game` into `day_finish.py`.

- [x] **Step 6: Keep `day.py` as explicit facade**

Re-export public node functions without wildcard imports.

- [x] **Step 7: Run focused day tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/runtime/test_day_discussion.py tests/runtime/test_vote_flow.py tests/runtime/test_pk_flow.py tests/runtime/test_day_deaths.py tests/runtime/test_day_vote_nodes.py -q
```

Expected: all selected tests pass.

- [x] **Step 8: Verify and commit**

Commit message:

```powershell
git commit -m "refactor: split day runtime nodes"
```

---

### Task 10: Split Games API Route Module

**Files:**
- Modify: `werewolf_agent/api/routes/games.py`
- Create: `werewolf_agent/api/routes/game_auth.py`
- Create: `werewolf_agent/api/routes/game_persistence.py`
- Create: `werewolf_agent/api/routes/game_cognition_views.py`
- Create: `werewolf_agent/api/routes/game_public_share.py`
- Test: `tests/api/test_game_auth_helpers.py`
- Test: `tests/api/test_game_persistence_helpers.py`
- Existing tests: `tests/api/test_api.py`, `tests/api/test_auth.py`, `tests/api/test_views.py`, `tests/api/test_pause_resume_lock.py`

- [x] **Step 1: Characterize route registration**

Verify `create_game_router` exposes the same routes before and after refactor.

- [x] **Step 2: Move auth helpers**

Move `_resolve_caller_role`, `_enforce_moderator_only`, and `_enforce_create_game_auth` into `game_auth.py`.

- [x] **Step 3: Move persistence helpers**

Move `_get_game` and `_persist` into `game_persistence.py`.

- [x] **Step 4: Move cognition view helpers**

Move `_build_cognition_data_for_viewer` and `_build_locked_config_snapshot` into `game_cognition_views.py`.

- [x] **Step 5: Move public share helpers**

Move `_event_is_public_for_share` and `_pick_public_mvp_candidate` into `game_public_share.py`.

- [x] **Step 6: Keep `games.py` as route declaration module**

`games.py` should define routes and dependency wiring only.

- [x] **Step 7: Run focused API tests**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/api/test_api.py tests/api/test_auth.py tests/api/test_views.py tests/api/test_pause_resume_lock.py tests/api/test_game_auth_helpers.py tests/api/test_game_persistence_helpers.py -q
```

Expected: all selected tests pass.

- [x] **Step 8: Verify and commit**

Commit message:

```powershell
git commit -m "refactor: split game route helpers"
```

---

### Task 11: Split Remaining Large Support Modules

**Files:**
- Modify: `werewolf_agent/evaluation/metrics.py`
- Modify: `werewolf_agent/model_gateway/router.py`
- Modify: `werewolf_agent/runtime/nodes/_shared.py`
- Modify: `werewolf_agent/runtime/game_runner.py`
- Modify: `werewolf_agent/engine/rule_engine.py`
- Modify: `werewolf_agent/runtime/nodes/sheriff.py`
- Modify: `werewolf_agent/rag/retriever.py`
- Modify: `werewolf_agent/rag/vector_store.py`
- Test: matching focused tests under `tests/evaluation`, `tests/runtime`, `tests/rules`, `tests/rag`

- [x] **Step 1: Re-rank these files after Tasks 1-10**

Run:

```powershell
$rows = Get-ChildItem -Recurse -File -Filter *.py werewolf_agent | ForEach-Object {
    $text = [System.IO.File]::ReadAllText($_.FullName)
    [PSCustomObject]@{
        Lines = ($text -split "`r?`n").Count
        Path = $_.FullName.Substring((Get-Location).Path.Length + 1)
    }
}
$rows | Where-Object { $_.Lines -ge 600 } | Sort-Object Lines -Descending
```

Expected: updated candidate list.

- [x] **Step 2: Split `metrics.py`**

Create `evaluation/claim_metrics.py`, `evaluation/pace_metrics.py`, and `evaluation/world_model_metrics.py`; keep `MetricsAggregator` facade.

- [x] **Step 3: Split `model_gateway/router.py`**

Create `model_gateway/provider_call.py`, `model_gateway/retry_policy.py`, and `model_gateway/usage_records.py`; keep `ModelRouter` facade.

- [x] **Step 4: Split `runtime/nodes/_shared.py`**

Create `runtime/nodes/runtime_state.py`, `runtime/nodes/action_audit.py`, and `runtime/nodes/node_helpers.py`; preserve compatibility imports for shared node helpers.

- [x] **Step 5: Split `runtime/game_runner.py`**

Create `runtime/game_runner_config.py`, `runtime/game_runner_setup.py`, and `runtime/game_runner_execution.py`; keep `GameRunner` facade.

- [x] **Step 6: Split `engine/rule_engine.py` only after rule tests are stable**

Create focused helpers including `rule_flow.py`, `rule_special_roles.py`, `rule_vote.py`, `rule_last_words.py`, `rule_visibility.py`, `rule_victory.py`, `rule_exile.py`, `rule_death.py`, and `rule_night.py`; keep `RuleEngine` facade.

- [x] **Step 7: Split `runtime/nodes/sheriff.py`**

Create `sheriff_registration_node.py`, `sheriff_vote_node.py`, and `sheriff_speech_node.py`; keep `sheriff.py` facade.

- [ ] **Step 8: Split RAG retriever/vector store modules**

Create focused modules for tokenization, local vector storage, embedding vector storage, and provider-specific vector stores.

- [ ] **Step 9: Run matching focused suites and commit per file group**

Commit each file group independently:

```powershell
git commit -m "refactor: split evaluation metrics"
git commit -m "refactor: split model gateway router"
git commit -m "refactor: split shared runtime node helpers"
git commit -m "refactor: split game runner helpers"
git commit -m "refactor: split rule engine responsibilities"
git commit -m "refactor: split sheriff runtime nodes"
git commit -m "refactor: split rag retrieval storage"
```

---

### Task 12: Split Oversized Test Files To Match Production Boundaries

**状态：不执行。** 2026-07-07 用户决策：`tests/` 目录不要拆分。保留本节作为历史候选记录，后续 agent 不应按下面步骤移动测试文件；只有在新增功能或修复缺陷时，才就地维护相关测试。

**Files:**
- Modify: `tests/agents/test_prompt_builder.py`
- Modify: `tests/runtime/test_strategy_directives.py`
- Modify: `tests/agents/test_player_agent.py`
- Modify: `tests/runtime/test_context.py`
- Modify: `tests/rag/test_rag.py`
- Create: focused test files matching newly created production modules.

- [x] **Step 1: Split prompt builder tests（已取消，不执行）**

Move section, persona, memory, strategy, and salience tests into dedicated files created in Task 4.

- [x] **Step 2: Split strategy directive tests（已取消，不执行）**

Group tests by directive module or role domain.

- [x] **Step 3: Split player agent tests（已取消，不执行）**

Group tests by success path, fallback/retry, latency, and provider failure behavior.

- [x] **Step 4: Split context tests（已取消，不执行）**

Move tests to files created in Task 2.

- [x] **Step 5: Split RAG tests（已取消，不执行）**

Group ingestion, retrieval, vector storage, prompt rendering, and hardening tests by module.

- [x] **Step 6: Run moved test suites（已取消，不执行）**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/agents tests/runtime tests/rag -q
```

Expected: same pass/fail baseline as before moving tests.

- [x] **Step 7: Verify and commit（已取消，不执行）**

Commit message:

```powershell
git commit -m "test: split oversized characterization suites"
```

---

### Task 13: Current Batch Final Verification And CodeGraph Sync

**Files:**
- All changed production and test files.

- [x] **Step 1: Run broad targeted suite**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/runtime tests/agents tests/skills tests/memory tests/api tests/evaluation tests/rag tests/rules -q
```

Expected: all relevant tests pass or known unrelated baseline failures are documented.

Result 2026-07-07: 初次运行发现 `tests/agents/test_model_router.py::TestModelRouter::test_usage_logging` 及兼容重导入用例仍硬编码 `openai` provider；当前 `config/models.yaml` 将 `p01/speech` 解析为 `minimax`，已改为按 `resolve_config()` 注册实际 provider。修复后本命令退出码 0。

- [x] **Step 2: Compile source**

Run:

```powershell
python -m compileall -q werewolf_agent
```

Expected: no output and exit code 0.

Result 2026-07-07: exit code 0.

- [x] **Step 3: Run touched-file ruff**

Run:

```powershell
$files = git diff --name-only HEAD~1..HEAD
python -m ruff check $files --select F401,F841
```

Expected: `All checks passed!`

Result 2026-07-07: `python -m ruff check tests/agents/test_model_router.py --select F401,F841` 通过。全量 tests 目录存在历史 F401/F841 债务，未作为本批次 gate。

- [x] **Step 4: Check whitespace**

Run:

```powershell
git diff --check
```

Expected: no output and exit code 0.

Result 2026-07-07: exit code 0.

- [x] **Step 5: Clean repo-local temp artifacts**

Run:

```powershell
$workspace = (Resolve-Path -LiteralPath .).Path
$tmpPath = Join-Path $workspace ".tmp"
if (Test-Path -LiteralPath $tmpPath) {
    $tmp = (Resolve-Path -LiteralPath $tmpPath).Path
    if (-not $tmp.StartsWith($workspace, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove outside workspace: $tmp"
    }
    Remove-Item -LiteralPath $tmp -Recurse -Force
}
```

Expected: `.tmp` removed.

Result 2026-07-07: `git status --short` 未显示 `.tmp`；repo-local pytest basetemp 未留下待清理的 untracked `.tmp`。

- [x] **Step 6: Sync CodeGraph**

Run:

```powershell
codegraph sync .
codegraph status .
```

Expected: `[OK] Index is up to date`.

Result 2026-07-07: `codegraph.cmd sync .` 完成，`codegraph.cmd status .` 显示 `[OK] Index is up to date`。

- [x] **Step 7: Final commit if needed**

Run:

```powershell
git status --short
git log --oneline -n 10
```

Expected: only intended changes committed; unrelated user changes remain untouched.

---

### Task 14: Split RAG Retrieval And Vector Storage

**Files:**
- Modify: `werewolf_agent/rag/retriever.py`
- Modify: `werewolf_agent/rag/vector_store.py`
- Candidate create: `werewolf_agent/rag/query_processing.py`
- Candidate create: `werewolf_agent/rag/retrieval_ranking.py`
- Candidate create: `werewolf_agent/rag/local_vector_store.py`
- Candidate create: `werewolf_agent/rag/embedding_vector_store.py`
- Test: `tests/rag/test_rag.py`
- Test: `tests/rag/test_retrieval_eval.py`
- Test: `tests/rag/test_clients.py`
- Candidate test: `tests/rag/test_query_processing.py`
- Candidate test: `tests/rag/test_vector_store_split.py`

- [x] **Step 1: Inspect current RAG boundaries with CodeGraph**

Run:

```powershell
codegraph.cmd explore "werewolf_agent/rag/retriever.py werewolf_agent/rag/vector_store.py responsibilities callers tests"
```

Expected: identify tokenization/query preparation, ranking, local storage, embedding storage, and provider-specific branches.

- [x] **Step 2: Characterize public imports and retrieval output**

Add or update tests proving existing imports from `retriever.py` and `vector_store.py` still work, and that representative query results keep the same order and metadata.

- [x] **Step 3: Move query processing**

Move query normalization, tokenization, filters, and lightweight scoring helpers into `query_processing.py`; re-export compatibility helpers where tests or callers rely on them.

- [x] **Step 4: Move retrieval ranking**

Move ranking, dedupe, score combination, and result shaping into `retrieval_ranking.py`.

- [x] **Step 5: Move vector store implementations**

Move local in-memory/file-backed behavior into `local_vector_store.py`; move embedding-backed/provider-specific behavior into `embedding_vector_store.py`.

- [x] **Step 6: Keep original modules as facades**

`retriever.py` should coordinate retrieval flow. `vector_store.py` should expose stable public store classes and delegate implementation details.

- [x] **Step 7: Verify and commit**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/rag/test_rag.py tests/rag/test_retrieval_eval.py tests/rag/test_clients.py tests/rag/test_query_processing.py tests/rag/test_vector_store_split.py -q
python -m compileall -q werewolf_agent/rag
python -m ruff check werewolf_agent/rag tests/rag --select F401,F841
git diff --check
git commit -m "refactor: split rag retrieval storage"
```

Expected: focused RAG tests pass; no import or syntax regressions.

---

### Task 15: Revisit Agent Adapter Facade

**Files:**
- Modify: `werewolf_agent/runtime/agent_adapter.py`
- Candidate not created: `werewolf_agent/runtime/agent_dispatch.py`
- Candidate not created: `werewolf_agent/runtime/agent_decision_contract.py`
- Create: `werewolf_agent/runtime/agent_action_pipeline.py`
- Test: `tests/runtime/test_agent_adapter.py`
- Test: `tests/runtime/test_agent_registry.py`
- Test: `tests/runtime/test_agent_action_audit.py`
- Test: `tests/runtime/test_agent_reflection_support.py`

- [x] **Step 1: Inspect remaining responsibilities**

Run:

```powershell
codegraph.cmd explore "werewolf_agent/runtime/agent_adapter.py remaining responsibilities dispatch decision contract action pipeline"
```

Expected: identify whether the file is a true facade or still owns dispatch, decision validation, and action orchestration.

- [x] 2026-07-07 result: `agent_adapter.py` still owned the runtime action adapter pipeline (`agent_night_*`, wolf discussion/consensus, day speech/vote, sheriff actions, hybrid/last-words/badge/hunter paths). No clean standalone dispatch or decision-contract boundary was identified in this pass.

- [x] **Step 2: Characterize facade compatibility**

Verify old imports for `AgentRegistry`, `SimpleAgentRegistry`, `_call_agent`, and action adapter helpers remain stable.

- [x] 2026-07-07 result: added characterization tests for facade-to-pipeline identity and adapter monkeypatch propagation; kept wolf kill helper compatibility through `wolf_kill_support` re-exports.

- [x] **Step 3: Split only if responsibilities remain mixed**

Move dispatch-only behavior into `agent_dispatch.py`, action contract normalization into `agent_decision_contract.py`, and multi-step action orchestration into `agent_action_pipeline.py`.

- [x] 2026-07-07 result: moved the concrete multi-step action orchestration into `agent_action_pipeline.py`; did not create `agent_dispatch.py` or `agent_decision_contract.py` because doing so would have produced speculative/empty modules.

- [x] **Step 4: Keep `agent_adapter.py` as compatibility entry**

Leave public import paths intact; avoid changing runtime graph callers unless required for explicit dependencies.

- [x] 2026-07-07 result: `agent_adapter.py` now re-exports the old symbols from `agent_action_pipeline.py`, delegates unknown compatibility symbols via `__getattr__`, and propagates facade monkeypatches back into the pipeline module.

- [x] **Step 5: Verify and commit**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/runtime/test_agent_adapter.py tests/runtime/test_agent_registry.py tests/runtime/test_agent_action_audit.py tests/runtime/test_agent_reflection_support.py -q
python -m compileall -q werewolf_agent/runtime
python -m ruff check werewolf_agent/runtime/agent_adapter.py werewolf_agent/runtime/agent_action_pipeline.py tests/runtime/test_agent_adapter.py --select F401,F841
git diff --check
git commit -m "refactor: split agent adapter action pipeline"
```

Expected: either a smaller facade commit or a documented decision that no additional split is justified.

2026-07-07 verification:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/runtime/test_agent_adapter.py -q
python -m pytest -n 0 --basetemp .tmp tests/runtime/test_agent_adapter.py tests/runtime/test_agent_registry.py tests/runtime/test_agent_action_audit.py tests/runtime/test_agent_reflection_support.py -q
python -m pytest -n 0 --basetemp .tmp tests/runtime/test_agent_adapter.py tests/runtime/test_witch_flow.py tests/runtime/test_night_flow.py tests/runtime/test_day_discussion.py tests/runtime/test_vote_flow.py tests/runtime/test_pk_flow.py tests/runtime/test_sheriff_flow.py tests/runtime/test_hunter_flow.py tests/runtime/test_strategy_directives.py -q
python -m compileall -q werewolf_agent/runtime
python -m ruff check werewolf_agent/runtime/agent_adapter.py werewolf_agent/runtime/agent_action_pipeline.py tests/runtime/test_agent_adapter.py --select F401,F841
git diff --check
compatibility import probe for 28 `agent_adapter` symbols
```

All listed commands returned exit code 0; broad runtime impact set covered 254 tests; compatibility probe reported `OK 28 agent_adapter compatibility names`.

---

### Task 16: Revisit Prompt Builder And Player Agent

**Files:**
- Modify: `werewolf_agent/agents/prompt_builder.py`
- Modify: `werewolf_agent/agents/player.py`
- Create: `werewolf_agent/agents/prompt_composer.py`
- Create: `werewolf_agent/agents/player_action_flow.py`
- Create: `werewolf_agent/agents/player_action_result.py`
- Candidate not created: `werewolf_agent/agents/player_result_mapping.py`（由更窄的 `player_action_result.py` 承担最终结果收尾）
- Test: `tests/agents/test_prompt_builder.py`
- Test: `tests/agents/test_prompt_sections.py`
- Test: `tests/agents/test_player_agent.py`
- Test: `tests/agents/test_player_generation.py`

- [x] **Step 1: Re-rank remaining agent files**

Run:

```powershell
codegraph.cmd explore "werewolf_agent/agents/prompt_builder.py werewolf_agent/agents/player.py remaining responsibilities facade coordinator"
```

Expected: distinguish high-level orchestration from leftover formatting, provider call, retry, and result mapping details.

- [x] 2026-07-07 result: `prompt_builder.py` already owns section rendering through mixins, but still owned the system/user prompt composition order. `player.py` still owned provider call, retry, fallback trace, metrics, and success/failure result mapping inside `PlayerAgent.act`.

- [x] **Step 2: Split prompt composition if still mixed**

Move prompt assembly orchestration into `prompt_composer.py` only if `prompt_builder.py` still mixes section rendering with prompt lifecycle decisions.

- [x] 2026-07-07 result: moved system/user prompt composition order into `prompt_composer.py`; `PlayerPromptBuilder` now delegates `build_system_prompt()` and `build_user_prompt()` while keeping all section builders import-compatible.

- [x] **Step 3: Split player action flow if still mixed**

Move action flow coordination into `player_action_flow.py`; move response/result mapping into `player_result_mapping.py` if `player.py` still owns those details.

- [x] 2026-07-07 result: moved `PlayerAgent.act` retry/action orchestration into `player_action_flow.py`; did not create `player_result_mapping.py` because the trace/result mapping state is still coupled to retry loop locals and would require a larger behavior-sensitive redesign.

- [x] **Step 4: Preserve public classes**

Keep `PlayerPromptBuilder` importable from `prompt_builder.py` and `PlayerAgent` importable from `player.py`.

- [x] 2026-07-07 result: public imports for `PlayerPromptBuilder` and `PlayerAgent` remain stable; characterization tests assert facade delegation to `prompt_composer` and `player_action_flow`.

- [x] **Step 5: Verify and commit**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/agents/test_prompt_builder.py tests/agents/test_prompt_sections.py tests/agents/test_prompt_persona.py tests/agents/test_prompt_memory.py tests/agents/test_prompt_strategy.py tests/agents/test_player_agent.py tests/agents/test_player_generation.py -q
python -m compileall -q werewolf_agent/agents
python -m ruff check werewolf_agent/agents/prompt_builder.py werewolf_agent/agents/prompt_composer.py werewolf_agent/agents/player.py werewolf_agent/agents/player_action_flow.py --select F401,F841
git diff --check
git commit -m "refactor: slim agent prompt and player facades"
```

Expected: public prompt/player behavior unchanged.

2026-07-07 verification:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/agents/test_prompt_builder.py::TestPromptComposerSplit -q
python -m pytest -n 0 --basetemp .tmp tests/agents/test_player_agent.py::TestPlayerActionFlowSplit -q
python -m pytest -n 0 --basetemp .tmp tests/agents/test_prompt_builder.py tests/agents/test_prompt_sections.py tests/agents/test_prompt_persona.py tests/agents/test_prompt_memory.py tests/agents/test_prompt_strategy.py tests/agents/test_player_agent.py tests/agents/test_player_generation.py -q
python -m pytest -n 0 --basetemp .tmp tests/agents/test_player_failures.py tests/agents/test_player_latency.py -q
python -m compileall -q werewolf_agent/agents
python -m ruff check werewolf_agent/agents/prompt_builder.py werewolf_agent/agents/prompt_composer.py werewolf_agent/agents/player.py werewolf_agent/agents/player_action_flow.py --select F401,F841
git diff --check
public import probe for `PlayerPromptBuilder`, `PlayerAgent`, `prompt_composer`, and `player_action_flow`
```

All listed commands returned exit code 0. The compatibility suite caught and guarded preserved `player.py` helper aliases after `PlayerAgent.act` moved to `player_action_flow.py`. A broader Ruff run including the two large touched test files surfaced pre-existing test-only F401/F841 debt and was not used as a Task 16 gate.

---

### Task 17: Split Large Role-Specific Runtime Helpers

**Files:**
- Modify: `werewolf_agent/skills/good_skill_handlers.py`
- Modify: `werewolf_agent/runtime/nodes/wolf_night_nodes.py`
- Candidate create: `werewolf_agent/skills/good_claim_handlers.py`
- Candidate create: `werewolf_agent/skills/good_vote_handlers.py`
- Candidate create: `werewolf_agent/skills/good_power_handlers.py`
- Candidate create: `werewolf_agent/runtime/nodes/wolf_discussion.py`
- Candidate create: `werewolf_agent/runtime/nodes/wolf_consensus.py`
- Test: `tests/skills/test_good_skill_handlers.py`
- Test: `tests/runtime/test_wolf_night_nodes.py`
- Test: `tests/runtime/test_wolf_flow.py`

- [ ] **Step 1: Inspect role helper cohesion**

Run:

```powershell
codegraph.cmd explore "werewolf_agent/skills/good_skill_handlers.py werewolf_agent/runtime/nodes/wolf_night_nodes.py role-specific helper responsibilities"
```

Expected: identify claim, vote, power-role, wolf discussion, and wolf consensus boundaries.

- [x] **Step 2: Split good-side skill handlers by intent**

Move claim/counter-claim handlers into `good_claim_handlers.py`, vote pressure handlers into `good_vote_handlers.py`, and power-role protection/search handlers into `good_power_handlers.py`.

2026-07-07 result: split `good_skill_handlers.py` into `good_claim_handlers.py`, `good_vote_handlers.py`, and `good_power_handlers.py`; kept `good_skill_handlers.py` as a compatibility facade so existing `werewolf_skills` imports and handler registration remain stable.

- [x] **Step 3: Split wolf night nodes by phase**

Move discussion/planning helpers into `wolf_discussion.py`; move consensus and fallback target resolution into `wolf_consensus.py`.

2026-07-07 result: split `wolf_discussion`, `_build_fallback_wolf_team_plan`, and `wolf_team_plan_node` into `wolf_discussion.py`; split `_legacy_wolf_consensus` and `wolf_consensus` into `wolf_consensus.py`; kept `wolf_night_nodes.py` as a compatibility facade.

- [x] **Step 4: Preserve compatibility registries and node imports**

Keep existing skill handler registration and node imports working through the old modules.

2026-07-07 result: old imports through `runtime.nodes.night` and `runtime.nodes.wolf_night_nodes` remain stable; compatibility test also covers old `night.py` monkeypatch paths for agent dispatch.

- [x] **Step 5: Verify and commit**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/skills/test_good_skill_handlers.py tests/skills/test_werewolf_skills.py tests/runtime/test_wolf_night_nodes.py tests/runtime/test_wolf_flow.py -q
python -m compileall -q werewolf_agent/skills werewolf_agent/runtime/nodes
python -m ruff check werewolf_agent/skills/good_skill_handlers.py werewolf_agent/skills/good_claim_handlers.py werewolf_agent/skills/good_vote_handlers.py werewolf_agent/skills/good_power_handlers.py werewolf_agent/runtime/nodes/wolf_night_nodes.py werewolf_agent/runtime/nodes/wolf_discussion.py werewolf_agent/runtime/nodes/wolf_consensus.py --select F401,F841
git diff --check
git commit -m "refactor: split role-specific helper modules"
```

Expected: handler registry and night graph behavior unchanged.

2026-07-07 partial verification for Step 2:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/skills/test_good_skill_handlers.py -q
python -m pytest -n 0 --basetemp .tmp tests/skills/test_good_skill_handlers.py tests/skills/test_werewolf_skills.py -q
python -m compileall -q werewolf_agent/skills
python -m ruff check werewolf_agent/skills/good_skill_handlers.py werewolf_agent/skills/good_claim_handlers.py werewolf_agent/skills/good_vote_handlers.py werewolf_agent/skills/good_power_handlers.py --select F401,F841
git diff --check
```

All listed Step 2 commands returned exit code 0.

2026-07-07 final verification:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/runtime/test_wolf_night_nodes.py -q
python -m pytest -n 0 --basetemp .tmp tests/skills/test_good_skill_handlers.py tests/skills/test_werewolf_skills.py tests/runtime/test_wolf_night_nodes.py tests/runtime/test_wolf_flow.py -q
python -m compileall -q werewolf_agent/skills werewolf_agent/runtime/nodes
python -m ruff check werewolf_agent/skills/good_skill_handlers.py werewolf_agent/skills/good_claim_handlers.py werewolf_agent/skills/good_vote_handlers.py werewolf_agent/skills/good_power_handlers.py werewolf_agent/runtime/nodes/wolf_night_nodes.py werewolf_agent/runtime/nodes/wolf_discussion.py werewolf_agent/runtime/nodes/wolf_consensus.py --select F401,F841
git diff --check
```

All listed final Task 17 commands returned exit code 0.

---

### Task 18: Split Agent Schema Definitions

**Files:**
- Modify: `werewolf_agent/agents/schemas.py`
- Candidate create: `werewolf_agent/agents/action_schemas.py`
- Candidate create: `werewolf_agent/agents/prompt_schemas.py`
- Candidate create: `werewolf_agent/agents/trace_schemas.py`
- Test: `tests/agents/test_schemas.py`
- Test: `tests/agents/test_action_contract.py`
- Test: `tests/agents/test_trace_builder.py`

- [x] **Step 1: Inspect schema callers**

Run:

```powershell
codegraph.cmd explore "werewolf_agent/agents/schemas.py classes imports action prompt trace schema callers"
```

Expected: map each schema group and public import path before moving definitions.

2026-07-07 result: CodeGraph and AST inspection mapped action, prompt, and trace schema groups; baseline schema tests passed before moving definitions.

- [x] **Step 2: Move action schemas**

Move action input/output and validation contract models into `action_schemas.py`.

2026-07-07 result: moved action enum/model variants, `JudgeBroadcast`, `FallbackAction`, and `WolfTeamPlan` into `action_schemas.py`.

- [x] **Step 3: Move prompt schemas**

Move prompt construction and prompt section data models into `prompt_schemas.py`.

2026-07-07 result: moved `TaskType`, `OutputMode`, and `AgentContext` into `prompt_schemas.py`.

- [x] **Step 4: Move trace schemas**

Move decision trace and audit-facing agent schemas into `trace_schemas.py`.

2026-07-07 result: moved `ActionTrace`, `PrivateIntent`, `FactionGoal`, `RiskFlag`, and `RetryInfo` into `trace_schemas.py`.

- [x] **Step 5: Keep `schemas.py` as public facade**

Re-export old public schema names from `schemas.py` to avoid breaking existing callers.

2026-07-07 result: `schemas.py` now re-exports old public names from the split modules; `tests/agents/test_schemas.py` covers new module import identity without splitting the test directory.

- [x] **Step 6: Verify and commit**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/agents/test_schemas.py tests/agents/test_action_contract.py tests/agents/test_trace_builder.py -q
python -m compileall -q werewolf_agent/agents
python -m ruff check werewolf_agent/agents/schemas.py werewolf_agent/agents/action_schemas.py werewolf_agent/agents/prompt_schemas.py werewolf_agent/agents/trace_schemas.py --select F401,F841
git diff --check
git commit -m "refactor: split agent schema groups"
```

Expected: all schema imports and serialization behavior remain stable.

2026-07-07 final verification:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/agents/test_schemas.py tests/agents/test_action_contract.py tests/agents/test_trace_builder.py -q
python -m compileall -q werewolf_agent/agents
python -m ruff check werewolf_agent/agents/schemas.py werewolf_agent/agents/action_schemas.py werewolf_agent/agents/prompt_schemas.py werewolf_agent/agents/trace_schemas.py --select F401,F841
git diff --check
```

All listed Task 18 commands returned exit code 0.

---

### Task 19: Second-Pass Evaluation Metrics Facade

**Files:**
- Modify: `werewolf_agent/evaluation/metrics.py`
- Candidate create: `werewolf_agent/evaluation/metric_aggregation.py`
- Candidate create: `werewolf_agent/evaluation/metric_reporting.py`
- Test: `tests/evaluation/test_feedback_metrics.py`
- Test: `tests/evaluation/test_metrics_split_helpers.py`
- Test: `tests/evaluation/test_remaining_metrics.py`

- [x] **Step 1: Inspect remaining metric responsibilities**

Run:

```powershell
codegraph.cmd explore "werewolf_agent/evaluation/metrics.py remaining responsibilities aggregator reporting callers"
```

Expected: decide whether `metrics.py` is an acceptable facade or still mixes aggregation and report formatting.

2026-07-07 result: CodeGraph and AST inspection showed `metrics.py` still carried the large `MetricsAggregator` implementation plus snapshot comparison payload shaping, so it was not yet an acceptable thin facade.

- [x] **Step 2: Move aggregation helpers if needed**

Move cross-metric aggregation and denominator handling into `metric_aggregation.py`.

2026-07-07 result: moved `MetricsAggregator` and cross-metric denominator/provenance aggregation into `metric_aggregation.py`.

- [x] **Step 3: Move reporting helpers if needed**

Move display/report shaping helpers into `metric_reporting.py`.

2026-07-07 result: moved `compare_snapshots` report payload shaping into `metric_reporting.py`; `MetricsAggregator.compare_snapshots` remains compatible through a staticmethod binding.

- [x] **Step 4: Verify and commit**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/evaluation/test_feedback_metrics.py tests/evaluation/test_metrics_split_helpers.py tests/evaluation/test_remaining_metrics.py -q
python -m compileall -q werewolf_agent/evaluation
python -m ruff check werewolf_agent/evaluation/metrics.py werewolf_agent/evaluation/metric_aggregation.py werewolf_agent/evaluation/metric_reporting.py --select F401,F841
git diff --check
git commit -m "refactor: slim evaluation metrics facade"
```

Expected: metric values and report payloads remain unchanged.

2026-07-07 final verification:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/evaluation/test_feedback_metrics.py tests/evaluation/test_metrics_split_helpers.py tests/evaluation/test_remaining_metrics.py -q
python -m compileall -q werewolf_agent/evaluation
python -m ruff check werewolf_agent/evaluation/metrics.py werewolf_agent/evaluation/metric_aggregation.py werewolf_agent/evaluation/metric_reporting.py --select F401,F841
git diff --check
```

All listed Task 19 commands returned exit code 0.

---

### Task 20: Second-Pass Game Route Facade

**Files:**
- Modify: `werewolf_agent/api/routes/games.py`
- Candidate create: `werewolf_agent/api/routes/game_lifecycle.py`
- Candidate create: `werewolf_agent/api/routes/game_commands.py`
- Candidate create: `werewolf_agent/api/routes/game_snapshots.py`
- Test: `tests/api/test_api.py`
- Test: `tests/api/test_routes.py`
- Test: `tests/api/test_pause_resume_lock.py`
- Test: `tests/api/test_game_auth_helpers.py`

- [x] **Step 1: Inspect route declaration density**

Run:

```powershell
codegraph.cmd explore "werewolf_agent/api/routes/games.py route declarations lifecycle commands snapshots dependencies"
```

Expected: determine whether route declaration itself should split by lifecycle, command, and snapshot endpoints.

2026-07-07 result: CodeGraph and route inspection showed `games.py` still owned lifecycle, command, snapshot/view, share, and audit route declarations after helper extraction, so route declarations remained too dense for one focused module.

- [x] **Step 2: Split route groups only if registration remains clear**

Move lifecycle endpoints into `game_lifecycle.py`, command/pause/resume endpoints into `game_commands.py`, and snapshot/share view endpoints into `game_snapshots.py`.

2026-07-07 result: lifecycle endpoints moved to `game_lifecycle.py`, start/step/pause/resume moved to `game_commands.py`, and state/replay/share/audit endpoints moved to `game_snapshots.py`.

- [x] **Step 3: Keep router factory stable**

Keep `create_game_router` and any public route registration API stable from `games.py`.

2026-07-07 result: `games.py` remains the public `create_game_router` facade and re-exports compatibility helpers. Route registration uses a current facade resolver wrapper so tests that monkeypatch `games._resolve_caller_role` keep working after router creation.

- [x] **Step 4: Verify and commit**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/api/test_api.py tests/api/test_routes.py tests/api/test_pause_resume_lock.py tests/api/test_game_auth_helpers.py tests/api/test_game_persistence_helpers.py -q
python -m compileall -q werewolf_agent/api
python -m ruff check werewolf_agent/api/routes/games.py werewolf_agent/api/routes/game_lifecycle.py werewolf_agent/api/routes/game_commands.py werewolf_agent/api/routes/game_snapshots.py --select F401,F841
git diff --check
git commit -m "refactor: split game route declarations"
```

Expected: route table and auth behavior stay unchanged.

2026-07-07 final verification in the managed sandbox:

```powershell
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -B -m pytest -o addopts= -p no:cacheprovider --basetemp .pytest_task20_focus tests/api/test_game_auth_helpers.py tests/api/test_game_persistence_helpers.py::test_game_route_group_modules_expose_registration_helpers tests/api/test_game_persistence_helpers.py::test_game_router_keeps_route_registration_surface tests/api/test_game_persistence_helpers.py::test_game_router_uses_patched_facade_role_resolver_after_registration -q
python -B -c "... route surface exact assertion ..."
python -B -c "... patched games._resolve_caller_role replay assertion ..."
python -B -c "... create/start/public-state/replay/pause/resume/rag-audit smoke ..."
python -B -c "... ast.parse touched route modules ..."
python -B -c "... _build_locked_config_snapshot path traversal assertion ..."
python -m ruff check --no-cache werewolf_agent/api/routes/games.py werewolf_agent/api/routes/game_lifecycle.py werewolf_agent/api/routes/game_commands.py werewolf_agent/api/routes/game_snapshots.py tests/api/test_game_persistence_helpers.py --select F401,F841
git diff --check
```

All listed sandbox-safe Task 20 commands returned exit code 0; the focused pytest command reported `9 passed`. The full pytest command with the plan's API test list executed 74 tests successfully, then errored only while setting up pytest's `tmp_path` fixture because the managed sandbox denied creating the requested basetemp directory (`PermissionError: [WinError 5]`). Sandboxed `compileall` also hit `WinError 5` writing bytecode cache, so AST parsing was used as the no-cache syntax check.

---

### Task 21: Second-Pass Model Router Facade

**Files:**
- Modify: `werewolf_agent/model_gateway/router.py`
- Candidate create: `werewolf_agent/model_gateway/router_config.py`
- Candidate create: `werewolf_agent/model_gateway/router_selection.py`
- Candidate create: `werewolf_agent/model_gateway/router_errors.py`
- Test: `tests/model_gateway/test_router.py`
- Test: `tests/model_gateway/test_router_split_helpers.py`
- Test: `tests/model_gateway/test_empty_response.py`
- Test: `tests/model_gateway/test_providers.py`

- [x] **Step 1: Inspect router responsibilities**

Run:

```powershell
codegraph.cmd explore "werewolf_agent/model_gateway/router.py remaining responsibilities config selection errors provider calls"
```

Expected: distinguish stable router orchestration from config parsing, provider selection, and error/fallback mapping.

2026-07-07 result: `router.py` still owned config validation and configured provider collection, provider/model selection, fallback model resolution, usage recording, and empty-result construction.

- [x] **Step 2: Move config helpers if needed**

Move profile/config parsing helpers into `router_config.py`.

2026-07-07 result: moved `_validate_config` and `_configured_provider_names` into `router_config.py`; `ModelRouter._validate_config()` and `_configured_provider_names()` now delegate while old helper imports remain available from `router.py`.

- [x] **Step 3: Move provider selection if needed**

Move provider selection and routing policy helpers into `router_selection.py`.

2026-07-07 result: moved `_resolve_config` and `_resolve_fallback_model` into `router_selection.py`; `ModelRouter.resolve_config()` and `_resolve_fallback_model()` keep the old method signatures.

- [x] **Step 4: Move error mapping if needed**

Move empty-response, retry exhaustion, and fallback error mapping into `router_errors.py`.

2026-07-07 result: moved `_record_success_usage`, `_record_failure_usage`, and `_empty_result` into `router_errors.py`; `ModelRouter.generate()` still owns provider call orchestration and delegates record/result construction.

- [x] **Step 5: Verify and commit**

Run:

```powershell
python -m pytest -n 0 --basetemp .tmp tests/model_gateway/test_router.py tests/model_gateway/test_router_split_helpers.py tests/model_gateway/test_empty_response.py tests/model_gateway/test_providers.py -q
python -m compileall -q werewolf_agent/model_gateway
python -m ruff check werewolf_agent/model_gateway/router.py werewolf_agent/model_gateway/router_config.py werewolf_agent/model_gateway/router_selection.py werewolf_agent/model_gateway/router_errors.py --select F401,F841
git diff --check
git commit -m "refactor: slim model gateway router facade"
```

Expected: provider routing, retries, and empty-response fallback behavior unchanged.

2026-07-07 verification in the managed sandbox:

```powershell
python -c "... router split import identity probe ..."
python -c "... manual router behavior suite ..."
python -c "... ruleset loader compatibility probe ..."
python -c "... rule engine behavior probe ..."
$env:PYTHONPYCACHEPREFIX='E:\NLP\agent\wofkill\.tmp\srp-task21-pycache'; python -m compileall -q werewolf_agent/model_gateway
$env:PYTHONPYCACHEPREFIX='E:\NLP\agent\wofkill\.tmp\srp-task21-pycache'; python -m compileall -q werewolf_agent/engine/rule_engine.py werewolf_agent/engine/ruleset_loader.py tests/rules/test_rule_engine_split.py
$env:RUFF_CACHE_DIR='E:\NLP\agent\wofkill\.tmp\srp-task21-ruff'; python -m ruff check werewolf_agent/model_gateway/router.py werewolf_agent/model_gateway/router_config.py werewolf_agent/model_gateway/router_selection.py werewolf_agent/model_gateway/router_errors.py tests/model_gateway/test_router_split_helpers.py --select F401,F841
$env:RUFF_CACHE_DIR='E:\NLP\agent\wofkill\.tmp\srp-task21-ruff'; python -m ruff check werewolf_agent/engine/rule_engine.py werewolf_agent/engine/ruleset_loader.py tests/rules/test_rule_engine_split.py --select F401,F841
git diff --check
```

All listed commands returned exit code 0. Direct RED checks first failed for missing `router_config` and missing `ruleset_loader`, then passed after the split. In this managed sandbox, pytest printed all selected `test_router_split_helpers.py` tests as passed (`....... [100%]`) but the pytest process did not reach normal session finish before timeout, so the final gate used direct behavior probes plus compileall, Ruff, import compatibility, and `git diff --check`.

---

## Completion Criteria

- Previously identified SRP candidates are either split or explicitly documented as cohesive enough to remain.
- Every original public module path still imports successfully.
- `agent_adapter.py`, `context.py`, `night.py`, `prompt_builder.py`, `werewolf_skills.py`, `output_parser.py`, `player.py`, `reflection.py`, `day.py`, and `games.py` are reduced to focused modules or compatibility facades.
- Focused tests exist next to each new responsibility module.
- Each commit is small, reversible, and tied to one responsibility group.
- CodeGraph is synced after the final commit.
