# Runtime Timeout、重试与 Deadline 移除实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 移除不可取消的 Runtime 线程超时与阶段 deadline，把 provider HTTP timeout 统一为 300 秒，在 ModelRouter 内实现确定性、按候选路由计数的指数退避，并让每个真实 provider timeout 可从 attempt 证据追溯到 ActionTrace 和最终报告。

**Architecture:** Runtime 直接在当前线程同步调用 Agent，唯一可终止网络 I/O 的边界是 provider 客户端的 300 秒 HTTP timeout。`retry_policy.py` 只负责分类、预算和等待值纯计算，`router.py` 负责真实调用、sleep、逐尝试证据和终态 `FailureDisposition`；Agent 层只对可修复输出进行下一轮语义修复。timeout 指标由 `execution_attempts` 单向派生，兼容入口只保留惰性的 `AGENT_TIMEOUTS` re-export。

**Tech Stack:** Python 3.11+、Pydantic、httpx、pytest、PowerShell、现有 Model Gateway / Runtime / evaluation 组件。

---

## 执行约定

- 所有命令从工作树 `E:\NLP\agent\wofkill\.worktrees\runtime-timeout-retry-deadline-removal` 执行。
- 每个 Python 文件发生非平凡修改时，检查中文模块 docstring 是否仍准确；新建 Python 文件使用编码声明、中文模块说明、作者和日期头。
- 测试不得真实 sleep；通过 monkeypatch 收集等待秒数，并为 HTTP-date 测试注入固定时钟。
- Windows 下统一禁用 pytest cacheprovider；涉及嵌套 pytest 的测试还要设置 `PYTEST_ADDOPTS` 让子进程继承。
- 每个任务完成后运行该任务列出的 GREEN 命令和 `git diff --check`，只提交本任务文件。
- 任何超时或没有最终 pytest summary 的测试运行都记为“未验证”，不能记为通过。

通用 PowerShell 测试前缀：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:PYTEST_ADDOPTS='-p no:cacheprovider'
$env:LANGCHAIN_TRACING_V2='false'
$env:LANGSMITH_TRACING='false'
$env:TMP='E:\NLP\agent\wofkill\.tmp'
$env:TEMP=$env:TMP
```

### Task 1: 建立纯重试策略和双预算合同

**Files:**

- Create: `tests/model_gateway/test_retry_policy.py`
- Modify: `werewolf_agent/model_gateway/retry_policy.py:1-124`
- Modify: `tests/model_gateway/test_router.py:1023-1033`

- [ ] **Step 1: 为确定性普通退避写 RED 测试**

在新测试文件中覆盖 primary `2, 4, 8, 16`、fallback `2, 4`，并断言计算不调用随机数。测试直接调用公开的纯函数，例如：

```python
def test_generic_retry_delays_are_route_specific() -> None:
    assert [retry_delay(RetryKind.GENERIC, RouteKind.PRIMARY, i) for i in range(4)] == [2, 4, 8, 16]
    assert [retry_delay(RetryKind.GENERIC, RouteKind.PROVIDER_FALLBACK, i) for i in range(2)] == [2, 4]
```

- [ ] **Step 2: 为 429 和 Retry-After 写 RED 测试**

覆盖 `16, 32, 64`、delta-seconds、未来/过去 HTTP-date、无效值、服务端值小于基线、300 秒封顶。固定 `now`，避免测试依赖真实时间：

```python
assert retry_delay(RetryKind.RATE_LIMIT, RouteKind.PRIMARY, 0, retry_after="45", now=NOW) == 45
assert retry_delay(RetryKind.RATE_LIMIT, RouteKind.PRIMARY, 2, retry_after="999", now=NOW) == 300
```

- [ ] **Step 3: 为每个 candidate 的混合错误双预算写 RED 测试**

定义 `RetryBudget`/`RetryBudgetState` 的可测试接口，验证：

- primary 普通错误最多 4 次重试；
- fallback 普通错误最多 2 次重试；
- primary/fallback 的 429 最多 3 次重试；
- `config_retry_count=0` 立即拒绝重试；
- fallback 默认总预算 4 时，`generic, generic, 429, 429` 可重试，但第五次失败不可重试；
- 错误类型切换不会重置 `total_retry_count` 或分类计数。

- [ ] **Step 4: 运行 RED 测试并确认失败原因是新 API 尚不存在**

```powershell
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\retry-policy-red tests/model_gateway/test_retry_policy.py tests/model_gateway/test_router.py::TestRetryHelpers -q
```

Expected: FAIL，缺少 `RetryKind`、预算对象或新纯函数；不能因收集错误以外的无关异常失败。

- [ ] **Step 5: 实现最小纯策略**

在 `retry_policy.py` 中：

- 保留并复用现有 `_is_retryable_exception()`；
- 新增普通/429 强类型分类；
- 解析异常的 status/headers 时保持现有 provider 兼容；
- 用 `email.utils.parsedate_to_datetime()` 解析 HTTP-date；
- 等待值实现为 `min(300.0, max(valid_retry_after, baseline))`；
- 删除 jitter 和 `random` 依赖；
- 预算对象同时维护 total/generic/rate-limit 三个计数；
- 预算属于单个 route candidate，函数本身不持有全局状态。

兼容 `_retry_delay_for_exception()` 的旧导入可以暂留为薄包装，但其结果必须确定性；更新旧测试名和断言，不再描述 jitter。

- [ ] **Step 6: 运行 GREEN 测试**

```powershell
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\retry-policy-green tests/model_gateway/test_retry_policy.py tests/model_gateway/test_router.py::TestRetryHelpers -q
git diff --check
```

Expected: 全部 PASS；普通与 429 序列没有真实等待。

- [ ] **Step 7: 提交纯策略**

```powershell
git add werewolf_agent/model_gateway/retry_policy.py tests/model_gateway/test_retry_policy.py tests/model_gateway/test_router.py
git commit -m "feat: define deterministic model retry policy"
```

### Task 2: 统一 300 秒 timeout、4 次缺省重试并接入 Router 候选预算

**Files:**

- Modify: `werewolf_agent/model_gateway/usage_records.py:28-52`
- Modify: `werewolf_agent/model_gateway/router_selection.py:43-69,180-205`
- Modify: `config/models.yaml:1-125`
- Modify: `werewolf_agent/model_gateway/router.py:338-535,600-741,790-793`
- Modify: `tests/model_gateway/test_router.py:174-229,432-783,1023-1033`
- Modify: `tests/model_gateway/test_providers.py:141-161`
- Modify: provider-specific tests under `tests/model_gateway/test_anthropic_provider.py` and `tests/model_gateway/test_minimax_provider_routing.py`

- [ ] **Step 1: 写缺省配置和 provider 透传 RED 测试**

断言：

```python
assert ModelConfig(provider="openai", model="m").timeout == 300
assert ModelConfig(provider="openai", model="m").retry_count == 4
```

再分别用 fake client 捕获 OpenAI、Anthropic、MiniMax、GLM 的实际请求 timeout，断言均使用 `config.timeout=300`。为 `_resolve_config()` 和 `_fallback_config()` 增加“profile 缺字段时 timeout=300、retry_count=4”的测试，避免 dataclass 默认值被旧 parser 默认覆盖。

- [ ] **Step 2: 写 Router 调用次数、sleep 和 attempt RED 测试**

在 `test_router.py` 用序列 provider 和 fake sleep 覆盖：

- primary 普通错误：5 次调用、4 次 retry、sleep `2/4/8/16`；
- fallback 普通错误：每个 candidate 3 次调用、sleep `2/4`；
- primary/fallback 429：每个 candidate 4 次调用、sleep `16/32/64`；
- explicit `retry_count=0`：只调用一次、不 sleep；
- fallback chain 切换 candidate 后预算重置；
- 单 candidate 混合普通/429 错误仍受总预算限制；
- timeout 每次追加 `RootCause.TIMEOUT`，429 每次追加 `RootCause.PROVIDER_ERROR`；
- ordinal 连续且每次真实 provider 调用恰好一条 record；
- 空响应只追加 `INVALID_OUTPUT`，不在 Router 内 sleep 或网络重试。

- [ ] **Step 3: 运行 RED 测试**

```powershell
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\router-budget-red tests/model_gateway/test_retry_policy.py tests/model_gateway/test_router.py tests/model_gateway/test_providers.py tests/model_gateway/test_anthropic_provider.py tests/model_gateway/test_minimax_provider_routing.py -q
```

Expected: 新的次数/等待/默认值断言 FAIL，证明仍在使用旧 jitter、旧 parser 默认或统一 fallback loop 预算。

- [ ] **Step 4: 修改配置默认值**

- `ModelConfig.timeout = 300`，`retry_count = 4`；
- `_resolve_config()` 缺省 timeout/retry 使用 `300/4`；
- `_fallback_config()` 缺省 timeout/retry 使用 `300/4`；
- `config/models.yaml` 中每个 model profile 的显式 timeout 改为 300，不改变模型、provider、fallback 顺序或显式 retry_count。

- [ ] **Step 5: 在 Router 中按 candidate 创建预算并消费策略**

primary 建立一个预算；fallback chain 每进入一个新的 provider/model candidate 就建立新预算。每次失败先追加 attempt，再由策略判断是否允许 retry，允许时计算 delay、写包含 route/attempt/max/category/delay 的日志并调用注入/monkeypatchable 的 `time.sleep()`。

不要在 `router.py` 重复指数公式。删除空响应的 Router 网络重试：空响应终止当前 route candidate 并进入现有 fallback/终态归因路径，不 sleep。

- [ ] **Step 6: 运行 GREEN 测试**

```powershell
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\router-budget-green tests/model_gateway/test_retry_policy.py tests/model_gateway/test_router.py tests/model_gateway/test_router_split_helpers.py tests/model_gateway/test_provider_fallback_policy.py tests/model_gateway/test_providers.py tests/model_gateway/test_anthropic_provider.py tests/model_gateway/test_minimax_provider_routing.py -q
git diff --check
```

Expected: 全部 PASS；fake sleep 精确匹配设计序列，未发生真实等待。

- [ ] **Step 7: 提交 Router 策略接入**

```powershell
git add config/models.yaml werewolf_agent/model_gateway/usage_records.py werewolf_agent/model_gateway/router_selection.py werewolf_agent/model_gateway/router.py tests/model_gateway
git commit -m "feat: apply route-aware exponential retries"
```

### Task 3: 增加强类型终态失败并隔离动作层网络重入

**Files:**

- Modify: `werewolf_agent/model_gateway/usage_records.py:136-177,236-249`
- Modify: `werewolf_agent/model_gateway/router_errors.py:23-142`
- Modify: `werewolf_agent/model_gateway/router.py:289-535,600-789`
- Modify: `werewolf_agent/agents/player_action_flow.py:261-366`
- Modify: `tests/model_gateway/test_execution_records.py:67-180`
- Modify: `tests/model_gateway/test_router.py:242-849`
- Modify: `tests/agents/test_player_retry.py:573-795`
- Modify: `tests/agents/test_player_agent.py` near empty-result and terminal-fallback cases

- [ ] **Step 1: 写 `FailureDisposition` RED 测试**

新增枚举往返和 `GenerateResult` 默认测试：默认 `NONE`；网关终退必须精确分类为 `TRANSPORT_EXHAUSTED`、`OUTPUT_REPAIRABLE`、`POLICY_REJECTED`、`ROUTE_UNAVAILABLE`。

特别覆盖：

- timeout/provider error 与 invalid output 混合时是 `TRANSPORT_EXHAUSTED`；
- 所有真实失败都是 `INVALID_OUTPUT` 时才是 `OUTPUT_REPAIRABLE`；
- 无 provider candidate 被调用时是 `ROUTE_UNAVAILABLE`；
- 决策性 reasoning/structured-output 策略拒绝是 `POLICY_REJECTED`。

- [ ] **Step 2: 写动作层“transport 终退不重入”RED 测试**

用计数 fake router 让第一次 `generate()` 返回空文本和 `TRANSPORT_EXHAUSTED`，断言动作结果进入安全 fallback 且 `generate()` 总调用数为 1。分别测试另外三个非 `NONE` disposition；只有 `OUTPUT_REPAIRABLE` 可以进入现有输出/语义修复循环。

- [ ] **Step 3: 运行 RED 测试**

```powershell
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\failure-disposition-red tests/model_gateway/test_execution_records.py tests/model_gateway/test_router.py tests/agents/test_player_retry.py tests/agents/test_player_agent.py -q
```

Expected: 缺少枚举/字段或动作层仍二次调用 Router 的断言 FAIL。

- [ ] **Step 4: 实现枚举和 Router 终态分类**

在 `usage_records.py` 定义 `str, Enum` 的 `FailureDisposition` 并给 `GenerateResult` 默认 `NONE`。让 `_empty_result()` 接受 disposition，Router 只在完整 route chain 终退边界根据真实 attempts 和决策性失败设置它；成功结果保持 `NONE`。保留 synthetic `RouteKind.SAFE_FALLBACK` record，不把它当成 provider attempt。

- [ ] **Step 5: 用强类型改写动作层分支**

在 `player_action_flow.py` 删除基于异常/错误文本猜测 transport failure 的分支。空结果时：

```python
if result.failure_disposition is FailureDisposition.OUTPUT_REPAIRABLE:
    # 进入现有解析/语义修复路径
else:
    # 保留全部 attempts，立即形成安全 fallback
```

确保安全 fallback 仍携带之前所有真实失败 attempts 和终态 synthetic record。

- [ ] **Step 6: 运行 GREEN 和动作回归**

```powershell
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\failure-disposition-green tests/model_gateway/test_execution_records.py tests/model_gateway/test_router.py tests/agents/test_player_retry.py tests/agents/test_player_agent.py tests/agents/test_player_generation.py -q
git diff --check
```

Expected: 全部 PASS；transport/output 混合终退没有动作层网络重入。

- [ ] **Step 7: 提交终态合同**

```powershell
git add werewolf_agent/model_gateway/usage_records.py werewolf_agent/model_gateway/router_errors.py werewolf_agent/model_gateway/router.py werewolf_agent/agents/player_action_flow.py tests/model_gateway tests/agents
git commit -m "fix: prevent action retry after transport exhaustion"
```

### Task 4: 移除不可取消的 Agent 线程 timeout 与 per-call timeout plumbing

**Files:**

- Modify: `werewolf_agent/runtime/nodes/node_helpers.py:39,96-161`
- Modify: `werewolf_agent/runtime/wolf_kill_support.py:35,140-155`
- Modify: `werewolf_agent/runtime/game_runner_config.py` at `agent_call_timeout`
- Modify: `werewolf_agent/runtime/game_runner_setup.py` at RuntimeState initialization
- Modify: `werewolf_agent/runtime/nodes/runtime_state.py:45-72`
- Modify: timeout_override call sites in `werewolf_agent/runtime/nodes/day_deaths.py`, `day_discussion.py`, `day_vote.py`, `night_specialists.py`, `night_witch_node.py`, `sheriff_endorse.py`, `sheriff_pk.py`, `sheriff_speech.py`, `skills.py`, `wolf_consensus.py`, `wolf_discussion.py`
- Modify: `werewolf_agent/runtime/nodes/__init__.py`
- Modify: `werewolf_agent/runtime/nodes/_shared.py:68-106`
- Modify: `tests/runtime/test_graph_lifecycle.py:255-335`
- Modify: `tests/runtime/test_wolf_kill_support.py`

- [ ] **Step 1: 把旧 timeout 行为测试改成同步线程合同 RED 测试**

替换 `test_single_wolf_vote_uses_global_agent_timeout`、`test_dispatch_agent_direct_call_when_timeout_zero`：记录调用方 thread id 与 Agent 内 thread id，断言相同；monkeypatch `threading.Thread` 为一旦实例化就失败，断言 `_dispatch_agent()` 和狼刀单人投票仍成功。删除对 `timed_call` monkeypatch 的依赖。

再加静态合同测试：`inspect.signature(_dispatch_agent)` 不含 `timeout_override`，Runtime 配置/状态不含 `agent_call_timeout` 或 `wolf_vote_timeout`。

- [ ] **Step 2: 运行 RED 测试**

```powershell
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\runtime-sync-red tests/runtime/test_graph_lifecycle.py tests/runtime/test_wolf_kill_support.py tests/runtime/test_game_runner.py -q
```

Expected: timeout 参数仍存在或 `timed_call` 仍创建线程，导致新断言 FAIL。

- [ ] **Step 3: 将 Runtime 调用改为直接同步调用**

- `_call_agent()` 直接执行 callable；若无独立价值则删除并让 `_dispatch_agent()` 直接调用；
- `_dispatch_agent()` 移除 `timeout_override`；
- 所有调用点删除 timeout 参数和活动 `AGENT_TIMEOUTS` 导入；
- `wolf_kill_support.py` 在已有 try/except 内直接 `agent.act(context)`；
- 删除 active Runtime 对 `agent_call_timeout`、`wolf_vote_timeout` 的配置/状态读取；
- 保留现有异常捕获和安全 fallback，不新增线程或 async 包装。

- [ ] **Step 4: 收紧兼容 facade**

`runtime.nodes._shared` 删除 `timed_call` re-export，因为 timer API 已获批准删除；`runtime.agent_adapter`、`runtime.agent_action_pipeline`、`runtime.nodes._shared` 仍 re-export 同一个 `runtime.timeouts.AGENT_TIMEOUTS` 对象。生产节点不得读取它。

- [ ] **Step 5: 运行 GREEN 测试**

```powershell
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\runtime-sync-green tests/runtime/test_graph_lifecycle.py tests/runtime/test_wolf_kill_support.py tests/runtime/test_agent_adapter.py tests/runtime/test_game_runner.py -q
git diff --check
```

Expected: 全部 PASS；Agent 与 Runtime 在同一线程，任何路径都不实例化 timeout worker thread。

- [ ] **Step 6: 提交同步 Runtime 调用**

```powershell
git add werewolf_agent/runtime tests/runtime/test_graph_lifecycle.py tests/runtime/test_wolf_kill_support.py tests/runtime/test_agent_adapter.py tests/runtime/test_game_runner.py
git commit -m "refactor: call runtime agents synchronously"
```

### Task 5: 删除整个 Runtime deadline/timer 系统

**Files:**

- Delete: `werewolf_agent/runtime/timers.py`
- Delete: `tests/runtime/test_timers.py`
- Modify: `werewolf_agent/runtime/nodes/node_helpers.py:96-108`
- Modify: `werewolf_agent/runtime/nodes/runtime_state.py:45-72`
- Modify: `werewolf_agent/runtime/nodes/day_discussion.py:160-207`
- Modify: `werewolf_agent/runtime/nodes/wolf_consensus.py` at expired short-circuit
- Modify: `werewolf_agent/runtime/nodes/wolf_discussion.py:145-181`
- Modify: `werewolf_agent/runtime/nodes/__init__.py`
- Modify: `tests/runtime/test_day_discussion.py:30-85,215-235`
- Modify: `tests/runtime/test_wolf_flow.py` around `ManualTimer`
- Modify: `tests/integration/test_post_july14_repair_closure.py:220-240,380-400,665-685`
- Modify: `tests/runtime/test_timeouts_config.py`
- Modify: `README.md:123-127`

- [ ] **Step 1: 先把阶段测试改写为“不会因 deadline 跳过”RED 测试**

删除对 `ManualTimer`、`RealTimer`、`NoopTimer` 的导入与 fixture。将白天发言、狼队讨论和狼队共识测试改为：满足正常前置状态时节点调用 Agent 并推进，而不是生成 `speech_timeout`、`wolf_no_kill_timeout` 或提前跳过。删除只验证 timer 自身实现的单元测试。

`tests/runtime/test_timeouts_config.py` 改成兼容测试：各 facade 的 `AGENT_TIMEOUTS` identity 相同，并扫描 active Runtime 模块不出现 `AGENT_TIMEOUTS.` 行为读取；常量具体数值不再是生产合同。

- [ ] **Step 2: 运行 RED 测试**

```powershell
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\deadline-removal-red tests/runtime/test_day_discussion.py tests/runtime/test_wolf_flow.py tests/runtime/test_timeouts_config.py tests/integration/test_post_july14_repair_closure.py -q
```

Expected: `runtime_timer`/expired 分支仍导致阶段跳过，或 active imports 扫描 FAIL。

- [ ] **Step 3: 删除 deadline 状态和分支**

- 删除 `timers.py` 及其所有导入；
- 删除 `_timer_expired()`、`_agent_timeout()` 和相关 exports；
- 从 `RuntimeState` 删除 `runtime_timer`、`speech_timed_out`、`speech_seconds_limit` 等仅服务 deadline 的字段；
- 删除 day speech expired/`speech_timeout` 分支；
- 删除 wolf consensus expired no-kill 分支；
- 删除 wolf discussion 的 `time.monotonic()` 总时限和 `AGENT_TIMEOUTS.wolf_discussion_total` break；
- 不保留隐藏 timer facade 或 always-false expired shim。

- [ ] **Step 4: 更新 README**

将 README 的 Timer 条目改成同步调用与 provider HTTP timeout 的真实边界说明，不再引用 `runtime.timers`、`ManualTimer` 或未来才会实现的取消能力。

- [ ] **Step 5: 运行 deadline GREEN 回归与残留扫描**

```powershell
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\deadline-removal-green tests/runtime/test_day_discussion.py tests/runtime/test_day_discussion_nodes.py tests/runtime/test_wolf_flow.py tests/runtime/test_wolf_night_nodes.py tests/runtime/test_timeouts_config.py tests/integration/test_post_july14_repair_closure.py -q
rg -n "timed_call|runtime_timer|_deadlines|ManualTimer|RealTimer|NoopTimer|_timer_expired|speech_timed_out|timeout_override" werewolf_agent tests README.md
git diff --check
```

Expected: pytest 全部 PASS；`rg` 无活动代码/测试/README 命中。若迁移说明必须保留文字，逐条确认它不是 import、状态字段或执行分支。

- [ ] **Step 6: 提交 deadline 删除**

```powershell
git add -A werewolf_agent/runtime tests/runtime tests/integration/test_post_july14_repair_closure.py README.md
git commit -m "refactor: remove runtime deadline system"
```

### Task 6: 从 attempts 派生 `runtime_timeout_count` 到 ActionTrace

**Files:**

- Modify: `werewolf_agent/runtime/decision_outcomes.py:100-260`
- Modify: `werewolf_agent/agents/trace_schemas.py:85-140`
- Modify: `werewolf_agent/agents/trace_builder.py:24-118`
- Modify: `werewolf_agent/agents/player_action_result.py:20-140` if required by centralized summary API
- Modify: `werewolf_agent/runtime/agent_action_audit.py`
- Modify: `tests/agents/test_decision_trace_v2.py:68-160`
- Modify: `tests/agents/test_schemas.py` near ActionTrace counter tests
- Modify: `tests/agents/test_trace_builder.py:34-135`
- Modify: `tests/runtime/test_decision_outcomes.py`
- Modify: `tests/runtime/test_agent_action_audit.py`

- [ ] **Step 1: 写 attempt 计数和 synthetic 排除 RED 测试**

为 `summarize_attempt_counts()` 增加断言：两个真实 `RootCause.TIMEOUT` + 一个 `SAFE_FALLBACK/TIMEOUT` synthetic record 的 `runtime_timeout_count == 2`。同时覆盖 mapping 和对象形式的 attempt。

- [ ] **Step 2: 写 ActionTrace 新旧 schema RED 测试**

覆盖四个合同：

1. 新 trace 从 attempts 自动得到 count，不允许调用者传入不一致值；
2. 旧 mapping 缺字段但有 attempts，normalize 后回填；
3. 缺字段且无 attempts，读取为 0；
4. 原输入显式带错误 count，严格一致性检查失败。

验证 `model_dump(exclude_unset=True)` 对旧输入不伪装成“显式携带字段”，一致性层必须保留原始 field-presence 信息。

- [ ] **Step 3: 写 moderator-only 审计 RED 测试**

构造带 timeout attempts 的动作，断言 `action_trace_audit` payload 含 `runtime_timeout_count`，事件可见性仍是 moderator/private，公开 exposure 中不出现该字段。

- [ ] **Step 4: 运行 RED 测试**

```powershell
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\timeout-trace-red tests/agents/test_decision_trace_v2.py tests/agents/test_schemas.py tests/agents/test_trace_builder.py tests/runtime/test_decision_outcomes.py tests/runtime/test_agent_action_audit.py tests/runtime/test_exposure_audit.py -q
```

Expected: summary/schema 尚无 `runtime_timeout_count` 或 synthetic record 被错误计入。

- [ ] **Step 5: 集中实现派生投影**

在 `decision_outcomes.py` 的 `AttemptCounts` 和 summary 中新增 count，权威判定为：`root_cause == RootCause.TIMEOUT` 且 `route_kind != RouteKind.SAFE_FALLBACK`。`trace_builder.py` 只使用这个 summary，不复制过滤逻辑。

在 `ActionTrace` 增加非负 `runtime_timeout_count: int = 0`，用 before/after validator 或显式 builder 保证新构造不能提供与 attempts 不同的值；normalize 保留“字段是否原本存在”的语义，缺失字段执行回填而不是报漂移。

- [ ] **Step 6: 运行 GREEN 测试**

```powershell
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\timeout-trace-green tests/agents/test_decision_trace_v2.py tests/agents/test_schemas.py tests/agents/test_trace_builder.py tests/runtime/test_decision_outcomes.py tests/runtime/test_agent_action_audit.py tests/runtime/test_exposure_audit.py -q
git diff --check
```

Expected: 全部 PASS；ActionTrace count 与 attempts 一致，公开边界不变。

- [ ] **Step 7: 提交 trace 指标**

```powershell
git add werewolf_agent/runtime/decision_outcomes.py werewolf_agent/agents/trace_schemas.py werewolf_agent/agents/trace_builder.py werewolf_agent/agents/player_action_result.py werewolf_agent/runtime/agent_action_audit.py tests/agents tests/runtime
git commit -m "feat: project provider timeouts into action traces"
```

### Task 7: 聚合 Runtime timeout 指标并输出真实游戏报告

**Files:**

- Modify: `werewolf_agent/evaluation/decision_execution_audit.py:32-205`
- Modify: `werewolf_agent/evaluation/trace_builder.py` projection field allowlist
- Verify/re-export: `werewolf_agent/evaluation/acceptance_audit.py`, `werewolf_agent/evaluation/balance_audit.py`
- Modify: `scripts/run_real_game_reports.py:99-189`
- Modify: `tests/evaluation/test_report_acceptance_metrics.py:182-338`
- Modify: `tests/evaluation/test_final_acceptance_integrity.py`
- Modify: `tests/scripts/test_run_real_game.py`

- [ ] **Step 1: 写聚合与一致性 RED 测试**

构造多条 action trace：真实 timeout attempts、429 provider error、invalid output、synthetic safe fallback。断言 `compute_decision_execution_metrics()` 只累计真实 timeout attempts，不信任传入 count。

分别验证：

- 缺少字段但有 attempts：回填且无 consistency error；
- 缺少字段也无 attempts：0；
- 显式错误字段：报告 consistency error；
- 429 和 empty response 均不增加 timeout count。

- [ ] **Step 2: 写 CLI 报告 RED 测试**

给 `print_usage_stats()`/报告入口提供两条 timeout attempts，捕获 stdout，断言包含精确一行：

```text
Runtime timeouts: 2
```

- [ ] **Step 3: 运行 RED 测试**

```powershell
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\timeout-report-red tests/evaluation/test_report_acceptance_metrics.py tests/evaluation/test_final_acceptance_integrity.py tests/scripts/test_run_real_game.py -q
```

Expected: 指标键或输出行不存在，相关新断言 FAIL。

- [ ] **Step 4: 实现 attempt-first 聚合**

在 `decision_execution_audit.py` 对规范化 attempts 重新计算总数并排除 SAFE_FALLBACK；不要直接求和 trace 派生字段。只有原始输入显式携带 `runtime_timeout_count` 时才将其与重算值比较并记录一致性错误。

把指标通过现有 `acceptance_audit` spread 和 `balance_audit` facade 自然暴露，避免新增第二套聚合入口。

- [ ] **Step 5: 实现真实游戏报告输出**

`run_real_game_reports.py` 使用同一结构化 attempt 事实或现有聚合结果输出 `Runtime timeouts: N`，不扫描 WARNING 文本，不统计旧 `Timed call exceeded` 日志。

- [ ] **Step 6: 运行 GREEN 测试**

```powershell
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\timeout-report-green tests/evaluation/test_report_acceptance_metrics.py tests/evaluation/test_final_acceptance_integrity.py tests/evaluation/test_balance_audit_srp_split.py tests/scripts/test_run_real_game.py -q
git diff --check
```

Expected: 全部 PASS；聚合值和 stdout 均来自 attempt 证据。

- [ ] **Step 7: 提交报告指标**

```powershell
git add werewolf_agent/evaluation scripts/run_real_game_reports.py tests/evaluation tests/scripts/test_run_real_game.py
git commit -m "feat: report runtime provider timeouts"
```

### Task 8: 全面回归、残留审计与最终文档一致性

**Files:**

- Modify only if a verified gap is found: files already in Tasks 1-7
- Verify: `docs/superpowers/specs/2026-07-23-runtime-timeout-retry-deadline-removal-design.md`
- Verify: `README.md`

- [ ] **Step 1: 运行 Model Gateway 全套**

```powershell
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\final-model-gateway tests/model_gateway -q
```

记录最终 summary 的 passed/failed 数量。

- [ ] **Step 2: 分批运行 Agent 与 Runtime 回归**

```powershell
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\final-agents tests/agents/test_player_retry.py tests/agents/test_player_agent.py tests/agents/test_player_generation.py tests/agents/test_decision_trace_v2.py tests/agents/test_schemas.py tests/agents/test_trace_builder.py -q
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\final-runtime tests/runtime -q
```

每条命令必须看到完成 summary；若聚合 Runtime suite 超时，按文件分批重跑并只报告完成批次。

- [ ] **Step 3: 运行 integration、evaluation 和 scripts 回归**

```powershell
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\final-integration tests/integration/test_post_july14_repair_closure.py tests/integration/test_live_runtime.py tests/integration/test_live_game_flow.py -q
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\final-evaluation tests/evaluation/test_report_acceptance_metrics.py tests/evaluation/test_final_acceptance_integrity.py tests/evaluation/test_balance_audit_srp_split.py -q
python -m pytest --override-ini="addopts=" -p no:cacheprovider --basetemp E:\NLP\agent\wofkill\.tmp\final-scripts tests/scripts/test_run_real_game.py -q
```

- [ ] **Step 4: 编译和活动残留检查**

```powershell
python -m compileall -q werewolf_agent scripts
rg -n "timed_call|runtime_timer|_deadlines|ManualTimer|RealTimer|NoopTimer|_timer_expired|speech_timed_out|timeout_override" werewolf_agent tests README.md
rg -n "AGENT_TIMEOUTS\." werewolf_agent/runtime
rg -n "timeout:\s*(?!300\b)" config/models.yaml
git diff --check
git status --short
```

Expected:

- compileall exit 0；
- timer/deadline 搜索无活动代码命中；
- `AGENT_TIMEOUTS.` 不出现在生产 Runtime 行为代码，只允许兼容对象的导入/re-export；
- YAML 不存在非 300 的 model timeout（PowerShell 的 ripgrep 若不支持 look-around，改用 `rg -n "timeout:" config/models.yaml` 并逐行确认）；
- diff check 无输出。

- [ ] **Step 5: 审计关键验收合同**

人工逐项对照 spec，确认：

- provider/model/fallback 顺序未改变；
- primary/fallback/429 次数和 sleep 序列由已通过测试证明；
- `FailureDisposition` 不靠文本匹配；
- terminal safe fallback 仍存在但不污染 timeout 指标；
- 旧 trace 缺字段回填，显式漂移报错；
- `AGENT_TIMEOUTS` facade identity 测试通过；
- README 不再宣传已删除 API。

- [ ] **Step 6: 若回归修复产生改动，做独立收尾提交**

仅在发现并修复真实遗漏时执行：先用 `git diff --name-only` 确认文件只属于本计划，
再按 Task 1-7 已列出的精确路径分别 `git add`，最后运行：

```powershell
git commit -m "test: close runtime retry regression gaps"
```

如果没有新改动，不创建空提交。

- [ ] **Step 7: 汇总交付证据**

最终汇总包含：每批测试的精确 passed/failed 数、任何超时或未完成批次、`git diff --check`、compileall、残留扫描结果、提交列表和工作树状态。不得把未运行或超时的全套测试写成通过。
