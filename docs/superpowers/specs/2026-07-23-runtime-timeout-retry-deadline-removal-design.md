# Runtime Timeout、重试与 Deadline 移除设计

## 1. 背景

最近真实游戏日志暴露了两类相互放大的运行时问题：

- `werewolf_agent.runtime.timers.timed_call()` 通过 daemon thread 执行 Agent，
  `join(timeout)` 超时后只让 Runtime 返回 fallback，无法取消仍在进行的 provider
  请求；旧线程会继续重试、写日志和消耗连接资源。
- 模型网关与动作层分别拥有重试循环。网络故障终退后，动作层可能再次发起完整
  模型调用，使尝试次数和总等待时间相乘。

本设计移除不可取消的线程超时模型和整个游戏阶段 deadline 体系，以 provider
HTTP timeout 作为唯一能够真实终止网络 I/O 的超时边界，并补齐可审计的 Runtime
timeout 指标。

## 2. 目标

1. Runtime 不再创建无法取消的 Agent 工作线程。
2. 删除 `RealTimer`、`ManualTimer`、`NoopTimer` 及所有阶段 deadline/expired 分支。
3. 所有 LLM provider/profile 的 HTTP timeout 统一为 300 秒。
4. 网络重试只由模型网关负责，使用确定性的指数退避。
5. 动作层只负责输出解析和语义修复，不在网关网络终退后重新发起网络调用。
6. 每次真正的 provider timeout 都进入逐尝试证据链，并汇总为
   `runtime_timeout_count`。
7. 保持历史执行证据、旧 ActionTrace 和 `AGENT_TIMEOUTS` 导入兼容。

## 3. 非目标

- 不把同步 provider 客户端整体迁移为 async。
- 不引入子进程隔离或强制终止 Python 线程。
- 不改变模型选择、prompt、游戏规则、角色平衡或 fallback 模型顺序。
- 不把空响应、JSON 解析失败或语义验证失败统计成 Runtime timeout。
- 不承诺 Runtime 层仍能限制非网络代码的最长执行时间。

## 4. 已确认的设计决策

### 4.1 单一同步调用生命周期

新的调用链为：

```text
Runtime -> Agent -> ModelRouter -> Provider HTTP client
                                      |
                                      +-- HTTP timeout = 300s
```

Runtime 直接同步调用 Agent。provider HTTP 客户端超时时抛出异常，当前调用栈负责
记录失败、执行重试或进入 fallback；不存在已经返回 Runtime、但仍在后台继续运行的
旧请求。

### 4.2 删除 Runtime deadline

删除以下活动能力：

- `werewolf_agent/runtime/timers.py`；
- `RealTimer`、`ManualTimer`、`NoopTimer`、`RuntimeTimer` 和 `timed_call`；
- `RuntimeState.runtime_timer`、`agent_call_timeout`、`wolf_vote_timeout`；
- `_timer_expired()`、`_agent_timeout()`、`timeout_override`；
- 所有 `runtime_timer.expired(...)`、阶段过期短路和 `timed_call(...)` 调用。

原先通过 `ManualTimer(expired_keys=...)` 测试的“阶段主动跳过”行为被删除。阶段节点在
满足游戏状态前置条件时正常执行，不再因 Runtime deadline 静默跳过。

`werewolf_agent.runtime.timeouts.AGENT_TIMEOUTS` 保留为 deprecated 兼容入口，避免
外部导入立即失败。现有兼容 facade（至少包括 `runtime.agent_adapter`、
`runtime.agent_action_pipeline` 和 `runtime.nodes._shared`）继续 re-export 同一个对象；
facade 导入不构成行为调用。生产 Runtime 的节点、调度和流程控制不得读取这些常量，
兼容入口不构成活动 deadline。

### 4.3 Provider HTTP timeout

- `ModelConfig.timeout` 默认值改为 `300`。
- YAML 中所有 model profile 的显式 `timeout` 改为 `300`。
- 路由配置解析缺省 timeout 时使用 `300`。
- OpenAI、Anthropic、MiniMax、GLM 等 provider 均继续把 `config.timeout` 传给实际
  HTTP 客户端。

HTTP timeout 是唯一超时边界。它必须由 provider 客户端真正中止网络请求，而不是
由另一个线程提前返回。

## 5. 网络重试策略

### 5.1 配置兼容与有效预算

`ModelConfig.retry_count` 保留，默认从 2 改为 4。所有“profile 未显式配置
`retry_count`”的解析入口必须同步使用 4：包括 `ModelConfig` dataclass 默认值、
primary `_resolve_config()` 当前的缺省值 2，以及 fallback `_fallback_config()` 当前的
缺省值 1。不能只改 dataclass 而让 YAML 路由解析继续保留旧默认。

profile 仍可显式降低重试次数或以 `retry_count: 0` 关闭重试。路由类型再施加上限：

| 错误类型 | 路由 | 最大重试次数 | 有效预算 |
| --- | --- | ---: | --- |
| 普通瞬时网络错误/timeout/5xx | primary | 4 | `min(config.retry_count, 4)` |
| 普通瞬时网络错误/timeout/5xx | fallback | 2 | `min(config.retry_count, 2)` |
| HTTP 429 | primary | 3 | `min(config.retry_count, 3)` |
| HTTP 429 | fallback | 3 | `min(config.retry_count, 3)` |

重试次数不包含第一次调用；例如 primary 普通网络错误最多产生 5 条 provider attempt。

预算按“单个 route candidate”计算。primary 是一个 candidate；fallback chain 中每个
不同的 provider/model candidate 在切入时获得自己的预算，切换 candidate 时重新计数。
同一 candidate 内不因错误类型从普通网络错误切换到 429（或反向切换）而重置计数。

每个 candidate 同时维护三个计数：

- `total_retry_count`：受 `config.retry_count` 约束；
- `generic_retry_count`：primary 上限 4，fallback 上限 2；
- `rate_limit_retry_count`：primary/fallback 上限均为 3。

一次失败只有在“总预算尚未耗尽”且“当前错误类别预算尚未耗尽”时才能重试。这样混合
错误序列也始终被 `config.retry_count` 限制。例如默认 `retry_count=4` 的 fallback
可以经历 2 次普通错误和 2 次 429 重试，但不能超过 4 次总重试；连续普通错误仍最多
2 次，连续 429 仍最多 3 次。

### 5.2 普通瞬时错误退避

普通 retryable 网络错误使用确定性序列，无随机 jitter：

- primary：`2s -> 4s -> 8s -> 16s`；
- fallback：`2s -> 4s`。

只有连接失败、HTTP timeout、429、5xx 等瞬时 transport/provider 错误可进入网络
重试。不可重试的 4xx、策略拒绝和本地编程错误立即终止当前路由。

### 5.3 HTTP 429 退避

429 使用独立序列：`16s -> 32s -> 64s`，primary 和 fallback 各自最多重试
3 次。

每次实际等待为：

```text
min(300s, max(parsed_retry_after, exponential_baseline))
```

其中：

- `Retry-After` 支持 delta-seconds 和 HTTP-date；
- 缺失、无效或已经过去的 HTTP-date 视为没有有效服务端等待值；
- 服务端值小于当前指数基线时使用指数基线；
- 服务端值或指数结果超过 300 秒时封顶到 300 秒。

重试日志必须同时包含 route、attempt、最大次数、错误类别和最终等待秒数。

### 5.4 动作层重试隔离

动作层的重试只处理：

- 空响应；
- JSON/structured-output 解析失败；
- 动作 schema 或游戏语义验证失败；
- 可修复的输出质量问题。

`GenerateResult` 新增强类型 `failure_disposition`，默认值为 `none`，终退时只能取：

- `transport_exhausted`：至少一个真实 provider attempt 为 `timeout` 或
  `provider_error`，且整条 route chain 没有成功；包括连接错误、5xx、429、不可重试
  provider 4xx，以及 transport/output 混合失败；
- `output_repairable`：所有真实 provider 失败 attempts 都是 `invalid_output`；
- `policy_rejected`：决策性失败来自 reasoning/structured-output 策略拒绝；
- `route_unavailable`：没有任何可调用的 provider candidate。

该字段由 ModelRouter 在 route chain 终退边界根据完整 attempts 设置，动作层不得重新
解析异常字符串。动作层仅对 `output_repairable` 继续输出/语义修复；
`transport_exhausted`、`policy_rejected` 和 `route_unavailable` 都立即进入安全
fallback，不再启动下一次完整模型生成。这样可以避免“网关重试次数 x 动作重试次数”
的乘法放大。transport/output 混合失败按 `transport_exhausted` 处理，优先阻止网络
故障后的再次调用。

空响应继续归因为 `invalid_output`，而不是 timeout；它可以进入动作层语义修复，但
不消耗网络重试预算。

## 6. Runtime Timeout 指标合同

### 6.1 权威事实

现有 `RootCause.TIMEOUT = "timeout"` 保持不变，避免破坏历史 JSON 和枚举兼容。
每个 `httpx.TimeoutException`、标准 `TimeoutError` 或明确的 provider timeout 异常，
都生成：

```text
AttemptExecutionRecord(
    attempt_outcome="attempt_failure",
    root_cause="timeout",
    ...
)
```

每次真实 provider 调用对应一条独立 attempt；retry 和 provider fallback 不得合并
真实 provider attempt。现有 synthetic `RouteKind.SAFE_FALLBACK` 记录继续保留，
因为 `DecisionOutcome.TERMINAL_FALLBACK` 依赖它表达决策边界；它不是 provider 调用，
不得计入 Runtime timeout provider-attempt 指标。

### 6.2 ActionTrace 投影

`ActionTrace` 新增：

```text
runtime_timeout_count: int = 0
```

它是 `execution_attempts` 中真实 provider attempts（排除
`RouteKind.SAFE_FALLBACK`）里 `root_cause == "timeout"` 的派生投影。

- 新 trace 构建时必须从 attempts 计算，不允许调用者手工提供不同值。
- 旧 trace 如果缺少该字段但带有 attempts，归一化时从 attempts 回填。
- 旧 trace 同时缺少字段和 attempts 时才按 0 读取。
- 只有输入中显式携带该字段时才执行“输入值必须等于 attempts 派生值”的严格一致性
  校验；字段缺失触发的是回填，不是漂移错误。

字段随 moderator-only 的 `action_trace_audit` 输出，不进入公开游戏状态。

### 6.3 报告聚合

- `compute_decision_execution_metrics()` 从真实 provider attempt 证据重新计算
  `runtime_timeout_count`，排除 synthetic safe-fallback record，不信任传入的派生
  计数。
- 报告一致性检查只对原始输入显式携带 `runtime_timeout_count` 的 trace 验证投影与
  attempt 事实相等；字段缺失的历史 trace 先归一化回填。
- `scripts/run_real_game_reports.py` 输出 `Runtime timeouts: N`。
- acceptance/report 指标使用结构化 attempts，不扫描 WARNING 日志文本。

429 继续归入 `provider_error`；空响应继续归入 `invalid_output`。被删除的旧
`Timed call exceeded` 日志不迁移为新指标，因为它不是由 HTTP 客户端确认终止的请求。

## 7. 组件职责

### `werewolf_agent/model_gateway/retry_policy.py`

- 唯一负责 retryable 分类、route-specific retry budget、指数序列和
  `Retry-After` 解析。
- 退避计算必须是可独立测试的纯函数。

### `werewolf_agent/model_gateway/router.py`

- 负责执行 primary/fallback attempt 和调用 sleep。
- 不自行复制指数公式。
- 每次调用后立即追加不可变 attempt 证据。

### `werewolf_agent/model_gateway/execution_records.py`

- 继续提供 `RootCause.TIMEOUT` 权威枚举。
- 不新增含义重复的 `runtime_timeout` root cause。

### `werewolf_agent/model_gateway/usage_records.py`

- 定义 `FailureDisposition` 强类型枚举，并在 `GenerateResult` 上提供默认 `none`。
- 保持旧 provider 构造调用兼容；只有 ModelRouter 终退边界设置非 `none` 值。

### `werewolf_agent/agents/*`

- 从 attempts 派生 `runtime_timeout_count`。
- 区分终态 transport failure 与可修复输出失败。
- transport failure 耗尽后直接安全 fallback。

### `werewolf_agent/runtime/*`

- 删除 deadline/timer 机制和线程 timeout 包装。
- 保留正常异常边界和安全 fallback，不承担 provider 重试。

### `werewolf_agent/evaluation/*` 与 `scripts/run_real_game_reports.py`

- 从结构化 attempt 证据汇总 Runtime timeout。
- 保持 moderator/private 边界和旧 trace 兼容。

## 8. 错误处理序列

### 普通 timeout

```text
provider call
  -> TimeoutException
  -> append root_cause=timeout attempt
  -> sleep 2/4/8/16 according to route budget
  -> primary exhausted
  -> provider fallback
  -> fallback exhausted
  -> terminal safe fallback
  -> action layer does not call ModelRouter again
```

终态仍追加一条 synthetic `RouteKind.SAFE_FALLBACK` 决策记录，但该记录不是 provider
attempt，不能增加 `runtime_timeout_count`。

### HTTP 429

```text
provider call
  -> HTTP 429
  -> append root_cause=provider_error attempt
  -> parse Retry-After
  -> sleep min(300, max(Retry-After, 16/32/64))
  -> retry at most 3 times for the current route
```

### Invalid output

```text
provider returns empty/invalid structured output
  -> append root_cause=invalid_output attempt
  -> no network backoff
  -> action parser/semantic repair path
```

## 9. 测试策略

所有行为修改遵循 RED-GREEN-REFACTOR，测试通过 monkeypatch/fake clock 捕获 sleep，
不得真实等待 2 秒或 300 秒。

### 重试策略单元测试

- primary 普通错误精确生成 `2/4/8/16`；
- fallback 普通错误精确生成 `2/4`；
- primary/fallback 429 各自生成 `16/32/64`；
- `Retry-After` 秒数和 HTTP-date 均取与基线的较大值；
- 等待值封顶 300 秒；
- `retry_count: 0` 关闭重试；
- 非 retryable 4xx 不 sleep。

### Router 测试

- primary、fallback、429 的真实调用次数和 attempt ordinal 正确；
- fallback chain 每个 candidate 重置预算，同一 candidate 的混合错误不重置总预算；
- 混合错误同时受总预算和分类预算限制；
- timeout attempts 全部为 `RootCause.TIMEOUT`；
- 429 attempts 为 `RootCause.PROVIDER_ERROR`；
- 空响应不触发网络 backoff；
- usage、retry_count 与 attempts 保持一致。
- safe-fallback synthetic record 继续支撑 terminal decision outcome，但不计入 Runtime
  timeout。

### Agent 测试

- transport failure 耗尽后只调用一次模型网关动作流程；
- `FailureDisposition` 的四种终退值均有动作层分支测试；
- transport/output 混合失败得到 `transport_exhausted`，不会重新调用网关；
- invalid output 仍能进入解析/语义修复；
- `runtime_timeout_count` 与 attempts 精确一致；
- terminal fallback 保留完整失败证据。

### Runtime 测试

- 删除 timer/deadline 行为测试和依赖 `ManualTimer` 的 fixtures；
- 验证 Agent 在当前线程同步执行；
- 验证 Runtime 不创建后台线程；
- 删除 `expired()` 导致阶段跳过的旧断言，改为验证阶段正常推进；
- `AGENT_TIMEOUTS` 仍可导入，但没有生产 Runtime 调用者。

### 指标和报告测试

- ActionTrace 新旧 schema 往返：缺字段但有 attempts 时回填，显式错误计数时拒绝或
  计入一致性错误，缺字段且无 attempts 时为 0；
- `action_trace_audit` 可见性保持 moderator-only；
- decision execution metrics 聚合 attempt-level timeout；
- 派生计数漂移会触发一致性错误；
- real-game report 输出 Runtime timeout 总数。

### 回归验证

- `tests/model_gateway`；
- 相关 `tests/agents`；
- `tests/runtime`；
- 相关 `tests/evaluation` 和 `tests/scripts/test_run_real_game.py`；
- `python -m compileall`；
- `git diff --check`；
- 全局检查活动代码不存在 `timed_call`、`runtime_timer`、`_deadlines` 或阶段
  `expired()` 残留。
- 更新 README 中对 `runtime.timers` 的模块说明，避免文档指向已删除模块。

## 10. 兼容性与迁移

- `RootCause.TIMEOUT` 不改名。
- `ActionTrace.runtime_timeout_count` 对旧记录按 attempts 回填；只有同时缺少 attempts
  时默认为 0。
- `ModelConfig.retry_count` 保留，默认提升为 4；显式低值继续生效。
- `AGENT_TIMEOUTS` 及既有 facade re-export 保留 deprecated 兼容导出；删除它们属于
  未来独立破坏性变更。只有兼容 import 可以引用该对象，活动 Runtime 行为不得读取。
- 删除 `timers.py` 会移除其类和函数导入，这是用户明确批准的 Runtime deadline
  API 移除范围。

## 11. 风险与约束

### 最坏等待时间

300 秒 HTTP timeout 与完整主/fallback 重试预算意味着单动作在持续故障下可能超过
40 分钟。429 在主/fallback 均耗尽时也可能累计较长等待。本设计按已确认需求接受
该代价，并通过结构化 attempts 和指标使等待原因可见。

### 非网络阻塞

移除 Runtime deadline 后，纯 Python 死循环、第三方同步代码死锁或不遵守 timeout
的客户端将不再有外层保护。本次不以线程或子进程重新包装，因为那会重新引入不可取消
或高复杂度的执行模型。

### 兼容测试

部分测试通过 `ManualTimer(expired_keys=...)` 模拟阶段跳过。这些不是生产业务规则，
必须随 deadline 能力删除而重写，不能保留隐藏的 timer facade。

## 12. 验收标准

1. 活动 Runtime 代码不创建 Agent timeout 线程。
2. 活动代码不存在 Runtime deadline/timer 状态或 expired 分支。
3. 所有 provider HTTP timeout 解析为 300 秒。
4. 普通 primary/fallback 和 429 退避序列、次数符合本设计。
5. 网关 transport failure 耗尽后动作层不重新调用网关。
6. 每个 HTTP timeout 都有 `RootCause.TIMEOUT` attempt 证据。
7. ActionTrace、审计事件和最终报告的 `runtime_timeout_count` 可追溯且一致。
8. fallback candidate 与混合错误序列的重试总数符合双预算合同。
9. synthetic safe-fallback record 继续支持终态决策，但不污染 provider timeout 指标。
10. 旧 trace 与 `AGENT_TIMEOUTS` 导入兼容。
11. README 不再引用已删除的 `runtime.timers` 活动能力。
12. 相关测试和静态检查全部完成并有最终汇总；超时的测试运行不得报告为通过。
