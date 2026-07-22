# Kimi-K2.6 替换为 DeepSeek-V4-Pro 设计

## 状态

- 日期：2026-07-23
- 状态：用户已批准
- 范围：模型路由配置与对应测试

## 背景与根因

最近三局 `g_1006227394`、`g_2521088742` 和进行中的
`g_2091187878` 共出现 103 条 retryable warning，其中 99 条来自分配给
`Kimi-K2.6` 的 `p02`、`p04`、`p09`、`p12`。三局还记录了 33 次
Kimi 终止级网络失败和 47 次外层 agent timeout。

主触发点是 Ark endpoint 上 `Kimi-K2.6` 在当前长提示负载下频繁超过
120 秒读取期限，并伴随 `RemoteProtocolError`。同一 endpoint 上的
DeepSeek 路由没有出现同等规模的问题，因此本次先替换模型，不把超时架构
改造混入同一变更。

现有超时配置会放大现象：模型单次 timeout 为 120 秒，默认最多尝试三次，
而投票和发言外层只等待 180/240 秒；`timed_call` 超时后不会取消工作线程。
这解释了外层已经兜底后，后台仍继续打印重试和终止失败日志。该机制属于
后续独立议题，不在本次实现范围内。

## 目标

1. 从活动模型配置中移除 `Kimi-K2.6`。
2. 将原 Kimi 流量改为精确模型 ID `DeepSeek-V4-Pro`。
3. 保留原 Kimi profile 的采样参数 `temperature: 0.5`、`top_p: 0.9`、
   `timeout: 120` 及结构化输出策略。
4. 消除 `ark_kimi_k26` 这类与实际模型不一致的命名。
5. 用路由测试锁定玩家、任务与模型的最终绑定关系。

## 非目标

- 不修改 retry_count、HTTP timeout、agent timeout 或线程取消语义。
- 不改变 native MiniMax 玩家分配。
- 不改变 `ark_deepseek_v4_pro_secondary` 的 `0.6/0.95` 采样通道。
- 不改变五类显式 `judge_*` 任务到 `minimax_m27_default` 的覆盖路由。
- 不接管、修改或提交当前未跟踪的 `scripts/probe_all_endpoints.py`。
- 不修改历史进度记录、既有审计文档或描述历史 Kimi 行为的测试注释。
- 不停止或重启当前进行中的游戏。

## 设计

### 1. 统一主 DeepSeek profile

保留现有 `ark_deepseek_v4_pro` profile，并将其实际模型 ID 从
`deepseek-v4-pro` 规范化为 `DeepSeek-V4-Pro`。该 profile 已使用
`temperature: 0.5` 和 `top_p: 0.9`，与被替换的 Kimi profile 一致，
因此能够在模型替换之外最大限度保留现有生成行为。

删除 `ark_kimi_k26` profile，避免留下错误的模型语义和重复配置。

### 2. 重接活动路由

把所有活动的 `ark_kimi_k26` 引用改为 `ark_deepseek_v4_pro`：

- `llm_profiles.ark_deepseek.default`
- `llm_profiles.minimax_default.default`
- `llm_profiles.minimax_default.tasks.speech`
- `llm_profiles.minimax_default.tasks.deception`
- `llm_profiles.minimax_default.tasks.night_action`

玩家与任务侧结果为：

- `p02`、`p04`、`p09`、`p12` 经 `minimax_default` 的 default、speech、
  deception、night_action 路由使用 `DeepSeek-V4-Pro`；reflection 继续使用
  `minimax_m27_reflection`。
- `judge` 的五类显式 `judge_*` 任务继续使用 `minimax_m27_default`。只有未来
  未列出、会落到 `minimax_default.default` 的任务才解析为 `DeepSeek-V4-Pro`。
- `p11` 经 `ark_glm` 使用同一主 profile。
- `p07` 继续使用 `ark_deepseek_v4_pro_secondary`，保留独立采样通道。
- 其他玩家路由不变。

`minimax_default.fallback` 继续使用 `ark_deepseek_v4_flash`。只纠正附近把
它描述成“不同 provider”的陈旧注释；实际是同一 Ark/OpenAI provider 下的
不同模型 profile，本次不改变 fallback 行为。

### 3. 保留用户未跟踪探测脚本

`scripts/probe_all_endpoints.py` 当前是未跟踪的用户文件，不属于已提交仓库内容。
本次不修改、不暂存也不接管该文件。脚本中残留的 Kimi 探测项不参与运行时路由，
且脚本当前统一使用 `temperature=0.0`、不传 `top_p`，本身不能验证两个 DeepSeek
profile 的采样通道差异。完成时把这一现状作为独立说明报告给用户。

### 4. 测试策略

按 TDD 顺序先修改路由测试，使其在旧配置上失败，再修改生产配置：

1. 断言活动配置不再包含 `ark_kimi_k26` 或 `Kimi-K2.6`。
2. 断言 `ark_deepseek_v4_pro.model == "DeepSeek-V4-Pro"`。
3. 断言 `minimax_default` 的 default/speech/deception/night_action 均解析到
   `ark_deepseek_v4_pro`。
4. 断言 reflection 与五类显式 `judge_*` 路由保持 MiniMax，不被本次替换波及。
5. 断言原有玩家分配、native MiniMax 路由和 secondary profile 不漂移。
6. 把陈旧的 `test_minimax_default_fallback_uses_different_provider` 改为验证
   “同一 Ark/OpenAI provider、不同模型 profile”，使测试名称和断言反映实际行为。

随后运行聚焦测试、相关 model_gateway 回归、YAML 路由加载检查、
`compileall` 和 `git diff --check`。

## 文件边界

- 修改 `config/models.yaml`：唯一生产配置变更。
- 修改 `tests/model_gateway/test_minimax_provider_routing.py`：路由回归契约。

不新增生产模块，不修改运行时 Python 文件，避免把配置替换扩大成架构重构。
未跟踪的 `scripts/probe_all_endpoints.py` 保持原状且不进入提交。

## 生效与运维说明

`ModelRouter` 在进程启动或游戏 runner 初始化时读取 YAML，本次变更不会热更新
当前 `g_2091187878`。需要在当前游戏结束后重启游戏进程，或启动一个读取新配置
的新进程，替换才会生效。

## 验收标准

- 活动配置中不存在 `Kimi-K2.6` 或 `ark_kimi_k26`。
- `p02`、`p04`、`p09`、`p12` 的 default/speech/deception/night_action 解析为
  `provider=openai`、`model=DeepSeek-V4-Pro`。
- reflection 与五类显式 `judge_*` 任务保持原 MiniMax 路由。
- 主 profile 保持 `temperature=0.5`、`top_p=0.9`、`timeout=120`。
- secondary profile 和非 Kimi 玩家路由保持不变。
- 聚焦测试与相关回归全部完成并通过。
- 当前游戏不被中止或修改。
