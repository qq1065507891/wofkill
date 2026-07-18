# 7 月 14 日后全量修复验证报告

> 状态：`PENDING_LIVE_SOAK`  
> 代码与离线门禁：通过  
> 固定十局真实模型 soak：尚未获准向外部模型提供商发送真实游戏提示，因此 Task 16 尚未最终关闭。

## 1. 验收范围与结论

本轮把 7 月 14 日后检查报告中的隐藏失败路径纳入自动门禁，不再仅依赖胜率、平安夜比例等表面指标。阈值评估器当前包含 44 项 fail-closed 硬门禁；本轮新增的十项分别覆盖：

1. `attempt_retry_consistency_error_count == 0`；
2. `malformed_resolution_batch_count == 0`；
3. `same_route_provider_fallback_count == 0`；
4. `saved_offline_quality_diff_count == 0`；
5. `finished_without_winner_count == 0`；
6. `abort_terminal_coverage_rate == 1.0`；
7. `majority_wolf_kill_execution_rate == 1.0` 且 `single_wolf_kill_execution_rate == 1.0`；
8. `empty_reflection_success_count == 0`；
9. `source_event_id_traceability_rate == 1.0`；
10. `public_skill_resolution_leak_count == 0`。

离线实现、正反例门禁、全量代码回归均已通过。真实十局数据尚未生成，所以这些门禁只能确认“评估器能正确 fail closed”，不能宣称“真实模型样本已全绿”。

## 2. 实现与隐藏问题修复

- 新增统一闭环聚合器 `post_july14_closure_metrics.py`，对缺字段、空样本和零分母显式返回 unsupported，避免用 0/0 伪装 100%。
- 保存后的 `quality_score` 通过完整 JSON 离线重算并逐字段归一化比较。
- 狼队合法多数和单狼执行分开计数、分开要求非零证据。
- soak 强制显式传入恰好十个唯一 seed；每个 seed 使用独立目录和 run-scoped game ID；只消费脚本本次返回的十个 JSON。
- aborted 保留原始产物和 `termination_reason`，不补跑成十个 finished；manifest 明确给出 launch/finished/aborted。
- 发现并修复旧测试 provider 未实现 `final_prompt_observer` 的兼容债务；生产 provider 仍保持 fail closed。
- GameRunner 新增两层进展守卫：连续 50 次相同 phase/day/night 快照，或连续 100 次存活集合不变，均以 `step_limit` 结构化中止，阻止“昼夜数字变化但无人出局”的伪进展。
- 反思节点不再把零个有效 V2 条目记录为成功；狼队公开预言家威胁只进入讨论提示，不恢复跨夜旧计划的执行权威。
- 集成测试的确定性 mock 按当前工具 schema 投影动作；预言家必选查验不再因旧提示文本解析而连续产生 `seer_check_skipped`。

## 3. 自动验证证据

以下均在 `codex/post-july14-full-repair` 工作树执行；退出码均为 0。默认工作树 `.pytest_cache` 存在已知 Windows ACL 警告，测试临时数据使用独立可写 `--basetemp`，警告未改变退出码或测试结果。

| 验证批次 | 精确结果 |
|---|---:|
| `python -m pytest -n 0 ... tests/agents tests/model_gateway -q` | 1056 passed，137.07s |
| `python -m pytest -n 0 ... tests/runtime -q` | 1695 passed，1 skipped，118.97s |
| `python -m pytest -n 0 ... tests/evaluation -q` | 451 passed，8.66s |
| integration（排除两个 live 文件） | 110 passed，1 skipped，19.38s |
| `tests/integration/test_live_runtime.py` | 6 passed，247.09s |
| `tests/integration/test_live_game_flow.py` | 8 passed，45.76s |
| `python -m pytest -n 0 ... tests/scripts -q` | 189 passed，46.47s |
| 最终聚焦闭环批次（threshold/soak/analyzer/runner/wolf-plan/GameRunner/reflection） | 329 passed，75.38s |
| K1–K3、N1–N12 映射结构与 pytest node 存在性 | 2 passed，15/15 case 已映射，0.94s |
| 额外 platform 批次（api/core/rules/storage/tools） | 582 passed，81.84s |
| 额外 experience 批次（customization/persona_runtime/ui） | 76 passed，22.49s |
| 额外 cognition | 150 passed，2.31s |
| 额外 memory | 262 passed，3.31s |
| 额外 rag | 298 passed，24.24s |
| 额外 skills | 159 passed，4.61s |

Task 16 计划要求的七个主要目录合计执行 3515 个 passed、2 个 skipped、0 failed。连同额外目录，本轮已有最终 summary 的非重叠批次合计 5042 个 passed、2 个 skipped、0 failed。

静态检查：

- 对本轮实际修改的生产文件和非兼容聚合测试运行 Ruff：`All checks passed!`，退出码 0。
- `git diff --check`：无输出，退出码 0。
- `tests/runtime/test_runtime.py` 是历史兼容聚合入口，`tests/runtime/test_graph_lifecycle.py` 存在本轮之前的未使用导入；未借本任务扩大为无关清理，二者由完整 pytest 覆盖。

## 4. K/N 闭环状态

K1–K3 与 N1–N12 共 15 项均至少映射一个当前存在的 pytest node，映射结构门禁为 15/15。对应节点已分布在本轮通过的 runtime、evaluation、integration、model_gateway 批次中。

递归包装测试 `test_mapped_audit_nodes_execute_as_nonrecursive_closure_batch` 会另起 pytest 子进程再次执行 16 个映射节点；当前 Windows 运行超过其常规观察窗口且尚无最终 summary，因此该包装器本次记为“未验证”，不伪报通过。它不改变上述 16 个节点在非递归批次中已通过的事实。

## 5. 固定十局 soak

计划命令：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command `
  "& '.\scripts\run_audit_closure_soak.ps1' `
  -Seeds @(714001,714002,714003,714004,714005,714006,714007,714008,714009,714010) `
  -ArtifactRoot 'artifacts/post_july14_repair_soak'"
```

当前有效结果：

| 项目 | 值 |
|---|---:|
| 有效模型 launch | 0 |
| finished | 0 |
| aborted | 0 |
| artifact root | 尚未生成 |
| hard gate 真实样本结果 | `PENDING_LIVE_SOAK` |

两次本地预启动均未调用模型：第一次被 PowerShell execution policy 拒绝；第二次因桌面沙箱阻止嵌套 PowerShell 写工作树 `artifacts/` 而失败。随后申请受控提权时，安全审查指出该命令会读取主检出目录 `.env`，并把真实游戏提示/响应发送到其中配置的外部模型目的地；由于尚无“这些外部目的地是受信系统”的证据，请求被拒绝。没有通过复制密钥、改变输出位置或其他方式绕过该限制。

## 6. 最终判定

- 自动门禁实现：PASS。
- 全量代码回归：PASS（0 failed；具体分组见上表）。
- K/N 映射：PASS，15/15。
- 精确十局真实模型 soak：`PENDING_LIVE_SOAK`。
- Task 16 总体：PENDING，不能提交“所有硬门禁全绿”的结论。
- Task 17 的 30 个 finished 平衡批次不得启动；按计划，P1/稳定性十局未关闭前不能进入数值平衡结论。

解除阻塞所需授权：用户明确同意读取 `E:\NLP\agent\wofkill\.env`，并向其中配置的外部模型提供商发送真实游戏提示与响应。授权后应重新执行上述精确十局命令，记录唯一 artifact root、10 次 launch、finished/aborted、44 项阈值 JSON，并据实更新本报告。
