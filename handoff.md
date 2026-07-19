# Werewolf Agent 项目交接说明

更新时间：2026-07-19 15:45（Asia/Shanghai）

## 1. 当前状态

- 仓库：`E:\NLP\agent\wofkill`
- 当前分支：`master`
- MiniMax 401 修复代码基线：`c05332c`（`docs: sync MiniMax auth verification status`）；
  `handoff.md` 的文档提交位于其后，实际 HEAD 以 `git log` 为准。
- MiniMax 401 修复分支已快进合并，临时分支与 worktree 已删除。
- 当前没有配置 Git remote，因此尚未推送远端。
- 主工作区仍保留以下用户/运行时未跟踪内容，不要擅自删除或提交：
  - `.claude/worktrees/`
  - `artifacts/`
  - `docs/audits/2026-07-15-post-july14-game-log-audit.md`

## 2. 这轮做了什么

### 2.1 7 月 14 日后问题全量修复

此前审计发现的狼队计划、空刀、终止状态、反思事务、事件证据、质量投影、
fallback 观测等问题已按设计实施，并通过修复分支合并到 `master`。

关键集成点：

- `4ea9390`：将 `codex/post-july14-full-repair` 与当时的 `master` 集成。
- `09008ca`：记录真实 soak 的策略/平台阻塞状态。
- `cb9d76a`：补齐 7 月 14 日后修复闭环门禁。
- 修复验证文档：
  `docs/audits/2026-07-15-post-july14-repair-verification.md`
- 修复设计：
  `docs/superpowers/specs/2026-07-15-post-july14-full-repair-design.md`

代码闭环已完成，但真实十局 soak 尚未完成，所以“修复代码已完成”不等于
“真实模型验收已完成”。

### 2.2 MiniMax 401 根因与修复

旧运行 `artifacts/audit_closure_soak/20260719-113342-eacbf50e56b54597b3c608cad979b75b/`
在 `seed-714001` 中出现 83 次 401，全部发往：

```text
https://api.minimaxi.com/v1/chat/completions
```

根因：

- `.env` 中没有 `MINIMAX_NATIVE_API_KEY` 或 `MINIMAX_API_KEY`。
- 现有 MiniMax 密钥配置在 `ANTHROPIC_API_KEY`，且
  `ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic`。
- 旧代码在 native MiniMax key 缺失时错误回退到 Ark 的
  `OPENAI_API_KEY`，因此稳定返回 401。

当前修复后的密钥优先级：

```text
MINIMAX_NATIVE_API_KEY
→ MINIMAX_API_KEY
→ 仅当 ANTHROPIC_BASE_URL 精确指向 api.minimaxi.com 时复用 ANTHROPIC_API_KEY
→ 全部缺失则 ProviderConfigError（请求前失败）
```

同时完成：

- 最终生效 URL 只计算一次，同时用于鉴权和 HTTP 请求。
- 只有 `https` 且 hostname 精确为 `api.minimaxi.com` 才进入 MiniMax 分支。
- `api.minimaxi.com.evil.example` 不会被误判。
- `http://api.minimaxi.com` 在发送密钥前直接失败。
- Native MiniMax 绝不再回退 Ark 的 `OPENAI_API_KEY`。
- 缺钥错误会列出可配置变量，但不会输出任何密钥值。

相关提交：

- `38dc1bb`：实现 scoped MiniMax key 复用和 fail-closed。
- `783c9f4`：补齐可操作的缺钥错误信息。
- `756f7fd`：锁定 scoped key 错误契约测试。
- `c05332c`：同步进度与真实验证状态。

设计和实施计划：

- `docs/superpowers/specs/2026-07-19-minimax-native-auth-reuse-design.md`
- `docs/superpowers/plans/2026-07-19-minimax-native-auth-reuse.md`

## 3. 已经完成并验证的内容

### 3.1 自动化验证

合并前和合并后的最终回归均完成。合并后在 `master` 上的新鲜证据：

```text
400 passed in 16.54s
Ruff: All checks passed!
git diff --check: passed
```

覆盖范围：

- `tests/model_gateway/`
- `tests/agents/test_model_router.py`
- `tests/scripts/test_run_real_game.py`

MiniMax 鉴权核心按 TDD 完成：旧实现下 9 项核心测试中有 6 项按预期失败；修复后
9/9 通过，相关三文件回归 61/61 通过。

### 3.2 实际 `.env` 的离线鉴权选择

没有输出、复制或提交任何密钥。离线加载实际 `.env` 后，resolver 结果为：

```text
selected_present=True
selected_is_anthropic_key=True
selected_is_openai_key=False
native_explicit_present=False
minimax_vendor_present=False
```

这证明当前代码会复用已有 MiniMax-scoped `ANTHROPIC_API_KEY`，不会选择 Ark key。

### 3.3 审查结果

- 规格审查：`Spec compliant`
- 代码质量审查：`Ready to merge: Yes`
- 最终整体审查：无 Critical / Important 问题
- 修复分支与 `master` 合并后差异：`0 / 0`

## 4. 当前卡在哪里

### 4.1 真实 MiniMax 请求没有完成

尝试最小真实请求时：

1. 沙箱内请求在 HTTP 之前被 `WinError 10013` 阻止。
2. 提权重试被平台的外部凭证披露策略拒绝。

请求没有到达 MiniMax，因此当前只能确认：

- 代码与实际 `.env` 会选择正确的 MiniMax key；
- 没有观察到修复后的新 401，因为请求根本未到供应商；
- 不能声称供应商已经返回 200，也不能把线上 401 标记为最终关闭。

这属于当前 Codex 执行环境限制，不是新的 provider 代码错误。

### 4.2 固定十局 soak 未完成

旧运行目录：

```text
artifacts/audit_closure_soak/20260719-113342-eacbf50e56b54597b3c608cad979b75b/
```

当前只有：

```text
seed-714001/runner.stdout.log
```

该日志包含 83 次修复前的历史 401，最后写入时间为 2026-07-19 13:56:08。
以下最终产物不存在：

```text
audit-closure-report.json
audit-closure-thresholds.json
```

旧运行不能续作或当作修复后证据，必须创建新的 run ID 从十个种子重新运行。

### 4.3 Task 17 / 最终真实平衡结论尚未开始

最终平衡分析依赖独立、完整、无 401 的真实对局样本。十局 soak 未完成前，不应基于
旧污染样本宣布阵营平衡、反思闭环或 acceptance thresholds 已通过。

## 5. 下一步怎么做

### 第一步：在普通 PowerShell 中运行最小真实探针

必须在允许向 `api.minimaxi.com` 发送凭证的终端环境中执行：

```powershell
Set-Location E:\NLP\agent\wofkill
$env:LANGCHAIN_TRACING_V2 = "false"
$env:LANGSMITH_TRACING = "false"

python -c "from werewolf_agent.model_gateway.providers.env import load_local_dotenv, get_env; load_local_dotenv('.env'); from werewolf_agent.model_gateway.providers.openai import OpenAIProvider; from werewolf_agent.model_gateway.usage_records import ModelConfig; p=OpenAIProvider(api_key=get_env('OPENAI_API_KEY')); r=p.generate('Reply with OK.', ModelConfig(provider='openai', model='MiniMax-M2.7', base_url='https://api.minimaxi.com/v1', max_tokens=16, timeout=30)); print('probe_ok=' + str(bool(r.text)) + ' provider=' + r.provider + ' model=' + r.model)"
```

成功标准：exit code 0、`probe_ok=True`、没有 401。不要打印回复原文或任何 key。

若仍返回 401：

- 不要恢复 Ark key fallback。
- 先检查 MiniMax key 是否失效、过期、账户无模型权限或 token plan 不适用。
- 如需新 key，优先配置 `MINIMAX_NATIVE_API_KEY` 或 `MINIMAX_API_KEY`。

### 第二步：探针成功后重新执行固定十局 soak

```powershell
Set-Location E:\NLP\agent\wofkill
$env:LANGCHAIN_TRACING_V2 = "false"
$env:LANGSMITH_TRACING = "false"

.\scripts\run_audit_closure_soak.ps1 `
  -Seeds (714001..714010) `
  -MaxSteps 500 `
  -TimeoutSeconds 120 `
  -DelayMilliseconds 0
```

新产物应位于：

```text
artifacts/audit_closure_soak/<new-run-id>/
```

运行期间先检查每个：

```text
seed-<seed>/runner.stdout.log
```

全部完成后必须存在：

```text
audit-closure-report.json
audit-closure-thresholds.json
```

### 第三步：基于新十局更新验收与平衡结论

1. 确认十局都进入终态，没有 `phase=night` 卡死或 step-limit 假完局。
2. 确认新日志中没有 401、胜负后调用、危险击杀、unsupported fact 或聚合差异。
3. 检查完局率、fallback、反思、预言家查验、猎人/女巫机会链等 acceptance 指标。
4. 更新 `docs/audits/2026-07-15-post-july14-repair-verification.md`。
5. 再启动 Task 17 的独立平衡分析，不要复用修复前污染样本。

## 6. 不要再踩的坑

1. **不要把 native MiniMax 缺钥回退到 `OPENAI_API_KEY`。** 该 key 属于 Ark，
   是本次 83 次 401 的直接原因。
2. **不要把请求到达 100% 当作 pytest 通过。** 必须看到 exit code 0 和最终
   `N passed` 汇总；本机曾出现测试到 100% 后退出阶段挂起。
3. **不要使用默认 pytest/ruff 缓存目录判断失败。** Windows ACL 可能导致
   `WinError 5`；使用已知可写的 `TMP`、`TEMP`、`--basetemp`、
   `RUFF_CACHE_DIR`，必要时加 `-p no:cacheprovider`。
4. **不要在 feature worktree 中默认依赖根仓库 `.env`。**
   `load_local_dotenv()` 按当前工作目录读取；worktree 没有 `.env` 时会表现为 provider
   未配置。运行真实游戏应从仓库根目录启动，或显式传入安全的 `.env` 路径。
5. **不要从任意目录传相对 `ArtifactRoot`。** 当前 soak 脚本的显式相对路径可能按
   调用者当前目录解析；从仓库根运行，或传绝对路径。
6. **不要续用旧的 401 soak 目录。** 新代码必须生成新的 run ID；旧日志只能作为
   根因证据，不能作为修复后验收样本。
7. **不要在真实 probe 未到供应商时声称线上 401 已解决。** 当前只有代码、测试和
   离线实际配置证据。
8. **不要降低门槛、随机补刀或把 fallback 当正常成功来制造“通过”。** 验收要看
   真实事件流和 decisive path，而不是只看汇总数字。
9. **不要提交或打印 `.env`、API key、Authorization header、完整敏感 prompt。**
10. **不要清理用户未跟踪文件。** `.claude/worktrees/`、`artifacts/` 和未跟踪审计
    文档当前均被刻意保留。
11. **不要把旧日志中的 83 次 401归因到新代码。** 它们发生在 `38dc1bb` 合并前；
    需要以新 run ID 的日志判断修复后表现。
12. **不要把“代码已合并”误写成“远端已发布”。** 当前没有 remote，也没有 push。

## 7. 接手时的快速检查

```powershell
Set-Location E:\NLP\agent\wofkill
git status --short --branch
git log -6 --oneline --decorate
python scripts/run_real_game.py --help
```

预期：

- 分支是 `master`，HEAD 是 `c05332c` 或其后继提交。
- 未跟踪内容仍只属于用户日志/产物/审计文件和本交接文件（若尚未提交）。
- 先跑最小探针，再决定是否花费真实模型额度执行十局。

## 8. 最终完成定义

本事项只有同时满足以下条件才可标记完全完成：

- 最小真实 MiniMax probe 到达供应商并成功，且没有 401。
- 新的固定十局全部生成终态游戏 JSON。
- 新 run 根目录生成 report 与 thresholds 文件。
- acceptance hard gates 全部通过，或对未通过项给出基于新日志的根因和修复。
- 修复验证文档更新为新 run ID 和真实指标。
- 最终 Task 17 平衡分析使用新、未污染样本完成。
