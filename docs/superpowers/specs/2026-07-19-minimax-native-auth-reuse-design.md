# MiniMax 原生端点鉴权复用修复设计

## 1. 背景与根因

2026-07-19 的固定十局 soak 在第一局 `seed-714001` 中停止。对应
`runner.stdout.log` 记录了 83 次 `401 Unauthorized`，全部发往
`https://api.minimaxi.com/v1/chat/completions`，涉及 `MiniMax-M3` 与
`MiniMax-M2.7`。

当前鉴权链路在 `OpenAIProvider` 遇到 `api.minimaxi.com` 时只读取
`MINIMAX_NATIVE_API_KEY`。该变量缺失后，代码回退到 provider 实例的
`OPENAI_API_KEY`；但当前 `OPENAI_API_KEY` 属于 Ark 火山端点，因此请求稳定
返回 401。仓库根 `.env` 已为 MiniMax Anthropic-compatible 端点配置
`ANTHROPIC_API_KEY` 与 `ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic`。
MiniMax 官方文档规定 Anthropic-compatible 与 OpenAI-compatible 接口使用同一
MiniMax API Key，因此用户已批准在可证明属于 MiniMax 的条件下复用该密钥。

## 2. 目标与边界

目标：

1. `api.minimaxi.com` 的 OpenAI-compatible 请求使用 MiniMax 密钥，不再发送
   Ark 的 `OPENAI_API_KEY`。
2. 保留显式专用密钥的最高优先级，并兼容未来迁移到 `MINIMAX_API_KEY`。
3. 缺少所有合法 MiniMax 密钥时，在发起 HTTP 请求前给出可操作的配置错误。
4. 用测试直接验证最终请求的 `Authorization` 选择，而不是只验证环境读取函数。

边界：

- 不修改五名玩家的模型分配、模型名称、endpoint 或 fallback 设计。
- 不复制、写入、打印或提交任何真实密钥。
- 不改变 Ark、OpenAI 及其他 OpenAI-compatible endpoint 的现有鉴权。
- 不通过重试掩盖 401，也不自动执行新的十局付费 soak。

## 3. 方案选择

采用端点感知的显式密钥解析，未采用以下备选方案：

- 启动时把 `ANTHROPIC_API_KEY` 隐式复制成 `MINIMAX_NATIVE_API_KEY`：来源不透明，
  且会制造进程级隐藏状态。
- 把五名玩家改回 Anthropic-compatible provider：会改变已批准的路由和输出协议，
  超出本次鉴权修复范围。

## 4. 鉴权解析设计

对 `ModelConfig.base_url` 使用 URL 解析得到 hostname。只有 hostname 精确等于
`api.minimaxi.com` 时进入 MiniMax 原生端点分支，避免使用可被相似恶意域名命中的
子串判断。

MiniMax 原生端点的密钥优先级固定为：

1. `MINIMAX_NATIVE_API_KEY`：端点专用显式覆盖。
2. `MINIMAX_API_KEY`：供应商级标准配置。
3. `ANTHROPIC_API_KEY`：仅当 `ANTHROPIC_BASE_URL` 的 hostname 也精确等于
   `api.minimaxi.com` 时复用。

三者均缺失时抛出 `ProviderConfigError`，错误消息只列出允许的变量名，不包含密钥
内容。此分支绝不回退到 `OPENAI_API_KEY`。非 MiniMax hostname 继续使用 provider
实例的默认密钥，保持 Ark 与其他端点行为不变。

数据流如下：

```text
ModelConfig.base_url
→ 解析 hostname
→ 非 api.minimaxi.com：使用 OpenAIProvider 默认密钥
→ api.minimaxi.com：按专用键、供应商键、受约束的 Anthropic 键依次解析
→ 缺钥：ProviderConfigError（HTTP 请求前）
→ 有键：构造 Authorization: Bearer <resolved key>
```

## 5. 测试设计

先修改测试捕获客户端，使其保存最终请求 headers，再按 TDD 增加并观察以下测试失败：

1. 专用键存在时，最终 `Authorization` 使用 `MINIMAX_NATIVE_API_KEY`。
2. 专用键缺失、`MINIMAX_API_KEY` 存在时使用供应商级键。
3. 前两者缺失且 `ANTHROPIC_BASE_URL` 指向 MiniMax 时，复用
   `ANTHROPIC_API_KEY`。
4. `ANTHROPIC_BASE_URL` 指向其他 hostname 时不得复用该键。
5. 所有 MiniMax 键缺失时抛出 `ProviderConfigError`，不得使用默认 Ark 键。
6. Ark endpoint 即使存在 MiniMax 键也继续使用默认 Ark 键。
7. 相似但非精确 hostname 不进入 MiniMax 鉴权分支。

实现后运行相关 provider、dotenv、路由与配置测试，并对修改文件执行 Ruff、
`git diff --check`。只有离线回归通过后，才执行一次最小真实 MiniMax 请求验证不再
返回 401；该验证不得输出密钥或完整敏感 prompt。固定十局 soak 仍需单独启动。

## 6. 成功标准

- 当前 `.env` 不新增或复制密钥也能为 `api.minimaxi.com` 选择已有 MiniMax 密钥。
- 单次真实验证不再返回 401。
- 非 MiniMax endpoint 的请求头与修复前一致。
- 缺钥场景在网络调用前失败，错误可定位且不泄露凭证。
- 相关自动化测试、Ruff 与差异检查全部通过。
