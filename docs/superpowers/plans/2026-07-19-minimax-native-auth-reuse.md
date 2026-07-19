# MiniMax Native Auth Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `api.minimaxi.com` 的 OpenAI-compatible 请求按显式优先级使用 MiniMax 密钥，彻底禁止回退到 Ark 的 `OPENAI_API_KEY`，并用一次最小真实请求证明 401 已消失。

**Architecture:** 保持路由、模型分配和 provider 类型不变，只在 `OpenAIProvider` 的单次请求边界计算最终生效 URL，并据此解析密钥。鉴权解析继续留在 `providers/openai.py` 的 OpenAI-compatible 责任域内；测试捕获最终请求 headers，直接验证网络边界行为。

**Tech Stack:** Python 3.12、pytest、httpx、`urllib.parse.urlparse`、Ruff、PowerShell

**Spec:** `docs/superpowers/specs/2026-07-19-minimax-native-auth-reuse-design.md`

---

## File Structure

- Modify: `tests/model_gateway/test_per_profile_url_and_extra_body.py`
  - 扩展现有捕获客户端以记录 headers。
  - 将当前“只检查环境变量”的弱测试改成最终 Authorization 断言。
  - 覆盖密钥优先级、受约束复用、缺钥、最终 URL、hostname 与 HTTPS 边界。
- Modify: `werewolf_agent/model_gateway/providers/openai.py`
  - 在请求前计算最终生效 URL。
  - 对 MiniMax 原生 endpoint 执行 fail-closed 密钥解析。
  - 保持非 MiniMax endpoint 和 OpenAI-compatible payload 行为不变。

不修改 `providers/base.py`：`PROVIDER_DOTENV_KEYS` 已包含
`MINIMAX_NATIVE_API_KEY`、`MINIMAX_API_KEY`、`ANTHROPIC_API_KEY` 与对应 base URL。

### Task 1: 用请求边界测试复现错误鉴权

**Files:**
- Modify: `tests/model_gateway/test_per_profile_url_and_extra_body.py:222-238`
- Modify: `tests/model_gateway/test_per_profile_url_and_extra_body.py:502-601`
- Test: `tests/model_gateway/test_per_profile_url_and_extra_body.py`

- [ ] **Step 1: 让捕获客户端保存最终 headers**

把 `_CapturingClient` 的职责说明和状态扩展为：

```python
class _CapturingClient:
    """记录最近一次 POST 的 URL、headers 与 payload。"""

    def __init__(self, response_data: dict | None = None) -> None:
        self.last_url: str | None = None
        self.last_headers: dict[str, str] | None = None
        self.last_payload: dict | None = None
        # 保留现有 response_data 初始化

    def post(self, url, *, json, **kwargs):  # noqa: A002
        self.last_url = url
        self.last_headers = dict(kwargs.get("headers") or {})
        self.last_payload = json
        return _FakeResponse(self._response)
```

- [ ] **Step 2: 隔离 provider 环境，避免真实 `.env` 污染测试**

在 `TestOpenAIProviderPerProfile` 中加入 autouse fixture，测试前后清理
`_ENV_OVERRIDES` 并设置 `MINIMAX_NATIVE_API_KEY=test-native-key`。该类已有三个
URL/payload 测试会向 MiniMax 示例 endpoint 发请求；显式测试键保证 fail-closed
不会把与鉴权无关的测试提前中断：

```python
@pytest.fixture(autouse=True)
def _configure_native_minimax_test_key(self, monkeypatch: pytest.MonkeyPatch):
    from werewolf_agent.model_gateway.providers.env import _ENV_OVERRIDES

    _ENV_OVERRIDES.clear()
    monkeypatch.setenv("MINIMAX_NATIVE_API_KEY", "test-native-key")
    yield
    _ENV_OVERRIDES.clear()
```

同时在 `TestNativeMiniMaxApiKeyRouting` 中加入独立 autouse fixture，测试前后清理
`_ENV_OVERRIDES`，并通过 `monkeypatch.delenv` 清理以下进程变量：

```python
@pytest.fixture(autouse=True)
def _isolate_provider_env(self, monkeypatch: pytest.MonkeyPatch):
    from werewolf_agent.model_gateway.providers.env import _ENV_OVERRIDES

    _ENV_OVERRIDES.clear()
    for key in (
        "MINIMAX_NATIVE_API_KEY",
        "MINIMAX_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    yield
    _ENV_OVERRIDES.clear()
```

移除现有 `test_native_endpoint_uses_minimax_native_key` 对
`load_local_dotenv()` 的调用，避免主工作区密钥影响单元测试。

- [ ] **Step 3: 写入密钥优先级和 fail-closed 的失败测试**

保留并强化专用键测试，使所有低优先级键同时存在，并断言：

```python
assert client.last_headers is not None
assert client.last_headers["Authorization"] == "Bearer dedicated-key"
```

新增独立测试：

```python
def test_native_endpoint_uses_minimax_api_key_when_dedicated_missing(...):
    # 同时设置 MINIMAX_API_KEY 与受约束的 ANTHROPIC_API_KEY
    # 断言最终 Authorization 使用 MINIMAX_API_KEY。

def test_native_endpoint_reuses_anthropic_key_for_minimax_base(...):
    # 仅设置 ANTHROPIC_API_KEY 与
    # ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic。
    # 断言最终 Authorization 使用该 MiniMax key。

def test_native_endpoint_without_minimax_key_fails_before_post(...):
    with pytest.raises(ProviderConfigError, match="MINIMAX_NATIVE_API_KEY"):
        provider.generate(...)
    assert client.last_url is None
```

删除旧的 `test_native_endpoint_with_no_native_key_falls_back`，因为“缺钥时回退
Ark key 并预期生产 401”正是本次缺陷。

- [ ] **Step 4: 写入最终 URL 与 endpoint 安全边界的失败测试**

新增：

```python
def test_provider_default_minimax_url_uses_minimax_key_when_config_url_missing(...):
    # provider base_url 是 https://api.minimaxi.com/v1，config.base_url=None。
    # 断言最终 Authorization 使用 MiniMax key。

def test_lookalike_minimax_hostname_keeps_default_key(...):
    # config.base_url=https://api.minimaxi.com.evil.example/v1。
    # 断言使用 default Ark key，不进入 MiniMax 分支。

def test_plain_http_minimax_url_fails_before_post(...):
    # config.base_url=http://api.minimaxi.com/v1。
    # 断言 ProviderConfigError 且 client.last_url is None。

def test_unrelated_anthropic_key_is_not_reused(...):
    # ANTHROPIC_BASE_URL=https://api.anthropic.com。
    # 断言 ProviderConfigError 且不发送请求。
```

现有 Ark endpoint 测试改为通过 `_CapturingClient` 发起请求并断言真实 header，
不再直接调用私有 resolver，证明非 MiniMax 流量保持默认 key。

- [ ] **Step 5: 运行新增测试并确认按预期失败**

Run:

```powershell
$p='C:\Users\10655\.codex\visualizations\2026\07\15\019f6456-40e2-76e2-8e97-8964d1ce2b5d\minimax-auth-red'
New-Item -ItemType Directory -Force -Path $p | Out-Null
$env:TMP=$p; $env:TEMP=$p
$env:LANGCHAIN_TRACING_V2='false'; $env:LANGSMITH_TRACING='false'
python -m pytest -n 0 --basetemp "$p\pytest" -p no:cacheprovider -o addopts='' `
  tests/model_gateway/test_per_profile_url_and_extra_body.py::TestNativeMiniMaxApiKeyRouting -q
```

Expected: FAIL。失败必须来自当前 resolver 仍回退默认 Ark key、忽略 provider 默认
MiniMax URL、使用 hostname 子串和允许 HTTP；不得是 fixture、导入或测试语法错误。

### Task 2: 实现端点感知的 MiniMax 密钥解析

**Files:**
- Modify: `werewolf_agent/model_gateway/providers/openai.py:1-108`
- Test: `tests/model_gateway/test_per_profile_url_and_extra_body.py`

- [ ] **Step 1: 更新模块说明和导入**

把模块 `修改日期` 更新为 `2026-07-19`，同步功能说明为“MiniMax 原生 endpoint
按专用键、供应商键、受约束 Anthropic 键解析，缺钥 fail closed”。新增：

```python
from urllib.parse import urlparse

from werewolf_agent.model_gateway.providers.base import (
    ProviderConfigError,
    _BaseHttpProvider,
)
```

- [ ] **Step 2: 只计算一次最终生效 URL**

在 `OpenAIProvider.generate()` 中只计算一次最终 URL，并把这个字符串同时传给鉴权
解析与 HTTP 请求：

```python
base_url = config.base_url or self._base_url
return _generate_openai_compatible(
    provider=self,
    base_url=base_url,
    api_key=_resolve_api_key_for_base_url(base_url, self._api_key),
    # 其余参数保持原样
)
```

- [ ] **Step 3: 实现精确 hostname、HTTPS 与密钥优先级**

用以下窄接口替换当前子串匹配与默认 key 回退：

```python
_MINIMAX_NATIVE_HOST = "api.minimaxi.com"


def _resolve_api_key_for_base_url(
    base_url: str,
    default_key: str,
) -> str:
    parsed = urlparse(base_url)
    if parsed.hostname != _MINIMAX_NATIVE_HOST:
        return default_key
    if parsed.scheme.lower() != "https":
        raise ProviderConfigError("MiniMax native endpoint requires HTTPS")

    for key in ("MINIMAX_NATIVE_API_KEY", "MINIMAX_API_KEY"):
        value = get_env(key)
        if value:
            return value

    anthropic_base = urlparse(get_env("ANTHROPIC_BASE_URL"))
    anthropic_key = get_env("ANTHROPIC_API_KEY")
    if anthropic_base.hostname == _MINIMAX_NATIVE_HOST and anthropic_key:
        return anthropic_key

    raise ProviderConfigError(
        "MiniMax native endpoint requires MINIMAX_NATIVE_API_KEY, "
        "MINIMAX_API_KEY, or a MiniMax-scoped ANTHROPIC_API_KEY"
    )
```

错误消息不得插入 URL query、header 或任何 key 值。

- [ ] **Step 4: 运行 RED 测试并确认转绿**

重复 Task 1 Step 5 命令。

Expected: PASS，所有新增 Authorization 与 fail-closed 断言成立。

- [ ] **Step 5: 运行相关 provider/config 回归**

Run:

```powershell
$p='C:\Users\10655\.codex\visualizations\2026\07\15\019f6456-40e2-76e2-8e97-8964d1ce2b5d\minimax-auth-green'
New-Item -ItemType Directory -Force -Path $p | Out-Null
$env:TMP=$p; $env:TEMP=$p
$env:LANGCHAIN_TRACING_V2='false'; $env:LANGSMITH_TRACING='false'
python -m pytest -n 0 --basetemp "$p\pytest" -p no:cacheprovider -o addopts='' `
  tests/model_gateway/test_per_profile_url_and_extra_body.py `
  tests/model_gateway/test_providers.py `
  tests/model_gateway/test_minimax_provider_routing.py -q
```

Expected: 至少保持基线 `57 passed`，加上本轮新增测试后全部通过。

- [ ] **Step 6: 提交行为修复**

```powershell
git add -- tests/model_gateway/test_per_profile_url_and_extra_body.py `
  werewolf_agent/model_gateway/providers/openai.py
git commit -m "fix: reuse scoped MiniMax key for native endpoint"
```

### Task 3: 扩大回归与静态验证

**Files:**
- Verify: `tests/model_gateway/`
- Verify: `tests/agents/test_model_router.py`
- Verify: `tests/scripts/test_run_real_game.py`

- [ ] **Step 1: 运行完整模型网关和调用方回归**

```powershell
$p='C:\Users\10655\.codex\visualizations\2026\07\15\019f6456-40e2-76e2-8e97-8964d1ce2b5d\minimax-auth-regression'
New-Item -ItemType Directory -Force -Path $p | Out-Null
$env:TMP=$p; $env:TEMP=$p
$env:LANGCHAIN_TRACING_V2='false'; $env:LANGSMITH_TRACING='false'
python -m pytest -n 0 --basetemp "$p\pytest" -p no:cacheprovider -o addopts='' `
  tests/model_gateway tests/agents/test_model_router.py tests/scripts/test_run_real_game.py -q
```

Expected: completed process、exit code 0、明确的 passed 汇总；到达 100% 但未退出不算通过。

- [ ] **Step 2: 运行修改文件静态检查**

```powershell
$env:RUFF_CACHE_DIR='C:\Users\10655\.codex\visualizations\2026\07\15\019f6456-40e2-76e2-8e97-8964d1ce2b5d\minimax-auth-ruff'
python -m ruff check `
  tests/model_gateway/test_per_profile_url_and_extra_body.py `
  werewolf_agent/model_gateway/providers/openai.py
git diff --check
```

Expected: `All checks passed!`，`git diff --check` 无输出。

- [ ] **Step 3: 若静态检查要求机械修正，修正后重跑并单独提交**

只有实际产生修正时执行：

```powershell
git add -- tests/model_gateway/test_per_profile_url_and_extra_body.py `
  werewolf_agent/model_gateway/providers/openai.py
git commit -m "style: clean MiniMax auth implementation"
```

### Task 4: 最小真实鉴权验证与交付

**Files:**
- Read only: `E:\NLP\agent\wofkill\.env`
- No production file changes expected

- [ ] **Step 1: 在用户已授权范围内执行一次最小真实请求**

从隔离 worktree 运行；显式读取主仓库 `.env`，不复制文件、不打印 key：

```powershell
$env:LANGCHAIN_TRACING_V2='false'; $env:LANGSMITH_TRACING='false'
python -c "from werewolf_agent.model_gateway.providers.env import load_local_dotenv, get_env; load_local_dotenv(r'E:\NLP\agent\wofkill\.env'); from werewolf_agent.model_gateway.providers.openai import OpenAIProvider; from werewolf_agent.model_gateway.usage_records import ModelConfig; p=OpenAIProvider(api_key=get_env('OPENAI_API_KEY')); r=p.generate('Reply with OK.', ModelConfig(provider='openai', model='MiniMax-M2.7', base_url='https://api.minimaxi.com/v1', max_tokens=16, timeout=30)); print('probe_ok=' + str(bool(r.text)) + ' provider=' + r.provider + ' model=' + r.model)"
```

该命令需要外部网络批准。Expected: exit code 0、`probe_ok=True`，日志中不再出现
401。若返回非 401 的额度、模型权限或限流错误，单独报告为供应商账户问题，不把它
伪装成代码通过。

- [ ] **Step 2: 检查分支状态和提交边界**

```powershell
git status --short --branch
git log -4 --oneline --decorate
git diff master...HEAD --check
```

Expected: 工作树干净；分支只包含计划和 MiniMax 鉴权修复相关提交。

- [ ] **Step 3: 使用 `superpowers:verification-before-completion` 复核所有完成声明**

确认 RED 曾按预期失败、GREEN 和扩大回归有最终退出码、真实 probe 没有 401，随后
使用 `superpowers:finishing-a-development-branch` 提供本地合并/PR/保留分支选项。
