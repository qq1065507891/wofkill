"""Tests for Anthropic Messages provider text-fallback robustness.

N3 (post-review-v2): the legacy text-fallback path injected a literal
``{"`` prefix into the conversation and re-attached it to the model
output via ``text = "{" + text if text[0] != "{"``. Both mechanisms
were brittle: any leading whitespace, BOM, or unexpected non-brace
character in the model's response would either get duplicated or
missed, producing invalid JSON for the downstream parser.

The new path returns the model's text verbatim and lets the consumer
parse it with the existing ``repair_json_text`` + ``json.loads`` chain.
"""

from __future__ import annotations

from typing import Any


class _FakeResponse:
    def __init__(self, json_payload: dict[str, Any]) -> None:
        self._json = json_payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._json


class _FakeHttpClient:
    """Captures the most recent POST body and returns a canned response."""

    def __init__(self, response_json: dict[str, Any]) -> None:
        self._response_json = response_json
        self.last_json: dict[str, Any] | None = None

    def post(self, url: str, *, json: dict[str, Any], **_: Any):
        self.last_json = json
        return _FakeResponse(self._response_json)


class TestAnthropicTextFallbackRobustness:
    """N3 (post-review-v2): text-fallback 处理换行/空格不应吃掉首字符。"""

    def test_lstrip_preserves_brace(self) -> None:
        """lstrip 移除前导空白后, 首字符仍是 ``{``。

        这是一个简单的 sanity check, 保证下游 json.loads 不会因为
        前导 ``\\n`` / 空格而把首字符 ``{`` 吃掉.
        """
        text = "\n{\"action_type\":\"speech\"}"
        assert text.lstrip().startswith("{")

    def test_text_fallback_does_not_inject_brace_prefix(self) -> None:
        """修复后: provider 不再在响应文本前注入 ``{"``。

        模拟 model 返回 ``"  \\n  {\\"action_type\\":\\"speech\\"}"`` (前后
        含空白), 修复前的代码会判定 ``text[0] != "{"``, 然后在前面再加
        一个 ``{"``, 得到 ``{"  \\n  {"action_type":...}`` (非法 JSON).
        修复后的代码应原样返回, 由下游 json.loads 解析.
        """
        from werewolf_agent.model_gateway.providers.anthropic import AnthropicProvider
        from werewolf_agent.model_gateway.router import ModelConfig

        model_text = "  \n  {\"action_type\":\"speech\"}"
        fake_response = {
            "content": [{"type": "text", "text": model_text}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        client = _FakeHttpClient(fake_response)
        provider = AnthropicProvider(
            api_key="sk-test",
            base_url="https://api.anthropic.com",
            http_client=client,
        )
        config = ModelConfig(
            provider="anthropic",
            model="claude-test",
            allow_text_tool_fallback=True,
        )
        # NOTE: 故意不传 tool_choice — 真正的 text-fallback 路径要求
        # ``not forcing_tool``, 即没有强制 tool call. 旧代码在这里
        # 会注入 ``{"`` 前缀.
        result = provider.generate(
            prompt="hello",
            config=config,
            system_prompt=None,
            tools=[{"name": "submit_player_action", "input_schema": {}}],
            tool_choice=None,
        )
        # 关键: 返回的 text 必须与模型输出一致, 不应被注入额外 ``{"``.
        assert result.text == model_text, (
            f"text was modified; got: {result.text!r}, expected: {model_text!r}"
        )
        # 文本能被 json.loads 正常解析.
        import json
        parsed = json.loads(result.text.lstrip())
        assert parsed.get("action_type") == "speech"

    def test_text_fallback_no_priming_message(self) -> None:
        """修复后: provider 不再发送 ``{"`` 作为前导 assistant 消息。

        修复前的代码会注入 ``{"role": "assistant", "content": "{"}``,
        试图"启动"模型的 JSON 输出. 这个机制脆且与 Anthropic 的
        tool_use API 不一致 — 模型在没有 tools 的情况下未必会响应
        ``{"`` 提示. 修复后完全移除该 priming 消息.
        """
        from werewolf_agent.model_gateway.providers.anthropic import AnthropicProvider
        from werewolf_agent.model_gateway.router import ModelConfig

        fake_response = {
            "content": [{"type": "text", "text": "{\"action_type\":\"speech\"}"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        client = _FakeHttpClient(fake_response)
        provider = AnthropicProvider(
            api_key="sk-test",
            base_url="https://api.anthropic.com",
            http_client=client,
        )
        config = ModelConfig(
            provider="anthropic",
            model="claude-test",
            allow_text_tool_fallback=True,
        )
        provider.generate(
            prompt="hello",
            config=config,
            system_prompt=None,
            tools=[{"name": "submit_player_action", "input_schema": {}}],
            tool_choice=None,
        )
        assert client.last_json is not None
        messages = client.last_json.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            assert content != "{", (
                f"priming message '{{' was injected into request: {msg!r}"
            )

    def test_thinking_block_is_detected_without_exposing_it_as_public_text(self) -> None:
        from werewolf_agent.model_gateway.providers.anthropic import AnthropicProvider
        from werewolf_agent.model_gateway.router import ModelConfig

        client = _FakeHttpClient({
            "content": [
                {"type": "thinking", "thinking": "private reasoning"},
                {"type": "text", "text": "final answer"},
            ],
            "usage": {
                "input_tokens": 2,
                "output_tokens": 7,
                "output_tokens_details": {"reasoning_tokens": 5},
            },
        })
        result = AnthropicProvider(
            api_key="k", base_url="https://api.example", http_client=client,
        ).generate("hello", ModelConfig(provider="anthropic", model="MiniMax-M2.7"))

        assert result.text == "final answer"
        assert result.reasoning_status == "confirmed"
        assert result.reasoning_tokens == 5

    def test_high_reasoning_request_sends_anthropic_thinking_contract(self) -> None:
        from werewolf_agent.model_gateway.providers.anthropic import AnthropicProvider
        from werewolf_agent.model_gateway.router import ModelConfig

        client = _FakeHttpClient({
            "content": [{"type": "text", "text": "answer"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })
        result = AnthropicProvider(
            api_key="k", base_url="https://api.example", http_client=client,
        ).generate("hello", ModelConfig(
            provider="anthropic", model="claude-test",
            reasoning_level="high", reasoning_requested=True,
            reasoning_capability="high", max_tokens=2048,
        ))
        assert client.last_json["thinking"]["type"] == "enabled"
        assert client.last_json["thinking"]["budget_tokens"] > 0
        assert result.reasoning_status == "requested_unconfirmed"


class TestAnthropicPromptCache:
    """2026-07-21 R2: Anthropic provider 加 cache_control: ephemeral 标记。

    把系统提示从裸 str 升级为 list-of-text-blocks 形式 (Anthropic 2026 协议),
    给首个 block 加 cache_control: {"type": "ephemeral"}, 让跨轮跨玩家的
    system prompt 复用 cache_read_input_tokens。
    """

    LONG_SYSTEM_PROMPT = "【提示词合同】id=werewolf-player-system;version=test\n" + (
        "稳定规则内容：" * 200
    )

    @staticmethod
    def _capture_client(response_json: dict[str, Any]) -> _FakeHttpClient:
        return _FakeHttpClient(response_json)

    def test_long_system_prompt_emits_cache_control_text_block(self) -> None:
        """system_prompt 足够长 (>= 1024 token 起步) 时, payload[\"system\"] 是 list.

        每个 text block 带 cache_control: {\"type\": \"ephemeral\"}. Anthropic
        将为该 prefix 创建 cache, 跨轮跨玩家复用 cache_read_input_tokens.
        """
        from werewolf_agent.model_gateway.providers.anthropic import AnthropicProvider
        from werewolf_agent.model_gateway.router import ModelConfig

        client = self._capture_client({
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1,
                      "cache_creation_input_tokens": 1500, "cache_read_input_tokens": 0},
        })
        result = AnthropicProvider(
            api_key="k", base_url="https://api.example", http_client=client,
        ).generate(
            prompt="hello",
            config=ModelConfig(provider="anthropic", model="claude-test"),
            system_prompt=self.LONG_SYSTEM_PROMPT,
        )
        sent_system = client.last_json["system"]
        assert isinstance(sent_system, list), (
            f"long system_prompt 必须走 list-of-text-blocks, got {type(sent_system).__name__}"
        )
        assert len(sent_system) == 1
        first = sent_system[0]
        assert first.get("type") == "text"
        assert first.get("text") == self.LONG_SYSTEM_PROMPT
        assert first.get("cache_control") == {"type": "ephemeral"}, (
            f"first text block 必须带 cache_control: ephemeral, got {first.get('cache_control')!r}"
        )
        assert result.text == "ok"

    def test_short_system_prompt_also_uses_cache_control_block(self) -> None:
        """短 system_prompt 也会走 list-of-text-blocks + cache_control marker.

        Anthropic 在 prefix < 1024 token 时不会创建 cache, 但发送 cache_control
        标记仍合法 (server 静默忽略). 这一行为保证 provider 始终按 Anthropic
        list 协议走, 不在 routing 层根据 prompt 长度切形态.
        """
        from werewolf_agent.model_gateway.providers.anthropic import AnthropicProvider
        from werewolf_agent.model_gateway.router import ModelConfig

        client = self._capture_client({
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })
        short_prompt = "你好"
        AnthropicProvider(
            api_key="k", base_url="https://api.example", http_client=client,
        ).generate(
            prompt="hello",
            config=ModelConfig(provider="anthropic", model="claude-test"),
            system_prompt=short_prompt,
        )
        sent_system = client.last_json["system"]
        assert isinstance(sent_system, list) and len(sent_system) == 1
        assert sent_system[0]["text"] == short_prompt
        assert sent_system[0]["cache_control"] == {"type": "ephemeral"}

    def test_no_system_prompt_does_not_emit_cache_control(self) -> None:
        """system_prompt 为 None 或空时, payload[\"system\"] 字段不出现."""
        from werewolf_agent.model_gateway.providers.anthropic import AnthropicProvider
        from werewolf_agent.model_gateway.router import ModelConfig

        client = self._capture_client({
            "content": [{"type": "text", "text": "ok"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })
        AnthropicProvider(
            api_key="k", base_url="https://api.example", http_client=client,
        ).generate(
            prompt="hello",
            config=ModelConfig(provider="anthropic", model="claude-test"),
            system_prompt=None,
        )
        assert "system" not in client.last_json

    def test_anthropic_usage_parsed_into_usage_record(self) -> None:
        """Anthropic response.usage.cache_creation_input_tokens /
        cache_read_input_tokens 必须写入 UsageRecord.
        """
        from werewolf_agent.model_gateway.providers.anthropic import AnthropicProvider
        from werewolf_agent.model_gateway.router import ModelConfig

        client = self._capture_client({
            "content": [{"type": "text", "text": "answer"}],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 1500,
                "cache_read_input_tokens": 700,
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        })
        result = AnthropicProvider(
            api_key="k", base_url="https://api.example", http_client=client,
        ).generate(
            prompt="hello",
            config=ModelConfig(provider="anthropic", model="claude-test"),
            system_prompt=self.LONG_SYSTEM_PROMPT,
        )
        assert result.usage is not None
        assert result.usage.cache_creation_input_tokens == 1500
        assert result.usage.cache_read_input_tokens == 700
        assert result.usage.prompt_tokens == 100

    def test_anthropic_usage_handles_missing_cache_fields(self) -> None:
        """老 vendor 或 fallback 路径不返回 cache_* 字段时, default 0."""
        from werewolf_agent.model_gateway.providers.anthropic import AnthropicProvider
        from werewolf_agent.model_gateway.router import ModelConfig

        client = self._capture_client({
            "content": [{"type": "text", "text": "answer"}],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
            },
        })
        result = AnthropicProvider(
            api_key="k", base_url="https://api.example", http_client=client,
        ).generate(
            prompt="hello",
            config=ModelConfig(provider="anthropic", model="claude-test"),
            system_prompt=None,
        )
        assert result.usage is not None
        assert result.usage.cache_creation_input_tokens == 0
        assert result.usage.cache_read_input_tokens == 0

