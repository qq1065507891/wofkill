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
