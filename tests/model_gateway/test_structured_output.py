from __future__ import annotations

from typing import Any

from werewolf_agent.model_gateway.providers.openai import OpenAIProvider
from werewolf_agent.model_gateway.router import ModelConfig, ModelRouter
from werewolf_agent.model_gateway.structured_output import (
    StructuredFailureStage,
    StructuredOutputMode,
    StructuredOutputPolicy,
    classify_structured_failure,
)


class _Response:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": '{"action_type":"no_action"}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }


class _Client:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def post(self, _url: str, *, json: dict[str, Any], **_kwargs: Any) -> _Response:
        self.payloads.append(json)
        return _Response()


def _tool() -> dict[str, Any]:
    return {
        "name": "submit_player_action",
        "description": "submit",
        "input_schema": {
            "type": "object",
            "properties": {"action_type": {"type": "string"}},
            "required": ["action_type"],
        },
    }


def test_explicit_policy_preserves_protocol_order() -> None:
    policy = StructuredOutputPolicy.from_model_profile(
        provider="openai",
        model_profile={
            "allow_text_tool_fallback": True,
            "structured_output": {
                "mode": "json_schema",
                "fallback_modes": ["json_object", "text_json"],
            },
        },
    )

    assert policy.modes == (
        StructuredOutputMode.JSON_SCHEMA,
        StructuredOutputMode.JSON_OBJECT,
        StructuredOutputMode.TEXT_JSON,
    )
    assert policy.next_mode(StructuredOutputMode.JSON_SCHEMA, "parse_error") == (
        StructuredOutputMode.JSON_OBJECT
    )
    assert policy.next_mode(StructuredOutputMode.JSON_SCHEMA, "illegal_action") == (
        StructuredOutputMode.JSON_SCHEMA
    )


def test_legacy_policy_maps_provider_capability() -> None:
    openai_policy = StructuredOutputPolicy.from_model_profile(
        provider="openai",
        model_profile={"allow_text_tool_fallback": True},
    )
    minimax_policy = StructuredOutputPolicy.from_model_profile(
        provider="minimax",
        model_profile={"allow_text_tool_fallback": True},
    )

    assert openai_policy.primary_mode == StructuredOutputMode.JSON_SCHEMA
    assert minimax_policy.primary_mode == StructuredOutputMode.TEXT_JSON


def test_failure_classification_separates_protocol_schema_and_semantic() -> None:
    assert classify_structured_failure("missing_tool_call") == StructuredFailureStage.PROTOCOL
    assert classify_structured_failure("parse_error") == StructuredFailureStage.PROTOCOL
    assert classify_structured_failure("schema_validation") == StructuredFailureStage.SCHEMA
    assert classify_structured_failure("vote_quality") == StructuredFailureStage.SEMANTIC
    assert classify_structured_failure("model_generation_failed") == StructuredFailureStage.PROVIDER


def test_router_resolves_explicit_structured_output_policy() -> None:
    router = ModelRouter(
        model_profiles={
            "model": {
                "provider": "openai",
                "model": "test",
                "allow_text_tool_fallback": True,
                "structured_output": {
                    "mode": "json_schema",
                    "fallback_modes": ["json_object", "text_json"],
                },
            },
        },
        llm_profiles={
            "profile": {
                "default": {"provider": "openai", "model_profile": "model"},
            },
        },
        player_assignments={"p01": "profile"},
    )

    config, _ = router.resolve_config("p01", "speech")
    assert config.structured_output_mode == "json_schema"
    assert config.structured_output_fallback_modes == ("json_object", "text_json")


def test_openai_provider_renders_each_structured_protocol() -> None:
    client = _Client()
    provider = OpenAIProvider(api_key="key", http_client=client)

    for mode in (
        StructuredOutputMode.NATIVE_TOOL,
        StructuredOutputMode.JSON_SCHEMA,
        StructuredOutputMode.JSON_OBJECT,
        StructuredOutputMode.TEXT_JSON,
    ):
        provider.generate(
            "prompt",
            ModelConfig(
                provider="openai",
                model="test",
                structured_output_mode=mode.value,
            ),
            tools=[_tool()],
            tool_choice={"type": "tool", "name": "submit_player_action"},
        )

    native, schema, obj, text = client.payloads
    assert "tools" in native and "tool_choice" in native
    assert schema["response_format"]["type"] == "json_schema"
    assert "tools" not in schema
    assert obj["response_format"] == {"type": "json_object"}
    assert "tools" not in obj
    assert "response_format" not in text
    assert "tools" not in text


def test_json_schema_preserves_nullable_target() -> None:
    client = _Client()
    provider = OpenAIProvider(api_key="key", http_client=client)
    tool = _tool()
    tool["input_schema"]["properties"]["target_id"] = {
        "type": ["string", "null"],
        "enum": ["p07", None],
    }
    tool["input_schema"]["required"].append("target_id")

    provider.generate(
        "prompt",
        ModelConfig(
            provider="openai",
            model="test",
            structured_output_mode="json_schema",
        ),
        tools=[tool],
    )

    target_schema = client.payloads[0]["response_format"]["json_schema"]["schema"][
        "properties"
    ]["target_id"]
    assert target_schema == {
        "anyOf": [
            {"type": "string", "enum": ["p07"]},
            {"type": "null"},
        ],
    }
