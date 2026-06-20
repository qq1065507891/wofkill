"""Probe Ark (Volcengine) OpenAI-compatible tool-call behavior.

This script makes small real API calls and reports whether each model returns
OpenAI-style ``tool_calls`` or only plain text JSON. It never prints API keys.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from werewolf_agent.model_gateway.providers import load_local_dotenv  # noqa: E402


DEFAULT_MODELS = ["minimax-m3", "deepseek-v4-flash", "deepseek-v4-pro", "minimax-m2.7"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe whether Ark (Volcengine) OpenAI-compatible models return tool_calls.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model names to probe. Defaults to Ark model_profiles in config/models.yaml.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL. Defaults to OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--only",
        choices=["auto", "forced", "none"],
        default=None,
        help="Run only one tool_choice mode.",
    )
    args = parser.parse_args()

    load_local_dotenv()
    from werewolf_agent.model_gateway.providers import get_env as _get_env
    api_key = _get_env("OPENAI_API_KEY")
    base_url = (args.base_url or _get_env("OPENAI_BASE_URL")).rstrip("/")
    if not api_key:
        print("ERROR: OPENAI_API_KEY is not set.")
        return 2
    if not base_url:
        print("ERROR: OPENAI_BASE_URL is not set.")
        return 2

    models = args.models or _ark_models_from_config() or DEFAULT_MODELS
    endpoint = _chat_completions_url(base_url)

    print(f"Endpoint: {endpoint}")
    print(f"Models: {', '.join(models)}")
    print("API key: SET (hidden)")
    print()

    modes = [args.only] if args.only else ["auto", "forced", "none"]
    with httpx.Client(timeout=args.timeout) as client:
        for model in models:
            print(f"=== {model} ===")
            for mode in modes:
                result = _probe_model(
                    client=client,
                    endpoint=endpoint,
                    api_key=api_key,
                    model=model,
                    mode=mode,
                )
                _print_result(mode, result)
            print()
    return 0


def _ark_models_from_config() -> list[str]:
    path = ROOT / "config" / "models.yaml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = []
    for profile_id, profile in (data.get("model_profiles") or {}).items():
        if not str(profile_id).startswith("ark_"):
            continue
        if profile.get("provider") == "openai" and profile.get("model"):
            models.append(str(profile["model"]))
    return models


def _chat_completions_url(base_url: str) -> str:
    if base_url.endswith("/v1") or "/v" in base_url.rsplit("/", 1)[-1]:
        return f"{base_url}/chat/completions"
    if base_url != "https://api.openai.com" and "/" in base_url.removeprefix("https://").removeprefix("http://"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def _probe_model(
    *,
    client: httpx.Client,
    endpoint: str,
    api_key: str,
    model: str,
    mode: str,
) -> dict[str, Any]:
    payload = _payload(model, mode)
    try:
        response = client.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            json=payload,
        )
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "transport_error": f"{type(exc).__name__}: {exc}",
        }

    result: dict[str, Any] = {
        "ok": response.is_success,
        "status_code": response.status_code,
    }
    try:
        data = response.json()
    except ValueError:
        result["raw_text"] = response.text[:500]
        return result

    if not response.is_success:
        result["error"] = _extract_error(data)
        return result

    message = (data.get("choices") or [{}])[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    content = message.get("content") or ""
    result.update({
        "tool_call_count": len(tool_calls),
        "tool_call_names": [
            ((call.get("function") or {}).get("name") or call.get("name") or "")
            for call in tool_calls
        ],
        "content_preview": str(content).replace("\n", " ")[:220],
        "content_json": _looks_like_json_object(content),
        "finish_reason": (data.get("choices") or [{}])[0].get("finish_reason"),
        "usage": data.get("usage") or {},
    })
    if tool_calls:
        first = tool_calls[0]
        function = first.get("function") or {}
        result["first_arguments_preview"] = str(function.get("arguments", ""))[:220]
    return result


def _payload(model: str, mode: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are testing function calling. Return exactly one structured action. "
                    "If tools are available, use the tool."
                ),
            },
            {
                "role": "user",
                "content": "Submit action_type=no_action, target_id=null, speech='', reason='probe', confidence=0.5.",
            },
        ],
        "temperature": 0,
        "max_tokens": 256,
    }
    if mode == "none":
        payload["response_format"] = {"type": "json_object"}
        return payload

    payload["tools"] = [_probe_tool()]
    if mode == "auto":
        payload["tool_choice"] = "auto"
    elif mode == "forced":
        payload["tool_choice"] = {
            "type": "function",
            "function": {"name": "submit_player_action"},
        }
    return payload


def _probe_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "submit_player_action",
            "description": "Submit one Werewolf player action.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "action_type": {"type": "string", "enum": ["no_action"]},
                    "target_id": {"type": ["string", "null"]},
                    "speech": {"type": "string"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["action_type", "target_id", "speech", "reason", "confidence"],
            },
        },
    }


def _looks_like_json_object(content: Any) -> bool:
    if not isinstance(content, str):
        return False
    text = content.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return False
    try:
        json.loads(text)
    except json.JSONDecodeError:
        return False
    return True


def _extract_error(data: Any) -> str:
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return json.dumps(error, ensure_ascii=False)
        if error:
            return str(error)
        return json.dumps(data, ensure_ascii=False)[:500]
    return str(data)[:500]


def _print_result(mode: str, result: dict[str, Any]) -> None:
    if not result.get("ok"):
        detail = result.get("transport_error") or result.get("error") or result.get("raw_text") or ""
        print(f"  [{mode}] FAIL status={result.get('status_code', '-')} {detail}")
        return
    usage = result.get("usage") or {}
    print(
        f"  [{mode}] OK tool_calls={result.get('tool_call_count', 0)} "
        f"names={result.get('tool_call_names', [])} "
        f"content_json={result.get('content_json')} "
        f"finish={result.get('finish_reason')} "
        f"tokens={usage.get('total_tokens', '-')}"
    )
    if result.get("first_arguments_preview"):
        print(f"       tool_args: {result['first_arguments_preview']}")
    elif result.get("content_preview"):
        print(f"       content: {result['content_preview']}")


if __name__ == "__main__":
    raise SystemExit(main())
