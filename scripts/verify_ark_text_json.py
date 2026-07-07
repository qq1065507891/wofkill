# -*- coding: utf-8 -*-
"""
验证 Ark text_json 模式能返回可解析 JSON。

作者: Project contributors
修改日期: 2026-07-07

使用示例:
    python scripts/verify_ark_text_json.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from werewolf_agent.model_gateway.providers import load_local_dotenv  # noqa: E402


def _probe(model: str, key: str, url: str) -> int:
    # text_json payload: NO response_format key at all (mirrors openai.py text_json path)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Output ONLY a single JSON object, no prose, no code fences. "
                    "Fields: action_type (string), target_id (string or null), "
                    "reason (string), confidence (number 0-1)."
                ),
            },
            {
                "role": "user",
                "content": "Submit no_action, target null, reason 'verify', confidence 0.5.",
            },
        ],
        "temperature": 0.5,
        "max_tokens": 256,
        "top_p": 0.9,
    }
    r = httpx.post(
        url,
        headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
        json=payload,
        timeout=60.0,
    )
    print(f"=== {model} ===")
    print(f"  STATUS: {r.status_code}")
    if not r.is_success:
        print(f"  BODY: {r.text[:400]}")
        return 1
    data = r.json()
    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    print(f"  CONTENT: {content[:300]}")
    try:
        parsed = json.loads(content.strip())
        print(f"  PARSED OK: {parsed}")
        return 0
    except json.JSONDecodeError as exc:
        print(f"  PARSE FAIL: {exc}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Ark text_json (no response_format).")
    parser.add_argument("--model", default="minimax-m2.7", help="Single model to probe.")
    parser.add_argument("--all", action="store_true", help="Probe all 4 ark models.")
    args = parser.parse_args()

    load_local_dotenv()
    from werewolf_agent.model_gateway.providers import get_env as _get_env
    key = _get_env("OPENAI_API_KEY")
    base = _get_env("OPENAI_BASE_URL").rstrip("/")
    if not key or not base:
        print("ERROR: OPENAI_API_KEY / OPENAI_BASE_URL not set in .env")
        return 2
    url = f"{base}/chat/completions"
    print(f"Endpoint: {url}")
    print(f"Mode: text_json (NO response_format in payload)")
    print()

    models = (
        ["minimax-m3", "deepseek-v4-flash", "deepseek-v4-pro", "minimax-m2.7"]
        if args.all
        else [args.model]
    )
    fails = 0
    for m in models:
        fails += _probe(m, key, url)
        print()
    print(f"{'PASS' if fails == 0 else 'FAIL'}: {len(models) - fails}/{len(models)} ok")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
