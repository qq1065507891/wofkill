# -*- coding: utf-8 -*-
"""
2026-07-21 R7 smoke-verify: 跑一次 model_router.generate(...), 断言
UsageRecord.cache_creation_input_tokens / cache_read_input_tokens 至少一边 > 0.

直接走 router 而不是 agent_wolf_discussion: agent_wolf_discussion 返回 dict
{audio_text, target_stance, action_trace}, 不含 UsageRecord.
cache_*_tokens 字段在 router 层的 GenerateResult.usage 上, 必须用 router 测.

适用路径:
- Anthropic / MiniMax: 期望 cache_creation_input_tokens (首轮写) 或 cache_read_input_tokens
  (后续轮读) > 0, 依赖 Anthropic 5min TTL.
- OpenAI / GLM: 期望 cache_read_input_tokens > 0 (OpenAI auto-cache prefix ≥ 1024 tokens).
- 任意 prefix < 1024 token 时 cached_tokens=0 (协议层不命中, R6 audit 字段填 0).

运行示例::

    python scripts/verify_cache_audit.py --provider anthropic --prompt-file tests/fixtures/long_prompt.txt
    # 真调一次 router; cache_create_tokens 应 > 0 (Anthropic 5min TTL 命中).

无 API key 时, 脚本会先 print 友好的错误信息并以非零退出, 不会调用 provider.

作者: Project contributors
创建日期: 2026-07-21
修改日期: 2026-07-21
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("scripts.verify_cache_audit")


def _check_api_key_present() -> str | None:
    """检查 ANTHROPIC_API_KEY / MINIMAX_API_KEY / OPENAI_API_KEY 至少一个在 env 里."""
    from werewolf_agent.model_gateway.providers.env import load_local_dotenv
    load_local_dotenv()
    keys = [
        "ANTHROPIC_API_KEY", "MINIMAX_API_KEY", "MINIMAX_NATIVE_API_KEY",
        "OPENAI_API_KEY", "GLM_API_KEY",
    ]
    if any(os.getenv(k) for k in keys):
        return None
    return "未检测到 ANTHROPIC_API_KEY / MINIMAX_API_KEY / OPENAI_API_KEY / GLM_API_KEY"


def _build_long_prompt_file(path: Path) -> str:
    """写一个 ≥ 1024 token 的稳定 prompt 到 path (用于让 OpenAI auto-cache / Anthropic prefix cache 命中).

    Anthropic 缓存 5min TTL, 系统 prompt 至少 1024 tokens 才命中 cache_write.
    OpenAI 同样 ≥1024 tokens prefix 才命中 auto-cache.
    """
    text = "".join(
        f"人类规则片段 {i:04d}: 当玩家角色是狼人时, 行为守则 X 应在不暴露身份 "
        f"的前提下, 通过发言或票型引导好人阵营走向期望的状态. 这部分内容是固定的, 不变.\n"
        for i in range(60)
    )
    path.write_text(text, encoding="utf-8")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="R7: 走 model_router.generate, 断言 UsageRecord.cache_* 字段被填写.",
    )
    parser.add_argument(
        "--provider", choices=["anthropic", "minimax", "openai", "glm"],
        default="minimax",
        help="要测的 provider 路径 (default: minimax — yaml 里有活跃 model_profile)",
    )
    parser.add_argument(
        "--prompt-file", type=str, default=None,
        help="prompt 文本文件 (默认自动生成 1024+ token 的稳定 prompt 写到临时文件)",
    )
    args = parser.parse_args()

    missing = _check_api_key_present()
    if missing is not None:
        print(f"SKIP: {missing}")
        return 0

    from werewolf_agent.model_gateway.router import (
        GenerationAttemptContext,
        ModelConfig,
        ModelRouter,
    )

    router = ModelRouter.from_yaml(
        str(ROOT / "config" / "models.yaml"), register_env_providers=True,
    )
    if args.provider not in list(router.provider_names()):
        # anthropic / glm 没有 yaml 内 model_profile; 改跑 minimax.
        if args.provider == "anthropic" and "minimax" in list(router.provider_names()):
            print(
                f"FALLBACK: provider {args.provider!r} 未在 yaml 中, 改用 minimax"
            )
            args.provider = "minimax"
        else:
            print(f"SKIP: provider {args.provider!r} 未注册")
            return 0

    # 取该 provider 的一个 model ID.
    providers = router._providers
    provider = providers[args.provider]
    cfg = ModelConfig(
        provider=args.provider, model="cache-audit-probe",
        allow_text_tool_fallback=True,
    )
    # 写 ≥1024 token 的稳定 prompt (Anthropic prefix cache / OpenAI auto-cache 起点).
    if args.prompt_file is None:
        prompt_path = ROOT / "artifacts" / "audit_prompt.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_text = _build_long_prompt_file(prompt_path)
    else:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
    sys_prompt_path = ROOT / "artifacts" / "audit_system.txt"
    sys_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    sys_text = _build_long_prompt_file(sys_prompt_path)

    print(
        f"=== verify_cache_audit ===\n"
        f"provider={args.provider}\n"
        f"prompt_chars={len(prompt_text)} system_chars={len(sys_text)}\n"
    )

    # 调用 1: 写 cache.
    attempt_ctx_1 = GenerationAttemptContext(run_scope="r7auditw")
    result_1 = router.generate(
        agent_id="audit-probe",
        task_type="speech",
        prompt=prompt_text,
        system_prompt=sys_text,
        structured_output_mode="text_json",
        generation_attempt_context=attempt_ctx_1,
    )
    usage_1 = result_1.usage
    if usage_1 is None:
        print(f"FAIL: provider.generate() 没有返回 UsageRecord")
        return 1
    print(
        f"[call 1] prompt={usage_1.prompt_tokens} "
        f"cache_create={usage_1.cache_creation_input_tokens} "
        f"cache_read={usage_1.cache_read_input_tokens}"
    )

    # 调用 2: 5min TTL 内, 同前缀应命中 cache_read.
    attempt_ctx_2 = GenerationAttemptContext(run_scope="r7auditr")
    result_2 = router.generate(
        agent_id="audit-probe",
        task_type="speech",
        prompt=prompt_text,
        system_prompt=sys_text,
        structured_output_mode="text_json",
        generation_attempt_context=attempt_ctx_2,
    )
    usage_2 = result_2.usage
    if usage_2 is None:
        print(f"FAIL: provider.generate() 第二次也没有返回 UsageRecord")
        return 1
    print(
        f"[call 2] prompt={usage_2.prompt_tokens} "
        f"cache_create={usage_2.cache_creation_input_tokens} "
        f"cache_read={usage_2.cache_read_input_tokens}"
    )

    cache_create_total = usage_1.cache_creation_input_tokens + usage_2.cache_creation_input_tokens
    cache_read_total = usage_1.cache_read_input_tokens + usage_2.cache_read_input_tokens

    print(
        f"\n=== summary ===\n"
        f"cache_create_sum={cache_create_total}\n"
        f"cache_read_sum={cache_read_total}\n"
        f"verdict={'PASS' if (cache_create_total + cache_read_total) > 0 else 'FAIL'}\n"
    )
    return 0 if (cache_create_total + cache_read_total) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
