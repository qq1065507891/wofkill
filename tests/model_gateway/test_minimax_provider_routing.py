# -*- coding: utf-8 -*-
"""锁定 ``config/models.yaml`` 中 MiniMax 与 Ark 模型的路由契约。

作者: Project contributors
修改日期: 2026-07-23
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from werewolf_agent.model_gateway.router import ModelRouter


CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "models.yaml"
)


@pytest.fixture(scope="module")
def yaml_config() -> dict:
    """Load ``config/models.yaml`` once per module.  Schema is
    versioned + reviewed; if parsing breaks, surface that clearly.
    """
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def model_router() -> ModelRouter:
    """从生产 YAML 构建真实模型路由器。"""
    return ModelRouter.from_yaml(CONFIG_PATH)


def test_minimax_default_default_provider_is_openai(yaml_config: dict) -> None:
    """默认 MiniMax 流量固定走 Ark DeepSeek V4 Pro。"""
    cfg = yaml_config["llm_profiles"]["minimax_default"]["default"]
    assert cfg["provider"] == "openai"
    assert cfg["model_profile"] == "ark_deepseek_v4_pro"


def test_ark_deepseek_default_uses_deepseek_v4_pro(yaml_config: dict) -> None:
    """``ark_deepseek`` 默认路由与主 DeepSeek V4 Pro 配置一致。"""
    cfg = yaml_config["llm_profiles"]["ark_deepseek"]["default"]
    assert cfg == {
        "provider": "openai",
        "model_profile": "ark_deepseek_v4_pro",
    }


@pytest.mark.parametrize("task_type", ["speech", "deception", "night_action"])
def test_minimax_default_tasks_use_deepseek_v4_pro(
    yaml_config: dict,
    task_type: str,
) -> None:
    """MiniMax 默认档的主要玩家任务固定走 DeepSeek V4 Pro。"""
    cfg = yaml_config["llm_profiles"]["minimax_default"]["tasks"][task_type]
    assert cfg == {
        "provider": "openai",
        "model_profile": "ark_deepseek_v4_pro",
    }


def test_minimax_default_fallback_uses_same_provider_different_model_profile(
    yaml_config: dict,
) -> None:
    """回退保持同 provider、不同 model profile 的既有配置。"""
    fb = yaml_config["llm_profiles"]["minimax_default"]["fallback"]
    primary = yaml_config["llm_profiles"]["minimax_default"]["default"]["provider"]
    assert fb == {
        "provider": "openai",
        "model_profile": "ark_deepseek_v4_flash",
    }
    assert fb["provider"] == primary
    assert fb["model_profile"] != "ark_deepseek_v4_pro"


def test_minimax_default_reflection_keeps_native_endpoint(yaml_config: dict) -> None:
    """``reflection`` task intentionally keeps ``minimax_m27_reflection``
    (native ``minimax`` provider / ``MiniMax-M2.7``).  This is a
    cross-version sampling choice: the reflection synthesizer benefits
    from diverse provider signals.
    """
    reflection = yaml_config["llm_profiles"]["minimax_default"]["tasks"]["reflection"]
    assert reflection["provider"] == "minimax"
    assert reflection["model_profile"] == "minimax_m27_reflection"


@pytest.mark.parametrize(
    "task_type",
    [
        "judge_vote_calling",
        "judge_skill_guide",
        "judge_vote_tally",
        "judge_exile",
        "judge_sheriff",
    ],
)
def test_judge_tasks_keep_minimax_m27_route(
    yaml_config: dict,
    model_router: ModelRouter,
    task_type: str,
) -> None:
    """五个 judge 任务继续使用原生 MiniMax M2.7。"""
    route = yaml_config["llm_profiles"]["minimax_default"]["tasks"][task_type]
    assert route == {
        "provider": "minimax",
        "model_profile": "minimax_m27_default",
    }
    resolved, _ = model_router.resolve_config("judge", task_type)
    assert resolved.provider == "minimax"
    assert resolved.model == "MiniMax-M2.7"


def test_reflection_resolves_to_minimax_m27(
    model_router: ModelRouter,
) -> None:
    """reflection 继续精确解析为原生 MiniMax M2.7。"""
    resolved, _ = model_router.resolve_config("p02", "reflection")
    assert resolved.provider == "minimax"
    assert resolved.model == "MiniMax-M2.7"


@pytest.mark.parametrize("player_id", ["p02", "p04", "p09", "p12"])
@pytest.mark.parametrize(
    "task_type",
    ["vote", "speech", "deception", "night_action"],
)
def test_minimax_default_players_resolve_to_deepseek_v4_pro(
    model_router: ModelRouter,
    player_id: str,
    task_type: str,
) -> None:
    """四名默认玩家的关键任务精确解析到 DeepSeek V4 Pro。"""
    resolved, _ = model_router.resolve_config(player_id, task_type)
    assert resolved.provider == "openai"
    assert resolved.model == "DeepSeek-V4-Pro"


def test_player_defaults_resolve_through_new_routing(yaml_config: dict) -> None:
    """Most players are still bound to ``minimax_default``; after
    Part C.1 every speech / night_action call from those players
    goes through Ark OpenAI + M3.  This test pins the player→profile
    assignment so a future refactor that re-binds to ``minimax_27_*``
    is caught.

    NEW (2026-07-15, native-minimax-routing): 5 players (p01/p03/p06/p08/p10)
    were rerouted to native ``minimax_native_m3`` / ``minimax_native_m2_7``
    profiles that hit ``https://api.minimaxi.com/v1``.  ``minimax_default``
    now retains 5 entries (p02/p04/p09/p12/judge); the rest sit on
    ``ark_*`` or the two new native profiles.
    """
    players = yaml_config["players"]
    minimax_default_players = {
        pid for pid, info in players.items()
        if info["llm_profile"] == "minimax_default"
    }
    # 5 native-routed + 3 ark_* + 5 minimax_default = 13 entries.
    assert minimax_default_players == {"p02", "p04", "p09", "p12", "judge"}, (
        f"minimax_default roster drifted: {sorted(minimax_default_players)}"
    )
    native_m3_players = {
        pid for pid, info in players.items()
        if info["llm_profile"] == "minimax_native_m3"
    }
    native_m27_players = {
        pid for pid, info in players.items()
        if info["llm_profile"] == "minimax_native_m2_7"
    }
    assert native_m3_players == {"p01", "p03", "p08"}, (
        f"minimax_native_m3 roster drifted: {sorted(native_m3_players)}"
    )
    assert native_m27_players == {"p06", "p10"}, (
        f"minimax_native_m2_7 roster drifted: {sorted(native_m27_players)}"
    )


def test_native_minimax_profiles_target_api_minimaxi_v1(yaml_config: dict) -> None:
    """``minimax_native_m3`` / ``minimax_native_m2_7`` must point at
    ``https://api.minimaxi.com/v1`` (the OpenAI-compatible native endpoint),
    not the Ark URL.  This is what makes them "native" rather than Ark.
    """
    for key in ("minimax_native_m3", "minimax_native_m2_7"):
        profile = yaml_config["model_profiles"][key]
        assert profile["provider"] == "openai", f"{key} not on openai provider"
        assert profile["base_url"] == "https://api.minimaxi.com/v1", (
            f"{key} base_url drifted: {profile.get('base_url')!r}"
        )
        assert profile["extra_body"] == {"reasoning_split": True}, (
            f"{key} missing reasoning_split extra_body"
        )

    # Cross-check llm_profile wiring: native profiles must reference the
    # native model_profiles, not ark_*.
    for key in ("minimax_native_m3", "minimax_native_m2_7"):
        default = yaml_config["llm_profiles"][key]["default"]
        assert default["provider"] == "openai"
        assert default["model_profile"] == key, (
            f"llm_profile {key} points to {default['model_profile']!r}, "
            f"expected {key!r}"
        )


def test_native_minimax_fallback_uses_anthropic_compatible_minimax(yaml_config: dict) -> None:
    """v1.1.4 cross-provider rule still holds for the new profiles:
    fallback uses ``provider: minimax`` (Anthropic-compatible native) so
    an ``api.minimaxi.com/v1`` outage does not silently flip back to the
    same endpoint via a same-provider fallback.
    """
    for key in ("minimax_native_m3", "minimax_native_m2_7"):
        fb = yaml_config["llm_profiles"][key]["fallback"]
        assert fb["provider"] == "minimax", (
            f"{key} fallback drifted to provider={fb['provider']!r}"
        )
        assert fb["model_profile"] == "minimax_m27_default", (
            f"{key} fallback model_profile drifted: {fb['model_profile']!r}"
        )


def test_primary_deepseek_v4_pro_model_profile_is_exact(yaml_config: dict) -> None:
    """主 DeepSeek V4 Pro 档固定关键采样与超时参数，且 Kimi 档已删除。"""
    profiles = yaml_config["model_profiles"]
    assert "ark_kimi_k26" not in profiles
    profile = profiles["ark_deepseek_v4_pro"]
    assert profile["provider"] == "openai"
    assert profile["model"] == "DeepSeek-V4-Pro"
    assert profile["temperature"] == 0.5
    assert profile["top_p"] == 0.9
    assert profile["timeout"] == 120
    assert profile["reasoning"] == {"level": "high"}
    assert profile["allow_text_tool_fallback"] is True
    assert profile["structured_output"] == {
        "mode": "text_json",
        "fallback_modes": [],
    }


def test_ark_deepseek_v4_pro_secondary_model_profile_exists(yaml_config: dict) -> None:
    """次级 DeepSeek V4 Pro 档保持独立采样参数。"""
    profile = yaml_config["model_profiles"]["ark_deepseek_v4_pro_secondary"]
    assert profile["provider"] == "openai"
    assert profile["model"] == "DeepSeek-V4-Pro"
    assert profile["temperature"] == 0.6
    assert profile["top_p"] == 0.95


def test_ark_dedup_no_longer_wraps_minimax_models(yaml_config: dict) -> None:
    """NEW (2026-07-15, ark-dedup): after the rename, no Ark profile
    should still wrap a ``minimax-*`` model id.  Native-minimax
    routing owns the MiniMax-M3 / MiniMax-M2.7 surface now; Ark must
    not double-cover it.
    """
    ark_profiles = yaml_config["model_profiles"]
    leaky = [
        name for name, cfg in ark_profiles.items()
        if name.startswith("ark_")
        and isinstance(cfg.get("model"), str)
        and cfg["model"].lower().startswith("minimax")
    ]
    assert leaky == [], f"Ark profiles still wrap MiniMax models: {leaky}"
    # And the old profile IDs must be gone.
    assert "ark_minimax_m3" not in ark_profiles
    assert "ark_minimax_m2_7" not in ark_profiles


def test_minimax_m27_profiles_still_defined_for_reflection_fallback(yaml_config: dict) -> None:
    """The native ``minimax`` provider profiles must still exist so
    the ``reflection`` task and any future ``minimax-only`` tests
    keep working.  We don't remove them, just stop routing primary
    traffic through them.
    """
    profiles = yaml_config["model_profiles"]
    for required in (
        "minimax_m27_default",
        "minimax_m27_reflection",
        "minimax_m27_fast",
    ):
        assert required in profiles, f"required profile {required!r} missing"
        assert profiles[required]["provider"] == "minimax"
        assert profiles[required]["model"] == "MiniMax-M2.7"


class _FakeResponse:
    def __init__(self, json_payload):
        self._json = json_payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._json


class _FakeHttpClient:
    def __init__(self, response_json):
        self._response_json = response_json
        self.last_json = None

    def post(self, url, *, json, **_):
        self.last_json = json
        return _FakeResponse(self._response_json)


def test_minimax_provider_payload_sync_with_anthropic_cache_control() -> None:
    """2026-07-21 R2: MiniMax payload.system 与 Anthropic 一致, 共享 cache marker.

    MiniMax.py 完全照抄 anthropic.py 的 system 字段拼装; R2 让两边都
    在 system_prompt 长度足够时切到 list-of-text-blocks + cache_control.
    """
    from werewolf_agent.model_gateway.providers.minimax import MiniMaxProvider
    from werewolf_agent.model_gateway.router import ModelConfig

    long_prompt = "【提示词合同】id=werewolf-player-system;version=test\n" + (
        "稳定规则内容：" * 200
    )
    client = _FakeHttpClient({
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })
    MiniMaxProvider(
        api_key="k", base_url="https://api.minimaxi.com/anthropic",
        http_client=client,
    ).generate(
        prompt="hello",
        config=ModelConfig(provider="minimax", model="MiniMax-M2.7"),
        system_prompt=long_prompt,
    )
    sent_system = client.last_json["system"]
    assert isinstance(sent_system, list) and len(sent_system) == 1
    assert sent_system[0]["cache_control"] == {"type": "ephemeral"}


def test_minimax_provider_short_system_also_uses_cache_control() -> None:
    """短 system_prompt 在 MiniMax 也走 list (与 anthropic 同步).

    Anthropic 在 prefix < 1024 token 时静默忽略 cache_control marker,
    不收费无副作用. R2 选择统一形态, provider 不做长度阈值判断.
    """
    from werewolf_agent.model_gateway.providers.minimax import MiniMaxProvider
    from werewolf_agent.model_gateway.router import ModelConfig

    client = _FakeHttpClient({
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })
    MiniMaxProvider(
        api_key="k", base_url="https://api.minimaxi.com/anthropic",
        http_client=client,
    ).generate(
        prompt="hello",
        config=ModelConfig(provider="minimax", model="MiniMax-M2.7"),
        system_prompt="hi",
    )
    sent_system = client.last_json["system"]
    assert isinstance(sent_system, list) and len(sent_system) == 1
    assert sent_system[0]["text"] == "hi"
    assert sent_system[0]["cache_control"] == {"type": "ephemeral"}
