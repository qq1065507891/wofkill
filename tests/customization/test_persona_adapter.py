"""Tests for adapting user persona packs to PersonaRouter config."""

from __future__ import annotations

from werewolf_agent.customization.persona_adapter import adapt_persona_pack
from werewolf_agent.persona_runtime.router import PersonaRouter


def test_persona_pack_adapter_outputs_router_profiles_and_assignments() -> None:
    pack = {
        "profile_pack_id": "custom_pack",
        "players": [
            {
                "seat": i,
                "name": f"P{i}",
                "archetype": "analyst",
                "speech_style": "calm",
                "risk_tolerance": "medium",
                "deception": "low",
                "cooperation": "high",
                "aggression": "low",
                "memory_focus": "medium",
                "logic_focus": "high",
                "emotionality": "low",
            }
            for i in range(1, 13)
        ],
    }

    adapted = adapt_persona_pack(pack)

    assert "persona_profiles" in adapted
    assert "player_assignments" in adapted
    assert adapted["player_assignments"]["p01"].startswith("custom_pack_seat_01")


def test_adapted_persona_pack_is_usable_by_persona_router() -> None:
    pack = {
        "profile_pack_id": "custom_pack",
        "players": [
            {
                "seat": i,
                "name": f"P{i}",
                "archetype": "leader",
                "speech_style": "structured",
                "risk_tolerance": "medium",
                "deception": "low",
                "cooperation": "high",
                "aggression": "medium",
                "memory_focus": "high",
                "logic_focus": "high",
                "emotionality": "low",
            }
            for i in range(1, 13)
        ],
    }
    adapted = adapt_persona_pack(pack)
    router = PersonaRouter(
        profiles=adapted["persona_profiles"],
        player_assignments=adapted["player_assignments"],
    )

    snapshot = router.resolve("p01", "speech")

    assert snapshot.display_name == "P1"
    assert snapshot.task_style == "structured_speech"
    assert snapshot.effective_params["logic_skill"] > 0.5


# ---------------------------------------------------------------------------
# C2 (post-review-v2): _slug 必须保留中文字符
# ---------------------------------------------------------------------------


def test_slug_preserves_chinese_archetype() -> None:
    """C2: 中文 archetype 名不应退化为 'default'。"""
    from werewolf_agent.customization.persona_adapter import _slug
    slug = _slug("冷静型")
    assert slug != "default", f"Chinese archetype collapsed: {slug!r}"
    # 应保留中文字符
    assert any("一" <= c <= "鿿" for c in slug), f"slug lost CJK: {slug!r}"


def test_slug_preserves_chinese_speech_style() -> None:
    """C2: speech_style 中的中文应保留，task_styles 命名应可读。"""
    from werewolf_agent.customization.persona_adapter import _slug
    slug = _slug("低调简洁")
    assert slug != "default"
    # 至少一个 CJK 字符
    assert any("一" <= c <= "鿿" for c in slug)
    # 不应只剩下划线/连字符
    assert any(c.isalnum() for c in slug)


def test_chinese_archetype_routes_to_readable_profile() -> None:
    """C2: 端到端 — 中文化 archetype 应生成可读 profile_id。"""
    pack = {
        "profile_pack_id": "chinese_pack",
        "players": [
            {
                "seat": i,
                "name": f"P{i}",
                "archetype": "冷静型" if i % 2 == 0 else "激进型",
                "speech_style": "简洁" if i % 2 == 0 else "激昂",
                "risk_tolerance": "medium",
                "deception": "low",
                "cooperation": "high",
                "aggression": "low",
                "memory_focus": "medium",
                "logic_focus": "high",
                "emotionality": "low",
            }
            for i in range(1, 13)
        ],
    }
    adapted = adapt_persona_pack(pack)
    # profile_id 不应为 ..._default_
    for pid in adapted["persona_profiles"]:
        assert "_default_" not in pid, (
            f"C2: profile_id {pid!r} contains 'default' (CJK not preserved)"
        )
    # 至少一个 profile_id 包含 CJK 字符
    has_cjk = any(
        any("一" <= c <= "鿿" for c in pid) for pid in adapted["persona_profiles"]
    )
    assert has_cjk, f"C2: no CJK in profile_ids: {list(adapted['persona_profiles'])}"

