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
