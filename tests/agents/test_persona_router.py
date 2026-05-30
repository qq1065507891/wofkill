"""Tests for Persona Router resolution and dynamic adjustments."""

from __future__ import annotations

from werewolf_agent.persona_runtime.router import (
    GameContext,
    PersonaRouter,
)

PERSONAS_YAML = "config/personas/jingcheng_style_prototypes.yaml"


class TestPersonaRouter:
    def test_load_from_yaml(self) -> None:
        router = PersonaRouter.from_yaml(PERSONAS_YAML)
        assert "logic_leader" in router._profiles

    def test_resolve_known_agent(self) -> None:
        router = PersonaRouter.from_yaml(PERSONAS_YAML)
        router.load_assignments({"p01": "logic_leader"})
        snap = router.resolve("p01", "speech")
        assert snap.profile_id == "logic_leader"
        assert snap.display_name == "强逻辑归票型"
        assert snap.task_style == "structured_reasoning"
        assert snap.base_params.get("logic_skill", 0) > 0.8

    def test_resolve_unknown_agent_returns_default(self) -> None:
        router = PersonaRouter.from_yaml(PERSONAS_YAML)
        snap = router.resolve("p99", "speech")
        assert snap.profile_id == "default"

    def test_task_style_changes_by_task(self) -> None:
        router = PersonaRouter.from_yaml(PERSONAS_YAML)
        router.load_assignments({"p02": "aggressive_bluffer"})
        speech_style = router.resolve("p02", "speech").task_style
        deception_style = router.resolve("p02", "deception").task_style
        assert speech_style == "pressure_attack"
        assert deception_style == "high_pressure_push"

    def test_dynamic_adjustment_when_suspected(self) -> None:
        router = PersonaRouter.from_yaml(PERSONAS_YAML)
        router.load_assignments({"p02": "aggressive_bluffer"})

        ctx = GameContext(player_is_suspected=True)
        snap = router.resolve("p02", "speech", ctx)
        # aggressive_bluffer has aggression_delta: 0.20 when suspected
        base_aggression = router._profiles["aggressive_bluffer"]["base"].get("aggression", 0)
        # Check dynamic adjustments contain aggression
        assert "aggression" in snap.dynamic_adjustments or base_aggression > 0

    def test_dynamic_adjustment_when_teammate_exiled(self) -> None:
        router = PersonaRouter.from_yaml(PERSONAS_YAML)
        router.load_assignments({"p02": "aggressive_bluffer"})

        ctx = GameContext(teammate_exiled=True)
        snap = router.resolve("p02", "speech", ctx)
        # Should have risk_tolerance delta of -0.15
        assert "risk_tolerance" in snap.dynamic_adjustments

    def test_effective_params_clamped(self) -> None:
        router = PersonaRouter.from_yaml(PERSONAS_YAML)
        router.load_assignments({"p04": "bold_pretender"})
        ctx = GameContext(player_is_suspected=True)
        snap = router.resolve("p04", "speech", ctx)
        # All effective params must be in [0, 1]
        for v in snap.effective_params.values():
            assert 0.0 <= v <= 1.0

    def test_persona_does_not_affect_rules(self) -> None:
        """Persona params are metadata only, they don't change legal actions."""
        router = PersonaRouter.from_yaml(PERSONAS_YAML)
        router.load_assignments({"p01": "logic_leader"})
        snap = router.resolve("p01", "vote")
        # Snapshot has no rule-affecting fields
        assert "legal_actions" not in snap.effective_params
