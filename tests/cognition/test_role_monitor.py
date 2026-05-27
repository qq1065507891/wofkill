"""Tests for RoleStateMonitor -- proactive role-specific alerts."""

from dataclasses import replace

import pytest

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.cognition.role_monitor import RoleStateMonitor, RoleAlert


def _make_gs(**kwargs):
    players = kwargs.pop("players", {
        f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager")
        for i in range(1, 13)
    })
    return GameState(game_id="test", players=players, **kwargs)


class TestSeerUnderPressure:
    def test_no_alert_when_seer_not_questioned(self):
        from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset
        engine = RuleEngine(Ruleset(raw={}))
        monitor = RoleStateMonitor(engine.ruleset)
        gs = _make_gs(day_number=2)
        gs = replace(gs, events=gs.events + [
            GameEvent(type="speech", payload={
                "speaker": "p02", "day_number": 2, "text": "p05是好人"
            }),
        ])
        alerts = monitor.assess(gs, "p05", "seer", "day")
        assert len([a for a in alerts if a.alert_type == "SEER_UNDER_PRESSURE"]) == 0

    def test_alerts_when_seer_questioned_by_two_players(self):
        from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset
        engine = RuleEngine(Ruleset(raw={}))
        monitor = RoleStateMonitor(engine.ruleset)
        gs = _make_gs(day_number=2)
        gs = replace(gs, events=gs.events + [
            GameEvent(type="speech", payload={
                "speaker": "p02", "day_number": 2,
                "text": "我怀疑p05有问题，他的查验逻辑不对",
            }),
            GameEvent(type="speech", payload={
                "speaker": "p03", "day_number": 2,
                "text": "p05就是狼，我投票出p05",
            }),
        ])
        alerts = monitor.assess(gs, "p05", "seer", "day")
        seer_alerts = [a for a in alerts if a.alert_type == "SEER_UNDER_PRESSURE"]
        assert len(seer_alerts) == 1
        assert seer_alerts[0].severity == "critical"


class TestWitchPoisonUnusedAtRisk:
    def test_alerts_when_witch_targeted_and_poison_unused(self):
        from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset
        engine = RuleEngine(Ruleset(raw={}))
        monitor = RoleStateMonitor(engine.ruleset)
        players = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager") for i in range(1, 9)}
        players["p02"] = PlayerState(id="p02", role="witch")
        gs = _make_gs(players=players, day_number=3, poison_used=False)
        gs = replace(gs, events=gs.events + [
            GameEvent(type="speech", payload={
                "speaker": "p03", "day_number": 3,
                "text": "p02的站边有问题，我怀疑她",
            }),
            GameEvent(type="speech", payload={
                "speaker": "p04", "day_number": 3,
                "text": "出p02，她是狼",
            }),
        ])
        alerts = monitor.assess(gs, "p02", "witch", "day")
        witch_alerts = [a for a in alerts if a.alert_type == "WITCH_POISON_UNUSED_AT_RISK"]
        assert len(witch_alerts) == 1

    def test_no_alert_when_poison_already_used(self):
        from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset
        engine = RuleEngine(Ruleset(raw={}))
        monitor = RoleStateMonitor(engine.ruleset)
        players = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager") for i in range(1, 9)}
        players["p02"] = PlayerState(id="p02", role="witch")
        gs = _make_gs(players=players, day_number=3, poison_used=True)
        gs = replace(gs, events=gs.events + [
            GameEvent(type="speech", payload={
                "speaker": "p03", "day_number": 3,
                "text": "我怀疑p02",
            }),
            GameEvent(type="speech", payload={
                "speaker": "p04", "day_number": 3,
                "text": "出p02",
            }),
        ])
        alerts = monitor.assess(gs, "p02", "witch", "day")
        witch_alerts = [a for a in alerts if a.alert_type == "WITCH_POISON_UNUSED_AT_RISK"]
        assert len(witch_alerts) == 0


class TestHunterMayDieSoon:
    def test_alerts_when_hunter_mentioned_in_late_game(self):
        from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset
        engine = RuleEngine(Ruleset(raw={}))
        monitor = RoleStateMonitor(engine.ruleset)
        players = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager") for i in range(1, 7)}
        players["p06"] = PlayerState(id="p06", role="villager", alive=False)
        players["p12"] = PlayerState(id="p12", role="hunter")
        gs = _make_gs(players=players, day_number=3)
        gs = replace(gs, events=gs.events + [
            GameEvent(type="speech", payload={
                "speaker": "p03", "day_number": 3,
                "text": "p12可能是猎人，发言很有底气",
            }),
        ])
        alerts = monitor.assess(gs, "p12", "hunter", "day")
        hunter_alerts = [a for a in alerts if a.alert_type == "HUNTER_MAY_DIE_SOON"]
        assert len(hunter_alerts) == 1


class TestHybridMasterDead:
    def test_alerts_when_hybrid_master_is_dead(self):
        from werewolf_agent.engine.rule_engine import RuleEngine, Ruleset
        engine = RuleEngine(Ruleset(raw={}))
        monitor = RoleStateMonitor(engine.ruleset)
        players = {f"p{i:02d}": PlayerState(id=f"p{i:02d}", role="villager") for i in range(1, 9)}
        players["p10"] = PlayerState(id="p10", role="hybrid")
        players["p01"] = PlayerState(id="p01", role="villager", alive=False)
        gs = _make_gs(players=players, day_number=3, hybrid_master_id="p01")
        alerts = monitor.assess(gs, "p10", "hybrid", "day")
        hybrid_alerts = [a for a in alerts if a.alert_type == "HYBRID_MASTER_DEAD"]
        assert len(hybrid_alerts) == 1
