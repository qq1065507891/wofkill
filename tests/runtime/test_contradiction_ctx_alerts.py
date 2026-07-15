"""P2 contradiction-prompt-section: ctx_alerts 收集全部 priority 测试。

验证 build_agent_context 的 contradiction_alerts 含 medium（vote_conflict），
且 must_address_alerts 仍只挑 high。
"""

from __future__ import annotations

from werewolf_agent.agents.schemas import TaskType
from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.engine.rule_engine import RuleEngine
from werewolf_agent.runtime.agent_adapter import build_agent_context

RULESET_PATH = "config/rulesets/pre_witch_hunter_idiot_mixed.yaml"


def _make_state() -> tuple[GameState, RuleEngine]:
    """claim_conflict(high): p01+p08 都跳预言家; vote_conflict(medium): p08 声称怀疑 p03 却投 p04。"""
    engine = RuleEngine.from_yaml(RULESET_PATH)
    roles = {
        "p01": "werewolf", "p02": "werewolf", "p03": "werewolf", "p04": "werewolf",
        "p05": "villager", "p06": "villager", "p07": "villager",
        "p08": "seer", "p09": "witch", "p10": "hunter", "p11": "idiot", "p12": "hybrid",
    }
    players = {pid: PlayerState(id=pid, role=role) for pid, role in roles.items()}
    gs = GameState(
        game_id="vc_test",
        players=players,
        phase="day",
        day_number=1,
        night_number=1,
        hybrid_master_id="p05",
        hybrid_master_faction="good",
        events=[
            GameEvent(type="speech", payload={"speaker": "p08", "day_number": 1, "text": "我是预言家 p03是狼人"}),
            GameEvent(type="speech", payload={"speaker": "p01", "day_number": 1, "text": "我是预言家"}),
            GameEvent(type="vote", payload={"voter": "p08", "target": "p04", "day_number": 1}),
        ],
    )
    return gs, engine


def test_ctx_alerts_includes_medium_vote_conflict():
    """ctx_alerts 应含 medium 优先级的 vote_conflict。"""
    gs, engine = _make_state()
    ctx = build_agent_context(engine, gs, "p05", TaskType.SPEECH)
    types = [a.get("alert_type") for a in ctx.contradiction_alerts]
    assert "vote_conflict" in types


def test_ctx_alerts_keeps_high_claim_conflict():
    """ctx_alerts 仍含 high 优先级的 claim_conflict（回归）。"""
    gs, engine = _make_state()
    ctx = build_agent_context(engine, gs, "p05", TaskType.SPEECH)
    types = [a.get("alert_type") for a in ctx.contradiction_alerts]
    assert "claim_conflict" in types


def test_must_address_includes_medium_priority():
    """v1.1.4 fallback-fix Part A.1: priority 门槛从只保留 high 放宽到
    high + medium;只有 low 仍被过滤。medium contradiction 既然已
    surface 给 LLM,它必须能在发言里回应,而不是被卡在 filler 上。
    """
    gs, engine = _make_state()
    ctx = build_agent_context(engine, gs, "p05", TaskType.SPEECH)
    must = ctx.strategy_directive.get("must_address_alerts", [])
    must_types = [e.get("alert_type") for e in must]
    # 高优先级 claim_conflict 仍进 must_address
    assert "claim_conflict" in must_types
    # medium 的 vote_conflict 现在也进(v1.1.4 行为变化)
    assert "vote_conflict" in must_types
    # priority 字段透传,low 仍未过滤后可识别
    for entry in must:
        assert entry.get("priority") in {"high", "medium"}, (
            f"low-priority alert leaked into must_address_alerts: {entry}"
        )
