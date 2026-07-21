# -*- coding: utf-8 -*-
"""
1a-verify: 调用真实 LLM 跑狼队 WOLF_DISCUSSION 任务, 断言 target_stance 必填
结构化字段 (2026-07-20 stance-text-align)。

运行示例::

    python scripts/verify_wolf_discussion.py \
        --wolf-ids p01 p02 p03 p04 --round 1 --round-count 1

    python scripts/verify_wolf_discussion.py --wolf-ids p01

无 API key 时, 脚本会先 print 友好的错误信息并以非零退出, 不会调用 provider。

作者: Project contributors
创建日期: 2026-07-20
修改日期: 2026-07-20
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from werewolf_agent.agents.schemas import (  # noqa: E402
    ActionType,
    AgentContext,
    TaskType,
)
from werewolf_agent.agents.player import PlayerAgent  # noqa: E402
from werewolf_agent.runtime.agent_registry import SimpleAgentRegistry  # noqa: E402
from werewolf_agent.agents.action_schemas import WolfTargetStanceAction  # noqa: E402
from werewolf_agent.core.models import GameState, PlayerState  # noqa: E402
from werewolf_agent.engine.rule_engine import RuleEngine  # noqa: E402
from werewolf_agent.model_gateway.providers import load_local_dotenv  # noqa: E402
from werewolf_agent.model_gateway.router import ModelRouter  # noqa: E402
from werewolf_agent.runtime.agent_wolf_actions import agent_wolf_discussion  # noqa: E402
from werewolf_agent.runtime.context import build_agent_context  # noqa: E402

logger = logging.getLogger("scripts.verify_wolf_discussion")


def _build_minimal_game_state(
    rule_engine: RuleEngine,
    *,
    night_number: int = 1,
    seed: int = 42,
) -> GameState:
    """构造最小 12 人游戏状态 (4 狼 + 8 非狼), 与 V1 标准板子一致。

    seed 默认 42 (脚本历史默认值), 可通过 main 的 --seed 覆盖, 用于验证
    不同 provider 分配下的 4 个狼。
    """
    player_ids = [f"p{i:02d}" for i in range(1, 13)]
    players = rule_engine.assign_roles(player_ids, seed=seed)
    return GameState(
        game_id=f"verify-{uuid.uuid4().hex[:8]}",
        players=players,
        night_number=night_number,
    )


def _build_registry(model_router: ModelRouter) -> SimpleAgentRegistry:
    """SimpleAgentRegistry 把 PlayerAgent 包成 dispatch 接口。"""
    registry = SimpleAgentRegistry()
    for pid in (f"p{i:02d}" for i in range(1, 13)):
        registry.register(pid, PlayerAgent(agent_id=pid, model_router=model_router))
    return registry


def _validate_target_stance(target_stance: Any) -> tuple[bool, str]:
    """验证 LLM 返回的 target_stance 是否符合协议必填契约。"""
    if target_stance is None:
        return False, "target_stance is None (LLM 缺省字段)"
    if not isinstance(target_stance, Mapping):
        return False, f"target_stance 类型错误: {type(target_stance).__name__}"
    stance = target_stance.get("stance")
    priority = target_stance.get("priority")
    target_id = target_stance.get("target_id")
    if stance not in {"propose", "support", "oppose", "abstain"}:
        return False, f"stance 不在合法枚举: {stance!r}"
    if priority not in {"primary", "backup"}:
        return False, f"priority 不在合法枚举: {priority!r}"
    if stance != "abstain" and target_id is None:
        return False, f"{stance} 形态必须给 target_id, 当前为 None"
    if stance == "abstain" and target_id is not None:
        return False, f"abstain 形态 target_id 必须为 null, 当前 {target_id!r}"
    return True, "OK"


def _run_one_wolf(
    *,
    wolf_id: str,
    game_state: GameState,
    engine: RuleEngine,
    registry: SimpleAgentRegistry,
    round_number: int,
) -> tuple[bool, str, dict[str, Any] | None]:
    """单次 WOLF_DISCUSSION 调用, 跑真实 LLM 路径。返回 (pass, msg, result)。

    2026-07-21: agent_wolf_discussion 在 stance 缺失时会重试一次。重试期间
    retry event 写入 exposure_collector；本函数通过同一 collector 同步读取
    retry 次数，让 verify 报告能区分"零次通过 / 一次重试通过 / 两次重试都空"。
    """
    legal_targets = [
        pid for pid, p in game_state.players.items()
        if p.alive and p.role != "werewolf"
    ]
    retry_collector = _build_retry_collector()
    try:
        state = {
            "game_state": game_state,
            "engine": engine,
            "agent_registry": registry,
            "wolf_discussion_round": round_number,
            "wolf_team_plan": None,
        }
        result = agent_wolf_discussion(
            state, engine, registry, wolf_id,
            decision_identity=_DISCRIMINATOR_IDENTITY,
            exposure_collector=retry_collector,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"WOLF_DISCUSSION 异常: {type(exc).__name__}: {exc}", None
    if result is None:
        return False, "agent_wolf_discussion 返回 None (registry 无 agent)", None

    speech = (result.get("speech_text") or "").strip()
    target_stance = result.get("target_stance")
    passed, why = _validate_target_stance(target_stance)
    retry_count = sum(
        1 for event in retry_collector.flush_events()
        if getattr(event, "type", "") == "wolf_target_stance_retry"
    )
    if retry_count > 0:
        why = f"{why} (retried {retry_count}x via jitter fallback)"
    summary = {
        "wolf_id": wolf_id,
        "round": round_number,
        "speech": speech[:160],
        "target_stance": target_stance,
        "verdict": "PASS" if passed else "FAIL",
        "reason": why,
        "retry_count": retry_count,
    }
    return passed, why, summary


def _build_retry_collector():
    from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector

    return ModuleExposureAuditCollector()


# Single shared identity is fine because verify_wolf_discussion runs each
# wolf_id once per invocation; the collector is per-call.
_DISCRIMINATOR_IDENTITY = __import__(
    "werewolf_agent.evaluation.trace_identity", fromlist=["DecisionIdentity"]
).DecisionIdentity(
    game_id="verify-wolf-discussion",
    player_id="verify",
    phase="wolf_discussion_verify",
    day_number=0,
    night_number=1,
    task_type="wolf_discussion",
    action_index=12,
)


def _check_api_key_present() -> str | None:
    load_local_dotenv()
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("GLM_API_KEY"):
        return None
    return "未检测到 ANTHROPIC_API_KEY 或 GLM_API_KEY, 跳过 1a-verify。"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "1a-verify: 跑真实 LLM 跑 WOLF_DISCUSSION, 断言 target_stance "
            "在 2026-07-20 stance-text-align 后是协议必填结构化字段。"
        ),
    )
    parser.add_argument(
        "--wolf-ids",
        nargs="+",
        default=["p01", "p02", "p03", "p04"],
        help="要验证的狼人 ID 列表 (默认 4 狼)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="WOLF_DISCUSSION 轮次 (默认 1)",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=str(ROOT / "config" / "models.yaml"),
        help="ModelRouter 配置路径 (默认 config/models.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Role-assignment seed (默认 42, 可改为其他以改变狼人分配)",
    )
    args = parser.parse_args()

    missing = _check_api_key_present()
    if missing is not None:
        print(f"SKIP: {missing}")
        return 0

    engine = RuleEngine.from_yaml(
        str(ROOT / "config" / "rulesets" / "pre_witch_hunter_idiot_mixed.yaml")
    )
    game_state = _build_minimal_game_state(engine, night_number=1, seed=args.seed)
    model_router = ModelRouter.from_yaml(
        args.model_config, register_env_providers=True
    )
    registry = _build_registry(model_router)

    print(
        f"=== verify_wolf_discussion ===\n"
        f"model_config={args.model_config}\n"
        f"round={args.round}\n"
        f"wolves={args.wolf_ids}\n"
    )

    failures: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for wolf_id in args.wolf_ids:
        if game_state.players.get(wolf_id) is None or game_state.players[wolf_id].role != "werewolf":
            print(f"SKIP {wolf_id}: not a werewolf in this seed")
            continue
        passed, why, summary = _run_one_wolf(
            wolf_id=wolf_id,
            game_state=game_state,
            engine=engine,
            registry=registry,
            round_number=args.round,
        )
        if summary is not None:
            summaries.append(summary)
        if not passed:
            failures.append({"wolf_id": wolf_id, "reason": why})
        print(
            f"[{wolf_id}] verdict={summary['verdict'] if summary else 'N/A'} "
            f"reason={why}\n  speech: {summary['speech'] if summary else ''!r}\n"
            f"  target_stance: {summary['target_stance'] if summary else None}"
        )

    total = len(summaries)
    failed = len(failures)
    print(
        f"\n=== summary ===\n"
        f"wolves_tested={total} failures={failed}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
