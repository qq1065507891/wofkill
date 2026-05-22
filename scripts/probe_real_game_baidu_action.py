"""Run one real-game-style PlayerAgent call through the configured model router.

This is a minimal live probe for the same path used by real games:
ModelRouter.from_yaml(..., register_env_providers=True) -> PlayerAgent.act().
It calls the configured model for one player/task and prints whether the result
was a real PlayerAction or a fallback. API keys are loaded from .env but never
printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from werewolf_agent.agents.player import PlayerAgent  # noqa: E402
from werewolf_agent.agents.schemas import ActionType, AgentContext, TaskType  # noqa: E402
from werewolf_agent.model_gateway.providers import load_local_dotenv  # noqa: E402
from werewolf_agent.model_gateway.router import ModelRouter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe one live PlayerAgent call through real game routing.",
    )
    parser.add_argument("--agent-id", default="p01")
    parser.add_argument("--player-name", default="陈思远")
    parser.add_argument("--task", default="sheriff_registration")
    parser.add_argument("--role", default="villager")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--strict-tool",
        action="store_true",
        help="Temporarily disable text JSON fallback for this probe.",
    )
    args = parser.parse_args()

    task_type = TaskType(args.task)
    load_local_dotenv()
    router = ModelRouter.from_yaml("config/models.yaml", register_env_providers=True)
    if args.strict_tool:
        _disable_text_tool_fallback(router)
    cfg, fallback = router.resolve_config(args.agent_id, task_type.value)

    print(f"providers={router.provider_names()}")
    print(
        "resolved="
        f"{cfg.provider}/{cfg.model} "
        f"allow_text_tool_fallback={cfg.allow_text_tool_fallback} "
        f"fallback={fallback}"
    )

    agent = PlayerAgent(
        agent_id=args.agent_id,
        model_router=router,
        max_retries=args.max_retries,
        player_name=args.player_name,
    )
    context = _context_for_task(args.agent_id, task_type, args.role)

    action, retry = agent.act(context)
    print(f"action_class={type(action).__name__}")
    print(f"action_type={action.action_type.value}")
    print(f"target_id={action.target_id}")
    print(f"reason={getattr(action, 'reason', '')}")
    print(f"speech={getattr(action, 'speech', '')}")
    print(f"retry={retry.model_dump()}")

    trace = getattr(action, "trace", None)
    if trace:
        trace_data = trace.model_dump(exclude={"raw_text"})
        print(f"trace={trace_data}")
        print(f"raw_text_preview={trace.raw_text[:500].replace(chr(10), ' ')}")
    return 0


def _disable_text_tool_fallback(router: ModelRouter) -> None:
    snapshot = router.config_snapshot()
    for profile in snapshot["model_profiles"].values():
        profile["allow_text_tool_fallback"] = False
    router._model_profiles = snapshot["model_profiles"]


def _context_for_task(agent_id: str, task_type: TaskType, role: str) -> AgentContext:
    alive_players = [f"p{i:02d}" for i in range(1, 13)]
    if task_type == TaskType.SHERIFF_REGISTRATION:
        legal_actions = [ActionType.SHERIFF_REGISTER, ActionType.NO_ACTION]
        legal_targets: list[str] = []
    elif task_type == TaskType.SPEECH:
        legal_actions = [ActionType.SPEECH]
        legal_targets = [pid for pid in alive_players if pid != agent_id]
    elif task_type == TaskType.VOTE:
        legal_actions = [ActionType.VOTE]
        legal_targets = [pid for pid in alive_players if pid != agent_id]
    else:
        legal_actions = [ActionType.NO_ACTION]
        legal_targets = []

    return AgentContext(
        agent_id=agent_id,
        task_type=task_type,
        phase="sheriff_election" if task_type == TaskType.SHERIFF_REGISTRATION else "day",
        day_number=1,
        night_number=1,
        own_role=role,
        legal_actions=legal_actions,
        legal_targets=legal_targets,
        public_summary="N1结束，进入警长竞选报名。",
        visible_world_state={"alive_players": alive_players},
    )


if __name__ == "__main__":
    raise SystemExit(main())
