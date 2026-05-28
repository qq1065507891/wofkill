"""Judge Agent: broadcasts phases, translates RuleEngine results, never adjudicates.

The judge is the human-facing interface for game flow. It calls RuleEngine
deterministically and translates results to natural language broadcasts.
It MUST NOT directly decide game outcomes.
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.agents.schemas import AgentContext, JudgeBroadcast, TaskType
from werewolf_agent.model_gateway.router import ModelRouter
from werewolf_agent.runtime.timeline import phase_label


class JudgeAgent:
    """Non-adjudicating judge agent for broadcast and flow control.

    Responsibilities:
    - Phase announcement
    - Death/announcement translation to natural language
    - Calling RuleEngine for deterministic decisions
    - Summarizing speech for context compression
    - Human-in-the-loop interaction
    """

    def __init__(
        self,
        model_router: ModelRouter,
    ) -> None:
        self.model_router = model_router

    def broadcast_phase(
        self,
        phase: str,
        day_number: int = 0,
        night_number: int = 0,
        public_data: dict[str, Any] | None = None,
    ) -> JudgeBroadcast:
        """Generate a phase transition broadcast."""
        templates: dict[str, str] = {
            "night": f"天黑请闭眼。{phase_label('night', night_number)} 开始。",
            "day": f"天亮了。{phase_label('day', day_number)} 开始。",
            "wolf_discussion": "狼人请睁眼，讨论击杀目标。",
            "witch_turn": "女巫请睁眼。",
            "seer_turn": "预言家请睁眼。",
            "vote": "投票阶段开始。",
            "sheriff_registration": "警长竞选开始，请上警玩家举手。",
            "sheriff_vote": "请警下玩家投票选举警长。",
            "free_discussion": "自由发言阶段开始。",
            "pk_speech": "平票PK发言开始。",
            "victory_good": "好人阵营获胜！",
            "victory_werewolf": "狼人阵营获胜！",
            "finished": "对局结束。",
        }

        message = templates.get(phase, f"进入 {phase} 阶段。")
        if public_data:
            deaths = public_data.get("deaths", [])
            if deaths:
                players_str = "、".join(d.get("player_id", "???") for d in deaths)
                message += f" 昨夜倒牌：{players_str}。"
            exiled = public_data.get("exiled")
            if exiled:
                message += f" {exiled} 被放逐。"
            revealed = public_data.get("revealed_idiot")
            if revealed:
                message += f" {revealed} 翻牌自证白痴身份。"

        return JudgeBroadcast(
            broadcast_type=phase,
            message=message,
            phase=phase,
            day_number=day_number,
            night_number=night_number,
            public_data=public_data or {},
        )

    def broadcast_death_announcement(
        self,
        deaths: list[dict[str, Any]],
        day_number: int,
    ) -> JudgeBroadcast:
        """Translate death records into a public announcement."""
        if not deaths:
            return JudgeBroadcast(
                broadcast_type="death_announcement",
                message=f"{phase_label('day', day_number)}：昨夜是平安夜，无人倒牌。",
                phase="day",
                day_number=day_number,
            )

        dead_names = []
        for d in deaths:
            name = d.get("player_id", "???")
            dead_names.append(name)

        msg = f"{phase_label('day', day_number)}：昨夜倒牌：{'、'.join(dead_names)}。"
        return JudgeBroadcast(
            broadcast_type="death_announcement",
            message=msg,
            phase="day",
            day_number=day_number,
            public_data={"deaths": deaths},
        )

    def summarize_speech(
        self,
        speeches: list[dict[str, str]],
        context: AgentContext | None = None,
    ) -> str:
        """Summarize a round of speeches for context compression.

        Uses LLM if model_router is configured, otherwise extracts key points.
        """
        if not speeches:
            return "本轮无人发言。"

        if context is not None:
            prompt_parts = ["请用中文简要总结以下发言的关键立场和观点：\n"]
            for s in speeches:
                speaker = s.get("speaker", "?")
                text = s.get("text", "")
                prompt_parts.append(f"[{speaker}]: {text[:200]}")
            prompt = "\n".join(prompt_parts)

            result = self.model_router.generate(
                agent_id="judge",
                task_type="speech",
                prompt=prompt,
            )
            if result.text:
                return result.text

        # Fallback: extract first sentence of each speech
        summaries = []
        for s in speeches:
            speaker = s.get("speaker", "?")
            text = s.get("text", "")
            first_sentence = text.split("。")[0][:80]
            summaries.append(f"[{speaker}] {first_sentence}")
        return "\n".join(summaries)

    def broadcast_vote_result(
        self,
        vote_result: dict[str, Any],
    ) -> JudgeBroadcast:
        """Translate vote resolution to broadcast."""
        exiled = vote_result.get("exiled_player_id")
        reason = vote_result.get("reason", "")

        if exiled:
            msg = f"投票结果：{exiled} 被放逐。"
        elif reason == "first_tie_pk":
            msg = "首次平票，进入PK发言。"
        elif reason == "second_tie_no_exile":
            msg = "再次平票，无人出局，进入夜晚。"
        else:
            msg = "投票结束。"

        return JudgeBroadcast(
            broadcast_type="vote_result",
            message=msg,
            phase="vote",
            public_data=vote_result,
        )

    def broadcast_sheriff_result(
        self,
        sheriff_id: str | None,
        badge_state: str,
    ) -> JudgeBroadcast:
        """Translate sheriff election result to broadcast."""
        if sheriff_id and badge_state == "active":
            msg = f"选举结果：{sheriff_id} 当选警长。"
        elif badge_state == "torn":
            msg = "警长撕掉了警徽，本局不再有警长。"
        else:
            msg = "未产生警长。"

        return JudgeBroadcast(
            broadcast_type="sheriff_result",
            message=msg,
            phase="sheriff_election",
            public_data={"sheriff_id": sheriff_id, "badge_state": badge_state},
        )
