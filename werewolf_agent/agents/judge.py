"""Judge Agent: broadcasts phases, translates RuleEngine results, never adjudicates.

The judge is the human-facing interface for game flow. It calls RuleEngine
deterministically and translates results to natural language broadcasts.
It MUST NOT directly decide game outcomes.
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.agents.schemas import (
    AgentContext,
    JudgeBroadcast,
    JudgeVoteCallingInput,
    JudgeSkillGuideInput,
    JudgeTallyInput,
    JudgeExileInput,
    TaskType,
)
from werewolf_agent.model_gateway.router import ModelRouter
from werewolf_agent.persona_runtime.judge_router import (
    JudgePersonaSnapshot,
    JudgeProfileRouter,
)
from werewolf_agent.runtime.timeline import phase_label


class JudgeAgent:
    """Non-adjudicating judge agent for broadcast and flow control.

    Responsibilities:
    - Phase announcement
    - Death/announcement translation to natural language
    - Calling RuleEngine for deterministic decisions
    - Summarizing speech for context compression
    - Human-in-the-loop interaction

    Layer 3: Accepts optional JudgeProfileRouter for persona-aware broadcasts.
    When a router is configured, each broadcast method resolves a persona
    snapshot and injects style context into LLM prompts.
    """

    def __init__(
        self,
        model_router: ModelRouter | None = None,
        profile_router: JudgeProfileRouter | None = None,
        profile_id: str = "tournament_referee",
    ) -> None:
        self.model_router = model_router
        self._profile_router = profile_router
        self._profile_id = profile_id

    # ------------------------------------------------------------------
    # Persona helpers (Layer 3)
    # ------------------------------------------------------------------

    def _resolve_persona(self, task_type: str = "judge_phase") -> JudgePersonaSnapshot | None:
        """Resolve the current judge persona snapshot, if a router is configured."""
        if self._profile_router is None:
            return None
        return self._profile_router.resolve(
            profile_id=self._profile_id, task_type=task_type,
        )

    def _persona_system_prompt(self, task_type: str = "judge_phase") -> str:
        """Build a persona-aware system prompt snippet for LLM calls."""
        persona = self._resolve_persona(task_type)
        if persona is None:
            return ""
        return persona.system_prompt

    def _persona_inject(self, prompt: str, task_type: str = "judge_phase") -> tuple[str, str | None]:
        """Resolve the persona system prompt and return (user_prompt, system_prompt).

        The persona is returned as a separate system prompt (not concatenated to
        the user prompt) so downstream LLM providers can route it through the
        native system role instead of burying it in user content.
        """
        sys_p = self._persona_system_prompt(task_type) or None
        return prompt, sys_p

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
                public_data={"death_count": 0, "death_ids": ""},
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
            public_data={
                "death_count": len(dead_names),
                "death_ids": ",".join(dead_names),
            },
        )

    def broadcast_vote_calling(
        self,
        voter_id: str,
        voter_name: str,
        candidates: list[str],
        position: int,
        total: int,
        day_number: int = 1,
        sheriff_weight: float = 1.0,
    ) -> JudgeBroadcast:
        """Per-player structured vote calling (唱票).

        Returns a ritualized calling phrase — e.g. '请p03号玩家陈思远投票，第3位/共10位'.
        Falls back to hardcoded template when no LLM is available.
        """
        label = phase_label("day", day_number)
        weight_note = f"（警长{ sheriff_weight }票）" if sheriff_weight > 1.0 else ""
        fallback = f"请{voter_name}玩家投票{weight_note}，第{position}/{total}位"
        if self.model_router is None:
            return JudgeBroadcast(
                broadcast_type="vote_calling",
                message=fallback,
                phase="vote",
                day_number=day_number,
                public_data={"voter_id": voter_id},
            )
        try:
            prompt = (
                f"你是狼人杀游戏的法官。现在是{label}投票阶段。\n"
                f"请用简洁的中文唱票，邀请第{position}/{total}位投票者{voter_name}投票。\n"
                f"{'该玩家是警长，拥有'+str(sheriff_weight)+'票。' if sheriff_weight > 1.0 else ''}"
                f"可投票目标：{', '.join(candidates) if candidates else '任意存活玩家'}。\n"
                f"只输出唱票台词，不要输出其他内容。"
            )
            prompt, system_prompt = self._persona_inject(prompt, "judge_vote_calling")
            result = self.model_router.generate(
                agent_id="judge",
                task_type="speech",
                prompt=prompt,
                system_prompt=system_prompt,
            )
            if result.text and result.text.strip():
                return JudgeBroadcast(
                    broadcast_type="vote_calling",
                    message=result.text.strip(),
                    phase="vote",
                    day_number=day_number,
                    public_data={"voter_id": voter_id},
                )
        except Exception:
            pass
        return JudgeBroadcast(
            broadcast_type="vote_calling",
            message=fallback,
            phase="vote",
            day_number=day_number,
            public_data={"voter_id": voter_id},
        )

    def guide_skill_use(
        self,
        role: str,
        player_id: str,
        player_name: str,
        available_actions: list[str],
        context_hints: dict[str, Any] | None = None,
    ) -> JudgeBroadcast:
        """Per-role skill interaction dialogue.

        Generates context-aware guidance framing the player's available options
        narratively. Falls back to hardcoded template when no LLM is available.
        """
        hints = context_hints or {}
        role_labels: dict[str, str] = {
            "witch": "女巫",
            "hunter": "猎人",
            "seer": "预言家",
            "idiot": "白痴",
            "hybrid": "混血儿",
            "werewolf": "狼人",
            "villager": "平民",
        }
        role_cn = role_labels.get(role, role)
        fallback = f"{role_cn}请睁眼。可用行动：{', '.join(available_actions)}。"
        if self.model_router is None:
            return JudgeBroadcast(
                broadcast_type="skill_guide",
                message=fallback,
                phase="night",
                public_data={"role": role, "player_id": player_id},
            )
        try:
            hints_text = ""
            if hints:
                hints_text = "当前信息：" + "；".join(
                    f"{k}: {v}" for k, v in hints.items()
                ) + "。\n"
            prompt = (
                f"你是狼人杀游戏的法官。{role_cn} {player_name} 睁眼。\n"
                f"{hints_text}"
                f"可用行动：{', '.join(available_actions)}。\n"
                f"请用叙事化的中文引导该玩家做出选择。不要替玩家做决定。\n"
                f"只输出引导台词，不要输出其他内容。"
            )
            prompt, system_prompt = self._persona_inject(prompt, "judge_skill_guide")
            result = self.model_router.generate(
                agent_id="judge",
                task_type="speech",
                prompt=prompt,
                system_prompt=system_prompt,
            )
            if result.text and result.text.strip():
                return JudgeBroadcast(
                    broadcast_type="skill_guide",
                    message=result.text.strip(),
                    phase="night",
                    public_data={"role": role, "player_id": player_id},
                )
        except Exception:
            pass
        return JudgeBroadcast(
            broadcast_type="skill_guide",
            message=fallback,
            phase="night",
            public_data={"role": role, "player_id": player_id},
        )

    def announce_vote_tally(
        self,
        tally: dict[str, float],
        player_names: dict[str, str],
        sheriff_id: str | None = None,
        sheriff_weight: float = 1.5,
        day_number: int = 1,
    ) -> JudgeBroadcast:
        """Announce vote tally with weighted counts and narrative flair."""
        label = phase_label("day", day_number)
        lines = []
        for pid, weight in sorted(tally.items(), key=lambda x: -x[1]):
            name = player_names.get(pid, pid)
            is_sheriff = pid == sheriff_id
            mark = f"（警长{ sheriff_weight }票）" if is_sheriff else ""
            lines.append(f"  {name}: {weight}票{mark}")
        fallback = f"{label} 投票结果：\n" + "\n".join(lines) if lines else f"{label} 投票结束。"
        # Flatten tally into scalar public_data (avoid nested dict per J-8)
        public_data: dict[str, str | int | float | bool] = {
            "tally_count": int(sum(tally.values())),
            "tally_top_id": (
                max(tally.items(), key=lambda x: x[1])[0] if tally else ""
            ),
            "tally_top_votes": (
                max(tally.values()) if tally else 0
            ),
        }
        if self.model_router is None:
            return JudgeBroadcast(
                broadcast_type="vote_tally",
                message=fallback,
                phase="vote",
                day_number=day_number,
                public_data=public_data,
            )
        try:
            tally_text = "；".join(
                f"{player_names.get(pid, pid)} {w}票" + (f"（警长{sheriff_weight}票）" if pid == sheriff_id else "")
                for pid, w in sorted(tally.items(), key=lambda x: -x[1])
            )
            prompt = (
                f"你是狼人杀游戏的法官。{label} 投票结束。\n"
                f"得票情况：{tally_text}。\n"
                f"请用简洁的中文宣布投票结果。只输出宣布台词，不要输出其他内容。"
            )
            prompt, system_prompt = self._persona_inject(prompt, "judge_vote_tally")
            result = self.model_router.generate(
                agent_id="judge",
                task_type="speech",
                prompt=prompt,
                system_prompt=system_prompt,
            )
            if result.text and result.text.strip():
                return JudgeBroadcast(
                    broadcast_type="vote_tally",
                    message=result.text.strip(),
                    phase="vote",
                    day_number=day_number,
                    public_data=public_data,
                )
        except Exception:
            pass
        return JudgeBroadcast(
            broadcast_type="vote_tally",
            message=fallback,
            phase="vote",
            day_number=day_number,
            public_data=public_data,
        )

    def announce_exile_result(
        self,
        exiled_player_id: str | None,
        exiled_player_name: str = "",
        reason: str = "",
        tied_player_ids: list[str] | None = None,
        day_number: int = 1,
    ) -> JudgeBroadcast:
        """Announce exile result with narrative flair."""
        label = phase_label("day", day_number)
        tied = tied_player_ids or []
        public_data: dict[str, str | int | float | bool] = {
            "exiled_player_id": exiled_player_id or "",
            "reason": reason,
            "tied_count": len(tied),
        }
        if exiled_player_id:
            name = exiled_player_name or exiled_player_id
            fallback = f"{label}：{name} 被放逐出局。"
        elif reason == "first_tie_pk":
            fallback = f"首次平票（{'、'.join(tied)}），进入PK发言。"
        elif reason == "second_tie_no_exile":
            fallback = f"再次平票，无人出局，进入夜晚。"
        else:
            fallback = f"{label} 投票结束。"
        if self.model_router is None:
            return JudgeBroadcast(
                broadcast_type="exile_result",
                message=fallback,
                phase="vote",
                day_number=day_number,
                public_data=public_data,
            )
        try:
            if exiled_player_id:
                n = exiled_player_name or exiled_player_id
                prompt = (
                    f"你是狼人杀游戏的法官。{label}，{n}被放逐出局。\n"
                    f"请用简洁的中文宣布放逐结果。只输出宣布台词，不要输出其他内容。"
                )
            elif reason == "first_tie_pk":
                prompt = (
                    f"你是狼人杀游戏的法官。{label}，投票出现平票（{'、'.join(tied)}），进入PK发言。\n"
                    f"请用简洁的中文宣布平票结果。只输出宣布台词，不要输出其他内容。"
                )
            elif reason == "second_tie_no_exile":
                prompt = (
                    f"你是狼人杀游戏的法官。{label}，再次平票，无人出局。\n"
                    f"请用简洁的中文宣布结果。只输出宣布台词，不要输出其他内容。"
                )
            else:
                prompt = f"你是狼人杀游戏的法官。{label}，投票结束。请用简洁的中文宣布。"
            prompt, system_prompt = self._persona_inject(prompt, "judge_exile")
            result = self.model_router.generate(
                agent_id="judge",
                task_type="speech",
                prompt=prompt,
                system_prompt=system_prompt,
            )
            if result.text and result.text.strip():
                return JudgeBroadcast(
                    broadcast_type="exile_result",
                    message=result.text.strip(),
                    phase="vote",
                    day_number=day_number,
                    public_data=public_data,
                )
        except Exception:
            pass
        return JudgeBroadcast(
            broadcast_type="exile_result",
            message=fallback,
            phase="vote",
            day_number=day_number,
            public_data=public_data,
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
            if self.model_router is None:
                # Fallback: extract first sentence of each speech
                summaries = []
                for s in speeches:
                    speaker = s.get("speaker", "?")
                    text = s.get("text", "")
                    first_sentence = text.split("。")[0][:80]
                    summaries.append(f"[{speaker}] {first_sentence}")
                return "\n".join(summaries)
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
            public_data={
                "sheriff_id": sheriff_id or "",
                "badge_state": badge_state,
            },
        )
