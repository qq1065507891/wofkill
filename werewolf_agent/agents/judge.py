# -*- coding: utf-8 -*-
"""
功能描述：**：法官作为面向人类的游戏流程接口，调用 RuleEngine 确定性结果并翻译为自然语言播报。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-08
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import logging
from typing import Any

from werewolf_agent.agents.schemas import (
    JudgeBroadcast,
)
from werewolf_agent.agents.judge_static_broadcasts import (
    build_death_announcement_broadcast,
    build_phase_broadcast,
)
from werewolf_agent.agents import judge_persona
from werewolf_agent.model_gateway.router import ModelRouter
from werewolf_agent.agents.judge_persona import JudgePersonaSnapshot, JudgeProfileRouter
from werewolf_agent.runtime.timeline import phase_label

logger = logging.getLogger(__name__)

_JUDGE_FACT_ONLY_SYSTEM_PROMPT = judge_persona.JUDGE_FACT_ONLY_SYSTEM_PROMPT
_JUDGE_FACT_ONLY_USER_BOUNDARY = judge_persona.JUDGE_FACT_ONLY_USER_BOUNDARY


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
        return judge_persona.resolve_persona(
            self._profile_router,
            self._profile_id,
            task_type,
        )

    def _persona_system_prompt(self, task_type: str = "judge_phase") -> str:
        """Build a persona-aware system prompt snippet for LLM calls."""
        return judge_persona.build_persona_system_prompt(
            self._profile_router,
            self._profile_id,
            task_type,
        )

    def _persona_inject(self, prompt: str, task_type: str = "judge_phase") -> tuple[str, str | None]:
        """Resolve persona prompts while preserving the legacy method name."""
        return judge_persona.inject_persona_prompt(
            prompt,
            profile_router=self._profile_router,
            profile_id=self._profile_id,
            task_type=task_type,
        )

    def broadcast_phase(
        self,
        phase: str,
        day_number: int = 0,
        night_number: int = 0,
        public_data: dict[str, Any] | None = None,
    ) -> JudgeBroadcast:
        """Generate a phase transition broadcast."""
        return build_phase_broadcast(
            phase=phase,
            day_number=day_number,
            night_number=night_number,
            public_data=public_data,
        )

    def broadcast_death_announcement(
        self,
        deaths: list[dict[str, Any]],
        day_number: int,
    ) -> JudgeBroadcast:
        """Translate death records into a public announcement."""
        return build_death_announcement_broadcast(
            deaths=deaths,
            day_number=day_number,
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
            # Phase-1 audit (C-fix5): zero jitter for judge broadcasts.
            # Judge calls are serial by design (one broadcast waits for
            # the previous to complete) and do not contend with the
            # 12-player concurrent generation burst.  Default jitter
            # of 0-0.8s adds latency without any throughput benefit.
            # Phase 2 P2-2: use the JUDGE_* task type so the
            # ``model_gateway`` router can route judge broadcasts to a
            # dedicated model profile (or override timeouts) per
            # the JUDGE_* keys in ``config/models.yaml``.  The task
            # type now matches the ``TaskType`` enum value
            # (``judge_vote_calling`` etc.) so the audit log can
            # distinguish judge calls from player speech calls.
            result = self.model_router.generate(
                agent_id="judge",
                task_type="judge_vote_calling",
                prompt=prompt,
                system_prompt=system_prompt,
                jitter_seconds=(0.0, 0.0),
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
            # Review U11: log the underlying failure with traceback
            # so the audit log captures why the LLM-driven broadcast
            # was abandoned. The fallback path is preserved so the
            # game continues to make forward progress.
            logger.warning("judge.broadcast_vote_calling failed", exc_info=True)
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
            # Phase 2 P2-2: see comment in ``broadcast_vote_calling``
            result = self.model_router.generate(
                agent_id="judge",
                task_type="judge_skill_guide",
                prompt=prompt,
                system_prompt=system_prompt,
                jitter_seconds=(0.0, 0.0),
            )
            if result.text and result.text.strip():
                return JudgeBroadcast(
                    broadcast_type="skill_guide",
                    message=result.text.strip(),
                    phase="night",
                    public_data={"role": role, "player_id": player_id},
                )
        except Exception:
            # Review U11: see comment in broadcast_vote_calling.
            logger.warning("judge.guide_skill_use failed", exc_info=True)
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
            # Phase 2 P2-2: see comment in ``broadcast_vote_calling``
            result = self.model_router.generate(
                agent_id="judge",
                task_type="judge_vote_tally",
                prompt=prompt,
                system_prompt=system_prompt,
                jitter_seconds=(0.0, 0.0),
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
            # Review U11: see comment in broadcast_vote_calling.
            logger.warning("judge.announce_vote_tally failed", exc_info=True)
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
                # P3-4: the pre-fix else-branch prompt was empty of any
                # context (just "投票结束") so the LLM had to hallucinate
                # what the result was.  Now we forward ``reason`` and
                # ``tied`` to the LLM so it can announce the correct
                # outcome (no-exile / generic-completion / custom).
                tied_str = f"平票玩家: {'、'.join(tied)}。" if tied else ""
                reason_str = f"（原因: {reason or '投票已结束'}）" if reason else "（原因: 投票已结束）"
                prompt = (
                    f"你是狼人杀游戏的法官。{label}，投票结束。"
                    f"{reason_str}{tied_str}\n"
                    f"请用简洁的中文宣布结果。只输出宣布台词，不要输出其他内容。"
                )
            prompt, system_prompt = self._persona_inject(prompt, "judge_exile")
            # Phase 2 P2-2: see comment in ``broadcast_vote_calling``
            result = self.model_router.generate(
                agent_id="judge",
                task_type="judge_exile",
                prompt=prompt,
                system_prompt=system_prompt,
                jitter_seconds=(0.0, 0.0),
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
            # Review U11: see comment in broadcast_vote_calling.
            logger.warning("judge.announce_exile_result failed", exc_info=True)
        return JudgeBroadcast(
            broadcast_type="exile_result",
            message=fallback,
            phase="vote",
            day_number=day_number,
            public_data=public_data,
        )

    def broadcast_sheriff_result(
        self,
        sheriff_id: str | None,
        badge_state: str,
    ) -> JudgeBroadcast:
        """Translate sheriff election result to broadcast.

        Phase 3 P3-5: this was the one broadcast that bypassed the
        LLM (purely hardcoded ``msg``).  The other 4 broadcasts
        (vote_calling, skill_guide, vote_tally, exile) all use
        ``_persona_inject + model_router.generate`` so the judge
        persona style (tournament_referee / variety_show_host /
        neutral_arbiter / ancient_mystic) is applied.  Now the
        sheriff result also routes through the LLM, so the
        tournament-style "pompous formal" persona and the
        variety-show "playful suspense" persona both apply to
        the sheriff-election announcement.
        """
        if sheriff_id and badge_state == "active":
            fallback = f"选举结果：{sheriff_id} 当选警长。"
        elif badge_state == "torn":
            fallback = "警长撕掉了警徽，本局不再有警长。"
        else:
            fallback = "未产生警长。"

        if self.model_router is None:
            return JudgeBroadcast(
                broadcast_type="sheriff_result",
                message=fallback,
                phase="sheriff_election",
                public_data={
                    "sheriff_id": sheriff_id or "",
                    "badge_state": badge_state,
                },
            )
        try:
            outcome_str = (
                f"{sheriff_id} 当选警长" if sheriff_id and badge_state == "active"
                else ("警长撕徽" if badge_state == "torn" else "未产生警长")
            )
            prompt = (
                f"你是狼人杀游戏的法官。请宣布警长选举结果：{outcome_str}。\n"
                f"请用简洁的中文宣布结果。只输出宣布台词，不要输出其他内容。"
            )
            prompt, system_prompt = self._persona_inject(prompt, "judge_sheriff")
            # Phase 2 P2-2: see comment in ``broadcast_vote_calling``
            result = self.model_router.generate(
                agent_id="judge",
                task_type="judge_sheriff",
                prompt=prompt,
                system_prompt=system_prompt,
                jitter_seconds=(0.0, 0.0),
            )
            if result.text and result.text.strip():
                return JudgeBroadcast(
                    broadcast_type="sheriff_result",
                    message=result.text.strip(),
                    phase="sheriff_election",
                    public_data={
                        "sheriff_id": sheriff_id or "",
                        "badge_state": badge_state,
                    },
                )
        except Exception:
            # Review U11: see comment in broadcast_vote_calling.
            logger.warning("judge.broadcast_sheriff_result failed", exc_info=True)
        return JudgeBroadcast(
            broadcast_type="sheriff_result",
            message=fallback,
            phase="sheriff_election",
            public_data={
                "sheriff_id": sheriff_id or "",
                "badge_state": badge_state,
            },
        )
