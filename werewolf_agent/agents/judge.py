# -*- coding: utf-8 -*-
"""
功能描述：**：法官作为面向人类的游戏流程接口，调用 RuleEngine 确定性结果并翻译为自然语言播报。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-08
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.agents.schemas import (
    JudgeBroadcast,
)
from werewolf_agent.agents import judge_persona
from werewolf_agent.agents.judge_dynamic_broadcasts import (
    build_exile_result_broadcast,
    build_sheriff_result_broadcast,
    build_skill_guide_broadcast,
    build_vote_calling_broadcast,
    build_vote_tally_broadcast,
)
from werewolf_agent.agents.judge_static_broadcasts import (
    build_death_announcement_broadcast,
    build_phase_broadcast,
)
from werewolf_agent.model_gateway.router import ModelRouter
from werewolf_agent.agents.judge_persona import JudgePersonaSnapshot, JudgeProfileRouter

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
        return build_vote_calling_broadcast(
            self.model_router,
            self._persona_inject,
            voter_id=voter_id,
            voter_name=voter_name,
            candidates=candidates,
            position=position,
            total=total,
            day_number=day_number,
            sheriff_weight=sheriff_weight,
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
        return build_skill_guide_broadcast(
            self.model_router,
            self._persona_inject,
            role=role,
            player_id=player_id,
            player_name=player_name,
            available_actions=available_actions,
            context_hints=context_hints,
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
        return build_vote_tally_broadcast(
            self.model_router,
            self._persona_inject,
            tally=tally,
            player_names=player_names,
            sheriff_id=sheriff_id,
            sheriff_weight=sheriff_weight,
            day_number=day_number,
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
        return build_exile_result_broadcast(
            self.model_router,
            self._persona_inject,
            exiled_player_id=exiled_player_id,
            exiled_player_name=exiled_player_name,
            reason=reason,
            tied_player_ids=tied_player_ids,
            day_number=day_number,
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
        return build_sheriff_result_broadcast(
            self.model_router,
            self._persona_inject,
            sheriff_id=sheriff_id,
            badge_state=badge_state,
        )
