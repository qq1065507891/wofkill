# -*- coding: utf-8 -*-
"""
玩家 Agent public facade，保留身份、persona、fallback 和解析辅助入口。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-07
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from werewolf_agent.agents.action_validation import (
    ActionValidator,
    DefaultActionValidator,
)
from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    FallbackAction,
    PlayerAction,
    RetryInfo,
    TaskType,
)
from werewolf_agent.agents.parse_dispatch import (
    select_output_mode as _select_output_mode,
)
from werewolf_agent.agents.player_failures import (
    # 兼容旧模块私有 helper 导入路径，实际流程已迁入 player_action_flow。
    categorize_failure_category as _categorize_failure_category,  # noqa: F401
    fallback_reason as _fallback_reason,
)
from werewolf_agent.agents.player_generation import (
    latest_generation_failure_reason as _latest_generation_failure_reason,
)
from werewolf_agent.agents.player_latency import (  # noqa: F401
    latency_from_result as _latency_from_result,
)
from werewolf_agent.agents.player_retry import (
    build_fallback_action as _build_fallback_action,
    check_repeat_error_signature as _check_repeat_error_signature,
    fallback_vote_target_from_context as _fallback_vote_target_from_context,
)
from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
from werewolf_agent.agents.planning import planning_envelope_to_action
from werewolf_agent.agents.metrics_collector import MetricsCollector
from werewolf_agent.agents.output_parser import (
    parse_action as _parse_action_impl,
    extract_decision_data as _extract_decision_impl,
    vote_choice_map as _vote_choice_map_impl,
    vote_candidate_summary as _vote_summary_impl,
    target_candidate_summary as _target_summary_impl,
)
from werewolf_agent.agents.tool_schema import (
    player_action_tool as _tool_impl,
    speech_quality_error as _speech_quality_impl,
    vote_quality_error as _vote_quality_impl,
)
from werewolf_agent.model_gateway.router import ModelRouter
from werewolf_agent.persona_runtime.router import (
    GameContext,
    PersonaRouter,
    sanitize_persona_snapshot,
)

_SHERIFF_VOTE_FORBIDDEN_AUDIT_FIELDS = (
    "seer_stance",
    "vote_basis",
    "standing_with_seer",
    "suspect_reason",
    "not_voting_reason",
    "private_reason",
)

class PlayerAgent:
    """Schema-constrained player agent with retry and fallback.

    Flow:
    1. Build prompt from AgentContext
    2. Call ModelRouter for LLM generation
    3. Parse into PlayerAction via Pydantic schema
    4. Validate against RuleEngine legal sets
    5. Retry with correction hints on failure
    6. Fallback to safe action after max retries
    """

    _CHOICE_TARGET_ACTIONS = DefaultActionValidator._TARGET_REQUIRED_ACTIONS
    _SPEECH_INTENT_TASKS = {
        TaskType.SPEECH,
        TaskType.SHERIFF_SPEECH,
        TaskType.DEFENSE_SPEECH,
        TaskType.PK_SPEECH,
        TaskType.LAST_WORDS,
    }

    def __init__(
        self,
        agent_id: str,
        model_router: ModelRouter,
        validator: ActionValidator | None = None,
        max_retries: int = 3,
        player_name: str | None = None,
        persona_key: str | None = None,
        persona_router: PersonaRouter | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.player_name = player_name or agent_id
        self.persona_key = persona_key
        self.persona_router = persona_router
        self.model_router = model_router
        self.validator = validator or DefaultActionValidator()
        self.max_retries = max_retries
        # Per-player failure profile: aggregates per-attempt outcomes so
        # developers can identify which persona's prompt template needs
        # tuning. Memory-only; not persisted across sessions.
        self.metrics_collector = MetricsCollector()

    def act(self, context: AgentContext) -> tuple[PlayerAction | FallbackAction, RetryInfo]:
        """Generate a constrained player action with retry/fallback."""
        from werewolf_agent.agents.player_action_flow import run_player_action_flow

        return run_player_action_flow(self, context)

    def _attach_persona_snapshot(self, context: AgentContext) -> AgentContext:
        """Resolve the per-turn persona before any prompt is rendered."""
        if context.persona_snapshot:
            sanitized = sanitize_persona_snapshot(
                context.persona_snapshot,
                own_role=context.own_role or "",
                task_type=context.task_type.value,
            )
            attached = (
                context
                if sanitized == context.persona_snapshot
                else context.model_copy(update={"persona_snapshot": sanitized})
            )
            self._record_persona_exposure(attached)
            return attached
        if not self.persona_key or self.persona_router is None:
            return context

        visible = context.visible_world_state or {}
        alive_players = set(visible.get("alive_players") or [])
        wolf_teammates = set(visible.get("wolf_teammates") or [])
        public_fragments = [
            str(item.get("text") or "")
            for item in context.recent_transcript
            if isinstance(item, dict)
        ]
        public_fragments.append(json.dumps(context.strategy_directive, ensure_ascii=False))
        player_id = re.escape(context.agent_id)
        suspicion_pattern = re.compile(
            rf"(?:(?:怀疑|质疑|施压).{{0,8}}{player_id}|"
            rf"{player_id}.{{0,8}}(?:可疑|狼面|有问题|矛盾|需要回应|承受压力))"
        )
        player_is_suspected = any(
            suspicion_pattern.search(fragment)
            for fragment in public_fragments
        )
        teammate_exiled = bool(
            context.own_role == "werewolf"
            and alive_players
            and any(teammate not in alive_players for teammate in wolf_teammates)
        )
        snapshot = self.persona_router.resolve(
            self.agent_id,
            context.task_type.value,
            GameContext(
                phase=context.phase,
                day_number=context.day_number,
                night_number=context.night_number,
                player_is_suspected=player_is_suspected,
                teammate_exiled=teammate_exiled,
                has_badge=visible.get("sheriff_id") == context.agent_id,
                own_role=context.own_role or "",
                alive=not alive_players or context.agent_id in alive_players,
            ),
        )
        data = asdict(snapshot)
        data.pop("agent_id", None)
        data.pop("base_params", None)
        attached = context.model_copy(update={"persona_snapshot": data})
        self._record_persona_exposure(attached)
        return attached

    @staticmethod
    def _record_persona_exposure(context: AgentContext) -> None:
        identity = getattr(context, "decision_identity", None)
        collector = getattr(context, "exposure_collector", None)
        if identity is not None and collector is not None:
            collector.record_persona(identity, context.persona_snapshot)

    def _latest_generation_failure_reason(self) -> str | None:
        return _latest_generation_failure_reason(self.model_router)

    def _check_repeat_error_signature(
        self,
        retry: RetryInfo,
        raw_text: str,
        attempt: int,
        last_signature: tuple[str, str, str] | None,
        *,
        structured_output_mode: str = "",
    ) -> tuple[bool, tuple[str, str, str] | None]:
        return _check_repeat_error_signature(
            retry,
            raw_text,
            attempt,
            last_signature,
            structured_output_mode=structured_output_mode,
        )

    # ── Delegated to output_parser.py ──

    def _parse_correction_hint(
        self,
        context: AgentContext,
        parse_error: str,
    ) -> str | None:
        if not self._is_sheriff_vote_audit_field_error(context, parse_error):
            return None
        forbidden = ", ".join(_SHERIFF_VOTE_FORBIDDEN_AUDIT_FIELDS)
        allowed = "action_type, target_id, speech, reason, confidence"
        return (
            "sheriff_vote is sheriff election voting, not exile voting. "
            f"Remove exile-vote audit fields: {forbidden}. "
            f"Only keep these fields: {allowed}."
        )

    @staticmethod
    def _is_sheriff_vote_audit_field_error(
        context: AgentContext,
        parse_error: str,
    ) -> bool:
        if ActionType.SHERIFF_VOTE not in context.legal_actions:
            return False
        if ActionType.VOTE in context.legal_actions:
            return False
        return any(
            field in parse_error
            for field in _SHERIFF_VOTE_FORBIDDEN_AUDIT_FIELDS
        )

    def _parse_action(self, text: str) -> tuple[PlayerAction | None, str | None]:
        return _parse_action_impl(text)

    def _parse_planning_action(
        self,
        text: str,
        context: AgentContext,
    ) -> tuple[PlayerAction | None, str | None, dict[str, Any] | None]:
        data, parse_error = _extract_decision_impl(text)
        if data is None:
            return None, None, None
        has_decision = "decision_plan" in data
        has_dialogue = "dialogue_plan" in data
        if not has_decision and not has_dialogue:
            return None, None, None
        if not has_decision or not has_dialogue:
            return None, "Planning envelope requires decision_plan and dialogue_plan", None
        try:
            action, audit = planning_envelope_to_action(data, context)
        except Exception as exc:
            return None, f"Planning envelope validation error: {exc}", None
        return action, None, audit

    def _vote_choice_map(self, context: AgentContext) -> dict[str, str]:
        return _vote_choice_map_impl(context.legal_targets)

    def _vote_candidate_summary(self, context: AgentContext, target_id: str) -> str:
        return _vote_summary_impl(context.salience_items, target_id)

    def _target_candidate_summary(self, context: AgentContext, target_id: str) -> str:
        return _target_summary_impl(context.legal_actions, context.salience_items, target_id)

    # ── Delegated to tool_schema.py ──

    def _player_action_tool(self, context: AgentContext) -> dict[str, Any]:
        output_mode = _select_output_mode(
            legal_actions=context.legal_actions,
            legal_targets=context.legal_targets,
            task_type=context.task_type,
            speech_intent_tasks=self._SPEECH_INTENT_TASKS,
        )
        return _tool_impl(
            context.legal_actions,
            context.legal_targets,
            context.task_type,
            output_mode,
        )

    def _speech_quality_error(self, context: AgentContext, action: PlayerAction) -> str | None:
        return _speech_quality_impl(
            context.task_type,
            action,
            context.recent_transcript,
            context.public_summary,
            context.strategy_directive,
            context.day_number,
        )

    def _vote_quality_error(self, context: AgentContext, action: PlayerAction) -> str | None:
        return _vote_quality_impl(
            context.task_type,
            action,
            context.strategy_directive,
            context.salience_items,
            context.recent_transcript,
        )

    # ── Remaining PlayerAgent methods ──

    def _fallback_action(self, context: AgentContext) -> FallbackAction:
        return _build_fallback_action(
            context,
            fallback_reason=_fallback_reason,
            fallback_speech=self._fallback_speech,
        )

    def _fallback_vote_target_from_context(
        self,
        context: AgentContext,
        candidates: list[str],
    ) -> str | None:
        return _fallback_vote_target_from_context(context, candidates)

    def _context_clues(self, context: AgentContext) -> str:
        clues: list[str] = []
        sheriff_id = context.visible_world_state.get("sheriff_id")
        alive_players = context.visible_world_state.get("alive_players", [])
        if sheriff_id and sheriff_id in alive_players:
            clues.append(f"当前警长是{sheriff_id}")
        for item in context.salience_items[:3]:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type") or item.get("event")
            if item_type == "seer_claim":
                speaker = item.get("speaker") or item.get("seer_id")
                target = item.get("target") or item.get("target_id")
                result = item.get("result") or item.get("alignment")
                if speaker and target and result:
                    clues.append(f"{speaker}报{target}为{result}")
            elif item_type in {"player_died", "death"}:
                player = item.get("player_id") or item.get("target_id")
                reason = item.get("reason")
                if player:
                    clues.append(f"{player}死亡" + (f"({reason})" if reason else ""))
            elif item_type == "vote_resolved":
                exiled = item.get("exiled")
                if exiled:
                    clues.append(f"上一轮放逐{exiled}")
        if context.recent_transcript:
            last = context.recent_transcript[-1]
            speaker = last.get("speaker")
            text = str(last.get("text") or "").strip()
            if speaker and text:
                clues.append(f"{speaker}最近发言：{text[:24]}")
        return "；".join(clues[:3])

    def _fallback_speech(self, context: AgentContext) -> str:
        import hashlib
        import logging
        _log = logging.getLogger(__name__)

        # P1-G3223805846-3: seer 在 PK 段 (task_type=PK_SPEECH) 必须有非空
        # 内容；优先用未报过的狼人查杀作为发言载体，否则给占位声明。
        # 通用 PK 模板是上警风格（"我上警"），对预言家 PK 完全错位且会丢
        # 身份标签，导致 fallback 掉链。
        if (getattr(context, "own_role", "") == "seer"
                and context.task_type == TaskType.PK_SPEECH):
            check_history = (context.strategy_directive or {}).get("my_check_history", []) or []
            wolf_checks = [
                c for c in check_history
                if c.get("alignment") == "wolf" and not c.get("reported")
            ]
            if wolf_checks:
                wc = wolf_checks[0]
                return (
                    f"我是预言家，N{wc.get('night', '?')} 验 "
                    f"{wc.get('target', '?')} 是狼人。"
                    f"我是真的预言家，请跟我投票。"
                )
            return "我是预言家，请给我一次发言机会详细说明查杀。"

        # Hash-based target selection: avoids all agents converging on legal_targets[0]
        seed_str = f"{context.agent_id}:{context.day_number}:{context.phase}"
        seed_hash = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
        # For wolves, prefer non-teammate targets to avoid self-damage
        legal = list(context.legal_targets) if context.legal_targets else []
        own_role = context.own_role or ""
        if own_role == "werewolf" and context.strategy_directive:
            wolf_plan = context.strategy_directive.get("wolf_team_plan", {})
            teammates = set()
            for k in ("fake_seer", "pusher", "deep_hook"):
                t = wolf_plan.get(k, "")
                if t and t in legal:
                    teammates.add(t)
            non_teammate = [t for t in legal if t not in teammates]
            if non_teammate:
                legal = non_teammate
        target = legal[seed_hash % len(legal)] if legal else None

        # Template index uses hash, not sum(ord(...)) — better distribution
        tmpl_idx = seed_hash % 7
        prefix = "[FALLBACK]"

        if context.task_type == TaskType.WOLF_DISCUSSION:
            if target:
                templates = [
                    f"{prefix}狼队夜聊我建议优先刀{{target}}，这个位置如果是神职能压缩好人信息。",
                    f"{prefix}狼队视角我倾向先处理{{target}}，今晚统一刀口，明天白天把压力转出去。",
                    f"{prefix}我建议本轮刀{{target}}，后续容易被好人认证，统一行动再分配冲锋倒钩。",
                    f"{prefix}从狼队视角看{{target}}威胁较大，建议集中票型统一处理，避免分刀。",
                    f"{prefix}今晚目标{{target}}，理由是这个位置存活越久越难处理，尽早解决。",
                    f"{prefix}狼队今晚刀{{target}}，明天我们安排一人引领讨论方向，一人补位配合。",
                    f"{prefix}我倾向于刀{{target}}，它在神职概率较高的位置，赌中收益很大。",
                ]
                return templates[tmpl_idx].format(target=target)
            templates = [
                f"{prefix}狼队夜聊先统一刀口，再分配明天的冲锋位和倒钩位，避免发言互相打架。",
                f"{prefix}狼队视角今晚先别分散意见，优先找神职或强势带队位，明天顺着信息推人。",
                f"{prefix}建议先整理每人明天的站位：一人带节奏，一人补逻辑，一人适度倒钩保护团队。",
                f"{prefix}今晚不宜空刀——连续空刀会暴露战术意图，至少制造一个刀口给女巫压力。",
                f"{prefix}狼队需要确定今晚行动，刀口一致才能最大化信息不对称优势。",
                f"{prefix}提醒队友注意发言一致性，不同人的站边不要互相矛盾以免被好人抓住破绽。",
                f"{prefix}狼队视角先定今晚目标，再决定明天谁冲锋谁潜伏，分工明确胜率更高。",
            ]
            return templates[tmpl_idx]

        if context.task_type in (TaskType.SHERIFF_SPEECH, TaskType.PK_SPEECH):
            templates = [
                f"{prefix}我上警是想给出自己的独立判断视角，重点关注前几位发言的逻辑一致性。",
                f"{prefix}我参加警长竞选，希望通过观察和提问帮好人理清局势。",
                f"{prefix}上警是为了确保好人阵营有人能带节奏，我会根据后续发言调整站边。",
                f"{prefix}我是好人视角上警，主要是防止狼人控场，请大家根据发言质量判断。",
                f"{prefix}上警竞选，我有信心带队——我会认真分析每个人的发言和投票逻辑。",
                f"{prefix}参选警长不是为了秀存在感，而是要让好人阵营有一个清晰的发言方向。",
                f"{prefix}我上警是对局势负责，不想看到警徽落入可疑玩家手中。",
            ]
            return templates[tmpl_idx]

        if context.task_type == TaskType.DEFENSE_SPEECH:
            templates = [
                f"{prefix}我确实不是狼人，请大家仔细分析我的发言和投票逻辑。",
                f"{prefix}我没有理由被推，关注我的人应该先看看自己的视角是否正确。",
                f"{prefix}我是好人，我的选择都是基于公开信息，没有任何隐藏动机。",
                f"{prefix}回顾我的发言和投票，没有任何矛盾之处，被推可能是狼人在带节奏。",
                f"{prefix}如果你们仔细看我的逻辑链，会发现我的站边和推理都是连贯且合理的。",
                f"{prefix}被质疑很正常，但我希望大家关注推我的人背后的动机——可能是狼人抗推。",
                f"{prefix}请好人看清局势，我不是狼，真正的问题可能在那些急于归票的人身上。",
            ]
            return templates[tmpl_idx]

        if context.task_type == TaskType.LAST_WORDS:
            if target:
                templates = [
                    f"{prefix}遗言重点关注{{target}}，发言逻辑存在明显矛盾，请大家后续留意。",
                    f"{prefix}走了，提醒大家注意{{target}}的立场和行为不一致，我对此有较大疑虑。",
                    f"{prefix}最后说一句，{{target}}的发言中有些关键点没有解释清楚，值得深挖。",
                    f"{prefix}遗言不多说，但{{target}}的投票路线和发言立场严重不符，建议重点观察。",
                    f"{prefix}我注意到{{target}}在关键轮次的站边突变，这不正常——好人阵营请留意。",
                    f"{prefix}临别前提醒一句：{{target}}可能是突破口，其逻辑链有明显断裂。",
                    f"{prefix}遗言：关注{{target}}，其行为模式与好人视角不符，建议后续深入盘查。",
                ]
                return templates[tmpl_idx].format(target=target)
            templates = [
                f"{prefix}遗言不多说了，请大家仔细分析每个人的站边逻辑和投票记录。",
                f"{prefix}我相信好人阵营能通过票型和发言找出狼人，加油。",
                f"{prefix}最后提醒一下，注意观察谁在关键投票中立场摇摆。",
                f"{prefix}遗言：希望大家冷静分析，不要被情绪化发言带偏，聚焦票型和逻辑链。",
                f"{prefix}走了。好人阵营请重点复盘关键轮的投票分布，那里有答案。",
                f"{prefix}我的身份是好人，希望我的出局能让你们更清晰地看清局势。",
                f"{prefix}遗言简短：信任逻辑不信任直觉，仔细对比每个人的发言与投票是否一致。",
            ]
            return templates[tmpl_idx]

        # SPEECH and other day discussion types
        if target:
            templates = [
                f"{prefix}我目前对{{target}}有较大疑虑，其发言逻辑不够连贯，需要进一步观察。",
                f"{prefix}从现有信息来看，{{target}}的立场和行为有矛盾，我倾向于关注这个方向。",
                f"{prefix}我分析了一下，{{target}}的发言中有些观点缺乏依据，我对此保持警惕。",
                f"{prefix}综合来看{{target}}在关键节点的表现比较可疑，值得进一步深挖其动机。",
                f"{prefix}{{target}}的投票行为和发言内容之间存在落差，这一点不太对劲。",
                f"{prefix}目前关注{{target}}——其逻辑推理链中有几处跳跃，不像自然的好人思维。",
                f"{prefix}{{target}}的站边轨迹值得关注，在关键轮次的变化缺乏充分解释。",
            ]
            return templates[tmpl_idx].format(target=target)
        templates = [
            f"{prefix}我目前还在整理信息，请大家注意分析发言中的逻辑矛盾和票型走向。",
            f"{prefix}暂时没有确定的目标，但我会重点关注后续发言中立场摇摆的人。",
            f"{prefix}根据现有公开信息，我建议大家都仔细梳理一下各人的站边逻辑。",
            f"{prefix}这一轮信息量较大，我需要时间消化——建议大家关注投票链和发言一致性。",
            f"{prefix}本轮我先听大家的分析，重点观察谁的逻辑链最严密、谁的立场有突变。",
            f"{prefix}我倾向于保持开放态度，不急于站边——让子弹飞一会儿，看后续发言质量。",
            f"{prefix}好人阵营需要团结，但也要警惕跟风——独立思考是我给大家的建议。",
        ]
        _log.warning(
            "fallback speech used for agent=%s day=%s phase=%s task=%s",
            context.agent_id, context.day_number, context.phase, context.task_type,
        )
        return templates[tmpl_idx]

    # ── Prompt building delegated to PlayerPromptBuilder (s10 pipeline) ──

    def _build_system_prompt(self, context: AgentContext) -> str:
        """Build system prompt via s10 pipeline: core + rules + role_guide + skills + output_contract."""
        return PlayerPromptBuilder(context, self.player_name).build_system_prompt()

    def _build_prompt(self, context: AgentContext, retry: RetryInfo) -> str:
        """Build user prompt via s10 pipeline: dynamic per-turn context and task instructions."""
        return PlayerPromptBuilder(context, self.player_name).build_user_prompt(retry)
