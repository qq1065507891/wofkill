# -*- coding: utf-8 -*-
"""
构建玩家提示词，将稳定规则和动态上下文分别写入对应消息。

作者: Mike
创建日期: 2025-01-15
修改日期: 2026-07-07

使用示例:
    >>> from werewolf_agent.agents.prompt_builder import PlayerPromptBuilder
    >>> PlayerPromptBuilder(...)
"""

from __future__ import annotations

import re
from typing import Any

from werewolf_agent.agents.directive_priority import (
    HARD_CONSTRAINT_KEYS,
    REFERENCE_KEYS,
    SUGGESTION_KEYS,
)
from werewolf_agent.agents.prompt_formatting import (
    clean_prompt_text,
    compact_json,
    structured_json_summary,
    summarize_json_value,
    truncate_text,
)
from werewolf_agent.agents import prompt_composer
from werewolf_agent.agents.prompt_memory import (
    REFLECTION_CARD_BUDGET,
    PromptMemoryMixin,
    _MAX_LEARNING_CONTEXT_CHARS,
    _MAX_LEARNING_TEXT_CHARS,
    _MAX_RAG_TEXT_CHARS,
)
from werewolf_agent.agents.prompt_persona import PromptPersonaMixin
from werewolf_agent.agents.prompt_output import (
    PromptOutputMixin,
    _ACTION_TYPE_GUARD,
    _OUTPUT_SCHEMA_SKILL_FIELDS,
    _OUTPUT_SCHEMA_SPEECH_FIELDS,
    _OUTPUT_SCHEMA_VOTE_FIELDS,
    _SPEECH_INTENT_TASKS,
    _VOTE_AUDIT_FIELDS,
    _VOTE_REASON_PRIVACY_GUARD,
)
from werewolf_agent.agents.prompt_salience import (
    PromptSalienceMixin,
    _MAX_SALIENCE_ITEMS,
    _SALIENCE_PRIVATE_KEYS,
    _SALIENCE_PUBLIC_FIELDS,
    _slim_salience_item,
)
from werewolf_agent.agents.prompt_sections import (
    PromptSectionMixin,
    _NEVER_DROP_TIER,
    _SectionSpec,
    _USER_PROMPT_BUDGET_CHARS,
)
from werewolf_agent.agents.prompt_strategy import (
    PromptStrategyMixin,
    _MAX_SKILL_TACTICAL_ADVICE_CHARS,
    _MAX_SKILL_TACTICAL_ADVICE_ITEMS,
    _STRATEGY_GROUP_ORDER,
)
from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    OutputMode,
    RetryInfo,
)

# P2-4: role name mapping was previously duplicated as
# ``_ROLE_NAMES`` here AND ``_ROLE_LABEL_CN`` in
# ``runtime.private_memory``.  Two definitions meant any role
# addition had to be edited in two places (and would silently drift
# if one was forgotten).  Now we import the single source of truth
# from private_memory; the duplicate here is deleted.
from werewolf_agent.runtime.private_memory import _ROLE_LABEL_CN as _ROLE_NAMES  # noqa: E402

__all__ = [
    "PlayerPromptBuilder",
    "HARD_CONSTRAINT_KEYS",
    "REFERENCE_KEYS",
    "REFLECTION_CARD_BUDGET",
    "SUGGESTION_KEYS",
    "_MAX_LEARNING_CONTEXT_CHARS",
    "_MAX_LEARNING_TEXT_CHARS",
    "_MAX_RAG_TEXT_CHARS",
    "_MAX_SALIENCE_ITEMS",
    "_MAX_SKILL_TACTICAL_ADVICE_CHARS",
    "_MAX_SKILL_TACTICAL_ADVICE_ITEMS",
    "_NEVER_DROP_TIER",
    "_SALIENCE_PRIVATE_KEYS",
    "_SALIENCE_PUBLIC_FIELDS",
    "_SectionSpec",
    "_STRATEGY_GROUP_ORDER",
    "_USER_PROMPT_BUDGET_CHARS",
    "_ACTION_TYPE_GUARD",
    "_OUTPUT_SCHEMA_SKILL_FIELDS",
    "_OUTPUT_SCHEMA_SPEECH_FIELDS",
    "_OUTPUT_SCHEMA_VOTE_FIELDS",
    "_SPEECH_INTENT_TASKS",
    "_VOTE_AUDIT_FIELDS",
    "_VOTE_REASON_PRIVACY_GUARD",
    "_slim_salience_item",
]

# P0-K1: skill catalog removed (tool path is dead code). Skill analyses
# are pre-injected via skill_analysis_hints — no separate tool catalog.

_MAX_JSON_CONTEXT_CHARS = 1800
_MAX_PUBLIC_SUMMARY_CHARS = 700
_MAX_TRANSCRIPT_ITEMS = 4
_MAX_TRANSCRIPT_TEXT_CHARS = 220
_MAX_PERSONA_LINE_CHARS = 180
def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_current_game_token(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip()
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars]


def _clean_current_game_list_items(
    value: Any,
    *,
    limit: int,
    max_chars: int,
) -> list[str]:
    """Clean a list of current-game text items, preserving player ids.

    Mirror of ``_clean_list_items`` but uses ``_clean_current_game_token``
    (no id substitution) so current-game ``p03`` references survive. Guards
    against non-list input (returns []) and skips empty items.
    """
    if not isinstance(value, list):
        return []
    return [
        _clean_current_game_token(item, max_chars=max_chars)
        for item in value[:limit]
        if str(item or "").strip()
    ]






class PlayerPromptBuilder(
    PromptSectionMixin,
    PromptSalienceMixin,
    PromptMemoryMixin,
    PromptStrategyMixin,
    PromptPersonaMixin,
    PromptOutputMixin,
):
    """Assembles player prompts as a pipeline of independently-built sections.

    Per s10:
      system_prompt = core + rules + role_guide + skills + output_contract
      user_prompt   = boundary + phase + belief + summary + state
                      + events + directive + persona + transcript
                      + retry + task + contract

    Each _build_* method owns exactly one data source.  Stable sections
    (identity, rules, output format) go into the system prompt; per-turn
    dynamic context goes into the user message.
    """

    def __init__(self, context: AgentContext, player_name: str = "") -> None:
        self.context = context
        self.player_name = player_name or context.agent_id

    # ═══════════════════════════════════════════════════════════════
    #  System prompt: stable identity, rules, skill catalog
    # ═══════════════════════════════════════════════════════════════

    def build_system_prompt(self) -> str:
        return prompt_composer.compose_system_prompt(self)

    def _build_core_identity(self) -> str:
        role_cn = _ROLE_NAMES.get(self.context.own_role or "", self.context.own_role or "")
        lines = [
            "你是一场狼人杀游戏的玩家。请用中文发言和思考。",
            f"你的玩家ID: {self.context.agent_id}",
            f"你的名字: {self.player_name}",
        ]
        if self.context.own_role:
            lines.append(f"你的角色: {role_cn}（{self.context.own_role}）")
        return "\n".join(lines)

    def _build_game_rules(self) -> str:
        return (
            "【禁止事项】本局只有以下7种角色：狼人、村民、预言家、女巫、猎人、白痴、混血儿。"
            "绝对禁止提及守卫、恋人、丘比特、白狼王、熊、乌鸦、狐狸、盗贼、吹笛者等任何不存在的角色。"
            "没有守卫，不存在被守护的可能。没有平安夜是由守卫造成的。平安夜只有两种可能：狼人空刀，或女巫使用解药救人。\n"
            "【平安夜与女巫规则】平安夜不等于无人被刀，只代表公开结果无人死亡。"
            "除狼人外，普通玩家不知道狼人是否空刀；除女巫外，普通玩家不知道女巫是否救人。"
            "不能用「平安夜没人死」反驳女巫知道刀口，也不能把「不公开救谁」直接等同于假女巫。"
            "可以质疑跳女巫玩家是否用药、为什么暂不公开银水、以及发言前后是否矛盾。"
            "不要跟风复述已有指控；每次发言必须给出独立证据、明确区分事实和推测。\n"
            "【公开记录引用约束】只有游戏概况、可见状态、关键事件、近期发言中明确出现的信息，才能称为公开记录。"
            "不要编造某玩家曾经说过的话、声称过的身份、投票理由或查验结论；不确定时必须写成推测或质疑。"
        )

    def _build_information_boundaries(self) -> str:
        section_names = "、".join(
            spec.display_name for spec in self._USER_SECTION_SPECS
        )
        labels = "、".join(
            dict.fromkeys(spec.label for spec in self._USER_SECTION_SPECS)
        )
        return (
            f"【信息边界】你会收到以下 user-prompt 段（外层标签包括{labels}；"
            "本轮任务/候选枚举或示例不加外层优先级标签）："
            f"{section_names}、本轮任务/候选枚举或示例。"
            "公开发言时，只有『当前局公开事实』『可见世界状态』『近期发言』『关键事件』"
            "可以称为「场上已知」或「公开记录」。"
            "私信可以用于决策，但不能伪装成公开事实。"
            "跨局学习参考包含知识库提示、跨局反思记忆、历史角色经验、认知校准摘要和跨局错误模式；"
            "它们只是历史经验或自我校准，不代表本局任何玩家真实身份，也不是当前局事实。"
            "本轮任务和最终输出约束只约束输出格式，不是公开记录；"
            "最终输出约束包含纠正提示和最终输出协议。"
            "技能战术建议只是辅助推理，不改变规则、身份或公开记录。"
            "不确定内容必须表达为推测。"
        )

    def _build_reasoning_method(self) -> str:
        # Phase-1 audit: 3-step actionable flow instead of 4 abstract lines.
        # The 4-line version (L213-219 historical) had only 1 actionable
        # step ("投票前比较证据链完整度") and no sequence.  LLM
        # observations show it was skimmed and ignored.  Numbered
        # steps are read more reliably and pair with the
        # ``_build_information_boundaries`` rule that private info
        # cannot be promoted to fact.
        return (
            "【推理方法-3 步】\n"
            "1) 分层：把每条信息标记为「事实 / 推测 / 立场 / 情绪」；"
            "私有信息可用于私有决策，但不能伪装成公开事实；"
            "跨局学习参考和技能建议均不能转成公开事实。\n"
            "2) 盘狼坑：按发言矛盾 > 票型关系 > 站边链条 > 收益动机 > 关键轮次行为 顺序排查；"
            "每条结论必须附公开记录出处或显式标注「推测」。\n"
            "3) 决策：投票前比较证据链完整度与误投成本；行动必须给出当前最优理由，不盲从多数归票。"
        )

    def _build_skill_policy(self) -> str:
        """Skill policy: 边界 with identity rules.

        M5-1: explicitly state that the role's identity rules
        (rendered above in role_guide) outrank skill advice on
        conflict. Without this, LLM may conflate 'skill said
        vote X' with 'role said vote X' — leading to the LLM
        prioritizing skill output over the deterministic role
        rules it was given in the system prompt.
        """
        return (
            "【技能与建议】策略指令中可能包含已计算的技能战术建议，"
            "请基于这些建议与当前局可见事实形成自己的判断，不要机械复述。"
            "【优先级边界】身份规则(role_guide)优先于技能建议，冲突时以身份规则为准。"
            "技能战术建议不是裁判真相；如果与公开事实冲突，以公开事实为准。"
        )

    def _build_role_guide(self) -> str:
        lines: list[str] = []
        role = self.context.own_role or ""
        # P1-S9: villager (3 of 12 players in V1) was missing from this
        # map. Audit identified this as a major gap — villagers were
        # seeing only the generic reasoning / information-boundary
        # sections, with no concrete day-time decision guidance.
        # Rules cover 4 day-time decision dimensions per the audit:
        # public stance, contradiction analysis, N1 antidote support,
        # and evidence-based voting.
        role_rules = {
            "hunter": "猎人规则：被狼人杀死或被放逐时可以开枪带走一人；被女巫毒杀时不能开枪。夜间无法自保。",
            "idiot": "白痴规则：被放逐时亮出身份证明为好人，可以发表遗言，随后出局；出局后不能再发言或投票。夜间无法自保。",
            "witch": (
                "女巫规则：有一瓶解药和一瓶毒药，不能在同一夜同时使用。解药不能自救。"
                "药水决策应结合目标价值、公开证据和保留药水的机会成本。"
            ),
            "seer": (
                "预言家规则：每晚可查验一人身份（好人/狼人），查验混血儿结果为好人。"
                "竞选警长或公开身份时，只能准确报告真实验人，并给出与当前局势相符的警徽流。"
            ),
            "werewolf": (
                "狼人规则：夜间与队友讨论击杀目标，可按合法行动选择击杀、自刀或空刀。"
                "白天可在规则允许时自爆；自爆后立即出局、无遗言，并中断当前白天。"
                "可以悍跳预言家上警对抗真预言家。"
            ),
            "hybrid": (
                "混血儿规则：N1 / 首夜选择一名主人，跟随主人阵营获胜。"
                "主人死亡后阵营不再改变，且不能再选新主人；"
                "如果主人是狼，狼队屠边仅需消灭3村民胜；"
                "如果主人是好人，狼队需消灭3村民+混血儿才算屠边。"
            ),
            "villager": (
                "村民规则：身份公开时表明好人立场；"
                "分析发言矛盾/票型；"
                "讨论解药选择时结合目标价值、公开证据与药水机会成本；"
                "归票基于证据链,不跟风。"
            ),
        }
        # P2-9: VOTE_BASIS_GUIDANCE was originally appended here for
        # non-seer roles. M2-2: the role_guide is part of the stable
        # system prompt and doesn't know task_type, so a wolf NIGHT
        # action was seeing "投票时 vote_basis 选用 speech_logic" —
        # irrelevant. Moved to per-turn strategy_directive injection
        # in agent_adapter.py for VOTE/SPEECH task types only. The
        # seer exemption also lives there (it still applies — seer
        # legitimately uses seer_check for their own checks).
        if role in role_rules:
            lines.append(role_rules[role])
        return "\n".join(lines) if lines else ""

    def _build_output_contract(self) -> str:
        """Stable protocol invariants; fields come from the turn contract."""
        return (
            "【结构化输出】当前回合 user prompt 中的 ActionContract 与"
            "「最终输出协议」是字段、枚举和必填项的唯一依据。"
            "若最终输出协议要求工具且工具可用，则使用工具提交；否则只输出一个"
            "符合当前 ActionContract 的 JSON 对象。不要解释、不要 Markdown、"
            "不要添加合同之外的字段。公开发言正文必须使用中文。"
        )

    # ═══════════════════════════════════════════════════════════════
    #  User prompt: per-turn dynamic context (system reminder)
    # ═══════════════════════════════════════════════════════════════

    def build_user_prompt(self, retry: RetryInfo) -> str:
        return prompt_composer.compose_user_prompt(self, retry)


    def _build_phase_context(self) -> str:
        ctx = self.context
        lines = [f"当前阶段: {ctx.phase}"]
        if ctx.legal_actions:
            lines.append(f"可用操作: {[a.value for a in ctx.legal_actions]}")
        if ctx.legal_targets:
            lines.append(f"可选目标: {ctx.legal_targets}")
        # Mandatory vote hints
        output_mode = self._select_output_mode()
        is_exile_vote_context = self._is_exile_vote_context()
        if ctx.legal_actions and is_exile_vote_context:
            if ActionType.NO_ACTION not in ctx.legal_actions:
                lines.append("重要：本轮投票必须选择一名玩家放逐，不能弃票！")
            if ctx.legal_actions == [ActionType.VOTE] and ctx.legal_targets:
                lines.append("你必须投出选票，从可选目标中选择一人。")
            json_label = (
                "choice 决策JSON"
                if output_mode == OutputMode.TARGET_CHOICE
                else "JSON"
            )
            lines.append(
                f"投票时必须先在心里完成判断，并在{json_label}中额外给出这些私有字段："
                "seer_stance（枚举：trust/distrust/undecided/no_claim）、"
                "vote_basis（枚举：seer_check/seer_siding/speech_logic/vote_pattern/pressure_test/anti_herd/fallback）、"
                "standing_with_seer（你站边哪个预言家/逻辑线，没有则写空字符串）、"
                "suspect_reason（为什么怀疑最终投票对象）、"
                "not_voting_reason（为什么不投其他主要候选人）、"
                "private_reason（完整内心活动：为什么投他、担心什么、最终如何决定）。"
                "这些字段不会公开发言，只给主持人审计。"
            )
            # P1-K6: anti-herd advice is role-gated. 抱团 (herding) is the
            # wolves' core day-vote coordination strategy — non-fake_seer
            # wolves are EXPECTED to follow the fake_seer's lead. Telling
            # wolves to be wary of herding inverts the wolf team's
            # actual coordination goal.
            #
            # Good-side roles (villager / seer / witch / hunter / idiot):
            # the existing anti-herd text — independent judgment is the
            # whole point of being a good player.
            #
            # Wolf-side roles (werewolf / hybrid, regardless of master):
            # a wolf-specific message that frames 抱团 as expected
            # coordination, with the 倒钩 (deep-hook) exception noted.
            #
            # Unknown / unset role: fall back to the good-side text
            # (safe default — better to over-warn than to silently
            # hand wolves the team-coordination cue).
            _GOOD_SIDE = {"villager", "seer", "witch", "hunter", "idiot"}
            _WOLF_SIDE = {"werewolf"}
            role = ctx.own_role or ""
            # Hybrid never receives hidden-faction coordination advice because
            # the role does not know its master's faction.
            if role == "hybrid":
                is_wolf_side = False
            else:
                is_wolf_side = role in _WOLF_SIDE
            if is_wolf_side:
                lines.append(
                    "狼队抱团是正常策略；投票时跟队友一致是预期行为；"
                    "只有在倒钩场景下需独立判断。"
                    "P2-5: 悍跳狼应跟悍跳队友的归票走，而非原狼队的票型。"
                )
            else:
                # Good side + unknown role (safe default).
                lines.append(
                    "反跟票警告：不要无条件跟随任何人的归票。如果多人集中投同一人，"
                    "检查是否可能是狼人抱团。独立判断优先级：发言逻辑矛盾 > 票型异常 > 谁说了什么。"
                    "不要仅因立场反复或票型矛盾机械抗推可能神职；涉及预言家、女巫、猎人、白痴时，"
                    "先核验证据链和技能风险。"
                )
        return "\n".join(lines)

    def _build_belief_state(self) -> str:
        ctx = self.context
        if not ctx.belief_state:
            return ""
        # P2-8: cap at top 3 suspects / top 3 trusted (was 5+5 = 10).
        # Combined with 3 salience items, the section stays under
        # 150 tokens in the typical case (was 200-400 with 5+5+4).
        _MAX_BELIEF_ITEMS = 3
        suspects = ctx.belief_state.get("my_suspects", [])
        trusted = ctx.belief_state.get("my_trusted", [])
        belief_lines = []
        if suspects:
            suspect_desc = ", ".join(
                f"{s['player']}(嫌疑{s['faction_lean']}, 猜{s['top_role_guess']})"
                for s in suspects[:_MAX_BELIEF_ITEMS]
            )
            belief_lines.append(f"我怀疑的玩家: {suspect_desc}")
        if trusted:
            trust_desc = ", ".join(
                f"{t['player']}(倾向{t['faction_lean']}, 信任{t['trust']})"
                for t in trusted[:_MAX_BELIEF_ITEMS]
            )
            belief_lines.append(f"我信任的玩家: {trust_desc}")
        if belief_lines:
            return "【我的判断（基于已有信息的推理，可能是错的）】" + " ".join(belief_lines)
        return ""

    def _build_contradiction_alerts(self) -> str:
        ctx = self.context
        alerts = ctx.contradiction_alerts
        if not alerts:
            return ""
        _MAX = 3
        _PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
        typed = [a for a in alerts if isinstance(a, dict)]
        if not typed:
            return ""
        typed.sort(key=lambda a: _PRIORITY_ORDER.get(str(a.get("priority", "")).lower(), 3))
        lines = ["公开矛盾点（从公开行为推出，可作为攻击/防守依据，非裁判定性）:"]
        for a in typed[:_MAX]:
            player = _clean_current_game_token(a.get("player_id") or "", max_chars=24)
            atype = _clean_current_game_token(a.get("alert_type") or "", max_chars=24)
            desc = _clean_current_game_token(a.get("description") or "", max_chars=120)
            line = f"- {player}"
            if atype:
                line += f" {atype}"
            if desc:
                line += f": {desc}"
            lines.append(line)
        return "\n".join(lines) if len(lines) > 1 else ""

    def _build_seer_credibility(self) -> str:
        ctx = self.context
        summary = ctx.seer_credibility
        if not summary:
            return ""
        lines_data = summary.get("seer_lines", [])
        if not lines_data:
            return ""
        lines = ["预言家线可信度（公开证据推断，非裁判真相，仅辅助推理）:"]
        for item in lines_data[:3]:
            claimant = _clean_current_game_token(item.get("claimant") or "", max_chars=12)
            status = _clean_current_game_token(item.get("status") or "", max_chars=12)
            score = _safe_float(item.get("score"), default=0.0)
            checks = ", ".join(
                _clean_current_game_token(c, max_chars=20)
                for c in item.get("checks", [])[:3]
            )
            evidence = ", ".join(
                _clean_current_game_token(e, max_chars=40)
                for e in (item.get("evidence") or [])[:3]
                if str(e or "").strip()
            )
            seg = [f"{claimant} {status} score={score:.2f}"]
            if checks:
                seg.append(f"查验[{checks}]")
            if evidence:
                seg.append(evidence)
            lines.append("- " + "; ".join(seg))
        return "\n".join(lines) if len(lines) > 1 else ""

    def _build_possible_worlds(self) -> str:
        ctx = self.context
        if not ctx.possible_worlds:
            return ""
        worlds = ctx.possible_worlds.get("top_worlds")
        if not isinstance(worlds, list) or not worlds:
            return ""
        warning = self._clean_prompt_text(
            ctx.possible_worlds.get(
                "warning",
                "These are hypotheses from visible evidence, not ground truth.",
            ),
            max_chars=160,
        )
        lines = [
            "可能世界假设: 以下是假设，不是裁判真相；只能用于私有推理，不能当作公开事实。"
        ]
        if warning:
            lines.append(warning)
        for idx, world in enumerate(worlds[:3], start=1):
            if not isinstance(world, dict):
                continue
            label = _clean_current_game_token(
                world.get("label") or f"World {idx}",
                max_chars=40,
            )
            probability = _safe_float(world.get("probability"), default=0.0)
            assignments = world.get("key_assignments")
            if isinstance(assignments, dict):
                assignment_text = ", ".join(
                    f"{_clean_current_game_token(pid, max_chars=16)}="
                    f"{_clean_current_game_token(role, max_chars=24)}"
                    for pid, role in sorted(assignments.items())[:4]
                )
            else:
                assignment_text = ""
            why = _clean_current_game_list_items(
                world.get("why"), limit=2, max_chars=80
            )
            watch_for = _clean_current_game_list_items(
                world.get("watch_for"), limit=2, max_chars=80
            )
            line = f"- {label}: prob={probability:.2f}"
            if assignment_text:
                line += f"; key={assignment_text}"
            if why:
                line += "; why=" + "；".join(why)
            if watch_for:
                line += "; watch=" + "；".join(watch_for)
            lines.append(line)
        return "\n".join(lines) if len(lines) > 1 else ""

    def _build_simulation_predictions(self) -> str:
        ctx = self.context
        simulation = ctx.simulation_predictions
        if not simulation or simulation.get("type") != "simulation":
            return ""
        predictions = simulation.get("predictions")
        if not isinstance(predictions, list) or not predictions:
            return ""
        warning = self._clean_prompt_text(
            simulation.get("warning", "Prediction, not fact."),
            max_chars=120,
        )
        horizon = self._clean_prompt_text(
            simulation.get("horizon", "next_turn"),
            max_chars=40,
        )
        lines = [
            f"Simulation predictions ({horizon}): use as private planning signals only."
        ]
        if warning:
            lines.append(warning)
        for item in predictions[:2]:
            if not isinstance(item, dict):
                continue
            event = _clean_current_game_token(item.get("event"), max_chars=48)
            if not event:
                continue
            probability = _safe_float(item.get("probability"), default=0.0)
            affected_raw = item.get("affected_players")
            affected = [
                _clean_current_game_token(player_id, max_chars=16)
                for player_id in (
                    affected_raw[:3] if isinstance(affected_raw, list) else []
                )
                if str(player_id or "").strip()
            ]
            rationale = _clean_current_game_token(
                item.get("rationale"),
                max_chars=100,
            )
            world_ids = _clean_current_game_list_items(
                item.get("world_ids"),
                limit=3,
                max_chars=32,
            )
            line = f"- {event}: prob={probability:.2f}"
            if affected:
                line += "; players=" + ", ".join(affected)
            if world_ids:
                line += "; worlds=" + ", ".join(world_ids)
            if rationale:
                line += "; why=" + rationale
            lines.append(line)
        return "\n".join(lines) if len(lines) > 1 else ""

    def _build_public_summary(self) -> str:
        ctx = self.context
        if not ctx.public_summary:
            return ""
        return "当前局公开事实:\n" + self._truncate_text(
            ctx.public_summary,
            _MAX_PUBLIC_SUMMARY_CHARS,
        )

    def _build_visible_state(self) -> str:
        ctx = self.context
        if not ctx.visible_world_state:
            return ""
        visible = dict(ctx.visible_world_state)
        visible.pop("private_memory", None)
        if not visible:
            return ""
        return "可见状态: " + self._compact_json(visible)
















    def _build_recent_transcript(self) -> str:
        ctx = self.context
        if not ctx.recent_transcript:
            return ""
        # P2-5: stable sort the items by (day_number, phase_order) so
        # re-entry speeches (appended later in the list) don't jump
        # to the top of the rendered block.  Pre-fix, ``[-N:]`` was
        # taken before any sort, so an out-of-order append could
        # make a re-entry speech appear BEFORE an earlier speech,
        # confusing the LLM's review of the conversation arc.
        #
        # The ``phase_order`` key sorts speeches within a single day
        # (pre-PK → PK → last_words); absent the key we fall back
        # to 0 (i.e. original list order) which is stable on
        # Python's sort.
        items = sorted(
            ctx.recent_transcript[-_MAX_TRANSCRIPT_ITEMS:],
            key=lambda it: (
                it.get("day_number", it.get("day", 0)),
                it.get("phase_order", 0),
            ),
        )
        lines: list[str] = ["近期发言:"]
        for item in items:
            speaker = item.get("speaker", "?")
            text = self._truncate_text(
                str(item.get("text", "")),
                _MAX_TRANSCRIPT_TEXT_CHARS,
            )
            lines.append(f"  [{speaker}] {text}")
        return "\n".join(lines)

    @staticmethod
    def _clean_prompt_text(
        value: Any,
        *,
        max_chars: int = _MAX_PERSONA_LINE_CHARS,
    ) -> str:
        return clean_prompt_text(value, max_chars=max_chars)

    # ── Utility ──

    def _compact_json(self, value: Any) -> str:
        return compact_json(value)

    @staticmethod
    def _structured_json_summary(value: Any) -> dict[str, Any] | None:
        return structured_json_summary(value)

    @staticmethod
    def _summarize_json_value(value: Any) -> Any:
        return summarize_json_value(value)

    @staticmethod
    def _truncate_text(
        text: str,
        max_chars: int,
        *,
        marker: str = "...（已截断）",
        prefer_sentence_boundary: bool = True,
    ) -> str:
        return truncate_text(
            text,
            max_chars,
            marker=marker,
            prefer_sentence_boundary=prefer_sentence_boundary,
        )
