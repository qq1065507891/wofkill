"""Player prompt builder: s10 pipeline architecture.

Assembles system and user prompts from clearly separated sources.
Follows s10: system_prompt = stable sections, user_prompt = dynamic sections.

Boundary: === DYNAMIC_BOUNDARY === marks where stable rules end and
per-turn context begins.  Stable sections go in the system prompt;
dynamic sections go in the user message (effectively a "system reminder").
"""

from __future__ import annotations

import json
from typing import Any

from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    OutputMode,
    RetryInfo,
    TaskType,
)

# Role name mapping for Chinese prompts
_ROLE_NAMES = {
    "werewolf": "狼人", "villager": "村民", "seer": "预言家",
    "witch": "女巫", "hunter": "猎人", "idiot": "白痴",
    "hybrid": "混血儿",
}

# P0-K1: skill catalog removed (tool path is dead code). Skill analyses
# are pre-injected via skill_analysis_hints — no separate tool catalog.

# Choice pipeline constants
_CHOICE_TARGET_ACTIONS = {
    ActionType.VOTE, ActionType.WOLF_KILL, ActionType.USE_POISON,
    ActionType.CHECK_ALIGNMENT, ActionType.CHOOSE_MASTER,
    ActionType.HUNTER_SHOT, ActionType.BADGE_TRANSFER,
    ActionType.SHERIFF_VOTE,
}

_SPEECH_INTENT_TASKS = {
    TaskType.SPEECH, TaskType.SHERIFF_SPEECH,
    TaskType.DEFENSE_SPEECH, TaskType.PK_SPEECH,
    TaskType.LAST_WORDS,
}

_SPEECH_INTENTS = {
    "self_clear": "表水",
    "question_target": "质疑/追问目标",
    "stand_with_seer": "站边预言家或逻辑线",
    "respond_pressure": "回应质疑",
    "push_vote": "提出投票倾向",
    "info_synthesis": "整合多人发言要点，提出综合判断",
    "anti_herd_call": "指出跟票风险，提醒大家独立判断",
}

_MAX_JSON_CONTEXT_CHARS = 1800
_MAX_TRANSCRIPT_ITEMS = 4
_MAX_TRANSCRIPT_TEXT_CHARS = 220
_MAX_SALIENCE_ITEMS = 4

# P0-S5: strategy_directive is split into 3 priority tiers so the LLM can
# distinguish hard constraints (must obey) from suggestions (recommended) and
# reference context (background). New keys not listed here default to 参考.
# Game trace g_3528592081 confirmed directives are received and acted on
# (e.g., p08 wolf followed `wolf_fake_seer_execution` to claim seer), so
# the LLM needs explicit priority labels to disambiguate.
HARD_CONSTRAINT_KEYS: frozenset[str] = frozenset({
    # Wolf fake-seer execution plan — set at night, must be acted on
    "wolf_fake_seer_execution",
    # Contradiction / role alerts that must be addressed
    "must_address_alerts",
    # N1 death identity (forced to be silent or to mention)
    "first_night_killed",
    # Speech must be empty / vote must be empty
    "speech_silent",
    "vote_silent",
    # Witch must use / not use antidote/poison this night
    "witch_night_action",
    # Per-role alerts (hunter, idiot, etc.)
    "role_alerts",
    # Hard vote pressure (e.g., must-vote target)
    "vote_pressure",
})

SUGGESTION_KEYS: frozenset[str] = frozenset({
    # Wolf speech style / universal rules
    "wolf_speech_directive",
    "wolf_universal_rules",
    # Good-side vote quality guard
    "good_vote_decision_guard",
    # Anti-herd / sheriff vote push
    "anti_herd",
    "sheriff_vote_push",
    # Speech style suggestions
    "speech_originality",
    "seer_speech_directive",
    "witch_speech_constraint",
    # Behavioral rules (anti-following, peace-night rule)
    "anti_following_and_peace_night_rule",
})

REFERENCE_KEYS: frozenset[str] = frozenset({
    # Tactical advice from skill analysis (read-only)
    "skill_tactical_advice",
    # Wolf target / plan hints (background, not binding)
    "wolf_day_push_target",
    "wolf_high_priority_target",
    "wolf_plan_target",
    # Hybrid master behavior summary
    "master_behavior_summary",
    # Witch pressure / strategy hints
    "witch_pressure",
    "witch_strategy_hint",
    # Public discussion summary
    "day_discussion_summary",
    # Vote pressure context (not hard)
    "vote_pressure_context",
})

_STRATEGY_GROUP_ORDER: tuple[frozenset[str], str, str] = (
    (HARD_CONSTRAINT_KEYS, "【硬约束】", "以下指令必须遵守（MUST）："),
    (SUGGESTION_KEYS, "【建议】", "以下指令为建议（SHOULD），偏离时需有充分理由："),
    (REFERENCE_KEYS, "【参考】", "以下为背景信息（REFERENCE），仅供决策参考："),
)


class PlayerPromptBuilder:
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
        parts: list[str] = []
        parts.append(self._build_core_identity())
        # P2-S10: _build_persona() moved to build_user_prompt() — persona
        # is per-turn (situation-driven) and should be a dynamic section
        # grouped with other per-turn context, not a stable section in
        # the system prompt.
        parts.append(self._build_information_boundaries())
        parts.append(self._build_game_rules())
        parts.append(self._build_role_guide())
        parts.append(self._build_reasoning_method())
        # P0-K1: skill tool path removed; policy about calling skill tools
        # is gone. Skill analyses are pre-injected (skill_analysis_hints).
        parts.append(self._build_skill_policy())
        parts.append(self._build_output_contract())
        return "\n\n".join(p for p in parts if p)

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
            "【平安夜与女巫规则硬约束】平安夜不等于无人被刀，只代表公开结果无人死亡。"
            "除狼人外，普通玩家不知道狼人是否空刀；除女巫外，普通玩家不知道女巫是否救人。"
            "不能用「平安夜没人死」反驳女巫知道刀口，也不能把「不公开救谁」直接等同于假女巫。"
            "可以质疑跳女巫玩家是否用药、为什么暂不公开银水、以及发言前后是否矛盾。"
            "不要跟风复述已有指控；每次发言必须给出独立证据、明确区分事实和推测。\n"
            "【公开记录引用约束】只有游戏概况、可见状态、关键事件、近期发言中明确出现的信息，才能称为公开记录。"
            "不要编造某玩家曾经说过的话、声称过的身份、投票理由或查验结论；不确定时必须写成推测或质疑。"
        )

    def _build_information_boundaries(self) -> str:
        return (
            "【信息边界】你收到的信息分为：当前局公开事实、当前局私有信息、知识库提示、"
            "跨局记忆、认知矩阵、技能分析结果、模型推测。"
            "公开发言时，只有当前局公开事实可以称为“场上已知”或“公开记录”。"
            "私有信息可以用于决策，但不能伪装成公开事实。"
            "知识库提示只是玩法经验，不是当前局发生的事。"
            "跨局记忆只是历史经验，不代表本局任何玩家真实身份。"
            "认知矩阵只是你自己的判断倾向，不是事实。"
            "技能分析只是辅助推理，不改变规则、身份或公开记录。"
            "不确定内容必须表达为推测。"
        )

    def _build_reasoning_method(self) -> str:
        return (
            "【推理方法】先区分事实、推测、立场、情绪。"
            "盘狼坑时优先看：发言矛盾、票型关系、站边链条、收益动机、关键轮次行为。"
            "投票前比较证据链完整度和出错成本，不盲从多数归票。"
            "允许保留不确定性，但行动必须给出当前最优理由。"
        )

    def _build_skill_policy(self) -> str:
        return (
            "【技能与建议】系统会在你的回合前注入已计算的技能分析结果，"
            "请基于这些分析与当前局可见事实形成自己的判断，不要机械复述。"
            "技能分析不是裁判真相；如果与公开事实冲突，以公开事实为准。"
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
            "idiot": "白痴规则：被放逐时亮出身份免死，但失去投票权且不能再被放逐；之后被狼人杀死才算真正死亡。夜间无法自保。",
            "witch": "女巫规则：有一瓶解药和一瓶毒药，不能在同一夜同时使用。解药不能自救。N1 / 首夜大概率应该救人。",
            "seer": "预言家规则：每晚可查验一人身份（好人/狼人），查验混血儿结果为好人。上警时必须留两夜警徽流。",
            "werewolf": "狼人规则：夜间与队友讨论击杀目标。可以悍跳预言家上警对抗真预言家。",
            "hybrid": "混血儿规则：N1 / 首夜选择一名主人，跟随主人阵营获胜。主人死亡后阵营不再改变。",
            "villager": (
                "村民规则：身份公开时积极表明好人立场；"
                "分析发言矛盾/票型关系；"
                "N1 用药决策推动解药救人；"
                "归票基于公开证据链而非情绪。"
            ),
        }
        if role in role_rules:
            lines.append(role_rules[role])
        return "\n".join(lines) if lines else ""

    def _build_output_contract(self) -> str:
        """Stable output format rules — same regardless of phase."""
        return (
            "请优先通过 submit_player_action 工具提交结构化行动。"
            "如果当前模型无法调用工具，则只输出一个JSON对象，不要解释、不要Markdown。"
            "字段必须包含 action_type、target_id、speech、reason、confidence。"
            "重要：speech字段必须使用中文，这是你在游戏中的公开发言。"
        )

    # ═══════════════════════════════════════════════════════════════
    #  User prompt: per-turn dynamic context (system reminder)
    # ═══════════════════════════════════════════════════════════════

    # P1-S3: section-level priority labels so the LLM can rank which
    # sections to attend to under tight token budget. The labels group
    # the 16 user-prompt sections into three priority tiers:
    #   - 硬约束 (HARD):     must be addressed / must be obeyed
    #   - 辅助 (AUXILIARY):  background context, ignore non-relevant
    #   - 可选 (OPTIONAL):   reference, may be skimmed or dropped
    # Note: this is the OUTER section label, distinct from the inner
    # 硬约束/建议/参考 sub-grouping already in P0-S5 for strategy_directive.
    _SECTION_PRIORITIES: dict[str, str] = {
        "_build_persona": "【辅助】",
        "_build_phase_context": "【辅助】",
        "_build_belief_state": "【辅助】",
        "_build_public_summary": "【辅助】",
        "_build_visible_state": "【辅助】",
        "_build_private_memory_hints": "【辅助】",
        "_build_salience_events": "【辅助】",
        "_build_rag_hints": "【辅助】",
        "_build_reflection_memory_hints": "【辅助】",
        "_build_profile_memory_hint": "【辅助】",
        "_build_cognition_matrix_hint": "【辅助】",
        "_build_strategy_directive": "【硬约束】",
        "_build_skill_analysis_hints": "【辅助】",
        "_build_recent_transcript": "【可选】",
        "_build_retry_hint": "【硬约束】",
        "_build_strict_output_contract": "【硬约束】",
        # Note: _build_task_prompt is intentionally unlabeled — the
        # task prompt is the action spec the LLM is executing.
    }

    def _label_section(self, builder_name: str, body: str) -> str:
        """Prepend the priority label to a section's body.

        P1-S3: Empty bodies are returned unchanged so the section
        just disappears from the prompt (preserving the existing
        `for p in parts if p` filter behavior).
        """
        if not body:
            return body
        label = self._SECTION_PRIORITIES.get(builder_name, "")
        if not label:
            return body
        return f"{label} {body}"

    def build_user_prompt(self, retry: RetryInfo) -> str:
        parts: list[str] = []
        # Boundary marker per s10: above = stable, below = dynamic
        parts.append("=== DYNAMIC_BOUNDARY ===")
        # P2-S10: persona (per-turn style/tone hint) lives in the user
        # message, right after the boundary marker, so it stays grouped
        # with other per-turn dynamic context and does not invalidate
        # the system-prompt cache on each turn.
        parts.append(self._label_section("_build_persona", self._build_persona()))
        # P1-S3: each section is wrapped with a [硬约束/辅助/可选]
        # priority label so the LLM can rank attention under tight
        # token budgets. The label is prepended at the section level
        # — internal sub-grouping (e.g., P0-S5 within strategy_directive)
        # is preserved.
        parts.append(self._label_section("_build_phase_context", self._build_phase_context()))
        parts.append(self._label_section("_build_belief_state", self._build_belief_state()))
        parts.append(self._label_section("_build_public_summary", self._build_public_summary()))
        parts.append(self._label_section("_build_visible_state", self._build_visible_state()))
        parts.append(self._label_section("_build_private_memory_hints", self._build_private_memory_hints()))
        parts.append(self._label_section("_build_salience_events", self._build_salience_events()))
        parts.append(self._label_section("_build_rag_hints", self._build_rag_hints()))
        parts.append(self._label_section("_build_reflection_memory_hints", self._build_reflection_memory_hints()))
        parts.append(self._label_section("_build_profile_memory_hint", self._build_profile_memory_hint()))
        parts.append(self._label_section("_build_cognition_matrix_hint", self._build_cognition_matrix_hint()))
        parts.append(self._label_section("_build_strategy_directive", self._build_strategy_directive()))
        parts.append(self._label_section("_build_skill_analysis_hints", self._build_skill_analysis_hints()))
        # P0-K1: skill tool path removed. Skill analyses are pre-injected
        # above (skill_analysis_hints) — no separate tool-catalog section.
        parts.append(self._label_section("_build_recent_transcript", self._build_recent_transcript()))
        # P0-S6: retry hint must come AFTER task prompt and BEFORE the
        # output contract. Old order put retry BEFORE task, so the LLM
        # read "纠正提示..." and then got distracted by the task
        # description that followed — easy to miss the correction.
        # New order (task → retry → contract) makes the correction the
        # last thing the LLM sees before the output contract.
        parts.append(self._build_task_prompt())
        parts.append(self._label_section("_build_retry_hint", self._build_retry_hint(retry)))
        parts.append(self._label_section("_build_strict_output_contract", self._build_strict_output_contract()))
        return "\n\n".join(p for p in parts if p)

    def _build_phase_context(self) -> str:
        ctx = self.context
        lines = [f"当前阶段: {ctx.phase}"]
        if ctx.legal_actions:
            lines.append(f"可用操作: {[a.value for a in ctx.legal_actions]}")
        if ctx.legal_targets:
            lines.append(f"可选目标: {ctx.legal_targets}")
        # Mandatory vote hints
        if ctx.legal_actions and ActionType.VOTE in ctx.legal_actions:
            if ActionType.NO_ACTION not in ctx.legal_actions:
                lines.append("重要：本轮投票必须选择一名玩家放逐，不能弃票！")
            if ctx.legal_actions == [ActionType.VOTE] and ctx.legal_targets:
                lines.append("你必须投出选票，从可选目标中选择一人。")
            lines.append(
                "投票时必须先在心里完成判断，并在JSON中额外给出这些私有字段："
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
            # P1-2: hybrid's bucket depends on master_faction. ~50% of
            # hybrids choose a good-side master and should see the
            # good-side anti-herd text, not the wolf-side coordination
            # message. Default to good-side when unset (safe default —
            # over-warn > silent team-coordination cue leak).
            if role == "hybrid":
                is_wolf_side = ctx.hybrid_master_faction == "werewolf"
            else:
                is_wolf_side = role in _WOLF_SIDE
            if is_wolf_side:
                lines.append(
                    "狼队抱团是正常策略；投票时跟队友一致是预期行为；"
                    "只有在倒钩场景下需独立判断。"
                )
            else:
                # Good side + unknown role (safe default).
                lines.append(
                    "反跟票警告：不要无条件跟随任何人的归票。如果多人集中投同一人，"
                    "检查是否可能是狼人抱团。独立判断优先级：发言逻辑矛盾 > 票型异常 > 谁说了什么。"
                )
        return "\n".join(lines)

    def _build_belief_state(self) -> str:
        ctx = self.context
        if not ctx.belief_state:
            return ""
        suspects = ctx.belief_state.get("my_suspects", [])
        trusted = ctx.belief_state.get("my_trusted", [])
        belief_lines = []
        if suspects:
            suspect_desc = ", ".join(
                f"{s['player']}(嫌疑{s['faction_lean']}, 猜{s['top_role_guess']})"
                for s in suspects[:5]
            )
            belief_lines.append(f"我怀疑的玩家: {suspect_desc}")
        if trusted:
            trust_desc = ", ".join(
                f"{t['player']}(倾向{t['faction_lean']}, 信任{t['trust']})"
                for t in trusted[:5]
            )
            belief_lines.append(f"我信任的玩家: {trust_desc}")
        if belief_lines:
            return "【我的判断（基于已有信息的推理，可能是错的）】" + " ".join(belief_lines)
        return ""

    def _build_public_summary(self) -> str:
        ctx = self.context
        if not ctx.public_summary:
            return ""
        return "当前局公开事实:\n" + self._truncate_text(ctx.public_summary, _MAX_JSON_CONTEXT_CHARS)

    def _build_visible_state(self) -> str:
        ctx = self.context
        if not ctx.visible_world_state:
            return ""
        visible = dict(ctx.visible_world_state)
        visible.pop("private_memory", None)
        if not visible:
            return ""
        return "可见状态: " + self._compact_json(visible)

    def _build_private_memory_hints(self) -> str:
        ctx = self.context
        # P0-M7: read only from private_memory_hints. The previous code
        # also fell back to ctx.visible_world_state.get("private_memory"),
        # which caused duplicate injection if both fields were populated
        # and risked surfacing stale data from an older code path.
        memory = ctx.private_memory_hints
        if not memory:
            return ""
        # P0-M1: prepend a "本局·第N轮·私有记忆" label so the LLM cannot
        # confuse this section with cross-game reflection memory or
        # with public speech.
        day_label = f"第{ctx.day_number}轮" if ctx.day_number else "首轮"
        # MEM-02: emit the P1-M10 caveat BEFORE the JSON payload so
        # the LLM sees the warning in the same paragraph as the
        # keyword-signal categories. Omitted when logic_flaws /
        # valid_points are both empty (caveat would be noise).
        caveat = ctx.private_memory_caveat or ""
        caveat_block = f"{caveat}\n" if caveat else ""
        return (
            f"【本局·{day_label}·私有记忆】以下只代表你在本局形成的观察、站边和私有思考，"
            "不是公开记录。"
            "【严禁】在公开发言中复述以下任何角色身份信息或暗示你从私有记忆中获知的身份。"
            "你在公开发言中只能使用公开可见的信息。\n"
            + caveat_block
            + self._compact_json(memory)
        )

    def _build_salience_events(self) -> str:
        ctx = self.context
        if not ctx.salience_items:
            return ""
        # P0-2 (defense in depth): explicitly whitelist public fields
        # so a future change that leaks a private key (seer_result,
        # witch_target, wolf_team, private_intent, moderator_full) into
        # ctx.salience_items cannot end up in the player-visible prompt.
        # The runtime (runtime/context.py:build_agent_context) does not
        # currently populate these, but the renderer should still
        # enforce the boundary.
        slimmed = [_slim_salience_item(item) for item in ctx.salience_items[:_MAX_SALIENCE_ITEMS]]
        slimmed = [item for item in slimmed if item is not None]
        if not slimmed:
            return ""
        return "关键事件: " + self._compact_json(slimmed)

    def _build_rag_hints(self) -> str:
        ctx = self.context
        if not ctx.rag_hints:
            return ""
        # P0-G2 defense in depth: even if a non-production code path
        # leaks full audit items into ``ctx.rag_hints``, the live prompt
        # must only see title / summary / key_decisions. Audit data
        # (relevance, quality, source, visibility, display annotation)
        # belongs in the audit log, not the LLM context window.
        slim_items = self._slim_rag_hint_items(ctx.rag_hints[:3])
        # P0-G3: hard-constraint prefix MUST come before the JSON
        # payload. Without this the LLM has been observed to parrot
        # case-specific player IDs (e.g., p04 / p09 in seed cases) as
        # if they were this game's player IDs, which is information
        # leakage and tactical error.
        warning = (
            "⚠️ RAG 案例中的玩家 ID 与本局无关；"
            "不得直接套用案例中具体玩家的发言或票型。\n"
        )
        return (
            "知识库提示: 知识库提示不是当前局事实，只能作为玩法经验和案例参考。\n"
            + warning
            + self._compact_json(slim_items)
        )

    @staticmethod
    def _slim_rag_hint_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop audit-only fields from RAG hint items in the live prompt.

        Mirrors :func:`werewolf_agent.rag.prompt_renderer.hits_to_prompt_lines`
        but operates on already-rendered dicts (so it works even when
        ``ctx.rag_hints`` was populated by a test or a non-default
        code path that bypassed :class:`RAGKnowledgeService`).

        The renderer treats a dict as a "slim line" if it already only
        has the three prompt-safe keys; otherwise it picks out those
        three keys, falling back to ``""`` / ``""`` / ``[]`` if absent.
        """
        slim: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            slim.append({
                "title": str(item.get("title", "") or ""),
                "summary": str(item.get("summary", "") or ""),
                "key_decisions": list(item.get("key_decisions") or [])[:3],
            })
        return slim


    def _build_reflection_memory_hints(self) -> str:
        ctx = self.context
        if not ctx.reflection_memory_hints:
            return ""
        return (
            "跨局反思记忆: 以下是你过往对局后的经验总结，不代表本局任何玩家真实身份。\n"
            + self._compact_json(ctx.reflection_memory_hints[:5])
        )

    def _build_profile_memory_hint(self) -> str:
        ctx = self.context
        if not ctx.profile_memory_hint:
            return ""
        return (
            "长期能力画像: 以下是你的历史能力画像和角色经历，只用于调整策略风格。\n"
            + self._compact_json(ctx.profile_memory_hint)
        )

    def _build_cognition_matrix_hint(self) -> str:
        ctx = self.context
        if not ctx.cognition_matrix_hint:
            return ""
        return (
            "我的认知矩阵: 以下是你自己的判断倾向，不是事实，也不包含其他玩家私密视角。\n"
            + self._compact_json(ctx.cognition_matrix_hint)
        )

    def _build_strategy_directive(self) -> str:
        """Render strategy_directive split into 3 priority sections.

        P0-S5: LLM previously saw a single flat JSON block with 20+ keys
        and no priority signal — had to guess hard vs soft. Now keys are
        grouped into 【硬约束】 (MUST), 【建议】 (SHOULD), 【参考】
        (REFERENCE). Unknown keys fall through to 参考 for forward-compat.
        """
        ctx = self.context
        if not ctx.strategy_directive:
            return ""

        grouped: dict[str, dict[str, Any]] = {
            header: {} for _, header, _ in _STRATEGY_GROUP_ORDER
        }
        reference_fallback_idx = next(
            i for i, (keys, header, _) in enumerate(_STRATEGY_GROUP_ORDER)
            if header == "【参考】"
        )
        reference_header = _STRATEGY_GROUP_ORDER[reference_fallback_idx][1]

        for key, value in ctx.strategy_directive.items():
            placed = False
            for i, (keys, header, _) in enumerate(_STRATEGY_GROUP_ORDER):
                if key in keys:
                    grouped[header][key] = value
                    placed = True
                    break
            if not placed:
                grouped[reference_header][key] = value

        parts: list[str] = ["本轮策略指令:"]
        for keys_set, header, label in _STRATEGY_GROUP_ORDER:
            section = grouped[header]
            if not section:
                continue
            parts.append(f"{header} {label}\n" + self._compact_json(section))
        return "\n\n".join(parts)

    def _build_skill_analysis_hints(self) -> str:
        ctx = self.context
        if not ctx.skill_analysis_hints:
            return ""
        return (
            "技能分析结果: 以下是本轮工具分析结果，只能作为辅助推理。\n"
            + self._compact_json(ctx.skill_analysis_hints)
        )

    def _build_persona(self) -> str:
        ctx = self.context
        if not ctx.persona_snapshot:
            return ""
        return "人格设定: " + self._compact_json(ctx.persona_snapshot)

    def _build_recent_transcript(self) -> str:
        ctx = self.context
        if not ctx.recent_transcript:
            return ""
        lines: list[str] = ["近期发言:"]
        for item in ctx.recent_transcript[-_MAX_TRANSCRIPT_ITEMS:]:
            speaker = item.get("speaker", "?")
            text = self._truncate_text(
                str(item.get("text", "")),
                _MAX_TRANSCRIPT_TEXT_CHARS,
            )
            lines.append(f"  [{speaker}] {text}")
        return "\n".join(lines)

    def _build_retry_hint(self, retry: RetryInfo) -> str:
        """Render the retry correction hint after task, before contract.

        P0-S6: When the LLM hits 3 retries on the same parse_error
        (e.g., game trace g_3528592081 Action 50, p10), a generic
        correction_hint like "只输出JSON..." is not actionable. We now
        surface the first 100 chars of the actual error_message so the
        LLM sees what specifically went wrong on the previous attempt.
        Truncation keeps the hint focused and avoids leaking long traces
        into the next prompt.

        P0-R2: When failure_category == "timeout" and error_code ==
        "empty_response", append a Chinese hint giving the LLM explicit
        permission to return `no_action` as a safe no-op. Without this,
        the model either retries and times out again or fabricates a
        vote target. Game trace g_3528592081 Action 57 (seer p03 vote)
        hit 3 empty retries and fell back to a default target — a
        '如果超时, 返回 no_action' hint would have let it safely no-op.
        """
        if not retry.correction_hint and not retry.error_message:
            return ""

        lines = [
            f"纠正提示（第{retry.attempt}/{retry.max_retries}次尝试）："
        ]
        if retry.error_message:
            snippet = retry.error_message[:100]
            if len(retry.error_message) > 100:
                snippet += "..."
            lines.append(f"上次错误: {snippet}")
        if retry.correction_hint:
            lines.append(f"修正建议: {retry.correction_hint}")
        # P0-R2: empty_response + timeout → safe no-op permission.
        if (
            retry.error_code == "empty_response"
            and retry.failure_category == "timeout"
        ):
            lines.append(
                "重要：如果你已经超时，请直接返回 no_action"
                "（action_type='no_action', target_id=null, "
                "reason='timeout - safe no-op'），"
                "不要再尝试长推理或构造JSON。"
            )
        return "\n".join(lines)

    def _build_task_prompt(self) -> str:
        """Task-specific prompt: choice enum, speech intent, or examples."""
        output_mode = self._select_output_mode()
        if output_mode == OutputMode.TARGET_CHOICE:
            return self._format_choice_prompt()
        if output_mode == OutputMode.SPEECH_INTENT:
            return self._format_speech_intent_prompt()
        return self._format_examples()

    def _format_examples(self) -> str:
        ctx = self.context
        parts: list[str] = []
        # P0-S7: claimed_view is documented as an identity-perspective
        # identifier (PrivateIntent schema), not a free-form Chinese
        # phrase. Use the canonical enum-style values so the LLM copies
        # a clean identifier instead of "我是好人" / "我是预言家".
        # Game trace g_3528592081 showed wolves writing
        # "我是好人，混水摸鱼" — a strategy note in natural Chinese —
        # when the example primed them to do so.
        _CLAIMED_VIEW_GOOD = "good_player_without_night_info"
        _CLAIMED_VIEW_SEER = "seer"
        if ctx.legal_actions and any(
            a in (ActionType.WOLF_KILL, ActionType.WOLF_NO_KILL) for a in ctx.legal_actions
        ):
            example_target = ctx.legal_targets[0] if ctx.legal_targets else "p05"
            parts.append("示例输出（狼人击杀场景）：")
            parts.append(
                f'{{"action_type": "wolf_kill", "target_id": "{example_target}", '
                f'"speech": "", '
                f'"reason": "选择击杀目标", "confidence": 0.8, '
                f'"private_intent": {{"true_role": "werewolf", '
                f'"faction_goal": "push_good_player_out", "claimed_view": "{_CLAIMED_VIEW_GOOD}", '
                f'"pressure_target": "{example_target}", "risk_flags": []}}}}'
            )
            parts.append("示例输出（狼人空刀场景）：")
            parts.append(
                '{"action_type": "wolf_no_kill", "target_id": null, '
                '"speech": "", '
                '"reason": "本轮空刀策略", "confidence": 0.6, '
                '"private_intent": {"true_role": "werewolf", '
                f'"faction_goal": "confuse_good", "claimed_view": "{_CLAIMED_VIEW_GOOD}", '
                '"pressure_target": null, "risk_flags": []}}'
            )
        elif ctx.legal_actions and ActionType.SHERIFF_REGISTER in ctx.legal_actions:
            parts.append("示例输出（上警报名场景）：")
            parts.append(
                '{"action_type": "sheriff_register", "target_id": null, '
                '"speech": "我报名竞选警长。", '
                '"reason": "希望参与警上发言并争取带队", "confidence": 0.6}'
            )
            if ActionType.NO_ACTION in ctx.legal_actions:
                parts.append("示例输出（不上警场景）：")
                parts.append(
                    '{"action_type": "no_action", "target_id": null, '
                    '"speech": "我不上警，先听警上发言再判断。", '
                    '"reason": "当前信息不足，先观察警上格局", "confidence": 0.6}'
                )
        else:
            role = ctx.own_role or "villager"
            # P0-1: example_role now follows ctx.own_role so the example's
            # `private_intent.true_role` reflects the agent's actual role
            # instead of being hardcoded to "villager". Wolves still get
            # the wolf-style example; every other role uses their own
            # role identifier. faction_goal and claimed_view stay role-aware.
            if role == "werewolf":
                example_role = "werewolf"
                example_goal = "confuse_good"
                example_view = _CLAIMED_VIEW_GOOD
            else:
                example_role = role
                example_goal = "find_wolves"
                example_view = _CLAIMED_VIEW_GOOD
            parts.append("示例输出（发言场景）：")
            parts.append('{"action_type": "speech", "target_id": null, '
                         '"speech": "我觉得p05很可疑，昨晚他的发言前后矛盾。", '
                         '"reason": "根据发言分析", "confidence": 0.7, '
                         f'"private_intent": {{"true_role": "{example_role}", '
                         f'"faction_goal": "{example_goal}", "claimed_view": "{example_view}", '
                         '"pressure_target": "p05", "risk_flags": []}}')
            vote_example_goal = "confuse_good" if example_role == "werewolf" else "find_wolves"
            # P0-S7: check the input role, not example_role. The original
            # code checked example_role, which was hardcoded to "villager"
            # for the seer case — so the seer claimed_view branch was
            # never actually triggered. Using role here makes the
            # claimed_view match the player's own identity.
            vote_example_view = _CLAIMED_VIEW_SEER if role == "seer" else _CLAIMED_VIEW_GOOD
            # P0-4: a seer agent stands with their OWN check, not with
            # another seer. The example must therefore set
            # `standing_with_seer=""` (own ID is implicit) and keep
            # `vote_basis="seer_check"` (now meaning "based on my own
            # check"). Non-seer roles continue to see the p03 example
            # since they do side with an external seer claim.
            if role == "seer":
                vote_standing_with_seer = ""
            else:
                vote_standing_with_seer = "p03"
            parts.append("示例输出（投票场景）：")
            parts.append('{"action_type": "vote", "target_id": "p05", '
                         '"speech": "", '
                         '"reason": "公开理由：p05发言可疑", '
                         '"seer_stance": "trust", '
                         '"vote_basis": "seer_check", '
                         f'"standing_with_seer": "{vote_standing_with_seer}", '
                         '"suspect_reason": "p05没有回应p03的查杀逻辑，发言前后不一致", '
                         '"not_voting_reason": "p07虽然被踩，但目前没有明确查验或票型证据", '
                         '"private_reason": "心里活动：我更信p03的预言家线，p05像狼队抗推失败后的防守位，所以投p05。", '
                         '"confidence": 0.8, '
                         f'"private_intent": {{"true_role": "{example_role}", '
                         f'"faction_goal": "{vote_example_goal}", "claimed_view": "{vote_example_view}", '
                         '"pressure_target": "p05", "risk_flags": []}}')
        return "\n".join(parts)

    def _build_strict_output_contract(self) -> str:
        """Per-turn output contract — adapts to task type."""
        ctx = self.context
        output_mode = self._select_output_mode()
        legal_actions = [a.value for a in ctx.legal_actions]
        legal_targets = list(ctx.legal_targets)

        if output_mode == OutputMode.TARGET_CHOICE:
            output_fields = "choice、reason、confidence"
            if ctx.legal_actions == [ActionType.VOTE]:
                output_fields = (
                    "choice、reason、seer_stance、vote_basis、standing_with_seer、suspect_reason、"
                    "not_voting_reason、private_reason、confidence"
                )
            lines = [
                "",
                "最终输出协议（必须遵守）：",
                "1. 只输出一个choice决策JSON对象；不要输出分析过程、解释、Markdown或多余文本。",
                "2. JSON必须以{开头、以}结尾，且只能有一个对象。",
                f"3. 最终输出字段：{output_fields}。",
                "4. choice只能取上方候选枚举中的字母，不要直接编写target_id。",
            ]
            if ctx.legal_actions == [ActionType.VOTE]:
                lines.append(
                    "5. 投票还必须包含seer_stance、vote_basis、standing_with_seer、"
                    "suspect_reason、not_voting_reason、private_reason，理由字段不能写「未说明」。"
                )
            lines.append("现在提交行动。")
            return "\n".join(lines)

        if output_mode == OutputMode.SPEECH_INTENT:
            lines = [
                "",
                "最终输出协议（必须遵守）：",
                "1. 只输出一个发言意图JSON对象；不要输出分析过程、解释、Markdown或多余文本。",
                "2. JSON必须以{开头、以}结尾，且只能有一个对象。",
                "3. 最终输出字段：intent、target_id、speech、reason、confidence。",
                "4. target_id没有目标时必须是null，不要写字符串\"null\"。",
            ]
            if legal_targets:
                lines.append(f"5. target_id只能取这些玩家之一或null：{legal_targets}。")
            lines.append("现在提交行动。")
            return "\n".join(lines)

        lines = [
            "",
            "最终输出协议（必须遵守）：",
            "1. 首选 submit_player_action 工具调用提交结构化参数。",
            "2. 如果当前模型无法工具调用，只输出一个JSON对象；不要输出分析过程、解释、Markdown或多余文本。",
            "3. JSON必须以{开头、以}结尾，且只能有一个对象。",
            "4. target_id没有目标时必须是null，不要写字符串\"null\"。",
            "5. 必填字段：action_type、target_id、speech、reason、confidence。",
        ]
        if legal_actions:
            lines.append(f"6. action_type只能取：{legal_actions}。")
        if legal_targets:
            lines.append(f"7. target_id只能取这些玩家之一或null：{legal_targets}。")
        if ActionType.VOTE in ctx.legal_actions:
            lines.append(
                "8. 投票还必须包含seer_stance、vote_basis、standing_with_seer、"
                "suspect_reason、not_voting_reason、private_reason，理由字段不能写「未说明」。"
            )
        lines.append("现在提交行动。")
        return "\n".join(lines)

    # ── Choice pipeline helpers ──

    def _format_choice_prompt(self) -> str:
        ctx = self.context
        is_vote = ctx.legal_actions == [ActionType.VOTE]
        header = "投票候选枚举" if is_vote else "目标候选枚举"
        choice_map = self._vote_choice_map()
        lines = [f"{header}（必须从中选择一个choice，不要直接编写target_id）："]
        for choice, target_id in choice_map.items():
            summary = (
                self._vote_candidate_summary(target_id)
                if is_vote
                else self._target_candidate_summary(target_id)
            )
            lines.append(f"{choice} = {target_id}，摘要：{summary}")
        if is_vote:
            example = (
                '{"choice":"A","reason":"投票公开理由",'
                '"seer_stance":"trust",'
                '"vote_basis":"seer_check",'
                '"standing_with_seer":"站边的预言家或逻辑线",'
                '"suspect_reason":"为什么怀疑该候选",'
                '"not_voting_reason":"为什么不投其他候选",'
                '"private_reason":"完整内心理由",'
                '"confidence":0.7}'
            )
        else:
            example = '{"choice":"A","reason":"选择该目标的简明理由","confidence":0.7}'
        lines.extend([
            "只需要输出choice决策JSON，程序会把choice映射为target_id并组装PlayerAction。",
            "示例：",
            example,
        ])
        return "\n".join(lines)

    def _format_speech_intent_prompt(self) -> str:
        ctx = self.context
        lines = ["发言意图枚举（先选intent，再写speech；不要输出分析过程）："]
        for intent, label in _SPEECH_INTENTS.items():
            lines.append(f"- {intent}: {label}")
        if ctx.legal_targets:
            lines.append(f"可围绕的目标玩家: {ctx.legal_targets}")
        lines.extend([
            "发言阶段只需要输出intent决策JSON，程序会组装为speech行动。",
            "speech必须是公开发言正文，不能留空，不能写「未发言」。",
            "示例：",
            (
                '{"intent":"question_target","target_id":"p05",'
                '"speech":"我想追问p05，你的站边和投票理由需要讲清楚。",'
                '"reason":"围绕可疑目标施压","confidence":0.7}'
            ),
        ])
        return "\n".join(lines)

    # ── Output mode selection ──

    def _select_output_mode(self) -> OutputMode:
        ctx = self.context
        if self._uses_choice_pipeline():
            return OutputMode.TARGET_CHOICE
        if self._uses_speech_intent_pipeline():
            return OutputMode.SPEECH_INTENT
        return OutputMode.FULL_ACTION

    def _uses_choice_pipeline(self) -> bool:
        ctx = self.context
        return (
            len(ctx.legal_actions) == 1
            and ctx.legal_actions[0] in _CHOICE_TARGET_ACTIONS
            and bool(ctx.legal_targets)
        )

    def _uses_speech_intent_pipeline(self) -> bool:
        ctx = self.context
        return (
            ctx.task_type in _SPEECH_INTENT_TASKS
            and ctx.legal_actions == [ActionType.SPEECH]
        )

    # ── Choice/target helpers ──

    def _vote_choice_map(self) -> dict[str, str]:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ctx = self.context
        return {
            letters[idx]: target
            for idx, target in enumerate(ctx.legal_targets[:len(letters)])
        }

    def _vote_candidate_summary(self, target_id: str) -> str:
        ctx = self.context
        clues: list[str] = []
        for item in ctx.salience_items:
            if not isinstance(item, dict):
                continue
            target = item.get("target") or item.get("target_id") or item.get("player_id")
            if target != target_id:
                continue
            item_type = item.get("type") or item.get("event")
            if item_type == "seer_claim":
                speaker = item.get("speaker") or item.get("seer_id")
                result = item.get("result") or item.get("alignment")
                if speaker and result:
                    clues.append(f"{speaker}报{target_id}为{result}")
            elif item_type in {"vote_resolved", "vote"}:
                clues.append(f"{target_id}出现在关键票型中")
            elif item_type in {"player_died", "death"}:
                clues.append(f"{target_id}关联死亡事件")
        if clues:
            return "；".join(clues[:2])
        return f"{target_id}是当前合法投票候选，需要基于发言、票型和站边继续施压"

    def _target_candidate_summary(self, target_id: str) -> str:
        ctx = self.context
        action = ctx.legal_actions[0] if ctx.legal_actions else ActionType.NO_ACTION
        action_reasons = {
            ActionType.WOLF_KILL: "作为狼队夜间击杀目标",
            ActionType.USE_POISON: "作为女巫毒药目标",
            ActionType.CHECK_ALIGNMENT: "作为预言家查验目标",
            ActionType.CHOOSE_MASTER: "作为混血儿主人选择目标",
            ActionType.HUNTER_SHOT: "作为猎人开枪目标",
            ActionType.BADGE_TRANSFER: "作为警徽移交目标",
            ActionType.SHERIFF_VOTE: "作为警长投票目标",
        }
        clues: list[str] = []
        for item in ctx.salience_items:
            if not isinstance(item, dict):
                continue
            item_text = json.dumps(item, ensure_ascii=False)
            if target_id in item_text:
                clues.append(item_text[:80])
        basis = f"；依据：{'；'.join(clues[:2])}" if clues else ""
        return f"{target_id}{action_reasons.get(action, '作为当前合法目标')}较合适{basis}"

    # ── Utility ──

    def _compact_json(self, value: Any) -> str:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return self._truncate_text(text, _MAX_JSON_CONTEXT_CHARS)

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + f"...（已截断，原长度{len(text)}）"


# P0-2 (defense): explicit field whitelist for salience items. The
# runtime currently does not populate private keys into salience_items,
# but the renderer should enforce the boundary so a future change
# cannot leak `seer_result`, `witch_target`, `wolf_team`, etc. into
# the player-visible prompt.
_SALIENCE_PUBLIC_FIELDS: frozenset[str] = frozenset({
    "weight", "bucket", "fact_type", "source", "target", "value",
    "day", "phase", "event_type",
})
_SALIENCE_PRIVATE_KEYS: frozenset[str] = frozenset({
    "seer_result", "witch_target", "wolf_team",
    "private_intent", "moderator_full",
})


def _slim_salience_item(item: Any) -> dict[str, Any] | None:
    """Return a salience item with only public fields, or None if it
    contains any private key.

    P0-2 (defense): explicit whitelist of public salience fields. Any
    item carrying a private key (seer_result, witch_target, wolf_team,
    private_intent, moderator_full) is dropped entirely. The runtime
    is not expected to populate these, but the renderer must enforce
    the boundary so a future change cannot leak through.
    """
    if not isinstance(item, dict):
        return None
    # If the item contains any private key, drop it entirely.
    if any(key in item for key in _SALIENCE_PRIVATE_KEYS):
        return None
    # Otherwise project to the public-field whitelist. Keep only known
    # public fields so future field additions don't silently leak.
    return {k: v for k, v in item.items() if k in _SALIENCE_PUBLIC_FIELDS}
