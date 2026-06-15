"""Player prompt builder: s10 pipeline architecture.

Assembles system and user prompts from clearly separated sources.
Follows s10: system_prompt = stable sections, user_prompt = dynamic sections.

Boundary: === DYNAMIC_BOUNDARY === marks where stable rules end and
per-turn context begins.  Stable sections go in the system prompt;
dynamic sections go in the user message (effectively a "system reminder").
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from werewolf_agent.agents.action_contract import ActionContract
from werewolf_agent.agents.directive_priority import (
    HARD_CONSTRAINT_KEYS,
    REFERENCE_KEYS,
    SUGGESTION_KEYS,
)
from werewolf_agent.agents.parse_dispatch import select_output_mode
from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    OutputMode,
    RetryInfo,
    TaskType,
)
from werewolf_agent.persona_runtime.router import sanitize_persona_snapshot

# P2-4: role name mapping was previously duplicated as
# ``_ROLE_NAMES`` here AND ``_ROLE_LABEL_CN`` in
# ``runtime.private_memory``.  Two definitions meant any role
# addition had to be edited in two places (and would silently drift
# if one was forgotten).  Now we import the single source of truth
# from private_memory; the duplicate here is deleted.
from werewolf_agent.runtime.private_memory import _ROLE_LABEL_CN as _ROLE_NAMES  # noqa: E402

# M4-1 (2026-06-09): single source of truth for the reflection-hint
# budget.  Imported from ``runtime.context`` so the prompt builder and
# the hint-generation helper cannot drift.  Previously this was a
# closure constant inside ``_reflection_memory_hints`` while the
# prompt builder hard-coded ``[:5]``, silently dropping 3 hints.
from werewolf_agent.runtime.context import REFLECTION_CARD_BUDGET  # noqa: E402

# M2-3 (2026-06-09): single source of truth for the output schema
# field set.  Both ``_build_output_contract`` (system prompt) and
# ``_build_strict_output_contract`` (per-turn strict) must reference
# these constants.  P2-3 attempted to sync them but two literal
# string lists still drift on future edits.  The constants are the
# MAX-set the stable contract advertises to the LLM; per-turn
# strict subsets (e.g. non-vote TARGET_CHOICE, SPEECH_INTENT) remain
# as literals because they are different schemas, not the same
# schema with fewer fields.
_OUTPUT_SCHEMA_VOTE_FIELDS: tuple[str, ...] = (
    "choice", "reason", "seer_stance", "vote_basis",
    "standing_with_seer", "suspect_reason", "not_voting_reason",
    "private_reason", "confidence",
)
_OUTPUT_SCHEMA_SPEECH_FIELDS: tuple[str, ...] = (
    "action_type", "target_id", "speech", "reason", "confidence",
)
_OUTPUT_SCHEMA_SKILL_FIELDS: tuple[str, ...] = (
    "action_type", "target_id", "speech", "reason", "confidence",
)

# M2-3 (2026-06-09): derived list of vote fields BEYOND choice/reason/confidence.
# Subtracted from _OUTPUT_SCHEMA_VOTE_FIELDS so future additions to
# the VOTE constant automatically propagate to the strict-contract
# audit check. The strict prompt advertises these as "additional
# required fields" on top of the stable choice/reason/confidence
# trio, so any field added to VOTE that's not in the subtracted set
# becomes a required audit field for free.
_VOTE_AUDIT_FIELDS: tuple[str, ...] = tuple(
    f for f in _OUTPUT_SCHEMA_VOTE_FIELDS
    if f not in ("choice", "reason", "confidence")
)

# P0-K1: skill catalog removed (tool path is dead code). Skill analyses
# are pre-injected via skill_analysis_hints — no separate tool catalog.

# Speech intent task set shared with parse dispatch.
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
_MAX_PUBLIC_SUMMARY_CHARS = 700
_MAX_TRANSCRIPT_ITEMS = 4
_MAX_TRANSCRIPT_TEXT_CHARS = 220
_MAX_SALIENCE_ITEMS = 3
_MAX_PERSONA_LINE_CHARS = 180
_MAX_LEARNING_TEXT_CHARS = 160
_MAX_RAG_TEXT_CHARS = 220
_MAX_LEARNING_CONTEXT_CHARS = 3_600
_MAX_SKILL_TACTICAL_ADVICE_ITEMS = 3
_MAX_SKILL_TACTICAL_ADVICE_CHARS = 180
_PLAYER_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:p\d{1,3}|player[_-]?\d{1,3}|agent[_-]?\d{1,3})(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
# P0-G3223805846-8: vote 阶段的 ``reason`` 字段是公开发言可见的（对所有
# 玩家公开），禁止任何私视角表述。游戏轨迹 g_3223805846 观察到 LLM 把
# ``我作为预言家`` / ``狼队 N1 刀了 p0X`` 等私有意图写入 ``reason`` 字段，
# 暴露自己的真实身份 / 狼队行动，破坏信息隔离。
_VOTE_REASON_PRIVACY_GUARD = (
    "\n【投票隐私 P0-G3223805846-8】你输出的 `reason` 字段是公开发言可见的"
    "（对所有玩家公开）。**禁止**在 `reason` 中出现以下私视角表述：\n"
    "- '我作为预言家'、'我是女巫'、'我查验了'、'我的金水'、'我的银水'\n"
    "- '我昨晚刀了'、'狼队 N1 刀了 p0X'、'我们狼队'\n"
    "- 任何 'private_intent' 字段中的内容\n"
    "如果你要表达'基于 X 的查验'，应改写为'基于场上 X 报出的查验结果'；"
    "如果你要表达'我作为预言家认为'，应改写为"
    "'X 报出的查验有 Y 漏洞，因此'\n"
)
# P1-5: global user-prompt budget. The cap follows each section's
# ``drop_tier`` in ``_USER_SECTION_SPECS``; never-drop sections remain.
_USER_PROMPT_BUDGET_CHARS = 20_000


@dataclass(frozen=True)
class _SectionSpec:
    """Single source of truth for user-prompt section metadata."""

    builder_name: str
    label: str
    display_name: str
    info_kind: str
    drop_tier: int | None
    public_record: bool = False


_NEVER_DROP_TIER: int | None = None

_STRATEGY_GROUP_ORDER: tuple[frozenset[str], str, str] = (
    (HARD_CONSTRAINT_KEYS, "【硬约束】", "以下指令必须遵守（MUST）："),
    (SUGGESTION_KEYS, "【建议】", "以下指令为建议（SHOULD），偏离时需有充分理由："),
    (REFERENCE_KEYS, "【参考】", "以下为背景信息（REFERENCE），仅供决策参考："),
)

# P1-G3223805846-4: 头部硬约束强调 JSON 顶层字段必须使用 ``action_type``。
# 游戏回放 g_3223805846 显示 LLM 在示例区之后仍输出 ``{"intent": "..."}``,
# 导致解析器拒绝并触发 fallback。 在示例块前注入这个 guard 提示,让模型在
# 模仿示例之前先看清字段名约束。 `_format_examples` 是示例输出的源头,所有
# skill action (wolf_kill / hunter_shot / use_antidote / use_poison /
# badge_transfer / badge_tear / choose_master) 和 speech / vote 示例都
# 走这条路径; 在入口注入一次即可覆盖所有示例。
_ACTION_TYPE_GUARD = (
    "\n【输出字段约束 P1-G3223805846-4】你输出的 JSON 顶层字段名必须使用 `action_type`，"
    "**不要**使用 `intent`、`command`、`action` 等其他名字。"
    "如果你写 `{\"intent\": \"...\"}` 整个输出会被解析器拒绝。\n"
    "正确的字段名：`action_type`, `target_id`, `speech`, `reason`, `confidence`, "
    "`private_intent`（可选嵌套对象）。\n"
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
        parts.append(self._build_game_rules())
        parts.append(self._build_role_guide())
        parts.append(self._build_information_boundaries())
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
            "idiot": "白痴规则：被放逐时亮出身份免死，但失去投票权且不能再被放逐；之后被狼人杀死才算真正死亡。夜间无法自保。",
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

    # Section-level metadata drives both prompt labels and budget
    # trimming. The visible label tells the LLM what kind of content
    # follows; ``drop_tier`` tells the trimmer what can disappear under
    # pressure. Keeping both in one table prevents drift such as a
    # section being labeled "optional" while also being never-dropped.
    _USER_SECTION_SPECS: tuple[_SectionSpec, ...] = (
        _SectionSpec("_build_phase_context", "【辅助】", "阶段上下文", "action_context", _NEVER_DROP_TIER),
        _SectionSpec("_build_public_summary", "【场上记录】", "当前局公开事实", "public_record", _NEVER_DROP_TIER, True),
        _SectionSpec("_build_visible_state", "【辅助】", "可见世界状态", "public_record", _NEVER_DROP_TIER, True),
        _SectionSpec("_build_salience_events", "【辅助】", "关键事件", "public_record", _NEVER_DROP_TIER, True),
        _SectionSpec("_build_recent_transcript", "【场上记录】", "近期发言", "public_record", _NEVER_DROP_TIER, True),
        _SectionSpec("_build_persona", "【人格】", "人格设定", "style", _NEVER_DROP_TIER),
        _SectionSpec("_build_belief_state", "【辅助】", "我的判断", "private_reasoning", 0),
        _SectionSpec("_build_private_memory_hints", "【辅助】", "本局·私有记忆", "private_memory", 1),
        _SectionSpec("_build_learning_context", "【参考】", "跨局学习参考", "cross_game_reference", 2),
        _SectionSpec("_build_strategy_directive", "【策略指令】", "策略指令", "directive", _NEVER_DROP_TIER),
        _SectionSpec("_build_final_output_guard", "【硬约束】", "最终输出约束", "output_constraint", _NEVER_DROP_TIER),
    )
    _SECTION_SPEC_BY_NAME: dict[str, _SectionSpec] = {
        spec.builder_name: spec for spec in _USER_SECTION_SPECS
    }
    _NEVER_DROP: frozenset[str] = frozenset(
        spec.builder_name
        for spec in _USER_SECTION_SPECS
        if spec.drop_tier is _NEVER_DROP_TIER
    )
    _GROUNDING_SECTIONS: frozenset[str] = frozenset(
        spec.builder_name
        for spec in _USER_SECTION_SPECS
        if spec.public_record or spec.info_kind == "action_context"
    )
    _LOW_VALUE_SECTIONS: frozenset[str] = frozenset(
        spec.builder_name
        for spec in _USER_SECTION_SPECS
        if spec.drop_tier == 0
    )
    _SECTION_PRIORITIES: dict[str, str] = {
        spec.builder_name: spec.label
        for spec in _USER_SECTION_SPECS
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
        # P1-5: build the full prompt first, then enforce the global
        # token budget by dropping sections with the lowest ``drop_tier``.
        # Current-game grounding, persona, strategy, and the final
        # output guard are marked never-drop in _USER_SECTION_SPECS.
        parts: list[tuple[str, str]] = []
        # Boundary marker per s10: above = stable, below = dynamic.
        # Boundary marker + task prompt are always kept (they are not
        # sections with a priority label).
        parts.append(("", "=== DYNAMIC_BOUNDARY ==="))
        # P1-S3: each section is wrapped with a label so the LLM can
        # rank attention under tight token budgets. The label is
        # prepended at the section level
        # — internal sub-grouping (e.g., P0-S5 within strategy_directive)
        # is preserved.
        parts.append(("_build_phase_context", self._label_section("_build_phase_context", self._build_phase_context())))
        parts.append(("_build_public_summary", self._label_section("_build_public_summary", self._build_public_summary())))
        parts.append(("_build_visible_state", self._label_section("_build_visible_state", self._build_visible_state())))
        parts.append(("_build_salience_events", self._label_section("_build_salience_events", self._build_salience_events())))
        parts.append(("_build_recent_transcript", self._label_section("_build_recent_transcript", self._build_recent_transcript())))
        # P2-S10: persona (per-turn style/tone hint) lives in the user
        # message so it stays dynamic, but it follows current-game public
        # grounding to avoid style hints interrupting the public record chain.
        parts.append(("_build_persona", self._label_section("_build_persona", self._build_persona())))
        parts.append(("_build_belief_state", self._label_section("_build_belief_state", self._build_belief_state())))
        parts.append(("_build_private_memory_hints", self._label_section("_build_private_memory_hints", self._build_private_memory_hints())))
        parts.append(("_build_learning_context", self._label_section("_build_learning_context", self._build_learning_context())))
        parts.append(("_build_strategy_directive", self._label_section("_build_strategy_directive", self._build_strategy_directive())))
        # NEW-S04-A: skill_analysis_hints render path dropped. The
        # structured skill_tactical_advice is the single source of
        # truth (rendered inside strategy_directive's 【参考】 group).
        # The opaque-dict dual render is gone. _build_skill_analysis_hints
        # is no longer called from here.
        # P0-S6: retry hint must come AFTER task prompt and BEFORE the
        # output contract. Old order put retry BEFORE task, so the LLM
        # read "纠正提示..." and then got distracted by the task
        # description that followed — easy to miss the correction.
        # New order (task → retry → contract) makes the correction the
        # last thing the LLM sees before the output contract.
        # task prompt has no priority label — it's the action spec.
        parts.append(("", self._build_task_prompt()))
        parts.append(("_build_final_output_guard", self._label_section("_build_final_output_guard", self._build_final_output_guard(retry))))
        return self._enforce_budget(parts)

    def _enforce_budget(
        self,
        parts: list[tuple[str, str]],
    ) -> str:
        """Join parts with blank-line separator, then trim if over budget.

        P1-5: when the joined prompt exceeds ``_USER_PROMPT_BUDGET_CHARS``,
        drop sections by ``_USER_SECTION_SPECS.drop_tier`` until it fits.
        Sections with ``drop_tier is None``, the boundary marker, and the
        task prompt are never dropped.

        Implementation note (carries over from prior fix): ``drop_indices``
        here is the EXCLUSION set — indices still in the set are EXCLUDED
        from the joined prompt. The loop iterates the droppable list in
        priority order (lowest priority first) and removes (drops) one
        section at a time, shrinking the joined. The loop exits as soon
        as the joined fits under the budget.
        """
        # Fast path: the full prompt is already under budget.
        full_joined = "\n\n".join(p for _, p in parts if p)
        if len(full_joined) <= _USER_PROMPT_BUDGET_CHARS:
            return full_joined
        # Build the drop order from the section registry. Sections
        # absent from the registry or marked never-drop are skipped.
        droppable: list[tuple[int, int]] = []
        for idx, (name, _) in enumerate(parts):
            if not name:
                continue
            spec = self._SECTION_SPEC_BY_NAME.get(name)
            if spec is None or spec.drop_tier is None:
                continue
            droppable.append((spec.drop_tier, idx))
        # Sort by tier first, then keep original order within a tier
        # (stable sort). Note: this means within a tier, sections
        # earlier in the parts list (e.g. phase_context, belief_state)
        # are dropped before later ones (e.g. retry_hint, output
        # contract). That's the opposite of "drop from the end"
        # but it's deterministic and predictable.
        droppable.sort(key=lambda x: x[0])
        # AUDIT-2-02 (corrected for G-R4-15): maintain a running total
        # of the joined length and update it O(1) per drop instead of
        # re-joining the full prompt on every iteration. ``drop_indices``
        # here is the INCLUSION set — indices in the set are KEPT in
        # the joined prompt, indices not in the set are dropped. We
        # start with drop_indices = set() (nothing kept, full prompt
        # would just be empty) — no wait, that doesn't match either.
        #
        # Re-deriving from scratch: the full_joined is len(13548)
        # in the test scenario. We want to drop sections until total
        # fits under budget (6250). So drop_indices is the set of
        # indices we have DROPPED so far. We start with all parts
        # included (drop_indices = set()) and ADD to drop_indices
        # as we drop. ``joined = total`` starts at full_joined and
        # decreases.
        total = len(full_joined)
        dropped: set[int] = set()
        # Walk droppable in stable tier+order; DROP each section
        # whose label is in the matching tier, until the budget
        # is satisfied or droppable is exhausted. The 硬约束
        # filter is applied up-front so they never appear here.
        for tier, idx in droppable:
            if total <= _USER_PROMPT_BUDGET_CHARS:
                break
            if idx in dropped:
                continue
            # Only drop if the section still maps to the same tier.
            spec = self._SECTION_SPEC_BY_NAME.get(parts[idx][0])
            if spec is None or spec.drop_tier != tier:
                continue
            # Mark as dropped and update running total: SUBTRACT
            # the body + 2 chars (one separator for the part that
            # just left the joined string). Empty body means the
            # section was already filtered out of the joined, so
            # the only effect is on dropped; the running total is
            # unchanged.
            body = parts[idx][1]
            if body:
                dropped.add(idx)
                total -= len(body) + 2
            else:
                # Empty body contributes nothing to joined; we
                # still mark it as "processed" so we don't try to
                # drop it again on a future iteration (the
                # ``idx in dropped`` guard handles this).
                dropped.add(idx)
        # AUDIT-2-02: reconstruct the joined string ONCE in O(N)
        # from the (post-drop) ``dropped`` set. The previous
        # implementation re-joined the whole prompt on every
        # iteration — O(N²) for tight budgets.
        active = [
            p for i, (_, p) in enumerate(parts) if i not in dropped and p
        ]
        return "\n\n".join(active)

    def _build_phase_context(self) -> str:
        ctx = self.context
        lines = [f"当前阶段: {ctx.phase}"]
        if ctx.legal_actions:
            lines.append(f"可用操作: {[a.value for a in ctx.legal_actions]}")
        if ctx.legal_targets:
            lines.append(f"可选目标: {ctx.legal_targets}")
        # Mandatory vote hints
        output_mode = self._select_output_mode()
        is_vote_context = ctx.task_type == TaskType.VOTE
        if ctx.legal_actions and is_vote_context:
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
        # Phase-1 audit: wrap the caveat in ``---`` markers so the
        # LLM cannot mistake the warning for a JSON key or treat it
        # as a hint payload entry.  Without the marker, the LLM has
        # been observed to fold caveat text into the JSON dict
        # (e.g. parsing "私有记忆" as a key).
        if caveat:
            caveat_block = f"---\n{caveat}\n---\n"
        else:
            caveat_block = ""
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
        # must only see title plus prompt-safe V2 tactical frame fields.
        # Audit data (summary, key_decisions, relevance, quality, source,
        # visibility, display annotation) belongs in the audit log, not
        # the LLM context window.
        #
        # G-R4-10: explicit ``type == "rag_hit"`` filter so a future
        # code path (or a test, or a manual debug call) that injects
        # auxiliary metadata (salience events, profile snapshots,
        # etc.) into ``ctx.rag_hints`` cannot leak into the prompt.
        # The previous code at runtime/context.py:231 used
        # ``if item.get("type") != "rag_hit"`` to *retain* non-rag
        # items, which is brittle — a stray non-rag item persists
        # across turns and the renderer would happily process it.
        # The prompt-side filter is explicit, defensive, and matches
        # the slim renderer's expectation that every line carries
        # the ``rag_hit`` discriminator.
        rag_only = [
            item for item in ctx.rag_hints
            if isinstance(item, dict) and item.get("type") == "rag_hit"
        ]
        if not rag_only:
            return ""
        # P2-6: cap on total hits uses the shared live-prompt
        # constant from rag.prompt_renderer (one source of truth for
        # the 3 retriever / slim-renderer / prompt-builder caps).
        from werewolf_agent.rag.prompt_renderer import RAG_LIVE_PROMPT_CAP
        slim_items = self._slim_rag_hint_items(rag_only[:RAG_LIVE_PROMPT_CAP])
        # P0-G3: hard-constraint prefix MUST come before the case
        # payload. Without this the LLM has been observed to parrot
        # case-specific player IDs (e.g., p04 / p09 in seed cases) as
        # if they were this game's player IDs, which is information
        # leakage and tactical error.
        # P2-11: extended the head warning to cover TACTIC reuse in
        # addition to player-ID reuse.  Pre-fix the warning only
        # flagged ID leakage, but the LLM was also observed to
        # wholesale-copy case tactics (e.g. "case used anti-herd and
        # won, so do anti-herd this game") regardless of local
        # context.  Add explicit anti-tactic-reuse framing.
        warning = (
            "⚠️ RAG 案例中的玩家 ID 与战术选择仅供启发；"
            "本局的玩家 ID、票型、遗言均与案例无关；"
            "不得直接套用案例中具体玩家的动作、票型或决策链。\n"
        )
        case_cards = self._render_rag_hint_cards(slim_items)
        # R19: a tail reminder after the case cards re-anchors the
        # model at the end of the section. The head warning only sets the
        # "do not parrot" frame at the start; without a tail the
        # LLM can still walk the section and treat the cards as a
        # hard assertion rather than reference material. Tail text
        # matches the head framing so the LLM sees a consistent
        # "this is reference only" message before it generates.
        tail = "（以上案例仅供参考，不得作为本局事实或硬性指令。）"
        if "…已截断" in case_cards:
            tail = tail + "（部分字段已截断，案例未完整呈现。）"
        return (
            "知识库提示: 知识库提示不是当前局事实，只能作为玩法经验和案例参考。\n"
            + warning
            + case_cards
            + "\n"
            + tail
        )

    def _build_learning_context(self) -> str:
        header = (
            "跨局学习参考: 以下内容只是历史经验与自我校准，"
            "不代表本局任何玩家真实身份。"
        )
        parts: list[tuple[str, int, str]] = []
        error_pattern = self._build_error_pattern_hint()
        if error_pattern:
            parts.append(("error_pattern", 4, error_pattern))
        reflection = self._build_reflection_memory_hints()
        if reflection:
            parts.append(("reflection", 3, reflection))
        profile = self._build_profile_memory_hint()
        if profile:
            parts.append(("profile", 2, profile))
        cognition = self._build_cognition_matrix_hint()
        if cognition:
            parts.append(("cognition", 1, cognition))
        rag = self._build_rag_hints()
        if rag:
            parts.append(("rag", 0, rag))
        if not parts:
            return ""

        active = list(parts)

        def render(items: list[tuple[str, int, str]]) -> str:
            return header + "\n\n" + "\n\n".join(text for _, _, text in items)

        while active and len(render(active)) > _MAX_LEARNING_CONTEXT_CHARS:
            lowest_priority = min(priority for _, priority, _ in active)
            for idx in range(len(active) - 1, -1, -1):
                if active[idx][1] == lowest_priority:
                    del active[idx]
                    break

        return render(active) if active else ""

    @staticmethod
    def _slim_rag_hint_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop audit-only fields from RAG hint items in the live prompt.

        Mirrors :func:`werewolf_agent.rag.prompt_renderer.hits_to_prompt_lines`
        but operates on already-rendered dicts (so it works even when
        ``ctx.rag_hints`` was populated by a test or a non-default
        code path that bypassed :class:`RAGKnowledgeService`).

        RAG V2 prompt shape is title plus prompt-safe tactical frame
        fields. Legacy dicts are routed through the shared tactical
        helper, which supplies a conservative fallback frame without
        exposing raw ``summary`` / ``key_decisions`` fields here.
        """
        from werewolf_agent.rag.tactical_text import prompt_safe_tactical_frame_dict

        slim: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            frame = prompt_safe_tactical_frame_dict(item)
            slim.append({
                "title": PlayerPromptBuilder._clean_prompt_text(
                    item.get("title", ""),
                    max_chars=_MAX_LEARNING_TEXT_CHARS,
                ),
                "situation_signature": PlayerPromptBuilder._clean_prompt_text(
                    frame["situation_signature"],
                    max_chars=_MAX_RAG_TEXT_CHARS,
                ),
                "transferable_lesson": PlayerPromptBuilder._clean_prompt_text(
                    frame["transferable_lesson"],
                    max_chars=_MAX_RAG_TEXT_CHARS,
                ),
                "applicability": [
                    PlayerPromptBuilder._clean_prompt_text(
                        value,
                        max_chars=_MAX_RAG_TEXT_CHARS,
                    )
                    for value in frame["applicability"]
                ],
                "counter_signals": [
                    PlayerPromptBuilder._clean_prompt_text(
                        value,
                        max_chars=_MAX_RAG_TEXT_CHARS,
                    )
                    for value in frame["counter_signals"]
                ],
                "recommended_use": PlayerPromptBuilder._clean_prompt_text(
                    frame["recommended_use"],
                    max_chars=_MAX_RAG_TEXT_CHARS,
                ),
                "misuse_risk": PlayerPromptBuilder._clean_prompt_text(
                    frame["misuse_risk"],
                    max_chars=_MAX_RAG_TEXT_CHARS,
                ),
            })
        return slim

    @staticmethod
    def _render_rag_hint_cards(items: list[dict[str, Any]]) -> str:
        """Render prompt-safe RAG items as readable low-priority cards."""
        def join_values(value: Any, *, fallback: str) -> str:
            if isinstance(value, list):
                text = "；".join(str(part).strip() for part in value if str(part).strip())
                return text or fallback
            text = str(value or "").strip()
            return text or fallback

        cards: list[str] = []
        for idx, item in enumerate(items, start=1):
            title = str(item.get("title") or "未命名案例")
            cards.append(
                f"案例 {idx}：{title}\n"
                f"- 适用局面：{join_values(item.get('situation_signature'), fallback='缺少局面描述。')}\n"
                f"- 可迁移原则：{join_values(item.get('transferable_lesson'), fallback='仅作为谨慎参考。')}\n"
                f"- 适用条件：{join_values(item.get('applicability'), fallback='仅当本局公开事实与案例局面相似时参考。')}\n"
                f"- 不适用信号：{join_values(item.get('counter_signals'), fallback='若当前局面不匹配则不要套用。')}\n"
                f"- 本局参考方式：{join_values(item.get('recommended_use'), fallback='作为参考，不作为直接指令。')}\n"
                f"- 误用风险：{join_values(item.get('misuse_risk'), fallback='过度套用可能误导判断。')}"
            )

        return "\n\n".join(cards)

    def _build_reflection_memory_hints(self) -> str:
        ctx = self.context
        if not ctx.reflection_memory_hints:
            return ""
        hints = self._slim_reflection_hints(
            ctx.reflection_memory_hints[:REFLECTION_CARD_BUDGET]
        )
        if not hints:
            return ""
        if any("recommended_action" in hint or "trigger_signals" in hint for hint in hints):
            return (
                "跨局反思记忆: 以下是你过往对局后的经验总结，不代表本局任何玩家真实身份。\n"
                "历史玩家 ID、身份真相和决策链不得映射到本局玩家。\n"
                + self._render_reflection_hint_cards(hints)
            )
        return (
            "跨局反思记忆: 以下是你过往对局后的经验总结，不代表本局任何玩家真实身份。\n"
            "历史玩家 ID、身份真相和决策链不得映射到本局玩家。\n"
            + self._compact_json(hints)
        )

    @staticmethod
    def _render_reflection_hint_cards(hints: list[dict[str, Any]]) -> str:
        def join_values(value: Any, *, fallback: str) -> str:
            if isinstance(value, list):
                text = "；".join(str(part).strip() for part in value if str(part).strip())
                return text or fallback
            text = str(value or "").strip()
            return text or fallback

        cards: list[str] = []
        for idx, hint in enumerate(hints, start=1):
            title = join_values(hint.get("theme"), fallback="历史反思")
            cards.append(
                f"\n\n反思 {idx}: {title}\n"
                f"- 触发信号: {join_values(hint.get('trigger_signals'), fallback='仅当本局公开事实相似时参考。')}\n"
                f"- 历史教训: {join_values(hint.get('lesson') or hint.get('summary') or hint.get('text'), fallback='历史经验仅供参考。')}\n"
                f"- 本局做法: {join_values(hint.get('recommended_action') or hint.get('actionable_advice'), fallback='先核验本局公开证据，再决定是否采用。')}\n"
                f"- 误用风险: {join_values(hint.get('misuse_risk'), fallback='不要把历史经验直接映射到本局玩家身份。')}"
            )
        return "".join(cards)

    def _build_error_pattern_hint(self) -> str:
        """reflect-cross-1: 跨局错误模式聚合,顶部强提示 LLM 自我修正。

        当 LLM 拿到"你历史最常犯的错误是 X"这种聚合信号,比单条反思更
        容易驱动行为改变。同时"保留优点 N 次"提醒 LLM 不要只学错误。
        """
        ctx = self.context
        ep = ctx.error_pattern_hint
        if not ep or ep.get("total_reflections", 0) == 0:
            return ""
        parts: list[str] = ["【跨局错误模式(基于你过往反思)】"]
        top = ep.get("top_mistakes") or []
        if top:
            mistake_strs = [
                f"{self._normalize_error_category(cat)}({count}次)"
                for cat, count in top
            ]
            parts.append(f"你最常犯的 2 类错误: {'、'.join(mistake_strs)}。")
        strengths = [
            self._clean_prompt_text(label, max_chars=40)
            for label in (ep.get("preserved_strength_labels") or [])
            if str(label or "").strip()
        ][:2]
        preserved = ep.get("preserved_strength_count", 0)
        if strengths:
            parts.append(f"可复用优点: {'、'.join(strengths)}。")
        elif preserved:
            parts.append(f"过去 {preserved} 局反思中出现过可保留优点,本局先核验再复用。")
        same = ep.get("same_role_reflections", 0)
        if same:
            parts.append(
                f"其中 {same} 局你拿过当前角色({ep.get('current_role', '?')}),"
                f"历史经验对当前角色特别相关。"
            )
        return "\n".join(parts)

    def _build_profile_memory_hint(self) -> str:
        ctx = self.context
        if not ctx.profile_memory_hint:
            return ""
        allowed_fields = (
            "games_played",
            "current_role",
            "current_role_games",
            "current_role_win_rate_pct",
            "win_rate_confidence",
            "summary",
        )
        live_hint = {}
        for key in allowed_fields:
            if key not in ctx.profile_memory_hint:
                continue
            value = ctx.profile_memory_hint[key]
            if isinstance(value, str):
                value = self._clean_prompt_text(
                    value,
                    max_chars=_MAX_LEARNING_TEXT_CHARS,
                )
            live_hint[key] = value
        if not live_hint:
            return ""
        return (
            "历史角色经验: 以下仅是样本量与当前角色经历，不代表本局能力高低。\n"
            + self._compact_json(live_hint)
        )

    def _build_cognition_matrix_hint(self) -> str:
        ctx = self.context
        if not ctx.cognition_matrix_hint:
            return ""
        suspects = ctx.cognition_matrix_hint.get("suspects")
        trusted = ctx.cognition_matrix_hint.get("trusted")
        if suspects is None and trusted is None:
            suspects = [
                value for value in ctx.cognition_matrix_hint.values()
                if isinstance(value, dict)
                and (
                    value.get("faction_lean") in {"wolf", "wolf_lean", "werewolf"}
                    or float(value.get("trust", 0.5) or 0.5) < 0.4
                )
            ]
            trusted = [
                value for value in ctx.cognition_matrix_hint.values()
                if isinstance(value, dict)
                and (
                    value.get("faction_lean") in {"good", "good_lean", "villager"}
                    or float(value.get("trust", 0.5) or 0.5) > 0.6
                )
            ]
        suspects = suspects or []
        trusted = trusted or []
        live_hint = {
            "tracked_suspect_count": len(suspects),
            "tracked_trusted_count": len(trusted),
            "open_question_count": sum(
                len(item.get("open_questions") or [])
                for item in list(suspects) + list(trusted)
                if isinstance(item, dict)
            ),
            "evidence_anchor_count": sum(
                len(item.get("key_evidence") or [])
                for item in list(suspects) + list(trusted)
                if isinstance(item, dict)
            ),
            "calibration_warning": "只复核证据充分性，不复用历史玩家名单。",
        }
        return (
            "认知校准摘要: 历史矩阵只用于提醒你重新核验判断，不提供本局嫌疑名单。\n"
            + self._compact_json(live_hint)
        )

    def _build_strategy_directive(self) -> str:
        """Render strategy_directive split into 3 priority sections.

        P0-S5: LLM previously saw a single flat JSON block with 20+ keys
        and no priority signal — had to guess hard vs soft. Now keys are
        grouped into 【硬约束】 (MUST), 【建议】 (SHOULD), 【参考】
        (REFERENCE). Unknown keys fall through to 参考 for forward-compat.

        NEW-R4-P1-1: ``skill_tactical_advice`` is a structured list of
        ``[{skill, advice, confidence}, ...]`` (S-07 contract). Running
        the whole section through ``_compact_json`` wraps it in a
        ``{"skill_tactical_advice":[{...}, ...]}`` JSON envelope that
        the LLM has to parse before reading the advice — wasted tokens
        and a parse-failure surface. We render that key with a
        dedicated human-readable bullet renderer:
        ``- [skill_name/confidence] advice_text``.
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
            parts.append(
                f"{header} {label}\n" + self._render_strategy_section(section)
            )
        return "\n\n".join(parts)

    def _render_strategy_section(self, section: dict[str, Any]) -> str:
        """Render a single strategy_directive section.

        NEW-R4-P1-1: if the section contains ``skill_tactical_advice``
        (a structured list per S-07), use a dedicated bullet renderer
        for that key and render the remaining keys via ``_compact_json``.
        This avoids the raw JSON envelope the LLM would otherwise have
        to parse before reading the advice.
        """
        advice = section.get("skill_tactical_advice")
        if not advice or not isinstance(advice, list):
            return self._compact_json(section)
        # Render the advice as a human-readable bullet list. Other keys
        # in the section still go through _compact_json — only the
        # structured advice is humanized.
        bullets = self._render_skill_tactical_advice(advice)
        rest = {k: v for k, v in section.items() if k != "skill_tactical_advice"}
        if not rest:
            return bullets
        return bullets + "\n" + self._compact_json(rest)

    @staticmethod
    def _render_skill_tactical_advice(advice: list[Any]) -> str:
        """Render the S-07 ``skill_tactical_advice`` list as bullets.

        Format: ``- [skill_name/confidence] advice_text`` (one per line).
        Entries that are not dicts or that lack the expected keys are
        rendered defensively via ``json.dumps`` so a single bad entry
        doesn't break the whole list.

        NEW-R4-P1-1: bypasses ``_compact_json`` so the LLM sees plain
        text rather than a JSON envelope.
        """
        lines: list[str] = ["技能战术建议:"]
        rendered = advice[:_MAX_SKILL_TACTICAL_ADVICE_ITEMS]
        for entry in rendered:
            if not isinstance(entry, dict):
                text = PlayerPromptBuilder._clean_prompt_text(
                    json.dumps(entry, ensure_ascii=False),
                    max_chars=_MAX_SKILL_TACTICAL_ADVICE_CHARS,
                )
                lines.append(f"- {text}")
                continue
            skill = PlayerPromptBuilder._clean_prompt_text(
                entry.get("skill", ""),
                max_chars=40,
            )
            conf = entry.get("confidence", "")
            text = PlayerPromptBuilder._clean_prompt_text(
                entry.get("advice", ""),
                max_chars=_MAX_SKILL_TACTICAL_ADVICE_CHARS,
            )
            try:
                conf_str = f"{float(conf):.2f}"
            except (TypeError, ValueError):
                conf_str = str(conf)
            frame_lines = [f"- [{skill}/{conf_str}] {text}"]
            situation = PlayerPromptBuilder._clean_prompt_text(
                entry.get("situation_signature", ""),
                max_chars=120,
            )
            recommended_use = PlayerPromptBuilder._clean_prompt_text(
                entry.get("recommended_use", ""),
                max_chars=_MAX_SKILL_TACTICAL_ADVICE_CHARS,
            )
            risk_alerts = [
                PlayerPromptBuilder._clean_prompt_text(item, max_chars=100)
                for item in list(entry.get("risk_alerts") or [])[:2]
            ]
            counter_signals = [
                PlayerPromptBuilder._clean_prompt_text(item, max_chars=100)
                for item in list(entry.get("counter_signals") or [])[:2]
            ]
            forbidden_use = PlayerPromptBuilder._clean_prompt_text(
                entry.get("forbidden_use", ""),
                max_chars=120,
            )
            if situation:
                frame_lines.append(f"  适用局面：{situation}")
            if recommended_use:
                frame_lines.append(f"  本轮建议：{recommended_use}")
            if risk_alerts:
                frame_lines.append(f"  风险：{'；'.join(risk_alerts)}")
            if counter_signals:
                frame_lines.append(f"  不适用信号：{'；'.join(counter_signals)}")
            if forbidden_use:
                frame_lines.append(f"  禁止套用：{forbidden_use}")
            lines.extend(frame_lines)
        omitted = len(advice) - len(rendered)
        if omitted > 0:
            lines.append(f"- 其余 {omitted} 条技能战术建议已省略。")
        return "\n".join(lines)

    def _build_skill_analysis_hints(self) -> str:
        # NEW-S04-A: this method is dead code. The dual-render path
        # was removed — skill_analysis_hints is no longer populated.
        # Kept (no-op) only to avoid removing a public method on a
        # shared class; remove in a follow-up cleanup if desired.
        return ""

    def _build_persona(self) -> str:
        ctx = self.context
        if not ctx.persona_snapshot:
            return ""
        sanitized_snapshot = sanitize_persona_snapshot(
            ctx.persona_snapshot,
            own_role=ctx.own_role or "",
            task_type=ctx.task_type.value,
        )
        lines = ["人格设定:"]
        text_fields = (
            ("personality", "人格核心"),
            ("speech_style", "表达风格"),
            ("task_style", "任务风格"),
            ("tone", "语气"),
        )
        for key, label in text_fields:
            value = sanitized_snapshot.get(key)
            if value:
                lines.append(f"- {label}: {self._clean_prompt_text(value)}")
        effective = self._slim_numeric_params(
            sanitized_snapshot.get("effective_params")
        )
        if effective:
            lines.append(f"- 稳定倾向: {self._compact_json(effective)}")
        adjustments = self._slim_numeric_params(
            sanitized_snapshot.get("dynamic_adjustments")
        )
        if adjustments:
            lines.append(f"- 本轮调整: {self._compact_json(adjustments)}")
        if len(lines) == 1:
            return ""
        lines.append("注意: 人格只影响表达和决策风格，不代表身份信息、公开事实或固定战术。")
        return "\n".join(lines)

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
            output_mode = self._select_output_mode()
            can_emit_no_action = (
                ActionType.NO_ACTION in self.context.legal_actions
                and output_mode == OutputMode.FULL_ACTION
            )
            if can_emit_no_action:
                lines.append(
                    "重要：如果你已经超时，请直接返回 no_action"
                    "（action_type='no_action', target_id=null, "
                    "reason='timeout - safe no-op'），"
                    "不要再尝试长推理或构造JSON。"
                )
            elif self.context.legal_targets:
                first_target = self.context.legal_targets[0]
                lines.append(
                    f"重要：如果你已经超时，请选择一个合法目标（例如 {first_target}）"
                    "并提交结构化JSON，不要再尝试长推理。"
                )
        return "\n".join(lines)

    def _build_final_output_guard(self, retry: RetryInfo) -> str:
        parts: list[str] = []
        retry_hint = self._build_retry_hint(retry).strip()
        if retry_hint:
            parts.append(retry_hint)
        contract = self._build_strict_output_contract().strip()
        if contract:
            parts.append(contract)
        if not parts:
            return ""
        return "最终输出约束（必须遵守）：\n" + "\n\n".join(parts)

    def _build_task_prompt(self) -> str:
        """Task-specific prompt: choice enum, speech intent, or examples."""
        output_mode = self._select_output_mode()
        if output_mode == OutputMode.TARGET_CHOICE:
            return self._format_choice_prompt()
        if output_mode == OutputMode.SPEECH_INTENT:
            return self._format_speech_intent_prompt()
        return self._format_examples()

    def _example_target(self, fallback: str = "pXX") -> str:
        return self.context.legal_targets[0] if self.context.legal_targets else fallback

    def _format_examples(self) -> str:
        ctx = self.context
        parts: list[str] = []
        # P1-G3223805846-4: 在所有示例块之前先告诉 LLM 顶层字段名必须是
        # ``action_type`` 而不是 ``intent``。 这条 guard 覆盖下面所有
        # 路径 (wolf_kill / wolf_no_kill / sheriff_register /
        # sheriff_withdraw / no_action / hunter_shot / use_antidote /
        # use_poison / badge_transfer / badge_tear / choose_master /
        # speech / vote), 在示例渲染前一次性注入,比每个分支都改更稳。
        parts.append(_ACTION_TYPE_GUARD)
        # P2 (post-review-v2): vote 阶段 FULL_ACTION 路径也注入
        # _VOTE_REASON_PRIVACY_GUARD。 之前该 guard 只在
        # ``_format_choice_prompt`` (单动作 [VOTE] → TARGET_CHOICE)
        # 注入；当 legal_actions 含 VOTE + 其他动作 (例如 VOTE +
        # NO_ACTION) 时路由到 FULL_ACTION 走 ``_format_examples``，
        # 隐私 guard 缺失，LLM 在 reason 字段会写入私视角表述
        # (g_3223805846 复现)。 与 _format_choice_prompt 路径对齐。
        if ctx.task_type == TaskType.VOTE:
            parts.append(_VOTE_REASON_PRIVACY_GUARD)
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
            # P2-3: explicit sheriff_withdraw example so the LLM has a
            # template to copy when the player decides to pull out of
            # the sheriff race. Game trace g_3528592081 action 41 (p05)
            # showed the LLM emitting `sheriff_register` when it meant
            # to withdraw — adding this example gives the model the
            # right action_type to pattern-match.
            if ActionType.SHERIFF_WITHDRAW in ctx.legal_actions:
                parts.append("示例输出（退警场景）：")
                parts.append(
                    '{"action_type": "sheriff_withdraw", "target_id": null, '
                    '"speech": "我退水，把警徽给更需要的人。", '
                    '"reason": "评估后觉得不适合继续竞选", "confidence": 0.6}'
                )
            if ActionType.NO_ACTION in ctx.legal_actions:
                parts.append("示例输出（不上警场景）：")
                parts.append(
                    '{"action_type": "no_action", "target_id": null, '
                    '"speech": "我不上警，先听警上发言再判断。", '
                    '"reason": "当前信息不足，先观察警上格局", "confidence": 0.6}'
                )
        # D4-5: 6 example blocks for action types that the legacy
        # implementation didn't render. The LLM has no template to
        # copy without these — game trace g_3528592081 showed
        # malformed hunter_shot / use_poison payloads that took
        # multiple retries to repair. Priority order matches the
        # natural night-action flow: hunter first (it interrupts
        # the day), then witch actions, then sheriff badge actions,
        # then hybrid's master choice (only on the first night).
        elif ActionType.HUNTER_SHOT in ctx.legal_actions:
            example_target = self._example_target()
            parts.append("示例输出（猎人开枪场景）：")
            parts.append(
                f'{{"action_type": "hunter_shot", "target_id": "{example_target}", '
                '"speech": "", '
                f'"reason": "我带走最可疑的{example_target}", "confidence": 0.7}}'
            )
        elif ActionType.USE_ANTIDOTE in ctx.legal_actions:
            example_target = self._example_target()
            parts.append("示例输出（女巫解药场景）：")
            parts.append(
                f'{{"action_type": "use_antidote", "target_id": "{example_target}", '
                '"speech": "", '
                f'"reason": "救下被刀的{example_target}", "confidence": 0.7}}'
            )
        elif ActionType.USE_POISON in ctx.legal_actions:
            example_target = self._example_target()
            parts.append("示例输出（女巫毒药场景）：")
            parts.append(
                f'{{"action_type": "use_poison", "target_id": "{example_target}", '
                '"speech": "", '
                f'"reason": "毒死确认的狼人{example_target}", "confidence": 0.7}}'
            )
        elif ActionType.BADGE_TRANSFER in ctx.legal_actions:
            example_target = self._example_target()
            parts.append("示例输出（警徽移交场景）：")
            parts.append(
                f'{{"action_type": "badge_transfer", "target_id": "{example_target}", '
                '"speech": "", '
                f'"reason": "把警徽传给更可信的{example_target}", "confidence": 0.7}}'
            )
        elif ActionType.BADGE_TEAR in ctx.legal_actions:
            parts.append("示例输出（撕毁警徽场景）：")
            parts.append(
                '{"action_type": "badge_tear", "target_id": null, '
                '"speech": "", '
                '"reason": "本局无合适人选，撕毁警徽", "confidence": 0.7}'
            )
        elif ActionType.CHOOSE_MASTER in ctx.legal_actions:
            example_target = self._example_target()
            parts.append("示例输出（混血儿选主场景）：")
            parts.append(
                f'{{"action_type": "choose_master", "target_id": "{example_target}", '
                '"speech": "", '
                f'"reason": "选择{example_target}作为我的主人", "confidence": 0.7}}'
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
                # P1-8: only a seer has a check. Non-seer roles don't —
                # they are standing with a (claimed) seer, so their
                # vote_basis is "seer_siding" not "seer_check".
                vote_basis = "seer_check"
            else:
                # C4: vote 段示例 standing_with_seer 用 pXX 占位符 —
                # 占位符标记此处填的是"本局真实预言家 ID"，由 LLM
                # 替换为当前局实际的预言家 ID（不应硬编码 p03，避免
                # LLM 把示例里的 ID 直接抄到输出里）。
                vote_standing_with_seer = "pXX"
                vote_basis = "seer_siding"
            # P1 (post-review-v2): vote 段示例所有 player ID 一律改占位符 pXX
            # — 任何出现在示例里的 p0X 都可能被 LLM 误抄到当前局输出。
            # LLM 看到 pXX 后应替换为"本局真实玩家 ID"，示例 ID 不是
            # 真实参考。覆盖 target_id、standing_with_seer、pressure_target
            # 三个 JSON 字段以及 reason / suspect_reason / not_voting_reason /
            # private_reason 四个文本字段里的 p0X 引用。
            parts.append("示例输出（投票场景）：")
            parts.append('{"action_type": "vote", "target_id": "pXX", '
                         '"speech": "", '
                         '"reason": "公开理由：pXX发言可疑", '
                         '"seer_stance": "trust", '
                         f'"vote_basis": "{vote_basis}", '
                         f'"standing_with_seer": "{vote_standing_with_seer}", '
                         '"suspect_reason": "pXX没有回应pXX的查杀逻辑，发言前后不一致", '
                         '"not_voting_reason": "pXX虽然被踩，但目前没有明确查验或票型证据", '
                         '"private_reason": "心里活动：我更信pXX的预言家线，pXX像狼队抗推失败后的防守位，所以投pXX。", '
                         '"confidence": 0.8, '
                         f'"private_intent": {{"true_role": "{example_role}", '
                         f'"faction_goal": "{vote_example_goal}", "claimed_view": "{vote_example_view}", '
                         '"pressure_target": "pXX", "risk_flags": []}}')
        return "\n".join(parts)

    def _build_strict_output_contract(self) -> str:
        """Per-turn output contract — adapts to task type.

        M2-3: the VOTE branch sources its 9-field list from the
        module-level ``_OUTPUT_SCHEMA_VOTE_FIELDS`` constant so the
        per-turn contract cannot drift from
        ``_build_output_contract`` (the system prompt's stable
        advertisement).  The non-vote TARGET_CHOICE branch (3
        fields: ``choice``, ``reason``, ``confidence``) and the
        SPEECH_INTENT branch (uses ``intent`` instead of
        ``action_type``) keep literal field lists because they are
        *different schemas*, not subsets of the VOTE/SKILL/SPEECH
        constants.  The full-action (SKILL) branch inherits its
        schema from the same constant the system prompt uses
        (``_OUTPUT_SCHEMA_SKILL_FIELDS``) — the per-turn branch
        only adds the *per-turn* rules (legal_actions,
        legal_targets, vote audit fields) per P1-4, but those
        per-turn rules reference the same underlying schema.
        """
        ctx = self.context
        output_mode = self._select_output_mode()
        legal_actions = [a.value for a in ctx.legal_actions]
        legal_targets = list(ctx.legal_targets)
        contract = ActionContract.build(
            output_mode=output_mode,
            task_type=ctx.task_type,
            legal_actions=ctx.legal_actions,
            legal_targets=ctx.legal_targets,
        )

        if output_mode == OutputMode.TARGET_CHOICE:
            output_fields = "、".join(contract.required_fields)
            lines = [
                "",
                "最终输出协议（必须遵守）：",
                "1. 只输出一个choice决策JSON对象；不要输出分析过程、解释、Markdown或多余文本。",
                "2. JSON必须以{开头、以}结尾，且只能有一个对象。",
                f"3. 最终输出字段：{output_fields}。",
                "4. choice只能取上方候选枚举中的字母，不要直接编写target_id。",
            ]
            if ctx.task_type == TaskType.VOTE:
                lines.append(
                    f"5. 投票还必须包含{'、'.join(_VOTE_AUDIT_FIELDS)}，理由字段不能写「未说明」。"
                )
            lines.append("现在提交行动。")
            return "\n".join(lines)

        if output_mode == OutputMode.SPEECH_INTENT:
            output_fields = "、".join(contract.required_fields)
            lines = [
                "",
                "最终输出协议（必须遵守）：",
                "1. 只输出一个发言意图JSON对象；不要输出分析过程、解释、Markdown或多余文本。",
                "2. JSON必须以{开头、以}结尾，且只能有一个对象。",
                f"3. 最终输出字段：{output_fields}。",
                "4. target_id没有目标时必须是null，不要写字符串\"null\"。",
            ]
            if legal_targets:
                lines.append(f"5. target_id只能取这些玩家之一或null：{legal_targets}。")
            lines.append("现在提交行动。")
            return "\n".join(lines)

        # Full-action (SKILL) branch.  P1-4 deliberately removed the
        # field-list duplication here — the system prompt's stable
        # ``_build_output_contract`` (which sources its field list
        # from ``_OUTPUT_SCHEMA_SKILL_FIELDS`` per M2-3) already
        # advertises the SKILL schema, so this per-turn branch only
        # adds the *per-turn* rules (legal_actions, legal_targets,
        # vote audit fields).
        lines = [
            "",
            "最终输出协议（必须遵守）：",
            "1. 首选 submit_player_action 工具调用提交结构化参数。",
            "2. 如果当前模型无法工具调用，只输出一个JSON对象；不要输出分析过程、解释、Markdown或多余文本。",
            "3. JSON必须以{开头、以}结尾，且只能有一个对象。",
            "4. target_id没有目标时必须是null，不要写字符串\"null\"。",
        ]
        if legal_actions:
            lines.append(f"5. action_type只能取：{legal_actions}。")
        if legal_targets:
            lines.append(f"6. target_id只能取这些玩家之一或null：{legal_targets}。")
        if ctx.task_type == TaskType.VOTE:
            lines.append(
                f"7. 投票还必须包含{'、'.join(_VOTE_AUDIT_FIELDS)}，理由字段不能写「未说明」。"
            )
        lines.append("现在提交行动。")
        return "\n".join(lines)

    # ── Choice pipeline helpers ──

    def _format_choice_prompt(self) -> str:
        ctx = self.context
        is_vote = ctx.task_type == TaskType.VOTE
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
            # D4-2: branch the example's vote_basis by own_role. Only a
            # seer has a check of their own — non-seer roles are
            # siding with a (claimed) seer. Mirrors the P1-8 fix in
            # `_format_examples`. Without this, a non-seer villager
            # voting via the TARGET_CHOICE pipeline would copy
            # ``vote_basis="seer_check"`` into the audit log and
            # fabricate a non-existent basis (game trace
            # g_3528592081 observed this leak).
            role = ctx.own_role or "villager"
            choice_vote_basis = "seer_check" if role == "seer" else "seer_siding"
            example = (
                '{"choice":"A","reason":"投票公开理由",'
                '"seer_stance":"trust",'
                f'"vote_basis":"{choice_vote_basis}",'
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
        # P0-G3223805846-8: vote 阶段的 ``reason`` 字段是公开可见的，
        # 注入隐私硬约束到 prompt 开头以保证 LLM 不会写入私视角表述。
        # 仅在 vote 阶段注入；其他 choice（wolf_kill/poison/...）不写
        # reason 公开字段，不需要此 guard。
        if is_vote:
            return _VOTE_REASON_PRIVACY_GUARD + "\n".join(lines)
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
        return select_output_mode(
            legal_actions=ctx.legal_actions,
            legal_targets=ctx.legal_targets,
            task_type=ctx.task_type,
            speech_intent_tasks=_SPEECH_INTENT_TASKS,
        )

    def _uses_choice_pipeline(self) -> bool:
        return self._select_output_mode() == OutputMode.TARGET_CHOICE

    def _uses_speech_intent_pipeline(self) -> bool:
        return self._select_output_mode() == OutputMode.SPEECH_INTENT

    @staticmethod
    def _clean_prompt_text(
        value: Any,
        *,
        max_chars: int = _MAX_PERSONA_LINE_CHARS,
    ) -> str:
        text = str(value or "").strip()
        text = _PLAYER_ID_RE.sub("历史玩家", text)
        text = text.replace("\r", " ").replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        if len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 6)] + "…已截断"

    @staticmethod
    def _slim_numeric_params(value: Any) -> dict[str, float]:
        if not isinstance(value, dict):
            return {}
        slim: dict[str, float] = {}
        for key, raw in value.items():
            if key == "deception_skill" or key.endswith("_rank"):
                continue
            if not isinstance(raw, (int, float)):
                continue
            slim[str(key)] = round(float(raw), 2)
        return slim

    @classmethod
    def _slim_reflection_hints(cls, hints: list[dict[str, Any]]) -> list[dict[str, Any]]:
        allowed_fields = (
            "role",
            "result",
            "theme",
            "summary",
            "text",
            "lesson",
            "trigger_signals",
            "recommended_action",
            "misuse_risk",
            "actionable_advice",
        )
        slimmed: list[dict[str, Any]] = []
        for item in hints:
            if not isinstance(item, dict):
                continue
            slim: dict[str, Any] = {}
            for key in allowed_fields:
                if key not in item:
                    continue
                value = item[key]
                if isinstance(value, str):
                    value = cls._clean_prompt_text(
                        value,
                        max_chars=_MAX_LEARNING_TEXT_CHARS,
                    )
                elif isinstance(value, list):
                    value = [
                        cls._clean_prompt_text(
                            part,
                            max_chars=_MAX_LEARNING_TEXT_CHARS,
                        )
                        for part in value
                        if str(part or "").strip()
                    ][:4]
                slim[key] = value
            if slim:
                slimmed.append(slim)
        return slimmed

    @staticmethod
    def _normalize_error_category(category: Any) -> str:
        raw = str(category or "").strip().lower()
        mapping = {
            "vote_mistake": "投票错误",
            "info_miss": "漏读信息",
            "role_execution": "角色执行",
            "speech_quality": "发言质量",
            "format_error": "格式错误",
            "parse_error": "格式错误",
        }
        if raw in mapping:
            return mapping[raw]
        cleaned = re.sub(r"[^a-z0-9_\-\u4e00-\u9fff]+", "", raw)
        cleaned = _PLAYER_ID_RE.sub("历史玩家", cleaned)
        return cleaned[:24] or "other"

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
        if len(text) <= _MAX_JSON_CONTEXT_CHARS:
            return text

        structured = self._structured_json_summary(value)
        if structured is not None:
            structured_text = json.dumps(
                structured,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if len(structured_text) <= _MAX_JSON_CONTEXT_CHARS:
                return structured_text

        prefix = text[: _MAX_JSON_CONTEXT_CHARS // 3]
        suffix = text[-(_MAX_JSON_CONTEXT_CHARS // 3):]
        while True:
            rendered = json.dumps(
                {
                    "truncated": True,
                    "original_type": type(value).__name__,
                    "original_length": len(text),
                    "omitted_middle": True,
                    "content_prefix": prefix,
                    "content_suffix": suffix,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if len(rendered) <= _MAX_JSON_CONTEXT_CHARS or (not prefix and not suffix):
                return rendered
            overflow = len(rendered) - _MAX_JSON_CONTEXT_CHARS
            trim_prefix = min(len(prefix), overflow // 2 + 8)
            trim_suffix = min(len(suffix), overflow - trim_prefix + 8)
            prefix = prefix[: max(0, len(prefix) - trim_prefix)]
            suffix = suffix[min(len(suffix), trim_suffix):]

    @staticmethod
    def _structured_json_summary(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            items = list(value.items())
            if not items:
                return None
            head_count = min(6, len(items))
            tail_count = min(4, max(0, len(items) - head_count))
            head = {
                key: PlayerPromptBuilder._summarize_json_value(val)
                for key, val in items[:head_count]
            }
            tail_items = items[-tail_count:] if tail_count else []
            tail = {
                key: PlayerPromptBuilder._summarize_json_value(val)
                for key, val in tail_items
            }
            return {
                "truncated": True,
                "original_type": "dict",
                "original_length": len(items),
                "omitted_middle": len(items) > head_count + tail_count,
                "head": head,
                "tail": tail,
            }
        if isinstance(value, list):
            if not value:
                return None
            head_count = min(4, len(value))
            tail_count = min(3, max(0, len(value) - head_count))
            return {
                "truncated": True,
                "original_type": "list",
                "original_length": len(value),
                "omitted_middle": len(value) > head_count + tail_count,
                "head": [
                    PlayerPromptBuilder._summarize_json_value(item)
                    for item in value[:head_count]
                ],
                "tail": [
                    PlayerPromptBuilder._summarize_json_value(item)
                    for item in (value[-tail_count:] if tail_count else [])
                ],
            }
        return None

    @staticmethod
    def _summarize_json_value(value: Any) -> Any:
        if isinstance(value, str):
            return PlayerPromptBuilder._clean_prompt_text(
                value,
                max_chars=_MAX_LEARNING_TEXT_CHARS,
            )
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            return [
                PlayerPromptBuilder._summarize_json_value(item)
                for item in value[:3]
            ]
        if isinstance(value, dict):
            slim: dict[str, Any] = {}
            for idx, (key, val) in enumerate(value.items()):
                if idx >= 5:
                    break
                slim[str(key)] = PlayerPromptBuilder._summarize_json_value(val)
            return slim
        return PlayerPromptBuilder._clean_prompt_text(
            value,
            max_chars=_MAX_LEARNING_TEXT_CHARS,
        )

    @staticmethod
    def _truncate_text(
        text: str,
        max_chars: int,
        *,
        marker: str = "...（已截断）",
        prefer_sentence_boundary: bool = True,
    ) -> str:
        # D4-8 (P2): two improvements on top of the P2-4 marker:
        #   1. Use the Chinese-parenthetical marker ``（已截断）`` (the
        #      default ``marker``) to distinguish prose truncation
        #      (here, default callers) from JSON truncation (P2-4
        #      used ``<已截断>`` in _compact_json, which still passes
        #      that marker explicitly). The LLM can now tell the two
        #      truncation contexts apart.
        #   2. Prefer a sentence-boundary cut (`.`/`。`/`!`/`?`/`！`/`？`
        #      plus newline) within the last ~10% of the budget so the
        #      LLM sees a clean stop rather than a mid-word break.
        #      JSON callers (which pass ``prefer_sentence_boundary=False``)
        #      bypass this: JSON has no sentences and the cut must
        #      land somewhere, not on a punctuation character that
        #      might be inside a string literal.
        if len(text) <= max_chars:
            return text
        if prefer_sentence_boundary:
            slack = max(1, max_chars // 10)
            search_start = max_chars - slack
            best_cut = text.rfind("\n", search_start, max_chars)
            if best_cut < search_start:
                # No newline in slack window; try sentence-end punctuation.
                for ch in ".。!！?？":
                    idx = text.rfind(ch, search_start, max_chars)
                    if idx > best_cut:
                        best_cut = idx
            if best_cut >= search_start:
                return text[: best_cut + 1] + marker
        return text[:max_chars] + marker


# P0-2 (defense): explicit field whitelist for salience items. The
# runtime currently does not populate private keys into salience_items,
# but the renderer should enforce the boundary so a future change
# cannot leak `seer_result`, `witch_target`, `wolf_team`, etc. into
# the player-visible prompt.
# D4-1: ``speaker`` / ``result`` / ``alignment`` are explicitly public
# fields. A seer_claim event's whole point is to broadcast who claimed
# what, so speaker (who) / result (claimed outcome) / alignment
# (claimed faction) are public-by-construction. The private
# ``seer_result`` (the raw check result the Seer actually saw) is
# distinct and stays in _SALIENCE_PRIVATE_KEYS.
_SALIENCE_PUBLIC_FIELDS: frozenset[str] = frozenset({
    "weight", "bucket", "fact_type", "source", "target", "value",
    "day", "phase", "event_type",
    "speaker", "result", "alignment",
    # Phase-1 audit: ``id`` and ``summary`` are public event identity
    # and human-readable description.  ``id`` is the dedup key used by
    # downstream cognition stages; ``summary`` is the only narrative
    # content the LLM sees for a salience event.  Without these the
    # LLM has no way to track which salience event is which across
    # turns (slim dropped everything except numeric weight).
    "id", "summary",
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
