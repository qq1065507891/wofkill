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

# Skill catalog: lightweight name + one-line description
_SKILL_CATALOG: dict[str, dict[str, str]] = {
    "wolf_pit": {
        "name": "盘狼坑",
        "desc": "系统性分析场上可能的狼人分布，输出嫌疑人区和排除区",
    },
    "find_power": {
        "name": "找神",
        "desc": "分析场上哪些玩家可能是神职（预言家/女巫/猎人等）",
    },
    "last_words": {
        "name": "分析遗言",
        "desc": "分析刚出局玩家的遗言，判断其身份可信度",
    },
}

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
}

_MAX_JSON_CONTEXT_CHARS = 1800
_MAX_TRANSCRIPT_ITEMS = 4
_MAX_TRANSCRIPT_TEXT_CHARS = 220
_MAX_SALIENCE_ITEMS = 4


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
        parts.append(self._build_persona())
        parts.append(self._build_game_rules())
        parts.append(self._build_role_guide())
        parts.append(self._build_skill_catalog())
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

    def _build_role_guide(self) -> str:
        lines: list[str] = []
        role = self.context.own_role or ""
        role_rules = {
            "hunter": "猎人规则：被狼人杀死或被放逐时可以开枪带走一人；被女巫毒杀时不能开枪。夜间无法自保。",
            "idiot": "白痴规则：被放逐时亮出身份免死，但失去投票权且不能再被放逐；之后被狼人杀死才算真正死亡。夜间无法自保。",
            "witch": "女巫规则：有一瓶解药和一瓶毒药，不能在同一夜同时使用。解药不能自救。N1 / 首夜大概率应该救人。",
            "seer": "预言家规则：每晚可查验一人身份（好人/狼人），查验混血儿结果为好人。上警时必须留两夜警徽流。",
            "werewolf": "狼人规则：夜间与队友讨论击杀目标。可以悍跳预言家上警对抗真预言家。",
            "hybrid": "混血儿规则：N1 / 首夜选择一名主人，跟随主人阵营获胜。主人死亡后阵营不再改变。",
        }
        if role in role_rules:
            lines.append(role_rules[role])
        return "\n".join(lines) if lines else ""

    def _build_skill_catalog(self) -> str:
        """Build the lightweight skill catalog (name + one-line desc only).

        Full analysis lives behind load_skill tool — injected on demand.
        """
        from werewolf_agent.skills.registry import SkillRegistry, faction_for_role
        from werewolf_agent.runtime.agent_adapter import _TOOL_SKILL_NAMES

        registry = SkillRegistry()
        role = self.context.own_role or ""
        phase = self.context.phase or ""
        role_faction = faction_for_role(role)
        allowed = {"common", "universal", role_faction.value}

        lines: list[str] = []
        for skill in registry.all_skills():
            if skill.faction.value not in allowed:
                continue
            if not skill.is_applicable(role, phase):
                continue
            name = skill.name.value
            if name not in _TOOL_SKILL_NAMES:
                continue
            catalog = _SKILL_CATALOG.get(name)
            if catalog:
                lines.append(f"- {catalog['name']}: {catalog['desc']}")

        if lines:
            return (
                "【可用技能目录】以下技能可通过 load_skill 工具按需加载完整分析：\n"
                + "\n".join(lines)
                + "\n在提交行动前，建议先调用 load_skill 加载相关分析以获得更准确的信息。"
            )
        return ""

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

    def build_user_prompt(self, retry: RetryInfo) -> str:
        parts: list[str] = []
        # Boundary marker per s10: above = stable, below = dynamic
        parts.append("=== DYNAMIC_BOUNDARY ===")
        parts.append(self._build_phase_context())
        parts.append(self._build_belief_state())
        parts.append(self._build_public_summary())
        parts.append(self._build_visible_state())
        parts.append(self._build_salience_events())
        parts.append(self._build_strategy_directive())
        parts.append(self._build_recent_transcript())
        parts.append(self._build_retry_hint(retry))
        parts.append(self._build_task_prompt())
        parts.append(self._build_strict_output_contract())
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
        return "游戏概况:\n" + self._truncate_text(ctx.public_summary, _MAX_JSON_CONTEXT_CHARS)

    def _build_visible_state(self) -> str:
        ctx = self.context
        if not ctx.visible_world_state:
            return ""
        return "可见状态: " + self._compact_json(ctx.visible_world_state)

    def _build_salience_events(self) -> str:
        ctx = self.context
        if not ctx.salience_items:
            return ""
        return "关键事件: " + self._compact_json(
            ctx.salience_items[:_MAX_SALIENCE_ITEMS],
        )

    def _build_strategy_directive(self) -> str:
        ctx = self.context
        if not ctx.strategy_directive:
            return ""
        return "策略建议: " + self._compact_json(ctx.strategy_directive)

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
        if not retry.correction_hint:
            return ""
        return (
            f"纠正提示（第{retry.attempt}/{retry.max_retries}次尝试）: {retry.correction_hint}\n"
            f"错误信息: {retry.error_message}"
        )

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
                f'"faction_goal": "push_good_player_out", "claimed_view": "我是好人", '
                f'"pressure_target": "{example_target}", "risk_flags": []}}}}'
            )
            parts.append("示例输出（狼人空刀场景）：")
            parts.append(
                '{"action_type": "wolf_no_kill", "target_id": null, '
                '"speech": "", '
                '"reason": "本轮空刀策略", "confidence": 0.6, '
                '"private_intent": {"true_role": "werewolf", '
                '"faction_goal": "confuse_good", "claimed_view": "我是好人", '
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
            parts.append("示例输出（发言场景）：")
            parts.append('{"action_type": "speech", "target_id": null, '
                         '"speech": "我觉得p05很可疑，昨晚他的发言前后矛盾。", '
                         '"reason": "根据发言分析", "confidence": 0.7, '
                         '"private_intent": {"true_role": "villager", '
                         '"faction_goal": "find_wolves", "claimed_view": "我是好人", '
                         '"pressure_target": "p05", "risk_flags": []}}')
            parts.append("示例输出（投票场景）：")
            parts.append('{"action_type": "vote", "target_id": "p05", '
                         '"speech": "", '
                         '"reason": "公开理由：p05发言可疑", '
                         '"seer_stance": "trust", '
                         '"vote_basis": "seer_check", '
                         '"standing_with_seer": "p03", '
                         '"suspect_reason": "p05没有回应p03的查杀逻辑，发言前后不一致", '
                         '"not_voting_reason": "p07虽然被踩，但目前没有明确查验或票型证据", '
                         '"private_reason": "心里活动：我更信p03的预言家线，p05像狼队抗推失败后的防守位，所以投p05。", '
                         '"confidence": 0.8, '
                         '"private_intent": {"true_role": "seer", '
                         '"faction_goal": "find_wolves", "claimed_view": "我是预言家", '
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
