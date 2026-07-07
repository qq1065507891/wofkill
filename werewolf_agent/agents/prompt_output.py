# -*- coding: utf-8 -*-
"""
渲染玩家行动输出协议、示例和最终提交约束。

作者: Project contributors
创建日期: 2026-07-07

使用示例:
    >>> from werewolf_agent.agents.prompt_output import PromptOutputMixin
"""

from __future__ import annotations

import json

from werewolf_agent.agents.action_contract import ActionContract
from werewolf_agent.agents.parse_dispatch import select_output_mode
from werewolf_agent.agents.schemas import (
    ActionType,
    OutputMode,
    RetryInfo,
    TaskType,
)

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
_VOTE_AUDIT_FIELDS: tuple[str, ...] = tuple(
    f for f in _OUTPUT_SCHEMA_VOTE_FIELDS
    if f not in ("choice", "reason", "confidence")
)

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

_ACTION_TYPE_GUARD = (
    "\n【输出字段约束 P1-G3223805846-4】你输出的 JSON 顶层字段名必须使用 `action_type`，"
    "**不要**使用 `intent`、`command`、`action` 等其他名字。"
    "如果你写 `{\"intent\": \"...\"}` 整个输出会被解析器拒绝。\n"
    "正确的字段名：`action_type`, `target_id`, `speech`, `reason`, `confidence`, "
    "`private_intent`（可选嵌套对象）。\n"
)


class PromptOutputMixin:
    """负责玩家最终输出协议、任务示例和 choice 候选摘要。"""

    def _build_retry_hint(self, retry: RetryInfo) -> str:
        """Render the retry correction hint after task, before contract."""
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
        parts.append(_ACTION_TYPE_GUARD)
        if self._is_exile_vote_context():
            parts.append(_VOTE_REASON_PRIVACY_GUARD)
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
        elif ActionType.SHERIFF_VOTE in ctx.legal_actions:
            example_target = self._example_target()
            parts.append("示例输出（警长投票场景）：")
            parts.append(
                f'{{"action_type": "sheriff_vote", "target_id": "{example_target}", '
                '"speech": "", '
                f'"reason": "选择{example_target}作为警长候选人", "confidence": 0.7}}'
            )
            if ActionType.NO_ACTION in ctx.legal_actions:
                parts.append("示例输出（警长投票弃权场景）：")
                parts.append(
                    '{"action_type": "no_action", "target_id": null, '
                    '"speech": "", '
                    '"reason": "候选人信息不足，暂不投出警长票", "confidence": 0.5}'
                )
        else:
            role = ctx.own_role or "villager"
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
                         '"speech": "我觉得pXX很可疑，昨晚他的发言前后矛盾。", '
                         '"reason": "根据发言分析", "confidence": 0.7, '
                         f'"private_intent": {{"true_role": "{example_role}", '
                         f'"faction_goal": "{example_goal}", "claimed_view": "{example_view}", '
                         '"pressure_target": "pXX", "risk_flags": []}}')
            vote_example_goal = "confuse_good" if example_role == "werewolf" else "find_wolves"
            vote_example_view = _CLAIMED_VIEW_SEER if role == "seer" else _CLAIMED_VIEW_GOOD
            if role == "seer":
                vote_standing_with_seer = ""
                vote_basis = "seer_check"
            else:
                vote_standing_with_seer = "pXX"
                vote_basis = "seer_siding"
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
        """Per-turn output contract — adapts to task type."""
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
            if self._is_exile_vote_context():
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
        if self._is_exile_vote_context():
            lines.append(
                f"7. 投票还必须包含{'、'.join(_VOTE_AUDIT_FIELDS)}，理由字段不能写「未说明」。"
            )
        lines.append("现在提交行动。")
        return "\n".join(lines)

    def _format_choice_prompt(self) -> str:
        ctx = self.context
        is_vote = self._is_exile_vote_context()
        header = "投票候选枚举" if is_vote else "目标候选枚举"
        choice_map = self._vote_choice_map()
        lines = [f"{header}（必须从中选择一个choice，不要直接编写target_id）："]
        if is_vote:
            lines.append(
                "候选枚举不是公开证据；摘要只用于识别候选，不能复制成 vote reason。"
                "reason 必须引用当前局公开事实：查验、对跳、警徽流、票型或具体发言。"
            )
        for choice, target_id in choice_map.items():
            summary = (
                self._vote_candidate_summary(target_id)
                if is_vote
                else self._target_candidate_summary(target_id)
            )
            lines.append(f"{choice} = {target_id}，摘要：{summary}")
        if is_vote:
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

    def _select_output_mode(self) -> OutputMode:
        ctx = self.context
        return select_output_mode(
            legal_actions=ctx.legal_actions,
            legal_targets=ctx.legal_targets,
            task_type=ctx.task_type,
            speech_intent_tasks=_SPEECH_INTENT_TASKS,
        )

    def _is_exile_vote_context(self) -> bool:
        ctx = self.context
        return ctx.task_type == TaskType.VOTE and ActionType.VOTE in ctx.legal_actions

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
        return (
            "暂无该候选的公开证据摘要；如选择该目标，必须另引公开发言、"
            "票型、查验或警徽流，不得把候选身份当作证据"
        )

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
