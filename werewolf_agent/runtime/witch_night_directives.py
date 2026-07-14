# -*- coding: utf-8 -*-
"""
构建女巫夜晚行动阶段的合法行动和策略指令。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-14

使用示例:
    >>> from werewolf_agent.runtime.witch_night_directives import build_witch_potion_status
    >>> build_witch_potion_status(antidote_used=False, poison_used=False)
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from werewolf_agent.agents.schemas import ActionType
from werewolf_agent.core.models import GameState
from werewolf_agent.engine.rule_engine import RuleEngine


def build_witch_legal_actions(
    gs: GameState,
    engine: RuleEngine,
    *,
    witch_id: str,
    wolf_kill_target_id: str | None,
) -> tuple[list[ActionType], list[str]]:
    """根据当前药水状态和规则集构建女巫可选行动。"""
    legal_actions = [ActionType.NO_ACTION]
    legal_targets: list[str] = []
    if wolf_kill_target_id and not gs.antidote_used:
        witch_cfg = engine.ruleset.raw["roles"]["witch"]["abilities"]
        can_self_save = witch_cfg["antidote"].get("can_self_save", False)
        if wolf_kill_target_id != witch_id or can_self_save:
            legal_actions.append(ActionType.USE_ANTIDOTE)
            legal_targets.append(wolf_kill_target_id)
    if not gs.poison_used:
        legal_actions.append(ActionType.USE_POISON)
        legal_targets.extend([
            pid for pid, player in gs.players.items()
            if player.alive and pid != witch_id
        ])
    return legal_actions, legal_targets


def build_witch_potion_status(*, antidote_used: bool, poison_used: bool) -> str:
    """构建女巫当前药水状态说明。"""
    potion_status = (
        f"当前药水状态：解药{'已用' if antidote_used else '可用'}，"
        f"毒药{'已用' if poison_used else '可用'}。"
    )
    if antidote_used and not poison_used:
        potion_status += "你只剩毒药，只能选择毒人或不用。"
    elif not antidote_used and poison_used:
        potion_status += "你只剩解药，只能选择救人或不用。"
    return potion_status


def build_witch_action_evidence(
    *,
    legal_targets: Sequence[str],
    antidote_targets: Sequence[str] | None = None,
    poison_targets: Sequence[str] | None = None,
    poison_candidates: Sequence[Mapping[str, Any]],
    wolf_kill_target_id: str | None,
) -> dict[str, Any]:
    """构造替代比较、保留药水收益与误伤风险证据，不替代模型决策。"""
    targets = list(dict.fromkeys(legal_targets))
    supported = {
        str(candidate.get("player_id"))
        for candidate in poison_candidates
        if candidate.get("player_id")
    }
    candidate_reasons = {
        str(candidate.get("player_id")): str(candidate.get("reason") or "").strip()
        for candidate in poison_candidates
        if candidate.get("player_id")
    }
    antidote = list(dict.fromkeys(
        antidote_targets
        if antidote_targets is not None
        else ([wolf_kill_target_id] if wolf_kill_target_id in targets else [])
    ))
    poison = list(dict.fromkeys(
        poison_targets if poison_targets is not None else targets
    ))
    has_action_target = bool(antidote or poison)
    retain_option = {
        "action": "no_action",
        "available": True,
        "required": not has_action_target,
        "reason": (
            "保留药水可等待更强公开证据并避免本夜误伤"
            if has_action_target else "无合法用药目标"
        ),
    }
    return {
        "antidote_targets": antidote,
        "poison_targets": poison,
        "ranked_targets": sorted(
            [
                {
                    "target": target,
                    "value": 1 if target in supported else 0,
                    "signals": [
                        candidate_reasons.get(target)
                        or "no_structured_public_support"
                    ],
                }
                for target in poison
            ],
            key=lambda item: (-item["value"], item["target"]),
        ),
        "alternative_comparison": {
            "legal_alternatives": targets,
            "no_legal_alternative": len(targets) <= 1,
            "alternative_target": targets[1] if len(targets) > 1 else None,
        },
        "retain_option": retain_option,
        # 保留旧字段供已有报表读取，值始终派生自 retain_option。
        "retain_skill_evidence": {
            "available": retain_option["available"],
            "reason": retain_option["reason"],
        },
        "friendly_fire_risk": {
            "status": "assessed",
            "targets": [target for target in poison if target not in supported],
            "basis": "缺少结构化公开毒杀依据",
        },
    }


def build_witch_night_action_directive(
    *,
    wolf_kill_target_id: str | None,
    witch_id: str,
    antidote_used: bool,
    poison_used: bool,
    can_use_antidote: bool,
    can_use_poison: bool,
) -> str:
    """构建女巫夜晚行动主指令文本。"""
    potion_status = build_witch_potion_status(
        antidote_used=antidote_used,
        poison_used=poison_used,
    )
    directive = f"你是女巫，现在是夜间行动阶段。{potion_status}\n你的选择：\n"
    options = []
    can_self = True
    if wolf_kill_target_id and not antidote_used and can_use_antidote:
        can_self = wolf_kill_target_id != witch_id
        save_hint = "（他被狼人杀害了）" if can_self else "（但是你不能自救！）"
        options.append(
            f"1) 使用解药救{wolf_kill_target_id}{save_hint} —— "
            f"action_type='use_antidote', target_id='{wolf_kill_target_id}'"
        )
    if not poison_used and can_use_poison:
        options.append(
            "2) 使用毒药毒杀某人 —— action_type='use_poison', target_id='目标玩家ID'"
        )
    no_action_label = "3) 暂不使用药水 —— action_type='no_action'"
    if not options:
        no_action_label = "1) 不使用药水（无可用行动）—— action_type='no_action'"
    options.append(no_action_label)
    directive += "\n".join(options)
    directive += "\n\n重要规则：不能在同一夜同时使用解药和毒药。"
    if not can_self:
        directive += "解药不能自救。"
    if wolf_kill_target_id and not antidote_used:
        directive += (
            "\n\n请结合下方 save_value_assessment、公开证据和药水机会成本决定是否救人；"
            "不要仅因存在刀口就机械使用解药。"
        )
    elif not poison_used:
        directive += (
            "\n\n你的毒药仍可用。只有存在可追溯的公开证据和高置信目标时才考虑使用；"
            "没有明确目标时保留毒药，避免用猜测制造额外好人损失。"
        )
    directive += "speech字段留空（夜间行动不需要发言）。"
    return directive


def build_witch_strategy_hint(
    save_value: Mapping[str, Any],
    *,
    poison_available: bool,
) -> str:
    """根据救人价值评估构建女巫策略提示。"""
    hint = ""
    if save_value.get("actionable"):
        if save_value.get("public_info_available"):
            score = save_value.get("save_value_score", 0)
            interp = save_value.get("interpretation", "")
            signals = "、".join(save_value.get("signals", []))
            hint = f"被杀者价值评估：得分{score}分（信号：{signals}）。{interp}"
        else:
            probability_framework = save_value.get("probability_framework", {})
            trade_off = save_value.get("trade_off", {})
            p_power = probability_framework.get("p_power_role", 0)
            hint = (
                f"首夜无公开信息。被杀者是神职的概率约{p_power:.0%}，"
                f"是村民的概率约{probability_framework.get('p_villager', 0):.0%}。"
                f"权衡：{trade_off.get('save_now', '')} | "
                f"{trade_off.get('save_later', '')} | "
                f"{trade_off.get('risk_no_save', '')}"
            )
    if poison_available:
        hint += " 毒药可用时，也可以考虑不救而保留毒药用于验证可疑目标。"
    return hint


def build_witch_poison_strategy(alive_count: int) -> dict[str, Any]:
    """根据存活人数构建唯一的女巫毒药策略分支。"""
    if alive_count <= 7:
        branch = "urgency_under_X_alive"
        text = (
            f"【紧急】场上仅存活{alive_count}人！你的毒药还没有使用！"
            "好人阵营需要主动性，但用毒仍必须引用具体公开来源："
            "可信查杀、多人明确指控、强票型或身份逻辑破产。"
            "如果没有结构化候选或说不清公开证据，默认 no_action，"
            "不要凭印象误毒好人。"
        )
    elif alive_count <= 9:
        branch = "evidence_required_threshold"
        text = (
            f"场上存活{alive_count}人，解药已用，你每夜只有毒药和空过两个选项。"
            "如果你有怀疑目标（即使证据不够硬），应积极考虑用毒——但需权衡误毒好人的风险。"
        )
    else:
        branch = "no_pressure_save_for_late"
        text = (
            "【毒药决策指引】毒药是好人阵营唯一的主动击杀手段。"
            "以下情况应优先使用毒药："
            "1) 可信预言家的明确查杀；2) 强票型证据（连续保狼、冲票、关键轮分票）；"
            "3) 对跳失败或身份逻辑明显破产；4) 场上存活人数减少，再不用毒药可能来不及。"
            "如果存在合理怀疑但证据不够硬，应权衡'不用毒药导致好人出局'vs'误毒好人'的风险。"
            "解药已用后，你每夜只剩毒药或空过——空过意味着好人失去一轮主动权。"
        )
    return {
        "branch": branch,
        "alive_count": alive_count,
        "text": text,
    }


def build_witch_poison_candidates_directive(
    candidates: Sequence[Mapping[str, Any]],
    *,
    alive_count: int,
) -> str:
    """构建女巫毒药候选目标或无候选防误毒提示。"""
    if candidates:
        cand_desc = "; ".join(
            f"{candidate['player_id']}({candidate['reason']})"
            for candidate in candidates[:5]
        )
        return (
            f"【毒药候选目标(按证据强度排序)】: {cand_desc}。"
            "如果你要用毒,请优先从以上候选中选(排在前面的证据更强)。"
            "如果你认为以上都不够硬,可选择 no_action 并在 reason 中说明理由。"
        )
    if alive_count > 9:
        return "【默认 no_action】当前公开信息不足,无明确高证据度狼目标。不要凭印象用毒。"
    if alive_count <= 7:
        return (
            "【紧急但证据不足】存活 ≤ 7 但无结构化候选。"
            "不推荐凭印象用毒；默认 no_action，除非 reason 能写清具体公开来源。"
        )
    return (
        "【证据不足】当前公开信息不足以构成用毒依据。"
        "如果没有强烈怀疑,默认 no_action。"
    )


def build_witch_first_night_killed_directive(
    *,
    wolf_kill_target_id: str | None,
    witch_id: str,
    poison_used: bool,
) -> str | None:
    """女巫首夜被刀且仍有毒药时，构建额外提醒。"""
    if wolf_kill_target_id != witch_id or poison_used:
        return None
    return (
        "你是女巫，N1 / 首夜就被狼人杀害了！你即将死亡，无法自救。"
        "若已有可追溯的高证据狼人目标，可以权衡使用毒药；"
        "若只有模糊怀疑，宁可不盲毒，避免在信息最少时额外击杀好人。"
        "遗言只能准确说明你实际采取的行动和当时依据。"
    )


def build_witch_pressure_directives(
    poison_pressure: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """把毒药压力目标渲染成策略指令。"""
    if not poison_pressure:
        return {}
    pressure_desc = "; ".join(
        f"{pressure['player_id']}({pressure['pressure_type']}: {pressure['description']})"
        for pressure in poison_pressure
    )
    return {
        "witch_pressure": f"存在毒药压力目标: {pressure_desc}",
        "required_evaluation": "如果选择不用毒药，必须在reason中解释为什么不用。",
    }
