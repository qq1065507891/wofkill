"""Death cause claim evaluation functions."""
from __future__ import annotations

import re
from typing import Any

from werewolf_agent.core.models import GameState


def evaluate_death_cause_claims(
    gs: GameState,
    player_id: str,
    player_role: str,
    wolf_kill_target_id: str | None = None,
) -> list[str]:
    """Evaluate trustworthiness of death cause claims from public speeches.

    Scans speech events for claims about how players died (poison, wolf-kill,
    saved). Evaluates each claim from the current player's perspective based
    on their private role knowledge.

    Returns a list of evaluation strings suitable for strategy_directive.
    """
    evaluations: list[str] = []
    if not player_role:
        return evaluations

    # Step 1: collect witch private data
    witch_actions: list[dict[str, Any]] = []
    for e in gs.events:
        if e.type == "witch_decision_audit":
            witch_actions.append(e.payload)

    # Step 2: collect all wolf kill targets (all nights, not just current)
    wolf_kill_targets: dict[int, str] = {}  # night_number -> target_id
    for e in gs.events:
        if e.type == "wolf_kill_selected":
            night = e.payload.get("night_number", 0)
            target = e.payload.get("target_id", "")
            if night and target:
                wolf_kill_targets[night] = target

    # Step 3: collect seer check results (seer_check events, visible only to seer)
    seer_checks: list[tuple[str, str]] = []
    if player_role == "seer":
        for e in gs.events:
            if e.type == "seer_check":
                seer_checks.append((e.payload["target_id"], e.payload["alignment"]))

    # Step 4: scan speeches for death cause claims
    _cause_patterns = [
        # (pattern, cause_type, label) — group 1 or 2 is the target player
        (r'(?:我|女巫).{0,4}(?:毒[杀了死]|撒毒).{0,4}(p\d+)', "poison", "自称毒杀"),
        (r'(p\d+).{0,6}(?:是|被)(?:女巫)?毒[杀了死]', "poison", "被指毒杀"),
        (r'(?:狼[刀杀人]|狼人[刀杀]).{0,4}(p\d+)|(p\d+).{0,4}(?:是|被)狼[刀杀了]', "wolf_kill", "被指狼刀"),
        (r'(?:我|女巫).{0,4}(?:救[了过]|用解药).{0,4}(p\d+)', "saved", "自称救了"),
        (r'(p\d+).{0,4}(?:是)?银水', "saved", "被指银水"),
    ]

    for e in gs.events:
        if e.type not in ("speech", "sheriff_speech"):
            continue
        speaker = e.payload.get("speaker", "")
        if speaker == player_id:
            continue
        text = str(e.payload.get("text", ""))

        for pattern, cause_type, label in _cause_patterns:
            m = re.search(pattern, text)
            if not m:
                continue
            target = m.group(1) or m.group(2)
            if not target:
                continue

            eval_text = None

            # -- Witch's own knowledge --
            if player_role == "witch":
                if cause_type == "poison":
                    poison_used = any(
                        a.get("action_taken") == "use_poison" and a.get("poison_target_id")
                        for a in witch_actions
                    )
                    poison_target = next(
                        (a["poison_target_id"] for a in witch_actions
                         if a.get("action_taken") == "use_poison" and a.get("poison_target_id")),
                        None,
                    )
                    if poison_target == target:
                        eval_text = f"[可信] {speaker}称{target}被毒杀——你确实毒了{target}"
                    elif poison_used and poison_target:
                        eval_text = f"[不可信] {speaker}称{target}被毒杀——你毒的是{poison_target}，不是{target}，{speaker}在撒谎或猜错"
                    elif not gs.poison_used or (gs.poison_used and not poison_used):
                        eval_text = f"[不可信] {speaker}称{target}被毒杀——你的毒药尚未使用"
                elif cause_type == "saved":
                    saved_target = next(
                        (a["wolf_kill_target_id"] for a in witch_actions
                         if a.get("action_taken") == "use_antidote"),
                        None,
                    )
                    if saved_target == target:
                        eval_text = f"[可信] {speaker}称{target}被救——你确实救了{target}"
                    elif gs.antidote_used and saved_target:
                        eval_text = f"[不可信] {speaker}称{target}被救——你救的是{saved_target}，不是{target}"
                elif cause_type == "wolf_kill":
                    wkt = wolf_kill_target_id or next(
                        (a["wolf_kill_target_id"] for a in witch_actions
                         if a.get("wolf_kill_target_id")),
                        None,
                    )
                    saved = any(
                        a.get("action_taken") == "use_antidote"
                        for a in witch_actions
                    )
                    if wkt == target and saved:
                        eval_text = f"[可信] {speaker}称{target}被狼刀——你救了{target}，他确实是狼刀目标"
                    elif wkt == target and not saved:
                        eval_text = f"[可信] {speaker}称{target}被狼刀——你没救{target}，他死于狼刀"

            # -- Werewolf's own knowledge --
            elif player_role == "werewolf":
                all_wolf_targets = set(wolf_kill_targets.values())
                if cause_type == "wolf_kill":
                    if target in all_wolf_targets:
                        nights = sorted(n for n, t in wolf_kill_targets.items() if t == target)
                        night_list = "、".join(f"N{n}" for n in nights)
                        eval_text = f"[可信] {speaker}称{target}被狼刀——狼队在{night_list}确实刀了{target}"
                    elif all_wolf_targets:
                        all_targets_str = "、".join(
                            f"N{n}:{t}" for n, t in sorted(wolf_kill_targets.items())
                        )
                        eval_text = f"[不可信] {speaker}称{target}被狼刀——狼队从未刀过{target}（狼队刀口: {all_targets_str}），{speaker}在转移视线"
                    else:
                        eval_text = f"[可信] {speaker}称{target}被狼刀——当前无狼刀记录，若{target}已死则可能属实"
                elif cause_type == "poison":
                    if target in all_wolf_targets:
                        eval_text = f"[需判断] {speaker}称{target}被毒杀——狼队刀过{target}，不排除毒杀，但{target}更可能是狼刀致死"
                    else:
                        eval_text = f"[需判断] {speaker}称{target}被毒杀——狼队未刀过{target}，若{target}已死则毒杀可能性高"
                elif cause_type == "saved":
                    if target in all_wolf_targets:
                        eval_text = f"[需判断] {speaker}称{target}被救——狼队刀过{target}，若无人死亡则可能属实"
                    else:
                        eval_text = f"[需判断] {speaker}称{target}被救——狼队未刀过{target}，{speaker}可能是假女巫在编造信息"

            # -- Seer's knowledge (from checks) --
            elif player_role == "seer":
                speaker_alignment = next((a for cid, a in seer_checks if cid == speaker), None)
                if speaker_alignment == "wolf":
                    eval_text = f"[不可信] {speaker}称{target}死因{label}——你查验{speaker}为狼人，此声明极可能是谎言"
                elif speaker_alignment == "good":
                    eval_text = f"[可信] {speaker}称{target}死因{label}——你查验{speaker}为好人，声明可信度较高"
                else:
                    eval_text = f"[需判断] {speaker}称{target}死因{label}——你未查验{speaker}，需从逻辑一致性判断"

            # -- Commoner perspective (villager / hunter / idiot / hybrid) --
            # D-5: rename the `[需判断]` label to `[公开判断]` and make
            # the villager no-private-info framing explicit.  Pre-fix
            # the label was misleading (it sounded like the evaluator
            # was unable to judge) and did not call out that the
            # villager-side branch is the public-only default.
            else:
                if player_role == "villager":
                    eval_text = (
                        f"[公开判断] {speaker}称{target}死因{label}——"
                        f"作为村民你无任何私有信息可验证，只能基于"
                        f"公开发言/票型/逻辑一致性判断{speaker}是否可信"
                    )
                elif player_role == "hunter":
                    eval_text = (
                        f"[公开判断] {speaker}称{target}死因{label}——"
                        f"作为猎人你无验人/药水信息，只能从公开逻辑判断"
                    )
                elif player_role == "idiot":
                    eval_text = (
                        f"[公开判断] {speaker}称{target}死因{label}——"
                        f"作为白痴你无私有信息，只能从公开逻辑判断"
                    )
                elif player_role == "hybrid":
                    eval_text = (
                        f"[公开判断] {speaker}称{target}死因{label}——"
                        f"作为混血儿你无验人/药水信息，只能从公开逻辑判断"
                    )
                else:
                    eval_text = (
                        f"[公开判断] {speaker}称{target}死因{label}——"
                        f"你无私有信息可验证，需判断{speaker}发言逻辑是否自洽、是否有矛盾"
                    )

            if eval_text:
                already_exists = any(eval_text in e for e in evaluations)
                if not already_exists:
                    evaluations.append(eval_text)
            break  # one label per speech

    return evaluations
