"""Werewolf skill definitions: 12 core gameplay skills per design doc §11.1.

Each skill provides a deterministic, game-state-aware tactical suggestion.
When game_state is provided, handlers analyze real signals and produce
actionable Chinese-language advice. When game_state is None, handlers fall
back to static output for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Callable

from werewolf_agent.skills.schemas import (
    SkillDefinition,
    SkillFaction,
    SkillInput,
    SkillName,
    SkillOutput,
)


# ---------------------------------------------------------------------------
# Skill handler registry (table-driven pattern)
# ---------------------------------------------------------------------------

_SKILL_HANDLERS: dict[SkillName, Callable] = {}


def register_handler(name: SkillName):
    """Decorator to register a skill handler in the dispatch table."""
    def decorator(fn: Callable) -> Callable:
        _SKILL_HANDLERS[name] = fn
        return fn
    return decorator


def get_handler(name: SkillName) -> Callable | None:
    """Look up a registered skill handler by name."""
    return _SKILL_HANDLERS.get(name)


# ---------------------------------------------------------------------------
# 12 core werewolf skills
# ---------------------------------------------------------------------------

def _parse_skill_frontmatter(text: str) -> dict[str, Any]:
    """Extract YAML frontmatter from a SKILL.md file (delimited by ---)."""
    import yaml
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    return yaml.safe_load(parts[1]) or {}


def _load_manifests(root: "Path | None" = None) -> list[SkillDefinition]:
    """Load skill metadata from SKILL.md files under skill directories.

    Each skill directory contains a SKILL.md with YAML frontmatter
    (name, description, applicable roles/phases, faction, tags).
    The dynamic analysis logic lives in Python handlers — only static
    metadata is stored here.

    The `root` parameter is a test seam: production callers omit it
    (it defaults to this module's parent directory); tests pass a
    `tmp_path` to load fixtures in isolation.
    """
    from pathlib import Path

    if root is None:
        root = Path(__file__).resolve().parent
    result: list[SkillDefinition] = []
    for skill_dir in sorted(root.iterdir()):
        if not skill_dir.is_dir() or skill_dir.name.startswith("_") or skill_dir.name.startswith("."):
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        data = _parse_skill_frontmatter(skill_md.read_text(encoding="utf-8"))
        if not data:
            continue
        try:
            result.append(SkillDefinition(
                name=SkillName(data["name"]),
                display_name=data.get("display_name", ""),
                description=data.get("description", ""),
                applicable_roles=data.get("applicable_roles", []),
                applicable_phases=data.get("applicable_phases", []),
                # S-01: precise task-type filter (P0-K2). Frontmatter
                # may declare `applies_to_task_types: [speech, ...]`
                # to scope the skill to specific task types instead
                # of cramming them into `applicable_phases`.
                applies_to_task_types=data.get("applies_to_task_types", []),
                faction=SkillFaction(data.get("faction", "common")),
                tags=data.get("tags", []),
            ))
        except (KeyError, ValueError) as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Failed to load skill %s: %s", skill_dir.name, exc,
            )
    return result


SKILL_DEFINITIONS: list[SkillDefinition] = _load_manifests()


# ---------------------------------------------------------------------------
# S-06: shared prompt_injectable length cap.
# ---------------------------------------------------------------------------

# Cap any prompt_injectable to this many characters. The renderer is
# the LLM's user prompt, and prompts that grow past ~1KB start
# bleeding into the model's context budget. Late-game review (last_words,
# review_correction, wolf_pit) historically produced 1-3KB prompts.
PROMPT_INJECTABLE_CAP = 800
# Truncation marker: appended to the end of a truncated prompt.  Kept
# short so it survives the cap itself. Uses ASCII "..." so the
# marker is preserved across all encodings (test cross-checks).
PROMPT_INJECTABLE_MARKER_TAIL = "...（已省略）"


def _cap_prompt_injectable(text: str, cap: int = PROMPT_INJECTABLE_CAP) -> str:
    """Truncate `text` to `cap` chars, appending a marker on truncation.

    S-06: late-game review output can exceed 1KB and bloat the
    renderer prompt. Cap to 800 chars and signal truncation to the
    LLM so it knows the advice is partial.
    """
    if not text:
        return text
    if len(text) <= cap:
        return text
    marker = PROMPT_INJECTABLE_MARKER_TAIL
    return text[: cap - len(marker)] + marker


# ---------------------------------------------------------------------------
# Shared helpers for game-state-aware analysis
# ---------------------------------------------------------------------------

def _count_seer_claimants(ws: Any) -> int:
    """Count distinct players who publicly claimed seer."""
    if ws is None:
        return 0
    claimants: set[str] = set()
    for f in ws.facts_of_type("claimed_role"):
        if f.value == "seer" and f.source_player:
            claimants.add(f.source_player)
    return len(claimants)


def _get_seer_claimants(ws: Any) -> list[str]:
    """Return list of players who publicly claimed seer."""
    if ws is None:
        return []
    claimants: set[str] = set()
    for f in ws.facts_of_type("claimed_role"):
        if f.value == "seer" and f.source_player:
            claimants.add(f.source_player)
    return sorted(claimants)


def _alive_wolves(gs: Any) -> list[str]:
    """Return alive wolf teammates."""
    if gs is None:
        return []
    return [
        pid for pid, p in gs.players.items()
        if p.alive and p.role == "werewolf"
    ]


def _alive_non_wolves(gs: Any) -> list[str]:
    """Return alive non-wolf players."""
    if gs is None:
        return []
    return [
        pid for pid, p in gs.players.items()
        if p.alive and p.role != "werewolf"
    ]


def _vote_targets_for_player(ws: Any, player_id: str) -> list[dict[str, Any]]:
    """Get vote facts targeting a specific player."""
    if ws is None:
        return []
    return [
        {"source": f.source_player, "day": f.day, "value": f.value}
        for f in ws.facts_of_type("vote")
        if f.target_player == player_id
    ]


def _seer_checks_on_target(ws: Any, target_id: str) -> list[dict[str, Any]]:
    """Get seer check claims targeting a specific player."""
    if ws is None:
        return []
    results = []
    for f in ws.facts_of_type("seer_check_claim"):
        if f.target_player == target_id:
            results.append({"source": f.source_player, "value": f.value, "day": f.day})
    return results


def _alerts_for_player(alerts: list[Any], player_id: str) -> list[Any]:
    """Filter contradiction alerts that mention the player."""
    if not player_id:
        return []
    return [
        a for a in alerts
        if player_id == a.player_id or ("," in a.player_id and player_id in a.player_id.split(","))
    ]


def _belief_top_suspects(bs: Any, count: int = 3) -> list[tuple[str, str, float]]:
    """Return top suspects from belief state (wolf_lean, lowest trust)."""
    if bs is None:
        return []
    suspects: list[tuple[str, str, float]] = []
    for pid, belief in bs.beliefs.items():
        if belief.faction_lean == "wolf_lean" or belief.trust < 0.35:
            suspects.append((pid, belief.faction_lean, belief.trust))
    suspects.sort(key=lambda x: x[2])
    return suspects[:count]


def _wolf_teammates_exposed(ws: Any, wolf_ids: list[str]) -> list[dict[str, Any]]:
    """Check which wolf teammates have been publicly seer-checked as wolf."""
    if ws is None:
        return []
    exposed = []
    for wid in wolf_ids:
        checks = _seer_checks_on_target(ws, wid)
        for c in checks:
            if "wolf" in c.get("value", "").lower() or "狼" in c.get("value", ""):
                exposed.append({"teammate": wid, "checked_by": c["source"]})
    return exposed


# ---------------------------------------------------------------------------
# Skill dispatch
# ---------------------------------------------------------------------------

def apply_skill(skill_name: SkillName, skill_input: SkillInput) -> SkillOutput:
    """Apply a skill to generate a tactical suggestion."""
    skill_def = _find_definition(skill_name)
    if skill_def is None:
        return SkillOutput(
            skill_name=skill_name.value,
            confidence=0.0,
            risk_alerts=["未知技能"],
        )

    handler = get_handler(skill_name)
    if handler is None:
        handler = _default_handler
    return handler(skill_input, skill_def)


def _find_definition(name: SkillName) -> SkillDefinition | None:
    for s in SKILL_DEFINITIONS:
        if s.name == name:
            return s
    return None


# ---------------------------------------------------------------------------
# Handlers — each merges static fallback (game_state is None) and dynamic
# analysis (game_state provided) into a single registered function.
# ---------------------------------------------------------------------------

def _default_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    return SkillOutput(
        skill_name=skill.name.value,
        confidence=0.5,
        reasoning=f"技能 {skill.display_name} 适用，需要更多局势信息",
    )


# --- BOLD_CLAIM (悍跳) ---

@register_handler(SkillName.BOLD_CLAIM)
def bold_claim_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        # static fallback
        risks = ["悍跳风险：如果对跳方是真预言家，可信度会大幅下降"]
        if inp.day > 2:
            risks.append("晚期悍跳风险更高：已发言轮次多，矛盾点容易被抓")
        conf = 0.6 if inp.day <= 1 else 0.3
        prompt = "悍跳建议：尽早跳预言家并报出假查验结果。构建完整的时间线和警徽流。" if conf >= 0.5 else "晚期悍跳风险极高，不建议此时悍跳。"
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["报查验结果", "声明警徽流", "攻击对立面逻辑"],
            risk_alerts=risks,
            confidence=conf,
            reasoning="悍跳需要前期执行，后期风险增大",
            prompt_injectable=_cap_prompt_injectable(prompt),
        )
    # dynamic analysis
    ws = inp.world_state
    day = gs.day_number
    seer_count = _count_seer_claimants(ws)
    wolves = _alive_wolves(gs)

    # If wolf_team_plan assigns a different wolf as fake_seer, skip bold claim advice
    wolf_plan = inp.extra.get("wolf_team_plan") if inp.extra else None
    if wolf_plan and wolf_plan.get("fake_seer") and wolf_plan["fake_seer"] != inp.player_id:
        # S-14: do NOT name the fake_seer teammate.  Role-neutral
        # phrasing — the teammate's player_id is a wolf-team secret
        # and must not leak into a prompt that may be inspected by
        # good-faction analysis tools.
        return SkillOutput(
            skill_name=skill.name.value,
            confidence=0.3,
            reasoning="已有队友占据预言家身份，你不需要悍跳",
            prompt_injectable=_cap_prompt_injectable(
                "悍跳建议：已有队友占据预言家身份，你不需要悍跳。"
                "配合TA的预言家身份进行站边和推人即可。"
            ),
        )

    risks: list[str] = ["悍跳风险：如果对跳方是真预言家，可信度会大幅下降"]

    if day > 2:
        risks.append("晚期悍跳风险更高：已发言轮次多，矛盾点容易被抓")

    if seer_count == 0:
        conf = 0.75 if day <= 1 else 0.55
        prompt = (
            f"悍跳建议：场上无人跳预言家（单边），悍跳窗口极佳。"
            f"建议立即跳预言家并报出假查验结果，同时声明警徽流。"
        )
        speech = ["报假查验结果", "声明完整警徽流", "攻击对立面逻辑"]
    elif seer_count == 1:
        claimant = _get_seer_claimants(ws)[0]
        conf = 0.55 if day <= 1 else 0.3
        prompt = (
            f"悍跳建议：场上已有 {claimant} 跳预言家，悍跳将形成对跳。"
            f"必须准备完整的假验人时间线来对抗。找出 {claimant} 发言的漏洞。"
        )
        speech = [f"指出{claimant}的发言漏洞", "报出完整假验人时间线", "声明警徽流对比"]
        risks.append(f"对跳 {claimant} 需要时间线高度一致，任何矛盾都会暴露")
    else:
        claimants = _get_seer_claimants(ws)
        conf = 0.3
        prompt = (
            f"悍跳建议：场上已有多人({', '.join(claimants)})跳预言家，"
            f"继续悍跳会导致多方混战，建议转为深水或倒钩策略。"
        )
        speech = ["暂不悍跳", "保持中立发言", "观察对跳结果"]
        risks.append("多方混战中悍跳极易被识破")

    if len(wolves) <= 2:
        risks.append(f"存活狼人仅{len(wolves)}人，悍跳失败代价极高")

    # Suggest a fake check target (alive non-wolf player)
    fake_target = None
    non_wolves = _alive_non_wolves(gs)
    if non_wolves and seer_count <= 1:
        fake_target = non_wolves[0]

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=speech,
        risk_alerts=risks,
        confidence=conf,
        reasoning="动态分析：根据场上预言家声明情况调整悍跳策略",
        prompt_injectable=_cap_prompt_injectable(prompt),
    )


# --- COUNTER_CLAIM (对跳) ---

@register_handler(SkillName.COUNTER_CLAIM)
def counter_claim_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    # S-10: branch on `inp.role`. A REAL seer countering a fake seer
    # needs to defend their check result and rip the faker's timeline.
    # A WOLF countering a real seer (i.e. doing the "悍跳" pair-up)
    # needs to fabricate a matching timeline and steer the room.
    # NEW-R4-P2-1: a HYBRID with `hybrid_master_faction='werewolf'`
    # is on the wolf side and gets the same 悍跳 framing as a real
    # werewolf (faking-seer is one of the hybrid's wolf-team jobs).
    is_seer = inp.role == "seer"
    is_wolf = inp.role == "werewolf"
    is_hybrid_wolf = (
        inp.role == "hybrid"
        and gs is not None
        and getattr(gs, "hybrid_master_faction", None) == "werewolf"
    )
    effective_wolf = is_wolf or is_hybrid_wolf
    if gs is None:
        # static fallback — role-tailored phrasing
        if is_seer:
            return SkillOutput(
                skill_name=skill.name.value,
                speech_structure=["展示自己的真查验结果", "攻击对方时间线漏洞", "对比警徽流"],
                risk_alerts=["对跳时一定要保持自己验人时间线的一致性"],
                confidence=0.6,
                reasoning="真预言家对跳：核心是守护自己的查验时间线",
                prompt_injectable=_cap_prompt_injectable(
                    "对跳建议（真预言家视角）：用你的真实查验结果逐条对比对方的"
                    "假时间线。任何不匹配都是暴露对方假预言家的机会。"
                    "重点攻击验人动机、警徽流矛盾。"
                ),
            )
        if is_wolf:
            return SkillOutput(
                skill_name=skill.name.value,
                speech_structure=["准备完整的假验人记录", "攻击真预言家的逻辑漏洞", "排坑占边"],
                risk_alerts=["悍跳风险：如果对方是真预言家，可信度会大幅下降"],
                confidence=0.55,
                reasoning="悍跳对跳：核心是构建与队友一致的假时间线",
                prompt_injectable=_cap_prompt_injectable(
                    "对跳建议（狼队悍跳视角）：你作为狼的悍跳者，需要准备完整的"
                    "假验人时间线来对跳真预言家。重点攻击对方的验人动机和警徽流漏洞，"
                    "并用排坑占边把节奏拉到自己这边。"
                ),
            )
        if is_hybrid_wolf:
            # NEW-R4-P2-1: hybrid-with-wolf-master gets the same 悍跳
            # framing as a real werewolf when faking seer. The wolf
            # team plan relies on the hybrid building a matching
            # fake-check timeline.
            return SkillOutput(
                skill_name=skill.name.value,
                speech_structure=["准备完整的假验人记录", "攻击真预言家的逻辑漏洞", "排坑占边"],
                risk_alerts=["悍跳风险：如果对方是真预言家，可信度会大幅下降"],
                confidence=0.55,
                reasoning="悍跳对跳（混合体视角）：核心是构建与狼队一致的假时间线",
                prompt_injectable=_cap_prompt_injectable(
                    "对跳建议（混合体悍跳视角）：你作为混合体且主人是狼，"
                    "你与狼队一起行动。你需要准备完整的假验人时间线来对跳真预言家。"
                    "重点攻击对方的验人动机和警徽流漏洞，"
                    "并用排坑占边把节奏拉到自己这边。"
                ),
            )
        # Other roles (villager/hunter/witch/...) — neutral counter-claim.
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["指出对方漏洞", "报自身查验信息", "建立时间线对比"],
            risk_alerts=["对跳风险：真预言家对跳时好人会倾向真预言家"],
            confidence=0.55,
            reasoning="对跳需要充分的逻辑支撑和时间线一致性",
            prompt_injectable=_cap_prompt_injectable("对跳建议：如果有人跳预言家，准备好完整的假验人记录来对跳。重点攻击对方的验人时间线和警徽流漏洞。"),
        )
    # dynamic analysis
    ws = inp.world_state
    claimants = _get_seer_claimants(ws)
    alerts = inp.contradiction_alerts

    if not claimants:
        return SkillOutput(
            skill_name=skill.name.value,
            confidence=0.35,
            reasoning="场上无人跳预言家，无需对跳",
        )

    target = claimants[0]
    target_alerts = _alerts_for_player(alerts, target)
    has_claim_conflict = any(a.alert_type == "claim_conflict" for a in target_alerts)

    # S-10: role-tailored dynamic prompt
    if is_seer:
        # Real seer: defend own timeline, rip faker's inconsistencies.
        if has_claim_conflict:
            prompt = (
                f"对跳分析（真预言家视角）：{target} 的发言存在矛盾点（claim_conflict）。"
                f"用你真实的查验结果逐条对照对方的假时间线，"
                f"任何不匹配都是暴露对方假预言家的机会。"
            )
            conf = 0.7
            speech = [
                f"展示你的真实查验结果对比{target}的假时间线",
                f"指出{target}验人时间线上的具体矛盾",
                "建立你自己的完整时间线作为锚点",
            ]
        else:
            prompt = (
                f"对跳分析（真预言家视角）：{target} 的发言看起来较一致。"
                f"重点从侧面找漏洞：验人动机、警徽流合理性、站边逻辑。"
                f"如果对方是悍跳者，时间线上一定会有逻辑漏洞。"
            )
            conf = 0.55
            speech = [
                f"质疑{target}的验人动机",
                f"分析{target}的警徽流是否合理",
                "用你自己的金水作为锚点",
            ]
        risks = [f"对跳 {target} 需保持查验时间线一致"]
    elif is_wolf:
        # Wolf (fake seer / counter-claimer): fabricate, attack, herd.
        if has_claim_conflict:
            prompt = (
                f"对跳分析（悍跳视角）：{target} 的发言已经暴露矛盾（claim_conflict）。"
                f"抓住这个矛盾猛攻，把场上风向拉到对你有利的方向。"
                f"用你准备的假验人时间线作为正面证据。"
            )
            conf = 0.65
            speech = [
                f"放大{target}的发言矛盾",
                "用你准备好的假时间线作为正面证据",
                "排坑占边把节奏拉到自己这边",
            ]
        else:
            prompt = (
                f"对跳分析（悍跳视角）：{target} 的发言暂时没明显漏洞。"
                f"不要正面硬刚，从侧面找漏洞（验人动机、警徽流合理性）。"
                f"如果真预言家时间线无漏洞，转为排坑占边策略。"
            )
            conf = 0.45
            speech = [
                f"侧面质疑{target}的验人动机",
                "准备完整的假时间线作为预案",
                "用排坑占边为团队创造空间",
            ]
        risks = [f"悍跳 {target} 需要完整的假验人记录，时间线断裂会暴露"]
    elif is_hybrid_wolf:
        # NEW-R4-P2-1: hybrid-with-wolf-master uses wolf 悍跳 framing
        # in the dynamic branch (mirrors the static fallback).
        if has_claim_conflict:
            prompt = (
                f"对跳分析（混合体悍跳视角）：{target} 的发言已经暴露矛盾（claim_conflict）。"
                f"作为混合体与狼队一起行动，抓住这个矛盾猛攻，"
                f"把场上风向拉到对你和狼队有利的方向。"
                f"用你准备的假验人时间线作为正面证据。"
            )
            conf = 0.6
            speech = [
                f"放大{target}的发言矛盾",
                "用你准备好的假时间线作为正面证据",
                "排坑占边把节奏拉到自己这边",
            ]
        else:
            prompt = (
                f"对跳分析（混合体悍跳视角）：{target} 的发言暂时没明显漏洞。"
                f"作为混合体与狼队一起行动，不要正面硬刚，从侧面找漏洞"
                f"（验人动机、警徽流合理性）。如果真预言家时间线无漏洞，"
                f"转为排坑占边策略。"
            )
            conf = 0.4
            speech = [
                f"侧面质疑{target}的验人动机",
                "准备完整的假时间线作为预案",
                "用排坑占边为狼队创造空间",
            ]
        risks = [f"悍跳 {target} 需要完整的假验人记录，时间线断裂会暴露"]
    else:
        # Other roles (villager/...) — neutral dynamic advice.
        if has_claim_conflict:
            prompt = (
                f"对跳分析：{target} 的发言存在矛盾点（claim_conflict）。"
                f"建议集中攻击以下疑点，质疑其预言家身份的真实性。"
            )
            conf = 0.65
            speech = [f"指出{target}验人时间线的矛盾", f"对比{target}前后不一致的发言", "建立自己的完整时间线"]
        else:
            prompt = (
                f"对跳分析：{target} 的发言和验人时间线较一致，直接对跳风险较高。"
                f"建议从侧面寻找漏洞：验人动机、警徽流合理性、站边逻辑。"
            )
            conf = 0.45
            speech = [f"质疑{target}的验人动机", f"分析{target}的警徽流是否合理", "找侧面漏洞而非正面硬刚"]
        risks = [f"对跳 {target} 需要完整的假验人记录，任何时间线断裂都会暴露"]

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=speech,
        risk_alerts=risks,
        confidence=conf,
        reasoning=f"动态分析（S-10: {inp.role} 对跳）：根据{target}的发言一致性调整策略",
        prompt_injectable=_cap_prompt_injectable(prompt),
    )


# --- PUSH_VOTE (归票) ---

@register_handler(SkillName.PUSH_VOTE)
def push_vote_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    # P1-K5: branch on task_type. push_vote during VOTE phase = "who to
    # actually vote for". push_vote during SPEECH phase = "how to
    # rhetorically rally others to vote for X". The two need different
    # advice and confidence framing.
    is_vote_task = inp.task_type == "vote"
    is_speech_task = inp.task_type == "speech"
    if gs is None:
        # static fallback — task_type-specific phrasing
        if is_vote_task:
            prompt = (
                "归票建议（投票阶段）：根据发言逻辑和验人信息，"
                "选择嫌疑最大的人作为你的投票目标。"
            )
            speech = ["确认你的最终投票目标", "回顾其嫌疑证据", "准备投出选票"]
        else:
            # Default (and speech-task): rhetoric-focused push.
            prompt = (
                "归票建议：根据发言逻辑和验人信息，选择嫌疑最大的人归票。"
                "陈述理由时需要有理有据，号召全场跟随。"
            )
            speech = ["陈述归票理由", "分析目标嫌疑", "号召全场归票"]
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=speech,
            risk_alerts=["归票错误目标可能导致好人损失"],
            confidence=0.6,
            reasoning="归票需要有充分的逻辑依据和说服力",
            prompt_injectable=_cap_prompt_injectable(prompt),
        )
    # dynamic analysis
    ws = inp.world_state
    bs = inp.belief_state

    top_suspects = _belief_top_suspects(bs, count=3)
    day = gs.day_number

    if not top_suspects:
        if is_vote_task:
            prompt = (
                "归票建议（投票阶段）：当前信息不足，没有明确嫌疑目标。"
                "选择一个相对最可疑的目标投票，避免弃票。"
            )
        elif is_speech_task:
            prompt = (
                "归票建议（发言阶段）：当前信息不足，没有明确嫌疑目标。"
                "在发言中表示需要观察，避免无依据地号召归票。"
            )
        else:
            prompt = "归票建议：当前信息不足，建议观察发言后再决定归票方向。"
        return SkillOutput(
            skill_name=skill.name.value,
            confidence=0.4,
            reasoning="当前无明确嫌疑目标",
            prompt_injectable=_cap_prompt_injectable(prompt),
        )

    primary, lean, trust = top_suspects[0]
    reasons = []
    if lean == "wolf_lean":
        reasons.append(f"行为模式偏向狼人")
    if trust < 0.3:
        reasons.append(f"可信度极低({trust:.0%})")

    checks = _seer_checks_on_target(ws, primary)
    for c in checks:
        if "wolf" in c.get("value", "").lower() or "狼" in c.get("value", ""):
            reasons.append(f"被{c['source']}查杀为狼")
            break

    votes_on = _vote_targets_for_player(ws, primary)
    if len(votes_on) >= 2:
        reasons.append(f"已有多人({len(votes_on)}票)指向该玩家")

    reason_text = "；".join(reasons) if reasons else "综合行为分析"
    if is_vote_task:
        # Vote phase: emphasize the target pick + evidence, no rhetoric.
        prompt = (
            f"归票建议（投票阶段）：{primary} 的嫌疑最大。"
            f"理由：{reason_text}。"
            f"请直接选 {primary} 作为你的投票目标。"
        )
        speech = [f"确认{primary}为最终投票目标", f"回顾{primary}的嫌疑证据", "投出选票"]
        risks = ["归票错误目标可能导致好人损失"]
    elif is_speech_task:
        # Speech phase: emphasize rhetoric, herd rallying.
        prompt = (
            f"归票建议（发言阶段）：{primary} 的嫌疑最大。"
            f"理由：{reason_text}。"
            f"在发言中陈述理由，号召全场集中票数归出 {primary}。"
        )
        speech = [f"陈述{primary}的嫌疑理由", "分析其行为链", "号召全场归票"]
        risks = ["归票错误目标可能导致好人损失"]
    else:
        # Unknown / default: keep original generic phrasing.
        prompt = (
            f"归票建议：根据场上信息，{primary} 的嫌疑最大。"
            f"理由：{reason_text}。号召全场集中票数归出 {primary}。"
        )
        speech = [f"陈述{primary}的嫌疑理由", "分析其行为链", "号召全场归票"]
        risks = ["归票错误目标可能导致好人损失"]

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=speech,
        risk_alerts=risks,
        confidence=0.6 + min(0.2, len(reasons) * 0.05),
        reasoning=f"动态分析：{primary} 有{len(reasons)}个嫌疑信号"
                  + ("（vote task）" if is_vote_task else
                     "（speech task）" if is_speech_task else ""),
        prompt_injectable=_cap_prompt_injectable(prompt),
    )


# --- SWING_VOTE (冲票) ---

@register_handler(SkillName.SWING_VOTE)
def swing_vote_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    # S-03: branch on task_type. `wolf_discussion` is a NIGHT phase —
    # the wolves are picking a night-kill target, not a day-vote target.
    # Default (day-vote) behavior is preserved.
    is_wolf_discussion = inp.task_type == "wolf_discussion"

    if gs is None:
        # static fallback
        risks = ["冲票暴露投票链：好人可能通过投票链锁定狼人"]
        if is_wolf_discussion:
            return SkillOutput(
                skill_name=skill.name.value,
                speech_structure=["在狼讨论中提出冲刀目标", "分析目标的投票压力", "协调队友分散或集中夜杀"],
                risk_alerts=risks,
                confidence=0.5,
                reasoning="冲刀需要考虑夜杀链暴露风险",
                prompt_injectable=_cap_prompt_injectable((
                    "冲刀建议（狼队夜杀讨论）：选择场上已有投票压力的好人作为冲刀目标。"
                    "与队友协调夜杀方向，避免夜杀链暴露狼人身份。"
                )),
            )
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["在狼讨论中提出冲票目标", "分析目标的投票压力", "协调队友分散或集中票"],
            risk_alerts=risks,
            confidence=0.5,
            reasoning="冲票需要考虑投票链暴露风险",
            prompt_injectable=_cap_prompt_injectable("冲票建议：选择场上已有投票压力的好人作为冲票目标。与队友协调投票方向，避免投票链暴露狼人身份。"),
        )
    # dynamic analysis
    ws = inp.world_state

    wolves = _alive_wolves(gs)
    non_wolves = _alive_non_wolves(gs)

    if not non_wolves:
        return SkillOutput(
            skill_name=skill.name.value,
            confidence=0.0,
            reasoning="无冲票目标" if not is_wolf_discussion else "无冲刀目标",
        )

    # Find non-wolf with most vote pressure
    # Weight by game phase: later phases have more settled opinions, so
    # votes and suspect claims carry more weight.
    day = gs.day_number
    vote_weight = 1.0 + min(day * 0.3, 1.5)    # 1.0 → 2.5 over 5 days
    suspect_weight = 1.5 + min(day * 0.3, 1.5)  # 1.5 → 3.0 over 5 days

    pressure: dict[str, float] = {pid: 0.0 for pid in non_wolves}
    for f in (ws.facts_of_type("vote") if ws else []):
        if f.target_player in pressure:
            pressure[f.target_player] += vote_weight
    for f in (ws.facts_of_type("claimed_suspect") if ws else []):
        if f.target_player in pressure:
            pressure[f.target_player] += suspect_weight

    best_target = max(non_wolves, key=lambda p: pressure.get(p, 0))
    best_pressure = pressure.get(best_target, 0)

    wolf_count = len(wolves)
    risks = ["冲票暴露投票链：好人可能通过投票链锁定狼人"]
    if wolf_count <= 2:
        risks.append(f"仅剩{wolf_count}狼，冲票暴露风险极高")

    if is_wolf_discussion:
        # Night-kill semantics: rephrase the prompt and action.
        if best_pressure > 0:
            prompt = (
                f"冲刀建议（狼队夜杀讨论）：集中狼队夜杀 {best_target}。"
                f"理由：{best_target} 已有{best_pressure}个怀疑信号，"
                f"冲刀成功率高，避免暴露狼队。"
            )
            conf = 0.5 + min(0.2, best_pressure * 0.04)
        else:
            prompt = (
                "冲刀建议（狼队夜杀讨论）：当前无明确冲刀目标，"
                "建议根据好人发言集中度选择夜杀目标。"
            )
            conf = 0.35
        return SkillOutput(
            skill_name=skill.name.value,
            risk_alerts=risks,
            confidence=conf,
            reasoning=(
                f"动态分析：{best_target} 有{best_pressure}个压力信号，"
                f"{wolf_count}狼存活（夜杀任务）"
            ),
            prompt_injectable=_cap_prompt_injectable(prompt),
        )

    if best_pressure > 0:
        prompt = (
            f"冲票建议：集中狼队票数冲 {best_target}。"
            f"理由：{best_target} 已有{best_pressure}个怀疑信号，冲票成功率高。"
        )
        conf = 0.5 + min(0.2, best_pressure * 0.04)
    else:
        prompt = "冲票建议：当前无明确冲票目标，建议分散投票避免暴露。"
        conf = 0.35

    return SkillOutput(
        skill_name=skill.name.value,
        risk_alerts=risks,
        confidence=conf,
        reasoning=f"动态分析：{best_target} 有{best_pressure}个压力信号，{wolf_count}狼存活",
        prompt_injectable=_cap_prompt_injectable(prompt),
    )


# --- DEEP_HOOK (倒钩) ---

@register_handler(SkillName.DEEP_HOOK)
def deep_hook_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        # static fallback
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["站边好人逻辑", "适度攻击可疑队友", "建立可信度"],
            risk_alerts=["倒钩策略需要长期维持一致性", "过度攻击队友可能被识别"],
            confidence=0.55,
            reasoning="倒钩核心是在好人阵营中建立长期可信度",
            prompt_injectable=_cap_prompt_injectable("倒钩建议：伪装成好人，站边好人逻辑线，适度攻击被怀疑的队友来建立信任。注意保持发言一致性，不要前后矛盾。"),
        )
    # dynamic analysis
    ws = inp.world_state
    bs = inp.belief_state

    wolves = _alive_wolves(gs)
    exposed = _wolf_teammates_exposed(ws, wolves)

    # If wolf role assignment is pusher or fake_seer, deep hook doesn't apply
    wolf_plan = inp.extra.get("wolf_team_plan") if inp.extra else None
    if wolf_plan:
        my_role_key = None
        for key in ("fake_seer", "pusher", "hooker", "deep_cover"):
            if wolf_plan.get(key) == inp.player_id:
                my_role_key = key
                break
        if my_role_key in ("fake_seer", "pusher"):
            return SkillOutput(
                skill_name=skill.name.value,
                confidence=0.3,
                reasoning=f"你是{my_role_key}，不需要倒钩策略",
                prompt_injectable=_cap_prompt_injectable("倒钩建议：你的角色分工是冲锋/悍跳，不需要倒钩。专注于你的主要任务。"),
            )

    day = gs.day_number
    risks = ["倒钩策略需要长期维持一致性", "过度攻击队友可能被识别"]

    if exposed:
        teammate = exposed[0]["teammate"]
        checker = exposed[0]["checked_by"]
        prompt = (
            f"倒钩建议：队友 {teammate} 已被 {checker} 查杀暴露。"
            f"建议适度攻击 {teammate} 来获取好人信任——但不要太用力，保持可信度。"
            f"可以用'我也觉得{teammate}可疑'的方式自然切入。"
        )
        conf = 0.65
        speech = [
            f"适度质疑{teammate}的发言",
            "站边好人逻辑但不过激",
            "贡献独立分析建立可信度",
        ]
    else:
        prompt = (
            f"倒钩建议：场上狼队暂无暴露，继续保持深水伪装。"
            f"保持中立发言，偶尔贡献独立分析，不要主动引起关注。"
            f"等待合适的时机（如队友暴露后）再适度切割。"
        )
        conf = 0.55
        speech = ["保持中立发言节奏", "贡献独立分析", "避免过度暴露信息优势"]

    if day > 3:
        risks.append("已过Day 3，倒钩需更加谨慎——好人的分析会越来越细")
        conf = max(0.4, conf - 0.1)

    # Suggest a target to attack (exposed teammate if applicable)
    hook_target = exposed[0]["teammate"] if exposed else None

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=speech,
        risk_alerts=risks,
        confidence=conf,
        reasoning="动态分析：根据队友暴露状态调整倒钩策略",
        prompt_injectable=_cap_prompt_injectable(prompt),
    )


# --- FIND_POWER (找神) ---

@register_handler(SkillName.FIND_POWER)
def find_power_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        # static fallback
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["分析发言信息量", "观察投票倾向", "识别保护行为"],
            confidence=0.5,
            reasoning="找神需要综合多个信号源进行推断",
            prompt_injectable=_cap_prompt_injectable("找神建议：关注发言中信息量异常的玩家（可能知道夜晚信息）、投票倾向保守的玩家、以及试图保护某些位置的玩家，这些可能是神职。"),
        )
    # dynamic analysis
    ws = inp.world_state
    bs = inp.belief_state

    # S-11: include `idiot` in power_roles. A revealed idiot is a
    # confirmed good player; protecting them keeps a vote-loss but
    # living role in play. Excluding idiot meant the post-exile idiot
    # (公开白露光后) was a free kill for wolves — the protection
    # skill was not flagging them as at-risk.
    power_roles = {"seer", "witch", "hunter", "idiot"}
    candidates: list[dict[str, Any]] = []

    if bs is not None:
        for pid, belief in bs.beliefs.items():
            # NEW-S19-D: skip dead players. A dead player with high
            # role probability would land in candidates and trip the
            # S-19 filter downstream. Mirror the wolf_pit belief-state
            # loop's alive guard.
            player = gs.players.get(pid)
            if not player or not player.alive:
                continue
            for role, prob in belief.role_probabilities.items():
                if role in power_roles and prob > 0.3:
                    candidates.append({
                        "player": pid,
                        "role": role,
                        "probability": prob,
                        "source": "belief",
                    })

    if ws is not None:
        for f in ws.facts_of_type("badge_flow_claim"):
            if f.source_player:
                candidates.append({
                    "player": f.source_player,
                    "role": "seer",
                    "probability": 0.6,
                    "source": "badge_flow",
                })

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for c in sorted(candidates, key=lambda x: x["probability"], reverse=True):
        key = f"{c['player']}_{c['role']}"
        if key not in seen:
            seen.add(key)
            unique.append(c)

    if not unique:
        return SkillOutput(
            skill_name=skill.name.value,
            confidence=0.3,
            reasoning="暂无足够信号推断神职",
            prompt_injectable=_cap_prompt_injectable("找神分析：当前信息不足，建议继续观察发言信息量和投票模式。"),
        )

    lines = []
    for c in unique[:3]:
        lines.append(f"{c['player']} 大概率是 {c['role']}（依据：{c['source']}，置信{c['probability']:.0%}）")
    prompt = f"找神分析：{'；'.join(lines)}"

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=["分析发言信息量", "观察投票倾向", "识别保护行为"],
        confidence=0.5 + min(0.2, len(unique) * 0.05),
        reasoning=f"动态分析：识别到{len(unique)}个疑似神职信号",
        prompt_injectable=_cap_prompt_injectable(prompt),
    )


# --- HIDE_IDENTITY (藏身份) ---

@register_handler(SkillName.HIDE_IDENTITY)
def hide_identity_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        # static fallback
        risks = ["藏身份过久可能导致无法在关键时刻发挥作用"]
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["保持中立发言", "避免暴露信息优势", "控制发言节奏"],
            risk_alerts=risks,
            confidence=0.6,
            reasoning="藏身份需要在隐匿和发挥作用之间找到平衡",
            prompt_injectable=_cap_prompt_injectable("藏身份建议：发言保持中立，不要暴露你知道的夜晚信息。如果被质疑，适度释放信息自证但不要全露底牌。"),
        )
    # dynamic analysis
    ws = inp.world_state
    pid = inp.player_id
    alerts = inp.contradiction_alerts
    day = gs.day_number

    my_claims = []
    if ws is not None:
        for f in ws.facts_of_type("claimed_role"):
            if f.source_player == pid:
                my_claims.append(f.value)

    my_alerts = _alerts_for_player(alerts, pid)
    under_pressure = len(my_alerts) > 0

    risks = ["藏身份过久可能导致无法在关键时刻发挥作用"]

    if my_claims:
        prompt = (
            f"藏身份建议：你已公开声明身份（{', '.join(my_claims)}），"
            f"身份已不完全隐蔽。建议继续维持已声明的人设，保持一致性。"
        )
        conf = 0.4
    elif under_pressure:
        pressure_desc = "; ".join(a.description for a in my_alerts[:2])
        prompt = (
            f"藏身份建议：你正被怀疑（{pressure_desc}），"
            f"需要适度释放信息来自证，但不要完全暴露身份。"
            f"可以给出部分信息来降低怀疑，同时保留核心身份信息。"
        )
        conf = 0.55
        risks.append("被怀疑时过度隐蔽反而加深嫌疑")
    else:
        prompt = (
            f"藏身份建议：你目前身份隐蔽状态良好，没有公开声明也没有被重点怀疑。"
            f"继续保持中立发言节奏，避免暴露信息优势。"
        )
        conf = 0.65

    if day > 3:
        risks.append("已过Day 3，继续藏身份可能导致无法在关键时刻发挥作用")
        conf = max(0.4, conf - 0.1)

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=["保持中立发言", "避免暴露信息优势", "控制发言节奏"],
        risk_alerts=risks,
        confidence=conf,
        reasoning="动态分析：根据自身暴露状态和被怀疑程度调整策略",
        prompt_injectable=_cap_prompt_injectable(prompt),
    )


# --- RESIST_PUSH (抗推) ---

@register_handler(SkillName.RESIST_PUSH)
def resist_push_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        # static fallback
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["承认疑虑合理性", "逻辑反驳关键指控", "提出建设性站边"],
            risk_alerts=["过度防御可能加深怀疑", "攻击质疑者会适得其反"],
            confidence=0.55,
            reasoning="抗推需要冷静的逻辑反驳，而非情绪对抗",
            prompt_injectable=_cap_prompt_injectable("抗推建议：冷静反驳质疑，针对关键指控逐一回应。如果被查杀，质疑预言家身份；如果仅被怀疑，补充自己的逻辑线和站边理由。"),
        )
    # dynamic analysis
    ws = inp.world_state
    pid = inp.player_id
    alerts = inp.contradiction_alerts

    # Check if player is seer-checked (查杀)
    seer_checks = _seer_checks_on_target(ws, pid)
    is_seer_checked = any(
        "wolf" in c.get("value", "").lower() or "狼" in c.get("value", "")
        for c in seer_checks
    )

    # Find who is pushing against us
    votes_against = _vote_targets_for_player(ws, pid)
    pushers = [v["source"] for v in votes_against if v.get("day", 0) == inp.game_state.day_number]

    risks = ["过度防御可能加深怀疑", "攻击质疑者会适得其反"]

    if is_seer_checked:
        checker = seer_checks[0]["source"] if seer_checks else "未知"
        prompt = (
            f"抗推建议：你被 {checker} 查杀为狼。这是最危险的推票信号。"
            f"必须质疑该玩家的预言家身份——指出其验人动机和逻辑漏洞。"
            f"可以考虑：1) 质疑其预言家身份 2) 提出自己的身份线 3) 分析其站边的可疑性。"
        )
        conf = 0.6
        speech = [f"质疑{checker}的预言家身份", "提出自身行为的一致性", "分析查杀动机的合理性"]
    elif pushers:
        prompt = (
            f"抗推建议：你被 {', '.join(pushers)} 等人怀疑/推票，但无查杀等实质证据。"
            f"冷静反驳：指出指控缺乏事实基础，转而质疑推你的人的逻辑。"
            f"用事实和逻辑回应，避免情绪化。"
        )
        conf = 0.55
        speech = ["指出指控缺乏实质证据", "逻辑反驳关键指控", "转而质疑推票者的动机"]
    else:
        prompt = (
            f"抗推建议：你目前被轻微怀疑但无明确推票压力。"
            f"保持冷静，适度解释自己的立场，避免过度反应引起更多怀疑。"
        )
        conf = 0.5
        speech = ["适度解释自身立场", "保持冷静客观", "贡献有价值分析转移注意力"]

    # Check for contradictions in pushers' behavior (counter-attack material)
    for pusher in pushers[:1]:
        pusher_alerts = _alerts_for_player(alerts, pusher)
        if pusher_alerts:
            prompt += f" 注意：{pusher} 自身存在发言矛盾，可作为反击材料。"
            conf += 0.05

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=speech,
        risk_alerts=risks,
        confidence=conf,
        reasoning="动态分析：根据推票来源和强度调整抗推策略",
        prompt_injectable=_cap_prompt_injectable(prompt),
    )


# --- WOLF_PIT_ANALYSIS (盘狼坑) ---

@register_handler(SkillName.WOLF_PIT_ANALYSIS)
def wolf_pit_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        # static fallback
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["列出嫌疑人", "分析各嫌疑人证据", "排除法缩小范围"],
            confidence=0.5,
            reasoning="盘狼坑需要系统性分析所有嫌疑人的行为链",
            prompt_injectable=_cap_prompt_injectable("盘狼坑建议：系统性分析所有嫌疑人的行为链。从发言矛盾、投票链异常、验人冲突等维度排查，用排除法缩小狼坑范围。"),
        )
    # dynamic analysis
    ws = inp.world_state
    bs = inp.belief_state

    # Build suspect and exclude lists
    suspects: list[tuple[str, str]] = []   # (player, reason)
    excluded: list[tuple[str, str]] = []   # (player, reason)

    alive_count = sum(1 for p in gs.players.values() if p.alive)

    # From belief state
    if bs is not None:
        for pid, belief in bs.beliefs.items():
            player = gs.players.get(pid)
            if not player or not player.alive:
                continue
            if belief.faction_lean == "wolf_lean":
                suspects.append((pid, f"行为偏向狼人(信任{belief.trust:.0%})"))
            elif belief.faction_lean == "good_lean" and belief.trust > 0.7:
                excluded.append((pid, f"行为像好人(信任{belief.trust:.0%})"))

    # From seer checks
    if ws is not None:
        for fact_type in ("seer_check_claim",):
            for f in ws.facts_of_type(fact_type):
                target = f.target_player
                val = (f.value or "").lower()
                # NEW-S19-B: skip dead players. A dead player with a
                # seer_check_claim would be added to suspects/excluded
                # and then dropped by the S-19 filter (or worse, the
                # prompt would carry an "illegal" target). Mirror the
                # belief-state loop above which already guards on
                # `player.alive`.
                target_player = gs.players.get(target) if target else None
                if not target_player or not target_player.alive:
                    continue
                if target and ("wolf" in val or "狼" in (f.value or "")):
                    suspects.append((target, f"被{f.source_player}查杀"))
                elif target and ("good" in val or "金水" in (f.value or "")):
                    excluded.append((target, f"被{f.source_player}发金水"))

    # Deduplicate
    suspect_ids = set()
    unique_suspects = []
    for pid, reason in suspects:
        if pid not in suspect_ids:
            suspect_ids.add(pid)
            unique_suspects.append((pid, reason))

    exclude_ids = set()
    unique_excluded = []
    for pid, reason in excluded:
        if pid not in exclude_ids:
            exclude_ids.add(pid)
            unique_excluded.append((pid, reason))

    suspect_lines = [f"{pid}({reason})" for pid, reason in unique_suspects[:5]]
    exclude_lines = [f"{pid}({reason})" for pid, reason in unique_excluded[:5]]

    prompt = (
        f"盘狼坑分析：当前存活{alive_count}人中，"
        f"嫌疑区({len(unique_suspects)}人)：{'；'.join(suspect_lines) if suspect_lines else '暂无明确嫌疑人'}。"
        f"排除区({len(unique_excluded)}人)：{'；'.join(exclude_lines) if exclude_lines else '暂无排除'}。"
        f"需要继续关注投票链和发言一致性来缩小范围。"
    )

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=["列出嫌疑人及其证据", "分析排除区", "缩小嫌疑范围"],
        confidence=0.5 + min(0.2, len(unique_suspects) * 0.05),
        reasoning=f"动态分析：{len(unique_suspects)}个嫌疑人，{len(unique_excluded)}个排除",
        prompt_injectable=_cap_prompt_injectable(prompt),
    )


# --- PROTECT_POWER (保护强神) ---

@register_handler(SkillName.PROTECT_POWER)
def protect_power_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        # static fallback
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["暗示关键角色需要保护", "引导怀疑方向远离神职", "分散狼队注意力"],
            risk_alerts=["过度保护某个玩家反而暴露其身份"],
            confidence=0.5,
            reasoning="保护强神需要隐蔽的引导而非明显的保护行为",
            prompt_injectable=_cap_prompt_injectable("保护强神建议：如果推测某玩家是神职且被推，用'我觉得他的逻辑没问题'等方式引导怀疑方向远离，不要直接说'保护他'。"),
        )
    # dynamic analysis
    ws = inp.world_state
    bs = inp.belief_state

    # S-11: include `idiot` in power_roles (see comment above in
    # find_power_handler).  Revealed idiot is a confirmed good role
    # to protect from wolf night-kill.
    power_roles = {"seer", "witch", "hunter", "idiot"}
    at_risk: list[dict[str, Any]] = []

    if bs is not None:
        for pid, belief in bs.beliefs.items():
            top_role, prob = belief.top_role_guess()
            if top_role in power_roles and prob > 0.3:
                votes_on = _vote_targets_for_player(ws, pid)
                # Also check social pressure: claimed_suspect against this player
                suspect_pressure = 0
                if ws is not None:
                    for f in ws.facts_of_type("claimed_suspect"):
                        if f.target_player == pid:
                            suspect_pressure += 1
                if votes_on or suspect_pressure > 0:
                    at_risk.append({
                        "player": pid,
                        "likely_role": top_role,
                        "votes": len(votes_on),
                        "suspect_claims": suspect_pressure,
                    })

    risks = ["过度保护某个玩家反而暴露其身份"]

    if at_risk:
        target = at_risk[0]
        pressure_desc = f"{target['votes']}票"
        if target["suspect_claims"] > 0:
            pressure_desc += f"、{target['suspect_claims']}次被公开怀疑"
        prompt = (
            f"保护强神建议：疑似{target['likely_role']}的 {target['player']} "
            f"正被施压（{pressure_desc}）。"
            f"建议发言引导怀疑方向远离TA：提出其他嫌疑人、质疑推票逻辑。"
            f"注意保护要隐蔽，不要让狼队察觉你在保人。"
        )
        conf = 0.6
    else:
        prompt = (
            f"保护强神建议：场上疑似神职暂时安全，无被推票压力。"
            f"继续观察，注意保护已识别的疑似神职不被狼队发现。"
        )
        conf = 0.45

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=["引导怀疑方向远离", "提出替代嫌疑人", "隐蔽保护"],
        risk_alerts=risks,
        confidence=conf,
        reasoning="动态分析：根据神职受压情况调整保护策略",
        prompt_injectable=_cap_prompt_injectable(prompt),
    )


# --- LAST_WORDS_ANALYSIS (遗言分析) ---

@register_handler(SkillName.LAST_WORDS_ANALYSIS)
def last_words_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    # S-18: unify the static fallback (gs is None) and the no-ws
    # branch (gs set but world_state is None) into a single early
    # return. Previously the two branches had duplicated code with
    # identical output — a divergence risk if one branch was edited
    # and the other was not.
    if gs is None or inp.world_state is None:
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["提取遗言关键信息", "分析遗言与已知信息的矛盾", "评估遗言可信度"],
            confidence=0.55,
            reasoning="遗言分析需要结合已有信息判断遗言内容的真实性",
            prompt_injectable=_cap_prompt_injectable("遗言分析建议：关注出局玩家最后发言中的角色声明、验人信息和站边逻辑。与已知信息交叉验证，判断遗言内容的可信度。"),
        )
    # dynamic analysis
    ws = inp.world_state
    alerts = inp.contradiction_alerts

    # Find recent deaths
    recent_deaths = list(ws.facts_of_type("player_died"))

    if not recent_deaths:
        return SkillOutput(
            skill_name=skill.name.value,
            confidence=0.3,
            reasoning="暂无死亡事件可分析",
            prompt_injectable=_cap_prompt_injectable("遗言分析：当前无遗言可分析。"),
        )

    # Analyze all deaths, prioritizing the most recent
    all_prompts: list[str] = []
    has_contradiction = False
    last_dead_player = None

    for death in reversed(recent_deaths):
        dead_player = death.target_player
        if dead_player is None:
            continue
        if last_dead_player is None:
            last_dead_player = dead_player

        # Find last speech of dead player
        speech_count = sum(
            1 for f in ws.facts_of_type("speech")
            if f.source_player == dead_player
        )

        # Check for contradictions
        dead_alerts = _alerts_for_player(alerts, dead_player)
        if dead_alerts:
            has_contradiction = True

        # Collect claims
        claims = [
            f"{f.value}(Day{f.day})"
            for f in ws.facts_of_type("claimed_role")
            if f.source_player == dead_player
        ]

        parts = [f"{dead_player}的遗言："]
        if claims:
            parts.append(f"身份声明：{', '.join(claims)}。")
        if dead_alerts:
            parts.append(f"发言矛盾：{'; '.join(a.description for a in dead_alerts[:2])}。")
        if speech_count:
            parts.append(f"有{speech_count}条发言记录。")
        # NEW-R4-P2-3: if the dead player has no claims, no
        # contradictions, and no speeches, parts is just the bare
        # "p05的遗言：" label with no body — a useless artifact that
        # wastes prompt budget. Fall back to a placeholder so the
        # LLM has something to read.
        if len(parts) == 1:
            parts.append("无具体遗言内容可分析。")
        all_prompts.append("".join(parts))

    if not all_prompts:
        return SkillOutput(
            skill_name=skill.name.value,
            confidence=0.3,
            reasoning="无有效遗言数据",
            prompt_injectable=_cap_prompt_injectable("遗言分析：无有效遗言可分析。"),
        )

    prompt = f"遗言分析（{len(all_prompts)}人死亡）：\n" + "\n".join(all_prompts)
    prompt += "\n需要结合已知信息判断遗言内容的真实性和意图。"

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=["提取遗言关键信息", "对比已知信息一致性", "评估可信度"],
        risk_alerts=["遗言可能是狼人的误导"] if has_contradiction else [],
        confidence=0.55 if not has_contradiction else 0.65,
        reasoning="动态分析：根据遗言内容与已知信息的对比判断可信度",
        prompt_injectable=_cap_prompt_injectable(prompt),
    )


# --- REVIEW_CORRECTION (复盘纠错) ---

@register_handler(SkillName.REVIEW_CORRECTION)
def review_correction_handler(inp: SkillInput, skill: SkillDefinition) -> SkillOutput:
    gs = inp.game_state
    if gs is None:
        # static fallback
        return SkillOutput(
            skill_name=skill.name.value,
            speech_structure=["回顾关键判断点", "识别错误和原因", "总结改进方向"],
            confidence=0.7,
            reasoning="复盘纠错以事实为基础，系统性地回顾决策过程",
            prompt_injectable=_cap_prompt_injectable("复盘建议：回顾每个Day的站边选择和投票决策。找出判断失误的关键节点，分析误判原因（信息不足？逻辑链断裂？被误导？），总结改进方向。"),
        )
    # dynamic analysis
    ws = inp.world_state

    winner = gs.winning_faction or "unknown"
    day = gs.day_number
    my_id = inp.player_id

    # Count deaths by cause
    deaths_by_wolf = 0
    deaths_by_exile = 0
    exiled_players: list[str] = []
    wolf_killed_players: list[str] = []
    if ws is not None:
        for f in ws.facts_of_type("player_died"):
            reason = f.value or ""
            if "wolf" in reason:
                deaths_by_wolf += 1
                if f.target_player:
                    wolf_killed_players.append(f.target_player)
            elif "exile" in reason:
                deaths_by_exile += 1
                if f.target_player:
                    exiled_players.append(f.target_player)

    # Analyze my vote accuracy: did I vote for wolves or good players?
    my_votes: list[str] = []
    if ws is not None:
        for f in ws.facts_of_type("vote"):
            if f.source_player == my_id and f.target_player:
                my_votes.append(f.target_player)

    # Check seer check accuracy if applicable
    seer_checks: list[str] = []
    if ws is not None:
        for fact_type in ("seer_check_claim",):
            for f in ws.facts_of_type(fact_type):
                if f.source_player == my_id and f.target_player:
                    val = f.value or ""
                    seer_checks.append(f"{f.target_player}={val}")

    # Build analysis
    parts: list[str] = []
    parts.append(f"复盘分析：游戏进行到Day {day}，狼刀{deaths_by_wolf}人，放逐{deaths_by_exile}人。")
    if winner != "unknown":
        parts.append(f"获胜方：{winner}。")

    correct_votes = 0
    if my_votes:
        parts.append(f"你共投出{len(my_votes)}票，投票目标：{'、'.join(my_votes[:6])}。")
        # Check if any vote hit a wolf (cross-reference with actual roles)
        wolf_ids = {pid for pid, p in gs.players.items() if p.role == "werewolf"}
        correct_votes = sum(1 for t in my_votes if t in wolf_ids)
        if correct_votes > 0:
            parts.append(f"其中{correct_votes}票命中狼人，投票准确率{correct_votes / len(my_votes):.0%}。")
        else:
            parts.append("所有投票均未命中狼人，需要反思站边和判断逻辑。")

    if seer_checks:
        parts.append(f"验人记录：{'、'.join(seer_checks[:4])}。")

    if exiled_players:
        parts.append(f"被放逐玩家：{'、'.join(exiled_players[:6])}。")
    if wolf_killed_players:
        parts.append(f"被狼刀玩家：{'、'.join(wolf_killed_players[:6])}。")

    parts.append("改进方向：检查站边选择是否正确、是否被狼人发言误导、投票链是否暴露了信息。")

    conf = 0.7
    if my_votes and correct_votes == 0:
        conf = 0.8  # High confidence in review when votes were all wrong

    return SkillOutput(
        skill_name=skill.name.value,
        speech_structure=["回顾关键判断点", "识别错误和原因", "总结改进方向"],
        confidence=conf,
        reasoning="动态分析：基于投票准确率和事件时间线进行复盘",
        prompt_injectable=_cap_prompt_injectable("\n".join(parts)),
    )
