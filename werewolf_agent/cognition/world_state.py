# -*- coding: utf-8 -*-
"""
功能描述：每个事实是带已知模式的冻结 dataclass。事实列表是所有下游认知模块
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-27
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
from typing import Any

from werewolf_agent.core.event_visibility import EventVisibility, event_visibility
from werewolf_agent.core.models import GameEvent, GameState


# ---------------------------------------------------------------------------
# Structured fact types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StructuredFact:
    fact_type: str
    source_player: str | None = None
    target_player: str | None = None
    day: int = 0
    night: int = 0
    phase: str = ""
    value: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# World state: collection of all structured facts
# ---------------------------------------------------------------------------

class StructuredWorldState:
    """Accumulates structured facts derived from GameEvents."""

    def __init__(self) -> None:
        self._facts: list[StructuredFact] = []

    @property
    def facts(self) -> list[StructuredFact]:
        return list(self._facts)

    def append(self, fact: StructuredFact) -> None:
        self._facts.append(fact)

    def extend(self, facts: list[StructuredFact]) -> None:
        self._facts.extend(facts)

    def facts_of_type(self, fact_type: str) -> list[StructuredFact]:
        return [f for f in self._facts if f.fact_type == fact_type]

    def facts_about(self, player_id: str) -> list[StructuredFact]:
        return [
            f for f in self._facts
            if f.source_player == player_id or f.target_player == player_id
        ]


# ---------------------------------------------------------------------------
# Event-to-fact extraction
# ---------------------------------------------------------------------------

# Map event types to their fact_type and extraction logic.
# Each extractor returns a list of StructuredFact.

def _extract_player_died(event: GameEvent, state: GameState) -> list[StructuredFact]:
    pid = event.payload.get("player_id", "?")
    reason = event.payload.get("reason", "unknown")
    timing = event.payload.get("timing", "unknown")
    return [StructuredFact(
        fact_type="player_died",
        target_player=pid,
        phase=timing,
        value=reason,
        metadata={"resolution_batch": event.payload.get("resolution_batch", "")},
    )]


def _extract_idiot_revealed(event: GameEvent, state: GameState) -> list[StructuredFact]:
    pid = event.payload.get("player_id", "?")
    return [StructuredFact(
        fact_type="idiot_revealed",
        target_player=pid,
        value="revealed_idiot",
    )]


def _extract_player_exiled(event: GameEvent, state: GameState) -> list[StructuredFact]:
    pid = event.payload.get("player_id", "?")
    return [StructuredFact(
        fact_type="player_exiled",
        target_player=pid,
        value="exiled",
    )]


def _extract_werewolf_self_destructed(event: GameEvent, state: GameState) -> list[StructuredFact]:
    pid = event.payload.get("player_id", "?")
    day = event.payload.get("day_number", 0)
    return [StructuredFact(
        fact_type="self_destruct",
        source_player=pid,
        target_player=pid,
        day=day,
        value="self_destruct",
    )]


def _extract_hybrid_master_chosen(event: GameEvent, state: GameState) -> list[StructuredFact]:
    hybrid_id = event.payload.get("hybrid_id", "?")
    master_id = event.payload.get("master_id", "?")
    return [StructuredFact(
        fact_type="hybrid_master_chosen",
        source_player=hybrid_id,
        target_player=master_id,
        night=1,
        value="chose_master",
    )]


def _extract_sheriff_elected(event: GameEvent, state: GameState) -> list[StructuredFact]:
    sid = event.payload.get("sheriff_id", "?")
    return [StructuredFact(
        fact_type="sheriff_elected",
        target_player=sid,
        value="elected_sheriff",
    )]


def _extract_sheriff_registered(event: GameEvent, state: GameState) -> list[StructuredFact]:
    candidates = event.payload.get("candidates", [])
    return [
        StructuredFact(
            fact_type="sheriff_registered",
            source_player=c,
            value="registered_for_sheriff",
        )
        for c in candidates
    ]


def _extract_sheriff_withdraw(event: GameEvent, state: GameState) -> list[StructuredFact]:
    withdrew = event.payload.get("withdrew", [])
    return [
        StructuredFact(
            fact_type="sheriff_withdraw",
            source_player=p,
            value="withdrew_from_sheriff",
        )
        for p in withdrew
    ]


def _extract_sheriff_vote_tie(event: GameEvent, state: GameState) -> list[StructuredFact]:
    tied = event.payload.get("tied", [])
    return [StructuredFact(
        fact_type="sheriff_vote_tie",
        value="tie",
        metadata={"tied_players": tied},
    )]


def _extract_sheriff_vote_tie_first(event: GameEvent, state: GameState) -> list[StructuredFact]:
    tied = event.payload.get("tied", [])
    return [StructuredFact(
        fact_type="sheriff_vote_tie_first",
        value="tie_first",
        metadata={"tied_players": tied},
    )]


def _extract_badge_transferred(event: GameEvent, state: GameState) -> list[StructuredFact]:
    new_id = event.payload.get("new_sheriff_id", "?")
    return [StructuredFact(
        fact_type="badge_transferred",
        target_player=new_id,
        value="badge_transfer",
    )]


def _extract_badge_torn(event: GameEvent, state: GameState) -> list[StructuredFact]:
    return [StructuredFact(
        fact_type="badge_torn",
        value="badge_torn",
    )]


def _extract_witch_antidote_used(event: GameEvent, state: GameState) -> list[StructuredFact]:
    target = event.payload.get("target_id", "?")
    return [StructuredFact(
        fact_type="witch_antidote_used",
        target_player=target,
        value="antidote_saved",
    )]


def _extract_witch_poison_used(event: GameEvent, state: GameState) -> list[StructuredFact]:
    target = event.payload.get("target_id", "?")
    return [StructuredFact(
        fact_type="witch_poison_used",
        target_player=target,
        value="poison_killed",
    )]


def _extract_wolf_kill_selected(event: GameEvent, state: GameState) -> list[StructuredFact]:
    target = event.payload.get("target_id", "?")
    night = event.payload.get("night_number", 0)
    return [
        StructuredFact(
            fact_type="wolf_kill_selected",
            target_player=target,
            night=night,
            value="kill_selected",
        ),
        StructuredFact(
            fact_type="witch_kill_target",
            target_player=target,
            night=night,
            value="wolf_kill_target",
        ),
    ]


def _extract_wolf_no_kill(event: GameEvent, state: GameState) -> list[StructuredFact]:
    return [StructuredFact(
        fact_type=event.type,
        night=event.payload.get("night_number", 0),
        value="no_kill",
        metadata={"reason": event.payload.get("reason", "")},
    )]


# Speech/vote events from agent interactions (generic)
def _extract_speech(event: GameEvent, state: GameState) -> list[StructuredFact]:
    speaker = event.payload.get("speaker", "?")
    text = event.payload.get("text", "")
    claims = event.payload.get("claims", [])
    facts: list[StructuredFact] = [StructuredFact(
        fact_type="speech",
        source_player=speaker,
        value=text[:500],
        phase=event.payload.get("phase", ""),
        day=event.payload.get("day_number", 0),
    )]
    for claim in claims:
        claim_type = claim.get("type", "claim")
        facts.append(StructuredFact(
            fact_type=f"claimed_{claim_type}",
            source_player=speaker,
            target_player=claim.get("target"),
            value=claim.get("value", ""),
            day=event.payload.get("day_number", 0),
        ))
    if not claims:
        facts.extend(_infer_claims_from_text(
            speaker=speaker,
            text=text,
            day=event.payload.get("day_number", 0),
        ))
    return facts


def _infer_claims_from_text(*, speaker: str, text: str, day: int) -> list[StructuredFact]:
    facts: list[StructuredFact] = []
    role_patterns = {
        "seer": ("我是预言家", "我跳预言家", "我认预言家", "我悍跳预言家"),
        "witch": ("我是女巫", "我认女巫"),
        "hunter": ("我是猎人", "我认猎人"),
        "villager": ("我是村民", "我是民", "我认民"),
    }
    for role, patterns in role_patterns.items():
        if any(pattern in text for pattern in patterns):
            facts.append(StructuredFact(
                fact_type="claimed_role",
                source_player=speaker,
                value=role,
                day=day,
            ))
    if _has_explicit_wolf_self_claim(text):
        facts.append(StructuredFact(
            fact_type="claimed_role",
            source_player=speaker,
            value="werewolf",
            day=day,
        ))

    # --- H-6: 先收集 seer_check_claim 匹配的 span，避免后续重复匹配 ---
    self_seer_context = _has_self_seer_context(text)
    seer_spans: list[tuple[int, int]] = []
    for match in re.finditer(r"(?:查验|验了?|验人)\s*(p\d{2})\s*(?:是|为)?\s*(狼人|查杀|狼|wolf)", text):
        if self_seer_context and not _is_third_party_seer_report(text, match.start(1)):
            facts.append(StructuredFact(
                fact_type="seer_check_claim",
                source_player=speaker,
                target_player=match.group(1),
                value="wolf",
                day=day,
                metadata={"claim_type": "seer_wolf_check"},
            ))
            seer_spans.append(match.span())

    for match in re.finditer(
        r"(?:查验|验了?|验人)\s*(p\d{2})\s*(?:[,，、]\s*)?"
        r"(?:结果\s*)?(?:是|为)\s*(好人|金水|good)",
        text,
    ):
        if self_seer_context and not _is_third_party_seer_report(text, match.start(1)):
            facts.append(StructuredFact(
                fact_type="seer_check_claim",
                source_player=speaker,
                target_player=match.group(1),
                value="good",
                day=day,
                metadata={"claim_type": "seer_good_check"},
            ))

    for match in re.finditer(r"(查验|验了|验人)?\s*(p\d{2})\s*(是|为)?\s*(狼人|查杀)", text):
        # 跳过已被 seer_check_claim 覆盖的区间
        if any(match.start() >= s and match.end() <= e for s, e in seer_spans):
            continue
        if _is_third_party_seer_report(text, match.start(2)):
            continue
        facts.append(StructuredFact(
            fact_type="claimed_suspect",
            source_player=speaker,
            target_player=match.group(2),
            value="wolf",
            day=day,
        ))
    # --- H-7: 右侧分支 p\d{2} 也放入捕获组 ---
    for match in re.finditer(r"(保|金水|好人)\s*(p\d{2})|(p\d{2})\s*(是|为)?\s*(金水|好人)", text):
        target_group = next(
            (index for index, group in enumerate(match.groups(), 1)
             if group and re.fullmatch(r"p\d{2}", group)),
            None,
        )
        if target_group is not None:
            target = match.group(target_group)
            if target and not _is_third_party_seer_report(text, match.start(target_group)):
                facts.append(StructuredFact(
                    fact_type="claimed_good",
                    source_player=speaker,
                    target_player=target,
                    value="good",
                    day=day,
                ))

    # Badge flow: 支持紧凑格式和逐行列出的 N2/N3 验人计划。
    for targets in _extract_badge_flow_targets(text):
        facts.append(StructuredFact(
            fact_type="badge_flow_claim",
            source_player=speaker,
            target_player=targets[0],
            day=day,
            night=0,
            phase="",
            value="badge_flow",
            metadata={"badge_flow_order": targets},
        ))

    # Gold claim: p05是金水, 给p05发金水
    gold_matches = [
        *re.finditer(r"(p\d{2})\s*(?:是金水|金水)", text),
        *re.finditer(r"给\s*(p\d{2})\s*(?:发)?金水", text),
    ]
    gold_matches.sort(key=lambda match: match.start(1))
    seen_gold_targets: set[int] = set()
    for match in gold_matches:
        target = match.group(1)
        target_start = match.start(1)
        if target_start in seen_gold_targets:
            continue
        seen_gold_targets.add(target_start)
        if self_seer_context and not _is_third_party_seer_report(
            text, target_start
        ):
            facts.append(StructuredFact(
                fact_type="seer_check_claim",
                source_player=speaker,
                target_player=target,
                day=day,
                night=0,
                phase="",
                value="good",
                metadata={"claim_type": "gold_claim"},
            ))

    return facts


def _has_explicit_wolf_self_claim(text: str) -> bool:
    if re.search(r"我(?:是|认)(?:狼人|狼)(?:$|[，。！？；、,\s])", text):
        return True
    return bool(re.search(r"我(?:要)?自爆(?:$|[，。！？；、,\s])", text))


def _has_self_seer_context(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if any(
        pattern in compact
        for pattern in ("我是预言家", "我跳预言家", "我认预言家", "我悍跳预言家")
    ):
        return True
    return bool(
        re.search(
            r"(?:我|昨晚|今晚|夜里|第[0-9一二三四五六七八九十]+夜).{0,8}(?:查验|查了?|验了?|验人)",
            compact,
        )
    )


def _is_third_party_seer_report(text: str, target_start: int) -> bool:
    """判断目标查验是否属于第三方转述，调用方统一传目标起点。"""
    if target_start < 0:
        return False
    # 只截取目标前的有限窗口，避免无标点长发言中每个目标都重复扫描全文。
    context_start = max(0, target_start - _THIRD_PARTY_CONTEXT_WINDOW)
    context = text[context_start:target_start]
    boundary_offset = max(
        context.rfind(mark)
        for mark in ("。", "！", "？", "!", "?", "；", ";", "\n", "，", ",")
    )
    if boundary_offset >= 0:
        boundary = context_start + boundary_offset
        prefix = context[boundary_offset + 1:]
    else:
        boundary = context_start - 1
        prefix = context
    prefix = re.sub(r"\s+", "", prefix)
    direct_self = re.search(
        r"(?:我(?:昨晚|今晚|夜里|首夜|刚刚)?(?:查验|查了?|验了?|验人|给)|给)$",
        prefix,
    )
    marker_in_clause = _contains_third_party_report_marker(prefix)
    if direct_self:
        # 同一子句已有“p01说/报”是被转述的第一人称；独立的“我验了”保留。
        if marker_in_clause:
            return True
        # 跨逗号引语的开头带引号时，前一子句的说话者仍然负责归因。
        if prefix.startswith(("“", '"', "「", "『")):
            return _previous_clause_has_marker(text, boundary)
        # 无引号的“p01说，我验了...”同样是 p01 的第一人称转述；
        # 带完整结果的“p01报p02查杀，我验了...”则保留当前玩家自述。
        return _previous_clause_has_marker(text, boundary, terminal_only=True)
    # 有明确玩家编号的“p02报/说/验了...”是转述；“你跳预言家说验了..."
    # 也属于对他人查验的描述，即使编号出现在前一个姓名子句中。
    if marker_in_clause:
        return True

    # “p01说，昨晚验了p02”中编号和查验动词跨逗号，检查紧邻的前一子句。
    return _previous_clause_has_marker(text, boundary)


def _previous_clause_has_marker(
    text: str, boundary: int, *, terminal_only: bool = False
) -> bool:
    if boundary < 0 or text[boundary] not in "，,":
        return False
    context_start = max(0, boundary - _THIRD_PARTY_CONTEXT_WINDOW)
    context = text[context_start:boundary]
    previous_offset = max(
        context.rfind(mark)
        for mark in ("。", "！", "？", "!", "?", "；", ";", "\n", "，", ",")
    )
    previous = re.sub(r"\s+", "", context[previous_offset + 1:])
    if terminal_only:
        return bool(_THIRD_PARTY_TERMINAL_MARKER_RE.search(previous))
    return _contains_third_party_report_marker(previous)


_THIRD_PARTY_MARKER_WINDOW = 120
_THIRD_PARTY_CONTEXT_WINDOW = 180
_THIRD_PARTY_REPORT_MARKER_RE = re.compile(
    rf"(?:"
    rf"(?:p\d{{2}}|你).{{0,{_THIRD_PARTY_MARKER_WINDOW}}}(?:"
    r"(?:报|说|称|讲|表示)了?(?=[:：]*[“\"「『]?(?:我|他|她|验|查|p\d{2}))"
    r")"
    rf"|(?:p\d{{2}}|你)(?:给|发)(?=(?:p\d{{2}}|我|他|她|金水|查杀|好人))"
    rf"|(?:p\d{{2}}|你)(?:验了?|查(?:了|验)?)(?=(?:p\d{{2}}|我))"
    rf")"
)
_THIRD_PARTY_TERMINAL_MARKER_RE = re.compile(
    rf"(?:p\d{{2}}|你).{{0,{_THIRD_PARTY_MARKER_WINDOW}}}(?:报|说|称|讲|表示)了?$"
)
_BADGE_FLOW_PLAN_LINE_RE = re.compile(
    r"^\s*[-*]?\s*"
    r"(?:第[一二三四五六七八九十0-9]+夜\s*)?N\d+\s*"
    r"[,:：，]?\s*(?:我\s*)?(?:计划\s*)?"
    r"(?:查验|验|查)\s*(p\d{2})(?![A-Za-z0-9_])"
)


def _contains_third_party_report_marker(prefix: str) -> bool:
    if not prefix:
        return False
    compact_prefix = re.sub(r"\s+", "", prefix)
    # 追加占位目标即可识别
    # “p01说p02/给p02”这种 marker 在目标起点前结束的紧凑写法，
    # 同时避免把目标之后的下一个“给/发”倒灌到当前目标。
    marker_match = _THIRD_PARTY_REPORT_MARKER_RE.search(compact_prefix + "p00")
    # 目标自身的 pXX 不能被当成报告者；报告者必须位于 target 起点之前。
    if marker_match and marker_match.start() < len(compact_prefix):
        return True
    return bool(_THIRD_PARTY_TERMINAL_MARKER_RE.search(compact_prefix))


def _extract_badge_flow_targets(text: str) -> list[list[str]]:
    """提取警徽流目标，限制在标记后的紧凑文本或连续计划行内。"""
    flows: list[list[str]] = []
    for marker in re.finditer(r"警徽流", text):
        tail = text[marker.end():]
        lines = tail.splitlines()
        if not lines:
            continue

        compact_match = re.match(
            r"^[\s:：]*(p\d{2}(?![A-Za-z0-9_])"
            r"(?:(?:\s+|[，、,]\s*)p\d{2}(?![A-Za-z0-9_]))*)",
            lines[0],
        )
        targets = (
            re.findall(r"p\d{2}(?![A-Za-z0-9_])", compact_match.group(1))
            if compact_match else []
        )
        if not targets:
            ordered_match = re.match(
                r"^[\s:：]*先\s*(?:验|查)?\s*(p\d{2}(?![A-Za-z0-9_]))"
                r"\s*[，、,]?\s*后\s*(?:验|查)?\s*(p\d{2}(?![A-Za-z0-9_]))"
                r"(?:\s*[，、,]?\s*再\s*(?:验|查)?\s*(p\d{2}(?![A-Za-z0-9_])))?"
                r"(?=$|[\s，、。！？；])",
                lines[0],
            )
            if ordered_match:
                targets = [group for group in ordered_match.groups() if group]
        if not targets:
            plan_lines = list(lines)
            plan_lines[0] = plan_lines[0].lstrip(" \t:：")
            if not plan_lines[0]:
                plan_lines = plan_lines[1:]
            for line in plan_lines:
                if not line.strip():
                    break
                plan_match = _BADGE_FLOW_PLAN_LINE_RE.match(line)
                if plan_match is None:
                    break
                targets.append(plan_match.group(1))
        if targets:
            flows.append(targets)
    return flows


def _extract_vote(event: GameEvent, state: GameState) -> list[StructuredFact]:
    voter = event.payload.get("voter", "?")
    target = event.payload.get("target", "?")
    return [StructuredFact(
        fact_type="vote",
        source_player=voter,
        target_player=target,
        day=event.payload.get("day_number", 0),
        value="voted_for",
    )]


def _extract_seer_check(event: GameEvent, seer_id: str) -> list[StructuredFact]:
    """E2 (post-review-v2): seer_id 由外部 dispatch 注入，函数体不做全表扫。"""
    target = event.payload.get("target_id", "?")
    alignment = event.payload.get("alignment") or event.payload.get("result", "?")
    night = event.payload.get("night_number", 0)
    return [StructuredFact(
        fact_type="seer_check",
        source_player=seer_id,
        target_player=target,
        night=night,
        value=str(alignment),
    )]


def _extract_sheriff_no_election(event: GameEvent, state: GameState) -> list[StructuredFact]:
    return [StructuredFact(
        fact_type="sheriff_no_election",
        value="no_sheriff_elected",
    )]


# Registry: event type → extractor
_EXTRACTORS: dict[str, Any] = {
    "player_died": _extract_player_died,
    "idiot_revealed": _extract_idiot_revealed,
    "player_exiled": _extract_player_exiled,
    "werewolf_self_destructed": _extract_werewolf_self_destructed,
    "hybrid_master_chosen": _extract_hybrid_master_chosen,
    "sheriff_elected": _extract_sheriff_elected,
    "sheriff_registered": _extract_sheriff_registered,
    "sheriff_withdraw": _extract_sheriff_withdraw,
    "sheriff_vote_tie": _extract_sheriff_vote_tie,
    "sheriff_vote_tie_first": _extract_sheriff_vote_tie_first,
    "badge_transferred": _extract_badge_transferred,
    "badge_torn": _extract_badge_torn,
    "witch_antidote_used": _extract_witch_antidote_used,
    "witch_poison_used": _extract_witch_poison_used,
    "wolf_kill_selected": _extract_wolf_kill_selected,
    "wolf_no_kill_declared": _extract_wolf_no_kill,
    "wolf_no_kill_timeout": _extract_wolf_no_kill,
    "speech": _extract_speech,
    "sheriff_speech": _extract_speech,
    "vote": _extract_vote,
    "seer_check": _extract_seer_check,
    "sheriff_no_election": _extract_sheriff_no_election,
}

_LAST_WORDS_EVENT_TYPES = frozenset({
    "exile_last_words",
    "night_death_last_words",
})
_CONDITIONAL_SPEECH_EVENT_TYPES = _LAST_WORDS_EVENT_TYPES | frozenset({
    "sheriff_pk_speech",
    "tie_pk_speech",
})

_CLAIM_FACT_TYPES = frozenset({
    "claimed_role",
    "claimed_good",
    "claimed_suspect",
    "seer_check_claim",
    "badge_flow_claim",
})


def _fact_provenance(fact: StructuredFact, event: GameEvent) -> dict[str, str]:
    """按事实类型和来源事件标记声明权威性与证据种类。"""
    is_claim = fact.fact_type in _CLAIM_FACT_TYPES or fact.fact_type.startswith("claimed_")
    return {
        "authority": "player_claim" if is_claim else "engine",
        "support_kind": (
            "last_words"
            if event.type in {"exile_last_words", "night_death_last_words"}
            else "public_speech"
            if event.type in {
                "speech",
                "sheriff_speech",
                "sheriff_pk_speech",
                "tie_pk_speech",
            }
            else "executed_action"
        ),
    }


def _has_usable_speech_payload(event: GameEvent) -> bool:
    """判断条件式公开发言事件是否包含可安全解析的说话者和文本。"""
    speaker = event.payload.get("speaker")
    text = event.payload.get("text")
    return (
        isinstance(speaker, str)
        and bool(speaker.strip())
        and isinstance(text, str)
    )


def extract_facts(event: GameEvent, state: GameState) -> list[StructuredFact]:
    """Extract structured facts from a single GameEvent."""
    extractor = _EXTRACTORS.get(event.type)
    if (
        extractor is None
        and event.type in _CONDITIONAL_SPEECH_EVENT_TYPES
        and _has_usable_speech_payload(event)
    ):
        extractor = _extract_speech
    if extractor is None:
        return [_attach_event_metadata(StructuredFact(
            fact_type=event.type,
            value=str(event.payload)[:200],
        ), event, state)]
    # E2 (post-review-v2): seer_check 走特殊路径，seer_id 在 dispatch 入口预计算
    if event.type == "seer_check":
        seer_id = next(
            (pid for pid, p in state.players.items() if p.role == "seer"),
            "?",
        )
        return [
            _attach_event_metadata(fact, event, state)
            for fact in _extract_seer_check(event, seer_id)
        ]
    return [_attach_event_metadata(fact, event, state) for fact in extractor(event, state)]


def _attach_event_metadata(
    fact: StructuredFact,
    event: GameEvent,
    state: GameState,
) -> StructuredFact:
    """把来源事件和 visibility 写入事实 metadata，供可见性策略使用。"""
    metadata = dict(fact.metadata)
    metadata.update(_fact_provenance(fact, event))
    metadata.setdefault("source_event", event.type)
    visibility = event_visibility(event)
    metadata.setdefault("visibility", visibility.value)
    if visibility in {
        EventVisibility.ACTOR_PRIVATE,
        EventVisibility.PRIVATE,
        EventVisibility.PLAYER_ONLY,
    }:
        metadata.setdefault(
            "visibility_actor_id",
            event.payload.get("visibility_actor_id") or fact.source_player,
        )
    elif visibility is EventVisibility.ROLE_PRIVATE:
        source = state.players.get(fact.source_player or "")
        metadata.setdefault(
            "visibility_role",
            event.payload.get("visibility_role") or (source.role if source else None),
        )
    return replace(fact, metadata=metadata)


def build_world_state(state: GameState) -> StructuredWorldState:
    """Build complete structured world state from GameState events."""
    ws = StructuredWorldState()
    for event in state.events:
        facts = extract_facts(event, state)
        ws.extend(facts)
    return ws
