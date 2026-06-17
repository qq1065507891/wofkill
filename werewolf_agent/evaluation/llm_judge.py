"""Semantic consistency judge for player actions.

The live game still uses deterministic validators for retries. This module is
for offline evaluation and regression gates: it checks whether a speech/action
stays consistent with the visible context before a future LLM-backed judge is
plugged in behind the same interface.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from werewolf_agent.cognition.public_evidence import PublicEvidenceIndex
from werewolf_agent.cognition.world_state import StructuredFact


_PLAYER_RE = re.compile(r"p\d{1,3}", re.IGNORECASE)

_SEER_TERMS = ("预言家", "seer")
_BLACK_CHECK_TERMS = (
    "查杀",
    "被查杀",
    "black-check",
    "black checked",
    "black_checked",
    "called wolf",
)
_GOLD_WATER_TERMS = ("金水", "good check", "checked good")
_WOLF_IDENTITY_TERMS = ("我是狼人", "我是狼", "我作为狼人", "我们狼队", "我的狼队")
_GOOD_WOLF_TASK_TERMS = ("狼队任务", "我的狼队", "我们狼队", "狼队友")


@dataclass(frozen=True)
class JudgeIssue:
    """One actionable semantic issue found by the judge."""

    code: str
    dimension: str
    detail: str
    severity: str = "error"

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpeechConsistencyJudgment:
    """Result returned by ``judge_speech_consistency``."""

    ok: bool
    consistency_score: float
    issues: list[JudgeIssue] = field(default_factory=list)
    checked_dimensions: tuple[str, ...] = (
        "identity_consistency",
        "faction_task_consistency",
        "public_fact_reference",
    )

    def has_issue(self, code: str) -> bool:
        return any(issue.code == code for issue in self.issues)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "consistency_score": self.consistency_score,
            "issues": [issue.to_json_dict() for issue in self.issues],
            "checked_dimensions": list(self.checked_dimensions),
        }


def judge_speech_consistency(
    context: dict[str, Any],
    action: dict[str, Any],
) -> SpeechConsistencyJudgment:
    """Judge whether a speech/action is semantically consistent.

    The checks intentionally use only the passed ``context`` and ``action`` so
    this can run offline on replay traces without querying hidden game state.
    """

    speech = _action_text(action)
    public_text = _public_context_text(context)
    public_index = _public_evidence_index(context)
    issues: list[JudgeIssue] = []

    if not speech:
        issues.append(JudgeIssue(
            code="empty_action",
            dimension="action_content",
            detail="Action has no speech/reason text to judge.",
        ))
    else:
        issues.extend(_identity_issues(context, speech))
        issues.extend(_faction_task_issues(context, speech))
        issues.extend(_public_fact_reference_issues(public_index, public_text, speech))

    dimensions_with_issues = {issue.dimension for issue in issues}
    score = _stable_score(1.0 - len(dimensions_with_issues) / 3.0)
    return SpeechConsistencyJudgment(
        ok=not issues,
        consistency_score=score,
        issues=issues,
    )


def _action_text(action: dict[str, Any]) -> str:
    parts = [
        action.get("speech"),
        action.get("reason"),
        action.get("public_story"),
    ]
    return " ".join(str(part).strip() for part in parts if str(part or "").strip())


def _public_context_text(context: dict[str, Any]) -> str:
    public_bits: list[Any] = []
    for key in (
        "public_facts",
        "public_summary",
        "public_record",
        "visible_facts",
        "recent_events",
    ):
        value = context.get(key)
        if value:
            public_bits.append(value)
    return json.dumps(public_bits, ensure_ascii=False).lower()


def _identity_issues(context: dict[str, Any], speech: str) -> list[JudgeIssue]:
    role = str(context.get("role", "")).lower()
    faction = str(context.get("faction", "")).lower()
    public_claim = str(context.get("public_claim", "")).lower()
    speech_text = speech.lower()

    real_wolf = role == "werewolf" or faction == "werewolf"
    claiming_wolf_publicly = any(term in speech for term in _WOLF_IDENTITY_TERMS)
    already_claimed_wolf = public_claim in {"werewolf", "wolf"}
    if real_wolf and claiming_wolf_publicly and not already_claimed_wolf:
        return [JudgeIssue(
            code="identity_consistency",
            dimension="identity_consistency",
            detail="Wolf speech leaks real identity while public claim is not wolf.",
        )]
    if not real_wolf and ("我是狼人" in speech or "我是狼" in speech_text):
        return [JudgeIssue(
            code="identity_consistency",
            dimension="identity_consistency",
            detail="Non-wolf role publicly claims to be wolf.",
        )]
    return []


def _faction_task_issues(context: dict[str, Any], speech: str) -> list[JudgeIssue]:
    faction = str(context.get("faction", "")).lower()
    if faction in {"good", "villager", "village"} and any(
        term in speech for term in _GOOD_WOLF_TASK_TERMS
    ):
        return [JudgeIssue(
            code="faction_task_consistency",
            dimension="faction_task_consistency",
            detail="Good-faction speech describes hidden wolf-team tasks.",
        )]
    return []


def _public_fact_reference_issues(
    public_index: PublicEvidenceIndex,
    public_text: str,
    speech: str,
) -> list[JudgeIssue]:
    issues: list[JudgeIssue] = []
    for player, concept, fragment in _extract_public_claims(speech):
        if not _public_text_supports(public_index, public_text, player, concept):
            issues.append(JudgeIssue(
                code="public_fact_reference",
                dimension="public_fact_reference",
                detail=(
                    f"Speech references public fact {concept!r} for {player}, "
                    f"but the visible context does not support it: {fragment!r}."
                ),
            ))
    return issues


def _public_evidence_index(context: dict[str, Any]) -> PublicEvidenceIndex:
    index = PublicEvidenceIndex()
    for item in _iter_public_items(context):
        if isinstance(item, StructuredFact):
            index.observe(item)
        elif isinstance(item, dict):
            fact = _fact_from_dict(item)
            if fact is not None:
                index.observe(fact)
        elif isinstance(item, str):
            for fact in _facts_from_text(item):
                index.observe(fact)
    return index


def _iter_public_items(context: dict[str, Any]) -> list[Any]:
    items: list[Any] = []
    for key in ("public_facts", "visible_facts", "recent_events"):
        value = context.get(key)
        if isinstance(value, list):
            items.extend(value)
        elif value:
            items.append(value)
    for key in ("public_summary", "public_record"):
        value = context.get(key)
        if value:
            items.append(value)
    return items


def _fact_from_dict(item: dict[str, Any]) -> StructuredFact | None:
    fact_type = item.get("fact_type") or item.get("type")
    if not fact_type:
        return None
    return StructuredFact(
        fact_type=str(fact_type),
        source_player=str(item.get("source_player") or item.get("source") or ""),
        target_player=str(item.get("target_player") or item.get("target") or ""),
        value=str(item.get("value") or item.get("result") or ""),
        day=int(item.get("day", 0) or 0),
    )


def _facts_from_text(text: str) -> list[StructuredFact]:
    facts: list[StructuredFact] = []
    for m in re.finditer(r"(p\d{1,3})\s+(?:black-checked|black checked|called wolf)\s+(p\d{1,3})", text, re.I):
        facts.append(StructuredFact(
            fact_type="seer_check_claim",
            source_player=m.group(1).lower(),
            target_player=m.group(2).lower(),
            value="wolf",
        ))
    for m in re.finditer(r"(p\d{1,3}).{0,8}(?:被查杀|查杀|是狼|狼人)", text, re.I):
        facts.append(StructuredFact(
            fact_type="seer_check_claim",
            target_player=m.group(1).lower(),
            value="wolf",
        ))
    for m in re.finditer(r"(?:金水|好人)\s*(p\d{1,3})|(p\d{1,3})\s*(?:是|为)?\s*(?:金水|好人)", text, re.I):
        target = next((g for g in m.groups() if g), "")
        if target:
            facts.append(StructuredFact(
                fact_type="seer_check_claim",
                target_player=target.lower(),
                value="good",
            ))
    for m in re.finditer(r"(p\d{1,3})\s+(?:claimed seer|跳预言家|认预言家|预言家)", text, re.I):
        facts.append(StructuredFact(
            fact_type="claimed_role",
            source_player=m.group(1).lower(),
            value="seer",
        ))
    return facts


def _extract_public_claims(speech: str) -> list[tuple[str, str, str]]:
    claims: list[tuple[str, str, str]] = []
    for match in re.finditer(
        r"(?:p\d{1,3}.{0,8}(?:被查杀|被黑|是狼|狼人|black-checked|black checked|called wolf)|"
        r"(?:查杀|黑)[:：]?\s*p\d{1,3}|p\d{1,3}.{0,8}(?:金水|好人|预言家|seer))",
        speech,
        re.I,
    ):
        frag = match.group(0)
        player_match = _PLAYER_RE.search(frag)
        if not player_match:
            continue
        player = player_match.group(0).lower()
        if any(term in frag for term in _BLACK_CHECK_TERMS) or "查杀" in frag or "黑" in frag:
            claims.append((player, "black_check", frag))
        elif any(term in frag for term in _GOLD_WATER_TERMS):
            claims.append((player, "gold_water", frag))
        elif any(term in frag.lower() for term in _SEER_TERMS):
            claims.append((player, "seer_claim", frag))
    return _dedupe_claims(claims)


def _dedupe_claims(claims: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str, str]] = []
    for player, concept, fragment in claims:
        key = (player, concept)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((player, concept, fragment))
    return deduped


def _public_text_supports(
    public_index: PublicEvidenceIndex,
    public_text: str,
    player: str,
    concept: str,
) -> bool:
    player = player.lower()
    if public_index.supports_reference(player, concept):
        return True
    if player not in public_text:
        return False
    return False


def _stable_score(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 4)
