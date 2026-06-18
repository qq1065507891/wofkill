"""Post-game attribution engine for evaluation feedback traces.

Annotates cognition-module exposures (rag / reflection / possible_worlds /
simulator) with cited_by_decision / aligned_with_decision / harmful_transfer,
and runs the consistency judge per trace with rebuilt public_facts. Pure
post-game: no runtime change, no audit payload growth.
"""

from __future__ import annotations

import dataclasses
import re as _re
from dataclasses import asdict
from typing import Any, Mapping

from werewolf_agent.core.models import GameEvent, GameState, PlayerState
from werewolf_agent.cognition.world_state import build_world_state
from werewolf_agent.cognition.visibility import VisibilityPolicy
from werewolf_agent.evaluation.feedback_schemas import DecisionOutcome, ModuleExposure
from werewolf_agent.evaluation.llm_judge import judge_speech_consistency
from werewolf_agent.evaluation.text_similarity import jaccard, tokenize


_RAG_TEXT_FIELDS = (
    "title",
    "situation_signature",
    "transferable_lesson",
    "recommended_action",
    "misuse_risk",
)
_REFLECTION_TEXT_FIELDS = (
    "theme",
    "lesson",
    "recommended_action",
    "misuse_risk",
)


class AttributionTextResolver:
    """Resolve compact exposure records to the prompt-safe card text.

    Production wiring wraps ``RAGRepository.get(entry_id)`` /
    ``ReflectionMemory.all_v2_entries()`` (or a service-level cache). Tests
    pass fixture dicts. Returns ``None`` when the entry cannot be resolved —
    the engine then marks that exposure ``MetricSupport.UNSUPPORTED``.
    """

    def __init__(
        self,
        *,
        rag_entries: Mapping[str, Mapping[str, Any]] | None = None,
        reflection_entries: Mapping[str, Mapping[str, Any]] | None = None,
        rag_provider=None,
        reflection_provider=None,
    ) -> None:
        self._rag_entries = rag_entries
        self._reflection_entries = reflection_entries
        self._rag_provider = rag_provider
        self._reflection_provider = reflection_provider

    def rag_text(self, exposure: ModuleExposure) -> str | None:
        data = self._resolve("rag", exposure.item_id)
        if not data:
            return None
        return " ".join(str(data.get(f, "") or "") for f in _RAG_TEXT_FIELDS).strip()

    def reflection_text(self, exposure: ModuleExposure) -> str | None:
        data = self._resolve("reflection", exposure.item_id)
        if not data:
            return None
        card = data.get("prompt_card", data)
        return " ".join(str(card.get(f, "") or "") for f in _REFLECTION_TEXT_FIELDS).strip()

    def _resolve(self, module: str, item_id: str) -> Mapping[str, Any] | None:
        if module == "rag":
            if self._rag_provider is not None:
                return self._rag_provider(item_id)
            if self._rag_entries is not None:
                return self._rag_entries.get(item_id)
        elif module == "reflection":
            if self._reflection_provider is not None:
                return self._reflection_provider(item_id)
            if self._reflection_entries is not None:
                return self._reflection_entries.get(item_id)
        return None


_CITED_THRESHOLD = 0.15
_ACTION_VERBS = ("先", "不要", "避免", "必须", "优先", "核验", "比较", "列")


def speech_from_decision(decision) -> str:
    """Read the public speech text from DecisionSnapshot.raw.

    EvaluationTrace does not retain a standalone parsed_action; speech lives
    in decision.raw (set by EvaluationTraceBuilder._decision_snapshot).
    """
    if decision is None:
        return ""
    raw = decision.raw or {}
    return str(raw.get("speech") or raw.get("public_story") or "")


def exposure_representative_text(
    exposure: ModuleExposure,
    resolver: AttributionTextResolver,
) -> str | None:
    """Prompt-safe representative text for an exposure, or None if unresolved.

    None signals the engine to mark the exposure MetricSupport.UNSUPPORTED
    (RAG/reflection whose card text cannot be resolved post-game).
    possible_worlds/simulator always resolve from their structured metadata.
    """
    module = exposure.module
    meta = exposure.metadata
    if module == "rag":
        return resolver.rag_text(exposure)
    if module == "reflection":
        return resolver.reflection_text(exposure)
    if module == "possible_worlds":
        assignments = meta.get("key_assignments") or {}
        return " ".join(f"{pid}={role}" for pid, role in assignments.items())
    if module == "simulator":
        affected = meta.get("affected_players") or []
        return f"{exposure.item_id} {' '.join(str(p) for p in affected)}".strip()
    return None


def cited(decision, exposure: ModuleExposure, resolver: AttributionTextResolver) -> bool:
    exp_text = exposure_representative_text(exposure, resolver)
    if not exp_text:
        return False  # unresolved → engine marks UNSUPPORTED; not cited
    decision_text = f"{decision.reason or ''} {speech_from_decision(decision)}"
    return jaccard(decision_text, exp_text) >= _CITED_THRESHOLD


_PLAYER_ID_RE = _re.compile(r"p\d{1,3}")
_WOLF_ROLES = frozenset({"werewolf", "wolf"})


def _reason_players(decision) -> set[str]:
    if decision is None or not decision.reason:
        return set()
    return set(_PLAYER_ID_RE.findall(decision.reason))


def aligned(
    decision,
    exposure: ModuleExposure,
    faction: str,
    resolver: AttributionTextResolver | None = None,
) -> bool:
    """Per-module direction rule: did the decision follow the exposure?"""
    module = exposure.module
    meta = exposure.metadata
    target = decision.target_id if decision else None
    mentioned = _reason_players(decision)
    relevant_players = ({target} if target else set()) | mentioned

    if module == "possible_worlds":
        assignments = meta.get("key_assignments") or {}
        wolves = {pid for pid, role in assignments.items() if role in _WOLF_ROLES}
        return bool(relevant_players & wolves)
    if module == "simulator":
        affected = set(meta.get("affected_players") or [])
        return bool(relevant_players & affected)
    if module in ("rag", "reflection"):
        if resolver is None:
            return False
        exp_text = exposure_representative_text(exposure, resolver)
        if not exp_text:
            return False
        reason = decision.reason or "" if decision else ""
        # the decision adopted a recommended action verb that the card also mentions
        return any(verb in reason and verb in exp_text for verb in _ACTION_VERBS)
    return False


_HARMFUL_ACTION_TYPES = frozenset({"vote", "use_poison", "hunter_shot", "sheriff_vote"})


def trace_outcome_is_bad(trace) -> bool:
    """Did this trace's decision produce a bad outcome? Uses the signals
    revived by monitoring-closure-fix (legal/leak) plus trace-level vote/
    wrong-target checks that need faction + action_type."""
    outcome = trace.outcome
    if outcome is None:
        return False
    if outcome.legal is False:
        return True
    if outcome.leaked_hidden_info:
        return True
    decision = trace.decision
    action_type = decision.action_type if decision else ""
    if trace.faction == "good" and action_type == "vote":
        if outcome.vote_hit_wolf is False:
            return True
    if action_type in _HARMFUL_ACTION_TYPES and trace.faction == "good":
        if outcome.target_faction == "good":
            return True
    return False


def is_harmful(exposure: ModuleExposure, trace) -> bool:
    return bool(
        exposure.cited_by_decision
        and exposure.aligned_with_decision
        and trace_outcome_is_bad(trace)
    )


def is_beneficial(exposure: ModuleExposure, trace) -> bool:
    return bool(
        exposure.cited_by_decision
        and exposure.aligned_with_decision
        and not trace_outcome_is_bad(trace)
    )


# ---------------------------------------------------------------------------
# Judge producer — rebuild public_facts, derive public_claim, run judge
# ---------------------------------------------------------------------------

_JUDGE_SENTINEL = "judge_consistency_scored"
_PHASE_ORDER = {
    "setup": 0,
    "night": 10,
    "wolf": 11,
    "witch": 12,
    "seer": 13,
    "sheriff": 20,
    "sheriff_speech": 21,
    "speech": 30,
    "day": 35,
    "day_vote": 40,
    "vote": 40,
}
_ROLE_CLAIMS = {
    "werewolf": ("我是狼人", "我是狼", "我们狼队", "狼队视角"),
    "seer": ("我是预言家", "我跳预言家", "认预言家"),
    "witch": ("我是女巫", "我认女巫"),
    "hunter": ("我是猎人", "我认猎人"),
    "villager": ("我是村民", "我是民", "我认民"),
}


def _event_payload(entry: dict) -> dict:
    payload = entry.get("payload")
    return dict(payload) if isinstance(payload, dict) else {}


def _event_day(entry: dict) -> int:
    payload = _event_payload(entry)
    value = entry.get("day_number", payload.get("day_number", 0))
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _event_phase(entry: dict) -> str:
    payload = _event_payload(entry)
    return str(entry.get("phase") or payload.get("phase") or "")


def _phase_rank(phase: str) -> int:
    # Unknown phases default to -1 (earliest): conservative inclusion is safer
    # than exclusion — an extra visible fact cannot create a false judge issue,
    # but a missing fact can (speech referencing a real-but-filtered fact).
    return _PHASE_ORDER.get(str(phase or ""), -1)


def _entry_is_in_trace_prefix(entry: dict, trace) -> bool:
    day = _event_day(entry)
    trace_day = trace.day_number or 0
    if day > trace_day:
        return False
    if day == trace_day and _phase_rank(_event_phase(entry)) > _phase_rank(trace.phase):
        return False
    return True


def _game_event_from_entry(entry: dict) -> GameEvent:
    payload = _event_payload(entry)
    if "day_number" not in payload and entry.get("day_number") is not None:
        payload["day_number"] = entry.get("day_number")
    if "phase" not in payload and entry.get("phase"):
        payload["phase"] = entry.get("phase")
    if "speaker" not in payload and payload.get("player_id"):
        payload["speaker"] = payload["player_id"]
    return GameEvent(type=str(entry.get("type") or ""), payload=payload)


def rebuild_visible_facts(result, trace):
    """Rebuild the public facts visible to ``trace``'s player at decision time.

    Filter result.event_log to the decision prefix using payload/top-level
    day_number and same-day phase rank, convert dicts to GameEvent, build a
    temporary GameState with concrete PlayerState objects, run
    build_world_state, then filter to what this player could see.
    """
    events = []
    for entry in result.event_log:
        if not isinstance(entry, dict):
            continue
        if not _entry_is_in_trace_prefix(entry, trace):
            continue
        events.append(_game_event_from_entry(entry))
    players = {
        pid: PlayerState(id=pid, role=role, faction=result.player_factions.get(pid))
        for pid, role in result.player_roles.items()
    }
    state = GameState(
        game_id=result.game_id,
        ruleset_id=result.ruleset_id,
        players=players,
        phase=trace.phase,
        day_number=trace.day_number,
        night_number=trace.night_number,
        events=events,
    )
    world_state = build_world_state(state)
    return VisibilityPolicy().filter_visible_facts(world_state, trace.player_id, trace.role)


def derive_public_claim(result, trace) -> str:
    """Return the player's latest prior public role claim, if one exists."""
    latest = ""
    for entry in result.event_log:
        if not isinstance(entry, dict):
            continue
        if not _entry_is_in_trace_prefix(entry, trace):
            continue
        if str(entry.get("type") or "") not in {"speech", "sheriff_speech"}:
            continue
        payload = _event_payload(entry)
        speaker = str(payload.get("speaker") or payload.get("player_id") or "")
        if speaker != trace.player_id:
            continue
        for claim in payload.get("claims", []) or []:
            if isinstance(claim, dict) and claim.get("type") == "role" and claim.get("value"):
                latest = str(claim["value"]).lower()
        # Text markers are a fallback ONLY when the event had no structured role claim.
        if not any(
            isinstance(c, dict) and c.get("type") == "role" and c.get("value")
            for c in (payload.get("claims") or [])
        ):
            text = str(payload.get("text") or payload.get("speech") or "")
            for role, markers in _ROLE_CLAIMS.items():
                if any(marker in text for marker in markers):
                    latest = role
    return latest


def judge_trace(trace, result):
    """Run the consistency judge on a trace with rebuilt public_facts.

    Returns a NEW EvaluationTrace whose outcome carries the consistency score
    and the ``judge_consistency_scored`` sentinel in outcome_refs. Traces with
    no speech/reason are returned unchanged (not judged).
    """
    speech = speech_from_decision(trace.decision)
    reason = trace.decision.reason if trace.decision else ""
    if not (speech.strip() or (reason or "").strip()):
        return trace
    visible_facts = rebuild_visible_facts(result, trace)
    public_facts_payload = [asdict(f) for f in visible_facts]
    context = {
        "role": trace.role,
        "faction": trace.faction,
        "public_claim": derive_public_claim(result, trace),
        "public_facts": public_facts_payload,
        "visible_facts": public_facts_payload,
    }
    action = {"speech": speech, "reason": reason}
    judgment = judge_speech_consistency(context, action)
    old_outcome = trace.outcome or DecisionOutcome()
    new_refs = list((old_outcome.outcome_refs if old_outcome else []) or [])
    if _JUDGE_SENTINEL not in new_refs:
        new_refs.append(_JUDGE_SENTINEL)
    new_outcome = dataclasses.replace(
        old_outcome,
        local_quality_score=judgment.consistency_score,
        outcome_refs=new_refs,
    )
    return dataclasses.replace(trace, outcome=new_outcome)
