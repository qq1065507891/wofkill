"""Schemas for evaluation feedback-loop traces and candidates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MetricSupport(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ModuleExposure:
    module: str
    item_id: str
    rank: int = 0
    score: float = 0.0
    prompt_visible: bool = False
    cited_by_decision: bool = False
    aligned_with_decision: bool = False
    support: MetricSupport = MetricSupport.SUPPORTED
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionSnapshot:
    action_type: str = ""
    target_id: str | None = None
    reason: str = ""
    confidence: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionOutcome:
    legal: bool | None = None
    target_role: str = ""
    target_faction: str = ""
    target_alive: bool | None = None
    vote_hit_wolf: bool | None = None
    protected_good: bool | None = None
    leaked_hidden_info: bool = False
    contradicted_visible_evidence: bool = False
    local_quality_score: float = 0.0
    outcome_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationTrace:
    trace_id: str
    game_id: str
    player_id: str
    role: str
    faction: str
    phase: str
    day_number: int = 0
    night_number: int = 0
    task_type: str = ""
    legal_actions: list[str] = field(default_factory=list)
    legal_targets: list[str] = field(default_factory=list)
    module_exposures: list[ModuleExposure] = field(default_factory=list)
    decision: DecisionSnapshot | None = None
    outcome: DecisionOutcome | None = None
    source_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FailureDiagnosis:
    diagnosis_id: str
    trace_id: str
    category: str
    severity: str
    primary_module: str = ""
    supporting_modules: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass(frozen=True)
class ImprovementCandidate:
    candidate_id: str
    source_diagnosis_ids: list[str]
    target_module: str
    operation: str
    priority: str
    prompt_safe_payload: dict[str, Any]
    audit_evidence: dict[str, Any] = field(default_factory=dict)
    moderator_notes: str = ""
    safety_notes: list[str] = field(default_factory=list)
    regression_seed_set: list[int] = field(default_factory=list)
    review_status: str = "pending"


_PLAYER_ID_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:p\d{1,3}|player[_-]?\d{1,3}|agent[_-]?\d{1,3})(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_PROMPT_UNSAFE_TOKENS = (
    "真实身份",
    "底牌",
    "target_role",
    "target_faction",
    "true_world_rank",
    "actual_role",
    "ground_truth",
)


def validate_candidate_prompt_safe(candidate: ImprovementCandidate) -> bool:
    """Return False when candidate prompt-visible payload contains audit truth."""
    text = json.dumps(candidate.prompt_safe_payload, ensure_ascii=False, sort_keys=True)
    if _PLAYER_ID_RE.search(text):
        return False
    lowered = text.lower()
    return not any(token.lower() in lowered for token in _PROMPT_UNSAFE_TOKENS)

