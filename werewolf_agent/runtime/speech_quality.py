"""Public speech quality validation.

Validates that speeches include required components:
- Identity perspective / stance
- Suspicion target(s)
- Protection/trusted target(s)
- Vote leaning
- Evidence basis

High-pressure phases (sheriff, PK, seer) have stronger requirements.
"""

from __future__ import annotations

import re
from typing import Any


# Filler patterns that indicate low-quality speech
_FILLER_PATTERNS = [
    r"^再观察",
    r"^先听[后下]",
    r"^信息太少",
    r"^我没什么",
    r"^没什么可说",
    r"^好人.*?相信",
    r"^大家.*?相信.*?就行",
]

# Suspicion indicators
_SUSPICION_PATTERNS = [
    r"怀疑\s*(p\d{2})", r"我觉得\s*(p\d{2})\s*是狼",
    r"(p\d{2})\s*是狼", r"(p\d{2})\s*有问题",
    r"(p\d{2})\s*发言.*?(?:矛盾|不合理)", r"推\s*(p\d{2})",
    r"(p\d{2})\s*必须出局", r"投\s*(p\d{2})",
]

# Protection/trust indicators
_PROTECTION_PATTERNS = [
    r"信任\s*(p\d{2})", r"保\s*(p\d{2})",
    r"(p\d{2})\s*是好人", r"(p\d{2})\s*逻辑.*?(?:完整|合理)",
    r"金水\s*(p\d{2})", r"(p\d{2})\s*可信",
]

# Vote leaning
_VOTE_PATTERNS = [
    r"(?:投|投票|归票).*?(p\d{2})",
    r"倾向.*?(?:投|票).*?(p\d{2})",
    r"(p\d{2})\s*(?:必须|应该).*?出局",
]

# Evidence basis types
_EVIDENCE_PATTERNS = [
    (r"矛盾", "contradiction"),
    (r"前后不一", "contradiction"),
    (r"不合理", "unreasonable"),
    (r"查杀", "seer_check"),
    (r"查验", "seer_check"),
    (r"警徽流", "badge_flow"),
    (r"对跳", "counterclaim"),
    (r"票数", "vote_tally"),
    (r"之前说", "speech_quote"),
    (r"刚才说", "speech_quote"),
]

# Stance indicators
_STANCE_PATTERNS = [
    r"好人", r"我是.*?(?:好人|村民|预言家|女巫|猎人)",
    r"狼人阵营", r"我站.*?边",
]


def extract_speech_quality(text: str, phase: str = "") -> dict[str, Any]:
    """Extract speech quality components from text.

    Returns dict with: suspicion_targets, protection_targets,
    vote_leaning, evidence_bases, has_stance.
    """
    if not text or not text.strip():
        return {
            "suspicion_targets": [],
            "protection_targets": [],
            "vote_leaning": None,
            "evidence_bases": [],
            "has_stance": False,
        }

    suspicion = set()
    for pattern in _SUSPICION_PATTERNS:
        for m in re.finditer(pattern, text):
            suspicion.add(m.group(1))

    protection = set()
    for pattern in _PROTECTION_PATTERNS:
        for m in re.finditer(pattern, text):
            protection.add(m.group(1))

    vote_target = None
    for pattern in _VOTE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            vote_target = m.group(1)
            break

    evidence = []
    for pattern, basis_type in _EVIDENCE_PATTERNS:
        if re.search(pattern, text):
            evidence.append(basis_type)

    has_stance = any(re.search(p, text) for p in _STANCE_PATTERNS)

    return {
        "suspicion_targets": sorted(suspicion),
        "protection_targets": sorted(protection),
        "vote_leaning": vote_target,
        "evidence_bases": evidence,
        "has_stance": has_stance,
    }


def validate_public_speech(
    text: str,
    phase: str = "",
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate that a public speech has required quality components.

    Returns dict with:
    - valid: bool
    - missing_fields: list of missing component names
    - quality: extracted quality components
    - hint: retry hint when invalid
    """
    context = context or {}

    # Check filler patterns
    if not text or not text.strip():
        return {
            "valid": False,
            "missing_fields": ["stance", "suspicion_target", "vote_leaning", "evidence"],
            "quality": {},
            "hint": "发言不能为空。必须包含立场、怀疑对象、投票倾向和依据。",
        }

    for pattern in _FILLER_PATTERNS:
        if re.match(pattern, text.strip()):
            quality = extract_speech_quality(text, phase)
            return {
                "valid": False,
                "missing_fields": ["stance", "suspicion_target", "vote_leaning", "evidence"],
                "quality": quality,
                "hint": "发言过于空洞。必须包含立场、怀疑对象、投票倾向和具体依据。",
            }

    quality = extract_speech_quality(text, phase)
    missing = []

    # Check stance (relaxed for some phases)
    if not quality["has_stance"] and phase not in ("pk_speech",):
        missing.append("stance")

    # Check suspicion targets
    if not quality["suspicion_targets"]:
        missing.append("suspicion_target")

    # Check vote leaning
    if not quality["vote_leaning"]:
        missing.append("vote_leaning")

    # Check evidence
    if not quality["evidence_bases"]:
        missing.append("evidence")

    # High-pressure phases have additional requirements
    is_high_pressure = phase in ("sheriff_speech", "pk_speech") or context.get("is_claiming_role")
    if is_high_pressure:
        # Must have claim logic or counterclaim
        has_claim = bool(re.search(r"我是.*?(?:预言家|女巫|猎人|白痴)", text))
        has_counterclaim = bool(re.search(r"对跳|反跳|假预言家", text))
        has_attack_defense = bool(re.search(r"(?:矛盾|不合理|逻辑不通|查杀|金水)", text))
        if not (has_claim or has_counterclaim or has_attack_defense):
            missing.append("claim_logic")

    # Check contradiction alerts from context
    must_address = context.get("must_address_alerts", [])
    if must_address:
        # Check if speech addresses any of the alerts
        addressed = False
        for alert in must_address:
            players = alert.get("players", [])
            for player in players:
                if player in text:
                    addressed = True
                    break
            if addressed:
                break
            # Also check if the alert description keywords are mentioned
            desc_keywords = re.findall(r"p\d{2}", alert.get("description", ""))
            for keyword in desc_keywords:
                if keyword in text:
                    addressed = True
                    break
            if addressed:
                break
        if not addressed:
            missing.append("contradiction_alert")

    is_valid = len(missing) == 0

    return {
        "valid": is_valid,
        "missing_fields": missing,
        "quality": quality,
        "hint": build_speech_retry_hint(missing) if missing else "",
    }


def build_speech_retry_hint(missing_fields: list[str]) -> str:
    """Build a retry hint based on missing speech components."""
    field_hints = {
        "stance": "需要表明你的身份立场（如'我是好人阵营'）",
        "suspicion_target": "需要指出至少一个怀疑对象（如'我怀疑pXX'）",
        "protection_target": "需要表明信任的玩家",
        "vote_leaning": "需要表达投票倾向（如'我倾向投pXX'）",
        "evidence": "需要给出具体依据（如矛盾点、查杀、发言引用等）",
        "claim_logic": "在警上/PK阶段需要包含角色声明、对跳分析或攻击/防守论点",
    }
    hints = [field_hints.get(f, f"缺少{f}") for f in missing_fields]
    return "发言不完整。" + "；".join(hints) + "。"


def fallback_speech_with_basis(context: dict[str, Any]) -> str:
    """Generate a minimum viable speech with concrete targets.

    Used when retries fail. Always names targets and uses available evidence.
    """
    own_id = context.get("own_id", "unknown")
    candidates = context.get("suspicion_candidates", [])
    day = context.get("day", 1)

    target = candidates[0] if candidates else "未知"
    reason = "发言信息不足以排除嫌疑"

    return (
        f"我是好人阵营。我怀疑{target}，{reason}。"
        f"我倾向投票{target}。其他玩家请关注{target}的表现。"
    )
