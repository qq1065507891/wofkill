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
    r"(p\d{2})\s*可疑", r"(p\d{2})\s*狼面",
    r"(p\d{2})\s*(?:是|像).{0,6}狼", r"标狼\s*(p\d{2})",
    r"(p\d{2})\s*行为.*?(?:反常|异常|不正常)",
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
    (r"(?:前后|昨天.*?今天|站边.*?投票).{0,12}(?:没|没有|未)(?:有)?对上", "contradiction"),
    (r"不合理", "unreasonable"),
    (r"查杀", "seer_check"),
    (r"查验", "seer_check"),
    (r"警徽流", "badge_flow"),
    (r"对跳", "counterclaim"),
    (r"票数", "vote_tally"),
    (r"票型", "vote_tally"),
    (r"投票记录", "vote_tally"),
    (r"之前说", "speech_quote"),
    (r"刚才说", "speech_quote"),
    (r"依据", "reasoning"),
    (r"理由", "reasoning"),
    (r"因为.*?(?:投|票|怀疑|保|杀|出局)", "reasoning"),
    (r"逻辑.*?(?:链|线|漏洞|矛盾|不通|问题)", "reasoning"),
    (r"跳(?:了?)?(?:预言家|女巫|猎人|白痴|身份)", "role_claim"),
]

# Stance indicators
_STANCE_PATTERNS = [
    r"好人", r"我是.*?(?:好人|村民|预言家|女巫|猎人)",
    r"狼人阵营", r"我是p\d{2}视角", r"我站.*?边", r"站边\s*p\d{2}",
]

_PEACE_NIGHT_WITCH_FALLACY_PATTERNS = [
    r"平安夜.{0,20}(?:根本)?没有人死.{0,40}女巫.{0,20}(?:不可能|怎么|凭什么).{0,20}(?:知道|知道狼人刀|知道刀口)",
    r"平安夜.{0,30}(?:根本)?没有人死.{0,40}(?:你|女巫).{0,20}怎么知道.{0,20}(?:狼人刀|刀了人|刀口)",
    r"平安夜.{0,40}(?:救了谁|救谁).{0,40}(?:说不出|不知道|不说).{0,20}(?:就是|说明).{0,20}(?:假女巫|不是女巫)",
    r"(?:说不出|不说|不知道).{0,12}(?:救了谁|救谁|银水).{0,20}(?:就是|说明).{0,20}(?:假女巫|不是女巫)",
    r"预言家.{0,20}(?:应该|必须).{0,12}质疑.{0,30}(?:女巫|救了谁|平安夜).{0,30}(?:而不是|不该).{0,20}(?:发金水|给.*金水)",
]

_ROLE_CLAIM_PATTERNS = [
    re.compile(
        r"(p\d{2}).{0,12}(?:声称自己是|说自己是|自称|认|跳)(狼人|预言家|女巫|猎人|白痴|村民|民|混血儿|hybrid)"
    ),
]

_ROLE_EVIDENCE_MARKERS = {
    "狼人": ("我是狼人", "认狼", "狼队视角", "我们狼队"),
    "预言家": ("我是预言家", "我跳预言家", "认预言家", "悍跳预言家"),
    "女巫": ("我是女巫", "我认女巫", "跳女巫"),
    "猎人": ("我是猎人", "我认猎人", "跳猎人"),
    "白痴": ("我是白痴", "我认白痴", "跳白痴"),
    "村民": ("我是村民", "我是民", "我认民"),
    "民": ("我是村民", "我是民", "我认民"),
    "混血儿": ("我是混血儿", "混血儿", "主人"),
    "hybrid": ("混血儿", "混血", "主人"),
}


def _has_peace_night_witch_fallacy(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in _PEACE_NIGHT_WITCH_FALLACY_PATTERNS)


def _context_public_texts(context: dict[str, Any]) -> list[tuple[str, str]]:
    public_texts: list[tuple[str, str]] = []
    for item in context.get("recent_transcript", []) or []:
        if isinstance(item, dict):
            public_texts.append((str(item.get("speaker", "")), str(item.get("text", ""))))
    public_summary = context.get("public_summary", "")
    if public_summary:
        public_texts.append(("", str(public_summary)))
    return public_texts


def _has_unsupported_public_record_claim(text: str, context: dict[str, Any]) -> bool:
    """Return True when a speech cites a public role claim not in public text."""
    public_texts = _context_public_texts(context)
    if not public_texts:
        return False
    for pattern in _ROLE_CLAIM_PATTERNS:
        for match in pattern.finditer(text):
            player_id, role = match.group(1), match.group(2)
            markers = _ROLE_EVIDENCE_MARKERS.get(role, (role,))
            supported = any(
                (not speaker or speaker == player_id) and any(marker in public_text for marker in markers)
                for speaker, public_text in public_texts
            )
            if not supported:
                return True
    return False


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


def _has_multi_entity_reference(text: str) -> bool:
    player_ids = set(re.findall(r"p\d{2}", text))
    if len(player_ids) >= 2:
        return True

    for match in re.finditer(r"(?:名单|包含|包括|上警|候选).{0,16}?[：:，,]?\s*([^。；;]+)", text):
        segment = match.group(1)
        parts = [
            part.strip()
            for part in re.split(r"[、,，/]", segment)
            if part.strip()
        ]
        named = [
            part for part in parts
            if part not in {"我", "自己", "大家", "其中"}
            and re.search(r"[\u4e00-\u9fff]{2,4}", part)
        ]
        if len(named) >= 2:
            return True
    return False


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
    intent = str(context.get("intent") or "")
    day_value = context.get("day_number")
    day = int(day_value) if day_value is not None else 2
    required = _required_components(intent, phase=phase, day=day)
    if phase == "pk_speech":
        required.discard("stance")

    # Check filler patterns
    if not text or not text.strip():
        return {
            "valid": False,
            "missing_fields": sorted(required),
            "quality": {},
            "hint": "发言不能为空。请按当前发言意图给出具体内容。",
        }

    for pattern in _FILLER_PATTERNS:
        if re.match(pattern, text.strip()):
            quality = extract_speech_quality(text, phase)
            return {
                "valid": False,
                "missing_fields": sorted(required),
                "quality": quality,
                "hint": "发言过于空洞。请按当前发言意图给出具体依据。",
            }

    quality = extract_speech_quality(text, phase)
    missing = []

    if _has_peace_night_witch_fallacy(text):
        missing.append("peace_night_witch_reasoning")
    if _has_unsupported_public_record_claim(text, context):
        missing.append("public_record_grounding")

    if "stance" in required and not quality["has_stance"]:
        missing.append("stance")

    if "suspicion_target" in required and not quality["suspicion_targets"]:
        missing.append("suspicion_target")

    if "vote_leaning" in required and not quality["vote_leaning"]:
        missing.append("vote_leaning")

    if "evidence" in required and not quality["evidence_bases"]:
        missing.append("evidence")

    mentioned_players = set(re.findall(r"p\d{2}", text))
    target_id = str(context.get("target_id") or "")
    if "target_reference" in required:
        if target_id and target_id not in mentioned_players:
            missing.append("target_reference")
        elif not target_id and not mentioned_players:
            missing.append("target_reference")
    if "multi_entity" in required and not _has_multi_entity_reference(text):
        missing.append("multi_entity")

    # High-pressure phases have additional requirements
    is_high_pressure = phase in ("sheriff_speech", "pk_speech") or context.get("is_claiming_role")
    if is_high_pressure:
        # Must have claim logic or counterclaim
        # D-12: the claim_pattern now also accepts 村民|混血儿 so a
        # villager / hybrid who claims their public role during a
        # high-pressure phase (sheriff / PK) is treated as a valid
        # claim.  Pre-fix, only 预言家|女巫|猎人|白痴 were accepted,
        # and a villager claiming "我是村民" silently failed the
        # check, polluting the missing_fields list.
        has_claim = bool(re.search(r"我是.*?(?:预言家|女巫|猎人|白痴|村民|混血儿)", text))
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


def _required_components(intent: str, *, phase: str = "", day: int = 0) -> set[str]:
    by_intent = {
        "stand_with_seer": {"stance", "evidence"},
        "question_target": {"target_reference", "evidence"},
        "push_vote": {"suspicion_target", "vote_leaning", "evidence"},
        "info_synthesis": {"multi_entity", "evidence"},
        "anti_herd_call": {"evidence"},
        "respond_pressure": {"stance", "evidence"},
        "self_clear": {"stance", "evidence"},
    }
    if intent and intent in by_intent:
        required = set(by_intent[intent])
    else:
        required = {"stance", "suspicion_target", "vote_leaning", "evidence"}

    # Per-phase relaxation: sheriff/PK speeches don't need vote_leaning
    # (no vote yet); D1 speeches don't need strict evidence.
    if phase in ("sheriff_speech", "pk_speech") or day <= 1:
        required.discard("vote_leaning")
    if day <= 1:
        required.discard("evidence")

    return required


def build_speech_retry_hint(missing_fields: list[str]) -> str:
    """Build a retry hint based on missing speech components."""
    field_hints = {
        "stance": "需要表明你的身份立场（如'我是好人阵营'）",
        "suspicion_target": "需要指出至少一个怀疑对象（如'我怀疑pXX'）",
        "protection_target": "需要表明信任的玩家",
        "vote_leaning": "需要表达投票倾向（如'我倾向投pXX'）",
        "evidence": "需要给出具体依据（如矛盾点、查杀、发言引用等）",
        "target_reference": "需要围绕当前追问目标展开，而不是转向无关玩家",
        "multi_entity": "信息整理需要对照至少两名玩家或两条公开信息",
        "claim_logic": "在警上/PK阶段需要包含角色声明、对跳分析或攻击/防守论点",
        "peace_night_witch_reasoning": (
            "平安夜不等于无人被刀；可能是狼人空刀，也可能是女巫用解药救人。"
            "不能用“平安夜没人死”反驳女巫知道刀口，也不能把“不公开救谁”直接等同于假女巫；"
            "请改为询问是否用药、为什么暂不公开银水、以及发言前后是否矛盾"
        ),
        "public_record_grounding": (
            "引用公开记录时，必须能在游戏概况或近期发言中找到对应原文；"
            "不要把推测写成“公开记录”，无法确认时改成“我推测/我质疑”"
        ),
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
