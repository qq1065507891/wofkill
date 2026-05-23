"""Wolf night discussion strategy: evidence-based consensus and plan building.

Wolves discuss targets and role assignments through multi-round private
discussion. The consensus is derived from discussion evidence, not seat order.
"""

from __future__ import annotations

import re
from typing import Any

from werewolf_agent.core.models import GameEvent


def round_requirements(night_number: int, round_number: int) -> dict[str, str]:
    """Return discussion requirements for a given night/round.

    Night 1: 3 rounds -- suspected gods, role proposals, target agreement.
    Later nights: 2 rounds -- review outcomes and adjust plan.
    """
    if night_number == 1:
        if round_number == 1:
            return {
                "focus": "suspected_gods",
                "required": "报告你怀疑的神职位置（预言家/女巫/猎人），并说明依据。",
                "role_assignment": True,
                "fake_seer": "proposed",
            }
        if round_number == 2:
            return {
                "focus": "role_proposals",
                "required": "提议角色分工：谁做假预言家、冲锋、倒钩、深水。",
                "role_assignment": True,
            }
        if round_number == 3:
            return {
                "focus": "target_agreement",
                "required": "确认击杀目标和备选目标，讨论明天的推人策略。",
            }
    else:
        if round_number == 1:
            return {
                "focus": "review_outcomes",
                "required": "回顾昨天的投票和发言结果，评估好人阵营的判断方向。",
            }
        if round_number == 2:
            return {
                "focus": "adjust_plan",
                "required": "根据新的信息调整击杀目标和推人策略。",
            }
    return {"focus": "general", "required": "讨论狼队策略。"}


def extract_wolf_proposal(text: str) -> dict[str, Any]:
    """Extract kill target and role assignment proposals from wolf speech.

    Uses regex to find player IDs (p01-p12) in kill/role context.
    Returns dict with optional 'target', 'role_assignment', 'support_for'.
    """
    result: dict[str, Any] = {
        "target": None,
        "role_assignment": None,
        "support_for": None,
    }
    if not text or not text.strip():
        return result

    # Extract kill target: patterns like "刀p05", "击杀p08", "杀p03"
    kill_match = re.search(r"[刀击杀杀]\s*(p\d{2})", text)
    if kill_match:
        result["target"] = kill_match.group(1)

    # Extract role assignments: patterns like "p01做假预言家", "p02你冲锋", "p03倒钩"
    role_map: dict[str, str] = {}
    # Patterns with explicit player ID: pXX followed by verb/pronoun + role keyword
    # Use negated character class [^，。！？、,\n] instead of .*? to prevent cross-clause matching
    role_patterns = [
        (r"(p\d{2})\s*[做当你去][^，。！？、,\n]*?假预言家", "fake_seer"),
        (r"(p\d{2})\s*[做当你去][^，。！？、,\n]*?冲锋", "pusher"),
        (r"(p\d{2})\s*[做当你去][^，。！？、,\n]*?倒钩", "hooker"),
        (r"(p\d{2})\s*[做当你去][^，。！？、,\n]*?深水", "deep_cover"),
        # Shorter: pXX directly followed by role keyword (no verb)
        (r"(p\d{2})\s*(?:做|当|去|是|负责)?\s*假预言家", "fake_seer"),
        (r"(p\d{2})\s*(?:做|当|去|是|负责)?\s*冲锋", "pusher"),
        (r"(p\d{2})\s*(?:做|当|去|是|负责)?\s*倒钩", "hooker"),
        (r"(p\d{2})\s*(?:做|当|去|是|负责)?\s*深水", "deep_cover"),
    ]
    for pattern, role_name in role_patterns:
        m = re.search(pattern, text)
        if m:
            role_map[role_name] = m.group(1)

    # Self-assignment: "我做假预言家", "我来做假预言家", "我冲锋" etc.
    self_role_patterns = [
        # Full patterns with optional verb chain — limit to same clause
        (r"我\s*(?:来\s*)?[做当去负责][^，。！？、,\n]*?假预言家", "fake_seer"),
        (r"我\s*(?:来\s*)?[做当去负责][^，。！？、,\n]*?冲锋", "pusher"),
        (r"我\s*(?:来\s*)?[做当去负责][^，。！？、,\n]*?倒钩", "hooker"),
        (r"我\s*(?:来\s*)?[做当去负责][^，。！？、,\n]*?深水", "deep_cover"),
        # Shorter forms: "假预言家" directly after "我"
        (r"我\s*(?:来\s*)?(?:做|当|去|负责)?\s*假预言家", "fake_seer"),
        (r"我\s*(?:来\s*)?(?:做|当|去|负责)?\s*冲锋", "pusher"),
        (r"我\s*(?:来\s*)?(?:做|当|去|负责)?\s*倒钩", "hooker"),
        (r"我\s*(?:来\s*)?(?:做|当|去|负责)?\s*深水", "deep_cover"),
    ]
    for pattern, role_name in self_role_patterns:
        m = re.search(pattern, text)
        if m:
            # Normalize "fake_seer_pusher" back to "pusher"
            normalized = "pusher" if role_name == "fake_seer_pusher" else role_name
            role_map[normalized] = "self"

    if role_map:
        result["role_assignment"] = role_map

    # Support for a target: "同意刀p08", "同意p08"
    support_match = re.search(r"同意[^，。！？、,\n]*?刀\s*(p\d{2})", text)
    if support_match:
        result["support_for"] = support_match.group(1)
        if result["target"] is None:
            result["target"] = support_match.group(1)

    return result


def summarize_wolf_consensus(
    events: list[GameEvent],
    alive_wolves: list[str],
    night_number: int | None = None,
) -> dict[str, Any]:
    """Build consensus summary from wolf discussion events.

    Aggregates kill target proposals and role assignments across rounds.
    Returns a plan dict with evidence_from_discussion field.
    """
    # Collect discussion events
    discussion_events = [
        e for e in events
        if e.type == "wolf_discussion" and e.payload.get("wolf_id") in alive_wolves
        and (night_number is None or e.payload.get("night_number") == night_number)
    ]

    # Track proposals per wolf
    target_votes: dict[str, int] = {}
    role_assignments: dict[str, str] = {}  # role_name -> wolf_id
    evidence: list[dict[str, Any]] = []

    for event in discussion_events:
        text = event.payload.get("text", "")
        wolf_id = event.payload.get("wolf_id", "")
        round_num = event.payload.get("round", 0)

        proposal = extract_wolf_proposal(text)

        # Handle self-assignments
        if proposal.get("role_assignment"):
            for role, assignee in proposal["role_assignment"].items():
                if assignee == "self":
                    proposal["role_assignment"][role] = wolf_id

        # Count target votes
        target = proposal.get("target")
        if target:
            target_votes[target] = target_votes.get(target, 0) + 1
            evidence.append({
                "wolf_id": wolf_id,
                "round": round_num,
                "target": target,
                "text_snippet": text[:80],
            })

        # Track role assignments (last proposal wins per role)
        if proposal.get("role_assignment"):
            for role, assignee in proposal["role_assignment"].items():
                if assignee != "self":
                    role_assignments[role] = assignee

    # Determine consensus target (most votes)
    primary_target = None
    backup_target = None
    if target_votes:
        sorted_targets = sorted(target_votes.items(), key=lambda x: x[1], reverse=True)
        primary_target = sorted_targets[0][0]
        if len(sorted_targets) > 1:
            backup_target = sorted_targets[1][0]
        else:
            backup_target = primary_target

    # Build consensus plan
    consensus: dict[str, Any] = {
        "night_kill_primary": primary_target,
        "night_kill_backup": backup_target,
        "evidence_from_discussion": evidence,
        "unresolved_disagreements": [],
    }

    # Add role assignments from discussion
    for role in ("fake_seer", "pusher", "hooker", "deep_cover"):
        if role in role_assignments:
            consensus[role] = role_assignments[role]

    # Check agreement level
    max_agreement = max(target_votes.values()) if target_votes else 0
    consensus["agreement_count"] = max_agreement
    consensus["total_wolves"] = len(alive_wolves)

    # Track disagreements
    if len(target_votes) > 1:
        for target_id, count in target_votes.items():
            if target_id != primary_target:
                consensus["unresolved_disagreements"].append({
                    "target": target_id,
                    "votes": count,
                })

    return consensus


def should_end_discussion_early(
    consensus: dict[str, Any],
    alive_wolves_count: int,
) -> bool:
    """Check if discussion can end early after consensus.

    Requires strict majority (>50%) agreement on kill target AND roles assigned.
    """
    if alive_wolves_count <= 2:
        return False  # Need discussion with 2 wolves

    agreement = consensus.get("agreement_count", 0)
    # Strict majority: more than half
    return agreement > alive_wolves_count / 2


def build_wolf_team_plan_from_discussion(
    gs: Any,  # GameState
    previous_plan: dict[str, Any] | None = None,
    consensus: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the wolf_team_plan from discussion consensus.

    Falls back to previous plan or default assignments when consensus lacks data.
    """
    if consensus is None:
        return previous_plan or {}
    primary = consensus.get("night_kill_primary")
    evidence = consensus.get("evidence_from_discussion", [])
    agreement = consensus.get("agreement_count", 0)
    total = consensus.get("total_wolves", 0)
    primary_evidence = [
        item for item in evidence
        if item.get("target") == primary
    ] if primary else []
    if not primary or not primary_evidence:
        evidence_quality = "none"
    elif total and agreement > total / 2:
        evidence_quality = "strong"
    else:
        evidence_quality = "weak"

    plan: dict[str, Any] = {
        "night_number": gs.night_number,
        "night_kill_primary": primary,
        "night_kill_backup": consensus.get("night_kill_backup") if evidence_quality != "none" else None,
        "evidence_from_discussion": evidence,
        "evidence_quality": evidence_quality,
        "unresolved_disagreements": consensus.get("unresolved_disagreements", []),
        "public_story": consensus.get("public_story", "执行讨论共识方案。"),
    }

    # Role assignments from consensus, fallback to previous plan
    previous = previous_plan or {}
    for role in ("fake_seer", "pusher", "hooker", "deep_cover"):
        plan[role] = consensus.get(role) or previous.get(role)

    # Day push target must come from current discussion evidence. Previous
    # targets are stale after a new private discussion starts.
    plan["day_push_target"] = (
        consensus.get("day_push_target")
        or (consensus.get("night_kill_backup") if evidence_quality != "none" else None)
    )

    # Rush vote opportunity (informational only)
    plan["rush_vote_opportunity"] = consensus.get("rush_vote_opportunity")

    return plan
