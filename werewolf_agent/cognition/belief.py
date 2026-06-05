"""Belief Updater: maintain role probabilities, faction leans, trust.

Updates beliefs using deterministic code logic — not LLM. The belief state
is an agent's subjective view of the game, never the ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from werewolf_agent.cognition.world_state import StructuredFact


# ---------------------------------------------------------------------------
# Belief data models
# ---------------------------------------------------------------------------

@dataclass
class PlayerBelief:
    """Agent's belief about one player."""
    player_id: str
    role_probabilities: dict[str, float] = field(default_factory=dict)
    faction_lean: str = "unknown"  # "good_lean", "wolf_lean", "unknown"
    trust: float = 0.5  # [0..1]
    open_questions: list[str] = field(default_factory=list)

    def top_role_guess(self) -> tuple[str, float]:
        if not self.role_probabilities:
            return ("unknown", 0.0)
        best = max(self.role_probabilities.items(), key=lambda x: x[1])
        return best


@dataclass
class BeliefState:
    """Complete belief state for one agent's perspective."""
    viewer_id: str
    beliefs: dict[str, PlayerBelief] = field(default_factory=dict)
    confirmed_wolves: set[str] = field(default_factory=set)
    global_assessment: str = ""
    last_updated_day: int = 0


# ---------------------------------------------------------------------------
# Belief Updater
# ---------------------------------------------------------------------------

# Role claim confidence modifiers
_CLAIM_ROLE_BOOST: dict[str, dict[str, float]] = {
    "seer": {"seer": 0.3, "werewolf": -0.1},
    "werewolf": {"werewolf": 0.4},
}

class BeliefUpdater:
    """Updates belief state from structured facts.

    Pure deterministic logic — no LLM calls. All probabilities are
    approximations from observable behavior, never ground truth.
    """

    def __init__(self, all_role_names: list[str] | None = None) -> None:
        self._role_names = all_role_names or [
            "villager", "seer", "witch", "hunter", "idiot", "werewolf", "hybrid",
        ]
        if not self._role_names:
            raise ValueError("all_role_names must not be empty")

    def initialize(self, player_ids: list[str], viewer_id: str) -> BeliefState:
        """Create initial uniform belief state."""
        state = BeliefState(viewer_id=viewer_id)
        uniform = 1.0 / len(self._role_names)
        for pid in player_ids:
            if pid == viewer_id:
                continue
            state.beliefs[pid] = PlayerBelief(
                player_id=pid,
                role_probabilities={r: uniform for r in self._role_names},
            )
        return state

    def update(
        self,
        belief_state: BeliefState,
        facts: list[StructuredFact],
        current_day: int,
    ) -> BeliefState:
        """Update beliefs from a list of visible facts.

        注意：此方法原地修改 belief_state。
        """
        for fact in facts:
            belief_state = self._apply_fact(belief_state, fact)
        belief_state.last_updated_day = current_day
        return belief_state

    def _apply_fact(
        self,
        state: BeliefState,
        fact: StructuredFact,
    ) -> BeliefState:
        """Apply a single fact to update beliefs."""
        if fact.fact_type == "player_died":
            return self._apply_death(state, fact)
        if fact.fact_type == "self_destruct":
            return self._apply_self_destruct(state, fact)
        if fact.fact_type == "idiot_revealed":
            return self._apply_idiot_reveal(state, fact)
        if fact.fact_type.startswith("claimed_role"):
            return self._apply_role_claim(state, fact)
        if fact.fact_type == "vote":
            return self._apply_vote(state, fact)
        if fact.fact_type == "speech":
            return self._apply_speech_signal(state, fact)
        if fact.fact_type == "seer_check_claim":
            return self._apply_seer_claim(state, fact)
        return state

    def _apply_death(self, state: BeliefState, fact: StructuredFact) -> BeliefState:
        """Dead player is removed from active beliefs."""
        pid = fact.target_player
        if pid and pid in state.beliefs:
            del state.beliefs[pid]
        return state

    def _apply_self_destruct(self, state: BeliefState, fact: StructuredFact) -> BeliefState:
        """Self-destruct confirms werewolf."""
        pid = fact.source_player
        if pid:
            state.confirmed_wolves.add(pid)
        if pid and pid in state.beliefs:
            del state.beliefs[pid]
        return state

    def _apply_idiot_reveal(self, state: BeliefState, fact: StructuredFact) -> BeliefState:
        """Idiot reveal confirms idiot role."""
        pid = fact.target_player
        if pid and pid in state.beliefs:
            belief = state.beliefs[pid]
            # Preserve all role keys (downstream consumers iterate the dict)
            belief.role_probabilities = {
                r: (1.0 if r == "idiot" else 0.0)
                for r in belief.role_probabilities
            }
            belief.faction_lean = "good_lean"
            belief.trust = 0.8
        return state

    def _apply_role_claim(self, state: BeliefState, fact: StructuredFact) -> BeliefState:
        """Role claim updates probability distribution."""
        pid = fact.source_player
        claimed = fact.value
        if not pid or pid not in state.beliefs:
            return state

        belief = state.beliefs[pid]
        boosts = _CLAIM_ROLE_BOOST.get(claimed, {})
        for role, delta in boosts.items():
            if role in belief.role_probabilities:
                belief.role_probabilities[role] = min(
                    1.0, belief.role_probabilities[role] + delta
                )

        # Normalize
        total = sum(belief.role_probabilities.values())
        if total > 0:
            for r in belief.role_probabilities:
                belief.role_probabilities[r] /= total
        return state

    def _apply_vote(self, state: BeliefState, fact: StructuredFact) -> BeliefState:
        """投票给已确认狼人时增加投票者的好人倾向。"""
        voter = fact.source_player
        target = fact.target_player
        if not voter or voter not in state.beliefs:
            return state

        # 只有投票给已确认狼人时才增加信任
        if target and target in state.confirmed_wolves:
            belief = state.beliefs[voter]
            belief.trust = min(1.0, belief.trust + 0.05)
        return state

    def _apply_speech_signal(self, state: BeliefState, fact: StructuredFact) -> BeliefState:
        """Speech length and style as weak signals."""
        pid = fact.source_player
        if not pid or pid not in state.beliefs:
            return state

        # Very long speeches correlate with information roles (weak signal)
        text_len = len(fact.value)
        if text_len > 200:
            state.beliefs[pid].trust = min(1.0, state.beliefs[pid].trust + 0.02)
        return state

    def _apply_seer_claim(self, state: BeliefState, fact: StructuredFact) -> BeliefState:
        """公开查杀声明：目标 faction_lean 偏狼，role_probabilities 也偏狼，声明者信任微增。"""
        target = fact.target_player
        source = fact.source_player
        val = (fact.value or "").lower()
        if "wolf" in val or "狼" in val:
            if target and target in state.beliefs:
                belief = state.beliefs[target]
                belief.faction_lean = "wolf_lean"
                belief.trust = max(0.0, belief.trust - 0.03)
                # Boost werewolf probability so top_role_guess converges
                if "werewolf" in belief.role_probabilities:
                    belief.role_probabilities["werewolf"] = min(
                        1.0, belief.role_probabilities["werewolf"] + 0.4
                    )
                    # Renormalize
                    total = sum(belief.role_probabilities.values())
                    if total > 0:
                        for r in belief.role_probabilities:
                            belief.role_probabilities[r] /= total
            if source and source in state.beliefs:
                state.beliefs[source].trust = min(1.0, state.beliefs[source].trust + 0.02)
        elif "good" in val or "好人" in val or "金水" in val:
            if target and target in state.beliefs:
                state.beliefs[target].faction_lean = "good_lean"
                state.beliefs[target].trust = min(1.0, state.beliefs[target].trust + 0.02)
        return state
