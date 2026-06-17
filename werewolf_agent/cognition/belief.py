"""Belief Updater: maintain role probabilities, faction leans, trust.

Updates beliefs using deterministic code logic — not LLM. The belief state
is an agent's subjective view of the game, never the ground truth.

Seer claims flow through a credibility engine (claim_credibility) so a
contested or low-credibility black check cannot directly set wolf_lean.
Facts are processed in order (observe + apply) so a later claim does not
reinterpret an earlier vote.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from werewolf_agent.cognition.public_evidence import PublicEvidenceIndex

from werewolf_agent.cognition.world_state import StructuredFact
from werewolf_agent.cognition.claim_credibility import SeerClaimCredibilityEngine


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

# Non-seer role claim confidence modifiers (seer claims go through credibility)
_CLAIM_ROLE_BOOST: dict[str, dict[str, float]] = {
    "werewolf": {"werewolf": 0.4},
}

# P0 belief-public-vote-signals: 投票 trust 增量，全部基于公开锚点而非 ground truth
# Seer black-check werewolf boost base, scaled by credibility score*confidence.
# spec §Belief Integration: supported=strong, uncontested=medium,
# contested=weak (no hard lean), weak=tiny, broken=none.
_SEER_CHECK_BOOST_BY_STATUS = {
    "supported": 0.40,
    "uncontested": 0.25,
    "contested": 0.12,
    "weak": 0.05,
    "broken": 0.0,
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
        credibility: SeerClaimCredibilityEngine | None = None,
        public_evidence: PublicEvidenceIndex | None = None,
    ) -> BeliefState:
        """Update beliefs from a list of visible facts, in fact order.

        observe(fact) then apply(fact): a vote is judged only against claims
        that came before it (no future anchors). ``credibility`` defaults to a
        fresh engine (recompute); persistent callers pass a restored engine.

        注意：此方法原地修改 belief_state。
        """
        if credibility is None:
            credibility = SeerClaimCredibilityEngine()
        if public_evidence is None:
            public_evidence = PublicEvidenceIndex()
        for fact in facts:
            credibility.observe(fact)
            public_evidence.observe(fact)
            belief_state = self._apply_fact(belief_state, fact, public_evidence, credibility)
        belief_state.last_updated_day = current_day
        return belief_state

    def _apply_fact(
        self,
        state: BeliefState,
        fact: StructuredFact,
        public_evidence: PublicEvidenceIndex,
        credibility: SeerClaimCredibilityEngine,
    ) -> BeliefState:
        """Apply a single fact to update beliefs."""
        if fact.fact_type == "player_died":
            return self._apply_death(state, fact)
        if fact.fact_type == "self_destruct":
            return self._apply_self_destruct(state, fact)
        if fact.fact_type == "idiot_revealed":
            return self._apply_idiot_reveal(state, fact)
        if fact.fact_type.startswith("claimed_role"):
            return self._apply_role_claim(state, fact, credibility)
        if fact.fact_type == "vote":
            return self._apply_vote(state, fact, public_evidence)
        if fact.fact_type == "speech":
            return self._apply_speech_signal(state, fact)
        if fact.fact_type == "seer_check_claim":
            return self._apply_seer_claim(state, fact, credibility)
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

    def _apply_role_claim(
        self,
        state: BeliefState,
        fact: StructuredFact,
        credibility: SeerClaimCredibilityEngine,
    ) -> BeliefState:
        """Role claim updates probability distribution.

        Seer claims are scaled by credibility (seer_role_boost = 0.30 * score
        * confidence; trust_delta = (score-0.50)*0.08), replacing one-size-
        fits-all boosts. Other role claims keep fixed boosts.
        """
        pid = fact.source_player
        claimed = (fact.value or "").lower()
        if not pid or pid not in state.beliefs:
            return state

        belief = state.beliefs[pid]
        if claimed == "seer":
            cred = credibility.score_for(pid)
            boost = 0.30 * cred.score * cred.confidence
            trust_delta = (cred.score - 0.50) * 0.08
            if "seer" in belief.role_probabilities:
                belief.role_probabilities["seer"] = min(
                    1.0, belief.role_probabilities["seer"] + boost
                )
            belief.trust = max(0.0, min(1.0, belief.trust + trust_delta))
        else:
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

    def _apply_vote(
        self,
        state: BeliefState,
        fact: StructuredFact,
        public_evidence: PublicEvidenceIndex,
    ) -> BeliefState:
        """投票信号：基于公开锚点更新 voter trust。

        anchors 是 running（vote 之前已 observe 的 claim 集合），保证 future
        anchor 不污染（后来的 claim 不影响该 vote 判断）。
        """
        voter = fact.source_player
        target = fact.target_player
        if not voter or voter not in state.beliefs:
            return state
        if not target:
            return state

        belief = state.beliefs[voter]

        if target in state.confirmed_wolves:
            belief.trust = min(1.0, belief.trust + 0.05)
            return state

        delta = public_evidence.vote_delta(fact)

        if delta > 0:
            belief.trust = min(1.0, belief.trust + delta)
        elif delta < 0:
            belief.trust = max(0.0, belief.trust + delta)
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

    def _apply_seer_claim(
        self,
        state: BeliefState,
        fact: StructuredFact,
        credibility: SeerClaimCredibilityEngine,
    ) -> BeliefState:
        """公开查杀/金水声明：依据声明者 credibility status 更新目标。

        effective_strength = score * confidence。wolf_lean 只在 supported 或
        uncontested 高分时设置；contested/weak/broken 不设 wolf_lean。
        """
        target = fact.target_player
        source = fact.source_player
        val = (fact.value or "").lower()
        cred = credibility.score_for(source) if source else None
        status = cred.status if cred else "weak"
        score = cred.score if cred else 0.0
        conf = cred.confidence if cred else 0.0
        effective = score * conf
        boost_base = _SEER_CHECK_BOOST_BY_STATUS.get(status, 0.0)

        if "wolf" in val or "狼" in val:
            if target and target in state.beliefs:
                belief = state.beliefs[target]
                # wolf_lean only when the line is supported or a strong uncontested line
                if status == "supported" or (status == "uncontested" and score >= 0.65):
                    belief.faction_lean = "wolf_lean"
                boost = boost_base * effective if boost_base else 0.0
                if boost > 0 and "werewolf" in belief.role_probabilities:
                    belief.role_probabilities["werewolf"] = min(
                        1.0, belief.role_probabilities["werewolf"] + boost
                    )
                    total = sum(belief.role_probabilities.values())
                    if total > 0:
                        for r in belief.role_probabilities:
                            belief.role_probabilities[r] /= total
                if effective > 0:
                    belief.trust = max(0.0, belief.trust - 0.03 * effective)
            if source and source in state.beliefs and conf > 0:
                state.beliefs[source].trust = min(
                    1.0, state.beliefs[source].trust + 0.02 * conf
                )
        elif "good" in val or "好人" in val or "金水" in val:
            if target and target in state.beliefs:
                # gold water: supported may set good_lean; contested/weak only slight trust
                if status == "supported":
                    state.beliefs[target].faction_lean = "good_lean"
                if effective > 0:
                    state.beliefs[target].trust = min(
                        1.0, state.beliefs[target].trust + 0.02 * effective
                    )
        return state
