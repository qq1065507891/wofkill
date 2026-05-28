"""Strategy Selector: choose strategy package from identity + persona + situation.

Selects from a predefined set of strategy packages based on:
- Own role and faction
- Belief state (who is suspected, who is trusted)
- Persona style (aggressive vs analytical vs deceptive)
- Game phase and risk level

Strategy packages are metadata instructions, not executable code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Strategy data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyPackage:
    """A named strategy package with goal and constraints."""
    name: str
    goal: str
    constraints: tuple[str, ...] = ()
    persona_compat: tuple[str, ...] = ()  # persona styles this works well with


# ---------------------------------------------------------------------------
# Predefined strategy packages
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, StrategyPackage] = {
    "aggressive_defense": StrategyPackage(
        name="aggressive_defense",
        goal="Defend against accusations with strong logical rebuttal",
        constraints=("maintain_claimed_identity", "avoid_self_contradiction"),
        persona_compat=("structured_logical", "authority_claim"),
    ),
    "deep_hook": StrategyPackage(
        name="deep_hook",
        goal="Play as deep-cover agent, subtly protect teammates while appearing pro-good",
        constraints=("never_directly_defend_wolf", "build_good_credibility"),
        persona_compat=("aggressive_short", "low_deception"),
    ),
    "protect_seer": StrategyPackage(
        name="protect_seer",
        goal="Shield the real seer from wolf attacks and votes",
        constraints=("do_not_out_seer", "create_noise_around_seer_suspects"),
        persona_compat=("structured_logical",),
    ),
    "push_counter_wagon": StrategyPackage(
        name="push_counter_wagon",
        goal="Push an alternative vote target to protect teammate",
        constraints=("avoid_obvious_telemarketing", "build_case_on_target"),
        persona_compat=("aggressive_short", "emotional_catalyst"),
    ),
    "find_wolves": StrategyPackage(
        name="find_wolves",
        goal="Analyze behavior and voting patterns to identify wolves",
        constraints=("trust_seer_claims_carefully", "cross_reference_votes"),
        persona_compat=("structured_logical", "data_driven_review"),
    ),
    "survive_lay_low": StrategyPackage(
        name="survive_lay_low",
        goal="Minimize visibility and avoid drawing attention",
        constraints=("short_speeches", "avoid_strong_claims"),
        persona_compat=("quiet_observer",),
    ),
    "claim_and_push": StrategyPackage(
        name="claim_and_push",
        goal="Make a strong role claim and push wolf suspects",
        constraints=("consistent_with_claim", "badge_flow_consistency"),
        persona_compat=("authority_claim", "aggressive_short"),
    ),
    "confuse_good": StrategyPackage(
        name="confuse_good",
        goal="Spread confusion among good players about roles and trusts",
        constraints=("avoid_direct_contradiction", "exploit_ambiguity"),
        persona_compat=("emotional_catalyst", "chaos_joker"),
    ),
}


# Role → default strategy mapping
_ROLE_DEFAULT_STRATEGY: dict[str, str] = {
    "villager": "find_wolves",
    "seer": "claim_and_push",
    "witch": "survive_lay_low",
    "hunter": "aggressive_defense",
    "idiot": "survive_lay_low",
    "werewolf": "deep_hook",
    "hybrid": "survive_lay_low",
}


# ---------------------------------------------------------------------------
# Strategy Selector
# ---------------------------------------------------------------------------

class StrategySelector:
    """Selects strategy package based on identity, persona, and situation."""

    def __init__(self) -> None:
        self._strategies = dict(STRATEGIES)

    def select(
        self,
        role: str,
        faction_goal: str = "",
        persona_style: str = "",
        is_suspected: bool = False,
        teammate_just_exiled: bool = False,
    ) -> StrategyPackage:
        """Select the best strategy package for the current situation."""
        # Start with role default
        base_strategy = _ROLE_DEFAULT_STRATEGY.get(role, "survive_lay_low")

        # Override based on situation
        if role == "werewolf":
            if is_suspected:
                base_strategy = "aggressive_defense"
            elif teammate_just_exiled:
                base_strategy = "push_counter_wagon"
            elif persona_style in ("aggressive_short", "emotional_catalyst"):
                base_strategy = "confuse_good"

        if role == "seer" and is_suspected:
            base_strategy = "aggressive_defense"

        if role == "hybrid" and faction_goal == "help_master_faction":
            base_strategy = "deep_hook"

        # Look up strategy package
        strategy = self._strategies.get(base_strategy)
        if strategy is None:
            strategy = self._strategies["survive_lay_low"]

        return strategy

    def get_strategy(self, name: str) -> StrategyPackage | None:
        return self._strategies.get(name)
