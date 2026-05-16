"""Cognitive Pipeline: top-level entry point for the cognition system.

Usage:
    pipeline = CognitivePipeline()
    context, budget = pipeline.build_context(game_state, viewer_id, ...)
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.agents.schemas import ActionType, AgentContext, TaskType
from werewolf_agent.cognition.attention import AttentionFilter
from werewolf_agent.cognition.belief import BeliefState, BeliefUpdater
from werewolf_agent.cognition.context import LocalContextBuilder, PromptBudgetReport
from werewolf_agent.cognition.contradiction import ContradictionEngine
from werewolf_agent.cognition.salience import SalienceEngine
from werewolf_agent.cognition.strategy import StrategySelector
from werewolf_agent.cognition.visibility import VisibilityPolicy
from werewolf_agent.core.models import GameState


class CognitivePipeline:
    """Complete cognition pipeline: world state → visibility → attention →
    salience → belief → contradiction → strategy → local context.
    """

    def __init__(
        self,
        visibility_config: dict[str, Any] | None = None,
        token_budget: int = 4096,
        all_role_names: list[str] | None = None,
    ) -> None:
        self._policy = VisibilityPolicy(visibility_config)
        self._attention = AttentionFilter(self._policy)
        self._salience = SalienceEngine()
        self._belief = BeliefUpdater(all_role_names)
        self._contradiction = ContradictionEngine()
        self._strategy = StrategySelector()
        self._builder = LocalContextBuilder(
            visibility_policy=self._policy,
            attention_filter=self._attention,
            salience_engine=self._salience,
            belief_updater=self._belief,
            contradiction_engine=self._contradiction,
            strategy_selector=self._strategy,
            token_budget=token_budget,
        )

    @property
    def visibility_policy(self) -> VisibilityPolicy:
        return self._policy

    @property
    def belief_updater(self) -> BeliefUpdater:
        return self._belief

    def build_context(
        self,
        game_state: GameState,
        viewer_id: str,
        viewer_role: str,
        task_type: TaskType,
        legal_actions: list[ActionType],
        legal_targets: list[str],
        current_phase: str = "",
        persona_snapshot: dict[str, Any] | None = None,
        belief_state: BeliefState | None = None,
    ) -> tuple[AgentContext, PromptBudgetReport]:
        """Build the complete local context for a player agent call."""
        return self._builder.build(
            game_state=game_state,
            viewer_id=viewer_id,
            viewer_role=viewer_role,
            task_type=task_type,
            legal_actions=legal_actions,
            legal_targets=legal_targets,
            current_phase=current_phase,
            persona_snapshot=persona_snapshot,
            belief_state=belief_state,
            current_day=game_state.day_number,
        )
