"""Memory Store: unified coordinator for all memory subsystems.

Manages short-term cognition matrix, structured relation graph,
long-term reflection memory, and player profiles.
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.cognition.belief import BeliefState
from werewolf_agent.cognition.world_state import StructuredWorldState
from werewolf_agent.memory.cognition_matrix import CognitionMatrix
from werewolf_agent.memory.profile import ProfileStore
from werewolf_agent.memory.reflection import ReflectionMemory
from werewolf_agent.memory.relation_graph import RelationGraph
from werewolf_agent.memory.review import ReviewGenerator
from werewolf_agent.memory.schemas import (
    CrossGameQuery,
    PlayerProfile,
    ReflectionEntry,
    RelationEvent,
    ReviewReport,
)


class MemoryStore:
    """Top-level coordinator for all memory subsystems."""

    def __init__(self) -> None:
        self.cognition_matrices: dict[str, CognitionMatrix] = {}
        self.relation_graph = RelationGraph()
        self.reflections = ReflectionMemory()
        self.profiles = ProfileStore()
        self._review_generator = ReviewGenerator()

    # --- Short-term: cognition matrix ---

    def init_matrix(
        self,
        viewer_id: str,
        player_ids: list[str],
        role_names: list[str] | None = None,
    ) -> CognitionMatrix:
        matrix = CognitionMatrix(viewer_id)
        matrix.initialize(player_ids, role_names)
        self.cognition_matrices[viewer_id] = matrix
        return matrix

    def get_matrix(self, viewer_id: str) -> CognitionMatrix | None:
        return self.cognition_matrices.get(viewer_id)

    def sync_matrix(self, viewer_id: str, belief_state: BeliefState) -> None:
        matrix = self.cognition_matrices.get(viewer_id)
        if matrix is not None:
            matrix.update_from_belief(belief_state)

    # --- Short-term: relation graph ---

    def add_relation(self, event: RelationEvent) -> None:
        self.relation_graph.add_event(event)

    def add_relations(self, events: list[RelationEvent]) -> None:
        self.relation_graph.add_events(events)

    def import_world_state(self, world_state: StructuredWorldState, day: int = 0) -> int:
        return self.relation_graph.import_from_world_state(world_state, day)

    # --- Long-term: reflection memory ---

    def store_reflection(self, entry: ReflectionEntry) -> None:
        self.reflections.store(entry)

    def query_reflections(self, query: CrossGameQuery) -> list[ReflectionEntry]:
        return self.reflections.query(query)

    def reflections_by_player(self, player_id: str) -> list[ReflectionEntry]:
        return self.reflections.by_player(player_id)

    def reflections_by_role(self, role: str) -> list[ReflectionEntry]:
        return self.reflections.by_role(role)

    # --- Profiles ---

    def get_profile(self, player_id: str) -> PlayerProfile | None:
        return self.profiles.get(player_id)

    def get_or_create_profile(self, player_id: str) -> PlayerProfile:
        return self.profiles.get_or_create(player_id)

    # --- Review ---

    def generate_review(
        self,
        game_id: str,
        player_id: str,
        role: str,
        faction_won: bool,
        ground_truth: dict[str, str],
        generate_reflection: bool = True,
    ) -> ReviewReport:
        """Generate review for a player, optionally storing reflection."""
        matrix = self.cognition_matrices.get(player_id)

        report = self._review_generator.generate(
            game_id=game_id,
            player_id=player_id,
            role=role,
            faction_won=faction_won,
            ground_truth=ground_truth,
            cognition_matrix=matrix,
            relation_graph=self.relation_graph,
        )

        # Update profile with deltas
        self.profiles.update_after_game(
            player_id=player_id,
            role=role,
            faction_won=faction_won,
            ability_deltas=report.ability_deltas,
            review_id=f"{game_id}_{player_id}",
        )

        # Store reflection
        if generate_reflection:
            self._store_review_reflection(report)

        return report

    def generate_reviews_for_game(
        self,
        game_id: str,
        player_ids: list[str],
        roles: dict[str, str],
        faction_won: bool,
        ground_truth: dict[str, str],
    ) -> list[ReviewReport]:
        """Generate reviews for all players in a game."""
        reports = []
        for pid in player_ids:
            role = roles.get(pid, "unknown")
            report = self.generate_review(
                game_id=game_id,
                player_id=pid,
                role=role,
                faction_won=faction_won,
                ground_truth=ground_truth,
            )
            reports.append(report)
        return reports

    def _store_review_reflection(self, report: ReviewReport) -> None:
        tags = [report.role, "review"]
        if report.faction_won:
            tags.append("win")
        else:
            tags.append("loss")
        if report.error_analysis:
            tags.append("had_errors")
        if report.deceived_by:
            tags.append("deceived")

        text_parts = []
        if report.summary:
            text_parts.append(report.summary)
        if report.error_analysis:
            text_parts.append("错误分析: " + "; ".join(report.error_analysis[:3]))
        if report.successful_strategies:
            text_parts.append("成功策略: " + "; ".join(report.successful_strategies[:3]))
        if report.improvement_suggestions:
            text_parts.append("改进建议: " + "; ".join(report.improvement_suggestions[:3]))

        entry = ReflectionEntry(
            entry_id=f"reflection_{report.game_id}_{report.player_id}",
            game_id=report.game_id,
            player_id=report.player_id,
            role=report.role,
            faction_won=report.faction_won,
            text=" | ".join(text_parts),
            tags=tags,
            situation=report.summary,
        )
        self.reflections.store(entry)

    # --- Cross-game retrieval ---

    def retrieve_experience(
        self,
        player_id: str,
        role: str = "",
        tags: list[str] | None = None,
        max_results: int = 5,
    ) -> list[ReflectionEntry]:
        """Retrieve historical experience for a player before a new game."""
        query = CrossGameQuery(
            player_id=player_id,
            role=role,
            tags=tags or [],
            max_results=max_results,
        )
        return self.reflections.query(query)

    # --- Reset ---

    def reset_game_memory(self) -> None:
        """Clear short-term memory (cognition matrices) for a new game."""
        self.cognition_matrices.clear()
        self.relation_graph = RelationGraph()

    def summary(self) -> dict[str, Any]:
        return {
            "cognition_matrices": len(self.cognition_matrices),
            "relation_events": self.relation_graph.count(),
            "reflection_count": self.reflections.count(),
            "profile_count": self.profiles.count(),
            "profile_summary": self.profiles.summary(),
        }
