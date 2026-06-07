"""Memory Store: unified coordinator for all memory subsystems.

Manages short-term cognition matrix, structured relation graph,
long-term reflection memory, and player profiles.
"""

from __future__ import annotations

import logging
import re
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

_LOG = logging.getLogger(__name__)

# P0-I4: scrub concrete player ids (e.g. ``p03``) from any text that
# gets written into long-term reflection. Cross-game memory must not
# carry concrete game identities — only role-based labels survive.
# Note: do NOT use \b — `\b` does not match between an ASCII letter and
# a CJK character, so "p03的预言家" wouldn't be detected.
_REFLECTION_PLAYER_ID_RE = re.compile(r"[Pp]\d{1,2}")
_REFLECTION_ID_REPLACEMENT = "[玩家ID已省略]"


def _scrub_player_ids(text: str) -> str:
    """Replace any ``p\\d{1,2}`` token with a neutral placeholder."""
    if not text:
        return text
    return _REFLECTION_PLAYER_ID_RE.sub(_REFLECTION_ID_REPLACEMENT, text)


def _scrub_player_ids_in_list(items: list[str]) -> list[str]:
    return [_scrub_player_ids(s) for s in items]


class MemoryStore:
    """Top-level coordinator for all memory subsystems."""

    def __init__(self, repo: Any = None) -> None:
        self.cognition_matrices: dict[str, CognitionMatrix] = {}
        self.relation_graph = RelationGraph()
        self.reflections = ReflectionMemory(repo=repo)
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
        faction: str | None = None,
    ) -> ReviewReport:
        """Generate review for a player, optionally storing reflection.

        MEM-08: pass ``faction`` to override the default role-based
        classification (used for hybrid whose master is on a known
        side). Callers should pass ``_player_faction(role,
        master_faction)`` for hybrid.
        """
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
            faction=faction,
        )

        # Store reflection
        if generate_reflection:
            self._store_review_reflection(report)

        return report

    _GOOD_ROLES = {"villager", "seer", "witch", "hunter", "idiot"}
    _WOLF_ROLES = {"werewolf"}

    @classmethod
    def _player_faction(cls, role: str | None, *, master_faction: str | None = None) -> str:
        if role == "hybrid":
            return master_faction or "unknown"
        if role in cls._GOOD_ROLES:
            return "good"
        if role in cls._WOLF_ROLES:
            return "werewolf"
        return "unknown"

    def generate_reviews_for_game(
        self,
        game_id: str,
        player_ids: list[str],
        roles: dict[str, str],
        winning_faction: str,
        ground_truth: dict[str, str],
        player_factions: dict[str, str] | None = None,
        hybrid_master_factions: dict[str, str] | None = None,
    ) -> list[ReviewReport]:
        """Generate reviews for all players in a game.

        Args:
            winning_faction: The faction that won (\"good\" or \"werewolf\").
            player_factions: Optional per-player faction override (e.g. for hybrid).
            hybrid_master_factions: Optional mapping of player_id →
                master's faction for hybrid roles. When set, the hybrid
                is classified as ``"good"`` or ``"werewolf"`` based on
                its master's faction; otherwise the legacy behavior
                treats hybrid as ``"unknown"``.

        MEM-22: if ``ground_truth`` is missing entries for some of
        the ``player_ids``, log a warning naming the missing players.
        The reviews are still generated (the missing ground-truth
        entry is treated as ``"unknown"``), but the discrepancy is
        surfaced so the upstream caller can fix the input — silent
        skips hide real bugs (the moderator might have failed to
        populate the ground truth, or the player list might have
        drifted).

        MEM-NEW-2: thread ``hybrid_master_factions`` into the
        per-player faction computation. Without it, every hybrid
        returns ``"unknown"`` from ``_player_faction`` and never
        gets a meaningful ``faction_won`` flag or profile side.
        """
        # MEM-22: detect the mismatch up-front so the warning is
        # emitted exactly once per call, regardless of how many
        # players are missing.
        player_set = set(player_ids)
        truth_set = set(ground_truth)
        missing = sorted(player_set - truth_set)
        extra = sorted(truth_set - player_set)
        if missing or extra:
            _LOG.warning(
                "generate_reviews_for_game %s: ground_truth player-id set "
                "does not match player_ids; missing=%s extra=%s",
                game_id, missing, extra,
            )
        reports = []
        for pid in player_ids:
            role = roles.get(pid, "unknown")
            # MEM-NEW-6: explicit None check, not ``or``. Caller
            # passing ``""`` must be respected, not silently
            # overridden by the role-based fallback.
            if player_factions and pid in player_factions:
                pf = player_factions[pid]
            else:
                # MEM-NEW-2: pass the hybrid master's faction so a
                # hybrid with master=werewolf is classified as
                # "werewolf" (not "unknown").
                master_faction = (hybrid_master_factions or {}).get(pid)
                pf = self._player_faction(role, master_faction=master_faction)
            faction_won = pf == winning_faction
            report = self.generate_review(
                game_id=game_id,
                player_id=pid,
                role=role,
                faction_won=faction_won,
                ground_truth=ground_truth,
                faction=pf,
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
            # MEM-12: deceived_by is a list of player_ids and must be
            # scrubbed of concrete pIDs before being persisted into
            # long-term reflection. The legacy code added the
            # ``deceived`` tag but never replaced the ids in the
            # source list — they leaked through ``ReportReview.deceived_by``
            # when the report was later re-serialized or quoted.
            # The fix scrubs in place so any downstream consumer of
            # the report also sees scrubbed ids.
            report.deceived_by = _scrub_player_ids_in_list(report.deceived_by)

        # P0-I4: scrub concrete player ids from every text fragment
        # that will land in the long-term reflection entry. ``player_id``
        # itself is the only structural id we keep (it identifies the
        # reflection owner, not the targets named in the text).
        text_parts = []
        if report.summary:
            text_parts.append(_scrub_player_ids(report.summary))
        if report.error_analysis:
            text_parts.append(
                "错误分析: " + "; ".join(
                    _scrub_player_ids_in_list(report.error_analysis[:3])
                )
            )
        if report.successful_strategies:
            text_parts.append(
                "成功策略: " + "; ".join(
                    _scrub_player_ids_in_list(report.successful_strategies[:3])
                )
            )
        if report.improvement_suggestions:
            text_parts.append(
                "改进建议: " + "; ".join(
                    _scrub_player_ids_in_list(report.improvement_suggestions[:3])
                )
            )

        entry = ReflectionEntry(
            entry_id=f"reflection_{report.game_id}_{report.player_id}",
            game_id=report.game_id,
            player_id=report.player_id,
            role=report.role,
            faction_won=report.faction_won,
            text=" | ".join(text_parts),
            tags=tags,
            # MEM-05: situation now holds the structured game context
            # (game_id / role / won) instead of duplicating the
            # summary. Previously it was a copy of ``summary`` and
            # doubled the per-entry storage cost.
            situation=(
                f"game={report.game_id} role={report.role} "
                f"won={report.faction_won}"
            ),
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

    # --- Reviews (P-U4) ---

    def save_review(
        self,
        game_id: str,
        player_id: str,
        review_data: dict[str, Any],
    ) -> str:
        """Persist a per-player review entry. Returns the review id.

        P-U4: used by ``tools.local_tools._write_review`` so the
        local tool actually writes somewhere instead of returning
        a stub.  Backed by an in-memory dict on this store; callers
        that need cross-process persistence can wire their own
        store or extend this method.

        Returns:
            A deterministic review id of the form ``"{game_id}:{player_id}"``.
        """
        if not hasattr(self, "_reviews") or self._reviews is None:
            self._reviews = {}
        review_id = f"{game_id}:{player_id}"
        self._reviews[review_id] = {
            "game_id": game_id,
            "player_id": player_id,
            "review_data": dict(review_data),
        }
        return review_id

    def get_review(
        self,
        game_id: str,
        player_id: str,
    ) -> dict[str, Any] | None:
        """Return the review for ``(game_id, player_id)`` or ``None``."""
        if not hasattr(self, "_reviews") or self._reviews is None:
            return None
        return self._reviews.get(f"{game_id}:{player_id}")

    def summary(self) -> dict[str, Any]:
        return {
            "cognition_matrices": len(self.cognition_matrices),
            "relation_events": self.relation_graph.count(),
            "reflection_count": self.reflections.count(),
            "profile_count": self.profiles.count(),
            "profile_summary": self.profiles.summary(),
        }
