"""Player profiles: ability scores with growth tracking.

Design doc §10: player profile tracks logic ability, deception ability,
leadership, credibility, learning speed, and risk preference.
Profiles are updated after each game's review.
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.memory.schemas import PlayerProfile


class ProfileStore:
    """Manages player profiles with ability scores and game history."""

    def __init__(self) -> None:
        self._profiles: dict[str, PlayerProfile] = {}

    def get_or_create(self, player_id: str) -> PlayerProfile:
        if player_id not in self._profiles:
            self._profiles[player_id] = PlayerProfile(player_id=player_id)
        return self._profiles[player_id]

    def store(self, profile: PlayerProfile) -> None:
        self._profiles[profile.player_id] = profile

    def get(self, player_id: str) -> PlayerProfile | None:
        return self._profiles.get(player_id)

    def all_profiles(self) -> list[PlayerProfile]:
        return list(self._profiles.values())

    def count(self) -> int:
        return len(self._profiles)

    def update_after_game(
        self,
        player_id: str,
        role: str,
        faction_won: bool,
        ability_deltas: dict[str, float] | None = None,
        review_id: str = "",
        faction: str | None = None,
    ) -> PlayerProfile:
        """Update profile after a game completes.

        MEM-08: ``faction`` overrides the default role-based
        classification. Pass ``_player_faction(role, master_faction)``
        for hybrid players so a hybrid with a wolf master counts
        as wolf, not good.
        """
        profile = self.get_or_create(player_id)
        profile.games_played += 1

        # MEM-08: explicit faction wins; otherwise fall back to the
        # role-based default. The default treats hybrid as unknown
        # to avoid double-counting when the master is not yet
        # determined.
        if faction == "werewolf":
            profile.games_as_wolf += 1
            if faction_won:
                profile.wolf_wins += 1
        elif faction == "good":
            profile.games_as_good += 1
            if faction_won:
                profile.good_wins += 1
        elif role == "werewolf":
            profile.games_as_wolf += 1
            if faction_won:
                profile.wolf_wins += 1
        elif role == "hybrid":
            # Hybrid with no explicit faction: do not count in
            # either bucket (master not yet determined).
            pass
        else:
            profile.games_as_good += 1
            if faction_won:
                profile.good_wins += 1

        if ability_deltas:
            profile.apply_deltas(ability_deltas)

        if review_id:
            profile.review_history.append(review_id)

        return profile

    def top_by(self, attribute: str, limit: int = 10) -> list[PlayerProfile]:
        """Return profiles sorted descending by an ability attribute.

        MEM-17: the secondary sort key is ``player_id`` so that ties
        resolve to a deterministic order regardless of the dict
        insertion order. This matters when the store is hydrated
        from a database (rows may come back in any order) or when
        callers ``get_or_create`` different players in different
        sequences. Without the secondary key, two profiles with the
        same ability value can swap places across runs, breaking
        downstream consumers that expect a stable ranking.
        """
        valid = list(self._profiles.values())
        # Sort ASCENDING by ``(-attribute, player_id)`` to get
        # descending by attribute and ascending by player_id on ties.
        valid.sort(key=lambda p: (-getattr(p, attribute, 0.0), p.player_id))
        return valid[:limit]

    def summary(self) -> dict[str, Any]:
        """Aggregate stats across all profiles.

        Observability-only — do NOT pass this directly into a player
        prompt. Aggregate ability distributions are an analytics
        surface and may leak the relative skill of other players
        (which is private info). Use ``per_player_observation(player_id)``
        for prompt-side per-player data.
        """
        profiles = list(self._profiles.values())
        if not profiles:
            return {"total_players": 0}
        return {
            "total_players": len(profiles),
            "total_games_played": sum(p.games_played for p in profiles),
            "avg_win_rate": sum(p.win_rate() for p in profiles) / len(profiles),
            "avg_logic": sum(p.logic for p in profiles) / len(profiles),
            "avg_deception": sum(p.deception for p in profiles) / len(profiles),
        }
