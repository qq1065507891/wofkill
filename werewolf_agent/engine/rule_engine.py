from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from werewolf_agent.core.models import (
    Action,
    AlignmentResult,
    BadgeDecisionOptions,
    Death,
    GameEvent,
    GameState,
    PlayerState,
    RuleResult,
    VisibleContext,
    VoteResult,
    VictoryResult,
)


def _stable_seed_val(*parts: object) -> int:
    import hashlib
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & 0xFFFFFFFF


@dataclass(frozen=True)
class Ruleset:
    raw: dict[str, Any]

    @property
    def player_count(self) -> int:
        return int(self.raw["player_count"])


class RuleEngine:
    def __init__(self, ruleset: Ruleset) -> None:
        self.ruleset = ruleset

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RuleEngine":
        ruleset_path = Path(path)
        data = yaml.safe_load(ruleset_path.read_text(encoding="utf-8"))
        return cls(Ruleset(raw=data))

    def role_count(self, role: str) -> int:
        return int(self.ruleset.raw["roles"][role]["count"])

    def assign_roles(
        self, player_ids: list[str], *, seed: int | None = None
    ) -> dict[str, PlayerState]:
        if len(player_ids) != self.ruleset.player_count:
            raise ValueError(
                f"Expected {self.ruleset.player_count} players, got {len(player_ids)}"
            )
        role_list: list[str] = []
        for role, cfg in self.ruleset.raw["roles"].items():
            role_list.extend([role] * int(cfg["count"]))
        rng = random.Random(seed)
        shuffled_roles = list(role_list)
        rng.shuffle(shuffled_roles)
        return {
            pid: PlayerState(id=pid, role=role)
            for pid, role in zip(player_ids, shuffled_roles)
        }

    def night_order(self) -> list[str]:
        return [item["node"] for item in self.ruleset.raw["night_flow"]["order"]]

    def day_flow(self, day_number: int) -> list[str]:
        if day_number != 1:
            return [
                node
                for node in self.ruleset.raw["day_flow"]["standard_order"]
                if node != "first_day_sheriff_election"
            ]
        return list(self.ruleset.raw["day_flow"]["standard_order"])

    # -- Seer --

    def check_alignment(self, state: GameState, *, seer_id: str, target_id: str) -> AlignmentResult:
        target = state.players[target_id]
        seer_result = self.ruleset.raw["roles"][target.role].get("seer_result", "good")
        return AlignmentResult(alignment=seer_result, role=None)

    # -- Hybrid master --

    def choose_master(
        self, state: GameState, *, hybrid_id: str, master_id: str
    ) -> tuple[GameState, GameEvent]:
        if state.hybrid_master_id is not None:
            raise ValueError("Hybrid has already chosen a master")
        master = state.players[master_id]
        master_faction = self.ruleset.raw["roles"][master.role]["faction"]
        if master_faction == "special_bound_to_master":
            raise ValueError("Hybrid cannot choose another hybrid as master")
        new_state = replace(
            state,
            hybrid_master_id=master_id,
            hybrid_master_faction=master_faction,
        )
        event = GameEvent(
            type="hybrid_master_chosen",
            payload={"hybrid_id": hybrid_id, "master_id": master_id},
        )
        return new_state, event

    # -- Witch --

    def legal_witch_actions(
        self,
        state: GameState,
        *,
        witch_id: str,
        night_number: int,
        wolf_kill_target_id: str | None,
    ) -> list[Action]:
        actions: list[Action] = [Action(type="no_action")]
        witch_cfg = self.ruleset.raw["roles"]["witch"]["abilities"]
        if (
            wolf_kill_target_id is not None
            and not state.antidote_used
            and witch_cfg["antidote"]["can_save_wolf_kill_target"]
        ):
            if wolf_kill_target_id != witch_id or witch_cfg["antidote"].get("can_self_save", False):
                actions.append(Action(type="use_antidote", target_id=wolf_kill_target_id))
        if not state.poison_used:
            actions.append(Action(type="use_poison"))
        return actions

    def resolve_witch_action(
        self,
        state: GameState,
        *,
        witch_id: str,
        night_number: int,
        wolf_kill_target_id: str | None,
        use_antidote: bool,
        poison_target_id: str | None,
        antidote_target_id: str | None = None,
    ) -> RuleResult:
        witch_cfg = self.ruleset.raw["roles"]["witch"]["abilities"]
        if use_antidote and wolf_kill_target_id is None:
            return RuleResult(accepted=False, error_code="witch_no_wolf_kill_target")
        if use_antidote and wolf_kill_target_id not in state.players:
            return RuleResult(accepted=False, error_code="witch_wolf_kill_target_not_found")
        if use_antidote and not state.players[wolf_kill_target_id].alive:
            return RuleResult(accepted=False, error_code="witch_wolf_kill_target_not_alive")
        if use_antidote and antidote_target_id is not None and antidote_target_id != wolf_kill_target_id:
            return RuleResult(accepted=False, error_code="witch_antidote_target_mismatch")
        if use_antidote and state.antidote_used:
            return RuleResult(accepted=False, error_code="witch_antidote_already_used")
        if poison_target_id is not None and poison_target_id not in state.players:
            return RuleResult(accepted=False, error_code="witch_poison_target_not_found")
        if poison_target_id is not None and not state.players[poison_target_id].alive:
            return RuleResult(accepted=False, error_code="witch_poison_target_not_alive")
        if poison_target_id is not None and state.poison_used:
            return RuleResult(accepted=False, error_code="witch_poison_already_used")
        if not witch_cfg["use_both_potions_same_night"] and use_antidote and poison_target_id is not None:
            return RuleResult(accepted=False, error_code="witch_cannot_use_both_potions_same_night")
        if use_antidote and wolf_kill_target_id == witch_id and not witch_cfg["antidote"].get("can_self_save", False):
            return RuleResult(accepted=False, error_code="witch_cannot_self_save")
        return RuleResult(accepted=True)

    # -- Hunter --

    def can_hunter_shoot(self, state: GameState, *, hunter_id: str, death_reason: str) -> bool:
        hunter_cfg = self.ruleset.raw["roles"]["hunter"]["abilities"]
        return death_reason in hunter_cfg["can_shoot_on_death_reasons"]

    # -- Exile / Idiot --

    def legal_exile_targets(self, state: GameState) -> list[str]:
        return [
            pid for pid, p in state.players.items()
            if p.alive and not p.exile_immune
        ]

    def resolve_exile(self, state: GameState, *, target_id: str) -> tuple[GameState, list[GameEvent]]:
        target = state.players[target_id]
        events: list[GameEvent] = []
        if not target.alive or target.exile_immune:
            return state, events

        if target.role == "idiot" and not target.revealed_idiot:
            idiot_cfg = self.ruleset.raw["roles"]["idiot"]["abilities"]
            after = idiot_cfg["state_after_reveal"]
            updated = replace(
                target,
                alive=after["alive"],
                revealed_idiot=after["revealed_idiot"],
                vote_enabled=not after["vote_disabled"],
                badge_eligible=not after["badge_ineligible"],
                exile_immune=after["exile_immune"],
            )
            new_players = {**state.players, target_id: updated}
            events.append(GameEvent(type="idiot_revealed", payload={"player_id": target_id}))
            return replace(state, players=new_players), events

        death = Death(
            player_id=target_id,
            reason="exile",
            timing="day_vote",
            resolution_batch=f"day_{state.day_number}_vote",
        )
        new_state = self.apply_death(state, death)
        events.append(GameEvent(
            type="player_exiled",
            payload={"player_id": target_id, "resolution_batch": death.resolution_batch},
        ))
        return new_state, events

    # -- Death --

    def apply_death(self, state: GameState, death: Death) -> GameState:
        target = state.players[death.player_id]
        can_leave_last_words = death.can_leave_last_words
        if can_leave_last_words is None:
            night_number = state.night_number
            if night_number == 0 and death.resolution_batch.startswith("night_"):
                try:
                    night_number = int(death.resolution_batch.removeprefix("night_"))
                except ValueError:
                    night_number = state.night_number
            can_leave_last_words = self.can_leave_last_words(
                death_reason=death.reason,
                timing=death.timing,
                night_number=night_number,
            )
        triggered_skills = list(death.triggered_skills)
        if target.role == "hunter" and self.can_hunter_shoot(
            state,
            hunter_id=death.player_id,
            death_reason=death.reason,
        ):
            triggered_skills.append("hunter_shot")
        recorded_death = replace(
            death,
            can_leave_last_words=can_leave_last_words,
            triggered_skills=triggered_skills,
        )
        updated = replace(target, alive=False)
        new_players = {**state.players, death.player_id: updated}
        new_deaths = state.deaths + [recorded_death]
        new_events = state.events + [GameEvent(
            type="player_died",
            payload={
                "player_id": recorded_death.player_id,
                "reason": recorded_death.reason,
                "timing": recorded_death.timing,
                "resolution_batch": recorded_death.resolution_batch,
                "source_player_id": recorded_death.source_player_id,
                "can_leave_last_words": recorded_death.can_leave_last_words,
                "triggered_skills": recorded_death.triggered_skills,
            },
        )]
        return replace(state, players=new_players, deaths=new_deaths, events=new_events)

    # -- Night resolution --

    @dataclass(frozen=True)
    class NightInput:
        wolf_kill_target_id: str | None = None
        use_antidote: bool = False
        poison_target_id: str | None = None
        seer_target_id: str | None = None
        hybrid_master_target_id: str | None = None

    def resolve_night(
        self,
        state: GameState,
        *,
        night_number: int,
        wolf_kill_target_id: str | None,
        use_antidote: bool = False,
        poison_target_id: str | None = None,
        seer_target_id: str | None = None,
    ) -> tuple[GameState, list[GameEvent]]:
        events: list[GameEvent] = []
        deaths: list[Death] = []
        antidote_used = state.antidote_used
        poison_used = state.poison_used
        batch = f"night_{night_number}"

        # 1. Wolf kill
        saved_by_antidote = False
        if wolf_kill_target_id is not None:
            self._validate_alive_target(state, wolf_kill_target_id, "wolf_kill_target")
            wolf_death = Death(
                player_id=wolf_kill_target_id,
                reason="wolf_kill",
                timing="night",
                resolution_batch=batch,
            )
            # 2. Witch antidote
            if use_antidote and not antidote_used:
                witch_id = next(
                    (pid for pid, p in state.players.items() if p.role == "witch" and p.alive),
                    None,
                )
                if witch_id is not None and wolf_kill_target_id != witch_id:
                    saved_by_antidote = True
                    antidote_used = True
                    events.append(GameEvent(
                        type="witch_antidote_used",
                        payload={"target_id": wolf_kill_target_id, "visibility": "witch_private"},
                    ))
            if not saved_by_antidote:
                deaths.append(wolf_death)

        # 3. Witch poison
        if poison_target_id is not None and not poison_used and not use_antidote:
            self._validate_alive_target(state, poison_target_id, "poison_target")
            poison_used = True
            deaths.append(Death(
                player_id=poison_target_id,
                reason="witch_poison",
                timing="night",
                resolution_batch=batch,
            ))
            events.append(GameEvent(
                type="witch_poison_used",
                payload={"target_id": poison_target_id, "visibility": "witch_private"},
            ))

        # Apply deaths
        new_state = state
        for death in deaths:
            new_state = self.apply_death(new_state, death)

        new_state = replace(new_state, antidote_used=antidote_used, poison_used=poison_used)

        # 4. Seer check
        if seer_target_id is not None:
            seer_id = next(
                (pid for pid, p in new_state.players.items() if p.role == "seer" and p.alive),
                None,
            )
            if seer_id is not None:
                alignment_result = self.check_alignment(new_state, seer_id=seer_id, target_id=seer_target_id)
                events.append(GameEvent(
                    type="seer_check",
                    payload={
                        "seer_id": seer_id,
                        "target_id": seer_target_id,
                        "alignment": alignment_result.alignment,
                        "night_number": night_number,
                        "visibility": "seer_only",
                    },
                ))

        return new_state, events

    # -- Self-destruct (wolf day action) --

    def resolve_self_destruct(
        self, state: GameState, *, wolf_id: str, day_number: int
    ) -> tuple[GameState, list[GameEvent]]:
        wolf = state.players[wolf_id]
        if not wolf.alive or wolf.role != "werewolf":
            return state, []
        death = Death(
            player_id=wolf_id,
            reason="self_destruct",
            timing="day_discussion",
            resolution_batch=f"day_{day_number}_self_destruct",
        )
        new_state = self.apply_death(state, death)
        event = GameEvent(
            type="werewolf_self_destructed",
            payload={"player_id": wolf_id, "day_number": day_number},
        )
        return new_state, [event]

    # -- Victory --

    def check_victory(self, state: GameState) -> VictoryResult:
        players = state.players
        wolves_alive = any(p.alive and p.role == "werewolf" for p in players.values())
        if not wolves_alive:
            return VictoryResult(winner="good", reason="all_werewolves_out")

        # Slaughter check
        villagers_alive = [pid for pid, p in players.items() if p.alive and p.role == "villager"]
        gods_alive = [
            pid for pid, p in players.items()
            if p.alive and p.role in ("seer", "witch", "hunter", "idiot")
        ]

        # God slaughter
        if not gods_alive:
            return VictoryResult(winner="werewolf", reason="slaughter_gods")

        # Villager slaughter (conditional on hybrid master)
        if not villagers_alive:
            hybrid = next((p for p in players.values() if p.role == "hybrid"), None)
            master_faction = state.hybrid_master_faction
            if master_faction == "good":
                if hybrid and not hybrid.alive:
                    return VictoryResult(winner="werewolf", reason="slaughter_villagers")
            else:
                return VictoryResult(winner="werewolf", reason="slaughter_villagers")

        return VictoryResult(winner=None, reason=None)

    # -- Sheriff / Badge --

    def badge_options_after_sheriff_death(
        self,
        state: GameState,
        *,
        sheriff_id: str,
        death_reason: str,
    ) -> BadgeDecisionOptions:
        cfg = self.ruleset.raw["sheriff"]["death_policy"]
        can_act = death_reason in cfg["may_transfer_or_tear_on_death_reasons"]
        if not can_act:
            return BadgeDecisionOptions(can_transfer=False, can_tear=False, transfer_targets=[])

        targets = [
            pid for pid, p in state.players.items()
            if p.alive and pid != sheriff_id and p.badge_eligible
        ]
        return BadgeDecisionOptions(can_transfer=True, can_tear=True, transfer_targets=targets)

    def resolve_badge_decision(
        self, state: GameState, *, decision: str, target_id: str | None = None
    ) -> GameState:
        if decision == "tear":
            return replace(state, sheriff_id=None, sheriff_badge_state="torn")
        if decision == "transfer" and target_id is not None:
            return replace(state, sheriff_id=target_id, sheriff_badge_state="active")
        return state

    def sheriff_register(
        self, state: GameState, *, candidates: list[str]
    ) -> tuple[GameState, GameEvent]:
        event = GameEvent(type="sheriff_registered", payload={"candidates": candidates})
        return state, event

    def sheriff_withdraw(
        self, state: GameState, *, candidates: list[str], withdrawing: list[str]
    ) -> tuple[GameState, GameEvent]:
        remaining = [c for c in candidates if c not in withdrawing]
        event = GameEvent(
            type="sheriff_withdraw",
            payload={"remaining": remaining, "withdrew": withdrawing},
        )
        return state, event

    def resolve_sheriff_vote(
        self, state: GameState, *, votes: dict[str, str], candidates: list[str]
    ) -> tuple[GameState, GameEvent]:
        tally: dict[str, int] = {}
        for target_id in votes.values():
            if target_id in candidates:
                tally[target_id] = tally.get(target_id, 0) + 1
        if not tally:
            return state, GameEvent(type="sheriff_no_election", payload={})
        max_votes = max(tally.values())
        top = [pid for pid, count in tally.items() if count == max_votes]
        if len(top) != 1:
            return state, GameEvent(type="sheriff_vote_tie", payload={"tied": top})
        winner = top[0]
        new_state = replace(state, sheriff_id=winner, sheriff_badge_state="active")
        return new_state, GameEvent(type="sheriff_elected", payload={"sheriff_id": winner})

    def speech_order_policy(self, state: GameState) -> str:
        cfg = self.ruleset.raw["day_flow"]["speech"]
        if state.sheriff_badge_state == "torn":
            return cfg["torn_badge_order_policy"]
        return cfg["no_sheriff_order_policy"]

    # -- Vote --

    def resolve_vote(
        self,
        state: GameState,
        *,
        votes: dict[str, str],
        revote: bool,
        consecutive_no_exile_days: int = 0,
        pk_candidates: list[str] | None = None,
        rng_seed: str | None = None,
    ) -> VoteResult:
        tally: dict[str, int] = {}
        legal_targets = set(self.legal_exile_targets(state))
        for voter_id, target_id in votes.items():
            voter = state.players.get(voter_id)
            if voter is None or not voter.alive or not voter.vote_enabled:
                continue
            if target_id == voter_id:
                continue
            if target_id not in legal_targets:
                continue
            weight = 3 if voter_id == state.sheriff_id and state.sheriff_badge_state == "active" else 2
            tally[target_id] = tally.get(target_id, 0) + weight

        cfg = self.ruleset.raw["day_flow"]["vote"]

        if not tally:
            if revote:
                # Anti-stall: force exile when tally is empty but consecutive no-exile is high
                pace_cfg = cfg.get("simulation_pace", {})
                max_allowed = pace_cfg.get("max_consecutive_no_exile_days", 0)
                if (
                    pace_cfg.get("enabled", False)
                    and max_allowed > 0
                    and consecutive_no_exile_days >= max_allowed
                ):
                    candidates = [
                        pid for pid in (pk_candidates or [])
                        if pid in legal_targets
                    ]
                    if candidates:
                        seed_val = _stable_seed_val(rng_seed or "anti-stall-empty", *candidates)
                        chosen = random.Random(seed_val).choice(candidates)
                        return VoteResult(
                            exiled_player_id=chosen,
                            next_phase="resolve_exile",
                            reason="anti_stall_empty_tally",
                        )
                return VoteResult(exiled_player_id=None, next_phase="night", reason="second_tie_no_exile")
            return VoteResult(exiled_player_id=None, next_phase="pk_speech", reason="first_tie_pk")

        max_votes = max(tally.values())
        top = [pid for pid, count in tally.items() if count == max_votes]

        if len(top) > 1:
            if not revote:
                return VoteResult(
                    exiled_player_id=None,
                    next_phase="pk_speech",
                    reason="first_tie_pk",
                    tied_player_ids=top,
                )
            if cfg["second_tie_policy"] == "no_exile_then_night":
                # Anti-stall: break repeated ties deterministically
                pace_cfg = cfg.get("simulation_pace", {})
                max_allowed = pace_cfg.get("max_consecutive_no_exile_days", 0)
                if (
                    pace_cfg.get("enabled", False)
                    and max_allowed > 0
                    and consecutive_no_exile_days >= max_allowed
                ):
                    return self._anti_stall_tie_break(state, top, rng_seed)
                return VoteResult(exiled_player_id=None, next_phase="night", reason="second_tie_no_exile")

        return VoteResult(exiled_player_id=top[0], next_phase="resolve_exile", reason="majority")

    def _anti_stall_tie_break(
        self, state: GameState, tied: list[str], rng_seed: str | None,
    ) -> VoteResult:
        """Deterministic tie-break for anti-stall: sheriff vote then seeded random."""
        # 1. If active sheriff voted for one of the tied candidates, exile that one
        if state.sheriff_id and state.sheriff_badge_state == "active":
            for ev in reversed(state.events):
                if ev.type == "vote_resolved":
                    break
            # Check if sheriff cast a vote for a tied candidate
            last_votes = {}
            for ev2 in state.events:
                if ev2.type == "speech" and ev2.payload.get("speaker") == state.sheriff_id:
                    text = ev2.payload.get("text", "")
                    for candidate in tied:
                        if candidate in text:
                            last_votes[state.sheriff_id] = candidate
            sheriff_vote = last_votes.get(state.sheriff_id)
            if sheriff_vote in tied:
                return VoteResult(
                    exiled_player_id=sheriff_vote,
                    next_phase="resolve_exile",
                    reason="anti_stall_tie_break",
                )

        # 2. Seeded random from tied candidates
        seed_val = _stable_seed_val(rng_seed or "anti-stall", *tied)
        rng = random.Random(seed_val)
        chosen = rng.choice(tied)
        return VoteResult(
            exiled_player_id=chosen,
            next_phase="resolve_exile",
            reason="anti_stall_tie_break",
        )

    # -- Last Words --

    def can_leave_last_words(self, *, death_reason: str, timing: str, night_number: int) -> bool:
        lw = self.ruleset.raw["last_words"]
        if death_reason == "exile" and timing == "day_vote":
            return lw["day_exile"]
        if timing == "night":
            if night_number == 1:
                return lw["first_night_death"]
            return lw["later_night_death"]
        if death_reason == "self_destruct":
            return lw["self_destruct"]
        if death_reason == "hunter_shot":
            return lw["hunter_shot_target"]
        return False

    # -- Visibility --

    def record_private_intent(
        self,
        state: GameState,
        *,
        player_id: str,
        private_intent: dict[str, Any],
    ) -> GameState:
        new_intents = {**state.private_intents, player_id: private_intent}
        return replace(state, private_intents=new_intents)

    def build_visible_context(self, state: GameState, *, viewer_id: str, view_mode: str) -> VisibleContext:
        forbidden = set(self.ruleset.raw["information_visibility"]["forbidden_for_player_agents"])
        sections: set[str] = set()
        if view_mode == "player_view":
            sections.add("public_state")
            sections.add("own_private_state")
            if viewer_id in state.private_intents:
                sections.add(f"{viewer_id}.private_intent")
            sections -= forbidden
        elif view_mode == "moderator_full":
            sections.add("moderator_full")
            sections.add("all_private_states")
        return VisibleContext(view_mode=view_mode, visible_sections=sections)

    # -- Event reducer --

    def reduce_event(self, state: GameState, event: GameEvent) -> GameState:
        etype = event.type
        payload = event.payload

        if etype == "player_died":
            pid = payload["player_id"]
            player = state.players[pid]
            if player.alive:
                updated = replace(player, alive=False)
                new_players = {**state.players, pid: updated}
                death = Death(
                    player_id=pid,
                    reason=payload.get("reason", "unknown"),
                    timing=payload.get("timing", "unknown"),
                    resolution_batch=payload.get("resolution_batch", "unknown"),
                    source_player_id=payload.get("source_player_id"),
                    can_leave_last_words=payload.get("can_leave_last_words"),
                    triggered_skills=list(payload.get("triggered_skills", [])),
                )
                return replace(
                    state,
                    players=new_players,
                    deaths=state.deaths + [death],
                    events=state.events + [event],
                )
            return replace(state, events=state.events + [event])

        if etype == "idiot_revealed":
            pid = payload["player_id"]
            player = state.players[pid]
            idiot_cfg = self.ruleset.raw["roles"]["idiot"]["abilities"]
            after = idiot_cfg["state_after_reveal"]
            updated = replace(
                player,
                alive=after["alive"],
                revealed_idiot=after["revealed_idiot"],
                vote_enabled=not after["vote_disabled"],
                badge_eligible=not after["badge_ineligible"],
                exile_immune=after["exile_immune"],
            )
            new_players = {**state.players, pid: updated}
            return replace(state, players=new_players, events=state.events + [event])

        if etype == "werewolf_self_destructed":
            pid = payload["player_id"]
            player = state.players[pid]
            if player.alive:
                updated = replace(player, alive=False)
                new_players = {**state.players, pid: updated}
                death = Death(
                    player_id=pid,
                    reason="self_destruct",
                    timing="day_discussion",
                    resolution_batch=f"day_{payload.get('day_number', '?')}_self_destruct",
                    can_leave_last_words=payload.get("can_leave_last_words"),
                    triggered_skills=list(payload.get("triggered_skills", [])),
                )
                return replace(
                    state,
                    players=new_players,
                    deaths=state.deaths + [death],
                    events=state.events + [event],
                )
            return replace(state, events=state.events + [event])

        if etype == "hybrid_master_chosen":
            return replace(
                state,
                hybrid_master_id=payload["master_id"],
                hybrid_master_faction=self._faction_for_player(state, payload["master_id"]),
                events=state.events + [event],
            )

        if etype == "sheriff_elected":
            return replace(
                state,
                sheriff_id=payload["sheriff_id"],
                sheriff_badge_state="active",
                events=state.events + [event],
            )

        if etype == "player_exiled":
            pid = payload["player_id"]
            player = state.players[pid]
            if player.alive and not player.exile_immune:
                updated = replace(player, alive=False)
                new_players = {**state.players, pid: updated}
                death = Death(
                    player_id=pid,
                    reason="exile",
                    timing="day_vote",
                    resolution_batch=payload.get("resolution_batch", "day_vote"),
                    can_leave_last_words=payload.get("can_leave_last_words"),
                    triggered_skills=list(payload.get("triggered_skills", [])),
                )
                return replace(
                    state,
                    players=new_players,
                    deaths=state.deaths + [death],
                    events=state.events + [event],
                )
            return replace(state, events=state.events + [event])

        if etype in {"victory", "victory_checked"}:
            winner = payload.get("winner") or payload.get("winning_faction")
            if winner is None:
                return replace(state, events=state.events + [event])
            hybrid_result = payload.get("hybrid_result")
            if hybrid_result is None:
                hybrid_result = self._hybrid_result(state, winner)
            return replace(
                state,
                winning_faction=winner,
                hybrid_result=hybrid_result,
                phase="finished",
                events=state.events + [event],
            )

        # Badge decisions
        if etype == "badge_torn":
            return replace(
                state,
                sheriff_id=None,
                sheriff_badge_state="torn",
                events=state.events + [event],
            )
        if etype == "badge_transferred":
            return replace(
                state,
                sheriff_id=payload["new_sheriff_id"],
                sheriff_badge_state="active",
                events=state.events + [event],
            )

        # Witch potion tracking
        if etype == "witch_antidote_used":
            return replace(
                state, antidote_used=True, events=state.events + [event]
            )
        if etype == "witch_poison_used":
            return replace(
                state, poison_used=True, events=state.events + [event]
            )

        # Game started — transition to night with initial players
        if etype == "game_started":
            new_players = {}
            for pid, pdata in payload.get("players", {}).items():
                if isinstance(pdata, dict):
                    new_players[pid] = PlayerState(**pdata)
                else:
                    new_players[pid] = pdata
            return replace(
                state,
                players=new_players or state.players,
                phase="night",
                events=state.events + [event],
            )

        # Pause / resume
        if etype == "game_paused":
            return replace(state, paused=True, events=state.events + [event])
        if etype == "game_resumed":
            return replace(state, paused=False, events=state.events + [event])

        # Default: just append event
        return replace(state, events=state.events + [event])

    def reduce_events(self, state: GameState, events: list[GameEvent]) -> GameState:
        for event in events:
            state = self.reduce_event(state, event)
        return state

    def _faction_for_player(self, state: GameState, player_id: str) -> str:
        role = state.players[player_id].role
        return self.ruleset.raw["roles"][role]["faction"]

    def _hybrid_result(self, state: GameState, winning_faction: str) -> str | None:
        if state.hybrid_master_faction is None:
            return None
        return "win" if state.hybrid_master_faction == winning_faction else "lose"

    def _validate_alive_target(self, state: GameState, target_id: str, label: str) -> None:
        target = state.players.get(target_id)
        if target is None:
            raise ValueError(f"{label}_not_found: {target_id}")
        if not target.alive:
            raise ValueError(f"{label}_not_alive: {target_id}")
