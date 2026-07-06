# -*- coding: utf-8 -*-
"""
功能描述：RuleEngine 是整个游戏的核心判决器，从 YAML 规则集加载配置，提供 assign_roles、resolve_night、
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-06
使用示例：内部模块，无对外接口
"""
from __future__ import annotations

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
from werewolf_agent.engine import (
    rule_death,
    rule_exile,
    rule_flow,
    rule_last_words,
    rule_special_roles,
    rule_victory,
    rule_visibility,
    rule_vote,
)
from werewolf_agent.engine.event_reducer import EventReducer, _apply_idiot_reveal
from werewolf_agent.engine.sheriff import SheriffRules


@dataclass(frozen=True)
class Ruleset:
    raw: dict[str, Any]

    @property
    def player_count(self) -> int:
        return int(self.raw["player_count"])


class RuleEngine:
    def __init__(self, ruleset: Ruleset) -> None:
        self.ruleset = ruleset if isinstance(ruleset, Ruleset) else Ruleset(raw=ruleset)
        raw = self.ruleset.raw
        self._sheriff = SheriffRules(raw)
        self._reducer = EventReducer(raw)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RuleEngine":
        ruleset_path = Path(path)
        data = yaml.safe_load(ruleset_path.read_text(encoding="utf-8"))
        return cls(Ruleset(raw=data))

    def role_count(self, role: str) -> int:
        return rule_flow.role_count(self.ruleset.raw, role)

    def assign_roles(
        self, player_ids: list[str], *, seed: int | None = None
    ) -> dict[str, PlayerState]:
        return rule_flow.assign_roles(
            self.ruleset.raw,
            player_ids,
            player_count=self.ruleset.player_count,
            seed=seed,
        )

    def night_order(self) -> list[str]:
        return rule_flow.night_order(self.ruleset.raw)

    def day_flow(self, day_number: int) -> list[str]:
        return rule_flow.day_flow(self.ruleset.raw, day_number)

    # -- Seer --

    def check_alignment(self, state: GameState, *, target_id: str) -> AlignmentResult:
        return rule_special_roles.check_alignment(
            self.ruleset.raw,
            state,
            target_id=target_id,
        )

    def _apply_idiot_reveal(self, state: GameState, player_id: str) -> GameState:
        """Apply idiot reveal state transitions. Delegates to engine-level helper."""
        return _apply_idiot_reveal(self.ruleset.raw, state, player_id)

    # -- Hybrid master --

    def choose_master(
        self, state: GameState, *, hybrid_id: str, master_id: str
    ) -> tuple[GameState, GameEvent]:
        return rule_special_roles.choose_master(
            self.ruleset.raw,
            state,
            hybrid_id=hybrid_id,
            master_id=master_id,
        )

    # -- Witch --

    def legal_witch_actions(
        self,
        state: GameState,
        *,
        witch_id: str,
        night_number: int,
        wolf_kill_target_id: str | None,
    ) -> list[Action]:
        return rule_special_roles.legal_witch_actions(
            self.ruleset.raw,
            state,
            witch_id=witch_id,
            night_number=night_number,
            wolf_kill_target_id=wolf_kill_target_id,
        )

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
        return rule_special_roles.resolve_witch_action(
            self.ruleset.raw,
            state,
            witch_id=witch_id,
            night_number=night_number,
            wolf_kill_target_id=wolf_kill_target_id,
            use_antidote=use_antidote,
            poison_target_id=poison_target_id,
            antidote_target_id=antidote_target_id,
        )

    # -- Hunter --

    def can_hunter_shoot(self, state: GameState, *, hunter_id: str, death_reason: str) -> bool:
        return rule_special_roles.can_hunter_shoot(
            self.ruleset.raw,
            death_reason,
        )

    # -- Exile / Idiot --

    def legal_exile_targets(self, state: GameState) -> list[str]:
        return rule_exile.legal_exile_targets(state)

    def resolve_exile(self, state: GameState, *, target_id: str) -> tuple[GameState, list[GameEvent]]:
        return rule_exile.resolve_exile(
            state,
            target_id=target_id,
            apply_death_fn=self.apply_death,
            apply_idiot_reveal_fn=self._apply_idiot_reveal,
        )

    # -- Death --

    def apply_death(self, state: GameState, death: Death) -> GameState:
        return rule_death.apply_death(
            state,
            death,
            can_leave_last_words_fn=self.can_leave_last_words,
            can_hunter_shoot_fn=self.can_hunter_shoot,
        )

    # -- Night resolution --

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
        witch_id = next(
            (pid for pid, p in state.players.items() if p.role == "witch" and p.alive),
            None,
        )
        if use_antidote or poison_target_id is not None:
            if witch_id is None:
                raise ValueError("witch_not_available")
            witch_result = self.resolve_witch_action(
                state,
                witch_id=witch_id,
                night_number=night_number,
                wolf_kill_target_id=wolf_kill_target_id,
                use_antidote=use_antidote,
                poison_target_id=poison_target_id,
            )
            if not witch_result.accepted:
                raise ValueError(witch_result.error_code or "witch_action_invalid")

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
                witch_cfg = self.ruleset.raw["roles"]["witch"]
                antidote_cfg = witch_cfg["abilities"].get("antidote", {})
                can_self_save = antidote_cfg.get("can_self_save", False)
                can_save_first_night = antidote_cfg.get("can_self_save_first_night", False)
                can_save = (
                    wolf_kill_target_id != witch_id
                    or can_self_save
                    or (can_save_first_night and night_number == 1)
                )
                if witch_id is not None and can_save:
                    saved_by_antidote = True
                    antidote_used = True
                    events.append(GameEvent(
                        type="witch_antidote_used",
                        payload={"target_id": wolf_kill_target_id, "visibility": "witch_private"},
                    ))
            if not saved_by_antidote:
                deaths.append(wolf_death)

        # 3. Witch poison
        witch_cfg = self.ruleset.raw["roles"]["witch"]["abilities"]
        use_both = witch_cfg.get("use_both_potions_same_night", False)
        if poison_target_id is not None and not poison_used:
            if not saved_by_antidote or use_both:
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

        # 4. Seer check (before apply_death so event appears before player_died)
        if seer_target_id is not None:
            seer_id = next(
                (pid for pid, p in state.players.items() if p.role == "seer" and p.alive),
                None,
            )
            if seer_id is not None:
                alignment_result = self.check_alignment(state, target_id=seer_target_id)
                # seer_id intentionally omitted from event payload (H-5)
                # to prevent leaking seer identity through event records.
                events.append(GameEvent(
                    type="seer_check",
                    payload={
                        "target_id": seer_target_id,
                        "alignment": alignment_result.alignment,
                        "night_number": night_number,
                        "visibility": "seer_only",
                    },
                ))

        # Apply deaths
        new_state = state
        for death in deaths:
            new_state = self.apply_death(new_state, death)

        new_state = replace(new_state, antidote_used=antidote_used, poison_used=poison_used)

        return new_state, events

    # -- Self-destruct (wolf day action) --

    def resolve_self_destruct(
        self, state: GameState, *, wolf_id: str, day_number: int
    ) -> tuple[GameState, list[GameEvent]]:
        return rule_exile.resolve_self_destruct(
            state,
            wolf_id=wolf_id,
            day_number=day_number,
            apply_death_fn=self.apply_death,
        )

    # -- Victory --

    def check_victory(self, state: GameState) -> VictoryResult:
        return rule_victory.check_victory(self.ruleset.raw, state)

    # -- Sheriff / Badge --

    def badge_options_after_sheriff_death(
        self,
        state: GameState,
        *,
        sheriff_id: str,
        death_reason: str,
    ) -> BadgeDecisionOptions:
        return self._sheriff.badge_options_after_sheriff_death(
            state, sheriff_id=sheriff_id, death_reason=death_reason,
        )

    def resolve_badge_decision(
        self, state: GameState, *, decision: str, target_id: str | None = None
    ) -> GameState:
        return self._sheriff.resolve_badge_decision(state, decision=decision, target_id=target_id)

    def sheriff_register(
        self, state: GameState, *, candidates: list[str]
    ) -> tuple[GameState, GameEvent]:
        return self._sheriff.sheriff_register(state, candidates=candidates)

    def sheriff_withdraw(
        self, state: GameState, *, candidates: list[str], withdrawing: list[str]
    ) -> tuple[GameState, GameEvent]:
        return self._sheriff.sheriff_withdraw(state, candidates=candidates, withdrawing=withdrawing)

    def resolve_sheriff_vote(
        self, state: GameState, *, votes: dict[str, str], candidates: list[str]
    ) -> tuple[GameState, GameEvent]:
        return self._sheriff.resolve_sheriff_vote(state, votes=votes, candidates=candidates)

    def speech_order_policy(self, state: GameState) -> str:
        return self._sheriff.speech_order_policy(state)

    # -- Vote --

    def base_vote_weight(self) -> int:
        return int(self.ruleset.raw.get("game_rules", {}).get("base_vote_weight", 2))

    def sheriff_vote_weight(self) -> int:
        return rule_vote.sheriff_vote_weight(
            self.ruleset.raw,
            base_vote_weight=self.base_vote_weight(),
        )

    def vote_weight(self, state: GameState, voter_id: str) -> int:
        return rule_vote.vote_weight(
            self.ruleset.raw,
            state,
            voter_id,
            base_vote_weight=self.base_vote_weight(),
        )

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
        return rule_vote.resolve_vote(
            self.ruleset.raw,
            state,
            votes=votes,
            revote=revote,
            base_vote_weight=self.base_vote_weight(),
            legal_targets=set(self.legal_exile_targets(state)),
            vote_weight_fn=self.vote_weight,
            anti_stall_tie_break_fn=self._anti_stall_tie_break,
            consecutive_no_exile_days=consecutive_no_exile_days,
            pk_candidates=pk_candidates,
            rng_seed=rng_seed,
        )

    def _anti_stall_tie_break(
        self, state: GameState, tied: list[str], rng_seed: str | None, votes: dict[str, str],
    ) -> VoteResult:
        """重复平票时优先按警长投票破局，否则使用稳定随机种子。"""
        return rule_vote.anti_stall_tie_break(state, tied, rng_seed, votes)

    # -- Last Words --

    def can_leave_last_words(self, *, death_reason: str, timing: str, night_number: int) -> bool:
        return rule_last_words.can_leave_last_words(
            self.ruleset.raw,
            death_reason=death_reason,
            timing=timing,
            night_number=night_number,
        )

    # -- Visibility --

    def record_private_intent(
        self,
        state: GameState,
        *,
        player_id: str,
        private_intent: dict[str, Any],
    ) -> GameState:
        return rule_visibility.record_private_intent(
            state,
            player_id=player_id,
            private_intent=private_intent,
        )

    def build_visible_context(self, state: GameState, *, viewer_id: str, view_mode: str) -> VisibleContext:
        return rule_visibility.build_visible_context(
            self.ruleset.raw,
            state,
            viewer_id=viewer_id,
            view_mode=view_mode,
        )

    # -- Event reducer (delegates to EventReducer) --

    def reduce_event(self, state: GameState, event: GameEvent) -> GameState:
        return self._reducer.reduce_event(state, event)

    def reduce_events(self, state: GameState, events: list[GameEvent]) -> GameState:
        return self._reducer.reduce_events(state, events)

    def _validate_alive_target(self, state: GameState, target_id: str, label: str) -> None:
        rule_death.validate_alive_target(state, target_id, label)
