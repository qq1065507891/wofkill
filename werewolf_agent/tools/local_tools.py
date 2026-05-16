"""Local LangGraph tools: internal rule/state queries per design doc §11.2.

These tools are highly coupled to local game state and rule engine.
They do NOT go through MCP — no RPC, no protocol conversion.
All results are deterministic and authoritative for game state.
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.core.models import GameState, PlayerState
from werewolf_agent.tools.schemas import (
    InternalToolName,
    ToolCall,
    ToolResult,
    ToolSource,
    ToolStatus,
)
from werewolf_agent.tools.tool_logger import ToolCallLogger


class LocalToolExecutor:
    """Executes internal LangGraph tools against local game state."""

    def __init__(self, logger: ToolCallLogger | None = None) -> None:
        self._logger = logger or ToolCallLogger()

    @property
    def logger(self) -> ToolCallLogger:
        return self._logger

    def execute(self, call: ToolCall, game_state: GameState) -> ToolResult:
        """Execute a local tool call. Returns deterministic result."""
        handler = self._HANDLERS.get(call.tool_name)
        if handler is None:
            result = ToolResult(
                tool_name=call.tool_name,
                source=ToolSource.LOCAL,
                status=ToolStatus.NOT_FOUND,
                error_message=f"Unknown local tool: {call.tool_name}",
            )
            self._logger.log(call, result)
            return result

        try:
            data = handler(self, call, game_state)
            result = ToolResult(
                tool_name=call.tool_name,
                source=ToolSource.LOCAL,
                status=ToolStatus.SUCCESS,
                data=data,
            )
        except Exception as exc:
            status = ToolStatus.UNAUTHORIZED if isinstance(exc, PermissionError) else ToolStatus.ERROR
            result = ToolResult(
                tool_name=call.tool_name,
                source=ToolSource.LOCAL,
                status=status,
                error_message=str(exc),
            )

        self._logger.log(call, result)
        return result

    # --- Tool implementations ---

    def _query_legal_actions(
        self, call: ToolCall, state: GameState,
    ) -> dict[str, Any]:
        """Query available actions for a player from RuleEngine."""
        player_id = call.params.get("player_id", "")
        if call.caller_id and call.caller_id != player_id:
            raise PermissionError(
                f"caller {call.caller_id} cannot query legal actions for {player_id}"
            )
        player = state.players.get(player_id)
        if player is None:
            return {"legal_actions": [], "legal_targets": {}, "reason": "player not found"}
        actions = self._compute_legal_actions(player, state)
        targets = self._compute_legal_targets(player, actions, state)
        return {"player_id": player_id, "legal_actions": actions, "legal_targets": targets}

    def _query_public_state(
        self, call: ToolCall, state: GameState,
    ) -> dict[str, Any]:
        """Query public game state visible to all players."""
        return {
            "day_number": state.day_number,
            "phase": state.phase,
            "sheriff_id": state.sheriff_id,
            "sheriff_badge_state": state.sheriff_badge_state,
            "alive_players": [
                p.id for p in state.players.values() if p.alive
            ],
            "dead_players": [
                {"id": p.id, "role": p.role}
                for p in state.players.values()
                if not p.alive and p.revealed_idiot
            ],
            "total_players": len(state.players),
        }

    def _query_private_state(
        self, call: ToolCall, state: GameState,
    ) -> dict[str, Any]:
        """Query private state for a specific player. Only own state."""
        player_id = call.params.get("player_id", "")
        player = state.players.get(player_id)
        if player is None:
            return {"error": "player not found"}
        if call.caller_id != player_id:
            raise PermissionError(
                f"caller {call.caller_id or '<missing>'} cannot access private state of {player_id}"
            )

        result: dict[str, Any] = {
            "player_id": player_id,
            "role": player.role,
            "alive": player.alive,
            "vote_enabled": player.vote_enabled,
            "badge_eligible": player.badge_eligible,
        }

        # Role-specific private info
        if player.role == "witch":
            result["antidote_available"] = not state.antidote_used
            result["poison_available"] = not state.poison_used
            result["current_wolf_kill_target_id"] = self._current_wolf_kill_target(state)
        if player.role == "hybrid" and state.hybrid_master_id:
            result["master_id"] = state.hybrid_master_id

        return result

    def _query_relation_graph(
        self, call: ToolCall, state: GameState,
    ) -> dict[str, Any]:
        """Query structured relation graph data."""
        # Returns event-derived relation data from state events
        filter_type = call.params.get("predicate", "")
        filter_source = call.params.get("source", "")
        filter_day = call.params.get("day")

        events = []
        for event in state.events:
            if event.type == "vote":
                voter = event.payload.get("voter", "")
                target = event.payload.get("target", "")
                day = event.payload.get("day_number", 0)
                if filter_source and voter != filter_source:
                    continue
                if filter_day is not None and day != filter_day:
                    continue
                events.append({
                    "predicate": "voted",
                    "source": voter,
                    "target": target,
                    "day": day,
                })
            elif event.type == "speech" and filter_type in ("", "speech"):
                speaker = event.payload.get("speaker", "")
                day = event.payload.get("day_number", 0)
                if filter_source and speaker != filter_source:
                    continue
                if filter_day is not None and day != filter_day:
                    continue
                events.append({
                    "predicate": "speech",
                    "source": speaker,
                    "day": day,
                })

        return {"events": events, "count": len(events)}

    def _query_cognition_matrix(
        self, call: ToolCall, state: GameState,
    ) -> dict[str, Any]:
        """Query cognition matrix for a viewer. Returns belief state."""
        # Returns empty if no matrix is stored — this is a read-only query
        viewer_id = call.params.get("viewer_id", "")
        return {
            "viewer_id": viewer_id,
            "note": "Cognition matrix is managed by MemoryStore. "
                    "This tool returns the query interface status.",
            "available": True,
        }

    def _write_review(
        self, call: ToolCall, state: GameState,
    ) -> dict[str, Any]:
        """Write review entry. Validates that game has ended."""
        if state.winning_faction is None:
            return {"error": "Cannot write review: game not ended"}

        player_id = call.params.get("player_id", "")
        review_text = call.params.get("review_text", "")
        return {
            "player_id": player_id,
            "review_text": review_text,
            "game_id": state.game_id,
            "status": "recorded",
        }

    def _call_evaluator(
        self, call: ToolCall, state: GameState,
    ) -> dict[str, Any]:
        """Call evaluation metrics for current game state."""
        alive_good = sum(
            1 for p in state.players.values()
            if p.alive and p.role != "werewolf"
        )
        alive_wolves = sum(
            1 for p in state.players.values()
            if p.alive and p.role == "werewolf"
        )
        return {
            "alive_good": alive_good,
            "alive_wolves": alive_wolves,
            "total_alive": alive_good + alive_wolves,
            "game_phase": state.phase,
            "day_number": state.day_number,
        }

    def _read_experiment_config(
        self, call: ToolCall, state: GameState,
    ) -> dict[str, Any]:
        """Read current experiment configuration."""
        return {
            "ruleset_id": state.ruleset_id,
            "game_id": state.game_id,
            "paused": state.paused,
        }

    def _generate_game_report(
        self, call: ToolCall, state: GameState,
    ) -> dict[str, Any]:
        """Generate a summary report of the current game."""
        return {
            "game_id": state.game_id,
            "ruleset_id": state.ruleset_id,
            "day_number": state.day_number,
            "night_number": state.night_number,
            "phase": state.phase,
            "winning_faction": state.winning_faction,
            "total_deaths": len(state.deaths),
            "total_events": len(state.events),
            "sheriff_id": state.sheriff_id,
            "badge_state": state.sheriff_badge_state,
        }

    def _compute_legal_actions(
        self, player: PlayerState, state: GameState,
    ) -> list[str]:
        """Compute legal actions based on player state and game phase."""
        actions: list[str] = []
        if not player.alive and not player.revealed_idiot:
            return actions

        if player.revealed_idiot:
            actions.append("speech")
            return actions

        actions.append("speech")

        if state.phase in ("day_discussion", "vote"):
            if player.vote_enabled:
                actions.append("vote")

        if player.role == "werewolf" and state.phase == "night":
            actions.append("wolf_kill")
            actions.append("wolf_no_kill")

        if player.role == "seer" and state.phase == "night":
            actions.append("check_alignment")

        if player.role == "witch" and state.phase == "night":
            if not state.antidote_used and self._current_wolf_kill_target(state) is not None:
                actions.append("use_antidote")
            if not state.poison_used:
                actions.append("use_poison")

        if player.role == "hunter":
            actions.append("hunter_shot")

        if state.phase == "sheriff_election" and player.badge_eligible:
            actions.extend(["sheriff_register", "sheriff_vote"])

        return actions

    def _compute_legal_targets(
        self,
        player: PlayerState,
        actions: list[str],
        state: GameState,
    ) -> dict[str, list[str]]:
        alive = [pid for pid, p in state.players.items() if p.alive]
        targets: dict[str, list[str]] = {}
        for action in actions:
            if action == "vote":
                targets[action] = [
                    pid for pid, p in state.players.items()
                    if p.alive and not p.exile_immune
                ]
            elif action == "wolf_kill":
                targets[action] = alive
            elif action == "use_antidote":
                current = self._current_wolf_kill_target(state)
                targets[action] = [current] if current else []
            elif action == "use_poison":
                targets[action] = alive
            elif action == "check_alignment":
                targets[action] = [pid for pid in alive if pid != player.id]
            elif action == "hunter_shot":
                targets[action] = [pid for pid in alive if pid != player.id]
            elif action in {"sheriff_vote", "badge_transfer"}:
                targets[action] = [pid for pid in alive if pid != player.id]
            else:
                targets[action] = []
        return targets

    def _current_wolf_kill_target(self, state: GameState) -> str | None:
        for event in reversed(state.events):
            if event.type == "wolf_kill_selected" and event.payload.get("night_number") == state.night_number:
                return event.payload.get("target_id")
            if event.type in {"wolf_no_kill_declared", "wolf_no_kill_timeout"} and event.payload.get("night_number") == state.night_number:
                return None
        return None

    _HANDLERS: dict[str, Any] = {
        InternalToolName.QUERY_LEGAL_ACTIONS.value: _query_legal_actions,
        InternalToolName.QUERY_PUBLIC_STATE.value: _query_public_state,
        InternalToolName.QUERY_PRIVATE_STATE.value: _query_private_state,
        InternalToolName.QUERY_RELATION_GRAPH.value: _query_relation_graph,
        InternalToolName.QUERY_COGNITION_MATRIX.value: _query_cognition_matrix,
        InternalToolName.WRITE_REVIEW.value: _write_review,
        InternalToolName.CALL_EVALUATOR.value: _call_evaluator,
        InternalToolName.READ_EXPERIMENT_CONFIG.value: _read_experiment_config,
        InternalToolName.GENERATE_GAME_REPORT.value: _generate_game_report,
    }
