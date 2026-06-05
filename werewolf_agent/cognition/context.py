"""Local Context Builder: assemble minimal LLM context within token budget.

Takes outputs from all cognition pipeline stages and produces a compact
AgentContext-ready structure that fits within a token budget. Visibility
boundaries are enforced — no forbidden information enters the context.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from werewolf_agent.agents.schemas import ActionType, AgentContext, TaskType
from werewolf_agent.cognition.attention import AttentionFilter
from werewolf_agent.cognition.belief import BeliefState, BeliefUpdater
from werewolf_agent.cognition.contradiction import ContradictionAlert, ContradictionEngine
from werewolf_agent.cognition.salience import SalienceEngine, SalientFact
from werewolf_agent.cognition.strategy import StrategyPackage, StrategySelector
from werewolf_agent.cognition.visibility import VisibilityPolicy
from werewolf_agent.cognition.world_state import StructuredFact, build_world_state
from werewolf_agent.core.models import GameState
from werewolf_agent.runtime.visible_state import (
    build_public_summary,
    build_visible_player_state,
)


# ---------------------------------------------------------------------------
# Budget report
# ---------------------------------------------------------------------------

@dataclass
class PromptBudgetReport:
    """Token budget tracking for the local context."""
    total_budget: int = 4096
    visible_state_tokens: int = 0
    salience_tokens: int = 0
    belief_tokens: int = 0
    contradiction_tokens: int = 0
    strategy_tokens: int = 0
    transcript_tokens: int = 0
    remaining: int = 4096

    @property
    def used(self) -> int:
        return (
            self.visible_state_tokens +
            self.salience_tokens + self.belief_tokens +
            self.contradiction_tokens + self.strategy_tokens +
            self.transcript_tokens
        )


# ---------------------------------------------------------------------------
# Local Context Builder
# ---------------------------------------------------------------------------

class LocalContextBuilder:
    """Assembles minimal LLM context from pipeline outputs.

    Enforces:
    - Visibility Policy hard boundary
    - Token budget
    - No forbidden information (moderator_full, other private state)
    - Strategy directive from Strategy Selector
    """

    def __init__(
        self,
        visibility_policy: VisibilityPolicy,
        attention_filter: AttentionFilter,
        salience_engine: SalienceEngine,
        belief_updater: BeliefUpdater,
        contradiction_engine: ContradictionEngine,
        strategy_selector: StrategySelector,
        token_budget: int = 4096,
    ) -> None:
        self._policy = visibility_policy
        self._attention = attention_filter
        self._salience = salience_engine
        self._belief = belief_updater
        self._contradiction = contradiction_engine
        self._strategy = strategy_selector
        self._budget = token_budget

    def build(
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
        current_day: int = 0,
    ) -> tuple[AgentContext, PromptBudgetReport]:
        """Build local context for a player agent call.

        Returns (AgentContext, PromptBudgetReport).
        """
        budget = PromptBudgetReport(total_budget=self._budget, remaining=self._budget)

        # 1. Build structured world state from events
        world_state = build_world_state(game_state)

        # 2. Apply attention filter (uses visibility policy internally)
        visible_facts = self._attention.filter(
            world_state, viewer_id, viewer_role, current_phase,
        )

        # 3. Compute salience
        weighted = self._salience.weight_facts(
            visible_facts, current_day, current_phase, viewer_role,
        )
        high_facts = self._salience.filter_by_bucket(weighted, "high")
        medium_facts = self._salience.filter_by_bucket(weighted, "medium")

        # 4. Update beliefs
        if belief_state is None:
            player_ids = list(game_state.players.keys())
            belief_state = self._belief.initialize(player_ids, viewer_id)
        belief_state = self._belief.update(belief_state, visible_facts, current_day)

        # 5. Detect contradictions
        contradictions = self._contradiction.detect(visible_facts, current_day)

        # 6. Select strategy
        persona_style = ""
        if persona_snapshot:
            persona_style = persona_snapshot.get("speech_style", "")
        # --- 判断是否被多数人怀疑（基于公开投票和发言中指向本玩家的比例） ---
        is_suspected = False
        teammate_exiled = False
        if game_state:
            # Count votes targeting this player in the most recent exile vote
            for e in reversed(game_state.events):
                if e.type == "vote_resolved":
                    for v in (e.payload.get("votes") or []):
                        if isinstance(v, dict) and v.get("target") == viewer_id:
                            is_suspected = True
                            break
                    break
            # Also check recent public speeches for accusations
            if not is_suspected:
                accuse_count = 0
                for e in reversed(game_state.events):
                    if e.type == "speech" and e.payload.get("speaker") != viewer_id:
                        text = e.payload.get("text", "")
                        if viewer_id in text and any(w in text for w in ("可疑", "狼", "查杀", "铁狼", "票", "不做好")):
                            accuse_count += 1
                    if accuse_count >= 2:
                        is_suspected = True
                        break

        # --- 狼人视角：队友是否被放逐 ---
        if viewer_role == "werewolf":
            for f in visible_facts:
                if f.fact_type == "player_exiled" and f.target_player:
                    exiled_role = (
                        game_state.players[f.target_player].role
                        if f.target_player in game_state.players else ""
                    )
                    if exiled_role == "werewolf":
                        teammate_exiled = True
                        break

        # --- Hybrid's faction_goal: help master's faction (any master) ---
        faction_goal = ""
        if viewer_role == "hybrid" and game_state.hybrid_master_faction:
            faction_goal = "help_master_faction"

        strategy = self._strategy.select(
            role=viewer_role,
            faction_goal=faction_goal,
            persona_style=persona_style,
            is_suspected=is_suspected,
            teammate_just_exiled=teammate_exiled,
        )

        # 7. Assemble visible world state (only public + role-visible info)
        visible_world_state = self._build_visible_state(
            game_state, viewer_id, viewer_role, visible_facts,
        )

        # 8. Build salience items
        salience_items = self._build_salience_items(high_facts + medium_facts, budget)

        # 9. Build belief state dict
        belief_dict = self._build_belief_dict(belief_state)

        # 10. Build contradiction alerts
        contradiction_alerts = self._build_contradiction_items(contradictions)

        # 11. Build strategy directive
        strategy_directive = self._build_strategy_directive(strategy)

        # 12. Build transcript from recent speeches
        transcript = self._build_transcript(visible_facts, current_day)

        # Compute budget
        budget.visible_state_tokens = self._estimate_tokens(json.dumps(visible_world_state, ensure_ascii=False))
        budget.salience_tokens = self._estimate_tokens(json.dumps(salience_items, ensure_ascii=False))
        budget.belief_tokens = self._estimate_tokens(json.dumps(belief_dict, ensure_ascii=False))
        budget.contradiction_tokens = self._estimate_tokens(json.dumps(contradiction_alerts, ensure_ascii=False))
        budget.strategy_tokens = self._estimate_tokens(json.dumps(strategy_directive, ensure_ascii=False))
        budget.transcript_tokens = self._estimate_tokens(
            "\n".join(t.get("text", "") for t in transcript)
        )
        budget.remaining = max(0, budget.total_budget - budget.used)

        # Trim if over budget
        if budget.used > budget.total_budget:
            salience_items, transcript, contradiction_alerts, strategy_directive = self._trim_to_budget(
                salience_items, transcript, budget,
                contradiction_alerts=contradiction_alerts,
                strategy_directive=strategy_directive,
            )
            budget.salience_tokens = self._estimate_tokens(json.dumps(salience_items, ensure_ascii=False))
            budget.transcript_tokens = self._estimate_tokens(
                "\n".join(t.get("text", "") for t in transcript)
            )
            if budget.used > budget.total_budget:
                belief_dict = {}
                budget.belief_tokens = self._estimate_tokens("{}")
            budget.remaining = max(0, budget.total_budget - budget.used)

        context = AgentContext(
            agent_id=viewer_id,
            task_type=task_type,
            phase=current_phase,
            day_number=current_day,
            night_number=game_state.night_number,
            public_summary=self._build_public_summary(game_state),
            own_role=viewer_role,
            legal_actions=legal_actions,
            legal_targets=legal_targets,
            visible_world_state=visible_world_state,
            salience_items=salience_items,
            belief_state=belief_dict,
            contradiction_alerts=contradiction_alerts,
            strategy_directive=strategy_directive,
            persona_snapshot=persona_snapshot or {},
            recent_transcript=transcript,
        )

        return context, budget

    def _build_visible_state(
        self,
        game_state: GameState,
        viewer_id: str,
        viewer_role: str,
        visible_facts: list[StructuredFact],
    ) -> dict[str, Any]:
        """Build visible world state dict — only what the viewer can see."""
        state: dict[str, Any] = build_visible_player_state(game_state)

        # Role-specific private state
        if viewer_role == "werewolf":
            state["wolf_teammates"] = [
                pid for pid, p in game_state.players.items()
                if p.role == "werewolf" and pid != viewer_id
            ]
        elif viewer_role == "hybrid" and game_state.hybrid_master_id:
            state["master_id"] = game_state.hybrid_master_id
        elif viewer_role == "seer":
            check_results = [
                {"target_id": f.target_player, "alignment": f.value, "night_number": f.night}
                for f in visible_facts
                if f.fact_type == "seer_check" and f.target_player
            ]
            state["check_results"] = check_results
        elif viewer_role == "witch":
            state["antidote_available"] = not game_state.antidote_used
            state["poison_available"] = not game_state.poison_used

        return state

    def _build_salience_items(
        self,
        facts: list[SalientFact],
        budget: PromptBudgetReport,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for sf in facts:
            items.append({
                "weight": round(sf.weight, 2),
                "bucket": sf.bucket,
                "fact_type": sf.fact.fact_type,
                "source": sf.fact.source_player,
                "target": sf.fact.target_player,
                "value": sf.fact.value[:200],
                "day": sf.fact.day,
            })
        return items

    def _build_belief_dict(self, belief_state: BeliefState) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for pid, belief in belief_state.beliefs.items():
            top_role, conf = belief.top_role_guess()
            result[pid] = {
                "faction_lean": belief.faction_lean,
                "trust": round(belief.trust, 2),
                "top_role_guess": top_role,
                "top_role_confidence": round(conf, 2),
            }
        return result

    def _build_contradiction_items(
        self,
        alerts: list[ContradictionAlert],
    ) -> list[dict[str, Any]]:
        return [
            {
                "player": a.player_id,
                "type": a.alert_type,
                "priority": a.priority,
                "description": a.description,
            }
            for a in alerts
        ]

    def _build_strategy_directive(
        self,
        strategy: StrategyPackage,
    ) -> dict[str, Any]:
        return {
            "package": strategy.name,
            "goal": strategy.goal,
            "constraints": list(strategy.constraints),
        }

    def _build_transcript(
        self,
        visible_facts: list[StructuredFact],
        current_day: int,
    ) -> list[dict[str, Any]]:
        speeches: list[dict[str, Any]] = []
        for f in visible_facts:
            if f.fact_type == "speech" and f.value:
                speeches.append({
                    "speaker": f.source_player,
                    "text": f.value[:300],
                    "day": f.day,
                })
        # Keep last 6 speeches
        return speeches[-6:]

    def _build_public_summary(self, game_state: GameState) -> str:
        return build_public_summary(game_state)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略 token 估算：中文文本约 1.4 字符/token。"""
        return max(1, int(len(text) * 0.7))

    def _trim_to_budget(
        self,
        salience_items: list[dict[str, Any]],
        transcript: list[dict[str, Any]],
        budget: PromptBudgetReport,
        contradiction_alerts: list[dict[str, Any]] | None = None,
        strategy_directive: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        """Trim context to fit within budget, prioritizing salience over transcript."""
        # First trim transcript to 3 entries
        transcript = [
            {**item, "text": str(item.get("text", ""))[:120]}
            for item in transcript[-3:]
        ]
        # Then trim salience to high-priority only
        salience_items = [s for s in salience_items if s.get("bucket") == "high"]
        # 裁剪 contradiction_alerts（保留前 N 条）
        if contradiction_alerts is not None and len(contradiction_alerts) > 3:
            contradiction_alerts = contradiction_alerts[:3]
        # 截断 strategy_directive 的值
        if strategy_directive is not None:
            for key in list(strategy_directive.keys()):
                val = strategy_directive[key]
                if isinstance(val, str) and len(val) > 300:
                    strategy_directive[key] = val[:300]
        return salience_items, transcript, contradiction_alerts or [], strategy_directive or {}
