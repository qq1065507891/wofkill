"""Post-game review generation: per-player analysis and ability delta computation.

Design doc §10.2: after each game, every player generates:
- key judgments (correct/incorrect)
- error analysis
- successful strategies
- deception analysis (who deceived them)
- improvement suggestions
- ability parameter changes

Review results feed into ReflectionMemory (unstructured) and ProfileStore (deltas).
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.cognition.belief import BeliefState
from werewolf_agent.memory.cognition_matrix import CognitionMatrix
from werewolf_agent.memory.relation_graph import RelationGraph
from werewolf_agent.memory.schemas import (
    CognitionMatrixEntry,
    PlayerProfile,
    RelationEvent,
    RelationType,
    ReviewJudgment,
    ReviewReport,
)


class ReviewGenerator:
    """Generates post-game review reports from memory and game state."""

    def generate(
        self,
        game_id: str,
        player_id: str,
        role: str,
        faction_won: bool,
        ground_truth: dict[str, str],
        cognition_matrix: CognitionMatrix | None = None,
        relation_graph: RelationGraph | None = None,
    ) -> ReviewReport:
        """Generate a review report for one player.

        ground_truth maps player_id → actual role.
        """
        report = ReviewReport(
            game_id=game_id,
            player_id=player_id,
            role=role,
            faction_won=faction_won,
        )

        if cognition_matrix is not None:
            self._evaluate_judgments(report, ground_truth, cognition_matrix)
            self._compute_ability_deltas(report, cognition_matrix, ground_truth)

        if relation_graph is not None:
            self._analyze_deception(report, player_id, ground_truth, relation_graph)

        self._generate_suggestions(report)
        report.summary = self._build_summary(report)

        return report

    def _evaluate_judgments(
        self,
        report: ReviewReport,
        ground_truth: dict[str, str],
        matrix: CognitionMatrix,
    ) -> None:
        """Compare cognition matrix role guesses against ground truth."""
        for entry in matrix.all_entries():
            actual = ground_truth.get(entry.player_id)
            if actual is None:
                continue

            guessed, confidence = self._top_role_guess(entry)
            correct = guessed == actual

            # MEM-07: key_evidence items may now be EvidenceItem or
            # bare str (back-compat). Render the claim string either way.
            evidence_claims = []
            for e in entry.key_evidence[-3:]:
                claim = getattr(e, "claim", None)
                evidence_claims.append(claim if claim is not None else str(e))
            report.key_judgments.append(ReviewJudgment(
                target_player=entry.player_id,
                judgment="correct" if correct else "incorrect",
                actual_role=actual,
                guessed_role=guessed,
                evidence="; ".join(evidence_claims),
            ))

            if not correct and confidence > 0.3:
                report.error_analysis.append(
                    f"误判 {entry.player_id} 为 {guessed}（实际 {actual}），"
                    f"置信度 {confidence:.2f}，faction_read={entry.faction_read}"
                )

    def _top_role_guess(self, entry: CognitionMatrixEntry) -> tuple[str, float]:
        if not entry.role_probabilities:
            return ("unknown", 0.0)
        best = max(entry.role_probabilities.items(), key=lambda x: x[1])
        return best

    def _compute_ability_deltas(
        self,
        report: ReviewReport,
        matrix: CognitionMatrix,
        ground_truth: dict[str, str],
    ) -> None:
        """Compute ability deltas based on judgment accuracy."""
        if not report.key_judgments:
            return

        correct = sum(1 for j in report.key_judgments if j.judgment == "correct")
        total = len(report.key_judgments)
        accuracy = correct / total if total > 0 else 0.0

        deltas: dict[str, float] = {}

        # Logic improves with correct judgments
        deltas["logic"] = (accuracy - 0.5) * 0.1

        # Credibility adjusts with faction win/loss
        if report.faction_won:
            deltas["credibility"] = 0.05
        else:
            deltas["credibility"] = -0.03

        # Deception ability for wolves
        if report.role == "werewolf":
            if report.faction_won:
                deltas["deception"] = 0.08
            else:
                deltas["deception"] = -0.02

        # Leadership for correct high-confidence calls
        high_conf_correct = sum(
            1 for j in report.key_judgments
            if j.judgment == "correct"
        )
        if high_conf_correct > total * 0.6:
            deltas["leadership"] = 0.05

        report.ability_deltas = deltas

    def _analyze_deception(
        self,
        report: ReviewReport,
        player_id: str,
        ground_truth: dict[str, str],
        graph: RelationGraph,
    ) -> None:
        """Analyze who deceived this player based on relations and votes."""
        # Method 1: Find werewolves who spoke against good players this player voted for
        votes = graph.by_source(player_id)
        votes = [e for e in votes if e.predicate == RelationType.VOTED]

        for vote in votes:
            target = vote.target
            if target is None:
                continue
            actual_target_role = ground_truth.get(target, "")
            if actual_target_role in ("villager", "seer", "witch", "hunter", "idiot", "hybrid"):
                attacks = graph.query(
                    predicate=RelationType.SPOKE_AGAINST,
                    target=target,
                )
                for attack in attacks:
                    if attack.source != player_id and attack.source not in report.deceived_by:
                        attacker_role = ground_truth.get(attack.source, "")
                        if attacker_role == "werewolf":
                            report.deceived_by.append(attack.source)

        # Method 2: Werewolves who voted the same way as this player on a good target
        if not report.deceived_by:
            for vote in votes:
                target = vote.target or ""
                if ground_truth.get(target, "") in ("villager", "seer", "witch", "hunter", "idiot", "hybrid"):
                    same_voters = graph.query(
                        predicate=RelationType.VOTED,
                        target=target,
                    )
                    for v in same_voters:
                        if v.source != player_id and v.source not in report.deceived_by:
                            if ground_truth.get(v.source, "") == "werewolf":
                                report.deceived_by.append(v.source)

    def _generate_suggestions(self, report: ReviewReport) -> None:
        """Generate improvement suggestions based on review findings."""
        if report.error_analysis:
            report.improvement_suggestions.append(
                "减少高置信度误判：在确信某玩家身份前，等待更多证据"
            )

        if report.deceived_by:
            report.improvement_suggestions.append(
                f"注意被 {len(report.deceived_by)} 名狼人引导投票："
                "检查发言者的站边一致性"
            )

        if not report.faction_won:
            report.improvement_suggestions.append(
                "复盘失败对局，关注关键转折点的信息缺失"
            )

        correct = sum(1 for j in report.key_judgments if j.judgment == "correct")
        total = len(report.key_judgments)
        if total > 0 and correct / total > 0.7:
            report.successful_strategies.append(
                "角色判断准确率高，继续保持基于证据的推理方式"
            )

    def _build_summary(self, report: ReviewReport) -> str:
        correct = sum(1 for j in report.key_judgments if j.judgment == "correct")
        total = len(report.key_judgments)
        accuracy = f"{correct}/{total}" if total > 0 else "N/A"
        won = "胜利" if report.faction_won else "失败"
        base = (
            f"角色={report.role}，阵营{won}，"
            f"判断准确率={accuracy}，"
            f"被欺骗={len(report.deceived_by)}次，"
            f"错误={len(report.error_analysis)}项"
        )
        if report.error_analysis:
            base += f" | 错误分析: {'; '.join(report.error_analysis[:3])}"
        if report.improvement_suggestions:
            base += f" | 改进建议: {'; '.join(report.improvement_suggestions[:3])}"
        return base
