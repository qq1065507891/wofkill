"""Offline golden-query evaluation tests for RAG retrieval."""

from __future__ import annotations

from pathlib import Path

from werewolf_agent.rag.schemas import (
    CaseType,
    QualityGrade,
    RAGHit,
    SourceType,
    VisibilityBoundary,
)


def _hit(entry_id: str, score: float = 0.9) -> RAGHit:
    return RAGHit(
        entry_id=entry_id,
        title=entry_id,
        summary=f"summary for {entry_id}",
        relevance_score=score,
        quality_grade=QualityGrade.EXPERT_REVIEW,
        source_type=SourceType.MANUAL_ENTRY,
        visibility_boundary=VisibilityBoundary.PLAYER_PERSPECTIVE,
        case_type=CaseType.EXTERNAL_TACTICS,
    )


class _FakeRetriever:
    def __init__(self, hits_by_situation: dict[str, list[RAGHit]]) -> None:
        self.hits_by_situation = hits_by_situation
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return self.hits_by_situation.get(query.situation, [])


def test_evaluate_golden_queries_computes_recall_mrr_ndcg_and_forbidden_hits() -> None:
    from werewolf_agent.rag.evaluation import GoldenQuery, evaluate_golden_queries

    queries = [
        GoldenQuery(
            query_id="seer_claim",
            role="seer",
            phase="sheriff_speech",
            situation="claim badge flow",
            expected_entry_ids=["seed_ext_seer_claim_01"],
            forbidden_entry_ids=["seed_ext_wolf_deep_hook_01"],
        ),
        GoldenQuery(
            query_id="wolf_cover",
            role="werewolf",
            phase="speech",
            situation="deep hook",
            expected_entry_ids=["seed_ext_wolf_deep_hook_01"],
            forbidden_entry_ids=["seed_ext_seer_claim_01"],
        ),
    ]
    retriever = _FakeRetriever({
        "claim badge flow": [
            _hit("seed_ext_wolf_deep_hook_01", 0.95),
            _hit("seed_ext_seer_claim_01", 0.88),
            _hit("seed_ext_vote_pressure_01", 0.5),
        ],
        "deep hook": [
            _hit("seed_ext_wolf_deep_hook_01", 0.97),
            _hit("seed_ext_seer_claim_01", 0.41),
        ],
    })

    report = evaluate_golden_queries(retriever, queries, k_values=(1, 3))

    assert report.total_queries == 2
    assert report.recall_at[1] == 0.5
    assert report.recall_at[3] == 1.0
    assert report.mrr == 0.75
    assert round(report.ndcg_at[3], 6) == round((1 / 1.5849625007211563 + 1.0) / 2, 6)
    assert report.forbidden_hit_count == 2
    assert report.results[0].retrieved_entry_ids[:3] == [
        "seed_ext_wolf_deep_hook_01",
        "seed_ext_seer_claim_01",
        "seed_ext_vote_pressure_01",
    ]
    assert report.results[0].first_relevant_rank == 2
    assert report.results[1].first_relevant_rank == 1
    assert retriever.queries[0].role == "seer"
    assert retriever.queries[0].phase == "sheriff_speech"
    assert retriever.queries[0].max_results == 3


def test_load_golden_queries_from_yaml(tmp_path: Path) -> None:
    from werewolf_agent.rag.evaluation import load_golden_queries

    yaml_path = tmp_path / "golden_queries.yaml"
    yaml_path.write_text(
        """
        - query_id: seer_claim
          role: seer
          phase: sheriff_speech
          situation: claim badge flow
          expected_entry_ids:
            - seed_ext_seer_claim_01
          forbidden_entry_ids:
            - seed_ext_wolf_deep_hook_01
          ruleset_id: pre_witch_hunter_idiot_mixed
          tags:
            - seer
            - badge_flow
        """,
        encoding="utf-8",
    )

    queries = load_golden_queries(yaml_path)

    assert len(queries) == 1
    assert queries[0].query_id == "seer_claim"
    assert queries[0].expected_entry_ids == ["seed_ext_seer_claim_01"]
    assert queries[0].forbidden_entry_ids == ["seed_ext_wolf_deep_hook_01"]
    assert queries[0].tags == ["seer", "badge_flow"]


def test_empty_golden_query_set_returns_zero_metrics() -> None:
    from werewolf_agent.rag.evaluation import evaluate_golden_queries

    report = evaluate_golden_queries(_FakeRetriever({}), [], k_values=(1, 3))

    assert report.total_queries == 0
    assert report.recall_at == {1: 0.0, 3: 0.0}
    assert report.ndcg_at == {3: 0.0}
    assert report.mrr == 0.0
    assert report.forbidden_hit_count == 0


def test_evaluation_retrieves_at_least_three_hits_for_ndcg_even_when_recall_k_is_one() -> None:
    from werewolf_agent.rag.evaluation import GoldenQuery, evaluate_golden_queries

    retriever = _FakeRetriever({"claim": [_hit("seed_ext_seer_claim_01")]})

    evaluate_golden_queries(
        retriever,
        [
            GoldenQuery(
                query_id="seer_claim",
                role="seer",
                phase="sheriff_speech",
                situation="claim",
                expected_entry_ids=["seed_ext_seer_claim_01"],
            )
        ],
        k_values=(1,),
    )

    assert retriever.queries[0].max_results == 3


def test_default_golden_queries_reference_existing_seed_entries() -> None:
    from werewolf_agent.rag.evaluation import load_golden_queries
    from werewolf_agent.rag.seed_data import create_seed_entries

    config_path = Path("config/rag_eval/golden_queries.yaml")
    queries = load_golden_queries(config_path)
    seed_ids = {entry.entry_id for entry in create_seed_entries()}

    referenced_ids = {
        entry_id
        for query in queries
        for entry_id in [*query.expected_entry_ids, *query.forbidden_entry_ids]
    }

    assert referenced_ids <= seed_ids
