"""Public evidence index tests.

The index is the single deterministic source for public claim/vote anchors used
by belief, seer credibility, and offline judges.
"""

from __future__ import annotations

from werewolf_agent.cognition.world_state import StructuredFact


def _fact(
    fact_type: str,
    *,
    source: str = "p08",
    target: str = "p01",
    value: str = "",
    day: int = 1,
) -> StructuredFact:
    return StructuredFact(
        fact_type=fact_type,
        source_player=source,
        target_player=target,
        value=value,
        day=day,
    )


def test_public_evidence_records_claim_anchors() -> None:
    from werewolf_agent.cognition.public_evidence import PublicEvidenceIndex

    index = PublicEvidenceIndex()
    index.observe(_fact("claimed_role", source="p08", target="", value="seer"))
    index.observe(_fact("seer_check_claim", source="p08", target="p01", value="wolf"))
    index.observe(_fact("seer_check_claim", source="p08", target="p07", value="good"))
    index.observe(_fact("claimed_suspect", source="p06", target="p04", value="wolf"))

    assert index.supports_reference("p01", "black_check")
    assert index.supports_reference("p07", "gold_water")
    assert index.supports_reference("p08", "seer_claim")
    assert index.supports_reference("p04", "public_suspect")
    assert not index.supports_reference("p05", "black_check")


def test_public_evidence_snapshot_round_trip() -> None:
    from werewolf_agent.cognition.public_evidence import PublicEvidenceIndex

    index = PublicEvidenceIndex()
    index.observe(_fact("claimed_role", source="p08", target="", value="seer"))
    index.observe(_fact("seer_check_claim", source="p08", target="p01", value="wolf"))
    index.observe(_fact("vote", source="p03", target="p01"))

    restored = PublicEvidenceIndex.from_snapshot(index.snapshot())

    assert restored.supports_reference("p01", "black_check")
    assert restored.supports_reference("p08", "seer_claim")
    assert restored.vote_targets("p03") == {"p01"}


def test_vote_signal_uses_only_prior_claims() -> None:
    from werewolf_agent.cognition.public_evidence import PublicEvidenceIndex

    index = PublicEvidenceIndex()
    vote_before_claim = _fact("vote", source="p03", target="p01")

    assert index.vote_delta(vote_before_claim) == 0.0

    index.observe(_fact("seer_check_claim", source="p08", target="p01", value="wolf"))

    assert index.vote_delta(vote_before_claim) > 0.0


def test_incremental_index_preserves_prior_claim_for_later_vote() -> None:
    from werewolf_agent.cognition.public_evidence import PublicEvidenceIndex

    index = PublicEvidenceIndex()
    index.observe(_fact("seer_check_claim", source="p08", target="p01", value="wolf"))
    later_vote = _fact("vote", source="p03", target="p01")

    assert index.vote_delta(later_vote) > 0.0
