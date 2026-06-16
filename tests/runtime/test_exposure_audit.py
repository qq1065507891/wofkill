from __future__ import annotations

from werewolf_agent.evaluation.trace_identity import DecisionIdentity
from werewolf_agent.runtime.exposure_audit import ModuleExposureAuditCollector


def _identity() -> DecisionIdentity:
    return DecisionIdentity("g1", "p01", "vote", 2, 1, "vote", 4)


def test_collector_builds_rag_reflection_skill_persona_events() -> None:
    collector = ModuleExposureAuditCollector()
    collector.record_rag(_identity(), [{"entry_id": "rag1", "rank": 1, "title": "safe"}])
    collector.record_reflection(_identity(), [{"entry_id": "ref1", "rank": 1}])
    collector.record_skill(_identity(), {"vote_analysis": "push p02"})
    collector.record_persona(_identity(), {"profile_id": "aggressive", "effective_params": {"risk": 0.8}})

    events = collector.flush_events()

    assert [event.type for event in events] == [
        "rag_exposure_audit",
        "reflection_exposure_audit",
        "skill_exposure_audit",
        "persona_exposure_audit",
    ]
    assert all(event.payload["trace_id"] == "g1:p01:vote:D2:N1:vote:4" for event in events)
    assert all(event.payload["visibility"] == "moderator_only" for event in events)


def test_collector_strips_forbidden_private_fields() -> None:
    collector = ModuleExposureAuditCollector()
    collector.record_rag(_identity(), [{"entry_id": "x", "target_role": "werewolf"}])
    payload = collector.flush_events()[0].payload
    assert "target_role" not in str(payload)
    assert "werewolf" not in str(payload)
