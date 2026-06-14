"""RAG schema tests.

Targets schema-level invariants (field existence, defaults) that are
not covered by the pipeline tests in test_rag.py. Specific to the
P2 polish batch (R14..R20); older schema tests live in test_rag.py.
"""

from __future__ import annotations

import importlib
import json

import pytest
from pydantic import ValidationError

from werewolf_agent.rag import schemas as rag_schemas
from werewolf_agent.rag import seed_data
from werewolf_agent.rag.schemas import RAGQuery


def test_rag_query_no_viewer_role_field() -> None:
    """R14: ``viewer_role`` was a dead field — set on RAGQuery but never
    read by the retriever or the injector. Removing it shrinks the
    schema and prevents future callers from setting it expecting
    downstream filtering (none exists).
    """
    assert "viewer_role" not in RAGQuery.model_fields, (
        "R14: RAGQuery.viewer_role is a dead field; remove it from the "
        "schema and from every call site that sets it."
    )


def test_rag_query_can_be_built_without_viewer_role() -> None:
    """R14: the public surface must let callers build a RAGQuery without
    ever mentioning viewer_role (default-only construction).
    """
    q = RAGQuery(role="seer", phase="speech")
    assert q.role == "seer"
    assert q.phase == "speech"
    # The attribute is not defined, so AttributeError would be expected
    # if a caller tries to read it.
    with pytest.raises(AttributeError):
        _ = q.viewer_role


def _metadata() -> rag_schemas.CaseMetadata:
    return rag_schemas.CaseMetadata(
        case_type=rag_schemas.CaseType.EXTERNAL_TACTICS,
        quality_grade=rag_schemas.QualityGrade.EXPERT_REVIEW,
        review_status=rag_schemas.ReviewStatus.APPROVED,
        reviewer="schema-test",
        ruleset_id="pre_witch_hunter_idiot_mixed",
        player_count=12,
        phase="day",
        role_perspective="werewolf",
        visibility_boundary=rag_schemas.VisibilityBoundary.PLAYER_PERSPECTIVE,
        source=rag_schemas.SourceMetadata(
            source_type=rag_schemas.SourceType.MANUAL_ENTRY,
        ),
        tags=["pressure", "seer-claim"],
    )


def _frame_model() -> type:
    frame_model = getattr(rag_schemas, "RAGTacticalFrame", None)
    if frame_model is None:
        pytest.fail("RAGTacticalFrame schema is missing")
    return frame_model


def _frame_kwargs() -> dict[str, object]:
    return {
        "situation_signature": "白天抗推预言家时票型过度集中",
        "transferable_lesson": "先解释票型动机，再把焦点转到发言矛盾",
        "applicability": ["狼人白天发言", "外置位需要拆票"],
        "counter_signals": ["真预言家已有强警徽流", "队友票型已经暴露"],
        "recommended_use": "用于白天发言时降低团队抱团感",
        "misuse_risk": "在低信息局强行套用会显得预设视角",
    }


def _frame(**overrides: object):
    kwargs = _frame_kwargs()
    kwargs.update(overrides)
    return _frame_model()(**kwargs)


def _tactical_text_module():
    try:
        return importlib.import_module("werewolf_agent.rag.tactical_text")
    except ModuleNotFoundError:
        pytest.fail("werewolf_agent.rag.tactical_text module is missing")


def test_v2_entry_requires_tactical_frame() -> None:
    with pytest.raises(ValidationError, match="tactical_frame") as exc_info:
        rag_schemas.RAGEntry(
            schema_version=2,
            entry_id="v2_missing_frame",
            title="V2 missing frame",
            summary="legacy summary should not be enough for V2",
            metadata=_metadata(),
        )
    error_text = str(exc_info.value)
    assert "v2_missing_frame" in error_text
    assert "schema_version=2" in error_text
    assert "tactical_frame" in error_text


def test_complete_v2_tactical_frame_can_create_entry() -> None:
    frame = _frame()
    entry = rag_schemas.RAGEntry(
        schema_version=2,
        entry_id="v2_complete",
        title="V2 complete",
        summary="legacy summary can still be present",
        tactical_frame=frame,
        metadata=_metadata(),
    )

    assert entry.schema_version == 2
    assert entry.tactical_frame == frame


def test_v2_entry_can_omit_legacy_summary() -> None:
    entry = rag_schemas.RAGEntry(
        schema_version=2,
        entry_id="v2_no_summary",
        title="V2 no summary",
        tactical_frame=_frame(),
        metadata=_metadata(),
    )

    assert entry.summary == ""


@pytest.mark.parametrize(
    ("field_name", "empty_value"),
    [
        ("situation_signature", "   "),
        ("transferable_lesson", ""),
        ("applicability", []),
        ("counter_signals", []),
        ("recommended_use", " "),
        ("misuse_risk", ""),
    ],
)
def test_tactical_frame_rejects_empty_required_values(
    field_name: str,
    empty_value: object,
) -> None:
    kwargs = _frame_kwargs()
    kwargs[field_name] = empty_value

    with pytest.raises(ValidationError, match=field_name):
        _frame_model()(**kwargs)


def test_build_rag_retrieval_text_uses_v2_values_not_field_names() -> None:
    module = _tactical_text_module()
    entry = rag_schemas.RAGEntry(
        schema_version=2,
        entry_id="v2_retrieval",
        title="V2 retrieval",
        tactical_frame=_frame(),
        metadata=_metadata(),
    )

    text = module.build_rag_retrieval_text(entry, max_chars=1500)

    for value in (
        "白天抗推预言家时票型过度集中",
        "先解释票型动机，再把焦点转到发言矛盾",
        "狼人白天发言",
        "外置位需要拆票",
        "真预言家已有强警徽流",
        "队友票型已经暴露",
        "用于白天发言时降低团队抱团感",
        "在低信息局强行套用会显得预设视角",
    ):
        assert value in text
    for field_name in _frame_kwargs():
        assert field_name not in text
    assert len(text) <= 1500


def test_get_prompt_tactical_frame_uses_safe_legacy_fallback() -> None:
    module = _tactical_text_module()
    legacy_summary = ("A" * 800) + "TAIL_SHOULD_NOT_APPEAR"
    entry = rag_schemas.RAGEntry(
        schema_version=1,
        entry_id="legacy_001",
        title="Legacy entry",
        summary=legacy_summary,
        key_decisions=["legacy decision"],
        metadata=_metadata(),
    )

    frame = module.get_prompt_tactical_frame(entry)
    encoded = json.dumps(frame, ensure_ascii=False)

    assert set(frame) == set(_frame_kwargs())
    assert "A" * 800 in encoded
    assert "TAIL_SHOULD_NOT_APPEAR" not in encoded
    for english_fallback in (
        "Legacy RAG entry",
        "Use only when",
        "Current situation",
        "Use as a cautious reference",
        "Legacy summary lacks",
    ):
        assert english_fallback not in encoded
    for audit_or_identity_field in (
        "entry_id",
        "quality_grade",
        "source_type",
        "visibility_boundary",
        "reviewer",
    ):
        assert audit_or_identity_field not in encoded


def test_seed_build_entry_treats_missing_schema_version_as_legacy() -> None:
    entry = seed_data._build_entry({
        "entry_id": "legacy_seed_missing_schema",
        "title": "Legacy seed",
        "summary": "Existing seed without tactical_frame",
        "metadata": {
            "case_type": "external_tactics",
            "quality_grade": "expert_review",
            "review_status": "approved",
            "ruleset_id": "pre_witch_hunter_idiot_mixed",
            "phase": "day",
            "role_perspective": "werewolf",
            "visibility_boundary": "player_perspective",
            "source": {"source_type": "manual_entry"},
        },
    })

    assert entry.schema_version == 1
    assert entry.tactical_frame is None
