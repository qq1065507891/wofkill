"""RAG schema tests.

Targets schema-level invariants (field existence, defaults) that are
not covered by the pipeline tests in test_rag.py. Specific to the
P2 polish batch (R14..R20); older schema tests live in test_rag.py.
"""

from __future__ import annotations

import pytest

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
