"""R15: elapsed_ms in the embedding/reranker HTTP clients is wired
to a debug log line so operators can see per-call latency without
having to scrape metrics endpoints. The clients are HTTP-only and
do not own the audit log, so a module-level logger is sufficient.
"""

from __future__ import annotations

import logging
from unittest import mock


def test_embedding_client_logs_elapsed_ms(caplog) -> None:
    """R15: SiliconFlowEmbeddingClient.embed must emit a DEBUG log
    line carrying ``elapsed_ms=`` so the per-call latency is visible
    to anyone running with log level DEBUG.
    """
    from werewolf_agent.rag.embedding_client import SiliconFlowEmbeddingClient

    # Build a fake httpx response so we don't need a real API key.
    fake_response = mock.Mock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "data": [
            {"index": 0, "embedding": [0.1, 0.2, 0.3]},
        ],
    }
    fake_http = mock.Mock()
    fake_http.post.return_value = fake_response

    client = SiliconFlowEmbeddingClient(
        api_key="fake-key",
        http_client=fake_http,
    )
    with caplog.at_level(logging.DEBUG, logger="werewolf_agent.rag.embedding_client"):
        client.embed(["hello world"])

    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("elapsed_ms" in r.getMessage() for r in debug_records), (
        "R15: SiliconFlowEmbeddingClient.embed must emit a DEBUG log "
        "with elapsed_ms=... for the HTTP call. "
        f"Got records: {[r.getMessage() for r in debug_records]}"
    )


def test_reranker_client_logs_elapsed_ms(caplog) -> None:
    """R15: SiliconFlowRerankerClient.rerank must emit a DEBUG log
    line carrying ``elapsed_ms=`` so the per-call latency is visible.
    """
    from werewolf_agent.rag.reranker_client import SiliconFlowRerankerClient

    fake_response = mock.Mock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "results": [
            {"index": 0, "document": {"text": "doc0"}, "relevance_score": 0.9},
        ],
    }
    fake_http = mock.Mock()
    fake_http.post.return_value = fake_response

    client = SiliconFlowRerankerClient(
        api_key="fake-key",
        http_client=fake_http,
    )
    with caplog.at_level(logging.DEBUG, logger="werewolf_agent.rag.reranker_client"):
        client.rerank("query", ["doc0"])

    debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("elapsed_ms" in r.getMessage() for r in debug_records), (
        "R15: SiliconFlowRerankerClient.rerank must emit a DEBUG log "
        "with elapsed_ms=... for the HTTP call. "
        f"Got records: {[r.getMessage() for r in debug_records]}"
    )
