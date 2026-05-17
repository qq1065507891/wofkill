"""SiliconFlow reranker API client.

Provides BAAI/bge-reranker-v2-m3 reranking over retrieved documents.
API key read from SILICONFLOW_API_KEY env var, loaded from .env if present.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from werewolf_agent.model_gateway.providers import load_local_dotenv


class RerankerClientError(RuntimeError):
    """Raised when reranker API call fails."""


class SiliconFlowRerankerClient:
    """HTTP client for SiliconFlow reranker API."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        model: str = "BAAI/bge-reranker-v2-m3",
        http_client: Any | None = None,
    ) -> None:
        load_local_dotenv()
        self._api_key = api_key or os.getenv("SILICONFLOW_API_KEY", "")
        if not self._api_key:
            raise RerankerClientError(
                "SILICONFLOW_API_KEY is required for SiliconFlow reranker. "
                "Set it in .env or environment."
            )
        self._base_url = (base_url or os.getenv(
            "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn"
        )).rstrip("/")
        self._model = model
        self._http = http_client or httpx.Client()

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank documents by relevance to the query.

        Returns list of {index, text, relevance_score} sorted by score desc.
        """
        if not documents:
            return []
        top_n = top_n or len(documents)
        start = time.monotonic()
        response = self._http.post(
            f"{self._base_url}/v1/rerank",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "return_documents": True,
            },
            timeout=30,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        if response.status_code != 200:
            raise RerankerClientError(
                f"Reranker API returned {response.status_code}: "
                f"{response.text[:500]}"
            )
        data = response.json()
        return [
            {
                "index": item["index"],
                "text": item.get("document", {}).get("text", ""),
                "relevance_score": item["relevance_score"],
            }
            for item in data["results"]
        ]

    def rerank_hits(
        self,
        query: str,
        documents: list[dict[str, Any]],
        text_key: str = "text",
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank structured document dicts, preserving metadata.

        Each document dict must have a text field (keyed by text_key).
        Returns the same dicts augmented with 'rerank_score'.
        """
        if not documents:
            return []
        texts = [d[text_key] for d in documents]
        results = self.rerank(query, texts, top_n=top_n)
        reranked: list[dict[str, Any]] = []
        for r in results:
            doc = dict(documents[r["index"]])
            doc["rerank_score"] = r["relevance_score"]
            reranked.append(doc)
        return reranked

    @property
    def model(self) -> str:
        return self._model
