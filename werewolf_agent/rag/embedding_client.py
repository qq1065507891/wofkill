# -*- coding: utf-8 -*-
"""
功能描述：调用 SiliconFlow 嵌入 API 获取语义向量，后端为 BAAI/bge-large-zh-v1.5。
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from werewolf_agent.model_gateway.providers import load_local_dotenv


logger = logging.getLogger(__name__)


class EmbeddingClientError(RuntimeError):
    """Raised when embedding API call fails."""


class SiliconFlowEmbeddingClient:
    """HTTP client for SiliconFlow embeddings API (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        model: str = "BAAI/bge-large-zh-v1.5",
        http_client: Any | None = None,
    ) -> None:
        load_local_dotenv()
        self._api_key = api_key or os.getenv("SILICONFLOW_API_KEY", "")
        if not self._api_key:
            raise EmbeddingClientError(
                "SILICONFLOW_API_KEY is required for SiliconFlow embedding. "
                "Set it in .env or environment."
            )
        self._base_url = (base_url or os.getenv(
            "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn"
        )).rstrip("/")
        self._model = model
        self._http = http_client or httpx.Client()
        self._owns_http = http_client is None

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Returns a list of embedding vectors, one per input text.
        """
        if not texts:
            return []
        start = time.monotonic()
        response = self._http.post(
            f"{self._base_url}/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "input": texts,
            },
            timeout=30,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        # R15: surface per-call latency to operators running with
        # log level DEBUG. Embedding is the most latency-sensitive
        # RAG hot path; the number is the wall-clock time from
        # request submission to response receipt.
        logger.debug(
            "embedding API call: model=%s batch_size=%d elapsed_ms=%d",
            self._model, len(texts), elapsed_ms,
        )
        if response.status_code != 200:
            raise EmbeddingClientError(
                f"Embedding API returned {response.status_code}: "
                f"{response.text[:500]}"
            )
        data = response.json()
        embeddings = [
            item["embedding"]
            for item in sorted(data["data"], key=lambda x: x["index"])
        ]
        return embeddings

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        results = self.embed([text])
        return results[0]

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        """BAAI/bge-large-zh-v1.5 outputs 1024-dimensional vectors."""
        return 1024
