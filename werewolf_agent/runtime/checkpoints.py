"""Checkpoint and pause/resume support using langgraph-checkpoint."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver


def make_checkpointer() -> MemorySaver:
    return MemorySaver()
