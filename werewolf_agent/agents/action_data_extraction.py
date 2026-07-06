# -*- coding: utf-8 -*-
"""
从 LLM 原始文本中提取 action 或轻量 decision 数据。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.agents.action_data_extraction import extract_decision_data
    >>> extract_decision_data('{"choice": "A"}')
"""

from __future__ import annotations

import json
import re
from html import unescape
from typing import Any

from werewolf_agent.agents.action_normalization import normalize_action_data
from werewolf_agent.agents.json_repair import extract_json_object_candidates


def extract_parameter_tag_action(text: str) -> dict[str, Any] | None:
    """Extract MiniMax-style <parameter name="...">value</parameter> tool payloads."""
    pairs = re.findall(
        r"<parameter\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</parameter>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not pairs:
        return None

    data: dict[str, Any] = {}
    for key, raw_value in pairs:
        value = unescape(raw_value.strip())
        if value.lower() in {"null", "none"}:
            data[key] = None
        elif key == "confidence":
            try:
                data[key] = float(value)
            except ValueError:
                data[key] = value
        else:
            data[key] = value
    return data if "action_type" in data else None


def extract_partial_decision_data(text: str) -> dict[str, Any] | None:
    """Recover discriminator fields from a truncated JSON object.

    This is intentionally narrow: only short completed scalar fields that
    identify a target-choice decision are recovered. Long free-form fields
    such as speech/reason are left to the existing repair pipeline to
    synthesize from the chosen legal target.
    """
    if "{" not in text:
        return None
    data: dict[str, Any] = {}
    for key in ("choice", "action_type", "target_id"):
        match = re.search(rf'"{key}"\s*:\s*"([^"\\]*)"', text)
        if match:
            data[key] = match.group(1)
    confidence = re.search(r'"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if confidence:
        try:
            data["confidence"] = float(confidence.group(1))
        except ValueError:
            pass
    if "choice" in data or "target_id" in data:
        return normalize_action_data(data)
    return None


def extract_decision_data(text: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return normalize_action_data(data), None
        return None, "Decision JSON must be an object"
    except json.JSONDecodeError as direct_error:
        parameter_data = extract_parameter_tag_action(cleaned)
        if parameter_data is not None:
            return normalize_action_data(parameter_data), None
        candidates = extract_json_object_candidates(cleaned)
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return normalize_action_data(data), None
        partial_data = extract_partial_decision_data(cleaned)
        if partial_data is not None:
            return partial_data, None
        return None, f"No JSON object found in output: {direct_error}"
