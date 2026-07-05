# -*- coding: utf-8 -*-
"""
修复 LLM 输出中的类 JSON 文本并提取动作 JSON 候选。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.agents.json_repair import repair_json_text
    >>> repair_json_text("{action_type:'vote',}")
"""

from __future__ import annotations

import re

from werewolf_agent.agents.schemas import ActionType


MOJIBAKE_REPLACEMENT_CHAR = "��"


def try_latin1_roundtrip(text: str) -> str | None:
    """通过 latin-1 往返尝试恢复被二次编码的 UTF-8 文本。"""
    try:
        roundtripped = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    if roundtripped == text:
        return None
    return roundtripped


def try_repair_mojibake(text: str) -> str | None:
    """尝试修复常见乱码；无法修复时返回 None。"""
    if MOJIBAKE_REPLACEMENT_CHAR not in text:
        return try_latin1_roundtrip(text)

    roundtripped = try_latin1_roundtrip(text)
    if roundtripped is not None and MOJIBAKE_REPLACEMENT_CHAR not in roundtripped:
        return roundtripped
    return text.replace(MOJIBAKE_REPLACEMENT_CHAR, '"')


def repair_json_text(raw: str) -> str:
    """修复 LLM 输出里常见的 JSON 语法瑕疵。"""
    text = raw.strip()
    text = text.replace("﻿", "").replace("​", "")
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"\bNaN\b", "null", text)
    text = re.sub(r"\bInfinity\b|\binf\b", "null", text, flags=re.IGNORECASE)

    mojibake_repaired = try_repair_mojibake(text)
    if mojibake_repaired is not None:
        text = mojibake_repaired

    text = re.sub(r"(?<!\\)'([^']*?)'", r'"\1"', text)
    text = re.sub(
        r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:',
        r'\1"\2":',
        text,
    )
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r",\s*,", ",", text)
    return text


def extract_json_object_candidates(text: str) -> list[str]:
    """从混合文本中提取看起来像玩家动作的平衡 JSON 对象。"""
    candidates: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False

    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                candidates.append(text[start:idx + 1])
                start = None

    action_candidates = [
        candidate for candidate in candidates
        if _looks_like_action_candidate(candidate)
    ]
    if not action_candidates:
        if not candidates:
            return []
        raise ValueError(
            "no_action_type_found: extract_json_object_candidates found "
            f"{len(candidates)} balanced JSON object(s) but none carried an "
            f"action_type discriminator (first-key form, or "
            f"action_type field). Refusing to fall back to non-action JSON."
        )
    return action_candidates


def _looks_like_action_candidate(candidate: str) -> bool:
    if '"action_type"' in candidate or "'action_type'" in candidate:
        return True
    if '"choice"' in candidate or "'choice'" in candidate:
        return True
    if '"intent"' in candidate or "'intent'" in candidate:
        return True

    first_key_match = re.match(
        r"""^\s*\{\s*(?:"([^"]+)"|'([^']+)'|([A-Za-z_][A-Za-z0-9_]*))\s*:""",
        candidate,
    )
    if first_key_match is None:
        return False
    first_key = first_key_match.group(1) or first_key_match.group(2) or first_key_match.group(3)
    return first_key in {action.value for action in ActionType}
