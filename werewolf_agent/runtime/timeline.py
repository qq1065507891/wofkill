"""Canonical player-facing timeline labels.

Internal counters stay numeric, but prompts and broadcasts should use these
labels so agents do not infer that D1 happens before N1.
"""

from __future__ import annotations

import re
from typing import Literal

PhaseKind = Literal["night", "day"]

TIMELINE_ORDER_NOTE = (
    "时间顺序为 N1 首夜 -> D1 第一天 -> N2 第二夜 -> D2 第二天；"
    "首夜发生在第一天之前，D1 是首夜之后的第一个白天。"
    "如果提到第一天，默认已经经历过首夜结算；不要把首夜理解为第一天之后的夜晚。"
)

TIMELINE_ABSOLUTE_ORDER = [
    "N1 / 首夜",
    "D1 / 第一天",
    "N2 / 第二夜",
    "D2 / 第二天",
]

NIGHT_ONE_ACTIONS = ["狼人刀人", "预言家验人", "女巫用药", "混血儿选主人"]


def phase_code(kind: PhaseKind, number: int) -> str:
    prefix = "N" if kind == "night" else "D"
    return f"{prefix}{number}"


def phase_name(kind: PhaseKind, number: int) -> str:
    if number <= 0:
        return "未开始"
    if number == 1:
        return "首夜" if kind == "night" else "第一天"
    return f"第{_chinese_ordinal(number)}{'夜' if kind == 'night' else '天'}"


def phase_label(kind: PhaseKind, number: int) -> str:
    return f"{phase_code(kind, number)} / {phase_name(kind, number)}"


def current_phase_label(phase: str, *, day_number: int, night_number: int) -> str:
    if phase == "night":
        return phase_label("night", night_number)
    if phase == "day":
        return phase_label("day", day_number)
    return ""


def build_timeline_facts(
    phase: str,
    *,
    day_number: int,
    night_number: int,
) -> dict[str, object]:
    """Return machine-readable timeline facts for agent prompts.

    The wording is intentionally repetitive: models often treat "第一天" as
    the start of the game unless the N1/D1 ordering is explicit and structured.
    """
    current_label = current_phase_label(
        phase,
        day_number=day_number,
        night_number=night_number,
    )
    previous_label = ""
    next_label = ""
    if phase == "day" and day_number == 1:
        previous_label = phase_label("night", 1)
        next_label = phase_label("night", 2)
    elif phase == "night" and night_number == 1:
        next_label = phase_label("day", 1)
    elif phase == "night" and night_number > 1:
        previous_label = phase_label("day", night_number - 1)
        next_label = phase_label("day", night_number)
    elif phase == "day" and day_number > 1:
        previous_label = phase_label("night", day_number)
        next_label = phase_label("night", day_number + 1)

    return {
        "current_phase_label": current_label,
        "previous_phase_label": previous_label,
        "next_phase_label": next_label,
        "absolute_order": list(TIMELINE_ABSOLUTE_ORDER),
        "first_night_before_first_day": True,
        "night_one_definition": "N1 首夜是游戏开始后的第一个夜晚，发生在 D1 第一天之前",
        "day_one_definition": "D1 是首夜 N1 结算后的第一个白天",
        "night_one_actions": list(NIGHT_ONE_ACTIONS),
        "anti_confusion_rule": "不要说或推理成第一天之后才发生首夜",
    }


def detect_timeline_confusion(text: str) -> list[dict[str, str]]:
    """Detect common first-night-after-first-day phrasing in model output."""
    if not text:
        return []

    patterns = [
        r"第一天.{0,20}(之后|后|结束后).{0,20}首夜",
        r"第一天.{0,20}(晚上|夜里).{0,12}(才|再)?(进入|开始|到)?首夜",
        r"警上.{0,20}(之后|后|结束后).{0,20}首夜",
        r"首夜.{0,12}(在|发生在).{0,20}第一天.{0,8}(之后|后)",
    ]
    findings: list[dict[str, str]] = []
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            findings.append({
                "type": "first_night_after_first_day",
                "matched_text": match.group(0),
                "correction": "首夜 N1 发生在第一天 D1 之前；第一天是首夜结算后的第一个白天。",
            })
    return findings


def _chinese_ordinal(number: int) -> str:
    numerals = {
        0: "零",
        1: "一",
        2: "二",
        3: "三",
        4: "四",
        5: "五",
        6: "六",
        7: "七",
        8: "八",
        9: "九",
        10: "十",
    }
    if number in numerals:
        return numerals[number]
    if number < 20:
        return f"十{numerals[number - 10]}"
    tens, ones = divmod(number, 10)
    if ones == 0:
        return f"{numerals[tens]}十"
    return f"{numerals[tens]}十{numerals[ones]}"
