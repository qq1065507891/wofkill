# -*- coding: utf-8 -*-
"""
提供 Agent 人设风格读取和任务风格提示。

作者: Project contributors
创建日期: 2026-07-06

使用示例:
    >>> from werewolf_agent.runtime.context_persona import _get_persona_speech_style
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_SPEECH_STYLE_HINTS = {
    "structured_logical": "用严谨的逻辑推理链分析场上信息，像法官一样条理清晰地展示判断依据。",
    "aggressive_short": "用简短犀利、一针见血的质疑制造压力，不需要长篇大论。",
    "moderate_calm": "用平和沉稳的语气，像旁观者一样冷静梳理场上各方的观点。",
    "confident_fake_claim": "表达坚定、自信、带有权威感，即使受到质疑也维持完整叙事并主动反压。",
    "subtle_helpful": "表面温和协作，先补充细节和可验证信息，再含蓄地引导讨论方向。",
    "emotional_vivid": "允许鲜明情绪和生活化表达，但最终要落回一个可以核对的判断依据。",
    "humorous_distracting": "用轻松、机敏和适量反问制造记忆点，同时避免让玩笑取代有效分析。",
    "evidence_based": "优先引用公开事件、原话和票型，用复盘式表达区分事实、推测与结论。",
    "brief_pointed": "少说套话，只抓一个关键细节，短句表达明确判断和后续观察点。",
    "mostly_quiet_then_explosive": "平时克制简短，发现关键矛盾时集中输出完整证据链并给出强判断。",
    "structured_slotting": "按玩家位置或信息链分组盘点，比较各组关系后再给出嫌疑排序。",
    "adaptive_varied": "根据局势切换长短、强弱和分析角度，避免连续使用同一种句式。",
    "emotional_intuitive": "用感性直觉的方式判断人，从'感觉不对'出发再找逻辑支撑。",
    "dramatic_theatrical": "用夸张、戏剧化的表达吸引注意力，善用比喻和反问。",
    "quiet_analytical": "不声不响地默默分析，发言内容重质不重量，专注关键细节。",
    "adaptable_flexible": "根据场上局势灵活调整发言策略，该激进时激进，该保守时保守。",
}

_SHERIFF_SPEECH_STYLE_OVERRIDES = {
    "aggressive_short": "用简短犀利、一针见血的质疑制造压力，不需要长篇大论，每句话都要有攻击性。",
    "moderate_calm": "用平和沉稳的语气，像旁观者一样冷静梳理场上各方的观点，指出其中的合理与矛盾。",
    "emotional_intuitive": "用感性直觉的方式判断人，从'感觉不对'出发再找逻辑支撑，可以适当表达情绪。",
    "dramatic_theatrical": "用夸张、戏剧化的表达吸引注意力，善用比喻和反问，让发言有记忆点。",
    "quiet_analytical": "不声不响地默默分析，发言内容重质不重量，专注于关键细节的挖掘。",
}

_TASK_STYLE_HINTS = {
    "authority_claim": "以领导者姿态出现，主动归纳场上信息，给出明确的方向性判断。",
    "forceful_claim": "用强烈的语气宣称自己的判断，对反对者直接施压。",
    "observation_first": "先全面观察再发言，重点分析别人的发言漏洞和信息差。",
    "mystery_hint": "暗示自己掌握关键信息但不直接亮底牌，制造悬念引导讨论方向。",
    "data_driven": "用事实和可验证的信息构建论证，避免空洞的定性判断。",
    "counterattack": "面对质疑时反击而不是解释，把压力转回给质疑者。",
}

_PERSONA_PROFILES_CACHE: dict[str, dict[str, Any]] | None = None
_PERSONA_PROFILES_LOCK: threading.Lock = threading.Lock()


def _load_persona_profile(persona_key: str) -> dict[str, Any]:
    global _PERSONA_PROFILES_CACHE
    with _PERSONA_PROFILES_LOCK:
        if _PERSONA_PROFILES_CACHE is None:
            _PERSONA_PROFILES_CACHE = {}
            try:
                path = (
                    Path(__file__).resolve().parent.parent.parent
                    / "config"
                    / "personas"
                    / "jingcheng_style_prototypes.yaml"
                )
                if path.exists():
                    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                    _PERSONA_PROFILES_CACHE = data.get("persona_profiles", {})
            except Exception:
                logger.warning(
                    "Failed to load persona profiles, all agents will use generic speech styles",
                    exc_info=True,
                )
    return _PERSONA_PROFILES_CACHE.get(persona_key, {})


def _get_persona_speech_style(agent: Any) -> str:
    if not agent or not getattr(agent, "persona_key", None):
        return ""
    profile = _load_persona_profile(agent.persona_key)
    return profile.get("base", {}).get("speech_style", "")


def _get_persona_task_style(agent: Any, task_key: str) -> str:
    if not agent or not getattr(agent, "persona_key", None):
        return ""
    profile = _load_persona_profile(agent.persona_key)
    task_styles = profile.get("task_styles", {})
    return task_styles.get(task_key, task_styles.get("speech", ""))
