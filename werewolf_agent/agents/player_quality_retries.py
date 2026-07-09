# -*- coding: utf-8 -*-
"""
构造玩家行动语义质量失败的重试提示。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-09

使用示例:
    >>> from werewolf_agent.agents.player_quality_retries import build_speech_quality_retry
    >>> build_speech_quality_retry("发言过于空洞", attempt=1, max_retries=3).error_code
    'speech_quality'
"""

from __future__ import annotations

from werewolf_agent.agents.schemas import RetryInfo


def _speech_quality_correction_hint(speech_quality_err: str) -> str:
    """按具体 speech_quality 失败类型生成更可执行的重试提示。"""
    if "公开记录" in speech_quality_err or "我推测/我质疑" in speech_quality_err:
        return (
            "把无法确认的公开记录改写为“我推测/我质疑”；"
            "不要继续声称公开记录已经证明。"
            "然后补一句身份立场和一个明确攻击或防御论点。"
        )
    if "身份立场" in speech_quality_err or "我是好人阵营" in speech_quality_err:
        return (
            "先补一句身份立场，例如“我是好人阵营”。"
            "再基于一条公开发言、票型或查验声明给出攻击或防御论点。"
        )
    return (
        f"发言缺少以下必填字段: {speech_quality_err}。"
        f"请基于公开记录重写发言，在 speech 字段中体现："
        f"1) 你的身份立场（至少引用一处公开事实）；"
        f"2) 攻击或防御的明确论点（PK 阶段必填）。"
        f"不要写「按公开信息判断」之类的占位文本。"
    )


def build_speech_quality_retry(
    speech_quality_err: str,
    *,
    attempt: int,
    max_retries: int,
) -> RetryInfo:
    return RetryInfo(
        attempt=attempt,
        max_retries=max_retries,
        error_code="speech_quality",
        error_message=speech_quality_err,
        correction_hint=_speech_quality_correction_hint(speech_quality_err),
    )


def build_vote_quality_retry(
    vote_quality_err: str,
    *,
    attempt: int,
    max_retries: int,
) -> RetryInfo:
    return RetryInfo(
        attempt=attempt,
        max_retries=max_retries,
        error_code="vote_quality",
        error_message=vote_quality_err,
        correction_hint=(
            f"投票理由缺少以下必填字段: {vote_quality_err}。"
            f"请基于以下公开来源重写 vote reason："
            f"1) 预言家查杀声明（金水/查杀 + 报验人+夜数）；"
            f"2) 票型异常（谁跟谁、票型突变）；"
            f"3) 警徽流状态（撕徽/未撕）；"
            f"4) 公开记录里的具体发言引用。"
            f"不要写「综合分析」之类的占位文本。"
        ),
    )


__all__ = [
    "build_speech_quality_retry",
    "build_vote_quality_retry",
]
