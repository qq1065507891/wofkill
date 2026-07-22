# -*- coding: utf-8 -*-
"""
功能描述：**：为反馈回路评估轨迹提供模块归因指标
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-22
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from werewolf_agent.evaluation.feedback_schemas import (
    EvaluationTrace,
    MetricSupport,
    ModuleExposure,
)


@dataclass(frozen=True)
class ModuleAttributionSummary:
    module: str
    exposure_count: int = 0
    supported_count: int = 0
    unsupported_count: int = 0
    prompt_visible_count: int = 0
    cited_count: int = 0
    aligned_count: int = 0
    harmful_count: int = 0
    # 2026-07-22 R9a: LLM prompt cache 字段 (R2/R6 写, R7 落 sink, R9a 折入统一模板).
    # 真实累计路径是 build_feedback_report 注入 "llm_cache" summary entry;
    # summarize_module_attribution 不会自动累计 (cache 数据来自 CostMetrics 不来自
    # EvaluationTrace). 默认 0 兼容既有 entry.
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def prompt_visible_rate(self) -> float:
        return _rate(self.prompt_visible_count, self.supported_count)

    @property
    def citation_rate(self) -> float:
        return _rate(self.cited_count, self.supported_count)

    @property
    def alignment_rate(self) -> float:
        return _rate(self.aligned_count, self.supported_count)

    @property
    def harmful_rate(self) -> float:
        return _rate(self.harmful_count, self.supported_count)

    @property
    def cache_hit_ratio(self) -> float:
        """R9a: cache_read / (cache_creation + cache_read). cache_creation 不计入
        因为首次写入不属于"命中". ModuleAttributionSummary 没有 prompt_tokens 字段,
        所以用 cache_creation 作"基线"代理 (无 LLM call 时 total=0 ratio=0)."""
        total = self.cache_creation_tokens + self.cache_read_tokens
        if total <= 0:
            return 0.0
        return self.cache_read_tokens / total


def summarize_module_attribution(
    traces: list[EvaluationTrace],
) -> dict[str, ModuleAttributionSummary]:
    """Aggregate module exposure attribution metrics by module name."""
    mutable: dict[str, dict[str, int]] = {}
    for trace in traces:
        for exposure in trace.module_exposures:
            module = exposure.module
            if not module:
                continue
            stats = mutable.setdefault(
                module,
                {
                    "exposure_count": 0,
                    "supported_count": 0,
                    "unsupported_count": 0,
                    "prompt_visible_count": 0,
                    "cited_count": 0,
                    "aligned_count": 0,
                    "harmful_count": 0,
                },
            )
            stats["exposure_count"] += 1
            if exposure.support == MetricSupport.UNSUPPORTED:
                stats["unsupported_count"] += 1
            else:
                stats["supported_count"] += 1
            if exposure.prompt_visible:
                stats["prompt_visible_count"] += 1
            if exposure.cited_by_decision:
                stats["cited_count"] += 1
            if exposure.aligned_with_decision:
                stats["aligned_count"] += 1
            if _is_harmful_transfer(exposure):
                stats["harmful_count"] += 1

    return {
        module: ModuleAttributionSummary(module=module, **stats)
        for module, stats in sorted(mutable.items())
    }


def _llm_cache_summary(cost: Any) -> "ModuleAttributionSummary":
    """R9a: 把 CostMetrics (R7) 合成 synthetic llm_cache module summary entry.

    真实数据源: R2 (Anthropic/MiniMax) + R6 (OpenAI/GLM) 解析的 cache_*_tokens.
    其它 exposure 字段 (supported/cited/harmful) 全部 0, 因为 cache 不是 exposure 路径.
    走 ModuleAttributionSummary 模板意味着 R8 的特设 cache_stats 字段被折入
    统一 module_metrics 模板, LangSmith dashboard 0 改动看到 cache_hit_ratio.

    cost=None 时返回空 entry (exposure_count=0, cache_* 字段=0).
    """
    if cost is None:
        return ModuleAttributionSummary(module="llm_cache")
    return ModuleAttributionSummary(
        module="llm_cache",
        # exposure_count = 1 (一次 module attribution snapshot, 让
        # _module_metric_to_dict 投影).
        exposure_count=1,
        supported_count=1,  # cache stats 总是 "supported" (有数据).
        cache_creation_tokens=cost.total_cache_creation_tokens,
        cache_read_tokens=cost.total_cache_read_tokens,
    )


def _is_harmful_transfer(exposure: ModuleExposure) -> bool:
    return (
        exposure.metadata.get("harmful_transfer") is True
        or exposure.metadata.get("harmful") is True
    )


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
