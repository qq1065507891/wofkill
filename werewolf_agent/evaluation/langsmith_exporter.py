# -*- coding: utf-8 -*-
"""
功能描述：**：将反馈报告导出至LangSmith，导入时避免引入外部追踪依赖
作者：Mike
创建日期：2025-01-15
修改日期：2026-07-05
使用示例：内部模块，无对外接口
"""

from __future__ import annotations

from typing import Any

from werewolf_agent.evaluation.feedback_report import FeedbackReport


PRIVATE_AUDIT_KEYS = {
    "actual_role",
    "ground_truth",
    "key_assignments",
    "target_faction",
    "target_role",
    "true_world_rank",
}


class LangSmithFeedbackExporter:
    """Build and optionally send LangSmith runs for feedback reports."""

    def __init__(self, project_name: str = "agent-feedback") -> None:
        self.project_name = project_name

    def build_payload(self, report: FeedbackReport) -> dict[str, Any]:
        data = report.to_json_dict()
        return {
            "project_name": self.project_name,
            "runs": [
                {
                    "name": report.report_id,
                    "run_type": "chain",
                    "inputs": {
                        "batch_id": report.batch_id,
                        "source_refs": data["source_refs"],
                    },
                    "outputs": {
                        "trace_count": report.trace_count,
                        "module_metrics": data["module_metrics"],
                        "failure_clusters": data["failure_clusters"],
                        "regression_summary": data["regression_summary"],
                        "candidates": _scrub_private_fields(data["candidates"]),
                        "ablations": data["ablations"],
                    },
                    "metadata": {
                        "batch_id": report.batch_id,
                        "schema_version": report.schema_version,
                        "generated_at": report.generated_at,
                    },
                }
            ],
        }

    def export(
        self,
        report: FeedbackReport,
        *,
        client: Any = None,
    ) -> list[Any]:
        """Export report runs via an injected or lazily imported LangSmith client."""
        langsmith_client = client if client is not None else self._new_client()
        payload = self.build_payload(report)
        results: list[Any] = []
        for run in payload["runs"]:
            results.append(
                langsmith_client.create_run(
                    project_name=payload["project_name"],
                    **run,
                )
            )
        return results

    def _new_client(self) -> Any:
        try:
            from langsmith import Client
        except ImportError as exc:
            raise RuntimeError(
                "LangSmith is not installed; install langsmith or pass an "
                "injected client to export()."
            ) from exc
        return Client()


def _scrub_private_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _scrub_private_fields(item)
            for key, item in value.items()
            if str(key).lower() not in PRIVATE_AUDIT_KEYS
        }
    if isinstance(value, list):
        return [_scrub_private_fields(item) for item in value]
    return value
