# -*- coding: utf-8 -*-
"""
构造 PlayerAgent 行动生成失败后的重试提示。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-21

使用示例:
    >>> from werewolf_agent.agents.player_retry_hints import build_empty_response_retry
"""

from __future__ import annotations

import re

from werewolf_agent.agents.schemas import (
    ActionType,
    AgentContext,
    OutputMode,
    RetryInfo,
)


def build_empty_response_retry(
    *,
    context: AgentContext,
    attempt: int,
    max_retries: int,
    failure_category: str,
    output_mode: OutputMode,
) -> RetryInfo:
    """为模型空响应构造下一轮重试提示。"""
    category_hint = f" (cause: {failure_category})" if failure_category else ""
    timeout_hint = ""
    if failure_category == "timeout":
        can_emit_no_action = (
            ActionType.NO_ACTION in context.legal_actions
            and output_mode == OutputMode.FULL_ACTION
        )
        if can_emit_no_action:
            timeout_hint = (
                " 如果超时，请直接返回 no_action 而非空响应"
                "（action_type='no_action', target_id=null,"
                "reason='timeout - safe no-op'）。"
            )
        elif context.legal_targets:
            first_target = context.legal_targets[0]
            timeout_hint = (
                f" 如果超时，请直接选择一个合法目标 "
                f"（例如 {first_target}）并提交结构化JSON。"
            )
    return RetryInfo(
        attempt=attempt,
        max_retries=max_retries,
        error_code="empty_response",
        error_message="Model returned empty text",
        failure_category=failure_category,
        correction_hint=(
            f"Please provide a valid JSON action{category_hint}. "
            f"If the model timed out, consider shorter reasoning."
            f"{timeout_hint}"
        ),
    )


def build_missing_tool_call_retry(
    *,
    attempt: int,
    max_retries: int,
    structured_failure_reason: str,
) -> RetryInfo:
    """为缺失 submit_player_action 工具调用构造重试提示。"""
    parse_error = "missing required tool call: submit_player_action"
    return RetryInfo(
        attempt=attempt,
        max_retries=max_retries,
        error_code=structured_failure_reason,
        error_message=parse_error,
        correction_hint=(
            "必须通过 submit_player_action 工具调用提交结构化参数；"
            "不要把JSON写在普通文本内容里。"
        ),
    )


# 2026-07-21 R1: 解析 Pydantic v2 ValidationError.__str__() 的字段路径.
#
# 真实 Pydantic 2 输出形如 (实测 pydantic 2.13) ::
#
#     2 validation errors for WolfDiscussionSpeechPlayerAction
#     target_stance
#       Field required [type=missing, input_value=None, input_type=NoneType]
#         For further information visit https://errors.pydantic.dev/2.13/v/missing
#     reason
#       Input should be a valid string [type=string_type, input_value=42, input_type=int]
#         For further information visit https://errors.pydantic.dev/2.13/v/string_type
#
# 我们按 `\n` 切块, 每块第一行是字段路径, 第二行是 msg + [type=...], 第三行是 url.
_PYDANTIC_VALIDATION_HEADER = re.compile(
    r"^\s*\d+\s+validation errors?\s+for\s+\S+\s*$"
)
_PYDANTIC_BLOCK_FIRST_LINE = re.compile(r"^\s*([\w.]+)\s*$")
_PYDANTIC_BLOCK_SECOND_LINE = re.compile(
    r"^\s+(?P<msg>[^\[]+?)\s*\[type=(?P<type>[^,]+),\s*input_value=(?P<input>[^\]]*?),\s*"
    r"input_type=(?P<input_type>[^\]]+?)\]\s*$"
)


def _extract_validation_error_items(parse_error: str) -> list[dict[str, str]]:
    """从 Pydantic ValidationError str 抽取 (loc, msg, type, input) 列表。

    不能解析时返回空列表, 调用方走 fallback (空 hint).
    """
    raw = parse_error.strip()
    # output_parser 把整段 f-string 拼成 "Schema validation error: <...>", 我们
    # 先去掉前缀, 再按 pydantic 2 标准 multi-error 格式切块.
    if raw.startswith("Schema validation error:"):
        raw = raw[len("Schema validation error:"):].lstrip()
    head_line = raw.splitlines()[0] if raw.splitlines() else ""
    if not _PYDANTIC_VALIDATION_HEADER.match(head_line):
        return []

    lines = raw.splitlines()
    body_lines: list[str] = []
    for line in lines[1:]:
        if _PYDANTIC_VALIDATION_HEADER.match(line):
            break
        body_lines.append(line)

    items: list[dict[str, str]] = []
    i = 0
    while i < len(body_lines):
        if "For further information visit" in body_lines[i]:
            i += 1
            continue
        m_first = _PYDANTIC_BLOCK_FIRST_LINE.match(body_lines[i])
        if not m_first:
            i += 1
            continue
        loc = m_first.group(1).strip()
        if i + 1 < len(body_lines):
            m_second = _PYDANTIC_BLOCK_SECOND_LINE.match(body_lines[i + 1])
            if m_second:
                items.append({
                    "loc": loc,
                    "msg": m_second.group("msg").strip(),
                    "type": m_second.group("type").strip(),
                    "input": m_second.group("input").strip(),
                    "input_type": m_second.group("input_type").strip(),
                })
                i += 2
                continue
        items.append({"loc": loc, "msg": "", "type": "", "input": "", "input_type": ""})
        i += 1
    return items


def build_schema_validation_hint(
    parse_error: str,
    *,
    max_items: int = 5,
    max_chars: int = 800,
) -> str:
    """Pydantic schema-validation 失败后, 把字段违规细节灌入 retry hint。

    输入是 ``output_parser.action_from_data`` 返回的 parse_error 字符串:
    ``"Schema validation error: <pydantic ValidationError str()>"``.

    输出形如::

        上一次 JSON 输出 Pydantic 校验失败。请保留 schema 其它字段,
        只修正下面违规字段, 不要重写整个 JSON。

        违规字段 (最多 5 条):
        - 路径 `target_stance`: Field required (当前值: None)
        - 路径 `reason`: Input should be a valid string (当前值: 42)

    非 Pydantic ValidationError 输入 (例如 ``truncated_json: ...``) 返回空字符串,
    caller 走原来的空泛兜底语分支.
    """
    items = _extract_validation_error_items(parse_error)
    if not items:
        return ""

    lines = [
        "上一次 JSON 输出 Pydantic 校验失败。请保留 schema 其它字段，"
        "只修正下面违规字段，不要重写整个 JSON。",
        "",
        f"违规字段 (最多 {max_items} 条):",
    ]
    head = items[:max_items]
    for it in head:
        if it.get("input"):
            lines.append(
                f"- 路径 `{it['loc']}`: {it['msg']} (当前值: {it['input']})"
            )
        else:
            lines.append(f"- 路径 `{it['loc']}`: {it['msg']}")
    if len(items) > max_items:
        lines.append(
            f"... 另有 {len(items) - max_items} 条违规未列出，请优先修正以上字段。"
        )
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars] + "…"
    return out


__all__ = [
    "build_empty_response_retry",
    "build_missing_tool_call_retry",
    "build_schema_validation_hint",
]
