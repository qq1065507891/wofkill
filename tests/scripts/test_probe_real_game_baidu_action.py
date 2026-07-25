# -*- coding: utf-8 -*-
"""
验证真实游戏行动探针的诊断输出不会泄露运行时修正提示。

作者: Project contributors
创建日期: 2026-07-25

使用示例:
    >>> python -m pytest tests/scripts/test_probe_real_game_baidu_action.py -q
"""

from __future__ import annotations

from types import SimpleNamespace

from werewolf_agent.agents.schemas import ActionType, FallbackAction, RetryInfo


def test_main_redacts_retry_correction_hint_from_stdout(
    monkeypatch,
    capsys,
) -> None:
    """主路径保留稳定审计字段，但不得打印含被拒发言的修正提示。"""
    from scripts import probe_real_game_baidu_action as probe

    sentinel = "REJECTED_SPEECH_SENTINEL"
    retry = RetryInfo(
        attempt=2,
        max_retries=3,
        error_code="speech_quality",
        error_message="缺少身份立场",
        reason_codes=["unsupported_public_claim"],
        correction_hint=f"请定点修改上一条被拒发言：{sentinel}",
        failure_category="unknown",
    )
    action = FallbackAction(
        action_type=ActionType.SPEECH,
        target_id="p05",
        speech="安全兜底发言",
        reason="fallback",
    )

    class FakeRouter:
        def provider_names(self) -> list[str]:
            return ["mock"]

        def resolve_config(self, _agent_id: str, _task_type: str):
            return (
                SimpleNamespace(
                    provider="mock",
                    model="mock-model",
                    allow_text_tool_fallback=True,
                ),
                False,
            )

    class FakeAgent:
        def __init__(self, **_kwargs) -> None:
            pass

        def act(self, _context):
            return action, retry

    monkeypatch.setattr(probe, "load_local_dotenv", lambda: None)
    monkeypatch.setattr(
        probe.ModelRouter,
        "from_yaml",
        lambda *_args, **_kwargs: FakeRouter(),
    )
    monkeypatch.setattr(probe, "PlayerAgent", FakeAgent)
    monkeypatch.setattr(
        probe.sys,
        "argv",
        ["probe_real_game_baidu_action.py", "--task", "speech"],
    )

    assert probe.main() == 0

    stdout = capsys.readouterr().out
    assert retry.correction_hint is not None
    assert sentinel in retry.correction_hint
    assert sentinel not in stdout
    assert "correction_hint" not in stdout
    assert "'error_code': 'speech_quality'" in stdout
    assert "'error_message': '缺少身份立场'" in stdout
    assert "'reason_codes': ['unsupported_public_claim']" in stdout
