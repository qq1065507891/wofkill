# -*- coding: utf-8 -*-
"""
测试运行时 JSON 对象提取工具。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.runtime.json_extract import extract_first_balanced_json_object
    >>> extract_first_balanced_json_object('prefix {"a": 1}')
"""

from werewolf_agent.runtime.json_extract import extract_first_balanced_json_object


def test_extract_first_balanced_json_object_reads_plain_object() -> None:
    """能够读取文本中的第一个完整 JSON 对象。"""
    assert extract_first_balanced_json_object('prefix {"a": 1} suffix') == {"a": 1}


def test_extract_first_balanced_json_object_keeps_braces_inside_strings() -> None:
    """字符串里的花括号不应该影响对象边界判断。"""
    text = 'noise {"msg": "has } in str", "nested": {"ok": true}} trailing'

    assert extract_first_balanced_json_object(text) == {
        "msg": "has } in str",
        "nested": {"ok": True},
    }


def test_extract_first_balanced_json_object_skips_invalid_candidate() -> None:
    """遇到无效候选对象后继续寻找下一个完整对象。"""
    text = 'bad {"a": } then good {"b": 2}'

    assert extract_first_balanced_json_object(text) == {"b": 2}


def test_extract_first_balanced_json_object_returns_none_when_missing() -> None:
    """没有完整对象时返回 None。"""
    assert extract_first_balanced_json_object("plain text") is None
    assert extract_first_balanced_json_object('{"a": 1') is None
