from werewolf_agent.runtime.timeline import (
    TIMELINE_ORDER_NOTE,
    build_timeline_facts,
    detect_timeline_confusion,
    phase_code,
    phase_label,
    phase_name,
)


def test_phase_labels_name_first_night_before_first_day() -> None:
    assert phase_code("night", 1) == "N1"
    assert phase_name("night", 1) == "首夜"
    assert phase_label("night", 1) == "N1 / 首夜"

    assert phase_code("day", 1) == "D1"
    assert phase_name("day", 1) == "第一天"
    assert phase_label("day", 1) == "D1 / 第一天"


def test_timeline_order_is_night_one_then_day_one() -> None:
    labels = [
        phase_label("night", 1),
        phase_label("day", 1),
        phase_label("night", 2),
        phase_label("day", 2),
    ]

    assert labels == ["N1 / 首夜", "D1 / 第一天", "N2 / 第二夜", "D2 / 第二天"]
    assert "N1 首夜 -> D1 第一天 -> N2 第二夜 -> D2 第二天" in TIMELINE_ORDER_NOTE
    assert "首夜发生在第一天之前" in TIMELINE_ORDER_NOTE


def test_timeline_facts_make_first_night_order_machine_readable() -> None:
    facts = build_timeline_facts("day", day_number=1, night_number=1)

    assert facts["current_phase_label"] == "D1 / 第一天"
    assert facts["absolute_order"] == ["N1 / 首夜", "D1 / 第一天", "N2 / 第二夜", "D2 / 第二天"]
    assert facts["first_night_before_first_day"] is True
    assert facts["day_one_definition"] == "D1 是首夜 N1 结算后的第一个白天"
    assert facts["night_one_actions"] == ["狼人刀人", "预言家验人", "女巫用药", "混血儿选主人"]


def test_detect_timeline_confusion_flags_first_night_after_first_day_claims() -> None:
    flagged = detect_timeline_confusion("第一天警上之后，晚上才进入首夜验人。")
    clean = detect_timeline_confusion("首夜验人结束后，第一天警上发言要报查验。")

    assert flagged
    assert flagged[0]["type"] == "first_night_after_first_day"
    assert clean == []
