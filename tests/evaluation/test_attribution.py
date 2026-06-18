from werewolf_agent.evaluation.attribution import AttributionTextResolver
from werewolf_agent.evaluation.feedback_schemas import ModuleExposure


def test_resolver_returns_rag_prompt_safe_text():
    resolver = AttributionTextResolver(
        rag_entries={"r1": {
            "title": "对跳局核验警徽流",
            "situation_signature": "seer counterclaim",
            "transferable_lesson": "先比较验人时间线",
            "recommended_action": "列对比表再站边",
            "misuse_risk": "不要套用历史身份",
        }},
    )
    exposure = ModuleExposure(module="rag", item_id="r1")
    text = resolver.rag_text(exposure)
    assert "对跳局核验警徽流" in text
    assert "列对比表再站边" in text


def test_resolver_returns_reflection_prompt_card_text():
    resolver = AttributionTextResolver(
        reflection_entries={"ref1": {
            "theme": "投票前核验",
            "lesson": "过早站边会误判",
            "recommended_action": "核验警徽流",
            "misuse_risk": "不映射本局玩家",
        }},
    )
    exposure = ModuleExposure(module="reflection", item_id="ref1")
    text = resolver.reflection_text(exposure)
    assert "投票前核验" in text
    assert "核验警徽流" in text


def test_resolver_returns_none_when_item_missing():
    resolver = AttributionTextResolver()
    exposure = ModuleExposure(module="rag", item_id="unknown")
    assert resolver.rag_text(exposure) is None


from werewolf_agent.evaluation.attribution import (
    speech_from_decision,
    exposure_representative_text,
    cited,
)
from werewolf_agent.evaluation.feedback_schemas import DecisionSnapshot


def test_speech_from_decision_reads_raw_speech():
    d = DecisionSnapshot(action_type="speech", raw={"speech": "我怀疑p03", "reason": "x"})
    assert speech_from_decision(d) == "我怀疑p03"


def test_speech_from_decision_falls_back_to_public_story():
    d = DecisionSnapshot(action_type="speech", raw={"public_story": "归票p07"})
    assert speech_from_decision(d) == "归票p07"


def test_exposure_representative_text_rag_uses_resolver():
    resolver = AttributionTextResolver(rag_entries={"r1": {
        "title": "对跳局核验警徽流", "recommended_action": "列对比表",
    }})
    text = exposure_representative_text(
        ModuleExposure(module="rag", item_id="r1"), resolver,
    )
    assert "对跳局核验警徽流" in text


def test_exposure_representative_text_possible_worlds_uses_assignments():
    exposure = ModuleExposure(
        module="possible_worlds", item_id="World A",
        metadata={"key_assignments": {"p03": "werewolf", "p07": "seer"}},
    )
    text = exposure_representative_text(exposure, AttributionTextResolver())
    assert "p03=werewolf" in text
    assert "p07=seer" in text


def test_exposure_representative_text_simulator_uses_item_id():
    exposure = ModuleExposure(
        module="simulator", item_id="p03 exile pressure",
        metadata={"affected_players": ["p03"]},
    )
    text = exposure_representative_text(exposure, AttributionTextResolver())
    assert "p03 exile pressure" in text
    assert "p03" in text


def test_cited_true_when_jaccard_above_threshold():
    decision = DecisionSnapshot(
        action_type="vote", reason="p03发言可疑，需核验警徽流", raw={},
    )
    resolver = AttributionTextResolver(rag_entries={"r1": {
        "title": "核验警徽流", "recommended_action": "核验",
    }})
    exposure = ModuleExposure(module="rag", item_id="r1")
    assert cited(decision, exposure, resolver) is True


def test_cited_false_when_no_overlap():
    decision = DecisionSnapshot(action_type="vote", reason="abcdefg", raw={})
    resolver = AttributionTextResolver(rag_entries={"r1": {"title": "完全不同的内容"}})
    exposure = ModuleExposure(module="rag", item_id="r1")
    assert cited(decision, exposure, resolver) is False
