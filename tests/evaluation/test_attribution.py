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
