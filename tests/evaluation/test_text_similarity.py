from werewolf_agent.evaluation.text_similarity import tokenize, jaccard


def test_tokenize_splits_ascii_and_cjk():
    assert tokenize("p03 vote 警徽流") == {"p03", "vote", "警", "徽", "流"}


def test_jaccard_identical_is_one():
    assert jaccard("警徽流 对跳", "警徽流 对跳") == 1.0


def test_jaccard_disjoint_is_zero():
    assert jaccard("abc", "xyz") == 0.0


def test_jaccard_partial():
    # {p03, vote} vs {p03, vote, extra} → 2/3
    assert round(jaccard("p03 vote", "p03 vote extra"), 4) == round(2 / 3, 4)
