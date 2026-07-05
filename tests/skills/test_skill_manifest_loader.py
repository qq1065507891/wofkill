# -*- coding: utf-8 -*-
"""
验证技能 SKILL.md manifest 加载器。

作者: Mike
创建日期: 2026-07-05
修改日期: 2026-07-05

使用示例:
    >>> from werewolf_agent.skills.manifest_loader import load_manifests
    >>> load_manifests(root)
"""

from werewolf_agent.skills.manifest_loader import (
    load_manifests,
    parse_skill_frontmatter,
)
from werewolf_agent.skills.schemas import SkillFaction, SkillName


def test_parse_skill_frontmatter_returns_meta_and_body():
    meta, body = parse_skill_frontmatter(
        "---\n"
        "name: bold_claim\n"
        "faction: wolf\n"
        "---\n"
        "技能正文"
    )

    assert meta == {"name": "bold_claim", "faction": "wolf"}
    assert body == "技能正文"


def test_load_manifests_reads_skill_md_frontmatter(tmp_path):
    skill_dir = tmp_path / "bold_claim"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: bold_claim\n"
        "display_name: 悍跳\n"
        "description: test\n"
        "applicable_roles:\n"
        "  - werewolf\n"
        "applicable_phases:\n"
        "  - speech\n"
        "applies_to_task_types:\n"
        "  - speech\n"
        "faction: wolf\n"
        "tags:\n"
        "  - test\n"
        "---\n"
        "正文",
        encoding="utf-8",
    )

    loaded = load_manifests(root=tmp_path)

    assert len(loaded) == 1
    skill = loaded[0]
    assert skill.name == SkillName.BOLD_CLAIM
    assert skill.faction == SkillFaction.WOLF
    assert skill.applies_to_task_types == ["speech"]
    assert skill.body == "正文"
