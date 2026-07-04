"""Skill schemas: structured types for werewolf agent skills.

Design doc §11.1: skills are abstracted gameplay capabilities.
Each skill takes role, phase, situation, persona, and experience as input,
and returns recommended action, speech structure, risk alerts, and confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Skill enums
# ---------------------------------------------------------------------------

class SkillName(str, Enum):
    BOLD_CLAIM = "bold_claim"            # 悍跳
    COUNTER_CLAIM = "counter_claim"      # 对跳
    PUSH_VOTE = "push_vote"              # 归票
    SWING_VOTE = "swing_vote"            # 冲票
    DEEP_HOOK = "deep_hook"              # 倒钩
    FIND_POWER = "find_power"            # 找神
    HIDE_IDENTITY = "hide_identity"      # 藏身份
    RESIST_PUSH = "resist_push"          # 抗推
    WOLF_PIT_ANALYSIS = "wolf_pit"       # 盘狼坑
    PROTECT_POWER = "protect_power"      # 保护强神
    LAST_WORDS_ANALYSIS = "last_words"   # 遗言分析
    REVIEW_CORRECTION = "review_correct" # 复盘纠错


class SkillFaction(str, Enum):
    WOLF = "wolf"
    GOOD = "good"
    NEUTRAL = "neutral"
    COMMON = "common"
    UNIVERSAL = "universal"


class SkillPhase(str, Enum):
    SPEECH = "speech"
    VOTE = "vote"
    NIGHT = "night_action"
    SHERIFF_SPEECH = "sheriff_speech"
    DEFENSE = "defense_speech"
    PK_SPEECH = "pk_speech"
    REVIEW = "review"
    WOLF_DISCUSSION = "wolf_discussion"
    SHERIFF_REGISTRATION = "sheriff_registration"
    HUNTER_SHOT = "hunter_shot"
    LAST_WORDS = "last_words"


# ---------------------------------------------------------------------------
# Skill input / output
# ---------------------------------------------------------------------------

@dataclass
class SkillInput:
    role: str
    phase: str
    situation: str = ""
    persona_params: dict[str, Any] = field(default_factory=dict)
    available_experience: list[dict[str, Any]] = field(default_factory=list)
    legal_actions: list[str] = field(default_factory=list)
    legal_targets: list[str] = field(default_factory=list)
    day: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
    game_state: Any | None = None
    world_state: Any | None = None
    belief_state: Any | None = None
    contradiction_alerts: list[Any] = field(default_factory=list)
    player_id: str = ""
    # P1-K5: precise task type (e.g. "speech", "vote", "night_action").
    # Handlers branch on this where it materially changes advice
    # (push_vote for vote vs speech, bold_claim for speech vs vote).
    task_type: str = ""


@dataclass
class SkillAdviceFrame:
    skill: str
    situation_signature: str
    recommended_use: str
    risk_alerts: list[str] = field(default_factory=list)
    counter_signals: list[str] = field(default_factory=list)
    forbidden_use: str = ""
    confidence: float = 0.5
    relevance: float = 0.5
    evidence_refs: list[str] = field(default_factory=list)


@dataclass
class SkillOutput:
    skill_name: str
    speech_structure: list[str] = field(default_factory=list)
    risk_alerts: list[str] = field(default_factory=list)
    confidence: float = 0.5
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt_injectable: str = ""
    advice_frame: SkillAdviceFrame | None = None


# ---------------------------------------------------------------------------
# Skill definition
# ---------------------------------------------------------------------------

@dataclass
class SkillDefinition:
    name: SkillName
    display_name: str
    description: str
    applicable_roles: list[str] = field(default_factory=list)
    applicable_phases: list[str] = field(default_factory=list)
    # P0-K2: precise task-type filter (e.g. 'speech', 'night_action',
    # 'sheriff_speech'). When non-empty, the skill is only applicable
    # if `task_type` is in this list. Use this instead of cramming
    # task-type values into `applicable_phases`.
    applies_to_task_types: list[str] = field(default_factory=list)
    faction: SkillFaction = SkillFaction.COMMON
    tags: list[str] = field(default_factory=list)
    # P-SK1: SKILL.md 正文（frontmatter 之后的 markdown 散文）。由
    # `werewolf_skills._load_manifests` 加载，registry 在
    # `dispatch_for_role` 处将其作为 "## 技能说明" 段拼接到
    # `prompt_injectable` 末尾。markdown-driven skills 的实际"驱动"
    # 由此字段承载 — Python handler 给出战术建议，body 给出散文背景。
    body: str = ""

    def is_applicable(
        self,
        role: str,
        phase: str = "",
        task_type: str = "",
    ) -> bool:
        """Check if this skill applies given role, phase, and task_type.

        SKILL.md frontmatter `applicable_phases` typically lists task-type
        values ('speech', 'night_action', 'sheriff_speech', etc.).
        Call sites can also pass `phase` ('day'/'night') as a coarse
        match, and `task_type` for precise phase-like matching.

        Match rule: the skill is applicable iff ALL of:
        - `role` is in `applicable_roles` (or `applicable_roles` is empty)
        - `phase` is in `applicable_phases` OR `task_type` is in
          `applicable_phases` (or `applicable_phases` is empty)
        - `task_type` is in `applies_to_task_types` (or
          `applies_to_task_types` is empty)

        If `applicable_phases` is empty, the skill applies always (subject
        to role filter). If `applies_to_task_types` is empty, the skill
        applies regardless of task_type.
        """
        if self.applicable_roles and role not in self.applicable_roles:
            return False
        if self.applicable_phases:
            phase_match = phase in self.applicable_phases
            task_match = task_type in self.applicable_phases
            if not (phase_match or task_match):
                return False
        # P0-K2: precise task-type filter
        if self.applies_to_task_types and task_type not in self.applies_to_task_types:
            return False
        return True
