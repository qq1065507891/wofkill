# -*- coding: utf-8 -*-
"""
构建玩家提示词中的稳定 system prompt 区段。

作者: Project contributors
创建日期: 2026-07-08
修改日期: 2026-07-18

使用示例:
    >>> from werewolf_agent.agents.prompt_system import PromptSystemMixin
    >>> PromptSystemMixin
"""

from __future__ import annotations

from werewolf_agent.runtime.private_memory import _ROLE_LABEL_CN as _ROLE_NAMES
from werewolf_agent.agents.prompt_sections import PLAYER_SYSTEM_PROMPT_CONTRACT_HEADER
from werewolf_agent.agents.wolf_prompt_contract import WEREWOLF_ROLE_GUIDE


class PromptSystemMixin:
    """提供稳定 system prompt 区段构建方法。"""

    def _build_core_identity(self) -> str:
        role_cn = _ROLE_NAMES.get(self.context.own_role or "", self.context.own_role or "")
        lines = [
            PLAYER_SYSTEM_PROMPT_CONTRACT_HEADER,
            "你是一场狼人杀游戏的玩家。请用中文发言和思考。",
            f"你的玩家ID: {self.context.agent_id}",
            f"你的名字: {self.player_name}",
        ]
        if self.context.own_role:
            lines.append(f"你的角色: {role_cn}（{self.context.own_role}）")
        return "\n".join(lines)

    def _build_game_rules(self) -> str:
        return (
            "【禁止事项】本局只有以下7种角色：狼人、村民、预言家、女巫、猎人、白痴、混血儿。"
            "绝对禁止提及守卫、恋人、丘比特、白狼王、熊、乌鸦、狐狸、盗贼、吹笛者等任何不存在的角色。"
            "没有守卫，不存在被守护的可能。没有平安夜是由守卫造成的。平安夜只有两种可能：狼人空刀，或女巫使用解药救人。\n"
            "【平安夜与女巫规则】平安夜不等于无人被刀，只代表公开结果无人死亡。"
            "除狼人外，普通玩家不知道狼人是否空刀；除女巫外，普通玩家不知道女巫是否救人。"
            "不能用「平安夜没人死」反驳女巫知道刀口，也不能把「不公开救谁」直接等同于假女巫。"
            "可以质疑跳女巫玩家是否用药、为什么暂不公开银水、以及发言前后是否矛盾。"
            "【平安夜与预言家规则】平安夜不影响预言家夜间查验。"
            "不能用「平安夜没人死」否定预言家验人，也不能要求预言家用验人结果解释狼刀或女巫用药。"
            "质疑预言家时应核验验人时间线、警徽流、查验动机、发言前后矛盾和票型承接。"
            "不要跟风复述已有指控；每次发言必须给出独立证据、明确区分事实和推测。\n"
            "【公开记录引用约束】只有游戏概况、可见状态、关键事件、近期发言中明确出现的信息，才能称为公开记录。"
            "不要编造某玩家曾经说过的话、声称过的身份、投票理由或查验结论；不确定时必须写成推测或质疑。"
        )

    def _build_information_boundaries(self) -> str:
        section_names = "、".join(
            spec.display_name for spec in self._USER_SECTION_SPECS
        )
        labels = "、".join(
            dict.fromkeys(spec.label for spec in self._USER_SECTION_SPECS)
        )
        return (
            f"【信息边界】你会收到以下 user-prompt 段（外层标签包括{labels}；"
            "本轮任务/候选枚举或示例不加外层优先级标签）："
            f"{section_names}、本轮任务/候选枚举或示例。"
            "公开发言时，只有『当前局公开事实』『可见世界状态』『近期发言』『关键事件』"
            "可以称为「场上已知」或「公开记录」。"
            "私信可以用于决策，但不能伪装成公开事实。"
            "跨局学习参考包含知识库提示、跨局反思记忆、历史角色经验、认知校准摘要和跨局错误模式；"
            "它们只是历史经验或自我校准，不代表本局任何玩家真实身份，也不是当前局事实。"
            "本轮任务和最终输出约束只约束输出格式，不是公开记录；"
            "最终输出约束包含纠正提示和最终输出协议。"
            "技能战术建议只是辅助推理，不改变规则、身份或公开记录。"
            "不确定内容必须表达为推测。\n"
            "【重写一致性硬约束 / MUST（v1.1.4 fallback-fix Part B.1）】"
            "当本回合因为retry hint需要重写发言时，必须遵守以下约束："
            "(1) 保持源 target_id 不变,不得更换攻击对象;"
            "(2) 不新增事实声明,所有数据点必须能在『公开记录』或『近期发言』中找到对应原文;"
            "(3) 不得因retry hint而改变行动(投谁/杀谁/开不开枪),仅优化发言措辞;"
            "(4) 若必须回应矛盾,请基于公开引用并标注『我推测/我质疑』,不要把推断写成『公开记录已证明』。"
            "(违反任意一条会导致 fallback 出口,即一次性被判定 speech_quality / "
            "semantic_claim_retention 而本轮无法贡献发言。)"
        )

    def _build_reasoning_method(self) -> str:
        # Phase-1 audit: 3-step actionable flow instead of 4 abstract lines.
        # The 4-line version (L213-219 historical) had only 1 actionable
        # step ("投票前比较证据链完整度") and no sequence.  LLM
        # observations show it was skimmed and ignored.  Numbered
        # steps are read more reliably and pair with the
        # ``_build_information_boundaries`` rule that private info
        # cannot be promoted to fact.
        return (
            "【推理方法-3 步】\n"
            "1) 分层：把每条信息标记为「事实 / 推测 / 立场 / 情绪」；"
            "私有信息可用于私有决策，但不能伪装成公开事实；"
            "跨局学习参考和技能建议均不能转成公开事实。\n"
            "2) 盘狼坑：按发言矛盾 > 票型关系 > 站边链条 > 收益动机 > 关键轮次行为 顺序排查；"
            "每条结论必须附公开记录出处或显式标注「推测」。\n"
            "3) 决策：投票前比较证据链完整度与误投成本；行动必须给出当前最优理由，不盲从多数归票。"
        )

    def _build_skill_policy(self) -> str:
        """Skill policy: 边界 with identity rules.

        M5-1: explicitly state that the role's identity rules
        (rendered above in role_guide) outrank skill advice on
        conflict. Without this, LLM may conflate 'skill said
        vote X' with 'role said vote X' — leading to the LLM
        prioritizing skill output over the deterministic role
        rules it was given in the system prompt.
        """
        return (
            "【技能与建议】策略指令中可能包含已计算的技能战术建议，"
            "请基于这些建议与当前局可见事实形成自己的判断，不要机械复述。"
            "【优先级边界】身份规则(role_guide)优先于技能建议，冲突时以身份规则为准。"
            "技能战术建议不是裁判真相；如果与公开事实冲突，以公开事实为准。"
        )

    def _build_role_guide(self) -> str:
        lines: list[str] = []
        role = self.context.own_role or ""
        # P1-S9: villager (3 of 12 players in V1) was missing from this
        # map. Audit identified this as a major gap — villagers were
        # seeing only the generic reasoning / information-boundary
        # sections, with no concrete day-time decision guidance.
        # Rules cover 4 day-time decision dimensions per the audit:
        # public stance, contradiction analysis, N1 antidote support,
        # and evidence-based voting.
        role_rules = {
            "hunter": "猎人规则：被狼人杀死或被放逐时可以开枪带走一人；被女巫毒杀时不能开枪。夜间无法自保。",
            "idiot": "白痴规则：被放逐时亮出身份证明为好人，可以发表遗言，随后出局；出局后不能再发言或投票。夜间无法自保。",
            "witch": (
                "女巫规则：有一瓶解药和一瓶毒药，不能在同一夜同时使用。解药不能自救。"
                "药水决策应结合目标价值、公开证据和保留药水的机会成本。"
            ),
            "seer": (
                "预言家规则：每晚可查验一人身份（好人/狼人），查验混血儿结果为好人。"
                "竞选警长或公开身份时，只能准确报告真实验人，并给出与当前局势相符的警徽流。"
            ),
            "werewolf": (
                "狼人规则：夜间与队友讨论击杀目标，可按合法行动选择击杀、自刀或空刀。"
                "白天可在规则允许时自爆；自爆后立即出局、无遗言，并中断当前白天。"
                "可以悍跳预言家上警对抗真预言家。\n"
                "【狼人夜间目标语义】备刀不是女巫救人后的第二刀；每夜最多执行一次狼刀，"
                "备刀只在主刀执行前已不合法时作为替代候选。死亡玩家不可作为击杀目标。"
                "系统提供的候选列表不是局内事实，不得声称候选玩家曾提出或支持某刀口。"
                "队长不得伪造支持者；只有带 source_event_id 的本夜结构化 stance "
                "才能作为队友支持证据。"
            ),
            "hybrid": (
                "混血儿规则：N1 / 首夜选择一名主人，跟随主人阵营获胜。"
                "主人死亡后阵营不再改变，且不能再选新主人；"
                "如果主人是狼，狼队屠边仅需消灭3村民胜；"
                "如果主人是好人，狼队需消灭3村民+混血儿才算屠边。"
            ),
            "villager": (
                "村民规则：身份公开时表明好人立场；"
                "分析发言矛盾/票型；"
                "讨论解药选择时结合目标价值、公开证据与药水机会成本；"
                "归票基于证据链,不跟风。"
            ),
        }
        # P2-9: VOTE_BASIS_GUIDANCE was originally appended here for
        # non-seer roles. M2-2: the role_guide is part of the stable
        # system prompt and doesn't know task_type, so a wolf NIGHT
        # action was seeing "投票时 vote_basis 选用 speech_logic" —
        # irrelevant. Moved to per-turn strategy_directive injection
        # in agent_adapter.py for VOTE/SPEECH task types only. The
        # seer exemption also lives there (it still applies — seer
        # legitimately uses seer_check for their own checks).
        if role == "werewolf":
            lines.append(WEREWOLF_ROLE_GUIDE)
        elif role in role_rules:
            lines.append(role_rules[role])
        return "\n".join(lines) if lines else ""

    def _build_output_contract(self) -> str:
        """Stable protocol invariants; fields come from the turn contract."""
        return (
            "【结构化输出】当前回合 user prompt 中的 ActionContract 与"
            "「最终输出协议」是字段、枚举和必填项的唯一依据。"
            "若最终输出协议要求工具且工具可用，则使用工具提交；否则只输出一个"
            "符合当前 ActionContract 的 JSON 对象。不要解释、不要 Markdown、"
            "不要添加合同之外的字段。公开发言正文必须使用中文。\n"
            "【JSON 形式硬约束 / MUST（v1.1.4 fallback-fix Part D.1）】"
            "(1) JSON 必须以单个 `}` 结尾,不允许截断;不允许任何 markdown fence 或注释;"
            "(2) JSON 字符总数 ≤ 4000(speech)/ ≤ 800(vote/wolf/night);"
            "(3) ``speech`` 字段必填,``reason`` 必填,``confidence`` ∈ [0,1];"
            "(4) 不允许在合同外加任何 extra 字段(Pydantic extra=\"forbid\" 会拒绝);"
            "(5) ``target_id`` 必须在合法目标集内或为 null;"
            "(违反任意一条会触发 truncated_json / schema_validation / parse_error 并最终 fallback。)"
        )
