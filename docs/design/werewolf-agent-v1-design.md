# 狼人杀智能体框架 V1 设计文档

## 1. 项目定位

本项目是一个用于学习高级智能体开发的狼人杀多智能体框架。它不是只做一个狼人杀小游戏，而是把狼人杀设计成一个 Agent 工程训练场，用来系统学习：

- LangGraph 状态图编排
- 多智能体协作与对抗
- 静态规则上下文与策略/案例 RAG
- 长短期记忆系统
- 大模型路由网关
- 人格运行时路由
- 认知协处理器与结构化局势压缩
- 上下文工程
- Skill 化能力拆分
- MCP/Tool 工具层
- Human-in-the-loop 观战与干预
- Agent 评测与自我进化

V1 固定采用 **12 名玩家 + 1 名法官**。法官不计入玩家数量。首个规则包采用 **预女猎白混**，后续其他板子只作为规则包扩展，不改变系统的玩家框架。

## 2. V1 核心目标

V1 的目标是落地一个“实验室 + 准生产”的智能体框架。

实验室侧重学习和可观测性：

- 能看到每个玩家的身份视角、公开视角、认知矩阵、记忆、检索结果、发言依据和复盘结果。
- 能对比不同人格路由、每玩家模型配置、策略 RAG 和记忆策略对胜率与行为的影响。
- 能复盘 Agent 为什么做出某个发言、投票或技能选择。

准生产侧重工程边界：

- 游戏规则、Agent 决策、RAG、记忆、评测、API、观战台彼此解耦。
- Agent 不直接拼装人格参数，所有人设、行为参数和表达策略统一经过 Persona Router。
- Agent 不直接调用模型，所有模型请求统一经过 LLM Router Gateway。
- Agent 不直接拼接完整历史发言，所有输入先经过 Cognitive Coprocessor 做信息隔离、显著性筛选、信念更新和局部上下文构建。
- 规则不通过 RAG 查询，基础规则作为确定性配置与 System Prompt 静态上下文常驻。
- 法官流程可暂停、恢复、回放。
- 数据可持久化，实验可重复运行。
- 所有 LLM 输出都经过结构化校验，不能直接污染规则结算。

## 3. 规则设计：12 人预女猎白混

### 3.1 玩家与角色配置

V1 首个规则包为 `pre_witch_hunter_idiot_mixed`，按 12 人预女猎白混标准规则落地。

规则包落地路径：

```text
config/rulesets/pre_witch_hunter_idiot_mixed.yaml
```

角色配置：

| 角色 | 数量 | 阵营 | 屠边归类 |
| --- | ---: | --- | --- |
| 平民 | 3 | 好人阵营 | 民边 |
| 预言家 | 1 | 好人阵营神职 | 神边 |
| 女巫 | 1 | 好人阵营神职 | 神边 |
| 猎人 | 1 | 好人阵营神职 | 神边 |
| 白痴 | 1 | 好人阵营神职 | 神边 |
| 狼人 | 4 | 狼人阵营 | 狼人 |
| 混血儿 | 1 | 特殊胜负角色，胜利条件跟随主人阵营 | 民边 |

总玩家数固定为 12。混血儿俗称“混混”，不进入狼人夜聊，不知道狼人身份，狼人也不知道混血儿身份。工程实现中必须把混血儿拆成四个独立维度建模：`role=hybrid`、`seer_result=good`、`slaughter_requirement=conditional_by_master_faction`、`win_binding=master_faction`。混血儿身份表记为民边，但屠民结算必须按主人阵营条件化处理：主人属于好人阵营时，狼人屠民需要让 3 名平民和混血儿全部出局；主人属于狼人阵营时，狼人屠民只需要让 3 名平民全部出局。混血儿的胜利结果由主人阵营决定，不单独形成第三方阵营。

### 3.2 角色技能

平民：

- 没有夜间技能，全程夜间闭眼。
- 通过白天发言、投票、站边和逻辑分析找出狼人。

狼人：

- 夜间共同睁眼，在限定时间内讨论并决定当晚行动：击杀一名玩家，或主动选择空刀。
- 狼人可以自刀。
- 狼人可以主动空刀。主动空刀是合法战术动作，可用于制造平安夜信息差并在白天抗推好人；系统必须记录为 `wolf_no_kill_declared`，不得把主动空刀视为异常或自动补刀。
- 狼人夜间讨论有时间限制。若计时结束时狼队没有形成合法最终动作，法官默认本夜空刀，并记录为 `wolf_no_kill_timeout`。超时空刀与主动空刀在结算上都不会产生狼刀死亡，但事件原因必须区分，便于复盘和评测。
- 狼人白天可以自爆；自爆后立即出局且无遗言，结束当前白天，跳过剩余发言和放逐投票，先检查胜负，若未结束则直接进入夜晚。
- 若自爆狼人持有警徽，视为警长出局，按警徽规则选择移交警徽或撕掉警徽。
- 白天通过伪装成好人、悍跳、倒钩、冲票、抗推等方式帮助狼人阵营获胜。

预言家：

- 每晚查验一名玩家的阵营属性，只返回“好人”或“狼人”阵营信息，不返回具体身份。
- 查验混血儿时返回好人信息。
- 标准打法要求预言家在警上起跳，报查验和警徽流；系统不强制玩家必须这样发言，但评测可记录其是否符合标准打法。

女巫：

- 拥有解药和毒药各一瓶，每瓶全局只能使用一次。
- 解药可以救活当晚被狼人击杀的玩家。
- 毒药可以在夜间毒死一名玩家。
- 同一晚不能同时使用解药和毒药。
- 女巫全程不能自救，包括首夜。

猎人：

- 猎人被狼人击杀或被白天放逐出局后，可以发动技能带走场上任意一名玩家。
- 猎人被女巫毒死时不能发动技能。
- 猎人开枪造成的死亡记为 `hunter_shot`，并进入死亡结算链。

白痴：

- 白痴夜间闭眼。
- 白痴被白天放逐后可以翻牌自证身份。
- 翻牌后的白痴证明为好人，可以发表遗言，随后出局。
- 翻牌后的白痴不能再参与常规发言或投票，不能成为警徽移交对象。
- 翻牌后的白痴状态应显式记录为 `dead + revealed_idiot + vote_disabled + badge_ineligible`；不再参与存活玩家发言顺序或投票，立即计入神边出局。
- 白痴被狼人击杀、女巫毒死或猎人带走时不触发翻牌。

混血儿：

- 混血儿开局首夜睁眼选择一名主人。
- 主人可以是好人，也可以是狼人。
- 主人不知道自己被混血儿选择。
- 混血儿不知道主人的身份和阵营。
- 混血儿的胜利条件是帮助主人所在阵营取得胜利。
- 主人死亡不改变混血儿的胜利归属，混血儿仍跟随主人原始阵营。
- 混血儿自己死亡也不改变个人胜负归属；最终仍按主人原始阵营是否获胜结算 `hybrid_result`。

### 3.3 夜间行动顺序

标准夜间流程：

1. 狼人睁眼，限时讨论并提交当晚行动：击杀一名玩家或空刀；若超时未提交合法行动，默认空刀。
2. 女巫睁眼，得知当晚刀口，选择是否使用解药或毒药。
3. 预言家睁眼，查验一名玩家阵营属性。
4. 首夜猎人、白痴睁眼，让法官确认身份和技能状态。
5. 首夜混血儿睁眼，选择一名主人。

工程实现中仍应把首夜身份确认和混血儿选主人建成独立节点，方便回放与审计；但裁判播报应遵循标准规则，不向其他玩家泄露任何夜间身份确认信息。

狼人夜间行动事件必须显式区分：

- `wolf_kill_selected`：狼队在时限内选择合法刀口。
- `wolf_no_kill_declared`：狼队在时限内主动选择空刀。
- `wolf_no_kill_timeout`：狼队讨论超时或未形成合法最终动作，系统默认空刀。

只有 `wolf_kill_selected` 会为后续女巫节点提供当晚刀口；主动空刀和超时空刀均不会产生狼刀死亡，也不会给女巫提供可救刀口。

### 3.4 白天流程

V1 区分首日（D1）与后续日（D2+）流程。

#### D1 — 警长尚未产生

1. 法官宣布"天亮了"。
2. 警长竞选：法官宣布竞选规则，存活玩家报名、退水、发言、投票，产生警长。
   - 狼人自爆中断：自爆后增加 `sheriff_interrupt_count`；若 `< 2` 次则回到警长竞选继续，若 `>= 2` 次则本局警徽永久流失（参见 §3.5）。
3. 法官公布昨夜死讯。
4. 死者按遗言规则发表遗言。
5. 警长决定发言顺序（无警长时随机选择发言起点，再按座次方向执行）。
6. 玩家按发言顺序发言。
7. 进入放逐投票（含 PK 发言与复投处理）。
8. 结算放逐、猎人开枪、白痴翻牌、警徽移交或撕徽等后续事件。
9. 检查胜负；若未结束，进入下一夜。

#### D2+ — 已有警长或警徽已流失

1. 法官宣布昨夜死讯。
2. 死者按遗言规则发表遗言。
3. 警长决定发言顺序（无警长时随机选择发言起点，再按座次方向执行）。
4. 玩家按发言顺序发言。
5. 进入放逐投票（含 PK 发言与复投处理）。
6. 结算放逐、猎人开枪、白痴翻牌、警徽移交或撕徽等后续事件。
7. 检查胜负；若未结束，进入下一夜。

发言与投票原则：

- 警长决定白天发言顺序。
- 无警长时随机选择发言起点，再按座次方向执行。
- V1 标准规则中，若警徽被撕掉导致本局无警长，后续所有白天发言均随机选择起点，再按座次方向执行。
- 每名玩家的白天发言有时间限制。到时仍未完成发言时，系统自动结束该玩家发言并进入下一位，记录 `speech_timeout`；超时结束是流程控制事件，不改变投票权、身份状态或胜负规则。
- 发言超时时不得伪造玩家发言内容；若需要在公开时间线展示，可显示为空发言或系统占位，但 Agent 复盘必须能区分真实发言和超时未发言。
- 白天投票出局者都有遗言。
- 夜晚倒牌只有第一晚有遗言。
- 狼人自爆后，该白天没有放逐投票阶段。
- 放逐投票首次平票时进入 PK 发言与再次投票；再次平票则无人出局，直接进入夜晚。

### 3.5 胜负规则

好人阵营胜利：

- 所有狼人出局。

狼人阵营胜利：

- 狼人完成屠边。
- 屠民：所有平民出局；如果混血儿主人属于好人阵营，混血儿也必须出局；如果混血儿主人属于狼人阵营，狼人只需让三名平民全部出局即可完成屠民。
- 屠神：预言家、女巫、猎人、白痴全部出局。白痴被放逐翻牌并发表遗言后，立即视为神边出局。

混血儿胜利：

- 混血儿不单独形成第三方胜利条件。
- 如果主人属于好人阵营，好人获胜时混血儿共同获胜。
- 如果主人属于狼人阵营，狼人获胜时混血儿共同获胜。
- 混血儿死亡、主人死亡或主人后来出局，都不改变混血儿最终跟随主人原始阵营结算的规则。
- 对局结果需要同时记录 `winning_faction`、`hybrid_master_id`、`hybrid_master_faction` 和 `hybrid_result`，避免只用一个 `winner` 字段丢失混血儿结算。

胜负检查时机：

- 每次夜间死亡结算后检查。
- 每次白天放逐、白痴翻牌、猎人开枪或狼人自爆后检查。
- 如果同一结算批次出现多名死亡，应先完成该批次全部死亡和技能触发，再统一检查胜负。

### 3.6 裁判原则

LLM 不能直接决定游戏规则结果。

确定性规则由 `RuleEngine` 执行：

- 身份分发
- 夜晚技能结算
- 查验结果
- 女巫救药/毒药限制
- 猎人开枪
- 白痴翻牌
- 白痴翻牌出局后的无投票权、不可接警徽和遗言状态
- 狼人自刀与自爆
- 放逐结算
- 平票 PK、再次投票和再次平票无人出局
- 遗言权
- 警长竞选、警徽权重、警徽移交与撕徽
- 死亡状态
- 胜负判定

法官 Agent 只负责：

- 流程播报
- 状态解释
- 异常提示
- 人类观战交互
- 调用确定性规则引擎

规则上下文原则：

- 预女猎白混完整规则、角色技能、结算顺序、胜负条件、警长规则、遗言规则和当前存活状态应由 `RuleEngine` 配置和 System Prompt 静态注入提供。
- RAG 不查询基础规则，不解释结算真相，不参与最终裁判。
- RAG 只用于历史对局案例、发言模板、高级战术和复盘经验。

### 3.7 警徽系统

警徽系统必须作为独立规则域建模，不能只塞进 `sheriff_election` 一个流程节点。

核心对象：

- `sheriff_id`：当前警长。
- `sheriff_badge_state`：`none`、`active`、`torn`。
- `sheriff_candidates`：上警玩家集合。
- `sheriff_speech_order`：警上发言顺序。
- `badge_flow_claims`：警徽流声明，例如“今晚验 X，如果死亡警徽给 Y”。
- `badge_transfer_policy`：警长死亡时传徽、撕徽和限制规则。

警徽流程：

1. 上警报名。
2. 警上发言。
3. 退水确认。
4. 警下投票。
5. 产生警长或进入平票处理。

警长权力：

- 白天投票权重为 `1.5` 票。
- 决定白天发言顺序。
- 可在归票阶段给出 `sheriff_endorse`。
- 警长出局后可以选择撕掉警徽，或将警徽移交给另一名玩家。
- 撕掉警徽后本局无警长，后续白天没有警长加票、归票和指定发言顺序；发言顺序改为随机起点。
- 白痴翻牌后不能成为警徽移交对象。

规则包必须显式配置：

- 警长因放逐、夜刀、女巫毒、猎枪带走或狼人自爆而出局时，均可选择传徽或撕徽。
- 撕徽后立刻失去警长加票。
- 警徽流既是公开声明，也必须进入结构化关系链作为后续评测对象。
- 若警长选择传徽，继任警长获得 `1.5` 票权重和发言顺序权；若警长选择撕徽，本局进入无警长状态。

### 3.8 死亡原因、时机与技能触发

`deaths` 不能只记录死亡玩家列表，必须记录死亡原因、死亡时机、结算批次和可触发技能。

建议结构：

```json
{
  "player_id": "player_08",
  "death_reason": "witch_poison",
  "death_timing": "night",
  "resolution_batch": "night_2",
  "source_player_id": "player_05",
  "can_leave_last_words": false,
  "triggered_skills": []
}
```

`death_reason` 至少包括：

- `wolf_kill`
- `witch_poison`
- `exile`
- `hunter_shot`
- `self_destruct`
- `rule_elimination`

`death_timing` 至少包括：

- `night`
- `day_vote`
- `day_discussion`
- `post_exile`

技能触发按 `(death_reason, death_timing)` 组合判断：

- 猎人被狼人击杀或白天放逐时可以开枪。
- 猎人被女巫毒死不能开枪。
- 同夜双死时，应先记录同一结算批次内的死亡，再按 V1 固定遗言规则和技能触发矩阵结算。
- 白痴只在被放逐时翻牌，被毒死、刀死或猎枪带走不触发翻牌。
- 夜晚倒牌只有第一晚有遗言，白天投票出局者都有遗言。

### 3.9 常见规则边界

首个规则包必须显式配置以下边界，避免引擎里出现隐式默认：

- 平安夜：夜间无人死亡时，白天仍进入公告、讨论和投票流程。
- 平安日：白天无人出局时，直接进入夜晚。
- 自爆：狼人白天可以自爆，自爆后立即出局且无遗言，先检查胜负；若未结束，直接进入夜晚，该白天没有放逐投票阶段。
- 平票：放逐投票首次平票时进入 PK 发言与再次投票；再次平票则无人出局，直接进入夜晚。
- 遗言权：白天投票出局者都有遗言，夜晚倒牌只有第一晚有遗言；自爆无遗言，猎枪带走者默认无遗言。
- 女巫自救：女巫全程不能自救。

## 4. 玩家人格设计

玩家人格采用“京城大师赛风格原型”，不直接复刻真人，不使用真实姓名作为 Agent 身份。

每个玩家由一组人格参数驱动：

| 字段 | 含义 |
| --- | --- |
| `personality` | 人格定位 |
| `speech_style` | 发言风格 |
| `risk_tolerance` | 风险偏好 |
| `deception_skill` | 欺骗能力 |
| `logic_skill` | 逻辑能力 |
| `leadership` | 带队能力 |
| `emotion_control` | 情绪稳定性 |
| `learning_rate` | 成长速度 |

建议 V1 预置 12 个匿名风格原型：

1. 强逻辑归票型
2. 压迫式发言型
3. 稳健站边型
4. 悍跳进攻型
5. 倒钩潜伏型
6. 情绪感染型
7. 娱乐搅局型
8. 复盘学习型
9. 低调观察型
10. 视角爆发型
11. 盘坑结构型
12. 临场博弈型

人格影响：

- 发言长度
- 发言攻击性
- 是否愿意带队
- 是否容易冲票
- 是否适合悍跳
- 是否倾向倒钩
- 是否容易被情绪影响
- 复盘后成长速度

### 4.1 Persona Router

Persona Router 是所有玩家人格配置的运行时入口。它不只返回静态人格参数，而是根据玩家、阶段、任务类型和局势压力，生成当前任务的人格运行时配置。

```text
PersonaRouter.resolve(agent_id, task_type, game_context)
```

Persona Router 职责：

- 按 `agent_id` 读取玩家自己的 `persona_profile`。
- 按 `task_type` 选择表达策略，例如 `sheriff_speech`、`defense_speech`、`wolf_deception`、`last_words`、`reflection`。
- 根据局势触发动态调整，例如被质疑、队友出局、拿到警徽、被抗推、夜间刀中关键神职。
- 输出稳定的行为参数快照，供 Agent prompt、Skill 和评测使用。
- 记录 persona 路由命中，便于复盘“为什么这个玩家突然变激进/保守”。

示例配置：

```yaml
players:
  player_02:
    persona_profile: aggressive_bluffer
    llm_profile: local_wolf

persona_profiles:
  aggressive_bluffer:
    base:
      logic_skill: 0.62
      deception_skill: 0.91
      aggression: 0.84
      risk_tolerance: 0.78
      speech_length: 0.72
    task_styles:
      sheriff_speech: pressure_claim
      defense_speech: counterattack
      wolf_deception: high_pressure_push
      last_words: emotional_counterpush
      reflection: blunt_self_review
    dynamic_policy:
      when_suspected:
        aggression_delta: 0.12
        speech_length_delta: 0.2
      when_teammate_exiled:
        risk_tolerance_delta: -0.15
      when_trusted_by_good_players:
        deception_skill_delta: 0.08
```

Persona Router 与 LLM Router 的分工：

- Persona Router 决定“这个玩家此刻像谁、偏好什么策略、用什么表达姿态”。
- LLM Router Gateway 决定“这次请求用哪个 provider 和 model 生成”。
- 二者都属于实验配置，不属于规则真相，不能修改可用动作和结算结果。

### 4.2 战术覆盖 (LLM Prompt Layer, 2026-06-08)

LLM 决策层在 action_type 已有但 prompt 缺引导时,会忽略合法战术 (4 局真实游戏审计发现)。本节列出 V1 显式注入 LLM prompt 的战术 directive,RuleEngine 仍只暴露合法 action,不强制 LLM 必须使用。

**狼队战术覆盖** (`agent_adapter.py:_single_wolf_vote` + `directives/wolf.py:build_wolf_directive`):

- `wolf_no_kill_conditions` — 显式列出 4 条空刀触发条件 (无高威胁目标 / 制造平安夜 / PK 前夜不出刀 / 自爆后续局),并提示"连续 2 夜空刀会强制出刀"上限。`_LEGACY_WOLF_CONSENSUS` 已有 `wolf_no_kill_declared` 事件类型但 LLM 不会主动选。
- `wolf_self_destruct_condition` — 狼处于被推/警徽流失危险位置时 (判定: 当前 day `vote_resolved` top tally == 本狼 OR 本狼持警徽) 才注入,4 条触发条件 (即将被放逐 / 持警徽被票 / 保护关键信息 / 屠边胜利在即)。`day.py:280` 已有自爆短路逻辑但 LLM 不会主动选。

**女巫毒药覆盖** (`agent_adapter.py:agent_night_witch` + `strategy/poison.py`):

- `witch_poison_candidates` — 按证据强度排序的 top 候选目标,来源: 公开查杀声明 (priority+10) + 多人(≥2)明确指控 (priority+6~8) + 死前投票异常。无证据时按存活人数给提示 (>9: 默认 no_action; ≤7: 紧急但证据不足)。`_collect_witch_poison_candidates` 纯 regex 实现,不调 LLM。

**核心约束**:

- 三个 directive 都是 **REFERENCE tier** (辅助信号),不是 HARD CONSTRAINT — LLM 仍可基于其他推理覆盖
- 注入位置: `strategy_directive` dict 的并列 key,prompt_builder 渲染到 strategy_directive 段内
- 单元测试覆盖: 10 个新测试断言 directive 在 AgentContext.strategy_directive 中存在
- 真实游戏效果需 5+ 局验证 (见 `PROGRESS.md` `balance-fix-tactic-coverage` phase)

## 5. 系统分层架构

### 5.1 总体分层

系统分为 13 层：

1. `config`：规则包、人格包、模型配置、实验配置。
2. `core`：领域模型、枚举、事件、动作、状态。
3. `engine`：确定性游戏规则引擎。
4. `runtime`：LangGraph 游戏流程编排。
5. `agents`：玩家 Agent、法官 Agent、复盘 Agent。
6. `persona_runtime`：人格路由、动态行为参数、表达策略、成长调整。
7. `model_gateway`：模型路由、每玩家 LLM 配置、fallback、限流、成本统计。
8. `cognition`：认知协处理器，负责结构化世界状态、注意力过滤、显著性排序、信念更新、矛盾发现、策略选择和局部上下文构建。
9. `context`：上下文构建、摘要、私有/公开信息隔离。
10. `rag`：策略、案例、历史对局和复盘经验检索，不承载基础规则。
11. `memory`：短期认知矩阵、结构化状态表、长期复盘经验、玩家画像。
12. `evaluation`：对局评分、玩家评分、实验对比。
13. `api/ui`：FastAPI 服务与观战台。

### 5.2 推荐目录结构

```text
werewolf_agent/
  api/
    app.py
    schemas.py
  agents/
    judge.py
    player.py
    reflection.py
  config/
    rulesets/
      pre_witch_hunter_idiot_mixed.yaml
    personas/
      jingcheng_style_prototypes.yaml
    models.yaml
    experiments.yaml
  persona_runtime/
    router.py
    profiles.py
    policies.py
    modifiers.py
  model_gateway/
    router.py
    providers.py
    policies.py
    usage.py
  core/
    enums.py
    models.py
    actions.py
    events.py
    context.py
    rule_prompt.py
  engine/
    rule_engine.py
    victory.py
    night_resolution.py
    voting.py
  runtime/
    graph.py
    checkpoints.py
    interrupts.py
  cognition/
    pipeline.py
    world_state.py
    attention_filter.py
    salience.py
    belief_updater.py
    contradiction_engine.py
    strategy_selector.py
    local_context.py
  rag/
    indexer.py
    retriever.py
    documents/
  memory/
    store.py
    cognition_matrix.py
    relation_graph.py
    reflection.py
    profile.py
  evaluation/
    metrics.py
    reports.py
  skills/
    registry.py
    werewolf_skills.py
  tools/
    mcp_registry.py
tests/
docs/
```

### 5.3 LLM Router Gateway

LLM Router Gateway 是所有 Agent 调用模型的唯一入口。玩家 Agent、法官 Agent、复盘 Agent 不直接依赖 OpenAI、Gemini、Ollama 等 provider SDK，而是统一调用：

```text
ModelRouter.generate(agent_id, task_type, context)
```

路由网关职责：

- 按 `agent_id` 读取玩家自己的 `llm_profile`。
- 按 `task_type` 选择模型，例如 `speech`、`vote`、`night_action`、`deception`、`reflection`。
- 统一注入 temperature、max_tokens、top_p、timeout、重试和 fallback。
- 统一记录 provider、model、prompt_tokens、completion_tokens、latency_ms、estimated_cost、fallback_reason。
- 支持按实验配置切换路由策略，方便比较“同人格不同模型”和“同模型不同人格”。

示例配置：

```yaml
players:
  player_01:
    persona_id: logic_leader
    llm_profile: pro_reasoner
  player_02:
    persona_id: aggressive_bluffer
    llm_profile: local_wolf

llm_profiles:
  pro_reasoner:
    default:
      provider: openai
      model_profile: strong_reasoner
    tasks:
      reflection:
        provider: openai
        model_profile: deep_reflection
  local_wolf:
    default:
      provider: ollama
      model_profile: local_fast
    tasks:
      deception:
        provider: openai
        model_profile: fast_deception
    fallback:
      provider: openai
      model_profile: low_cost_fallback
```

路由边界：

- `llm_profile` 属于实验配置，不属于游戏规则。
- 文档中的 `model_profile` 是逻辑档位，不是固定模型名；真实 provider/model 必须集中写在 `models.yaml`，并随实验快照持久化，避免设计文档里的示例模型过期后污染实现。
- 模型路由可以影响 Agent 推理质量和发言风格，但不能影响 `RuleEngine` 的合法动作集合和结算结果。
- 每次模型调用都必须可复现：记录配置快照、模型版本、随机种子或采样参数。

## 6. LangGraph 流程设计

### 6.1 对局主状态

`GameState` 是 LangGraph 的核心状态对象。

关键字段：

- `game_id`
- `ruleset_id`
- `persona_router_config_snapshot`
- `model_router_config_snapshot`
- `phase`
- `day_number`
- `night_number`
- `players`
- `public_events`
- `public_summary`
- `recent_public_transcript`
- `private_events`
- `night_actions`
- `votes`
- `deaths`
- `sheriff_id`
- `sheriff_badge_state`
- `sheriff_candidates`
- `badge_flow_claims`
- `speech_order`
- `hybrid_master_id`
- `hybrid_master_faction`
- `structured_world_state`
- `agent_cognition_matrices`
- `agent_salience_snapshots`
- `agent_contradiction_alerts`
- `persona_runtime_snapshots`
- `model_usage`
- `winning_faction`
- `hybrid_result`
- `paused`
- `human_commands`

状态更新原则：

- `GameState` 不应被 LangGraph 节点随意原地修改；所有规则结果先写入 `GameEvent`，再由确定性 reducer 归约成新的 `GameState`。
- `public_events`、`private_events`、`deaths`、`votes`、`night_actions`、`sheriff_badge_state` 和 `hybrid_result` 都必须能从事件日志重放出来。
- 人工干预也必须写成审计事件，再进入 reducer；不得绕过事件日志直接改胜负、死亡、警徽或身份状态。
- replay、pause/resume、评测复现实验均以 `initial_seed + ruleset_snapshot + event_log` 为准。

### 6.2 主流程节点

建议 LangGraph 节点：

1. `setup_game`
2. `assign_roles`
3. `wolf_discussion`
4. `wolf_team_plan`
5. `wolf_consensus`
6. `night_witch`
7. `night_seer`
8. `night_hunter_idiot_status`
9. `first_night_hybrid_master`
10. `resolve_night`
11. `announce_deaths`
12. `night_death_last_words`
13. `sheriff_registration`
14. `sheriff_speech`
15. `sheriff_withdraw`
16. `sheriff_vote`
17. `free_discussion`
18. `summarize_positions`
19. `sheriff_endorse`
20. `day_vote`
21. `tie_pk_speech`
22. `tie_revote`
23. `resolve_exile`
24. `post_exile_skills`
25. `check_victory`
26. `sheriff_badge_transfer`
27. `summarize_context`
28. `reflection`
29. `finish_game`

`sheriff_badge_transfer` 不是只属于放逐后的节点；任何结算批次中只要当前警长因夜刀、女巫毒、猎枪带走、白天放逐或狼人自爆出局，都必须先进入 `check_victory`。若 `check_victory` 判定对局尚未结束，再进入 `sheriff_badge_transfer` 让警长选择传徽或撕徽；若警徽已被撕掉或本局尚未产生警长，则跳过该节点。

夜晚沟通原则：

- 狼人夜晚不应只提交单点刀人动作，而应先进入 `wolf_discussion`。
- `wolf_discussion` 支持狼人内部多轮讨论，讨论内容只进入狼人私有视角，并受 `wolf_discussion_seconds` 限制。
- `wolf_team_plan` 节点紧接 `wolf_discussion`：由 alive werewolves 排序首位作为队长调用一次 LLM，产出 Pydantic `WolfTeamPlan`（含 4 角色分工 `fake_seer`/`pusher`/`hooker`/`deep_cover` + 击杀目标 `night_kill_primary`/`backup` + `public_story` + `reasoning`）。LLM 失败时回退到正则抽取 + 静态分配（`werewolf_agent/runtime/wolf_strategy.py` + `_build_wolf_team_plan`），并 emit `wolf_team_plan_fallback` 审计事件标记 `reason`。`reasoning` 字段含队长决策依据，仅 `werewolf_team_only` 可见，绝不进入公开视角。
- `wolf_consensus` 负责达成最终夜间行动，合法输出只有 `kill(target_id)` 或 `no_kill(reason)`。`kill` 必须给出合法存活目标；`no_kill` 表示狼队主动空刀。
- 如果 `wolf_discussion_seconds` 到期时仍未形成合法最终行动，V1 默认本夜空刀并记录 `wolf_no_kill_timeout`，不再随机兜底刀人。该策略保留狼人通过空刀制造信息差的玩法，同时避免运行时无限等待。
- 女巫节点必须在狼人夜间行动结算之后执行；只有当晚存在 `wolf_kill_selected` 时，女巫才能得知刀口并选择是否使用解药。若当晚是主动空刀或超时空刀，女巫没有可救刀口，但仍可按规则选择是否使用毒药；预言家查验在女巫行动之后执行。
- `first_night_hybrid_master` 只在首夜身份确认阶段执行，用于记录 `hybrid_master_id` 和不可泄露的主人绑定关系；该节点产物不得进入狼队夜聊或公开事件。
- 预言家、女巫、猎人、白痴等角色不能看到狼人讨论，只能看到符合自身视角的夜间结果。
- `night_death_last_words` 只在第一晚存在夜间死亡时触发；第二晚及之后的夜间死亡不触发遗言。
- `sheriff_registration` 到 `sheriff_vote` 只在首个白天执行；首日之后不再进入警长竞选流程。

白天流程原则：

- 发言顺序是局势的一部分，必须由 `speech_order` 记录。
- 有警长时，发言顺序由警长选择并写入 `speech_order`；若警长未给出有效选择，则由 `RuleEngine` 随机选择起点兜底。
- 无警长时，V1 标准规则随机选择发言起点，再按座次方向执行；若警徽被撕掉，本局后续白天均使用该规则。
- 后发言玩家拥有更完整的公开发言上下文；先发言玩家不能看到尚未发生的发言。
- 每个 `day_speech` 节点必须绑定发言人和发言时长上限。到时未产出有效公开发言时，系统记录 `speech_timeout` 并进入下一位发言人；若已有部分发言内容，可记录为该玩家公开发言，同时附带超时结束标记。
- 白天至少包含自由讨论、多轮发言摘要和最终投票；有警长且警徽未撕时才进入 `sheriff_endorse` 归票动作。
- 放逐投票首次平票后进入 `tie_pk_speech` 和 `tie_revote`；再次平票则无人出局，直接进入夜晚。
- 自爆、平票、无人出局、警长死亡等分支必须通过条件边进入对应节点，而不是在 Agent 自然语言里临时裁判。

白天并发原则：

- `free_discussion` 内的 `day_speech` 可以按发言顺序生成公开发言，但“听完发言后更新自身认知”的步骤必须通过 LangGraph `Send` API 并行触发所有存活玩家。
- `vote` 前只等待所有玩家的内部状态更新完成，再进入投票决策。
- 这样把顺序依赖限制在公开发言本身，避免 12 个 Agent 的状态更新串行放大延迟。

上下文剪裁原则：

- `public_events` 是审计日志，可以持久化完整事件，但不应完整塞进每次 Agent 调用。
- `summarize_context` 在每天结束后生成结构化摘要，包括站边变化、投票关系、关键矛盾、死亡与技能线索。
- Agent 上下文只保留当天逐字稿、最近关键事件和历史结构化摘要。

认知协处理器原则：

- 每次玩家 Agent 调用前，先执行 `build_agent_context(engine, game_state, agent_id, task_type)`。
- 管线产物包括 `structured_world_state_view`、`salience_items`、`belief_state`、`contradiction_alerts`、`strategy_directive` 和 `local_context`。
- LLM 不负责记账、算身份概率或从长文本里寻找矛盾，这些工作由确定性代码或轻量模型在认知管线中完成。
- `local_context` 才是进入 LLM 的主要输入，完整公开事件和私有事件只用于审计、回放和认知管线重算。

### 6.3 Human-in-the-loop

以下位置允许人工暂停：

- 身份分发后
- 每个夜间角色行动前
- 白天发言前
- 投票前
- 放逐结算前
- 胜负判定前
- 复盘生成前

人工干预能力：

- 查看公开状态
- 查看某个玩家私有状态
- 查看玩家上下文包
- 查看 RAG 命中
- 查看长期记忆
- 注入实验提示
- 修改非核心实验参数
- 恢复运行

核心规则状态修改必须被记录为审计事件。

## 7. Agent 设计

### 7.1 玩家 Agent 输入

`AgentContext` 包含：

- 当前阶段
- 公开局势摘要
- 认知协处理器生成的局部上下文
- 当前 Agent 可见的结构化世界状态
- 显著事件列表
- 逻辑矛盾警报
- 策略选择器给出的宏观策略指令
- 规则静态上下文
- 自己的私有身份信息
- 自己的短期记忆
- 自己的局势认知矩阵
- 结构化关系链
- 长期复盘经验检索结果
- 策略/案例 RAG 检索结果
- 当前可用动作
- 当前任务的 `persona_profile` 运行时摘要
- 当前任务的 `llm_profile` 摘要
- 输出格式要求

### 7.2 玩家 Agent 输出

所有输出必须结构化。

结构化输出必须优先使用 provider 支持的 `structured output`、JSON schema 或 function calling。合法动作、合法目标、阶段枚举和技能枚举必须编码到 schema 的 enum 中，不能只在提示词里用自然语言要求模型遵守规则。

建议把复杂输出拆成三步：

1. 生成 `private_intent`：只写入私有审计视图，用结构化字段表达真实身份、阵营目标、伪装视角和风险标记；不保存长篇自然语言私有推理文本。
2. 生成 `action` 与 `target`：通过 function calling 或严格 schema 输出，目标必须来自 `RuleEngine` 提供的合法动作集合。
3. 生成 `speech`：在动作已确定后生成自然语言公开发言。

弱模型或本地模型无法稳定一次性生成完整 JSON 时，必须启用三步生成，避免把自然语言发言、动作选择和私有策略字段混在同一次输出里。

典型输出：

```json
{
  "action_type": "vote",
  "target_id": "player_07",
  "private_intent": {
    "true_role": "werewolf",
    "faction_goal": "push_good_player_out",
    "claimed_view": "good_player_without_night_info",
    "pressure_target": "player_07",
    "risk_flags": ["avoid_night_kill_leak"]
  },
  "speech": "我今天归 7。",
  "reason": "7 的视角没有解释清楚，且投票行为像冲票狼。",
  "confidence": 0.72
}
```

`private_intent` 是对外不可见的私有策略快照，只进入调试追踪和观战台的私有审计视图，不写入公开发言时间线，不进入其他玩家上下文。

**专用决议性 Schema**：除上述通用 `PlayerAction` 外，少数高复杂度的"团队决议"输出有独立 Pydantic schema：

- `WolfTeamPlan`（`werewolf_agent/agents/schemas.py`）：狼队队长在 `wolf_team_plan` 节点一次性产出本夜完整战术计划，字段包括 4 个角色分工（`fake_seer` / `pusher` / `hooker` / `deep_cover`，必须 alive werewolf 且互不相同）、击杀目标主备（必须 alive 非狼或 null）、`public_story`（白天对外口径，1~120 字）、`evidence_quality` 枚举、`reasoning`（队长决策依据，1~200 字，**仅 werewolf_team_only 可见**）。schema validator 在解析时拒绝重复角色分配和"击杀目标 = 狼队成员"，game-state validator 在调用方再做 alive 集合校验。LLM 失败时调用方回退到正则抽取 + 静态分配（fallback path），保证流程不中断。

这种"专用决议 schema"模式适用于：(a) 输出字段强相关、(b) schema 校验能在结构层就拒绝大量非法输出、(c) 失败时存在 deterministic fallback 路径的场景。一般的玩家逐动作输出仍走 `PlayerAction` Union。

欺骗能力原则：

- 狼人、倒钩、悍跳、抗推等高欺骗任务必须先生成 `private_intent`，再生成公开 `speech`。
- `private_intent` 必须显式写出真实身份、当前阵营目标、准备伪装的视角和要压迫的对象，但只能使用短字段和枚举，不输出长链路推理文本。
- 公开发言不得泄露 `private_intent` 内容，只保留符合角色视角的自然语言表达。

非法输出处理：

1. JSON 解析失败：重试。
2. 动作不在可选范围：纠偏提示后重试。
3. 目标不存在或已死亡：纠偏提示后重试。
4. 多次失败：降级为规则兜底动作。

### 7.3 法官 Agent

法官 Agent 不直接裁判。

法官职责：

- 播报阶段
- 汇总发言
- 调用规则引擎
- 将规则结果转成自然语言事件
- 与观战台交互
- 处理人工暂停与恢复

## 8. 认知协处理器设计

Cognitive Coprocessor 是 Agent 调用 LLM 前的认知管线。它的目标是把 LLM 从“全能大脑”降级为“推理与生成引擎”：LLM 不直接吃完整历史，不负责维护关系图，不负责概率计算，也不负责判断自己该不该知道某条信息。

```text
build_agent_context(engine, game_state, agent_id, task_type) -> AgentContext
```

### 8.1 管线阶段

1. Structured World State：把自然语言发言、投票、身份声明、警徽流、死亡、技能结果等转成结构化事实。例如 `suspected(source, target, reason, day)`、`claimed_role(player, role, day)`、`badge_flow_claimed(seer, inspect_target, badge_target)`。
2. Visibility Policy：先按确定性权限表为每类事件计算可见范围，例如 `public`、`wolf_team_private`、`witch_private`、`seer_private`、`hybrid_private`、`moderator_only`，并记录可见性审计结果。
3. Attention Filter：按 Visibility Policy 和当前 Agent 的身份视角裁剪信息。平民不能看到夜间刀口和狼队讨论；女巫只看到与药、刀口、银水和求药相关的信息；狼人能看到队友和狼队夜聊；混血儿只能看到自己选择的主人编号，不能得知主人阵营和身份。
4. Salience Engine：为可见事实打权重。高权重事实直接进入 Prompt，中权重事实摘要进入，低权重事实只留存在结构化状态或长期记忆。
5. Belief Updater：用代码逻辑或轻量概率模型更新 `role_probabilities`、阵营倾向、信任度和开放问题。LLM 只接收计算结果，不直接盘概率。
6. Contradiction Engine：从关系图中发现矛盾和攻击抓手，例如“跳预言家的玩家没有投自己的查杀”或“警徽流与后续站边冲突”。
7. Strategy Selector：结合身份、阵营任务、Persona Router 产物和局势风险，选择策略包，例如 `aggressive_defense`、`deep_hook`、`protect_seer`、`push_counter_wagon`。
8. Context Builder（`build_agent_context`）：组装给 LLM 的极简上下文，包含身份视角、显著事实、信念摘要、矛盾警报、可用动作、策略指令和输出 schema。
9. LLM：基于压缩后的结构化前提生成行动建议、投票理由和自然语言发言。

### 8.2 输入输出边界

认知管线输入：

- `GameState`
- `RuleEngine` 当前合法动作集合
- Visibility Policy 权限表与事件可见性标记
- 当前 Agent 私有视角
- 公开事件与私有事件
- 结构化关系链
- 短期认知矩阵
- Persona Router 运行时结果
- RAG 检索结果

认知管线输出：

- `local_context`
- `visible_world_state`
- `salience_items`
- `belief_state`
- `contradiction_alerts`
- `strategy_directive`
- `prompt_budget_report`

边界原则：

- 认知管线可以筛选、排序和压缩信息，但不能改变规则结算结果。
- Attention Filter 的输出必须可审计，任何私有信息泄露都应能定位到具体过滤器规则。
- Visibility Policy 是硬边界，必须先于 Attention Filter、Salience Engine、RAG 注入和 Local Context Builder 执行；任何后续模块都不得扩大事件可见范围。
- Belief Updater 更新的是 Agent 视角，不是游戏真相。
- Contradiction Engine 只提供攻击抓手，不直接替 Agent 发言。
- Strategy Selector 受 Persona Router 约束，避免所有 Agent 在同一局势下做出同质化选择。

### 8.3 示例 LocalContext

```json
{
  "identity_view": {
    "self_role": "werewolf",
    "faction_task": "push_good_player_out",
    "persona_strategy": "deep_hook"
  },
  "belief_state": {
    "player_03": {"role_read": "seer_lean", "confidence": 0.9},
    "player_04": {"faction_read": "good_lean", "confidence": 0.67}
  },
  "salience_items": [
    {"weight": "high", "fact": "player_04 defended player_03 yesterday, then attacked player_03 today"}
  ],
  "contradiction_alerts": [
    {"priority": "high", "claim": "player_04 stance reversal lacks explanation"}
  ],
  "strategy_directive": {
    "package": "deep_hook",
    "goal": "attack player_04 while implicitly protecting player_03"
  }
}
```

## 9. RAG 设计

### 9.1 知识库类型

V1 规划 6 类 RAG 知识库：

1. 角色策略库：预言家、女巫、狼人、混血儿等角色打法。
2. 发言案例库：不同局势下的发言模板和案例。
3. 外部高端对局案例库：来自公开赛事、公开复盘、高手实战解说和高质量训练局的关键局面、发言、投票、夜间决策和复盘结论。
4. 外部高端战术库：从高端对局中抽取的悍跳、倒钩、抗推、冲票、站边、警徽流、女巫用药、混血儿隐匿等打法模式。
5. 项目内历史对局案例库：项目内已运行对局的关键局面、发言与结果，用于个性化成长和错误纠正。
6. 项目内复盘经验库：Agent 自身对局复盘沉淀出的经验、误区、人格偏差和模型失败案例。

知识库不能只依赖项目内自举。外部高端对局是 V1 的高优先级学习源，用于给 Agent 提供“强玩家如何处理相似局面”的范例；项目内历史对局主要用于个体成长、错误纠正和评测对比。

外部高端对局入库要求：

- 来源必须公开可访问或已获得授权，记录 `source_type`、`source_url`、`source_title`、`source_author`、`publish_date` 和采集时间。
- 必须标注规则包、人数版型、角色配置、局面阶段、玩家视角、公开信息边界和结局。
- 不直接整段搬运长篇原文；优先保存结构化摘要、关键决策点、短引用、时间戳和回看链接。
- 每个案例必须标注质量等级，例如 `pro_match`、`expert_review`、`high_rank_game`、`community_case`，并记录人工审核人和审核状态。
- 外部案例只作为策略与表达参考，不得作为规则真相，不得覆盖 `RuleEngine`。

禁止进入 RAG 的内容：

- 基础规则。
- 角色技能真相。
- 胜负结算。
- 特殊结算优先级。
- 当前存活状态。

这些内容必须来自 `RuleEngine`、规则配置和 System Prompt 静态上下文。

### 9.2 检索策略

检索条件：

- 当前角色
- 当前阶段
- 当前局势
- 当前规则包和版型
- 当前可见信息边界
- 人格类型
- 需要解决的问题
- 历史相似局面
- 案例质量等级
- 来源类型

检索优先级：

1. 先检索同规则包、同阶段、同角色视角的外部高端对局案例。
2. 再检索外部高端战术库，补充可迁移打法。
3. 再检索项目内历史对局与自身复盘，用于纠正该 Agent 的惯性错误。
4. 最后检索通用发言模板，避免所有 Agent 发言同质化。

RAG 输出进入 Agent 上下文，但必须标注来源和用途。

RAG 的重要原则：

- 只回答“怎么打得更像人”和“相似局面里别人怎么发言/决策”。
- 不回答“规则是什么”和“这个技能怎么结算”。
- 不可覆盖 `RuleEngine` 的确定性判定。
- 任何 RAG 命中都必须展示在观战台，便于分析它是否诱发幻觉。
- 外部高端案例和项目内复盘必须分开展示来源，避免 Agent 把自身低质量自举经验误认为高端打法。
- 检索注入必须经过 Visibility Policy；外部案例中的上帝视角复盘只能在复盘阶段使用，不能在对局中泄露给玩家 Agent。

### 9.2.1 RAG 防御层 (rag-hardening, 2026-06-09)

4 道防线的现状 (从 ingestion 到 live prompt):

| 层 | 职责 | 实现 |
|---|---|---|
| 1. Schema 封闭 | RAGEntry 字段封闭 (无 `private_intent` / `wolf_team` / `seer_result` 等) | `rag/schemas.py:115` |
| 2. Ingestion 校验 | 4 层验证: 禁止内容 (4 个英文 keyword) / 来源元数据 / 质量等级 / 基础规则正则 (16 模式) + PII 过滤 (`\bp\d{2}\b` 拒绝 p01-p12) + catch-all 模式 (`pNN is <role>` / `pNN 查杀` 等泛指身份断言) | `rag/ingestion.py:64-183` |
| 3. Visibility 过滤 | retriever `_filter_candidates` 过滤 GOD_VIEW/MODERATOR_ONLY | `rag/retriever.py:407-450` |
| 4. Renderer | `_FORBIDDEN_LIVE_FIELDS` 14 字段 + 3 字段白名单 + `hits_to_prompt_lines` 二次过滤 `allowed_in_live_context=False` (defense-in-depth) | `rag/prompt_renderer.py:32-48, 223-289` |

**4 道防线的关键约束** (避免 LLM prompt 层被穿透):

- **`role_perspective` 过滤的精细度**: `any` / `general` 是"通用视角"标签,**不等同于"内容无角色专享"**。任何带狼内部战术的 seed 必须用 `werewolf` 而非 `any`,否则村民 LLM 也会看到。例如 `seed_foundation_peace_night` (公共观察视角) 与 `seed_foundation_peace_night_wolf` (狼队内部战术) 拆分。
- **PII 拒绝**: RAG 是策略库/公共知识,**禁止命名具体 player slot**。`\bp\d{2}\b` 过滤 p01-p12 引用,因为跨局 player ID 匹配会泄漏身份。
- **catch-all 模式**: 即使 PII 过滤漏过,`_validate_not_rule_truth` 还会拒绝 "X 是狼" / "X 查杀" / "X 金水" 等泛指身份断言 (双层防御)。
- **renderer 不只是字段白名单**: `hits_to_prompt_lines` 还做 `allowed_in_live_context` 二次过滤,即使 retriever 失效,GOD_VIEW 内容也不会到达 live prompt。

**反思记忆的同步防御** (`memory/reflection.py` + `agent_adapter.py:_GOOD_REFLECTION_TEMPLATE / _WOLF_REFLECTION_TEMPLATE`): 反思模板末尾有【PII】段(2026-06-10 起从 8 行精简为 1 行,理由:模板里 8 行铺陈对 LLM 行为影响极小,反而吃 context 预算),LLM 收到"不要写具体玩家 ID"提示;**后处理兜底** 由 `agent_adapter._agent_reflection` 返回 `reflection_text` 前调 `memory/store._scrub_player_ids` 正则替换 `p\d+ → [玩家ID已省略]`,ReflectionMemory 写入前再过一遍(`memory/store._store_review_reflection`)。理由:反思文本会跨局注入到下一局 player 的 prompt,如果对方玩家 ID 在下局匹配到会造成跨局信息泄漏。两层防御:模板 1 行提示(LLM 自律) + 后处理正则(强制生效)。

### 9.3 冷启动方案

RAG 不应只依赖人工大规模编写案例库，也不能只靠纯 LLM 自举。V1 采用四阶段冷启动：

1. Phase A：外部高端对局种子。优先整理公开高质量对局、公开赛事复盘、高手解说和训练局片段，形成第一批强样本。
2. Phase B：规则驱动种子。从 `RuleEngine` 的合法动作集合、结算结果和常见局势标签自动生成基础策略文档，例如“预言家拿警徽后如何声明警徽流”“平票 PK 时狼队如何冲票”。
3. Phase C：对局自举。运行一批纯 LLM 对局，不接 RAG，筛选“决策明显导致胜利/失败”的片段作为候选案例，但默认质量低于外部高端对局。
4. Phase D：人工精选。人工标注高质量片段，区分正面教材、反面教材、角色视角、适用阶段、信息边界和风险边界。

冷启动产物进入 RAG 前必须标注来源、适用规则包、适用阶段、质量等级和可见信息边界。任何从外部案例或对局自举产生的案例都不能被当作规则真相。

## 10. 记忆系统设计

### 10.1 记忆类型

短期记忆：

- 本局发言
- 本局投票
- 本局怀疑对象
- 当前站边
- 当前身份判断
- 局势认知矩阵
- 结构化关系链

长期记忆：

- 历史错误判断
- 成功策略
- 被欺骗案例
- 常见玩家风格
- 自己的成长记录
- **跨局错误模式聚合** (2026-06-09): 反思文本含【投票错误】/【悍跳分析】等 section header,纯 regex 解析为 category,跨局统计 top 错误类别 + 保留优点段计数,作为强信号注入下一局 LLM prompt

玩家画像：

- 逻辑能力
- 欺骗能力
- 领导能力
- 发言可信度
- 学习速度
- 风险偏好变化

短期记忆不使用向量库盘逻辑。每个 Agent 在本局内维护 JSON 格式的局势认知矩阵：

```json
{
  "player_07": {
    "role_probabilities": {
      "werewolf": 0.46,
      "villager": 0.24,
      "seer": 0.08,
      "witch": 0.06,
      "hunter": 0.05,
      "idiot": 0.05,
      "hybrid": 0.06
    },
    "faction_read": "wolf_lean",
    "trust": 0.31,
    "key_evidence": ["day1_voted_player_03", "day2_attacked_player_09"],
    "open_questions": ["是否在倒钩狼队友"]
  }
}
```

结构化关系链记录为可查询事件表或轻量知识图谱：

- `spoke_against(A, B, day)`
- `voted(A, B, round)`
- `claimed_role(A, role, day)`
- `defended(A, B, day)`
- `night_result_claimed(A, target, result)`

向量数据库只存储非结构化反思，例如“上次轻信情绪化发言导致误站边”。盘狼坑、投票链、攻防关系必须优先查询结构化状态表或知识图谱。

### 10.2 复盘机制

每局结束后，每个玩家生成：

- 本局关键判断
- 错误判断原因
- 成功策略
- 被谁欺骗或误导
- 下局改进建议
- 能力参数变化

**反思 LLM 调用的 task_type** (2026-06-10 修复,`P0-RF1`):
`_agent_reflection` 用 `TaskType.REFLECTION` (而非 `SPEECH`),
避免 `speech_quality_phase` 跑 4 字段公开发言质量检查
(反思文本没有 `stance` / `suspicion_target` / `vote_leaning` /
`evidence`)。`TaskType.REFLECTION` 在 `schemas.py:64` 存在,
`speech_quality_phase` 映射表里没它 → 返回 None → 短路退出。

**复盘模板必须按角色族分支** (2026-06-09 反馈,详见 `PROGRESS.md` `reflection-role-specific` phase):

- **好人阵营** (villager / seer / witch / hunter / idiot / hybrid-master-good): 反思方向
  - 投票错误 / 站错边 / 信息缺失 / 神职执行(预言家/女巫/猎人/白痴专项)
- **狼人阵营** (werewolf / hybrid-master-wolf): 反思方向
  - 悍跳分析(为什么没人信) / 暴露原因(发言/票型留痕) / 角色分工(深水/冲锋/倒钩)
- **混血儿** (master 未知时): 通用模板
- **共同强制项**: 末尾必须含"【保留的优点】"段,要求列出 1-2 个本局做对的具体策略下局复用

不分角色的通用 prompt 会让反思质量下降,因为不同角色的"错误"维度完全不同(好人不会悍跳失败,狼人不会站错边)。

**结构化与 prompt 边界补充**（2026-06-18，`reflection-synthesis-upgrade`）：

- LLM 自评中【保留的优点】段由 `ReflectionSynthesizer._extract_llm_strengths` 解析为 `preserved_strengths`（fact-free，含真相 token 的条目丢弃，玩家 ID scrub），与确定性 `successful_strategies` 经 Jaccard 去重合并（总上限 3）。这是 spec `Synthesis` rule 1 的 strengths 部分（mistake 提取需 det 校正，留后续）。
- `mistake_patterns[].wrong_action` 承载确定性真相（`review.py:108` 写 `实际 {actual}`），是审计字段，**不**属 prompt-visible；`prompt_card` 仍是唯一面向玩家 prompt 的摘要（spec `prompt_card` 仅 prompt-facing 约束）。
- 混血儿 master 阵营未知时走通用模板仍是设计意图（见上），本次不动。
- `corrected_from_llm` 否认检测关键词已扩宽（`没什么问题`/`判断都挺准`/`没(有)?失误`/`都对` 等），但收紧以避免匹配错误自述（如`我的判断都错了`）。

**反思 context 裁剪 + LLM mistake 解析**（2026-06-21，`fix-reflection-memory-dead-chain`，补 2026-06-18 留的"mistake 提取后续"）:

- `_agent_reflection` 在 `build_agent_context(REFLECTION)` 后用 allowlist `_POST_GAME_KEEP = {reflection_task, game_outcome}` 剥离赛内决策 directive（`role_alerts`/`skill_tactical_advice` 等），让赛后反思指令成为压倒性内容——修复 LLM 曾输出赛内决策噪声（狼人写"今晚刀X"、好人写"第N夜观察"）而非 section 化反思。
- `ReflectionSynthesizer._extract_llm_mistakes` 解析 6 个 section（投票错误/信息缺失/神职执行/悍跳分析/暴露原因/角色分工）bullet 为 `ReflectionMistakePattern`，category 按 header 映射（不走关键词 `_category`）；镜像 `_extract_llm_strengths`。merge 进 `synthesize`：deterministic 优先、LLM 补充、Jaccard≥0.6 去重、总上限 3。
- **安全约束**：LLM mistake 恒设 `auto_verified=False` + `fact_basis="llm_transferable"`。`auto_verified=True` 会令 `_has_unsafe_truth_claim` 直接返回 False，旁路 truth-token 防护；故 LLM mistake 必须为 False，与 deterministic mistake（`auto_verified=True`，因其 wrong_action 是确定性真相、audit-only）严格区分。truth-token bullet 经 per-bullet drop（`_LLM_TRUTH_TOKENS`）+ entry-level gate 双重过滤。
- **score 结论**（`scripts/analyze_reflection_score.py` 5 场景演算）：1b 后全部 approved（最低 0.75）。`short_prompt_card`（<80）罚分仅在"全 fallback"路径（patterns/success/improvement 皆空 + 非 seer/werewolf）触发，但该路径 score 天花板本身 <0.70，不触及 approved 门槛。故 `fix-belief-cognition.md`（defect B）降级为满分兜底，非 approved 门槛硬依赖。

**跨局检索与排序** (`context.py:_reflection_memory_hints`, 2026-06-09 增强):

- 排序 key: `(-priority, -faction_won, neg_game_id, entry_id)`
  - priority: 同角色=2, 同阵营=1, 异阵营=0
  - faction_won=True 在同 priority 内排前(成功模式可复用,优先级高于失败教训)
- hint 预算 5 → 8 (覆盖 4 个角色族 × 2 hint/族)
- 同 player 反反思按 (player_id, role) 自然累积;跨 player 同角色 (priority=2) 和同阵营 (priority=1) 是主要学习通路
- V1 板子每局角色随机分配,同 player+同角色罕见 (~1/12),所以"跨 player 同角色"是更现实的学习路径

**错误模式聚合提示** (新 `error_pattern_hint` 字段, 2026-06-09):

- `_compute_error_pattern(reflections, current_role)` 统计 top 2 错误类别 + 保留优点段数 + 同角色反思数 + dominant 错误占比
- 渲染为 LLM prompt 顶部独立段: "【跨局错误模式】你最常犯的 2 类错误: vote_mistake(3次)、role_execution(2次)。"
- 不调 LLM,纯 section header regex 提取 — 0 额外成本

**反思记忆的提示优先级** (`prompt_builder.py:_SECTION_PRIORITIES`, M4-2, 2026-06-09):

- `_build_reflection_memory_hints` 标记为 `【参考】` (参考 tier, 中优先级)
- `_build_rag_hints` 标记为 `【辅助】` (辅助 tier, 低优先级,budget 紧张时先被裁掉)
- 排序依据: **per-player 同角色累积反思 (本玩家本角色经验) 价值 > 通用 RAG 知识 (社区语料)**。M4-2 是 G-R4-15 的反向决议 — G-R4-15 当时以"检索成本高"为由把 RAG 升到参考 tier,M4-2 视角是"检索成本 ≠ prompt 价值":RAG 替换自语料库,丢了不重复犯同样错;反思是本玩家本角色私有积累,丢了会在多局重复同一个错误。
- 预算裁剪顺序 (新): 辅助 (persona / phase / belief / public_summary / visible_state / private_memory / salience / **rag_hints** / profile / cognition / error_pattern) → 参考 (**reflection_memory_hints**) → 硬约束 (strategy_directive / retry_hint / output_contract, 永不被裁)

**Schema 边界**: `ReflectionEntry` (memory/schemas.py:254) 字段不动 (entry_id / game_id / player_id / role / faction_won / text / tags / situation);只改 `text` 字段的内容质量。新功能不引入新列。

**Storage 边界**: PostgreSQL `reflections` 表 (entry_id, game_id, player_id, entry_json) 不动;LLM 改进的反思文本作为 `entry_json.text` 字段值序列化,旧数据无影响。

复盘中的经验段落进入长期向量记忆，并在后续对局中被检索。结构化事实进入关系表，不写成向量后再反查。

## 11. Skill 与 MCP 设计

### 11.1 Skill 设计

狼人杀技能可抽象为 Agent Skill：

- 悍跳
- 对跳
- 归票
- 冲票
- 倒钩
- 找神
- 藏身份
- 抗推
- 盘狼坑
- 保护强神
- 遗言分析
- 复盘纠错

Skill 输入：

- 当前角色
- 当前阶段
- 当前局势
- 人格参数
- 可用历史经验

Skill 输出：

- 推荐行动
- 推荐发言结构
- 风险提示
- 置信度

### 11.2 MCP/Tool 设计

内部 LangChain/LangGraph Tool：

- 查询 `RuleEngine` 可用动作
- 查询当前公开/私有状态
- 查询结构化关系链
- 查询 Agent 认知矩阵
- 写入复盘
- 调用评测器
- 读取当前实验配置
- 生成对局报告

MCP 只用于外部扩展能力：

- 查询外部历史对局库
- 查询外部玩家画像库
- 接入用户自定义提示词库
- 接入外部战术库、Notion、Slack 或本地 IDE
- 接入玩家自己写的“外挂顾问”或实验插件

MCP 定位原则：

- 高度耦合的本地规则查询、状态读写、对局推进不走 MCP，避免不必要的 RPC 和协议转换复杂度。
- MCP 层只提供外部工具能力，不拥有游戏状态真相。
- 所有外部 MCP 结果都必须标注来源，并被视为建议而非裁判事实。

## 12. API 与观战台设计

### 12.1 API

核心 API：

- `POST /games`：创建对局
- `POST /games/{game_id}/start`：开始对局
- `POST /games/{game_id}/pause`：暂停
- `POST /games/{game_id}/resume`：恢复
- `GET /games/{game_id}/public-state`：公开状态
- `GET /games/{game_id}/players/{player_id}/private-state`：私有状态
- `GET /games/{game_id}/timeline`：时间线
- `GET /games/{game_id}/replay`：回放
- `GET /games/{game_id}/evaluation`：评测结果

API 权限边界：

- `public-state`、`timeline` 默认只返回公开事件、公开发言、公开投票和公开死亡信息。
- `private-state` 必须要求 `moderator`、`debugger` 或对应 `player_agent` 自身权限；玩家 Agent 只能读取自己的私有视角，不能读取他人身份、夜间结果、`private_intent` 或认知矩阵。
- `replay` 和 `evaluation` 必须支持 `public`、`player_view`、`moderator_full` 三种视图；对局进行中默认禁止向玩家 Agent 暴露 `moderator_full`。
- 所有越权访问尝试必须写入审计日志，并在评测中标记为信息泄露风险。

### 12.2 观战台

观战台展示：

- 当前阶段
- 12 名玩家状态
- 发言时间线
- 投票关系
- 死亡记录
- 身份视角
- RAG 命中
- 认知管线输出
- 显著事件权重
- 逻辑矛盾警报
- 记忆调用
- 人格路由记录
- 模型路由记录
- Agent 输出结构
- Agent `private_intent` 审计视图
- 认知差异图
- 复盘报告
- 实验指标

Killer UI：认知差异图。

- 左侧展示真实身份底牌、真实阵营和关键夜间结果。
- 右侧展示所选 Agent 脑内的玩家身份概率分布、信任度、阵营判断和证据链。
- 时间轴支持回放某个 Agent 如何从“认好人”逐步误判成“铁狼”，或如何被悍跳/倒钩误导。
- 该视图是项目区别于普通聊天记录 UI 的核心卖点，用于观察 Agent 幻觉、欺骗和逻辑崩盘。

认知差异图属于 `moderator_full` 调试视图，只能在观战者、研究者或复盘阶段展示；对局进行中不得把真实身份、他人私有夜间信息、他人 `private_intent` 或完整认知矩阵注入任何玩家 Agent 上下文。

模型可观测性：

- 每个玩家面板展示当前绑定的 `llm_profile`、provider 和 model。
- 每次发言、投票、夜间行动、复盘都展示实际命中的模型、耗时、token、估算成本和是否触发 fallback。
- 对局结束后展示按玩家、按任务类型、按 provider 聚合的成本与延迟。

人格可观测性：

- 每个玩家面板展示当前绑定的 `persona_profile`、基础人格参数和动态修正项。
- 每次发言、投票、夜间行动、复盘都展示命中的 task style 和 dynamic policy。
- 时间轴支持对比“人格路由变化”和“模型路由变化”，避免把模型能力差异误判成人格差异。

认知可观测性：

- 每次 Agent 调用展示 Attention Filter 过滤前后的信息类别统计。
- 展示 Salience Engine 选入 Prompt 的高权重事实，以及被摘要或仅留存的低权重事实。
- 展示 Belief Updater 对身份概率、信任度和阵营倾向的变化。
- 展示 Contradiction Engine 发现的矛盾警报和是否被 Agent 采纳。
- 展示 Local Context Builder 的 token 预算，说明为什么某些历史事件没有进入本次上下文。

## 13. 数据存储设计

准生产默认：

- PostgreSQL：对局、玩家、事件、死亡记录、警徽记录、评测结果、配置快照、人格路由配置快照、人格运行时快照、认知管线快照、模型路由配置快照、模型调用日志、成本统计。
- Redis：运行时缓存、任务状态、临时锁。
- Qdrant 或 pgvector：策略、外部高端对局案例、外部高端战术、项目内历史对局案例、发言模板、复盘经验向量检索。
- PostgreSQL：外部案例元数据、来源链接、版权/授权状态、质量等级、人工审核状态、适用规则包、适用阶段和可见信息边界。
- PostgreSQL 表或轻量知识图谱：投票链、攻防关系、身份声称、警徽流声明、认知矩阵快照、矛盾警报。
- 对象存储或本地文件：完整对局日志、报告、调试追踪。

V1 可以先用本地文件或 SQLite 做开发替代，但接口边界要按准生产设计。

Lite 模式：

- 支持每个玩家绑定自己的 `llm_profile`，法官和关键神职可用强模型，普通村民可用本地 Ollama 或低成本 Flash 模型。
- 支持同一玩家按任务切模型：日间发言用低成本模型，悍跳、夜间关键动作和复盘用强模型。
- 提供 `docker-compose.yml` 的低成本配置，默认可以用本地/轻量模型跑通一局。
- 所有模型调用必须可记录 token、耗时和成本，便于开发者评估实验门槛。

## 14. 评测体系

基础指标：

- 阵营胜率
- 玩家胜率
- 狼人胜率
- 平民抗推率
- 谎言被识破率
- 站边准确率
- 投票命中率
- 发言影响力
- 身份伪装成功率
- 悍跳成功率
- 倒钩收益
- 混血儿主人选择收益
- 混血儿共同获胜率
- 女巫用药收益
- 预言家警徽流质量
- 警徽传递/撕徽决策质量
- 发言顺序利用率
- 狼队夜间共识质量
- 玩家成长曲线
- 人格路由稳定性
- 动态人格修正收益
- 认知管线压缩率
- 私有信息泄露率
- 矛盾警报命中率
- 单玩家模型成本
- 单玩家模型延迟

实验维度：

- 不同模型对比
- 不同玩家模型组合对比
- 不同任务路由策略对比
- 不同人格对比
- 不同人格路由策略对比
- 有无 Cognitive Coprocessor 对比
- 不同显著性排序策略对比
- 不同策略 RAG 对比
- 不同记忆策略对比
- 有无复盘进化对比

标准化 Benchmark：

- 将项目包装为狼人杀 Eval，用固定规则、固定初始配置、固定随机种子集合评测模型推理和欺骗能力。
- 输出模型排行榜所需的标准 JSON 报告。
- Leaderboard 维度包括狼人胜率、好人胜率、平民抗推率、谎言被识破率、站边准确率、非法动作率、平均单局成本和平均单局延迟。

## 15. 分阶段落地路线

### Phase 0：设计与项目骨架

交付：

- 项目目录结构
- 规则包配置格式
- 人格包配置格式
- 每玩家 `persona_profile` 配置格式
- 每玩家 `llm_profile` 配置格式
- 核心数据模型设计
- API 草案
- Lite 模式配置草案
- 测试计划

学习目标：

- 理解一个 Agent 项目的分层方式。
- 理解规则引擎和 LLM Agent 为什么必须解耦。
- 理解低成本本地运行对开源项目传播的意义。

### Phase 1：规则闭环

交付：

- 预女猎白混规则引擎
- 12 人身份分发
- 夜晚行动流程
- 死亡原因、死亡时机和技能触发矩阵
- 女巫全程不能自救、猎人毒死不能开枪、白痴被放逐翻牌后可遗言并出局等规则
- 平安夜、平安日、平票和自爆分支
- 警徽竞选、警徽流、传徽和撕徽规则
- 白天发言占位流程
- 投票与放逐
- 胜负判定

学习目标：

- 理解确定性状态机。
- 理解复杂游戏流程如何建模。

### Phase 2：LangGraph 编排

交付：

- 游戏主流程图
- checkpoint
- pause/resume
- replay timeline
- 狼人夜间讨论与共识节点
- 警长竞选拆分节点
- 多轮白天讨论、有警长时的归票和投票节点
- 发言顺序生成和记录
- 白天 Agent 状态并行更新

学习目标：

- 理解 LangGraph 的状态、节点、边、条件跳转和持久化。
- 理解 `Send` API 的 Map-Reduce 式并行更新。

### Phase 3：全员 LLM Agent

交付：

- 12 个玩家 Agent
- 法官 Agent
- Persona Router
- LLM Router Gateway
- 每玩家人格绑定
- 每玩家模型绑定
- 结构化输出校验
- 非法动作重试
- 人格化发言
- `private_intent` 与公开发言分离
- 按玩家与任务类型异构人格路由
- 按玩家与任务类型异构模型路由

学习目标：

- 理解多智能体交互。
- 理解 Agent 输出约束与安全边界。
- 理解人格路由如何隔离静态人设、动态局势和复盘成长。
- 理解模型路由如何隔离 provider 差异和成本策略。
- 理解欺骗型任务为什么需要结构化私有策略缓冲。

### Phase 4：上下文工程

交付：

- Cognitive Coprocessor 基础管线
- 结构化世界状态
- Attention Filter
- Salience Engine
- Belief Updater
- Contradiction Engine
- Strategy Selector
- Local Context Builder
- 公开上下文
- 私有上下文
- 阶段任务上下文
- 局势摘要器
- token 预算策略
- 当前日逐字稿与历史摘要分层

学习目标：

- 理解上下文隔离、摘要、动态构建和结构化局势压缩。
- 理解为什么 LLM 不应承担记账、算概率和私有信息过滤。

### Phase 5：RAG

交付：

- 策略知识库
- 案例知识库
- 外部高端对局案例库
- 外部高端战术库
- 历史对局案例库
- 高级战术库
- 检索器
- 外部公开高质量对局导入与标注流程
- 规则驱动 RAG 种子
- 纯 LLM 对局自举案例
- 人工精选与正反例标注
- RAG 命中展示

学习目标：

- 理解 RAG 的索引、检索、重排、注入和评估。
- 理解封闭规则系统中为什么不能用 RAG 查基础规则。

### Phase 6：记忆与复盘

交付：

- 短期记忆
- 局势认知矩阵
- 结构化关系链
- 长期记忆
- 玩家画像
- 复盘记录
- 跨局检索

学习目标：

- 理解 Agent memory 的写入、筛选、压缩、检索和遗忘。
- 理解向量记忆与结构化关系记忆的边界。

### Phase 7：Skill 与 MCP

交付：

- 狼人杀技能注册表
- 关键玩法 Skill
- 内部 LangGraph Tool
- 外部 MCP 扩展接口
- 工具调用日志

学习目标：

- 理解 Agent 能力模块化和外部工具化。
- 理解 MCP 适合外部系统连接，不适合承载本地核心规则闭环。

### Phase 8：观战台

交付：

- 对局时间线
- 玩家状态面板
- 每玩家人格配置面板
- 每玩家模型配置面板
- 发言与投票可视化
- 记忆、RAG 和私有审计摘要查看，并按 `public`、`player_view`、`moderator_full` 权限隔离
- 人格路由、task style、dynamic policy 追踪
- 模型调用、fallback、成本和延迟追踪
- 认知差异图
- Agent 身份概率分布表
- 人工暂停与恢复

学习目标：

- 理解 Human-in-the-loop Agent 系统。
- 理解可观测性如何成为项目差异化能力。

### Phase 9：评测实验室

交付：

- 批量对局
- 指标统计
- 实验对比
- 模型路由策略对比
- 成长曲线
- Leaderboard 标准报告
- 单局成本统计
- 报告导出

学习目标：

- 理解 Agent 应用如何评测、优化和迭代。
- 理解如何把狼人杀包装成模型推理与欺骗能力基准。

## 16. 测试计划

规则测试：

- 12 人配置正确。
- 混血儿首夜选择主人。
- 预言家查验混血儿返回好人。
- 混血儿跟随主人阵营共同胜利。
- 混血儿或主人出局后，混血儿仍按主人原始阵营结算个人胜负。
- 混血儿主人为好人时，狼人屠民必须让三名平民和混血儿全部出局；混血儿主人为狼人时，狼人屠民只需让三名平民全部出局。
- 女巫救药和毒药限制正确。
- 猎人死亡后触发开枪条件正确。
- 白痴被放逐后翻牌逻辑正确。
- 死亡记录包含死亡原因、死亡时机和结算批次。
- 猎人被女巫毒死不能开枪。
- 白痴被毒死不触发翻牌。
- 白痴被放逐翻牌后可以发表遗言，随后出局；不能再发言、投票或接警徽，并立即计入神边出局。
- 警徽竞选、退水、投票、传徽和撕徽流程正确；撕徽后本局无警长且后续发言随机起点。
- 平安夜、平安日、首次平票 PK、再次平票无人出局、自爆无遗言和遗言权按规则配置生效。
- 女巫首夜和后续夜晚均不能自救。
- 屠民、屠神、狼人全灭判定正确。

Graph 测试：

- 夜晚按狼人、女巫、预言家、首夜猎人/白痴确认、首夜混血儿主人绑定的规则顺序落地，并和 LangGraph 条件节点一致。
- 狼人夜间讨论和共识流程正确隔离私有信息。
- 混血儿主人绑定不进入狼人夜聊、公共事件或其他玩家上下文。
- 警长竞选拆分节点顺序正确。
- 白天多轮讨论、有警长时归票、无警长时跳过归票和投票流程正确。
- 警长因夜刀、女巫毒、猎枪、放逐或自爆出局时，均能进入传徽/撕徽结算；撕徽后后续白天保持无警长状态。
- 发言顺序能被警长或 V1 无警长随机起点规则决定并写入状态。
- 发言到投票流程正确。
- 放逐到技能触发流程正确。
- `GameEvent` 能通过确定性 reducer 重放出 `GameState`，暂停恢复和人工干预不绕过事件日志。
- checkpoint 可恢复。
- 人工干预后可继续执行。

Agent 测试：

- 玩家不能看到不该看的私有身份。
- API 与观战台权限正确隔离：玩家 Agent 只能读取自己的 `player_view`，不能读取 `moderator_full`、他人私有状态、他人 `private_intent` 或真实身份底牌。
- 行动必须在可用动作集合内。
- 投票目标必须合法。
- 发言符合人格参数。
- 欺骗任务必须生成对外不可见的结构化 `private_intent`，且不保存长篇自然语言私有推理文本。
- Agent 人格参数必须经过 Persona Router。
- 每个玩家能绑定不同的 `persona_profile`。
- Agent 模型调用必须经过 LLM Router Gateway。
- 每个玩家能绑定不同的 `llm_profile`。
- 输出必须可结构化解析。
- function calling 或 schema enum 能限制非法动作和非法目标。
- 三步生成能隔离 `private_intent`、结构化动作和公开发言。

认知管线测试：

- Visibility Policy 先于 Attention Filter、RAG 注入和 Local Context Builder 执行，且不泄露当前 Agent 不应知道的私有信息。
- Attention Filter 的输出可审计，并能定位到具体可见性规则。
- Salience Engine 能把高权重事实放入本次上下文。
- Belief Updater 能随死亡、跳身份、投票和发言更新认知矩阵。
- Contradiction Engine 能发现结构化关系链中的明显矛盾。
- Local Context Builder 能在 token 预算内输出完整任务上下文。
- Cognitive Coprocessor 产物能在观战台审计。

RAG 测试：

- 基础规则不能进入 RAG 检索路径。
- 策略检索与当前局势相关。
- 错误检索不能改变规则引擎判定。
- 外部高端对局案例必须带来源、链接、规则包、阶段、质量等级、审核状态和可见信息边界。
- 同等相关性下，检索优先命中外部高端对局，再命中项目内自举案例。
- 上帝视角复盘内容不能注入对局中玩家 Agent 的实时上下文，只能用于复盘阶段。
- 冷启动 Phase B 生成的规则驱动策略不进入规则裁判路径。
- 自举案例必须带来源、适用阶段和正反例标注。

记忆测试：

- 本局事件能写入短期记忆。
- 投票、踩人、保人、身份声称能写入结构化关系链。
- 认知矩阵能随发言和投票更新。
- 复盘能写入长期记忆。
- 下一局能检索历史经验。
- 人格成长能影响后续行为。

评测测试：

- 能统计单局结果。
- 能统计多局胜率。
- 能比较不同人格、模型、策略 RAG。
- 能比较每玩家人格组合和动态人格路由策略。
- 能比较每玩家模型组合和任务路由策略。
- 能比较有无 Cognitive Coprocessor 的上下文长度、泄密率和胜率差异。
- 能统计矛盾警报命中率和被采纳率。
- 能输出 Leaderboard 标准指标。
- 能输出成长曲线。

## 17. 风险与边界

主要风险：

- 全员 LLM Agent 调用成本高。
- 12 人多轮发言容易导致上下文过长。
- LLM 容易输出非法动作或泄露身份。
- 认知过滤器如果实现错误，会把夜间私有信息泄露给错误角色。
- 显著性排序如果过度压缩，会丢失关键站边和投票细节。
- 规则如果依赖自然语言会导致裁判不稳定。
- 用 RAG 查询规则会导致关键结算幻觉。
- 用向量库承载关系链会导致盘狼坑逻辑崩盘。
- 串行更新 12 个 Agent 会导致延迟和成本膨胀。
- 人格路由如果缺少快照，会导致行为变化不可解释。
- 模型路由如果缺少日志，会导致实验不可复现、成本不可解释。
- 长期记忆如果不筛选，会污染后续推理。

边界原则：

- 规则必须确定性。
- 基础规则必须常驻 System Prompt 和规则配置，不能依赖 RAG。
- LLM 只负责推理、发言和决策建议。
- LLM 不负责维护完整世界状态、身份概率计算和私有信息过滤。
- Agent 输入必须先经过 Cognitive Coprocessor，不能直接拼接完整公开/私有历史。
- 认知管线输出必须随对局快照持久化，便于回放和审计。
- Agent 不直接拼装 persona prompt，必须通过人格路由运行时。
- 人格路由配置和运行时结果必须随对局快照持久化。
- Agent 不直接调用 provider SDK，必须通过模型路由网关。
- 模型路由配置必须随对局快照持久化。
- 私有信息必须隔离。
- 关系逻辑必须结构化，向量记忆只承载复盘经验。
- 记忆写入必须有筛选、摘要和结构化归档。
- RAG 不得覆盖裁判，也不得作为规则真相来源。
- 人工干预必须可审计。

## 18. 参考来源

- 烂柯游艺社经典版型：https://lanke.fun/classic-templates/
- 口袋狼人杀预女猎白混规则：https://www.langrensha.net/strategy/2021091702.html
- 游民岛 13 人变体说明，仅作后续规则包扩展对照，不作为 V1 预女猎白混裁判依据：https://www.youmindao.com/langrensha/488.html
- 京城大师赛复盘公开内容搜索参考：https://search.bilibili.com/all?keyword=%E4%BA%AC%E5%9F%8E%E5%A4%A7%E5%B8%88%E8%B5%9B%E5%A4%8D%E7%9B%98

V1 的规则裁判权威顺序固定为：本文第 3 章规则设计、`config/rulesets/pre_witch_hunter_idiot_mixed.yaml`、`RuleEngine` 单元测试。外部链接只用于交叉参考和案例学习，不得覆盖本文已固化的标准规则。
