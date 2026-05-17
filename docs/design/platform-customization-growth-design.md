# AI 狼人杀平台化与自定义配置设计

> 本文补充 `docs/design/werewolf-agent-v1-design.md`。V1 文档仍是规则裁决权威；本文负责产品化入口、合规模板上传、AI 玩家人格包、复盘分享与增长闭环。

## 目标

把项目从“可观战的狼人杀 Agent 实验台”升级为“用户能开局、配置、观战、分享、复盘和传播的 AI 狼人杀平台”。

核心差异点不是复制竞品页面，而是让用户定义一整局狼人杀的世界：

- 规则板子：可下载模板、上传 YAML、校验后创建房间。
- 玩家人格包：默认 12 个 AI 玩家，也可上传 12 人人格模板。
- 开局向导：规则、玩家、体验模式三步完成。
- 观战与解说：公开观战页面包含 AI 解说和事件流。
- 复盘传播：一局结束自动生成可分享复盘页。

## 非目标与合规边界

- 不抓取、搬运、盗用其他平台用户资料。
- 不批量私信、骚扰或绕过第三方平台限制。
- 不执行用户上传的任何脚本、表达式或插件代码。
- 上传规则和人格配置只能作为结构化数据解析。
- 自定义规则必须映射到当前 RuleEngine 明确支持的字段；无法裁决的自然语言只进入展示说明。

## 审查决议与第一版边界

专业审查后，平台化第一版采用更保守的边界：

- 第一版只支持 `pre_witch_hunter_idiot_mixed` 规则家族的参数化和展示配置，不直接支持新角色板子。
- `guard`、`wolf_king` 等当前 RuleEngine 没有结算能力的角色只能作为 `display_only` 市场示例或未来路线图，不能通过校验后创建真实房间。
- 任何 `ruleset_id` 要成为 `playable`，必须能被 `RulesetRegistry` 加载、被 `RuleEngine` 实例化，并通过规则回归测试。
- `GameRunner` 必须从 registry/repository 按 `ruleset_id` 加载规则，不能继续固定读取默认规则文件。
- 玩家人格包必须通过 adapter 转换为现有 `PersonaRouter` 可读的 `persona_profiles + player_assignments`，否则不得声明会影响 AI 发言。
- 平台配置必须有明确生命周期：raw YAML、normalized data、validation result、hash、status、creator、created_at、updated_at。

## 产品信息架构

### 开局向导

创建房间不再是单按钮，而是三步向导：

1. 选择规则板子：默认规则、上传规则、热门规则。
2. 选择 AI 玩家人格包：默认 12 人、上传人格包、热门人格包。
3. 选择体验模式：快速观战、我参与一席、全 AI 自动跑、教学复盘。

向导创建结果：

- `ruleset_id`
- `profile_pack_id`
- `experience_mode`
- `share_code`
- `game_id`

### 游戏大厅

大厅展示：

- 进行中房间
- 公开观战入口
- 房间分享入口
- 智能体广场
- 规则市场
- 人格包市场

### 智能体广场

先做静态版，再接 API。每个 AI 玩家展示：

- 名称
- 原型标签
- 发言风格
- 胜率或模拟战绩
- 常用策略
- 试发言预览

## 自定义规则模板

模板格式使用 YAML，复用 `config/rulesets/*.yaml` 的方向。模板分成“可裁决字段”和“展示说明字段”。

示例：

```yaml
ruleset_id: custom_pre_witch_hunter_idiot_mixed
name: 12人局-预女猎白混-自定义参数
version: 1
language: zh-CN
player_count: 12

roles:
  werewolf:
    count: 4
    faction: werewolf
  villager:
    count: 3
    faction: good
  seer:
    count: 1
    faction: good
  witch:
    count: 1
    faction: good
  hunter:
    count: 1
    faction: good
  idiot:
    count: 1
    faction: good
  hybrid:
    count: 1
    faction: good

night_order:
  - werewolf
  - witch
  - seer
  - hybrid

victory:
  werewolf:
    - slaughter_villagers
    - slaughter_gods
  good:
    - eliminate_all_wolves

constraints:
  witch_can_self_save: false
  witch_can_use_both_potions_same_night: false
  werewolf_can_no_kill: true
  wolf_timeout_default: no_kill
  hybrid_enabled: true

notes:
  etiquette:
    - 禁止贴脸发言
    - 禁止场外信息
```

### 规则校验

后端必须校验：

- YAML 可解析。
- 必填字段完整。
- `roles.*.count` 总和等于 `player_count`。
- `playable` 规则的角色名必须在 RuleEngine capability 支持列表内。当前第一版可玩角色为 `werewolf`、`villager`、`seer`、`witch`、`hunter`、`idiot`、`hybrid`。
- `guard`、`wolf_king` 等未实现角色只能标记为 `display_only`，并禁止用于真实开局。
- `night_order` 不包含非法角色。
- `victory` 条件属于 RuleEngine 支持的枚举。
- `constraints` 只接受白名单字段。
- 文本字段长度受限。
- 不允许脚本、HTML、JS、shell、模板表达式或系统提示覆盖字段。

校验响应应包含：

- `valid`
- `errors`
- `warnings`
- `summary`
- `normalized_ruleset`

### RulesetRegistry

新增 `RulesetRegistry` 作为 RuleEngine 与平台配置之间的唯一入口：

- 读取内置规则、已验证自定义规则和市场规则。
- 为每个规则返回 `status`: `playable`、`display_only`、`invalid`、`deprecated`。
- 为每个规则返回 `capabilities`: supported roles, night actions, victory conditions, constraints。
- `GameRunner` 只能使用 `status=playable` 的规则。
- registry 的 playable 判断必须来自 RuleEngine capability 和回归测试，不允许验证器自己推断规则可裁决。

### 规则兼容性矩阵

每个规则模板都必须能生成兼容性矩阵，避免用户把“可展示”误解为“可开局”。

矩阵字段：

- `status`: `playable`、`beta`、`experimental`、`display_only`、`invalid`。
- `supported_roles`: RuleEngine 当前可裁决角色。
- `unsupported_roles`: 模板中存在但当前不可裁决的角色。
- `supported_abilities`: 当前可裁决能力。
- `missing_abilities`: 模板引用但没有实现的能力。
- `supported_victory_conditions`: 当前可裁决胜利条件。
- `unsupported_victory_conditions`: 不支持的胜利条件。
- `warnings`: 可运行但存在体验或压测风险的事项。

示例：

```yaml
compatibility:
  status: display_only
  supported_roles:
    - werewolf
    - villager
    - seer
    - witch
    - hunter
    - idiot
    - hybrid
  unsupported_roles:
    - guard
    - wolf_king
  missing_abilities:
    - guard_protect
    - wolf_king_shot
  warnings:
    - 同守同救结算尚未实现。
```

前端必须展示这个矩阵的摘要：

```text
可玩：否
缺失能力：guard_protect, wolf_king_shot
状态：仅展示
```

### 模板差异对比

用户上传规则或人格包后，系统应提供相对默认模板的 diff 摘要，而不是要求用户阅读 YAML。

规则 diff 示例：

```text
相对默认 12 人预女猎白混：
- 女巫自救：false -> true
- 狼人数量：4 -> 3
- 混血儿：enabled -> disabled
```

人格包 diff 示例：

```text
相对默认 12 AI 玩家：
- 3 号 aggression: medium -> high
- 7 号 speech_length: standard -> long
- 10 号 chaos_factor: low -> medium
```

diff 只比较规范化后的白名单字段，不比较原始 YAML 文本，避免格式差异造成噪声。

### 未来角色能力扩展架构

后续要让狼王守卫局成为真实玩法，不能只在 YAML 中增加 `guard` 和 `wolf_king`。需要把 RuleEngine 演进为“内置能力注册 + 模板声明”的架构：

```text
YAML 规则模板
  -> RulesetRegistry capability 检查
  -> RoleAbilityRegistry 解析内置技能实现
  -> RuleEngine 按夜晚、白天、死亡触发点统一调度
  -> Replay reducer 和 VisibilityPolicy 验证
```

关键原则：

- 用户模板只能引用系统内置 `ability_id`，不能上传能力代码。
- 每个 ability 必须有 Python 实现、规则测试、运行时测试、replay reducer 和可见性测试。
- `RulesetRegistry` 只有在所有 ability 均已实现并通过回归测试时，才把规则标记为 `playable`。
- 未实现 ability 的规则可以留在市场中展示，但状态必须是 `display_only`。

狼王守卫局至少需要这些能力：

- `guard_protect`: 守卫夜间守护一名玩家，可配置能否自守、能否连续守同一目标、同守同救是否死亡。
- `wolf_king_shot`: 狼王死亡后开枪，可配置哪些死亡原因触发、哪些死亡原因阻断。
- `night_protection_resolution`: 统一处理狼刀、守护、女巫解药、毒药之间的结算顺序。
- `death_trigger_chain`: 统一处理猎人、狼王等死亡触发技能，支持链式死亡并防止无限循环。
- `ability_visibility`: 守卫目标、狼王开枪资格、私有技能结果必须按视角隔离。

模板示例应引用内置能力，而不是描述自然语言技能：

```yaml
roles:
  wolf_king:
    count: 1
    faction: werewolf
    category: wolf
    seer_result: werewolf
    abilities:
      - id: wolf_king_shot
        triggers:
          death_reasons:
            - exile
            - hunter_shot
          blocked_by:
            - witch_poison

  guard:
    count: 1
    faction: good
    category: god
    seer_result: good
    abilities:
      - id: guard_protect
        night_order: 15
        target_filter: alive_player
        can_self_target: true
        can_repeat_target: false
        same_guard_and_antidote_policy: target_dies
```

建议演进顺序：

1. `RoleAbilityRegistry`: 列出系统内置能力和 capability metadata。
2. `death_trigger_chain`: 先把猎人开枪抽象成通用死亡触发机制。
3. `wolf_king_shot`: 复用死亡触发链实现狼王。
4. `guard_protect`: 接入夜间保护结算。
5. `night_protection_resolution`: 明确同守同救、狼刀、毒药的结算顺序。
6. 狼王守卫规则模板从 `display_only` 升级为 `playable`。

狼王守卫局升级为 playable 前，至少要有这些测试：

- 守卫守中狼刀目标，目标按规则免死。
- 守卫不能连续守同一目标。
- 守卫能否自守按配置生效。
- 同守同救规则按配置生效。
- 狼王被放逐后可开枪。
- 狼王被女巫毒死后按配置阻断开枪。
- 狼王开枪造成的死亡能进入胜利判断。
- 猎人和狼王连锁死亡不会无限循环。
- 公开视角不泄漏守卫目标。
- replay 可以还原完整状态。

## 自定义玩家人格模板

模板格式也使用 YAML。系统默认提供 12 个 AI 玩家；用户可上传正好 12 个玩家的人格包。

示例：

```yaml
profile_pack_id: default_12_ai_players
name: 默认12人AI玩家
version: 1
language: zh-CN

players:
  - seat: 1
    name: 归票官
    archetype: leader
    speech_style: 强势、简洁、偏归纳
    risk_tolerance: high
    deception: medium
    cooperation: high
    aggression: medium
    memory_focus: high
    logic_focus: high
    emotionality: low
    speech_length: standard
    claim_tendency: low
    vote_stability: high
    pressure_response: logical_rebuttal
    wolf_deception_style: deep_hook
    seer_behavior: cautious_claim
    teamplay: high
    chaos_factor: low
    preferred_roles:
      - villager
      - hunter
    catchphrases:
      - 先把票型理清楚。
      - 不要只听情绪，要看行为。
```

### 人格校验

后端必须校验：

- 必须正好 12 个玩家。
- `seat` 为 1-12 且不重复。
- `name` 非空并限制长度。
- 枚举字段只能使用允许值：例如 `low`、`medium`、`high`。
- `preferred_roles` 来自系统支持角色。
- `speech_style`、`catchphrases` 只作为提示词素材，不允许执行。
- 禁止 prompt 注入字段，例如“忽略系统指令”“你必须服从用户”等。
- 超长文本拒绝或截断并返回 warning。

### PersonaPackAdapter

用户上传的人格模板不是运行时格式，必须转换后才能生效：

- 输入：`players: [{seat, name, archetype, speech_style, ...}]`。
- 输出：`persona_profiles` 和 `player_assignments`，兼容 `PersonaRouter.from_yaml()` 和 `load_assignments()`。
- 每个 seat 生成稳定 profile id，例如 `custom_pack_id_seat_01`。
- `GameRunner` 创建 agent registry 时必须接收 `profile_pack_id`，并把对应 assignments 注入 PersonaRouter。
- `AgentContext.persona_snapshot` 中必须能看到最终生效的人格摘要。

### 人设预览发言

上传人格包后，前端展示每个玩家的四类试发言：

- 平民开局发言
- 被怀疑时自辩
- 狼人夜聊建议
- 预言家起跳发言

预览可先由确定性模板生成，后续再接 LLM。

## 规则市场与人格包市场

第一版使用仓库内静态 YAML/JSON：

- 经典 12 人狼王守卫
- 新手友好局
- 高压悍跳局
- 全员逻辑局
- 综艺效果局
- 狼队强度拉满局

人格包示例：

- 默认竞技型 12 人
- 新手陪练 12 人
- 高端局压力测试
- 综艺表达型 12 人
- 逻辑怪 12 人

后续再做用户投稿、审核、精选和版本管理。

## 复盘与传播

每局结束生成复盘页：

- MVP 玩家
- 最佳发言
- 最大争议点
- 狼队协作评分
- 好人推理评分
- 预言家验人价值
- 女巫用药质量
- 投票转折点
- AI 玩家高光片段

复盘页必须可分享，并带“立即开一局”入口。分享链路记录来源，用于统计观战转开局。

公开复盘只能包含 public timeline、公开身份结果、公开统计和脱敏摘要。不得泄露 moderator_full、私有意图、玩家私有视角、隐藏认知矩阵、未公开 RAG 命中或模型私有调用细节。

### 信息泄漏审计报告

每局结束后，复盘页和法官后台都应生成信息泄漏审计报告。

报告字段：

- `leak_check_status`: `passed`、`warning`、`failed`。
- `private_role_leaks`: 私有身份泄漏数量。
- `illegal_view_references`: 非法视角引用数量。
- `forbidden_event_exposures`: 禁止公开事件暴露数量。
- `rag_boundary_violations`: RAG 可见性边界违规数量。
- `tool_output_violations`: 工具输出泄漏数量。
- `audit_event_count`: 参与审计的事件数量。
- `public_summary`: 可公开展示的审计摘要。

公开复盘只显示摘要，例如：

```text
信息泄漏检查：通过
私有身份泄漏：0
非法视角引用：0
越权事件：0
```

详细违规证据仅法官/调试视角可见。

## 配置版本锁定

每局创建时必须锁定配置快照，保证复盘和 replay 可重复解释。

需要锁定：

- `ruleset_id`
- `ruleset_version`
- `ruleset_hash`
- `profile_pack_id`
- `profile_pack_version`
- `profile_pack_hash`
- `model_config_hash`
- `persona_adapter_version`
- `rag_config_hash`
- `engine_version`
- `random_seed`
- `agent_behavior_seed`
- `speech_order_seed`

这些字段应进入 game config snapshot，而不是只存在于 UI。旧局 replay 必须使用创建时的配置快照解释，不受后续模板更新影响。

## 用户参与模式

平台需要支持“我参与一席”：

- 用户选择 1 个座位。
- 其余 11 个由 AI 扮演。
- 用户可选择默认规则/自定义规则和默认人格/自定义人格。
- 结束后生成个人评分和复盘建议。

这是留存核心，因为用户无需等待真人局，也能练发言、练抿身份、练悍跳。

## 安全策略

- 上传文件大小限制。
- YAML 使用安全解析器。
- 仅允许白名单字段。
- 所有自定义文本进入 prompt 前做长度限制与风险词检查。
- 自定义规则必须映射到 RuleEngine 支持能力，不能以自然语言覆盖裁决。
- 公开分享内容需要基础敏感词过滤。
- 所有上传配置保存原文、规范化版本、校验结果和创建者信息，便于审计。

安全要求必须进入自动化测试：

- 恶意 YAML tag。
- 超大 body。
- 未知字段。
- HTML/JS/shell/template 表达式。
- prompt 注入短语。
- 未授权上传。
- 审计字段缺失。

## 存储与配置生命周期

自定义规则和人格包使用独立配置生命周期，不直接塞进普通游戏事件：

- `raw_yaml`: 原始上传内容。
- `normalized`: 白名单规范化结果。
- `validation_result`: 错误、警告、摘要。
- `content_hash`: 去重和版本定位。
- `status`: `playable`、`display_only`、`invalid`、`deprecated`。
- `creator_id`: 本地开发可为 `mod1`，后续接真实账号。
- `created_at` / `updated_at`。
- `version`
- `maturity`: `stable`、`beta`、`experimental`、`display_only`。
- `compatibility_matrix`
- `diff_against_default`

游戏创建时保存引用：

- `ruleset_id`
- `profile_pack_id`
- `experience_mode`
- `human_seat`
- `share_code`
- locked config snapshot

这些字段需要进入 API schema、存储快照或明确的 game metadata；不能只存在于前端内存。

## 前端维护边界

后续平台化 UI 不再继续堆叠到单个 `dashboard.html` 中。至少拆分为：

- `dashboard.html`: DOM 骨架。
- `dashboard.css`: 视觉样式。
- `dashboard.js`: API 调用、上传校验、渲染逻辑。

现有字符串存在性测试只能作为基础护栏，上传、校验、创建房间、模板下载需要交互级测试和浏览器验证。

## 未来平台路线图

以下能力不进入第一版平台化切片，但应作为后续演进方向保留在产品和架构计划中。

### 房间体验预设

用户不应被迫理解所有规则和人格字段。平台应提供可直接选择的体验预设：

- 新手教学局。
- 标准竞技局。
- 高压悍跳局。
- 狼队优势局。
- 好人优势局。
- 综艺效果局。
- 复盘训练局。

每个预设绑定：

- 规则模板。
- 人格包。
- AI 强度。
- 解说模式。
- 对局节奏。
- 成本预算。

### AI 玩家强度模型

人格描述“怎么说话”，强度描述“会不会玩”。强度需要与人格分离，形成独立配置：

- `novice`: 少量逻辑，容易漏视角。
- `normal`: 基本票型和身份判断。
- `advanced`: 会盘坑、听爆点、做身份。
- `competitive`: 会悍跳、倒钩、冲锋、抗推。
- `stress_test`: 专门测试规则、提示词和信息泄漏边界。

同一人格包可以在不同强度下运行，方便用户选择轻松局、训练局或高压局。

### 玩家训练模式

`我参与一席` 后续应扩展为训练产品：

- 练预言家起跳。
- 练女巫用药。
- 练狼人悍跳。
- 练平民表水。
- 练猎人拍身份。
- 练归票发言。

每局结束给出：

- 个人评分。
- 关键发言点评。
- 投票质量。
- 身份隐藏/识别能力。
- 下一局训练建议。

### 多视角复盘

同一局应支持多种复盘视角：

- 公开视角。
- 某玩家视角。
- 狼队视角。
- 法官视角。
- 上帝视角。
- AI 内心策略视角。

公开分享默认只能使用公开视角。私有视角和 AI 内心策略视角只能在权限允许时查看。

### 关键节点时间轴

复盘页应自动标记关键节点：

- 首次跳预言家。
- 首次对跳。
- 女巫用药。
- 狼队冲票。
- 归票位形成。
- 票型分裂。
- 胜负转折点。

这些节点用于提升观战可读性，也可作为训练反馈和分享摘要素材。

### 对局质量评分

每局结束生成质量评分：

- 规则完整性。
- 发言质量。
- 票型质量。
- 狼队协作。
- 好人推理。
- 角色技能价值。
- 信息泄漏风险。
- 戏剧性指数。

质量评分用于筛选值得分享的局，也用于规则模板和人格包的健康分。

### AI 裁判与解说分层

裁判和解说必须分离：

```text
Judge = 流程控制、规则执行、信息边界，不创造规则，不泄漏私有信息。
Commentator = 只基于允许视角解释局势，提高观赏性。
```

解说不能直接访问 moderator_full 或未授权私有信息。不同解说模式：

- 新手解说：解释规则和行为含义。
- 高端复盘：身份逻辑、票型、狼坑。
- 综艺解说：更有戏剧性。
- 静默模式：只看日志。

### 配置沙盒验证

用户上传规则或人格包后，平台应能自动跑若干局模拟，生成沙盒验证报告：

- 是否能正常结束。
- 是否出现非法动作。
- 是否胜率极端。
- 是否有信息泄漏。
- 平均局长是否过长。
- 成本是否超预算。

通过沙盒验证的配置才推荐公开分享或进入市场。

### 模板健康分

规则模板和人格包应拥有健康分：

- 可玩性。
- 平衡性。
- 结束率。
- 平均局长。
- 非法动作率。
- 信息泄漏风险。
- 用户收藏/复用。

健康分用于市场排序、推荐和审核。

### Agent 行为回归测试集

为每种角色建立固定场景，防止 prompt、模型或人格包更新导致 AI 水平倒退：

- 预言家被对跳时怎么说。
- 女巫首夜吃刀能否自救。
- 狼人队友被查杀时怎么打。
- 平民被抗推时怎么表水。
- 猎人被毒死是否不开枪。

每次修改模型路由、prompt、人设模板或策略选择器，都应运行这些场景。

### Prompt 与模型 A/B 实验系统

基于现有 evaluation 能力，后续支持：

- 不同模型对比。
- 不同人格包对比。
- 不同 RAG 策略对比。
- 不同解说风格对比。
- 不同规则板子平衡性对比。

指标包括胜率、非法动作率、局长、信息泄漏、成本、复盘质量。

### 运营后台

平台化后需要运营后台：

- 热门规则模板。
- 热门人格包。
- 对局数量。
- 分享转化。
- 用户占座局数量。
- 异常局监控。
- 上传内容审核。
- 成本监控。

### 内容审核与版权边界

规则模板、人设、发言和复盘都可能包含用户内容。需要设计：

- 敏感词过滤。
- 举报。
- 下架。
- 作者署名。
- 版权声明。
- 模板 fork 来源。

### AI 玩家记忆成长

AI 玩家可在未来形成长期风格和成长：

- 记住自己过去常被抗推。
- 学习降低冲动归票。
- 狼人逐渐学会倒钩和深水。

默认公开房间应使用固定快照，保证公平性和可解释性。成长型 AI 应作为单独模式或实验模式。

### 成本预算模式

创建房间时支持成本/质量选择：

- 低成本快速局。
- 标准质量局。
- 高质量复盘局。
- 极致推理局。

背后绑定模型、token budget、RAG 深度、复盘生成深度和解说详细程度。

## 推荐实施优先级

1. RulesetRegistry 和 RuleEngine capability 边界。
2. 规则模板下载、上传、校验。
3. 玩家人格模板下载、上传、校验与 PersonaPackAdapter。
4. 开局向导：规则 + 玩家人格 + 模式。
5. 默认 12 人人格包与人设预览发言。
6. 规则市场与人格包市场静态版。
7. 规则兼容性矩阵与模板差异对比。
8. 配置版本锁定。
9. 信息泄漏审计报告。
10. 复盘分享页。
11. 用户占座参与模式。
12. 观战 AI 解说。
13. 邀请、分享、转化统计。
14. 社区投稿和审核。
15. RoleAbilityRegistry 与狼王守卫真实玩法扩展。
16. 配置沙盒验证。
17. AI 玩家强度模型。
18. 玩家训练模式。
19. 多视角复盘。
20. Agent 行为回归测试集。
