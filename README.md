# Werewolf Agent

Werewolf Agent 是一个面向 12 人「预女猎白混」规则的多智能体狼人杀实验项目。项目将确定性裁判、玩家与法官智能体、认知状态、RAG、跨局记忆、评测审计和 Observer Dashboard 组织为可测试、可复现的模块。

项目的首要边界是：**规则真值只来自 `RuleEngine` 与当前规则集**。LLM 输出、玩家发言、RAG 案例和长期记忆只能影响策略，不能覆盖角色能力、结算顺序、投票结果或胜负判定。

当前规则依据：

- [V1 设计文档](docs/design/werewolf-agent-v1-design.md)
- [预女猎白混规则集](config/rulesets/pre_witch_hunter_idiot_mixed.yaml)

## 核心能力

- **确定性裁判**：角色分配、夜间结算、警长竞选、白天投票、平票复投、白痴翻牌、猎人开枪和胜负判断。
- **多智能体运行时**：使用 LangGraph 编排夜晚、白天、公开发言、投票、技能、自爆、checkpoint 和终局复盘。
- **任务专用输出契约**：公开发言、讨论摘要、投票、狼人协作和赛后 Reflection 使用各自的 Schema、验证与 fallback 边界。
- **公开证据与隐私隔离**：公开声明只能引用可见事件；玩家私有状态、狼人协作内容和被拒绝的模型草稿不会进入公开事件。
- **认知与策略**：可见事实、注意力、显著性、信念更新、矛盾检测、策略技能和 suggestion-only RAG。
- **记忆与复盘**：认知矩阵、关系图、玩家画像、结构化 Reflection、匿名化和跨局经验持久化。
- **API 与观战台**：FastAPI 提供 public、player_view、moderator_full 分层视图，Dashboard 展示时间线、投票、认知、RAG、模型路由和调用成本。
- **评测与审计**：支持单局质量评分、批量平衡分析、回放、语义安全指标和结构化审计报告。

## 当前关键契约

### 投票单位与显示

规则层使用整数“半票单位”避免浮点裁判：普通玩家为 `2` 单位，警长为 `3` 单位。规则比较、平票和放逐判定始终使用整数单位；日志、事件、API 和 Dashboard 对外显示为实际 `1` 票和 `1.5` 票。

新版投票事件同时保留 `*_units` 与实际票数字段，并带 `base_vote_weight`。消费者通过统一 helper 读取，兼容旧事件且不会重复换算。

### Discussion Summary

内部讨论摘要使用独立的 `TaskType.DISCUSSION_SUMMARY` 和 `DiscussionSummary` Schema：

- 不进入公开 `SPEECH` 质量门；
- 不产生公开发言事件；
- 模型失败时使用确定性摘要；
- 私有摘要只进入当前玩家的策略上下文，并以哈希/计数形式参与暴露审计。

### Public Speech Repair

公开发言重试会累计质量门和公开证据门产生的结构化约束。后续尝试必须同时保留目标、允许的公开证据及禁止声明；被拒绝的原始文本和 correction hint 不会写入持久 trace。重试耗尽后只允许生成基于公开事实的安全 fallback。

### Reflection

赛后复盘直接生成严格的 `ReflectionDraft`，不再借用通用 `PlayerAction` 或 `SPEECH` 管线。草稿通过 Schema 校验、事实验证、匿名化和质量门后才可持久化；失败日志只保留安全错误码、字段路径和错误类型。

### 指标支持状态

依赖完整事件集的 fallback 指标会显式报告支持状态。遇到 `json_item_limit_exceeded`、部分事件导出或缺失事件时：

- 指标值为 `null`，不会伪装成 `0`；
- `fallback_metrics_supported=false`；
- `fallback_metrics_unsupported_reason` 记录原因；
- stdout 诊断与正式结构化指标保持分离。

## 目录结构

```text
werewolf_agent/
  agents/            # 玩家、法官、任务 Schema 与重试契约
  api/               # FastAPI 工厂、路由、权限和视图
  cognition/         # 认知状态、信念和矛盾检测
  core/              # 核心游戏数据模型与事件可见性
  engine/            # RuleEngine 与确定性规则结算
  evaluation/        # 质量、平衡、验收和审计指标
  memory/            # 游戏内记忆、Reflection 与跨局经验
  model_gateway/     # Provider、路由、重试和结构化输出
  rag/               # 策略案例检索和 suggestion-only 注入
  runtime/           # LangGraph 游戏流程与节点
  skills/            # 策略技能注册和执行
  storage/           # 内存、SQLite 和 PostgreSQL Repository
  ui/                # Observer Dashboard 静态资源
config/              # 规则集、模型路由和 persona 配置
docs/                # 设计、操作、审计和实施文档
scripts/             # 真实对局、审计、平衡分析和诊断脚本
tests/               # pytest 测试套件
```

## 快速开始

### 1. 创建环境

项目使用 Python 3.12。推荐通过 Conda 安装：

```powershell
conda env create -f environment.yml
conda run -n wofkill python --version
```

已有环境可以更新：

```powershell
conda env update -n wofkill -f environment.yml --prune
```

### 2. 配置环境变量

```powershell
Copy-Item .env.example .env
```

在本地 `.env` 中填写 `config/models.yaml` 当前路由所需的 Provider key。不要提交 `.env` 或任何真实凭据。

常用变量：

| 变量 | 用途 |
| --- | --- |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` | Anthropic-compatible Provider |
| `GLM_API_KEY` / `GLM_BASE_URL` | GLM/OpenAI-compatible Provider |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | OpenAI-compatible 或 Ark 路由 |
| `WEREWOLF_MODEL_CONFIG` | 模型路由配置，默认 `config/models.yaml` |
| `WEREWOLF_STORAGE_BACKEND` | `sqlite` 或 `postgres` |
| `POSTGRES_DSN` | PostgreSQL 连接串 |
| `WEREWOLF_VECTOR_BACKEND` | RAG 向量后端，如 `pgvector` |
| `SILICONFLOW_API_KEY` | 可选语义嵌入与 reranker |

Provider、模型和玩家分配以 [config/models.yaml](config/models.yaml) 为准，不应在应用代码或 README 中硬编码。

### 3. 运行测试

```powershell
conda run -n wofkill python -m pytest -q -o addopts=''
```

Windows 默认临时目录出现 `WinError 5` 时，可显式使用不被嵌套测试管理的仓库目录：

```powershell
conda run -n wofkill python -m pytest -q -o addopts='' --basetemp .pytest-tmp/full-suite
```

常用聚焦测试：

```powershell
conda run -n wofkill python -m pytest tests/rules/test_rule_engine_v1.py -q
conda run -n wofkill python -m pytest tests/api/test_api.py -q
conda run -n wofkill python -m pytest tests/ui/test_dashboard.py -q
conda run -n wofkill python -m pytest tests/scripts/test_run_real_game.py -q
```

## 启动 API 与 Dashboard

本地开发默认使用内存存储，进程退出后数据丢失：

```powershell
uvicorn werewolf_agent.api.app:create_app --factory --reload --port 8000
```

浏览器访问 <http://localhost:8000/>。

需要本地 SQLite 时：

```powershell
$env:WEREWOLF_DB_PATH = "data/wofkill.db"
uvicorn werewolf_agent.api.app:create_app --factory --reload --port 8000
```

Dashboard 支持 Public、Player View、Moderator 和 Debug 视图。开发环境预置的主持人调用方为 `mod1`，调试调用方为 `dbg1`；生产部署必须设置安全的 `WEREWOLF_AUTH_SECRET` 并按部署策略管理身份。

## Docker Compose

Docker 默认启动 API 与 PostgreSQL/pgvector，并将宿主机 `18000` 映射到容器 `8000`：

```powershell
docker compose up --build
```

访问 <http://localhost:18000/>。

启用 Redis 适配器：

```powershell
docker compose --profile with-redis up --build
```

修改宿主机端口：

```powershell
$env:WEREWOLF_API_PORT = "8000"
docker compose up --build
```

详细生产配置、健康检查、备份和存储选择见 [部署指南](docs/operations/deployment-guide.md)。`docker-compose.yml` 中的开发密码和默认鉴权 secret 不适用于生产环境。

## 运行真实 12 人对局

真实对局会调用外部模型，可能持续较长时间并消耗 Provider 额度。先确认 `.env` 与 `config/models.yaml` 匹配；当前 CLI 预检还要求至少设置 `ANTHROPIC_API_KEY` 或 `GLM_API_KEY`。

```powershell
python scripts/run_real_game.py `
  --seed 42 `
  --max-steps 500 `
  --output-dir artifacts/real-game
```

可用参数：

```powershell
python scripts/run_real_game.py --help
```

- `--game-id`：显式指定本次运行 ID；默认由 seed 生成。
- `--delay -1`：取消额外的调用间延迟；不会绕过 Provider timeout 或重试。
- `--output-dir`：隔离本局 JSON 产物，推荐真实验收时总是指定。

只有完整结束并生成正式 JSON 的新对局，才能作为生产行为改善的验收证据。进程超时、人工终止或仅有 stdout 片段时，应标记为未完成。

## 对局产物与审计

### 单局 JSON

真实对局写入：

```text
<output-dir>/game_<game-id>.json
```

高 fallback 且指标受支持的低质量对局可能写入 `<output-dir>/low_quality_games/`。JSON 包含终局状态、玩家、死亡、事件、耗时、步数、质量指标和 acceptance 支持状态。

### 运行日志

`scripts/run_real_game.py` 默认把详细运行日志写入仓库根目录 `game_stdout.log`。可以通过 `WEREWOLF_GAME_LOG_PATH` 指定其他位置：

```powershell
$env:WEREWOLF_GAME_LOG_PATH = "artifacts/real-game/game_stdout.log"
```

stdout 可用于诊断 Provider 重试和运行阶段，但不能替代缺失的结构化事件指标。

### 审计报告

```powershell
python scripts/print_game_audit.py artifacts/real-game/game_g_42.json
python scripts/print_game_audit.py artifacts/real-game/game_g_42.json `
  --output artifacts/real-game/game_g_42-audit.md
```

批量平衡分析：

```powershell
python scripts/analyze_recent_balance.py `
  artifacts/real-game/game_g_42.json `
  artifacts/real-game/game_g_43.json
```

报告消费者必须读取 `*_supported` 和 `*_unsupported_reason`，不能把 `null` 转成零。

## 存储与 RAG

`create_app()` 的存储选择顺序：

1. 设置 `WEREWOLF_STORAGE_BACKEND` 时使用生产存储工厂；
2. 仅设置 `WEREWOLF_DB_PATH` 时使用 SQLite；
3. 均未设置时使用内存存储。

Docker Compose 默认使用 PostgreSQL。Repository 接口统一覆盖游戏状态、事件、死亡、模型用量、评测和配置快照。

RAG 始终是策略建议，不是规则来源。`WEREWOLF_VECTOR_BACKEND` 未配置或外部向量服务不可用时，运行时会使用可用的本地/规则 fallback；任何 RAG 内容都不能修改裁判真值。

## 开发约束

- `RuleEngine` 与规则集是唯一裁判真值来源。
- 所有公开内容必须通过 `EventVisibility` 和公开证据边界。
- 玩家不能读取其他玩家私有状态、隐藏身份或禁用的夜间信息。
- 兼容迁移应保留现有 facade、re-export 和旧事件读取能力。
- Provider 错误、被拒绝草稿和 correction hint 不得进入公开日志或持久审计。
- 依赖事件全集的指标必须先检查支持状态。
- 不提交 `.env`、Provider key、数据库凭据、pytest 缓存或生成的游戏日志。

## 进一步阅读

- [V1 系统设计](docs/design/werewolf-agent-v1-design.md)
- [生产部署指南](docs/operations/deployment-guide.md)
- [规则集配置](config/rulesets/pre_witch_hunter_idiot_mixed.yaml)
- [模型路由配置](config/models.yaml)
- [环境变量示例](.env.example)
