# Werewolf Agent

Werewolf Agent 是一个面向 12 人「预女猎白混」狼人杀规则的多智能体实验项目。项目目标是把游戏裁判、玩家智能体、认知状态、RAG 策略参考、长期记忆和评测回放拆成可测试、可复现的模块。

当前规则权威来自：

- `docs/design/werewolf-agent-v1-design.md`
- `config/rulesets/pre_witch_hunter_idiot_mixed.yaml`

外部资料、RAG 案例和玩家发言只能作为策略参考，不能覆盖 RuleEngine 的确定性裁判结果。

## 功能概览

- 确定性 `RuleEngine`：角色分配、夜间结算、白天投票、警长、白痴翻牌、猎人开枪、胜负判断。
- LangGraph 运行时：编排夜晚、白天、发言、投票、自爆、胜负检查和 checkpoint。
- `GameRunner` 对局编排器：串联 LangGraph + RuleEngine + 持久化，支持全量运行和逐步推进。
- 玩家/法官智能体接口：结构化行动输出、非法行动重试、安全 fallback、公开广播。
- 认知流水线：可见事实、注意力、显著性、信念更新、矛盾检测和策略选择。
- RAG 边界：策略案例可注入，基础规则真值和裁判结论禁止进入 RAG。支持 SiliconFlow 语义嵌入 + Reranker。
- 记忆系统：认知矩阵、关系图、玩家画像、复盘报告和跨局经验。自动持久化到 SQLite。
- API 视图：public、player_view、moderator_full 分层权限与审计。HMAC session 认证。
- 评测实验：批量对局、指标聚合（含语音影响力、认知压缩率等）、replay 验证和 leaderboard 报告。
- 观战台：增强版 dashboard，含认知差异图、RAG 命中面板、模型/人格路由面板、成本延迟追踪。

## 目录结构

```text
werewolf_agent/
  agents/            # 玩家与法官智能体接口
  api/               # FastAPI 视图、权限和响应模型
  cognition/         # 认知流水线
  core/              # 核心数据模型
  engine/            # 确定性规则引擎
  evaluation/        # 批量评测、指标和报告
  memory/            # 游戏内外记忆
  model_gateway/     # 模型路由和调用抽象
  rag/               # 案例摄取、检索和注入
  runtime/           # LangGraph 游戏流程
  skills/            # 策略技能注册与执行
  storage/           # 持久化：Repository 接口、内存和 SQLite 实现
  ui/                # Observer Dashboard 静态文件

config/              # 规则集、模型和 persona 配置
docs/                # 设计文档和开发说明
harness/             # 开发控制台账、计划和检查清单
tests/               # pytest 测试套件
```

## 环境准备

项目使用 Conda 环境，配置文件在根目录：

```powershell
conda env create -f environment.yml
conda activate wofkill
```

如果环境已经存在，可以更新：

```powershell
conda env update -f environment.yml --prune
```

## 运行测试

```powershell
python -m pytest -q
```

在当前开发环境中也可以直接使用：

```powershell
D:/Miniforge3/envs/wofkill/python.exe -m pytest -q
```

## Observer Dashboard

启动 API 后访问浏览器查看 Dashboard：

```powershell
uvicorn werewolf_agent.api.app:create_app --factory --reload
# 浏览器打开 http://localhost:8000/
```

Dashboard 功能：

- 左侧边栏：游戏列表、创建新游戏
- 中间面板：当前阶段、12 人玩家状态、死亡记录、投票记录、事件时间线
- 右侧面板（Moderator/Debug 模式）：私有审计（角色信息）、认知差异可视化、模型路由信息、RAG/记忆审计
- 顶部切换 Public / Player View / Moderator / Debug 视角
- Pause/Resume 控制按钮

权限：Public 模式不显示任何私密信息；Moderator/Debug 模式需要 `caller_id=mod1` 或 `dbg1` 并设置对应角色。

```powershell
# Dashboard 测试
python -m pytest tests/ui/test_dashboard.py -q
```

## Docker 快速启动

```bash
# 构建并启动 API 服务
docker compose up

# 访问观战台
# http://localhost:8000

# 包含 Redis 的完整启动
docker compose --profile with-redis up

# 可选生产依赖占位服务（适配器仍需后续实现）
docker compose --profile production-adapters up postgres qdrant redis
```

## V1.1 硬化边界

V1.1 增加了本地运行硬化和生产适配边界，但默认开发路径仍保持轻量：

- Runtime：`LocalRuntimeExecutor` 提供单进程 per-game lock、后台运行状态和 paused-game step 拦截；多进程部署仍需 Redis/task-state。
- Timer：`runtime.timers` 提供 `ManualTimer` / `NoopTimer`，用于狼队讨论和白天发言的流程超时测试；真实 provider 调用取消仍是后续工作。
- MCP：`TransportMCPProvider` 可以包装真实 transport，但 MCP 结果仍统一标注为 suggestion-only。
- RAG：`create_vector_store()` 支持 `auto`、`local`、`embedding`、`siliconflow`，并对 `qdrant` / `pgvector` 给出显式配置错误。
- Storage：`ProductionStorageConfig` / `create_game_repository()` 默认返回 SQLite；PostgreSQL/Redis 生产适配器尚未实现，会显式报错而不是静默 fallback。

## 持久化模式

API 默认使用内存存储（进程退出后数据丢失）。可通过 `repository` 参数启用持久化：

```python
from werewolf_agent.api.app import create_app
from werewolf_agent.storage.sqlite_store import SqliteGameRepository

# 内存模式（默认）
app = create_app()

# SQLite 持久化模式
repo = SqliteGameRepository("data/games.db")
app = create_app(repository=repo)
```

`GameRepository` 协议定义了统一接口：`save_game`、`load_game`、`append_events`、`load_events`、`save_deaths`、`save_model_usage`、`save_evaluation`、`save_config_snapshot` 等。`InMemoryGameRepository` 用于测试，`SqliteGameRepository` 用于本地持久化开发。设计文档 §13 允许 V1 先用 SQLite 或本地文件做开发替代，接口边界按准生产设计。

```powershell
# 持久化测试
python -m pytest tests/storage/test_storage.py -q
```

## 开发约束

- RuleEngine 是裁判真值来源；LLM、RAG、外部案例不能修改规则判断。
- 玩家视角不能看到 `moderator_full`、其他玩家私有状态、隐藏身份或禁用夜间信息。
- 狼队可以主动空刀；狼队夜间讨论超时且未形成合法操作时默认空刀。
- 只有 `wolf_kill_selected` 会向女巫暴露当前夜刀口；主动空刀和超时空刀不会产生可救刀口。
- 白天每名玩家发言有时间限制，超时记录 `speech_timeout` 并推进发言队列。
- 评测 replay 必须使用记录中的 `ruleset_snapshot` 复原规则。

## 评测报告导出

评测系统支持 JSON 报告导出，可供 Observer Dashboard 或外部工具消费：

```python
from werewolf_agent.evaluation.metrics import MetricsAggregator
from werewolf_agent.evaluation.reports import ReportGenerator
from werewolf_agent.evaluation.schemas import BatchConfig

# 运行批量对局
config = BatchConfig(batch_id="my_experiment", num_games=10)
# ... run games and collect results ...

# 计算指标并导出
agg = MetricsAggregator(config)
agg.add_results(results)
snap = agg.compute_snapshot()

gen = ReportGenerator()
gen.add_snapshot(snap)
full_report = gen.export_full_report(snap)

# full_report 是 JSON-serializable dict
import json
print(json.dumps(full_report, indent=2, ensure_ascii=False))
```

导出内容包括：
- `metrics`: 完整指标快照（阵营胜率、玩家/角色指标、质量指标、安全指标、成本指标、成长曲线、溯源信息）
- `leaderboard`: 排行榜（含 overall_score、win_rate、stance_accuracy 等）
- `provenance`: 每个指标的溯源信息（计算方法、数据来源、贡献游戏数）

## 常用命令

```powershell
# 全量测试
python -m pytest -q

# 规则引擎测试
python -m pytest tests/rules/test_rule_engine_v1.py -q

# API 权限测试
python -m pytest tests/api/test_api.py -q

# 运行本地 API 示例
uvicorn werewolf_agent.api.app:create_app --factory --reload
```

## 模型 Provider 与 API Key

真实模型调用通过 `ModelRouter` 统一接入。`config/models.yaml` 只保存 provider、model、玩家和法官的路由配置，不保存密钥。

本地复制 `.env.example` 为 `.env`，填写需要使用的 provider：

```env
ANTHROPIC_API_KEY=你的 Anthropic key
GLM_API_KEY=你的 GLM / 智谱 key
OPENAI_API_KEY=你的 OpenAI key
```

可选 base URL：

```env
ANTHROPIC_BASE_URL=https://api.anthropic.com
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
OPENAI_BASE_URL=https://api.openai.com
```

在代码里启用真实 provider：

```python
from werewolf_agent.model_gateway.router import ModelRouter

router = ModelRouter.from_yaml(
    "config/models.yaml",
    register_env_providers=True,
)
```

玩家模型在 `config/models.yaml` 的 `players` 中配置：

```yaml
players:
  p01:
    persona_id: logic_leader
    llm_profile: pro_reasoner
  p02:
    persona_id: aggressive_bluffer
    llm_profile: local_wolf
```

法官也使用同一个模型路由，当前已配置为虚拟 agent：

```yaml
players:
  judge:
    persona_id: judge
    llm_profile: pro_reasoner
```

使用法官时传入同一个 router：

```python
from werewolf_agent.agents.judge import JudgeAgent

judge = JudgeAgent(model_router=router)
```

当前内置 provider：

- `anthropic`: Anthropic Messages API，读取 `ANTHROPIC_API_KEY`
- `glm`: OpenAI-compatible Chat Completions，读取 `GLM_API_KEY`
- `openai`: OpenAI Chat Completions，读取 `OPENAI_API_KEY`

### 真实 Provider 冒烟测试

默认测试套件不会调用真实模型。需要主动设置环境变量后，才会运行真实 provider smoke test：

```powershell
$env:WEREWOLF_RUN_REAL_LLM_SMOKE = "1"
$env:OPENAI_API_KEY = "..."
# 或 ANTHROPIC_API_KEY / GLM_API_KEY
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/integration/test_real_llm_smoke.py -q --basetemp .pytest-tmp
```

该测试只验证真实 provider 能通过 `ModelRouter` 和 `PlayerAgent` 的 schema/fallback 边界，不作为完整 12 人真实 LLM 对局验收。

## Git

本仓库忽略 Python 缓存、pytest 缓存、虚拟环境、本地 `.env` 和覆盖率输出。不要提交本地密钥、供应商 API key 或机器相关路径。
