# Werewolf Agent

Werewolf Agent 是一个面向 12 人「预女猎白混」狼人杀规则的多智能体实验项目。项目目标是把游戏裁判、玩家智能体、认知状态、RAG 策略参考、长期记忆和评测回放拆成可测试、可复现的模块。

当前规则权威来自：

- `docs/design/werewolf-agent-v1-design.md`
- `config/rulesets/pre_witch_hunter_idiot_mixed.yaml`

外部资料、RAG 案例和玩家发言只能作为策略参考，不能覆盖 RuleEngine 的确定性裁判结果。

## 功能概览

- 确定性 `RuleEngine`：角色分配、夜间结算、白天投票、警长、白痴翻牌、猎人开枪、胜负判断。
- LangGraph 运行时：编排夜晚、白天、发言、投票、自爆、胜负检查和 checkpoint。
- 玩家/法官智能体接口：结构化行动输出、非法行动重试、安全 fallback、公开广播。
- 认知流水线：可见事实、注意力、显著性、信念更新、矛盾检测和策略选择。
- RAG 边界：策略案例可注入，基础规则真值和裁判结论禁止进入 RAG。
- 记忆系统：认知矩阵、关系图、玩家画像、复盘报告和跨局经验。
- API 视图：public、player_view、moderator_full 分层权限与审计。
- 评测实验：批量对局、指标聚合、replay 验证和 leaderboard 报告。

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

## 开发约束

- RuleEngine 是裁判真值来源；LLM、RAG、外部案例不能修改规则判断。
- 玩家视角不能看到 `moderator_full`、其他玩家私有状态、隐藏身份或禁用夜间信息。
- 狼队可以主动空刀；狼队夜间讨论超时且未形成合法操作时默认空刀。
- 只有 `wolf_kill_selected` 会向女巫暴露当前夜刀口；主动空刀和超时空刀不会产生可救刀口。
- 白天每名玩家发言有时间限制，超时记录 `speech_timeout` 并推进发言队列。
- 评测 replay 必须使用记录中的 `ruleset_snapshot` 复原规则。

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

## Git

本仓库忽略 Python 缓存、pytest 缓存、虚拟环境、本地 `.env` 和覆盖率输出。不要提交本地密钥、供应商 API key 或机器相关路径。
