# README Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 更新项目根目录 README，使其准确描述当前安装、运行、测试、输出和安全契约。

**Architecture:** 采用单文件增量重写，保留仍然有效的项目概览和部署信息，删除重复与过时段落。所有命令、路径和环境变量均从当前仓库入口与配置验证，README 不承担完整架构手册职责。

**Tech Stack:** Markdown、PowerShell、Python 3.12、Conda、Docker Compose、pytest、FastAPI/Uvicorn

---

### Task 1: 核对当前文档事实

**Files:**
- Read: `README.md`
- Read: `.env.example`
- Read: `environment.yml`
- Read: `docker-compose.yml`
- Read: `config/models.yaml`
- Read: `scripts/run_real_game.py`
- Read: `scripts/print_game_audit.py`
- Read: `werewolf_agent/api/app.py`

- [ ] **Step 1: 核对命令入口**

Run: `python scripts/run_real_game.py --help`

Expected: 显示 `--seed`、`--game-id`、`--max-steps`、`--delay` 和 `--output-dir`。

- [ ] **Step 2: 核对 API 与 Docker 入口**

Run: `python -c "from werewolf_agent.api.app import create_app; assert callable(create_app)"`

Expected: 退出码为 0；`docker-compose.yml` 映射默认主机端口 `18000`。

- [ ] **Step 3: 核对 README 引用目标**

确认规则设计、规则集、部署指南、dotenv 示例和审计脚本路径存在，不引用未跟踪的本地脚本。

### Task 2: 增量重写 README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 重写项目概览与关键契约**

保留 RuleEngine 作为唯一裁判真值来源，增加当前票权显示、discussion summary、speech repair、Reflection 和事件截断指标支持状态说明。

- [ ] **Step 2: 整理快速开始**

依次给出 Conda 安装、`.env.example` 配置、自动化测试、API Dashboard、Docker Compose 和真实对局命令。

- [ ] **Step 3: 补充输出与诊断路径**

说明 `--output-dir`、游戏 JSON、`game_stdout.log`、`print_game_audit.py` 和 balance report 的用途；明确未完成或截断事件不能报告为零 fallback。

- [ ] **Step 4: 删除重复与过时内容**

合并重复测试/API 命令，删除固定模型名称、过时 provider 分配和无法由当前配置保证的生产结论。

### Task 3: 验证文档

**Files:**
- Verify: `README.md`

- [ ] **Step 1: 验证本地 Markdown 链接**

使用 PowerShell 提取 README 中的相对 Markdown 链接，逐一确认目标存在。

Expected: 不存在缺失的本地链接。

- [ ] **Step 2: 验证关键命令**

Run: `python scripts/run_real_game.py --help`

Run: `python -c "from werewolf_agent.api.app import create_app; assert callable(create_app)"`

Expected: 两条命令退出码均为 0。

- [ ] **Step 3: 运行文档相关回归**

Run: `python -m pytest tests/ui/test_dashboard.py tests/scripts/test_run_real_game.py -q -o addopts='' --basetemp E:\NLP\agent\wofkill\.tmp\pytest-readme-final`

Expected: 全部测试通过。

- [ ] **Step 4: 检查差异并提交**

Run: `git diff --check`

Expected: 无空白错误。

```powershell
git add README.md
git commit -m "docs: refresh project README"
```

