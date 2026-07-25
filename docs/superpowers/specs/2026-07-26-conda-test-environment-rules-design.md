# Conda 测试环境项目规则设计

## 目标

将仓库根目录 `environment.yml` 定义的 `wofkill` Conda 环境设为本项目唯一默认的 Python 开发与测试环境，并通过 Claude 与 Codex 各自原生读取的项目规则文件持续约束命令执行。

## 规则入口

- 新增根目录 `AGENTS.md`，供 Codex 自动读取。
- 更新根目录 `CLAUDE.md`，供 Claude Code 自动读取。
- 两个文件保留相同的核心环境约束，避免任一工具依赖二级文档才能发现规则。

## 核心约束

1. Python、pytest、ruff、mypy 和项目 Python 脚本必须通过 `conda run -n wofkill` 执行。
2. 测试命令统一采用以下形式：

   ```bash
   conda run -n wofkill python -m pytest ...
   ```

3. 不得直接调用当前 shell 的 `python`、`pytest`、`ruff` 或 `mypy`，除非任务明确要求验证其他解释器或环境。
4. 环境不存在时执行：

   ```bash
   conda env create -f environment.yml
   ```

5. `environment.yml` 发生变化或环境需要同步时执行：

   ```bash
   conda env update -n wofkill -f environment.yml --prune
   ```

6. 不依赖 `conda activate` 的跨命令状态；每条命令显式指定环境，以适配 agent 的独立 shell 进程。

## 文件变更

### `AGENTS.md`

创建简洁的项目级环境规则，明确默认环境、标准命令、禁止的裸命令以及环境创建和同步方式。

### `CLAUDE.md`

保留现有环境创建和同步说明，将依赖 `conda activate wofkill` 的执行方式改成显式 `conda run -n wofkill`，并补充与 `AGENTS.md` 一致的强制约束。

## 验证

实施后执行以下检查：

```bash
conda env list
conda run -n wofkill python --version
conda run -n wofkill python -m pytest --version
```

同时检查：

- `AGENTS.md` 与 `CLAUDE.md` 均包含相同的核心命令和禁止项。
- 文档中不再把 `conda activate wofkill` 作为 agent 执行测试的默认前提。
- Git 暂存区只包含本任务文件，不包含 `.superpowers/brainstorm` 的既有删除。

## 边界

- 不修改 `environment.yml` 的依赖集合。
- 不配置目录进入时自动激活环境。
- 不新增测试包装脚本。
- 不提交或恢复当前工作区中与本任务无关的删除和 `.DS_Store` 文件。
