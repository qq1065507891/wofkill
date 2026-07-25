# Conda Test Environment Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `wofkill` Conda environment the explicit default for all Python development and test commands run by Claude and Codex in this repository.

**Architecture:** Use each agent's native repository instruction file: create `AGENTS.md` for Codex and update `CLAUDE.md` for Claude. Both files carry the same command-level invariant, so every independent shell command names the Conda environment without relying on `conda activate` state.

**Tech Stack:** Markdown project rules, Conda, Python 3.12, pytest, ruff, mypy

---

### Task 1: Add the Codex project rule

**Files:**
- Create: `AGENTS.md`

- [ ] **Step 1: Verify the Codex rule is currently absent**

Run:

```bash
test -f AGENTS.md && rg -n "conda run -n wofkill" AGENTS.md
```

Expected: non-zero exit because the repository does not yet contain `AGENTS.md`.

- [ ] **Step 2: Create the minimal Codex rule**

Create `AGENTS.md` with exactly this environment section:

```markdown
# Project Agent Rules

## Default Conda Environment

The repository root `environment.yml` defines the required `wofkill` environment. Treat it as the only default environment for Python development and testing in this project.

- Run Python, pytest, ruff, mypy, and project Python scripts through `conda run -n wofkill`.
- Use `conda run -n wofkill python -m pytest ...` for tests.
- Use `conda run -n wofkill python -m ruff ...` and `conda run -n wofkill python -m mypy ...` for static checks.
- Do not call bare `python`, `pytest`, `ruff`, or `mypy` for project work unless the task explicitly requires another interpreter or environment.
- If the environment is missing, run `conda env create -f environment.yml` from the repository root.
- If `environment.yml` changes or the environment must be synchronized, run `conda env update -n wofkill -f environment.yml --prune`.
- Do not rely on `conda activate` state across commands; each agent shell command must name the environment explicitly.
```

- [ ] **Step 3: Verify the Codex rule contains every invariant**

Run:

```bash
rg -n "conda run -n wofkill|Do not call bare|conda env create|conda env update|Do not rely on `conda activate`" AGENTS.md
```

Expected: matches for the explicit command prefix, bare-command prohibition, environment creation, environment update, and activation-state prohibition.

- [ ] **Step 4: Commit the Codex rule**

```bash
git add AGENTS.md
git diff --cached --check
git commit -m "docs: set Codex conda test environment"
```

Expected: one commit containing only `AGENTS.md`.

### Task 2: Align the Claude project rule

**Files:**
- Modify: `CLAUDE.md:16-24`

- [ ] **Step 1: Verify the Claude rule still relies on activation**

Run:

```bash
rg -n "conda activate wofkill" CLAUDE.md
```

Expected: one match in the shared Conda environment section.

- [ ] **Step 2: Replace the existing Conda section with the explicit command rule**

Replace the current block beginning with `Use the shared Conda environment` and ending with the update command sentence with:

````markdown
Use the repository root `environment.yml` as the only default environment for Python development and testing. Agent commands must name the environment explicitly because shell state is not guaranteed to persist between commands.

```bash
conda run -n wofkill python -m pytest ...
conda run -n wofkill python -m ruff ...
conda run -n wofkill python -m mypy ...
```

- Run Python, pytest, ruff, mypy, and project Python scripts through `conda run -n wofkill`.
- Do not call bare `python`, `pytest`, `ruff`, or `mypy` for project work unless the task explicitly requires another interpreter or environment.
- If the environment is missing, run `conda env create -f environment.yml` from the repository root.
- If `environment.yml` changes or the environment must be synchronized, run `conda env update -n wofkill -f environment.yml --prune`.
- Do not rely on `conda activate` state across commands.
````

- [ ] **Step 3: Verify Claude and Codex expose the same core commands**

Run:

```bash
rg -n "conda run -n wofkill python -m pytest|conda env create -f environment.yml|conda env update -n wofkill -f environment.yml --prune|Do not call bare" AGENTS.md CLAUDE.md
```

Expected: every invariant appears in both files.

Run:

```bash
if rg -n "conda activate wofkill" AGENTS.md CLAUDE.md; then exit 1; fi
```

Expected: exit 0 with no matches.

- [ ] **Step 4: Commit the Claude rule**

```bash
git add CLAUDE.md
git diff --cached --check
git commit -m "docs: set Claude conda test environment"
```

Expected: one commit containing only `CLAUDE.md`.

### Task 3: Provision and verify the default environment

**Files:**
- Verify: `environment.yml`
- Verify: `AGENTS.md`
- Verify: `CLAUDE.md`

- [ ] **Step 1: Detect whether the `wofkill` environment exists**

Run:

```bash
conda env list
```

Expected: an environment named `wofkill` is listed. If it is absent, continue with Step 2; if present, continue with Step 3.

- [ ] **Step 2: Create the missing environment**

Run only when Step 1 does not list `wofkill`:

```bash
conda env create -f environment.yml
```

Expected: Conda creates an environment named `wofkill` without dependency-resolution errors.

- [ ] **Step 3: Synchronize an existing environment**

Run when Step 1 already lists `wofkill`:

```bash
conda env update -n wofkill -f environment.yml --prune
```

Expected: Conda completes successfully and retains the environment name `wofkill`.

- [ ] **Step 4: Verify interpreter and test runner resolution**

Run:

```bash
conda run -n wofkill python --version
conda run -n wofkill python -m pytest --version
```

Expected: Python reports version 3.12.x and pytest prints its installed version.

- [ ] **Step 5: Run a focused repository test through the required environment**

Run:

```bash
conda run -n wofkill python -m pytest tests/rag/test_schemas.py -q
```

Expected: all tests in `tests/rag/test_schemas.py` pass.

- [ ] **Step 6: Confirm unrelated work remains untouched**

Run:

```bash
git status --short
git diff --cached --name-only
```

Expected: `.superpowers/brainstorm` deletions and `.DS_Store` files remain unstaged; no unrelated file is included in either rules commit.
