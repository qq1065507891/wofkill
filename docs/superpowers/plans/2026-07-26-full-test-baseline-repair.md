# Full Test Baseline Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the complete serial pytest suite pass on macOS through the project-level `wofkill` Conda environment, with PowerShell validation remaining an optional platform check.

**Architecture:** Close the declared Conda dependency set first, then update only the stale tests that no longer match current framework or application contracts. Keep production behavior unchanged, make platform-tool discovery explicit, and use pytest's system-managed temporary directories for the final complete-suite run.

**Tech Stack:** Conda, Python 3.12, pytest, pytest-xdist, NumPy, FastAPI, Pydantic

---

### Task 1: Close the Conda test dependency set

**Files:**
- Modify: `environment.yml:1-22`

- [ ] **Step 1: Verify the declared environment is missing both dependencies**

Run:

```bash
rg -n "numpy|pytest-xdist" environment.yml
conda run -n wofkill python -c "import importlib.util as u; print('numpy', u.find_spec('numpy')); print('xdist', u.find_spec('xdist'))"
```

Expected: `rg` has no matches and both module specifications print `None`.

- [ ] **Step 2: Add the dependencies beside the existing pytest tools**

Change the Conda dependency block to include:

```yaml
  - numpy
  - pytest
  - pytest-cov
  - pytest-xdist
```

Do not move Python packages between Conda and pip sections and do not add PowerShell.

- [ ] **Step 3: Synchronize the existing project environment**

Run:

```bash
conda env update -n wofkill -f environment.yml --prune
```

Expected: Conda completes without dependency-resolution errors and retains the environment name `wofkill`.

- [ ] **Step 4: Verify the newly declared modules and the NumPy-backed behavior**

Run:

```bash
conda run -n wofkill python -c "import numpy, xdist; print(numpy.__version__); print(xdist.__version__)"
conda run -n wofkill python -m pytest tests/rag/test_rag_hardening.py::TestAutoVectorStore::test_auto_selects_embedding_when_numpy_available -q -o addopts=''
```

Expected: both packages print versions and the focused test passes.

- [ ] **Step 5: Commit the dependency closure**

```bash
git add environment.yml
git diff --cached --check
git commit -m "build: complete conda test dependencies"
```

Expected: one commit containing only `environment.yml`.

### Task 2: Make FastAPI route-surface verification public-API based

**Files:**
- Modify: `tests/api/test_game_persistence_helpers.py:1-85`

- [ ] **Step 1: Run the stale route-surface test and retain the failure**

Run:

```bash
conda run -n wofkill python -m pytest tests/api/test_game_persistence_helpers.py::test_game_router_keeps_route_registration_surface -q -o addopts=''
```

Expected: FAIL because FastAPI 0.140 `_IncludedRouter` has no `path` attribute.

- [ ] **Step 2: Update the module date and inspect the OpenAPI route surface**

Add this line to the existing Chinese module docstring without changing its functional description:

```python
修改日期: 2026-07-26
```

Replace the `app.routes` comprehension with:

```python
    route_methods = {"get", "post", "put", "patch", "delete", "options", "head"}
    routes = {
        (method.upper(), path)
        for path, operations in app.openapi()["paths"].items()
        for method in operations
        if method in route_methods
        and (path == "/" or path.startswith(("/auth", "/games")))
    }
```

This verifies the public HTTP contract and does not depend on FastAPI's internal router wrapper classes.

- [ ] **Step 3: Run the focused API module**

Run:

```bash
conda run -n wofkill python -m pytest tests/api/test_game_persistence_helpers.py -q -o addopts=''
```

Expected: the module passes, including the exact expected route set.

- [ ] **Step 4: Commit the FastAPI compatibility test**

```bash
git add tests/api/test_game_persistence_helpers.py
git diff --cached --check
git commit -m "test: use public FastAPI route surface"
```

Expected: one commit containing only the API test module.

### Task 3: Repair the audit regression-node mapping

**Files:**
- Modify: `tests/regression/test_post_july14_contract_cases.py:1-70`

- [ ] **Step 1: Reproduce the missing audit node**

Run:

```bash
conda run -n wofkill python -m pytest tests/regression/test_post_july14_contract_cases.py::test_every_audit_issue_maps_to_an_existing_pytest_node -q -o addopts=''
```

Expected: FAIL naming `test_runtime_no_kill_event_has_complete_v2_audit_identity` as missing.

- [ ] **Step 2: Point N7 at the existing provider-failure contract**

Update the module docstring date:

```python
修改日期: 2026-07-26
```

Replace only the N7 function-name fragment with:

```python
        "test_provider_failure_no_kill_event_has_complete_v2_audit_identity",
```

- [ ] **Step 3: Verify both mapping and nested closure execution**

Run:

```bash
conda run -n wofkill python -m pytest tests/regression/test_post_july14_contract_cases.py::test_every_audit_issue_maps_to_an_existing_pytest_node tests/regression/test_post_july14_contract_cases.py::test_mapped_audit_nodes_execute_as_nonrecursive_closure_batch -q -o addopts=''
```

Expected: both tests pass. The nested command recognizes `-n0` because Task 1 installed `pytest-xdist`, produces JUnit, and executes every mapped node.

- [ ] **Step 4: Commit the audit mapping repair**

```bash
git add tests/regression/test_post_july14_contract_cases.py
git diff --cached --check
git commit -m "test: repair audit closure node mapping"
```

Expected: one commit containing only the regression test module.

### Task 4: Align reflection fakes with the structured draft API

**Files:**
- Modify: `tests/runtime/test_strategy_directives.py:1-10`
- Modify: `tests/runtime/test_strategy_directives.py:3580-3700`

- [ ] **Step 1: Reproduce both obsolete-agent failures**

Run:

```bash
conda run -n wofkill python -m pytest tests/runtime/test_strategy_directives.py::TestReflectionRoleSpecific::test_agent_reflection_uses_REFLECTION_task_type tests/runtime/test_strategy_directives.py::TestReflectionRoleSpecific::test_agent_reflection_rejects_unstructured_draft_without_exposure -q -o addopts=''
```

Expected: both tests fail with `agent_error` because their fake agents have `act(...)` but no `generate_reflection(...)`.

- [ ] **Step 2: Update the module date and import the stable reflection error**

Change the existing docstring date to:

```python
修改日期: 2026-07-26
```

Inside each affected test, import:

```python
from werewolf_agent.agents.player import ReflectionDraftGenerationError
```

- [ ] **Step 3: Replace the obsolete fake action API in the task-type test**

Replace its `FakeAgent` class with:

```python
        class FakeAgent:
            def generate_reflection(self, context, prompt):
                raise ReflectionDraftGenerationError("invalid_structured_draft")
```

Keep the existing assertions that `build_agent_context` received `TaskType.REFLECTION`, the verification status is `invalid_structured_draft`, and no `reflection_text` is exposed.

- [ ] **Step 4: Replace the obsolete free-text fake with the structured failure contract**

Rename the second test to:

```python
    def test_agent_reflection_rejects_invalid_structured_draft_without_exposure(
        self,
        monkeypatch,
    ) -> None:
```

Replace its `FakeAgent` and `FakeRegistry` with:

```python
        class FakeAgent:
            def generate_reflection(self, context, prompt):
                raise ReflectionDraftGenerationError("invalid_structured_draft")

        class FakeRegistry:
            def get_agent(self, player_id):
                return FakeAgent()
```

Call `_agent_reflection` with `registry=FakeRegistry()` and retain the assertions that the failure is `invalid_structured_draft`, verified lessons are empty, and raw reflection text is absent.

- [ ] **Step 5: Run the reflection-focused regression set**

Run:

```bash
conda run -n wofkill python -m pytest tests/runtime/test_strategy_directives.py::TestReflectionRoleSpecific tests/runtime/test_reflection_security_contract.py tests/agents/test_reflection_generation.py -q -o addopts=''
```

Expected: all tests pass and the current `generate_reflection` production contract remains unchanged.

- [ ] **Step 6: Commit the reflection test migration**

```bash
git add tests/runtime/test_strategy_directives.py
git diff --cached --check
git commit -m "test: align reflection fakes with draft API"
```

Expected: one commit containing only `tests/runtime/test_strategy_directives.py`.

### Task 5: Make PowerShell syntax validation portable

**Files:**
- Modify: `tests/scripts/test_analyze_recent_balance.py:1-10`
- Modify: `tests/scripts/test_analyze_recent_balance.py:96-116`

- [ ] **Step 1: Reproduce the missing executable failure on macOS**

Run:

```bash
conda run -n wofkill python -m pytest tests/scripts/test_analyze_recent_balance.py::test_soak_script_has_valid_powershell_ast -q -o addopts=''
```

Expected: FAIL with `FileNotFoundError` because neither `powershell` nor `pwsh` is installed.

- [ ] **Step 2: Add the required Python module header and imports**

Add this header before imports:

```python
# -*- coding: utf-8 -*-
"""
验证近期平衡报告与审计闭环 PowerShell 脚本契约。

作者: Project contributors
修改日期: 2026-07-26
"""
```

Keep `from __future__ import annotations` immediately after the docstring. Add imports in standard-library/third-party order:

```python
import shutil
import subprocess

import pytest
```

- [ ] **Step 3: Resolve both platform command names and skip explicitly**

At the start of `test_soak_script_has_valid_powershell_ast`, add:

```python
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("需要 pwsh 或 powershell 才能验证 PowerShell AST")
```

Change the subprocess executable from the literal `"powershell"` to `powershell`:

```python
    result = subprocess.run(
        [powershell, "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )
```

- [ ] **Step 4: Verify the script test module**

Run:

```bash
conda run -n wofkill python -m pytest tests/scripts/test_analyze_recent_balance.py -q -o addopts=''
```

Expected on this macOS host: all ordinary tests pass and exactly the PowerShell AST test skips with the declared reason. On a host with `pwsh` or `powershell`, the AST test executes and passes.

- [ ] **Step 5: Commit the platform-aware test**

```bash
git add tests/scripts/test_analyze_recent_balance.py
git diff --cached --check
git commit -m "test: detect PowerShell across platforms"
```

Expected: one commit containing only the script test module.

### Task 6: Document the safe complete-suite command

**Files:**
- Modify: `README.md:120-138`

- [ ] **Step 1: Record the stale bare commands and shared temporary path**

Run:

```bash
rg -n "python -m pytest|--basetemp \.tmp/pytest" README.md
```

Expected: the test section uses bare Python and the Windows fallback reuses `.tmp/pytest`, which nested regression tests also use.

- [ ] **Step 2: Prefix project test commands with the project Conda environment**

Change the complete-suite example to:

```powershell
conda run -n wofkill python -m pytest -q -o addopts=''
```

Change each focused example in the same section to use `conda run -n wofkill python -m pytest ...`.

- [ ] **Step 3: Isolate the Windows-only writable temporary directory**

Replace the fallback command and explanation with:

````markdown
Windows 默认临时目录出现 `WinError 5` 时，可显式使用不被嵌套测试管理的仓库目录：

```powershell
conda run -n wofkill python -m pytest -q -o addopts='' --basetemp .pytest-tmp/full-suite
```
````

The canonical macOS/Linux command must remain free of `--basetemp`.

- [ ] **Step 4: Verify README consistency with both project agent rules**

Run:

```bash
rg -n "conda run -n wofkill python -m pytest" README.md AGENTS.md CLAUDE.md
! rg -n -- "--basetemp \.tmp/pytest" README.md
```

Expected: all three documents expose the explicit Conda command and the obsolete shared path is absent.

- [ ] **Step 5: Commit the verification documentation**

```bash
git add README.md
git diff --cached --check
git commit -m "docs: use safe conda test command"
```

Expected: one commit containing only `README.md`.

### Task 7: Run focused and complete verification

**Files:**
- Verify: `environment.yml`
- Verify: `AGENTS.md`
- Verify: `CLAUDE.md`
- Verify: `README.md`
- Verify: the four modified test modules

- [ ] **Step 1: Run every formerly failing test in one focused command**

Run:

```bash
conda run -n wofkill python -m pytest -q -o addopts='' \
  tests/api/test_game_persistence_helpers.py::test_game_router_keeps_route_registration_surface \
  tests/rag/test_rag_hardening.py::TestAutoVectorStore::test_auto_selects_embedding_when_numpy_available \
  tests/regression/test_post_july14_contract_cases.py::test_every_audit_issue_maps_to_an_existing_pytest_node \
  tests/regression/test_post_july14_contract_cases.py::test_mapped_audit_nodes_execute_as_nonrecursive_closure_batch \
  tests/runtime/test_strategy_directives.py::TestReflectionRoleSpecific::test_agent_reflection_uses_REFLECTION_task_type \
  tests/runtime/test_strategy_directives.py::TestReflectionRoleSpecific::test_agent_reflection_rejects_invalid_structured_draft_without_exposure \
  tests/scripts/test_analyze_recent_balance.py::test_soak_script_has_valid_powershell_ast
```

Expected on this host: six pass and the PowerShell test skips, with no failures or errors.

- [ ] **Step 2: Run static checks on modified Python test modules**

Run:

```bash
conda run -n wofkill python -m ruff check tests/api/test_game_persistence_helpers.py tests/regression/test_post_july14_contract_cases.py tests/runtime/test_strategy_directives.py tests/scripts/test_analyze_recent_balance.py
```

Expected: Ruff exits 0 with no findings.

- [ ] **Step 3: Run the complete serial suite without a shared basetemp**

Run:

```bash
conda run -n wofkill python -m pytest -q -o addopts=''
```

Expected: all collected non-platform tests pass, the existing optional skips remain skips, and there are zero failures and zero errors.

- [ ] **Step 4: Verify dependency resolution and repository cleanliness**

Run:

```bash
conda run -n wofkill python --version
conda run -n wofkill python -c "import numpy, pytest, xdist; print(numpy.__version__, pytest.__version__, xdist.__version__)"
git diff --check 0ebec44..HEAD
git status --short --branch
git diff --cached --name-only
```

Expected: Python is 3.12.x; all package versions print; the branch worktree is clean; no staged files remain.

- [ ] **Step 5: Confirm the main worktree remains untouched**

Run from the main repository root:

```bash
git status --short --branch
```

Expected: the user's existing `.superpowers/brainstorm` deletions and `.DS_Store` files remain unstaged, with no files from this repair work present as uncommitted changes.

### Task 8: Review and integrate locally

**Files:**
- Review: branch range `0ebec44..HEAD`

- [ ] **Step 1: Run independent specification and quality reviews**

Dispatch a specification reviewer against the approved design and this plan, then dispatch a separate quality reviewer only after specification approval.

Expected: no open Critical or Important findings. Any finding is fixed by the task implementer and re-reviewed before continuing.

- [ ] **Step 2: Run a final whole-branch review**

Review `git diff 0ebec44..HEAD` for scope, test-contract correctness, dependency completeness, cross-platform behavior, documentation consistency, and unrelated changes.

Expected: the branch is approved for local integration.

- [ ] **Step 3: Merge only after fresh pre-merge verification**

Use `superpowers:verification-before-completion` and `superpowers:finishing-a-development-branch`. Execute the already selected local-merge option only when the complete serial suite exits 0.

Expected: `codex/conda-env-rules` merges into `master` without discarding the main worktree's existing unstaged files.

- [ ] **Step 4: Verify the merged result and clean the owned worktree**

Run on merged `master`:

```bash
conda run -n wofkill python -m pytest -q -o addopts=''
```

Expected: the complete suite exits 0. Then remove `/Users/zengyilin/NLP/wofkill/.worktrees/conda-env-rules`, prune worktree registrations, and delete the merged `codex/conda-env-rules` branch.
