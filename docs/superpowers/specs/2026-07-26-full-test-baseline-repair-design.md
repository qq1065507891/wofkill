# Full Test Baseline Repair Design

## Goal

Restore a reliable macOS test baseline in the project-level `wofkill` Conda
environment so the Conda rule branch can pass the complete serial test suite
before local integration.

## Confirmed Scope

- Add the missing Python test dependencies `numpy` and `pytest-xdist` to
  `environment.yml` and synchronize the existing `wofkill` environment.
- Keep PowerShell optional on macOS. A PowerShell syntax test may use either
  `pwsh` or `powershell`; when neither executable is installed, it must skip
  with an explicit reason.
- Repair stale tests that no longer match the current FastAPI route model,
  reflection draft API, or audit-node name.
- Repair the complete-suite temporary-directory command so nested pytest runs
  cannot invalidate the parent suite's `basetemp`.
- Preserve application behavior unless investigation proves that a production
  defect, rather than a stale test contract, caused a failure.

## Root-Cause Groups

### Conda dependency closure

`AutoVectorStore` intentionally selects its embedding backend when NumPy is
available, but `environment.yml` does not install NumPy. The nested audit
closure invokes pytest with `-n0`, but `pytest.ini` also supplies xdist options
and the environment does not install `pytest-xdist`. These are environment
declaration gaps, so the dependencies belong in `environment.yml` rather than
being mocked or suppressed in tests.

### Cross-platform PowerShell validation

The repository contains a `.ps1` script whose AST can only be validated by a
PowerShell runtime. macOS Terminal normally runs `zsh`, which cannot parse
PowerShell syntax. The test currently hard-codes the Windows-style
`powershell` command. It will resolve `pwsh` first, then `powershell`, and skip
when neither is installed. Installing PowerShell remains an optional platform
prerequisite, not a Python project dependency.

### FastAPI route introspection

FastAPI 0.140 exposes included routers through `_IncludedRouter` entries that
do not have a `path` attribute. The route-surface test assumes every entry in
`app.routes` is a flattened route. The test must inspect the public route
surface without relying on every framework-internal entry having `path`.

### Reflection API migration

The reflection runtime now calls `agent.generate_reflection(...)` and verifies
a structured draft. Two older tests still provide fake agents with only
`act(...)`, causing `AttributeError` to be normalized as `agent_error`. The
tests must use fakes that implement the current structured-reflection contract
and assert the intended invalid-draft behavior without reverting production
code to the obsolete action path.

### Audit regression-node rename

The N7 audit mapping references
`test_runtime_no_kill_event_has_complete_v2_audit_identity`, while the actual
test is named
`test_provider_failure_no_kill_event_has_complete_v2_audit_identity`. The
mapping must point to the existing contract test.

### Shared `basetemp` collision

The README full-suite command uses the repository-relative
`.tmp/pytest` directory. Some regression tests start nested pytest processes
under `.tmp` and clean their temporary directory, which can remove the parent
suite's base directory and produce cascading setup errors. The canonical full
suite command will omit a shared repository `--basetemp`, allowing pytest to
allocate isolated system temporary directories. Focused commands may continue
to use unique temporary directories when required.

## Implementation Boundaries

- Dependency changes stay in `environment.yml`.
- Cross-platform command discovery stays in the PowerShell-specific test.
- FastAPI compatibility stays in its route-surface test unless a public route
  is genuinely absent.
- Reflection compatibility stays in test fakes and assertions unless a fresh
  structured-agent reproduction demonstrates a runtime defect.
- Audit mapping changes only the stale node identifier.
- README changes only the canonical full-suite verification command and nearby
  explanation.
- Python files touched non-trivially must retain or update their Chinese module
  docstrings and modification date according to `AGENTS.md`; new comments are
  written in Chinese.

## Verification

Each root-cause group receives a focused red-green verification before its
change is committed. The final gate is:

```bash
conda env update -n wofkill -f environment.yml --prune
conda run -n wofkill python -m pytest -q -o addopts=''
```

Acceptance requires:

- Python 3.12, NumPy, pytest, and pytest-xdist resolve from `wofkill`.
- All non-platform-specific tests pass.
- The PowerShell AST test passes when `pwsh` or `powershell` is installed and
  skips with an explicit reason otherwise.
- No test depends on a shared repository `basetemp` for the complete suite.
- The feature worktree is clean after commits and contains no unrelated files.
- The original main-worktree `.superpowers` deletions and `.DS_Store` files
  remain untouched and unstaged.

## Integration

After focused and complete-suite verification, run independent specification,
quality, and final branch reviews. Then merge `codex/conda-env-rules` into
`master` locally, rerun the complete serial suite on the merged result, remove
the Superpowers-owned worktree, and delete the merged feature branch.
