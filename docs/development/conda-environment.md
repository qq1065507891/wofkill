# Conda Development Environment

Use the repository root `environment.yml` to create the shared development environment.

## Create

```powershell
conda env create -f environment.yml
```

## Activate

```powershell
conda activate wofkill
```

## Verify

```powershell
python -m pytest -q
```

Expected: all tests pass (807 tests as of latest run). The current test state is recorded in `PROGRESS.md`.

## Key Commands

```powershell
# Full test suite
python -m pytest -q

# Rule engine tests
python -m pytest tests/rules/ -q

# API + permissions
python -m pytest tests/api/ -q

# Evaluation + integration
python -m pytest tests/evaluation/ tests/integration/ -q

# Start API server
uvicorn werewolf_agent.api.app:create_app --factory --reload
# Open http://localhost:8000/ for dashboard

# Start API with SQLite persistence
python -c "from werewolf_agent.api.app import create_app; from werewolf_agent.storage.sqlite_store import SqliteGameRepository; import uvicorn; app = create_app(repository=SqliteGameRepository('data/games.db')); uvicorn.run(app, host='0.0.0.0', port=8000)"
```

## Update

After adding dependencies, update the existing environment with:

```powershell
conda env update -f environment.yml --prune
```

## API Keys

Copy `.env.example` to `.env` and fill in provider keys for real LLM games:

```env
ANTHROPIC_API_KEY=sk-ant-...
GLM_API_KEY=...
OPENAI_API_KEY=sk-...
```

Without API keys, all tests run with mock providers. Real provider configuration is documented in `config/models.yaml`.

## Optional Real Provider Smoke Test

Real-provider smoke tests are skipped by default so normal development never spends API quota or requires network access.

```powershell
$env:WEREWOLF_RUN_REAL_LLM_SMOKE = "1"
$env:OPENAI_API_KEY = "..."
# or ANTHROPIC_API_KEY / GLM_API_KEY
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/integration/test_real_llm_smoke.py -q --basetemp .pytest-tmp
```

Expected without `WEREWOLF_RUN_REAL_LLM_SMOKE=1`: the smoke test is skipped.

## V1.1 Local Hardening Commands

Use the workspace temp directory on Windows when the default user temp path is locked down:

```powershell
D:/Miniforge3/envs/wofkill/python.exe -m pytest -q --basetemp .pytest-tmp
```

Targeted V1.1 checks:

```powershell
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/runtime/test_game_runner.py tests/runtime/test_runtime.py -q --basetemp .pytest-tmp
D:/Miniforge3/envs/wofkill/python.exe -m pytest tests/tools/test_mcp_hardening.py tests/rag/test_rag_hardening.py tests/storage/test_storage.py -q --basetemp .pytest-tmp
```

Do not commit local secrets, provider API keys, or machine-specific Conda paths.
