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
python --version
pytest
```

The current expected test state during early RuleEngine development is recorded in `PROGRESS.md`. Do not treat failing tests as unexpected unless they differ from the progress ledger.

## Update

After adding dependencies, update the existing environment with:

```powershell
conda env update -f environment.yml --prune
```

Do not commit local secrets, provider API keys, or machine-specific Conda paths. Put credentials in local environment variables or an untracked `.env` file.
