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
