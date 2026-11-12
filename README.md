# GUI Agent

Base repository for a modular Python GUI agent focused on screen observation, visual tagging, VLM-based planning, and operating system-level automation.

## Structure

- `agent/`: planner (Claude), prompts, the agent loop and the run recorder.
- `vision/`: screen capture, Set-of-Mark grid and frame diffs.
- `controller/`: mouse/keyboard control (pynput) and the emergency switch.
- `schemas/`: typed action models for the agent.
- `ui/`: floating command bar.
- `main.py`: entry point.

## Setup

Create and activate the virtual environment before installing dependencies or running tests (the global Python does not have them installed):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest
```

Dependencies live in `pyproject.toml` (`dev` extra: pytest and ruff). Tests that touch the real screen, input devices or external APIs are marked `integration` and skipped by default; run them with `python -m pytest -m integration`.

Before pushing, run the same checks as CI:

```powershell
ruff check .
ruff format --check .
python -m pytest
```

## Usage

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` (the API is billed separately from a Claude subscription). Then:

```powershell
python main.py
```

opens a floating bar: type a goal, press Enter, and the agent works while a small status pill shows its progress. Or run one goal in the console:

```powershell
python main.py --goal "Open Notepad and type Hello World"
```

While the agent runs, **Ctrl+Alt+Q** pauses it; while paused, **Ctrl+Alt+R** resumes and **Ctrl+Alt+Q** again stops it. The console shows every step, its cost, and where the run log was saved (`runs/`, gitignored: it holds real screenshots).

## Precision bench

`bench/precision.py` measures where the agent's clicks land on ~20 synthetic scenes (`tests/bench_scenes.py`: large targets, small icons and menu items, and targets at the edge of a non-maximized window). For each scene it runs the real pipeline without moving the mouse: grid, `Planner.decide`, the state machine's zoom and `Planner.refine`, then scores the click point against the target rectangle.

```powershell
python -m bench.precision --scenes 20 --budget 1.0
python -m bench.precision --model claude-haiku-5-5 --env-file .env
```

It reports hit rate, mean and median distance to the target center, calls and estimated cost, and stops once the cost reaches `--budget` (USD). Results go to `bench/results/<timestamp>-<model>.json` (gitignored); `--baseline` also rewrites `bench/baseline.json`, the tracked reference that later changes must not fall below. It makes real API calls, so it needs `ANTHROPIC_API_KEY`.
