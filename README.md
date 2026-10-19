# GUI Agent

Base repository for a modular Python GUI agent focused on screen observation, visual tagging, VLM-based planning, and operating system-level automation.

## Initial Structure

- `agent/`: orchestration, planner, and prompts.
- `vision/`: screen capture, visual tagging, and frame diffs.
- `controller/`: system control abstraction and safety.
- `schemas/`: typed action models for the agent.
- `main.py`: CLI entrypoint.

## Setup

Create and activate the virtual environment before installing dependencies or running tests (the global Python does not have them installed):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest
```

## Current State

This commit only provides the project scaffold. There is no functional implementation yet.

## Next Steps

1. Implement `schemas/actions.py` first.
2. Then add `vision/capture.py` and `vision/som_tagger.py`.
3. After that, implement `controller/base.py` and `controller/os_controller.py`.
4. Finally, wire up `agent/planner.py`, `agent/state_machine.py`, and `main.py`.
