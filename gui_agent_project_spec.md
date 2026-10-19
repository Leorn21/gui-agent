# SPECIFICATION & SYSTEM PROMPT: GUI-Agent Architecture Blueprint

## Context & Goal
You are an expert Principal Python & Systems Engineer. 
The objective is to scaffold and implement a production-grade, highly modular, clean-architecture GUI Agent in Python.
The agent observes the user's desktop screen, tags interactive zones/grid coordinates using computer vision (Set-of-Mark style), consults a Vision-Language Model (VLM), executes precise OS-level mouse/keyboard actions, and validates the resulting visual state through an autonomous feedback loop.

---

## Technical Stack & Constraints
- Language: Python 3.14+
- Type Hinting & Validation: Strict typing with Pydantic v2
- Image & Screen Capture: `mss` (high-performance in-memory screen capturing) + `OpenCV` (cv2) / `Pillow` (PIL)
- OS Automation: `pynput` (mouse and keyboard, Unicode typing) with abstract base controller architecture
- Safety: Global emergency switch (Ctrl+Alt+Q pauses; while paused Ctrl+Alt+R resumes and Ctrl+Alt+Q aborts)
- Modularity: Separation of Concerns (Vision, Controller, Agent/FSM, Schemas, Telemetry)

---

## Directory & File Structure to Scaffold

```text
gui-agent-core/
│
├── .gitignore
├── .env.example
├── pyproject.toml
├── README.md
│
├── agent/                             # Decision engine & orchestration
│   ├── __init__.py
│   ├── state_machine.py               # FSM Loop (Perception -> Decision -> Action -> Evaluation)
│   ├── planner.py                     # VLM Interface & structured tool caller
│   └── prompts.py                     # System prompt templates & few-shot examples
│
├── vision/                            # Perception & visual processing
│   ├── __init__.py
│   ├── capture.py                     # Ultra-fast screen grabber (mss to numpy/PIL)
│   ├── som_tagger.py                  # Set-of-Mark & grid overlay generator
│   └── diff.py                        # Frame difference detector to confirm UI changes
│
├── controller/                        # OS level peripheral emulation
│   ├── __init__.py
│   ├── base.py                        # Abstract Base Class (BaseController)
│   ├── os_controller.py               # Concrete OS input emulator
│   └── safety.py                      # Global threaded emergency kill-switch
│
├── schemas/                           # Typed contracts & Pydantic models
│   ├── __init__.py
│   └── actions.py                     # Action schemas (Click, Type, Drag, Wait, Finish, Fail)
│
└── main.py                            # CLI entrypoint & execution harness
```

---

## Module Specifications & Responsibilities

### 1. `schemas/actions.py`
Define strict Pydantic models for all agent outputs:
- `ActionType` (Enum: `CLICK`, `DOUBLE_CLICK`, `RIGHT_CLICK`, `TYPE`, `PRESS_KEY`, `DRAG`, `WAIT`, `FINISH`, `FAIL`)
- `AgentAction` (Pydantic model):
  - `thought: str` (Chain of thought explaining the rationale)
  - `action_type: ActionType`
  - `target_id: Optional[int]` (Tagged box/grid ID if using Set-of-Mark)
  - `coordinate: Optional[tuple[int, int]]` (Absolute or normalized x, y)
  - `text: Optional[str]` (Payload for typing)
  - `key: Optional[str]` (Special key stroke, e.g. "enter", "tab")
  - `duration: Optional[float]` (Wait time or drag duration)
  - `is_terminal: bool` (Flag indicating the task is completed)

### 2. `vision/capture.py`
- Class `ScreenCapture`:
  - Uses `mss` for zero-overhead primary monitor screen capture.
  - Returns frames as NumPy arrays (`cv2` BGR format) and PIL Images.
  - Provides utility to capture specific bounding boxes (ROI) or scaled down previews.

### 3. `vision/som_tagger.py`
- Class `VisualTagger`:
  - Method `apply_labeled_grid(image, grid_cols=20, grid_rows=12)`: Draws an alpha-blended high-contrast coordinate grid with clean alphanumeric or numeric markers.
  - Method `overlay_boxes(image, detected_boxes)`: Supports labeling detected interactive contours or UI bounding boxes.
  - Translates grid IDs back to normalized or absolute screen pixel coordinates $(x, y)$.

### 4. `vision/diff.py`
- Class `FrameDiff`:
  - Compares `frame_before` and `frame_after` with a thresholded pixel delta (OpenCV), merging nearby changes into regions and ignoring regions below a minimum area, to verify if an action produced a visual change.

### 5. `controller/base.py` & `controller/os_controller.py`
- `BaseController` (ABC):
  - `click(x, y, button="left")`
  - `double_click(x, y)`
  - `type_text(text)`
  - `press_key(key)`
  - `drag(start_x, start_y, end_x, end_y, duration)`
- `OSController` implementation:
  - Implements the methods using `pynput`.
  - Normalizes coordinate scaling (e.g., Retina / High-DPI screens).

### 6. `controller/safety.py`
- Class `EmergencyKillSwitch`:
  - Runs a lightweight background daemon thread (`pynput.keyboard`).
  - Monitors the pause / resume hotkeys (`Ctrl+Alt+Q`, `Ctrl+Alt+R`); a second `Ctrl+Alt+Q` while paused aborts.
  - Implements an atomic `is_aborted()` check that immediately stops the FSM loop and raises `InterruptedError`.

### 7. `agent/planner.py` & `agent/prompts.py`
- `prompts.py`: Defines the system instructions, guiding the VLM to observe the tagged image and return exclusively valid JSON adhering to the `AgentAction` schema.
- `planner.py`:
  - Accepts the user goal, execution history, and current annotated screenshot.
  - Calls Claude (default `claude-haiku-5-5`, configurable) through the Anthropic SDK with structured outputs, then refines each click on an enlarged region with a finer grid.
  - Validates and parses the response into an `AgentAction` instance.

### 8. `agent/state_machine.py`
- Class `AgentStateMachine`:
  - Encapsulates the execution loop:
    1. **Check Safety:** Ensure kill-switch is not triggered.
    2. **Perceive:** Capture screen and apply visual tags.
    3. **Decide:** Send context + tagged screenshot to `Planner`.
    4. **Act:** Execute action via `OSController`.
    5. **Evaluate:** Capture post-action frame, compute diff, update short-term memory.
    6. **Terminate:** Break when `is_terminal` is True or max steps exceeded.

### 9. `main.py`
- Rich CLI interface (using `argparse` or `typer` + `rich` for formatting).
- Accepts a natural language prompt from the user (e.g., `python main.py --goal "Open Notepad and type Hello World"`).
- Initializes safety listener, runs the agent loop with informative logging.

---

## Instructions for the Assistant
1. Initialize the complete repository structure with all mentioned folders and empty `__init__.py` files.
2. Generate all core classes with clean docstrings, comprehensive type annotations, robust error handling, and separation of concerns.
3. Keep external dependencies well-isolated and declare them all in `pyproject.toml` (runtime dependencies plus a `dev` extra).