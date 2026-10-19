"""Floating command bar: type a goal, press Enter, watch the agent work.

The bar sits bottom-center, borderless and always on top. When a goal starts
it hides (so the keyboard focus goes back to the previous window) and a small
status pill appears bottom-right above the taskbar. The pill would show up in
the agent's screenshots, so ``MaskedScreen`` paints its area gray before the
model sees it and ``GuardedController`` skips any click aimed at it. When the
run ends, the bar comes back showing only the final status.

Everything except the tkinter window itself (``CommandBar``) is plain logic
covered by unit tests; the window is checked by hand.
"""

from __future__ import annotations

import ctypes
import math
import queue
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from agent.state_machine import RunResult, RunStatus, StepEvent
from controller.base import BaseController
from vision.capture import Frame

PILL_SIZE = (360, 44)  # width, height in screen pixels
PILL_MARGIN = 12
MASK_COLOR = (128, 128, 128)  # BGR gray painted over the pill in captures
BAR_WIDTH = 640
BAR_HEIGHT = 52
DEFAULT_START_DELAY = 1.0  # seconds before the agent takes the mouse

STATUS_WORDS = {
    RunStatus.FINISHED: "Listo",
    RunStatus.FAILED: "No pudo completarse",
    RunStatus.STUCK: "Se atascó",
    RunStatus.ABORTED: "Detenido",
    RunStatus.MAX_STEPS: "Límite de pasos",
}

Rect = tuple[int, int, int, int]  # x, y, width, height in screen pixels


def format_hotkey(hotkey: str) -> str:
    """``"ctrl+alt+q"`` -> ``"Ctrl+Alt+Q"``."""

    return "+".join(part.strip().capitalize() for part in hotkey.split("+"))


def pill_rect(
    work_area: Rect, size: tuple[int, int] = PILL_SIZE, margin: int = PILL_MARGIN
) -> Rect:
    """Bottom-right corner of the work area (the screen minus the taskbar)."""

    left, top, width, height = work_area
    pill_w, pill_h = size
    return (
        left + width - margin - pill_w,
        top + height - margin - pill_h,
        pill_w,
        pill_h,
    )


def step_text(event: StepEvent, pause_hotkey: str = "ctrl+alt+q") -> str:
    """Pill text while the agent works."""

    return (
        f"Paso {event.number} · {event.action.action_type} · "
        f"{format_hotkey(pause_hotkey)} pausa"
    )


def result_text(result: RunResult) -> str:
    """Bar text when a run ends: only the status."""

    return STATUS_WORDS[result.status]


def _inside(rect: Rect, x: int, y: int) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + rh


class MaskedScreen:
    """Wraps a screen so the pill's area is painted gray in every capture."""

    def __init__(self, screen: Any, rect: Rect) -> None:
        self._screen = screen
        self._rect = rect

    def capture(self) -> Frame:
        return self._mask(self._screen.capture())

    def capture_region(self, x: int, y: int, width: int, height: int) -> Frame:
        return self._mask(self._screen.capture_region(x, y, width, height))

    def _mask(self, frame: Frame) -> Frame:
        rx, ry, rw, rh = self._rect
        ox, oy = frame.origin
        width, height = frame.size
        x0 = max(0, math.floor((rx - ox) * frame.scale))
        y0 = max(0, math.floor((ry - oy) * frame.scale))
        x1 = min(width, math.ceil((rx + rw - ox) * frame.scale))
        y1 = min(height, math.ceil((ry + rh - oy) * frame.scale))
        if x0 >= x1 or y0 >= y1:
            return frame

        image = frame.image.copy()
        image[y0:y1, x0:x1] = MASK_COLOR
        return Frame(
            image=image,
            original_size=frame.original_size,
            scale=frame.scale,
            origin=frame.origin,
        )


class GuardedController(BaseController):
    """Forwards actions, except mouse actions that land on the pill.

    A skipped action raises nothing: the screen does not change, so the agent
    sees "no visible change" in its history and tries something else.
    """

    def __init__(
        self,
        inner: Any,
        rect: Rect,
        on_blocked: Callable[[str], None] | None = None,
    ) -> None:
        self._inner = inner
        self._rect = rect
        self._on_blocked = on_blocked

    def _blocked(self, what: str, *points: tuple[int, int]) -> bool:
        if any(_inside(self._rect, x, y) for x, y in points):
            if self._on_blocked:
                self._on_blocked(f"{what} on the status pill skipped")
            return True
        return False

    def click(self, x: int, y: int, button: str = "left") -> None:
        if not self._blocked("click", (x, y)):
            self._inner.click(x, y, button=button)

    def double_click(self, x: int, y: int) -> None:
        if not self._blocked("double_click", (x, y)):
            self._inner.double_click(x, y)

    def type_text(self, text: str) -> None:
        self._inner.type_text(text)

    def press_key(self, key: str) -> None:
        self._inner.press_key(key)

    def drag(
        self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float
    ) -> None:
        if not self._blocked("drag", (start_x, start_y), (end_x, end_y)):
            self._inner.drag(start_x, start_y, end_x, end_y, duration)

    def scroll(self, x: int, y: int, amount: int) -> None:
        if not self._blocked("scroll", (x, y)):
            self._inner.scroll(x, y, amount)


class AgentRunner:
    """Runs one goal at a time in a background thread."""

    def __init__(
        self,
        build: Callable[[Callable[[StepEvent], None]], Any],
        on_event: Callable[[StepEvent], None],
        on_done: Callable[[RunResult | None, BaseException | None], None],
        start_delay: float = DEFAULT_START_DELAY,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """
        Args:
            build: Given the step callback, returns an object whose
                ``run(goal)`` returns a ``RunResult``.
            on_event: Called (from the worker thread) for every step.
            on_done: Called (from the worker thread) with the result, or with
                ``(None, error)`` if the run raised.
            start_delay: Silent pause before the agent starts.
        """

        self._build = build
        self._on_event = on_event
        self._on_done = on_done
        self._start_delay = start_delay
        self._sleep = sleep
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, goal: str) -> bool:
        """Start ``goal`` unless it is empty or another run is in progress."""

        goal = goal.strip()
        with self._lock:
            if not goal or self._running:
                return False
            self._running = True
        self._thread = threading.Thread(target=self._run, args=(goal,), daemon=True)
        self._thread.start()
        return True

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self, goal: str) -> None:
        result: RunResult | None = None
        error: BaseException | None = None
        try:
            self._sleep(self._start_delay)
            result = self._build(self._on_event).run(goal)
        except Exception as exc:  # reported to the UI, never raised in a thread
            error = exc
        finally:
            self._running = False
        self._on_done(result, error)


# --------------------------------------------------------------------- window


def work_area() -> Rect:
    """Primary monitor minus the taskbar, in physical pixels (Windows)."""

    if sys.platform == "win32":

        class _Rect(ctypes.Structure):
            _fields_ = [(n, ctypes.c_long) for n in ("left", "top", "right", "bottom")]

        r = _Rect()
        if ctypes.windll.user32.SystemParametersInfoW(0x30, 0, ctypes.byref(r), 0):
            return r.left, r.top, r.right - r.left, r.bottom - r.top
    import tkinter

    root = tkinter.Tk()
    root.withdraw()
    width, height = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return 0, 0, width, height - 48


def _no_activate(window: Any) -> None:
    """Keep a window from taking the keyboard focus (Windows)."""

    if sys.platform != "win32":
        return
    gwl_exstyle, ws_ex_noactivate, ws_ex_toolwindow = -20, 0x08000000, 0x00000080
    hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
    style = ctypes.windll.user32.GetWindowLongW(hwnd, gwl_exstyle)
    ctypes.windll.user32.SetWindowLongW(
        hwnd, gwl_exstyle, style | ws_ex_noactivate | ws_ex_toolwindow
    )


class CommandBar:
    """The tkinter window. Run it with ``CommandBar(...).run()``."""

    BG = "#1f1f1f"
    FG = "#f2f2f2"
    MUTED = "#a0a0a0"

    def __init__(
        self,
        build: Callable[[Callable[[StepEvent], None], Rect], Any],
        pause_hotkey: str,
        area: Rect | None = None,
        on_done: Callable[[RunResult | None, BaseException | None], None] | None = None,
    ) -> None:
        """
        Args:
            build: Given the step callback and the pill rectangle, returns an
                object with ``run(goal)`` (masking and guarding included).
            pause_hotkey: Shown in the pill.
            area: Work area; read from Windows when omitted.
            on_done: Extra callback when a run ends (e.g. console output).
        """

        import tkinter as tk

        self._tk = tk
        self._area = area or work_area()
        self._pill = pill_rect(self._area)
        self._pause_hotkey = pause_hotkey
        self._extra_done = on_done
        self._events: queue.Queue = queue.Queue()
        self._runner = AgentRunner(
            lambda on_step: build(on_step, self._pill),
            on_event=lambda e: self._events.put(("step", e)),
            on_done=lambda r, err: self._events.put(("done", r, err)),
        )

        self.root = tk.Tk()
        self.root.title("GUI Agent")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=self.BG)
        left, top, width, height = self._area
        x = left + (width - BAR_WIDTH) // 2
        y = top + height - BAR_HEIGHT - 3 * PILL_MARGIN
        self.root.geometry(f"{BAR_WIDTH}x{BAR_HEIGHT}+{x}+{y}")

        frame = tk.Frame(self.root, bg=self.BG, padx=14, pady=10)
        frame.pack(fill="both", expand=True)
        self.entry = tk.Entry(
            frame,
            bg=self.BG,
            fg=self.FG,
            insertbackground=self.FG,
            relief="flat",
            font=("Segoe UI", 12),
        )
        self.entry.pack(side="left", fill="x", expand=True)
        self.status = tk.Label(
            frame, text="", bg=self.BG, fg=self.MUTED, font=("Segoe UI", 10)
        )
        self.status.pack(side="left", padx=(10, 6))
        close = tk.Label(
            frame, text="×", bg=self.BG, fg=self.MUTED, font=("Segoe UI", 14)
        )
        close.pack(side="right")
        close.bind("<Button-1>", lambda _e: self.root.destroy())

        self.entry.bind("<Return>", self._submit)
        self.root.bind("<Escape>", lambda _e: self.root.destroy())
        for widget in (frame, self.status):
            widget.bind("<ButtonPress-1>", self._drag_start)
            widget.bind("<B1-Motion>", self._drag_move)

        self.pill = tk.Toplevel(self.root)
        self.pill.overrideredirect(True)
        self.pill.attributes("-topmost", True)
        self.pill.configure(bg=self.BG)
        px, py, pw, ph = self._pill
        self.pill.geometry(f"{pw}x{ph}+{px}+{py}")
        self.pill_label = tk.Label(
            self.pill, text="", bg=self.BG, fg=self.FG, font=("Segoe UI", 10)
        )
        self.pill_label.pack(fill="both", expand=True, padx=10)
        self.pill.withdraw()
        self.root.update_idletasks()
        _no_activate(self.pill)

        self.root.after(100, self._poll)

    def run(self) -> None:
        self.entry.focus_force()
        self.root.mainloop()

    def _submit(self, _event: Any = None) -> None:
        if not self._runner.start(self.entry.get()):
            return
        self.status.configure(text="")
        self.root.withdraw()  # gives the focus back to the previous window
        self.pill_label.configure(text="Trabajando…")
        self.pill.deiconify()

    def _poll(self) -> None:
        while True:
            try:
                item = self._events.get_nowait()
            except queue.Empty:
                break
            if item[0] == "step":
                self.pill_label.configure(text=step_text(item[1], self._pause_hotkey))
            else:
                _, result, error = item
                self.pill.withdraw()
                text = "Error (ver consola)" if error else result_text(result)
                self.status.configure(text=text)
                self.entry.delete(0, "end")
                self.root.deiconify()
                self.entry.focus_force()
                if self._extra_done:
                    self._extra_done(result, error)
        self.root.after(100, self._poll)

    def _drag_start(self, event: Any) -> None:
        self._drag_from = (
            event.x_root - self.root.winfo_x(),
            event.y_root - self.root.winfo_y(),
        )

    def _drag_move(self, event: Any) -> None:
        dx, dy = self._drag_from
        self.root.geometry(f"+{event.x_root - dx}+{event.y_root - dy}")


__all__ = [
    "BAR_HEIGHT",
    "BAR_WIDTH",
    "DEFAULT_START_DELAY",
    "MASK_COLOR",
    "PILL_MARGIN",
    "PILL_SIZE",
    "STATUS_WORDS",
    "AgentRunner",
    "CommandBar",
    "GuardedController",
    "MaskedScreen",
    "format_hotkey",
    "pill_rect",
    "result_text",
    "step_text",
    "work_area",
]
