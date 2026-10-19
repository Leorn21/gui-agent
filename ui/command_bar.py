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
from controller.safety import SafetyState
from vision.capture import Frame

PILL_SIZE = (360, 44)  # width, height in screen pixels
PILL_MARGIN = 12
MASK_COLOR = (128, 128, 128)  # BGR gray painted over the pill in captures
BAR_WIDTH = 640
BAR_HEIGHT = 56
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


# -------------------------------------------------------------------- styling

# Look inspired by React Bits' "Prompt Bar" (dark, rounded, circular send
# button) and "Star Border" (a light moving on the edge while working).
BG = "#1c1c21"
FG = "#f2f2f5"
MUTED = "#8b8b96"
BAR_BORDER = "#3a3a44"
ACCENT = "#8b7bff"
PULSE_BASE = "#3d3566"
PULSE_PEAK = "#a89bff"
PAUSED_BORDER = "#f5a524"
PLACEHOLDER = "¿Qué hago?"
PULSE_PERIOD = 2.0  # seconds per pulse
PULSE_FRAME_MS = 50

# Windows 11 DWM window attributes.
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWCP_ROUND = 2
DWMWA_BORDER_COLOR = 34


def hex_to_colorref(color: str) -> int:
    """``"#rrggbb"`` -> Windows COLORREF (``0x00BBGGRR``)."""

    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    return (b << 16) | (g << 8) | r


def _mix(a: str, b: str, f: float) -> str:
    ca = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(
        f"{round(x + (y - x) * f):02x}" for x, y in zip(ca, cb, strict=True)
    )


def pulse_color(phase: float) -> str:
    """Border color for a pulse ``phase`` (cycles every 1.0): base, peak, base."""

    f = (1 - math.cos(2 * math.pi * (phase % 1.0))) / 2
    return _mix(PULSE_BASE, PULSE_PEAK, f)


def pill_border(state: SafetyState, phase: float) -> str:
    """Pulsing while running, steady amber while paused."""

    if state is SafetyState.PAUSED:
        return PAUSED_BORDER
    return pulse_color(phase)


def paused_text(pause_hotkey: str, resume_hotkey: str) -> str:
    return (
        f"Pausado · {format_hotkey(resume_hotkey)} reanuda · "
        f"{format_hotkey(pause_hotkey)} detiene"
    )


class _DwmApi:
    """Thin wrapper over ``DwmSetWindowAttribute`` (Windows only)."""

    def set_attribute(self, hwnd: int, attribute: int, value: int) -> None:
        data = ctypes.c_int(value)
        result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, attribute, ctypes.byref(data), ctypes.sizeof(data)
        )
        if result != 0:
            raise OSError(f"DwmSetWindowAttribute({attribute}) failed: {result:#x}")


def default_dwm() -> _DwmApi | None:
    return _DwmApi() if sys.platform == "win32" else None


_DEFAULT = object()


def apply_rounded_corners(hwnd: int, border: str, dwm: Any = _DEFAULT) -> bool:
    """Ask Windows 11 to round a window's corners and color its border.

    Returns ``False`` (window left as is) where that is not supported.
    """

    if dwm is _DEFAULT:
        dwm = default_dwm()
    if dwm is None:
        return False
    try:
        dwm.set_attribute(hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND)
        dwm.set_attribute(hwnd, DWMWA_BORDER_COLOR, hex_to_colorref(border))
    except OSError:
        return False
    return True


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


def _hwnd(window: Any) -> int:
    if sys.platform != "win32":
        return 0
    return ctypes.windll.user32.GetParent(window.winfo_id())


def _no_activate(window: Any) -> None:
    """Keep a window from taking the keyboard focus (Windows)."""

    if sys.platform != "win32":
        return
    gwl_exstyle, ws_ex_noactivate, ws_ex_toolwindow = -20, 0x08000000, 0x00000080
    hwnd = _hwnd(window)
    style = ctypes.windll.user32.GetWindowLongW(hwnd, gwl_exstyle)
    ctypes.windll.user32.SetWindowLongW(
        hwnd, gwl_exstyle, style | ws_ex_noactivate | ws_ex_toolwindow
    )


def _send_icon(size: int, fill: str, arrow: str) -> Any:
    """Anti-aliased circular send button with an up arrow (Pillow)."""

    from PIL import Image, ImageDraw, ImageTk

    scale = 4
    big = size * scale
    img = Image.new("RGBA", (big, big), BG)
    draw = ImageDraw.Draw(img)
    draw.ellipse((0, 0, big - 1, big - 1), fill=fill)
    c, w = big / 2, round(big * 0.07)
    draw.line((c, big * 0.72, c, big * 0.30), fill=arrow, width=w)
    draw.line((c, big * 0.28, big * 0.33, big * 0.45), fill=arrow, width=w)
    draw.line((c, big * 0.28, big * 0.67, big * 0.45), fill=arrow, width=w)
    return ImageTk.PhotoImage(img.resize((size, size), Image.LANCZOS))


class CommandBar:
    """The tkinter window. Run it with ``CommandBar(...).run()``."""

    def __init__(
        self,
        build: Callable[[Callable[[StepEvent], None], Rect], Any],
        pause_hotkey: str,
        resume_hotkey: str = "ctrl+alt+r",
        area: Rect | None = None,
        on_done: Callable[[RunResult | None, BaseException | None], None] | None = None,
    ) -> None:
        """
        Args:
            build: Given the step callback and the pill rectangle, returns an
                object with ``run(goal)`` (masking and guarding included).
            pause_hotkey, resume_hotkey: Shown in the pill.
            area: Work area; read from Windows when omitted.
            on_done: Extra callback when a run ends (e.g. console output).
        """

        import tkinter as tk

        self._area = area or work_area()
        self._pill = pill_rect(self._area)
        self._pause_hotkey = pause_hotkey
        self._resume_hotkey = resume_hotkey
        self._extra_done = on_done
        self._events: queue.Queue = queue.Queue()
        self._safety = SafetyState.RUNNING
        self._step_text = ""
        self._pulsing = False
        self._runner = AgentRunner(
            lambda on_step: build(on_step, self._pill),
            on_event=lambda e: self._events.put(("step", e)),
            on_done=lambda r, err: self._events.put(("done", r, err)),
        )

        # ---------------------------------------------------------------- bar
        self.root = tk.Tk()
        self.root.title("GUI Agent")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=BG)
        left, top, width, height = self._area
        x = left + (width - BAR_WIDTH) // 2
        y = top + height - BAR_HEIGHT - 3 * PILL_MARGIN
        self.root.geometry(f"{BAR_WIDTH}x{BAR_HEIGHT}+{x}+{y}")

        frame = tk.Frame(self.root, bg=BG, padx=18, pady=10)
        frame.pack(fill="both", expand=True)

        self._send_img = _send_icon(32, ACCENT, "#ffffff")
        send = tk.Label(frame, image=self._send_img, bg=BG, cursor="hand2")
        send.pack(side="right", padx=(8, 0))
        send.bind("<Button-1>", self._submit)

        close = tk.Label(frame, text="×", bg=BG, fg=MUTED, font=("Segoe UI", 13))
        close.pack(side="right", padx=(6, 4))
        close.bind("<Button-1>", lambda _e: self.root.destroy())
        close.bind("<Enter>", lambda _e: close.configure(fg=FG))
        close.bind("<Leave>", lambda _e: close.configure(fg=MUTED))

        self.status = tk.Label(frame, text="", bg=BG, fg=MUTED, font=("Segoe UI", 10))
        self.status.pack(side="right", padx=(8, 4))

        self.entry = tk.Entry(
            frame,
            bg=BG,
            fg=FG,
            insertbackground=FG,
            relief="flat",
            highlightthickness=0,
            font=("Segoe UI", 12),
        )
        self.entry.pack(side="left", fill="x", expand=True)
        self.placeholder = tk.Label(
            frame, text=PLACEHOLDER, bg=BG, fg=MUTED, font=("Segoe UI", 12)
        )
        self.placeholder.place(in_=self.entry, x=1, rely=0.5, anchor="w")
        self.placeholder.bind("<Button-1>", lambda _e: self.entry.focus_set())

        self.entry.bind("<Return>", self._submit)
        self.entry.bind("<KeyRelease>", lambda _e: self._update_placeholder())
        self.root.bind("<Escape>", lambda _e: self.root.destroy())
        for widget in (frame, self.status):
            widget.bind("<ButtonPress-1>", self._drag_start)
            widget.bind("<B1-Motion>", self._drag_move)

        # --------------------------------------------------------------- pill
        self.pill = tk.Toplevel(self.root)
        self.pill.overrideredirect(True)
        self.pill.attributes("-topmost", True)
        self.pill.configure(bg=BG)
        px, py, pw, ph = self._pill
        self.pill.geometry(f"{pw}x{ph}+{px}+{py}")
        self.pill_frame = tk.Frame(
            self.pill, bg=BG, highlightthickness=0, highlightbackground=PULSE_BASE
        )
        self.pill_frame.pack(fill="both", expand=True)
        self.pill_label = tk.Label(
            self.pill_frame, text="", bg=BG, fg=FG, font=("Segoe UI", 10)
        )
        self.pill_label.pack(fill="both", expand=True, padx=14)

        self.root.update_idletasks()
        self.pill.withdraw()
        _no_activate(self.pill)
        apply_rounded_corners(_hwnd(self.root), BAR_BORDER)
        self._pill_native = apply_rounded_corners(_hwnd(self.pill), PULSE_BASE)
        if not self._pill_native:  # no DWM border: draw one with tkinter
            self.pill_frame.configure(highlightthickness=2)

        self.root.after(100, self._poll)

    # ------------------------------------------------------------------- run

    def run(self) -> None:
        self.entry.focus_force()
        self.root.mainloop()

    def notify_safety(self, state: SafetyState) -> None:
        """Thread-safe: called from the keyboard hook thread."""

        self._events.put(("safety", state))

    def _update_placeholder(self) -> None:
        if self.entry.get():
            self.placeholder.place_forget()
        else:
            self.placeholder.place(in_=self.entry, x=1, rely=0.5, anchor="w")

    def _submit(self, _event: Any = None) -> None:
        if not self._runner.start(self.entry.get()):
            return
        self.status.configure(text="")
        self._safety = SafetyState.RUNNING
        self._step_text = "Trabajando…"
        self.root.withdraw()  # gives the focus back to the previous window
        self.pill_label.configure(text=self._step_text)
        self.pill.deiconify()
        self._pulsing = True
        self._pulse()

    def _pulse(self) -> None:
        if not self._pulsing:
            return
        phase = (time.monotonic() % PULSE_PERIOD) / PULSE_PERIOD
        color = pill_border(self._safety, phase)
        if self._pill_native:
            apply_rounded_corners(_hwnd(self.pill), color)
        else:
            self.pill_frame.configure(highlightbackground=color)
        self.root.after(PULSE_FRAME_MS, self._pulse)

    def _poll(self) -> None:
        while True:
            try:
                item = self._events.get_nowait()
            except queue.Empty:
                break
            kind = item[0]
            if kind == "step":
                self._step_text = step_text(item[1], self._pause_hotkey)
                if self._safety is not SafetyState.PAUSED:
                    self.pill_label.configure(text=self._step_text)
            elif kind == "safety":
                self._safety = item[1]
                if self._safety is SafetyState.PAUSED:
                    text = paused_text(self._pause_hotkey, self._resume_hotkey)
                else:
                    text = self._step_text
                self.pill_label.configure(text=text)
            else:
                _, result, error = item
                self._pulsing = False
                self.pill.withdraw()
                text = "Error (ver consola)" if error else result_text(result)
                self.status.configure(text=text)
                self.entry.delete(0, "end")
                self._update_placeholder()
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
    "ACCENT",
    "BAR_BORDER",
    "BAR_HEIGHT",
    "BAR_WIDTH",
    "DEFAULT_START_DELAY",
    "DWMWA_BORDER_COLOR",
    "DWMWA_WINDOW_CORNER_PREFERENCE",
    "DWMWCP_ROUND",
    "MASK_COLOR",
    "PAUSED_BORDER",
    "PILL_MARGIN",
    "PILL_SIZE",
    "PULSE_BASE",
    "PULSE_PEAK",
    "STATUS_WORDS",
    "AgentRunner",
    "CommandBar",
    "GuardedController",
    "MaskedScreen",
    "apply_rounded_corners",
    "format_hotkey",
    "hex_to_colorref",
    "paused_text",
    "pill_border",
    "pill_rect",
    "pulse_color",
    "result_text",
    "step_text",
    "work_area",
]
