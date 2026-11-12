"""Synthetic scenes for the precision bench (``python -m bench.precision``).

Every scene is a 1920x1080 BGR image drawn with OpenCV (no real screenshots),
a goal text and the screen rectangle of the element the agent must click.
Scenes come in three kinds:

* ``large``: big targets (buttons, text fields).
* ``small``: icons, menu items, checkboxes and other targets under ~40 px.
* ``edge``: targets close to the border of a non-maximized window, where the
  zoom region reaches outside the window (a real run clicked ~60 px below a
  chat window's message box this way).

The generation is deterministic: the same call always returns the same images.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from tests import scenes as base

WIDTH, HEIGHT = base.WIDTH, base.HEIGHT
Rect = tuple[int, int, int, int]  # x, y, w, h in screen pixels
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_DARK = (40, 40, 40)


@dataclass(frozen=True, eq=False)
class Scene:
    """One bench scene.

    Attributes:
        name: Unique short identifier.
        kind: ``large``, ``small`` or ``edge``.
        goal: What the agent is asked to click.
        image: BGR ``uint8`` array, 1080x1920.
        target: ``(x, y, w, h)`` of the element to click, in screen pixels.
    """

    name: str
    kind: str
    goal: str
    image: NDArray[np.uint8]
    target: Rect

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.target
        return x + w // 2, y + h // 2

    def contains(self, px: float, py: float) -> bool:
        """Whether a screen point lies inside the target rectangle."""

        x, y, w, h = self.target
        return x <= px < x + w and y <= py < y + h


# ------------------------------------------------------------------ drawing


def _text(img, s, x, y, scale=0.7, color=_DARK):
    cv2.putText(img, s, (x, y), _FONT, scale, color, 2, cv2.LINE_AA)


def _rect(img, r: Rect, fill, border=None):
    x, y, w, h = r
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), fill, -1)
    if border is not None:
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), border, 1)


def _label_in(img, r: Rect, label, color=_DARK, scale=0.6):
    x, y, w, h = r
    (tw, th), _ = cv2.getTextSize(label, _FONT, scale, 2)
    _text(img, label, x + (w - tw) // 2, y + (h + th) // 2, scale, color)


def _button(img, r: Rect, label, fill):
    _rect(img, r, fill)
    _label_in(img, r, label, _WHITE)


def _desktop(seed: int = 0) -> NDArray[np.uint8]:
    """A wallpaper with a few desktop icons and a taskbar."""

    rng = np.random.default_rng(seed)
    img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    top = np.array((120 + 10 * seed, 80, 40), np.float32)
    bottom = np.array((200, 150, 90 + 10 * seed), np.float32)
    ramp = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None, None]
    img[:] = (top + (bottom - top) * ramp).astype(np.uint8)
    for i in range(4):
        x, y = 30, 30 + i * 110
        _rect(img, (x, y, 56, 56), tuple(int(v) for v in rng.integers(60, 230, 3)))
        _text(img, f"App {i + 1}", x - 2, y + 80, 0.5, _WHITE)
    _rect(img, (0, HEIGHT - 40, WIDTH, 40), (35, 35, 35))
    _rect(img, (6, HEIGHT - 36, 36, 32), (215, 120, 0))
    return img


def _window(img, r: Rect, title, body=(250, 250, 250)):
    """Draw a window with a title bar and a close button; return the close rect."""

    x, y, w, h = r
    cv2.rectangle(img, (x + 6, y + 6), (x + w + 6, y + h + 6), (20, 20, 20), -1)
    _rect(img, r, body, (90, 90, 90))
    _rect(img, (x, y, w, 36), (70, 60, 55))
    _text(img, title, x + 12, y + 25, 0.6, _WHITE)
    close = (x + w - 46, y, 46, 36)
    _rect(img, close, (60, 60, 190))
    _label_in(img, close, "X", _WHITE)
    return close


_COLORS = {
    "red": (40, 40, 220),
    "green": (60, 180, 60),
    "blue": (220, 120, 20),
    "orange": (30, 140, 250),
    "purple": (180, 60, 140),
}


def _icon(img, r: Rect, shape: str, color):
    x, y, w, h = r
    _rect(img, r, (225, 225, 225), (170, 170, 170))
    cx, cy = x + w // 2, y + h // 2
    q = min(w, h) // 3
    if shape == "circle":
        cv2.circle(img, (cx, cy), q, color, -1)
    elif shape == "square":
        cv2.rectangle(img, (cx - q, cy - q), (cx + q, cy + q), color, -1)
    elif shape == "triangle":
        pts = np.array([[cx, cy - q], [cx - q, cy + q], [cx + q, cy + q]])
        cv2.fillPoly(img, [pts], color)
    elif shape == "cross":
        cv2.line(img, (cx - q, cy - q), (cx + q, cy + q), color, 3)
        cv2.line(img, (cx - q, cy + q), (cx + q, cy - q), color, 3)
    elif shape == "diamond":
        pts = np.array([[cx, cy - q], [cx + q, cy], [cx, cy + q], [cx - q, cy]])
        cv2.fillPoly(img, [pts], color)


def _toolbar(img, origin, size, gap, icons, bg=(240, 240, 240)):
    """Draw a row of icons; return the rect of each by index."""

    x0, y0 = origin
    _rect(img, (x0 - 8, y0 - 8, len(icons) * (size + gap) + 8, size + 16), bg)
    rects = []
    for i, (shape, color) in enumerate(icons):
        r = (x0 + i * (size + gap), y0, size, size)
        _icon(img, r, shape, _COLORS[color])
        rects.append(r)
    return rects


_ICON_SET = [
    ("circle", "red"),
    ("square", "green"),
    ("triangle", "blue"),
    ("cross", "orange"),
    ("diamond", "purple"),
    ("circle", "blue"),
    ("square", "red"),
    ("triangle", "green"),
]


# ------------------------------------------------------------------- scenes


def _notes_scenes() -> list[Scene]:
    img = base.notes_app()
    return [
        Scene(
            "notes_save_button",
            "large",
            "Click the blue 'Guardar' (Save) button in the toolbar.",
            img,
            base.SAVE_BUTTON,
        ),
        Scene(
            "notes_export_button",
            "large",
            "Click the grey 'Exportar' button in the toolbar.",
            img,
            (630, 78, 140, 36),
        ),
        Scene(
            "notes_editor",
            "large",
            "Click inside the large white text editor to place the cursor.",
            img,
            (340, 400, 1520, 560),
        ),
    ]


def _form_scenes() -> list[Scene]:
    img = np.full((HEIGHT, WIDTH, 3), (240, 238, 235), np.uint8)
    _text(img, "Sign in to your account", 740, 300, 1.2)
    _text(img, "Username", 700, 392, 0.6)
    user = (700, 405, 520, 48)
    _rect(img, user, _WHITE, (150, 150, 150))
    _text(img, "Password", 700, 492, 0.6)
    _rect(img, (700, 505, 520, 48), _WHITE, (150, 150, 150))
    box = (700, 580, 20, 20)
    _rect(img, box, _WHITE, (90, 90, 90))
    _text(img, "Remember me", 735, 597, 0.6)
    login = (700, 640, 200, 52)
    _button(img, login, "Sign in", (215, 120, 0))
    _button(img, (930, 640, 200, 52), "Cancel", (130, 130, 130))
    return [
        Scene(
            "form_username_field",
            "large",
            "Click the 'Username' text field.",
            img,
            user,
        ),
        Scene(
            "form_sign_in_button",
            "large",
            "Click the blue 'Sign in' button.",
            img,
            login,
        ),
        Scene(
            "form_remember_checkbox",
            "small",
            "Click the 'Remember me' checkbox.",
            img,
            box,
        ),
    ]


def _menu_scenes() -> list[Scene]:
    img = np.full((HEIGHT, WIDTH, 3), (245, 245, 245), np.uint8)
    _rect(img, (0, 0, WIDTH, 34), (225, 225, 225))
    for i, name in enumerate(["File", "Edit", "View", "Help"]):
        _text(img, name, 16 + i * 80, 24, 0.6)
    items = ["New", "Open...", "Save", "Save as...", "Print", "Exit"]
    _rect(img, (10, 34, 240, len(items) * 30 + 8), _WHITE, (160, 160, 160))
    rects = {}
    for i, name in enumerate(items):
        r = (12, 38 + i * 30, 236, 30)
        rects[name] = r
        _text(img, name, 24, 38 + i * 30 + 21, 0.6)

    ctx = np.full((HEIGHT, WIDTH, 3), (250, 250, 250), np.uint8)
    _rect(ctx, (900, 500, 220, 6 * 28 + 8), _WHITE, (150, 150, 150))
    cmds = ["Cut", "Copy", "Paste", "Rename", "Delete", "Properties"]
    ctx_rects = {}
    for i, name in enumerate(cmds):
        ctx_rects[name] = (902, 504 + i * 28, 216, 28)
        _text(ctx, name, 914, 504 + i * 28 + 20, 0.55)

    return [
        Scene(
            "menu_save_as",
            "small",
            "In the open File menu, click the 'Save as...' entry.",
            img,
            rects["Save as..."],
        ),
        Scene(
            "menu_exit",
            "small",
            "In the open File menu, click the 'Exit' entry.",
            img,
            rects["Exit"],
        ),
        Scene(
            "context_delete",
            "small",
            "In the context menu, click the 'Delete' entry.",
            ctx,
            ctx_rects["Delete"],
        ),
    ]


def _icon_scenes() -> list[Scene]:
    out = []
    specs = [
        # name, origin, size, gap, icon index, goal
        ("icon_top_left", (30, 70), 32, 14, 3, "orange cross"),
        ("icon_top_right", (1520, 70), 28, 12, 4, "purple diamond"),
        ("icon_center", (820, 520), 36, 16, 2, "blue triangle"),
        ("icon_bottom_row", (700, 960), 24, 10, 1, "green square"),
    ]
    for name, origin, size, gap, idx, desc in specs:
        img = np.full((HEIGHT, WIDTH, 3), (232, 232, 236), np.uint8)
        _text(img, "Workspace", 760, 40, 0.8, (120, 120, 120))
        rects = _toolbar(img, origin, size, gap, _ICON_SET)
        out.append(
            Scene(
                name,
                "small",
                f"Click the {desc} icon in the row of icons.",
                img,
                rects[idx],
            )
        )
    return out


def _edge_scenes() -> list[Scene]:
    out = []

    # Chat window, message box on the bottom edge (the real failure).
    img = _desktop(0)
    win = (560, 180, 800, 640)
    _window(img, win, "Chat - Maria")
    for i in range(5):
        side = 590 if i % 2 == 0 else 900
        _rect(img, (side, 230 + i * 70, 340, 48), (225, 235, 250), (190, 190, 190))
        _text(img, f"Message {i + 1}", side + 14, 262 + i * 70, 0.6)
    msg = (576, 750, 600, 54)
    _rect(img, msg, _WHITE, (150, 150, 150))
    _text(img, "Type a message", 592, 784, 0.6, (150, 150, 150))
    send = (1190, 750, 150, 54)
    _button(img, send, "Send", (215, 120, 0))
    out.append(
        Scene(
            "edge_chat_message_box",
            "edge",
            "Click the message box at the bottom of the chat window.",
            img,
            msg,
        )
    )
    out.append(
        Scene(
            "edge_chat_send_button",
            "edge",
            "Click the blue 'Send' button at the bottom right of the chat window.",
            img,
            send,
        )
    )

    # Window close button in the top-right corner.
    img = _desktop(1)
    win = (900, 300, 700, 460)
    close = _window(img, win, "Settings")
    _text(img, "Preferences", 930, 380, 0.8)
    out.append(
        Scene(
            "edge_close_button",
            "edge",
            "Click the red close button in the top-right corner of the window.",
            img,
            close,
        )
    )

    # Window flush against the left edge of the screen, OK at its bottom edge.
    img = _desktop(2)
    win = (0, 400, 640, 380)
    _window(img, win, "Confirm")
    _text(img, "Save changes before closing?", 40, 520, 0.8)
    ok = (500, 720, 120, 44)
    _button(img, ok, "OK", (215, 120, 0))
    _button(img, (360, 720, 120, 44), "Discard", (130, 130, 130))
    out.append(
        Scene(
            "edge_dialog_ok_left",
            "edge",
            "Click the blue 'OK' button of the dialog.",
            img,
            ok,
        )
    )

    # Small dialog with its OK button right above the window's bottom border.
    img = _desktop(3)
    win = (710, 390, 500, 300)
    _window(img, win, "Warning")
    _text(img, "The file already exists.", 750, 500, 0.8)
    ok = (1060, 638, 130, 40)
    _button(img, ok, "Replace", (60, 60, 190))
    _button(img, (910, 638, 130, 40), "Keep both", (130, 130, 130))
    out.append(
        Scene(
            "edge_dialog_replace",
            "edge",
            "Click the red 'Replace' button of the warning dialog.",
            img,
            ok,
        )
    )

    # Start button in the screen's bottom-left corner.
    img = _desktop(0)
    out.append(
        Scene(
            "edge_start_button",
            "edge",
            "Click the blue start button at the left end of the taskbar.",
            img,
            (6, HEIGHT - 36, 36, 32),
        )
    )

    # Tab on the top edge of a non-maximized window.
    img = _desktop(1)
    win = (300, 150, 1100, 700)
    _window(img, win, "Browser")
    tabs = ["Home", "Settings", "Profile"]
    tab_rects = {}
    for i, name in enumerate(tabs):
        r = (308 + i * 150, 190, 144, 34)
        tab_rects[name] = r
        _rect(img, r, (235, 235, 235) if i else _WHITE, (170, 170, 170))
        _label_in(img, r, name)
    out.append(
        Scene(
            "edge_tab_settings",
            "edge",
            "Click the 'Settings' tab at the top of the window.",
            img,
            tab_rects["Settings"],
        )
    )
    return out


def _slider_scene() -> Scene:
    img = np.full((HEIGHT, WIDTH, 3), (238, 238, 238), np.uint8)
    _text(img, "Volume", 600, 520, 0.8)
    _rect(img, (700, 508, 520, 6), (170, 170, 170))
    _rect(img, (700, 508, 300, 6), (215, 120, 0))
    handle = (992, 500, 16, 22)
    _rect(img, handle, (215, 120, 0), (120, 70, 0))
    return Scene(
        "slider_handle",
        "small",
        "Click the handle of the Volume slider.",
        img,
        handle,
    )


def all_scenes() -> list[Scene]:
    """The full bench set, in a fixed order that mixes the kinds."""

    groups = [
        _notes_scenes(),
        _form_scenes(),
        _menu_scenes(),
        _icon_scenes(),
        _edge_scenes(),
        [_slider_scene()],
    ]
    # Interleave so that a truncated run (--scenes N) still covers every kind.
    ordered: list[Scene] = []
    for i in range(max(len(g) for g in groups)):
        ordered.extend(g[i] for g in groups if i < len(g))
    return ordered


__all__ = ["HEIGHT", "WIDTH", "Rect", "Scene", "all_scenes"]
