"""Synthetic screens for tests and figures (no real screen content).

``notes_app()`` draws a fake 1920x1080 notes application: a title bar, a
sidebar, a toolbar with a blue "Guardar" button and an editor.
"""

import cv2
import numpy as np

from vision.capture import Frame

WIDTH, HEIGHT = 1920, 1080
SAVE_BUTTON = (330, 78, 140, 36)  # x, y, w, h in screen pixels
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, s, x, y, scale=0.8, color=(40, 40, 40)):
    cv2.putText(img, s, (x, y), _FONT, scale, color, 2, cv2.LINE_AA)


def _button(img, x, y, w, h, label, fill):
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1)
    (tw, th), _ = cv2.getTextSize(label, _FONT, 0.7, 2)
    _text(img, label, x + (w - tw) // 2, y + (h + th) // 2, 0.7, (255, 255, 255))


def notes_app() -> np.ndarray:
    """BGR image of the fake notes app at full resolution."""

    img = np.full((HEIGHT, WIDTH, 3), (235, 235, 235), np.uint8)
    cv2.rectangle(img, (0, 0), (WIDTH, 60), (60, 50, 45), -1)
    _text(img, "Notas - documento_sin_titulo.txt", 24, 40, 0.9, (240, 240, 240))
    cv2.rectangle(img, (0, 60), (300, HEIGHT - 50), (250, 248, 245), -1)
    for i, name in enumerate(["Inicio", "Recientes", "Favoritos", "Papelera"]):
        _text(img, name, 40, 120 + i * 60)
    cv2.rectangle(img, (300, 60), (WIDTH, 130), (245, 245, 245), -1)
    _button(img, *SAVE_BUTTON, "Guardar", (215, 120, 0))
    _button(img, 490, 78, 120, 36, "Abrir", (120, 120, 120))
    _button(img, 630, 78, 140, 36, "Exportar", (120, 120, 120))
    cv2.rectangle(img, (340, 170), (1860, 960), (255, 255, 255), -1)
    for i, line in enumerate(["Hola Mundo", "Lista de tareas:"]):
        _text(img, line, 370, 220 + i * 46, 0.9)
    cv2.rectangle(img, (0, HEIGHT - 50), (WIDTH, HEIGHT), (45, 45, 45), -1)
    return img


def notes_app_frame(max_width: int = 1280) -> Frame:
    """The fake app as ScreenCapture would return it (downscaled)."""

    img = notes_app()
    scale = max_width / WIDTH
    small = cv2.resize(
        img, (max_width, round(HEIGHT * scale)), interpolation=cv2.INTER_AREA
    )
    return Frame(image=small, original_size=(WIDTH, HEIGHT), scale=scale, origin=(0, 0))


def region_frame(x: int, y: int, width: int, height: int) -> Frame:
    """A full-resolution region of the fake app, as capture_region() returns it."""

    img = notes_app()[y : y + height, x : x + width].copy()
    return Frame(image=img, original_size=(width, height), scale=1.0, origin=(x, y))


def inside_save_button(x: int, y: int) -> bool:
    bx, by, bw, bh = SAVE_BUTTON
    return bx <= x < bx + bw and by <= y < by + bh
