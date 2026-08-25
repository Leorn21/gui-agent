# GUI Agent

Repositorio base para un agente GUI modular en Python, orientado a observación de pantalla, etiquetado visual, planificación con VLM y automatización a nivel de sistema operativo.

## Estructura inicial

- `agent/`: orquestación, planner y prompts.
- `vision/`: captura de pantalla, tagger visual y diffs.
- `controller/`: abstracción de control del sistema y seguridad.
- `schemas/`: modelos tipados de acciones del agente.
- `main.py`: entrypoint del CLI.

## Estado actual

Este commit deja solo el andamiaje del proyecto. No hay implementación funcional todavía.

## Próximos pasos

1. Implementar primero `schemas/actions.py`.
2. Luego `vision/capture.py` y `vision/som_tagger.py`.
3. Después `controller/base.py` y `controller/os_controller.py`.
4. Finalmente `agent/planner.py`, `agent/state_machine.py` y `main.py`.
