# Plan de optimización: hacia un agente casi en tiempo real

Estado: plan aprobado para implementar por paquetes. Nada de esto está implementado.

Las latencias citadas son **estimaciones leídas del código, no mediciones**. El
paquete P0 existe para reemplazarlas por números reales.

## Límite realista

Con un VLM en la nube por paso, el piso es de alrededor de 0,5 a 1,5 s por
decisión. Optimizando el bucle actual se llega a tareas de escritorio fluidas y
a juegos por turnos o lentos. Para juegos de acción hace falta la arquitectura
de dos velocidades (fase 5), y aun así no alcanza para FPS o competitivos.

## Dónde se va el tiempo hoy

Un paso con click hace todo esto en serie (`agent/state_machine.py`, bucle de
`run`):

| Etapa | Costo estimado | Causa |
|---|---|---|
| Llamada `decide` | 1,5–4 s | Imagen de 1280 px, `thought` libre, `max_tokens` 4096 |
| Llamada `refine` (zoom) | 1–3 s | Segunda llamada completa por cada click; dos en un drag |
| Movimiento del mouse | 0,2 s | `DEFAULT_MOVE_DURATION` en `controller/os_controller.py` |
| Espera de cambio | hasta 2 s | Sondeo cada 0,1 s con tope de 2 s en `vision/diff.py`; si nada cambia se pagan los 2 s |
| PNG y grabación | 50–150 ms | Codificación PNG y `record_step` síncronos |

Total estimado: 4 a 9 s por click. El cuello de botella son las dos llamadas al
modelo.

## Fases

### Fase 0: medir

- **Tiempos por etapa:** captura, codificación, `decide`, `refine`, ejecución y
  espera, en `StepEvent` y en el registro de la corrida.
- **Banco de precisión:** sobre las escenas sintéticas de `tests/scenes.py`,
  mide si el click cae dentro del objetivo. Sin esto no se puede validar que
  las fases 1 a 3 no rompan la puntería.

### Fase 1: ajustes baratos (estimado: −30 a −40 %)

- **Imagen:** JPEG de calidad ~80 en vez de PNG, y probar 1024 px de ancho.
  Verificar con el banco que los números de la grilla sigan legibles.
- **Salida:** `thought` de una frase corta, `max_tokens` ~300 y quitar
  `thought` de `ZoomChoice`. Los tokens de salida son lo que más pesa.
- **Controlador:** `move_duration` y `type_interval` configurables por entorno,
  con 0 en un modo rápido.
- **Espera de cambio:** sondeo cada 30 ms y tope según la acción (corto para
  teclas, más largo para abrir programas).
- **Grabación:** escritura de imágenes en un hilo con cola.
- **Caché:** el prompt de sistema probablemente no llega al mínimo cacheable de
  Haiku, así que `cache_control` quizá no hace nada. Confirmar mirando
  `cache_read_input_tokens` en una corrida real.
- **Conexión:** crear el cliente de Anthropic al abrir la barra y reutilizarlo
  entre corridas, no uno por `_Session`.

### Fase 2: eliminar la segunda llamada (estimado: −40 % en clicks)

- **Zoom condicional (primero):** el modelo devuelve `needs_zoom`; si el
  objetivo es grande, se hace click en el centro de la celda sin `refine`.
- **Coordenadas directas (a evaluar):** pedir x,y en píxeles y comparar contra
  grilla más zoom en el banco. Contradice la decisión de grilla SoM: solo se
  adopta si el banco lo justifica y el usuario lo aprueba.
- **UI Automation (opcional):** marcas sobre elementos reales, click exacto sin
  zoom. No sirve en juegos.

### Fase 3: varias acciones por llamada (estimado: 2–3× menos llamadas)

- **Esquema:** `AgentPlan` con una lista de 1 a 5 acciones (por ejemplo win,
  escribir, enter).
- **Ejecución:** en serie, con verificación de cambio entre cada una; si la
  pantalla diverge de lo esperado, se corta el lote y se vuelve a percibir.
- **Regla de seguridad:** tras una acción que cambia la pantalla, dentro del
  mismo lote solo se permiten acciones sin objetivo (teclas, texto).

### Fase 4: percepción más barata

- **Recorte:** mandar solo la ventana activa o la región que cambió cuando el
  cambio es local.
- **Captura continua:** un hilo que mantiene el último cuadro, y Desktop
  Duplication (`dxcam`) en vez de `mss` para 60 fps o más.

### Fase 5: modo juego (dos velocidades)

- **Estratega lento (VLM, cada 1–3 s, asíncrono):** devuelve una política
  corta: teclas a mantener, macros con tiempos y la caja de un objetivo a
  seguir.
- **Reflejo rápido (local, 30–60 Hz):** ejecuta la política sobre el último
  cuadro y sigue el objetivo con seguimiento local (plantilla o color).
- **Controlador:** `key_down`, `key_up`, mantener por milisegundos y movimiento
  relativo del mouse por `SendInput` con scan codes.
- **Llamadas solapadas:** lanzar la siguiente consulta antes de que termine la
  anterior y descartar respuestas viejas.
- **Seguridad:** el interruptor se chequea en el bucle rápido; al pausar o
  abortar se sueltan todas las teclas mantenidas.
- **Alcance:** solo juegos de un jugador.

## Metas estimadas

| Después de | Paso con click | Sirve para |
|---|---|---|
| Hoy | 4–9 s | Tareas de escritorio lentas |
| Fases 1–2 | 1,5–3 s | Escritorio fluido, juegos por turnos |
| Fase 3 | ~1 s efectivo por acción | Puzzles, point-and-click, estrategia lenta |
| Fase 5 | Reacción local 20–50 ms, estrategia cada 1–3 s | Arcade simple, plataformas lentos |

## Paquetes de trabajo

Reglas para todos los paquetes:

- Una rama `feat/...` y un PR a `main` por paquete; el usuario hace el merge.
- Tests primero (TDD), con mocks; tests de integración opcionales.
- Commits atómicos con formato convencional, sin atribución a Claude.
- No tocar archivos fuera de los listados sin avisar.
- Verificar los detalles de la API de Anthropic en la documentación antes de
  implementar (salida estructurada, mínimos de caché, formatos de imagen).
- Criterio común: tests en verde, el banco de precisión (P1) no baja respecto
  de la línea base, y los tiempos de P0 muestran la mejora en una corrida real.

| # | Paquete | Archivos | Depende de | Modelo sugerido |
|---|---|---|---|---|
| P0 | Tiempos por etapa | `agent/state_machine.py`, `agent/recorder.py` | — | Haiku |
| P1 | Banco de precisión | `tests/`, script nuevo | — | Sonnet |
| P2 | Controlador rápido | `controller/os_controller.py`, `main.py` | — | Haiku |
| P3 | Espera de cambio adaptativa | `vision/diff.py` | — | Haiku |
| P4 | Grabación asíncrona | `agent/recorder.py` | P0 | Haiku |
| P5 | JPEG, salida corta, cliente reutilizado | `agent/planner.py`, `agent/prompts.py`, `main.py` | P1, P2 | Haiku |
| P6 | Zoom condicional | `schemas/actions.py`, `agent/prompts.py`, `agent/planner.py`, `agent/state_machine.py` | P0, P1, P5 | Sonnet |
| P7 | Lotes de acciones | los mismos que P6 | P6 | Sonnet |
| P8 | Captura continua y `dxcam` | `vision/capture.py`, módulo nuevo | — | Sonnet |
| P9 | Entradas de juego y soltar teclas | `controller/os_controller.py`, `controller/safety.py`, `schemas/actions.py` | P2, P7 | Sonnet |
| P10 | Bucle de dos velocidades | `agent/realtime.py` nuevo, `main.py` | P7, P8, P9 | Sonnet, sobre el diseño de C3 |

P0, P1, P2, P3 y P8 pueden correr en paralelo desde el inicio: no comparten
archivos. El resto va en serie según la columna de dependencias (varias
dependencias existen solo porque dos paquetes tocan el mismo archivo).

## Orquestación

El orquestador es **Opus**. Su trabajo es mecánico: lanzar un subagente por
paquete con el modelo de la tabla, respetar las dependencias, verificar que el
PR cumple el criterio de aceptación y frenar para que el usuario haga el merge.
El orquestador no implementa paquetes ni toma decisiones de diseño.

Cada subagente recibe como instrucción: este archivo, el número de su paquete,
los archivos que puede tocar y su criterio de aceptación.

Tres tareas son de criterio y el orquestador las lanza con **Fable 5.2**
(`claude-fable-5-2`). Son puntos de control: el trabajo que depende de ellas no
arranca hasta que terminan y el usuario aprueba el resultado.

| Punto de control | Cuándo | Qué hace el subagente Fable | Qué bloquea |
|---|---|---|---|
| C1: replanteo con mediciones | Al terminar P0 y P1 | Lee los tiempos reales y la línea base de precisión, corrige las estimaciones de este plan y propone cambios de orden o de alcance. Solo edita este archivo. | P5 en adelante |
| C2: revisión de P6 y P7 | Con el PR de cada uno abierto | Revisa el diff buscando errores en el esquema, los prompts, el bucle y la regla de seguridad del lote. Reporta hallazgos; no implementa. | El merge de P6 y de P7 |
| C3: diseño de P10 | Antes de implementar P10 | Escribe el diseño del bucle de dos velocidades: hilos, formato de la política, llamadas solapadas, teclas mantenidas e interruptor de seguridad. | La implementación de P10 |

P2, P3 y P4 son cambios chicos y aislados: el subagente los resuelve directo
desde este plan, sin revisión de Fable.

### Aceptación por paquete

- **P0:** cada `StepEvent` trae los tiempos por etapa; `runs/<corrida>` los
  guarda; una corrida real imprime el desglose.
- **P1:** un comando corre N escenas y reporta tasa de acierto y distancia
  media al objetivo; se guarda la línea base actual.
- **P2:** `GUI_AGENT_MOVE_DURATION` y `GUI_AGENT_TYPE_INTERVAL` funcionan; con
  0 el mouse salta y el interruptor de seguridad sigue respondiendo.
- **P3:** el tope y el intervalo dependen del tipo de acción; un paso sin
  cambio de una tecla espera bastante menos de 2 s.
- **P4:** `record_step` vuelve sin escribir a disco; `finish` espera a que la
  cola se vacíe; no se pierden imágenes al abortar.
- **P5:** formato y ancho de imagen configurables; precisión del banco sin
  caída; latencia de `decide` menor en P0; se documenta si la caché acierta.
- **P6:** los clicks sobre objetivos grandes hacen una sola llamada; precisión
  del banco sin caída.
- **P7:** "abrir un programa" (win, texto, enter) sale en una llamada; un lote
  se corta si la pantalla diverge.
- **P8:** un hilo entrega el último cuadro sin bloquear; se mide el fps de
  captura; `mss` queda como respaldo.
- **P9:** teclas mantenidas y mouse relativo funcionan; pausar o abortar suelta
  todo lo mantenido (con test).
- **P10:** demo en un juego simple de un jugador con el bucle rápido a 30 Hz o
  más y el interruptor de seguridad operativo.
