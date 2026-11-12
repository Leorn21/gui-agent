# Plan de optimización: hacia un agente casi en tiempo real

Estado: P0, P1, P2, P3 y P8 están integrados en `main`. El punto de control C1
(replanteo con mediciones) revisó este archivo; **los paquetes nuevos (P11 a
P13) y P4 en adelante esperan la aprobación del usuario sobre C1**. Los cambios de C1 están resumidos al final.

Las latencias marcadas como **medido** salen de una corrida real y del banco de
precisión (ver "Mediciones"). Todo lo demás sigue siendo **estimación** y está
rotulado así.

## Mediciones (base de C1)

- **Corrida real:** Windows 11, 1920×1080, `claude-sonnet-5-5` con esfuerzo
  `low`, objetivo "enviar un mensaje corto en una app de chat". 12 pasos
  (9 clicks, 1 texto, 2 teclas), abortada por el usuario, US$ 0,097
  (~US$ 0,008 por paso, 21 llamadas). No terminó la tarea: 4 de los 9 clicks
  cayeron en la barra de tareas (ver "Fallo del zoom en bordes").
- **Banco de precisión** (`bench/baseline.json`): 21 escenas sintéticas, mismo
  modelo y esfuerzo. 19/21 aciertos (90 %), distancia mediana 12 px, media
  39 px, 42 llamadas, US$ 0,148. Sin el zoom acertarían solo 7/21 (3/5
  grandes, 4/7 de borde, 0/9 chicos).
- **Límites de estas mediciones:** una sola corrida real y una sola pasada del
  banco (no se midió la varianza; una escena equivale a 4,8 puntos). La corrida
  no guarda tokens por llamada ni las etapas `record` y `total` (hueco de P0,
  se corrige en P4). El banco no mide latencia. No hay tiempos medidos con
  Haiku.

## Límite realista

Con un VLM en la nube por paso, el piso estimado es de 0,5 a 1,5 s por
decisión. Medido hoy con Sonnet `low`: `decide` nunca bajó de 1,6 s y `refine`
nunca bajó de 2,4 s. Optimizando el bucle actual se llega a tareas de
escritorio fluidas y a juegos por turnos o lentos. Para juegos de acción hace
falta la arquitectura de dos velocidades (fase 5), y aun así no alcanza para
FPS o competitivos.

## Dónde se va el tiempo hoy (medido)

Un paso con click hace todo esto en serie (`agent/state_machine.py`, bucle de
`run`):

| Etapa | Medido | Notas |
|---|---|---|
| Captura y grilla | ~0 s (3–4 ms; 42 ms la primera) | El cuadro se reutiliza de la espera anterior |
| Llamada `decide` | 1,6–3,6 s, media 2,35 s | 43 % del tiempo total |
| Llamada `refine` (zoom) | 2,4–3,6 s, media 2,75 s por click | 38 % del total. Más lenta que `decide` pese a la imagen chica y la respuesta mínima |
| Ejecución | 0,2 s por click; 0,5 s el texto; ~0 las teclas | El 0,2 s es `DEFAULT_MOVE_DURATION`; P2 ya permite 0 por `.env` |
| Espera con cambio | 0,24–0,77 s, media 0,44 s | Es sobre todo el tiempo de reacción de la interfaz |
| Espera sin cambio | 2,0–2,1 s (3 de 12 pasos) | 61 % de todo el tiempo de espera. P3 no está conectado al bucle |
| PNG y grabación | sin medir | `record` y `total` no llegan a `steps.jsonl` |

Paso con click: **4,8 a 7,2 s, media 5,9 s**. Paso de texto o tecla: 3,0 a
5,3 s. Las dos llamadas al modelo son el 81 % del tiempo; la espera, el 15 %;
la ejecución, el 4 %.

Conclusiones que cambian el plan:

- **`refine` cuesta tanto como `decide`.** Su latencia parece dominada por el
  costo fijo de la llamada, no por los tokens (inferencia: no hay tokens por
  llamada guardados). Eliminar llamadas rinde más que achicarlas.
- **La percepción local ya es gratis.** Ni `dxcam` ni el recorte mejoran el
  bucle actual por el lado de la captura.
- **La espera solo se puede recortar cuando no hay cambio.** Pasar el sondeo de
  100 a 30 ms ahorra como mucho ~70 ms por paso.
- **El primer bloqueo es de precisión, no de velocidad.** La corrida real no
  terminó por clicks mal ubicados, no por lentitud.

## Fallo del zoom en bordes

Visto dos veces en corridas reales. La captura y el mapeo de coordenadas se
verificaron y son correctos: el error es de elección de la celda fina.

- **Ventana no maximizada:** la celda gruesa era correcta, pero `refine` eligió
  una celda fina del margen, ~60 px debajo de la ventana. Los clicks cayeron en
  el escritorio y la corrida se trabó. El banco lo reproduce
  (`edge_chat_message_box`: celda de la última fila del zoom, 40 px debajo del
  objetivo).
- **Ventana maximizada:** el campo de mensaje está en la última fila de la
  grilla, pegado a la barra de tareas. La celda gruesa (226) es mitad campo y
  mitad barra. La región del zoom se recorta contra el borde inferior de la
  pantalla (135 px de alto en vez de 180), las celdas finas quedan de ~17 px y
  las etiquetas tapan el campo, que es fino y oscuro. El modelo eligió celdas
  de la barra (y 1054 y 1071), se perdió el foco y se abrió el menú Inicio.

Dato para el diseño del arreglo: 4 de los 19 aciertos del banco caen **fuera**
de la celda gruesa, dentro del margen. El margen es necesario; no se puede
limitar la elección a la celda gruesa.

El otro fallo del banco (`icon_top_right`, ícono de 28 px, error de 8 px) es de
resolución: la celda fina mide 24×22,5 px y el click va a su centro.

## Fases

### Fase 0: medir (hecha)

- **Tiempos por etapa:** P0, integrado. Falta `record` y `total` en
  `steps.jsonl` (pasa a P4) y los tokens por llamada (pasa a P5).
- **Banco de precisión:** P1, integrado. Falta medir latencia por llamada y
  cubrir los bordes de pantalla (pasa a P12).

### Fase 0b: precisión y cableado (nueva, antes de la fase 1)

- **Conectar la espera adaptativa:** el bucle llama a `wait_for_change` sin
  `**wait_profile(action.action_type)`. Es un cambio de una línea. Ojo: el
  perfil actual sube el tope de CLICK de 2 a 3 s, lo que empeora el click sin
  cambio; los clicks con cambio tardaron como mucho 0,77 s.
- **Zoom en bordes:** mitigaciones a evaluar por separado en el banco, en este
  orden, quedándose solo con las que mejoran:
  1. Desplazar la región del zoom hacia adentro en vez de recortarla contra el
     borde de la pantalla, para que las celdas finas no se achiquen.
  2. Marcar en la imagen del zoom el contorno de la celda gruesa elegida y
     decir en el prompt que el objetivo se ubicó ahí y que el margen es
     contexto.
  3. Etiquetas más livianas en la grilla del zoom (más chicas o translúcidas),
     para no tapar elementos finos.
  4. Regla general en el prompt del zoom: elegir una celda sobre el elemento,
     nunca más allá del borde de su ventana; si el objetivo no se ve, no
     elegir. Una regla específica sobre la barra de tareas queda como último
     recurso: arregla un síntoma de un solo sistema.
- **Coordenadas directas (experimento):** medir en el banco, sin tocar el
  agente, si pedir x,y en píxeles iguala a grilla más zoom. Se adelanta porque
  su resultado define el diseño de la fase 2.

### Fase 1: ajustes baratos (estimado: −10 a −25 % por paso)

La estimación original era −30 a −40 %. Se baja porque lo medido indica que la
latencia de las llamadas depende poco del tamaño de la entrada y la salida.

- **Imagen:** JPEG de calidad ~80 en vez de PNG, y probar 1024 px de ancho.
  Verificar con el banco que los números de la grilla sigan legibles, en
  especial en las escenas de borde.
- **Salida:** `thought` de una frase corta y `max_tokens` ~300 en `decide`. El
  costo por llamada (~US$ 0,0046) sugiere que la salida ya es corta, así que
  la ganancia esperada es chica (estimación). **No se quita `thought` de
  `ZoomChoice` en este paso:** con el fallo de bordes abierto, ese campo puede
  estar ayudando a elegir. Se prueba aparte, con y sin, y solo se quita si el
  banco ampliado no baja en dos pasadas.
- **Controlador:** hecho (P2). Falta usar 0 en `.env` para el modo rápido.
- **Espera de cambio:** hecho en `vision/diff.py` (P3); se conecta en P11.
- **Grabación:** escritura de imágenes en un hilo con cola.
- **Caché:** **sí acierta en `decide`**: una corrida de prueba con Haiku mostró
  `cache_read_input_tokens=2177`. El prompt del zoom es demasiado corto para
  cachearse. El mínimo cacheable puede ser distinto en Sonnet: verificarlo en
  la documentación y en `usage`, no asumirlo. El ahorro es sobre todo de costo.
- **Conexión:** crear el cliente de Anthropic al abrir la barra y reutilizarlo
  entre corridas, no uno por `_Session`.

### Fase 2: eliminar la segunda llamada (estimado: −2,75 s en cada click que la evita)

- **Zoom condicional:** el modelo devuelve `needs_zoom`; si el objetivo es
  grande, se hace click en el centro de la celda sin `refine`. El banco le pone
  techo: sin zoom aciertan 7/21, y dos botones de 36 px de alto fallan. Solo
  sirve para objetivos más grandes que una celda (~96×90 px). Ganancia media
  estimada: −15 a −20 % en clicks, no −40 %.
- **Coordenadas directas:** si el experimento P13 iguala al zoom, evita
  `refine` en todos los clicks y pasa a ser la opción principal. Contradice la
  decisión de grilla SoM: solo se adopta si el banco lo justifica y el usuario
  lo aprueba.
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
  cambio es local. La captura ya no cuesta; la ganancia posible es por menos
  tokens de imagen en `decide`, y está sin medir.
- **Captura continua:** hecha (P8, `vision/live_capture.py`: `dxcam` ~91 fps
  contra `mss` ~39 fps). No está conectada al bucle y no hace falta conectarla
  hasta P10: en el bucle actual la captura no es un costo.

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

## Metas

Solo la primera fila es medida; el resto son estimaciones a partir de ella.

| Después de | Paso con click | Paso de tecla o texto | Sirve para |
|---|---|---|---|
| Hoy (medido) | 4,8–7,2 s, media 5,9 s | 3,0–5,3 s | Tareas de escritorio lentas |
| P11 y modo rápido en `.env` | 4,6–7 s | 2,5–3,5 s | Lo mismo, con menos espera muerta |
| P5 | 4–6 s | 2–3 s | Lo mismo |
| P6, zoom condicional | 2–3,5 s sin zoom; igual que antes con zoom; media ~4–5 s | 2–3 s | Escritorio aceptable |
| P6, si se adoptan coordenadas directas | 2–3,5 s en todos los clicks | 2–3 s | Escritorio fluido, juegos por turnos |
| Fase 3 | — | ~1–1,5 s efectivo por acción dentro de un lote | Puzzles, point-and-click, estrategia lenta |
| Fase 5 | Reacción local 20–50 ms, estrategia cada 1–3 s | — | Arcade simple, plataformas lentos |

La meta original de 1,5 a 3 s por click tras las fases 1 y 2 no se alcanza solo
con zoom condicional: depende de que la mayoría de los clicks evite `refine`.

## Paquetes de trabajo

Reglas para todos los paquetes:

- Una rama `feat/...` y un PR a `main` por paquete; el usuario hace el merge.
- Tests primero (TDD), con mocks; tests de integración opcionales.
- Commits atómicos con formato convencional, sin atribución a Claude.
- No tocar archivos fuera de los listados sin avisar.
- Verificar los detalles de la API de Anthropic en la documentación antes de
  implementar (salida estructurada, mínimos de caché, formatos de imagen).
- Criterio común: tests en verde, el banco de precisión no baja respecto de la
  línea base vigente, y los tiempos de P0 muestran la mejora en una corrida
  real.
- **"No baja" significa:** en dos pasadas del banco, los aciertos totales no
  caen más de una escena y ninguna escena de borde que acertaba pasa a fallar.
  Se compara tasa de acierto y distancia mediana; la media no sirve porque los
  objetivos grandes la inflan.
- **Acoplamiento con el banco:** `bench/precision.py` llama al método privado
  `AgentStateMachine._locate`. Ya rompió una vez (P0 cambió su firma, arreglo
  en #18). Todo paquete que toque `_locate`, `_point` o `_zoom_region` incluye
  `bench/precision.py` y `tests/test_bench_precision.py` entre sus archivos y
  corre esos tests.

La columna "Modelo sugerido" es el modelo del **subagente que implementa** el
paquete. No es el VLM del agente: ese es configurable y hoy corre con Sonnet
5.5 por decisión del usuario.

| # | Paquete | Archivos | Depende de | Modelo sugerido | Estado |
|---|---|---|---|---|---|
| P0 | Tiempos por etapa | `agent/state_machine.py`, `agent/recorder.py` | — | Haiku | Integrado (#16) |
| P1 | Banco de precisión | `tests/`, `bench/` | — | Sonnet | Integrado (#17, arreglo #18) |
| P2 | Controlador rápido | `controller/os_controller.py`, `main.py` | — | Haiku | Integrado (#13) |
| P3 | Espera de cambio adaptativa | `vision/diff.py` | — | Haiku | Integrado (#14), sin conectar |
| P8 | Captura continua y `dxcam` | `vision/live_capture.py` | — | Sonnet | Integrado (#15), sin conectar |
| P11 | Conectar la espera adaptativa | `agent/state_machine.py`, `vision/diff.py` (solo el tope de CLICK), sus tests | P3 | Haiku | Nuevo |
| P12 | Zoom en bordes y banco ampliado | `agent/state_machine.py`, `vision/som_tagger.py`, `agent/prompts.py`, `agent/planner.py`, `tests/bench_scenes.py`, `bench/precision.py`, `bench/baseline.json`, sus tests | P1, P11 | Sonnet | Nuevo |
| P13 | Experimento de coordenadas directas | `bench/direct_coords.py` nuevo y su test | P12 | Sonnet | Nuevo |
| P4 | Grabación asíncrona | `agent/recorder.py`, `agent/state_machine.py` (solo el registro de `record` y `total`) | P0, P12 | Haiku | Pendiente |
| P5 | JPEG, salida corta, cliente reutilizado | `agent/planner.py`, `agent/prompts.py`, `agent/recorder.py`, `main.py` | P2, P4, P12 | Haiku | Pendiente |
| P6 | Zoom condicional o coordenadas directas | `schemas/actions.py`, `agent/prompts.py`, `agent/planner.py`, `agent/state_machine.py`, `bench/precision.py` | P5, P13 | Sonnet | Pendiente |
| P7 | Lotes de acciones | los mismos que P6 | P6 | Sonnet | Pendiente |
| P9 | Entradas de juego y soltar teclas | `controller/os_controller.py`, `controller/safety.py`, `schemas/actions.py` | P2, P7 | Sonnet | Pendiente |
| P10 | Bucle de dos velocidades | `agent/realtime.py` nuevo, `main.py` | P7, P8, P9 | Sonnet, sobre el diseño de C3 | Pendiente |

Orden propuesto: **P11 → P12 → P13 → P4 → P5 → P6 → P7 → P9 → P10**. Casi todo
va en serie: P11, P12, P4, P6 y P7 tocan `agent/state_machine.py`; P4 y P5 tocan
`agent/recorder.py`. P13 solo crea archivos nuevos y puede correr en paralelo
con P4 y P5. Conectar P8 al bucle queda dentro de P10.

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
| C1: replanteo con mediciones | Al terminar P0 y P1 (hecho, falta aprobar) | Lee los tiempos reales y la línea base de precisión, corrige las estimaciones de este plan y propone cambios de orden o de alcance. Solo edita este archivo. | P5 en adelante, los paquetes nuevos P11 a P13 y P4 (cambió su alcance) |
| C2: revisión de P6 y P7 | Con el PR de cada uno abierto | Revisa el diff buscando errores en el esquema, los prompts, el bucle y la regla de seguridad del lote. Reporta hallazgos; no implementa. | El merge de P6 y de P7 |
| C3: diseño de P10 | Antes de implementar P10 | Escribe el diseño del bucle de dos velocidades: hilos, formato de la política, llamadas solapadas, teclas mantenidas e interruptor de seguridad. | La implementación de P10 |

Además hay una **decisión del usuario antes de P6**: con el informe de P13 a la
vista, elige entre zoom condicional y coordenadas directas. No requiere un
subagente Fable.

P2, P3, P4 y P11 son cambios chicos y aislados: el subagente los resuelve
directo desde este plan, sin revisión de Fable.

### Aceptación por paquete

- **P0:** cada `StepEvent` trae los tiempos por etapa; `runs/<corrida>` los
  guarda; una corrida real imprime el desglose. (Cumplido salvo `record` y
  `total` en `steps.jsonl`.)
- **P1:** un comando corre N escenas y reporta tasa de acierto y distancia
  media al objetivo; se guarda la línea base actual. (Cumplido.)
- **P2:** `GUI_AGENT_MOVE_DURATION` y `GUI_AGENT_TYPE_INTERVAL` funcionan; con
  0 el mouse salta y el interruptor de seguridad sigue respondiendo.
  (Cumplido.)
- **P3:** el tope y el intervalo dependen del tipo de acción; un paso sin
  cambio de una tecla espera bastante menos de 2 s. (Cumplido en
  `vision/diff.py`; en el bucle lo cumple P11.)
- **P11:** el bucle pasa `wait_profile` del tipo de acción a `wait_for_change`
  (con test); el tope de CLICK no supera los 2 s de hoy; en una corrida real,
  un paso de tecla o texto sin cambio espera ~0,5 s en vez de ~2 s.
- **P12:**
  - El banco suma al menos 4 escenas de borde de pantalla: un campo fino y
    oscuro en la última fila de la grilla, pegado a una barra de tareas de
    ~48 px en una ventana maximizada, y objetivos contra los bordes superior,
    izquierdo y derecho.
  - La región del zoom mantiene su tamaño en los bordes de la pantalla y nunca
    sale de ella (con test).
  - En dos pasadas aciertan las escenas nuevas y `edge_chat_message_box`, y las
    21 escenas originales no bajan de 19 aciertos.
  - El PR informa el resultado de cada mitigación por separado y deja solo las
    que mejoran.
  - El banco guarda la latencia de `decide` y `refine` por escena y deja de
    depender de un método privado del bucle.
  - Se guarda la línea base nueva.
  - En una corrida real, la tarea de enviar un mensaje con la ventana
    maximizada termina sin clicks en la barra de tareas.
- **P13:** un comando corre las mismas escenas pidiendo x,y directos en una
  sola llamada y reporta aciertos, distancia mediana, latencia y costo junto a
  la línea base de P12. No cambia el agente. El informe termina en una
  recomendación para la decisión del usuario.
- **P4:** `record_step` vuelve sin escribir a disco; `finish` espera a que la
  cola se vacíe; no se pierden imágenes al abortar; `steps.jsonl` incluye
  `record` y `total`.
- **P5:** formato y ancho de imagen configurables; precisión del banco sin
  caída; latencia de `decide` y `refine` menor en el banco y en una corrida
  real; `steps.jsonl` guarda los tokens de entrada, salida y caché por llamada;
  se documenta si la caché acierta en `decide` con el modelo en uso;
  `ZoomChoice` conserva `thought` salvo que la prueba con y sin lo justifique.
- **P6:** los clicks que no necesitan zoom hacen una sola llamada; precisión
  del banco sin caída; el PR informa qué fracción de las escenas evitó
  `refine`.
- **P7:** "abrir un programa" (win, texto, enter) sale en una llamada; un lote
  se corta si la pantalla diverge.
- **P8:** un hilo entrega el último cuadro sin bloquear; se mide el fps de
  captura; `mss` queda como respaldo. (Cumplido.)
- **P9:** teclas mantenidas y mouse relativo funcionan; pausar o abortar suelta
  todo lo mantenido (con test).
- **P10:** demo en un juego simple de un jugador con el bucle rápido a 30 Hz o
  más y el interruptor de seguridad operativo.

## Cambios de C1

- **Latencias medidas en lugar de estimadas.** La tabla de etapas, el límite
  realista y las metas usan la corrida real. El paso con click mide 4,8 a
  7,2 s (la estimación era 4 a 9 s).
- **`refine` pesa más de lo previsto** (2,75 s por click contra 1 a 3 s
  estimados, y más que `decide`). Por eso se bajó la ganancia esperada de la
  fase 1 (de −30/−40 % a −10/−25 %) y se la concentró en eliminar llamadas.
- **Zoom condicional con techo.** El banco muestra que solo 7/21 clicks
  aciertan sin zoom, así que su ganancia media se bajó de −40 % a −15/−20 %.
- **Fase 0b nueva, antes de P5 y P6:** P11 (conectar la espera de P3 al bucle,
  una línea), P12 (arreglo del zoom en bordes con escenas nuevas) y P13
  (experimento de coordenadas directas, solo en el banco). Motivo: la corrida
  real falló por precisión, y optimizar sobre una línea base que no cubre ese
  fallo arriesga empeorarlo sin verlo.
- **P13 adelantado.** La evaluación de coordenadas directas ya estaba en la
  fase 2; pasa a ser un experimento previo a P6 porque su resultado decide el
  diseño de P6. Sigue requiriendo aprobación del usuario para adoptarse.
- **Caché corregida:** acierta en `decide` (visto con Haiku); el prompt del
  zoom es demasiado corto. Falta verificar el mínimo de Sonnet.
- **`thought` de `ZoomChoice` no se quita en P5.** Se decide con una prueba en
  el banco ampliado, después de P12.
- **"Modelo sugerido" aclarado:** es el modelo del subagente que implementa, no
  el VLM del agente.
- **Regla de la barra de tareas degradada** a último recurso, a favor de una
  regla general sobre bordes de ventana. No se limita la elección a la celda
  gruesa porque 4 aciertos del banco usan el margen.
- **Tope de CLICK en P11:** el perfil de P3 lo sube de 2 a 3 s; se pide que no
  empeore lo actual.
- **Criterio "no baja" definido** (dos pasadas, tolerancia de una escena,
  mediana en vez de media), porque la línea base es una sola pasada de 21
  escenas.
- **Riesgo de acoplamiento anotado:** el banco usa `_locate`, que es privado.
  P12 lo desacopla y P6 y P7 incluyen el banco entre sus archivos.
- **P4 y P5 amplían su aceptación:** `record` y `total` en `steps.jsonl` (P4) y
  tokens por llamada (P5), para poder atribuir la latencia.
- **P8 no se conecta al bucle por ahora:** la captura cuesta ~0 s; queda para
  P10.
- **Fuera de alcance, anotado:** en la corrida real los clicks en la barra de
  tareas cambiaban la pantalla, así que el contador de "trabado" no saltó. El
  arreglo de precisión ataca la causa; revisar esa detección queda pendiente.
