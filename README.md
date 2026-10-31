# Motor de Reposición de Inventario

Producto de ML que transforma las ventas diarias de una cadena de supermercados en una
**recomendación de pedido semanal** por tienda y familia de producto, con seguimiento del
desempeño de los pronósticos ya publicados.

> Laboratorio 1 — Product Development, Maestría en Data Science, Universidad Galileo.
> Datos: [Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
> (Corporación Favorita, Ecuador). 54 tiendas × 33 familias = **1 782 series**.

---

## Qué decide el producto

La pregunta de negocio no es "¿cuánto se va a vender?" sino **"¿cuánto hay que pedir?"**, y no
son la misma pregunta. Pedir la venta esperada es la decisión equivocada en casi todas las
familias:

- Si falta producto se pierde el margen y, a veces, el cliente.
- Si sobra se pierde capital, espacio de góndola y, en frescos, el producto entero.

Esos dos costos casi nunca son iguales, así que el pedido óptimo **no es el pronóstico central**
sino un cuantil de la distribución de demanda — el del vendedor de periódicos:

$$q^\* = \frac{c_u}{c_u + c_o}$$

Por eso el modelo no estima una media: estima **cuantiles** (0,5 · 0,8 · 0,9 · 0,95), y la
política elige cuál usar según los costos de cada familia. Con `costo_quiebre = 3` y
`costo_merma = 1` el objetivo es el 0,75; en `PRODUCE`, que se descompone en días, baja al 0,43.

Todos esos números viven en [`config/costs.yaml`](config/costs.yaml). **Cambiar un costo cambia
la recomendación sin tocar una línea de Python ni reentrenar el modelo** — ver
[la demostración](#cambiar-un-costo-sin-tocar-código).

## Las tres salidas

| Salida | Dónde | Para quién |
|---|---|---|
| Pronóstico cuantílico diario por tienda × familia | `data/pronosticos/asof=<fecha>/` | insumo interno |
| Pedido sugerido de la semana | `data/pedidos/asof=<fecha>/` (Parquet + CSV) | gerente de tienda y centro de distribución |
| Métricas y decisión de promoción del modelo | `reportes/asof=<fecha>/` (en Git) | quien mantiene el modelo |

---

## Las cinco reglas que no se negocian

Están completas en [`CLAUDE.md`](CLAUDE.md) y son el contrato del proyecto. En resumen:

1. **Corrección punto-en-el-tiempo.** Un pronóstico para la fecha `D` sólo puede usar lo que
   *había llegado* en `D`. Toda lectura filtra por `arrival_date`, nunca por `date`. No existe
   `datetime.now()` en `src/` ni en `dags/`, y hay una prueba que falla si aparece.
2. **La bandeja es un sistema externo de sólo lectura.** El producto lee `landing/` y jamás la
   escribe. El único autorizado es el simulador, que no es parte del producto.
3. **Idempotencia.** Correr una tarea N veces da el mismo resultado que correrla una vez:
   escritura atómica, particiones reescritas completas, nunca *append*.
4. **El histórico es inmutable.** Una corrección llega como registro nuevo con `arrival_date`
   posterior; la vista vigente se resuelve en lectura. Así se puede responder "¿qué sabíamos el
   15 de julio?" meses después.
5. **Dato inválido ≠ situación de negocio.** Una columna faltante va a cuarentena y la tarea
   falla. Unas ventas en cero por feriado, una tienda nueva o una fuente que no llegó hoy se
   procesan con normalidad y **nunca** hacen fallar el DAG.

La distinción entre `date` y `arrival_date` es la que sostiene todo lo demás: `date` es *a qué
día se refieren los datos*, `arrival_date` es *cuándo los supimos*.

---

## Puesta en marcha

Requisitos: [uv](https://docs.astral.sh/uv/), Git y ~1,5 GB de disco. No hace falta Docker para
nada de lo que sigue.

```bash
uv sync                      # crea el entorno a partir de uv.lock
uv run pytest                # 237 pruebas, sin Airflow y sin Docker
uv run ruff check .
```

Los CSV de Kaggle no están en Git (se versionan con DVC). Hay dos caminos:

```bash
uv run dvc pull              # si tienes acceso a la caché o al remoto
```

o bien descargarlos de Kaggle y colocarlos en `data/kaggle/` (`train.csv`, `stores.csv`,
`oil.csv`, `holidays_events.csv`, `transactions.csv`). A partir de ahí el pipeline reconstruye
todo lo demás:

```bash
uv run dvc repro
```

---

## El pipeline

Siete etapas declaradas en [`dvc.yaml`](dvc.yaml). La fecha lógica de la emisión está en
[`config/pipeline.yaml`](config/pipeline.yaml), versionada: **el commit describe un pronóstico
concreto.**

| Etapa | Qué hace | Salida |
|---|---|---|
| `bandeja` | Simula el sistema de origen: reparte los CSV de Kaggle en entregas diarias con latencias e incidentes reales. **No es parte del producto.** | `landing/` |
| `ingesta` | Valida contra contrato y materializa el almacén propio en Parquet. Lo que incumple va a cuarentena. | `data/raw/` |
| `panel` | Genera ejemplos de entrenamiento simulando 105 emisiones pasadas, sin fuga temporal. | `data/procesado/` |
| `validacion` | Los tres modelos compiten con los mismos pliegues; decide quién opera. | `reportes/`, `data/seguimiento/` |
| `modelo` | Ajusta el modelo promovido sobre el panel completo y lo registra con su ficha. | `models/` |
| `pronostico` | Emite los cuantiles del horizonte. Las series sin historia las atiende el baseline. | `data/pronosticos/` |
| `pedido` | Aplica los costos y produce el pedido sugerido. | `data/pedidos/` |

Cada etapa se invoca por la misma orden que usará Airflow, que es la única frontera entre
orquestación y producto:

```bash
uv run python -m reposicion pronosticar --asof 2017-08-14
```

`--asof` es obligatorio en todas y nunca tiene valor por defecto: la fecha lógica entra desde
fuera o no se ejecuta nada.

---

## Reproducir un pronóstico pasado

El requerimiento central del versionamiento. La pregunta a contestar es: *«¿qué pedido publicamos
el 14 de agosto, y se puede demostrar?»*

Hay dos emisiones en el historial, que se distinguen por un solo valor:

```console
$ git log --oneline -S'asof: "2017-08-14"' -- config/pipeline.yaml
87fb20f Emite el pronostico del 2017-07-17 cambiando solo la fecha logica
25fc898 Versiona el pipeline con DVC y emite el pronostico del 2017-08-14
```

Buscar por el *contenido* del parámetro (`git log -S`) y no por la fecha del commit es a propósito:
la fecha lógica es un dato versionado, así que el propio Git sabe cuándo cambió.

### Paso 1 — volver al commit de esa emisión

```console
$ git checkout 25fc898
$ grep '^asof' config/pipeline.yaml
asof: "2017-08-14"
```

### Paso 2 — recuperar los datos que ese commit describe

```console
$ uv run dvc checkout
A       data/pedidos/asof=2017-08-14/
A       models/asof=2017-08-14/
A       data/procesado/asof=2017-08-14/entrenamiento.parquet
A       data/pronosticos/asof=2017-08-14/pronostico.parquet
A       data/seguimiento/asof=2017-08-14/validacion.csv
D       data/pedidos/asof=2017-07-17/
D       models/asof=2017-07-17/
D       data/procesado/asof=2017-07-17/entrenamiento.parquet
D       data/pronosticos/asof=2017-07-17/pronostico.parquet
D       data/seguimiento/asof=2017-07-17/validacion.csv
```

Repone la emisión de agosto **y retira la de julio**: el espacio de trabajo queda exactamente
como lo describe el commit, no como una mezcla de los dos.

### Paso 3 — comprobar que es el pronóstico de agosto y no otro

```console
OK  c9dd259328cb31412f8066efb20c7d36  data/pronosticos/asof=2017-08-14/pronostico.parquet
OK  6882ffb1933913a57f576f72f31ec76a  data/pedidos/asof=2017-08-14/pedido.parquet
OK  1fdefa433ae795f553e0631fb2e3834b  models/asof=2017-08-14/modelo.pkl
```

Y el almacén también retrocedió, que es lo que hace honesto al pronóstico:

```console
ventas vigentes: 2.997.324  hasta 2017-08-13
```

No llega al 14 porque no podía: con latencia D+1, el 14 de agosto sólo se conocían las ventas
hasta el 13. El pronóstico reproducido usa la información que **había llegado** entonces, no la
que existe hoy.

### Paso 4 — regenerarlo, no sólo recuperarlo

Los pasos anteriores restauran archivos desde la caché. Eso prueba trazabilidad, pero no prueba
que el resultado sea *reproducible*: podría ser que el pipeline ya no fuera capaz de producirlo.
Así que se recalcula desde cero:

```console
$ uv run dvc repro --force -s modelo pronostico pedido
Running stage 'modelo':
lgbm_cuantilico ajustado con 1,297,296 ejemplos
-> models/asof=2017-08-14/modelo.pkl  (32.0 s)
Running stage 'pronostico':
12,474 filas  ·  origen 2017-08-13
-> data/pronosticos/asof=2017-08-14/pronostico.parquet  (3.1 s)
Running stage 'pedido':
1,782 series  ·  5,554,337 unidades  ·  68 linea(s) recortada(s) por el tope
```

```
IDENTICO   1fdefa433ae795f553e0631fb2e3834b  models/asof=2017-08-14/modelo.pkl
IDENTICO   c9dd259328cb31412f8066efb20c7d36  data/pronosticos/asof=2017-08-14/pronostico.parquet
IDENTICO   6882ffb1933913a57f576f72f31ec76a  data/pedidos/asof=2017-08-14/pedido.parquet
```

**Los tres artefactos vuelven a salir byte a byte iguales**, incluidos los 7,3 MB del modelo
entrenado desde cero sobre 1 297 296 ejemplos. El determinismo no es casualidad: los tres
modelos fijan semilla, y hay una prueba que entrena cada uno dos veces y compara el md5 del
artefacto (`tests/test_registro.py`).

### Lo que hace que esto funcione

DVC compara **contenido**, no fechas de modificación. Durante el desarrollo se refactorizó
`io.py` y bastó para que el lock quedara desfasado:

```console
$ uv run dvc status
ingesta:
        changed deps:
                modified:           src/reposicion

$ uv run dvc repro
Stage 'bandeja' didn't change, skipping
Running stage 'ingesta':
-> data/raw  (2.1 s)
Stage 'panel' didn't change, skipping
Stage 'validacion' didn't change, skipping
Stage 'modelo' didn't change, skipping
Stage 'pronostico' didn't change, skipping
Stage 'pedido' didn't change, skipping
```

Volvió a ejecutar la ingesta en 2 segundos, obtuvo un `data/raw` **idéntico** —porque el
refactor no cambiaba el comportamiento— y al ver el mismo contenido se saltó las cinco etapas
siguientes, incluidos los catorce minutos de panel y validación. Con comparación por fecha de
modificación habría recalculado todo sin necesidad.

### El remoto

```console
$ uv run dvc push
280 files pushed
```

El remoto es una **carpeta local** (`../dvc-remote-lab1`, 200 MB), configurada así a propósito:
el evaluador no necesita credenciales de ninguna nube para reproducir nada, y la mecánica que se
demuestra (`push`, `pull`, `checkout`) es la misma. La contrapartida, explícita: no sirve para
compartir entre máquinas; para eso habría que cambiar una línea de `.dvc/config`. Y como la
etapa `bandeja` regenera la bandeja desde los CSV de Kaggle, quien clone el repositorio puede
reconstruirlo todo con `dvc repro` aunque no tenga acceso a esta caché.

---

## Cambiar un costo sin tocar código

El requerimiento 2 del producto. Se cambia **un número** en `config/costs.yaml` y nada más:

```diff
 defaults:
-  costo_quiebre: 3.0
+  costo_quiebre: 9.0
   costo_merma: 1.0
```

DVC dice entonces qué quedó obsoleto:

```console
$ uv run dvc status
validacion:
        changed deps:
                config/costs.yaml:
                        modified:           defaults
pedido:
        changed deps:
                config/costs.yaml:
                        modified:           defaults
```

**Dos etapas, no una,** y conviene entender por qué: un costo tiene dos efectos de horizonte
distinto. Cambia el pedido *hoy* (`pedido`) y puede cambiar *qué modelo opera* en el siguiente
ciclo, porque la métrica que decide la promoción **es** el costo de negocio (`validacion`). El
grafo los separa en vez de confundirlos. Para obtener sólo el pedido nuevo:

```console
$ uv run dvc repro -s pedido
Running stage 'pedido':
1,782 series  ·  5,993,661 unidades  ·  187 linea(s) recortada(s) por el tope
-> data/pedidos/asof=2017-08-14/explicacion.csv  (0.1 s)
```

**0,1 segundos. Cero líneas de Python. Cero reentrenamientos.**

| Familia | q\* antes | Cuantil usado | q\* después | Cuantil usado | Pedido |
|---|---|---|---|---|---|
| PRODUCE | 0,429 | 0,5 | 0,692 | 0,8 | +5,8 % |
| GROCERY I | 0,857 | 0,9 | 0,947 | 0,95 | +6,0 % |
| CLEANING | 0,857 | 0,9 | 0,947 | 0,95 | +19,2 % |
| **EGGS** | 0,727 | 0,8 | **0,727** | **0,8** | **sin cambio** |

```
total antes:   5.554.337 unidades
total después: 5.993.661 unidades      (+7,9 %)
líneas topadas:       68  ->  187
```

El detalle de `EGGS` es el que confirma que la mecánica es la correcta: no se mueve porque tiene
su propio `costo_quiebre: 4.0` en la configuración, y los ajustes por familia tienen precedencia
sobre el valor por defecto.

Al restaurar el costo, DVC devuelve el pedido anterior **sin recalcularlo**, desde su caché de
ejecuciones, y el archivo es idéntico byte a byte (`md5 6882ffb1…`). Esa es la misma propiedad
que sostiene la reproducción de un pronóstico pasado.

---

## Modelado

Tres modelos compiten bajo **validación de origen móvil**, con los mismos pliegues, el mismo
panel y las mismas series. La métrica que decide la promoción es el **costo de negocio**, no el
error estadístico, y el criterio exige ganar en **todos** los pliegues: un modelo que mejora
mucho en uno y empeora en otro es inestable, y el promedio lo esconde.

1 297 296 ejemplos de 105 orígenes semanales, 3 pliegues de 28 días:

| Modelo | Costo negocio | Pinball | MAE | RMSLE | Cob. 0,9 | Cob. 0,95 |
|---|---|---|---|---|---|---|
| **lgbm cuantílico** | **105,95** | **20,21** | **58,74** | **0,401** | 0,903 | 0,951 |
| baseline estacional | 130,52 | 25,22 | 69,12 | 0,507 | 0,858 | 0,909 |
| lineal cuantílico | 263,06 | 40,09 | 86,16 | 0,434 | 0,953 | 0,978 |

```
lgbm   vs baseline: PROMOVER  (p1: +14,7 %, p2: +17,5 %, p3: +23,8 %)
lineal vs baseline: MANTENER  (p1: -169,6 %, p2: -94,5 %, p3: -43,0 %)
```

El resultado que justifica toda la arquitectura de decisión: **el modelo lineal gana al baseline
en RMSLE (0,434 frente a 0,507) y pierde en costo (263 frente a 131).** Las dos métricas señalan
modelos distintos, en direcciones opuestas. Elegir por el error estadístico habría promovido el
lineal y duplicado el costo de operación.

La causa está en la cobertura: el lineal sobrecubre (0,978 donde debería dar 0,95). En error
logarítmico pasarse apenas cuenta; en inventario real se traduce en merma.

**Deep learning fue evaluado y descartado** con cuatro evidencias medidas sobre estos datos
(saturación de capacidad, distribución del error, autocorrelación residual y el resultado del
propio criterio de promoción). El análisis completo está en
[`DECISIONES.md`](DECISIONES.md#1-deep-learning-evaluado-y-descartado).

---

## Estructura del repositorio

```
config/           decisiones de negocio en YAML: costos, horizonte, cuantiles, umbrales
dags/             orquestación de Airflow — sólo tareas y dependencias, cero lógica
src/reposicion/   el producto
  config.py         rutas y carga de configuración
  contracts.py      contratos de datos (Pandera) por fuente
  landing.py        lectura as-of de la bandeja
  io.py             escritura atómica y particiones
  ingesta.py        validación, cuarentena y almacén propio
  features.py       variables explicativas ancladas al origen
  models/           baseline, LightGBM y regresión cuantílica lineal
  metrics.py        pinball, cobertura y costo de negocio
  validacion.py     pliegues móviles y criterio de promoción
  policy.py         del pronóstico al pedido (vendedor de periódicos)
  pronostico.py     emisión, con respaldo para series sin historia
  registro.py       artefactos de modelo con su ficha
  cli.py            puntos de entrada del pipeline
tests/            237 pruebas; corren sin Airflow y sin Docker
tools/            simulador del sistema de origen (NO es parte del producto)
reportes/         métricas y decisiones de promoción (sí van a Git)
```

## Pruebas

```bash
uv run pytest                  # todo
uv run pytest -m "not slow"    # sin las que entrenan modelos
```

Las irrenunciables, que son las que justifican el resto:

| Qué vigila | Dónde |
|---|---|
| Que no haya fuga temporal en las variables | `test_features.py` |
| Que la lectura as-of ignore lo que llegó después | `test_landing.py` |
| Que una corrección retroactiva cambie la vista vigente sin borrar el histórico | `test_landing.py` |
| Que correr una etapa dos veces dé el mismo byte | `test_cli.py` |
| Que entrenar dos veces dé el mismo artefacto | `test_registro.py` |
| Que subir el costo de quiebre aumente el pedido | `test_policy.py` |
| Que no aparezca `datetime.now()` en el producto | `test_invariantes.py` |
| Que no haya datos ni modelos versionados en Git | `test_invariantes.py` |
| Que ninguna etapa de DVC omita el código como dependencia | `test_invariantes.py` |

## Documentación

- [`CLAUDE.md`](CLAUDE.md) — las reglas del proyecto: invariantes, arquitectura, datos, Git.
- [`DECISIONES.md`](DECISIONES.md) — cada decisión relevante con su alternativa descartada.
