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
| Desempeño realizado de los pedidos ya publicados | `data/seguimiento/asof=<fecha>/` | quien responde por el inventario |

Y una cuarta que no es una salida de negocio sino la **pista de auditoría del modelo**: las
métricas y la decisión de promoción en `reportes/asof=<fecha>/`, que sí van a Git para que el
historial conteste con qué números opera cada modelo y desde cuándo.

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
uv run pytest                # 312 pruebas, sin Airflow y sin Docker
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
uv run python -m reposicion pronosticar --asof 2017-07-17
```

`--asof` es obligatorio en todas y nunca tiene valor por defecto: la fecha lógica entra desde
fuera o no se ejecuta nada.

Hay **dos órdenes más que no son etapas de DVC**, porque no participan en producir una emisión:
`auditar` registra el estado de las fuentes y `seguir` mide el desempeño de lo ya publicado. Son
tareas de operación y viven sólo en los DAGs; la razón está en
[Operación con Airflow 3](#operación-con-airflow-3).

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

El remoto es una **carpeta local** (200 MB), configurada así a propósito: el evaluador no
necesita credenciales de ninguna nube para reproducir nada, y la mecánica que se demuestra
(`push`, `pull`, `checkout`) es la misma.

Dos contrapartidas, explícitas. La primera: **no sirve para compartir entre máquinas.** Quien
clone este repositorio no tiene la carpeta, así que `dvc pull` no le dará nada; el camino para
reconstruir es poner los CSV de Kaggle en `data/kaggle/` y lanzar `dvc repro`, porque la etapa
`bandeja` regenera la bandeja desde ahí. La segunda es una trampa de la ruta:

```ini
['remote "local"']
    url = ../../dvc-remote-lab1
```

**DVC resuelve las rutas relativas respecto a `.dvc/`, no a la raíz del repositorio**, y de ahí
el `../../`. La consecuencia práctica es que el remoto queda atado a *dónde* esté el clon: si el
repositorio se mueve de sitio, la ruta apunta a otra parte y `dvc pull` falla sin decir por qué.
Para usar otro remoto basta una línea, que además no se versiona:

```bash
dvc remote modify --local local url /ruta/a/tu/remoto
```

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
1,782 series  ·  6,874,460 unidades  ·  127 linea(s) recortada(s) por el tope
-> data/pedidos/asof=2017-07-17/explicacion.csv  (0.2 s)
```

**Dos décimas de segundo. Cero líneas de Python. Cero reentrenamientos.**

| Familia | q\* antes | Cuantil usado | q\* después | Cuantil usado | Pedido |
|---|---|---|---|---|---|
| PRODUCE | 0,429 | 0,5 | 0,692 | 0,8 | +7,7 % |
| DAIRY | 0,500 | 0,5 | 0,750 | 0,8 | +7,3 % |
| GROCERY I | 0,857 | 0,9 | 0,947 | 0,95 | +5,5 % |
| CLEANING | 0,857 | 0,9 | 0,947 | 0,95 | +14,5 % |
| **EGGS** | 0,727 | 0,8 | **0,727** | **0,8** | **sin cambio** |

```
total antes:   6.351.166 unidades
total después: 6.874.460 unidades      (+8,2 %)
líneas topadas:        8  ->  127
```

El detalle de `EGGS` es el que confirma que la mecánica es la correcta: no se mueve porque tiene
su propio `costo_quiebre: 4.0` en la configuración, y los ajustes por familia tienen precedencia
sobre el valor por defecto.

Al restaurar el costo, DVC devuelve el pedido anterior **sin recalcularlo**, desde su caché de
ejecuciones, y el archivo es idéntico byte a byte (`md5 6882ffb1…`). Esa es la misma propiedad
que sostiene la reproducción de un pronóstico pasado.

---

## Operación con Airflow 3

DVC y Airflow resuelven dos problemas que se confunden a menudo. **DVC reproduce; Airflow
opera.** El grafo de DVC es estático y contesta *«¿esta salida corresponde a estas entradas?»*.
El de Airflow es temporal y contesta *«¿corrió lo que tocaba el 25 de julio?»*.

Lo que mantiene honestos a los dos es que **ejecutan exactamente la misma orden**:

```bash
python -m reposicion pronosticar --asof 2017-07-17
```

Si Airflow llamara a una función de Python y DVC a un script propio, «lo que se reproduce» y «lo
que opera» serían dos cosas distintas y podrían separarse sin que nada avisara. Hay una prueba
que falla si un DAG importa `reposicion` ([`test_dags.py`](tests/test_dags.py)).

Airflow tampoco llama a `dvc repro`, y también es deliberado: DVC se saltaría la etapa cuando su
salida ya está al día —y una tarea programada tiene que correr cuando le toca, no cuando a la
caché le parece— y además escribiría `dvc.lock`, que es un archivo de Git. Un orquestador que
modifica archivos versionados en cada corrida convierte el repositorio en un diario de
ejecuciones.

### Puesta en marcha

Requiere Docker. La imagen extiende la oficial de Airflow 3.3.2 con las dependencias del
producto, fijadas desde `uv.lock`.

```bash
cp docker/.env.ejemplo docker/.env
```

```bash
cd docker && docker compose up -d --build
```

La interfaz queda en `http://localhost:8080`, usuario y contraseña `airflow`. Si ya tienes algo
escuchando en ese puerto —otro Airflow, por ejemplo, que es más común de lo que parece en cuanto
se tiene más de un proyecto con Docker— cambia `AIRFLOW_PORT` en `docker/.env`.

El repositorio se monta en `/opt/proyecto` y `REPOSICION_HOME` apunta ahí, así que **lo que corre
Airflow es el commit que está en tu disco**: no hay que reconstruir la imagen para probar un
cambio de código. La imagen lleva las dependencias; el volumen, el producto.

Para comprobar que quedó bien, dos órdenes. La primera no necesita credenciales:

```bash
curl -s http://localhost:8080/api/v2/monitor/health
```

```
{"metadatabase":{"status":"healthy"},"scheduler":{"status":"healthy",...},
 "triggerer":{"status":"healthy",...},"dag_processor":{"status":"healthy",...}}
```

La segunda es la verificación que las pruebas de `pytest` **no pueden hacer**, porque corren sin
Airflow: que los cuatro DAGs se importen de verdad.

```bash
docker compose run --rm cli dags list
```

```
dag_id                | fileloc                                    | owners
======================+============================================+============
forecast_pedido       | /opt/airflow/dags/forecast_pedido.py       | reposicion
ingesta_fuentes       | /opt/airflow/dags/ingesta_fuentes.py       | reposicion
reentrenamiento       | /opt/airflow/dags/reentrenamiento.py       | reposicion
seguimiento_desempeno | /opt/airflow/dags/seguimiento_desempeno.py | reposicion
```

Las pruebas estáticas de `test_dags.py` comprueban que ningún DAG importe el producto ni contenga
lógica, pero no ejecutan nada: un DAG puede pasarlas y fallar al importarse. De hecho pasó —ver
`libgomp` en [`DECISIONES.md` §3.6](DECISIONES.md)—, y por eso esta orden es parte de la puesta en
marcha y no un extra.

### Los cuatro DAGs

Uno por cadencia real. Ninguno existe para rellenar un hueco del enunciado.

| DAG | Cuándo | Tareas | Qué produce |
|---|---|---|---|
| [`ingesta_fuentes`](dags/ingesta_fuentes.py) | diario 06:00 | `ingesta` → `auditar` | el almacén y el reporte de calidad del día |
| [`forecast_pedido`](dags/forecast_pedido.py) | lunes 07:00 | `auditar` → `pronosticar` → `pedir` | el pronóstico y el pedido de la semana |
| [`seguimiento_desempeno`](dags/seguimiento_desempeno.py) | diario 08:00 | `seguir` | el desempeño realizado de lo ya publicado |
| [`reentrenamiento`](dags/reentrenamiento.py) | día 1 del mes 03:00 | `panel` → `validar` → `modelo` | la decisión de promoción y el modelo |

**Por qué no uno solo.** Las cadencias son distintas y no divisibles: las fuentes llegan a
diario, el pedido se emite los lunes y reentrenar cuesta quince minutos. Un único DAG obligaría
a la cadencia más fina, y seis de cada siete corridas serían quince minutos de cómputo para
confirmar lo que ya operaba.

**Por qué el panel vive en el reentrenamiento y no en un DAG diario.** El panel sólo lo consume
la validación. Construirlo cada mañana serían cinco minutos diarios para un consumidor mensual;
el pronóstico no lo necesita, porque calcula sus propias variables sobre el almacén.

### Dos relojes: la emisión y el modelo

El pedido sale cada lunes y el modelo se reentrena una vez al mes, así que **la mayoría de los
lunes no existe ningún `models/asof=<ese lunes>`**. `pronosticar` carga el *modelo vigente*: el
último registrado con fecha ≤ la fecha lógica.

La segunda mitad de esa regla es la que importa. Tomar simplemente el último modelo haría que un
backfill de julio usara el modelo entrenado en agosto, y el resultado sería mejor que el que el
producto habría dado de verdad ese día: **el backfill mentiría a su favor**. Por eso el filtro
punto-en-el-tiempo se aplica también al registro de modelos.

Una sola regla cubre las dos situaciones. Cuando el pipeline de DVC acaba de entrenar para
`asof`, esa versión *es* la más reciente ≤ `asof`, de modo que DVC y Airflow siguen ejecutando
lo mismo. Qué versión atendió cada emisión queda escrito en `emision.json`, y es lo que permite
al seguimiento contestar *«¿empeoró desde que cambiamos de modelo?»*.

### Un hueco de entrega no es un fallo

El INVARIANTE 5 puesto en código. La tarea `auditar` separa dos cosas que desde fuera se ven
igual —en ambas no hay datos nuevos— y piden arreglos opuestos:

| Situación | Veredicto | Qué pasa |
|---|---|---|
| El petróleo no cotiza el domingo | `hueco` | se anota y se sigue con lo último conocido |
| El POS central no entregó hoy | `hueco` | igual: el run termina en verde |
| Las ventas llevan ocho días sin dato nuevo | `atrasado` | **la tarea falla**: la ingesta no corrió |
| Una fuente no tiene ni un registro | `sin datos` | **la tarea falla**: el despliegue está incompleto |

El umbral que las separa es una decisión de negocio y vive en
[`config/seguimiento.yaml`](config/seguimiento.yaml), no en el código.

El retraso se mide sobre `date` —a qué día se refieren los datos— y nunca sobre `arrival_date`.
Medido sobre la llegada, una entrega de hoy con datos de hace dos semanas daría retraso cero, y
la auditoría daría luz verde a un almacén al que le faltan trece días.

`auditar` aparece en dos DAGs, y no es un descuido: en el de ingesta es el registro diario del
estado de las fuentes; en el de emisión es una compuerta. Sin ella, emitir un lunes cuya ingesta
nunca corrió daría un pronóstico impecable calculado sobre datos de la semana anterior, y nada
en el resultado delataría el problema.

### El seguimiento es la tarea que DVC no puede expresar

Hay una etapa del producto que **no aparece en `dvc.yaml`**, y la razón es la mejor
justificación de por qué el laboratorio pide las dos herramientas.

La entrada del seguimiento es *«todas las emisiones pasadas cuyo horizonte ya se realizó»*. Eso
no es una arista fija de un grafo: es una consulta que depende de la fecha en que se corre. DVC
declara dependencias entre archivos concretos y no tiene forma de escribirla. Un orquestador con
fecha lógica, sí.

Lo que mide:

- el **costo realizado** de cada pedido publicado, en la misma moneda con la que se eligió el
  modelo: `cu · faltante + co · sobrante`;
- la **cobertura efectiva** del cuantil que usó la política: si pide cubrir el 90 % y en
  operación cubre el 78 %, el pedido se queda corto más seguido de lo que el negocio aceptó, y
  el error medio puede verse perfectamente bien;
- el desvío contra el **costo con el que se validó el modelo que atendió esa emisión**, no
  contra la emisión anterior: lo que se vigila es si el modelo cumple la promesa con la que se
  promovió, no cómo vino la semana.

Tres atajos que habrían convertido la métrica en propaganda, cada uno con su prueba:

1. **Rellenar con cero** el día cuyo dato aún no ha llegado. Inventaría una merma que nadie tuvo
   y haría ver mal a un modelo que acertó. El cruce es interno y se informa qué fracción del
   horizonte ya se puede calificar.
2. **Medir con lo que sabemos hoy** en vez de con lo que se sabía en la fecha lógica. Daría una
   cifra más exacta y **no reproducible**: cambiaría sola cada vez que llega una corrección, y
   una métrica de monitoreo que se mueve por su cuenta no detecta nada.
3. **Calificar una emisión posterior** a la fecha lógica, que durante un backfill ya existe en
   el disco.

Y un modelo que se degrada **no hace fallar el DAG**. Es una situación de negocio: la alerta
queda en el log y en `desempeno.json`, y la decisión de reentrenar es de una persona. Hacer caer
la tarea dejaría al día siguiente sin medición, que es justo cuando hace falta mirarla.

### El backfill, corrido de verdad

No es una captura de pantalla de la interfaz: son las corridas que produjeron los archivos que
están en `data/`.

```bash
airflow backfill create --dag-id ingesta_fuentes --from-date 2017-07-18 --to-date 2017-08-20
```

#### Un hueco de entrega declarado, resuelto sin intervención

`config/sources.yaml` inyecta a propósito una caída del POS central el 2017-07-25. El reporte de
calidad de ese día y del siguiente, tal como los dejó el DAG:

```
2017-07-25   fuente        estado   ultimo_dato  retraso  filas_en_asof
             transactions  hueco    2017-07-22         3              0
             sales         al dia   2017-07-24         1          1.782

2017-07-26   transactions  al dia   2017-07-24         2            108
```

El run del 25 terminó **en verde** con la fuente ausente, y el 26 llegaron 108 filas: las dos
entregas acumuladas, 54 de cada día. Es exactamente lo que describe el incidente, y nadie tuvo
que tocar nada.

#### Una corrección retroactiva, contada con precisión

El otro incidente declarado: el POS de la tienda 44 reportó ceros el 2017-07-05 y el dato real se
reenvía quince días después, cuando el pronóstico de esa semana ya estaba publicado. El reporte
del 2017-07-20:

```
sales reenvio 1.782 fila(s) ya conocidas, de las que 30 cambian de valor
```

Las dos cifras dicen cosas distintas y por eso están las dos. El sistema origen reenvió **el día
completo** —1.782 filas, como hace un sistema real—, pero sólo 30 traían un valor distinto. Y 30,
no 33: la tienda 44 tiene 33 familias, y BABY CARE, BOOKS y HOME APPLIANCES vendieron de verdad
cero ese día, así que para esas tres el «cero» original era correcto.

La primera versión de esta auditoría devolvía un solo número e informó de 1.782 correcciones. Un
reporte que dice que se corrigieron 1.782 filas cuando cambiaron 30 manda a investigar un
incidente que no existe, y a la tercera vez nadie lo mira.

#### La auditoría encontró sola el final de los datos

El backfill llegó al 2017-08-20 y ese run **falló**. No es un defecto: los datos públicos de
Kaggle terminan el 2017-08-15, así que las ventas acumulaban cinco días de retraso sobre una
tolerancia de cuatro.

```
2017-08-19   sales   hueco      2017-08-15   retraso 4   (tolerado 4)  -> verde
2017-08-20   sales   atrasado   2017-08-15   retraso 5   (tolerado 4)  -> ROJO
```

Es la tarea haciendo su trabajo: a partir de ahí, seguir emitiendo pedidos habría sido calcular
sobre un almacén que dejó de alimentarse. La ventana de operación con datos completos es
2017-07-18 → 2017-08-19.

#### El lazo cerrado: qué pasó con el pedido del 17 de julio

Siete días después, el horizonte completo de la primera emisión ya se había realizado:

```
2017-07-17   100.0% realizado   costo 95.68 vs 113.12 validado   cobertura 76.6% (la politica pide 76%)
sin alertas
```

Las dos cifras que importan:

- **Costo realizado 95,68 frente a 113,12 de validación.** El modelo operó un 15 % *mejor* que en
  los pliegues. Conviene no celebrarlo: una semana no demuestra nada, y la métrica no está ahí
  para declarar mejoras sino para detectar degradación sostenida. Lo que sí demuestra es que el
  número de validación no estaba inflado.
- **Cobertura efectiva 76,6 % contra el 76 % que pide la política.** Los cuantiles están
  calibrados en operación, no sólo en validación. Esta es la cifra que delataría un pedido que se
  queda corto aunque el error medio se viera bien.

El desglose por familia muestra por qué el detalle se guarda aparte del total:

| familia | cuantil que usa | cobertura lograda | costo realizado |
|---|---|---|---|
| PRODUCE | 0,50 | 50,8 % | 685,52 |
| BEVERAGES | 0,90 | 86,0 % | 629,36 |
| GROCERY I | 0,90 | 86,5 % | 429,84 |
| DAIRY | 0,50 | 62,2 % | 220,51 |
| CLEANING | 0,90 | 89,7 % | 197,74 |

PRODUCE cubre el 50 % **a propósito**: es perecedero, su merma es cara y el cuantil crítico que
sale de sus costos es 0,5. Un reporte que midiera todo contra el 90 % lo marcaría como un
problema cuando es la política funcionando.

#### El reentrenamiento mensual decidió solo

El DAG `reentrenamiento` corrió para el 2017-08-01: construyó el panel, hizo competir a los tres
modelos con los mismos pliegues y escribió su decisión en `reportes/asof=2017-08-01/`.

| modelo | costo | pinball | RMSLE | cobertura 0,9 | decisión |
|---|---|---|---|---|---|
| `lgbm_cuantilico` | **105,89** | 20,17 | 0,401 | 0,900 | PROMOVER |
| `baseline_estacional` | 126,35 | 24,16 | 0,547 | 0,863 | — (vigente) |
| `lineal_cuantilico` | 142,95 | 27,49 | 0,429 | 0,916 | MANTENER |

El lgbm gana en **los tres pliegues** (+19,3 %, +13,0 %, +15,5 %), que es lo que exige
`en_todos_los_pliegues: true`. El lineal pierde en los tres (−10,9 %, −24,3 %, −4,7 %) **pese a
tener mejor RMSLE que el baseline** (0,429 contra 0,547): la demostración de por qué el criterio
mide costo y no error estadístico, esta vez tomada por el orquestador y no a mano.

Nadie eligió el modelo: lo eligió el criterio escrito en `config/model.yaml`. La tarea `validar`
dejó la decisión en `promocion.json` y la tarea `modelo` obedeció ese archivo.

#### Cinco emisiones, dos modelos, y el baremo correcto para cada una

El backfill emitió los cinco lunes de la ventana. Qué modelo atendió cada emisión, según el
`emision.json` que escribió cada run:

| emisión | modelo que la atendió | origen |
|---|---|---|
| 2017-07-17 | `lgbm_cuantilico` de **2017-07-17** | 2017-07-16 |
| 2017-07-24 | `lgbm_cuantilico` de **2017-07-17** | 2017-07-23 |
| 2017-07-31 | `lgbm_cuantilico` de **2017-07-17** | 2017-07-30 |
| 2017-08-07 | `lgbm_cuantilico` de **2017-08-01** | 2017-08-06 |
| 2017-08-14 | `lgbm_cuantilico` de **2017-08-01** | 2017-08-13 |

Es la regla del modelo vigente funcionando: el cambio ocurre exactamente en la fecha del
reentrenamiento, **aunque los cinco runs se ejecutaran el mismo día y con los dos modelos ya en
el disco**. Cada emisión usó el modelo que existía en su fecha lógica, no el más nuevo.

Y el seguimiento del 2017-08-19 mide cada una contra la promesa del modelo que de verdad la
produjo:

```
emision       realizado     costo  validado  cobertura   pide   baremo
2017-07-17      100.0%      95.68    113.12     76.6%     76%   validado en 2017-07-17
2017-07-24      100.0%     109.58    113.12     71.8%     76%   validado en 2017-07-17
2017-07-31      100.0%     118.59    113.12     78.1%     76%   validado en 2017-07-17
2017-08-07      100.0%     124.85    105.89     81.1%     76%   validado en 2017-08-01
2017-08-14       28.6%      86.95    105.89     67.5%     76%   validado en 2017-08-01
```

El baremo cambia solo a partir del 7 de agosto. La última emisión va por el 28,6 % porque sólo
dos de sus siete días han llegado, y eso es correcto: el resto todavía no se puede calificar.

**El costo sube cuatro semanas seguidas y ninguna alerta se dispara.** De 95,68 a 124,85, un
30 %, con la peor emisión a +17,9 % sobre una tolerancia del 25 %. Con cuatro puntos no se puede
distinguir un modelo que se degrada de cuatro semanas que vinieron más difíciles —agosto trae el
feriado del Primer Grito de Independencia y el efecto de quincena—, y no poder decidirlo es el
resultado. Lo que sí queda demostrado es un límite del diseño, discutido en
[`DECISIONES.md` §3.16](DECISIONES.md): una alerta que mira cada emisión por separado no ve una
tendencia.

#### La reproducibilidad byte a byte es de una plataforma

El backfill regeneró en Linux la misma emisión del 2017-07-17 que `dvc.lock` describe con el hash
que produjo Windows, y **los hashes no coinciden**. Midiendo la diferencia:

| | windows | linux |
|---|---|---|
| un `q0.5` cualquiera | `4.129039181173225` | `4.1290391811732245` |

Diferencia relativa máxima: **3,0 · 10⁻¹⁶**, que es un bit —el épsilon de un `float64` vale
2,2 · 10⁻¹⁶—. Es el orden en que se suman los flotantes, que depende de cómo reparta el trabajo la
biblioteca de álgebra. Las dependencias son idénticas: están fijadas desde `uv.lock`.

**Pero el pedido publicado sí es idéntico bit a bit**: las 12.474 cantidades, el total de
6.351.166,32 unidades y las 8 líneas recortadas. El motivo es `politica.decimales: 2`, puesto
porque nadie despacha 0,0000000026 unidades, que resulta absorber el ruido de la última cifra.

La precisión que esto obliga a hacer: **la decisión es reproducible entre plataformas; el archivo
no.** Y como DVC compara hashes de archivos, `dvc repro` en otro sistema operativo dará todas las
etapas por obsoletas. El detalle está en [`DECISIONES.md` §3.15](DECISIONES.md).
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

**Y la conclusión aguanta en tres paneles distintos.** El reentrenamiento que corrió Airflow sobre
el panel del 2017-08-01 dio una tercera medición independiente:

| panel | costo del lineal | RMSLE del lineal | RMSLE del baseline | ¿gana en costo? |
|---|---|---|---|---|
| 2017-07-17 | 166,89 | 0,456 | 0,634 | no |
| 2017-08-01 | 142,95 | 0,429 | 0,547 | no |
| 2017-08-14 | 263,06 | 0,434 | 0,507 | no |

El costo del lineal se mueve un 84 % entre paneles mientras su RMSLE se queda entre 0,43 y 0,46:
es la firma de un modelo que no converge en las 60 épocas que se le dan, y está analizado en
[`DECISIONES.md` §0](DECISIONES.md#0-los-tres-modelos-y-por-qué-gana-el-que-gana). Lo que no se
mueve es el orden: gana al baseline en RMSLE y pierde en costo **las tres veces**.

**Deep learning fue evaluado y descartado** con cuatro evidencias medidas sobre estos datos
(saturación de capacidad, distribución del error, autocorrelación residual y el resultado del
propio criterio de promoción). El análisis completo está en
[`DECISIONES.md`](DECISIONES.md#1-deep-learning-evaluado-y-descartado).

---

## Estructura del repositorio

```
config/           decisiones de negocio en YAML: costos, horizonte, cuantiles, umbrales
dags/             orquestación de Airflow — sólo tareas y dependencias, cero lógica
docker/           imagen y compose de Airflow 3 (LocalExecutor, sin Celery)
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
  registro.py       artefactos de modelo con su ficha y el modelo vigente
  calidad.py        auditoría del almacén: huecos de entrega y correcciones
  seguimiento.py    desempeño realizado de los pedidos ya publicados
  cli.py            puntos de entrada del pipeline
tests/            312 pruebas; corren sin Airflow y sin Docker
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
| Que ningún DAG importe el producto ni contenga lógica | `test_dags.py` |
| Que un hueco de entrega no se confunda con una ingesta parada | `test_calidad.py` |
| Que el seguimiento no rellene con cero lo que aún no ha llegado | `test_seguimiento.py` |
| Que un backfill no use un modelo entrenado después de su fecha | `test_registro.py` |
| Que las salidas de texto usen LF en cualquier sistema operativo | `test_invariantes.py`, `test_build_landing.py` |

## Dónde está cada requerimiento del enunciado

| # | Requerimiento | Dónde |
|---|---|---|
| 1 | Repositorio con ramas y Pull Requests, `main` protegida | un PR por fase, ninguno directo a `main`, que exige PR, historial lineal y los dos checks |
| 1 | `pytest` en GitHub Actions | [`.github/workflows/`](.github/workflows) — matriz Ubuntu + Windows |
| 1 | `CLAUDE.md` con las reglas del proyecto | [`CLAUDE.md`](CLAUDE.md) |
| 2 | Ni datos ni modelos en Git | `git ls-files -- data landing models` devuelve sólo `data/kaggle.dvc`; hay prueba que lo vigila |
| 2 | Remoto de DVC | carpeta local, justificada en [el remoto](#el-remoto) |
| 2 | Reproducir un pronóstico pasado, con comandos y resultados | [Reproducir un pronóstico pasado](#reproducir-un-pronóstico-pasado) |
| 3 | Airflow 3 con Docker Compose | [`docker/`](docker) — `LocalExecutor`, cinco servicios |
| 3 | Tareas idempotentes | escritura atómica y partición por fecha lógica; prueba de byte idéntico |
| 3 | Cuarentena del dato inválido | [`src/reposicion/ingesta.py`](src/reposicion/ingesta.py) — partición todo-o-nada |
| 3 | Las situaciones de negocio no hacen fallar el DAG | [Un hueco de entrega no es un fallo](#un-hueco-de-entrega-no-es-un-fallo), con el incidente del 2017-07-25 medido |
| 3 | Backfill | 33 días de ingesta, 5 emisiones y 1 reentrenamiento, en [el backfill](#el-backfill-corrido-de-verdad) |
| 3 | Los DAGs sólo orquestan | [`dags/`](dags) — prueba estática que falla si un DAG importa el producto o tiene un `if` |
| 3 | Seguimiento del desempeño | [`src/reposicion/seguimiento.py`](src/reposicion/seguimiento.py) y [la tarea que DVC no puede expresar](#el-seguimiento-es-la-tarea-que-dvc-no-puede-expresar) |
| 4 | Baseline y un modelo que lo gane | [Modelado](#modelado) — lgbm gana en los tres pliegues |
| 4 | Deep learning justificado en costo/beneficio | [`DECISIONES.md` §1](DECISIONES.md#1-deep-learning-evaluado-y-descartado) — evaluado y descartado con cuatro evidencias |
| 4 | Misma validación temporal para todos | origen móvil, mismos pliegues y mismo panel; prohibido `KFold` sin orden |
| 4 | Métricas técnicas **y** de negocio | pinball, MAE, RMSLE, cobertura **y** costo de negocio, juntas en `reportes/` |
| 4 | Criterio de promoción explícito | `promocion` en [`config/model.yaml`](config/model.yaml): costo, mejora mínima del 3 %, en todos los pliegues |
| 5 | README y DECISIONES | este archivo y [`DECISIONES.md`](DECISIONES.md) |

## Documentación

- [`CLAUDE.md`](CLAUDE.md) — las reglas del proyecto: invariantes, arquitectura, datos, Git.
- [`DECISIONES.md`](DECISIONES.md) — cada decisión relevante con su alternativa descartada.
