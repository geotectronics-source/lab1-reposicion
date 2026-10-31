# Evidencia de funcionamiento

Capturas de la interfaz de Airflow y los reportes que produjeron las corridas descritas en
[`README.md` § Operación con Airflow 3](../../README.md#operación-con-airflow-3). Todo lo de
esta carpeta se tomó de un despliegue real —`docker compose up` sobre este repositorio— después
del backfill; nada está dibujado ni editado.

## Por qué viven en `docs/` y no en `data/` ni en `reportes/`

Son tres sitios con tres significados y mezclarlos costaría la trazabilidad:

- **`data/`** es salida del producto y **no entra en Git** (requerimiento 2). Hay una prueba que
  falla si algo bajo `data/`, `landing/` o `models/` aparece rastreado, salvo `data/kaggle.dvc`.
- **`reportes/`** son salidas declaradas de la etapa `validacion` en [`dvc.yaml`](../../dvc.yaml),
  con `cache: false` para que sí viajen en Git. Las escribe el pipeline y DVC las compara; una
  copia puesta a mano ahí haría que `dvc status` mintiera.
- **`docs/evidencia/`** es esto: el testimonio de que el sistema corrió. No es salida de ninguna
  etapa, nadie lo reproduce y por eso puede estar en Git sin contradecir nada.

La consecuencia honesta de esa separación: **el reporte de seguimiento que el producto escribe
vive en `data/seguimiento/` y no llega al repositorio.** Lo que hay aquí es una copia fechada,
no la salida viva.

## Las capturas

| # | Captura | Qué prueba |
|---|---|---|
| 01 | [Los cuatro DAGs y sus cadencias](01-dags-y-cadencias.jpg) | Un DAG por cadencia real: `0 7 * * 1` semanal, `0 6 * * *` y `0 8 * * *` diarios, `0 3 1 * *` mensual. En la barra de `ingesta_fuentes` ya se ve el único rojo. |
| 02 | [35 ejecuciones de la ingesta](02-ingesta-35-ejecuciones.jpg) | El backfill de 34 días más una corrida manual de prueba. Todas `Exitoso` salvo `backfill__2017-08-20`, roja **por diseño**. |
| 03 | [Cinco emisiones semanales](03-forecast-cinco-emisiones.jpg) | Los cinco lunes del 2017-07-17 al 2017-08-14, con `auditar`, `pronosticar` y `pedir` en verde en los cinco. |
| 04 | [El grafo del pronóstico](04-forecast-grafo-tres-tareas.jpg) | `auditar → pronosticar → pedir`, los tres `BashOperator`: el DAG orquesta y no calcula. |
| 05 | [La auditoría falla tras tres intentos](05-auditoria-falla-con-tres-intentos.jpg) | `ingesta` en verde y `auditar` en rojo en la misma corrida, con los tres intentos que fija `retries: 2`. El fallo no es del operador: es del dato. |
| 06 | [Huecos tolerados, retraso no](06-log-huecos-tolerados-retraso-no.jpg) | **La captura que más dice.** Tres fuentes con hueco se registran y el flujo sigue; `sales` con 5 días de retraso sobre un máximo de 4 detiene la tarea con un mensaje que dice qué hacer. Es la INVARIANTE 5 en una pantalla. |
| 07 | [El reentrenamiento mensual](07-reentrenamiento-mensual.jpg) | `panel → validar → modelo` el 2017-08-01, la única fecha en que el cron mensual dispara dentro de la ventana. |
| 08 | [El seguimiento diario](08-seguimiento-diario.jpg) | La tarea `seguir`, que mide lo ya publicado contra lo que de verdad pasó. No es etapa de DVC y por eso sólo existe aquí. |
| 09 | [Tres pliegues y la promoción](09-validacion-tres-pliegues-y-promocion.jpg) | La comparación de modelos tal como la imprime la tarea: los tres pliegues, la tabla con métricas técnicas **y** de negocio, `lgbm_cuantilico PROMOVER` y `lineal_cuantilico MANTENER`. |

## Los reportes

| Archivo | Qué contiene |
|---|---|
| [`desempeno-2017-08-19.json`](desempeno-2017-08-19.json) | El resumen del seguimiento: las cinco emisiones con su costo realizado frente al validado, la cobertura lograda y el baremo —qué modelo la emitió— de cada una. `"alertas": []`. |
| [`desempeno-2017-08-19.csv`](desempeno-2017-08-19.csv) | El detalle que respalda ese resumen: 33 familias × 5 emisiones, con faltante, sobrante y pinball por línea. |
| [`calidad-2017-08-19.csv`](calidad-2017-08-19.csv) | La auditoría del día anterior al fallo: `sales` con `retraso_dias` 4 y `retraso_tolerado` 4. Pasa por un día. |
| [`calidad-2017-08-20.csv`](calidad-2017-08-20.csv) | La del día del fallo: el mismo `sales` con 5. La diferencia entre las dos filas es todo el incidente. |

## Lo que estas capturas no dicen

El costo realizado sube cuatro semanas seguidas —95,68 → 109,58 → 118,59 → 124,85— y
`"alertas"` sigue vacío, porque el peor caso individual (+17,9 %) no llega a la tolerancia del
25 % que fija [`config/seguimiento.yaml`](../../config/seguimiento.yaml). La alerta mira
emisiones sueltas, no tendencias. Con cuatro puntos no se distingue un modelo que se degrada de
cuatro semanas difíciles, así que el diseño prefiere callar antes que afirmar de más. Está
razonado en [`DECISIONES.md` §3.16](../../DECISIONES.md).
