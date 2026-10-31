"""### Reentrenamiento y promoción mensual

El único DAG que decide **qué modelo opera**. Reconstruye el panel de
entrenamiento, hace competir a los tres modelos con los mismos pliegues y
registra al ganador según el criterio versionado en `config/model.yaml`.

| tarea | qué hace | cuánto tarda |
|---|---|---|
| `panel` | genera los ejemplos simulando emisiones pasadas | ~5 min |
| `validar` | valida con origen móvil y decide la promoción | ~9 min |
| `modelo` | ajusta al ganador sobre el panel completo y lo registra | ~35 s |

**Mensual y no semanal.** Reentrenar cada semana suena a diligencia y es
perseguir ruido: con siete días nuevos sobre dos años de historia, la
diferencia entre modelos queda por debajo del `mejora_minima` del 3 % que
exige el criterio, de modo que el resultado habitual sería quince minutos de
cómputo para confirmar lo que ya operaba. Un mes de datos nuevos sí puede
mover la decisión.

**Es el único DAG que puede cambiar el pedido sin que cambie un dato.** Por
eso la promoción no la decide quien lanza el run: la decide el criterio
escrito en `config/model.yaml`, la tarea `validar` la deja en
`reportes/asof=<ds>/promocion.json` y `modelo` obedece ese archivo. Quien
quiera cambiar el criterio tiene que hacerlo en un commit, que es donde se
revisa.

**Si este DAG falla, el producto sigue emitiendo.** `forecast_pedido` carga el
último modelo registrado con fecha ≤ su fecha lógica, así que un
reentrenamiento roto deja el modelo anterior operando en vez de dejar al
negocio sin pedido.
"""

from airflow.sdk import DAG
from comun import ARGUMENTOS, FIN, INICIO, etapa

with DAG(
    dag_id="reentrenamiento",
    description="Valida los modelos, promueve al ganador y lo registra",
    # Día 1 de cada mes a las 03:00: fuera de la ventana en que se emite, y
    # con horas de sobra antes del lunes siguiente.
    schedule="0 3 1 * *",
    start_date=INICIO,
    end_date=FIN,
    catchup=False,
    # Dos reentrenamientos a la vez se pelearían por la memoria y por el
    # archivo del modelo registrado.
    max_active_runs=1,
    default_args=ARGUMENTOS,
    tags=["reposicion", "mensual", "modelo"],
    doc_md=__doc__,
):
    etapa("panel") >> etapa("validar") >> etapa("modelo")
