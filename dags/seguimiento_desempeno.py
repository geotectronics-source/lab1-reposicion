"""### Seguimiento diario del desempeño

Cierra el lazo: toma los pedidos ya publicados, los cruza con las ventas que
fueron llegando y mide **cuánto costó de verdad** cada decisión, en la misma
moneda con la que se eligió el modelo.

**Es la tarea que justifica tener Airflow y no sólo DVC.** El grafo de DVC es
estático: una etapa declara de qué archivos depende y eso no cambia nunca. La
entrada de esta medición es "todas las emisiones pasadas cuyo horizonte ya se
realizó", que no es una arista fija sino una consulta que depende de la fecha
en que se corre. DVC no puede expresarla. Por eso esta etapa no aparece en
`dvc.yaml` y sólo existe aquí.

**Corre a diario aunque se emita una vez por semana**, porque el horizonte se
realiza de a un día: el lunes se publican siete días y el martes sólo se puede
calificar el primero. La columna `realizado` del reporte sube cada mañana
hasta llegar al 100 %.

**Un modelo que se degrada no hace fallar este DAG.** Es una situación de
negocio (INVARIANTE 5): la alerta queda en el log y en `desempeno.json`, y la
decisión de reentrenar es de una persona. Hacer caer la tarea dejaría al día
siguiente sin medición, que es precisamente cuando hace falta mirarla.
"""

from airflow.sdk import DAG
from comun import ARGUMENTOS, FIN, INICIO, etapa

with DAG(
    dag_id="seguimiento_desempeno",
    description="Compara los pedidos publicados con la venta realizada",
    # A las 08:00, después de la ingesta del día: mide con lo que acaba de
    # llegar y nunca con lo que llegará mañana.
    schedule="0 8 * * *",
    start_date=INICIO,
    end_date=FIN,
    catchup=False,
    max_active_runs=1,
    default_args=ARGUMENTOS,
    tags=["reposicion", "diario", "seguimiento"],
    doc_md=__doc__,
):
    etapa("seguir")
