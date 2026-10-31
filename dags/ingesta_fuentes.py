"""### Ingesta diaria de la bandeja

Materializa en el almacén propio lo que el sistema origen **entregó** en la
fecha lógica del run, y deja constancia de en qué estado quedó.

| tarea | qué hace | cuándo falla |
|---|---|---|
| `ingesta` | valida y escribe la partición del día | un archivo rompe su contrato |
| `auditar` | anota qué llegó y qué se corrigió | una fuente acumula demasiado retraso |

**Un run toca una partición.** La tarea ingiere `arrival_date == ds`, no
`<= ds`. Así repetir el run reescribe exactamente lo mismo (INVARIANTE 3) y un
backfill de seis semanas son cuarenta y dos runs independientes en vez de
cuarenta y dos reprocesados cada vez más largos.

**Que una fuente no entregue no hace fallar nada.** El 2017-07-25 el POS
central no entrega `transactions`: el run de ese día termina en verde, el
hueco queda anotado y el pronóstico siguiente usa lo último conocido
(INVARIANTE 5). Lo que sí falla es que el retraso acumulado delate que la
ingesta no corrió, que visto desde fuera se parece mucho y significa algo
completamente distinto.
"""

from airflow.sdk import DAG
from comun import ARGUMENTOS, FIN, INICIO, etapa

with DAG(
    dag_id="ingesta_fuentes",
    description="Lee la entrega del dia y audita el estado del almacen",
    # Las ventas del día se cierran de madrugada; a las 06:00 ya están.
    schedule="0 6 * * *",
    start_date=INICIO,
    end_date=FIN,
    # Los backfills se piden a mano y acotados (`airflow backfill create`).
    # Con `catchup=True`, habilitar el DAG lanzaría de golpe toda la ventana
    # sin que nadie lo haya pedido, y eso no es una decisión del planificador.
    catchup=False,
    # De uno en uno. Las particiones son independientes y podrían ir en
    # paralelo, pero la carga inicial la hace el primer run que la encuentre
    # ausente: dos runs simultáneos al empezar la harían dos veces.
    max_active_runs=1,
    default_args=ARGUMENTOS,
    tags=["reposicion", "diario", "ingesta"],
    doc_md=__doc__,
):
    etapa("ingesta", opciones="--incremental") >> etapa("auditar")
