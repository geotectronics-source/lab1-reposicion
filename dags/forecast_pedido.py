"""### Emisión semanal del pronóstico y del pedido

Corre los lunes y produce lo que el negocio recibe: el pronóstico cuantílico
de los siete días que empiezan ese lunes y el pedido sugerido que sale de
aplicarle los costos.

| tarea | qué hace |
|---|---|
| `auditar` | comprueba que el almacén está al día antes de emitir |
| `pronosticar` | emite el pronóstico con el **modelo vigente** |
| `pedir` | aplica el cuantil crítico de cada familia y publica el pedido |

**`auditar` aparece también en el DAG de ingesta, y no es un descuido.** Allí
es el registro diario del estado de las fuentes; aquí es una compuerta. Sin
ella, emitir un lunes cuya ingesta nunca corrió daría un pronóstico
impecable, calculado sobre datos de la semana anterior, y nada en el resultado
delataría el problema: los números se verían perfectamente razonables. Cuesta
un segundo y convierte una dependencia invisible entre dos DAGs en una tarea
roja con un mensaje que dice qué fechas faltan.

**El modelo no se entrena aquí.** `pronosticar` carga el último modelo
registrado con fecha ≤ `ds`, que normalmente viene del DAG de
reentrenamiento, de hace semanas. Son dos relojes distintos a propósito: el
pedido tiene que salir cada lunes aunque el último reentrenamiento haya
fallado. Qué versión atendió la emisión queda escrito en `emision.json`.
"""

from airflow.sdk import DAG
from comun import ARGUMENTOS, FIN, INICIO, etapa

with DAG(
    dag_id="forecast_pedido",
    description="Pronostico cuantilico y pedido sugerido de la semana",
    # Lunes a las 07:00, después de que la ingesta del día haya terminado. El
    # pedido tiene que estar en el centro de distribución antes del fin de
    # semana, que es cuando se concentra la venta.
    schedule="0 7 * * 1",
    start_date=INICIO,
    end_date=FIN,
    catchup=False,
    max_active_runs=1,
    default_args=ARGUMENTOS,
    tags=["reposicion", "semanal", "emision"],
    doc_md=__doc__,
):
    etapa("auditar") >> etapa("pronosticar") >> etapa("pedir")
