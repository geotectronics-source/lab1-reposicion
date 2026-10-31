"""Piezas compartidas por los DAGs. **Aquí no hay una sola regla de negocio.**

Un archivo de `dags/` define tareas, dependencias y programación, y nada más
(CLAUDE.md §2). Toda la lógica vive en `src/reposicion/` y se alcanza por una
única puerta: `python -m reposicion <etapa> --asof <fecha>`, la misma orden
que ejecutan las etapas de `dvc.yaml`.

**Por qué `BashOperator` y no `PythonOperator`.** Con `PythonOperator` se
importaría `reposicion` dentro del proceso de Airflow y se llamaría a la
función directamente, lo que da mejores mensajes de error y permite pasar
objetos por XCom. Se descartó porque abre una segunda puerta: habría una
forma de ejecutar el producto que DVC no usa, y entonces "lo que reproduce
DVC" y "lo que corre en operación" podrían separarse sin que nada avise. Con
`BashOperator` la frontera es una sola y es verificable: hay una prueba que
falla si un DAG importa `reposicion`.

El precio está asumido: un fallo llega como código de salida y un traceback
en el log, no como una excepción de Python en la interfaz.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator

#: Raíz del repositorio. **No se escribe aquí**, ni siquiera como valor por
#: defecto: la fija el volumen declarado en `docker/docker-compose.yml`, que
#: exporta `REPOSICION_HOME`, y ni `src/` ni `dags/` llevan rutas absolutas
#: (CLAUDE.md §2). Con un valor por defecto escrito en el código habría dos
#: fuentes de verdad, y la del código ganaría el día que alguien cambiara el
#: volumen y se olvidara de este archivo.
PROYECTO = os.environ.get("REPOSICION_HOME")
if not PROYECTO:
    raise RuntimeError(
        "REPOSICION_HOME no esta definida. La exporta docker/docker-compose.yml; "
        "sin ella los DAGs no saben donde esta montado el repositorio."
    )

#: Ventana en la que existe el sistema origen simulado. La bandeja arranca el
#: 2017-07-01 y su última entrega es del 2017-09-01, pero sólo de `oil`: las
#: ventas públicas de Kaggle terminan el 2017-08-15, de modo que la operación
#: con datos completos llega hasta el 2017-08-19. Del 20 en adelante la tarea
#: `auditar` declara las ventas atrasadas y hace fallar el run, que es
#: exactamente su trabajo: el producto no debe seguir emitiendo pedidos sobre
#: un almacén que dejó de alimentarse.
#:
#: `end_date` no es decoración: sin él, habilitar un DAG diario con fecha de
#: inicio en 2017 pondría al planificador a crear miles de corridas hasta hoy,
#: todas sobre una bandeja que no tiene nada que entregar. En un despliegue
#: real se quitaría, porque la fecha lógica sería la de hoy.
INICIO = datetime(2017, 7, 1)
FIN = datetime(2017, 9, 2)

#: Reintentos. **Sólo son seguros porque las tareas son idempotentes**
#: (INVARIANTE 3): la salida se particiona por fecha lógica y se escribe con
#: renombrado atómico, así que un segundo intento reescribe exactamente lo
#: mismo. En un pipeline que hiciera *append*, este mismo ajuste duplicaría
#: datos en silencio cada vez que la red tosiera.
ARGUMENTOS = {
    "owner": "reposicion",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

#: Plantilla Jinja de la fecha lógica, en una constante con nombre y no
#: incrustada en una f-string. Dentro de una f-string habría que escribirla
#: `{{{{ ds }}}}`, que es ilegible y, peor, imposible de buscar: una prueba
#: que compruebe que todas las tareas reciben la fecha lógica tendría que
#: buscar el texto escapado.
FECHA_LOGICA = "{{ ds }}"


def etapa(orden: str, *, task_id: str | None = None, opciones: str = "") -> BashOperator:
    """Tarea que ejecuta una etapa del producto con la fecha lógica del run.

    `{{ ds }}` es la fecha lógica que inyecta Airflow, y es la **única** fuente
    de "hoy" que el producto acepta (INVARIANTE 1). Por eso un backfill de
    julio produce hoy el mismo resultado que habría producido en julio: la
    tarea no tiene forma de preguntarle la fecha al reloj.

    Cuidado con un detalle que no se ve: **`ds` es la fecha lógica en UTC**,
    mientras que el cron se interpreta en la zona de `DEFAULT_TIMEZONE`
    (America/Guayaquil, UTC-5). Las cuatro programaciones de este proyecto
    están de madrugada o por la mañana, así que al pasar a UTC siguen cayendo
    en el mismo día del calendario. Un cron de las 20:00 locales serían las
    01:00 UTC del día siguiente y `ds` adelantaría un día sin avisar, que es
    una forma silenciosa de romper la corrección punto-en-el-tiempo.
    """
    return BashOperator(
        task_id=task_id or orden,
        bash_command=" ".join(
            parte
            for parte in ("python -m reposicion", orden, "--asof", FECHA_LOGICA, opciones)
            if parte
        ),
        cwd=PROYECTO,
    )
