"""Auditoría diaria del almacén: qué llegó, qué falta y qué se corrigió.

Esta es la respuesta del producto a dos exigencias que parecen la misma y no
lo son:

- *"Una fuente no llegó hoy"* es una **situación de negocio**: se sigue
  adelante con lo último conocido y **se registra el hueco** (INVARIANTE 5).
  Registrarlo no es cortesía: sin un rastro, el día que alguien pregunte por
  qué el pronóstico de esa semana salió raro no habrá nada que mirar.
- *"La ingesta no corrió"* es un **fallo**. Y desde fuera se parece mucho a lo
  anterior: en los dos casos no hay datos nuevos. La diferencia es el
  **retraso acumulado**: que `transactions` no entregue un martes es normal;
  que su último dato sea de hace ocho días significa que el flujo está roto y
  que el pronóstico del lunes se emitiría con información vieja sin que nadie
  se enterara.

El umbral que separa las dos cosas es una decisión de negocio y vive en
`config/seguimiento.yaml`, no aquí.

**Por qué el umbral no se lee de `config/sources.yaml`.** Ahí están declaradas
las latencias reales de cada fuente, que es exactamente el dato que haría
falta. Pero ese archivo lo consume el simulador del sistema origen y el
producto no lo mira (INVARIANTE 2): si el producto dedujera sus expectativas
del archivo que describe al simulador, dejaría de ser un producto que opera
contra un sistema ajeno. El producto declara **qué retraso tolera**; si el
origen real es más lento, eso es una conversación con el negocio y no un
cambio de código.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from reposicion.config import cargar_config, rutas
from reposicion.contracts import CLAVES
from reposicion.ingesta import FUENTES, leer_raw_asof
from reposicion.io import COLUMNA_LLEGADA, normalizar_fecha
from reposicion.landing import vista_vigente

#: Estados posibles de una fuente. El orden es el de gravedad.
AL_DIA = "al dia"
HUECO = "hueco"
ATRASADO = "atrasado"
VACIO = "sin datos"


def auditar(asof: str | pd.Timestamp, *, config: dict | None = None) -> pd.DataFrame:
    """Estado de cada fuente del almacén en la fecha lógica `asof`.

    Una fila por fuente. No lanza nada: decidir qué hacer con un estado es de
    quien llama, porque el mismo hecho —no hay datos nuevos— significa cosas
    distintas según cuánto lleve pasando.
    """
    dia = normalizar_fecha(asof)
    cfg = config or cargar_config("seguimiento")
    tolerado = cfg.get("retraso_maximo_dias") or {}

    filas = []
    for fuente in FUENTES:
        historia = leer_raw_asof(fuente, dia)
        limite = tolerado.get(fuente)

        if historia.empty:
            filas.append(
                {
                    "fuente": fuente,
                    "particiones": 0,
                    "llego_en_asof": False,
                    "filas_en_asof": 0,
                    "reenviadas_en_asof": 0,
                    "correcciones_en_asof": 0,
                    "filas_vigentes": 0,
                    "ultimo_dato": pd.NaT,
                    "retraso_dias": pd.NA,
                    "retraso_tolerado": limite,
                    "estado": VACIO,
                }
            )
            continue

        llegada = historia[COLUMNA_LLEGADA]
        hoy = historia[llegada == dia]
        antes = historia[llegada < dia]
        reenviadas, corregidas = _reenvios(hoy, antes, fuente)

        filas.append(
            {
                "fuente": fuente,
                "particiones": int(llegada.nunique()),
                "llego_en_asof": bool(len(hoy)),
                "filas_en_asof": int(len(hoy)),
                "reenviadas_en_asof": reenviadas,
                "correcciones_en_asof": corregidas,
                "filas_vigentes": int(len(vista_vigente(historia, fuente))),
                **_retraso(historia, dia, limite),
            }
        )

    return pd.DataFrame(filas).sort_values("fuente").reset_index(drop=True)


def _reenvios(hoy: pd.DataFrame, antes: pd.DataFrame, fuente: str) -> tuple[int, int]:
    """Cuántas filas de hoy ya se conocían, y de ésas cuántas traen otro valor.

    Las dos cuentas son la medida observable del INVARIANTE 4: una corrección
    no sobrescribe nada, llega como registro nuevo, y lo único que delata que
    hubo corrección es que su clave ya estaba.

    **Pero reenviar no es corregir, y contarlo junto exagera el problema.** La
    primera versión de esta función devolvía un solo número y, el día de la
    corrección retroactiva del dataset, informó de 1.782 correcciones: el
    sistema origen reenvió el día completo —como hace un sistema real— y sólo
    33 filas traían un valor distinto. Un reporte que dice que se corrigieron
    1.782 filas cuando cambiaron 33 manda a investigar un incidente que no
    existe, y la próxima vez nadie lo mira.

    - `reenviadas`: la clave ya se conocía. Normal: muchas fuentes republican.
    - `corregidas`: la clave ya se conocía **y algún valor cambió**. Esto es lo
      que puede invalidar algo que ya se publicó.

    La comparación se hace contra la vista **vigente** de lo anterior, no
    contra todas las llegadas: la pregunta es si cambia lo que hoy creemos.
    """
    clave = CLAVES[fuente]
    if hoy.empty or antes.empty or any(columna not in hoy.columns for columna in clave):
        return 0, 0

    previo = vista_vigente(antes, fuente)
    cruce = hoy.merge(previo, on=clave, how="inner", suffixes=("", "_previo"))
    if cruce.empty:
        return 0, 0

    # Se comparan los valores, no la clave ni la fecha de llegada: la llegada
    # es distinta por definición y compararla daría que todo cambió siempre.
    difiere = pd.Series(False, index=cruce.index)
    for columna in hoy.columns:
        if columna in clave or columna == COLUMNA_LLEGADA:
            continue
        anterior = f"{columna}_previo"
        if anterior not in cruce.columns:
            continue
        actual, pasado = cruce[columna], cruce[anterior]
        # Dos nulos son iguales para esto: `oil` no cotiza el fin de semana y
        # un nulo reenviado como nulo no es ninguna corrección.
        difiere |= ~((actual == pasado) | (actual.isna() & pasado.isna()))

    return int(len(cruce)), int(difiere.sum())


def _retraso(historia: pd.DataFrame, asof: pd.Timestamp, limite: int | None) -> dict:
    """Antigüedad del dato más reciente y veredicto sobre ella.

    Se mide sobre `date` (a qué día se refieren los datos) y no sobre
    `arrival_date` (cuándo los supimos). Confundirlas aquí daría siempre
    retraso cero: la última llegada es, por definición, reciente.
    """
    if "date" not in historia.columns:
        # `stores` es un catálogo sin fecha: su antigüedad no significa nada.
        return {
            "ultimo_dato": pd.NaT,
            "retraso_dias": pd.NA,
            "retraso_tolerado": limite,
            "estado": AL_DIA,
        }

    # `holidays_events` informa del futuro, así que su retraso sale negativo;
    # se recorta a cero porque "el calendario llega con 60 días de adelanto"
    # no es un estado que haya que vigilar.
    ultimo = historia["date"].max()
    retraso = max(0, int((asof - ultimo).days))

    if limite is None:
        estado = AL_DIA
    elif retraso > int(limite):
        estado = ATRASADO
    elif historia[COLUMNA_LLEGADA].max() < asof:
        estado = HUECO
    else:
        estado = AL_DIA

    return {
        "ultimo_dato": ultimo,
        "retraso_dias": retraso,
        "retraso_tolerado": limite,
        "estado": estado,
    }


def atrasadas(reporte: pd.DataFrame) -> list[str]:
    """Fuentes cuyo retraso ya no se explica por la operación normal."""
    return sorted(reporte.loc[reporte["estado"].isin({ATRASADO, VACIO}), "fuente"])


def huecos(reporte: pd.DataFrame) -> list[str]:
    """Fuentes que hoy no entregaron pero siguen dentro de lo tolerado."""
    return sorted(reporte.loc[reporte["estado"] == HUECO, "fuente"])


def ruta_reporte(asof: str | pd.Timestamp) -> Path:
    """Dónde se materializa el reporte de una fecha lógica."""
    dia = normalizar_fecha(asof)
    return rutas().seguimiento / f"asof={dia:%Y-%m-%d}" / "calidad.csv"
