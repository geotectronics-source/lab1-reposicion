"""Emisión del pronóstico: del panel a los cuantiles publicados.

Esta capa existe por una razón de negocio, no de ingeniería. Un modelo de
aprendizaje sólo puede opinar sobre series de las que vio historia; una tienda
que abrió hace tres semanas no la tiene. Para esas series el pronóstico no se
omite —el gerente necesita pedir igual— sino que lo produce el modelo de
referencia, que sólo necesita las últimas semanas observadas.

Es la INVARIANTE 5 aplicada: "tienda nueva sin historia suficiente" es una
**situación de negocio**, no un dato inválido, así que se resuelve con un
respaldo documentado y jamás hace fallar la tarea.

La columna `modelo` registra quién produjo cada fila. Sin ella, una serie
atendida por el respaldo sería indistinguible de una atendida por el modelo
promovido, y el seguimiento de desempeño compararía peras con manzanas.
"""

from __future__ import annotations

import pandas as pd

from reposicion.config import cargar_config
from reposicion.features import CLAVE_SERIE
from reposicion.io import marcar_fecha
from reposicion.models.baseline import BaselineEstacional

#: Fecha de emisión. Separada de `origen` a propósito: el origen es el último
#: día con datos y la emisión es el día en que se publica el pedido. Entre
#: ambos hay latencia, y el seguimiento necesita saber cuándo se publicó.
COLUMNA_EMISION = "emitida"


def emitir(
    panel: pd.DataFrame,
    modelo,
    asof: str | pd.Timestamp,
    *,
    respaldo=None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Pronostica todo el panel, derivando al respaldo lo que el modelo no cubre.

    Devuelve el panel con una columna por cuantil, la fecha de emisión y el
    nombre del modelo que atendió cada fila.
    """
    cfg = config or cargar_config("model")
    cuantiles = cfg["pronostico"]["cuantiles"]
    minimo = int(cfg["entrenamiento"].get("minimo_dias_historia", 0))

    if panel.empty:
        return panel

    referencia = respaldo if respaldo is not None else BaselineEstacional(cuantiles)

    # Sin la columna no hay forma de saber qué series son nuevas, así que se
    # asume que todas tienen historia: es el caso de un panel sintético de
    # prueba, no el de producción.
    if "dias_historia" in panel.columns:
        atendible = panel["dias_historia"] >= minimo
    else:
        atendible = pd.Series(True, index=panel.index)

    partes = []
    for subconjunto, quien in ((atendible, modelo), (~atendible, referencia)):
        if not subconjunto.any():
            continue
        prediccion = quien.predecir(panel[subconjunto].copy())
        prediccion["modelo"] = getattr(quien, "nombre", type(quien).__name__)
        partes.append(prediccion)

    emitido = marcar_fecha(pd.concat(partes, ignore_index=True), COLUMNA_EMISION, asof)
    return emitido.sort_values([*CLAVE_SERIE, "date"]).reset_index(drop=True)


def cobertura_del_respaldo(emitido: pd.DataFrame) -> pd.DataFrame:
    """Cuántas series atendió cada modelo.

    Es la cifra que hay que vigilar: si el respaldo empieza a cubrir una parte
    creciente del catálogo, el problema no es el pronóstico sino que el
    reentrenamiento dejó de alcanzar a las series nuevas.
    """
    if emitido.empty:
        return pd.DataFrame(columns=["modelo", "series", "filas"])

    filas = emitido.groupby("modelo", observed=True).agg(filas=("date", "size")).reset_index()
    series = (
        emitido.drop_duplicates(["modelo", *CLAVE_SERIE])
        .groupby("modelo", observed=True)
        .size()
        .rename("series")
        .reset_index()
    )
    return filas.merge(series, on="modelo").sort_values("modelo").reset_index(drop=True)
