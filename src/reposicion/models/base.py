"""Contrato común de los modelos y utilidades compartidas.

Que todos los modelos expongan la misma interfaz es lo que permite validarlos
con el mismo código y los mismos pliegues. Comparar modelos evaluados con
rutinas distintas es la forma más fácil de llegar a una conclusión falsa.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

# Columnas que describen el ejemplo pero no pueden usarse para predecir.
# `y` es la respuesta; `date` y `origen` son fechas absolutas, y dejarlas
# entrar permitiría al modelo memorizar periodos concretos en lugar de
# aprender el patrón que los explica.
NO_PREDICTORAS = ("y", "date", "origen")

CATEGORICAS = ("family", "city", "state", "type", "store_nbr", "cluster")


@runtime_checkable
class Modelo(Protocol):
    """Lo mínimo que debe ofrecer un modelo para entrar en la comparación."""

    nombre: str
    cuantiles: list[float]

    def entrenar(self, panel: pd.DataFrame) -> Modelo: ...

    def predecir(self, panel: pd.DataFrame) -> pd.DataFrame: ...


def columna(q: float) -> str:
    return f"q{q}"


def predictoras(panel: pd.DataFrame) -> list[str]:
    """Columnas utilizables como variables explicativas."""
    return [c for c in panel.columns if c not in NO_PREDICTORAS and not c.startswith("q")]


def preparar_matriz(panel: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Deja el panel listo para LightGBM, con las categóricas tipadas."""
    X = panel[columnas].copy()
    for col in CATEGORICAS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype("int8")
    return X


def ordenar_cuantiles(panel: pd.DataFrame, cuantiles: list[float]) -> pd.DataFrame:
    """Impide que los cuantiles se crucen entre sí.

    Cada cuantil se estima con un modelo independiente, así que nada garantiza
    que el 0.9 salga por encima del 0.5. Un cruce no es sólo feo: haría que
    subir el costo de quiebre en la configuración produjera un pedido MENOR,
    y la política dejaría de ser interpretable.

    Se corrige con un máximo acumulado sobre los cuantiles ordenados, que es
    la proyección isotónica más simple y sólo mueve los valores incoherentes.
    """
    ordenados = sorted(cuantiles)
    columnas = [columna(q) for q in ordenados]
    valores = np.maximum.accumulate(panel[columnas].to_numpy(dtype=float), axis=1)
    panel[columnas] = valores
    return panel
