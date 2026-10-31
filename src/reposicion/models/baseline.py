"""Modelo de referencia: cuantiles empíricos por día de la semana.

El piso contra el que se mide todo lo demás. No aprende nada: para cada serie
y cada día de la semana devuelve los cuantiles observados en las últimas
semanas, que ya vienen calculados en el panel.

Elegirlo como referencia no es una concesión. En retail el día de la semana
explica la mayor parte de la variación, así que es un rival sorprendentemente
duro, y cuesta microsegundos. Un modelo de aprendizaje que no lo supere con
holgura no justifica su costo de mantenimiento, su riesgo de fallo ni el
tiempo de cómputo que consume cada semana.
"""

from __future__ import annotations

import pandas as pd

from reposicion.models.base import columna, ordenar_cuantiles


class BaselineEstacional:
    """Repite lo que la serie suele hacer ese día de la semana."""

    nombre = "baseline_estacional"

    def __init__(self, cuantiles: list[float]):
        self.cuantiles = list(cuantiles)

    def entrenar(self, panel: pd.DataFrame) -> BaselineEstacional:
        """No hay nada que ajustar: su conocimiento está en el propio panel.

        Se mantiene el método para que la validación pueda tratar a todos los
        modelos igual y los pliegues sean idénticos.
        """
        return self

    def predecir(self, panel: pd.DataFrame) -> pd.DataFrame:
        salida = panel.copy()

        for q in self.cuantiles:
            empirico = f"emp_q{q}"
            prediccion = (
                salida[empirico]
                if empirico in salida.columns
                else pd.Series(float("nan"), index=salida.index)
            )
            # Una serie sin suficientes observaciones del día pedido (tienda
            # recién abierta, producto nuevo) cae al nivel reciente, y si
            # tampoco lo hay, a cero: no vender nada es la mejor conjetura
            # cuando no hay ninguna evidencia.
            respaldo = salida.get("media_7", pd.Series(0.0, index=salida.index))
            salida[columna(q)] = prediccion.fillna(respaldo).fillna(0.0).clip(lower=0.0)

        return ordenar_cuantiles(salida, self.cuantiles)
