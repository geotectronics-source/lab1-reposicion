"""Modelo de aprendizaje: LightGBM con pérdida pinball, un modelo por cuantil.

Tres decisiones lo definen.

**Un único modelo global para las 1.782 series.** Un modelo por serie tendría
muy pocas observaciones cada uno y no podría aprender nada que no estuviera en
esa serie. El global aprende patrones compartidos —el efecto del día de pago,
de los feriados, de la estacionalidad semanal— y los aplica incluso a series
con poca historia. La identidad de la serie entra como variable categórica,
así que el modelo puede especializarse donde haga falta.

**Entrenamiento en escala logarítmica.** Las ventas van de 0 a 124.717: sin
transformar, la pérdida la dominarían unas pocas series enormes y el resto
quedaría mal ajustado. La transformación es válida para cuantiles por una
razón precisa::

    Q_q(log(1 + Y)) = log(1 + Q_q(Y))

porque `log1p` es monótona creciente y los cuantiles se preservan bajo
transformaciones monótonas. Deshacerla con `expm1` devuelve el cuantil
correcto, sin el sesgo de retransformación que aparecería si se estimara una
media en vez de un cuantil.

**Pérdida pinball, que es la métrica de negocio.** Como el costo de reposición
equivale a la pérdida pinball en el cuantil crítico (ver `metrics`), entrenar
con esta pérdida optimiza directamente lo que se quiere minimizar en dinero.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from reposicion.models.base import columna, ordenar_cuantiles, predictoras, preparar_matriz

PARAMS_POR_DEFECTO = {
    "objective": "quantile",
    "n_estimators": 300,
    "learning_rate": 0.08,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "n_jobs": -1,
}


class LightGBMCuantilico:
    """Un estimador por cuantil, todos sobre el mismo conjunto de ejemplos."""

    nombre = "lgbm_cuantilico"

    def __init__(
        self,
        cuantiles: list[float],
        params: dict | None = None,
        semilla: int = 42,
    ):
        self.cuantiles = list(cuantiles)
        self.params = {**PARAMS_POR_DEFECTO, **(params or {})}
        self.semilla = semilla
        self.modelos: dict[float, lgb.LGBMRegressor] = {}
        self.columnas: list[str] = []

    def entrenar(self, panel: pd.DataFrame) -> LightGBMCuantilico:
        if "y" not in panel.columns:
            raise ValueError("El panel de entrenamiento necesita la columna objetivo 'y'.")

        self.columnas = predictoras(panel)
        X = preparar_matriz(panel, self.columnas)
        y = np.log1p(panel["y"].clip(lower=0.0))

        for q in self.cuantiles:
            modelo = lgb.LGBMRegressor(**self.params, alpha=q, random_state=self.semilla)
            modelo.fit(X, y, categorical_feature="auto")
            self.modelos[q] = modelo

        return self

    def predecir(self, panel: pd.DataFrame) -> pd.DataFrame:
        if not self.modelos:
            raise RuntimeError("El modelo no ha sido entrenado.")

        salida = panel.copy()
        # Las columnas de predicción deben coincidir con las de entrenamiento,
        # en el mismo orden: LightGBM asocia por posición, no por nombre.
        faltantes = [c for c in self.columnas if c not in salida.columns]
        if faltantes:
            raise ValueError(f"Faltan variables usadas al entrenar: {faltantes}")

        X = preparar_matriz(salida, self.columnas)
        for q, modelo in self.modelos.items():
            # Se deshace el logaritmo y se recorta en cero: una venta negativa
            # no existe, y la política la interpretaría como pedir de menos.
            salida[columna(q)] = np.maximum(0.0, np.expm1(modelo.predict(X)))

        return ordenar_cuantiles(salida, self.cuantiles)

    def importancias(self, top: int = 20) -> pd.DataFrame:
        """Variables más usadas por el estimador de la mediana.

        Sirve para comprobar que el modelo se apoya en lo que tiene sentido de
        negocio y no en un artefacto de construcción del panel.
        """
        if 0.5 not in self.modelos:
            raise RuntimeError("No hay un estimador de la mediana entrenado.")
        modelo = self.modelos[0.5]
        return (
            pd.DataFrame(
                {"variable": self.columnas, "importancia": modelo.feature_importances_}
            )
            .sort_values("importancia", ascending=False)
            .head(top)
            .reset_index(drop=True)
        )
