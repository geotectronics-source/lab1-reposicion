"""Regresión cuantílica lineal: el segundo modelo de aprendizaje.

Añadir otro gradient boosting (XGBoost, CatBoost) habría sido poco útil:
comparten con LightGBM el mismo sesgo inductivo —particiones perpendiculares
a los ejes— y producirían resultados casi idénticos. Una comparación sólo
informa cuando los modelos se equivocan de formas distintas.

Un modelo lineal aporta tres cosas que el conjunto no tenía:

1. **Diversidad real.** Si se acerca al modelo de árboles, significa que la
   señal es mayormente aditiva; si queda muy por detrás, que las
   interacciones importan. En ambos casos se aprende algo sobre el problema.
2. **Degradación suave.** Ante un cambio de régimen —inflación, una familia
   nueva, un rango de precios no visto— un lineal extrapola de forma
   predecible, mientras que un árbol devuelve el valor de la hoja más cercana
   y se queda clavado en el rango que conoció.
3. **Interpretabilidad.** Los coeficientes se leen directamente: cuánto suma
   un punto de promoción, cuánto resta un feriado.

**Por qué una implementación propia.** `sklearn.linear_model.QuantileRegressor`
resuelve un programa lineal exacto, con un costo que lo hace inviable sobre
1,3 millones de ejemplos. Aquí se minimiza la misma pérdida con descenso de
gradiente, que escala linealmente. La pérdida pinball es convexa, así que el
óptimo es el mismo; lo que cambia es cómo se llega. Hay una prueba que
compara los coeficientes de ambas implementaciones en un caso pequeño, que es
la forma de confiar en código propio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from reposicion.models.base import columna, ordenar_cuantiles, predictoras

# Sólo se codifican `store_nbr` y `family`.
#
# `city`, `state`, `type` y `cluster` son funciones determinísticas de la
# tienda: cada tienda tiene una ciudad fija. Al codificar `store_nbr` con
# variables indicadoras, esas cuatro pasan a ser combinaciones lineales
# exactas de las columnas ya presentes, es decir colinealidad perfecta. No
# añaden información y vuelven inestables los coeficientes, repartiendo un
# mismo efecto entre columnas equivalentes de forma arbitraria.
#
# Al modelo de árboles no le estorban —puede partir por cualquiera de ellas—,
# y por eso allí sí se conservan. Es un ejemplo concreto de que el
# preprocesamiento correcto depende de la familia de modelos.
A_CODIFICAR = ("store_nbr", "family")

# Variables expresadas en unidades de venta, de transacciones o de precio, que
# se transforman con log1p antes de estandarizar.
#
# El modelo predice `log1p(y)`, así que la relación correcta con el nivel de
# la serie es logarítmica: duplicar la venta histórica suma una constante al
# logaritmo, no un múltiplo del nivel. Pasar `media_56` en unidades (0 a
# 11.594) contra un objetivo logarítmico es un error de especificación, y el
# modelo responde ignorando esas variables y prediciendo casi sólo por
# familia.
#
# El modelo de árboles no lo sufre: aprende cualquier relación monótona
# partiendo el rango, de modo que la transformación le resulta indiferente.
# Es otro caso donde el preprocesamiento depende de la familia de modelos.
EN_UNIDADES = (
    "ventas_",
    "media_",
    "mediana_",
    "max_",
    "desv_",
    "emp_q",
    "trans_media_",
    "promo_media_",
    "petroleo_",
)


class RegresionCuantilicaLineal:
    """Minimiza la pérdida pinball sobre un modelo lineal, con descenso de gradiente."""

    nombre = "lineal_cuantilico"

    def __init__(
        self,
        cuantiles: list[float],
        *,
        epocas: int = 60,
        tam_lote: int = 8192,
        tasa: float = 0.05,
        l2: float = 1e-4,
        semilla: int = 42,
    ):
        self.cuantiles = list(cuantiles)
        self.epocas = epocas
        self.tam_lote = tam_lote
        self.tasa = tasa
        self.l2 = l2
        self.semilla = semilla

        self.coeficientes: dict[float, np.ndarray] = {}
        self.sesgos: dict[float, float] = {}
        self.columnas: list[str] = []
        self.categorias: dict[str, list] = {}
        self.medias: np.ndarray | None = None
        self.escalas: np.ndarray | None = None
        self.nombres_columnas: list[str] = []

    # -- preparación --------------------------------------------------------

    def _matriz(self, panel: pd.DataFrame, ajustar: bool) -> np.ndarray:
        # Las columnas de texto que no se codifican se descartan: son las
        # redundantes con `store_nbr` y un modelo lineal no puede usarlas.
        numericas = [
            c
            for c in self.columnas
            if c not in A_CODIFICAR and pd.api.types.is_numeric_dtype(panel[c])
        ]
        bruto = panel[numericas].astype(float).to_numpy()
        # Las columnas en unidades pasan a escala logarítmica, la misma del
        # objetivo. Sin esto el modelo no puede usar el nivel de la serie.
        en_unidades = np.array([c.startswith(EN_UNIDADES) for c in numericas])
        if en_unidades.any():
            bruto[:, en_unidades] = np.log1p(np.maximum(0.0, bruto[:, en_unidades]))

        bloques = [bruto]
        nombres = list(numericas)

        for col in A_CODIFICAR:
            if col not in self.columnas:
                continue
            if ajustar:
                self.categorias[col] = sorted(panel[col].dropna().unique().tolist())
            valores = self.categorias[col]
            # Se omite la primera categoría para no crear colinealidad exacta
            # con el término independiente.
            codigos = panel[col].to_numpy()[:, None] == np.array(valores[1:], dtype=object)
            bloques.append(codigos.astype(float))
            nombres.extend(f"{col}={v}" for v in valores[1:])

        X = np.hstack(bloques)

        if ajustar:
            self.nombres_columnas = nombres
            # La mediana resiste los valores extremos de las series grandes
            # mejor que la media, y aquí hay ventas de hasta 124.717.
            self.medias = np.nanmedian(X, axis=0)
            escalas = np.nanstd(X, axis=0)
            # Una columna constante no aporta y dividir por su escala daría
            # infinito: se neutraliza con escala 1.
            self.escalas = np.where(escalas < 1e-8, 1.0, escalas)

        # Un valor ausente pasa a ser "el valor típico" tras centrar: es la
        # imputación más neutra para un modelo lineal.
        X = np.where(np.isnan(X), self.medias, X)
        return (X - self.medias) / self.escalas

    # -- ajuste -------------------------------------------------------------

    def _ajustar_un_cuantil(self, X: np.ndarray, y: np.ndarray, q: float) -> tuple:
        rng = np.random.default_rng(self.semilla)
        n, p = X.shape
        w = np.zeros(p)
        b = float(np.quantile(y, q))  # arrancar en el cuantil marginal acelera mucho

        # Adam: la pérdida pinball tiene gradiente constante a cada lado del
        # cero, así que un paso fijo oscila. Adam adapta el paso por
        # coordenada y estabiliza la convergencia.
        m_w, v_w = np.zeros(p), np.zeros(p)
        m_b = v_b = 0.0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        paso = 0

        for _ in range(self.epocas):
            orden = rng.permutation(n)
            for inicio in range(0, n, self.tam_lote):
                idx = orden[inicio : inicio + self.tam_lote]
                Xb, yb = X[idx], y[idx]

                residuo = yb - (Xb @ w + b)
                # Subgradiente de la pinball: -q donde se queda corto,
                # (1-q) donde se pasa.
                g = np.where(residuo > 0, -q, 1.0 - q)

                grad_w = Xb.T @ g / len(idx) + self.l2 * w
                grad_b = float(g.mean())

                paso += 1
                m_w = beta1 * m_w + (1 - beta1) * grad_w
                v_w = beta2 * v_w + (1 - beta2) * grad_w**2
                m_b = beta1 * m_b + (1 - beta1) * grad_b
                v_b = beta2 * v_b + (1 - beta2) * grad_b**2

                corr1 = 1 - beta1**paso
                corr2 = 1 - beta2**paso
                w -= self.tasa * (m_w / corr1) / (np.sqrt(v_w / corr2) + eps)
                b -= self.tasa * (m_b / corr1) / (np.sqrt(v_b / corr2) + eps)

        return w, b

    def entrenar(self, panel: pd.DataFrame) -> RegresionCuantilicaLineal:
        if "y" not in panel.columns:
            raise ValueError("El panel de entrenamiento necesita la columna objetivo 'y'.")

        self.columnas = predictoras(panel)
        X = self._matriz(panel, ajustar=True)
        # Misma escala logarítmica que el modelo de árboles, para que la
        # comparación entre ambos sea de modelos y no de transformaciones.
        y = np.log1p(panel["y"].clip(lower=0.0).to_numpy(dtype=float))

        for q in self.cuantiles:
            self.coeficientes[q], self.sesgos[q] = self._ajustar_un_cuantil(X, y, q)

        return self

    # -- predicción ---------------------------------------------------------

    def predecir(self, panel: pd.DataFrame) -> pd.DataFrame:
        if not self.coeficientes:
            raise RuntimeError("El modelo no ha sido entrenado.")

        faltantes = [c for c in self.columnas if c not in panel.columns]
        if faltantes:
            raise ValueError(f"Faltan variables usadas al entrenar: {faltantes}")

        salida = panel.copy()
        X = self._matriz(salida, ajustar=False)

        for q in self.cuantiles:
            bruto = X @ self.coeficientes[q] + self.sesgos[q]
            salida[columna(q)] = np.maximum(0.0, np.expm1(bruto))

        return ordenar_cuantiles(salida, self.cuantiles)

    # -- interpretación -----------------------------------------------------

    def coeficientes_ordenados(self, q: float = 0.5, top: int = 20) -> pd.DataFrame:
        """Variables con más peso, en unidades de desviación estándar.

        Como las columnas están estandarizadas, los coeficientes son
        comparables entre sí: miden el efecto de moverse una desviación típica.
        """
        if q not in self.coeficientes:
            raise RuntimeError(f"No hay un estimador entrenado para el cuantil {q}.")
        return (
            pd.DataFrame(
                {"variable": self.nombres_columnas, "coeficiente": self.coeficientes[q]}
            )
            .assign(magnitud=lambda d: d["coeficiente"].abs())
            .sort_values("magnitud", ascending=False)
            .drop(columns="magnitud")
            .head(top)
            .reset_index(drop=True)
        )
