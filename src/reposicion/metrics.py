"""Métricas del pronóstico y del negocio.

Hay una identidad que conviene tener presente porque une las dos mitades del
producto. El costo de una decisión de reposición es::

    costo = cu * max(0, real - pedido) + co * max(0, pedido - real)

donde `cu` es el costo de quedarse sin producto y `co` el de que sobre. La
pérdida pinball en el cuantil q es::

    L_q = q * max(0, real - pred) + (1 - q) * max(0, pred - real)

Sustituyendo q* = cu / (cu + co) se obtiene::

    costo = (cu + co) * L_q*

Es decir: **minimizar la pérdida pinball en el cuantil crítico es exactamente
minimizar el costo de negocio**. No son dos objetivos que haya que equilibrar,
son el mismo objetivo en distinta escala. Por eso el modelo se entrena con
pérdida pinball y se evalúa con dinero sin que haya contradicción entre ambos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from reposicion.config import cargar_config

# ---------------------------------------------------------------------------
# Costos de negocio
# ---------------------------------------------------------------------------


def costos_de_familia(familia: str, config: dict | None = None) -> tuple[float, float]:
    """Costos (quiebre, merma) de una familia, con los valores por defecto."""
    cfg = config or cargar_config("costs")
    base = cfg["defaults"]
    ajuste = cfg.get("familias", {}).get(familia, {}) or {}
    return (
        float(ajuste.get("costo_quiebre", base["costo_quiebre"])),
        float(ajuste.get("costo_merma", base["costo_merma"])),
    )


def cuantil_critico(costo_quiebre: float, costo_merma: float) -> float:
    """Proporción del riesgo que conviene cubrir: `cu / (cu + co)`.

    Si quebrar duele el triple que tirar producto, se cubre hasta el 75 %: el
    punto donde el costo marginal de pedir una unidad más iguala al de no
    pedirla.
    """
    total = costo_quiebre + costo_merma
    if total <= 0:
        raise ValueError("Los costos deben sumar más de cero para definir un cuantil.")
    return costo_quiebre / total


def costo_negocio(
    real: pd.Series | np.ndarray,
    pedido: pd.Series | np.ndarray,
    costo_quiebre: float,
    costo_merma: float,
) -> float:
    """Costo medio por serie-día de una decisión de reposición.

    Es la métrica que decide qué modelo gana. Un error estadístico menor no
    sirve de nada si se equivoca en la dirección cara.
    """
    real = np.asarray(real, dtype=float)
    pedido = np.asarray(pedido, dtype=float)
    faltante = np.maximum(0.0, real - pedido)
    sobrante = np.maximum(0.0, pedido - real)
    return float((costo_quiebre * faltante + costo_merma * sobrante).mean())


def costo_negocio_por_familia(
    df: pd.DataFrame, columna_pedido: str, config: dict | None = None
) -> float:
    """Costo total ponderando cada familia con sus propios costos.

    Evaluar todo con los costos por defecto escondería justo lo que importa:
    equivocarse en PRODUCE (perecedero) cuesta distinto que en GROCERY I.
    """
    cfg = config or cargar_config("costs")
    total = 0.0
    for familia, bloque in df.groupby("family", observed=True):
        cu, co = costos_de_familia(str(familia), cfg)
        total += costo_negocio(bloque["y"], bloque[columna_pedido], cu, co) * len(bloque)
    return total / len(df) if len(df) else 0.0


# ---------------------------------------------------------------------------
# Métricas del pronóstico
# ---------------------------------------------------------------------------


def pinball(real, pred, q: float) -> float:
    """Pérdida pinball: penaliza de forma asimétrica según el cuantil.

    Es la métrica propia de un pronóstico cuantílico. Para q=0.9 quedarse
    corto cuesta nueve veces más que pasarse, que es justo el comportamiento
    que se le pide a un cuantil alto.
    """
    real = np.asarray(real, dtype=float)
    pred = np.asarray(pred, dtype=float)
    diferencia = real - pred
    return float(np.maximum(q * diferencia, (q - 1) * diferencia).mean())


def pinball_medio(df: pd.DataFrame, cuantiles: list[float], prefijo: str = "q") -> float:
    """Promedio de la pérdida pinball sobre todos los cuantiles estimados.

    Aproxima el CRPS y resume en un solo número la calidad de toda la
    distribución predicha, no sólo la de su centro.
    """
    return float(np.mean([pinball(df["y"], df[f"{prefijo}{q}"], q) for q in cuantiles]))


def cobertura(real, pred, q: float) -> float:
    """Proporción de observaciones por debajo del cuantil predicho.

    Debería acercarse a q. Una cobertura del 70 % en el cuantil 0.9 significa
    que el modelo subestima la incertidumbre y el pedido se quedará corto más
    a menudo de lo previsto; por encima, se está pidiendo de más.
    """
    return float((np.asarray(real, dtype=float) <= np.asarray(pred, dtype=float)).mean())


def mae(real, pred) -> float:
    return float(np.abs(np.asarray(real, dtype=float) - np.asarray(pred, dtype=float)).mean())


def rmsle(real, pred) -> float:
    """Error cuadrático medio en escala logarítmica.

    La métrica de la competencia de Kaggle. Trata el error de forma relativa,
    de modo que fallar por 10 unidades en una serie que vende 20 pesa más que
    fallar por 10 en una que vende 5.000. Se reporta por comparabilidad, pero
    no es la que decide la promoción de un modelo.
    """
    real = np.maximum(0.0, np.asarray(real, dtype=float))
    pred = np.maximum(0.0, np.asarray(pred, dtype=float))
    return float(np.sqrt(((np.log1p(pred) - np.log1p(real)) ** 2).mean()))


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------


def evaluar(
    df: pd.DataFrame,
    cuantiles: list[float],
    *,
    prefijo: str = "q",
    config_costos: dict | None = None,
) -> dict:
    """Panel de métricas de un conjunto de predicciones ya evaluadas.

    Espera `y` con el valor observado y una columna por cuantil estimado.
    Devuelve las técnicas y la de negocio juntas, porque leerlas por separado
    invita a elegir el modelo por la que conviene.
    """
    mediana = f"{prefijo}0.5"
    resumen: dict[str, float] = {
        "n": float(len(df)),
        "mae": mae(df["y"], df[mediana]),
        "rmsle": rmsle(df["y"], df[mediana]),
        "pinball_medio": pinball_medio(df, cuantiles, prefijo),
    }

    for q in cuantiles:
        resumen[f"cobertura_{q}"] = cobertura(df["y"], df[f"{prefijo}{q}"], q)

    # Costo del pedido que saldría de aplicar la política a este pronóstico.
    resumen["costo_negocio"] = costo_negocio_por_familia(
        df.assign(_pedido=_pedido_por_cuantil_critico(df, cuantiles, prefijo, config_costos)),
        "_pedido",
        config_costos,
    )
    return resumen


def _pedido_por_cuantil_critico(
    df: pd.DataFrame, cuantiles: list[float], prefijo: str, config: dict | None
) -> pd.Series:
    """Pedido que resultaría de la política, para poder medir su costo.

    La política definitiva vive en `policy.py`; aquí se reproduce lo mínimo
    para que la comparación entre modelos se haga en la moneda del negocio y
    no sólo en error estadístico.
    """
    cfg = config or cargar_config("costs")
    pedido = pd.Series(np.nan, index=df.index)

    for familia, bloque in df.groupby("family", observed=True):
        cu, co = costos_de_familia(str(familia), cfg)
        objetivo = cuantil_critico(cu, co)
        # Se usa el cuantil estimado más cercano: interpolar entre dos daría
        # una precisión que el modelo no tiene.
        elegido = min(cuantiles, key=lambda q: abs(q - objetivo))
        pedido.loc[bloque.index] = bloque[f"{prefijo}{elegido}"]

    return pedido
