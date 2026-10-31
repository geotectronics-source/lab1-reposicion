"""Política de reposición: del pronóstico al pedido sugerido.

La decisión de cuánto pedir es el problema del vendedor de periódicos. Con un
costo `cu` por quedarse sin producto y `co` por que sobre, el pedido óptimo es
el cuantil::

    q* = cu / (cu + co)

La intuición: pedir una unidad más cuesta `co` si sobra y ahorra `cu` si
falta. El punto de equilibrio está donde la probabilidad de que falte iguala
esa proporción. Si quebrar duele el triple que tirar producto, conviene
cubrirse hasta el 75 %.

**Aquí no hay números de negocio.** Los costos viven en `config/costs.yaml`.
Subir `costo_quiebre` en ese archivo aumenta el pedido sin tocar una línea de
Python ni reentrenar el modelo, que es el requisito central del producto.

**Granularidad.** El pronóstico es diario y el pedido se emite una vez por
semana, de modo que el resultado es un *plan diario de reposición*: el centro
de distribución despacha cada día según ese plan. Por eso el cuantil diario
es la decisión correcta. El total semanal que se reporta al gerente es la
suma de esas siete decisiones, no el cuantil de la demanda semanal: si el
abastecimiento pasara a ser un único envío, habría que estimar el cuantil de
la suma, que es menor que la suma de los cuantiles porque la variabilidad
diaria se compensa en parte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from reposicion.config import cargar_config
from reposicion.metrics import costos_de_familia, cuantil_critico

CLAVE_SERIE = ["store_nbr", "family"]


def cuantil_objetivo(familia: str, config: dict | None = None) -> float:
    """Proporción del riesgo que conviene cubrir para una familia."""
    cu, co = costos_de_familia(familia, config)
    return cuantil_critico(cu, co)


def cuantil_disponible(objetivo: float, cuantiles: list[float]) -> float:
    """Cuantil estimado más cercano al objetivo teórico.

    No se interpola entre dos cuantiles. El modelo se entrenó para estimar
    unos valores concretos y nada garantiza que la distribución entre ellos
    sea lineal: interpolar daría una precisión que el modelo no tiene. Se
    documenta la diferencia en la columna `cuantil_usado` para que la
    aproximación quede a la vista.

    Ante un empate se elige el cuantil más alto: cubrirse de más cuesta merma,
    cubrirse de menos cuesta la venta y, a veces, el cliente.

    La distancia se redondea antes de comparar porque en coma flotante
    `abs(0.85 - 0.8)` vale 0,049999…93 y `abs(0.9 - 0.85)` vale 0,050000…04.
    Sin redondear, un empate exacto nunca se detecta y gana siempre el cuantil
    bajo, que es el lado arriesgado.
    """
    return min(cuantiles, key=lambda q: (round(abs(q - objetivo), 9), -q))


def calcular_pedido(
    pronostico: pd.DataFrame,
    *,
    cuantiles: list[float] | None = None,
    config_costos: dict | None = None,
    config_modelo: dict | None = None,
) -> pd.DataFrame:
    """Convierte un pronóstico cuantílico en un pedido sugerido por serie y día.

    Devuelve el pronóstico con cuatro columnas añadidas que hacen auditable la
    decisión: el cuantil que pedían los costos, el que realmente se usó, el
    pedido y si se aplicó el tope de seguridad. Un pedido que nadie puede
    explicar no se sigue, por bueno que sea.
    """
    costos = config_costos or cargar_config("costs")
    modelo = config_modelo or cargar_config("model")
    disponibles = cuantiles or modelo["pronostico"]["cuantiles"]
    politica = costos.get("politica", {}) or {}

    salida = pronostico.copy()
    salida["cuantil_objetivo"] = np.nan
    salida["cuantil_usado"] = np.nan
    salida["pedido_sugerido"] = np.nan

    for familia, bloque in salida.groupby("family", observed=True):
        objetivo = cuantil_objetivo(str(familia), costos)
        elegido = cuantil_disponible(objetivo, disponibles)

        salida.loc[bloque.index, "cuantil_objetivo"] = objetivo
        salida.loc[bloque.index, "cuantil_usado"] = elegido
        salida.loc[bloque.index, "pedido_sugerido"] = bloque[f"q{elegido}"].to_numpy()

    salida["pedido_sugerido"] = salida["pedido_sugerido"].clip(lower=0.0)
    return _aplicar_tope(salida, politica)


def _aplicar_tope(pedido: pd.DataFrame, politica: dict) -> pd.DataFrame:
    """Limita la sugerencia a un múltiplo de la venta media reciente.

    Es una red de seguridad, no parte del modelo: protege contra un pronóstico
    desbocado por un dato anómalo que haya superado la validación. Sin ella,
    un único valor extremo podría traducirse en un pedido que llena el almacén
    de una tienda.

    El recorte se registra en `tope_aplicado` en vez de aplicarse en silencio:
    si una tienda aparece recortada semana tras semana, no es un pronóstico
    anómalo sino un modelo que se quedó corto con esa serie, y conviene verlo.
    """
    multiplo = politica.get("tope_multiplo_media")
    pedido["tope_aplicado"] = False

    if not multiplo or "media_28" not in pedido.columns:
        return pedido

    # Un mínimo absoluto evita que una serie de venta casi nula quede con un
    # tope de cero y no pueda reponerse nunca.
    limite = np.maximum(float(multiplo) * pedido["media_28"].fillna(0.0), 1.0)
    excede = pedido["pedido_sugerido"] > limite

    pedido.loc[excede, "tope_aplicado"] = True
    pedido.loc[excede, "pedido_sugerido"] = limite[excede]
    return pedido


def resumen_por_serie(pedido: pd.DataFrame) -> pd.DataFrame:
    """Total a pedir por tienda y familia en el horizonte completo.

    Es la vista que recibe el gerente: una línea por serie con el total de la
    semana. El desglose diario sigue disponible para el centro de
    distribución, que despacha día a día.
    """
    agregado = (
        pedido.groupby(CLAVE_SERIE, observed=True)
        .agg(
            dias=("pedido_sugerido", "size"),
            pedido_semana=("pedido_sugerido", "sum"),
            pronostico_central=("q0.5", "sum"),
            cuantil_usado=("cuantil_usado", "first"),
            recortes=("tope_aplicado", "sum"),
        )
        .reset_index()
    )
    # Cuánto se pide por encima del escenario central: es el colchón que
    # compra la política, y conviene que sea explícito.
    agregado["colchon"] = agregado["pedido_semana"] - agregado["pronostico_central"]
    return agregado.sort_values(CLAVE_SERIE).reset_index(drop=True)


def explicar(pedido: pd.DataFrame, config_costos: dict | None = None) -> pd.DataFrame:
    """Tabla de una línea por familia explicando de dónde sale su cuantil.

    Sirve para responder a la pregunta que siempre llega de operaciones:
    "¿por qué para PRODUCE pides menos que para GROCERY I?".
    """
    costos = config_costos or cargar_config("costs")
    filas = []
    for familia in sorted(pedido["family"].unique()):
        cu, co = costos_de_familia(str(familia), costos)
        bloque = pedido[pedido["family"] == familia]
        filas.append(
            {
                "family": familia,
                "costo_quiebre": cu,
                "costo_merma": co,
                "cuantil_objetivo": cuantil_critico(cu, co),
                "cuantil_usado": bloque["cuantil_usado"].iloc[0],
                "pedido_total": bloque["pedido_sugerido"].sum(),
            }
        )
    return pd.DataFrame(filas).sort_values("cuantil_objetivo").reset_index(drop=True)
