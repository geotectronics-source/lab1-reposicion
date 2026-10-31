"""Validación temporal y criterio de promoción de modelos.

Un `KFold` aleatorio sobre series temporales entrena con el futuro y evalúa
con el pasado: produce métricas excelentes y carentes de sentido. Aquí cada
pliegue entrena con lo anterior a un corte y evalúa con los días siguientes,
que es exactamente lo que ocurre en operación.

Todos los modelos se validan con **los mismos pliegues y los mismos
ejemplos**. Comparar modelos evaluados por separado es la forma más fácil de
llegar a una conclusión falsa.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from reposicion.config import cargar_config
from reposicion.metrics import evaluar


@dataclass(frozen=True)
class Pliegue:
    """Un corte temporal: se entrena con lo de antes y se evalúa con lo de después."""

    numero: int
    entrena_hasta: pd.Timestamp
    evalua_hasta: pd.Timestamp

    def __str__(self) -> str:
        return (
            f"pliegue {self.numero}: entrena <= {self.entrena_hasta:%Y-%m-%d}, "
            f"evalúa <= {self.evalua_hasta:%Y-%m-%d}"
        )


def pliegues_moviles(hasta: pd.Timestamp, numero: int, dias: int) -> list[Pliegue]:
    """Cortes de origen móvil, el más reciente al final.

    Cada pliegue avanza el corte `dias` hacia adelante, de modo que el modelo
    se evalúa varias veces sobre periodos distintos. Un único corte podría
    caer en una semana atípica y dar una conclusión que no se sostiene.
    """
    hasta = pd.Timestamp(hasta).normalize()
    pliegues = []
    for i in range(numero, 0, -1):
        evalua_hasta = hasta - pd.Timedelta(days=(i - 1) * dias)
        pliegues.append(
            Pliegue(
                numero=numero - i + 1,
                entrena_hasta=evalua_hasta - pd.Timedelta(days=dias),
                evalua_hasta=evalua_hasta,
            )
        )
    return pliegues


def partir(panel: pd.DataFrame, pliegue: Pliegue) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa entrenamiento y evaluación por la fecha que se predice.

    El criterio es la fecha objetivo y no la del origen: un ejemplo cuyo
    objetivo cae después del corte pertenece al futuro, por muy antiguo que
    sea su origen.
    """
    entrena = panel[panel["date"] <= pliegue.entrena_hasta]
    evalua = panel[
        (panel["date"] > pliegue.entrena_hasta) & (panel["date"] <= pliegue.evalua_hasta)
    ]
    return entrena, evalua


def validar(
    modelos: list,
    panel: pd.DataFrame,
    pliegues: list[Pliegue],
    cuantiles: list[float],
    *,
    config_costos: dict | None = None,
    verboso: bool = False,
) -> pd.DataFrame:
    """Evalúa cada modelo en cada pliegue y devuelve una fila por combinación."""
    filas = []

    for pliegue in pliegues:
        entrena, evalua = partir(panel, pliegue)
        if entrena.empty or evalua.empty:
            continue
        if verboso:
            print(f"{pliegue}  (entrena {len(entrena):,} / evalúa {len(evalua):,})")

        for modelo in modelos:
            ajustado = modelo.entrenar(entrena)
            predicciones = ajustado.predecir(evalua)
            metricas = evaluar(predicciones, cuantiles, config_costos=config_costos)
            filas.append({"modelo": modelo.nombre, "pliegue": pliegue.numero, **metricas})
            if verboso:
                print(
                    f"  {modelo.nombre:<22} costo={metricas['costo_negocio']:.4f} "
                    f"pinball={metricas['pinball_medio']:.4f} rmsle={metricas['rmsle']:.4f}"
                )

    return pd.DataFrame(filas)


def resumir(resultados: pd.DataFrame) -> pd.DataFrame:
    """Promedia las métricas de cada modelo sobre los pliegues."""
    numericas = resultados.select_dtypes("number").columns.drop("pliegue")
    return resultados.groupby("modelo")[list(numericas)].mean().reset_index()


# ---------------------------------------------------------------------------
# Promoción
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """Resultado de comparar un retador con el modelo vigente."""

    promover: bool
    razon: str
    mejoras: dict[int, float]

    def __str__(self) -> str:
        detalle = ", ".join(f"p{n}: {m:+.1%}" for n, m in sorted(self.mejoras.items()))
        return f"{'PROMOVER' if self.promover else 'MANTENER'} — {self.razon} ({detalle})"


def decidir_promocion(
    resultados: pd.DataFrame,
    vigente: str,
    retador: str,
    *,
    config: dict | None = None,
) -> Decision:
    """Decide si el retador reemplaza al vigente, con un criterio explícito.

    Dos elecciones deliberadas, ambas versionadas en `model.yaml`:

    - **Manda el costo de negocio**, no el error estadístico. Un modelo con
      mejor RMSLE que se equivoca en la dirección cara es peor producto.
    - **Debe ganar en todos los pliegues**, no en el promedio. Un modelo que
      mejora mucho en uno y empeora en otro es inestable, y la media lo
      esconde: en producción se notaría como semanas buenas y semanas malas
      sin explicación aparente.
    """
    cfg = (config or cargar_config("model"))["promocion"]
    metrica = cfg["metrica"]
    umbral = float(cfg["mejora_minima"])
    todos = bool(cfg.get("en_todos_los_pliegues", True))

    tabla = resultados.pivot(index="pliegue", columns="modelo", values=metrica)
    if vigente not in tabla.columns or retador not in tabla.columns:
        raise KeyError(f"Faltan resultados de '{vigente}' o '{retador}' en la comparación.")

    # La métrica es un costo: mejorar significa reducirlo.
    mejoras = ((tabla[vigente] - tabla[retador]) / tabla[vigente]).to_dict()

    suficientes = [m >= umbral for m in mejoras.values()]
    gana = all(suficientes) if todos else (sum(suficientes) > len(suficientes) / 2)

    if gana:
        alcance = "todos los" if todos else "la mayoría de"
        razon = f"mejora {metrica} al menos {umbral:.0%} en {alcance} pliegues"
    else:
        peor = min(mejoras.values())
        razon = f"la mejora mínima observada ({peor:+.1%}) no alcanza el umbral de {umbral:.0%}"

    return Decision(promover=gana, razon=razon, mejoras=mejoras)


def elegir_campeon(
    resultados: pd.DataFrame,
    *,
    config: dict | None = None,
) -> tuple[str, dict[str, Decision]]:
    """Qué modelo pasa a producción, decidido con el criterio versionado.

    Hay más de un retador, así que el criterio se aplica dos veces y en este
    orden:

    1. Cada retador tiene que **ganarle al vigente** según `decidir_promocion`.
       Un retador que no supera el umbral queda fuera aunque sea el mejor de
       los retadores: el vigente no se cambia por algo que no demostró ser
       mejor.
    2. Entre los que pasan, se promueve el de **menor costo medio**.

    El orden no es intercambiable. Comparar primero a los retadores entre sí
    y luego al ganador contra el vigente permitiría promover un modelo que
    gana por poco al vigente sólo porque otro retador era peor todavía.

    Devuelve el nombre del modelo promovido y la decisión frente al vigente de
    cada retador, para que el pipeline la registre entera: saber *por qué no*
    se promovió un modelo vale tanto como saber por qué sí.
    """
    cfg = (config or cargar_config("model"))["promocion"]
    vigente = cfg["vigente"]
    retadores = [r for r in cfg["retadores"] if r in set(resultados["modelo"])]

    decisiones = {r: decidir_promocion(resultados, vigente, r, config=config) for r in retadores}
    medias = resumir(resultados).set_index("modelo")[cfg["metrica"]]

    promovibles = [r for r, d in decisiones.items() if d.promover]
    campeon = min(promovibles, key=lambda r: medias[r]) if promovibles else vigente

    return campeon, decisiones
