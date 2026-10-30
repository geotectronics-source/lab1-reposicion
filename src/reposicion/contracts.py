"""Contratos de datos: qué debe cumplir cada fuente para entrar al producto.

El INVARIANTE 5 exige distinguir un dato inválido de una situación de negocio.
Aquí eso se concreta en tres niveles, decididos tras perfilar los CSV reales:

1. **Estructura** — falta una columna, la fecha es ilegible, la clave primaria
   se repite. Es un defecto del sistema de origen: cuarentena y la tarea falla.
2. **Valor imposible** — una venta negativa, un precio de petróleo negativo.
   No puede ocurrir en el negocio, así que es un defecto: cuarentena y falla.
3. **Vocabulario nuevo** — una tienda que no existía, un tipo de feriado no
   visto antes. Es negocio: se procesa y se registra un aviso.

La tentación de escribir `Check.isin([...])` sobre el tipo de feriado sería un
error: si el evaluador aporta un feriado de un tipo nuevo, el DAG fallaría por
una situación perfectamente normal. Se exige la estructura, no el vocabulario.

**Regla que ordena los nulos**: las columnas que forman la clave de negocio y
la variable objetivo no admiten nulos (sin ellas no hay nada que calcular);
las variables explicativas sí los admiten y se imputan aguas abajo.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema


class ErrorDeContrato(Exception):
    """Los datos no cumplen el contrato de su fuente.

    Lleva el detalle de Pandera y el origen (el archivo) para que el mensaje
    en los logs de Airflow sirva sin tener que reproducir el fallo.
    """

    def __init__(self, fuente: str, origen: str, detalle: str):
        self.fuente = fuente
        self.origen = origen
        self.detalle = detalle
        super().__init__(
            f"La fuente '{fuente}' no cumple su contrato"
            f"{f' en {origen}' if origen else ''}:\n{detalle}"
        )


# ---------------------------------------------------------------------------
# Claves de negocio
# ---------------------------------------------------------------------------
# Identifican unívocamente un hecho. Son también la clave con la que se
# resuelven las correcciones por last-write-wins (INVARIANTE 4).
#
# `holidays_events` no usa `date` sola a propósito: el perfilado encontró 31
# fechas con varias filas (un feriado nacional y uno local el mismo día).
# Deduplicar por fecha descartaría los feriados regionales, que son justo los
# que explican los picos de venta de una ciudad.
CLAVES: dict[str, list[str]] = {
    "sales": ["date", "store_nbr", "family"],
    "transactions": ["date", "store_nbr"],
    "oil": ["date"],
    "holidays_events": ["date", "type", "locale", "locale_name", "description"],
    "stores": ["store_nbr"],
}


# ---------------------------------------------------------------------------
# Esquemas
# ---------------------------------------------------------------------------

_FECHA = Column("datetime64[ns]", nullable=False, coerce=True)

ESQUEMAS: dict[str, DataFrameSchema] = {
    # El perfilado confirmó: 0 nulos, 0 negativos, 0 duplicados y rejilla
    # completa de 54 tiendas x 33 familias. La rejilla NO se exige: una tienda
    # recién abierta no vende todas las familias desde el primer día.
    "sales": DataFrameSchema(
        {
            "date": _FECHA,
            "store_nbr": Column(int, Check.gt(0), nullable=False, coerce=True),
            "family": Column(str, nullable=False, coerce=True),
            # Objetivo del modelo: un nulo aquí no se puede imputar sin inventar.
            # Hay 462k valores decimales (productos vendidos por peso), así que
            # el tipo es float y no int.
            "sales": Column(float, Check.ge(0), nullable=False, coerce=True),
            # Variable explicativa: admite nulos, se imputa a 0 aguas abajo.
            "onpromotion": Column("Int64", Check.ge(0), nullable=True, coerce=True),
        },
        unique=CLAVES["sales"],
        strict=False,  # se permiten columnas extra como `id` o `arrival_date`
        name="sales",
    ),
    # No es rejilla completa (83,488 de 90,828 posibles): una tienda cerrada
    # simplemente no aparece ese día. La ausencia es negocio, no error.
    "transactions": DataFrameSchema(
        {
            "date": _FECHA,
            "store_nbr": Column(int, Check.gt(0), nullable=False, coerce=True),
            "transactions": Column("Int64", Check.ge(0), nullable=True, coerce=True),
        },
        unique=CLAVES["transactions"],
        strict=False,
        name="transactions",
    ),
    # 43 nulos reales, y el primero está en la fila 1 (2013-01-01): el petróleo
    # no cotiza en fines de semana ni feriados bursátiles. `nullable=True` no es
    # una concesión, es el contrato correcto de esta fuente.
    "oil": DataFrameSchema(
        {
            "date": _FECHA,
            "dcoilwtico": Column(float, Check.gt(0), nullable=True, coerce=True),
        },
        unique=CLAVES["oil"],
        strict=False,
        name="oil",
    ),
    # Única fuente que informa sobre el futuro. El vocabulario de `type` y
    # `locale` NO se restringe: ver el aviso de negocio más abajo.
    "holidays_events": DataFrameSchema(
        {
            "date": _FECHA,
            "type": Column(str, nullable=False, coerce=True),
            "locale": Column(str, nullable=False, coerce=True),
            "locale_name": Column(str, nullable=False, coerce=True),
            "description": Column(str, nullable=False, coerce=True),
            "transferred": Column(bool, nullable=False, coerce=True),
        },
        unique=CLAVES["holidays_events"],
        strict=False,
        name="holidays_events",
    ),
    "stores": DataFrameSchema(
        {
            "store_nbr": Column(int, Check.gt(0), nullable=False, coerce=True),
            "city": Column(str, nullable=False, coerce=True),
            "state": Column(str, nullable=False, coerce=True),
            "type": Column(str, nullable=False, coerce=True),
            "cluster": Column(int, Check.gt(0), nullable=False, coerce=True),
        },
        unique=CLAVES["stores"],
        strict=False,
        name="stores",
    ),
}


# ---------------------------------------------------------------------------
# Vocabulario conocido (nivel 3: avisos, nunca errores)
# ---------------------------------------------------------------------------
# Lo observado en los datos públicos. Un valor fuera de estas listas no invalida
# nada: significa que el negocio cambió y conviene enterarse.
VOCABULARIO_CONOCIDO: dict[str, dict[str, set[str]]] = {
    "holidays_events": {
        "type": {"Additional", "Bridge", "Event", "Holiday", "Transfer", "Work Day"},
        "locale": {"Local", "National", "Regional"},
    },
    "stores": {
        "type": {"A", "B", "C", "D", "E"},
    },
}

# Rangos observados en los datos públicos, para detectar expansión del negocio.
TIENDAS_CONOCIDAS = 54
FAMILIAS_CONOCIDAS = 33


def avisos_de_negocio(df: pd.DataFrame, fuente: str) -> list[str]:
    """Detecta cambios del negocio que merecen registrarse pero no detener nada.

    Se calcula de forma explícita en vez de usar `raise_warning` de Pandera
    para que los avisos sean un valor de retorno (fácil de registrar en los
    logs del DAG y de probar) y no un efecto colateral del sistema de warnings.
    """
    avisos: list[str] = []

    for columna, conocidos in VOCABULARIO_CONOCIDO.get(fuente, {}).items():
        if columna not in df.columns:
            continue
        nuevos = sorted(set(df[columna].dropna().astype(str)) - conocidos)
        if nuevos:
            avisos.append(
                f"{fuente}.{columna}: valores no vistos antes {nuevos}. "
                "Se procesan igual; revisar si el negocio cambió."
            )

    if "store_nbr" in df.columns:
        nuevas = sorted(int(s) for s in df["store_nbr"].dropna().unique() if s > TIENDAS_CONOCIDAS)
        if nuevas:
            avisos.append(
                f"{fuente}: tiendas fuera del rango conocido 1-{TIENDAS_CONOCIDAS}: {nuevas}. "
                "Sin historia suficiente, el pronóstico caerá al baseline."
            )

    if fuente == "sales" and "family" in df.columns:
        familias = df["family"].nunique()
        if familias > FAMILIAS_CONOCIDAS:
            avisos.append(
                f"sales: {familias} familias, más de las {FAMILIAS_CONOCIDAS} conocidas."
            )

    return avisos


def validar(df: pd.DataFrame, fuente: str, *, origen: str = "") -> pd.DataFrame:
    """Aplica el contrato de `fuente` y devuelve el DataFrame con tipos fijados.

    Lanza `ErrorDeContrato` si no lo cumple. `lazy=True` recolecta todos los
    fallos en vez de detenerse en el primero: cuando un archivo del sistema
    origen viene mal, interesa ver todo lo que está mal de una vez.
    """
    if fuente not in ESQUEMAS:
        raise KeyError(
            f"No hay contrato definido para la fuente '{fuente}'. Conocidas: {sorted(ESQUEMAS)}"
        )

    try:
        return ESQUEMAS[fuente].validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise ErrorDeContrato(fuente, origen, str(exc.failure_cases)) from exc
    except pa.errors.SchemaError as exc:  # pragma: no cover - lazy casi siempre agrupa
        raise ErrorDeContrato(fuente, origen, str(exc)) from exc
