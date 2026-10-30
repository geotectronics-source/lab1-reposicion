"""Lectura punto-en-el-tiempo de la bandeja.

Responde a una sola pregunta, que es la que sostiene todo el producto:
**¿qué sabíamos de esta fuente en la fecha D?**

    leer_asof("sales", "2017-07-15")   # la tienda 44 aparece en ceros
    leer_asof("sales", "2017-07-25")   # ya llegó la corrección: valores reales

La misma función, dos respuestas distintas, ambas correctas. Eso es lo que
permite entrenar y evaluar sin fuga temporal, y reconstruir meses después el
pronóstico que se publicó en una fecha concreta.

Dos decisiones de diseño sostienen el módulo:

1. **El filtro es por `arrival_date`, nunca por `date`** (INVARIANTE 1). La
   fecha de llegada vive en el nombre de la carpeta, así que decidir si un
   archivo entra al cómputo no requiere abrirlo.

2. **Las correcciones se resuelven en lectura, no en escritura** (INVARIANTE 4).
   Todo lo recibido se conserva; `vista_vigente` elige la versión más reciente
   según `arrival_date`. Sobrescribir al recibir sería más simple y haría
   imposible auditar qué se sabía antes de la corrección.

Este módulo SÓLO LEE de `landing/` (INVARIANTE 2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from reposicion.config import rutas
from reposicion.contracts import CLAVES, validar
from reposicion.io import COLUMNA_LLEGADA, a_fecha, marcar_llegada, normalizar_fecha

# El histórico es la carga inicial: por definición precede a cualquier entrega
# de la bandeja. Se le asigna una fecha de llegada anterior a cualquier dato
# real para que siempre pierda frente a una corrección posterior.
ARRIVAL_HISTORICO = pd.Timestamp("1900-01-01")

_PATRON_ARRIVAL = re.compile(r"^arrival_date=(\d{4}-\d{2}-\d{2})$")



@dataclass(frozen=True)
class Entrega:
    """Un archivo encontrado en la bandeja."""

    arrival_date: pd.Timestamp
    fuente: str
    ruta: Path

    @property
    def nombre(self) -> str:
        return self.ruta.name


def _leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee un CSV de la bandeja dejando `date` siempre como datetime.

    El tipo de salida del módulo NO puede depender de si se validó el contrato.
    Si la conversión la hiciera sólo Pandera, una lectura con
    `validar_contrato=False` devolvería `date` como texto, y comparar un
    Timestamp con una cadena no lanza error: simplemente no coincide nunca.
    Es un fallo silencioso que produce resultados vacíos en vez de una
    excepción, que es la peor clase de fallo en un flujo automatizado.
    """
    df = pd.read_csv(ruta)
    if "date" in df.columns:
        # pandas 3 parsea en microsegundos, pero el contrato (y Parquet aguas
        # abajo) esperan nanosegundos. Fijarlo aquí evita que una lectura
        # validada y una sin validar devuelvan dtypes distintos.
        df["date"] = a_fecha(df["date"])
    return df


def descubrir_entregas(
    fuente: str, asof: str | pd.Timestamp, *, landing: Path | None = None
) -> list[Entrega]:
    """Entregas de `fuente` que YA HABÍAN LLEGADO en `asof`.

    Las carpetas que no siguen el patrón `arrival_date=YYYY-MM-DD` se ignoran:
    la bandeja es un sistema externo y puede contener archivos ajenos al
    producto. Sólo se consideran `.csv`, de modo que un `.tmp` de una escritura
    en curso del sistema origen nunca se lee a medias.
    """
    base = landing or rutas().landing
    limite = normalizar_fecha(asof)
    encontradas: list[Entrega] = []

    if not base.exists():
        return encontradas

    for carpeta in sorted(base.iterdir()):
        if not carpeta.is_dir():
            continue
        coincidencia = _PATRON_ARRIVAL.match(carpeta.name)
        if coincidencia is None:
            continue

        llegada = pd.Timestamp(coincidencia.group(1))
        # EL filtro que define la corrección punto-en-el-tiempo.
        if llegada > limite:
            continue

        for archivo in sorted((carpeta / fuente).glob("*.csv")):
            encontradas.append(Entrega(arrival_date=llegada, fuente=fuente, ruta=archivo))

    return sorted(encontradas, key=lambda e: (e.arrival_date, e.nombre))


def entregas_del_dia(
    fuente: str, arrival_date: str | pd.Timestamp, *, landing: Path | None = None
) -> list[Entrega]:
    """Entregas llegadas EXACTAMENTE en `arrival_date`.

    La ingesta trabaja por partición: el run del día D procesa sólo lo que
    llegó en D, no todo lo anterior. Así cada ejecución toca una única
    partición, la idempotencia es trivial y el backfill de un rango es
    simplemente un run por día.
    """
    base = landing or rutas().landing
    dia = normalizar_fecha(arrival_date)
    carpeta = base / f"arrival_date={dia:%Y-%m-%d}" / fuente

    if not carpeta.is_dir():
        return []

    return [
        Entrega(arrival_date=dia, fuente=fuente, ruta=archivo)
        for archivo in sorted(carpeta.glob("*.csv"))
    ]


def leer_historico(
    fuente: str, *, landing: Path | None = None, validar_contrato: bool = True
) -> pd.DataFrame:
    """Carga inicial de una fuente. Siempre disponible, sea cual sea `asof`."""
    base = landing or rutas().landing
    ruta = base / "historico" / f"{fuente}.csv"
    if not ruta.exists():
        return pd.DataFrame()

    df = _leer_csv(ruta)
    if validar_contrato:
        df = validar(df, fuente, origen=str(ruta))
    return marcar_llegada(df, ARRIVAL_HISTORICO)


def leer_asof(
    fuente: str,
    asof: str | pd.Timestamp,
    *,
    landing: Path | None = None,
    incluir_historico: bool = True,
    validar_contrato: bool = True,
) -> pd.DataFrame:
    """Todo lo conocido de `fuente` hasta `asof`, con su fecha de llegada.

    Devuelve el historial COMPLETO de lo recibido, con duplicados cuando un
    dato fue corregido. Para quedarse con la versión que rige en `asof`, pasar
    el resultado por `vista_vigente`.

    Cada archivo se valida por separado para que el mensaje de error indique
    cuál incumplió el contrato. `validar_contrato=False` existe para las
    lecturas repetidas de datos ya validados en la ingesta: validar 2.9
    millones de filas en cada tarea sería un costo inútil.
    """
    partes: list[pd.DataFrame] = []

    if incluir_historico:
        historico = leer_historico(fuente, landing=landing, validar_contrato=validar_contrato)
        if not historico.empty:
            partes.append(historico)

    for entrega in descubrir_entregas(fuente, asof, landing=landing):
        df = _leer_csv(entrega.ruta)
        if validar_contrato:
            df = validar(df, fuente, origen=str(entrega.ruta))
        partes.append(marcar_llegada(df, entrega.arrival_date))

    if not partes:
        return pd.DataFrame()

    return pd.concat(partes, ignore_index=True)


def vista_vigente(df: pd.DataFrame, fuente: str) -> pd.DataFrame:
    """Resuelve las correcciones: para cada clave, gana la llegada más reciente.

    `keep="last"` sobre un orden estable por `arrival_date` implementa
    last-write-wins sin perder nada: el historial completo sigue en `df`, esto
    sólo elige qué versión rige.
    """
    if df.empty:
        return df

    clave = CLAVES[fuente]
    faltantes = [c for c in clave if c not in df.columns]
    if faltantes:
        raise KeyError(f"Faltan columnas de la clave de '{fuente}': {faltantes}")

    return (
        df.sort_values(COLUMNA_LLEGADA, kind="stable")
        .drop_duplicates(subset=clave, keep="last")
        .sort_values(clave, kind="stable")
        .reset_index(drop=True)
    )


def leer_vigente(
    fuente: str,
    asof: str | pd.Timestamp,
    *,
    landing: Path | None = None,
    validar_contrato: bool = True,
) -> pd.DataFrame:
    """Atajo para el caso habitual: lo que rige en `asof`, ya resuelto."""
    return vista_vigente(
        leer_asof(fuente, asof, landing=landing, validar_contrato=validar_contrato), fuente
    )


def historial_de(df: pd.DataFrame, fuente: str, **filtros) -> pd.DataFrame:
    """Todas las versiones recibidas de un hecho concreto, en orden de llegada.

    Es la herramienta de auditoría que justifica conservar las correcciones en
    vez de sobrescribirlas: permite responder "¿qué nos dijeron y cuándo?"::

        historial_de(df, "sales", date="2017-07-05", store_nbr=44)
    """
    if df.empty:
        return df

    mascara = pd.Series(True, index=df.index)
    for columna, valor in filtros.items():
        objetivo = normalizar_fecha(valor) if columna == "date" else valor
        mascara &= df[columna] == objetivo

    return df.loc[mascara].sort_values(COLUMNA_LLEGADA, kind="stable").reset_index(drop=True)


def fuentes_en_bandeja(asof: str | pd.Timestamp, *, landing: Path | None = None) -> list[str]:
    """Fuentes con al menos una entrega hasta `asof`.

    Sirve al DAG de ingesta para registrar qué llegó y qué no, sin tratar la
    ausencia de una fuente como un fallo (INVARIANTE 5).
    """
    base = landing or rutas().landing
    limite = normalizar_fecha(asof)
    fuentes: set[str] = set()

    if not base.exists():
        return []

    for carpeta in base.iterdir():
        coincidencia = _PATRON_ARRIVAL.match(carpeta.name) if carpeta.is_dir() else None
        if coincidencia is None or pd.Timestamp(coincidencia.group(1)) > limite:
            continue
        fuentes.update(sub.name for sub in carpeta.iterdir() if sub.is_dir())

    return sorted(fuentes)
