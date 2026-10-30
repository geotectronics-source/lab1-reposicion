"""Escritura y lectura de datos particionados por fecha de llegada.

Aquí vive la mecánica que hace idempotentes a las tareas de Airflow
(INVARIANTE 3): escritura atómica y particiones que se reescriben completas,
nunca a trozos.

Las dos zonas tienen estructuras distintas a propósito::

    landing/arrival_date=2017-07-06/sales/sales_2017-07-05.csv
    data/raw/sales/arrival_date=2017-07-06/datos.parquet

La bandeja se organiza por fecha porque así deposita un sistema de origen
real: todo lo que mandó hoy, junto. El almacén propio se organiza por fuente
porque así se consulta: todo el histórico de ventas, junto. Imitar la
estructura del origen en el almacén obligaría a recorrer 59 carpetas para
leer una sola fuente.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd

PATRON_PARTICION = re.compile(r"^arrival_date=(\d{4}-\d{2}-\d{2})$")
ARCHIVO_PARTICION = "datos.parquet"

#: Columna que registra cuándo se supo cada fila. Es la que sostiene la
#: corrección punto-en-el-tiempo, así que su nombre vive en un solo lugar.
COLUMNA_LLEGADA = "arrival_date"

#: Resolución temporal del proyecto. pandas 3 infiere microsegundos al parsear
#: y al construir desde un escalar, pero el contrato y Parquet esperan
#: nanosegundos. Fijarlo en un único sitio evita que dos caminos distintos
#: produzcan dtypes distintos, un fallo que no lanza error: sólo hace que las
#: comparaciones dejen de coincidir.
DTYPE_FECHA = "datetime64[ns]"


def normalizar_fecha(fecha: str | pd.Timestamp) -> pd.Timestamp:
    """Acepta texto o Timestamp y devuelve una fecha sin hora ni zona.

    Airflow entrega `logical_date` como datetime con zona horaria; el negocio
    opera en una sola zona y los datos son diarios, así que todo se reduce a
    la fecha del calendario.
    """
    ts = pd.Timestamp(fecha)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def a_fecha(serie: pd.Series) -> pd.Series:
    """Convierte una columna a la resolución temporal del proyecto."""
    return pd.to_datetime(serie).astype(DTYPE_FECHA)


def marcar_llegada(df: pd.DataFrame, arrival_date: str | pd.Timestamp) -> pd.DataFrame:
    """Sella el DataFrame con la fecha en que se conoció.

    Asignar un `Timestamp` escalar dejaría la columna en microsegundos; se
    construye la Serie con el dtype explícito para que todas las rutas de
    lectura entreguen exactamente el mismo tipo.
    """
    df[COLUMNA_LLEGADA] = pd.Series(
        normalizar_fecha(arrival_date), index=df.index, dtype=DTYPE_FECHA
    )
    return df


def ruta_particion(base: Path, fuente: str, arrival_date: str | pd.Timestamp) -> Path:
    """Dónde vive la partición de una fuente para una fecha de llegada."""
    return base / fuente / f"arrival_date={normalizar_fecha(arrival_date):%Y-%m-%d}"


def escribir_particion(df: pd.DataFrame, destino: Path) -> Path:
    """Escribe una partición completa de forma atómica.

    Se escribe a un archivo temporal y se renombra al destino. `Path.replace`
    es atómico dentro del mismo volumen, así que un fallo a mitad de escritura
    deja la partición anterior intacta en lugar de un Parquet truncado que la
    siguiente tarea leería como válido.

    La partición se REESCRIBE completa. No existe modo de agregado: ejecutar
    la tarea dos veces debe dar el mismo resultado que ejecutarla una
    (INVARIANTE 3), y un `append` duplicaría las filas en el segundo intento.
    """
    destino.mkdir(parents=True, exist_ok=True)
    final = destino / ARCHIVO_PARTICION
    tmp = destino / f".{ARCHIVO_PARTICION}.tmp"

    df.to_parquet(tmp, index=False, engine="pyarrow", compression="snappy")
    tmp.replace(final)
    return final


def listar_particiones(
    base: Path, fuente: str, *, hasta: str | pd.Timestamp | None = None
) -> list[tuple[pd.Timestamp, Path]]:
    """Particiones existentes de una fuente, opcionalmente hasta una fecha.

    `hasta` implementa el filtro punto-en-el-tiempo sobre el almacén propio:
    es el mismo criterio que en la bandeja, por fecha de llegada y nunca por
    fecha del dato.
    """
    carpeta = base / fuente
    if not carpeta.exists():
        return []

    limite = normalizar_fecha(hasta) if hasta is not None else None
    encontradas: list[tuple[pd.Timestamp, Path]] = []

    for sub in sorted(carpeta.iterdir()):
        if not sub.is_dir():
            continue
        coincidencia = PATRON_PARTICION.match(sub.name)
        if coincidencia is None:
            continue
        llegada = pd.Timestamp(coincidencia.group(1))
        if limite is not None and llegada > limite:
            continue
        archivo = sub / ARCHIVO_PARTICION
        if archivo.exists():
            encontradas.append((llegada, archivo))

    return encontradas


def leer_particiones(
    base: Path, fuente: str, *, hasta: str | pd.Timestamp | None = None
) -> pd.DataFrame:
    """Concatena las particiones de una fuente hasta una fecha de llegada."""
    particiones = listar_particiones(base, fuente, hasta=hasta)
    if not particiones:
        return pd.DataFrame()

    partes = [pd.read_parquet(archivo, engine="pyarrow") for _, archivo in particiones]
    return pd.concat(partes, ignore_index=True)


def copiar_a_cuarentena(origen: Path, destino_dir: Path, motivo: str) -> Path:
    """Aparta un archivo que no cumple su contrato, con el motivo al lado.

    Se copia en vez de moverse porque el original está en la bandeja, que es
    un sistema externo de sólo lectura (INVARIANTE 2). La copia permite
    inspeccionar el archivo aunque el origen lo reemplace después.
    """
    destino_dir.mkdir(parents=True, exist_ok=True)
    copia = destino_dir / origen.name
    shutil.copy2(origen, copia)
    copia.with_suffix(copia.suffix + ".motivo.txt").write_text(motivo, encoding="utf-8")
    return copia
