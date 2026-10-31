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

import json
import re
import shutil
from collections.abc import Callable
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


def marcar_fecha(df: pd.DataFrame, columna: str, fecha: str | pd.Timestamp) -> pd.DataFrame:
    """Añade una columna con la misma fecha en todas las filas.

    Existe porque asignar un `Timestamp` escalar deja la columna en
    microsegundos y el proyecto exige nanosegundos. Es un fallo silencioso:
    no lanza error, sólo hace que las comparaciones con otras columnas de
    fecha dejen de coincidir y el resultado salga vacío.

    Esta función es la cuarta versión del mismo arreglo. Las tres anteriores
    fueron parches en el sitio donde apareció el problema —la ingesta, el
    panel, la emisión— hasta que quedó claro que el patrón es general:
    *sellar un DataFrame con una fecha*. Mientras hubiera un camino que no
    pasara por aquí, el error volvería.
    """
    df[columna] = pd.Series(normalizar_fecha(fecha), index=df.index, dtype=DTYPE_FECHA)
    return df


def marcar_llegada(df: pd.DataFrame, arrival_date: str | pd.Timestamp) -> pd.DataFrame:
    """Sella el DataFrame con la fecha en que se conoció."""
    return marcar_fecha(df, COLUMNA_LLEGADA, arrival_date)


def ruta_particion(base: Path, fuente: str, arrival_date: str | pd.Timestamp) -> Path:
    """Dónde vive la partición de una fuente para una fecha de llegada."""
    return base / fuente / f"arrival_date={normalizar_fecha(arrival_date):%Y-%m-%d}"


def escribir_atomico(destino: Path, escribir: Callable[[Path], None]) -> Path:
    """Materializa `destino` sin que exista nunca a medias.

    `escribir` recibe una ruta temporal; si termina sin excepción, el temporal
    se renombra al destino. `Path.replace` es atómico dentro del mismo
    volumen, así que un fallo a mitad de escritura deja el archivo anterior
    intacto en lugar de uno truncado que la siguiente tarea leería como
    válido.

    Es la mecánica de la INVARIANTE 3, y está en un solo sitio para que
    ninguna etapa del pipeline tenga que recordarla. El temporal lleva punto
    inicial para que ni el descubridor de particiones ni DVC lo confundan con
    una salida.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(f".{destino.name}.tmp")
    try:
        escribir(tmp)
        tmp.replace(destino)
    finally:
        # Tras un renombrado correcto ya no existe: esto sólo limpia el
        # residuo de un fallo, que si no quedaría confundiendo al siguiente run.
        tmp.unlink(missing_ok=True)
    return destino


def escribir_parquet(df: pd.DataFrame, destino: Path) -> Path:
    """Parquet atómico. Es el formato de intercambio interno porque preserva
    los tipos; en CSV las fechas volverían como texto."""
    return escribir_atomico(
        destino,
        lambda ruta: df.to_parquet(ruta, index=False, engine="pyarrow", compression="snappy"),
    )


#: Fin de línea de **todas** las salidas de texto del producto, en cualquier
#: sistema operativo.
#:
#: Hay que decirlo explícitamente porque ni `Path.write_text` ni
#: `DataFrame.to_csv` lo hacen: los dos abren el archivo en modo texto y
#: traducen `\n` al fin de línea de la plataforma, de modo que el mismo
#: pipeline producía CRLF en Windows y LF en Linux.
#:
#: No es cosmético por dos razones concretas. **DVC compara bytes**: un CSV
#: de pedido con 1.783 líneas cambia de hash entre plataformas sin que cambie
#: un dato, y `dvc repro` en otro sistema da la etapa por obsoleta. Y los JSON
#: de `reportes/` son salidas `cache: false` **rastreadas en Git**, así que
#: cada `dvc repro` en Windows los dejaba marcados como modificados aunque
#: `git diff` saliera vacío, con el árbol de trabajo permanentemente sucio.
FIN_DE_LINEA = "\n"


def escribir_csv(df: pd.DataFrame, destino: Path) -> Path:
    """CSV atómico, sólo para lo que va a leer una persona.

    Dentro del producto se usa Parquet. El CSV existe porque el gerente abre
    el pedido en una hoja de cálculo, y una lectura humana no necesita tipos.
    """
    return escribir_atomico(
        destino,
        lambda ruta: df.to_csv(
            ruta, index=False, encoding="utf-8", lineterminator=FIN_DE_LINEA
        ),
    )


def escribir_json(contenido: dict, destino: Path) -> Path:
    """JSON atómico, con las claves ordenadas y fin de línea fijo.

    Ninguna de las dos cosas es estética: **DVC compara la salida como
    contenido**, así que un orden de claves que cambie entre ejecuciones —o un
    fin de línea que cambie entre sistemas operativos— haría aparecer
    diferencias donde no las hay.
    """
    texto = json.dumps(contenido, indent=2, ensure_ascii=False, sort_keys=True, default=str)
    return escribir_atomico(
        destino,
        lambda ruta: ruta.write_text(texto + "\n", encoding="utf-8", newline=FIN_DE_LINEA),
    )


def escribir_particion(df: pd.DataFrame, destino: Path) -> Path:
    """Escribe una partición completa de forma atómica.

    La partición se REESCRIBE completa. No existe modo de agregado: ejecutar
    la tarea dos veces debe dar el mismo resultado que ejecutarla una
    (INVARIANTE 3), y un `append` duplicaría las filas en el segundo intento.
    """
    destino.mkdir(parents=True, exist_ok=True)
    return escribir_parquet(df, destino / ARCHIVO_PARTICION)


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
