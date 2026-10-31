"""Registro de modelos: guardar un modelo entrenado y volver a usarlo.

Separar el entrenamiento de la predicción no es un capricho de diseño: el
modelo se ajusta una vez por semana y se consulta cada día, y el pronóstico
tiene que poder emitirse aunque el reentrenamiento falle. Para eso el modelo
tiene que ser un artefacto con identidad propia.

**Cada artefacto lleva su ficha.** El `.pkl` no se puede interpretar solo: no
dice con qué datos se ajustó ni por qué se eligió. El `metadata.json` de al
lado responde eso, y es texto plano para que se pueda leer sin Python —
incluso dentro de un contenedor de Airflow al que sólo se entra a mirar logs.

**Por qué pickle.** El artefacto lo produce y lo consume el mismo pipeline, en
el mismo entorno fijado por `uv.lock`. La alternativa era un formato por
modelo (el texto nativo de LightGBM, un JSON de coeficientes para el lineal),
que es más portable pero obliga a escribir y mantener serialización a medida
para cada clase y rompe el que los tres modelos sean intercambiables. Se
asume la contrapartida y se acota: **sólo se cargan artefactos producidos por
este pipeline**, nunca uno recibido de fuera, porque deserializar un pickle
ajeno es ejecutar código ajeno.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from reposicion.io import escribir_atomico, escribir_json, normalizar_fecha

ARCHIVO_MODELO = "modelo.pkl"
ARCHIVO_FICHA = "metadata.json"

#: Prefijo de las carpetas de versión (`models/asof=2017-08-14`).
ETIQUETA_VERSION = "asof="


def guardar(modelo: Any, destino: Path, ficha: dict) -> Path:
    """Materializa el modelo y su ficha en `destino`, de forma atómica.

    El orden importa: primero el modelo, después la ficha. Si el proceso muere
    entre ambos, queda un modelo sin ficha —detectable— en vez de una ficha
    que promete un modelo que no existe.
    """
    destino.mkdir(parents=True, exist_ok=True)
    ruta = escribir_atomico(
        destino / ARCHIVO_MODELO,
        lambda tmp: tmp.write_bytes(pickle.dumps(modelo, protocol=5)),
    )
    escribir_json({**ficha, "clase": type(modelo).__name__}, destino / ARCHIVO_FICHA)
    return ruta


def cargar(destino: Path) -> Any:
    """Recupera el modelo guardado en `destino`."""
    ruta = destino / ARCHIVO_MODELO
    if not ruta.exists():
        raise FileNotFoundError(
            f"No hay modelo en {destino}. Ejecuta la etapa `modelo` del pipeline antes "
            "de pronosticar."
        )
    return pickle.loads(ruta.read_bytes())


def versiones(base: Path) -> list[str]:
    """Fechas de las versiones completas registradas en `base`, en orden.

    Se exige que exista el `.pkl`: una carpeta a medio escribir —un
    entrenamiento que murió, o una que DVC acaba de crear vacía— no es una
    versión y no debe poder quedar operando.
    """
    if not base.exists():
        return []
    return sorted(
        carpeta.name.removeprefix(ETIQUETA_VERSION)
        for carpeta in base.iterdir()
        if carpeta.is_dir()
        and carpeta.name.startswith(ETIQUETA_VERSION)
        and (carpeta / ARCHIVO_MODELO).exists()
    )


def version_vigente(base: Path, asof: str | pd.Timestamp) -> Path:
    """Carpeta del modelo que opera en `asof`: el más reciente con fecha ≤ `asof`.

    Esta función existe porque **la versión del modelo y la fecha de la
    emisión son dos relojes distintos**. El pronóstico sale cada lunes y el
    reentrenamiento corre una vez al mes: la mayoría de los lunes no hay
    ningún `models/asof=<ese lunes>`, y exigirlo dejaría al producto sin poder
    emitir seis semanas de cada siete. Lo que opera es el último modelo
    registrado, igual que en cualquier sistema real.

    **El filtro punto-en-el-tiempo también se aplica al registro de modelos**
    (INVARIANTE 1). Un backfill de julio que cargara el modelo entrenado en
    agosto estaría usando información del futuro para decidir el pedido de
    julio, y el resultado del backfill sería una mentira cómoda: mejor de lo
    que el producto habría hecho de verdad ese día. Por eso se descartan las
    versiones posteriores a `asof` en vez de tomar simplemente la última.

    No hay dos caminos de código según quién llame. Cuando el pipeline de DVC
    acaba de entrenar para `asof`, esa versión *es* la más reciente ≤ `asof` y
    la regla devuelve exactamente la carpeta que DVC declaró como salida; en
    operación devuelve la del último reentrenamiento. Una sola regla cubre las
    dos situaciones, y por eso DVC y Airflow siguen ejecutando lo mismo.
    """
    dia = f"{normalizar_fecha(asof):%Y-%m-%d}"
    # Las fechas ISO ordenan igual como texto que como fecha, así que la
    # comparación no necesita volver a convertirlas.
    anteriores = [version for version in versiones(base) if version <= dia]
    if not anteriores:
        posteriores = versiones(base)
        detalle = (
            f"Las unicas versiones registradas son posteriores: {posteriores}. "
            "Usar una seria mirar al futuro."
            if posteriores
            else "No hay ninguna version registrada."
        )
        raise FileNotFoundError(
            f"No hay modelo vigente en {dia} dentro de {base}. {detalle} "
            "Ejecuta la etapa `modelo` (o el DAG de reentrenamiento) con una "
            "fecha logica anterior o igual."
        )
    return base / f"{ETIQUETA_VERSION}{anteriores[-1]}"


def leer_ficha(destino: Path) -> dict:
    """Ficha del modelo guardado, sin deserializarlo.

    Permite responder "¿qué modelo está vigente y desde cuándo?" sin cargar
    LightGBM ni el artefacto, que es lo que necesita el seguimiento.
    """
    ruta = destino / ARCHIVO_FICHA
    if not ruta.exists():
        raise FileNotFoundError(f"No hay ficha de modelo en {destino}.")
    return json.loads(ruta.read_text(encoding="utf-8"))
