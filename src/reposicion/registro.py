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

from reposicion.io import escribir_atomico, escribir_json

ARCHIVO_MODELO = "modelo.pkl"
ARCHIVO_FICHA = "metadata.json"


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


def leer_ficha(destino: Path) -> dict:
    """Ficha del modelo guardado, sin deserializarlo.

    Permite responder "¿qué modelo está vigente y desde cuándo?" sin cargar
    LightGBM ni el artefacto, que es lo que necesita el seguimiento.
    """
    ruta = destino / ARCHIVO_FICHA
    if not ruta.exists():
        raise FileNotFoundError(f"No hay ficha de modelo en {destino}.")
    return json.loads(ruta.read_text(encoding="utf-8"))
