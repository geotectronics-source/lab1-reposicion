"""Resolución de rutas y carga de configuración.

El mismo código se ejecuta en la máquina del estudiante (Windows), en el CI
(Ubuntu y Windows) y dentro de los contenedores de Airflow (Linux), donde el
proyecto está montado en otra ruta. Por eso nada de esto se escribe fijo:
la raíz sale de la variable de entorno `REPOSICION_HOME` y, si no está, se
deduce de la ubicación del propio paquete.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

VARIABLE_RAIZ = "REPOSICION_HOME"


def raiz_proyecto() -> Path:
    """Devuelve la raíz del repositorio.

    `REPOSICION_HOME` manda: es como Airflow le dice al producto dónde está
    montado el proyecto dentro del contenedor. Sin ella se deduce subiendo
    desde este archivo (src/reposicion/config.py -> raíz), que es lo correcto
    con una instalación editable como la que crea `uv sync`.
    """
    desde_entorno = os.environ.get(VARIABLE_RAIZ)
    if desde_entorno:
        return Path(desde_entorno).resolve()
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Rutas:
    """Ubicaciones del proyecto, todas derivadas de una sola raíz.

    La separación importa y no es decorativa:

    - `landing` es el sistema de origen: SÓLO LECTURA (INVARIANTE 2).
    - `data` es lo que produce el producto y se versiona con DVC.
    - `cuarentena` recibe lo que no cumple el contrato, para poder
      inspeccionarlo sin que contamine ningún cálculo.
    """

    raiz: Path

    @property
    def landing(self) -> Path:
        return self.raiz / "landing"

    @property
    def historico(self) -> Path:
        return self.landing / "historico"

    @property
    def data(self) -> Path:
        return self.raiz / "data"

    @property
    def kaggle(self) -> Path:
        return self.data / "kaggle"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def procesado(self) -> Path:
        return self.data / "procesado"

    @property
    def pronosticos(self) -> Path:
        return self.data / "pronosticos"

    @property
    def pedidos(self) -> Path:
        return self.data / "pedidos"

    @property
    def seguimiento(self) -> Path:
        return self.data / "seguimiento"

    @property
    def cuarentena(self) -> Path:
        return self.data / "cuarentena"

    @property
    def reportes(self) -> Path:
        """Métricas y decisiones de promoción. **Esto SÍ va a Git.**

        Es la excepción deliberada a "nada de salidas en el repositorio", y no
        contradice el requerimiento: no son datos ni modelos, son unos cuantos
        kilobytes de JSON que dicen qué modelo se promovió y por qué. Tenerlos
        versionados hace que el historial de Git conteste "¿con qué métricas
        se tomó esa decisión?" sin necesidad de descargar la caché de DVC, y
        hace que el cambio de métricas aparezca en el diff de un Pull Request,
        que es donde se revisa.
        """
        return self.raiz / "reportes"

    @property
    def modelos(self) -> Path:
        return self.raiz / "models"

    @property
    def config(self) -> Path:
        return self.raiz / "config"


def rutas() -> Rutas:
    """Rutas del proyecto. No se cachea: `REPOSICION_HOME` puede cambiar entre
    tareas de Airflow y las pruebas la reescriben con `monkeypatch`."""
    return Rutas(raiz=raiz_proyecto())


@cache
def _leer_yaml(ruta: str) -> dict:
    with Path(ruta).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def cargar_config(nombre: str) -> dict:
    """Carga `config/<nombre>.yaml`.

    Toda decisión de negocio (costos, horizonte, cuantiles, umbrales) vive en
    estos archivos y no en el código, para que cambiarla no requiera tocar
    Python ni reconstruir la imagen de Airflow.
    """
    ruta = rutas().config / f"{nombre}.yaml"
    if not ruta.exists():
        disponibles = sorted(p.stem for p in rutas().config.glob("*.yaml"))
        raise FileNotFoundError(
            f"No existe la configuración '{nombre}' en {rutas().config}. "
            f"Disponibles: {disponibles or 'ninguna'}."
        )
    # El cacheo usa la ruta como texto porque Path no siempre es hashable
    # de forma estable entre sistemas operativos.
    return _leer_yaml(str(ruta))


def limpiar_cache_config() -> None:
    """Olvida los YAML ya leídos. Lo usan las pruebas tras cambiar la raíz."""
    _leer_yaml.cache_clear()
