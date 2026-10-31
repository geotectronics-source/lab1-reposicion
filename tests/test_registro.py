"""Pruebas del registro de modelos.

Lo que se verifica aquí no es que pickle funcione —eso es de la biblioteca
estándar— sino las dos propiedades que el pipeline necesita del artefacto:
que un modelo recuperado prediga **exactamente** lo mismo que el original, y
que el entrenamiento sea **determinista**, porque si no lo fuera el hash que
DVC guarda en `dvc.lock` describiría una ejecución concreta y no un resultado
reproducible.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from reposicion import registro
from reposicion.models.baseline import BaselineEstacional
from reposicion.models.lgbm import LightGBMCuantilico
from reposicion.models.lineal import RegresionCuantilicaLineal

CUANTILES = [0.5, 0.9]


def _panel(filas: int = 400, semilla: int = 0) -> pd.DataFrame:
    """Panel pequeño con señal real: nivel por tienda y efecto de fin de semana."""
    generador = np.random.default_rng(semilla)
    fechas = pd.date_range("2017-05-01", periods=filas // 4, freq="D")
    marco = pd.DataFrame(
        [
            {
                "store_nbr": tienda,
                "family": familia,
                "date": fecha,
                "h": 1,
                "media_7": 100.0 * tienda,
                "media_28": 100.0 * tienda,
                "dias_historia": 200,
                "emp_q0.5": 100.0 * tienda,
                "emp_q0.9": 130.0 * tienda,
                "dia_semana": fecha.day_of_week,
            }
            for fecha in fechas
            for tienda in (1, 2)
            for familia in ("GROCERY I", "PRODUCE")
        ]
    )
    marco["y"] = marco["media_7"] * np.where(marco["dia_semana"] >= 5, 1.5, 1.0) + generador.normal(
        0, 3, len(marco)
    )
    return marco


@pytest.fixture
def ajustado():
    return LightGBMCuantilico(CUANTILES).entrenar(_panel())


def test_el_modelo_recuperado_predice_lo_mismo(ajustado, tmp_path):
    """La propiedad que hace útil separar entrenamiento de predicción."""
    registro.guardar(ajustado, tmp_path / "m", {"nombre": ajustado.nombre})
    recuperado = registro.cargar(tmp_path / "m")

    panel = _panel(semilla=7).drop(columns="y")
    pd.testing.assert_frame_equal(ajustado.predecir(panel), recuperado.predecir(panel))


def test_la_ficha_queda_legible_sin_python(ajustado, tmp_path):
    """El JSON de al lado responde "¿qué modelo es y con qué se ajustó?" sin
    deserializar nada: es lo único que se puede leer dentro de un contenedor
    al que sólo se entra a mirar logs."""
    registro.guardar(ajustado, tmp_path / "m", {"nombre": ajustado.nombre, "ejemplos": 400})
    ficha = json.loads((tmp_path / "m" / "metadata.json").read_text("utf-8"))

    assert ficha["nombre"] == "lgbm_cuantilico"
    assert ficha["ejemplos"] == 400
    assert ficha["clase"] == "LightGBMCuantilico"  # lo añade el registro, no quien llama


def test_cargar_sin_modelo_dice_que_falta_la_etapa(tmp_path):
    """Un `FileNotFoundError` pelado mandaría a buscar el archivo; el mensaje
    tiene que mandar a ejecutar la etapa que lo produce."""
    with pytest.raises(FileNotFoundError, match="etapa"):
        registro.cargar(tmp_path / "no-existe")


def test_un_fallo_al_guardar_no_deja_archivo_a_medias(tmp_path):
    """Escritura atómica (INVARIANTE 3): un modelo truncado es peor que
    ninguno, porque la etapa siguiente lo cargaría como válido."""

    class NoSerializable:
        def __reduce__(self):
            raise RuntimeError("no se puede serializar")

    with pytest.raises(RuntimeError):
        registro.guardar(NoSerializable(), tmp_path / "m", {})

    assert not (tmp_path / "m" / "modelo.pkl").exists()
    assert not list((tmp_path / "m").glob(".*tmp")), "quedó el temporal de la escritura fallida"


# ---------------------------------------------------------------------------
# Determinismo: lo que hace que el hash de DVC signifique algo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "clase", [BaselineEstacional, LightGBMCuantilico, RegresionCuantilicaLineal]
)
def test_entrenar_dos_veces_da_el_mismo_artefacto(clase, tmp_path):
    """Dos entrenamientos con los mismos datos tienen que dar el mismo archivo.

    Si no lo dieran, `dvc repro --force` sobre un commit antiguo produciría un
    hash distinto al registrado y la reproducibilidad del laboratorio sería
    una afirmación sin respaldo. La semilla fija de cada modelo es lo que
    sostiene esto, y conviene que se rompa en una prueba y no en la defensa.
    """
    panel = _panel()
    huellas = []
    for intento in ("a", "b"):
        destino = tmp_path / intento
        registro.guardar(clase(CUANTILES).entrenar(panel), destino, {})
        huellas.append(hashlib.md5((destino / "modelo.pkl").read_bytes()).hexdigest())

    assert huellas[0] == huellas[1]
