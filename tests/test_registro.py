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


# ---------------------------------------------------------------------------
# El modelo vigente: dos relojes distintos
# ---------------------------------------------------------------------------
# La versión del modelo y la fecha de la emisión no coinciden casi nunca: se
# emite cada lunes y se reentrena una vez al mes. Resolver "cuál opera hoy"
# mal tiene dos formas, y las dos son caras: exigir la versión de la fecha
# exacta deja al producto sin emitir seis semanas de cada siete, y tomar
# simplemente la última hace que un backfill de julio use el modelo de agosto
# y se califique con información que ese día no existía.


def _version(tmp_path, fecha: str, nombre: str = "baseline_estacional"):
    """Registra un modelo entrenado bajo `models/asof=<fecha>`."""
    modelo = BaselineEstacional(CUANTILES).entrenar(_panel())
    registro.guardar(modelo, tmp_path / f"asof={fecha}", {"nombre": nombre, "asof": fecha})


def test_vigente_es_la_version_mas_reciente_que_no_mira_al_futuro(tmp_path):
    for fecha in ("2017-06-01", "2017-07-01", "2017-08-01"):
        _version(tmp_path, fecha)

    assert registro.version_vigente(tmp_path, "2017-07-17").name == "asof=2017-07-01"


def test_la_version_de_la_fecha_exacta_gana_cuando_existe(tmp_path):
    """Es el caso del pipeline de DVC, que entrena para la fecha que emite.

    Importa que la misma regla cubra las dos situaciones: si hubiera un camino
    de código para DVC y otro para Airflow, "lo que se reproduce" y "lo que
    opera" podrían separarse sin que nada avisara.
    """
    _version(tmp_path, "2017-07-01")
    _version(tmp_path, "2017-07-17")

    assert registro.version_vigente(tmp_path, "2017-07-17").name == "asof=2017-07-17"


def test_nunca_se_usa_un_modelo_entrenado_despues_de_la_fecha_logica(tmp_path):
    """Un backfill de julio con un modelo de agosto sería hacer trampa.

    El resultado no fallaría: daría un pronóstico mejor del que el producto
    habría podido dar ese día, y el backfill mentiría a su favor.
    """
    _version(tmp_path, "2017-08-01")

    with pytest.raises(FileNotFoundError, match="futuro"):
        registro.version_vigente(tmp_path, "2017-07-17")


def test_una_carpeta_sin_artefacto_no_cuenta_como_version(tmp_path):
    """Un entrenamiento que murió a mitad no puede quedar operando.

    La carpeta existe y lleva la fecha más reciente, así que una búsqueda que
    sólo mire nombres la elegiría y el pronóstico fallaría al cargarla, o peor,
    cargaría la ficha sin el modelo.
    """
    _version(tmp_path, "2017-07-01")
    (tmp_path / "asof=2017-07-10").mkdir()

    assert registro.version_vigente(tmp_path, "2017-07-17").name == "asof=2017-07-01"
    assert registro.versiones(tmp_path) == ["2017-07-01"]


def test_sin_ninguna_version_el_mensaje_dice_que_hacer(tmp_path):
    with pytest.raises(FileNotFoundError, match="etapa `modelo`|DAG de reentrenamiento"):
        registro.version_vigente(tmp_path, "2017-07-17")


def test_las_versiones_salen_en_orden_cronologico(tmp_path):
    """El orden alfabético de las fechas ISO coincide con el cronológico, y de
    eso depende que `version_vigente` pueda comparar como texto."""
    for fecha in ("2017-09-01", "2017-07-01", "2017-12-31", "2017-08-15"):
        _version(tmp_path, fecha)

    assert registro.versiones(tmp_path) == [
        "2017-07-01",
        "2017-08-15",
        "2017-09-01",
        "2017-12-31",
    ]
