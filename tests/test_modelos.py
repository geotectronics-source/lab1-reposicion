"""Pruebas de los modelos y de la validación temporal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reposicion.models.base import ordenar_cuantiles, predictoras
from reposicion.models.baseline import BaselineEstacional
from reposicion.models.lgbm import LightGBMCuantilico
from reposicion.validacion import (
    Pliegue,
    decidir_promocion,
    partir,
    pliegues_moviles,
    resumir,
    validar,
)

CUANTILES = [0.5, 0.9]

CFG_PROMOCION = {
    "promocion": {
        "metrica": "costo_negocio",
        "mejora_minima": 0.03,
        "en_todos_los_pliegues": True,
    }
}


def _panel(n: int = 400, semilla: int = 0) -> pd.DataFrame:
    """Panel sintético con una señal aprendible y ruido."""
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2017-01-01", periods=n, freq="D")
    nivel = 50 + 20 * np.sin(np.arange(n) / 7 * 2 * np.pi)
    return pd.DataFrame(
        {
            "date": fechas,
            "origen": fechas - pd.Timedelta(days=1),
            "store_nbr": 1,
            "family": "GROCERY I",
            "h": rng.integers(1, 8, n),
            "dia_semana": fechas.day_of_week,
            "media_7": nivel,
            "emp_q0.5": nivel,
            "emp_q0.9": nivel * 1.3,
            "y": np.maximum(0, nivel + rng.normal(0, 5, n)),
        }
    )


# ---------------------------------------------------------------------------
# Contrato común
# ---------------------------------------------------------------------------


def test_las_fechas_no_entran_como_variables():
    """Dejarlas pasar permitiría memorizar periodos concretos en vez de
    aprender el patrón que los explica."""
    columnas = predictoras(_panel())
    assert "date" not in columnas
    assert "origen" not in columnas
    assert "y" not in columnas


def test_los_cuantiles_no_pueden_cruzarse():
    """Cada cuantil se estima por separado, así que nada garantiza el orden.

    Un cruce haría que subir el costo de quiebre produjera un pedido MENOR, y
    la política dejaría de ser interpretable.
    """
    df = pd.DataFrame({"q0.5": [100.0, 10.0], "q0.9": [80.0, 50.0]})  # la primera cruza
    ordenado = ordenar_cuantiles(df, [0.5, 0.9])

    assert (ordenado["q0.9"] >= ordenado["q0.5"]).all()
    assert ordenado.loc[1, "q0.9"] == 50.0  # la coherente no se toca


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


def test_el_baseline_usa_los_cuantiles_empiricos():
    panel = _panel()
    pred = BaselineEstacional(CUANTILES).entrenar(panel).predecir(panel)

    assert (pred["q0.5"] == panel["emp_q0.5"]).all()


def test_el_baseline_respeta_el_orden_de_los_cuantiles():
    pred = BaselineEstacional(CUANTILES).entrenar(_panel()).predecir(_panel())
    assert (pred["q0.9"] >= pred["q0.5"]).all()


def test_el_baseline_cae_al_nivel_reciente_sin_historia():
    """Una tienda recién abierta no tiene cuantiles de su día de la semana."""
    panel = _panel(50)
    panel["emp_q0.5"] = np.nan
    panel["emp_q0.9"] = np.nan

    pred = BaselineEstacional(CUANTILES).entrenar(panel).predecir(panel)
    assert (pred["q0.5"] == panel["media_7"]).all()


def test_el_baseline_nunca_predice_negativo():
    panel = _panel(50)
    panel["emp_q0.5"] = -10.0
    pred = BaselineEstacional(CUANTILES).entrenar(panel).predecir(panel)
    assert (pred["q0.5"] >= 0).all()


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_el_modelo_aprende_la_senal():
    panel = _panel(600)
    entrena, evalua = panel.iloc[:500], panel.iloc[500:]

    modelo = LightGBMCuantilico(CUANTILES, params={"n_estimators": 60}).entrenar(entrena)
    pred = modelo.predecir(evalua)

    # Con una señal tan marcada, el error debe quedar claramente por debajo de
    # la desviación de la propia serie.
    assert np.abs(pred["q0.5"] - evalua["y"]).mean() < evalua["y"].std()


@pytest.mark.slow
def test_el_modelo_respeta_el_orden_de_los_cuantiles():
    panel = _panel(400)
    modelo = LightGBMCuantilico(CUANTILES, params={"n_estimators": 40}).entrenar(panel)
    pred = modelo.predecir(panel)
    assert (pred["q0.9"] >= pred["q0.5"]).all()


@pytest.mark.slow
def test_el_modelo_nunca_predice_negativo():
    """`expm1` puede devolver valores negativos y la política los leería como
    pedir de menos."""
    panel = _panel(300)
    modelo = LightGBMCuantilico(CUANTILES, params={"n_estimators": 40}).entrenar(panel)
    assert (modelo.predecir(panel)["q0.5"] >= 0).all()


def test_predecir_sin_entrenar_falla_claramente():
    with pytest.raises(RuntimeError, match="no ha sido entrenado"):
        LightGBMCuantilico(CUANTILES).predecir(_panel())


def test_entrenar_sin_objetivo_falla_claramente():
    with pytest.raises(ValueError, match="columna objetivo"):
        LightGBMCuantilico(CUANTILES).entrenar(_panel().drop(columns="y"))


@pytest.mark.slow
def test_predecir_sin_las_variables_de_entrenamiento_falla():
    """LightGBM asocia columnas por posición: un panel distinto daría
    predicciones silenciosamente erróneas."""
    panel = _panel(200)
    modelo = LightGBMCuantilico(CUANTILES, params={"n_estimators": 20}).entrenar(panel)

    with pytest.raises(ValueError, match="Faltan variables"):
        modelo.predecir(panel.drop(columns="media_7"))


# ---------------------------------------------------------------------------
# Validación temporal
# ---------------------------------------------------------------------------


def test_los_pliegues_avanzan_en_el_tiempo():
    pliegues = pliegues_moviles(pd.Timestamp("2017-08-13"), 3, 28)

    assert len(pliegues) == 3
    assert pliegues[-1].evalua_hasta == pd.Timestamp("2017-08-13")
    cortes = [p.entrena_hasta for p in pliegues]
    assert cortes == sorted(cortes) and len(set(cortes)) == 3


def test_cada_pliegue_entrena_solo_con_el_pasado():
    """La prueba que distingue una validación temporal de un KFold."""
    panel = _panel(400)
    pliegue = Pliegue(1, pd.Timestamp("2017-06-01"), pd.Timestamp("2017-06-29"))
    entrena, evalua = partir(panel, pliegue)

    assert entrena["date"].max() <= pliegue.entrena_hasta
    assert evalua["date"].min() > pliegue.entrena_hasta
    assert not set(entrena.index) & set(evalua.index)


def test_el_corte_usa_la_fecha_objetivo_no_la_del_origen():
    """Un ejemplo cuyo objetivo cae tras el corte pertenece al futuro, por
    antiguo que sea su origen."""
    panel = _panel(400)
    pliegue = Pliegue(1, pd.Timestamp("2017-06-01"), pd.Timestamp("2017-06-29"))
    _, evalua = partir(panel, pliegue)
    assert (evalua["date"] > pliegue.entrena_hasta).all()


def test_todos_los_modelos_ven_los_mismos_pliegues():
    """Comparar modelos evaluados por separado lleva a conclusiones falsas."""
    panel = _panel(300)
    pliegues = pliegues_moviles(panel["date"].max(), 2, 28)
    resultados = validar([BaselineEstacional(CUANTILES)], panel, pliegues, CUANTILES)

    assert sorted(resultados["pliegue"].unique()) == [1, 2]


# ---------------------------------------------------------------------------
# Criterio de promoción
# ---------------------------------------------------------------------------


def _resultados(costos_vigente: list[float], costos_retador: list[float]) -> pd.DataFrame:
    filas = []
    for i, (v, r) in enumerate(zip(costos_vigente, costos_retador, strict=True), 1):
        filas.append({"modelo": "vigente", "pliegue": i, "costo_negocio": v})
        filas.append({"modelo": "retador", "pliegue": i, "costo_negocio": r})
    return pd.DataFrame(filas)


def test_promueve_cuando_mejora_en_todos_los_pliegues():
    decision = decidir_promocion(
        _resultados([100, 100, 100], [90, 85, 92]), "vigente", "retador", config=CFG_PROMOCION
    )
    assert decision.promover


def test_no_promueve_si_falla_en_un_solo_pliegue():
    """Un modelo que mejora mucho en uno y empeora en otro es inestable, y el
    promedio lo esconde: en producción se vería como semanas buenas y malas
    sin explicación."""
    decision = decidir_promocion(
        _resultados([100, 100, 100], [50, 50, 105]), "vigente", "retador", config=CFG_PROMOCION
    )
    assert not decision.promover
    assert "no alcanza el umbral" in decision.razon


def test_no_promueve_por_una_mejora_marginal():
    """Por debajo del umbral, el cambio no compensa el riesgo de tocar un
    modelo que ya opera bien."""
    decision = decidir_promocion(
        _resultados([100, 100], [99, 99]), "vigente", "retador", config=CFG_PROMOCION
    )
    assert not decision.promover


def test_la_decision_explica_cada_pliegue():
    """El criterio debe ser auditable, no un veredicto sin justificación."""
    decision = decidir_promocion(
        _resultados([100, 100], [90, 80]), "vigente", "retador", config=CFG_PROMOCION
    )
    assert decision.mejoras == {1: pytest.approx(0.10), 2: pytest.approx(0.20)}
    assert "p1: +10.0%" in str(decision)


def test_faltar_un_modelo_falla_claramente():
    with pytest.raises(KeyError, match="Faltan resultados"):
        decidir_promocion(
            _resultados([100], [90]), "vigente", "inexistente", config=CFG_PROMOCION
        )


def test_resumir_promedia_sobre_los_pliegues():
    resumen = resumir(_resultados([100, 200], [90, 110]))
    costos = resumen.set_index("modelo")["costo_negocio"]
    assert costos["vigente"] == pytest.approx(150.0)
    assert costos["retador"] == pytest.approx(100.0)
