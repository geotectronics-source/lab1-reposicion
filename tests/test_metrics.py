"""Pruebas de las métricas y de los costos de negocio."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reposicion.metrics import (
    cobertura,
    costo_negocio,
    costo_negocio_por_familia,
    costos_de_familia,
    cuantil_critico,
    evaluar,
    mae,
    pinball,
    rmsle,
)

COSTOS = {
    "defaults": {"costo_quiebre": 3.0, "costo_merma": 1.0},
    "familias": {"PRODUCE": {"costo_merma": 4.0}},
}


# ---------------------------------------------------------------------------
# La identidad que une pronóstico y negocio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cu,co", [(3.0, 1.0), (1.0, 1.0), (4.0, 1.5), (1.0, 5.0)])
def test_el_costo_de_negocio_es_la_perdida_pinball_escalada(cu: float, co: float):
    """costo = (cu + co) * L_q*, con q* = cu / (cu + co).

    No son dos objetivos que haya que equilibrar: son el mismo objetivo en
    distinta escala. Es la razón de que entrenar con pérdida pinball en el
    cuantil crítico minimice directamente el dinero perdido, y por eso la
    identidad merece una prueba y no sólo un comentario.
    """
    rng = np.random.default_rng(0)
    real = rng.gamma(2.0, 50.0, size=500)
    pedido = rng.gamma(2.0, 50.0, size=500)

    q = cuantil_critico(cu, co)
    esperado = (cu + co) * pinball(real, pedido, q)
    assert costo_negocio(real, pedido, cu, co) == pytest.approx(esperado)


# ---------------------------------------------------------------------------
# Cuantil crítico
# ---------------------------------------------------------------------------


def test_el_cuantil_critico_crece_con_el_costo_de_quebrar():
    """Si quebrar duele más, hay que cubrirse más."""
    assert cuantil_critico(3.0, 1.0) == pytest.approx(0.75)
    assert cuantil_critico(1.0, 1.0) == pytest.approx(0.50)
    assert cuantil_critico(1.0, 4.0) == pytest.approx(0.20)


def test_costos_invalidos_fallan_explicitamente():
    with pytest.raises(ValueError, match="deben sumar más de cero"):
        cuantil_critico(0.0, 0.0)


def test_un_perecedero_pide_con_mas_cautela_que_un_abarrote():
    """La diferencia no está en el margen, sino en qué pasa con lo que sobra."""
    cu_p, co_p = costos_de_familia("PRODUCE", COSTOS)
    cu_g, co_g = costos_de_familia("GROCERY I", COSTOS)

    assert cuantil_critico(cu_p, co_p) < cuantil_critico(cu_g, co_g)


def test_una_familia_sin_ajuste_usa_los_valores_por_defecto():
    assert costos_de_familia("FAMILIA NUEVA", COSTOS) == (3.0, 1.0)


# ---------------------------------------------------------------------------
# Costo de negocio
# ---------------------------------------------------------------------------


def test_acertar_exactamente_no_cuesta_nada():
    assert costo_negocio([10.0, 20.0], [10.0, 20.0], 3.0, 1.0) == 0.0


def test_quedarse_corto_cuesta_el_costo_de_quiebre():
    assert costo_negocio([10.0], [6.0], 3.0, 1.0) == pytest.approx(12.0)  # 4 faltantes x 3


def test_pasarse_cuesta_el_costo_de_merma():
    assert costo_negocio([10.0], [14.0], 3.0, 1.0) == pytest.approx(4.0)  # 4 sobrantes x 1


def test_cada_familia_se_evalua_con_sus_propios_costos():
    """Evaluar todo con los costos por defecto escondería lo que importa."""
    df = pd.DataFrame(
        {
            "family": ["PRODUCE", "GROCERY I"],
            "y": [10.0, 10.0],
            "pedido": [14.0, 14.0],  # ambas se pasan en 4
        }
    )
    total = costo_negocio_por_familia(df, "pedido", COSTOS)
    # PRODUCE: 4 x 4.0 = 16 ; GROCERY I: 4 x 1.0 = 4 ; media = 10
    assert total == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Métricas del pronóstico
# ---------------------------------------------------------------------------


def test_el_pinball_penaliza_de_forma_asimetrica():
    """Para q=0.9, quedarse corto debe costar nueve veces más que pasarse."""
    corto = pinball([100.0], [90.0], 0.9)
    pasado = pinball([100.0], [110.0], 0.9)
    assert corto == pytest.approx(9 * pasado)


def test_el_pinball_es_simetrico_en_la_mediana():
    assert pinball([100.0], [90.0], 0.5) == pytest.approx(pinball([100.0], [110.0], 0.5))


def test_la_cobertura_mide_cuantos_quedan_por_debajo():
    assert cobertura([1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 0.0], 0.9) == pytest.approx(0.75)


def test_el_rmsle_es_relativo_no_absoluto():
    """Fallar por 10 unidades pesa mucho más en una serie pequeña."""
    pequeno = rmsle([20.0], [30.0])
    grande = rmsle([5000.0], [5010.0])
    assert pequeno > grande


def test_el_mae_es_el_error_absoluto_medio():
    assert mae([10.0, 20.0], [12.0, 16.0]) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------


def test_evaluar_devuelve_tecnicas_y_negocio_juntas():
    """Leerlas por separado invita a elegir el modelo por la que conviene."""
    df = pd.DataFrame(
        {
            "family": ["GROCERY I"] * 4,
            "y": [10.0, 20.0, 30.0, 40.0],
            "q0.5": [12.0, 18.0, 28.0, 44.0],
            "q0.9": [15.0, 25.0, 35.0, 50.0],
        }
    )
    resumen = evaluar(df, [0.5, 0.9], config_costos=COSTOS)

    assert {"mae", "rmsle", "pinball_medio", "costo_negocio"} <= set(resumen)
    assert resumen["n"] == 4
    assert resumen["cobertura_0.9"] == pytest.approx(1.0)
