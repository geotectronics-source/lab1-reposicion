"""Pruebas de la política de reposición.

La prueba central es `test_subir_el_costo_de_quiebre_aumenta_el_pedido`: el
enunciado exige que un cambio en los costos modifique la recomendación sin
modificar el código, y eso debe verificarse, no afirmarse.
"""

from __future__ import annotations

import pandas as pd
import pytest

from reposicion.policy import (
    calcular_pedido,
    cuantil_disponible,
    cuantil_objetivo,
    explicar,
    resumen_por_serie,
)

CUANTILES = [0.5, 0.8, 0.9, 0.95]

COSTOS = {
    "defaults": {"costo_quiebre": 3.0, "costo_merma": 1.0},  # q* = 0,75
    "familias": {
        "PRODUCE": {"costo_merma": 4.0},  # q* = 0,43 -> perecedero
        "GROCERY I": {"costo_merma": 0.5},  # q* = 0,86 -> no perece
    },
    "politica": {"tope_multiplo_media": 5.0},
}

MODELO = {"pronostico": {"cuantiles": CUANTILES}}


def _pronostico(familias=("PRODUCE", "GROCERY I"), dias: int = 3) -> pd.DataFrame:
    filas = []
    for familia in familias:
        for dia in range(dias):
            filas.append(
                {
                    "store_nbr": 1,
                    "family": familia,
                    "date": pd.Timestamp("2017-07-17") + pd.Timedelta(days=dia),
                    "media_28": 100.0,
                    "q0.5": 100.0,
                    "q0.8": 130.0,
                    "q0.9": 150.0,
                    "q0.95": 180.0,
                }
            )
    return pd.DataFrame(filas)


def _pedido(costos=None, pronostico=None) -> pd.DataFrame:
    return calcular_pedido(
        pronostico if pronostico is not None else _pronostico(),
        config_costos=costos or COSTOS,
        config_modelo=MODELO,
    )


# ---------------------------------------------------------------------------
# LA prueba: la configuración manda, no el código
# ---------------------------------------------------------------------------


def test_subir_el_costo_de_quiebre_aumenta_el_pedido():
    """El requisito central del producto, verificado.

    Cambiar un número en la configuración debe cambiar la recomendación sin
    tocar una línea de Python ni reentrenar el modelo.
    """
    barato = {**COSTOS, "defaults": {"costo_quiebre": 1.0, "costo_merma": 1.0}}
    caro = {**COSTOS, "defaults": {"costo_quiebre": 9.0, "costo_merma": 1.0}}
    sin_ajustes = {"familias": {}, "politica": {}}

    pedido_barato = _pedido({**barato, **sin_ajustes})
    pedido_caro = _pedido({**caro, **sin_ajustes})

    assert pedido_caro["pedido_sugerido"].sum() > pedido_barato["pedido_sugerido"].sum()
    assert pedido_barato["cuantil_objetivo"].iloc[0] == pytest.approx(0.50)
    assert pedido_caro["cuantil_objetivo"].iloc[0] == pytest.approx(0.90)


def test_el_pedido_es_monotono_en_el_costo_de_quiebre():
    """Más caro quebrar nunca puede producir un pedido menor.

    Si esto fallara, la política sería ininterpretable: subir la penalización
    por faltante daría menos inventario.
    """
    totales = []
    for cu in (1.0, 2.0, 3.0, 5.0, 9.0, 20.0):
        costos = {
            "defaults": {"costo_quiebre": cu, "costo_merma": 1.0},
            "familias": {},
            "politica": {},
        }
        totales.append(_pedido(costos)["pedido_sugerido"].sum())

    assert totales == sorted(totales)


def test_un_perecedero_pide_menos_que_un_abarrote():
    """La diferencia no está en el margen sino en qué pasa con lo que sobra:
    la fruta se tira, el arroz se vende la semana siguiente."""
    pedido = _pedido()
    por_familia = pedido.groupby("family")["pedido_sugerido"].sum()

    assert por_familia["PRODUCE"] < por_familia["GROCERY I"]


def test_cada_familia_usa_su_propio_cuantil():
    pedido = _pedido()
    objetivos = pedido.groupby("family")["cuantil_objetivo"].first()

    assert objetivos["PRODUCE"] == pytest.approx(3.0 / 7.0)  # 3 / (3 + 4)
    assert objetivos["GROCERY I"] == pytest.approx(3.0 / 3.5)  # 3 / (3 + 0,5)


# ---------------------------------------------------------------------------
# Elección del cuantil
# ---------------------------------------------------------------------------


def test_se_elige_el_cuantil_estimado_mas_cercano():
    assert cuantil_disponible(0.86, CUANTILES) == 0.9
    assert cuantil_disponible(0.43, CUANTILES) == 0.5
    assert cuantil_disponible(0.95, CUANTILES) == 0.95


def test_ante_un_empate_se_elige_el_cuantil_mas_alto():
    """Con la misma distancia, cubrirse de más cuesta merma y cubrirse de
    menos cuesta ventas perdidas y clientes: se prefiere el lado conservador."""
    assert cuantil_disponible(0.85, [0.8, 0.9]) == 0.9


def test_el_cuantil_objetivo_sale_de_los_costos():
    assert cuantil_objetivo("PRODUCE", COSTOS) == pytest.approx(3.0 / 7.0)
    assert cuantil_objetivo("FAMILIA NUEVA", COSTOS) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Tope de seguridad
# ---------------------------------------------------------------------------


def test_el_tope_recorta_un_pronostico_desbocado():
    """Red de seguridad ante un dato anómalo que superó la validación."""
    desbocado = _pronostico(familias=("GROCERY I",), dias=1)
    desbocado.loc[:, ["q0.9", "q0.95"]] = 99_999.0
    desbocado.loc[:, "media_28"] = 10.0  # el tope queda en 50

    pedido = _pedido(pronostico=desbocado)

    assert pedido["tope_aplicado"].all()
    assert pedido["pedido_sugerido"].iloc[0] == pytest.approx(50.0)


def test_el_recorte_queda_registrado():
    """Recortar en silencio escondería que el modelo se queda corto con una
    serie semana tras semana."""
    normal = _pedido()
    assert not normal["tope_aplicado"].any()
    assert "tope_aplicado" in normal.columns


def test_una_serie_casi_sin_venta_puede_reponerse():
    """Con media cero el tope sería cero y la serie no se repondría nunca."""
    minima = _pronostico(familias=("GROCERY I",), dias=1)
    minima.loc[:, "media_28"] = 0.0

    pedido = _pedido(pronostico=minima)
    assert pedido["pedido_sugerido"].iloc[0] >= 1.0


def test_sin_tope_configurado_no_se_recorta():
    sin_tope = {**COSTOS, "politica": {}}
    desbocado = _pronostico(familias=("GROCERY I",), dias=1)
    desbocado.loc[:, ["q0.9", "q0.95"]] = 99_999.0

    pedido = _pedido(sin_tope, desbocado)
    assert not pedido["tope_aplicado"].any()


# ---------------------------------------------------------------------------
# Trazabilidad
# ---------------------------------------------------------------------------


def test_el_pedido_explica_de_donde_sale():
    """Un pedido que nadie puede explicar no se sigue, por bueno que sea."""
    pedido = _pedido()
    assert {
        "cuantil_objetivo",
        "cuantil_usado",
        "pedido_sugerido",
        "tope_aplicado",
    } <= set(pedido.columns)


def test_el_pedido_nunca_es_negativo():
    negativo = _pronostico(familias=("GROCERY I",), dias=1)
    negativo.loc[:, ["q0.5", "q0.8", "q0.9", "q0.95"]] = -5.0

    assert (_pedido(pronostico=negativo)["pedido_sugerido"] >= 0).all()


def test_el_pronostico_original_se_conserva():
    """La política añade columnas, no reemplaza el pronóstico: hace falta
    para el seguimiento del desempeño."""
    pedido = _pedido()
    assert {"q0.5", "q0.9"} <= set(pedido.columns)


# ---------------------------------------------------------------------------
# Vistas agregadas
# ---------------------------------------------------------------------------


def test_el_resumen_suma_el_horizonte_por_serie():
    resumen = resumen_por_serie(_pedido())

    assert len(resumen) == 2  # dos familias, una tienda
    assert (resumen["dias"] == 3).all()


def test_el_resumen_muestra_el_colchon_sobre_el_escenario_central():
    """Cuánto se pide por encima de la mediana es la apuesta de la política,
    y conviene que sea explícita."""
    resumen = resumen_por_serie(_pedido())
    grocery = resumen[resumen["family"] == "GROCERY I"].iloc[0]

    assert grocery["colchon"] == pytest.approx(
        grocery["pedido_semana"] - grocery["pronostico_central"]
    )
    assert grocery["colchon"] > 0  # q* = 0,86 pide por encima de la mediana


def test_explicar_responde_por_que_una_familia_pide_menos():
    tabla = explicar(_pedido(), COSTOS)

    assert list(tabla["family"]) == ["PRODUCE", "GROCERY I"]  # ordenado por cuantil
    assert set(tabla.columns) >= {"costo_quiebre", "costo_merma", "cuantil_objetivo"}
