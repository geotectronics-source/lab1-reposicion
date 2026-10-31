"""Pruebas de la emisión del pronóstico.

Lo único que añade esta capa sobre los modelos es el reparto: decidir qué
series puede atender el modelo promovido y cuáles van al respaldo. Es poca
lógica y muy cara de equivocar, porque un fallo aquí no lanza error: deja
series sin pronóstico, y lo que no se pronostica no se pide.
"""

from __future__ import annotations

import pandas as pd
import pytest

from reposicion.pronostico import cobertura_del_respaldo, emitir

CUANTILES = [0.5, 0.9]

CFG = {
    "pronostico": {"cuantiles": CUANTILES},
    "entrenamiento": {"minimo_dias_historia": 56},
}


class ModeloFalso:
    """Devuelve una constante. Basta: lo que se prueba es el reparto."""

    nombre = "modelo_falso"

    def predecir(self, panel: pd.DataFrame) -> pd.DataFrame:
        salida = panel.copy()
        for q in CUANTILES:
            salida[f"q{q}"] = 999.0
        return salida


def _panel(*historias: int) -> pd.DataFrame:
    """Una serie por valor de `dias_historia`, con tres días de horizonte."""
    filas = [
        {
            "store_nbr": indice + 1,
            "family": "GROCERY I",
            "date": pd.Timestamp("2017-07-17") + pd.Timedelta(days=dia),
            "dias_historia": dias,
            "media_7": 10.0,
            "emp_q0.5": 10.0,
            "emp_q0.9": 15.0,
        }
        for indice, dias in enumerate(historias)
        for dia in range(3)
    ]
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# LA prueba: ninguna serie se queda sin pronóstico
# ---------------------------------------------------------------------------


def test_una_serie_sin_historia_la_atiende_el_respaldo():
    """INVARIANTE 5: una tienda nueva es negocio, no un dato inválido."""
    emitido = emitir(_panel(200, 10), ModeloFalso(), "2017-07-17", config=CFG)

    por_serie = emitido.groupby("store_nbr")["modelo"].first()
    assert por_serie[1] == "modelo_falso"
    assert por_serie[2] == "baseline_estacional"
    assert len(emitido) == 6, "alguna serie se quedó sin pronóstico"


def test_el_umbral_sale_de_la_configuracion():
    """Es una decisión de negocio, no una constante del código: bajarlo debe
    hacer que el modelo atienda series que antes iban al respaldo."""
    permisivo = {**CFG, "entrenamiento": {"minimo_dias_historia": 5}}
    emitido = emitir(_panel(10), ModeloFalso(), "2017-07-17", config=permisivo)

    assert emitido["modelo"].unique().tolist() == ["modelo_falso"]


def test_sin_la_columna_de_historia_atiende_todo_el_modelo():
    """Un panel de prueba puede no traerla; no es motivo para no pronosticar."""
    panel = _panel(200).drop(columns="dias_historia")
    emitido = emitir(panel, ModeloFalso(), "2017-07-17", config=CFG)

    assert emitido["modelo"].unique().tolist() == ["modelo_falso"]


def test_registra_la_fecha_de_emision():
    """La emisión no es el origen: entre ambos hay latencia, y el seguimiento
    necesita saber cuándo se publicó el pedido."""
    emitido = emitir(_panel(200), ModeloFalso(), "2017-07-17", config=CFG)

    assert emitido["emitida"].unique().tolist() == [pd.Timestamp("2017-07-17")]
    assert emitido["emitida"].dtype == "datetime64[ns]"


def test_un_panel_vacio_no_rompe():
    """Puede pasar legítimamente: una emisión antes de que llegue nada."""
    assert emitir(pd.DataFrame(), ModeloFalso(), "2017-07-17", config=CFG).empty


def test_el_respaldo_se_puede_sustituir():
    """Hace falta para probar y para poder cambiar el modelo de referencia sin
    tocar esta capa."""

    class OtroRespaldo(ModeloFalso):
        nombre = "otro_respaldo"

    emitido = emitir(
        _panel(10), ModeloFalso(), "2017-07-17", respaldo=OtroRespaldo(), config=CFG
    )
    assert emitido["modelo"].unique().tolist() == ["otro_respaldo"]


def test_el_pronostico_sale_ordenado_por_serie_y_fecha():
    """El reparto concatena dos bloques; sin reordenar, la salida alternaría
    entre modelos y sería ilegible para quien la revise."""
    emitido = emitir(_panel(200, 10, 300), ModeloFalso(), "2017-07-17", config=CFG)
    esperado = emitido.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(emitido, esperado)


# ---------------------------------------------------------------------------
# La cifra que hay que vigilar
# ---------------------------------------------------------------------------


def test_la_cobertura_cuenta_series_y_no_filas():
    """Si el respaldo empieza a cubrir una parte creciente del catálogo, el
    problema no es el pronóstico sino que el reentrenamiento dejó de alcanzar
    a las series nuevas. Contar filas lo escondería: el horizonte multiplica."""
    cobertura = cobertura_del_respaldo(
        emitir(_panel(200, 200, 10), ModeloFalso(), "2017-07-17", config=CFG)
    ).set_index("modelo")

    assert cobertura.loc["modelo_falso", "series"] == 2
    assert cobertura.loc["modelo_falso", "filas"] == 6
    assert cobertura.loc["baseline_estacional", "series"] == 1


def test_la_cobertura_de_un_pronostico_vacio_es_una_tabla_vacia():
    vacia = cobertura_del_respaldo(pd.DataFrame())
    assert vacia.empty
    assert list(vacia.columns) == ["modelo", "series", "filas"]


@pytest.mark.parametrize("cuantil", CUANTILES)
def test_todas_las_filas_traen_todos_los_cuantiles(cuantil: float):
    """El respaldo y el modelo tienen que producir las mismas columnas: si no,
    la política encontraría nulos justo en las series nuevas."""
    emitido = emitir(_panel(200, 10), ModeloFalso(), "2017-07-17", config=CFG)
    assert emitido[f"q{cuantil}"].notna().all()
