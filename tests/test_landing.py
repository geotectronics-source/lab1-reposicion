"""Pruebas de la lectura punto-en-el-tiempo.

Es el módulo que sostiene la validez del producto: si la lectura as-of deja
entrar información del futuro, todo lo que venga después hereda la fuga sin
que ninguna otra prueba lo note.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reposicion.landing import (
    ARRIVAL_HISTORICO,
    descubrir_entregas,
    fuentes_en_bandeja,
    historial_de,
    leer_asof,
    leer_historico,
    leer_vigente,
    vista_vigente,
)


def _ventas_tienda44(df: pd.DataFrame) -> float:
    sel = df[(df["date"] == pd.Timestamp("2017-07-05")) & (df["store_nbr"] == 44)]
    return float(sel["sales"].sum())


# ---------------------------------------------------------------------------
# INVARIANTE 1 — sólo lo que había llegado
# ---------------------------------------------------------------------------


def test_no_se_ve_lo_que_aun_no_habia_llegado(bandeja: Path):
    """La prueba central del laboratorio.

    El 15 de julio la corrección todavía no existía, así que el producto debe
    ver los ceros. Verla corregida significaría estar usando el futuro.
    """
    vigente = leer_vigente("sales", "2017-07-15", landing=bandeja)
    assert _ventas_tienda44(vigente) == 0.0


def test_la_correccion_aparece_cuando_ya_habia_llegado(bandeja: Path):
    """El 25 de julio la corrección ya llegó: ahora sí debe regir."""
    vigente = leer_vigente("sales", "2017-07-25", landing=bandeja)
    assert _ventas_tienda44(vigente) == 880.0  # 2 familias x 10.0 x 44


def test_la_misma_funcion_responde_distinto_segun_la_fecha(bandeja: Path):
    """Lo que hace auditable al sistema: el resultado depende de `asof`, no de
    cuándo se ejecute la consulta."""
    antes = _ventas_tienda44(leer_vigente("sales", "2017-07-15", landing=bandeja))
    despues = _ventas_tienda44(leer_vigente("sales", "2017-07-25", landing=bandeja))
    assert antes != despues


def test_el_filtro_es_por_llegada_no_por_fecha_del_dato(bandeja: Path):
    """Un `asof` anterior a la llegada excluye el archivo aunque el dato sea
    de una fecha muy anterior. Es la diferencia entre `date` y `arrival_date`."""
    entregas = descubrir_entregas("sales", "2017-07-10", landing=bandeja)
    llegadas = sorted({e.arrival_date for e in entregas})
    assert llegadas == [pd.Timestamp("2017-07-06"), pd.Timestamp("2017-07-07")]
    # El archivo del 2017-07-20 contiene datos del 2017-07-05 y aun así queda fuera.
    assert all(e.arrival_date <= pd.Timestamp("2017-07-10") for e in entregas)


def test_asof_antes_de_toda_entrega_devuelve_solo_el_historico(bandeja: Path):
    df = leer_asof("sales", "2017-07-01", landing=bandeja)
    assert not df.empty
    assert (df["arrival_date"] == ARRIVAL_HISTORICO).all()
    assert df["date"].max() == pd.Timestamp("2017-06-30")


def test_acepta_timestamp_y_texto_indistintamente(bandeja: Path):
    por_texto = leer_vigente("sales", "2017-07-15", landing=bandeja)
    por_timestamp = leer_vigente("sales", pd.Timestamp("2017-07-15"), landing=bandeja)
    pd.testing.assert_frame_equal(por_texto, por_timestamp)


def test_ignora_la_hora_de_la_fecha_logica(bandeja: Path):
    """Airflow entrega `logical_date` con hora; los datos son diarios."""
    sin_hora = leer_vigente("sales", "2017-07-15", landing=bandeja)
    con_hora = leer_vigente("sales", pd.Timestamp("2017-07-15 23:59:59"), landing=bandeja)
    pd.testing.assert_frame_equal(sin_hora, con_hora)


# ---------------------------------------------------------------------------
# INVARIANTE 4 — el historial se conserva
# ---------------------------------------------------------------------------


def test_la_correccion_no_borra_lo_que_se_supo_antes(bandeja: Path):
    """Ambas versiones siguen en el historial; sólo cambia cuál rige."""
    df = leer_asof("sales", "2017-07-25", landing=bandeja)
    historial = historial_de(df, "sales", date="2017-07-05", store_nbr=44)

    llegadas = historial["arrival_date"].unique()
    assert len(llegadas) == 2, "deben convivir la entrega original y la corrección"

    por_llegada = historial.groupby("arrival_date")["sales"].sum()
    assert por_llegada[pd.Timestamp("2017-07-06")] == 0.0
    assert por_llegada[pd.Timestamp("2017-07-20")] == 880.0


def test_vista_vigente_elige_la_llegada_mas_reciente(bandeja: Path):
    df = leer_asof("sales", "2017-07-25", landing=bandeja)
    vigente = vista_vigente(df, "sales")

    filas = vigente[(vigente["date"] == pd.Timestamp("2017-07-05")) & (vigente["store_nbr"] == 44)]
    assert len(filas) == 2, "una fila por familia, no las dos versiones"
    assert (filas["arrival_date"] == pd.Timestamp("2017-07-20")).all()


def test_vista_vigente_no_duplica_claves(bandeja: Path):
    vigente = leer_vigente("sales", "2017-07-25", landing=bandeja)
    assert not vigente.duplicated(subset=["date", "store_nbr", "family"]).any()


def test_el_historico_siempre_pierde_ante_una_entrega(bandeja: Path):
    """Su fecha de llegada es anterior a cualquier dato real por construcción."""
    assert ARRIVAL_HISTORICO < pd.Timestamp("1970-01-01")


# ---------------------------------------------------------------------------
# Contrato de tipos
# ---------------------------------------------------------------------------


def test_el_tipo_de_date_no_depende_de_si_se_valido(bandeja: Path):
    """Regresión: cuando la conversión la hacía sólo Pandera, leer sin validar
    devolvía `date` como texto. Comparar texto con Timestamp no falla, sólo
    deja de coincidir, y el error aparecía como un resultado vacío."""
    validado = leer_asof("sales", "2017-07-25", landing=bandeja, validar_contrato=True)
    sin_validar = leer_asof("sales", "2017-07-25", landing=bandeja, validar_contrato=False)

    assert validado["date"].dtype == "datetime64[ns]"
    assert sin_validar["date"].dtype == "datetime64[ns]"


def test_siempre_hay_columna_de_llegada(bandeja: Path):
    df = leer_asof("sales", "2017-07-25", landing=bandeja)
    assert "arrival_date" in df.columns
    assert df["arrival_date"].notna().all()


# ---------------------------------------------------------------------------
# INVARIANTE 5 — ausencias que no son errores
# ---------------------------------------------------------------------------


def test_una_fuente_sin_entregas_devuelve_vacio_sin_fallar(bandeja: Path):
    """`transactions` no existe en esta bandeja: eso no es un error."""
    assert leer_asof("transactions", "2017-07-25", landing=bandeja).empty


def test_bandeja_inexistente_no_revienta(tmp_path: Path):
    assert descubrir_entregas("sales", "2017-07-25", landing=tmp_path / "no_existe") == []
    assert fuentes_en_bandeja("2017-07-25", landing=tmp_path / "no_existe") == []


def test_sin_historico_se_lee_solo_la_bandeja(bandeja: Path):
    (bandeja / "historico" / "sales.csv").unlink()
    df = leer_asof("sales", "2017-07-25", landing=bandeja)
    assert not df.empty
    assert (df["arrival_date"] > ARRIVAL_HISTORICO).all()


def test_carpetas_ajenas_al_producto_se_ignoran(bandeja: Path):
    """La bandeja es un sistema externo: puede tener archivos que no son nuestros."""
    (bandeja / "_staging_interno").mkdir()
    (bandeja / "arrival_date=no-es-fecha").mkdir()
    entregas = descubrir_entregas("sales", "2017-07-25", landing=bandeja)
    assert len(entregas) == 3


def test_los_temporales_no_se_leen_a_medias(bandeja: Path):
    """Un `.tmp` del sistema origen escribiendo no debe entrar al cómputo."""
    destino = bandeja / "arrival_date=2017-07-08" / "sales"
    destino.mkdir(parents=True)
    (destino / "sales_2017-07-07.csv.tmp").write_text("basura a medio escribir")

    entregas = descubrir_entregas("sales", "2017-07-25", landing=bandeja)
    assert all(e.ruta.suffix == ".csv" for e in entregas)
    assert len(entregas) == 3


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def test_fuentes_en_bandeja_respeta_el_asof(bandeja: Path):
    assert fuentes_en_bandeja("2017-07-06", landing=bandeja) == ["sales"]
    assert fuentes_en_bandeja("2017-07-01", landing=bandeja) == []


def test_historico_sin_archivo_devuelve_vacio(tmp_path: Path):
    assert leer_historico("sales", landing=tmp_path).empty


def test_vista_vigente_exige_la_clave_completa():
    df = pd.DataFrame(
        {"date": [pd.Timestamp("2017-07-05")], "arrival_date": [pd.Timestamp("2017-07-06")]}
    )
    with pytest.raises(KeyError, match="Faltan columnas de la clave"):
        vista_vigente(df, "sales")


def test_vista_vigente_tolera_vacio():
    assert vista_vigente(pd.DataFrame(), "sales").empty
