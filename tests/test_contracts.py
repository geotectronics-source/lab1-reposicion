"""Pruebas de los contratos de datos.

Un contrato sirve si rechaza lo que debe rechazar Y deja pasar lo que debe
pasar. Lo segundo importa tanto como lo primero: un contrato demasiado
estricto convierte situaciones normales del negocio en caídas del DAG, que es
justo lo que el INVARIANTE 5 prohíbe.
"""

from __future__ import annotations

import pandas as pd
import pytest

from reposicion.contracts import (
    CLAVES,
    ErrorDeContrato,
    avisos_de_negocio,
    validar,
)


def _sales(**overrides) -> pd.DataFrame:
    base = {
        "date": ["2017-07-05", "2017-07-05"],
        "store_nbr": [1, 44],
        "family": ["GROCERY I", "PRODUCE"],
        "sales": [10.5, 20.0],
        "onpromotion": [0, 3],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# Lo que debe pasar
# ---------------------------------------------------------------------------


def test_acepta_datos_correctos():
    validado = validar(_sales(), "sales")
    assert len(validado) == 2
    assert validado["date"].dtype == "datetime64[ns]"


def test_acepta_ventas_decimales():
    """462k valores del dataset real tienen decimales: productos por peso."""
    validar(_sales(sales=[10.537, 0.001]), "sales")


def test_acepta_ventas_en_cero():
    """Un feriado o una tienda cerrada producen ceros legítimos."""
    validar(_sales(sales=[0.0, 0.0]), "sales")


def test_acepta_columnas_extra():
    """`id` de Kaggle y `arrival_date` nuestro no deben estorbar."""
    df = _sales()
    df["id"] = [1, 2]
    df["arrival_date"] = pd.Timestamp("2017-07-06")
    validar(df, "sales")


def test_oil_acepta_nulos():
    """43 nulos reales: el petróleo no cotiza fines de semana ni feriados.

    Es el caso donde un contrato ingenuo (`nullable=False`) haría fallar el
    DAG todos los lunes.
    """
    oil = pd.DataFrame({"date": ["2017-07-01", "2017-07-02"], "dcoilwtico": [46.0, None]})
    validado = validar(oil, "oil")
    assert validado["dcoilwtico"].isna().sum() == 1


def test_onpromotion_acepta_nulos():
    """Es variable explicativa: se imputa aguas abajo, no detiene nada."""
    validar(_sales(onpromotion=[0, None]), "sales")


def test_holidays_admite_varias_filas_por_fecha():
    """31 fechas reales tienen feriado nacional y local a la vez.

    Si `date` fuera la clave primaria, deduplicar perdería los feriados
    regionales, que son los que explican los picos de venta por ciudad.
    """
    holidays = pd.DataFrame(
        {
            "date": ["2017-07-05", "2017-07-05"],
            "type": ["Holiday", "Holiday"],
            "locale": ["National", "Local"],
            "locale_name": ["Ecuador", "Quito"],
            "description": ["Primer Grito", "Fundacion de Quito"],
            "transferred": [False, False],
        }
    )
    assert len(validar(holidays, "holidays_events")) == 2


# ---------------------------------------------------------------------------
# Lo que debe fallar: estructura
# ---------------------------------------------------------------------------


def test_rechaza_columna_faltante():
    df = _sales().drop(columns=["sales"])
    with pytest.raises(ErrorDeContrato, match="sales"):
        validar(df, "sales")


def test_rechaza_clave_primaria_duplicada():
    """Dos filas para el mismo (date, store, family) hacen ambigua la verdad."""
    df = _sales(store_nbr=[1, 1], family=["GROCERY I", "GROCERY I"])
    with pytest.raises(ErrorDeContrato):
        validar(df, "sales")


def test_rechaza_fecha_ilegible():
    with pytest.raises(ErrorDeContrato):
        validar(_sales(date=["2017-07-05", "no-es-fecha"]), "sales")


def test_rechaza_objetivo_nulo():
    """Sin ventas no hay nada que modelar: no se puede imputar sin inventar."""
    with pytest.raises(ErrorDeContrato):
        validar(_sales(sales=[10.0, None]), "sales")


# ---------------------------------------------------------------------------
# Lo que debe fallar: valores imposibles
# ---------------------------------------------------------------------------


def test_rechaza_ventas_negativas():
    with pytest.raises(ErrorDeContrato):
        validar(_sales(sales=[10.0, -5.0]), "sales")


def test_rechaza_precio_de_petroleo_negativo():
    oil = pd.DataFrame({"date": ["2017-07-03"], "dcoilwtico": [-1.0]})
    with pytest.raises(ErrorDeContrato):
        validar(oil, "oil")


def test_rechaza_tienda_cero_o_negativa():
    with pytest.raises(ErrorDeContrato):
        validar(_sales(store_nbr=[1, 0]), "sales")


def test_el_error_dice_que_archivo_fallo():
    """En los logs de Airflow el mensaje debe bastar para actuar."""
    with pytest.raises(ErrorDeContrato, match="sales_2017-07-05.csv"):
        validar(_sales(sales=[10.0, -1.0]), "sales", origen="landing/.../sales_2017-07-05.csv")


def test_fuente_desconocida_falla_claramente():
    with pytest.raises(KeyError, match="No hay contrato"):
        validar(_sales(), "inventada")


# ---------------------------------------------------------------------------
# Lo que sólo debe avisar (nivel 3)
# ---------------------------------------------------------------------------


def test_una_tienda_nueva_avisa_pero_no_falla():
    """El negocio abre tiendas. Eso no puede tumbar el flujo."""
    df = _sales(store_nbr=[1, 99])
    validar(df, "sales")  # no lanza

    avisos = avisos_de_negocio(df, "sales")
    assert any("99" in a for a in avisos)


def test_un_tipo_de_feriado_nuevo_avisa_pero_no_falla():
    """La razón de no usar `Check.isin` sobre el vocabulario del negocio."""
    holidays = pd.DataFrame(
        {
            "date": ["2017-07-05"],
            "type": ["Pandemia"],
            "locale": ["National"],
            "locale_name": ["Ecuador"],
            "description": ["Cierre sanitario"],
            "transferred": [False],
        }
    )
    validar(holidays, "holidays_events")  # no lanza

    avisos = avisos_de_negocio(holidays, "holidays_events")
    assert any("Pandemia" in a for a in avisos)


def test_sin_novedades_no_hay_avisos():
    assert avisos_de_negocio(_sales(), "sales") == []


# ---------------------------------------------------------------------------
# Coherencia interna
# ---------------------------------------------------------------------------


def test_holidays_no_usa_la_fecha_como_clave():
    assert CLAVES["holidays_events"] != ["date"]
    assert len(CLAVES["holidays_events"]) == 5
