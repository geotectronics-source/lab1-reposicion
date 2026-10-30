"""Pruebas de las variables explicativas.

La prueba central es `test_los_valores_futuros_no_alteran_las_features`: si
una venta posterior al origen cambia aunque sea un decimal de una variable,
hay fuga de ventana y todo lo que se construya encima queda invalidado.

El resto vigila el calendario de Favorita, que tiene sutilezas (traslados,
alcance local, días laborales compensatorios) donde es fácil meter ruido sin
notarlo.
"""

from __future__ import annotations

import pandas as pd
import pytest

from reposicion.features import (
    agregar_objetivo,
    construir_panel,
    features_calendario,
    features_de_serie,
    features_feriados,
    features_petroleo,
    features_transacciones,
    matriz_de_ventas,
    media_por_dia_de_semana,
    origen_de,
    origenes_de_entrenamiento,
)
from reposicion.io import marcar_llegada, ruta_particion
from reposicion.landing import ARRIVAL_HISTORICO

CONFIG = {
    "pronostico": {"dias": 7},
    "features": {"ventanas": [7, 14], "semanas_mismo_dia": 4, "dias_de_pago": [15, "ultimo"]},
}

VENTANAS = [7, 14]
ORIGEN = pd.Timestamp("2017-07-16")


def _ventas(desde: str, hasta: str, valor: float = 10.0, tienda: int = 1) -> pd.DataFrame:
    fechas = pd.date_range(desde, hasta, freq="D")
    return pd.DataFrame(
        {
            "date": fechas,
            "store_nbr": tienda,
            "family": "GROCERY I",
            "sales": valor,
            "onpromotion": 0,
        }
    )


def _escribir_fuente(base, fuente: str, df: pd.DataFrame) -> None:
    from reposicion.io import escribir_particion

    escribir_particion(
        marcar_llegada(df, ARRIVAL_HISTORICO), ruta_particion(base, fuente, ARRIVAL_HISTORICO)
    )


@pytest.fixture
def raw_minimo_solo_ventas(tmp_path):
    """Almacén con ventas y tiendas únicamente: el mínimo para pronosticar."""
    base = tmp_path / "raw"
    ventas = pd.concat(
        [
            _ventas("2017-05-01", "2017-07-16", tienda=1),
            _ventas("2017-05-01", "2017-07-16", tienda=2, valor=20.0),
        ],
        ignore_index=True,
    )
    _escribir_fuente(base, "sales", ventas)
    _escribir_fuente(
        base,
        "stores",
        pd.DataFrame(
            [
                {"store_nbr": 1, "city": "Quito", "state": "Pichincha", "type": "A", "cluster": 1},
                {"store_nbr": 2, "city": "Guayaquil", "state": "Guayas", "type": "B", "cluster": 2},
            ]
        ),
    )
    return base


@pytest.fixture
def raw_minimo(raw_minimo_solo_ventas, tmp_path):
    """El anterior más las fuentes de contexto."""
    base = raw_minimo_solo_ventas
    _escribir_fuente(
        base,
        "oil",
        pd.DataFrame(
            {"date": pd.date_range("2017-05-01", "2017-07-14"), "dcoilwtico": 48.0}
        ),
    )
    _escribir_fuente(
        base,
        "transactions",
        pd.DataFrame(
            {
                "date": list(pd.date_range("2017-05-01", "2017-07-14")) * 2,
                "store_nbr": [1] * 75 + [2] * 75,
                "transactions": 100,
            }
        ),
    )
    _escribir_fuente(
        base,
        "holidays_events",
        pd.DataFrame([_feriado("2017-07-20")]),
    )
    return base


# ---------------------------------------------------------------------------
# LA prueba: fuga de ventana
# ---------------------------------------------------------------------------


def test_los_valores_futuros_no_alteran_las_features():
    """Ninguna variable del origen puede depender de lo que pasa después.

    Se construyen dos escenarios idénticos hasta el origen y radicalmente
    distintos a partir de él. Si alguna variable difiere, una ventana está
    alcanzando el futuro: el modelo parecería excelente en validación y
    fracasaría en operación, sin que nada falle de forma visible.
    """
    pasado = _ventas("2017-05-01", "2017-07-16", valor=10.0)

    futuro_tranquilo = _ventas("2017-07-17", "2017-07-31", valor=10.0)
    futuro_extremo = _ventas("2017-07-17", "2017-07-31", valor=99_999.0)

    izq = pd.concat([pasado, futuro_tranquilo], ignore_index=True)
    der = pd.concat([pasado, futuro_extremo], ignore_index=True)

    features_izq = features_de_serie(matriz_de_ventas(izq, ORIGEN), ORIGEN, VENTANAS, 4)
    features_der = features_de_serie(matriz_de_ventas(der, ORIGEN), ORIGEN, VENTANAS, 4)

    pd.testing.assert_frame_equal(features_izq, features_der)


def test_la_media_por_dia_de_semana_tampoco_mira_al_futuro():
    pasado = _ventas("2017-05-01", "2017-07-16", valor=10.0)
    con_futuro = pd.concat(
        [pasado, _ventas("2017-07-17", "2017-07-31", valor=99_999.0)], ignore_index=True
    )

    sin = media_por_dia_de_semana(matriz_de_ventas(pasado, ORIGEN), ORIGEN, 4)
    con = media_por_dia_de_semana(matriz_de_ventas(con_futuro, ORIGEN), ORIGEN, 4)

    pd.testing.assert_frame_equal(sin, con)


def test_la_matriz_se_corta_en_el_origen():
    ventas = _ventas("2017-07-01", "2017-07-31")
    matriz = matriz_de_ventas(ventas, ORIGEN)
    assert matriz.index.max() == ORIGEN


# ---------------------------------------------------------------------------
# Corrección de las ventanas
# ---------------------------------------------------------------------------


def test_la_ventana_incluye_el_origen():
    """El día del origen ya se conoce al emitir: excluirlo desperdiciaría el
    dato más reciente y más informativo."""
    ventas = _ventas("2017-07-01", "2017-07-15", valor=10.0)
    ventas = pd.concat([ventas, _ventas("2017-07-16", "2017-07-16", valor=80.0)])

    features = features_de_serie(matriz_de_ventas(ventas, ORIGEN), ORIGEN, [7], 4)
    # Seis días a 10 y uno a 80 -> (6*10 + 80) / 7
    assert features["media_7"].iloc[0] == pytest.approx((6 * 10 + 80) / 7)


def test_la_ventana_tiene_la_longitud_pedida():
    ventas = pd.concat(
        [
            _ventas("2017-07-01", "2017-07-09", valor=100.0),  # fuera de la ventana de 7
            _ventas("2017-07-10", "2017-07-16", valor=10.0),  # exactamente los 7 últimos
        ]
    )
    features = features_de_serie(matriz_de_ventas(ventas, ORIGEN), ORIGEN, [7], 4)
    assert features["media_7"].iloc[0] == pytest.approx(10.0)


def test_los_dias_faltantes_se_rellenan_con_cero():
    """Faltan cuatro fechas en el histórico: las navidades, cuando cierran
    todas las tiendas. Sin densificar, siete registros abarcarían ocho días."""
    ventas = pd.concat(
        [_ventas("2017-07-10", "2017-07-12"), _ventas("2017-07-15", "2017-07-16")]
    )
    matriz = matriz_de_ventas(ventas, ORIGEN)

    assert len(matriz) == 7  # del 10 al 16, sin huecos
    assert matriz.loc[pd.Timestamp("2017-07-13")].iloc[0] == 0.0


def test_no_se_inventa_historia_antes_de_la_primera_venta():
    """Una tienda que abrió hace un mes no tiene ceros de hace un año.

    Rellenarlos haría que su nivel reciente pareciera mucho más bajo de lo que
    es, y el pedido saldría sistemáticamente corto.
    """
    antigua = _ventas("2017-05-01", "2017-07-16", valor=10.0, tienda=1)
    nueva = _ventas("2017-07-10", "2017-07-16", valor=10.0, tienda=2)
    matriz = matriz_de_ventas(pd.concat([antigua, nueva]), ORIGEN)

    columna_nueva = [c for c in matriz.columns if c[0] == 2][0]
    assert matriz[columna_nueva].loc[: pd.Timestamp("2017-07-09")].isna().all()
    assert matriz[columna_nueva].loc[pd.Timestamp("2017-07-10") :].notna().all()


def test_la_proporcion_de_ceros_distingue_baja_rotacion():
    """Un producto que vende poco todos los días no es lo mismo que uno que
    vende mucho de vez en cuando: se reponen distinto."""
    ventas = pd.concat(
        [_ventas("2017-07-03", "2017-07-09", valor=0.0), _ventas("2017-07-10", "2017-07-16")]
    )
    features = features_de_serie(matriz_de_ventas(ventas, ORIGEN), ORIGEN, [7, 14], 4)
    assert features["prop_ceros_14"].iloc[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Origen
# ---------------------------------------------------------------------------


def test_el_origen_sale_de_los_datos_no_de_la_fecha_de_emision():
    """Si el sistema origen se retrasa, el horizonte se alarga solo."""
    assert origen_de(_ventas("2017-07-01", "2017-07-14")) == pd.Timestamp("2017-07-14")


def test_sin_ventas_no_hay_origen_posible():
    with pytest.raises(ValueError, match="no se puede fijar un origen"):
        origen_de(pd.DataFrame({"date": []}))


# ---------------------------------------------------------------------------
# Calendario
# ---------------------------------------------------------------------------


def test_marca_los_dias_de_cobro():
    fechas = pd.date_range("2017-07-01", "2017-07-31", freq="D")
    cal = features_calendario(fechas, [15, "ultimo"])

    pagos = cal.loc[cal["es_dia_pago"], "date"].dt.day.tolist()
    assert pagos == [15, 31]


def test_el_ultimo_dia_del_mes_se_ajusta_a_cada_mes():
    """Febrero no tiene 31, y en año bisiesto tiene 29."""
    cal = features_calendario(pd.date_range("2016-02-01", "2016-02-29"), ["ultimo"])
    assert cal.loc[cal["es_dia_pago"], "date"].dt.day.tolist() == [29]


def test_cuenta_los_dias_desde_el_ultimo_cobro():
    cal = features_calendario(pd.date_range("2017-07-14", "2017-07-18"), [15, "ultimo"])
    serie = cal.set_index("date")["dias_desde_pago"]
    assert serie[pd.Timestamp("2017-07-15")] == 0
    assert serie[pd.Timestamp("2017-07-18")] == 3


def test_cuenta_los_dias_aunque_el_cobro_quede_fuera_del_horizonte():
    """Regresión: la versión anterior buscaba el último cobro sólo entre las
    fechas pedidas, así que una semana sin día 15 ni fin de mes dejaba la
    columna entera en nulo. En producción eso es lo habitual: la mayoría de
    las semanas no contienen un día de pago.

    La prueba anterior no lo detectaba porque su rango incluía el día 15.
    """
    cal = features_calendario(pd.date_range("2017-08-07", "2017-08-13"), [15, "ultimo"])

    assert cal["dias_desde_pago"].notna().all()
    # El último cobro fue el 31 de julio: el 7 de agosto son 7 días.
    assert cal.set_index("date")["dias_desde_pago"][pd.Timestamp("2017-08-07")] == 7


def test_el_cobro_anterior_puede_estar_en_el_mes_pasado():
    cal = features_calendario(pd.date_range("2017-03-02", "2017-03-03"), [15, "ultimo"])
    # 28 de febrero de 2017 fue el último día del mes.
    assert cal.set_index("date")["dias_desde_pago"][pd.Timestamp("2017-03-02")] == 2


def test_identifica_el_fin_de_semana():
    cal = features_calendario(pd.date_range("2017-07-17", "2017-07-23"), [15])
    finde = cal.loc[cal["es_fin_semana"], "date"].dt.day_name().tolist()
    assert finde == ["Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# Feriados: las tres sutilezas del calendario de Favorita
# ---------------------------------------------------------------------------


TIENDAS = pd.DataFrame(
    [
        {"store_nbr": 1, "city": "Quito", "state": "Pichincha"},
        {"store_nbr": 2, "city": "Guayaquil", "state": "Guayas"},
    ]
)


def _feriado(fecha: str, tipo="Holiday", locale="National", nombre="Ecuador", transferido=False):
    return {
        "date": pd.Timestamp(fecha),
        "type": tipo,
        "locale": locale,
        "locale_name": nombre,
        "description": "prueba",
        "transferred": transferido,
    }


def test_un_feriado_trasladado_no_cuenta_en_su_fecha_original():
    """`transferred=True` significa que ESE día no se celebró: se movió.

    Marcarlo como feriado enseñaría al modelo a esperar una caída de ventas en
    un día que fue laborable normal.
    """
    holidays = pd.DataFrame([_feriado("2017-07-20", transferido=True)])
    df = features_feriados(pd.date_range("2017-07-20", "2017-07-20"), holidays, TIENDAS)
    assert not df["es_feriado"].any()


def test_el_dia_al_que_se_traslado_si_cuenta():
    holidays = pd.DataFrame([_feriado("2017-07-24", tipo="Transfer")])
    df = features_feriados(pd.date_range("2017-07-24", "2017-07-24"), holidays, TIENDAS)
    assert df["es_feriado"].all()


def test_un_dia_laboral_compensatorio_no_es_feriado():
    """`Work Day` es lo contrario: un sábado que se trabaja."""
    holidays = pd.DataFrame([_feriado("2017-07-22", tipo="Work Day")])
    df = features_feriados(pd.date_range("2017-07-22", "2017-07-22"), holidays, TIENDAS)

    assert not df["es_feriado"].any()
    assert df["es_dia_laboral_extra"].all()


def test_un_feriado_local_solo_afecta_a_su_ciudad():
    """Tratarlo como nacional metería ruido en 53 tiendas para explicar una."""
    holidays = pd.DataFrame([_feriado("2017-07-25", locale="Local", nombre="Quito")])
    df = features_feriados(pd.date_range("2017-07-25", "2017-07-25"), holidays, TIENDAS)

    por_tienda = df.set_index("store_nbr")["es_feriado_local"]
    assert por_tienda[1]  # Quito
    assert not por_tienda[2]  # Guayaquil


def test_un_feriado_regional_aplica_por_provincia():
    holidays = pd.DataFrame([_feriado("2017-07-25", locale="Regional", nombre="Guayas")])
    df = features_feriados(pd.date_range("2017-07-25", "2017-07-25"), holidays, TIENDAS)

    por_tienda = df.set_index("store_nbr")["es_feriado_local"]
    assert not por_tienda[1]
    assert por_tienda[2]


def test_un_feriado_nacional_aplica_a_todas():
    holidays = pd.DataFrame([_feriado("2017-08-10")])
    df = features_feriados(pd.date_range("2017-08-10", "2017-08-10"), holidays, TIENDAS)
    assert df["es_feriado_nacional"].all()


# ---------------------------------------------------------------------------
# Exógenas
# ---------------------------------------------------------------------------


def test_el_petroleo_arrastra_el_ultimo_valor_conocido():
    """No cotiza fines de semana: es lo que un analista vería ese día."""
    oil = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-07-13", "2017-07-14"]),
            "dcoilwtico": [46.0, 48.0],
        }
    )
    valores = features_petroleo(oil, ORIGEN, [7])
    assert valores["petroleo_origen"] == 48.0  # el 16 es domingo


def test_el_petroleo_no_propaga_nulos():
    """El primer registro del histórico es nulo; tras el arrastre podría
    quedar un hueco que contaminaría las 1.782 series."""
    oil = pd.DataFrame(
        {"date": pd.to_datetime(["2017-07-10", "2017-07-14"]), "dcoilwtico": [None, 48.0]}
    )
    valores = features_petroleo(oil, ORIGEN, [7])
    assert all(pd.notna(v) for v in valores.values())


def test_el_petroleo_ausente_no_rompe_nada():
    assert features_petroleo(pd.DataFrame(), ORIGEN, [7]) == {}


def test_las_transacciones_declaran_su_retraso():
    """Llegan con dos días de latencia: el retraso se expone, no se disimula."""
    trans = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-07-14", "2017-07-14"]),
            "store_nbr": [1, 2],
            "transactions": [100, 200],
        }
    )
    df = features_transacciones(trans, ORIGEN, [7])
    assert (df["dias_retraso_trans"] == 2).all()


def test_las_transacciones_ausentes_no_rompen_nada():
    assert features_transacciones(pd.DataFrame(), ORIGEN, [7]).empty


# ---------------------------------------------------------------------------
# Orígenes de entrenamiento
# ---------------------------------------------------------------------------


def test_el_panel_tiene_una_fila_por_serie_y_dia(raw_minimo):
    panel = construir_panel("2017-07-17", raw=raw_minimo, config=CONFIG)

    series = panel.groupby(["store_nbr", "family"]).ngroups
    assert len(panel) == series * 7
    assert sorted(panel["h"].unique()) == [1, 2, 3, 4, 5, 6, 7]


def test_el_panel_no_trae_la_variable_objetivo(raw_minimo):
    """Las features se construyen sin mirar lo que hay que predecir."""
    panel = construir_panel("2017-07-17", raw=raw_minimo, config=CONFIG)
    assert "y" not in panel.columns
    assert "sales" not in panel.columns


def test_el_horizonte_se_mide_desde_el_origen(raw_minimo):
    panel = construir_panel("2017-07-17", raw=raw_minimo, config=CONFIG)
    assert ((panel["date"] - panel["origen"]).dt.days == panel["h"]).all()


def test_el_panel_se_construye_sin_las_fuentes_opcionales(raw_minimo_solo_ventas):
    """INVARIANTE 5: que falte oil, holidays o transactions es operación
    normal, no un fallo. El pronóstico se emite igual, con menos contexto."""
    panel = construir_panel("2017-07-17", raw=raw_minimo_solo_ventas, config=CONFIG)

    assert not panel.empty
    assert "media_7" in panel.columns


def test_agregar_objetivo_une_por_serie_y_fecha(raw_minimo):
    panel = construir_panel("2017-07-17", raw=raw_minimo, config=CONFIG)
    reales = _ventas("2017-07-17", "2017-07-23", valor=42.0)

    con_objetivo = agregar_objetivo(panel, reales)
    assert (con_objetivo["y"] == 42.0).all()
    assert len(con_objetivo) == 7  # sólo la serie que existe en `reales`


def test_los_origenes_siguen_la_cadencia_de_emision():
    """Entrenar con los mismos orígenes en que se emite evita que el modelo
    aprenda sobre horizontes que nunca se dan en operación."""
    origenes = origenes_de_entrenamiento(pd.Timestamp("2017-07-16"), 7, 28)

    assert origenes[-1] == pd.Timestamp("2017-07-16")
    separaciones = {(b - a).days for a, b in zip(origenes, origenes[1:], strict=False)}
    assert separaciones == {7}
