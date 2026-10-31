"""Pruebas del simulador del sistema de origen.

El simulador no forma parte del producto, pero su lógica de fechas define si
el resto del laboratorio puede ser correcto punto-en-el-tiempo. Si la bandeja
se construye mal, todo lo que se apoye en ella hereda el error en silencio.

Las pruebas usan datos sintéticos minúsculos: lo que se verifica es el
calendario de llegadas, no el contenido de Kaggle.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tools.build_landing import (
    Entrega,
    aplicar_incidentes,
    calcular_corte,
    planificar,
)

# --- Fixtures de datos sintéticos ------------------------------------------

FUENTES_HECHO = {
    "sales": {
        "archivo": "train.csv",
        "tipo": "hecho_diario",
        "latencia_dias": 1,
        "clave": ["date", "store_nbr", "family"],
    },
    "transactions": {
        "archivo": "transactions.csv",
        "tipo": "hecho_diario",
        "latencia_dias": 2,
        "clave": ["date", "store_nbr"],
    },
}


def _hechos(desde: str, hasta: str, tiendas=(1, 44)) -> pd.DataFrame:
    fechas = pd.date_range(desde, hasta, freq="D")
    filas = [
        {"date": f, "store_nbr": t, "family": "GROCERY I", "sales": 10.0}
        for f in fechas
        for t in tiendas
    ]
    return pd.DataFrame(filas)


@pytest.fixture
def datos() -> dict[str, pd.DataFrame]:
    return {
        "sales": _hechos("2017-06-25", "2017-07-10"),
        "transactions": _hechos("2017-06-25", "2017-07-10"),
    }


# --- Corte histórico --------------------------------------------------------


def test_corte_se_deriva_de_dias_de_operacion(datos):
    """Sin corte explícito, se cuenta hacia atrás desde la última venta."""
    corte = calcular_corte(datos["sales"], {"dias_operacion": 10, "corte_historico": None})
    assert corte == pd.Timestamp("2017-06-30")  # 2017-07-10 menos 10 días


def test_corte_explicito_tiene_prioridad(datos):
    corte = calcular_corte(
        datos["sales"], {"dias_operacion": 10, "corte_historico": "2017-07-01"}
    )
    assert corte == pd.Timestamp("2017-07-01")


def test_corte_se_adapta_a_datos_mas_recientes():
    """El evaluador usará datos no publicados: el corte debe seguirlos.

    Es la razón por la que `corte_historico` se deja en null en la config.
    """
    extendidos = _hechos("2017-06-25", "2017-10-31")
    corte = calcular_corte(extendidos, {"dias_operacion": 10, "corte_historico": None})
    assert corte == pd.Timestamp("2017-10-21")


# --- Calendario de llegadas -------------------------------------------------


def test_la_latencia_define_la_fecha_de_llegada(datos):
    """INVARIANTE 1: arrival_date = date + latencia, por fuente."""
    plan = planificar(datos, FUENTES_HECHO, pd.Timestamp("2017-06-30"))

    ventas = {e.nombre: e.arrival_date for e in plan if e.fuente == "sales"}
    assert ventas["sales_2017-07-01.csv"] == pd.Timestamp("2017-07-02")  # latencia 1

    trans = {e.nombre: e.arrival_date for e in plan if e.fuente == "transactions"}
    assert trans["transactions_2017-07-01.csv"] == pd.Timestamp("2017-07-03")  # latencia 2


def test_el_historico_no_se_reenvia_en_la_bandeja(datos):
    """Lo anterior al corte ya está en la carga inicial: no debe duplicarse."""
    corte = pd.Timestamp("2017-06-30")
    plan = planificar(datos, FUENTES_HECHO, corte)

    fechas_de_datos = [
        pd.Timestamp(e.nombre.removeprefix(f"{e.fuente}_").removesuffix(".csv")) for e in plan
    ]
    assert min(fechas_de_datos) > corte


def test_cada_entrega_contiene_solo_su_propio_dia(datos):
    plan = planificar(datos, FUENTES_HECHO, pd.Timestamp("2017-06-30"))
    entrega = next(e for e in plan if e.nombre == "sales_2017-07-03.csv")
    assert entrega.datos["date"].unique().tolist() == [pd.Timestamp("2017-07-03")]


def test_bandeja_vacia_falla_con_mensaje_claro(datos):
    """Un corte posterior a los datos es un error de configuración, no un caso
    silencioso: debe detenerse con una explicación accionable."""
    with pytest.raises(ValueError, match="bandeja quedaría vacía"):
        planificar(datos, FUENTES_HECHO, pd.Timestamp("2018-01-01"))


# --- Catálogos --------------------------------------------------------------


def test_catalogo_publica_el_futuro_conocido(datos):
    """Los feriados son la única fuente que informa sobre el futuro.

    Un snapshot publicado el 2017-07-01 con 60 días de anticipación debe
    incluir feriados de agosto, y eso NO viola la corrección punto-en-el-tiempo
    porque el calendario efectivamente se conoce por adelantado.
    """
    fuentes = dict(FUENTES_HECHO)
    fuentes["holidays_events"] = {
        "archivo": "holidays_events.csv",
        "tipo": "catalogo",
        "cadencia_dias": 30,
        "anticipacion_dias": 60,
        "clave": ["date"],
    }
    con_feriados = dict(datos)
    con_feriados["holidays_events"] = pd.DataFrame(
        {"date": pd.to_datetime(["2017-06-01", "2017-08-10", "2017-12-25"])}
    )

    plan = planificar(con_feriados, fuentes, pd.Timestamp("2017-06-30"))
    snapshot = next(e for e in plan if e.fuente == "holidays_events")

    fechas = snapshot.datos["date"].tolist()
    assert pd.Timestamp("2017-08-10") in fechas, "debe anticipar el feriado de agosto"
    assert pd.Timestamp("2017-12-25") not in fechas, "no debe filtrar más allá de la anticipación"


# --- Incidentes de operación ------------------------------------------------


def test_correccion_retroactiva_conserva_la_entrega_original(datos):
    """INVARIANTE 4: la corrección se añade, no sobrescribe.

    Deben quedar DOS entregas del mismo archivo con fechas de llegada distintas:
    la degradada que se conoció primero y la verdadera que llegó después.
    """
    plan = planificar(datos, FUENTES_HECHO, pd.Timestamp("2017-06-30"))
    incidente = {
        "tipo": "correccion_retroactiva",
        "fuente": "sales",
        "fecha_datos": "2017-07-05",
        "corregido_en": "2017-07-20",
        "entrega_original": {"donde": {"store_nbr": 44}, "asignar": {"sales": 0.0}},
    }
    plan = aplicar_incidentes(plan, [incidente])

    versiones = sorted(
        (e for e in plan if e.nombre == "sales_2017-07-05.csv"),
        key=lambda e: e.arrival_date,
    )
    assert len(versiones) == 2

    primera, segunda = versiones
    assert primera.arrival_date == pd.Timestamp("2017-07-06")  # date + latencia
    assert segunda.arrival_date == pd.Timestamp("2017-07-20")

    t44_primera = primera.datos.loc[primera.datos.store_nbr == 44, "sales"]
    t44_segunda = segunda.datos.loc[segunda.datos.store_nbr == 44, "sales"]
    assert (t44_primera == 0.0).all(), "la entrega original llega degradada"
    assert (t44_segunda > 0.0).all(), "la corrección trae el dato real"


def test_correccion_retroactiva_no_toca_otras_tiendas(datos):
    plan = planificar(datos, FUENTES_HECHO, pd.Timestamp("2017-06-30"))
    plan = aplicar_incidentes(
        plan,
        [
            {
                "tipo": "correccion_retroactiva",
                "fuente": "sales",
                "fecha_datos": "2017-07-05",
                "corregido_en": "2017-07-20",
                "entrega_original": {"donde": {"store_nbr": 44}, "asignar": {"sales": 0.0}},
            }
        ],
    )
    degradada = min(
        (e for e in plan if e.nombre == "sales_2017-07-05.csv"), key=lambda e: e.arrival_date
    )
    assert (degradada.datos.loc[degradada.datos.store_nbr == 1, "sales"] == 10.0).all()


def test_fuente_que_no_llega_se_acumula_al_dia_siguiente(datos):
    """INVARIANTE 5: la ausencia de una entrega es una situación de negocio.

    El dato no se pierde: aparece junto al del día siguiente, que es como se
    comporta un sistema real tras recuperarse de una caída.
    """
    plan = planificar(datos, FUENTES_HECHO, pd.Timestamp("2017-06-30"))
    antes = sum(1 for e in plan if e.fuente == "transactions")

    plan = aplicar_incidentes(
        plan,
        [{"tipo": "fuente_no_llega", "fuente": "transactions", "arrival_date": "2017-07-05"}],
    )

    por_llegada: dict[pd.Timestamp, int] = {}
    for e in plan:
        if e.fuente == "transactions":
            por_llegada[e.arrival_date] = por_llegada.get(e.arrival_date, 0) + 1

    assert pd.Timestamp("2017-07-05") not in por_llegada, "ese día no hubo entrega"
    assert por_llegada[pd.Timestamp("2017-07-06")] == 2, "se acumulan dos archivos"
    assert sum(por_llegada.values()) == antes, "ningún dato se pierde"


def test_incidente_desconocido_falla_explicitamente(datos):
    plan = planificar(datos, FUENTES_HECHO, pd.Timestamp("2017-06-30"))
    with pytest.raises(ValueError, match="Tipo de incidente desconocido"):
        aplicar_incidentes(plan, [{"tipo": "terremoto"}])


# --- Contrato de rutas ------------------------------------------------------


def test_la_ruta_codifica_la_fecha_de_llegada():
    """La corrección punto-en-el-tiempo depende de poder filtrar por carpeta."""
    entrega = Entrega(
        arrival_date=pd.Timestamp("2017-07-06"),
        fuente="sales",
        nombre="sales_2017-07-05.csv",
        datos=pd.DataFrame(),
    )
    assert entrega.ruta_relativa.as_posix() == (
        "arrival_date=2017-07-06/sales/sales_2017-07-05.csv"
    )


def test_la_bandeja_se_escribe_con_lf_en_cualquier_sistema(tmp_path):
    """`landing/` es la salida de la etapa 0, y DVC compara bytes.

    `to_csv` traduce el salto de línea al de la plataforma, así que sin fijarlo
    el mismo simulador producía CRLF en Windows y LF en Linux. El producto no
    se entera —el CSV se parsea igual— pero el hash de la etapa `bandeja`
    cambiaba por plataforma, y eso obsoletaba la cadena entera desde el primer
    eslabón.

    El simulador declara su propia constante en vez de importarla del producto:
    representa un sistema ajeno y no depende de él (INVARIANTE 2).
    """
    from tools.build_landing import escribir_csv

    destino = tmp_path / "sales" / "sales_2017-07-05.csv"
    escribir_csv(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2017-07-05", "2017-07-05"]),
                "store_nbr": [1, 2],
                "sales": [10.5, 20.0],
            }
        ),
        destino,
    )

    crudo = destino.read_bytes()
    assert b"\r" not in crudo, "la bandeja lleva retorno de carro: su hash dependería del sistema"
    assert crudo.count(b"\n") == 3, "cabecera y dos filas"
