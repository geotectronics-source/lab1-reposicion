"""Pruebas de la auditoría del almacén.

Lo que se prueba aquí es una sola distinción, y es la que más fácil se hace
mal: **"hoy no entregaron" y "la ingesta no corrió" se ven igual desde fuera**
—en los dos casos no hay datos nuevos— y significan cosas opuestas. Confundir
el primero con el segundo hace caer el DAG cada fin de semana; confundir el
segundo con el primero deja el producto emitiendo pedidos sobre datos viejos
sin que nadie se entere, que es el fallo caro.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from reposicion import calidad
from reposicion.config import limpiar_cache_config
from reposicion.io import escribir_particion, marcar_llegada, ruta_particion

RAIZ_REAL = Path(__file__).resolve().parents[1]

ASOF = pd.Timestamp("2017-07-17")
TIENDAS = [1, 2]
FAMILIAS = ["GROCERY I", "PRODUCE"]


@pytest.fixture
def almacen(tmp_path: Path, monkeypatch) -> Path:
    """Proyecto vacío con la configuración real de seguimiento.

    Se copia el YAML del proyecto y no uno inventado: así la prueba también
    comprueba que el archivo de verdad tiene las claves que el código espera,
    que es un fallo que ya ocurrió en este repositorio.
    """
    raiz = tmp_path / "proyecto"
    (raiz / "config").mkdir(parents=True)
    shutil.copy(RAIZ_REAL / "config" / "seguimiento.yaml", raiz / "config" / "seguimiento.yaml")

    monkeypatch.setenv("REPOSICION_HOME", str(raiz))
    limpiar_cache_config()
    yield raiz
    limpiar_cache_config()


def _ventas(fechas: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": fecha,
                "store_nbr": tienda,
                "family": familia,
                "sales": 100.0 + tienda,
                "onpromotion": 0,
            }
            for fecha in fechas
            for tienda in TIENDAS
            for familia in FAMILIAS
        ]
    )


def _llegada(raiz: Path, fuente: str, arrival: str, df: pd.DataFrame) -> None:
    """Materializa una partición del almacén como lo haría la ingesta."""
    escribir_particion(
        marcar_llegada(df, arrival), ruta_particion(raiz / "data" / "raw", fuente, arrival)
    )


def _fila(reporte: pd.DataFrame, fuente: str) -> dict:
    return reporte.set_index("fuente").loc[fuente].to_dict()


# ---------------------------------------------------------------------------
# INVARIANTE 5 — lo esperado del negocio no es un fallo
# ---------------------------------------------------------------------------


def test_una_fuente_que_no_entrego_hoy_es_un_hueco(almacen: Path):
    """Latencia D+1 y hoy sin entrega: queda anotado y no hay nada que fallar."""
    _llegada(almacen, "sales", "2017-07-16", _ventas(pd.date_range("2017-07-01", "2017-07-15")))

    reporte = calidad.auditar(ASOF)
    fila = _fila(reporte, "sales")

    assert fila["estado"] == calidad.HUECO
    assert fila["llego_en_asof"] is False
    assert fila["retraso_dias"] == 2
    assert calidad.huecos(reporte) == ["sales"]
    assert "sales" not in calidad.atrasadas(reporte), (
        "un hueco tolerado no debe hacer fallar la tarea"
    )


def test_una_fuente_al_dia_no_genera_ningun_aviso(almacen: Path):
    _llegada(almacen, "sales", ASOF, _ventas(pd.date_range("2017-07-01", "2017-07-16")))

    reporte = calidad.auditar(ASOF)

    assert _fila(reporte, "sales")["estado"] == calidad.AL_DIA
    assert calidad.huecos(reporte) == []
    assert "sales" not in calidad.atrasadas(reporte)


# ---------------------------------------------------------------------------
# Lo que sí es un fallo
# ---------------------------------------------------------------------------


def test_un_retraso_mayor_que_el_tolerado_si_es_un_fallo(almacen: Path):
    """Ocho días sin datos nuevos de ventas no es un fin de semana largo."""
    _llegada(almacen, "sales", "2017-07-10", _ventas(pd.date_range("2017-07-01", "2017-07-09")))

    reporte = calidad.auditar(ASOF)

    assert _fila(reporte, "sales")["retraso_dias"] == 8
    assert _fila(reporte, "sales")["estado"] == calidad.ATRASADO
    assert "sales" in calidad.atrasadas(reporte)


def test_una_fuente_sin_un_solo_dato_se_distingue_de_una_atrasada(almacen: Path):
    """"Nunca llegó nada" y "lleva días sin llegar" piden arreglos distintos.

    El primero es un despliegue incompleto; el segundo, un flujo que se paró.
    Darles el mismo nombre manda a depurar al sitio equivocado.
    """
    reporte = calidad.auditar(ASOF)

    assert _fila(reporte, "sales")["estado"] == calidad.VACIO
    assert _fila(reporte, "sales")["particiones"] == 0
    assert "sales" in calidad.atrasadas(reporte)


# ---------------------------------------------------------------------------
# INVARIANTE 1 — `date` y `arrival_date` no son lo mismo
# ---------------------------------------------------------------------------


def test_el_retraso_se_mide_sobre_date_y_no_sobre_la_llegada(almacen: Path):
    """La trampa: una entrega de hoy que trae datos de hace dos semanas.

    Medido sobre `arrival_date` el retraso saldría cero —acaba de llegar— y la
    auditoría daría luz verde a un almacén que no tiene los últimos trece
    días. Es el mismo error que invalidaría un pronóstico entero, aquí en
    pequeño.
    """
    _llegada(almacen, "sales", ASOF, _ventas(pd.date_range("2017-06-20", "2017-07-04")))

    fila = _fila(calidad.auditar(ASOF), "sales")

    assert fila["llego_en_asof"] is True, "la entrega de hoy existe"
    assert fila["retraso_dias"] == 13, "pero el dato más reciente es del día 4"
    assert fila["estado"] == calidad.ATRASADO


def test_no_mira_las_llegadas_posteriores_a_la_fecha_logica(almacen: Path):
    """Un backfill de julio no puede verse beneficiado por lo que llegó en agosto."""
    _llegada(almacen, "sales", "2017-07-16", _ventas(pd.date_range("2017-07-01", "2017-07-15")))
    _llegada(almacen, "sales", "2017-08-20", _ventas(pd.date_range("2017-07-16", "2017-08-19")))

    fila = _fila(calidad.auditar(ASOF), "sales")

    assert fila["particiones"] == 1
    assert fila["ultimo_dato"] == pd.Timestamp("2017-07-15")


# ---------------------------------------------------------------------------
# INVARIANTE 4 — una corrección se añade y se puede contar
# ---------------------------------------------------------------------------


def test_cuenta_las_correcciones_retroactivas(almacen: Path):
    """Reenviar un día ya conocido con otro valor deja un rastro medible.

    Sin esta cuenta, un reenvío de datos pasados es indistinguible de datos
    nuevos, y la pregunta "¿cambió algo de lo que ya habíamos publicado?" no
    tiene respuesta.
    """
    _llegada(almacen, "sales", "2017-07-16", _ventas(pd.date_range("2017-07-01", "2017-07-15")))

    corregido = _ventas(pd.DatetimeIndex([pd.Timestamp("2017-07-05")]))
    corregido["sales"] = 999.0
    _llegada(almacen, "sales", ASOF, corregido)

    series = len(TIENDAS) * len(FAMILIAS)
    fila = _fila(calidad.auditar(ASOF), "sales")

    assert fila["filas_en_asof"] == series
    assert fila["reenviadas_en_asof"] == series
    assert fila["correcciones_en_asof"] == series
    # La vista vigente no crece: la corrección reemplaza, no duplica.
    assert fila["filas_vigentes"] == 15 * series


def test_un_reenvio_identico_no_es_una_correccion(almacen: Path):
    """La distinción que la primera versión de esta auditoría no hacía.

    Un sistema origen real reenvía el día completo, no sólo las filas que
    cambiaron. Contar todo el reenvío como corrección informó de 1.782
    correcciones el día en que cambiaron 33 filas, y un reporte que exagera
    así manda a investigar un incidente que no existe.
    """
    dias = pd.date_range("2017-07-01", "2017-07-15")
    _llegada(almacen, "sales", "2017-07-16", _ventas(dias))

    # Mismos datos, exactamente, en una llegada nueva.
    _llegada(almacen, "sales", ASOF, _ventas(dias))

    fila = _fila(calidad.auditar(ASOF), "sales")

    assert fila["reenviadas_en_asof"] == len(dias) * len(TIENDAS) * len(FAMILIAS)
    assert fila["correcciones_en_asof"] == 0, "nada cambió de valor: no hubo corrección"


def test_los_datos_nuevos_no_cuentan_como_reenvio(almacen: Path):
    _llegada(almacen, "sales", "2017-07-16", _ventas(pd.date_range("2017-07-01", "2017-07-14")))
    _llegada(almacen, "sales", ASOF, _ventas(pd.DatetimeIndex([pd.Timestamp("2017-07-16")])))

    fila = _fila(calidad.auditar(ASOF), "sales")

    assert fila["filas_en_asof"] == len(TIENDAS) * len(FAMILIAS)
    assert fila["reenviadas_en_asof"] == 0
    assert fila["correcciones_en_asof"] == 0


# ---------------------------------------------------------------------------
# Fuentes que no se parecen a las ventas
# ---------------------------------------------------------------------------


def test_un_catalogo_sin_fecha_no_inventa_retraso(almacen: Path):
    """`stores` es estático: su antigüedad no significa nada."""
    _llegada(
        almacen,
        "stores",
        "2017-01-01",
        pd.DataFrame([{"store_nbr": 1, "city": "Quito", "state": "Pichincha"}]),
    )

    fila = _fila(calidad.auditar(ASOF), "stores")

    assert fila["estado"] == calidad.AL_DIA
    assert pd.isna(fila["retraso_dias"])


def test_un_calendario_que_informa_del_futuro_no_esta_atrasado(almacen: Path):
    """`holidays_events` es la única fuente que habla del futuro.

    Su último dato está a sesenta días por delante de la fecha lógica, así que
    la resta sale negativa. Un retraso negativo no es un estado que haya que
    vigilar, y dejarlo pasar tal cual ensuciaría el reporte con números que no
    quieren decir nada.
    """
    _llegada(
        almacen,
        "holidays_events",
        "2017-07-01",
        pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2017-09-15"),
                    "type": "Holiday",
                    "locale": "National",
                    "locale_name": "Ecuador",
                    "description": "Fiesta",
                    "transferred": False,
                }
            ]
        ),
    )

    fila = _fila(calidad.auditar(ASOF), "holidays_events")

    assert fila["retraso_dias"] == 0
    assert fila["estado"] == calidad.HUECO


def test_el_reporte_cubre_todas_las_fuentes_conocidas(almacen: Path):
    """Una fuente nueva aparece en el reporte sin tocar la auditoría."""
    from reposicion.ingesta import FUENTES

    assert sorted(calidad.auditar(ASOF)["fuente"]) == sorted(FUENTES)
