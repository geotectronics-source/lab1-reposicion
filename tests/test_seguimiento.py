"""Pruebas del seguimiento de desempeño.

El riesgo de un módulo que mide el pasado es que mida de más. Hay tres formas
de hacerlo, las tres cómodas, y las tres convierten la métrica en propaganda:

1. **Rellenar con cero** la venta de un día cuyo dato aún no ha llegado.
   Inventa una merma que nadie tuvo.
2. **Medir con lo que sabemos hoy** en vez de con lo que se sabía en la fecha
   de la medición. Da una cifra más exacta y no reproducible: cambia sola cada
   vez que llega una corrección.
3. **Calificar una emisión posterior** a la fecha lógica, que en el disco ya
   existe durante un backfill.

Cada una tiene su prueba aquí, y la de reproducibilidad es la que las ata:
volver a medir el 25 de julio tiene que dar el mismo número siempre.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from reposicion import seguimiento
from reposicion.config import cargar_config, limpiar_cache_config
from reposicion.io import (
    escribir_json,
    escribir_parquet,
    escribir_particion,
    marcar_llegada,
    ruta_particion,
)
from reposicion.metrics import costos_de_familia
from reposicion.policy import calcular_pedido

RAIZ_REAL = Path(__file__).resolve().parents[1]

EMISION = "2017-07-17"
HORIZONTE = pd.date_range(EMISION, periods=7, freq="D")
FAMILIA = "GROCERY I"
TIENDA = 1
CUANTILES = [0.5, 0.8, 0.9, 0.95]

#: Valor que se pone en TODOS los cuantiles del pronóstico sintético. Así el
#: pedido vale 12 sea cual sea el cuantil que elija la política, y la prueba
#: mide el costo realizado sin depender de la elección de cuantil, que ya
#: tiene sus propias pruebas en `test_policy`.
PEDIDO = 12.0


@pytest.fixture
def proyecto(tmp_path: Path, monkeypatch) -> Path:
    """Proyecto vacío con los tres YAML reales del producto."""
    raiz = tmp_path / "proyecto"
    (raiz / "config").mkdir(parents=True)
    for nombre in ("model", "costs", "seguimiento"):
        shutil.copy(RAIZ_REAL / "config" / f"{nombre}.yaml", raiz / "config" / f"{nombre}.yaml")

    monkeypatch.setenv("REPOSICION_HOME", str(raiz))
    limpiar_cache_config()
    yield raiz
    limpiar_cache_config()


def _publicar_pedido(raiz: Path, emision: str, fechas: pd.DatetimeIndex) -> pd.DataFrame:
    """Escribe un pedido como lo haría la etapa `pedir`.

    Se construye con `calcular_pedido` y no a mano: si el esquema del pedido
    cambiara, esta prueba se enteraría en vez de seguir midiendo un formato
    que ya no existe.
    """
    pronostico = pd.DataFrame(
        {
            "date": fechas,
            "store_nbr": TIENDA,
            "family": FAMILIA,
            "origen": fechas[0] - pd.Timedelta(days=1),
            "modelo": "lgbm_cuantilico",
            "emitida": pd.Timestamp(emision),
            **{f"q{q}": PEDIDO for q in CUANTILES},
        }
    )
    pedido = calcular_pedido(pronostico, cuantiles=CUANTILES)
    escribir_parquet(
        pedido, raiz / "data" / "pedidos" / f"asof={emision}" / seguimiento.ARCHIVO_PEDIDO
    )
    return pedido


def _publicar_ventas(raiz: Path, fechas: pd.DatetimeIndex, cantidad: float, llegada: str) -> None:
    ventas = pd.DataFrame(
        {
            "date": fechas,
            "store_nbr": TIENDA,
            "family": FAMILIA,
            "sales": cantidad,
            "onpromotion": 0,
        }
    )
    escribir_particion(
        marcar_llegada(ventas, llegada),
        ruta_particion(raiz / "data" / "raw", "sales", llegada),
    )


def _emision(resumen: dict, etiqueta: str) -> dict:
    return next(f for f in resumen["emisiones"] if f["emision"] == etiqueta)


# ---------------------------------------------------------------------------
# Lo que todavía no se puede calificar
# ---------------------------------------------------------------------------


def test_una_emision_sin_horizonte_realizado_se_informa_y_no_falla(proyecto: Path):
    """El lunes se publican siete días y no hay ni uno medible todavía.

    Es la situación normal, no un error: la medición tiene que decirlo y
    seguir (INVARIANTE 5).
    """
    _publicar_pedido(proyecto, EMISION, HORIZONTE)

    detalle, resumen = seguimiento.medir(EMISION)

    assert detalle.empty
    assert _emision(resumen, EMISION)["estado"] == "horizonte aun sin realizar"
    assert resumen["alertas"] == []


def test_sin_ninguna_emision_publicada_lo_dice_en_vez_de_fallar(proyecto: Path):
    detalle, resumen = seguimiento.medir(EMISION)

    assert detalle.empty
    assert resumen["emisiones"] == []
    assert "nota" in resumen


def test_las_fechas_cuyo_dato_no_ha_llegado_no_se_rellenan_con_cero(proyecto: Path):
    """La prueba más importante del módulo.

    Se publican siete días y sólo han llegado las ventas de los tres
    primeros. Rellenar los otros cuatro con cero inventaría una merma de 48
    unidades que nadie tuvo, y haría ver mal a un modelo que acertó. Lo
    correcto es medir tres días y decir que va por el 43 %.
    """
    _publicar_pedido(proyecto, EMISION, HORIZONTE)
    _publicar_ventas(proyecto, HORIZONTE[:3], 10.0, "2017-07-20")

    detalle, resumen = seguimiento.medir("2017-07-20")
    fila = _emision(resumen, EMISION)

    assert fila["lineas"] == 3
    assert fila["publicadas"] == 7
    assert fila["realizado"] == pytest.approx(3 / 7, abs=1e-4)
    assert detalle["lineas"].sum() == 3
    assert fila["sobrante_unidades"] == pytest.approx(3 * (PEDIDO - 10.0))


# ---------------------------------------------------------------------------
# INVARIANTE 1 — se mide con lo que se sabía, no con lo que se sabe
# ---------------------------------------------------------------------------


def test_no_mide_emisiones_posteriores_a_la_fecha_logica(proyecto: Path):
    """Durante un backfill, la emisión de agosto ya existe en el disco.

    Calificarla desde una medición de julio sería usar el futuro para evaluar
    el pasado, y el reporte del backfill mostraría un desempeño que en julio
    nadie podía conocer.
    """
    _publicar_pedido(proyecto, EMISION, HORIZONTE)
    _publicar_pedido(proyecto, "2017-08-14", pd.date_range("2017-08-14", periods=7, freq="D"))
    _publicar_ventas(proyecto, HORIZONTE, 10.0, "2017-07-25")

    _, resumen = seguimiento.medir("2017-07-25")

    assert [f["emision"] for f in resumen["emisiones"]] == [EMISION]


def test_ignora_las_ventas_que_llegaron_despues_de_la_fecha_logica(proyecto: Path):
    """Una corrección posterior no puede cambiar una medición ya hecha.

    Las ventas reales del horizonte llegan el día 25 y una corrección el 30.
    Medir el 25 tiene que dar el número del día 25, para siempre: es lo que
    permite comparar la serie de mediciones entre sí.
    """
    _publicar_pedido(proyecto, EMISION, HORIZONTE)
    _publicar_ventas(proyecto, HORIZONTE, 10.0, "2017-07-25")
    _publicar_ventas(proyecto, HORIZONTE, 999.0, "2017-07-30")

    _, el_25 = seguimiento.medir("2017-07-25")
    _, el_30 = seguimiento.medir("2017-07-30")

    assert _emision(el_25, EMISION)["sobrante_unidades"] == pytest.approx(7 * (PEDIDO - 10.0))
    # El día 30 sí se ve la corrección: la vista vigente cambió, y eso es
    # correcto. Lo que no puede cambiar es la medición del día 25.
    assert _emision(el_30, EMISION)["faltante_unidades"] == pytest.approx(7 * (999.0 - PEDIDO))


def test_volver_a_medir_la_misma_fecha_da_el_mismo_numero(proyecto: Path):
    """Idempotencia de la medición (INVARIANTE 3)."""
    _publicar_pedido(proyecto, EMISION, HORIZONTE)
    _publicar_ventas(proyecto, HORIZONTE, 10.0, "2017-07-25")

    primero, resumen_1 = seguimiento.medir("2017-07-25")
    segundo, resumen_2 = seguimiento.medir("2017-07-25")

    pd.testing.assert_frame_equal(primero, segundo)
    assert resumen_1 == resumen_2


# ---------------------------------------------------------------------------
# El costo realizado, en la moneda del negocio
# ---------------------------------------------------------------------------


def test_lo_que_sobra_se_cobra_a_precio_de_merma(proyecto: Path):
    """Pedimos 12, se vendieron 10: sobran 2 por día a `costo_merma`."""
    _publicar_pedido(proyecto, EMISION, HORIZONTE)
    _publicar_ventas(proyecto, HORIZONTE, 10.0, "2017-07-25")

    _, costo_merma = costos_de_familia(FAMILIA, cargar_config("costs"))
    detalle, _ = seguimiento.medir("2017-07-25")

    assert detalle["costo_realizado"].iloc[0] == pytest.approx(costo_merma * 2.0)


def test_lo_que_falta_se_cobra_a_precio_de_quiebre(proyecto: Path):
    """Pedimos 12, se vendieron 20: faltan 8 por día a `costo_quiebre`.

    La asimetría es el punto entero de la métrica: el mismo error de 8
    unidades cuesta distinto según el lado por el que se falle.
    """
    _publicar_pedido(proyecto, EMISION, HORIZONTE)
    _publicar_ventas(proyecto, HORIZONTE, 20.0, "2017-07-25")

    costo_quiebre, _ = costos_de_familia(FAMILIA, cargar_config("costs"))
    detalle, _ = seguimiento.medir("2017-07-25")

    assert detalle["costo_realizado"].iloc[0] == pytest.approx(costo_quiebre * 8.0)


def test_el_detalle_se_desagrega_por_familia(proyecto: Path):
    """Un total que se ve bien puede esconder una familia que va mal."""
    _publicar_pedido(proyecto, EMISION, HORIZONTE)
    _publicar_ventas(proyecto, HORIZONTE, 10.0, "2017-07-25")

    detalle, _ = seguimiento.medir("2017-07-25")

    assert set(detalle.columns) >= {"emision", "family", "costo_realizado", "cobertura_usada"}
    assert detalle["evaluado_en"].unique().tolist() == ["2017-07-25"]


def test_la_cobertura_mide_el_cuantil_que_uso_la_politica(proyecto: Path):
    """Con la venta siempre por debajo del pedido, la cobertura es del 100 %.

    No es un detalle contable: si la política pide cubrir el 90 % y en
    operación cubre el 60 %, el pedido se queda corto más seguido de lo que el
    negocio aceptó, y el error medio puede verse perfectamente bien.
    """
    _publicar_pedido(proyecto, EMISION, HORIZONTE)
    _publicar_ventas(proyecto, HORIZONTE, 1.0, "2017-07-25")

    _, resumen = seguimiento.medir("2017-07-25")

    assert _emision(resumen, EMISION)["cobertura_usada"] == 1.0


# ---------------------------------------------------------------------------
# Alertas: avisan, no interrumpen
# ---------------------------------------------------------------------------


def _baremo(raiz: Path, version: str, costo: float, modelo: str = "lgbm_cuantilico") -> None:
    escribir_json(
        {"campeon": modelo, "modelos": {modelo: {"costo_negocio": costo}}},
        raiz / "reportes" / f"asof={version}" / "metricas.json",
    )


def test_un_desempeno_mucho_peor_que_lo_validado_alerta_sin_lanzar(proyecto: Path):
    """La degradación es negocio: se avisa y se sigue.

    Hacer caer la tarea dejaría al día siguiente sin medición, que es
    exactamente cuando hace falta mirarla.
    """
    _publicar_pedido(proyecto, EMISION, HORIZONTE)
    _publicar_ventas(proyecto, HORIZONTE, 20.0, "2017-07-25")
    _baremo(proyecto, EMISION, costo=0.5)

    cfg = cargar_config("seguimiento") | {"minimo_lineas": 1}
    _, resumen = seguimiento.medir("2017-07-25", config=cfg)

    assert resumen["alertas"], "un costo cuarenta veces peor tenía que avisar"
    assert _emision(resumen, EMISION)["estado"] == "alerta"
    assert "costo realizado" in resumen["alertas"][0]


def test_un_desempeno_acorde_a_lo_validado_no_alerta(proyecto: Path):
    _publicar_pedido(proyecto, EMISION, HORIZONTE)
    _publicar_ventas(proyecto, HORIZONTE, 10.0, "2017-07-25")

    _, costo_merma = costos_de_familia(FAMILIA, cargar_config("costs"))
    _baremo(proyecto, EMISION, costo=costo_merma * 2.0)

    cfg = cargar_config("seguimiento") | {"minimo_lineas": 1}
    _, resumen = seguimiento.medir("2017-07-25", config=cfg)

    assert resumen["alertas"] == []
    assert _emision(resumen, EMISION)["estado"] == "normal"


def test_con_pocas_lineas_medidas_no_se_alerta(proyecto: Path):
    """Un martes, con un solo día realizado, cualquier cosa parece una crisis.

    El umbral de líneas evita que un feriado dispare una falsa alarma el
    primer día de cada horizonte, que es cuando la muestra es más pequeña.
    """
    _publicar_pedido(proyecto, EMISION, HORIZONTE)
    _publicar_ventas(proyecto, HORIZONTE[:1], 20.0, "2017-07-18")
    _baremo(proyecto, EMISION, costo=0.5)

    _, resumen = seguimiento.medir("2017-07-18")

    assert resumen["alertas"] == []


def test_el_baremo_es_el_del_modelo_que_atendio_la_emision(proyecto: Path):
    """La emisión puede estar atendida por un modelo de hace un mes.

    Comparar contra el reporte de validación de la fecha de la emisión sería
    comparar contra un reporte que no existe, o peor, contra el de otro
    modelo. `emision.json` dice qué versión operó.
    """
    _publicar_pedido(proyecto, EMISION, HORIZONTE)
    _publicar_ventas(proyecto, HORIZONTE, 10.0, "2017-07-25")

    escribir_json(
        {"asof": EMISION, "modelo": "lgbm_cuantilico", "modelo_asof": "2017-07-01"},
        proyecto / "data" / "pronosticos" / f"asof={EMISION}" / seguimiento.ARCHIVO_EMISION,
    )
    _baremo(proyecto, "2017-07-01", costo=7.25)

    _, resumen = seguimiento.medir("2017-07-25")
    fila = _emision(resumen, EMISION)

    assert fila["costo_validado"] == pytest.approx(7.25)
    assert "2017-07-01" in fila["baremo"]


def test_sin_reporte_de_validacion_se_mide_igual_y_se_dice(proyecto: Path):
    """Falta el baremo, no la medición: el costo realizado sigue valiendo."""
    _publicar_pedido(proyecto, EMISION, HORIZONTE)
    _publicar_ventas(proyecto, HORIZONTE, 10.0, "2017-07-25")

    _, resumen = seguimiento.medir("2017-07-25")
    fila = _emision(resumen, EMISION)

    assert fila["costo_validado"] is None
    assert "sin reporte" in fila["baremo"]
    assert fila["costo_realizado"] > 0


def test_la_ventana_limita_cuantas_emisiones_se_miden(proyecto: Path):
    """El costo de la tarea no puede crecer con la antigüedad del producto."""
    for semana in range(6):
        emision = pd.Timestamp("2017-06-05") + pd.Timedelta(weeks=semana)
        fechas = pd.date_range(emision, periods=7, freq="D")
        _publicar_pedido(proyecto, f"{emision:%Y-%m-%d}", fechas)
        _publicar_ventas(proyecto, fechas, 10.0, "2017-07-25")

    cfg = cargar_config("seguimiento") | {"ventana_emisiones": 2}
    _, resumen = seguimiento.medir("2017-07-25", config=cfg)

    assert [f["emision"] for f in resumen["emisiones"]] == ["2017-07-03", "2017-07-10"]


def test_el_resumen_es_json_serializable(proyecto: Path):
    """Lo escribe la CLI como `desempeno.json`; un tipo de numpy lo rompería."""
    _publicar_pedido(proyecto, EMISION, HORIZONTE)
    _publicar_ventas(proyecto, HORIZONTE, 10.0, "2017-07-25")

    _, resumen = seguimiento.medir("2017-07-25")

    json.dumps(resumen)
