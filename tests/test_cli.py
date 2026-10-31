"""Pruebas de extremo a extremo del pipeline, por sus puntos de entrada reales.

Estas son las pruebas que más valen de todo el proyecto y también las más
lentas, y conviene entender por qué existen aparte de las de cada módulo.

Las pruebas unitarias comprueban que cada función hace lo suyo. No detectan el
error de cableado: una etapa que escribe donde la siguiente no lee, una columna
que se renombra en el camino, un archivo de configuración al que le falta una
clave que el código da por hecha. Ese error sólo aparece recorriendo la cadena
completa, y es exactamente el que rompe un pipeline en producción.

**Corren sin Airflow y sin Docker**, que es un requisito del laboratorio. El
proyecto completo se monta en un directorio temporal: configuración propia,
bandeja sintética de siete meses y almacén vacío. `REPOSICION_HOME` es lo que
hace esto posible, y es la razón por la que no hay ni una ruta absoluta en
`src/`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from reposicion.cli import app
from reposicion.config import limpiar_cache_config

RAIZ_REAL = Path(__file__).resolve().parents[1]

#: Fecha de emisión de la prueba. Fija y sin relación con el reloj: si el
#: resultado dependiera de cuándo se ejecuta la prueba, el producto violaría la
#: INVARIANTE 1 y la prueba lo taparía.
ASOF = "2017-07-17"

TIENDAS = [1, 2]
FAMILIAS = ["GROCERY I", "PRODUCE"]
PRIMER_DIA = "2017-01-01"
CORTE_HISTORICO = pd.Timestamp("2017-06-30")
ULTIMA_VENTA = pd.Timestamp("2017-07-16")

#: Recortes sobre la configuración real para que la cadena completa quepa en
#: una prueba. Se parte de los YAML del proyecto y sólo se reducen los tamaños:
#: así la prueba también comprueba que la configuración de verdad tiene todas
#: las claves que el código espera, que es un fallo que ya ocurrió.
RECORTES = {
    "features": {"ventanas": [7, 14], "semanas_mismo_dia": 4, "semanas_cuantiles": 8},
    "entrenamiento": {"ventana_dias": 120, "minimo_dias_historia": 28},
    "validacion": {"pliegues": 2, "dias_por_pliegue": 7},
}


def _ventas(fechas: pd.DatetimeIndex) -> pd.DataFrame:
    """Ventas con nivel por tienda y estacionalidad semanal.

    La forma importa: con una serie plana todos los modelos empatan y la
    comparación no distingue nada. El patrón por día de la semana es el que de
    verdad domina en retail.
    """
    filas = []
    for fecha in fechas:
        for tienda in TIENDAS:
            for familia in FAMILIAS:
                nivel = 100.0 * tienda + (20.0 if familia == "PRODUCE" else 0.0)
                finde = 1.6 if fecha.day_of_week >= 5 else 1.0
                filas.append(
                    {
                        "date": fecha,
                        "store_nbr": tienda,
                        "family": familia,
                        "sales": round(nivel * finde + fecha.day, 2),
                        "onpromotion": 1 if fecha.day % 10 == 0 else 0,
                    }
                )
    return pd.DataFrame(filas)


def _escribir(df: pd.DataFrame, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False, date_format="%Y-%m-%d")


@pytest.fixture
def proyecto(tmp_path: Path, monkeypatch) -> Path:
    """Un proyecto completo en un directorio temporal."""
    raiz = tmp_path / "proyecto"
    (raiz / "config").mkdir(parents=True)

    for nombre in ("model", "costs", "seguimiento"):
        original = yaml.safe_load((RAIZ_REAL / "config" / f"{nombre}.yaml").read_text("utf-8"))
        if nombre == "model":
            for seccion, valores in RECORTES.items():
                original[seccion].update(valores)
        (raiz / "config" / f"{nombre}.yaml").write_text(
            yaml.safe_dump(original, allow_unicode=True), encoding="utf-8"
        )

    ventas = _ventas(pd.date_range(PRIMER_DIA, ULTIMA_VENTA, freq="D"))
    landing = raiz / "landing"

    _escribir(ventas[ventas["date"] <= CORTE_HISTORICO], landing / "historico" / "sales.csv")
    _escribir(
        pd.DataFrame(
            [
                {"store_nbr": t, "city": c, "state": e, "type": "A", "cluster": 1}
                for t, c, e in ((1, "Quito", "Pichincha"), (2, "Guayaquil", "Guayas"))
            ]
        ),
        landing / "historico" / "stores.csv",
    )
    _escribir(
        pd.DataFrame(
            {"date": pd.date_range(PRIMER_DIA, ULTIMA_VENTA, freq="D"), "dcoilwtico": 48.0}
        ),
        landing / "historico" / "oil.csv",
    )

    # `transactions` existe porque el producto conoce la fuente, y un almacén
    # al que le falta una fuente entera no es un proyecto realista: la
    # auditoría lo detecta como "sin datos" y hace fallar la tarea, que es
    # justo lo que tiene que hacer.
    transacciones = pd.DataFrame(
        [
            {"date": fecha, "store_nbr": tienda, "transactions": 500 + tienda}
            for fecha in pd.date_range(PRIMER_DIA, ULTIMA_VENTA, freq="D")
            for tienda in TIENDAS
        ]
    )
    _escribir(
        transacciones[transacciones["date"] <= CORTE_HISTORICO],
        landing / "historico" / "transactions.csv",
    )
    _escribir(
        pd.DataFrame(
            [
                {
                    "date": fecha,
                    "type": "Holiday",
                    "locale": "National",
                    "locale_name": "Ecuador",
                    "description": descripcion,
                    "transferred": False,
                }
                # El calendario se publica con anticipación, así que su último
                # dato está en el futuro de la emisión. Es lo que hace que la
                # auditoría no lo vea como una fuente atrasada.
                for fecha, descripcion in (
                    ("2017-05-24", "Batalla de Pichincha"),
                    ("2017-08-10", "Primer Grito de Independencia"),
                )
            ]
        ),
        landing / "historico" / "holidays_events.csv",
    )

    # Entregas diarias con latencia D+1: las ventas del día D llegan en D+1.
    posteriores = ventas[ventas["date"] > CORTE_HISTORICO]
    for fecha, bloque in posteriores.groupby("date"):
        llegada = fecha + pd.Timedelta(days=1)
        _escribir(
            bloque,
            landing
            / f"arrival_date={llegada:%Y-%m-%d}"
            / "sales"
            / f"sales_{fecha:%Y-%m-%d}.csv",
        )

    # `transactions` llega un día más tarde que las ventas. La asimetría es
    # deliberada: obliga al producto a tolerar que en la fecha de emisión una
    # fuente esté más atrasada que otra.
    for fecha, bloque in transacciones[transacciones["date"] > CORTE_HISTORICO].groupby("date"):
        llegada = fecha + pd.Timedelta(days=2)
        _escribir(
            bloque,
            landing
            / f"arrival_date={llegada:%Y-%m-%d}"
            / "transactions"
            / f"transactions_{fecha:%Y-%m-%d}.csv",
        )

    monkeypatch.setenv("REPOSICION_HOME", str(raiz))
    limpiar_cache_config()
    yield raiz
    limpiar_cache_config()


def _ejecutar(*orden: str) -> None:
    """Lanza una orden del pipeline y falla con su salida si no termina bien."""
    resultado = CliRunner().invoke(app, list(orden))
    if resultado.exit_code != 0:
        raise AssertionError(
            f"`{' '.join(orden)}` salió con código {resultado.exit_code}\n"
            f"{resultado.output}\n{resultado.exception!r}"
        )


def _huella(ruta: Path) -> str:
    return hashlib.md5(ruta.read_bytes()).hexdigest()


@pytest.fixture
def pipeline_corrido(proyecto: Path) -> Path:
    """Ejecuta la cadena completa una vez y devuelve la raíz del proyecto."""
    _ejecutar("ingesta", "--asof", ASOF)
    _ejecutar("panel", "--asof", ASOF)
    _ejecutar("validar", "--asof", ASOF)
    _ejecutar("modelo", "--asof", ASOF)
    _ejecutar("pronosticar", "--asof", ASOF)
    _ejecutar("pedir", "--asof", ASOF)
    return proyecto


# ---------------------------------------------------------------------------
# LA prueba: la cadena entera produce un pedido
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_la_cadena_completa_produce_un_pedido(pipeline_corrido: Path):
    """De la bandeja al pedido sugerido, por los puntos de entrada reales."""
    pedido = pd.read_parquet(
        pipeline_corrido / "data" / "pedidos" / f"asof={ASOF}" / "pedido.parquet"
    )

    series = len(TIENDAS) * len(FAMILIAS)
    assert len(pedido) == series * 7  # cuatro series por los siete días
    assert (pedido["pedido_sugerido"] >= 0).all()
    assert pedido["date"].min() > ULTIMA_VENTA  # sólo se pronostica el futuro
    assert set(pedido["cuantil_usado"].unique()) <= {0.5, 0.8, 0.9, 0.95}


@pytest.mark.slow
def test_el_pronostico_no_mira_mas_alla_del_origen(pipeline_corrido: Path):
    """La prueba de fuga, a nivel de pipeline.

    El origen es el último día con datos, y con latencia D+1 nunca puede
    alcanzar la fecha de emisión. Si una etapa colara datos posteriores, el
    origen se adelantaría.
    """
    emitido = pd.read_parquet(
        pipeline_corrido / "data" / "pronosticos" / f"asof={ASOF}" / "pronostico.parquet"
    )
    assert emitido["origen"].max() == ULTIMA_VENTA
    assert emitido["emitida"].unique().tolist() == [pd.Timestamp(ASOF)]


@pytest.mark.slow
def test_la_decision_de_promocion_queda_registrada(pipeline_corrido: Path):
    """Quién opera y por qué tiene que ser legible sin ejecutar nada."""
    reportes = pipeline_corrido / "reportes" / f"asof={ASOF}"
    promocion = json.loads((reportes / "promocion.json").read_text("utf-8"))
    metricas = json.loads((reportes / "metricas.json").read_text("utf-8"))

    assert promocion["campeon"] in metricas["modelos"]
    assert promocion["decisiones"], "sin decisiones no se puede auditar la promoción"
    for decision in promocion["decisiones"].values():
        assert decision["razon"]
        assert isinstance(decision["promover"], bool)

    ficha = json.loads(
        (pipeline_corrido / "models" / f"asof={ASOF}" / "metadata.json").read_text("utf-8")
    )
    assert ficha["nombre"] == promocion["campeon"], "opera un modelo distinto del promovido"


@pytest.mark.slow
def test_el_costo_de_negocio_decide_la_promocion(pipeline_corrido: Path):
    """El criterio versionado manda, no el mejor número de la tabla.

    Se comprueba la implicación que de verdad importa: si se promovió a un
    retador, su costo tiene que ser menor que el del vigente. Que gane en
    RMSLE no basta y el proyecto ya tiene un modelo que lo demuestra.
    """
    reportes = pipeline_corrido / "reportes" / f"asof={ASOF}"
    metricas = json.loads((reportes / "metricas.json").read_text("utf-8"))
    campeon, vigente = metricas["campeon"], metricas["vigente"]

    if campeon != vigente:
        costos = {m: v["costo_negocio"] for m, v in metricas["modelos"].items()}
        assert costos[campeon] < costos[vigente]


# ---------------------------------------------------------------------------
# INVARIANTE 3 — idempotencia de punta a punta
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_repetir_la_cadena_da_el_mismo_resultado(pipeline_corrido: Path):
    """Ejecutar dos veces tiene que dar lo mismo, byte a byte.

    Es el requisito que permite reintentar una tarea de Airflow sin pensar y
    el que hace que `dvc repro` signifique algo: si una segunda ejecución
    cambiara el archivo, el hash del lock describiría una de las dos y no se
    sabría cuál.
    """
    rutas = {
        "pronostico": pipeline_corrido
        / "data"
        / "pronosticos"
        / f"asof={ASOF}"
        / "pronostico.parquet",
        "pedido": pipeline_corrido / "data" / "pedidos" / f"asof={ASOF}" / "pedido.parquet",
        "metricas": pipeline_corrido / "reportes" / f"asof={ASOF}" / "metricas.json",
    }
    antes = {nombre: _huella(ruta) for nombre, ruta in rutas.items()}

    _ejecutar("pronosticar", "--asof", ASOF)
    _ejecutar("pedir", "--asof", ASOF)
    _ejecutar("validar", "--asof", ASOF)

    for nombre, ruta in rutas.items():
        assert _huella(ruta) == antes[nombre], f"{nombre} cambió al repetir la ejecución"


@pytest.mark.slow
def test_la_ingesta_no_duplica_al_repetirse(pipeline_corrido: Path):
    """Una partición se reescribe completa; jamás se le agregan filas."""
    from reposicion.ingesta import leer_raw_asof

    antes = len(leer_raw_asof("sales", ASOF))
    _ejecutar("ingesta", "--asof", ASOF)
    assert len(leer_raw_asof("sales", ASOF)) == antes


# ---------------------------------------------------------------------------
# INVARIANTE 1 — la fecha entra desde fuera
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "orden",
    ["ingesta", "auditar", "panel", "validar", "modelo", "pronosticar", "pedir", "seguir"],
)
def test_ninguna_orden_funciona_sin_asof(orden: str, proyecto: Path):
    """Sin fecha lógica no se ejecuta nada.

    Si alguna orden tuviera un valor por defecto tomado del reloj, un backfill
    de julio lanzado hoy produciría el pronóstico de hoy sin avisar. Mejor que
    falle.
    """
    assert CliRunner().invoke(app, [orden]).exit_code != 0


def test_una_fecha_sin_datos_falla_con_un_mensaje_util(proyecto: Path):
    """Pedir una emisión anterior a cualquier dato no debe reventar con un
    rastro de pandas, sino decir qué falta."""
    _ejecutar("ingesta", "--asof", "2017-01-05")
    resultado = CliRunner().invoke(app, ["panel", "--asof", "2016-01-01"])

    assert resultado.exit_code != 0
    assert "ingesta" in resultado.output.lower() or "ingesta" in str(resultado.exception).lower()


# ---------------------------------------------------------------------------
# El respaldo de la INVARIANTE 5
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_una_serie_sin_historia_la_atiende_el_respaldo(pipeline_corrido: Path):
    """"Tienda nueva" es una situación de negocio: se pronostica igual.

    Se añade una serie con tres días de venta —menos del mínimo exigido— y se
    comprueba que recibe pronóstico y que la columna `modelo` dice que lo
    produjo el modelo de referencia. Sin esa columna, una serie atendida por
    el respaldo sería indistinguible y el seguimiento mezclaría dos cosas.
    """
    nueva = pd.DataFrame(
        {
            "date": pd.date_range("2017-07-14", "2017-07-16", freq="D"),
            "store_nbr": 9,
            "family": "GROCERY I",
            "sales": 5.0,
            "onpromotion": 0,
        }
    )
    llegada = pipeline_corrido / "landing" / "arrival_date=2017-07-17" / "sales"
    _escribir(nueva, llegada / "sales_nueva.csv")
    _escribir(
        pd.DataFrame(
            [{"store_nbr": 9, "city": "Cuenca", "state": "Azuay", "type": "C", "cluster": 3}]
        ),
        pipeline_corrido / "landing" / "arrival_date=2017-07-17" / "stores" / "stores.csv",
    )

    _ejecutar("ingesta", "--asof", ASOF)
    _ejecutar("pronosticar", "--asof", ASOF)

    emitido = pd.read_parquet(
        pipeline_corrido / "data" / "pronosticos" / f"asof={ASOF}" / "pronostico.parquet"
    )
    serie = emitido[emitido["store_nbr"] == 9]

    assert not serie.empty, "la tienda nueva se quedó sin pronóstico"
    assert serie["modelo"].unique().tolist() == ["baseline_estacional"]
    assert (serie["q0.9"] >= 0).all()


# ---------------------------------------------------------------------------
# Requerimiento 3 — operación: auditoría, modelo vigente y seguimiento
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_la_auditoria_registra_el_estado_de_cada_fuente(pipeline_corrido: Path):
    """Con latencia D+1, las ventas van un día por detrás y eso es lo normal.

    La auditoría tiene que decirlo sin alarmarse: es la tarea que distingue un
    retraso esperado de un flujo parado, y equivocarse hacia el lado estricto
    haría caer el DAG todos los días.
    """
    _ejecutar("auditar", "--asof", ASOF)

    reporte = pd.read_csv(
        pipeline_corrido / "data" / "seguimiento" / f"asof={ASOF}" / "calidad.csv"
    )
    ventas = reporte.set_index("fuente").loc["sales"]

    assert ventas["estado"] == "al dia"
    assert ventas["ultimo_dato"] == f"{ULTIMA_VENTA:%Y-%m-%d}"
    assert ventas["retraso_dias"] == 1


@pytest.mark.slow
def test_la_emision_usa_el_modelo_vigente_aunque_no_sea_de_su_fecha(pipeline_corrido: Path):
    """Se emite la semana siguiente sin reentrenar: opera el modelo anterior.

    Es la situación normal en operación —se emite cada lunes y se reentrena
    una vez al mes— y la que haría fallar a un producto que exigiera un modelo
    de la fecha exacta de la emisión.
    """
    siguiente = "2017-07-24"
    _ejecutar("pronosticar", "--asof", siguiente)

    emision = json.loads(
        (
            pipeline_corrido / "data" / "pronosticos" / f"asof={siguiente}" / "emision.json"
        ).read_text("utf-8")
    )

    assert not (pipeline_corrido / "models" / f"asof={siguiente}").exists()
    assert emision["modelo_asof"] == ASOF, "no usó el último modelo registrado"


@pytest.mark.slow
def test_el_seguimiento_califica_la_emision_de_la_semana_pasada(pipeline_corrido: Path):
    """El lazo completo: se publicó un pedido, llegaron las ventas, se mide.

    Es el requerimiento 3 del laboratorio de punta a punta, y la única prueba
    que recorre la cadena en el orden en que la recorre el negocio: emitir,
    esperar, comparar.
    """
    realizadas = _ventas(pd.date_range(ASOF, periods=7, freq="D"))
    for fecha, bloque in realizadas.groupby("date"):
        llegada = fecha + pd.Timedelta(days=1)
        _escribir(
            bloque,
            pipeline_corrido
            / "landing"
            / f"arrival_date={llegada:%Y-%m-%d}"
            / "sales"
            / f"sales_{fecha:%Y-%m-%d}.csv",
        )

    medido = "2017-07-24"
    _ejecutar("ingesta", "--asof", medido)
    _ejecutar("seguir", "--asof", medido)

    salida = pipeline_corrido / "data" / "seguimiento" / f"asof={medido}"
    resumen = json.loads((salida / "desempeno.json").read_text("utf-8"))
    fila = next(f for f in resumen["emisiones"] if f["emision"] == ASOF)

    series = len(TIENDAS) * len(FAMILIAS)
    assert fila["lineas"] == series * 7, "el horizonte completo tenía que estar realizado"
    assert fila["realizado"] == 1.0
    assert fila["costo_realizado"] > 0
    assert fila["costo_validado"] is not None, "el baremo sale del reporte del modelo que operó"

    detalle = pd.read_csv(salida / "desempeno.csv")
    assert set(detalle["family"]) == set(FAMILIAS), "el detalle se desagrega por familia"
