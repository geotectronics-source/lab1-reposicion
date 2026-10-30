"""Pruebas de la ingesta y de la escritura particionada.

Lo que se verifica aquí es sobre todo *operativo*: que repetir una tarea no
cambie el resultado, que el orden de un backfill sea irrelevante y que un
archivo malo se detenga antes de entrar al almacén.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from reposicion.ingesta import (
    FUENTES,
    IngestaRechazada,
    ingerir,
    ingerir_historico,
    leer_raw_asof,
    leer_raw_vigente,
)
from reposicion.io import (
    DTYPE_FECHA,
    a_fecha,
    escribir_particion,
    listar_particiones,
    marcar_llegada,
    normalizar_fecha,
    ruta_particion,
)
from reposicion.landing import ARRIVAL_HISTORICO


@pytest.fixture
def almacen(tmp_path: Path) -> dict[str, Path]:
    return {"raw": tmp_path / "raw", "cuarentena": tmp_path / "cuarentena"}


def _ingerir_todo(bandeja: Path, almacen: dict[str, Path], dias: list[str]) -> None:
    ingerir_historico("sales", landing=bandeja, raw=almacen["raw"])
    for dia in dias:
        ingerir("sales", dia, landing=bandeja, raw=almacen["raw"],
                cuarentena=almacen["cuarentena"])


def _ventas_tienda44(df: pd.DataFrame) -> float:
    sel = df[(df["date"] == pd.Timestamp("2017-07-05")) & (df["store_nbr"] == 44)]
    return float(sel["sales"].sum())


# ---------------------------------------------------------------------------
# Ingesta básica
# ---------------------------------------------------------------------------


def test_materializa_la_particion_de_la_fecha(bandeja: Path, almacen: dict[str, Path]):
    resultado = ingerir("sales", "2017-07-06", landing=bandeja, **almacen)

    assert resultado.hubo_entrega
    assert resultado.filas == 4
    assert resultado.particion is not None and resultado.particion.exists()
    assert resultado.particion.parent.name == "arrival_date=2017-07-06"


def test_solo_procesa_su_propia_fecha(bandeja: Path, almacen: dict[str, Path]):
    """El run del día D no debe arrastrar lo de días anteriores."""
    ingerir("sales", "2017-07-06", landing=bandeja, **almacen)
    ingerir("sales", "2017-07-07", landing=bandeja, **almacen)

    particiones = listar_particiones(almacen["raw"], "sales")
    assert [p[0] for p in particiones] == [
        pd.Timestamp("2017-07-06"),
        pd.Timestamp("2017-07-07"),
    ]


def test_varios_archivos_del_mismo_dia_van_a_una_particion(
    bandeja: Path, almacen: dict[str, Path]
):
    """El 2017-07-20 llegan la entrega del día y una corrección atrasada."""
    extra = bandeja / "arrival_date=2017-07-20" / "sales"
    pd.DataFrame(
        {
            "date": ["2017-07-19"],
            "store_nbr": [1],
            "family": ["GROCERY I"],
            "sales": [5.0],
            "onpromotion": [0],
        }
    ).to_csv(extra / "sales_2017-07-19.csv", index=False)

    resultado = ingerir("sales", "2017-07-20", landing=bandeja, **almacen)
    assert resultado.archivos == 2
    assert resultado.filas == 5


def test_el_historico_es_la_particion_mas_antigua(bandeja: Path, almacen: dict[str, Path]):
    """Así el resto del producto no necesita un camino especial para leerlo."""
    resultado = ingerir_historico("sales", landing=bandeja, raw=almacen["raw"])

    assert resultado.arrival_date == ARRIVAL_HISTORICO
    particiones = listar_particiones(almacen["raw"], "sales")
    assert particiones[0][0] == ARRIVAL_HISTORICO


# ---------------------------------------------------------------------------
# INVARIANTE 3 — idempotencia
# ---------------------------------------------------------------------------


def test_repetir_la_tarea_no_cambia_nada(bandeja: Path, almacen: dict[str, Path]):
    """Ejecutar N veces debe dar el mismo resultado que ejecutar una."""
    primera = ingerir("sales", "2017-07-06", landing=bandeja, **almacen)
    hash_inicial = hashlib.sha256(primera.particion.read_bytes()).hexdigest()

    for _ in range(3):
        repetida = ingerir("sales", "2017-07-06", landing=bandeja, **almacen)
        assert repetida.filas == primera.filas
        assert hashlib.sha256(repetida.particion.read_bytes()).hexdigest() == hash_inicial


def test_la_particion_no_acumula_archivos(bandeja: Path, almacen: dict[str, Path]):
    """Se reescribe completa: ni se agrega ni quedan temporales huérfanos."""
    resultado = ingerir("sales", "2017-07-06", landing=bandeja, **almacen)
    ingerir("sales", "2017-07-06", landing=bandeja, **almacen)

    assert len(list(resultado.particion.parent.iterdir())) == 1


def test_reingerir_no_duplica_filas(bandeja: Path, almacen: dict[str, Path]):
    _ingerir_todo(bandeja, almacen, ["2017-07-06", "2017-07-07", "2017-07-20"])
    antes = len(leer_raw_asof("sales", "2017-07-25", raw=almacen["raw"]))

    ingerir("sales", "2017-07-20", landing=bandeja, **almacen)
    assert len(leer_raw_asof("sales", "2017-07-25", raw=almacen["raw"])) == antes


def test_el_orden_del_backfill_es_irrelevante(bandeja: Path, almacen: dict[str, Path],
                                              tmp_path: Path):
    """Un backfill hacia atrás debe producir lo mismo que uno hacia adelante."""
    adelante = {"raw": tmp_path / "a_raw", "cuarentena": tmp_path / "a_cuar"}
    atras = {"raw": tmp_path / "b_raw", "cuarentena": tmp_path / "b_cuar"}
    dias = ["2017-07-06", "2017-07-07", "2017-07-20"]

    for dia in dias:
        ingerir("sales", dia, landing=bandeja, **adelante)
    for dia in reversed(dias):
        ingerir("sales", dia, landing=bandeja, **atras)

    izq = leer_raw_vigente("sales", "2017-07-25", raw=adelante["raw"])
    der = leer_raw_vigente("sales", "2017-07-25", raw=atras["raw"])
    pd.testing.assert_frame_equal(izq, der)


# ---------------------------------------------------------------------------
# Validación y cuarentena
# ---------------------------------------------------------------------------


def _archivo_invalido(bandeja: Path, nombre: str = "sales_corrupto.csv") -> Path:
    destino = bandeja / "arrival_date=2017-07-06" / "sales" / nombre
    pd.DataFrame(
        {
            "date": ["2017-07-05"],
            "store_nbr": [2],
            "family": ["PRODUCE"],
            "sales": [-5.0],  # venta negativa: valor imposible
            "onpromotion": [0],
        }
    ).to_csv(destino, index=False)
    return destino


def test_un_archivo_invalido_detiene_la_ingesta(bandeja: Path, almacen: dict[str, Path]):
    _archivo_invalido(bandeja)
    with pytest.raises(IngestaRechazada):
        ingerir("sales", "2017-07-06", landing=bandeja, **almacen)


def test_la_particion_es_todo_o_nada(bandeja: Path, almacen: dict[str, Path]):
    """Con un archivo malo no se escribe NINGUNO, ni siquiera los buenos.

    Una partición parcial parece completa y no lo es: contamina en silencio,
    que es peor que fallar.
    """
    _archivo_invalido(bandeja)
    with pytest.raises(IngestaRechazada):
        ingerir("sales", "2017-07-06", landing=bandeja, **almacen)

    assert not (almacen["raw"] / "sales").exists()


def test_el_archivo_malo_queda_con_su_motivo(bandeja: Path, almacen: dict[str, Path]):
    _archivo_invalido(bandeja)
    with pytest.raises(IngestaRechazada) as exc:
        ingerir("sales", "2017-07-06", landing=bandeja, **almacen)

    copia = exc.value.cuarentena[0]
    assert copia.exists()
    motivo = copia.with_suffix(copia.suffix + ".motivo.txt")
    assert motivo.exists() and "sales" in motivo.read_text(encoding="utf-8")


def test_la_cuarentena_no_toca_la_bandeja(bandeja: Path, almacen: dict[str, Path]):
    """INVARIANTE 2: el archivo se copia, el original sigue en su sitio."""
    original = _archivo_invalido(bandeja)
    with pytest.raises(IngestaRechazada):
        ingerir("sales", "2017-07-06", landing=bandeja, **almacen)

    assert original.exists()


def test_tras_corregir_el_archivo_la_ingesta_pasa(bandeja: Path, almacen: dict[str, Path]):
    """El reproceso de datos corregidos no requiere limpiar nada a mano."""
    malo = _archivo_invalido(bandeja)
    with pytest.raises(IngestaRechazada):
        ingerir("sales", "2017-07-06", landing=bandeja, **almacen)

    malo.unlink()  # el sistema origen retira el archivo defectuoso
    resultado = ingerir("sales", "2017-07-06", landing=bandeja, **almacen)
    assert resultado.hubo_entrega


# ---------------------------------------------------------------------------
# INVARIANTE 5 — ausencias que no son errores
# ---------------------------------------------------------------------------


def test_una_fuente_que_no_llego_no_falla(bandeja: Path, almacen: dict[str, Path]):
    resultado = ingerir("transactions", "2017-07-06", landing=bandeja, **almacen)

    assert not resultado.hubo_entrega
    assert resultado.filas == 0
    assert "sin entrega" in str(resultado)


def test_un_dia_sin_ninguna_entrega_no_falla(bandeja: Path, almacen: dict[str, Path]):
    resultado = ingerir("sales", "2017-07-25", landing=bandeja, **almacen)
    assert not resultado.hubo_entrega


def test_historico_inexistente_no_falla(tmp_path: Path):
    resultado = ingerir_historico("sales", landing=tmp_path, raw=tmp_path / "raw")
    assert not resultado.hubo_entrega


# ---------------------------------------------------------------------------
# Lectura del almacén
# ---------------------------------------------------------------------------


def test_leer_raw_respeta_el_asof(bandeja: Path, almacen: dict[str, Path]):
    """La misma garantía punto-en-el-tiempo, ahora sobre el almacén propio."""
    _ingerir_todo(bandeja, almacen, ["2017-07-06", "2017-07-07", "2017-07-20"])

    assert _ventas_tienda44(leer_raw_vigente("sales", "2017-07-15", raw=almacen["raw"])) == 0.0
    assert _ventas_tienda44(leer_raw_vigente("sales", "2017-07-25", raw=almacen["raw"])) == 880.0


def test_leer_raw_preserva_los_tipos(bandeja: Path, almacen: dict[str, Path]):
    """Es la ventaja de Parquet sobre CSV: no hay que re-inferir nada."""
    _ingerir_todo(bandeja, almacen, ["2017-07-06"])
    df = leer_raw_asof("sales", "2017-07-25", raw=almacen["raw"])

    assert df["date"].dtype == "datetime64[ns]"
    assert df["arrival_date"].dtype == "datetime64[ns]"
    assert df["sales"].dtype == float


def test_almacen_vacio_devuelve_vacio(tmp_path: Path):
    assert leer_raw_asof("sales", "2017-07-25", raw=tmp_path).empty


# ---------------------------------------------------------------------------
# Utilidades de io
# ---------------------------------------------------------------------------


def test_escritura_atomica_no_deja_temporales(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2]})
    destino = tmp_path / "particion"
    escribir_particion(df, destino)

    assert [p.name for p in destino.iterdir()] == ["datos.parquet"]


def test_listar_particiones_filtra_por_fecha(tmp_path: Path):
    for dia in ["2017-07-05", "2017-07-06", "2017-07-20"]:
        escribir_particion(pd.DataFrame({"a": [1]}), ruta_particion(tmp_path, "sales", dia))

    hasta = listar_particiones(tmp_path, "sales", hasta="2017-07-10")
    assert [p[0] for p in hasta] == [pd.Timestamp("2017-07-05"), pd.Timestamp("2017-07-06")]


def test_listar_particiones_ignora_carpetas_ajenas(tmp_path: Path):
    escribir_particion(pd.DataFrame({"a": [1]}), ruta_particion(tmp_path, "sales", "2017-07-05"))
    (tmp_path / "sales" / "_temporal").mkdir()

    assert len(listar_particiones(tmp_path, "sales")) == 1


def test_marcar_llegada_fija_la_resolucion_temporal():
    """Regresión: asignar un Timestamp escalar deja la columna en microsegundos.

    pandas 3 infiere `[us]` tanto al parsear como al construir desde un
    escalar, mientras que el contrato y Parquet esperan `[ns]`. Un dtype
    distinto no lanza error: sólo hace que las comparaciones dejen de
    coincidir, y el síntoma es un resultado vacío en vez de una excepción.
    Apareció tres veces en sitios distintos antes de centralizarse aquí.
    """
    df = marcar_llegada(pd.DataFrame({"a": [1, 2]}), "2017-07-06")
    assert df["arrival_date"].dtype == DTYPE_FECHA
    assert (df["arrival_date"] == pd.Timestamp("2017-07-06")).all()


def test_a_fecha_fija_la_resolucion_temporal():
    assert a_fecha(pd.Series(["2017-07-05"])).dtype == DTYPE_FECHA
    assert a_fecha(pd.Series([pd.Timestamp("2017-07-05 10:00")])).dtype == DTYPE_FECHA


def test_normalizar_fecha_quita_hora_y_zona():
    assert normalizar_fecha("2017-07-05 18:30:00") == pd.Timestamp("2017-07-05")
    con_zona = pd.Timestamp("2017-07-05 23:00:00", tz="UTC")
    assert normalizar_fecha(con_zona) == pd.Timestamp("2017-07-05")


# ---------------------------------------------------------------------------
# Coherencia
# ---------------------------------------------------------------------------


def test_las_fuentes_ingeribles_son_las_que_tienen_contrato():
    """Evita que una fuente entre al almacén sin validarse."""
    assert set(FUENTES) == {"sales", "transactions", "oil", "holidays_events", "stores"}
