"""Pruebas que vigilan los invariantes del proyecto (ver CLAUDE.md).

No prueban una función concreta: prueban que el repositorio en su conjunto no
se desvíe de las reglas que hacen válido al producto. Son baratas y detectan
de inmediato el tipo de error que no se nota revisando un diff.
"""

from __future__ import annotations

from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]

# `src/` contiene la lógica del producto y `dags/` la orquestación. Ambos se
# ejecutan en nombre de una fecha lógica que Airflow inyecta, nunca "ahora".
DIRECTORIOS_DEL_PRODUCTO = ("src", "dags")


def _fuentes_del_producto() -> list[Path]:
    archivos: list[Path] = []
    for directorio in DIRECTORIOS_DEL_PRODUCTO:
        archivos.extend((RAIZ / directorio).rglob("*.py"))
    return archivos


# ---------------------------------------------------------------------------
# INVARIANTE 1 — corrección punto-en-el-tiempo
# ---------------------------------------------------------------------------

# Formas de preguntar "qué hora es" que rompen la reproducibilidad: un backfill
# sobre julio no debe comportarse distinto por ejecutarse hoy.
RELOJES_PROHIBIDOS = (
    "datetime.now(",
    "datetime.today(",
    "date.today(",
    "Timestamp.now(",
    "Timestamp.today(",
    "time.time(",
    "pd.to_datetime('today')",
    'pd.to_datetime("today")',
)


@pytest.mark.parametrize("reloj", RELOJES_PROHIBIDOS)
def test_el_producto_no_consulta_el_reloj_del_sistema(reloj: str):
    """La única fuente de "hoy" es la fecha lógica del run.

    Si el código pregunta la hora al sistema operativo, un backfill del
    2017-07-10 ejecutado hoy produciría un resultado distinto al que produjo
    entonces, y la reproducibilidad exigida por el laboratorio se pierde.
    """
    infractores = [
        f"{ruta.relative_to(RAIZ)}:{numero}"
        for ruta in _fuentes_del_producto()
        for numero, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), 1)
        if reloj in linea
    ]
    assert not infractores, (
        f"`{reloj}` aparece en el código del producto: {infractores}. "
        "Usa la fecha lógica del run (parámetro `asof`), no el reloj del sistema."
    )


# ---------------------------------------------------------------------------
# INVARIANTE 2 — la bandeja es de sólo lectura
# ---------------------------------------------------------------------------

# Llamadas que escribirían en el sistema de origen. El producto sólo lee de
# `landing/`; el único autorizado a escribir es tools/build_landing.py, que no
# forma parte del producto y por eso no se inspecciona aquí.
ESCRITURAS_PROHIBIDAS = (
    "shutil.move",
    "shutil.rmtree",
    "os.remove",
    "os.rename",
    ".unlink(",
    ".to_csv(",
    ".to_parquet(",
)


def test_el_producto_no_escribe_en_la_bandeja():
    """Detecta escrituras cuyo destino menciona `landing`.

    Es una heurística de línea, no un análisis de flujo: busca el caso realista
    (alguien escribe directo a una ruta de landing) sin pretender demostrar la
    ausencia total de escrituras.
    """
    infractores = [
        f"{ruta.relative_to(RAIZ)}:{numero}: {linea.strip()}"
        for ruta in _fuentes_del_producto()
        for numero, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), 1)
        if "landing" in linea.lower()
        and any(escritura in linea for escritura in ESCRITURAS_PROHIBIDAS)
    ]
    assert not infractores, (
        f"El producto intenta escribir en la bandeja: {infractores}. "
        "`landing/` simula un sistema externo: se lee, nunca se modifica."
    )


# ---------------------------------------------------------------------------
# INVARIANTE de arquitectura — sin rutas absolutas
# ---------------------------------------------------------------------------


def test_el_producto_no_usa_rutas_absolutas():
    """El mismo código corre en Windows, en Linux y dentro del contenedor."""
    sospechosas = ("C:\\\\", "C:/", "/home/", "/Users/")
    infractores = [
        f"{ruta.relative_to(RAIZ)}:{numero}: {linea.strip()}"
        for ruta in _fuentes_del_producto()
        for numero, linea in enumerate(ruta.read_text(encoding="utf-8").splitlines(), 1)
        if any(patron in linea for patron in sospechosas)
    ]
    assert not infractores, (
        f"Ruta absoluta en el código del producto: {infractores}. "
        "Resuelve las rutas desde la configuración o el entorno."
    )
