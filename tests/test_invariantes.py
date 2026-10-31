"""Pruebas que vigilan los invariantes del proyecto (ver CLAUDE.md).

No prueban una función concreta: prueban que el repositorio en su conjunto no
se desvíe de las reglas que hacen válido al producto. Son baratas y detectan
de inmediato el tipo de error que no se nota revisando un diff.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

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
# El código fuente tiene que estar en el repositorio
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("git") is None, reason="requiere git")
def test_ningun_archivo_de_codigo_esta_excluido_por_gitignore():
    """Regresión: un patrón sin anclar en .gitignore borró un paquete entero.

    La regla `models/`, pensada para los modelos entrenados de la raíz,
    coincide a cualquier profundidad y excluía también
    `src/reposicion/models/`. El paquete desapareció del repositorio sin aviso:
    en la máquina de desarrollo todo seguía funcionando porque los archivos
    estaban en disco, y sólo un clon limpio podía detectarlo.

    Se comprueba la exclusión y no el seguimiento. Un archivo recién creado y
    todavía sin añadir es normal mientras se trabaja; uno **ignorado** nunca
    lo es, porque jamás llegará al repositorio aunque se intente añadirlo.
    """
    archivos = [
        ruta
        for directorio in ("src", "dags", "tests")
        for ruta in (RAIZ / directorio).rglob("*.py")
        if "__pycache__" not in ruta.parts
    ]
    if not archivos:
        pytest.skip("no hay código que comprobar")

    # `check-ignore` devuelve en stdout sólo las rutas que están ignoradas.
    resultado = subprocess.run(
        ["git", "check-ignore", "--no-index", *[str(p) for p in archivos]],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        check=False,
    )
    # Código 0: hay ignorados. 1: no hay ninguno. 128: no es un repositorio.
    if resultado.returncode == 128:
        pytest.skip("el directorio no es un repositorio git")

    ignorados = sorted(linea for linea in resultado.stdout.splitlines() if linea)
    assert not ignorados, (
        f"Código fuente excluido por .gitignore: {ignorados}. "
        "Un patrón sin barra inicial coincide a cualquier nivel: "
        "`models/` también afecta a `src/reposicion/models/`."
    )


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
    # `/opt/` entra en la lista por la fase de Airflow: la raíz dentro del
    # contenedor es `/opt/proyecto`, y escribirla en un DAG —aunque sea como
    # valor por defecto— crearía una segunda fuente de verdad que ganaría el
    # día que alguien cambiara el volumen del compose. La ruta se exporta una
    # sola vez, como `REPOSICION_HOME`.
    sospechosas = ("C:\\\\", "C:/", "/home/", "/Users/", "/opt/")
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


# ---------------------------------------------------------------------------
# Requerimiento 2 del laboratorio — ni datos ni modelos en Git
# ---------------------------------------------------------------------------

#: Lo único que estas carpetas pueden aportar al repositorio: los punteros de
#: DVC y los `.gitignore` que DVC genera. Todo lo demás son datos o modelos.
RASTREABLES_EN_ZONAS_DE_DATOS = (".dvc", ".gitignore")

ZONAS_DE_DATOS = ("data", "landing", "models", "docker/logs")


@pytest.mark.skipif(shutil.which("git") is None, reason="requiere git")
def test_no_hay_datos_ni_modelos_en_el_repositorio():
    """El enunciado lo exige y es fácil romperlo sin darse cuenta.

    Basta un `git add -A` antes de que DVC haya registrado una salida nueva
    para meter 118 MB de CSV en el historial, y una vez dentro no se van: hay
    que reescribir el historial de una rama protegida. La prueba es la red
    que hace que el error se detecte en el primer `pytest` y no en el PR.

    Se mira lo que Git **rastrea**, no lo que hay en disco. Que los archivos
    existan en la máquina de desarrollo es lo normal; lo que no puede pasar es
    que estén versionados.
    """
    resultado = subprocess.run(
        ["git", "ls-files", "--", *ZONAS_DE_DATOS],
        cwd=RAIZ,
        capture_output=True,
        text=True,
        check=False,
    )
    if resultado.returncode != 0:
        pytest.skip("el directorio no es un repositorio git")

    intrusos = [
        linea
        for linea in resultado.stdout.splitlines()
        if linea and not linea.endswith(RASTREABLES_EN_ZONAS_DE_DATOS)
    ]
    assert not intrusos, (
        f"Hay datos o modelos versionados en Git: {intrusos}. "
        "Se versionan con DVC: `dvc add` o como salida de una etapa de dvc.yaml."
    )


# ---------------------------------------------------------------------------
# El lock de DVC no puede mentir
# ---------------------------------------------------------------------------


def test_toda_etapa_del_pipeline_depende_del_codigo_del_producto():
    """Una etapa que no declara el código como dependencia produce un lock falso.

    DVC daría la salida por actualizada después de cambiar una función, y el
    hash registrado describiría un artefacto que ya no corresponde al código
    que dice haberlo producido. Eso anula el motivo de usar DVC, así que la
    dependencia gruesa sobre `src/reposicion` es obligatoria y se comprueba.

    La etapa del simulador queda fuera a propósito: no ejecuta el producto
    sino `tools/build_landing.py`, que declara por separado.
    """
    pipeline = yaml.safe_load((RAIZ / "dvc.yaml").read_text(encoding="utf-8"))
    sin_codigo = [
        nombre
        for nombre, etapa in pipeline["stages"].items()
        if "reposicion" in etapa["cmd"] and "src/reposicion" not in (etapa.get("deps") or [])
    ]
    assert not sin_codigo, (
        f"Etapas que ejecutan el producto sin declararlo como dependencia: {sin_codigo}. "
        "Añade `src/reposicion` a sus `deps` o el lock dejará de decir la verdad."
    )


def test_ninguna_etapa_consulta_la_fecha_del_sistema():
    """Toda orden del pipeline recibe `--asof` de forma explícita.

    Si una etapa lo omitiera, tomaría la fecha por defecto —si existiera— y
    `dvc repro` sobre un commit antiguo produciría el pronóstico de hoy con el
    código de entonces: lo peor de los dos mundos.
    """
    pipeline = yaml.safe_load((RAIZ / "dvc.yaml").read_text(encoding="utf-8"))
    sin_fecha = [
        nombre
        for nombre, etapa in pipeline["stages"].items()
        if "reposicion" in etapa["cmd"] and "--asof" not in etapa["cmd"]
    ]
    assert not sin_fecha, f"Etapas sin fecha logica explicita: {sin_fecha}."


def test_la_ruta_del_contenedor_se_declara_una_sola_vez():
    """El compose es el único sitio que sabe dónde se monta el repositorio.

    Contrapartida de la prueba anterior: si `/opt/` no puede aparecer en el
    código, alguien tiene que decir la ruta, y ese alguien es el archivo que
    crea el volumen. Sin esta comprobación, prohibirla en el código sólo
    garantizaría que no está escrita en ningún sitio.
    """
    compose = (RAIZ / "docker" / "docker-compose.yml").read_text(encoding="utf-8")

    assert "REPOSICION_HOME: /opt/proyecto" in compose, (
        "el compose tiene que exportar REPOSICION_HOME: es como el producto "
        "averigua dónde está montado sin llevar la ruta dentro"
    )
    assert "..:/opt/proyecto" in compose, "falta el volumen con el repositorio"
