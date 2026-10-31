"""Pruebas de los DAGs **sin instalar Airflow**.

El laboratorio exige que `pytest` corra sin Airflow y sin Docker (CLAUDE.md
§6), de modo que aquí no se importa ningún DAG: se leen como texto y se
analizan con `ast`. Lo que se comprueba no es que Airflow los acepte —de eso
se encarga `airflow dags list`, que corre en el contenedor— sino las
propiedades que convierten un archivo de `dags/` en orquestación y no en
producto escondido.

**Por qué merece la pena una prueba estática.** "Los DAGs sólo orquestan" es
una regla que se respeta el primer día y se erosiona el tercero: alguien
necesita un `if` para saltarse un domingo, luego una función para calcular una
fecha, y al mes la lógica de negocio vive en dos sitios y sólo uno tiene
pruebas. Un árbol de sintaxis detecta esa deriva en el momento en que ocurre.

La contrapartida honesta: estas pruebas no ejecutan nada. Un DAG puede pasar
las ocho y seguir fallando al importarse en Airflow. La verificación real es
`airflow dags list` dentro del contenedor, documentada en el README.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
CARPETA = RAIZ / "dags"

#: Módulo de ayuda compartido: no define DAGs y sí define una función, así que
#: queda fuera de las comprobaciones que aplican a los archivos de DAG.
AYUDA = "comun.py"


def _archivos_de_dag() -> list[Path]:
    return sorted(p for p in CARPETA.glob("*.py") if p.name != AYUDA)


def _arbol(ruta: Path) -> ast.Module:
    return ast.parse(ruta.read_text(encoding="utf-8"), filename=str(ruta))


def _llamada_dag(arbol: ast.Module) -> ast.Call:
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Call) and getattr(nodo.func, "id", "") == "DAG":
            return nodo
    raise AssertionError("el archivo no construye ningún DAG")


def _argumentos(llamada: ast.Call) -> dict[str, ast.expr]:
    return {kw.arg: kw.value for kw in llamada.keywords if kw.arg}


def _texto(nodo: ast.expr | None) -> object:
    return nodo.value if isinstance(nodo, ast.Constant) else None


IDS = [p.stem for p in _archivos_de_dag()]


def test_hay_dags():
    """Si la carpeta se vacía, el resto de las pruebas pasarían por vacuidad."""
    assert _archivos_de_dag(), "no hay ningún DAG en dags/"


@pytest.mark.parametrize("archivo", _archivos_de_dag(), ids=IDS)
def test_el_dag_se_analiza_sin_errores(archivo: Path):
    """Un DAG con un error de sintaxis no llega ni a aparecer en la interfaz."""
    _arbol(archivo)


# ---------------------------------------------------------------------------
# La regla central: los DAGs sólo orquestan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("archivo", _archivos_de_dag(), ids=IDS)
def test_ningun_dag_importa_el_producto(archivo: Path):
    """La frontera entre orquestación y producto es la línea de comando.

    Si un DAG importara `reposicion`, habría una segunda forma de ejecutar el
    producto —la que usa Airflow— distinta de la que reproduce DVC, y las dos
    podrían separarse sin que nada avisara. Es la prueba que sostiene la
    decisión de usar `BashOperator` en vez de `PythonOperator`.
    """
    importados = {
        alias.name.split(".")[0]
        for nodo in ast.walk(_arbol(archivo))
        if isinstance(nodo, ast.Import)
        for alias in nodo.names
    } | {
        (nodo.module or "").split(".")[0]
        for nodo in ast.walk(_arbol(archivo))
        if isinstance(nodo, ast.ImportFrom)
    }
    assert "reposicion" not in importados, (
        f"{archivo.name} importa el producto. La lógica se invoca por su línea de "
        "comando, que es la misma que ejecuta dvc.yaml."
    )


@pytest.mark.parametrize("archivo", _archivos_de_dag(), ids=IDS)
def test_ningun_dag_contiene_logica(archivo: Path):
    """Ni funciones, ni condicionales, ni bucles, ni manejo de excepciones.

    No es purismo: cualquiera de esas cuatro cosas en un archivo de `dags/` es
    una decisión que se toma fuera de `src/`, donde no hay pruebas unitarias
    que la cubran ni DVC que registre de qué depende.
    """
    prohibidos = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.If,
        ast.For,
        ast.While,
        ast.Try,
        ast.ClassDef,
    )
    encontrados = sorted(
        {type(nodo).__name__ for nodo in ast.walk(_arbol(archivo)) if isinstance(nodo, prohibidos)}
    )
    assert not encontrados, (
        f"{archivo.name} contiene lógica ({encontrados}). Esa decisión va en "
        "src/reposicion/, donde se puede probar."
    )


@pytest.mark.parametrize("archivo", _archivos_de_dag(), ids=IDS)
def test_ningun_dag_construye_operadores_a_mano(archivo: Path):
    """Todas las tareas pasan por `comun.etapa`.

    Un `BashOperator` escrito a mano en un DAG es la puerta por la que entra
    una orden con la fecha fija, otro directorio de trabajo o una variable de
    entorno distinta. Con una sola fábrica, esas decisiones están en un sitio.
    """
    llamadas = {
        getattr(nodo.func, "id", "")
        for nodo in ast.walk(_arbol(archivo))
        if isinstance(nodo, ast.Call)
    }
    assert "BashOperator" not in llamadas, f"{archivo.name} crea un operador sin pasar por etapa()"
    assert "etapa" in llamadas, f"{archivo.name} no declara ninguna tarea"


# ---------------------------------------------------------------------------
# INVARIANTE 1 — la fecha entra desde fuera, también en la orquestación
# ---------------------------------------------------------------------------


def test_la_fabrica_de_tareas_usa_la_fecha_logica():
    """`comun.etapa` tiene que pasar `--asof {{ ds }}` y nada más.

    Es el único sitio donde se construye la orden, así que basta comprobarlo
    aquí para saber que ninguna tarea del proyecto puede ejecutarse con una
    fecha que no venga de Airflow.
    """
    texto = (CARPETA / AYUDA).read_text(encoding="utf-8")
    assert 'FECHA_LOGICA = "{{ ds }}"' in texto, (
        "la fábrica de tareas no declara la plantilla de la fecha lógica"
    )
    assert "FECHA_LOGICA" in texto.split("def etapa")[1], (
        "etapa() no usa la plantilla de la fecha lógica"
    )


@pytest.mark.parametrize("archivo", sorted(CARPETA.glob("*.py")), ids=lambda p: p.stem)
def test_ningun_dag_fija_una_fecha_en_una_orden(archivo: Path):
    """Una fecha escrita en un `bash_command` convertiría el backfill en una
    mentira: todos los runs producirían la emisión del mismo día."""
    for numero, linea in enumerate(archivo.read_text(encoding="utf-8").splitlines(), 1):
        assert "--asof 2" not in linea, f"{archivo.name}:{numero} fija la fecha de la emisión"


# ---------------------------------------------------------------------------
# Programación: lo que decide qué corre y cuándo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("archivo", _archivos_de_dag(), ids=IDS)
def test_cada_dag_declara_su_programacion_completa(archivo: Path):
    """`schedule`, `start_date` y `catchup` explícitos, y el id igual al archivo.

    Dejar `catchup` al valor por defecto (que es `True`) es el error caro de
    Airflow: habilitar un DAG diario con fecha de inicio en 2017 pone al
    planificador a crear miles de corridas que nadie pidió.
    """
    argumentos = _argumentos(_llamada_dag(_arbol(archivo)))

    assert _texto(argumentos.get("dag_id")) == archivo.stem, (
        "el dag_id tiene que coincidir con el nombre del archivo: es lo que "
        "permite ir del log a la fuente sin buscar"
    )
    for obligatorio in ("schedule", "start_date", "catchup", "max_active_runs", "doc_md"):
        assert obligatorio in argumentos, f"{archivo.name} no declara {obligatorio}"

    assert _texto(argumentos["catchup"]) is False, f"{archivo.name} dejaría catchup en True"
    assert _texto(argumentos["max_active_runs"]) == 1


@pytest.mark.parametrize("archivo", _archivos_de_dag(), ids=IDS)
def test_cada_dag_se_explica_en_la_interfaz(archivo: Path):
    """El docstring del módulo es lo que se lee en la pestaña de documentación.

    Quien mire un DAG rojo a las seis de la mañana no va a abrir el
    repositorio. `doc_md=__doc__` evita que la explicación y el código se
    separen.
    """
    arbol = _arbol(archivo)
    assert ast.get_docstring(arbol), f"{archivo.name} no tiene docstring de módulo"

    doc = _argumentos(_llamada_dag(arbol))["doc_md"]
    assert getattr(doc, "id", "") == "__doc__", (
        f"{archivo.name} documenta la interfaz con un texto aparte del docstring"
    )


def test_los_dag_id_no_se_repiten():
    """Dos DAGs con el mismo id: Airflow carga uno y descarta el otro en
    silencio, y el que falta no aparece como error en ninguna parte."""
    ids = [_texto(_argumentos(_llamada_dag(_arbol(p))).get("dag_id")) for p in _archivos_de_dag()]
    assert len(ids) == len(set(ids)), f"dag_id repetido en {ids}"
