"""Puntos de entrada del pipeline: una orden por etapa.

Estas órdenes son la frontera entre la orquestación y el producto. No
contienen lógica de negocio: leen la configuración, llaman a `src/reposicion`
y materializan el resultado. Es el mismo contrato que se exige a los DAGs
(CLAUDE.md §2), y por eso los dos orquestadores pueden invocar lo mismo: DVC
las ejecuta para reproducir un pronóstico pasado y Airflow las ejecutará cada
semana para operar.

Que exista una sola frontera es lo que hace que el laboratorio se sostenga.
Si DVC llamara a una función y Airflow a otra, "lo que reproduce DVC" y "lo
que corre en producción" serían dos cosas distintas y la reproducibilidad
sería decorativa.

**`--asof` es obligatorio en todas y nunca tiene valor por defecto.** Ninguna
orden le pregunta la fecha al sistema operativo (INVARIANTE 1): la fecha
lógica entra desde fuera —el parámetro de DVC o la `logical_date` de Airflow—
y es eso lo que hace que un backfill de julio dé hoy el mismo resultado que
dio en julio.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import typer

from reposicion import features, registro, validacion
from reposicion.config import cargar_config, rutas
from reposicion.ingesta import FUENTES, ingerir, ingerir_historico
from reposicion.io import escribir_csv, escribir_json, escribir_parquet, normalizar_fecha
from reposicion.landing import descubrir_entregas
from reposicion.models.baseline import BaselineEstacional
from reposicion.models.lgbm import LightGBMCuantilico
from reposicion.models.lineal import RegresionCuantilicaLineal
from reposicion.policy import calcular_pedido, explicar, resumen_por_serie
from reposicion.pronostico import cobertura_del_respaldo, emitir

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Motor de reposicion: una orden por etapa del pipeline.",
)

#: Modelos disponibles, indexados por el nombre con el que aparecen en
#: `config/model.yaml`. El nombre vive en la clase y no aquí para que no haya
#: dos fuentes de verdad sobre cómo se llama un modelo.
CATALOGO = {
    clase.nombre: clase
    for clase in (BaselineEstacional, LightGBMCuantilico, RegresionCuantilicaLineal)
}

ASOF = typer.Option(
    ...,
    "--asof",
    help="Fecha logica de la emision, YYYY-MM-DD. Obligatoria: el producto no "
    "consulta el reloj del sistema.",
)

ARCHIVO_PANEL = "entrenamiento.parquet"
ARCHIVO_DECISION = "promocion.json"
ARCHIVO_PRONOSTICO = "pronostico.parquet"


@app.callback()
def preparar_salida() -> None:
    """Fuerza UTF-8 en la salida antes de ejecutar cualquier orden.

    En Windows, Python usa la codificación de la consola (cp1252) para stdout.
    Un mensaje con tilde lanza entonces `UnicodeEncodeError` y **la etapa
    falla por un acento**, no por el cálculo. Pasó ya una vez con el simulador
    de la bandeja.

    Va en el `callback` y no en el módulo porque reconfigurar los flujos es un
    efecto de ejecutable, no de importación: una prueba que importe `cli` no
    debe alterar la salida del proceso de pytest.
    """
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8", errors="replace")


def _carpeta(base: Path, asof: pd.Timestamp) -> Path:
    """Carpeta de salida de una emisión.

    Las salidas se particionan por fecha lógica, no por fecha de ejecución: es
    lo que permite reejecutar una emisión antigua sin pisar las demás
    (INVARIANTE 3) y lo que hace que un backfill sea seguro.
    """
    return base / f"asof={asof:%Y-%m-%d}"


def _hecho(destino: Path, desde: float) -> None:
    typer.secho(f"-> {destino}  ({time.monotonic() - desde:.1f} s)", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------
# 1. Ingesta
# ---------------------------------------------------------------------------


@app.command()
def ingesta(
    asof: str = ASOF,
    solo_el_dia: bool = typer.Option(
        False,
        "--solo-el-dia",
        help="Ingiere unicamente la llegada de asof, no el historico anterior. "
        "Es la forma en que lo llamara Airflow una vez al dia.",
    ),
) -> None:
    """Materializa en el almacén propio todo lo que había llegado en `asof`.

    El almacén **acumula**: no se borra lo ingerido después de `asof`, porque
    el filtro punto-en-el-tiempo se aplica al leer (`leer_raw_asof`) y no al
    escribir. Borrarlo destruiría el histórico que permite auditar
    (INVARIANTE 4).
    """
    dia = normalizar_fecha(asof)
    inicio = time.monotonic()

    for fuente in FUENTES:
        resultados = []
        if solo_el_dia:
            fechas = [dia]
        else:
            resultados.append(ingerir_historico(fuente))
            fechas = sorted({e.arrival_date for e in descubrir_entregas(fuente, dia)})

        for fecha in fechas:
            resultados.append(ingerir(fuente, fecha))

        entregados = [r for r in resultados if r.hubo_entrega]
        filas = sum(r.filas for r in entregados)
        sin_entrega = "" if entregados else "   sin entrega (situacion de negocio)"
        typer.echo(
            f"{fuente:<17} {len(entregados):>3} particion(es), {filas:>10,} filas{sin_entrega}"
        )
        for resultado in entregados:
            for aviso in resultado.avisos:
                typer.secho(
                    f"  aviso {resultado.arrival_date:%Y-%m-%d}: {aviso}",
                    fg=typer.colors.YELLOW,
                )

    _hecho(rutas().raw, inicio)


# ---------------------------------------------------------------------------
# 2. Panel de entrenamiento
# ---------------------------------------------------------------------------


@app.command()
def panel(asof: str = ASOF) -> None:
    """Genera los ejemplos de entrenamiento simulando emisiones pasadas.

    Es la etapa cara del pipeline: reconstruye el estado de la información en
    cada origen histórico. Se materializa en un archivo propio justamente para
    no repetirla cada vez que se compara un modelo.
    """
    dia = normalizar_fecha(asof)
    inicio = time.monotonic()

    ejemplos = features.construir_entrenamiento(dia, progreso=True)
    if ejemplos.empty:
        raise typer.BadParameter(
            f"No se genero ningun ejemplo para asof={dia:%Y-%m-%d}. "
            "Revisa que la etapa de ingesta haya materializado data/raw."
        )

    destino = escribir_parquet(ejemplos, _carpeta(rutas().procesado, dia) / ARCHIVO_PANEL)
    series = len(ejemplos.drop_duplicates(features.CLAVE_SERIE))
    typer.echo(
        f"{len(ejemplos):,} ejemplos  ·  {ejemplos['origen'].nunique()} origenes  ·  "
        f"{series:,} series"
    )
    _hecho(destino, inicio)


# ---------------------------------------------------------------------------
# 3. Validación y promoción
# ---------------------------------------------------------------------------


@app.command()
def validar(asof: str = ASOF) -> None:
    """Compara todos los modelos con los mismos pliegues y decide quién opera.

    Escribe tres salidas con propósitos distintos y en dos sitios distintos:

    - `data/seguimiento/`: la tabla completa por modelo y pliegue. Es el
      detalle para auditar la comparación; va a la caché de DVC.
    - `reportes/`: el resumen de métricas y la decisión de promoción. Son unos
      kilobytes y **sí van a Git**, para que el historial conteste con qué
      números se promovió un modelo y para que el cambio aparezca en el diff
      del Pull Request.
    """
    dia = normalizar_fecha(asof)
    inicio = time.monotonic()
    cfg = cargar_config("model")
    cuantiles = cfg["pronostico"]["cuantiles"]

    ejemplos = pd.read_parquet(_carpeta(rutas().procesado, dia) / ARCHIVO_PANEL)

    # Los pliegues se anclan a la última fecha OBJETIVO del panel, no a `asof`:
    # entre el origen y la fecha emitida hay horizonte, y anclar a `asof`
    # dejaría el último pliegue sin nada que evaluar.
    pliegues = validacion.pliegues_moviles(
        ejemplos["date"].max(),
        int(cfg["validacion"]["pliegues"]),
        int(cfg["validacion"]["dias_por_pliegue"]),
    )
    competidores = [cfg["promocion"]["vigente"], *cfg["promocion"]["retadores"]]
    modelos = [CATALOGO[nombre](cuantiles) for nombre in competidores]

    resultados = validacion.validar(modelos, ejemplos, pliegues, cuantiles, verboso=True)
    resumen = validacion.resumir(resultados)
    campeon, decisiones = validacion.elegir_campeon(resultados, config=cfg)

    typer.echo("")
    typer.echo(resumen.to_string(index=False))
    typer.echo("")
    for nombre, decision in decisiones.items():
        typer.echo(f"{nombre:<22} {decision}")
    typer.secho(f"\nmodelo promovido: {campeon}", fg=typer.colors.GREEN, bold=True)

    escribir_csv(resultados, _carpeta(rutas().seguimiento, dia) / "validacion.csv")

    salida = _carpeta(rutas().reportes, dia)
    escribir_json(
        {
            "campeon": campeon,
            "vigente": cfg["promocion"]["vigente"],
            "pliegues": len(pliegues),
            "ejemplos": int(len(ejemplos)),
            "modelos": resumen.set_index("modelo").to_dict(orient="index"),
        },
        salida / "metricas.json",
    )
    destino = escribir_json(
        {
            "campeon": campeon,
            "decisiones": {
                nombre: {
                    "promover": decision.promover,
                    "razon": decision.razon,
                    "mejoras": {str(numero): mejora for numero, mejora in decision.mejoras.items()},
                }
                for nombre, decision in decisiones.items()
            },
        },
        salida / ARCHIVO_DECISION,
    )
    _hecho(destino, inicio)


# ---------------------------------------------------------------------------
# 4. Modelo promovido
# ---------------------------------------------------------------------------


@app.command()
def modelo(asof: str = ASOF) -> None:
    """Ajusta el modelo promovido sobre el panel completo y lo registra.

    La validación usa recortes del panel para poder medir contra un futuro que
    el modelo no vio. El que opera se ajusta con TODO el panel: no hay nada que
    medir y descartar la parte más reciente sería tirar justo la información
    más parecida a la semana que viene.
    """
    dia = normalizar_fecha(asof)
    inicio = time.monotonic()
    cfg = cargar_config("model")

    ejemplos = pd.read_parquet(_carpeta(rutas().procesado, dia) / ARCHIVO_PANEL)
    ruta_decision = _carpeta(rutas().reportes, dia) / ARCHIVO_DECISION
    decision = json.loads(ruta_decision.read_text(encoding="utf-8"))
    nombre = decision["campeon"]

    ajustado = CATALOGO[nombre](cfg["pronostico"]["cuantiles"]).entrenar(ejemplos)
    destino = registro.guardar(
        ajustado,
        _carpeta(rutas().modelos, dia),
        {
            "nombre": nombre,
            "asof": f"{dia:%Y-%m-%d}",
            "ejemplos": int(len(ejemplos)),
            "cuantiles": cfg["pronostico"]["cuantiles"],
            "ultimo_objetivo": f"{ejemplos['date'].max():%Y-%m-%d}",
            "promocion": decision["decisiones"],
        },
    )
    typer.echo(f"{nombre} ajustado con {len(ejemplos):,} ejemplos")
    _hecho(destino, inicio)


# ---------------------------------------------------------------------------
# 5. Pronóstico
# ---------------------------------------------------------------------------


@app.command()
def pronosticar(asof: str = ASOF) -> None:
    """Emite el pronóstico cuantílico del horizonte que arranca en `asof`."""
    dia = normalizar_fecha(asof)
    inicio = time.monotonic()
    cfg = cargar_config("model")

    futuro = features.construir_panel(dia, config=cfg)
    if futuro.empty:
        raise typer.BadParameter(
            f"No hay datos para pronosticar en asof={dia:%Y-%m-%d}: el almacen esta vacio."
        )

    emitido = emitir(futuro, registro.cargar(_carpeta(rutas().modelos, dia)), dia, config=cfg)

    typer.echo(f"{len(emitido):,} filas  ·  origen {emitido['origen'].max():%Y-%m-%d}")
    typer.echo(cobertura_del_respaldo(emitido).to_string(index=False))

    destino = escribir_parquet(emitido, _carpeta(rutas().pronosticos, dia) / ARCHIVO_PRONOSTICO)
    _hecho(destino, inicio)


# ---------------------------------------------------------------------------
# 6. Pedido
# ---------------------------------------------------------------------------


@app.command()
def pedir(asof: str = ASOF) -> None:
    """Convierte el pronóstico en pedido sugerido aplicando los costos.

    Esta etapa no depende del modelo, sólo del pronóstico ya emitido y de
    `config/costs.yaml`. Por eso cambiar un costo recalcula el pedido en
    segundos sin reentrenar nada: es una propiedad del grafo de etapas, no una
    promesa.
    """
    dia = normalizar_fecha(asof)
    inicio = time.monotonic()
    costos = cargar_config("costs")

    emitido = pd.read_parquet(_carpeta(rutas().pronosticos, dia) / ARCHIVO_PRONOSTICO)
    pedido = calcular_pedido(emitido, config_costos=costos)
    resumen = resumen_por_serie(pedido)
    explicacion = explicar(pedido, costos)

    recortadas = int(pedido["tope_aplicado"].sum())
    typer.echo(
        f"{len(resumen):,} series  ·  {pedido['pedido_sugerido'].sum():,.0f} unidades  ·  "
        f"{recortadas} linea(s) recortada(s) por el tope"
    )
    typer.echo(explicacion.to_string(index=False))

    salida = _carpeta(rutas().pedidos, dia)
    escribir_parquet(pedido, salida / "pedido.parquet")
    escribir_csv(resumen, salida / "pedido_semana.csv")
    destino = escribir_csv(explicacion, salida / "explicacion.csv")
    _hecho(destino, inicio)


if __name__ == "__main__":  # pragma: no cover
    app()
