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

from reposicion import calidad, features, registro, seguimiento, validacion
from reposicion.config import cargar_config, rutas
from reposicion.ingesta import FUENTES, ingerir, ingerir_historico, tiene_carga_inicial
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
    incremental: bool = typer.Option(
        False,
        "--incremental",
        help="Ingiere solo la llegada de asof, suponiendo que el almacen ya "
        "refleja los dias anteriores. Es como lo llama Airflow cada dia.",
    ),
) -> None:
    """Materializa en el almacén propio todo lo que había llegado en `asof`.

    El almacén **acumula**: no se borra lo ingerido después de `asof`, porque
    el filtro punto-en-el-tiempo se aplica al leer (`leer_raw_asof`) y no al
    escribir. Borrarlo destruiría el histórico que permite auditar
    (INVARIANTE 4).

    Hay dos modos, uno por orquestador, y los dos dejan el almacén en el mismo
    estado para una misma fecha:

    - **completo** (por defecto): rehace todo lo llegado hasta `asof`. Es el de
      DVC, porque una etapa reproducible tiene que ser función de sus entradas
      y no del orden en que se ejecutó antes.
    - **incremental** (`--incremental`): añade sólo la partición de `asof`. Es
      el de Airflow, un run por día, porque reprocesar seis semanas de
      llegadas cada mañana sería trabajo cuadrático a cambio de nada.

    El modo incremental hace la **carga inicial si falta**. Dejar ese paso
    fuera obligaría a una orden manual antes del primer run del despliegue, y
    es justo el paso que nadie recuerda: el almacén quedaría con una semana de
    llegadas y sin los dos años de historia que necesitan las ventanas.
    """
    dia = normalizar_fecha(asof)
    inicio = time.monotonic()

    for fuente in FUENTES:
        resultados = []
        if incremental:
            if not tiene_carga_inicial(fuente):
                resultados.append(ingerir_historico(fuente))
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
# 1b. Auditoría del almacén
# ---------------------------------------------------------------------------


@app.command()
def auditar(asof: str = ASOF) -> None:
    """Registra el estado de cada fuente y distingue un hueco de un flujo roto.

    Falla **sólo** si alguna fuente acumula más retraso del tolerado o no
    tiene ni un dato. Que hoy no haya entregado no la hace fallar: eso es
    negocio (INVARIANTE 5) y queda anotado en `calidad.csv`.

    Es la tarea que convierte una dependencia implícita en una explícita. Sin
    ella, pronosticar un lunes cuya ingesta nunca corrió daría un pronóstico
    perfectamente válido calculado sobre datos viejos, y nadie se enteraría.
    """
    dia = normalizar_fecha(asof)
    inicio = time.monotonic()

    reporte = calidad.auditar(dia)
    columnas = ["fuente", "estado", "ultimo_dato", "retraso_dias", "filas_en_asof"]
    typer.echo(reporte[columnas].to_string(index=False))

    for fuente in calidad.huecos(reporte):
        typer.secho(
            f"  hueco: {fuente} no entrego en {dia:%Y-%m-%d}; se sigue con lo ultimo conocido",
            fg=typer.colors.YELLOW,
        )
    for _, fila in reporte.loc[reporte["reenviadas_en_asof"] > 0].iterrows():
        typer.secho(
            f"  {fila['fuente']} reenvio {fila['reenviadas_en_asof']:,} fila(s) ya "
            f"conocidas, de las que {fila['correcciones_en_asof']:,} cambian de valor",
            fg=typer.colors.CYAN,
        )

    destino = escribir_csv(reporte, calidad.ruta_reporte(dia))
    _hecho(destino, inicio)

    if rotas := calidad.atrasadas(reporte):
        raise typer.BadParameter(
            f"Fuentes con retraso inadmisible en {dia:%Y-%m-%d}: {rotas}. "
            "Esto no es un hueco de entrega: la ingesta de esos dias no corrio. "
            "Ejecuta el DAG `ingesta_fuentes` para las fechas que faltan."
        )


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
    """Emite el pronóstico cuantílico del horizonte que arranca en `asof`.

    Usa el **modelo vigente**: el último registrado con fecha ≤ `asof`, no uno
    entrenado para `asof`. El pronóstico sale cada semana y el reentrenamiento
    corre una vez al mes, así que exigir un modelo de la misma fecha dejaría al
    producto sin poder emitir casi ningún lunes. Qué versión atendió la emisión
    queda escrito en `emision.json`, que es lo que después permite al
    seguimiento preguntar "¿empeoró desde que cambiamos de modelo?".
    """
    dia = normalizar_fecha(asof)
    inicio = time.monotonic()
    cfg = cargar_config("model")

    futuro = features.construir_panel(dia, config=cfg)
    if futuro.empty:
        raise typer.BadParameter(
            f"No hay datos para pronosticar en asof={dia:%Y-%m-%d}: el almacen esta vacio."
        )

    version = registro.version_vigente(rutas().modelos, dia)
    ficha = registro.leer_ficha(version)
    emitido = emitir(futuro, registro.cargar(version), dia, config=cfg)
    respaldo = cobertura_del_respaldo(emitido)

    typer.echo(
        f"{len(emitido):,} filas  ·  origen {emitido['origen'].max():%Y-%m-%d}  ·  "
        f"modelo {ficha['nombre']} de {ficha['asof']}"
    )
    typer.echo(respaldo.to_string(index=False))

    salida = _carpeta(rutas().pronosticos, dia)
    escribir_parquet(emitido, salida / ARCHIVO_PRONOSTICO)
    destino = escribir_json(
        {
            "asof": f"{dia:%Y-%m-%d}",
            "modelo": ficha["nombre"],
            "modelo_asof": ficha["asof"],
            "origen": f"{emitido['origen'].max():%Y-%m-%d}",
            "filas": int(len(emitido)),
            "cobertura": respaldo.to_dict(orient="records"),
        },
        salida / seguimiento.ARCHIVO_EMISION,
    )
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


# ---------------------------------------------------------------------------
# 7. Seguimiento del desempeño
# ---------------------------------------------------------------------------


@app.command()
def seguir(asof: str = ASOF) -> None:
    """Mide qué pasó de verdad con los pedidos ya publicados.

    **Esta orden no falla por un modelo que se degrada.** Un desempeño peor
    que el validado es una situación de negocio, no un dato inválido
    (INVARIANTE 5): hacer caer el DAG por eso dejaría al día siguiente sin
    medición, que es justo cuando hace falta. La alerta queda en el log y en
    `desempeno.json`, y la decisión de reentrenar es de una persona.

    Lo único que la hace fallar es que el almacén no se pueda leer, es decir,
    un problema de verdad.
    """
    dia = normalizar_fecha(asof)
    inicio = time.monotonic()

    detalle, resumen = seguimiento.medir(dia)

    for fila in resumen["emisiones"]:
        if not fila["lineas"]:
            typer.echo(f"{fila['emision']}   {fila['estado']}")
            continue
        validado = f"{fila['costo_validado']:8.2f}" if fila["costo_validado"] else "       -"
        typer.echo(
            f"{fila['emision']}   {fila['realizado']:>6.1%} realizado   "
            f"costo {fila['costo_realizado']:8.2f} vs {validado} validado   "
            f"cobertura {fila['cobertura_usada']:>6.1%} (la politica pide "
            f"{fila['cuantil_usado']:.0%})"
        )

    for aviso in resumen["alertas"]:
        typer.secho(f"  ALERTA  {aviso}", fg=typer.colors.YELLOW, bold=True)
    if resumen["emisiones"] and not resumen["alertas"]:
        typer.secho("sin alertas", fg=typer.colors.GREEN)
    if nota := resumen.get("nota"):
        typer.echo(nota)

    salida = _carpeta(rutas().seguimiento, dia)
    if not detalle.empty:
        escribir_csv(detalle, salida / "desempeno.csv")
    destino = escribir_json(resumen, salida / "desempeno.json")
    _hecho(destino, inicio)


if __name__ == "__main__":  # pragma: no cover
    app()
