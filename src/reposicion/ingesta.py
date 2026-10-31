"""Ingesta: de la bandeja al almacén propio, validando por el camino.

Toma los archivos que llegaron en una fecha, los contrasta con su contrato y
materializa una partición de Parquet. Lo que no cumple el contrato se aparta
en cuarentena y detiene la tarea antes de contaminar nada.

Tres decisiones gobiernan el módulo:

1. **La partición es todo-o-nada.** Si uno de los archivos de la fecha es
   inválido, no se escribe ninguno. Escribir los buenos y fallar dejaría una
   partición que parece completa y no lo es: contaminación silenciosa, que es
   peor que la caída.

2. **Se procesa una sola fecha por ejecución** (`arrival_date == D`, no `<=`).
   Cada run toca una partición, así que repetirlo reescribe exactamente lo
   mismo (INVARIANTE 3) y un backfill es un run por día.

3. **La ausencia de una fuente no es un fallo** (INVARIANTE 5). Si el sistema
   origen no entregó hoy, se registra y el flujo continúa con lo último
   conocido.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from reposicion.config import rutas
from reposicion.contracts import ESQUEMAS, ErrorDeContrato, avisos_de_negocio, validar
from reposicion.io import (
    copiar_a_cuarentena,
    escribir_particion,
    leer_particiones,
    marcar_llegada,
    normalizar_fecha,
    ruta_particion,
)
from reposicion.landing import (
    ARRIVAL_HISTORICO,
    entregas_del_dia,
    leer_historico,
    vista_vigente,
)

#: Fuentes que el producto sabe ingerir: exactamente las que tienen contrato.
#: Derivarlo de los contratos en vez de repetirlo en un YAML evita que ambas
#: listas se desincronicen sin que nadie lo note.
FUENTES = tuple(sorted(ESQUEMAS))


@dataclass(frozen=True)
class ResultadoIngesta:
    """Qué pasó al ingerir una fuente en una fecha. Se registra en el log."""

    fuente: str
    arrival_date: pd.Timestamp
    archivos: int = 0
    filas: int = 0
    particion: Path | None = None
    avisos: list[str] = field(default_factory=list)

    @property
    def hubo_entrega(self) -> bool:
        return self.archivos > 0

    def __str__(self) -> str:
        if not self.hubo_entrega:
            return (
                f"{self.fuente} {self.arrival_date:%Y-%m-%d}: sin entrega "
                "(situación de negocio, se continúa con lo último conocido)"
            )
        resumen = (
            f"{self.fuente} {self.arrival_date:%Y-%m-%d}: "
            f"{self.archivos} archivo(s), {self.filas:,} filas"
        )
        if self.avisos:
            resumen += "\n  aviso: " + "\n  aviso: ".join(self.avisos)
        return resumen


class IngestaRechazada(Exception):
    """Algún archivo de la fecha no cumplió su contrato.

    La partición no se escribió. Los archivos problemáticos están en
    cuarentena con el motivo al lado, para poder revisarlos sin reproducir
    la ejecución.
    """

    def __init__(self, fuente: str, arrival_date: pd.Timestamp, motivos: list[str],
                 cuarentena: list[Path]):
        self.fuente = fuente
        self.arrival_date = arrival_date
        self.motivos = motivos
        self.cuarentena = cuarentena
        detalle = "\n".join(motivos)
        ubicaciones = "\n".join(f"  - {p}" for p in cuarentena)
        super().__init__(
            f"Ingesta de '{fuente}' rechazada para {arrival_date:%Y-%m-%d}. "
            f"No se escribió ninguna partición.\n{detalle}\n"
            f"Archivos en cuarentena:\n{ubicaciones}"
        )


def ingerir(
    fuente: str,
    arrival_date: str | pd.Timestamp,
    *,
    landing: Path | None = None,
    raw: Path | None = None,
    cuarentena: Path | None = None,
) -> ResultadoIngesta:
    """Materializa la partición de `fuente` correspondiente a `arrival_date`.

    Devuelve un resultado vacío si no hubo entrega; lanza `IngestaRechazada`
    si algún archivo incumple el contrato.
    """
    dia = normalizar_fecha(arrival_date)
    destino_raw = raw or rutas().raw
    destino_cuarentena = cuarentena or rutas().cuarentena

    entregas = entregas_del_dia(fuente, dia, landing=landing)
    if not entregas:
        return ResultadoIngesta(fuente=fuente, arrival_date=dia)

    partes: list[pd.DataFrame] = []
    motivos: list[str] = []
    apartados: list[Path] = []

    def apartar(entrega, motivo: str) -> None:
        motivos.append(f"{entrega.nombre}: {motivo}")
        apartados.append(
            copiar_a_cuarentena(
                entrega.ruta, ruta_particion(destino_cuarentena, fuente, dia), motivo=motivo
            )
        )

    for entrega in entregas:
        # Las dos formas de fallar se capturan por separado y de forma
        # acotada. Envolver lectura y validación en un mismo `except
        # ValueError` haría que un defecto NUESTRO se registrara como "el
        # origen mandó basura", y el log mandaría a depurar al lugar
        # equivocado. Cualquier excepción que no sea una de estas dos se
        # propaga: un error del producto debe romper ruidosamente.
        try:
            df = pd.read_csv(entrega.ruta)
        except (ValueError, OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
            apartar(entrega, f"no se pudo leer como CSV: {exc}")
            continue

        try:
            # La conversión de `date` la hace el contrato con `coerce`: así una
            # fecha ilegible se reporta con el detalle de Pandera (columna y
            # fila) en vez de con un ValueError suelto de pandas.
            partes.append(validar(df, fuente, origen=str(entrega.ruta)))
        except ErrorDeContrato as exc:
            apartar(entrega, str(exc))
            continue

    # Todo-o-nada: con un solo archivo inválido no se materializa la partición.
    if motivos:
        raise IngestaRechazada(fuente, dia, motivos, apartados)

    df = marcar_llegada(pd.concat(partes, ignore_index=True), dia)

    particion = escribir_particion(df, ruta_particion(destino_raw, fuente, dia))

    return ResultadoIngesta(
        fuente=fuente,
        arrival_date=dia,
        archivos=len(entregas),
        filas=len(df),
        particion=particion,
        avisos=avisos_de_negocio(df, fuente),
    )


def tiene_carga_inicial(fuente: str, *, raw: Path | None = None) -> bool:
    """Si el almacén ya contiene la carga histórica de `fuente`.

    Lo consulta el modo incremental: añadir la llegada de un día sobre un
    almacén que nunca recibió el histórico daría un almacén que parece válido
    y no tiene con qué calcular una media móvil de 56 días.
    """
    return ruta_particion(raw or rutas().raw, fuente, ARRIVAL_HISTORICO).exists()


def ingerir_historico(
    fuente: str, *, landing: Path | None = None, raw: Path | None = None
) -> ResultadoIngesta:
    """Materializa la carga inicial como la partición más antigua.

    Tratar el histórico como una partición más (con fecha de llegada anterior
    a cualquier dato real) evita que el resto del producto necesite un camino
    especial para leerlo: `leer_raw_asof` lo recoge como a cualquier otra.
    """
    destino_raw = raw or rutas().raw
    df = leer_historico(fuente, landing=landing, validar_contrato=True)

    if df.empty:
        return ResultadoIngesta(fuente=fuente, arrival_date=ARRIVAL_HISTORICO)

    particion = escribir_particion(df, ruta_particion(destino_raw, fuente, ARRIVAL_HISTORICO))

    return ResultadoIngesta(
        fuente=fuente,
        arrival_date=ARRIVAL_HISTORICO,
        archivos=1,
        filas=len(df),
        particion=particion,
        avisos=avisos_de_negocio(df, fuente),
    )


def leer_raw_asof(
    fuente: str, asof: str | pd.Timestamp, *, raw: Path | None = None
) -> pd.DataFrame:
    """Todo lo ingerido de `fuente` hasta `asof`, con su fecha de llegada.

    Equivale a `landing.leer_asof` pero sobre el almacén propio: ya validado,
    en Parquet y con los tipos preservados. Es la lectura que usan las fases
    posteriores; la de la bandeja sólo la usa la ingesta.
    """
    return leer_particiones(raw or rutas().raw, fuente, hasta=asof)


def leer_raw_vigente(
    fuente: str, asof: str | pd.Timestamp, *, raw: Path | None = None
) -> pd.DataFrame:
    """Lo que rige en `asof`, con las correcciones ya resueltas."""
    return vista_vigente(leer_raw_asof(fuente, asof, raw=raw), fuente)
