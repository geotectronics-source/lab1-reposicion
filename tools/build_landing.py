"""Simulador del sistema de origen de la empresa.

Toma los CSV originales de Kaggle y los convierte en:

1. Una **carga histórica** (`landing/historico/`): lo que ya existía cuando el
   producto entró en operación.
2. Una **bandeja** (`landing/arrival_date=YYYY-MM-DD/<fuente>/`): archivos que
   llegan día a día, cada fuente con su propia frecuencia y latencia.

ESTE ARCHIVO NO FORMA PARTE DEL PRODUCTO. Es el único código autorizado a
escribir en `landing/` (ver INVARIANTE 2 en CLAUDE.md). El producto trata la
bandeja como un sistema externo de sólo lectura.

La ruta codifica la fecha de llegada y el nombre codifica la fecha de los datos::

    landing/arrival_date=2017-07-06/sales/sales_2017-07-05.csv
                         ^ cuándo lo supimos      ^ a qué día se refiere

Esa separación es lo que hace que la corrección punto-en-el-tiempo sea
estructural y no un acto de disciplina: "usar sólo lo disponible en D" se
convierte en un filtro de carpetas.

Uso::

    uv run python tools/build_landing.py
    uv run python tools/build_landing.py --config config/sources.yaml --limpiar
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import yaml

# Fuentes cuyo CSV tiene una columna `date` que identifica el día del dato.
_CON_FECHA = {"hecho_diario", "catalogo"}


@dataclass(frozen=True)
class Entrega:
    """Un archivo que el sistema origen deposita en la bandeja.

    `arrival_date` es cuándo llega; `datos` es lo que contiene. Mantener esto
    como un objeto en memoria (en vez de escribir directo a disco) permite
    inyectar incidentes sobre el plan y probar la lógica de fechas sin E/S.
    """

    arrival_date: pd.Timestamp
    fuente: str
    nombre: str
    datos: pd.DataFrame

    @property
    def ruta_relativa(self) -> Path:
        return Path(f"arrival_date={self.arrival_date:%Y-%m-%d}") / self.fuente / self.nombre


# ---------------------------------------------------------------------------
# Lectura de configuración y de los CSV de origen
# ---------------------------------------------------------------------------


def cargar_config(ruta: Path) -> dict:
    with ruta.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def leer_fuentes(kaggle_dir: Path, fuentes: dict) -> dict[str, pd.DataFrame]:
    """Carga cada CSV de Kaggle y normaliza la columna de fecha."""
    datos: dict[str, pd.DataFrame] = {}
    for nombre, cfg in fuentes.items():
        ruta = kaggle_dir / cfg["archivo"]
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró {ruta}. Coloca los CSV de Kaggle en {kaggle_dir} "
                "(ver README.md) antes de construir la bandeja."
            )
        df = pd.read_csv(ruta)
        if cfg["tipo"] in _CON_FECHA:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date", kind="stable").reset_index(drop=True)
        datos[nombre] = df
    return datos


def calcular_corte(ventas: pd.DataFrame, origen: dict) -> pd.Timestamp:
    """Determina la fecha de corte entre carga histórica y bandeja.

    Si la configuración fija `corte_historico` se respeta; si es `null` se
    deriva como (última fecha de ventas - dias_operacion). Lo segundo es el
    modo por defecto para que la bandeja siga teniendo contenido cuando el
    evaluador aporte datos más recientes que los públicos.
    """
    if origen.get("corte_historico"):
        return pd.Timestamp(origen["corte_historico"])
    ultima = ventas["date"].max()
    return ultima - pd.Timedelta(days=int(origen["dias_operacion"]))


# ---------------------------------------------------------------------------
# Plan de entregas
# ---------------------------------------------------------------------------


def planificar(
    datos: dict[str, pd.DataFrame], fuentes: dict, corte: pd.Timestamp
) -> list[Entrega]:
    """Construye el calendario completo de llegadas posteriores al corte."""
    plan: list[Entrega] = []

    # Primero las fuentes de hechos: definen hasta cuándo llega la operación.
    for nombre, cfg in fuentes.items():
        if cfg["tipo"] != "hecho_diario":
            continue
        latencia = pd.Timedelta(days=int(cfg["latencia_dias"]))
        posteriores = datos[nombre][datos[nombre]["date"] > corte]
        # Un `groupby` por fecha evita recorrer 3M de filas una vez por día.
        for fecha, bloque in posteriores.groupby("date", sort=True):
            plan.append(
                Entrega(
                    arrival_date=fecha + latencia,
                    fuente=nombre,
                    nombre=f"{nombre}_{fecha:%Y-%m-%d}.csv",
                    datos=bloque.reset_index(drop=True),
                )
            )

    if not plan:
        raise ValueError(
            f"No hay datos posteriores al corte {corte:%Y-%m-%d}: la bandeja quedaría vacía. "
            "Revisa `dias_operacion` en config/sources.yaml."
        )

    ultimo_arrival = max(e.arrival_date for e in plan)

    # Luego los catálogos, que se re-publican periódicamente dentro de ese rango.
    for nombre, cfg in fuentes.items():
        if cfg["tipo"] != "catalogo":
            continue
        cadencia = pd.Timedelta(days=int(cfg["cadencia_dias"]))
        anticipacion = pd.Timedelta(days=int(cfg["anticipacion_dias"]))
        arrival = corte + pd.Timedelta(days=1)
        while arrival <= ultimo_arrival:
            # Un catálogo se publica completo, incluyendo el futuro conocido.
            vigente = datos[nombre][datos[nombre]["date"] <= arrival + anticipacion]
            plan.append(
                Entrega(
                    arrival_date=arrival,
                    fuente=nombre,
                    nombre=f"{nombre}_snapshot.csv",
                    datos=vigente.reset_index(drop=True),
                )
            )
            arrival = arrival + cadencia

    return sorted(plan, key=lambda e: (e.arrival_date, e.fuente, e.nombre))


# ---------------------------------------------------------------------------
# Incidentes de operación
# ---------------------------------------------------------------------------


def _corregir_retroactivamente(plan: list[Entrega], inc: dict) -> list[Entrega]:
    """La entrega original sale degradada; el dato real llega días después.

    Se construye así (y no al revés) para que el estado final de la bandeja
    coincida con los datos reales de Kaggle: lo que el sistema termina sabiendo
    es la verdad, y lo que supo primero fue el error.
    """
    fuente, fecha = inc["fuente"], pd.Timestamp(inc["fecha_datos"])
    esperado = f"{fuente}_{fecha:%Y-%m-%d}.csv"
    indice = next(
        (i for i, e in enumerate(plan) if e.fuente == fuente and e.nombre == esperado),
        None,
    )
    if indice is None:
        print(f"  ! incidente omitido: no hay entrega {esperado} en el plan", file=sys.stderr)
        return plan

    original = plan[indice]

    # La corrección reenvía el MISMO archivo con los valores verdaderos.
    correccion = replace(original, arrival_date=pd.Timestamp(inc["corregido_en"]))

    # La entrega original se degrada según `entrega_original`.
    degradado = original.datos.copy()
    mascara = pd.Series(True, index=degradado.index)
    for columna, valor in inc["entrega_original"]["donde"].items():
        mascara &= degradado[columna] == valor
    for columna, valor in inc["entrega_original"]["asignar"].items():
        degradado.loc[mascara, columna] = valor

    plan[indice] = replace(original, datos=degradado)
    plan.append(correccion)
    print(
        f"  · corrección retroactiva en {fuente} {fecha:%Y-%m-%d}: "
        f"{int(mascara.sum())} filas degradadas, se corrigen el {inc['corregido_en']}"
    )
    return plan


def _retrasar_entrega(plan: list[Entrega], inc: dict) -> list[Entrega]:
    """Un día sin entrega: los archivos se acumulan al día siguiente."""
    fuente, arrival = inc["fuente"], pd.Timestamp(inc["arrival_date"])
    movidas = 0
    for i, e in enumerate(plan):
        if e.fuente == fuente and e.arrival_date == arrival:
            plan[i] = replace(e, arrival_date=arrival + pd.Timedelta(days=1))
            movidas += 1
    print(f"  · {fuente} no entrega el {arrival:%Y-%m-%d}: {movidas} archivo(s) al día siguiente")
    return plan


def aplicar_incidentes(plan: list[Entrega], incidentes: list[dict] | None) -> list[Entrega]:
    if not incidentes:
        return plan
    manejadores = {
        "correccion_retroactiva": _corregir_retroactivamente,
        "fuente_no_llega": _retrasar_entrega,
    }
    print("Incidentes de operación inyectados:")
    for inc in incidentes:
        manejador = manejadores.get(inc["tipo"])
        if manejador is None:
            raise ValueError(f"Tipo de incidente desconocido: {inc['tipo']}")
        plan = manejador(plan, inc)
    return sorted(plan, key=lambda e: (e.arrival_date, e.fuente, e.nombre))


# ---------------------------------------------------------------------------
# Materialización
# ---------------------------------------------------------------------------


#: Fin de línea de la bandeja, fijo en cualquier sistema operativo.
#:
#: `to_csv` traduce `\n` al de la plataforma, de modo que sin esto la bandeja
#: salía con CRLF en Windows y LF en Linux. El producto no se entera —el CSV se
#: parsea igual— pero `landing/` es la salida de la etapa `bandeja` y DVC
#: compara bytes: el mismo simulador producía un hash distinto por plataforma,
#: y eso obsoletaba la cadena entera desde la etapa 0.
#:
#: Se declara aquí y no se importa de `reposicion.io` a propósito: el simulador
#: representa un sistema ajeno y no depende del producto (INVARIANTE 2). Que
#: las dos constantes coincidan es correcto, y que vivan separadas también.
FIN_DE_LINEA = "\n"


def escribir_csv(df: pd.DataFrame, destino: Path) -> None:
    """Escritura atómica: archivo temporal y renombrado.

    Evita que una interrupción deje un CSV a medio escribir que el producto
    leería como válido.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".tmp")
    df.to_csv(tmp, index=False, date_format="%Y-%m-%d", lineterminator=FIN_DE_LINEA)
    tmp.replace(destino)


def materializar_historico(
    datos: dict[str, pd.DataFrame], fuentes: dict, corte: pd.Timestamp, landing: Path
) -> int:
    destino = landing / "historico"
    for nombre, cfg in fuentes.items():
        df = datos[nombre]
        if cfg["tipo"] in _CON_FECHA:
            df = df[df["date"] <= corte]
        escribir_csv(df, destino / f"{nombre}.csv")
    return len(fuentes)


def materializar_bandeja(plan: list[Entrega], landing: Path) -> None:
    for entrega in plan:
        escribir_csv(entrega.datos, landing / entrega.ruta_relativa)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("config/sources.yaml"))
    parser.add_argument("--kaggle-dir", type=Path, default=None)
    parser.add_argument("--landing-dir", type=Path, default=None)
    parser.add_argument(
        "--limpiar",
        action="store_true",
        help="Borra la bandeja antes de reconstruirla (hace el simulador idempotente).",
    )
    args = parser.parse_args(argv)

    # La consola de Windows usa cp1252 y aborta al imprimir acentos o flechas.
    # El evaluador correrá esto en Windows y en Linux, así que se fuerza UTF-8
    # con reemplazo en vez de confiar en la codificación de la terminal.
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8", errors="replace")

    cfg = cargar_config(args.config)
    origen, fuentes = cfg["origen"], cfg["fuentes"]
    kaggle_dir = args.kaggle_dir or Path(origen["kaggle_dir"])
    landing = args.landing_dir or Path(origen["landing_dir"])

    print(f"Leyendo CSV originales de {kaggle_dir} ...")
    datos = leer_fuentes(kaggle_dir, fuentes)
    for nombre, df in datos.items():
        print(f"  {nombre:<18} {len(df):>9,} filas")

    corte = calcular_corte(datos["sales"], origen)
    print(f"\nCorte histórico: {corte:%Y-%m-%d}")

    plan = planificar(datos, fuentes, corte)
    plan = aplicar_incidentes(plan, cfg.get("incidentes"))

    if args.limpiar and landing.exists():
        shutil.rmtree(landing)
        print(f"\nBandeja anterior eliminada: {landing}")

    materializar_historico(datos, fuentes, corte, landing)
    materializar_bandeja(plan, landing)

    arrivals = sorted({e.arrival_date for e in plan})
    historico = landing / "historico"
    print(
        f"\nListo.\n"
        f"  carga histórica : {historico} "
        f"({len(fuentes)} fuentes, hasta {corte:%Y-%m-%d})\n"
        f"  bandeja         : {len(plan)} archivos en {len(arrivals)} fechas de llegada\n"
        f"  rango de llegada: {arrivals[0]:%Y-%m-%d} -> {arrivals[-1]:%Y-%m-%d}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
