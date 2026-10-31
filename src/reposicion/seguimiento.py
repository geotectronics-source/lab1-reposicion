"""Seguimiento del desempeño: qué pasó de verdad con los pedidos publicados.

Un pronóstico que nadie contrasta con la realidad no es un producto, es una
opinión. Este módulo cierra el lazo: toma los pedidos que el producto ya
emitió, los cruza con las ventas que fueron llegando y mide **cuánto costó de
verdad** la decisión, en la misma moneda con la que se eligió el modelo.

Tres decisiones de diseño que conviene poder defender:

**Es la tarea que justifica tener Airflow y no sólo DVC.** El grafo de DVC es
estático: una etapa declara de qué archivos depende y eso no cambia. La
entrada de esta medición es "todas las emisiones pasadas cuyo horizonte ya se
realizó", que no es una arista fija sino una consulta que depende de la fecha
en que se corre. DVC no puede expresarla; un orquestador con fecha lógica sí.
Por eso esta etapa vive sólo en los DAGs.

**Se mide con lo que se sabía en la fecha lógica de la medición, no con lo
que se sabe hoy.** La lectura de ventas pasa por la vista vigente en `asof`,
que filtra por `arrival_date` (INVARIANTE 1). Tiene una consecuencia que
parece un detalle y no lo es: volver a medir el desempeño del 20 de julio da
hoy el mismo número que dio el 20 de julio, porque la foto de ese día está
congelada. Medir contra "lo mejor que sabemos ahora" daría una cifra más
exacta y **no reproducible**, y una métrica de monitoreo que cambia sola cada
vez que llega una corrección no sirve para detectar nada.

**El baremo es el costo con el que se validó el modelo que atendió la
emisión.** No la emisión anterior. Lo que se vigila es si el modelo cumple en
operación la promesa con la que se lo promovió; comparar contra la semana
pasada mediría la semana, no el modelo.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from reposicion.config import cargar_config, rutas
from reposicion.ingesta import leer_raw_vigente
from reposicion.io import normalizar_fecha
from reposicion.metrics import cobertura, costo_negocio, costos_de_familia, pinball_medio
from reposicion.registro import ETIQUETA_VERSION

ARCHIVO_PEDIDO = "pedido.parquet"
ARCHIVO_EMISION = "emision.json"

#: Clave con la que se cruza un pedido publicado con la venta realizada.
CLAVE_LINEA = ["date", "store_nbr", "family"]

#: Nombre que `metrics` espera para el valor observado.
COLUMNA_REAL = "y"


def emisiones_publicadas(base: Path, asof: str | pd.Timestamp) -> list[str]:
    """Emisiones con pedido publicado y fecha ≤ `asof`, en orden.

    El filtro por fecha no es cosmético: en un backfill del 20 de julio, la
    carpeta de la emisión de agosto ya existe en el disco, y medirla sería
    usar el futuro para calificar el pasado.
    """
    if not base.exists():
        return []
    dia = f"{normalizar_fecha(asof):%Y-%m-%d}"
    return sorted(
        etiqueta
        for carpeta in base.iterdir()
        if carpeta.is_dir()
        and carpeta.name.startswith(ETIQUETA_VERSION)
        and (carpeta / ARCHIVO_PEDIDO).exists()
        and (etiqueta := carpeta.name.removeprefix(ETIQUETA_VERSION)) <= dia
    )


def unir_realizado(pedido: pd.DataFrame, ventas: pd.DataFrame) -> pd.DataFrame:
    """Añade a cada línea del pedido la venta que de verdad ocurrió.

    El cruce es interno a propósito. Una línea sin venta correspondiente
    significa que ese día **todavía no ha llegado**: con latencia D+1, el
    pedido del lunes no se puede calificar hasta el martes. Rellenar con cero
    convertiría "no lo sé" en "no se vendió nada", que es el error más caro
    posible aquí: inventaría una merma que nadie tuvo y haría ver mal a un
    modelo que acertó.
    """
    if ventas.empty:
        # Almacén sin ninguna venta todavía: es el estado de la primera
        # emisión, antes de que llegue el primer día del horizonte. Pedirle
        # columnas a una tabla vacía y fallar por no tenerlas confundiría ese
        # estado normal con una fuente que incumple su contrato, que es la
        # distinción que sostiene la INVARIANTE 5.
        sin_realizar = pedido.iloc[0:0].copy()
        sin_realizar[COLUMNA_REAL] = pd.Series(dtype=float)
        return sin_realizar

    faltan = [c for c in CLAVE_LINEA if c not in ventas.columns]
    if faltan or "sales" not in ventas.columns:
        raise KeyError(f"Las ventas no traen las columnas necesarias: {faltan or ['sales']}")

    real = ventas[[*CLAVE_LINEA, "sales"]].rename(columns={"sales": COLUMNA_REAL})
    return pedido.merge(real, on=CLAVE_LINEA, how="inner")


def medir_emision(
    lineas: pd.DataFrame,
    *,
    cuantiles: list[float],
    config_costos: dict | None = None,
) -> pd.DataFrame:
    """Desempeño realizado de una emisión, una fila por familia.

    Se desagrega por familia y no sólo en total porque los costos son por
    familia: un total que se ve bien puede esconder que en PRODUCE, donde la
    merma es cara, el pedido viene sistemáticamente alto.
    """
    costos = config_costos or cargar_config("costs")
    filas = []

    for familia, bloque in lineas.groupby("family", observed=True):
        cu, co = costos_de_familia(str(familia), costos)
        usado = float(bloque["cuantil_usado"].iloc[0])
        real = bloque[COLUMNA_REAL].to_numpy(dtype=float)
        pedido = bloque["pedido_sugerido"].to_numpy(dtype=float)

        filas.append(
            {
                "family": familia,
                "lineas": int(len(bloque)),
                "dias": int(bloque["date"].nunique()),
                "real": float(real.sum()),
                "pedido": float(pedido.sum()),
                # Las dos caras del costo, en unidades y no en dinero: es lo
                # que operaciones puede reconocer de su propio inventario.
                "faltante_unidades": float(np.maximum(0.0, real - pedido).sum()),
                "sobrante_unidades": float(np.maximum(0.0, pedido - real).sum()),
                "costo_realizado": costo_negocio(real, pedido, cu, co),
                "pinball_medio": pinball_medio(bloque, cuantiles),
                "cuantil_usado": usado,
                "cobertura_usada": cobertura(real, bloque[f"q{usado}"], usado),
                # Positivo = el pronóstico central iba por encima de la venta.
                "sesgo_central": float((bloque["q0.5"].to_numpy(dtype=float) - real).mean()),
            }
        )

    return pd.DataFrame(filas).sort_values("family").reset_index(drop=True)


def costo_validado(
    emision: str, *, pronosticos: Path | None = None, reportes: Path | None = None
) -> tuple[float | None, str]:
    """Costo con el que se validó el modelo que atendió `emision`.

    Devuelve también de dónde salió, porque una comparación cuyo baremo no se
    puede señalar no es auditable.

    Hay dos casos y los dos son legítimos. En operación, la emisión registra
    en `emision.json` qué versión del modelo la atendió, y el baremo es el
    reporte de **esa** versión: el modelo pudo haberse entrenado un mes antes.
    En una emisión producida por el pipeline de DVC, el modelo se entrenó para
    esa misma fecha, de modo que el reporte de la emisión es el correcto; es
    también el camino que siguen las emisiones anteriores a que existiera
    `emision.json`.
    """
    carpeta_reportes = reportes or rutas().reportes
    ficha = (pronosticos or rutas().pronosticos) / f"{ETIQUETA_VERSION}{emision}" / ARCHIVO_EMISION

    version, modelo = emision, None
    if ficha.exists():
        contenido = json.loads(ficha.read_text(encoding="utf-8"))
        version = contenido.get("modelo_asof") or emision
        modelo = contenido.get("modelo")

    reporte = carpeta_reportes / f"{ETIQUETA_VERSION}{version}" / "metricas.json"
    if not reporte.exists():
        return None, f"sin reporte de validacion para {version}"

    metricas = json.loads(reporte.read_text(encoding="utf-8"))
    nombre = modelo or metricas.get("campeon")
    bloque = metricas.get("modelos", {}).get(nombre)
    if not bloque or "costo_negocio" not in bloque:
        return None, f"el reporte de {version} no mide a {nombre}"
    return float(bloque["costo_negocio"]), f"{nombre} validado en {version}"


def _alertas(fila: dict, cfg: dict) -> list[str]:
    """Avisos de una emisión ya medida. Son avisos, nunca excepciones."""
    if fila["lineas"] < int(cfg["minimo_lineas"]):
        return []

    avisos = []
    esperado = fila["costo_validado"]
    if esperado:
        exceso = fila["costo_realizado"] / esperado - 1.0
        if exceso > float(cfg["tolerancia_costo"]):
            avisos.append(
                f"el costo realizado supera en {exceso:.1%} el validado "
                f"({fila['costo_realizado']:.2f} frente a {esperado:.2f})"
            )

    desvio = fila["cuantil_usado"] - fila["cobertura_usada"]
    if desvio > float(cfg["tolerancia_cobertura"]):
        avisos.append(
            f"el pedido cubre el {fila['cobertura_usada']:.1%} donde la politica "
            f"pide el {fila['cuantil_usado']:.0%}: se esta quedando corto"
        )
    return avisos


def medir(
    asof: str | pd.Timestamp,
    *,
    config: dict | None = None,
    config_modelo: dict | None = None,
    config_costos: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Mide el desempeño realizado de las emisiones vigentes en `asof`.

    Devuelve el detalle por emisión y familia, y un resumen con las alertas.
    Una emisión cuyo horizonte aún no se realizó no es un problema: se informa
    y se sigue (INVARIANTE 5).
    """
    dia = normalizar_fecha(asof)
    cfg = config or cargar_config("seguimiento")
    modelo = config_modelo or cargar_config("model")
    costos = config_costos or cargar_config("costs")
    cuantiles = modelo["pronostico"]["cuantiles"]

    base = rutas().pedidos
    candidatas = emisiones_publicadas(base, dia)[-int(cfg["ventana_emisiones"]) :]

    if not candidatas:
        return pd.DataFrame(), {
            "asof": f"{dia:%Y-%m-%d}",
            "emisiones": [],
            "alertas": [],
            "nota": "no hay ninguna emision publicada con fecha anterior o igual",
        }

    # Una sola lectura de ventas para todas las emisiones: es la parte cara.
    ventas = leer_raw_vigente("sales", dia)

    detalle, resumen_emisiones, alertas = [], [], []
    for emision in candidatas:
        pedido = pd.read_parquet(base / f"{ETIQUETA_VERSION}{emision}" / ARCHIVO_PEDIDO)
        lineas = unir_realizado(pedido, ventas)

        if lineas.empty:
            resumen_emisiones.append(
                {
                    "emision": emision,
                    "lineas": 0,
                    "publicadas": int(len(pedido)),
                    "estado": "horizonte aun sin realizar",
                }
            )
            continue

        por_familia = medir_emision(lineas, cuantiles=cuantiles, config_costos=costos)
        detalle.append(por_familia.assign(emision=emision, evaluado_en=f"{dia:%Y-%m-%d}"))

        esperado, procedencia = costo_validado(emision)
        real = lineas[COLUMNA_REAL].to_numpy(dtype=float)
        pedido_total = lineas["pedido_sugerido"].to_numpy(dtype=float)
        # A nivel de emisión el cuantil es un PROMEDIO: cada familia usa el
        # suyo según sus costos, así que este número puede no corresponder a
        # ningún cuantil estimado. Sirve para comparar contra la cobertura
        # agregada y para la alerta; el valor exacto por familia está en el
        # detalle, que es donde se investiga.
        usado = float(lineas["cuantil_usado"].mean())

        fila = {
            "emision": emision,
            "lineas": int(len(lineas)),
            "publicadas": int(len(pedido)),
            # Qué parte del horizonte ya se puede calificar. Sube cada día.
            "realizado": round(len(lineas) / len(pedido), 4) if len(pedido) else 0.0,
            "costo_realizado": round(
                float((por_familia["costo_realizado"] * por_familia["lineas"]).sum())
                / int(por_familia["lineas"].sum()),
                4,
            ),
            "costo_validado": round(esperado, 4) if esperado else None,
            "baremo": procedencia,
            "cuantil_usado": round(usado, 4),
            "cobertura_usada": round(cobertura(real, _cuantil_de_la_politica(lineas), usado), 4),
            "faltante_unidades": float(np.maximum(0.0, real - pedido_total).sum()),
            "sobrante_unidades": float(np.maximum(0.0, pedido_total - real).sum()),
        }
        avisos = _alertas(fila, cfg)
        fila["estado"] = "alerta" if avisos else "normal"
        resumen_emisiones.append(fila)
        alertas.extend(f"{emision}: {aviso}" for aviso in avisos)

    tabla = (
        pd.concat(detalle, ignore_index=True)
        if detalle
        else pd.DataFrame(columns=["emision", "family"])
    )
    return tabla, {
        "asof": f"{dia:%Y-%m-%d}",
        "emisiones": resumen_emisiones,
        "alertas": alertas,
    }


def _cuantil_de_la_politica(lineas: pd.DataFrame) -> np.ndarray:
    """Valor del cuantil que la política usó en cada línea.

    Cada familia puede usar un cuantil distinto, así que no hay una sola
    columna `q…` que sirva para todas: se arma la serie línea por línea.
    """
    valores = np.empty(len(lineas), dtype=float)
    posicion = pd.Series(np.arange(len(lineas)), index=lineas.index)
    for usado, bloque in lineas.groupby("cuantil_usado", observed=True):
        valores[posicion.loc[bloque.index].to_numpy()] = bloque[f"q{float(usado)}"].to_numpy()
    return valores
