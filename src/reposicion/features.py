"""Construcción de variables explicativas sin fuga temporal.

Hay **dos** fugas temporales distintas y conviene no confundirlas:

- **De disponibilidad**: usar un dato que todavía no había llegado. Se resuelve
  leyendo con `asof` (módulos `landing` e `ingesta`).
- **De ventana**: usar el valor del día que se está prediciendo, o posteriores.
  Es la que se ataca aquí, y es traicionera porque no produce un error sino un
  modelo sospechosamente bueno en validación que luego falla en operación.

La defensa es estructural: **todas las variables de la serie se calculan en el
origen** (el último día con datos disponibles) y el horizonte entra como una
variable más::

    features(O) + h  ->  predice sales[O + h]

Un `lag_1` sería incorrecto aquí: para h=1 vale `sales[O]`, pero para h=2
valdría `sales[O+1]`, que es justo lo que se quiere predecir. Anclando al
origen, ninguna ventana puede alcanzar el futuro por construcción.

El origen no se supone, se **calcula** a partir de lo que ha llegado. Si el
sistema de origen se retrasa un día, `h` crece y el resto sigue funcionando.
"""

from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

from reposicion.config import cargar_config
from reposicion.ingesta import leer_raw_asof, leer_raw_vigente
from reposicion.io import COLUMNA_LLEGADA, a_fecha
from reposicion.landing import vista_vigente

CLAVE_SERIE = ["store_nbr", "family"]

# Tipos de registro del calendario que suprimen la actividad comercial normal.
# `Work Day` es lo contrario: un sábado que se trabaja para compensar un puente.
TIPOS_FERIADO = {"Holiday", "Additional", "Transfer", "Bridge", "Event"}


# ---------------------------------------------------------------------------
# Origen
# ---------------------------------------------------------------------------


def origen_de(ventas: pd.DataFrame) -> pd.Timestamp:
    """Último día con ventas conocidas.

    Se deriva de los datos y no de la fecha de emisión porque la latencia del
    sistema de origen puede variar: si hoy no llegó la entrega, el origen es
    anterior y el horizonte se alarga solo.
    """
    if ventas.empty:
        raise ValueError("No hay ventas disponibles: no se puede fijar un origen.")
    return pd.Timestamp(ventas["date"].max())


# ---------------------------------------------------------------------------
# Matriz de la serie
# ---------------------------------------------------------------------------


def matriz_de_ventas(ventas: pd.DataFrame, origen: pd.Timestamp) -> pd.DataFrame:
    """Pivota las ventas a (fechas x series) con la rejilla de días completa.

    Densificar el calendario importa: faltan cuatro fechas en el histórico (las
    navidades, cuando cierran todas las tiendas). Sin rellenarlas, una ventana
    de 7 registros abarcaría 8 días naturales y las medias quedarían
    desalineadas respecto al calendario.

    El relleno con cero se aplica **sólo desde la primera venta observada** de
    cada serie. Rellenar antes inventaría historia para una tienda que todavía
    no había abierto, y haría que su nivel reciente pareciera más bajo de lo
    que es.
    """
    previas = ventas[ventas["date"] <= origen]
    if previas.empty:
        return pd.DataFrame()

    matriz = previas.pivot_table(
        index="date", columns=CLAVE_SERIE, values="sales", aggfunc="sum"
    )
    calendario = pd.date_range(matriz.index.min(), origen, freq="D")
    matriz = matriz.reindex(calendario)

    # `cumsum` de "ya hubo un dato" marca, por serie, desde cuándo existe.
    ya_existia = matriz.notna().cumsum() > 0
    return matriz.where(~(ya_existia & matriz.isna()), 0.0)


def _ventana(matriz: pd.DataFrame, origen: pd.Timestamp, dias: int) -> pd.DataFrame:
    """Los últimos `dias` hasta el origen, inclusive.

    Incluir el origen es correcto: sus ventas ya se conocen en el momento de
    emitir. El futuro empieza en O+1.
    """
    return matriz.loc[origen - pd.Timedelta(days=dias - 1) : origen]


def features_de_serie(
    matriz: pd.DataFrame, origen: pd.Timestamp, ventanas: list[int], semanas_mismo_dia: int
) -> pd.DataFrame:
    """Resumen del comportamiento de cada serie tal como se ve en el origen."""
    if matriz.empty:
        return pd.DataFrame()

    columnas: dict[str, pd.Series] = {
        "ventas_origen": matriz.loc[origen],
        "dias_historia": (matriz.notna() & (matriz.index <= origen)[:, None]).sum(),
    }

    for dias in ventanas:
        bloque = _ventana(matriz, origen, dias)
        columnas[f"media_{dias}"] = bloque.mean()
        columnas[f"mediana_{dias}"] = bloque.median()
        columnas[f"max_{dias}"] = bloque.max()
    # La dispersión reciente alimenta la incertidumbre: una serie errática
    # necesita un cuantil alto más separado de la mediana que una estable.
    corto = min(ventanas)
    columnas[f"desv_{corto}"] = _ventana(matriz, origen, corto).std()

    # Proporción de días sin venta: distingue un producto de baja rotación de
    # uno que simplemente vende poco, y son casos de reposición distintos.
    largo = max(ventanas)
    bloque_largo = _ventana(matriz, origen, largo)
    columnas[f"prop_ceros_{largo}"] = (bloque_largo == 0).mean()

    # Tendencia: hacia dónde va el nivel reciente respecto al de referencia.
    columnas["razon_corto_largo"] = columnas[f"media_{corto}"] / (
        columnas[f"media_{largo}"].replace(0, np.nan)
    )

    features = pd.DataFrame(columnas)
    features.index.names = CLAVE_SERIE
    return features.reset_index()


def _apilar(matriz: pd.DataFrame, origen: pd.Timestamp, semanas: int) -> pd.DataFrame:
    """Pasa la ventana de la matriz a formato largo con el día de la semana."""
    bloque = _ventana(matriz, origen, semanas * 7)
    apilado = bloque.stack(CLAVE_SERIE, future_stack=True).rename("sales").reset_index()
    apilado = apilado.rename(columns={"level_0": "date"})
    apilado["dia_semana"] = apilado["date"].dt.day_of_week
    return apilado


def cuantiles_por_dia_de_semana(
    matriz: pd.DataFrame, origen: pd.Timestamp, semanas: int, cuantiles: list[float]
) -> pd.DataFrame:
    """Cuantiles empíricos de cada serie en cada día de la semana.

    Son la predicción del modelo de referencia y, a la vez, variables muy
    informativas para el modelo de aprendizaje: resumen el nivel **y** la
    dispersión del día concreto que se va a predecir, que es exactamente lo
    que hay que estimar.

    Se usan más semanas que para la media porque un cuantil necesita masa: con
    cuatro observaciones, el 0.95 degenera en el máximo de la muestra.
    """
    if matriz.empty:
        return pd.DataFrame()

    apilado = _apilar(matriz, origen, semanas).dropna(subset=["sales"])
    agregados = (
        apilado.groupby([*CLAVE_SERIE, "dia_semana"])["sales"]
        .quantile(cuantiles)
        .unstack()
        .rename(columns=lambda q: f"emp_q{q}")
    )
    return agregados.reset_index()


def media_por_dia_de_semana(
    matriz: pd.DataFrame, origen: pd.Timestamp, semanas: int
) -> pd.DataFrame:
    """Media de cada serie en cada día de la semana, en las últimas `semanas`.

    En retail el día de la semana domina la variación: un sábado se parece
    mucho más al sábado anterior que al viernes inmediato. Esta tabla permite
    que el modelo reciba el nivel del día concreto que va a predecir.
    """
    if matriz.empty:
        return pd.DataFrame()

    return (
        _apilar(matriz, origen, semanas)
        .groupby([*CLAVE_SERIE, "dia_semana"])["sales"]
        .mean()
        .rename("media_mismo_dia")
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Calendario
# ---------------------------------------------------------------------------


def _dias_de_pago(fechas: pd.DatetimeIndex, configurados: list) -> pd.Series:
    """Marca los días de cobro. En Ecuador se paga a mitad y a fin de mes."""
    marca = pd.Series(False, index=fechas)
    for dia in configurados:
        if dia == "ultimo":
            ultimo = fechas.map(lambda f: calendar.monthrange(f.year, f.month)[1])
            marca |= fechas.day == ultimo
        else:
            marca |= fechas.day == int(dia)
    return marca


def features_calendario(fechas: pd.DatetimeIndex, dias_de_pago: list) -> pd.DataFrame:
    """Variables deducibles del calendario, conocidas con cualquier antelación."""
    df = pd.DataFrame({"date": fechas})
    df["dia_semana"] = df["date"].dt.day_of_week
    df["dia_mes"] = df["date"].dt.day
    df["mes"] = df["date"].dt.month
    df["semana_anio"] = df["date"].dt.isocalendar().week.astype(int)
    df["es_fin_semana"] = df["dia_semana"].isin([5, 6])
    df["es_dia_pago"] = _dias_de_pago(fechas, dias_de_pago).to_numpy()

    # Días transcurridos desde el último cobro: el gasto decae a lo largo de
    # la quincena, así que la distancia importa más que la marca binaria.
    #
    # El último cobro casi nunca cae dentro del horizonte: una semana de
    # pronóstico rara vez contiene un día 15 o un fin de mes. Hay que buscarlo
    # hacia atrás, en un tramo que cubra con holgura el ciclo de pago
    # anterior; limitarse a las fechas pedidas deja la columna entera en nulo.
    previas = pd.date_range(fechas.min() - pd.Timedelta(days=62), fechas.max(), freq="D")
    candidatos = previas[_dias_de_pago(previas, dias_de_pago).to_numpy()]

    df["dias_desde_pago"] = df["date"].map(
        lambda f: (f - candidatos[candidatos <= f].max()).days
        if (candidatos <= f).any()
        else np.nan
    )
    return df


def features_feriados(
    fechas: pd.DatetimeIndex, holidays: pd.DataFrame, tiendas: pd.DataFrame
) -> pd.DataFrame:
    """Marca feriados por tienda, respetando el alcance y los traslados.

    El calendario de Favorita tiene tres sutilezas que cambian el resultado:

    - `transferred=True` significa que ESE día el feriado **no** se celebró:
      se movió a otra fecha. Marcarlo como feriado sería un error.
    - `type='Transfer'` es el día al que se trasladó, y ése sí lo es.
    - `type='Work Day'` es lo contrario de un feriado: un sábado que se
      trabaja para compensar un puente.

    El alcance importa igual: un feriado local de Quito no afecta a Guayaquil,
    y tratarlo como nacional metería ruido en 53 tiendas para explicar una.
    """
    vigentes = holidays[~holidays["transferred"].astype(bool)]

    nacionales = set(
        vigentes.loc[
            (vigentes["locale"] == "National") & (vigentes["type"].isin(TIPOS_FERIADO)), "date"
        ]
    )
    laborales = set(vigentes.loc[vigentes["type"] == "Work Day", "date"])

    regionales = vigentes[
        (vigentes["locale"] == "Regional") & (vigentes["type"].isin(TIPOS_FERIADO))
    ]
    locales = vigentes[
        (vigentes["locale"] == "Local") & (vigentes["type"].isin(TIPOS_FERIADO))
    ]
    por_estado = set(zip(regionales["locale_name"], regionales["date"], strict=True))
    por_ciudad = set(zip(locales["locale_name"], locales["date"], strict=True))

    filas = []
    for _, tienda in tiendas.iterrows():
        for fecha in fechas:
            filas.append(
                {
                    "store_nbr": tienda["store_nbr"],
                    "date": fecha,
                    "es_feriado_nacional": fecha in nacionales,
                    "es_feriado_local": (tienda["city"], fecha) in por_ciudad
                    or (tienda["state"], fecha) in por_estado,
                    "es_dia_laboral_extra": fecha in laborales,
                }
            )

    df = pd.DataFrame(filas)
    df["es_feriado"] = df["es_feriado_nacional"] | df["es_feriado_local"]
    return df


# ---------------------------------------------------------------------------
# Variables exógenas
# ---------------------------------------------------------------------------


def features_petroleo(oil: pd.DataFrame, origen: pd.Timestamp, ventanas: list[int]) -> dict:
    """Nivel del crudo conocido en el origen.

    El precio no cotiza fines de semana ni feriados bursátiles, así que se
    arrastra el último valor (`ffill`): es lo que un analista tendría a la
    vista ese día. El primer registro del histórico es nulo, de modo que tras
    el arrastre puede quedar un hueco al inicio; se cubre con la mediana
    conocida para no propagar un nulo a todas las series.
    """
    if oil.empty:
        return {}

    serie = (
        oil[oil["date"] <= origen]
        .sort_values("date")
        .set_index("date")["dcoilwtico"]
        .asfreq("D")
        .ffill()
    )
    if serie.empty:
        return {}
    serie = serie.fillna(serie.median())

    valores = {"petroleo_origen": float(serie.iloc[-1])}
    for dias in ventanas:
        valores[f"petroleo_media_{dias}"] = float(serie.tail(dias).mean())
    return valores


def features_transacciones(
    transacciones: pd.DataFrame, origen: pd.Timestamp, ventanas: list[int]
) -> pd.DataFrame:
    """Afluencia reciente por tienda.

    Esta fuente llega con dos días de latencia, así que en el origen sus datos
    más recientes suelen faltar. No se rellena ni se supone: se agrega lo que
    hay. Que una fuente vaya más atrasada que otra es una situación normal de
    operación, no un error.
    """
    if transacciones.empty:
        return pd.DataFrame(columns=["store_nbr"])

    previas = transacciones[transacciones["date"] <= origen]
    if previas.empty:
        return pd.DataFrame(columns=["store_nbr"])

    salida = previas.groupby("store_nbr")["date"].max().rename("ult_fecha_trans").reset_index()
    for dias in ventanas:
        desde = origen - pd.Timedelta(days=dias - 1)
        media = (
            previas[previas["date"] >= desde]
            .groupby("store_nbr")["transactions"]
            .mean()
            .rename(f"trans_media_{dias}")
        )
        salida = salida.merge(media, on="store_nbr", how="left")

    # Cuántos días de retraso acumula la fuente en este origen: deja constancia
    # de la latencia real en vez de esconderla.
    salida["dias_retraso_trans"] = (origen - salida["ult_fecha_trans"]).dt.days
    return salida.drop(columns="ult_fecha_trans")


def features_promocion(
    ventas: pd.DataFrame, origen: pd.Timestamp, ventanas: list[int]
) -> pd.DataFrame:
    """Intensidad promocional reciente por serie.

    Las promociones del futuro no están disponibles: el sistema de origen las
    entrega junto con las ventas, con la misma latencia. En una cadena real
    vendrían del sistema de planificación comercial con semanas de antelación;
    aquí sólo se puede usar el historial reciente como aproximación, y así
    queda documentado en DECISIONES.md.
    """
    previas = ventas[ventas["date"] <= origen]
    if previas.empty or "onpromotion" not in previas.columns:
        return pd.DataFrame(columns=CLAVE_SERIE)

    salida = previas[CLAVE_SERIE].drop_duplicates()
    for dias in ventanas:
        desde = origen - pd.Timedelta(days=dias - 1)
        media = (
            previas[previas["date"] >= desde]
            .groupby(CLAVE_SERIE)["onpromotion"]
            .mean()
            .rename(f"promo_media_{dias}")
            .reset_index()
        )
        salida = salida.merge(media, on=CLAVE_SERIE, how="left")
    return salida


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


def datos_vigentes(asof: str | pd.Timestamp, *, raw=None) -> dict[str, pd.DataFrame]:
    """Lee de una vez todas las fuentes tal como se conocían en `asof`."""
    return {
        fuente: leer_raw_vigente(fuente, asof, raw=raw)
        for fuente in ("sales", "stores", "holidays_events", "transactions", "oil")
    }


def construir_panel(
    asof: str | pd.Timestamp,
    *,
    dias: int | None = None,
    raw=None,
    config: dict | None = None,
    datos: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Panel de predicción: una fila por (serie, día a pronosticar).

    Sólo usa lo que había llegado en `asof` y, dentro de eso, sólo información
    anterior o igual al origen. El resultado no contiene la variable objetivo:
    para entrenar se añade por separado con `agregar_objetivo`.

    `datos` permite reutilizar fuentes ya leídas y recortadas al `asof`
    correspondiente. Generar los ejemplos de entrenamiento exige construir un
    panel por cada origen simulado, y releer tres millones de filas cada vez
    haría el entrenamiento inviable.
    """
    cfg = config or cargar_config("model")
    ventanas = cfg["features"]["ventanas"]
    horizonte = dias or cfg["pronostico"]["dias"]

    fuentes = datos if datos is not None else datos_vigentes(asof, raw=raw)
    ventas = fuentes.get("sales", pd.DataFrame())
    if ventas.empty:
        return pd.DataFrame()

    emision = pd.Timestamp(asof).normalize()

    # El origen nunca puede alcanzar la fecha de emisión: las ventas del día en
    # que se emite todavía no han terminado de ocurrir, y mucho menos de
    # transmitirse.
    #
    # El tope no es decorativo. La carga histórica se ingiere como una única
    # partición con fecha de llegada muy anterior, de modo que al reconstruir
    # una emisión pasada el filtro por `arrival_date` no la recorta: sin este
    # tope, simular una emisión de febrero daría como origen el final del
    # histórico, y el modelo entrenaría con meses de futuro obteniendo
    # métricas magníficas y sin valor.
    origen = min(origen_de(ventas), emision - pd.Timedelta(days=1))

    # Se pronostica la ventana que arranca en la fecha de emisión; si el origen
    # viene retrasado, `h` lo absorbe y el horizonte efectivo se alarga.
    objetivo = pd.date_range(emision, periods=horizonte, freq="D")

    matriz = matriz_de_ventas(ventas, origen)
    serie = features_de_serie(matriz, origen, ventanas, cfg["features"]["semanas_mismo_dia"])
    if serie.empty:
        return pd.DataFrame()

    # Esqueleto: cada serie cruzada con cada día a pronosticar.
    panel = serie.merge(pd.DataFrame({"date": objetivo}), how="cross")
    panel["origen"] = origen
    panel["h"] = (panel["date"] - origen).dt.days

    calendario = features_calendario(objetivo, cfg["features"]["dias_de_pago"])
    panel = panel.merge(calendario, on="date", how="left")

    por_dia = media_por_dia_de_semana(matriz, origen, cfg["features"]["semanas_mismo_dia"])
    if not por_dia.empty:
        panel = panel.merge(por_dia, on=[*CLAVE_SERIE, "dia_semana"], how="left")

    empiricos = cuantiles_por_dia_de_semana(
        matriz,
        origen,
        cfg["features"].get("semanas_cuantiles", 12),
        cfg["pronostico"]["cuantiles"],
    )
    if not empiricos.empty:
        panel = panel.merge(empiricos, on=[*CLAVE_SERIE, "dia_semana"], how="left")

    tiendas = fuentes.get("stores", pd.DataFrame())
    if not tiendas.empty:
        panel = panel.merge(
            tiendas[["store_nbr", "city", "state", "type", "cluster"]],
            on="store_nbr",
            how="left",
        )
        holidays = fuentes.get("holidays_events", pd.DataFrame())
        if not holidays.empty:
            holidays = holidays.assign(date=a_fecha(holidays["date"]))
            panel = panel.merge(
                features_feriados(objetivo, holidays, tiendas), on=["store_nbr", "date"], how="left"
            )

    panel = panel.merge(
        features_transacciones(fuentes.get("transactions", pd.DataFrame()), origen, ventanas),
        on="store_nbr",
        how="left",
    )
    panel = panel.merge(
        features_promocion(ventas, origen, ventanas), on=CLAVE_SERIE, how="left"
    )

    for nombre, valor in features_petroleo(
        fuentes.get("oil", pd.DataFrame()), origen, ventanas
    ).items():
        panel[nombre] = valor

    return panel.sort_values([*CLAVE_SERIE, "date"]).reset_index(drop=True)


def agregar_objetivo(panel: pd.DataFrame, ventas_reales: pd.DataFrame) -> pd.DataFrame:
    """Añade las ventas observadas al panel, para entrenar o evaluar.

    Se usa con ventas leídas **después** de que ocurrieran: al entrenar sobre
    el pasado eso es legítimo, porque el objetivo de un ejemplo histórico ya se
    conoce. Las variables explicativas siguen viniendo sólo del origen.
    """
    objetivo = ventas_reales[[*CLAVE_SERIE, "date", "sales"]].rename(columns={"sales": "y"})
    return panel.merge(objetivo, on=[*CLAVE_SERIE, "date"], how="inner")


def construir_entrenamiento(
    asof: str | pd.Timestamp,
    *,
    raw=None,
    config: dict | None = None,
    progreso: bool = False,
) -> pd.DataFrame:
    """Genera ejemplos históricos simulando emisiones pasadas.

    La distinción que hace honesto el entrenamiento:

    - Las **variables explicativas** salen de lo que se sabía en cada emisión
      simulada. Si la tienda 44 reportó ceros ese día, el ejemplo lleva ceros:
      es exactamente lo que el modelo verá en producción.
    - La **variable objetivo** sale de la verdad final, ya corregida. Es lo que
      de verdad ocurrió, y es contra eso que debe aprender.

    Entrenar las variables con los datos corregidos sería fuga: el modelo
    aprendería con una calidad de información que nunca tendrá al predecir.
    """
    cfg = config or cargar_config("model")
    ent = cfg["entrenamiento"]

    # Historial completo de lo recibido, con su fecha de llegada, leído una
    # sola vez: cada origen se obtiene recortando esto en memoria.
    crudos = {
        fuente: leer_raw_asof(fuente, asof, raw=raw)
        for fuente in ("sales", "stores", "holidays_events", "transactions", "oil")
    }
    if crudos["sales"].empty:
        return pd.DataFrame()

    verdad = vista_vigente(crudos["sales"], "sales")
    origenes = origenes_de_entrenamiento(
        pd.Timestamp(asof).normalize(),
        int(ent["cadencia_origenes_dias"]),
        int(ent["ventana_dias"]),
    )

    paneles: list[pd.DataFrame] = []
    for numero, emision in enumerate(origenes, 1):
        datos = {
            fuente: vista_vigente(df[df[COLUMNA_LLEGADA] <= emision], fuente)
            for fuente, df in crudos.items()
            if not df.empty
        }
        if datos.get("sales", pd.DataFrame()).empty:
            continue

        panel = construir_panel(emision, config=cfg, datos=datos)
        if panel.empty:
            continue

        con_objetivo = agregar_objetivo(panel, verdad)
        if not con_objetivo.empty:
            paneles.append(con_objetivo)
        if progreso:
            print(f"  origen {numero}/{len(origenes)} {emision:%Y-%m-%d}: {len(con_objetivo):,}")

    if not paneles:
        return pd.DataFrame()

    entrenamiento = pd.concat(paneles, ignore_index=True)

    # Series sin historia suficiente no entran: no hay con qué estimar su
    # nivel ni su estacionalidad, y meterlas sólo añade ruido. En producción
    # esas series caen al modelo de referencia.
    minimo = int(ent.get("minimo_dias_historia", 0))
    return entrenamiento[entrenamiento["dias_historia"] >= minimo].reset_index(drop=True)


def origenes_de_entrenamiento(
    hasta: pd.Timestamp, cadencia_dias: int, ventana_dias: int
) -> list[pd.Timestamp]:
    """Fechas de emisión simuladas para generar ejemplos de entrenamiento.

    Se usan orígenes espaciados como las emisiones reales (semanales) para que
    el modelo entrene en las mismas condiciones en que va a operar, en vez de
    sobre una mezcla de horizontes que nunca se dan en producción.
    """
    desde = hasta - pd.Timedelta(days=ventana_dias)
    return list(pd.date_range(desde, hasta, freq=f"{cadencia_dias}D"))
