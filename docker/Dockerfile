# =============================================================================
# Imagen de Airflow con las dependencias del producto
# =============================================================================
# La imagen oficial de Airflow no trae pandas, LightGBM ni pandera, así que
# hay que extenderla. Las dos alternativas y por qué se descartaron:
#
#   `_PIP_ADDITIONAL_REQUIREMENTS` en el compose -> instala en cada arranque
#   de cada contenedor. La propia documentación de Airflow dice que es sólo
#   para pruebas rápidas: tarda minutos cada vez y la versión instalada
#   depende de qué haya en PyPI ese día, que es lo contrario de reproducible.
#
#   Instalar el paquete dentro de la imagen (`pip install .`) -> la imagen
#   llevaría una copia del código. Entonces "lo que corre Airflow" sería la
#   versión que se congeló al construir, no la del commit que está
#   desplegado, y reconstruir la imagen pasaría a ser un paso obligatorio
#   antes de cada prueba.
#
# Lo que se hace: **la imagen lleva las dependencias y el repositorio se monta
# como volumen**. Las dependencias cambian con `uv.lock` (raro, y entonces se
# reconstruye); el código cambia a diario y se lee del mismo árbol de trabajo
# que Git y DVC están versionando. Así "lo que corre Airflow" es, literal, el
# commit que está en el disco.
#
# Las versiones salen de `requirements.txt`, exportado de `uv.lock` con
#
#     uv export --no-dev --no-emit-project --no-hashes \
#               --format requirements-txt -o docker/requirements.txt
#
# de modo que el contenedor usa EXACTAMENTE las versiones con las que se
# desarrolló y se probó. No es un detalle: el proyecto ya se topó cuatro veces
# con que pandas infiere `datetime64[us]` donde el dominio exige `[ns]`, y una
# versión distinta dentro del contenedor reabriría ese tipo de diferencia en
# el peor sitio posible, con el pipeline ya corriendo.
#
# La etiqueta está fijada (`3.3.2-python3.12`) y no es `latest`: Python 3.12
# es el que declara `pyproject.toml`, y los artefactos del registro de modelos
# son pickles, que no cruzan versiones de Python.
# =============================================================================
FROM apache/airflow:3.3.2-python3.12

# -----------------------------------------------------------------------------
# Dependencia NATIVA de LightGBM
# -----------------------------------------------------------------------------
# LightGBM es C++ y paraleliza con OpenMP: al importarse carga `libgomp.so.1`
# en tiempo de ejecución. La imagen de Airflow no la trae, así que sin esta
# línea `pip install lightgbm` termina en verde y la tarea muere más tarde con
#
#     OSError: libgomp.so.1: cannot open shared object file
#
# Se descubrió aquí y no antes porque la rueda de LightGBM para Windows
# empaqueta el runtime de OpenMP dentro del `.whl`: en el equipo de desarrollo
# la biblioteca estaba siempre disponible. Es el caso de libro de "funciona en
# mi máquina", y la lección es que una lista de dependencias de Python no
# describe el entorno completo de un paquete con extensiones nativas.
USER root
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
USER airflow

COPY requirements.txt /tmp/reposicion-requirements.txt

# `--no-cache-dir` deja la imagen ~300 MB más pequeña: la caché de pip no
# sirve de nada en una imagen que ya está construida.
RUN pip install --no-cache-dir -r /tmp/reposicion-requirements.txt
