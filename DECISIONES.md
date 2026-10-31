# Decisiones técnicas y de producto

Registro de las decisiones que no se deducen leyendo el código: qué se eligió, qué se descartó
y por qué. Donde hay números, salen de medir sobre los datos reales del proyecto, no de la
intuición ni de la costumbre.

---

## 0. Los tres modelos y por qué gana el que gana

Validación de origen móvil, 3 pliegues de 28 días, 1.297.296 ejemplos de 105 orígenes
semanales. Los tres modelos ven exactamente los mismos pliegues y los mismos ejemplos.

Las cifras son las que produce el pipeline y están registradas en
`reportes/asof=2017-08-14/metricas.json`, con el hash de todas sus entradas en `dvc.lock`. No
son de una medición a mano: se pueden volver a obtener con `dvc repro`.

| Modelo | Costo negocio | Pinball | MAE | RMSLE | Cob. 0,9 | Cob. 0,95 |
|---|---|---|---|---|---|---|
| **lgbm cuantílico** | **105,95** | **20,21** | **58,74** | **0,401** | 0,903 | 0,951 |
| baseline estacional | 130,52 | 25,22 | 69,12 | 0,507 | 0,858 | 0,909 |
| lineal cuantílico | 263,06 | 40,09 | 86,16 | 0,434 | 0,953 | 0,978 |

```
lgbm   vs baseline: PROMOVER  (p1: +14,7 %, p2: +17,5 %, p3: +23,8 %)
lineal vs baseline: MANTENER  (p1: -169,6 %, p2: -94,5 %, p3: -43,0 %)
```

### El resultado que justifica toda la arquitectura de decisión

El modelo lineal **gana al baseline en RMSLE** (0,434 frente a 0,507) y **pierde en costo de
negocio** (263 frente a 131). Las dos métricas señalan modelos distintos, y en direcciones
opuestas.

Elegir por el error estadístico habría promovido el lineal y duplicado el costo de operación.
Es la demostración empírica de por qué el criterio de promoción está definido sobre el costo y
no sobre el error, y de por qué el enunciado pide una métrica de negocio además de las técnicas.

La explicación está en la cobertura: el lineal sobrecubre (0,978 donde debería dar 0,95 y 0,953
donde debería dar 0,90), es decir predice cuantiles demasiado altos. En error relativo eso
penaliza poco; en inventario real se traduce en merma. El RMSLE mide la distancia en escala
logarítmica, donde pasarse en una serie grande apenas cuenta; el costo la mide en unidades, que
es donde el dinero ocurre.

### El pipeline invalidó una cifra que yo ya había escrito

Conviene dejarlo escrito porque es el primer beneficio concreto que dio DVC, y no fue el
esperado.

La primera versión de esta tabla daba al lineal un costo de **186,97**, medido con un script de
exploración. La primera ejecución del pipeline dio **263,06** con el mismo código, el mismo
panel y los mismos pliegues: el baseline y el lgbm coincidieron hasta el quinto decimal, así
que el panel era idéntico y la diferencia estaba sólo en el lineal.

Buscando la causa apareció algo peor que un número mal copiado. Reentrenando el lineal en el
pliegue 2 cambiando **únicamente el orden de las filas** del panel de entrenamiento:

| Orden de las filas | Costo | Pinball | RMSLE |
|---|---|---|---|
| el del panel (ordenado por serie y fecha) | 222,18 | 32,93 | 0,4089 |
| barajado, semilla 0 | 135,19 | 24,65 | 0,4009 |
| barajado, semilla 1 | 167,65 | 27,42 | 0,4048 |

El modelo es **determinista** —misma entrada, misma salida, y hay una prueba que lo comprueba—
pero **no es estable**: su resultado varía casi un 40 % según qué lotes vea el descenso de
gradiente. Con 60 épocas y paso 0,05 sobre 1,2 millones de filas no ha convergido, y acaba
donde lo dejó el camino de muestreo. Las dos cifras, 187 y 263, son dos tiradas de la misma
distribución; la del script no se puede recuperar porque el script no estaba versionado.

Dos consecuencias que sí importan:

1. **La cifra canónica es la del pipeline**, porque es la única con procedencia: `dvc.lock`
   registra el hash de cada entrada que la produjo. Una cifra de un script sin versionar no es
   una medición, es una anécdota. Ésta es exactamente la clase de error que el requerimiento 2
   del laboratorio existe para prevenir, y apareció en el primer `dvc repro`.
2. **La conclusión no cambia, y ahora se puede defender mejor.** En las tres tiradas el lineal
   gana al baseline en RMSLE (0,401–0,409 frente a 0,429 en ese pliegue) y pierde en costo
   (135–222 frente a 114). La contradicción entre las dos métricas no es un artefacto de una
   ejecución afortunada: se sostiene en todas.

Limitación que queda abierta, declarada: para que el número del lineal sea una medición y no una
tirada habría que hacerlo converger —más épocas o paso decreciente— y verificar que la
dispersión entre semillas desaparece. No se hizo porque el modelo no se promueve en ninguna
tirada y el costo de cómputo no se justifica; si alguna vez fuera candidato a producción, ese
sería el primer paso.

### Por qué se conserva el lineal aunque no se promueva

No entra en producción, pero sí en la comparación, por tres motivos:

1. **Audita el preprocesamiento.** Ver la sección 0.1: detectó un error de especificación que el
   modelo de árboles absorbía en silencio.
2. **Acota la no linealidad.** Que un modelo aditivo quede por detrás incluso del baseline
   —y en su mejor tirada apenas lo iguale— indica que las interacciones entre nivel, día de la
   semana y horizonte son esenciales, no accesorias. El lgbm no gana por capacidad bruta: gana
   porque la estructura del problema es de interacciones.
3. **Es interpretable.** Sus coeficientes se leen uno por uno, y eso ya sirvió para encontrar un
   defecto que el modelo de árboles tapaba (sección 0.1). Es la pieza de diagnóstico del
   proyecto, no un candidato a producción.

### 0.1 El modelo lineal encontró un error en el preprocesamiento

En su primera medición el lineal dio un costo de 911, siete veces peor que el baseline. (Esa
cifra es de las exploratorias anteriores al pipeline, con la salvedad de la sección anterior;
el orden de magnitud es lo que importa aquí, y la diferencia con cualquier tirada posterior al
arreglo es de un factor cuatro, muy por encima de la dispersión del optimizador.) Los
coeficientes explicaron por qué: las doce variables de mayor peso eran todas indicadoras de
`family`, y **ninguna variable de nivel**. El modelo predecía el típico de cada familia
ignorando cuánto vende cada serie concreta.

La causa era un error de especificación: el objetivo es `log1p(y)` pero las variables de nivel
se pasaban en unidades originales (`media_56` recorre de 0 a 11.594). La relación real entre el
nivel de una serie y el logaritmo de sus ventas es logarítmica, y al forzar una relación lineal
el ajuste óptimo consiste en ignorar esas variables.

Tras transformar las variables en unidades a escala logarítmica, el costo bajó de 911 a 187 y
los coeficientes pasaron a ser los esperables: `emp_q0.5`, `media_mismo_dia`, `media_7`.

Lo relevante es que **el error llevaba ahí desde que se construyó el panel y el modelo de
árboles no lo manifestaba**: un GBM aprende cualquier relación monótona partiendo el rango, así
que la transformación le resulta indiferente. Hizo falta un modelo que no pudiera compensar la
mala especificación para que el defecto se hiciera visible.

Es el argumento más sólido para tener dos familias de modelos: la segunda no sirve sólo para
comparar rendimiento, sino para **auditar el preprocesamiento de la primera**.

---

## 1. Deep learning: evaluado y descartado

El enunciado permite usar modelos de aprendizaje profundo y pide justificar la decisión en
términos de costo y beneficio. Se evaluó y **se descartó**, con evidencia medida.

### Qué modelo se habría usado

De haberse adoptado, la elección habría sido un **Temporal Fusion Transformer**, no por ser el
más citado sino por una razón estructural: distingue arquitectónicamente tres tipos de
covariable, que es exactamente la estructura de este problema.

| Tipo de covariable | En este producto |
|---|---|
| Estáticas | `store_nbr`, `family`, `city`, `state`, `type`, `cluster` |
| Conocidas en el futuro | calendario, feriados, día de pago |
| Observadas sólo en el pasado | `sales`, `oil`, `transactions` |

Esa distinción es el mismo problema que aquí se resuelve anclando todas las variables al origen
(ver sección 4). TFT además tiene pérdida cuantílica multi-horizonte nativa, que coincide con la
salida que necesita el producto.

Se descartaron otras arquitecturas por motivos concretos: **DeepAR** es autoregresivo y acumula
error a lo largo de siete días; **N-BEATS** y **N-HiTS** son univariados y no aprovechan
promociones, feriados ni atributos de tienda; **PatchTST** y **TimesNet** están diseñados para
horizontes largos sobre series continuas; los modelos fundacionales (**Chronos**, **Moirai**)
ofrecen predicción sin entrenamiento pero ignoran las covariables de negocio, que son la señal
principal aquí.

### Evidencia 1 — el modelo actual ya satura

Se midió el efecto de aumentar la capacidad del modelo de gradient boosting sobre el pliegue más
reciente (1.247.400 ejemplos de entrenamiento, 49.896 de evaluación):

| Configuración | Pinball q0.5 | Pinball q0.9 | Tiempo |
|---|---|---|---|
| 100 árboles, 31 hojas | 32,76 | 18,93 | 5,8 s |
| **300 árboles, 63 hojas** (actual) | 31,29 | 17,98 | 16,0 s |
| 900 árboles, 127 hojas | **30,18** | 18,26 | 54,0 s |
| 2500 árboles, 255 hojas | 30,75 | 18,43 | 169,0 s |

Multiplicar la capacidad por ocho mejora un 3,5 %; multiplicarla por veinticinco **empeora** el
resultado. El modelo satura y después sobreajusta.

Es el argumento central: el factor limitante no es la capacidad del modelo. Un TFT aporta más
capacidad, de modo que empuja en la dirección que ya demostró no ayudar.

### Evidencia 2 — el error está en la escala, no en la dificultad

| Escala de la serie | Filas | Error medio | % del error total |
|---|---|---|---|
| casi nula (<1) | 7.292 | 0,28 | 0,07 % |
| baja (1–10) | 10.613 | 3,50 | 1,19 % |
| media (10–100) | 12.813 | 11,80 | 4,84 % |
| alta (>100) | 19.178 | 152,87 | **93,90 %** |

Las series intermitentes (más del 30 % de días sin venta) son el **17 % de los casos y aportan
el 0,1 % del error absoluto**. El dato importa porque la intermitencia es uno de los escenarios
donde el aprendizaje profundo sí suele aportar —DeepAR con verosimilitud negativa binomial, por
ejemplo—; aquí ese beneficio sería nulo.

El error se concentra en las series de mayor volumen y es proporcional a su magnitud: fallar un
10 % en una serie que vende 1.500 unidades son 150 unidades. Eso es ruido multiplicativo, no
estructura oculta que una arquitectura más expresiva pueda descubrir.

### Evidencia 3 — apenas queda estructura temporal sin capturar

Autocorrelación media de los residuos por serie:

```
lag 1: +0,1513
lag 2: +0,0276
lag 7: +0,0528
```

Los lags 2 y 7 están prácticamente en cero: la estacionalidad semanal y la persistencia de medio
plazo ya están explicadas.

El lag 1 (+0,15) sí es señal real, y conviene decir de dónde viene: es consecuencia del esquema
de horizonte directo. Al predecir `h=7`, el modelo no sabe qué ocurrió en los días `h=1..6` del
propio horizonte, porque trata cada día de forma independiente. Una arquitectura secuencial
podría modelar esa dependencia.

Dicho con rigor: una autocorrelación de 0,15 explica alrededor del 2,3 % de la varianza
residual. Es un beneficio real y pequeño.

### Costos, concretos para este proyecto

| Costo | Impacto |
|---|---|
| Dependencias | `torch` + `lightning` + `pytorch-forecasting` ≈ 2,5–3 GB; la imagen de Airflow pasaría de ~1 GB a ~5 GB |
| Tiempo de entrenamiento | 3–8 h estimadas en CPU (no hay GPU en el entorno de evaluación) frente a **27 s** del modelo actual |
| Reproducibilidad | PyTorch no es determinista por defecto. El enunciado exige reproducir un pronóstico publicado en el pasado, lo que obligaría a `use_deterministic_algorithms`, semillas en todas las fuentes de aleatoriedad y control de versiones de CUDA, sin garantía plena entre entornos |
| Artefactos | pesos de 50–200 MB por versión frente a ~5 MB, multiplicado por cada versión almacenada en DVC |
| Riesgo de evaluación | el equipo docente levanta Airflow con Docker y ejecuta el flujo; una imagen de 5 GB y un DAG de horas convierte una ejecución rutinaria en una apuesta |

### Evidencia externa

La competencia **M5** (Walmart: ventas diarias por tienda y producto, promociones, calendario
jerárquico) es el análogo más cercano a este problema, con **42.840 series**, veinticuatro veces
las 1.782 de aquí. Los cinco primeros puestos en M5 Accuracy usaron gradient boosting, y los
métodos ganadores de M5 Uncertainty tampoco fueron de aprendizaje profundo. Si el beneficio no
apareció con 42.000 series, es menos probable que aparezca con 1.782.

### Decisión

**No se adopta deep learning.** El beneficio medible está acotado por debajo del 4 %, y al
comprobarlo con la métrica real ni siquiera ese margen se sostiene (ver evidencia 4), mientras
que el costo incluye multiplicar por cinco el tamaño de la imagen, pasar de segundos a horas de
entrenamiento y comprometer la reproducibilidad, que es un requisito explícito del producto.

### Evidencia 4 — ni siquiera más capacidad en el modelo actual se sostiene

El experimento de capacidad sugería subir a 900 árboles para capturar un 3,5 %. Antes de
adoptarlo se sometió esa configuración al **mismo criterio de promoción que cualquier retador**:
cuatro cuantiles, tres pliegues y costo de negocio con el cuantil crítico de cada familia.

| Modelo | Costo negocio | Pinball | RMSLE | Cobertura 0,9 | Cobertura 0,95 |
|---|---|---|---|---|---|
| **lgbm actual** (300 árb) | 105,95 | 20,21 | 0,401 | **0,903** | **0,951** |
| lgbm 900 árboles | 105,82 | 20,11 | 0,398 | 0,885 | 0,936 |

```
MANTENER - la mejora mínima observada (-2,1 %) no alcanza el umbral de 3 %
           (p1: +0,5 %, p2: -2,1 %, p3: +1,6 %)
```

El resultado deja dos lecciones.

**El experimento barato sobreestimó el beneficio.** Medido en un solo pliegue y con una métrica
aproximada, el margen parecía del 3,5 %; evaluado con la métrica real sobre tres pliegues, se
queda en 0,1 % de media y **empeora** en uno de ellos. Si el criterio de promoción se hubiera
aplicado sólo al modelo rival y no a la propia propuesta, se habría adoptado un cambio que no
mejora nada.

**Más capacidad ajusta mejor el centro y calibra peor las colas.** El modelo grande gana en
RMSLE (0,398 frente a 0,401) y pierde en cobertura de los cuantiles altos (0,885 frente a 0,903).
Para un producto cuya política opera en cuantiles altos, la calibración de la cola importa más
que el ajuste central: es un recordatorio de por qué la decisión la toma el costo de negocio y
no el error estadístico.

Y refuerza el descarte del aprendizaje profundo: si aumentar la capacidad dentro de la misma
familia de modelos ya no rinde de forma consistente, una arquitectura con mucha más capacidad
tiene aún menos probabilidades de hacerlo.

### Qué se hace en su lugar

1. **Mantener la configuración actual** (300 árboles, 63 hojas). Es la que mejor calibra los
   cuantiles altos, que es lo que usa la política de pedido, y la más barata de entrenar.
2. **Atacar la dependencia intra-horizonte con variables, no con arquitectura**: el lag 1
   residual sugiere que el modelo ignora la relación entre días del mismo horizonte. Es
   abordable con variables de interacción o un ajuste por horizonte, sin cambiar de familia de
   modelos.
3. **Revisar la decisión si cambian los supuestos**: si el catálogo creciera un orden de
   magnitud, si hubiera GPU en el entorno de ejecución o si el horizonte pasara a varias
   semanas, el balance podría invertirse. La decisión está atada a estas condiciones, no es
   permanente.

### Limitación de la medición

La columna de costo de negocio del experimento de capacidad se calculó usando el cuantil 0,9
para todas las familias, en lugar del cuantil crítico de cada una, porque sólo se entrenaron dos
cuantiles. Sirve para comparar configuraciones entre sí, pero no es comparable con las cifras de
costo de la sección de modelado, que sí usan el cuantil crítico por familia.
---

## 2. Versionamiento de datos y reproducibilidad con DVC

El requerimiento es "reproducir un pronóstico pasado". Conviene empezar por por qué Git no
alcanza, porque de ahí sale todo lo demás.

Un commit de Git describe **código**. Dice qué se ejecutó, no con qué datos ni qué produjo. Dos
personas en el mismo commit pueden obtener pronósticos distintos y ninguna puede demostrar quién
tiene razón. El problema del tamaño de los archivos es secundario: lo grave es que falta la
mitad de la descripción del resultado.

DVC añade esa mitad guardando en Git el **hash** de cada entrada y cada salida de cada etapa,
dentro de `dvc.lock`. Con eso, un commit sí describe un resultado.

### 2.1 La fecha lógica como parámetro versionado

Esta es la decisión central de la fase y la menos obvia.

El producto es *point-in-time*: todo lo que lee depende de `asof`. DVC, en cambio, tiene un
grafo **estático**: no sabe nada de fechas. Había que decidir por dónde entra la fecha.

| Alternativa | Qué pasa | Veredicto |
|---|---|---|
| Argumento de línea de comando (`dvc repro -- --asof …`) | DVC no detecta el cambio, da el pipeline por actualizado y `dvc.lock` queda describiendo una fecha distinta de la que produjo las salidas | **Descartada.** Un lock que miente es peor que no tener lock |
| Una etapa de `dvc.yaml` por fecha | 59 etapas escritas a mano, y hay que editar el archivo cada semana | **Descartada.** Ingobernable |
| `asof` en `config/pipeline.yaml`, declarado como `params` | Cambiar la fecha invalida el grafo; la fecha queda en Git y aparece literalmente en `dvc.lock` | **Elegida** |

Con la tercera, **el commit describe un pronóstico concreto**. Reproducir el del 17 de julio es
volver al commit en que ese valor era `2017-07-17`. No hay que recordar qué argumentos se
pasaron: están versionados.

### 2.2 El grafo hace demostrable el requerimiento de los costos

Un efecto que no se buscaba y que resultó ser el más útil para defender el producto. Las
dependencias declaradas son:

```
config/model.yaml  ──params──>  panel · validacion · pronostico
config/costs.yaml  ──params──>  validacion · pedido
```

La etapa `pedido` no depende del modelo ni del panel, sólo del pronóstico ya emitido y de los
costos. Así que el requerimiento 2 del producto deja de ser una afirmación del README y pasa a
ser una propiedad verificable: `dvc repro -s pedido` recalcula el pedido en **0,1 s** tras
cambiar un costo, sin tocar el modelo.

**Y aquí el grafo corrigió un error de razonamiento mío**, que vale más que la demostración en
sí. La primera versión de este documento afirmaba que cambiar un costo invalida *sólo* la etapa
`pedido`. Al ejecutarlo, `dvc status` dijo otra cosa:

```
$ dvc status                         # tras cambiar costo_quiebre de 3.0 a 9.0
validacion:
        changed deps:
                config/costs.yaml:
                        modified:           defaults
pedido:
        changed deps:
                config/costs.yaml:
                        modified:           defaults
```

Dos etapas, no una, y tiene todo el sentido: **la métrica que decide la promoción ES el costo de
negocio**. Cambiar los costos cambia la vara con la que se mide a los modelos, y por tanto puede
cambiar cuál se promueve. Declararlo es lo correcto, y verlo escrito deja clara una propiedad
del producto que es fácil pasar por alto: un costo tiene dos efectos de horizonte muy distinto
—el pedido cambia hoy, el modelo que opera puede cambiar en el siguiente ciclo de
reentrenamiento— y el grafo los separa en lugar de confundirlos.

La afirmación correcta, entonces: para obtener el pedido nuevo basta `dvc repro -s pedido`; un
`dvc repro` completo revalidará además la promoción, que es lo que corresponde.

### 2.3 Dependencia gruesa sobre el código: corrección antes que granularidad

Cada etapa declara `src/reposicion` **entero** como dependencia, no los módulos que usa.

Es deliberadamente tosco: tocar `policy.py` invalida también el entrenamiento, que no lo
necesita, y obliga a recalcular cinco minutos de panel. La alternativa —listar módulo por
módulo— es más fina y más rápida, y se descartó por una razón concreta: basta añadir un `import`
nuevo y olvidar actualizar los `deps` para que DVC dé por actualizada una salida que ya no
corresponde al código. El fallo es **silencioso** y degrada justo lo único que se le pide a
DVC. Se paga el recálculo.

La demostración de los costos no se ve afectada, porque cambiar un YAML no cambia el código.

### 2.4 El simulador de la bandeja es la etapa 0

`tools/build_landing.py` entra al pipeline como primera etapa, con los CSV de Kaggle como
dependencia. Esto no contradice la INVARIANTE 2 —"el producto no escribe en `landing/`"— porque
el simulador no es parte del producto y es el único autorizado a escribirla.

La razón para incluirlo: sin esa etapa, la bandeja sería un regalo sin procedencia. Nadie podría
comprobar que las latencias y los incidentes son los declarados en `config/sources.yaml`, y la
cadena reproducible empezaría a mitad de camino. Con ella, el pronóstico se reconstruye desde
los CSV públicos originales, cuyo hash queda fijado en `data/kaggle.dvc`.

### 2.5 Remoto en carpeta local

El remoto es una carpeta del sistema de archivos, no S3 ni Google Drive.

El criterio fue quién tiene que poder reproducir esto: el evaluador. Un remoto en la nube exige
credenciales que no se pueden versionar y que caducan; una carpeta local funciona sin configurar
nada y demuestra la misma mecánica (`dvc push`, `dvc pull`, `dvc checkout`). Y como la etapa 0
regenera la bandeja desde los CSV de Kaggle, quien clone el repositorio puede reconstruir todo
con `dvc repro` aunque no tenga acceso a la caché.

La contrapartida, explícita: este remoto no sirve para compartir entre máquinas. Para eso habría
que configurar uno real, y el único cambio sería una línea en `.dvc/config`.

### 2.6 Las métricas sí van a Git

Hay una excepción deliberada a "ninguna salida en el repositorio": la carpeta `reportes/`, que
contiene el resumen de métricas y la decisión de promoción de cada emisión, declarados en
`dvc.yaml` con `cache: false`.

No contradice el requerimiento: no son datos ni modelos, son unos kilobytes de JSON. El motivo
es que la pregunta "¿con qué métricas se promovió este modelo?" tiene que poder contestarse
leyendo el historial de Git, sin descargar la caché de DVC, y porque así el cambio de métricas
aparece en el diff del Pull Request, que es donde se revisa.

### 2.7 El determinismo es una propiedad que hay que construir

Todo lo anterior se apoya en un supuesto: que las mismas entradas producen las mismas salidas.
Si no fuera cierto, `dvc repro --force` daría un hash distinto al registrado y la
reproducibilidad sería una afirmación sin respaldo.

No es gratis. LightGBM y el modelo lineal tienen aleatoriedad —submuestreo de filas y columnas,
inicialización y orden de los lotes en el descenso de gradiente— y ambos fijan semilla. Para que
eso sea una garantía y no una esperanza, hay una prueba que entrena cada modelo dos veces y
compara el **md5 del artefacto serializado**
(`tests/test_registro.py::test_entrenar_dos_veces_da_el_mismo_artefacto`). Que se rompa en
`pytest` y no en la defensa.

---

### 2.8 La trampa de `.gitignore` que habría dejado DVC en nada

Apareció al registrar los CSV de Kaggle y merece quedar escrita porque es la segunda vez que
el mismo mecanismo rompe algo en este proyecto.

El `.gitignore` traía, de la fase 0:

```gitignore
/data/
!*.dvc          # "pero los punteros de DVC sí van a Git"
```

La excepción **no tiene ningún efecto**. Git no puede reincluir un archivo si un directorio
padre está excluido, y `/data/` excluye el directorio entero. Comprobado antes de arreglarlo:

```
$ git check-ignore -v data/kaggle.dvc
.gitignore:18:/data/    data/kaggle.dvc
```

Es decir: `dvc add data/kaggle` habría funcionado, habría dicho que todo salió bien, y el
puntero jamás habría llegado al repositorio. Quien clonara el repo no tendría forma de saber
qué datos usó el pipeline, y todo el versionamiento con DVC sería decorativo. Sin error, sin
aviso.

El arreglo aparente es excluir el **contenido** en vez de la carpeta, que sí admite excepciones:

```gitignore
/data/*
!/data/*.dvc
```

Y eso arregló Git, pero rompió DVC de otra forma, que tardó más en aparecer.

#### El segundo fallo: DVC no entra en carpetas ignoradas por Git

Con `/data/*`, el puntero sí llegaba al repositorio. Pero al volver a un commit anterior para
reproducir una emisión pasada, `dvc checkout` **borró los 123 MB de CSV de Kaggle**:

```
$ uv run dvc checkout
...
D       data/kaggle/
```

Y los borraba cada vez, aunque `data/kaggle.dvc` estuviera ahí mismo y `dvc status
data/kaggle.dvc` dijera que todo estaba al día. La traza detallada dio la causa:

```
$ uv run dvc checkout -vv
DEBUG: Removing '.../data/kaggle' as unused link.
...
TRACE:  199.58 ms in collecting stages from .../lab1-reposicion
TRACE:  249.00 mks in collecting stages from .../lab1-reposicion/.github
TRACE:    5.20 mks in collecting stages from .../lab1-reposicion/config
TRACE:    1.30 mks in collecting stages from .../lab1-reposicion/dags
TRACE:  236.90 mks in collecting stages from .../lab1-reposicion/reportes
TRACE:  276.40 mks in collecting stages from .../lab1-reposicion/src
TRACE:   12.90 mks in collecting stages from .../lab1-reposicion/tests
```

El recorrido visita `.github`, `config`, `dags`, `reportes`, `src`, `tests` — **y nunca `data`**.
DVC no desciende a directorios ignorados por Git cuando busca archivos `.dvc`. Así que no
encontraba el puntero, no consideraba `data/kaggle` parte de la salida esperada, y lo retiraba
"por enlace sin usar". Lógico desde dentro de DVC, y destructivo desde fuera.

Primero probé la hipótesis equivocada —que el conflicto venía de declarar `data/kaggle` a la vez
como salida de un `.dvc` y como dependencia de la etapa `bandeja`— y la descarté quitando la
dependencia: seguía borrándolo. Fue la traza, no el razonamiento, la que dio la respuesta.

#### El arreglo

Enumerar las subcarpetas, de modo que `data/` quede visible para el recorrido de DVC:

```gitignore
/data/kaggle/
/data/raw/
/data/procesado/
/data/pronosticos/
/data/pedidos/
/data/seguimiento/
/data/cuarentena/
```

`landing/` y `models/` sí pueden ir en bloque: son salidas de etapas, su hash vive en
`dvc.lock` y no hay ningún `.dvc` dentro que DVC deba encontrar.

La contrapartida, declarada: una subcarpeta nueva bajo `data/` no quedaría cubierta. De eso
avisa `test_no_hay_datos_ni_modelos_en_el_repositorio`, que falla si Git rastrea algo que no sea
un puntero.

#### Lo que hay que llevarse de aquí

Tres veces en este proyecto —`models/` sin anclar en la fase 5, y las dos de arriba— un patrón
de `.gitignore` rompió algo importante **sin lanzar ningún error**, y las tres veces el síntoma
apareció lejos de la causa. La regla práctica: **ninguna regla de `.gitignore` se da por
entendida leyéndola; se verifica con `git check-ignore`.** Y si un archivo tiene que ser visible
para otra herramienta, hay que comprobar que esa herramienta lo ve, no sólo que Git lo deja
pasar.

Hay dos pruebas que lo vigilan en cada `pytest`: una comprueba que ningún `.py` esté ignorado y
la otra que ningún dato esté versionado.

