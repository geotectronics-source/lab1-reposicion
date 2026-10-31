# Decisiones técnicas y de producto

Registro de las decisiones que no se deducen leyendo el código: qué se eligió, qué se descartó
y por qué. Donde hay números, salen de medir sobre los datos reales del proyecto, no de la
intuición ni de la costumbre.

El orden es el de las fases en que se tomaron, no el de lectura. Para entender el producto de
abajo arriba, el camino es **4 → 5 → 6 → 7 → 0 → 1 → 2 → 3**: primero de dónde vienen los datos,
después qué modelo opera y por qué, y al final cómo se versiona y se orquesta.

| § | Sobre qué | Lo que contesta |
|---|---|---|
| [0](#0-los-tres-modelos-y-por-qué-gana-el-que-gana) | los tres modelos | por qué gana el que gana, y por qué el criterio mide dinero |
| [1](#1-deep-learning-evaluado-y-descartado) | deep learning | por qué se evaluó y se descartó, con cuatro evidencias |
| [2](#2-versionamiento-de-datos-y-reproducibilidad-con-dvc) | DVC | cómo se reproduce un pronóstico pasado |
| [3](#3-orquestación-con-airflow-3) | Airflow | cómo se opera, y qué destapó operarlo de verdad |
| [4](#4-la-bandeja-el-tiempo-de-llegada-va-en-la-ruta) | la bandeja | cómo se separa «a qué día se refiere» de «cuándo lo supimos» |
| [5](#5-contratos-de-datos-tres-niveles-de-severidad) | contratos | qué es un dato inválido y qué es una situación de negocio |
| [6](#6-ingesta-todo-o-nada-una-fecha-por-run) | ingesta | por qué una partición es todo-o-nada |
| [7](#7-variables-explicativas-hay-dos-fugas-no-una) | variables | las dos fugas temporales, y por qué una no da error |

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

**Confirmación posterior, desde la fase 8.** El reentrenamiento que corrió Airflow sobre el panel
del 2017-08-01 —un panel distinto, generado por otra orquestación— dio al lineal un costo de
142,95. Son ya tres mediciones independientes:

| panel | costo del lineal | RMSLE del lineal | RMSLE del baseline | ¿gana en costo? |
|---|---|---|---|---|
| 2017-07-17 | 166,89 | 0,456 | 0,634 | no |
| 2017-08-01 | 142,95 | 0,429 | 0,547 | no |
| 2017-08-14 | 263,06 | 0,434 | 0,507 | no |

El costo se mueve en un rango de 143 a 263 —un 84 % de dispersión— mientras el RMSLE se queda
clavado entre 0,43 y 0,46. Es exactamente la firma de un modelo que no converge: la calidad
media del ajuste es estable y el reparto del error entre los dos lados del pedido no lo es.

Y la conclusión que importa aguanta las tres veces: **el lineal gana al baseline en RMSLE y
pierde en costo de negocio, siempre.** Eso es lo que hace que el criterio de promoción mida
costo y no error estadístico, y ahora está apoyado en tres paneles en vez de uno.

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

### 2.9 Normalizar los finales de línea invalidó el lock, y es culpa del diseño de §2.3

Error cometido en este proyecto, no hipotético.

El commit que normalizó los finales de línea a LF tocó dos archivos de `src/reposicion/` —`io.py`
y `registro.py`— sin cambiar una sola instrucción: `git diff --ignore-cr-at-eol` sale vacío. Pero
**`src/reposicion` es dependencia declarada de cinco etapas** (§2.3), y DVC hashea el contenido
de los archivos, no su significado. Al cambiar los bytes cambió el hash del directorio, y
`dvc status` quedó así:

```
ingesta:
	changed deps:
		modified:           src\reposicion
```

El commit normalizó también `dvc.lock`, de modo que el archivo parecía atendido. No lo estaba: el
lock seguía describiendo el hash del código **anterior** a la normalización, y por tanto afirmaba
que unas salidas correspondían a un código que ya no estaba en el repositorio. Es exactamente la
situación que §2.3 pretendía evitar.

#### Las dos reparaciones, y por qué una no vale

- **`dvc commit`** registra el estado actual de dependencias y salidas en el lock sin ejecutar
  nada. Habría tardado un segundo y habría sido defendible en voz alta —«sólo cambiaron los
  finales de línea, las salidas siguen valiendo»— pero es una **afirmación**, no una
  demostración: el lock pasaría a declarar un hash que nadie ha comprobado que este código
  produzca. Todo el valor del requerimiento 2 depende de que el lock no afirme nada que no se
  pueda regenerar.
- **`dvc repro`** vuelve a ejecutarlo todo. Cuesta unos veinte minutos y deja el lock *ganado*.

Se eligió la segunda, y de paso contestó una pregunta que la primera habría dejado abierta: tras
regenerar, **las salidas vuelven a tener exactamente los mismos hashes**. Eso demuestra que los
finales de línea del código fuente no alteran el resultado, que es lo que la vía rápida habría
dado por supuesto.

#### Lo que hay que llevarse

§2.3 declaró el costo de la dependencia gruesa sobre `src/reposicion` en abstracto: «tocar
`policy.py` invalida también el entrenamiento, que no lo necesita». Aquí está el costo en
concreto, y en su forma más incómoda: **un cambio de espacios en blanco invalida el pipeline
entero**. La decisión sigue siendo la correcta —un lock que miente anula el motivo de usar DVC—
pero conviene saber lo que se paga.

Y la regla práctica, que es nueva: **cualquier cambio masivo y mecánico sobre archivos que DVC
hashea exige `dvc repro` en el mismo commit.** Un reformateo, un cambio de finales de línea o un
renombrado de variables no son «cambios inocuos» para una herramienta que compara bytes. Si el
commit de normalización hubiera llevado el `dvc repro` dentro, no habría habido nada que
reparar.
### 2.10 El producto fija el fin de línea de sus salidas de texto

La otra cara de §2.9. Allí el problema era que DVC hashea los archivos de **código** como
dependencia; aquí es que hashea los de **salida**, y el producto los escribía distinto en cada
sistema operativo.

Ni `Path.write_text` ni `DataFrame.to_csv` fijan el fin de línea: los dos abren el archivo en
modo texto y traducen `\n` al de la plataforma. Medido sobre las salidas reales en Windows:

| salida | retornos de carro |
|---|---|
| `reportes/asof=…/metricas.json` | 41 de 41 líneas |
| `data/pedidos/asof=…/pedido_semana.csv` | 1.783 de 1.783 |
| `data/seguimiento/asof=…/validacion.csv` | una por fila |

Dos consecuencias, ninguna cosmética:

1. **El mismo pipeline producía hashes distintos en Windows y en Linux** para archivos cuyo
   contenido es idéntico. `dvc repro` en otro sistema operativo daba las etapas por obsoletas sin
   que hubiera cambiado un dato. Es el mismo síntoma que §3.15 —donde la causa es el último bit
   de un flotante y no hay nada que hacer— pero aquí la causa es evitable, y dejarla habría
   mezclado un problema real con uno autoinfligido.
2. **Los JSON de `reportes/` son salidas `cache: false` rastreadas en Git.** Cada `dvc repro` en
   Windows los dejaba marcados como modificados aunque `git diff` saliera vacío, porque
   `.gitattributes` normaliza al comparar. El árbol de trabajo quedaba permanentemente sucio, y
   un árbol que siempre está sucio es un árbol en el que nadie mira si está sucio.

La corrección es `newline="\n"` y `lineterminator="\n"` explícitos, en una constante con nombre
—`io.FIN_DE_LINEA`— para que el siguiente escritor que se añada no tenga que acordarse.

#### El simulador tenía el mismo defecto, y es el que más pesa

`tools/build_landing.py` escribe la bandeja con `to_csv` y arrastraba lo mismo: 55 retornos de
carro en `stores.csv`, 1.783 en un archivo de ventas de un día. El producto no se entera —un CSV
se parsea igual con cualquier fin de línea— pero **`landing/` es la salida de la etapa
`bandeja`**, la etapa 0, y DVC compara bytes. El mismo simulador producía un hash distinto por
plataforma, de modo que la cadena quedaba obsoleta **desde el primer eslabón**: justo lo que
§2.4 pretendía evitar al meter el simulador dentro del grafo.

El simulador declara su propia constante en vez de importar `io.FIN_DE_LINEA`. Representa un
sistema ajeno y no depende del producto (INVARIANTE 2); que las dos constantes coincidan es
correcto, y que vivan separadas también.

Y aquí la medición corrigió lo que yo daba por hecho. Esperaba que corregir el simulador
obligara a reejecutar la cadena entera, porque `bandeja` está al principio. No fue así:

```
Running stage 'bandeja':
Running stage 'ingesta':
Stage 'panel' didn't change, skipping
Stage 'validacion' didn't change, skipping
Stage 'modelo' didn't change, skipping
Stage 'pronostico' didn't change, skipping
Stage 'pedido' didn't change, skipping
```

El cambio se propagó **dos etapas y se detuvo en seco**. `data/raw` volvió con el mismo hash
—`b99f346b…`— y con eso las cinco etapas siguientes quedaron al día. Dos minutos en vez de
veinte.

La razón es que **la ingesta es el cortafuegos de la representación**: convierte CSV en Parquet
tipado, y un fin de línea no sobrevive a esa frontera. Es exactamente lo que §6.1 argumentaba al
justificar por qué se copia la bandeja a un almacén propio —«si el origen cambia de formato, sólo
cambia la ingesta»— pero esto ya no es un argumento, es una medición: DVC, comparando contenido,
enseña dónde muere un cambio de forma.

Dicho de otro modo: el defecto del simulador era real pero **no podía alterar ningún pronóstico**,
sólo el hash de dos etapas. Se corrigió igual, porque dejarlo habría convertido §2.10 en un
principio que el propio proyecto viola una etapa más arriba.

#### Comprobado entre plataformas, no sólo arreglado

Las pruebas verifican que no hay retornos de carro, lo cual en Linux es verdad por defecto y no
prueba nada: son un guardia para Windows. La pregunta que de verdad importaba —*¿el mismo
escritor produce ahora los mismos bytes en los dos sistemas?*— se contestó ejecutando los tres
escritores sobre el mismo DataFrame en Windows y dentro del contenedor Linux, que lleva las
dependencias fijadas desde `uv.lock`:

| escritor | md5 en Windows | md5 en Linux |
|---|---|---|
| `io.escribir_csv` | `9bae9670fc30fa56210d7afbd6dc72cb` | `9bae9670fc30fa56210d7afbd6dc72cb` |
| `io.escribir_json` | `2cbad5b1abaf92e2f3be7c9321c16794` | `2cbad5b1abaf92e2f3be7c9321c16794` |
| `build_landing.escribir_csv` | `9bae9670fc30fa56210d7afbd6dc72cb` | `9bae9670fc30fa56210d7afbd6dc72cb` |

Idénticos. Conviene notar el contraste con §3.15: allí la diferencia entre plataformas **no se
puede eliminar**, porque nace del orden en que se suman los flotantes dentro de LightGBM. Aquí sí
se podía, y por eso había que hacerlo: con las dos mezcladas, la primera habría quedado tapada
por la segunda, y la afirmación «esto no se puede arreglar» habría sido verdad por el motivo
equivocado.

#### Por qué la prueba mira a los escritores y no al repositorio

La tentación es comprobar que los archivos de `reportes/` no llevan CR. Daría **siempre verde** y
no probaría nada: `.gitattributes` normaliza lo que entra al índice, así que lo que se lee del
repositorio es LF por construcción, viniera como viniera del disco. La prueba llama a
`escribir_json` y `escribir_csv` y mira los bytes que dejan.

Se comprobó que falla con el código anterior antes de darla por buena. Una prueba de regresión
que nunca se vio fallar no es una prueba de regresión, es una expresión de deseo.
---

## 3. Orquestación con Airflow 3

### 3.1 Cuatro DAGs, uno por cadencia

El producto tiene siete etapas y cuatro cadencias distintas: las fuentes llegan a diario, el
pedido se emite los lunes, el desempeño se mide cada mañana y reentrenar cuesta quince minutos.

Un DAG único habría sido más simple de explicar y peor producto: obligaría a la cadencia más
fina, de modo que seis de cada siete corridas gastarían quince minutos en reentrenar para
confirmar lo que ya operaba. Y al revés: un DAG por etapa daría siete grafos de una tarea, con
siete sitios donde mirar cuando algo falla y ninguna dependencia expresada.

El criterio que se usó: **un DAG por cadencia, y dentro de cada uno las tareas que comparten
fecha lógica y tienen que pasar o fallar juntas.** De ahí salen cuatro.

Hubo una quinta idea que no sobrevivió: un `preparacion_datos` diario que construyera el panel de
entrenamiento. Al dibujar las dependencias se vio que el panel sólo lo consume la validación,
que corre una vez al mes, y que el pronóstico no lo necesita porque calcula sus propias
variables sobre el almacén. Un DAG diario de cinco minutos para un consumidor mensual. El panel
pasó a ser la primera tarea del reentrenamiento, que es quien lo usa.

### 3.2 `BashOperator` y no `PythonOperator`: una sola frontera

`PythonOperator` era la opción natural: importar `reposicion`, llamar a la función, mejores
mensajes de error y objetos por XCom. Se descartó por una razón de arquitectura y no de estilo.

Con `PythonOperator` habría **dos formas de ejecutar el producto**: la de Airflow, que llama a
funciones, y la de DVC, que llama a la línea de comando. Nada garantizaría que siguen siendo
equivalentes. El día que alguien añadiera un parámetro en la función y no en la CLI, «lo que se
reproduce» y «lo que opera» serían cosas distintas y la reproducibilidad del requerimiento 2
pasaría a ser decorativa.

Con `BashOperator` la frontera es una sola y es **verificable**: hay una prueba que falla si un
archivo de `dags/` importa `reposicion`. El precio está asumido y declarado: un fallo llega como
código de salida y un traceback en el log, no como una excepción de Python en la interfaz.

### 3.3 Airflow no llama a `dvc repro`

Parecía elegante —el pipeline ya está declarado, que lo ejecute el orquestador— y es un error por
dos motivos independientes.

El primero: `dvc repro` **se salta** la etapa cuando su salida ya está al día. Una tarea
programada tiene que correr cuando le toca, no cuando a la caché le parece; un DAG cuyas tareas
deciden solas si trabajar es un DAG que no se puede razonar.

El segundo: `dvc repro` escribe `dvc.lock`, que es un archivo de Git. Un orquestador que modifica
archivos versionados en cada corrida convierte el repositorio en un diario de ejecuciones, y el
historial deja de contar decisiones para contar cron jobs.

La división queda: **DVC reproduce, Airflow opera, y los dos ejecutan la misma orden.** Esa
última parte es la que hace que las dos primeras signifiquen algo.

### 3.4 `LocalExecutor` y no `CeleryExecutor`

El `docker-compose.yaml` oficial de Airflow trae Celery, Redis y un contenedor de worker, porque
está pensado para repartir tareas entre máquinas. Aquí la tarea más cara tarda nueve minutos y no
hay nada que repartir entre nodos.

Celery habría añadido dos servicios, un broker que puede quedarse colgado y una fuente más de
«la tarea se quedó en *queued*» sin resolver ningún problema que tengamos. `LocalExecutor`
ejecuta en subprocesos del planificador y da el paralelismo que sí necesitamos, el de dentro de
la máquina. Se limita a dos tareas simultáneas porque la etapa del panel levanta dos años de
ventas en memoria y dos de esas a la vez tumbarían el contenedor.

### 3.5 La imagen lleva las dependencias; el volumen, el código

Tres formas de meter el producto en el contenedor:

1. `_PIP_ADDITIONAL_REQUIREMENTS` en el compose. Instala en cada arranque de cada contenedor; la
   propia documentación de Airflow dice que es sólo para pruebas rápidas. Tarda minutos cada vez
   y la versión instalada depende de qué haya en PyPI ese día.
2. `pip install .` dentro de la imagen. Entonces la imagen lleva una **copia** del código, y «lo
   que corre Airflow» es la versión que se congeló al construir, no la del commit desplegado.
   Reconstruir pasaría a ser obligatorio antes de cada prueba.
3. La imagen lleva las dependencias y **el repositorio se monta como volumen**.

Se eligió la tercera. Las dependencias cambian con `uv.lock`, que es raro, y entonces se
reconstruye; el código cambia a diario y se lee del mismo árbol de trabajo que Git y DVC están
versionando. Así «lo que corre Airflow» es, literalmente, el commit que está en el disco.

Las versiones se exportan de `uv.lock` a `docker/requirements.txt`, de modo que el contenedor usa
exactamente las que se probaron. No es celo innecesario: este proyecto ya se topó cuatro veces
con que pandas infiere `datetime64[us]` donde el dominio exige `[ns]`, y una versión distinta
dentro del contenedor reabriría esa clase de diferencia con el pipeline ya corriendo.

### 3.6 `libgomp1`: la dependencia que no estaba en la lista

La primera corrida del DAG de ingesta falló así:

```
OSError: libgomp.so.1: cannot open shared object file: No such file or directory
```

LightGBM es C++ y paraleliza con OpenMP: al importarse carga `libgomp.so.1` en tiempo de
ejecución. `pip install lightgbm` terminó en verde y la tarea murió más tarde.

No apareció antes porque **la rueda de LightGBM para Windows empaqueta el runtime de OpenMP
dentro del `.whl`**: en el equipo de desarrollo la biblioteca estaba siempre disponible. Es el
caso de libro de «funciona en mi máquina», y la lección es concreta: una lista de
dependencias de Python no describe el entorno completo de un paquete con extensiones nativas.
La imagen instala `libgomp1` con `apt-get` antes de instalar nada de Python.

### 3.7 `catchup=False` y `end_date`: el backfill se pide, no se hereda

El valor por defecto de `catchup` en Airflow es `True`, y es el error caro de la herramienta:
habilitar un DAG diario con fecha de inicio en 2017 pone al planificador a crear **miles** de
corridas que nadie pidió, sobre una bandeja que no tiene nada que entregar.

Los cuatro DAGs declaran `catchup=False` y acotan su ventana con `end_date`, porque el sistema
origen es una simulación que termina el 2017-09-01. Los backfills se piden explícitamente y
acotados:

```bash
airflow backfill create --dag-id ingesta_fuentes --from-date 2017-07-18 --to-date 2017-08-21
```

En un despliegue real se quitaría el `end_date` y `catchup` dejaría de importar, porque la fecha
lógica sería la de hoy.

Hay una prueba estática que falla si un DAG deja `catchup` sin declarar.

**Trampa del rango, que costó un intento.** `--from-date` y `--to-date` son *instantes*, no días
de calendario, y el backfill sólo crea los runs cuyo disparo del cron cae dentro. Para el DAG
mensual, que dispara a las 03:00 del día 1:

```
--from-date 2017-08-01 --to-date 2017-08-01
Warning: No runs to create for Dag reentrenamiento in the range [2017-08-01 00:00:00-05:00, ...]
```

El rango es de medianoche a medianoche y las 03:00 quedan fuera. Hay que pedir
`--to-date 2017-08-02`. El aviso lo dice, pero es un `Warning` y el comando sale con código 0, de
modo que un backfill que no hizo nada parece haber funcionado.

### 3.8 El modelo vigente: dos relojes y el filtro punto-en-el-tiempo

El pedido sale cada lunes y el modelo se reentrena una vez al mes. La consecuencia es que **la
mayoría de los lunes no existe ningún `models/asof=<ese lunes>`**, y la primera versión del
pronóstico —la que escribió la fase de DVC— exigía exactamente esa carpeta. Funcionaba porque
DVC entrenaba y emitía en la misma fecha; en operación habría dejado al producto sin emitir seis
semanas de cada siete.

El arreglo obvio —cargar el último modelo registrado— tiene un defecto peor y más silencioso: un
backfill de julio usaría el modelo entrenado en agosto, y el pronóstico resultante sería **mejor
que el que el producto habría podido dar ese día**. El backfill mentiría a su favor, y el
seguimiento mediría un desempeño que en julio nadie podía conseguir.

La regla que se implementó: *el modelo vigente en `asof` es el más reciente registrado con fecha
≤ `asof`*. El filtro punto-en-el-tiempo se aplica también al registro de modelos, no sólo a los
datos.

Lo que hace que no haya dos caminos de código: cuando el pipeline de DVC acaba de entrenar para
`asof`, esa versión *es* la más reciente ≤ `asof`, de modo que la misma regla devuelve la carpeta
que DVC declaró como salida. Una regla, dos situaciones.

Detalle que costó una prueba: una versión sólo cuenta si tiene el `.pkl`. Un entrenamiento que
muere a mitad deja una carpeta con la fecha más reciente, y una búsqueda que sólo mirara nombres
la elegiría.

### 3.9 El seguimiento no es una etapa de DVC

Hay una etapa del producto que no aparece en `dvc.yaml`, y es la mejor justificación de por qué
el laboratorio pide las dos herramientas.

La entrada del seguimiento es *«todas las emisiones pasadas cuyo horizonte ya se realizó»*. No es
una arista fija entre archivos: es una consulta que depende de la fecha en que se corre. DVC
declara dependencias entre rutas concretas y no tiene forma de expresarla. Un orquestador con
fecha lógica, sí.

Sus salidas tampoco van a Git. Es un DAG diario: versionar su reporte convertiría el historial en
un registro de ejecuciones. Las decisiones de promoción sí van a Git porque son raras y se
revisan en un Pull Request; el monitoreo diario es un dato, no una decisión.

### 3.10 La alerta avisa; no interrumpe

Un modelo que se degrada **no hace fallar el DAG**. Es la aplicación directa del INVARIANTE 5:
degradarse es una situación de negocio, no un dato inválido.

El argumento concreto, más allá del principio: hacer caer la tarea de medición dejaría al día
siguiente sin medición, que es exactamente cuando hace falta mirarla. La alerta queda en el log y
en `desempeno.json`, y reentrenar lo decide una persona.

Lo que sí se dejó fuera, declarado: **escalar la alerta** —disparar el reentrenamiento
automáticamente, o mandar un correo— no está implementado. No hay SMTP en el laboratorio, y un
modelo que empeora merece una revisión humana antes de que el sistema se reentrene solo sobre
los datos que lo degradaron.

### 3.11 Pruebas estáticas de los DAGs: lo que cubren y lo que no

`pytest` tiene que correr sin Airflow (CLAUDE.md §6), así que las pruebas de los DAGs **no los
importan**: los leen como texto y los analizan con `ast`. Comprueban que ninguno importe el
producto, que ninguno contenga funciones, condicionales ni bucles, que todos declaren
`schedule`, `start_date`, `catchup` y `max_active_runs`, que el `dag_id` coincida con el nombre
del archivo y que ninguno fije una fecha en una orden.

Por qué merece la pena: «los DAGs sólo orquestan» es una regla que se respeta el primer día y se
erosiona el tercero. Alguien necesita un `if` para saltarse un domingo, luego una función para
calcular una fecha, y al mes la lógica de negocio vive en dos sitios y sólo uno tiene pruebas.
Un árbol de sintaxis detecta esa deriva el día que ocurre.

La contrapartida, declarada sin adornos: **estas pruebas no ejecutan nada**. Un DAG puede pasar
las nueve y fallar al importarse en Airflow —de hecho ocurrió, con `libgomp`. La verificación
real es `airflow dags list` dentro del contenedor, y está documentada en el README como parte de
la puesta en marcha.

### 3.12 `ds` es UTC y el cron no

Trampa que se revisó antes de que diera problemas. El cron se interpreta en la zona de
`DEFAULT_TIMEZONE` (America/Guayaquil, UTC−5), mientras que `{{ ds }}` es la fecha lógica **en
UTC**. El backfill lo confirma: la corrida del 19 de agosto tiene `logical_date` a las 11:00 UTC,
que son las 06:00 de Guayaquil.

Las cuatro programaciones del proyecto están de madrugada o por la mañana, así que al pasar a UTC
siguen cayendo en el mismo día del calendario. Un cron de las 20:00 locales serían las 01:00 UTC
del día siguiente y `ds` adelantaría un día **sin avisar**, que es una forma silenciosa de
romper la corrección punto-en-el-tiempo: el pedido del lunes se calcularía con la fecha lógica
del martes.

### 3.13 El almacén es acumulativo, y eso tensiona el grafo de DVC

Interacción entre las dos herramientas que conviene tener declarada, porque aparece en cuanto se
usan juntas de verdad.

La etapa `ingesta` declara `data/raw` como salida. Desde un árbol limpio es una función de sus
entradas: `ingesta --asof D` produce exactamente las llegadas ≤ D, y el hash que registra
`dvc.lock` es reproducible.

Pero el almacén **acumula por invariante**. `ingesta` no borra nada: el filtro
punto-en-el-tiempo se aplica al leer y no al escribir, porque borrar lo posterior destruiría el
histórico que permite auditar (INVARIANTE 4). Cuando Airflow opera —un run por día durante cinco
semanas— el almacén crece más allá del `asof` que describe `dvc.lock`, y el hash deja de
coincidir.

No es una contradicción: es que el árbol de trabajo ya no es el que el lock describe, igual que
después de editar un archivo. La vuelta es la que define DVC:

```bash
dvc checkout            # el almacén vuelve al estado que describe el lock
dvc repro               # y el pipeline se reproduce desde ahí
```

Las dos alternativas que se consideraron y por qué no:

- **Que `ingesta` limpie lo posterior a `asof`.** Haría la etapa idempotente respecto al árbol,
  y rompería el INVARIANTE 4: un backfill de julio borraría lo que se sabe de agosto, y la
  pregunta «¿qué sabíamos el 15 de julio?» dejaría de tener respuesta en cuanto alguien
  reprodujera una fecha antigua.
- **Sacar `data/raw` del grafo** y tratar el almacén como una fuente externa. El pipeline dejaría
  de ser reproducible de punta a punta, que es justo el requerimiento 2.

Se asume la contrapartida y se documenta el comando. Lo que importa es que el lock nunca
describa algo que no se pueda regenerar: con `dvc checkout` de por medio, sí se puede.

### 3.14 El contenedor no cuesta lo mismo que el equipo, y conviene medirlo

Tiempos de las tareas del reentrenamiento del 2017-08-01, dentro del contenedor, comparados con
los de la misma etapa corriendo en Windows sobre el mismo equipo:

| etapa | qué la limita | en Windows | en el contenedor | factor |
|---|---|---|---|---|
| `panel` | pandas, un hilo | 5 min 16 s | 4 min 48 s | 0,9× |
| `validar` | LightGBM, 9 ajustes | 8 min 47 s | 31 min 30 s | 3,6× |
| `modelo` | LightGBM, panel completo | 35 s | 5 min 40 s | 9,7× |

El patrón es informativo: la etapa que va limitada por pandas tarda lo mismo o menos, y las dos
que van limitadas por LightGBM tardan entre cuatro y diez veces más. El contenedor no es «más
lento» en general; lo es para lo que paraleliza.

Que el factor no sea el mismo en las dos etapas de LightGBM —3,6× y 9,7×— es algo que no
conseguí explicar, y queda dicho: no se aisló la variable.

La causa más probable, y se declara como probable y no como comprobada: **sobresuscripción de
hilos**. LightGBM crea tantos hilos como núcleos le reporta la máquina virtual de WSL2, y durante
la medición había además otro Airflow del mismo equipo compitiendo por CPU. El contador marcaba
1.848 % de uso —unos dieciocho núcleos ocupados— para ir casi cuatro veces más lento, que es la
firma de hilos peleándose en vez de trabajando.

**No se ajustó**, y el motivo es de producto, no de pereza: la palanca sería fijar
`OMP_NUM_THREADS` o `num_threads` en el compose, y afirmar que eso lo arregla exige volver a
medir media hora. La tarea corre una vez al mes a las 03:00; treinta y un minutos ahí no le
quitan el sueño a nadie. Si pasara a ser semanal, este sería el primer sitio donde mirar.

Lo que sí se deja dicho: **los tiempos de un pipeline no se heredan del equipo de desarrollo.**
Diseñar la cadencia de un DAG con los números del portátil es cómo se acaba con una tarea
nocturna que todavía corre a las nueve de la mañana.

### 3.15 La reproducibilidad byte a byte es de una plataforma, no del proyecto

El hallazgo que más matiza lo que afirmaba la fase 7.

El backfill de Airflow regeneró, dentro del contenedor Linux, la misma emisión del 2017-07-17 que
`dvc.lock` describe con el hash que produjo Windows. **Los hashes no coinciden.**

Antes de escribir nada hubo que averiguar si cambiaban los valores o sólo los bytes del archivo,
porque las dos cosas significan lo contrario. Midiendo:

| columna | filas que difieren | dif. absoluta máx. | dif. **relativa** máx. |
|---|---|---|---|
| `q0.5` | 1.760 de 12.474 (14,1 %) | 1,8 · 10⁻¹² | 3,0 · 10⁻¹⁶ |
| `q0.8` | 1.739 (13,9 %) | 1,8 · 10⁻¹² | 3,2 · 10⁻¹⁶ |
| `q0.9` | 1.704 (13,7 %) | 1,8 · 10⁻¹² | 2,7 · 10⁻¹⁶ |
| `q0.95` | 2.205 (17,7 %) | 1,8 · 10⁻¹² | 2,8 · 10⁻¹⁶ |

Una diferencia relativa de 3 · 10⁻¹⁶ es **un bit**: el épsilon de un `float64` vale
2,2 · 10⁻¹⁶. Un valor concreto:

```
windows: 4.129039181173225
linux:   4.1290391811732245
```

Es el orden en que se suman los flotantes, que depende de cómo reparta el trabajo la biblioteca
de álgebra y de cuántos hilos use. No es un defecto del producto ni una diferencia de versiones:
las dependencias están fijadas desde `uv.lock` y son idénticas.

**Pero el pedido publicado sí es idéntico, bit a bit.** Las 12.474 cantidades, el total de
6.351.166,32 unidades y las 8 líneas recortadas por el tope coinciden exactamente.

La razón es una decisión de negocio que se tomó por un motivo completamente distinto:
`politica.decimales: 2` en `costs.yaml` está ahí porque nadie despacha 0,0000000026 unidades, y
resulta que redondear a dos decimales **absorbe el ruido de la última cifra**. El redondeo que se
puso para que el pedido fuera despachable es lo que lo hace estable entre plataformas.

#### Qué se afirma y qué no

Esto obliga a precisar la afirmación de la fase 7, que decía «byte a byte» sin decir dónde:

- **La decisión es reproducible entre plataformas**, exactamente. Es la afirmación que le importa
  al negocio: el pedido que se publicó se puede volver a obtener en cualquier máquina.
- **El archivo no lo es.** Y como DVC compara hashes de archivos, `dvc repro` en otro sistema
  operativo dará todas las etapas por obsoletas y volverá a ejecutarlas. No es un fallo del
  pipeline: es una propiedad de hashear artefactos de coma flotante.
- La afirmación «byte a byte» de la fase 7 era correcta **en la misma máquina**, que es donde se
  midió, y eso es lo que hace que `dvc repro` detecte un cambio real de código. No se puede
  extender a otra plataforma, y ahora está medido en vez de supuesto.

Lo que se haría si el proyecto tuviera que garantizar identidad entre plataformas: materializar
sólo la decisión redondeada como salida versionada y dejar los cuantiles crudos fuera del hash.
No se hizo porque los cuantiles crudos son justo lo que necesita el seguimiento para medir
cobertura, y perderlos costaría más de lo que vale la identidad del archivo.

### 3.16 La alerta mira emisiones, no tendencias

El seguimiento de las cinco emisiones del backfill, medido el 2017-08-19:

| emisión | realizado | costo | validado | cobertura | baremo |
|---|---|---|---|---|---|
| 2017-07-17 | 100 % | 95,68 | 113,12 | 76,6 % | validado en 2017-07-17 |
| 2017-07-24 | 100 % | 109,58 | 113,12 | 71,8 % | validado en 2017-07-17 |
| 2017-07-31 | 100 % | 118,59 | 113,12 | 78,1 % | validado en 2017-07-17 |
| 2017-08-07 | 100 % | 124,85 | **105,89** | 81,1 % | validado en 2017-08-01 |
| 2017-08-14 | 28,6 % | 86,95 | **105,89** | 67,5 % | validado en 2017-08-01 |

Dos cosas funcionan como se diseñaron. El baremo **cambia solo** a partir del 2017-08-07, porque
`emision.json` dice que esas emisiones las atendió el modelo del 2017-08-01: cada emisión se mide
contra la promesa del modelo que de verdad la produjo. Y la cobertura se mueve alrededor del 76 %
que pide la política, entre el 67 % y el 81 %.

**Y una no.** El costo sube cuatro semanas seguidas —95,68 → 109,58 → 118,59 → 124,85, un 30 %— y
**ninguna emisión dispara la alerta**, porque cada una se compara por separado contra su baremo y
la peor se queda en +17,9 % sobre una tolerancia del 25 %.

Conviene no sobreinterpretarlo en ninguna de las dos direcciones. Con cuatro puntos no se puede
distinguir un modelo que se degrada de cuatro semanas que vinieron más difíciles: mediados de
agosto trae el feriado del Primer Grito de Independencia y un efecto de quincena, y el horizonte
se mete en ellos. Que no se pueda decidir con estos datos es precisamente el resultado.

Lo que sí queda demostrado es el límite del diseño: **una alerta que mira cada emisión por
separado no ve una tendencia.** Lo que lo arreglaría es un contraste sobre la ventana —la media
de las últimas cuatro contra el baremo, o la pendiente— y no se implementó por dos razones: con
cinco emisiones no hay con qué estimar una pendiente, y el reentrenamiento mensual ya vuelve a
fijar el baremo antes de que una tendencia tenga tiempo de consolidarse. Queda anotado como lo
primero que haría falta si el producto operara un año.

---

## 4. La bandeja: el tiempo de llegada va en la ruta

El sistema de origen simulado es la pieza de la que depende todo lo demás, porque es la que hace
que la corrección punto-en-el-tiempo sea comprobable en vez de prometida.

### 4.1 `date` y `arrival_date` son dos tiempos, y la ruta los separa

```
landing/arrival_date=2017-07-06/sales/sales_2017-07-05.csv
                     ^ cuándo lo supimos      ^ a qué día se refiere
```

Filtrar por `date <= D` parece correcto y **es** fuga temporal: el 6 de julio no se conocen
todavía las ventas del 6. Llevando la fecha de llegada en la ruta, la lectura point-in-time se
convierte en un filtro de carpetas en lugar de un acto de disciplina del programador, y un
descuido deja de ser posible.

**Alternativa descartada: una columna `arrival_date` dentro del CSV.** Se rechazó por dos
motivos. Un sistema de origen real no declara cuándo te envió un archivo —eso lo sabe el
receptor, por cuándo lo recibió—, y además obligaría a abrir y leer cada archivo para saber si
corresponde al periodo, en lugar de decidirlo por el nombre de la carpeta.

Las dos zonas se organizan al revés a propósito: la bandeja por fecha
(`arrival_date=D/fuente/`), porque así deposita un sistema real que entrega lo del día; el
almacén por fuente (`fuente/arrival_date=D/`), porque así se consulta.

### 4.2 Latencias asimétricas, puestas a propósito

| fuente | tipo | latencia | por qué |
|---|---|---|---|
| `sales` | hecho diario | D+1 | se cierran de madrugada y se transmiten al día siguiente |
| `transactions` | hecho diario | **D+2** | el POS central consolida más lento |
| `oil` | hecho diario | D+1 | sólo días hábiles: los huecos son esperados |
| `holidays_events` | catálogo | cada 30 d | con 60 de anticipación: la única fuente que informa del futuro |
| `stores` | catálogo estático | — | sólo carga inicial |

Que `transactions` vaya un día por detrás de `sales` **no es realismo decorativo**: obliga al
producto a tolerar que en la fecha de emisión una fuente esté más atrasada que otra. Un diseño
que asumiera latencia uniforme funcionaría en la simulación y se rompería contra un origen real
el primer día. La tarea `auditar` tiene un umbral distinto por fuente justamente por esto.

`holidays_events` merece su propia nota: es la única fuente cuyo último dato está en el **futuro**
de la fecha lógica. Por eso un pronóstico puede usar feriados que aún no han ocurrido sin violar
el INVARIANTE 1 —ya se sabían— y por eso su retraso, medido como `asof − último dato`, sale
negativo y se recorta a cero.

### 4.3 El corte histórico se deriva, no se fija

`corte_historico: null` en `config/sources.yaml`, calculado como *(última venta −
`dias_operacion`)*, con `dias_operacion: 46`. Con los datos públicos de Kaggle da 2017-06-30.

El enunciado advierte que el flujo se ejecutará sobre un periodo de datos **no publicado**. Un
corte fijo en `2017-06-30` dejaría la bandeja vacía o desalineada si los CSV del evaluador llegan
más lejos que los públicos. Derivarlo es lo que hace que el proyecto funcione sobre datos que no
he visto, que es la condición que de verdad hay que cumplir.

### 4.4 Los incidentes se inyectan para poder demostrar que se resisten

Dos situaciones declaradas en `config/sources.yaml`, no accidentes:

1. **Corrección retroactiva.** El POS de la tienda 44 reporta ceros el 2017-07-05 y el dato real
   se reenvía el 2017-07-20, cuando el pronóstico de esa semana ya se publicó. El archivo queda
   en **dos** `arrival_date` distintos: la corrección se añade, no sobrescribe (INVARIANTE 4).
2. **Fuente que no llega.** `transactions` no entrega el 2017-07-25 y se acumula al día
   siguiente. Es una situación de negocio y no debe hacer fallar el DAG (INVARIANTE 5).

Las dos están medidas en operación en la sección de Airflow del README: el run del 25 terminó en
verde y el del 26 recibió 108 filas —las dos entregas acumuladas—, y el reporte del 20 de julio
distingue 1.782 filas reenviadas de 30 corregidas de verdad.

### 4.5 El simulador no es parte del producto

`tools/build_landing.py` es el único código autorizado a escribir en `landing/`, y vive fuera de
`src/` a propósito. `landing/` representa el sistema de la empresa: el producto **lo lee y jamás
lo escribe** (INVARIANTE 2). Hay una prueba que falla si algo de `src/` o `dags/` abre esa ruta
en modo escritura.

La tensión que esto crea con DVC está resuelta en §2.4: el simulador es la etapa 0 del pipeline,
para que la cadena sea reproducible desde los CSV públicos y las latencias declaradas se puedan
comprobar en vez de creer.

---

## 5. Contratos de datos: tres niveles de severidad

### 5.1 El contrato salió del perfilado, no de la intuición

Antes de escribir un solo `Check` se perfilaron las cinco fuentes. Cinco hallazgos cambiaron el
diseño, y cada uno habría causado un fallo concreto si se hubiera supuesto lo contrario:

| hallazgo | consecuencia en el contrato | qué habría pasado sin él |
|---|---|---|
| `oil` tiene **43 nulos**, el primero en la fila 1 | `nullable=True` | el DAG fallaría todos los lunes: el petróleo no cotiza el fin de semana |
| `holidays`: **31 fechas con varias filas** | la clave es la tupla de 5 campos, no `date` | deduplicar por fecha perdería los feriados regionales, que explican los picos de una ciudad |
| `transactions` **no** es rejilla completa (83.488 de 90.828) | la ausencia de una tienda es negocio | una tienda cerrada un día tumbaría la ingesta |
| `sales` es rejilla perfecta: 0 nulos, 0 negativos, 0 duplicados | aquí la clave primaria sí es invariante duro | una PK duplicada pasaría sin avisar |
| `sales` tiene **462.113 valores decimales** | tipo `float`, no `int` | redondear distorsionaría las familias que se venden por peso |

### 5.2 Estructura sí, vocabulario no

El INVARIANTE 5 exige distinguir dato inválido de situación de negocio. Se concreta en tres
niveles:

| nivel | ejemplo | acción |
|---|---|---|
| estructura | falta una columna, fecha ilegible, PK duplicada | cuarentena + la tarea falla |
| valor imposible | venta negativa, petróleo a precio negativo | cuarentena + la tarea falla |
| vocabulario nuevo | tienda 99, tipo de feriado no visto | **avisa y procesa** |

**Decisión deliberada: no se usa `Check.isin([...])` sobre `type` ni `locale`.** Si el evaluador
aporta un feriado de un tipo que no estaba en los datos públicos, un vocabulario cerrado tumbaría
el DAG por una situación perfectamente normal. Se exige la estructura del dato, no el vocabulario
del negocio: el negocio puede abrir una tienda o inventarse un feriado sin pedirnos permiso.

**Regla que ordena los nulos:** las columnas de la clave de negocio y la variable objetivo no los
admiten, porque sin ellas no hay nada que calcular y no se pueden imputar sin inventar. Las
variables explicativas sí, y se imputan aguas abajo de forma documentada.

### 5.3 Las correcciones se resuelven al leer, no al recibir

`vista_vigente` aplica *last-write-wins* por `arrival_date` en el momento de la lectura, en vez de
sobrescribir cuando llega la corrección. Sobrescribir sería más simple de implementar y haría
**imposible** auditar qué se sabía antes:

```
historial_de(df, "sales", date="2017-07-05", store_nbr=44)

arrival_date   filas   suma_ventas
2017-07-06        33         0.000     <- lo que supimos primero
2017-07-20        33    46,791.723     <- la corrección
```

Las dos filas conviven. La vigente es la segunda, y la primera sigue ahí para contestar «¿con qué
datos publicamos el pedido de esa semana?» meses después. Es el INVARIANTE 4, y la operación lo
ejercita de verdad: el pronóstico del 2017-07-17 se emitió con los ceros, porque el 17 de julio
los ceros eran lo que se sabía.

---

## 6. Ingesta: todo-o-nada, una fecha por run

### 6.1 ¿Por qué copiar si la bandeja ya tiene los datos?

| razón | detalle |
|---|---|
| formato | CSV no preserva tipos; Parquet sí, y se lee mucho más rápido |
| garantía | lo que está en `data/raw/` ya pasó el contrato; lo de `landing/` no se sabe |
| propiedad | `landing/` es externo y no se versiona; `data/raw/` es nuestro y va a DVC |
| desacople | si el origen cambia de formato, sólo cambia la ingesta |

### 6.2 La partición es todo-o-nada

Si uno de los archivos de la fecha es inválido **no se escribe ninguno**, ni siquiera los
válidos.

Una partición parcial *parece* completa y no lo es: contamina en silencio, y eso es peor que
caerse. Hay prueba de que `data/raw/sales` ni siquiera llega a existir tras un rechazo.

**Alternativa descartada: escribir lo bueno y fallar.** Además de la contaminación, rompería la
idempotencia: al reprocesar tras la corrección, la partición tendría que *fusionarse* con lo
escrito antes en lugar de reescribirse entera, y el INVARIANTE 3 dejaría de valer.

### 6.3 Una sola fecha por ejecución

La ingesta del día D procesa `arrival_date == D`, no `<= D`. Cada run toca exactamente una
partición, así que repetirlo reescribe lo mismo y un backfill es un run por día **en cualquier
orden**. Hay prueba de que ingerir hacia adelante y hacia atrás produce DataFrames idénticos.

Esta decisión es la que hace que los reintentos de Airflow sean seguros sin pensar (§3.1) y la
que permitió correr 33 días de backfill sin coordinación entre runs.

El modo completo que usa DVC —rehacer todo lo llegado hasta `asof`— convive con este: la
diferencia y su tensión con el grafo están en §3.13.

### 6.4 El histórico es la partición más antigua

Se materializa con `arrival_date = 1900-01-01` en lugar de tratarse como un caso especial. Así
`leer_raw_asof` lo recoge como a cualquier otra partición y **ningún módulo aguas abajo necesita
saber que existe una carga inicial**. Un `if es_historico` en la lectura se habría propagado a
todas las fases siguientes.

### 6.5 Las fuentes ingeribles se derivan de los contratos

`FUENTES = tuple(sorted(ESQUEMAS))`, en vez de repetir la lista en un YAML. Evita el fallo de
que una fuente entre al almacén sin validarse porque alguien la añadió en un sitio y no en el
otro. La misma idea aparece en el catálogo de modelos de la CLI, que se construye leyendo el
nombre de cada clase en lugar de repetirlo.

---

## 7. Variables explicativas: hay dos fugas, no una

### 7.1 La distinción

| fuga | qué es | dónde se resuelve |
|---|---|---|
| de disponibilidad | usar un dato que aún no había llegado | la bandeja y `arrival_date` (§4) |
| **de ventana** | usar el valor del día que se predice, o posteriores | aquí |

La segunda es la traicionera: no produce un error, produce un modelo **sospechosamente bueno en
validación** que luego falla en operación. Y no la detecta ninguna prueba de contratos, porque
los datos son perfectamente válidos; sólo la detecta razonar sobre qué se conoce cuándo.

### 7.2 Anclar al origen y meter el horizonte como variable

El instinto es usar `lag_1`. Al predecir siete días se rompe de inmediato:

```
lag_1 para h=1  ->  sales[O]     conocido
lag_1 para h=2  ->  sales[O+1]   es justo lo que se quiere predecir
```

Así que **todas** las variables de la serie se calculan en el origen y el horizonte entra como
una variable más:

```
features(O) + h  ->  predice sales[O + h]
```

Un solo modelo para los siete días, sin acumulación de error y **sin que ninguna ventana pueda
alcanzar el futuro por construcción**. Eso último es lo que importa: no se evita la fuga con
cuidado, se evita porque no hay forma de expresarla.

**El origen se calcula, no se supone**: es la última fecha con ventas disponibles en `asof`. Si
el sistema de origen se retrasa un día, `h` crece y el resto sigue funcionando. Un origen fijo
en `asof − 1` habría fallado el primer día que la entrega llegara tarde.

### 7.3 Densificar el calendario, pero no inventar historia

Faltan cuatro fechas en el histórico, y son las cuatro navidades que cubre el dataset
—2013-12-25, 2014-12-25, 2015-12-25 y 2016-12-25—, cuando cierran todas las tiendas. Sin
rellenarlas, una ventana de 7 registros abarcaría 8 días naturales y las medias quedarían
desalineadas del calendario.

Pero el relleno con cero se aplica **sólo desde la primera venta observada de cada serie**.
Rellenar antes crearía historia para una tienda que no había abierto, hundiría su nivel reciente
y el pedido saldría sistemáticamente corto justo para las tiendas nuevas, que son las que menos
margen de error tienen.

### 7.4 Las tres sutilezas del calendario de Favorita

- `transferred=True` → ese día el feriado **no** se celebró: se movió. Marcarlo enseñaría al
  modelo a esperar una caída en un día laborable normal.
- `type='Transfer'` → el día al que se trasladó, y ése sí cuenta.
- `type='Work Day'` → lo contrario de un feriado: un sábado que se trabaja.

El alcance importa igual: un feriado local de Quito marcado como nacional metería ruido en 53
tiendas para explicar una.

### 7.5 Las promociones futuras no están disponibles, y se dice

El sistema de origen entrega `onpromotion` **junto con las ventas**, con la misma latencia D+1.
En una cadena real vendrían del sistema de planificación comercial con semanas de antelación, y
serían una de las variables más útiles del modelo.

Aquí sólo se usa la **intensidad promocional reciente** como aproximación: medias de
`onpromotion` en las mismas ventanas que las ventas, ancladas al origen. Es una aproximación
pobre y conviene que esté dicho, porque es la primera mejora que daría el negocio si conectara
su sistema de planificación: no un modelo mejor, un dato que hoy no llega.

Limitación declarada, no descubierta a posteriori: el modelo no puede anticipar una promoción
futura. Si operaciones lanza una campaña la semana que viene, el pedido se quedará corto, y eso
no es un defecto del modelo sino del dato disponible.
