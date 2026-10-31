# Decisiones técnicas y de producto

Registro de las decisiones que no se deducen leyendo el código: qué se eligió, qué se descartó
y por qué. Donde hay números, salen de medir sobre los datos reales del proyecto, no de la
intuición ni de la costumbre.

---

## 0. Los tres modelos y por qué gana el que gana

Validación de origen móvil, 3 pliegues de 28 días, 1.297.296 ejemplos de 105 orígenes
semanales. Los tres modelos ven exactamente los mismos pliegues y los mismos ejemplos.

| Modelo | Costo negocio | Pinball | MAE | RMSLE | Cobertura 0,95 |
|---|---|---|---|---|---|
| **lgbm cuantílico** | **105,95** | **20,21** | **58,74** | **0,401** | 0,951 |
| baseline estacional | 130,52 | 25,22 | 69,12 | 0,507 | 0,909 |
| lineal cuantílico | 186,97 | 34,49 | 105,51 | 0,436 | 0,976 |

```
lineal vs baseline: MANTENER  (p1: -100,7 %, p2: -19,3 %, p3: -8,3 %)
lgbm   vs baseline: PROMOVER  (p1: +14,7 %, p2: +17,5 %, p3: +23,8 %)
lgbm   vs lineal:   PROMOVER  (p1: +57,5 %, p2: +30,8 %, p3: +29,6 %)
```

### El resultado que justifica toda la arquitectura de decisión

El modelo lineal **gana al baseline en RMSLE** (0,436 frente a 0,507) y **pierde en costo de
negocio** (187 frente a 131). Las dos métricas señalan modelos distintos.

Elegir por el error estadístico habría promovido el lineal y empeorado el resultado del negocio
en unos 56 puntos de costo por serie y día. Es la demostración empírica de por qué el criterio
de promoción está definido sobre el costo y no sobre el error, y de por qué el enunciado pide
una métrica de negocio además de las técnicas.

La explicación está en la cobertura: el lineal sobrecubre (0,976 donde debería dar 0,95), es
decir predice cuantiles demasiado altos. En error relativo eso penaliza poco; en inventario
real se traduce en merma. El RMSLE mide la distancia en escala logarítmica, donde pasarse en
una serie grande apenas cuenta; el costo la mide en unidades, que es donde el dinero ocurre.

### Por qué se conserva el lineal aunque no se promueva

No entra en producción, pero sí en la comparación, por tres motivos:

1. **Audita el preprocesamiento.** Ver la sección 0.1: detectó un error de especificación que el
   modelo de árboles absorbía en silencio.
2. **Acota la no linealidad.** Que un modelo aditivo quede un 43 % por detrás indica que las
   interacciones entre nivel, día de la semana y horizonte son esenciales, no accesorias.
3. **Es el plan de contingencia.** Si el modelo de árboles fallara o se desajustara, existe un
   tercer escalón auditable entre el baseline y el GBM.

### 0.1 El modelo lineal encontró un error en el preprocesamiento

En su primera medición el lineal dio un costo de 911, siete veces peor que el baseline. Los
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
