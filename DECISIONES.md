# Decisiones técnicas y de producto

Registro de las decisiones que no se deducen leyendo el código: qué se eligió, qué se descartó
y por qué. Donde hay números, salen de medir sobre los datos reales del proyecto, no de la
intuición ni de la costumbre.

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

**No se adopta deep learning.** El beneficio medible está acotado por debajo del 4 % —y una
parte de ese margen se obtiene simplemente aumentando el número de árboles— mientras que el
costo incluye multiplicar por cinco el tamaño de la imagen, pasar de segundos a horas de
entrenamiento y comprometer la reproducibilidad, que es un requisito explícito del producto.

### Qué se hace en su lugar

1. **Subir el modelo a ~900 árboles y 127 hojas**: captura la mejora del 3,5 % medida, con un
   costo de 54 s frente a 16 s. Es casi todo el margen disponible, a una fracción del precio.
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
