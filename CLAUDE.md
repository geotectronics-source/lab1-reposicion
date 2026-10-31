# CLAUDE.md — Reglas del proyecto

Motor de reposición de inventario: pronóstico cuantílico diario por tienda × familia y pedido
sugerido semanal, orquestado con Airflow 3 y versionado con Git + DVC.

Este archivo es el contrato de trabajo para cualquier agente de programación que toque el
repositorio. Las reglas marcadas como **INVARIANTE** no se negocian: romperlas invalida el
producto aunque las pruebas pasen.

---

## 1. Invariantes del dominio

### INVARIANTE 1 — Corrección punto-en-el-tiempo (*point-in-time*)
Un pronóstico emitido para la fecha lógica `D` sólo puede usar información que **había llegado**
en `D`. En la práctica:

- Toda lectura de la bandeja filtra por `arrival_date <= D`. Nunca por `date <= D`.
- `date` es *a qué día se refieren los datos*; `arrival_date` es *cuándo los supimos*. No son
  intercambiables y confundirlas es el error más caro del proyecto.
- Prohibido `datetime.now()`, `date.today()`, `pd.Timestamp.now()` y equivalentes en `src/` y en
  `dags/`. La única fuente de "hoy" es la fecha lógica que Airflow inyecta (`logical_date`) o el
  parámetro `asof` de la función. Hay una prueba que falla si aparecen.

### INVARIANTE 2 — La bandeja es un sistema externo de sólo lectura
`landing/` simula el sistema de origen de la empresa. El producto **la lee y jamás la escribe,
mueve, renombra ni borra**. Nada en `src/` ni en `dags/` abre `landing/` en modo escritura.
El único código autorizado a escribir ahí es `tools/build_landing.py`, que es el simulador del
sistema origen y **no forma parte del producto**.

### INVARIANTE 3 — Idempotencia
Ejecutar una tarea N veces produce el mismo resultado que ejecutarla una vez.

- La salida se particiona por la fecha lógica del run.
- Se escribe a un archivo temporal y se renombra al destino (escritura atómica); nunca se escribe
  en el lugar final de forma incremental.
- Prohibido el *append* a un archivo de salida existente. Una partición se reescribe completa.

### INVARIANTE 4 — El histórico es inmutable y auditable
Las correcciones **no sobrescriben** lo ya recibido. Un dato corregido llega como un registro
nuevo con `arrival_date` posterior. La vista "vigente" se resuelve en lectura con
*last-write-wins* por clave de negocio según `arrival_date`. Esto permite responder
"¿qué sabíamos el 15 de julio?" meses después.

### INVARIANTE 5 — Dato inválido ≠ situación de negocio
Son cosas distintas y se tratan distinto:

| Situación | Clasificación | Acción |
|---|---|---|
| Columna faltante, tipo incorrecto, fecha ilegible, PK duplicada | **dato inválido** | cuarentena + la tarea falla |
| Ventas en 0 por feriado o cierre de tienda | negocio | se procesa normal |
| Tienda nueva sin historia suficiente | negocio | *fallback* al baseline |
| `oil` sin valor en fin de semana o feriado | negocio | *forward fill* documentado |
| Una fuente no llegó hoy | negocio | se usa lo último conocido; se registra el hueco |

Los datos inválidos se detienen **antes** de contaminar el resultado. Las situaciones esperadas
del negocio **nunca** provocan una falla del DAG.

---

## 2. Reglas de arquitectura

- **Los DAGs sólo orquestan.** Un archivo en `dags/` define tareas, dependencias y
  programación. Toda la lógica vive en `src/reposicion/` e se importa. Si un DAG tiene más de
  ~15 líneas de lógica dentro de una tarea, esa lógica está en el lugar equivocado.
- **Todo lo que es decisión de negocio vive en `config/`, no en el código.** Costos, horizonte,
  cuantiles, umbrales de validación y criterio de promoción de modelos son YAML versionado.
  Cambiar un costo debe cambiar la recomendación sin tocar un `.py`.
- **Sin rutas absolutas** en `src/` ni en `dags/`. Las rutas se resuelven desde la configuración
  o desde variables de entorno, para que el proyecto funcione igual en Windows, en Linux y
  dentro del contenedor de Airflow.
- **Sin estado global mutable.** Las funciones reciben lo que necesitan y devuelven resultados.

## 3. Reglas de datos

- Formato de intercambio interno: **Parquet** (preserva tipos; CSV no).
- La bandeja llega en CSV porque así llega de un sistema real.
- Clave de negocio de ventas: `(date, store_nbr, family)`.
- Nombres en `snake_case`, en el idioma del dato de origen (`store_nbr`, `family`, `sales`,
  `onpromotion`) para no romper la trazabilidad con Kaggle.
- Fechas siempre `datetime64[ns]` sin zona horaria; el negocio opera en una sola zona (Ecuador).

## 4. Reglas de modelado

- Todo modelo compite contra el **baseline** bajo el **mismo esquema de validación temporal**
  (origen móvil, mismos folds, mismas series).
- Prohibido validar con `train_test_split` aleatorio o `KFold` sin orden temporal.
- Las features de ventana (lags, medias móviles) se calculan **sólo con pasado**: cualquier
  ventana centrada o `shift` negativo es un defecto.
- La métrica que decide la promoción de un modelo es el **costo de negocio**, no el error
  estadístico. El error estadístico se reporta, pero no manda.

## 5. Reglas de Git

- `main` está protegida. Nada se commitea directo a `main`.
- Una rama y un Pull Request por fase: `feat/…`, `fix/…`, `docs/…`.
- Los PR requieren que `pytest` pase en GitHub Actions.
- Mensajes de commit en imperativo y en español, explicando el *por qué* cuando no es obvio.
- Nunca commitear: datos, modelos, `.venv`, credenciales, logs de Airflow.
- Un cambio **mecánico y masivo** sobre archivos que DVC hashea —reformateo, finales de línea,
  renombrado automático— lleva `dvc repro` **en el mismo commit**. DVC compara bytes, no
  significado: un cambio de espacios en blanco en `src/` invalida el lock de todas las etapas
  que dependen del código, y un lock que describe un código que ya no está es un lock que miente.

## 6. Pruebas

- Toda función en `src/` con lógica de negocio tiene prueba.
- Hay pruebas específicas e irrenunciables para:
  - que no exista fuga temporal (*leakage*) en las features;
  - que la lectura as-of ignore lo que llegó después de la fecha lógica;
  - que correr una tarea dos veces dé el mismo resultado (idempotencia);
  - que una corrección retroactiva cambie la vista vigente sin borrar el histórico;
  - que no aparezca `datetime.now()` en `src/` ni en `dags/`.
- `pytest` debe correr **sin Airflow y sin Docker**. Si una prueba necesita Airflow, está mal
  diseñada.

## 7. Comandos

```bash
uv sync                      # instalar/actualizar el entorno
uv run pytest                # pruebas
uv run ruff check . --fix    # lint
uv run dvc repro             # reproducir el pipeline de datos
```

## 8. Estilo de respuesta esperado del agente

Este proyecto es un trabajo académico: el estudiante debe poder **explicar y defender** cada
línea. Al proponer código, explicar la decisión de diseño y su alternativa descartada, no sólo
entregar el resultado. Las decisiones relevantes se registran en `DECISIONES.md`.
