# Aprendizaje Automático en la Nube

Repositorio para el Trabajo final de **aprendizaje automático en la nube**. Este
proyecto toma como base el repositorio [`MLOps-Course`](https://github.com/dpalacioj/MLOps-Course)
para estructura, convenciones y configuración.

Predice la concentración horaria de **PM2.5** en 12 estaciones de monitoreo de
Beijing, a partir de meteorología y otros contaminantes.

## Integrantes

- Juan Pablo Arango
- Alejandro Taborda
- Daniel Gomez
- Santiago Gomez

## Empezando

> **Requisito de shell.** Los comandos `make` están escritos para **Bash**
> (`Makefile` usa `SHELL := /bin/bash` y herramientas de Unix como `grep`,
> `awk` y `find`).
> - **macOS/Linux:** funcionan tal cual en la terminal normal.
> - **Windows:** ejecútalos desde **Git Bash**, no desde PowerShell/CMD. Desde
>   PowerShell `make` usa `cmd.exe`, que no entiende esas herramientas y falla.

```bash
make setup        # instala dependencias y los hooks de pre-commit
make check        # lint + tipos + tests, lo mismo que verifica el CI
```

En Windows: abrir **Git Bash** desde el menú Inicio, ir al proyecto y ejecutar:

```bash
cd aprendizaje-automatico-en-la-nube
make setup
make check
```

## Estructura

```
aprendizaje-automatico-en-la-nube/
├── pyproject.toml           las dependencias del proyecto
├── uv.lock                  exactamente qué quedó instalado
├── Makefile                 los comandos del proyecto, con nombre corto
├── .pre-commit-config.yaml  hooks de calidad instalados con make setup
├── .github/workflows/       el CI: lo que se verifica en cada push
├── src/BeijingAir/          el código de verdad, como paquete instalable
│   ├── config.py              las decisiones en un solo lugar
│   ├── data/                  descargar, cargar y VALIDAR datos
│   ├── features/              construir variables
│   ├── models/                entrenar y evaluar
│   ├── flows/                 el pipeline orquestado con Prefect
│   ├── api/                   servir el modelo
│   └── monitoring/            vigilarlo
├── notebooks/               exploración y narrativa — importa de src/, no define lógica
├── tests/                   lo que protege a src/ de nosotros mismos
├── data/                    los datos NO se versionan; la carpeta sí existe
│   ├── raw/                   tal como llegaron, intocables
│   │   └── metadata.json      la procedencia SÍ se versiona: url, hash, licencia
│   └── processed/             lo que produce el pipeline
├── reports/                 reportes generados (el de drift sí se versiona)
├── models/                  artefactos locales — tampoco se versionan
└── docs/                    decisiones, fichas de datos y del modelo
```

## Comandos frecuentes

```bash
make setup         # instala dependencias y los hooks de pre-commit
make data          # descarga y extrae el dataset Beijing
make lint          # revisa estilo con ruff
make format        # formatea con ruff
make typecheck     # verifica tipos con mypy
make test          # corre todos los tests
make test-fast     # corre solo los tests sin red ni servicios
make check         # lint + tipos + tests, en local
make validate-data # descarga y valida las particiones reales contra el contrato
make mlflow        # inicia MLflow en http://127.0.0.1:5001 (dejar esta terminal abierta)
make train         # entrena baseline y bosque, y registra las corridas en MLflow
make flow          # pipeline de entrenamiento orquestado con Prefect
make drift         # reporte de drift entre referencia y producción simulada
make clean         # borra caches y artefactos temporales
```

## Los datos

**Beijing Multi-Site Air Quality Data** — UCI Machine Learning Repository,
dataset 501. 420.768 filas × 18 columnas, horario, de 2013-03-01 a 2017-02-28,
12 estaciones de monitoreo. Licencia CC BY 4.0.

```bash
make data
```

**El dataset no se versiona.** Lo que sí va al repositorio es
`data/raw/metadata.json`, con la URL, el SHA-256, el tamaño y la licencia. Cada
integrante corre la descarga en su máquina y compara el hash: si coincide,
todos trabajan con exactamente el mismo dato.

Las particiones son **rangos de fechas fijos** declarados en `config.py`, nunca
`datetime.now()`:

| Partición | Rango | Uso |
|---|---|---|
| `train` | 2013-03-01 a 2015-06-30 | entrenamiento |
| `valid` | 2015-07-01 a 2015-12-31 | selección de hiperparámetros |
| `test` | 2016-01-01 a 2016-06-30 | holdout fijo, juez del gate |
| `produccion` | 2016-07-01 a 2017-02-28 | producción simulada para monitoreo |

Esquema, unidades por columna, conteo y naturaleza de los nulos, sesgos y
limitaciones están en [`docs/dataset-card.md`](docs/dataset-card.md).

## Pipeline de entrenamiento

El entrenamiento está orquestado con Prefect. El flow **no reimplementa** la
lógica de ML: llama a `models/train.py` y a `data/`. Duplicarla produciría dos
versiones que se desincronizan.

### Cómo se ejecuta

Necesita dos servicios corriendo, cada uno en su propia terminal:

```bash
uv run prefect server start   # terminal 1 — UI en http://127.0.0.1:4200
make mlflow                   # terminal 2 — UI en http://127.0.0.1:5001
make flow                     # terminal 3 — el pipeline
```

### Las seis tasks

```
extraer ──► validar ──► entrenar ──► evaluar ──┬──► registrar_candidato
                                               └──► publicar_reporte
```

| Task | Qué hace | Detalle |
|---|---|---|
| `extraer` | Descarga el ZIP y registra su hash | `retries=3` con backoff `[10, 30, 60]` |
| `validar` | Corre el contrato sobre el crudo | `cache_key_fn=task_input_hash` |
| `entrenar` | Llama a `entrenar_y_registrar()` | Registra en MLflow |
| `evaluar` | Elige el mejor candidato por RMSE | |
| `registrar_candidato` | Pone el alias `@candidate` | **No toca `@champion`** |
| `publicar_reporte` | Tabla de métricas como artifact | Visible en la UI |

El orden **no está escrito a mano**: sale de los datos que cada task le pasa a la
siguiente. Por eso `validar` recibe la ruta que devuelve `extraer` aunque no la
use.

### El flow registra, no promueve

`registrar_candidato` marca la versión nueva con el alias `@candidate` y el tag
`validation_status=pending`. **Nunca mueve `@champion`.** La promoción es
responsabilidad del gate, no del entrenamiento: un modelo no llega a producción
por el hecho de que el entrenamiento no lanzó excepciones.

### Por qué el backoff es `[10, 30, 60]` y no `[2, 2, 2]`

Reintentar cada dos segundos contra un servicio caído solo le agrega carga. La
lista da control explícito por intento y da tiempo real a que el proveedor se
recupere.

### Caching: medición y diagnóstico

Dos ejecuciones consecutivas, cronometradas:

| Task | 1ª (caché frío) | 2ª (caché caliente) | Estado |
|---|---:|---:|---|
| `extraer` | 0,26 s | 0,27 s | Completed |
| **`validar`** | **0,765 s** | **0,009 s** | **Cached** |
| `entrenar` | 23,7 s | 32,2 s | Completed |
| **Total** | **30,1 s** | **38,2 s** | |

**El caching funciona: `validar` es 85 veces más rápido en la segunda corrida**, y
Prefect lo reporta explícitamente como `Cached(type=COMPLETED)`.

**Pero el tiempo total no baja, y esa es la observación importante.** El ahorro es
de 0,76 s; `entrenar` varió 8,5 s entre las dos corridas por su cuenta. La
variabilidad de una task grande se come el ahorro de una pequeña.

`entrenar` es el 90-95 % del pipeline, y parte de ese tiempo es MLflow exportando
las 227 dependencias del proyecto para guardarlas junto al modelo. **Ninguna de
las dos cosas se debe cachear**: cachear el entrenamiento significa "no vuelvas a
entrenar", y la exportación de dependencias es la trazabilidad que hace
reproducible el artefacto.

**Pendiente declarado:** la preparación de datos ocurre dentro de `entrenar`
(en `data/loaders.py`), no como task independiente. Sacarla haría el ahorro
visible en el total.

### Nota de portabilidad

`make flow` fuerza `PYTHONUTF8=1`. MLflow imprime emojis en sus mensajes, y
Windows usa `cp1252` cuando la salida no va a una consola — lo que hacía fallar
el pipeline con `UnicodeEncodeError` al redirigir la salida. **Habría roto el CI**,
que captura la salida igual.

## Monitoreo de drift

Compara la partición de referencia (`train`, 2013-03 a 2015-06) contra la
producción simulada (2016-07 a 2017-02) y genera un reporte navegable.

```bash
make drift        # o: uv run python -m BeijingAir.monitoring.check_drift
```

Produce dos cosas:

- Una tabla por columna en la terminal: test, p-valor, tamaño del efecto y veredicto.
- `reports/drift-report.html`, el reporte navegable de Evidently.

Y termina con un **exit code** utilizable en CI: `0` sin drift, `1` con drift,
`2` si falla la infraestructura.

> `make drift` **termina con error cuando detecta drift**. Es el comportamiento
> esperado: es lo que permite usarlo como gate en CI. No envolverlo en `|| true`.

### Cómo se decide

No basta con que el cambio sea estadísticamente significativo: con 245.376 filas
de referencia, **todas** las columnas dan p < 0.05, incluso las que cambian un
0,7 %. Por eso se exigen dos condiciones: significancia **y** un tamaño de efecto
mínimo (KS ≥ 0.10). El umbral global es el 30 % de columnas con drift.

`mes` y `temporada` se excluyen del check porque driftean por construcción: dos
ventanas temporales distintas siempre tienen mezcla distinta de meses. `hora`,
`dia_semana` y `station` sí se conservan como control — que den efecto ~0
verifica que la partición está bien armada.

### Qué encontramos

36 % de las columnas driftean, pero **no todo es del mismo tipo**. Comparando el
mismo mes en años distintos (enero, con la estacionalidad constante):

| | 2014 | 2015 | 2016 | 2017 | Patrón |
|---|---:|---:|---:|---:|---|
| SO2 | 53,4 | 34,3 | 19,9 | 18,5 | ↓ monótono, **−65 %** |
| O3 | 22,5 | 23,6 | 30,2 | 33,9 | ↑ monótono, **+51 %** |
| PM2.5 | 98,0 | 96,4 | 66,9 | 113,3 | sin tendencia |

SO2 y O3 tienen **tendencia estructural**. PM2.5, PM10, CO y NO2 se mueven al
unísono sin dirección: variabilidad meteorológica interanual. PRES, TEMP y `wd`
driftean por estacionalidad esperada.

Los umbrales, su justificación y el análisis completo están en
[`docs/politica-de-reentrenamiento.md`](docs/politica-de-reentrenamiento.md).

## Documentación

| Documento | Qué contiene |
|---|---|
| [`docs/dataset-card.md`](docs/dataset-card.md) | Procedencia, licencia, esquema, nulos, particiones, sesgos |
| [`docs/model-card.md`](docs/model-card.md) | El modelo, sus métricas y sus límites |
| [`docs/politica-de-reentrenamiento.md`](docs/politica-de-reentrenamiento.md) | Trigger, umbrales, rollback, alertas |
| [`docs/adr/`](docs/adr/) | Registro de decisiones de arquitectura |

## Créditos

Basado en el repositorio [`MLOps-Course`](https://github.com/dpalacioj/MLOps-Course)
para estructura, convenciones y configuración.

## Contribuir

Pendiente de definir las convenciones de contribución.