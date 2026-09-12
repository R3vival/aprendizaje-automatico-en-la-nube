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
│   ├── models/                entrenar, evaluar y promover
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
make mlflow         # inicia MLflow en http://127.0.0.1:5001 (dejar esta terminal abierta)
make train          # entrena baseline y bosque, y registra las corridas en MLflow
make prefect-server # inicia Prefect en http://127.0.0.1:4200
make flow           # pipeline de entrenamiento orquestado con Prefect
make serve-flow     # deja servido el schedule mensual de entrenamiento en Prefect
make serve          # inicia la API de predicción en http://127.0.0.1:8000
make promote-check  # evalúa candidate contra el gate, sin mover el alias champion
make drift          # reporte de drift entre referencia y producción simulada
make clean          # borra caches y artefactos temporales
make up             # levanta el stack: MLflow + API en contenedores
make down           # detiene el stack
make promote        # evalúa el gate y mueve @champion si aprueba
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

> Si el stack de Docker está arriba (`make up`), **no corras `make mlflow`**: el
> contenedor ya ocupa el puerto 5001. Usa uno u otro, no los dos.

### El flujo de entrenamiento

```
asegurar_origen ──► preparar_y_validar ──► entrenar_y_registrar ──► evaluar
                                                                         │
                            ┌────────────────────────────────────────────┘
                            ▼
                   publicar_reporte ──► marcar_candidato ──► guardar_estado
```

| Task | Qué hace | Detalle |
|---|---|---|
| `asegurar_origen` | Descarga el ZIP si falta y registra su hash | 2 reintentos con esperas de 5 y 15 segundos |
| `preparar_y_validar` | Valida contratos y prepara train, validación y test | Cache por entradas durante un día |
| `entrenar_y_registrar` | Entrena los modelos y los registra en MLflow | Conserva el id de la corrida de Prefect |
| `evaluar` | Comprueba que el candidato supere al baseline | Si no lo supera, el flow falla |
| `publicar_reporte` | Tabla de métricas como artifact | Visible en la UI |
| `marcar_candidato` | Deja la versión pendiente de promoción | **No toca `@champion`** |
| `guardar_estado` | Guarda el hash del último dato procesado | Evita reentrenar sin datos nuevos |

El orden sale de las dependencias entre tareas: por ejemplo, una versión solo se
marca como candidata después de superar la evaluación.

### El flow registra, no promueve

`marcar_candidato` deja la versión nueva con el alias `@candidate` y el tag
`validation_status=pending`. **Nunca mueve `@champion`.** La promoción es
responsabilidad del gate, no del entrenamiento: un modelo no llega a producción
por el hecho de que el entrenamiento no lanzó excepciones.


### Caché y reintentos

La preparación de datos se cachea por un día usando el hash del dataset y la
muestra solicitada. Prefect muestra si una tarea se reutilizó en cada corrida;
no se publican tiempos fijos porque cambian entre equipos, red e infraestructura.







El entrenamiento no se cachea a propósito: una corrida nueva debe conservar sus
métricas y artefactos en MLflow para que sea trazable.

### El flow no reentrena si los datos no cambiaron

`debe_reentrenar()` compara el SHA-256 del dataset contra el de la última
corrida. Si es idéntico, el flow termina sin entrenar y lo dice en el log. El
schedule mensual despierta el pipeline; los datos deciden si vale la pena
ejecutarlo.

Para forzar una corrida, al probar cambios de código o hiperparámetros:

```bash
uv run python -m BeijingAir.flows.training --forzar
```

### Nota de portabilidad

El `Makefile` exporta `PYTHONUTF8=1` para todos los targets. MLflow imprime
emojis en sus mensajes, y Windows usa `cp1252` cuando la salida no va a una
consola — lo que hacía fallar el pipeline con `UnicodeEncodeError` al redirigir
la salida. **Habría roto el CI**, que captura la salida igual.

Se usa la directiva `export` de make y no el prefijo `VARIABLE=valor comando`,
que es sintaxis de shell: en Windows `make` cae a `cmd.exe`, que no la entiende
y falla con `'PYTHONUTF8' is not recognized`.

## Despliegue local con Docker

El stack levanta dos servicios en contenedores: el servidor de MLflow (tracking + registry) y la API de inferencia.

```bash
make up      # levanta MLflow + API
make down    # los detiene
```

| Servicio | URL | Que hace |
|---|---|---|
| MLflow | http://127.0.0.1:5001 | Tracking y Model Registry (SQLite + artefactos en volumen) |
| API | http://127.0.0.1:8000/docs | Sirve el modelo resuelto por alias `@champion` |

### Cadena completa hasta una prediccion

`make up` deja los contenedores arriba, pero el registry arranca vacio: la API
responde `degradado` hasta que exista un modelo con alias `@champion`.

```bash
make up                       # 1. contenedores arriba
make prefect-server           # 2. en otra terminal
make flow                     # 3. entrena y registra v1 como @candidate
make promote                  # 4. evalua el gate y mueve @champion
docker compose restart api    # 5. la API resuelve el alias al arrancar
```

El paso 5 hace falta porque la API resuelve el alias **al iniciar**, no en cada
peticion. Y `make flow` no reentrena si el SHA-256 del dataset no cambio; para
forzarlo: `uv run python -m BeijingAir.flows.training --forzar`.

### Verificacion

```bash
curl http://127.0.0.1:8000/health
```

```json
{"estado":"ok","modelo_cargado":true,"model_uri":"models:/beijing-air-pm25@champion","model_version":"1"}
```

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"datetime":"2017-01-15T08:00:00","station":"Aotizhongxin","wd":"NW","PM10":180.0,"SO2":25.0,"NO2":70.0,"CO":1800.0,"O3":20.0,"TEMP":-3.5,"PRES":1025.0,"DEWP":-15.0,"RAIN":0.0,"WSPM":1.8}'
```

```json
{"prediccion_pm25":165.58487430162435,"model_name":"beijing-air-pm25","model_version":"1","latencia_ms":118.7653560000399}
```

En PowerShell `curl` es un alias de `Invoke-WebRequest`: usa `curl.exe` y pasa el
cuerpo desde un archivo con `-d "@archivo.json"`.

La imagen no corre como root:

```bash
docker run --rm --entrypoint sh beijing-air-api -c 'id -u'
# 10001
```

### Dos decisiones de configuracion del servidor MLflow

**`--allowed-hosts`** — MLflow 3.x responde 403 a las peticiones cuyo `Host` no
reconoce (proteccion contra DNS rebinding). La API se conecta como
`http://mlflow:5001`, asi que ese nombre debe estar en la lista. Se listan los
hosts necesarios en vez de `*`, que la propia documentacion desaconseja.

**`--serve-artifacts`** — el servidor recibe y sirve los archivos del modelo por
HTTP (URIs `mlflow-artifacts:/`). Con `--default-artifact-root` apuntando a una
ruta local, cada cliente escribe el modelo en esa ruta **en su propia maquina**:
entrenando desde Windows el modelo termina en `C:\mlflow\artifacts` y el
contenedor no lo encuentra.

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


## Contrato de la API

La API recibe una lectura horaria cruda y deriva las variables de calendario con
el mismo código que el entrenamiento. Rechaza columnas desconocidas
(`extra="forbid"`) y valida rangos por columna: un `PM10` de 5000 o un campo de
más devuelven 422, no una predicción silenciosamente mala.

Resuelve exclusivamente `models:/beijing-air-pm25@champion`. Si no hay champion
registrado, `GET /health` responde `degradado` y `POST /predict` responde 503.
Es deliberado: la API prefiere degradarse antes que servir un `candidate` sin
aprobar, y antes que entrar en un ciclo de reinicios.

Para desarrollo, sin Docker:

```bash
make serve   # http://127.0.0.1:8000/docs
```

Detalles y decisión en
[`docs/adr/0003-serving-api-y-registry.md`](docs/adr/0003-serving-api-y-registry.md).

## Promoción controlada del modelo

El flow deja un modelo como `candidate`; nunca toca `champion`. Cada corrida
actualizada mide además `mae_test` y `r2_test` en la partición temporal que no
se usa para elegir el modelo. El gate compara dichas métricas con límites
explícitos y con el champion actual. Para revisar el resultado localmente, sin
mover ningún alias, inicia MLflow y ejecuta:

```bash
uv run python -m BeijingAir.models.promote --dry-run
```
Contra el MLflow local (el del stack de Docker), la promoción se ejecuta a mano:

```bash
make promote-check   # solo evalúa e imprime el veredicto
make promote         # evalúa y, si aprueba, mueve @champion
```

En el registry compartido la mutación no se hace desde una máquina personal:

La mutación real se hace solo desde el workflow manual **Promover modelo** de
GitHub Actions, en el entorno `production`. Antes de usarlo, el administrador
del repositorio debe crear allí el secreto `MLFLOW_TRACKING_URI` con una URL de
un MLflow Registry remoto; `http://127.0.0.1:5001` es local y GitHub no puede
alcanzarlo. Los criterios y la decisión están documentados en
[`docs/adr/0004-gate-de-promocion.md`](docs/adr/0004-gate-de-promocion.md).

## Créditos

Basado en el repositorio [`MLOps-Course`](https://github.com/dpalacioj/MLOps-Course)
para estructura, convenciones y configuración.

## Contribuir

1. Parte de `main` actualizado y crea una rama con un nombre descriptivo.
2. Antes de abrir un PR ejecuta `uv run ruff format --check .`,
   `uv run ruff check .`, `uv run mypy` y
   `uv run pytest -m "not slow and not integration"`.
3. Describe en el PR qué cambió, cómo lo validaste y espera la revisión de al
   menos un integrante antes de fusionarlo a `main`.
