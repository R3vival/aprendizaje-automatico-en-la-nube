# Aprendizaje Automático en la Nube

Repositorio para el Trabajo final de **aprendizaje automático en la nube**. Este
proyecto toma como base el repositorio [`MLOps-Course`](https://github.com/dpalacioj/MLOps-Course)
para estructura, convenciones y configuración.

# Integrantes: 
- Juan Pablo Arango
- Alejandro Taborda
- Daniel Gomez
- Santiago Gomez

## Empezando

> **Requisito de shell.** Los comandos `make` están escritos para **Bash**
> (`Makefile` usa `SHELL := /bin/bash` y herramientas de Unix como `grep`,
> `awk` y `find`).
> - **macOS/Linux:** funcionan tal cual en la terminal normal.
> - **Windows:** ejecutalos desde **Git Bash** (ver abajo), no desde
>   PowerShell/CMD. Desde PowerShell `make` usa `cmd.exe`, que no entiende esas
>   herramientas y falla.

```bash
make setup        # instala dependencias y los hooks de pre-commit
make check        # lint + tipos + tests, lo mismo que verifica el CI
# Windows: abrir "Git Bash" desde el menú Inicio, ir al proyecto y ejecutar:
cd aprendizaje-automatico-en-la-nube
make setup
make check
```

## Estructura

```
aprendizaje-automatico-en-la-nube/
├── pyproject.toml        las dependencias del proyecto
├── uv.lock               exactamente qué quedó instalado
├── Makefile              los comandos del proyecto, con nombre corto
├── .pre-commit-config.yaml  hooks de calidad instalados con make setup
├── .github/workflows/    el CI: lo que se verifica en cada push
├── src/BeijingAir/       el código de verdad, como paquete instalable
│   ├── config.py           las decisiones en un solo lugar
│   ├── data/               cargar y VALIDAR datos
│   ├── features/           construir variables
│   ├── models/             entrenar y evaluar
│   ├── flows/              orquestar entrenamiento con Prefect
│   ├── api/                servir el modelo
│   └── monitoring/         vigilarlo
├── notebooks/            exploración y narrativa — importa de src/, no define lógica
├── tests/                lo que protege a src/ de nosotros mismos
├── data/                 los datos NO se versionan; la carpeta sí existe
│   ├── raw/                tal como llegaron, intocables
│   └── processed/          lo que produce el pipeline
├── configs/              parámetros por entorno, si los hay
├── models/               artefactos locales — tampoco se versionan
└── docs/                 decisiones, fichas de datos y del modelo
```

## Comandos frecuentes

```bash
make setup        # instala dependencias y los hooks de pre-commit
make lint         # revisa estilo con ruff
make format       # formatea con ruff
make typecheck    # verifica tipos con mypy
make test         # corre todos los tests
make test-fast    # corre solo los tests sin red ni servicios
make check        # lint + tipos + tests, en local
make validate-data # descarga y valida las particiones reales contra el contrato
make mlflow       # inicia MLflow en http://127.0.0.1:5001 (dejar esta terminal abierta)
make train        # entrena baseline y bosque, y registra las corridas en MLflow
make prefect-server # inicia Prefect en http://127.0.0.1:4200
make flow          # valida, entrena y registra un candidato con Prefect
make serve-flow    # deja servido el schedule mensual de Prefect
make serve         # inicia la API de prediccion en http://127.0.0.1:8000
make clean        # borra caches y artefactos temporales
```

Para entrenar, abre dos terminales Git Bash: en la primera ejecuta `make mlflow`
y en la segunda `make train`. MLflow registra el hash del dataset, las
particiones, el commit, parámetros, métricas globales y por estación, además del
modelo con su firma de entrada.

Para la sesión de orquestación usa tres terminales: `make mlflow`,
`make prefect-server` y `make flow`. El flow reintenta solo la descarga, valida
los datos antes de entrenar y registra el bosque como `candidate` en MLflow. El
schedule mensual de `make serve-flow` despierta el flow, pero este solo
reentrena si cambió el hash del dataset; no promueve modelos automáticamente.

## Deployment: API y Docker

La API se ejecuta con `uv run uvicorn BeijingAir.api.main:app --host 127.0.0.1 --port 8000`
(o `make serve` donde `make` esté disponible). Abre
`http://127.0.0.1:8000/docs` para probarla. Su contrato recibe una lectura
cruda: la API deriva las variables de calendario mediante el mismo código que
el entrenamiento y no acepta columnas desconocidas.

El servicio busca exclusivamente `models:/beijing-air-pm25@champion` en MLflow.
Como la promoción aún corresponde a la siguiente etapa, es normal que al inicio
`GET /health` responda `degradado` y `POST /predict` responda 503: es una
protección para no entregar el alias `candidate` a usuarios. Para usar otro
Registry o URI se configura `MODELO_URI` antes de arrancar.

```bash
# Requiere Docker Desktop encendido. No incluye datos ni modelos en la imagen.
docker build -t beijing-air-api .
docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5001 \
  beijing-air-api
```

Consulta los detalles y la decisión en
[`docs/adr/0003-serving-api-y-registry.md`](docs/adr/0003-serving-api-y-registry.md).

## Créditos

Basado en el repositorio [`MLOps-Course`](https://github.com/dpalacioj/MLOps-Course)
para estructura, convenciones y configuración.

## Contribuir

Pendiente de definir las convenciones de contribución.
