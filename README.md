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
make clean        # borra caches y artefactos temporales
```

Para entrenar, abre dos terminales Git Bash: en la primera ejecuta `make mlflow`
y en la segunda `make train`. MLflow registra el hash del dataset, las
particiones, el commit, parámetros, métricas globales y por estación, además del
modelo con su firma de entrada.

## Créditos

Basado en el repositorio [`MLOps-Course`](https://github.com/dpalacioj/MLOps-Course)
para estructura, convenciones y configuración.

## Contribuir

Pendiente de definir las convenciones de contribución.

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

### Cómo se decide

No basta con que el cambio sea estadísticamente significativo: con 245.376 filas
de referencia, **todas** las columnas dan p < 0.05, incluso las que cambian un
0,7 %. Por eso se exigen dos condiciones: significancia **y** un tamaño de efecto
mínimo (KS ≥ 0.10). El umbral global es el 30 % de columnas con drift.

`mes` y `temporada` se excluyen del check porque driftean por construcción: dos
ventanas temporales distintas siempre tienen mezcla distinta de meses. `hora`,
`dia_semana` y `station` sí se conservan como control — que den efecto ~0
verifica que la partición está bien armada.

Los umbrales y su justificación están en
[`docs/politica-de-reentrenamiento.md`](docs/politica-de-reentrenamiento.md),
junto con el análisis que separa la estacionalidad del cambio estructural real.