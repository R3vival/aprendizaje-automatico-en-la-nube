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

```bash
make setup        # instala dependencias y los hooks de pre-commit
make check        # lint + tipos + tests, lo mismo que verifica el CI
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
make clean        # borra caches y artefactos temporales
```

## Créditos

Basado en el repositorio [`MLOps-Course`](https://github.com/dpalacioj/MLOps-Course)
para estructura, convenciones y configuración.

## Contribuir

Pendiente de definir las convenciones de contribución.
