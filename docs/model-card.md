# Model card — Estimación horaria de PM2.5

## Propósito

Estimar la concentración horaria de `PM2.5` (µg/m³) de una estación de Beijing
cuando la lectura del sensor objetivo no está disponible. Es un caso de
**nowcasting**: las demás mediciones de la misma hora deben haber llegado antes
de solicitar la estimación.

## Datos y particiones

El entrenamiento usa `train` (2013-03 a 2015-06), la selección se realiza en
`valid` (2015-07 a 2015-12) y `test` queda reservado como holdout. El hash del
ZIP, el commit y los rangos de las particiones se registran como tags en MLflow.

## Features y protección contra leakage

El pipeline toma las columnas declaradas en `features/contract.py`: estación,
dirección del viento, temporada, calendario, otros contaminantes y
meteorología. `PM2.5` no pertenece a `FEATURES`; es exclusivamente el target.
El split es temporal y cada transformación se ajusta solo con `train` dentro
del `Pipeline` de scikit-learn.

## Modelos comparados

1. `baseline`: `DummyRegressor` que predice la media de train.
2. `bosque`: `RandomForestRegressor` con categóricas codificadas e imputación
   dentro del pipeline.

La selección se hace por MAE de validación. También se registra el RMSE, R² y
el MAE de la estación peor atendida, para no ocultar un mal desempeño local tras
una métrica global buena.

## Reproducibilidad y trazabilidad

Ejecutar `make mlflow` y luego `make train`. Cada corrida guarda parámetros,
métricas, tags de procedencia, un ejemplo de entrada, la firma del modelo, el
artefacto serializado y un JSON con el MAE por estación. Los valores concretos
de la corrida aprobada se consultan en MLflow, no se copian manualmente aquí.

## Limitaciones

- El modelo no es un pronóstico de varios días ni reemplaza una medición
  regulatoria.
- Las correlaciones entre contaminantes no demuestran causalidad.
- La cobertura se limita a 12 estaciones y a los años 2013–2017.
- Antes de promover un modelo se debe evaluar una sola vez sobre `test` y
  documentar el resultado de esa corrida.
