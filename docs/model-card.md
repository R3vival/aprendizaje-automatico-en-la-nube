# Model Card — beijing-air-pm25

<!-- ARCHIVO GENERADO. No lo edites a mano: `make model-card` lo sobrescribe.
     Si necesitas cambiar el texto, edita scripts/model_card.py. -->

_Generada automaticamente el 2026-09-12 16:54 UTC por `scripts/model_card.py`._

## 1. Identificacion y version

| campo | valor |
|---|---|
| Nombre registrado | `beijing-air-pm25` |
| Version | **1** |
| Alias | `@champion` |
| URI de referencia | `models:/beijing-air-pm25@champion` |
| Run de MLflow | `2720505b22ba43129c586fec5432ed0d` |
| Registrada el | 2026-09-12 16:52 UTC |
| `validation_status` | passed |

El modelo se referencia siempre por **alias**, nunca por numero de version ni
por ruta de archivo. Mover el alias es la operacion de despliegue y, en sentido
inverso, la de rollback.

## 2. Uso previsto

Estimar la **concentracion horaria de PM2.5** (ug/m3) en las 12 estaciones de
monitoreo de Beijing, a partir de la meteorologia y los demas contaminantes
registrados en la misma hora.

Casos de uso contemplados:

- Vigilancia de la calidad del aire a nivel horario por estacion.
- Estudios de correlacion entre contaminantes y meteorologia.
- Caso de estudio del curso de MLOps: es el objeto sobre el que se practica
  tracking, registry, promocion, despliegue y monitoreo.

## 3. Uso NO previsto

Esta seccion es la mas importante de la card y la que mas se omite. Un modelo
usado fuera de su contexto de entrenamiento falla de forma silenciosa.

- **No es un modelo de pronostico.** Predice el valor *concurrente* a partir de
  las magnitudes de esa misma hora; no proyecta el futuro.
- **No sirve para otra ciudad ni para otra red de sensores.** Fue entrenado con
  12 estaciones de Beijing en un periodo de 2013-2017; otra ciudad o un sensor
  distinto tienen distribuciones y calibraciones distintas.
- **No debe usarse para decisiones sobre personas.** No fue disenado, medido ni
  auditado para asignar responsabilidades, ventilar culpas ni tomar medidas
  administrativas sobre individuos u organizaciones.
- **No es un sistema de alerta en tiempo real.** No recibe lecturas en vivo ni
  estados de falla del sensor; ante un episodio de contaminacion atipico su
  error crece y el modelo no lo sabe.

## 4. Datos de entrenamiento

Fuente: **UCI Machine Learning Repository, dataset 501** (https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip), licencia CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). Las particiones son **fijas y del pasado** por decision
de diseno: un pipeline que calcula el periodo con `datetime.now()` se rompe y
rompe la comparabilidad entre corridas.

| rol | particion | rango | SHA-256 (16 primeros) |
|---|---|---|---|
| Entrenamiento | train | `2013-03-01` a `2015-06-30` | `b04da438b2f331ac...` |
| Validacion (seleccion de hiperparametros) | valid | `2015-07-01` a `2015-12-31` | `b04da438b2f331ac...` |
| Holdout (juez del gate de promocion) | test | `2016-01-01` a `2016-06-30` | `b04da438b2f331ac...` |
| Produccion simulada (monitoreo) | produccion | `2016-07-01` a `2017-02-28` | `b04da438b2f331ac...` |

Muestreo determinista de 50 000 filas por particion con semilla
`42`. La division es **temporal**, no aleatoria: se entrena con
periodos anteriores y se evalua con periodos posteriores, porque en produccion
el modelo siempre predice sobre el futuro.

## 5. Contrato de features

| grupo | columnas |
|---|---|
| crudas_requeridas | `datetime`, `station`, `wd`, `PM10`, `SO2`, `NO2`, `CO`, `O3`, `TEMP`, `PRES`, `DEWP`, `RAIN`, `WSPM` |
| features_categoricas | `temporada`, `station`, `wd` |
| features_numericas | `PM10`, `SO2`, `NO2`, `CO`, `O3`, `TEMP`, `PRES`, `DEWP`, `RAIN`, `WSPM`, `hora`, `dia_semana`, `mes` |
| target | `PM2.5` |

El consumidor envia las columnas crudas (`station`, `wd`, contaminantes y
meteorologia); las features de calendario (`hora`, `dia_semana`, `mes`,
`temporada`) se derivan dentro del pipeline con el mismo codigo que el
entrenamiento. Eso es lo que evita el train/serving skew.

Target: `beijing-air-pm25` predice `PM2.5` en ug/m3
(regresion).

## 6. Metricas

Leidas del run de MLflow que produjo esta version. Los prefijos indican sobre
que particion se midio cada una (`*_valid` sobre valid, `*_test` sobre test).
El gate de promocion decide con `mae_test` y `r2_test`.

| metrica | valor |
|---|---|
| `mae_test` | 14.0452 |
| `mae_valid` | 16.4162 |
| `peor_mae_estacion_test` | 15.5822 |
| `peor_mae_estacion_valid` | 23.1627 |
| `r2_test` | 0.8585 |
| `r2_valid` | 0.9073 |
| `rmse_test` | 26.2038 |
| `rmse_valid` | 28.4881 |

Las metricas por estacion se guardan como artifact del run en
`evaluacion/metricas_por_estacion.json`.

## 7. Limitaciones conocidas

- **Valores imputados.** Cuando un sensor calla, la magnitud se imputa con la
  mediana y se deja el indicador `<col>_era_nulo`. En horas con muchos sensores
  caidos, la entrada al modelo no describe el aire real.
- **Estaciones no vistas.** Si una estacion nueva no estaba en el entrenamiento,
  `OneHotEncoder(handle_unknown="ignore")` la codifica como cero y la prediccion
  se apoya solo en las demas features; el modelo no lo senala.
- **Deriva temporal y estructural.** Entrenado con datos de 2013-2017. Cambios
  de regulacion o de medicion (p. ej. la tendencia a la baja de SO2) degradan el
  desempeno con el tiempo. Es el problema que se monitorea en el drift.
- **El PM2.5 no se imputa.** Las horas con target nulo se descartan: el modelo
  no puede aprender a "inventar" un sensor que no existio.
- **Representatividad.** Una estacion con mas lecturas aporta mas a la metrica
  agregada; el error por estacion no es uniforme (ver `peor_mae_estacion`).

## 8. Consideraciones eticas

- **Sesgo geografico.** El error no se distribuye igual entre las 12 estaciones.
  Las estaciones con menos lecturas o con episodios extremos tienen peor
  estimacion. Cualquier uso que asocie la prediccion a un barrio debe tenerlo en
  cuenta.
- **Datos publicos, sin datos personales.** El dataset es agregado por estacion
  y hora; no contiene identificadores de personas.
- **Transparencia.** El artefacto es trazable de punta a punta: SHA-256 de los
  datos, run de MLflow, version del registry y tag de validacion.
- **Uso no regulatorio.** La prediccion no sustituye a un instrumento de
  medicion de referencia ni debe usarse para certificar la calidad del aire.

## 9. Clasificacion tentativa bajo el EU AI Act

**Clasificacion propuesta: riesgo minimo** (fuera de las categorias de riesgo
alto del Anexo III). Estimar una concentracion de contaminante no decide sobre
acceso a empleo, educacion, credito, servicios esenciales, migracion ni
justicia; no es identificacion biometrica ni infraestructura critica.

> Aviso: esta clasificacion es un **ejercicio didactico** del curso, no
> asesoramiento legal. Una clasificacion vinculante requiere analisis juridico
> del caso de uso concreto y del rol de quien lo opera.
