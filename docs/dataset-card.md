# Dataset card

## Identificación

| Campo | Valor |
|---|---|
| Nombre | Beijing Multi-Site Air Quality Data |
| Fuente y URL | UCI Machine Learning Repository, dataset 501 — <https://archive.ics.uci.edu/dataset/501> |
| Licencia (nombre exacto + URL) | CC BY 4.0 — <https://creativecommons.org/licenses/by/4.0/> |
| Fecha de descarga | 2026-08-25 |
| Tamaño | 420.768 filas × 18 columnas · 8.192.212 bytes comprimido |
| Hash de las particiones | ver `data/raw/metadata.json` |

## Cumplimiento de los requisitos duros del curso

| Requisito | ¿Cumple? | Evidencia o justificación |
|---|---|---|
| Eje temporal explícito | Sí | `year`/`month`/`day`/`hour` horario, de 2013-03-01 a 2017-02-28. Se compone en la columna `datetime` en `data/descarga.py` |
| ≥2 particiones separables | Sí | Cuatro rangos declarados en `config.py`: train, valid, test y produccion simulada |
| ≤500 MB, descargable sin autenticación | Sí | 8,2 MB comprimido, HTTP 200 sin credenciales |
| ≥3 categóricas y ≥3 numéricas con nulos reales | **Parcial** | Numéricas: 11 con nulos reales. Categóricas nativas: solo 2 (`station`, `wd`). Se deriva una tercera (TODO: cuál) y se declara aquí |
| Métrica de negocio articulable | Sí | TODO: escríbela tú, ver abajo |
| Licencia que permite uso educativo | Sí | CC BY 4.0, requiere atribución |

> Nota sobre el metadata de UCI: la ficha oficial declara `has_missing_values: no`
> y **es falso**. Los CSV traen `NA` en todas las columnas de contaminantes. Los
> conteos reales están más abajo.

## Esquema y significado de las columnas

| Columna | Tipo | Unidades | Significado |
|---|---|---|---|
| `No` | entero | — | Índice del registro dentro de su estación |
| `year`, `month`, `day`, `hour` | entero | — | Componentes del eje temporal |
| `PM2.5` | real | µg/m³ | Partículas ≤2,5 µm |
| `PM10` | real | µg/m³ | Partículas ≤10 µm |
| `SO2` | real | µg/m³ | Dióxido de azufre |
| `NO2` | real | µg/m³ | Dióxido de nitrógeno |
| `CO` | real | µg/m³ | Monóxido de carbono |
| `O3` | real | µg/m³ | Ozono |
| `TEMP` | real | °C | Temperatura |
| `PRES` | real | hPa | Presión atmosférica |
| `DEWP` | real | °C | Punto de rocío |
| `RAIN` | real | mm | Precipitación en la hora |
| `wd` | categórica | 16 niveles | Dirección del viento |
| `WSPM` | real | m/s | Velocidad del viento |
| `station` | categórica | 12 niveles | Estación de monitoreo |

## Nulos: por qué los hay y qué se hace con ellos

Conteos reales medidos sobre las 420.768 filas (12 estaciones, 2013-03 a 2017-02):

| Columna | Nulos | % | Naturaleza |
|---|---|---|---|
| `CO` | 20.701 | 4,9 % | Fallo de captura |
| `O3` | 13.277 | 3,2 % | Fallo de captura |
| `NO2` | 12.116 | 2,9 % | Fallo de captura |
| `SO2` | 9.021 | 2,1 % | Fallo de captura |
| `PM2.5` | 8.739 | 2,1 % | Fallo de captura |
| `PM10` | 6.449 | 1,5 % | Fallo de captura |
| `wd` | 1.822 | 0,4 % | Fallo de captura |
| `DEWP` | 403 | 0,1 % | Fallo de captura |
| `TEMP` | 398 | 0,1 % | Fallo de captura |
| `PRES` | 393 | 0,1 % | Fallo de captura |
| `RAIN` | 390 | 0,1 % | Fallo de captura |
| `WSPM` | 318 | 0,1 % | Fallo de captura |

**Ninguno de estos nulos es estructural.** Todas las columnas aplican a todos los
registros: una estación de monitoreo siempre tiene una temperatura y siempre
tiene una concentración de CO. Un `NA` significa que **el sensor no reportó esa
hora**, no que la magnitud no exista.

Esa distinción manda la estrategia. `fillna(0)` en `PM2.5` afirmaría que hubo aire
perfectamente limpio en 8.739 horas en las que en realidad no sabemos nada — y el
modelo aprendería esos ceros como reales.

Se nota el patrón: los contaminantes (1,5-4,9 %) fallan mucho más que la
meteorología (0,1 %). Son instrumentos distintos, con mantenimiento y
calibraciones distintas.

TODO: escribe aquí la estrategia que eligieron y por qué.


Conteos reales medidos sobre las 420.768 filas:

<PEGA AQUÍ LA SALIDA DEL COMANDO>

TODO: para cada columna, di si el nulo es **estructural** (el campo no aplica a
ese registro) o por **fallo de captura**, y qué estrategia aplicas.

## Particiones

| Partición | Rango temporal | Uso |
|---|---|---|
| train | 2013-03-01 a 2015-06-30 | entrenamiento |
| valid | 2015-07-01 a 2015-12-31 | selección de hiperparámetros |
| test | 2016-01-01 a 2016-06-30 | holdout fijo, juez del gate |
| produccion | 2016-07-01 a 2017-02-28 | producción simulada para monitoreo |

## Población representada y sesgos conocidos

TODO

## Limitaciones y usos no previstos

TODO