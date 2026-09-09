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

## Cumplimiento de los requisitos del proyecto

| Requisito | ¿Cumple? | Evidencia o justificación |
|---|---|---|
| Eje temporal explícito | Sí | `year`/`month`/`day`/`hour` horario, de 2013-03-01 a 2017-02-28. Se compone en la columna `datetime` en `data/descarga.py` |
| ≥2 particiones separables | Sí | Cuatro rangos declarados en `config.py`: train, valid, test y produccion simulada |
| ≤500 MB, descargable sin autenticación | Sí | 8,2 MB comprimido, HTTP 200 sin credenciales |
| ≥3 categóricas y ≥3 numéricas con nulos reales | Sí | Numéricas: 11 con nulos reales. Categóricas nativas: `station` y `wd`; el pipeline deriva `temporada` a partir del mes |
| Métrica de negocio articulable | Sí | MAE de `PM2.5` en µg/m³, reportado además por estación. El modelo sirve para estimar la lectura cuando el sensor de `PM2.5` no está disponible; un error se interpreta directamente en la unidad de calidad del aire |
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

**Estrategia aplicada en `data/loaders.py`:** se descartan las filas con
`PM2.5` nulo porque inventar el target enseñaría al modelo una medición que no
existió. Las variables numéricas se imputan con la mediana de su partición y
conservan un indicador `<col>_era_nulo`; así el modelo puede aprender que el
sensor falló. La dirección del viento `wd` se completa como `desconocido`.
Esta decisión se ejecuta después de validar el crudo y antes de construir las
features.

## Calibración de controles del contrato

| Regla de distribución | Valor medido | Umbral del contrato | Margen y motivo |
|---|---:|---:|---|
| Volumen del dataset completo | 420.768 filas | ≥100 filas por lote | El umbral no intenta validar el tamaño histórico: detecta una ingesta truncada |
| Cobertura de CO (la menor) | 95,1 % | ≥70 % | Margen de 25,1 puntos porcentuales para tolerar ruido real sin aceptar un sensor caído |
| Cobertura de los demás contaminantes | 96,8–98,5 % | ≥70 % | Misma regla, con margen amplio y uniforme para el proveedor |
| Variación de `PM2.5` | Más de un valor observado | >1 valor distinto | Un target constante no permite entrenar una regresión útil |
| Regla física `DEWP ≤ TEMP` | 0 violaciones con ambas mediciones presentes | 0 violaciones | Un punto de rocío mayor que la temperatura indica un problema de medición o parseo |
| Clave `station` + `datetime` | 420.768 pares distintos | 0 duplicados | Dos lecturas para la misma estación y hora suelen revelar una descarga repetida o un merge defectuoso |

## Particiones

| Partición | Rango temporal | Uso |
|---|---|---|
| train | 2013-03-01 a 2015-06-30 | entrenamiento |
| valid | 2015-07-01 a 2015-12-31 | selección de hiperparámetros |
| test | 2016-01-01 a 2016-06-30 | holdout fijo, juez del gate |
| produccion | 2016-07-01 a 2017-02-28 | producción simulada para monitoreo |

La validación reproducible del lote real se ejecuta con `make validate-data`.
El comando descarga el ZIP solo si no está disponible, crea el cache local y
valida cada partición completa contra `RegistrosCrudos`.

## Población representada y sesgos conocidos

La población representada son **lecturas horarias de calidad del aire y
meteorología**, no personas: 12 estaciones de monitoreo de Beijing entre marzo
de 2013 y febrero de 2017. Cada fila describe una estación y una hora; no mide la
exposición personal de quienes viven o trabajan cerca de ella.

Sesgos y riesgos conocidos:

1. **Cobertura espacial limitada.** Doce estaciones no representan todos los
   microambientes de Beijing ni la exposición dentro de viviendas, escuelas o
   medios de transporte. Un modelo puede rendir distinto lejos de una estación.
2. **Cobertura temporal histórica.** El periodo termina en 2017. Cambios
   posteriores en movilidad, industria, regulación o clima no están en los
   datos y no se deben interpretar como comportamiento actual.
3. **Ausencia no aleatoria de sensores.** Los contaminantes concentran más
   nulos que la meteorología; por tanto, imputar sin registrar la ausencia
   ocultaría posibles periodos de fallo de instrumentos.
4. **Cola pesada, no saturación.** El máximo observado en `PM2.5` y `PM10` es
   999, pero solo 1 fila alcanza ese valor en `PM2.5` y 3 en `PM10` sobre
   420.768: son valores extremos aislados, no un tope del instrumento que
   censure sistemáticamente las horas más contaminadas. El riesgo real es
   distinto: la distribución tiene cola pesada (p99 de `PM2.5` en ~342 µg/m³ en
   train, máximo ~844), así que los episodios severos están poco representados
   y el error del modelo se concentra ahí. Se verifica en el EDA, sección 12.

## Limitaciones y usos no previstos

- Este dataset permite **estimación horaria al cierre de la hora** (nowcasting
  de `PM2.5` con las demás mediciones disponibles), no un pronóstico de varios
  días. Para pronosticar el futuro se necesitarían variables disponibles antes
  de la hora objetivo, como pronósticos meteorológicos y rezagos.
- No permite atribuir causalidad entre contaminantes y meteorología: las
  correlaciones del EDA son descriptivas y pueden compartir fuentes o
  estacionalidad.
- No es adecuado para decisiones clínicas, regulatorias individuales ni para
  inferir exposición de una persona concreta.

## Disponibilidad temporal y prevención de leakage

| Feature o grupo | Disponible para estimar la hora `t` | Decisión |
|---|---|---|
| `station` | Antes de `t` | Identifica el sensor; se usa como categórica |
| Calendario (`hora`, `dia_semana`, `mes`, `temporada`) | Antes de `t` | Se deriva solo del timestamp y no usa el target |
| Meteorología y otros contaminantes | Al cierre de `t` | Se usan para estimar `PM2.5` si sus sensores reportaron esa hora; son un caso de nowcasting |
| `PM2.5` | Al cierre de `t` | Es el target y nunca entra en `FEATURES` |
| Mediciones posteriores a `t` | No | Se excluyen: introducirían información del futuro |

Las particiones se cortan por fecha y permanecen fijas. La validación y el
test son posteriores al train, por lo que el holdout no participa en decisiones
de selección ni de imputación del entrenamiento.
