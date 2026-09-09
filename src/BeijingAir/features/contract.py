"""Contrato de features - la UNICA definicion de features del proyecto.

Problema que resuelve: en un proyecto sin este modulo, el notebook, el flow de
entrenamiento y la API acaban con tres listas de features distintas. El sintoma
tipico es un ``KeyError`` en el mejor caso, y en el peor una feature que se
codifica como categorica en entrenamiento y como numerica en serving.

La distincion importante, y la razon por la que este archivo esta separado del
contrato de datos: hay que separar explicitamente las columnas que **llegan** en
el dato crudo de las que el pipeline **deriva**. Confundirlas es lo que produce
train/serving skew.

Caso guia: Beijing Multi-Site Air Quality. 12 estaciones, 2013-03 a 2017-02.
El target es ``PM2.5`` (regresion); el resto de contaminantes y la meteorologia
son las features de entrada.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

# =============================================================================
# Columnas que LLEGAN en el dato crudo
# =============================================================================
#: Eje temporal. Se compone en ``data/descarga.py`` a partir de
#: year/month/day/hour. Es la columna sobre la que se hace el split y el drift.
COL_TIEMPO: Final[str] = "datetime"

#: Categoricas crudas. ``station`` identifica la estacion (12 niveles) y ``wd``
#: la direccion del viento (16 rumbos). Van como string: "NNE" no es mayor que
#: "N". Nulos en ``wd`` son fallo de captura.
CRUDAS_CATEGORICAS: Final[list[str]] = ["station", "wd"]

#: Numericas crudas (excluye el target ``PM2.5``). Contaminantes en ug/m3,
#: excepto ``CO`` en ug/m3 (100..10000); meteorologia en unidades SI (TEMP en
#: grados C, PRES en hPa, RAIN en mm, WSPM en m/s, DEWP en grados C).
#: Un cambio silencioso de unidades degrada la metrica sin lanzar excepciones.
CRUDAS_NUMERICAS: Final[list[str]] = [
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
]

COLUMNAS_CRUDAS_REQUERIDAS: Final[list[str]] = [
    COL_TIEMPO,
    *CRUDAS_CATEGORICAS,
    *CRUDAS_NUMERICAS,
]

# =============================================================================
# Columnas DERIVADAS por el pipeline
# =============================================================================
#: Temporada del anio, derivada del mes. Es la TERCERA categorica: el dataset
#: solo trae dos categoricas nativas (station, wd), y el proyecto requiere minimo
#: tres. Captura el ciclo anual de la contaminacion (invierno peor).
COL_TEMPORADA: Final[str] = "temporada"
DERIVADAS_CATEGORICAS: Final[list[str]] = [COL_TEMPORADA]

#: Valores que genera ``_mes_a_temporada``. El contrato procesado y la API
#: comparten esta constante para que no aparezcan reglas distintas de negocio.
TEMPORADAS_VALIDAS: Final[frozenset[str]] = frozenset({"invierno", "primavera", "verano", "otonio"})

#: Features de calendario. La contaminacion de un martes a las 9am no se parece
#: a la de un domingo a las 3am, y el modelo no puede inferirlo de un timestamp.
COL_HORA: Final[str] = "hora"
COL_DIA_SEMANA: Final[str] = "dia_semana"
COL_MES: Final[str] = "mes"
DERIVADAS_NUMERICAS: Final[list[str]] = [COL_HORA, COL_DIA_SEMANA, COL_MES]

# =============================================================================
# Features que consume el modelo
# =============================================================================
FEATURES_CATEGORICAS: Final[list[str]] = [*DERIVADAS_CATEGORICAS, *CRUDAS_CATEGORICAS]
FEATURES_NUMERICAS: Final[list[str]] = [*CRUDAS_NUMERICAS, *DERIVADAS_NUMERICAS]
FEATURES: Final[list[str]] = [*FEATURES_CATEGORICAS, *FEATURES_NUMERICAS]

# =============================================================================
# Target
# =============================================================================
#: Concentracion horaria de particulas PM2.5 (ug/m3). Regresion. Nunca entra
#: como feature: predecir PM2.5 con PM2.5 es leakage puro.
TARGET: Final[str] = "PM2.5"

#: Columna por la que se calculan metricas por subgrupo en el gate de promocion.
#: Una categorica con pocos niveles y sentido de negocio: comparar el error por
#: estacion dice si el modelo se porta mal solo en algunas.
COL_SUBGRUPO: Final[str] = "station"


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva las features a partir del dataframe crudo.

    Es **idempotente**: llamarla dos veces sobre el mismo dataframe da el mismo
    resultado. Eso importa porque el orquestador puede cachear el resultado y
    porque los tests la llaman varias veces.

    Args:
        df: dataframe con al menos ``COLUMNAS_CRUDAS_REQUERIDAS``.

    Returns:
        Copia del dataframe con las columnas derivadas agregadas.

    Raises:
        KeyError: si falta alguna columna cruda requerida.
    """
    faltantes = [c for c in COLUMNAS_CRUDAS_REQUERIDAS if c not in df.columns]
    if faltantes:
        raise KeyError(
            f"Faltan columnas crudas requeridas: {faltantes}. "
            f"Columnas presentes: {sorted(df.columns.tolist())}"
        )

    out = df.copy()

    # Las categoricas se castean a string SOLO al construir features, nunca
    # sobre el dataframe crudo completo: castear el crudo convierte tambien las
    # numericas y el vectorizador acaba one-hot-encodeando una concentracion.
    for col in CRUDAS_CATEGORICAS:
        out[col] = out[col].astype("string").fillna("desconocido").astype(str)

    out[COL_TEMPORADA] = pd.to_datetime(out[COL_TIEMPO]).dt.month.map(_mes_a_temporada)

    tiempo = pd.to_datetime(out[COL_TIEMPO])
    out[COL_HORA] = tiempo.dt.hour.astype("int16")
    out[COL_DIA_SEMANA] = tiempo.dt.dayofweek.astype("int16")
    out[COL_MES] = tiempo.dt.month.astype("int16")

    return out


def _mes_a_temporada(mes: int) -> str:
    """Mapea el numero de mes (1-12) a la temporada en el hemisferio norte."""
    if mes in (12, 1, 2):
        return "invierno"
    if mes in (3, 4, 5):
        return "primavera"
    if mes in (6, 7, 8):
        return "verano"
    return "otonio"


def a_diccionarios(df: pd.DataFrame) -> list[dict]:
    """Convierte el dataframe al formato que espera ``DictVectorizer``.

    Se usa ``DictVectorizer`` y no ``OneHotEncoder`` a proposito: cuando una
    categorica tiene muchos valores y algunos aparecen SOLO en produccion,
    ``DictVectorizer`` ignora las claves que no vio en ``fit``, que es
    exactamente el comportamiento deseado.
    """
    faltantes = [c for c in FEATURES if c not in df.columns]
    if faltantes:
        raise KeyError(
            f"Faltan features derivadas: {faltantes}. "
            f"Llama a construir_features(df) antes de a_diccionarios(df)."
        )
    return df[FEATURES].to_dict(orient="records")
