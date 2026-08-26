"""Contrato de datos ejecutable para Beijing Multi-Site Air Quality.

Un contrato de datos es un esquema **versionado junto al codigo** que describe
como se ve un dato valido, y que se valida en la FRONTERA del pipeline: donde el
dato entra, no donde se usa.

Por que hace falta: los errores de datos casi nunca lanzan excepciones, degradan
metricas. Si una columna cambia de unidades, o llega una categoria nueva, o el
proveedor empieza a mandar nulos donde antes no habia, el pipeline entrena sin
quejarse y sirve predicciones malas. Todo verde, todo mal. El contrato convierte
ese fallo silencioso en un fallo ruidoso.

Que va en el contrato y que va en el test:

- el **contrato** describe como es un dato valido;
- el **test** verifica que el pipeline se comporta como debe ante un dato
  invalido (ver ``tests/data/test_contrato.py``).

Los limites de abajo se midieron sobre las 420.768 filas reales (12 estaciones,
2013-03 a 2017-02). No son inventados: son el rango observado mas un margen,
para que un dato razonable no alerte en falso pero un cambio de unidades o una
lectura de sensor rota si.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from BeijingAir.features import contract as fc

# =============================================================================
# Limites de negocio. Aqui, no repartidos por el codigo.
# =============================================================================
#: Volumen minimo por particion. Una particion con 12 filas casi siempre
#: significa que la ingesta se corto, no que hubo 12 lecturas ese mes.
VOLUMEN_MINIMO: Final[int] = 100

#: Contaminantes en ug/m3. Rango observado mas margen. PM2.5/PM10 se saturan en
#: 999 (cap del instrumento); CO va 100..10000.
PM_MIN: Final[float] = 0.0
PM_MAX: Final[float] = 1000.0
SO2_MAX: Final[float] = 600.0
NO2_MAX: Final[float] = 400.0
CO_MAX: Final[float] = 15_000.0
O3_MAX: Final[float] = 1_200.0

#: Meteorologia. TEMP y DEWP en grados C, PRES en hPa, RAIN en mm, WSPM en m/s.
TEMP_MIN: Final[float] = -30.0
TEMP_MAX: Final[float] = 50.0
PRES_MIN: Final[float] = 900.0
PRES_MAX: Final[float] = 1_100.0
DEWP_MIN: Final[float] = -50.0
DEWP_MAX: Final[float] = 40.0
RAIN_MAX: Final[float] = 200.0
WSPM_MAX: Final[float] = 50.0

#: Direcciones de viento validas (los 16 rumbos que documenta el dataset).
#: Un valor fuera de esta lista es un error del proveedor o del parseo.
RUMBOS_VALIDOS: Final[frozenset[str]] = frozenset(
    {
        "N",
        "NNE",
        "NE",
        "ENE",
        "E",
        "ESE",
        "SE",
        "SSE",
        "S",
        "SSW",
        "SW",
        "WSW",
        "W",
        "WNW",
        "NW",
        "NNW",
    }
)

#: Contaminantes y meteorologia crudas, para no repetir los limites.
_CONTA_MIN: Final[dict[str, float]] = dict.fromkeys(
    ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"], PM_MIN
)
_CONTA_MAX: Final[dict[str, float]] = {
    "PM2.5": PM_MAX,
    "PM10": PM_MAX,
    "SO2": SO2_MAX,
    "NO2": NO2_MAX,
    "CO": CO_MAX,
    "O3": O3_MAX,
}


class RegistrosCrudos(pa.DataFrameModel):
    """Contrato del dato tal como llega del proveedor.

    ``strict = False`` a proposito: la fuente trae columnas que no usamos
    (``No``, ``year``..``hour``), y exigir ausencia de columnas extra entrena al
    equipo a ignorar el contrato. ``coerce = True`` convierte tipos compatibles
    sin inventar datos.
    """

    # Frontera de entrada: validamos las columnas que el pipeline consume.
    datetime: Series[pd.Timestamp] = pa.Field(nullable=False)
    station: Series[str] = pa.Field(nullable=False)
    wd: Series[str] = pa.Field(nullable=True, isin=set(RUMBOS_VALIDOS))
    PM2_5: Series[float] = pa.Field(ge=PM_MIN, le=PM_MAX, nullable=True, alias="PM2.5")
    PM10: Series[float] = pa.Field(ge=PM_MIN, le=PM_MAX, nullable=True)
    SO2: Series[float] = pa.Field(ge=PM_MIN, le=SO2_MAX, nullable=True)
    NO2: Series[float] = pa.Field(ge=PM_MIN, le=NO2_MAX, nullable=True)
    CO: Series[float] = pa.Field(ge=PM_MIN, le=CO_MAX, nullable=True)
    O3: Series[float] = pa.Field(ge=PM_MIN, le=O3_MAX, nullable=True)
    TEMP: Series[float] = pa.Field(ge=TEMP_MIN, le=TEMP_MAX, nullable=True)
    PRES: Series[float] = pa.Field(ge=PRES_MIN, le=PRES_MAX, nullable=True)
    DEWP: Series[float] = pa.Field(ge=DEWP_MIN, le=DEWP_MAX, nullable=True)
    RAIN: Series[float] = pa.Field(ge=PM_MIN, le=RAIN_MAX, nullable=True)
    WSPM: Series[float] = pa.Field(ge=PM_MIN, le=WSPM_MAX, nullable=True)

    class Config:
        strict = False
        coerce = True

    @pa.dataframe_check(name="volumen_minimo")
    def volumen_minimo(cls, df: pd.DataFrame) -> bool:
        """Alerta si el volumen cae drasticamente (ingesta cortada)."""
        return len(df) >= VOLUMEN_MINIMO

    @pa.dataframe_check(name="eje_temporal_sin_futuro")
    def eje_temporal_sin_futuro(cls, df: pd.DataFrame) -> Series[bool]:
        """Ninguna lectura puede tener fecha futura.

        Un timestamp futuro suele ser un bug de unidades (ms leidos como s) o
        de zona horaria. Si entra al entrenamiento, el split temporal queda mal
        y la metrica reportada es optimista sin que nada falle.
        """
        return df["datetime"] <= pd.Timestamp.now()

    @pa.dataframe_check(name="punto_rocio_no_mayor_que_temperatura")
    def punto_rocio_no_mayor_que_temperatura(cls, df: pd.DataFrame) -> Series[bool]:
        """Regla fisica: el punto de rocio nunca supera la temperatura.

        Solo se evalua donde ambas magnitudes estan presentes: si TEMP o DEWP
        faltan, la comparacion no tiene sentido (el sensor no reporto esa hora).
        Se cumplio sin excepciones en las 420.768 filas reales.
        """
        ambas = df["TEMP"].notna() & df["DEWP"].notna()
        return ~ambas | df["TEMP"].ge(df["DEWP"])

    @pa.dataframe_check(name="contaminantes_medianamente_cubiertos")
    def contaminantes_medianamente_cubiertos(cls, df: pd.DataFrame) -> bool:
        """Menos del 30 % de nulos en cada contaminante.

        Un sensor que falla el 60 % de las horas ya no reporta ruido: esta
        muerto, y el fillna imputaria valores inventados sobre casi todo el
        registro. Mejor fallar alto y avisar que imputar en silencio.
        """
        return all(df[c].notna().mean() >= 0.7 for c in _CONTA_MIN)

    @pa.dataframe_check(name="variacion_de_target")
    def variacion_de_target(cls, df: pd.DataFrame) -> bool:
        """El target tiene que variar dentro de la particion.

        Si PM2.5 es constante en un tramo, casi siempre es un tramo con el
        sensor saturado o con la ingesta filtrada, no aire perfectamente igual.
        """
        return df[fc.TARGET].dropna().nunique() > 1


class RegistrosProcesados(pa.DataFrameModel):
    """Contrato del dataset listo para entrenar.

    Los rangos son mas duros que en el crudo: si algo llega hasta aqui fuera de
    rango, el bug esta en NUESTRO pipeline y no en el proveedor. El target ya
    no tiene nulos: los que faltaban se descartaron en ``limpiar``.
    """

    station: Series[str] = pa.Field(nullable=False)
    wd: Series[str] = pa.Field(nullable=False, isin=set(RUMBOS_VALIDOS) | {"desconocido"})
    temporada: Series[str] = pa.Field(
        nullable=False, isin={"invierno", "primavera", "verano", "otonio"}
    )
    # Numericas ya imputadas: sin nulos y dentro de rango.
    PM10: Series[float] = pa.Field(ge=PM_MIN, le=PM_MAX, nullable=False)
    SO2: Series[float] = pa.Field(ge=PM_MIN, le=SO2_MAX, nullable=False)
    NO2: Series[float] = pa.Field(ge=PM_MIN, le=NO2_MAX, nullable=False)
    CO: Series[float] = pa.Field(ge=PM_MIN, le=CO_MAX, nullable=False)
    O3: Series[float] = pa.Field(ge=PM_MIN, le=O3_MAX, nullable=False)
    TEMP: Series[float] = pa.Field(ge=TEMP_MIN, le=TEMP_MAX, nullable=False)
    PRES: Series[float] = pa.Field(ge=PRES_MIN, le=PRES_MAX, nullable=False)
    DEWP: Series[float] = pa.Field(ge=DEWP_MIN, le=DEWP_MAX, nullable=False)
    RAIN: Series[float] = pa.Field(ge=PM_MIN, le=RAIN_MAX, nullable=False)
    WSPM: Series[float] = pa.Field(ge=PM_MIN, le=WSPM_MAX, nullable=False)
    hora: Series[int] = pa.Field(ge=0, le=23, nullable=False)
    dia_semana: Series[int] = pa.Field(ge=0, le=6, nullable=False)
    mes: Series[int] = pa.Field(ge=1, le=12, nullable=False)
    PM2_5: Series[float] = pa.Field(ge=PM_MIN, le=PM_MAX, nullable=False, alias="PM2.5")

    class Config:
        strict = False
        coerce = True

    @pa.dataframe_check(name="target_no_constante")
    def target_no_constante(cls, df: pd.DataFrame) -> bool:
        """El target tiene que variar (un modelo sobre constante es inutil)."""
        return df[fc.TARGET].nunique() > 1

    @pa.dataframe_check(name="punto_rocio_no_mayor_que_temperatura")
    def punto_rocio_no_mayor_que_temperatura(cls, df: pd.DataFrame) -> Series[bool]:
        """La regla fisica se respeta mientras no se haya imputado la magnitud.

        La imputacion rellena TEMP y DEWP con sus medianas de forma
        independiente, y una mediana puede quedar por debajo de un punto de
        rocio real alto. Por eso se salta la fila cuando alguna de las dos fue
        imputada (marcada por la columna ``<col>_era_nulo`` de ``limpiar``).
        """
        imputada = df["TEMP_era_nulo"].eq(1) | df["DEWP_era_nulo"].eq(1)
        return imputada | df["TEMP"].ge(df["DEWP"])


def validar_crudos(df: pd.DataFrame, *, lazy: bool = True) -> pd.DataFrame:
    """Valida el dataframe crudo contra el contrato.

    Args:
        df: dataframe recien leido de la fuente.
        lazy: si es True, acumula TODOS los errores antes de fallar.

    Raises:
        pandera.errors.SchemaError | SchemaErrors: con el detalle de que fallo.
    """
    return RegistrosCrudos.validate(df, lazy=lazy)


def validar_procesados(df: pd.DataFrame, *, lazy: bool = True) -> pd.DataFrame:
    """Valida el dataframe procesado contra el contrato."""
    return RegistrosProcesados.validate(df, lazy=lazy)


def resumen_contrato() -> dict[str, list[str]]:
    """Devuelve el contrato como diccionario, para documentarlo o loguearlo."""
    return {
        "crudas_requeridas": fc.COLUMNAS_CRUDAS_REQUERIDAS,
        "features_categoricas": fc.FEATURES_CATEGORICAS,
        "features_numericas": fc.FEATURES_NUMERICAS,
        "target": [fc.TARGET],
    }
