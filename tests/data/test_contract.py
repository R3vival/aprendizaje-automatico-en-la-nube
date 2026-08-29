"""Pruebas del contrato de datos sin descargar el dataset real.

Los fixtures recrean fallos silenciosos del proveedor: el dataframe sigue siendo
usable por pandas, pero el contrato debe impedir que llegue al entrenamiento.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from BeijingAir.data.contract import validar_crudos

ERRORES_CONTRATO = (SchemaError, SchemaErrors)
FILAS_VALIDAS = 120


def construir_crudo_valido(*, desplazamiento: float = 0.0) -> pd.DataFrame:
    """Construye un lote horario válido, independiente de red y de archivos."""
    indice = range(FILAS_VALIDAS)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2016-01-01", periods=FILAS_VALIDAS, freq="h"),
            "station": ["Aotizhongxin"] * FILAS_VALIDAS,
            "wd": ["N"] * FILAS_VALIDAS,
            "PM2.5": [20.0 + desplazamiento + (i % 8) for i in indice],
            "PM10": [40.0 + desplazamiento + (i % 8) for i in indice],
            "SO2": [10.0 + (i % 3) for i in indice],
            "NO2": [30.0 + (i % 5) for i in indice],
            "CO": [800.0 + (i % 20) for i in indice],
            "O3": [50.0 + (i % 7) for i in indice],
            "TEMP": [15.0 + (i % 4) for i in indice],
            "PRES": [1_010.0 + (i % 3) for i in indice],
            "DEWP": [8.0 + (i % 3) for i in indice],
            "RAIN": [0.0] * FILAS_VALIDAS,
            "WSPM": [2.0 + (i % 2) for i in indice],
        }
    )


@pytest.fixture
def crudo_valido() -> pd.DataFrame:
    """Lote que cumple los límites físicos y de cobertura del proveedor."""
    return construir_crudo_valido()


@pytest.fixture
def pm2_5_fuera_de_rango(crudo_valido: pd.DataFrame) -> pd.DataFrame:
    """Simula un cambio de escala que aún conserva un dtype numérico válido."""
    roto = crudo_valido.copy()
    roto.loc[0, "PM2.5"] = 1_200.0
    return roto


@pytest.fixture
def sensor_co_caido(crudo_valido: pd.DataFrame) -> pd.DataFrame:
    """Simula un sensor de CO que deja de reportar más del 30 % del lote."""
    roto = crudo_valido.copy()
    roto.loc[:47, "CO"] = None
    return roto


@pytest.fixture
def punto_rocio_imposible(crudo_valido: pd.DataFrame) -> pd.DataFrame:
    """Simula valores físicamente inconsistentes que pandas puede procesar."""
    roto = crudo_valido.copy()
    roto.loc[0, "DEWP"] = roto.loc[0, "TEMP"] + 1.0
    return roto


@pytest.fixture
def lectura_estacion_hora_duplicada(crudo_valido: pd.DataFrame) -> pd.DataFrame:
    """Simula una descarga que repite una lectura de la misma estación y hora."""
    roto = crudo_valido.copy()
    roto.loc[1, "datetime"] = roto.loc[0, "datetime"]
    roto.loc[1, "station"] = roto.loc[0, "station"]
    return roto


@pytest.mark.parametrize(
    "nombre_fixture",
    [
        "pm2_5_fuera_de_rango",
        "sensor_co_caido",
        "punto_rocio_imposible",
        "lectura_estacion_hora_duplicada",
    ],
)
def test_contrato_rechaza_datos_rotos(
    request: pytest.FixtureRequest, nombre_fixture: str
) -> None:
    """Los tres incidentes silenciosos se rechazan antes del entrenamiento."""
    dataframe_roto = request.getfixturevalue(nombre_fixture)

    with pytest.raises(ERRORES_CONTRATO):
        validar_crudos(dataframe_roto)


@pytest.mark.parametrize("desplazamiento", [0.0, 5.0, 10.0])
def test_control_negativo_lotes_validos_independientes(desplazamiento: float) -> None:
    """Un contrato útil no inventa fallos sobre varios lotes correctos."""
    validar_crudos(construir_crudo_valido(desplazamiento=desplazamiento))
