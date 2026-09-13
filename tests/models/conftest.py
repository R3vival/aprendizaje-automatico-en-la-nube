"""Fixtures compartidas por las pruebas de ``models/``.

El holdout sintetico vive aqui y no dentro de un modulo de tests porque lo usan
tanto la prueba de la politica (``test_evaluate.py``) como la de la capa que
habla con MLflow (``test_promote.py``). Duplicarlo garantizaria que un dia
divergieran y que una de las dos dejara de probar lo que cree probar.
"""

from __future__ import annotations

import pandas as pd
import pytest

from BeijingAir.data import contract as dc


@pytest.fixture
def holdout_valido() -> pd.DataFrame:
    """Un holdout sintetico que cumple el contrato de datos procesados.

    120 filas, mitad invierno y mitad verano, para que ambos subgrupos superen
    el minimo de filas que exige ``metricas_por_subgrupo``.
    """
    n = 120
    df = pd.DataFrame(
        {
            "station": ["Aotizhongxin"] * n,
            "wd": ["N"] * n,
            "temporada": ["invierno"] * (n // 2) + ["verano"] * (n // 2),
            "PM10": [40.0] * n,
            "SO2": [10.0] * n,
            "NO2": [30.0] * n,
            "CO": [800.0] * n,
            "O3": [50.0] * n,
            "TEMP": [15.0] * n,
            "PRES": [1010.0] * n,
            "DEWP": [8.0] * n,
            "RAIN": [0.0] * n,
            "WSPM": [2.0] * n,
            "hora": list(range(24)) * 5,
            "dia_semana": [0] * n,
            "mes": [7] * n,
            "PM2.5": [float(20 + (i % 40)) for i in range(n)],
            "TEMP_era_nulo": [0] * n,
            "DEWP_era_nulo": [0] * n,
        }
    )
    dc.validar_procesados(df)
    return df
