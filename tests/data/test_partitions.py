"""Pruebas de los límites temporales de las particiones."""

from __future__ import annotations

import pandas as pd

from BeijingAir.config import COL_TIEMPO, Particion
from BeijingAir.data.descarga import filtrar


def test_particion_incluye_todas_las_horas_del_dia_final() -> None:
    """La fecha final representa un día completo, no solo su medianoche."""
    dataframe = pd.DataFrame(
        {
            COL_TIEMPO: pd.to_datetime(
                ["2015-06-30 00:00", "2015-06-30 23:00", "2015-07-01 00:00"]
            ),
            "valor": [1, 2, 3],
        }
    )

    resultado = filtrar(dataframe, Particion("validacion", "2015-06-30", "2015-06-30"))

    assert resultado["valor"].tolist() == [1, 2]
