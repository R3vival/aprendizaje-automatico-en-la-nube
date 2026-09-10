"""Pruebas de la preparacion de datos: limpieza y split temporal."""

from __future__ import annotations

import pandas as pd
import pytest

from BeijingAir.data.loaders import limpiar, split_temporal
from BeijingAir.features import contract as fc


def _df_minimo() -> pd.DataFrame:
    """Un dataframe crudo minimo: una fila con target nulo, un nulo numerico y
    un nulo categorico en filas DISTINTAS, para ejercitar la estrategia de
    imputacion."""
    return pd.DataFrame(
        {
            fc.COL_TIEMPO: pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "PM2.5": [None, 10.0, 30.0],  # fila 0: target nulo (se descarta)
            "PM10": [20.0, None, 40.0],  # fila 1: nulo numerico
            "wd": ["N", "S", None],  # fila 2: nulo categorico
            "TEMP": [1.0, 2.0, 3.0],
        }
    )


def test_limpiar_descarta_filas_con_target_nulo() -> None:
    out = limpiar(_df_minimo())
    assert out[fc.TARGET].notna().all()
    assert len(out) == 2  # la fila con PM2.5 nulo se descarto


def test_limpiar_agrega_indicador_de_ausencia() -> None:
    out = limpiar(_df_minimo())
    assert "PM10_era_nulo" in out.columns
    fila_con_nulo = out.loc[out["PM10_era_nulo"] == 1]
    assert len(fila_con_nulo) == 1
    assert int(fila_con_nulo["PM10_era_nulo"].iloc[0]) == 1


def test_limpiar_imputa_numericas_con_la_mediana() -> None:
    out = limpiar(_df_minimo())
    # PM10 = [20, None, 40]; al descartar la fila 0 (target nulo) quedan
    # [None, 40] -> mediana 40
    assert out.loc[out["PM10_era_nulo"] == 1, "PM10"].iloc[0] == 40.0


def test_limpiar_rellena_categorica_con_desconocido() -> None:
    out = limpiar(_df_minimo())
    assert "desconocido" in set(out["wd"])


def test_split_temporal_rechaza_datos_desordenados() -> None:
    df = _df_minimo().iloc[::-1]  # invertido a proposito
    with pytest.raises(ValueError):
        split_temporal(df)


def test_split_temporal_corta_por_posicion() -> None:
    df = _df_minimo().sort_values(fc.COL_TIEMPO)
    train, test = split_temporal(df, fraccion_train=0.67)
    assert len(train) == 2
    assert len(test) == 1


def test_los_indicadores_de_ausencia_no_son_features_del_modelo() -> None:
    """Decision declarada en docs/dataset-card.md, blindada aqui.

    ``limpiar`` produce ``<col>_era_nulo`` como trazabilidad de la imputacion,
    no como senal para el modelo. Promoverlos a feature sin que la API acepte
    valores ausentes los volveria constantes en serving y variables en
    entrenamiento: train/serve skew introducido a mano. Si alguien los agrega a
    FEATURES, este test lo obliga a resolver antes el contrato de la API.
    """
    indicadores = {f"{columna}_era_nulo" for columna in fc.CRUDAS_NUMERICAS}

    assert not indicadores & set(fc.FEATURES)
