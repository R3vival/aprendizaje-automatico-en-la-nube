"""Pruebas de la construccion de features."""

from __future__ import annotations

import pandas as pd
import pytest

from BeijingAir.features import contract as fc


def _df_crudo() -> pd.DataFrame:
    """Un dataframe crudo minimo con las columnas requeridas."""
    return pd.DataFrame(
        {
            fc.COL_TIEMPO: pd.to_datetime(["2023-01-01 08:00"]),
            "station": ["dongsi"],
            "wd": ["N"],
            "PM10": [20.0],
            "SO2": [5.0],
            "NO2": [30.0],
            "CO": [600.0],
            "O3": [40.0],
            "TEMP": [1.0],
            "PRES": [1020.0],
            "DEWP": [-2.0],
            "RAIN": [0.0],
            "WSPM": [2.0],
        }
    )


def test_construir_features_es_idempotente() -> None:
    una_vez = fc.construir_features(_df_crudo())
    dos_veces = fc.construir_features(una_vez)
    pd.testing.assert_frame_equal(una_vez, dos_veces)


def test_construir_features_deriva_temporada_de_enero() -> None:
    out = fc.construir_features(_df_crudo())
    assert out.loc[0, fc.COL_TEMPORADA] == "invierno"  # enero


def test_construir_features_deriva_temporada_de_verano() -> None:
    df = _df_crudo()
    df[fc.COL_TIEMPO] = pd.to_datetime(["2023-07-01 12:00"])
    out = fc.construir_features(df)
    assert out.loc[0, fc.COL_TEMPORADA] == "verano"


def test_construir_features_agrega_la_feature_de_hora() -> None:
    out = fc.construir_features(_df_crudo())
    assert int(out.loc[0, fc.COL_HORA]) == 8


def test_construir_features_falla_si_falta_columna() -> None:
    df = _df_crudo().drop(columns=["TEMP"])
    with pytest.raises(KeyError):
        fc.construir_features(df)


def test_a_diccionarios_requiere_features_derivadas() -> None:
    with pytest.raises(KeyError):
        fc.a_diccionarios(_df_crudo())  # falta llamar a construir_features
