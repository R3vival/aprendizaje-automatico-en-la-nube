"""Pruebas del pipeline de modelos sin descargar el dataset Beijing."""

from __future__ import annotations

import pandas as pd
import pytest

from BeijingAir.features import contract as fc
from BeijingAir.models.pipeline import crear_pipeline, evaluar_regresion, separar_features_target


def dataframe_modelo() -> pd.DataFrame:
    """Construye un dataset mínimo con el esquema de features del proyecto."""
    filas = 16
    data: dict[str, list[float] | list[str]] = {}
    for posicion, columna in enumerate(fc.FEATURES_NUMERICAS):
        data[columna] = [float(posicion + fila) for fila in range(filas)]

    data["station"] = ["Aotizhongxin", "Changping"] * (filas // 2)
    data["wd"] = ["N", "SE"] * (filas // 2)
    data["temporada"] = ["invierno", "verano"] * (filas // 2)
    data[fc.TARGET] = [25.0 + float(fila % 6) for fila in range(filas)]
    return pd.DataFrame(data)


def test_bosque_predice_y_reporta_metricas_por_estacion() -> None:
    """El modelo usa las features declaradas y conserva métricas por subgrupo."""
    datos = dataframe_modelo()
    x, y = separar_features_target(datos)
    pipeline = crear_pipeline("bosque", n_estimators=10)
    pipeline.fit(x, y)

    predicciones = pipeline.predict(x)
    resultado = evaluar_regresion(y, predicciones, datos[fc.COL_SUBGRUPO])

    assert len(predicciones) == len(datos)
    assert set(resultado.mae_por_estacion) == {"Aotizhongxin", "Changping"}
    assert resultado.mae >= 0
    assert resultado.rmse >= 0


def test_pipeline_acepta_categoria_no_vista() -> None:
    """Una estación nueva no rompe la predicción gracias a OneHotEncoder."""
    datos = dataframe_modelo()
    x, y = separar_features_target(datos)
    pipeline = crear_pipeline("baseline")
    pipeline.fit(x, y)

    entrada_nueva = x.iloc[[0]].copy()
    entrada_nueva.loc[:, "station"] = "Estacion-nueva"

    assert len(pipeline.predict(entrada_nueva)) == 1


def test_evaluacion_rechaza_longitudes_distintas() -> None:
    """No se deben asociar predicciones con estaciones de otra partición."""
    with pytest.raises(ValueError, match="misma longitud"):
        evaluar_regresion(
            pd.Series([1.0, 2.0]),
            predicciones=[1.0],
            estaciones=pd.Series(["A", "B"]),
        )
