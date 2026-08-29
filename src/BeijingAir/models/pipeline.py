"""Pipeline reproducible para estimar PM2.5.

La logica de entrenamiento vive en ``src/`` y no en un notebook: asi los tests,
MLflow y el futuro orquestador ejecutan exactamente el mismo codigo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from BeijingAir.config import SEMILLA
from BeijingAir.features import contract as fc

TipoModelo = Literal["baseline", "bosque"]


@dataclass(frozen=True)
class ResultadoEvaluacion:
    """Metricas globales y por estacion de una corrida de regresion."""

    mae: float
    rmse: float
    r2: float
    peor_mae_estacion: float
    mae_por_estacion: dict[str, float]

    def como_dict(self) -> dict[str, float | dict[str, float]]:
        """Serializa el resultado para MLflow y para el model card."""
        return asdict(self)


def separar_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa las columnas declaradas del target sin permitir leakage."""
    requeridas = [*fc.FEATURES, fc.TARGET]
    faltantes = [columna for columna in requeridas if columna not in df.columns]
    if faltantes:
        raise KeyError(f"Faltan columnas para entrenar: {faltantes}")
    return df[fc.FEATURES].copy(), df[fc.TARGET].astype(float).copy()


def crear_pipeline(
    modelo: TipoModelo,
    *,
    semilla: int = SEMILLA,
    n_estimators: int = 300,
) -> Pipeline:
    """Crea un pipeline que aprende transformaciones solo sobre el train.

    ``OneHotEncoder(handle_unknown='ignore')`` evita que una estacion o rumbo
    nuevo rompa el serving. El imputador dentro del pipeline conserva la misma
    transformacion cuando el modelo llegue a una API.
    """
    preprocesador = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                Pipeline(
                    steps=[
                        ("imputar", SimpleImputer(strategy="most_frequent")),
                        ("codificar", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                fc.FEATURES_CATEGORICAS,
            ),
            ("numericas", SimpleImputer(strategy="median"), fc.FEATURES_NUMERICAS),
        ],
        remainder="drop",
    )

    estimador: DummyRegressor | RandomForestRegressor
    if modelo == "baseline":
        estimador = DummyRegressor(strategy="mean")
    elif modelo == "bosque":
        estimador = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=semilla,
        )
    else:
        raise ValueError(f"Modelo no soportado: {modelo}")

    return Pipeline(steps=[("preprocesar", preprocesador), ("modelo", estimador)])


def evaluar_regresion(
    y_real: pd.Series,
    predicciones: np.ndarray,
    estaciones: pd.Series,
) -> ResultadoEvaluacion:
    """Calcula metricas globales y el MAE de la estacion peor atendida."""
    if len(y_real) != len(predicciones) or len(y_real) != len(estaciones):
        raise ValueError("Target, predicciones y estaciones deben tener la misma longitud.")

    errores_absolutos = pd.Series(
        np.abs(y_real.to_numpy() - predicciones), index=estaciones.index, name="error_absoluto"
    )
    mae_por_estacion = errores_absolutos.groupby(estaciones.astype(str)).mean().sort_index()

    return ResultadoEvaluacion(
        mae=float(mean_absolute_error(y_real, predicciones)),
        rmse=float(root_mean_squared_error(y_real, predicciones)),
        r2=float(r2_score(y_real, predicciones)),
        peor_mae_estacion=float(mae_por_estacion.max()),
        mae_por_estacion={estacion: float(mae) for estacion, mae in mae_por_estacion.items()},
    )
