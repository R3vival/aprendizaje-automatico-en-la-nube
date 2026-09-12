"""Carga perezosa y auditable del modelo desde MLflow Model Registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from BeijingAir.config import MLFLOW_TRACKING_URI, MODELO_REGISTRADO, MODELO_URI


class Predictor(Protocol):
    """Lo minimo que necesita el servicio de un modelo cargado."""

    def predict(self, datos: pd.DataFrame) -> object:
        """Devuelve una prediccion para cada fila de ``datos``."""


@dataclass(frozen=True)
class ModeloCargado:
    """Modelo en memoria y la version resuelta en el Registry."""

    predictor: Predictor
    nombre: str
    version: str
    uri: str


def _nombre_y_alias(uri: str) -> tuple[str, str] | None:
    """Extrae ``nombre`` y ``alias`` solo de una URI models:/name@alias."""
    prefijo = "models:/"
    if not uri.startswith(prefijo):
        return None
    referencia = uri.removeprefix(prefijo)
    if "@" not in referencia:
        return None
    nombre, alias = referencia.rsplit("@", maxsplit=1)
    if not nombre or not alias:
        return None
    return nombre, alias


class CargadorModelo:
    """Mantiene el servicio vivo aunque MLflow aun no tenga un champion.

    La carga se intenta al iniciar la aplicacion. Si el Registry no esta
    disponible o aun falta promocion, ``/health`` sigue respondiendo y
    ``/predict`` devuelve 503, en vez de iniciar un ciclo de reinicios.
    """

    def __init__(
        self,
        *,
        uri: str = MODELO_URI,
        cargar_modelo: Callable[[str], Predictor] = mlflow.pyfunc.load_model,
    ) -> None:
        self.uri = uri
        self._cargar_modelo = cargar_modelo
        self._modelo: ModeloCargado | None = None
        self._error: str | None = None

    @property
    def listo(self) -> bool:
        """Indica si se puede atender una prediccion."""
        return self._modelo is not None

    @property
    def error(self) -> str | None:
        """Motivo resumido del estado degradado, util solo en logs."""
        return self._error

    @property
    def version(self) -> str | None:
        """Version resuelta, o ``None`` si el modelo no esta disponible."""
        return None if self._modelo is None else self._modelo.version

    @property
    def nombre(self) -> str:
        """Nombre del modelo aun cuando la URI no sea del Registry."""
        referencia = _nombre_y_alias(self.uri)
        return MODELO_REGISTRADO if referencia is None else referencia[0]

    def cargar(self) -> None:
        """Resuelve el alias y carga el predictor sin exponer fallos al cliente."""
        self._modelo = None
        self._error = None
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            referencia = _nombre_y_alias(self.uri)
            version = "desconocida"
            if referencia is not None:
                nombre, alias = referencia
                version = str(MlflowClient().get_model_version_by_alias(nombre, alias).version)
            predictor = self._cargar_modelo(self.uri)
            self._modelo = ModeloCargado(predictor, self.nombre, version, self.uri)
        except Exception as error:  # La disponibilidad del Registry es externa.
            self._error = f"{type(error).__name__}: {error}"

    def predecir(self, datos: pd.DataFrame) -> float:
        """Predice una sola fila o avisa que el modelo aun no esta disponible."""
        if self._modelo is None:
            raise RuntimeError("El modelo no esta cargado.")
        predicciones = np.asarray(self._modelo.predictor.predict(datos)).reshape(-1)
        if len(predicciones) != 1:
            raise ValueError("El predictor debe devolver exactamente una prediccion por solicitud.")
        return float(predicciones[0])

    def predecir_lote(self, datos: pd.DataFrame) -> list[float]:
        """Predice varias filas de una vez (batch).

        A diferencia de ``predecir``, no exige que la salida tenga una sola fila:
        amortiza el costo de inferencia por lote en una sola llamada.
        """
        if self._modelo is None:
            raise RuntimeError("El modelo no esta cargado.")
        predicciones = np.asarray(self._modelo.predictor.predict(datos)).reshape(-1)
        if len(predicciones) != len(datos):
            raise ValueError(
                f"El predictor devolvio {len(predicciones)} predicciones para {len(datos)} filas."
            )
        return [float(p) for p in predicciones]
