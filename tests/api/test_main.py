"""Pruebas de contrato HTTP sin requerir un servidor de MLflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from BeijingAir.api.main import (
    DETALLE_ERROR_INTERNO,
    DETALLE_MODELO_NO_DISPONIBLE,
    crear_app,
)
from BeijingAir.api.modelo import CargadorModelo, Predictor
from BeijingAir.features import contract as fc


class ModeloFalso:
    """Predictor determinista que deja comprobar el dataframe recibido."""

    def __init__(self) -> None:
        self.columnas_recibidas: list[str] = []

    def predict(self, datos: pd.DataFrame) -> np.ndarray:
        self.columnas_recibidas = datos.columns.tolist()
        return np.array([42.5])


def _modelo_falso(_: str) -> Predictor:
    return ModeloFalso()


def _fallar_carga(_: str) -> Predictor:
    raise RuntimeError("Registry no disponible")


def _payload_valido() -> dict[str, str | float]:
    return {
        "datetime": "2016-07-01T08:00:00",
        "station": "Aotizhongxin",
        "wd": "n",
        "PM10": 40.0,
        "SO2": 10.0,
        "NO2": 30.0,
        "CO": 800.0,
        "O3": 50.0,
        "TEMP": 15.0,
        "PRES": 1010.0,
        "DEWP": 8.0,
        "RAIN": 0.0,
        "WSPM": 2.0,
    }


def test_predict_deriva_features_y_reporta_version() -> None:
    """La API reutiliza las features del paquete y responde con trazabilidad."""
    predictor = ModeloFalso()
    cargador = CargadorModelo(uri="runs:/prueba/modelo", cargar_modelo=lambda _: predictor)

    with TestClient(crear_app(cargador)) as cliente:
        respuesta = cliente.post("/predict", json=_payload_valido())

    assert respuesta.status_code == 200
    assert respuesta.json()["prediccion_pm25"] == 42.5
    assert respuesta.json()["model_version"] == "desconocida"
    assert predictor.columnas_recibidas == fc.FEATURES


def test_health_degradada_y_predict_rechaza_sin_modelo() -> None:
    """No se sirven candidatos ni se cae el proceso cuando falta champion."""
    cargador = CargadorModelo(uri="runs:/prueba/modelo", cargar_modelo=_fallar_carga)

    with TestClient(crear_app(cargador)) as cliente:
        salud = cliente.get("/health")
        respuesta = cliente.post("/predict", json=_payload_valido())

    assert salud.status_code == 200
    assert salud.json()["estado"] == "degradado"
    assert respuesta.status_code == 503
    assert respuesta.json()["detail"] == DETALLE_MODELO_NO_DISPONIBLE


def test_predict_rechaza_datos_fuera_del_contrato() -> None:
    """Los limites del contrato de datos se aplican antes de llamar al modelo."""
    cargador = CargadorModelo(uri="runs:/prueba/modelo", cargar_modelo=_modelo_falso)
    payload = _payload_valido()
    payload["PM10"] = 1_001.0

    with TestClient(crear_app(cargador)) as cliente:
        respuesta = cliente.post("/predict", json=payload)

    assert respuesta.status_code == 422


class ModeloQueFalla:
    """Predictor que revienta con un mensaje que NO debe llegar al cliente."""

    SECRETO = "/home/equipo/.mlflow/credenciales.json"

    def predict(self, _: pd.DataFrame) -> np.ndarray:
        raise RuntimeError(f"no pude abrir {self.SECRETO}")


def test_un_fallo_interno_no_filtra_el_detalle_al_cliente() -> None:
    """Antipatron que la API evita a proposito: ``HTTPException(detail=str(e))``.

    El mensaje de una excepcion suele traer rutas del servidor, nombres de
    columnas y a veces credenciales. El detalle va al log; al cliente le llega
    una respuesta estable.
    """
    cargador = CargadorModelo(uri="runs:/prueba/modelo", cargar_modelo=lambda _: ModeloQueFalla())

    with TestClient(crear_app(cargador), raise_server_exceptions=False) as cliente:
        respuesta = cliente.post("/predict", json=_payload_valido())

    assert respuesta.status_code == 500
    assert respuesta.json()["detail"] == DETALLE_ERROR_INTERNO
    assert ModeloQueFalla.SECRETO not in respuesta.text


def test_metrics_expone_las_metricas_propias_del_servicio() -> None:
    """Prometheus mide el SERVICIO; Evidently mide los DATOS. Son dos preguntas.

    El contador se etiqueta por resultado para poder distinguir una API sana de
    una que responde 200 sirviendo errores.
    """
    cargador = CargadorModelo(uri="runs:/prueba/modelo", cargar_modelo=_modelo_falso)

    with TestClient(crear_app(cargador)) as cliente:
        cliente.post("/predict", json=_payload_valido())
        cuerpo = cliente.get("/metrics").text

    assert "beijing_air_predicciones_total" in cuerpo
    assert "beijing_air_prediccion_latencia_ms" in cuerpo
    assert 'resultado="ok"' in cuerpo
