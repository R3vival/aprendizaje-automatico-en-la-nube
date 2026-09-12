"""Esquemas HTTP de la API de prediccion.

La peticion representa una lectura cruda de una estacion. Las features de
calendario se derivan con el mismo modulo que se uso durante el entrenamiento;
el cliente nunca tiene que adivinar ``hora``, ``mes`` o ``temporada``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from BeijingAir.data import contract as dc
from BeijingAir.features import contract as fc


class PrediccionEntrada(BaseModel):
    """Una observacion horaria lista para estimar PM2.5."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    datetime: datetime
    station: str = Field(min_length=1, max_length=80)
    wd: str = Field(min_length=1, max_length=3)
    PM10: float = Field(ge=dc.PM_MIN, le=dc.PM_MAX)
    SO2: float = Field(ge=dc.PM_MIN, le=dc.SO2_MAX)
    NO2: float = Field(ge=dc.PM_MIN, le=dc.NO2_MAX)
    CO: float = Field(ge=dc.PM_MIN, le=dc.CO_MAX)
    O3: float = Field(ge=dc.PM_MIN, le=dc.O3_MAX)
    TEMP: float = Field(ge=dc.TEMP_MIN, le=dc.TEMP_MAX)
    PRES: float = Field(ge=dc.PRES_MIN, le=dc.PRES_MAX)
    DEWP: float = Field(ge=dc.DEWP_MIN, le=dc.DEWP_MAX)
    RAIN: float = Field(ge=dc.PM_MIN, le=dc.RAIN_MAX)
    WSPM: float = Field(ge=dc.PM_MIN, le=dc.WSPM_MAX)

    @field_validator("wd")
    @classmethod
    def validar_rumbo(cls, valor: str) -> str:
        """Normaliza y comprueba la direccion documentada por el proveedor."""
        rumbo = valor.upper()
        if rumbo not in dc.RUMBOS_VALIDOS:
            permitidos = ", ".join(sorted(dc.RUMBOS_VALIDOS))
            raise ValueError(f"wd debe ser uno de: {permitidos}")
        return rumbo

    def a_dataframe(self) -> pd.DataFrame:
        """Construye exactamente las features que recibio el pipeline entrenado."""
        lectura = pd.DataFrame([self.model_dump()])
        con_features = fc.construir_features(lectura)
        return con_features[fc.FEATURES]


class PrediccionSalida(BaseModel):
    """Respuesta auditable de una prediccion individual."""

    prediccion_pm25: float
    model_name: str
    model_version: str
    latencia_ms: float = Field(ge=0)


#: Tope de predicciones por lote. Acota la memoria por request (un lote sin
#: limite es un vector de denegacion de servicio) y la latencia de cola. Vive en
#: el schema para que aparezca en OpenAPI y el cliente lo vea antes de mandar.
MAX_LOTE: int = 500


class LotePrediccion(BaseModel):
    """Lote de lecturas horarias para predecir en una sola llamada."""

    model_config = ConfigDict(extra="forbid")

    lecturas: list[PrediccionEntrada] = Field(min_length=1, max_length=MAX_LOTE)


class LotePrediccionSalida(BaseModel):
    """Respuesta agregada de un lote, con la version que lo produjo."""

    predicciones: list[PrediccionSalida]
    model_name: str
    model_version: str


class SaludSalida(BaseModel):
    """Estado de vida y disponibilidad del modelo, sin filtrar detalles internos."""

    estado: Literal["ok", "degradado"]
    modelo_cargado: bool
    model_uri: str
    model_version: str | None = None


class ModeloSalida(BaseModel):
    """Metadatos del contrato del modelo que la API intenta servir."""

    model_name: str
    model_uri: str
    model_version: str | None = None
    features_entrada: list[str]
