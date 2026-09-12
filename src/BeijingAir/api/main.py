"""Aplicacion FastAPI para servir estimaciones de PM2.5."""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter

import pandas as pd
from fastapi import FastAPI, HTTPException, Response, status

from BeijingAir.api.metricas import contenido_prometheus, registrar_prediccion
from BeijingAir.api.modelo import CargadorModelo
from BeijingAir.api.schemas import (
    LotePrediccion,
    LotePrediccionSalida,
    ModeloSalida,
    PrediccionEntrada,
    PrediccionSalida,
    SaludSalida,
)
from BeijingAir.config import PROYECTO, VERSION
from BeijingAir.features import contract as fc

LOGGER = logging.getLogger(__name__)
DETALLE_MODELO_NO_DISPONIBLE = "Modelo no disponible; falta cargar un modelo promovido."
#: Respuesta estable ante un fallo interno. Es constante a proposito: devolver
#: ``str(excepcion)`` filtra rutas, nombres de columnas y a veces credenciales
#: al cliente. El detalle va al log, no a la respuesta.
DETALLE_ERROR_INTERNO = "No fue posible calcular la prediccion."


def _error_interno() -> HTTPException:
    """Construye un 500 estable con un id de correlacion para rastrear el log.

    El detalle va al log (con ``logger.exception`` en el llamador); al cliente
    le llega un mensaje estable mas el ``id_correlacion``, que permite encontrar
    la traza exacta del incidente sin filtrar el interior del servicio.
    """
    id_correlacion = uuid.uuid4().hex[:8]
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={"detalle": DETALLE_ERROR_INTERNO, "id_correlacion": id_correlacion},
    )


@asynccontextmanager
async def vida_util(app: FastAPI) -> AsyncIterator[None]:
    """Carga el modelo una vez sin convertir una ausencia de Registry en crash loop."""
    cargador: CargadorModelo = app.state.cargador
    cargador.cargar()
    if not cargador.listo:
        LOGGER.warning("La API inicia degradada: %s", cargador.error)
    yield


def crear_app(cargador: CargadorModelo | None = None) -> FastAPI:
    """Crea la app, permitiendo inyectar un cargador falso en pruebas."""
    app = FastAPI(
        title="Beijing Air PM2.5 API",
        version=VERSION,
        description="Sirve el modelo promovido en MLflow para estimar PM2.5.",
        lifespan=vida_util,
    )
    app.state.cargador = cargador or CargadorModelo()

    @app.get("/health", response_model=SaludSalida, tags=["operacion"])
    def health() -> SaludSalida:
        """Liveness: 200 incluso en degradado para facilitar diagnostico operativo."""
        servicio: CargadorModelo = app.state.cargador
        return SaludSalida(
            estado="ok" if servicio.listo else "degradado",
            modelo_cargado=servicio.listo,
            model_uri=servicio.uri,
            model_version=servicio.version,
        )

    @app.get("/modelo", response_model=ModeloSalida, tags=["operacion"])
    def modelo() -> ModeloSalida:
        """Expone contrato y procedencia sin revelar errores internos de MLflow."""
        servicio: CargadorModelo = app.state.cargador
        return ModeloSalida(
            model_name=servicio.nombre,
            model_uri=servicio.uri,
            model_version=servicio.version,
            features_entrada=fc.FEATURES,
        )

    @app.get("/metrics", tags=["operacion"], include_in_schema=False)
    def metrics() -> Response:
        """Entrega metricas en el formato scrapeable de Prometheus."""
        contenido, content_type = contenido_prometheus()
        return Response(content=contenido, media_type=content_type)

    @app.post("/predict", response_model=PrediccionSalida, tags=["prediccion"])
    def predict(entrada: PrediccionEntrada) -> PrediccionSalida:
        """Estima PM2.5 con la misma construccion de features del entrenamiento."""
        servicio: CargadorModelo = app.state.cargador
        if not servicio.listo:
            registrar_prediccion(resultado="modelo_no_disponible")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=DETALLE_MODELO_NO_DISPONIBLE,
            )

        inicio = perf_counter()
        try:
            estimacion = servicio.predecir(entrada.a_dataframe())
        except Exception:
            LOGGER.exception("Fallo al ejecutar una prediccion")
            registrar_prediccion(resultado="error")
            raise _error_interno() from None

        latencia_ms = (perf_counter() - inicio) * 1_000
        registrar_prediccion(resultado="ok", latencia_ms=latencia_ms)
        return PrediccionSalida(
            prediccion_pm25=estimacion,
            model_name=servicio.nombre,
            model_version=servicio.version or "desconocida",
            latencia_ms=latencia_ms,
        )

    @app.post("/predict/batch", response_model=LotePrediccionSalida, tags=["prediccion"])
    def predict_batch(entrada: LotePrediccion) -> LotePrediccionSalida:
        """Estima PM2.5 para un lote de lecturas en una sola llamada."""
        servicio: CargadorModelo = app.state.cargador
        if not servicio.listo:
            registrar_prediccion(resultado="modelo_no_disponible")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=DETALLE_MODELO_NO_DISPONIBLE,
            )

        inicio = perf_counter()
        try:
            dataframe = pd.concat([lectura.a_dataframe() for lectura in entrada.lecturas])
            estimaciones = servicio.predecir_lote(dataframe)
        except Exception:
            LOGGER.exception("Fallo al ejecutar un lote de predicciones")
            registrar_prediccion(resultado="error")
            raise _error_interno() from None

        latencia_ms = (perf_counter() - inicio) * 1_000
        version = servicio.version or "desconocida"
        predicciones = [
            PrediccionSalida(
                prediccion_pm25=valor,
                model_name=servicio.nombre,
                model_version=version,
                latencia_ms=latencia_ms,
            )
            for valor in estimaciones
        ]
        registrar_prediccion(resultado="ok", latencia_ms=latencia_ms)
        return LotePrediccionSalida(
            predicciones=predicciones,
            model_name=servicio.nombre,
            model_version=version,
        )

    @app.get("/", tags=["operacion"])
    def raiz() -> dict[str, str]:
        """Punto de entrada humano y breve de la API."""
        return {"proyecto": PROYECTO, "documentacion": "/docs"}

    return app


app = crear_app()
