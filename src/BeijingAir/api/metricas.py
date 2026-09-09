"""Metricas de servicio expuestas en formato Prometheus."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

SOLICITUDES_PREDICCION = Counter(
    "beijing_air_predicciones_total",
    "Solicitudes de prediccion atendidas por resultado.",
    ("resultado",),
)
LATENCIA_PREDICCION_MS = Histogram(
    "beijing_air_prediccion_latencia_ms",
    "Latencia de prediccion en milisegundos.",
)


def registrar_prediccion(*, resultado: str, latencia_ms: float | None = None) -> None:
    """Registra el resultado y, cuando aplica, la latencia de una solicitud."""
    SOLICITUDES_PREDICCION.labels(resultado=resultado).inc()
    if latencia_ms is not None:
        LATENCIA_PREDICCION_MS.observe(latencia_ms)


def contenido_prometheus() -> tuple[bytes, str]:
    """Devuelve el cuerpo y content type esperados por Prometheus."""
    return generate_latest(), CONTENT_TYPE_LATEST
