"""Pruebas de la busqueda de hiperparametros y del pipeline con overrides."""

from __future__ import annotations

from BeijingAir.models.pipeline import crear_pipeline
from BeijingAir.models.train import ESPACIO_RANDOM_FOREST


def test_espacio_cubre_los_hiperparametros_clave() -> None:
    """El espacio de busqueda debe incluir n_estimators y profundidad."""
    assert "n_estimators" in ESPACIO_RANDOM_FOREST
    assert "max_depth" in ESPACIO_RANDOM_FOREST
    assert "min_samples_leaf" in ESPACIO_RANDOM_FOREST


def test_crear_pipeline_acepta_overrides_de_hiperparametros() -> None:
    """``crear_pipeline`` deja pasar kwargs extra al RandomForest sin romperse."""
    pipeline = crear_pipeline("bosque", n_estimators=10, max_depth=8, min_samples_leaf=2)
    estimador = pipeline.named_steps["modelo"]
    assert estimador.n_estimators == 10
    assert estimador.max_depth == 8
    assert estimador.min_samples_leaf == 2
