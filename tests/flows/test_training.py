"""Pruebas del trigger de Continuous Training sin levantar Prefect ni MLflow."""

from __future__ import annotations

from pathlib import Path

from BeijingAir.flows.training import (
    EstadoEntrenamiento,
    cargar_estado,
    debe_reentrenar,
    guardar_estado,
)


def test_primer_dataset_si_requiere_entrenamiento() -> None:
    """Sin una corrida previa, la primera version debe producir un candidato."""
    assert debe_reentrenar("hash-nuevo", None)


def test_mismo_dataset_no_reentrena_por_cron() -> None:
    """Un schedule no basta: sin dato nuevo se evita entrenar otra vez."""
    estado = EstadoEntrenamiento("hash-igual", "2026-09-06T00:00:00+00:00")
    assert not debe_reentrenar("hash-igual", estado)


def test_forzar_permite_evaluar_cambio_de_codigo() -> None:
    """El equipo puede repetir una corrida de forma explicita y auditable."""
    estado = EstadoEntrenamiento("hash-igual", "2026-09-06T00:00:00+00:00")
    assert debe_reentrenar("hash-igual", estado, forzar=True)


def test_estado_se_persiste_fuera_de_git(tmp_path: Path) -> None:
    """El estado se puede regenerar y no requiere versionar artefactos locales."""
    ruta = tmp_path / "ultimo_entrenamiento.json"
    esperado = EstadoEntrenamiento("hash-123", "2026-09-06T00:00:00+00:00")

    guardar_estado(esperado, ruta)

    assert cargar_estado(ruta) == esperado
