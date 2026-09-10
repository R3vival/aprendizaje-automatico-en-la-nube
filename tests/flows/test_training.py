"""Pruebas del trigger de Continuous Training sin levantar Prefect ni MLflow."""

from __future__ import annotations

from pathlib import Path

import pytest

from BeijingAir.flows.training import (
    EstadoEntrenamiento,
    cargar_estado,
    debe_reentrenar,
    elegir_candidato,
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


def _resultados(mae_bosque: float, mae_baseline: float = 30.0) -> dict[str, dict[str, float]]:
    """Resultados minimos con la forma que devuelve ``entrenar_y_registrar``."""
    return {
        "baseline": {"mae": mae_baseline, "rmse": mae_baseline * 1.7, "r2": 0.0},
        "bosque": {"mae": mae_bosque, "rmse": mae_bosque * 1.7, "r2": 0.5},
    }


def test_la_compuerta_deja_pasar_un_bosque_que_supera_al_baseline() -> None:
    """Si el modelo aporta sobre predecir la media, se marca como candidato."""
    assert elegir_candidato(_resultados(mae_bosque=16.0))[0] == "bosque"


def test_la_compuerta_falla_si_el_bosque_no_supera_al_baseline() -> None:
    """Un modelo que no le gana a la media no debe llegar al Registry.

    Falla en vez de avisar: un candidato registrado se lee como un candidato
    valido, y un WARNING en los logs no lo lee nadie.
    """
    with pytest.raises(ValueError, match="no supera al baseline"):
        elegir_candidato(_resultados(mae_bosque=31.0))


def test_la_compuerta_falla_si_falta_la_metrica_decisoria() -> None:
    """Sin la metrica que decide, el flow no puede afirmar que hay mejora."""
    with pytest.raises(KeyError, match="mae"):
        elegir_candidato({"baseline": {"rmse": 50.0}, "bosque": {"rmse": 28.0}})
