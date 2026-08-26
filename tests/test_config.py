"""Tests del módulo de configuración central."""

from BeijingAir import config


def test_proyecto_tiene_nombre_esperado() -> None:
    assert config.PROYECTO == "beijing-air"


def test_version_sigue_el_formato_semver() -> None:
    partes = config.VERSION.split(".")
    assert len(partes) == 3
    assert all(p.isdigit() for p in partes)


def test_config_expone_las_constantes_publicas() -> None:
    assert set(config.__all__) <= {"PROYECTO", "VERSION"}
