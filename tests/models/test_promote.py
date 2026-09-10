"""Pruebas del gate de promocion sin levantar MLflow."""

from __future__ import annotations

from dataclasses import dataclass

from mlflow.exceptions import MlflowException

from BeijingAir.models.promote import (
    CriteriosPromocion,
    MetricasModelo,
    PromotorModelo,
    evaluar_candidato,
)


def test_gate_rechaza_candidato_que_empeora_champion() -> None:
    """Un R2 aceptable no compensa degradar demasiado el error del servicio."""
    decision = evaluar_candidato(
        MetricasModelo(mae_test=25.0, r2_test=0.7),
        MetricasModelo(mae_test=20.0, r2_test=0.6),
        CriteriosPromocion(max_mae_test=30.0, min_r2_test=0.0, max_empeoramiento_mae=0.05),
    )

    assert not decision.aprobada
    assert "empeora frente a champion" in decision.razones[0]


def test_gate_acepta_primer_champion_si_cumple_limites() -> None:
    """El primer modelo no tiene champion previo, pero si debe superar el minimo."""
    decision = evaluar_candidato(
        MetricasModelo(mae_test=20.0, r2_test=0.4),
        None,
        CriteriosPromocion(max_mae_test=30.0, min_r2_test=0.2, max_empeoramiento_mae=0.05),
    )

    assert decision.aprobada


@dataclass
class _VersionFalsa:
    version: str
    run_id: str


@dataclass
class _DatosFalsos:
    metrics: dict[str, float]


@dataclass
class _CorridaFalsa:
    data: _DatosFalsos


class _ClienteFalso:
    def __init__(self) -> None:
        self.aliases: dict[str, _VersionFalsa] = {"candidate": _VersionFalsa("7", "run-candidate")}
        self.corridas = {
            "run-candidate": _CorridaFalsa(_DatosFalsos({"mae_test": 20.0, "r2_test": 0.5}))
        }
        self.alias_asignado: tuple[str, str, str] | None = None
        self.tags: dict[str, str] = {}

    def get_model_version_by_alias(self, _: str, alias: str) -> _VersionFalsa:
        if alias not in self.aliases:
            raise MlflowException("Alias ausente")
        return self.aliases[alias]

    def get_model_version(self, _: str, version: str) -> _VersionFalsa:
        return _VersionFalsa(version, "run-candidate")

    def get_run(self, run_id: str) -> _CorridaFalsa:
        return self.corridas[run_id]

    def set_registered_model_alias(self, nombre: str, alias: str, version: str) -> None:
        self.alias_asignado = (nombre, alias, version)

    def set_model_version_tag(self, _: str, __: str, key: str, value: str) -> None:
        self.tags[key] = value


def test_promotor_mueve_alias_solo_despues_de_aprobar() -> None:
    """La unica mutacion de Registry es asignar champion al candidato aprobado."""
    cliente = _ClienteFalso()
    resultado = PromotorModelo(cliente).promover(
        criterios=CriteriosPromocion(max_mae_test=30.0, min_r2_test=0.0),
    )

    assert resultado.decision.aprobada
    assert resultado.promovido
    assert cliente.alias_asignado == ("beijing-air-pm25", "champion", "7")


def test_una_aprobacion_deja_el_tag_de_validacion() -> None:
    """El tag acompana al alias: la version aprobada queda marcada como passed."""
    cliente = _ClienteFalso()
    PromotorModelo(cliente).promover(
        criterios=CriteriosPromocion(max_mae_test=30.0, min_r2_test=0.0),
    )

    assert cliente.tags["validation_status"] == "passed"


def test_un_rechazo_tambien_deja_evidencia_en_la_version() -> None:
    """La evidencia de por que NO se promovio vale tanto como la de por que si.

    Sin este comportamiento, "por que no se promovio aquel candidato" no tiene
    respuesta tres semanas despues: el alias no se movio y no queda rastro.
    """
    cliente = _ClienteFalso()
    resultado = PromotorModelo(cliente).promover(
        criterios=CriteriosPromocion(max_mae_test=10.0, min_r2_test=0.0),
    )

    assert not resultado.decision.aprobada
    assert not resultado.promovido
    assert cliente.alias_asignado is None
    assert cliente.tags["validation_status"] == "failed"
    assert cliente.tags["gate_motivo"]


def test_dry_run_no_muta_el_registry() -> None:
    """Un ensayo audita la decision sin escribir alias ni tags."""
    cliente = _ClienteFalso()
    resultado = PromotorModelo(cliente).promover(
        criterios=CriteriosPromocion(max_mae_test=30.0, min_r2_test=0.0),
        dry_run=True,
    )

    assert resultado.decision.aprobada
    assert not resultado.promovido
    assert cliente.alias_asignado is None
    assert cliente.tags == {}
