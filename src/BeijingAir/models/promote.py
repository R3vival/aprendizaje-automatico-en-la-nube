"""Gate auditable para promover un candidato de MLflow a ``champion``.

La decision usa exclusivamente la particion temporal ``test``. Los valores se
leen de la corrida que produjo la version del Model Registry: si falta una
metrica, el gate falla en lugar de asumir que el candidato es seguro.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from BeijingAir.config import (
    MAX_EMPEORAMIENTO_MAE,
    MAX_MAE_TEST,
    MIN_R2_TEST,
    MLFLOW_TRACKING_URI,
    MODELO_ALIAS,
    MODELO_ALIAS_CANDIDATO,
    MODELO_REGISTRADO,
    TAG_VALIDACION,
)


class VersionRegistry(Protocol):
    """Atributos de una version que usa el gate, independiente del SDK concreto."""

    @property
    def version(self) -> str | int:
        """Identificador de version que devuelve MLflow."""
        ...

    @property
    def run_id(self) -> str | None:
        """Corrida MLflow que creo esta version."""
        ...


class DatosCorrida(Protocol):
    """Datos minimos de una corrida MLflow."""

    @property
    def metrics(self) -> Mapping[str, float]:
        """Metricas registradas en la corrida."""
        ...


class CorridaMLflow(Protocol):
    """Resultado de recuperar una corrida en MLflow."""

    @property
    def data(self) -> DatosCorrida:
        """Datos asociados a la corrida."""
        ...


class ClienteRegistry(Protocol):
    """Operaciones de Registry que necesita la promocion."""

    def get_model_version_by_alias(self, name: str, alias: str) -> VersionRegistry: ...

    def get_model_version(self, name: str, version: str) -> VersionRegistry: ...

    def get_run(self, run_id: str) -> CorridaMLflow: ...

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> object: ...

    def set_model_version_tag(self, name: str, version: str, key: str, value: str) -> object: ...


@dataclass(frozen=True)
class MetricasModelo:
    """Metricas del holdout temporal que protegen una promocion."""

    mae_test: float
    r2_test: float

    @classmethod
    def desde_mapping(cls, metricas: Mapping[str, float]) -> MetricasModelo:
        """Extrae las metricas obligatorias o falla antes de mover un alias."""
        requeridas = ("mae_test", "r2_test")
        faltantes = [metrica for metrica in requeridas if metrica not in metricas]
        if faltantes:
            raise ValueError(f"La corrida no tiene metricas de test requeridas: {faltantes}")

        valores = cls(mae_test=float(metricas["mae_test"]), r2_test=float(metricas["r2_test"]))
        if not all(math.isfinite(valor) for valor in (valores.mae_test, valores.r2_test)):
            raise ValueError("Las metricas de test deben ser numeros finitos.")
        return valores


@dataclass(frozen=True)
class CriteriosPromocion:
    """Reglas explicitas para aceptar un modelo en produccion."""

    max_mae_test: float = MAX_MAE_TEST
    min_r2_test: float = MIN_R2_TEST
    max_empeoramiento_mae: float = MAX_EMPEORAMIENTO_MAE

    def __post_init__(self) -> None:
        if self.max_mae_test <= 0:
            raise ValueError("max_mae_test debe ser mayor que cero.")
        if self.max_empeoramiento_mae < 0:
            raise ValueError("max_empeoramiento_mae no puede ser negativo.")


@dataclass(frozen=True)
class DecisionPromocion:
    """Resultado explicable del gate, apto para logs de CI."""

    aprobada: bool
    razones: tuple[str, ...]


@dataclass(frozen=True)
class VersionModelo:
    """Version de Registry junto a las metricas de la corrida que la produjo."""

    version: str
    metricas: MetricasModelo


@dataclass(frozen=True)
class ResultadoPromocion:
    """Resultado completo de revisar o ejecutar una promocion."""

    candidato: VersionModelo
    champion_anterior: VersionModelo | None
    decision: DecisionPromocion
    promovido: bool

    def como_dict(self) -> dict[str, object]:
        """Serializa el resultado para logs estructurados de GitHub Actions."""
        return asdict(self)


def evaluar_candidato(
    candidato: MetricasModelo,
    champion: MetricasModelo | None,
    criterios: CriteriosPromocion,
) -> DecisionPromocion:
    """Decide con limites absolutos y sin degradar al champion existente."""
    razones: list[str] = []
    if candidato.mae_test > criterios.max_mae_test:
        razones.append(
            f"mae_test={candidato.mae_test:.3f} supera max_mae_test={criterios.max_mae_test:.3f}"
        )
    if candidato.r2_test < criterios.min_r2_test:
        razones.append(
            f"r2_test={candidato.r2_test:.3f} es menor que min_r2_test={criterios.min_r2_test:.3f}"
        )
    if champion is not None:
        limite_champion = champion.mae_test * (1 + criterios.max_empeoramiento_mae)
        if candidato.mae_test > limite_champion:
            razones.append(
                "mae_test del candidato empeora frente a champion: "
                f"{candidato.mae_test:.3f} > {limite_champion:.3f}"
            )
    return DecisionPromocion(aprobada=not razones, razones=tuple(razones))


class PromotorModelo:
    """Conecta el gate puro con MLflow Registry y mueve solo el alias champion."""

    def __init__(
        self,
        cliente: ClienteRegistry,
        *,
        nombre_modelo: str = MODELO_REGISTRADO,
        alias_candidato: str = MODELO_ALIAS_CANDIDATO,
        alias_champion: str = MODELO_ALIAS,
    ) -> None:
        self.cliente = cliente
        self.nombre_modelo = nombre_modelo
        self.alias_candidato = alias_candidato
        self.alias_champion = alias_champion

    def _version_con_metricas(self, version_registry: VersionRegistry) -> VersionModelo:
        run_id = version_registry.run_id
        if not run_id:
            raise ValueError("La version del modelo no esta asociada a una corrida MLflow.")
        corrida = self.cliente.get_run(run_id)
        return VersionModelo(
            version=str(version_registry.version),
            metricas=MetricasModelo.desde_mapping(corrida.data.metrics),
        )

    def _candidato(self, version: str | None) -> VersionModelo:
        if version is None:
            registro = self.cliente.get_model_version_by_alias(
                self.nombre_modelo, self.alias_candidato
            )
        else:
            registro = self.cliente.get_model_version(self.nombre_modelo, version)
        return self._version_con_metricas(registro)

    def _champion(self) -> VersionModelo | None:
        try:
            registro = self.cliente.get_model_version_by_alias(
                self.nombre_modelo, self.alias_champion
            )
        except MlflowException:
            return None
        return self._version_con_metricas(registro)

    def promover(
        self,
        *,
        criterios: CriteriosPromocion,
        version_candidata: str | None = None,
        dry_run: bool = False,
    ) -> ResultadoPromocion:
        """Evalua y, solo si aprueba, mueve ``champion`` a la version candidata."""
        candidato = self._candidato(version_candidata)
        champion = self._champion()
        decision = evaluar_candidato(
            candidato.metricas,
            None if champion is None else champion.metricas,
            criterios,
        )
        promovido = decision.aprobada and not dry_run

        # Los tags se escriben SIEMPRE, tambien cuando se rechaza: la evidencia
        # de por que un modelo NO llego a produccion vale tanto como la de por
        # que si. Sin ellos, "por que no se promovio aquel candidato" no tiene
        # respuesta tres semanas despues.
        if not dry_run:
            estado = "passed" if decision.aprobada else "failed"
            self.cliente.set_model_version_tag(
                self.nombre_modelo, candidato.version, TAG_VALIDACION, estado
            )
            self.cliente.set_model_version_tag(
                self.nombre_modelo,
                candidato.version,
                "gate_motivo",
                "; ".join(decision.razones),
            )

        if promovido:
            self.cliente.set_registered_model_alias(
                self.nombre_modelo, self.alias_champion, candidato.version
            )
        return ResultadoPromocion(candidato, champion, decision, promovido)


def _argumentos(argumentos: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua y promociona un candidato MLflow.")
    parser.add_argument(
        "--candidate-version",
        help="Version a evaluar; si se omite, se usa el alias candidate.",
    )
    parser.add_argument("--max-mae-test", type=float, default=MAX_MAE_TEST)
    parser.add_argument("--min-r2-test", type=float, default=MIN_R2_TEST)
    parser.add_argument(
        "--max-empeoramiento-mae",
        type=float,
        default=MAX_EMPEORAMIENTO_MAE,
        help="Fraccion maxima permitida de empeoramiento frente a champion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evalua el gate y muestra el resultado, sin mover el alias champion.",
    )
    return parser.parse_args(argumentos)


def main(argumentos: Sequence[str] | None = None) -> None:
    """Punto de entrada para la ejecucion manual protegida de CI/CD."""
    opciones = _argumentos(argumentos)
    criterios = CriteriosPromocion(
        max_mae_test=opciones.max_mae_test,
        min_r2_test=opciones.min_r2_test,
        max_empeoramiento_mae=opciones.max_empeoramiento_mae,
    )
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    resultado = PromotorModelo(MlflowClient()).promover(
        criterios=criterios,
        version_candidata=opciones.candidate_version,
        dry_run=opciones.dry_run,
    )
    print(json.dumps(resultado.como_dict(), ensure_ascii=False, indent=2))
    if not resultado.decision.aprobada:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
