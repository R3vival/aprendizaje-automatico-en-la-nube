"""Gate auditable para promover un candidato de MLflow a ``champion`` (Sesion 6).

La decision usa exclusivamente la particion temporal ``test`` (el holdout fijo).
El champion **se reevalua sobre el holdout actual** en cada corrida: si el
holdout o el codigo de la metrica cambiaron desde que se entreno, los numeros
guardados en su run no son comparables. La politica (los criterios) vive en
``models/evaluate.py`` como funcion pura; este modulo es la capa de
presentacion: carga los modelos, los evalúa, escribe los tags, mueve el alias y
reporta exit codes.

Tres exit codes, y por que son tres:
- ``0`` promovido (o el candidato ya era el champion).
- ``1`` rechazado por los criterios. ``@champion`` no se toca.
- ``2`` no pudo medir (MLflow no responde, falta el holdout, no hay candidato).

La distincion entre 1 y 2 importa: "el modelo no es lo bastante bueno" es un
resultado exitoso del gate; "no pude medir" es una falla del gate. Confundirlos
hace que un MLflow caido se lea como un modelo malo y alguien acabe reentrenando
para arreglar un problema de red.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from BeijingAir.config import (
    MEJORA_MINIMA_RELATIVA,
    MLFLOW_TRACKING_URI,
    MODELO_ALIAS,
    MODELO_ALIAS_CANDIDATO,
    MODELO_REGISTRADO,
    PARTICION_TEST,
    TAG_VALIDACION,
    UMBRAL_DEGRADACION_SUBGRUPO,
)

EXIT_PROMOVIDO: Final[int] = 0
EXIT_RECHAZADO: Final[int] = 1
EXIT_INFRA: Final[int] = 2


@dataclass(frozen=True)
class CriteriosPromocion:
    """Umbrales del gate: margen de mejora y tolerancia por subgrupo."""

    mejora_minima: float = MEJORA_MINIMA_RELATIVA
    umbral_subgrupo: float = UMBRAL_DEGRADACION_SUBGRUPO

    def __post_init__(self) -> None:
        if self.mejora_minima < 0:
            raise ValueError("mejora_minima no puede ser negativa.")
        if self.umbral_subgrupo < 0:
            raise ValueError("umbral_subgrupo no puede ser negativo.")


@dataclass(frozen=True)
class ResultadoPromocion:
    """Resultado completo de revisar o ejecutar una promocion."""

    version_candidato: str
    version_champion: str | None
    decision: Any  # models.evaluate.DecisionGate
    promovido: bool

    def como_dict(self) -> dict[str, object]:
        """Serializa el resultado para logs estructurados de GitHub Actions."""
        return asdict(self)


def _fallar_rapido() -> None:
    """Baja el timeout y los reintentos HTTP de MLflow para las consultas de
    metadatos. Con los defaults (7 reintentos, 120 s de timeout) un registry
    caido cuelga el gate varios minutos antes de devolver el exit 2. Un fallback
    que tarda minutos en activarse no es un fallback."""
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")


def _cargar_holdout() -> Any:
    """Prepara el holdout fijo sobre el que se mide champion y candidato."""
    from BeijingAir.data.loaders import preparar_particion

    return preparar_particion(PARTICION_TEST)


def _modelo_y_metadatos(cliente: MlflowClient, nombre: str, referencia: str) -> tuple[Any, str]:
    """Carga un modelo por ``models:/nombre@alias`` o ``models:/nombre/version``.

    Resuelve la version que referencia (el alias es mutable, la version no) y
    devuelve ``(modelo_pyfunc, version)``.
    """
    if "@" in referencia:
        _, alias = referencia.rsplit("@", 1)
        version = cliente.get_model_version_by_alias(nombre, alias)
        uri = f"models:/{nombre}@{alias}"
    else:
        numero = referencia
        version = cliente.get_model_version(nombre, numero)
        uri = f"models:/{nombre}/{numero}"
    modelo = mlflow.pyfunc.load_model(uri)
    return modelo, str(version.version)


def evaluar_candidato(
    *,
    version_candidata: str | None = None,
    criterios: CriteriosPromocion = CriteriosPromocion(),
    dry_run: bool = False,
    nombre_modelo: str = MODELO_REGISTRADO,
) -> ResultadoPromocion:
    """Evalua el gate y, solo si aprueba y no es dry-run, mueve ``champion``.

    Carga candidato y champion por alias, los reevalua sobre el holdout actual y
    aplica la politica de ``models.evaluate.decidir_promocion``.

    Args:
        version_candidata: version a evaluar; ``None`` usa el alias candidate.
        criterios: margen de mejora y tolerancia por subgrupo.
        dry_run: evalua e informa sin escribir tag ni mover alias.
        nombre_modelo: modelo registrado en el registry.

    Returns:
        ``ResultadoPromocion`` con la decision y si se promovio.

    Raises:
        MlflowException: si el registry no responde (infraestructura).
    """
    from BeijingAir.models import evaluate

    _fallar_rapido()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    cliente = MlflowClient()

    # El candidato se resuelve ANTES de preparar el holdout: si MLflow esta
    # caido, la consulta de metadatos falla rapido (timeout 3 s) sin cargar datos.
    referencia_candidato = version_candidata or f"{nombre_modelo}@{MODELO_ALIAS_CANDIDATO}"
    modelo_candidato, version_candidato = _modelo_y_metadatos(
        cliente, nombre_modelo, referencia_candidato
    )

    version_champion: str | None = None
    met_champion: Mapping[str, float] | None = None
    sub_champion: Mapping[str, float] | None = None
    modelo_champion = None
    try:
        modelo_champion, version_champion = _modelo_y_metadatos(
            cliente, nombre_modelo, f"{nombre_modelo}@{MODELO_ALIAS}"
        )
    except MlflowException:
        # Sin champion (primer modelo) o champion no existe: ambas son "no hay
        # con que comparar", que es un resultado valido del gate. Un registry
        # caido no llega aqui: habria fallado antes, en el candidato.
        version_champion = None

    holdout = _cargar_holdout()
    met_candidato, sub_candidato = evaluate.evaluar_modelo(modelo_candidato, holdout)
    if modelo_champion is not None:
        met_champion, sub_champion = evaluate.evaluar_modelo(modelo_champion, holdout)

    decision = evaluate.decidir_promocion(
        holdout,
        met_candidato,
        sub_candidato,
        met_champion,
        sub_champion,
        mejora_minima=criterios.mejora_minima,
        umbral_subgrupo=criterios.umbral_subgrupo,
    )

    promovido = decision.promover and not dry_run

    # Los tags se escriben SIEMPRE (tambien al rechazar): la evidencia de por que
    # un modelo NO llego a produccion vale tanto como la de por que si.
    if not dry_run:
        estado = "passed" if decision.promover else "failed"
        cliente.set_model_version_tag(nombre_modelo, version_candidato, TAG_VALIDACION, estado)
        cliente.set_model_version_tag(
            nombre_modelo, version_candidato, "gate_motivo", decision.motivo
        )

    if promovido:
        cliente.set_registered_model_alias(nombre_modelo, MODELO_ALIAS, version_candidato)

    return ResultadoPromocion(
        version_candidato=version_candidato,
        version_champion=version_champion,
        decision=decision,
        promovido=promovido,
    )


def _argumentos(argumentos: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua y promociona un candidato MLflow.")
    parser.add_argument(
        "--candidate-version",
        help="Version a evaluar; si se omite, se usa el alias candidate.",
    )
    parser.add_argument(
        "--mejora-minima",
        type=float,
        default=MEJORA_MINIMA_RELATIVA,
        help="Margen relativo minimo de mejora del MAE sobre el holdout.",
    )
    parser.add_argument(
        "--umbral-subgrupo",
        type=float,
        default=UMBRAL_DEGRADACION_SUBGRUPO,
        help="Degradacion relativa maxima tolerada por subgrupo.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evalua el gate y muestra el resultado, sin escribir tag ni mover alias.",
    )
    return parser.parse_args(argumentos)


def main(argumentos: Sequence[str] | None = None) -> int:
    """Punto de entrada de ``python -m BeijingAir.models.promote``."""
    opciones = _argumentos(argumentos)
    criterios = CriteriosPromocion(
        mejora_minima=opciones.mejora_minima,
        umbral_subgrupo=opciones.umbral_subgrupo,
    )
    try:
        resultado = evaluar_candidato(
            version_candidata=opciones.candidate_version,
            criterios=criterios,
            dry_run=opciones.dry_run,
        )
    except MlflowException as error:
        print(
            f"No se pudo hablar con MLflow en {MLFLOW_TRACKING_URI} "
            f"({type(error).__name__}). No es un problema del modelo: es "
            "infraestructura. Levanta el tracking server y vuelve a correr el gate."
        )
        return EXIT_INFRA
    except Exception as error:
        print(
            f"No se pudo medir el gate ({type(error).__name__}): {error}. "
            "No es una decision del modelo."
        )
        return EXIT_INFRA

    resumen = {
        "version_candidato": resultado.version_candidato,
        "version_champion": resultado.version_champion,
        "promover": resultado.decision.promover,
        "motivo": resultado.decision.motivo,
        "criterios": [
            {"nombre": c.nombre, "estado": c.estado, "detalle": c.detalle}
            for c in resultado.decision.criterios
        ],
    }
    print(json.dumps(resumen, ensure_ascii=False, indent=2))

    if not resultado.decision.promover:
        print(
            f"RECHAZADO — @champion no se toca. Sigue en la version {resultado.version_champion}."
        )
        return EXIT_RECHAZADO
    if opciones.dry_run:
        print("--dry-run: habria promovido, pero no se escribio el alias.")
    else:
        print(f"PROMOVIDO — @champion ahora apunta a la version {resultado.version_candidato}.")
    return EXIT_PROMOVIDO


if __name__ == "__main__":
    sys.exit(main())
