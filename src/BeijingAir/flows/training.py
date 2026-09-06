"""Flow de entrenamiento reproducible para BeijingAir.

Prefect responde cuando y como corrio el pipeline; MLflow conserva los
parametros, las metricas y el modelo que produjo cada corrida. Este modulo no
promueve modelos: solo registra el candidato que despues evaluara el gate de
CI/CD.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from prefect import flow, get_run_logger, task
from prefect.cache_policies import INPUTS
from prefect.runtime import flow_run
from prefect.schedules import Cron

from BeijingAir.config import (
    ESTADO_ENTRENAMIENTO,
    FILAS_POR_PARTICION,
    PARTICIONES_TRAIN,
    PARTICION_VALID,
    PREFECT_SCHEDULE_CRON,
    PREFECT_TIMEZONE,
)
from BeijingAir.data.loaders import asegurar_crudo, preparar_particion, preparar_particiones
from BeijingAir.models.train import entrenar_y_registrar, hash_dataset

MetricasModelo = dict[str, float | dict[str, float]]


@dataclass(frozen=True)
class EstadoEntrenamiento:
    """Huella de la ultima corrida exitosa, persistida fuera de Git."""

    dataset_sha256: str
    ejecutado_en_utc: str


@dataclass(frozen=True)
class ResultadoFlujo:
    """Resumen serializable de una ejecucion del flujo."""

    ejecutado: bool
    dataset_sha256: str
    filas_validadas: dict[str, int]
    resultados: dict[str, MetricasModelo]


def cargar_estado(ruta: Path = ESTADO_ENTRENAMIENTO) -> EstadoEntrenamiento | None:
    """Lee el estado local; no tenerlo implica que aun no se ha entrenado."""
    if not ruta.exists():
        return None

    contenido: object = json.loads(ruta.read_text(encoding="utf-8"))
    if not isinstance(contenido, dict):
        return None
    dataset_sha256 = contenido.get("dataset_sha256")
    ejecutado_en_utc = contenido.get("ejecutado_en_utc")
    if not isinstance(dataset_sha256, str) or not isinstance(ejecutado_en_utc, str):
        return None
    return EstadoEntrenamiento(dataset_sha256, ejecutado_en_utc)


def guardar_estado(estado: EstadoEntrenamiento, ruta: Path = ESTADO_ENTRENAMIENTO) -> None:
    """Persiste la huella solo despues de que MLflow registro el candidato."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(
        json.dumps(asdict(estado), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def debe_reentrenar(
    dataset_sha256: str,
    estado_anterior: EstadoEntrenamiento | None,
    *,
    forzar: bool = False,
) -> bool:
    """Decide por llegada de datos, no por una frecuencia arbitraria."""
    return forzar or estado_anterior is None or estado_anterior.dataset_sha256 != dataset_sha256


@task(name="asegurar-origen", retries=2, retry_delay_seconds=[5, 15])
def asegurar_origen() -> str:
    """Descarga el origen si falta; los reintentos son para fallos de red."""
    asegurar_crudo()
    huella = hash_dataset()
    if huella == "no-disponible":
        raise RuntimeError("No se encontro metadata.json con el hash del dataset.")
    return huella


@task(
    name="validar-particiones",
    cache_policy=INPUTS,
    cache_expiration=timedelta(days=1),
    persist_result=True,
)
def validar_particiones(dataset_sha256: str, filas: int | None) -> dict[str, int]:
    """Valida train y valid; el cache cambia si cambia el hash o el muestreo."""
    _ = dataset_sha256
    train = preparar_particiones(PARTICIONES_TRAIN, filas=filas)
    valid = preparar_particion(PARTICION_VALID, filas=filas)
    return {"train": len(train), "valid": len(valid)}


@task(name="entrenar-y-registrar")
def ejecutar_entrenamiento(
    filas: int | None,
    n_estimators: int,
    prefect_flow_run_id: str,
) -> dict[str, MetricasModelo]:
    """Entrena y deja el run de Prefect como tag en las corridas de MLflow."""
    resultados = entrenar_y_registrar(
        filas=filas,
        n_estimators=n_estimators,
        registrar=True,
        tags_adicionales={"prefect_flow_run_id": prefect_flow_run_id},
    )
    return {nombre: evaluacion.como_dict() for nombre, evaluacion in resultados.items()}


@task(name="guardar-estado")
def persistir_estado(dataset_sha256: str) -> None:
    """Marca como procesada esta version del dataset tras una corrida exitosa."""
    guardar_estado(
        EstadoEntrenamiento(
            dataset_sha256=dataset_sha256,
            ejecutado_en_utc=datetime.now(timezone.utc).isoformat(),
        )
    )


@flow(name="entrenamiento-beijing-pm25", log_prints=True)
def flujo_entrenamiento(
    *,
    filas: int | None = FILAS_POR_PARTICION,
    n_estimators: int = 300,
    forzar: bool = False,
) -> ResultadoFlujo:
    """Orquesta validar -> entrenar -> registrar candidato -> guardar estado.

    El schedule mensual despierta el flow, pero este solo reentrena si la huella
    del dataset es nueva. ``--forzar`` existe para probar cambios de codigo o
    hiperparametros de manera intencional.
    """
    logger = get_run_logger()
    dataset_sha256 = asegurar_origen()
    estado_anterior = cargar_estado()
    if not debe_reentrenar(dataset_sha256, estado_anterior, forzar=forzar):
        logger.info("No hay datos nuevos: se omite el reentrenamiento.")
        return ResultadoFlujo(False, dataset_sha256, {}, {})

    filas_validadas = validar_particiones(dataset_sha256, filas)
    resultados = ejecutar_entrenamiento(filas, n_estimators, str(flow_run.id))
    persistir_estado(dataset_sha256)
    logger.info("Candidato registrado; la promocion la decide CI/CD, no este flow.")
    return ResultadoFlujo(True, dataset_sha256, filas_validadas, resultados)


def servir_flujo() -> None:
    """Sirve el schedule mensual en Prefect; esta llamada queda en primer plano."""
    flujo_entrenamiento.serve(
        name="entrenamiento-beijing-mensual",
        schedules=[Cron(PREFECT_SCHEDULE_CRON, timezone=PREFECT_TIMEZONE)],
        parameters={"filas": FILAS_POR_PARTICION, "n_estimators": 300, "forzar": False},
    )


def _argumentos(argumentos: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orquesta el entrenamiento de PM2.5 con Prefect.")
    parser.add_argument("--filas", type=int, default=FILAS_POR_PARTICION)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--forzar", action="store_true")
    parser.add_argument("--serve", action="store_true", help="Deja servido el schedule mensual.")
    return parser.parse_args(argumentos)


def main(argumentos: Sequence[str] | None = None) -> None:
    """Punto de entrada de ``python -m BeijingAir.flows.training``."""
    opciones = _argumentos(argumentos)
    if opciones.serve:
        servir_flujo()
        return

    filas = None if opciones.filas == 0 else opciones.filas
    resultado = flujo_entrenamiento(
        filas=filas,
        n_estimators=opciones.n_estimators,
        forzar=opciones.forzar,
    )
    estado = "ejecutado" if resultado.ejecutado else "omitido: sin datos nuevos"
    print(f"Flow {estado}. Dataset: {resultado.dataset_sha256}")


if __name__ == "__main__":
    main()
