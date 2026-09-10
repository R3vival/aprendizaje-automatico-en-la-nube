"""Flow de entrenamiento reproducible para BeijingAir.

Prefect responde cuando y como corrio el pipeline; MLflow conserva los
parametros, las metricas y el modelo que produjo cada corrida. Este modulo NO
implementa logica de ML: la orquesta. El entrenamiento vive en
``models/train.py`` y la preparacion de datos en ``data/loaders.py``.

Este flow registra un CANDIDATO y NO lo promueve: la promocion la decide el
gate de CI/CD (``models/promote.py``). Para decidir si reentrenar usa el hash
del dataset (llegada de datos), no una frecuencia arbitraria.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_table_artifact
from prefect.cache_policies import INPUTS
from prefect.runtime import flow_run
from prefect.schedules import Cron

from BeijingAir.config import (
    ESTADO_ENTRENAMIENTO,
    FILAS_POR_PARTICION,
    MODELO_ALIAS_CANDIDATO,
    MODELO_REGISTRADO,
    PARTICION_TEST,
    PARTICION_VALID,
    PARTICIONES_TRAIN,
    PREFECT_SCHEDULE_CRON,
    PREFECT_TIMEZONE,
    TAG_VALIDACION,
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
    name="preparar-y-validar-particiones",
    cache_policy=INPUTS,
    cache_expiration=timedelta(days=1),
    persist_result=True,
)
def preparar_datos(
    dataset_sha256: str, filas: int | None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepara train, valid y test una vez; el cache depende del hash y muestra."""
    _ = dataset_sha256
    train = preparar_particiones(PARTICIONES_TRAIN, filas=filas)
    valid = preparar_particion(PARTICION_VALID, filas=filas)
    test = preparar_particion(PARTICION_TEST, filas=filas)
    return train, valid, test


@task(name="entrenar-y-registrar")
def ejecutar_entrenamiento(
    datos_train: pd.DataFrame,
    datos_valid: pd.DataFrame,
    datos_test: pd.DataFrame,
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
        datos_train=datos_train,
        datos_valid=datos_valid,
        datos_test=datos_test,
    )
    return {nombre: evaluacion.como_dict() for nombre, evaluacion in resultados.items()}


#: Metrica que decide, en validacion. Es la misma que corta el gate sobre el
#: holdout (``mae_test``) y la que declara la model card: un proyecto con dos
#: criterios distintos no puede explicar por que un modelo llego a produccion.
METRICA_DECISORIA = "mae"


def _metrica(metricas: MetricasModelo, clave: str = METRICA_DECISORIA) -> float:
    """Lee una metrica global del resultado, o falla diciendo que falta."""
    valor = metricas.get(clave)
    if not isinstance(valor, float):
        raise KeyError(f"No encuentro '{clave}' en {sorted(metricas)}")
    return valor


def elegir_candidato(resultados: dict[str, MetricasModelo]) -> tuple[str, float]:
    """Compuerta de calidad. Funcion **pura**: no toca Prefect, MLflow ni disco.

    No "elige el mejor" entre baseline y bosque, porque el baseline nunca se
    registra: existe para ser batido. Lo que se decide aqui es si el bosque
    merece llegar al Registry, y si no lo merece se **lanza excepcion** en lugar
    de dejar pasar un candidato que no le gana a predecir la media.

    Falla en vez de avisar a proposito: un WARNING en los logs de Prefect no lo
    lee nadie, y un candidato registrado se lee como un candidato valido.

    Esta separada de la task por la misma razon que ``promote.evaluar_candidato``
    lo esta de ``promote.promover``: asi la politica se prueba en milisegundos y
    sin levantar infraestructura.

    Returns:
        El modelo elegido y su mejora relativa sobre el baseline.

    Raises:
        KeyError: si falta la metrica que decide.
        ValueError: si el bosque no supera al baseline.
    """
    baseline = _metrica(resultados["baseline"])
    bosque = _metrica(resultados["bosque"])

    if bosque >= baseline:
        raise ValueError(
            f"El bosque ({METRICA_DECISORIA}={bosque:.3f}) no supera al baseline "
            f"({METRICA_DECISORIA}={baseline:.3f}). No se registra candidato: "
            "el problema no esta en los hiperparametros."
        )
    return "bosque", (baseline - bosque) / baseline


@task(name="evaluar", description="El candidato debe superar al baseline o el flow falla.")
def evaluar(resultados: dict[str, MetricasModelo]) -> str:
    """Aplica la compuerta y deja la mejora en el log de la corrida."""
    elegido, mejora = elegir_candidato(resultados)
    get_run_logger().info(
        "%s supera al baseline en %.1f%% de %s", elegido, mejora * 100, METRICA_DECISORIA
    )
    return elegido


@task(name="marcar-candidato", description="Marca la version candidata con el tag de validacion.")
def marcar_candidato(modelo_elegido: str) -> str:
    """Deja el tag de validacion ``pending`` en la ultima version registrada.

    El alias ``@candidate`` ya lo asigna ``models/train.py`` al registrar el
    bosque. Esta task agrega el tag que el gate usa para auditar por que no se
    promovio aun; no mueve ``@champion``.

    Recibe ``modelo_elegido`` de ``evaluar``: es lo que le dice a Prefect que
    esta task depende de la compuerta de calidad. Sin ese argumento el grafo no
    tiene la arista y la version se marcaria aunque el candidato no superara al
    baseline.
    """
    from mlflow import MlflowClient

    logger = get_run_logger()
    cliente = MlflowClient()
    versiones = cliente.search_model_versions(f"name='{MODELO_REGISTRADO}'")
    if not versiones:
        raise RuntimeError(f"No hay versiones registradas de {MODELO_REGISTRADO}")

    ultima = max(versiones, key=lambda v: int(v.version))
    cliente.set_model_version_tag(MODELO_REGISTRADO, ultima.version, TAG_VALIDACION, "pending")
    logger.info(
        "version %s (%s) marcada como @%s (champion sin tocar)",
        ultima.version,
        modelo_elegido,
        MODELO_ALIAS_CANDIDATO,
    )
    return str(ultima.version)


@task(name="publicar-reporte", description="Publica la tabla de metricas.")
def publicar_reporte(resultados: dict[str, MetricasModelo]) -> None:
    """Deja la tabla de metricas junto a la corrida, en la UI de Prefect."""
    filas = [{"modelo": nombre, **metricas} for nombre, metricas in resultados.items()]
    create_table_artifact(
        key="metricas-entrenamiento",
        table=filas,
        description="Metricas de los candidatos de esta corrida.",
    )


@task(name="guardar-estado")
def persistir_estado(dataset_sha256: str) -> None:
    """Marca como procesada esta version del dataset tras una corrida exitosa."""
    guardar_estado(
        EstadoEntrenamiento(
            dataset_sha256=dataset_sha256,
            ejecutado_en_utc=datetime.now(UTC).isoformat(),
        )
    )


@flow(name="entrenamiento-beijing-pm25", log_prints=True)
def flujo_entrenamiento(
    *,
    filas: int | None = FILAS_POR_PARTICION,
    n_estimators: int = 300,
    forzar: bool = False,
) -> ResultadoFlujo:
    """Orquesta extraer -> validar -> preparar -> entrenar -> evaluar -> registrar.

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

    datos_train, datos_valid, datos_test = preparar_datos(dataset_sha256, filas)
    filas_validadas = {
        "train": len(datos_train),
        "valid": len(datos_valid),
        "test": len(datos_test),
    }
    resultados = ejecutar_entrenamiento(
        datos_train,
        datos_valid,
        datos_test,
        filas,
        n_estimators,
        str(flow_run.id),
    )
    modelo_elegido = evaluar(resultados)
    publicar_reporte(resultados)
    marcar_candidato(modelo_elegido)
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
