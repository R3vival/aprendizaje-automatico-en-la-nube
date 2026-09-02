"""Pipeline de entrenamiento orquestado con Prefect.

Este modulo NO implementa logica de ML: la orquesta. El entrenamiento vive en
`models/train.py` y la preparacion de datos en `data/loaders.py`. Duplicar esa
logica aqui produciria dos versiones que se desincronizan.

Tres garantias que aporta la orquestacion:

1. **Resiliencia**: `retries` con backoff en la unica task que habla con la red.
2. **Trazabilidad**: cada corrida queda en la UI, con quien la lanzo y cuanto
   tardo cada paso.
3. **Separacion de responsabilidades**: el flow registra un CANDIDATO y NO lo
   promueve. La promocion es del gate.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import timedelta
from typing import Any

from prefect import flow, get_run_logger, task
from prefect.artifacts import create_table_artifact
from prefect.tasks import task_input_hash

from BeijingAir.config import (
    ALIAS_CANDIDATO,
    MODELO_REGISTRADO,
    PARTICIONES_TRAIN,
    TAG_VALIDACION,
)


def _a_dict(obj: Any) -> dict[str, Any]:
    """Convierte un ResultadoEvaluacion a diccionario, sea dataclass o no."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return dict(vars(obj))


@task(
    name="extraer",
    description="Descarga el ZIP de UCI y registra su hash en metadata.json.",
    retries=3,
    retry_delay_seconds=[10, 30, 60],
)
def extraer() -> str:
    """Unica task que habla con la red, y por eso la unica con reintentos.

    El backoff es [10, 30, 60] y no [2, 2, 2] a proposito: reintentar cada dos
    segundos contra un servidor caido solo le agrega carga. La lista da control
    explicito por intento.
    """
    from BeijingAir.data.descarga import descargar
    from BeijingAir.data.descarga import extraer as descomprimir

    ruta = descargar()
    descomprimir(ruta)
    get_run_logger().info("dataset listo en %s", ruta)
    return str(ruta)


@task(
    name="validar",
    description="Corre el contrato sobre los datos crudos.",
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=1),
)
def validar(_ruta: str) -> dict[str, int]:
    """Falla temprano si el proveedor cambio el esquema.

    Recibe `_ruta` sin usarla: es lo que le dice a Prefect que esta task
    depende de `extraer`. Asi el grafo sale de los datos, no de un orden
    escrito a mano.
    """
    from BeijingAir.data.descarga import cargar_crudo

    df = cargar_crudo()
    filas, columnas = df.shape
    get_run_logger().info("crudo validado: %d filas x %d columnas", filas, columnas)
    return {"filas": int(filas), "columnas": int(columnas)}


@task(name="entrenar", description="Entrena los candidatos y los registra en MLflow.")
def entrenar(_stats: dict[str, int]) -> dict[str, dict[str, Any]]:
    """Llama al entrenamiento que ya existe. No reimplementa nada."""
    from BeijingAir.models.train import entrenar_y_registrar

    resultados = entrenar_y_registrar(registrar=True)
    get_run_logger().info("modelos entrenados: %s", ", ".join(resultados))
    return {nombre: _a_dict(res) for nombre, res in resultados.items()}


@task(name="evaluar", description="Elige el mejor candidato segun la metrica.")
def evaluar(resultados: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Selecciona el mejor por RMSE. Si tu metrica es otra, cambia la clave."""
    logger = get_run_logger()

    def _rmse(item: tuple[str, dict[str, Any]]) -> float:
        metricas = item[1]
        for clave in ("rmse", "RMSE", "rmse_valid"):
            if clave in metricas:
                return float(metricas[clave])
        raise KeyError(f"No encuentro el RMSE en {list(metricas)}")

    nombre, metricas = min(resultados.items(), key=_rmse)
    logger.info("mejor candidato: %s", nombre)
    return nombre, metricas


@task(
    name="registrar_candidato",
    description="Marca la ultima version como @candidate. NO la promueve.",
)
def registrar_candidato(mejor: tuple[str, dict[str, Any]]) -> str:
    """Pone el alias @candidate y el tag de validacion pendiente.

    NO toca @champion. Un modelo no llega a produccion por el hecho de que el
    entrenamiento no lanzo excepciones: esa decision es del gate de promocion.
    """
    from mlflow import MlflowClient

    logger = get_run_logger()
    cliente = MlflowClient()
    versiones = cliente.search_model_versions(f"name='{MODELO_REGISTRADO}'")
    if not versiones:
        raise RuntimeError(f"No hay versiones registradas de {MODELO_REGISTRADO}")

    ultima = max(versiones, key=lambda v: int(v.version))
    cliente.set_registered_model_alias(MODELO_REGISTRADO, ALIAS_CANDIDATO, ultima.version)
    cliente.set_model_version_tag(MODELO_REGISTRADO, ultima.version, TAG_VALIDACION, "pending")
    logger.info("version %s marcada como @%s (champion sin tocar)", ultima.version, ALIAS_CANDIDATO)
    return str(ultima.version)


@task(name="publicar_reporte", description="Publica la tabla de metricas.")
def publicar_reporte(resultados: dict[str, dict[str, Any]]) -> None:
    """Deja la tabla de metricas junto a la corrida, en la UI de Prefect."""
    filas = [{"modelo": nombre, **metricas} for nombre, metricas in resultados.items()]
    create_table_artifact(
        key="metricas-entrenamiento",
        table=filas,
        description="Metricas de los candidatos de esta corrida.",
    )


@flow(name="entrenamiento-beijing", log_prints=True)
def entrenamiento_flow() -> None:
    """Pipeline completo: extraer, validar, entrenar, evaluar y registrar."""
    logger = get_run_logger()
    logger.info("particiones de entrenamiento: %s", ", ".join(str(p) for p in PARTICIONES_TRAIN))

    ruta = extraer()
    stats = validar(ruta)
    resultados = entrenar(stats)
    mejor = evaluar(resultados)
    version = registrar_candidato(mejor)
    publicar_reporte(resultados)

    logger.info("listo. Version %s registrada como @%s, sin promover.", version, ALIAS_CANDIDATO)


if __name__ == "__main__":
    entrenamiento_flow()
