"""Pipeline batch de predicciones de PM2.5 (Sesion 5).

Que problema resuelve
---------------------
No todo modelo necesita una API. Si el consumo es un reporte diario o una carga
a un data warehouse, el batch es mas simple, mas barato y mas facil de operar:
no hay servicio que mantener arriba, no hay latencia p95 que vigilar, y si falla
se re-corre.

Lo que hace este flow: lee una particion de "produccion simulada", valida el
contrato, carga el modelo **por alias** desde el registry, predice de forma
vectorizada y persiste cada prediccion **con la version del modelo que la
produjo** y su timestamp.

Por que la trazabilidad es el punto central
-------------------------------------------
Si en la tabla de predicciones no queda registrado *que version* produjo cada
fila, no se puede responder ninguna de las preguntas que importan cuando algo
sale mal: que predicciones hay que revisar tras un rollback, si la degradacion
empezo con la version 7 o con el cambio de datos, o que se le respondio a un
cliente en una fecha dada. La trazabilidad datos -> modelo -> prediccion es lo
que hace auditable el sistema.

Trampas que este flow evita:
1. **Datos sinteticos con semilla fija.** La "produccion" son meses reales
   (``PARTICIONES_PRODUCCION``), que traen drift de verdad.
2. **``iterrows()`` en el bucle de prediccion.** Se predice sobre el DataFrame
   completo, de forma vectorizada.
3. **Un literal ``'stage': 'Production'``** en la fila. Se persiste el **alias**
   consultado y la **version** que ese alias resolvia en el momento de la corrida.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.artifacts import create_table_artifact
from prefect.exceptions import MissingContextError
from prefect.logging.loggers import LoggingAdapter

from BeijingAir.config import (
    DATA_DIR,
    MLFLOW_TRACKING_URI,
    MODELO_ALIAS,
    MODELO_REGISTRADO,
    PARTICIONES_PRODUCCION,
    Particion,
)
from BeijingAir.data.loaders import preparar_particion
from BeijingAir.features import contract as fc

TABLA: Final[str] = "predicciones"

#: Columnas de la fila persistida. Se declaran aqui para que el esquema de la
#: tabla, las consultas SQL y los tests hablen del mismo contrato.
COLUMNAS_FILA: Final[tuple[str, ...]] = (
    "batch_id",
    "prediction_timestamp",
    "particion",
    "model_name",
    "model_version",
    "model_alias",
    "model_uri",
    "prediccion_pm25",
    "station",
    "wd",
    "hora",
    "dia_semana",
    "mes",
    "temporada",
)

_DDL: Final[str] = f"""
CREATE TABLE IF NOT EXISTS {TABLA} (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id             TEXT    NOT NULL,
    prediction_timestamp TEXT    NOT NULL,
    particion            TEXT    NOT NULL,
    model_name           TEXT    NOT NULL,
    model_version        TEXT    NOT NULL,
    model_alias          TEXT    NOT NULL,
    model_uri            TEXT    NOT NULL,
    prediccion_pm25      REAL    NOT NULL,
    station              TEXT    NOT NULL,
    wd                   TEXT    NOT NULL,
    hora                 INTEGER NOT NULL,
    dia_semana           INTEGER NOT NULL,
    mes                  INTEGER NOT NULL,
    temporada            TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_{TABLA}_batch   ON {TABLA} (batch_id);
CREATE INDEX IF NOT EXISTS idx_{TABLA}_version ON {TABLA} (model_version);
"""


def _log() -> logging.Logger | LoggingAdapter:
    """Logger del run si hay contexto de Prefect; logger de modulo si no.

    ``get_run_logger()`` lanza ``MissingContextError`` fuera de un run, y eso
    impediria invocar las tasks con ``.fn()`` en los tests unitarios.
    """
    try:
        return get_run_logger()
    except MissingContextError:
        return logging.getLogger(__name__)


def ruta_sqlite() -> Path:
    """Ruta de la base SQLite de predicciones.

    Configurable por entorno (``PREDICCIONES_DB``) para que los tests no
    escriban en la base del estudiante.
    """
    return Path(os.getenv("PREDICCIONES_DB", DATA_DIR / "predicciones.db"))


def destino() -> str:
    """Describe donde se persiste: Postgres si hay ``DATABASE_URL``, si no SQLite."""
    return os.getenv("DATABASE_URL") or f"sqlite:///{ruta_sqlite()}"


def construir_filas(
    df: pd.DataFrame,
    predicciones: Any,
    *,
    batch_id: str,
    particion: str,
    model_name: str,
    model_version: str,
    model_alias: str,
    model_uri: str,
    momento: datetime | None = None,
) -> pd.DataFrame:
    """Construye el DataFrame a persistir, una fila por prediccion.

    Funcion pura para poder testear el contrato de la fila sin base de datos ni
    modelo. Lanza si el numero de predicciones no coincide con el de filas:
    silenciar ese desalineamiento produce filas con la prediccion de otra
    observacion, el peor error posible en una tabla de auditoria.
    """
    if len(df) != len(predicciones):
        raise ValueError(
            f"Desalineamiento: {len(df)} filas de features y {len(predicciones)} predicciones."
        )
    marca = (momento or datetime.now(UTC)).isoformat(timespec="seconds")
    filas = pd.DataFrame(
        {
            "batch_id": batch_id,
            "prediction_timestamp": marca,
            "particion": particion,
            "model_name": model_name,
            "model_version": model_version,
            # El alias consultado y la version que resolvia, no un literal
            # 'Production' escrito a mano.
            "model_alias": model_alias,
            "model_uri": model_uri,
            "prediccion_pm25": [float(p) for p in predicciones],
            "station": df["station"].astype(str).to_numpy(),
            "wd": df["wd"].astype(str).to_numpy(),
            "hora": df[fc.COL_HORA].astype(int).to_numpy(),
            "dia_semana": df[fc.COL_DIA_SEMANA].astype(int).to_numpy(),
            "mes": df[fc.COL_MES].astype(int).to_numpy(),
            "temporada": df[fc.COL_TEMPORADA].astype(str).to_numpy(),
        }
    )
    return filas[list(COLUMNAS_FILA)]


def escribir(filas: pd.DataFrame, *, url_destino: str | None = None) -> int:
    """Persiste las filas y devuelve cuantas escribio."""
    url = url_destino or destino()
    if url.startswith("sqlite"):
        ruta = Path(url.replace("sqlite:///", "", 1))
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(ruta) as conexion:
            conexion.executescript(_DDL)
            filas.to_sql(TABLA, conexion, if_exists="append", index=False)
        return len(filas)

    from sqlalchemy import create_engine, text  # import tardio

    motor = create_engine(url)
    with motor.begin() as conexion:
        for sentencia in _DDL.strip().split(";"):
            if sentencia.strip():
                conexion.execute(text(sentencia))
        filas.to_sql(TABLA, conexion, if_exists="append", index=False)
    return len(filas)


@task(
    name="leer-produccion",
    description="Lee y valida una particion de produccion simulada.",
    retries=3,
    retry_delay_seconds=[10, 30, 60],
)
def leer_produccion(particion: Particion) -> pd.DataFrame:
    """Descarga la particion, valida el contrato y deriva las features."""
    logger = _log()
    df = preparar_particion(particion)
    logger.info("Particion de produccion %s: %d filas validadas", particion.etiqueta, len(df))
    return df


@task(name="cargar-modelo", description="Carga el modelo por alias desde el registry.")
def cargar_modelo(
    nombre_modelo: str = MODELO_REGISTRADO,
    alias: str = MODELO_ALIAS,
) -> tuple[Any, dict[str, str]]:
    """Carga el modelo por alias y resuelve **que version** resolvio el alias."""
    import mlflow
    from mlflow import MlflowClient

    logger = _log()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    uri = f"models:/{nombre_modelo}@{alias}"
    version = MlflowClient().get_model_version_by_alias(nombre_modelo, alias)
    modelo = mlflow.pyfunc.load_model(uri)
    metadatos = {
        "model_name": nombre_modelo,
        "model_version": str(version.version),
        "model_alias": alias,
        "model_uri": uri,
    }
    logger.info("Cargado %s -> version %s", uri, version.version)
    return modelo, metadatos


@task(name="predecir", description="Predice sobre la particion completa.")
def predecir(modelo: Any, df: pd.DataFrame) -> Any:
    """Predice de forma vectorizada (una llamada con el DataFrame completo)."""
    logger = _log()
    predicciones = modelo.predict(df[fc.FEATURES])
    logger.info("Generadas %d predicciones", len(predicciones))
    return predicciones


@task(name="persistir", description="Escribe las predicciones con su trazabilidad.")
def persistir(
    df: pd.DataFrame,
    predicciones: Any,
    metadatos: dict[str, str],
    particion: Particion,
    batch_id: str,
) -> int:
    """Construye las filas y las escribe en el destino configurado."""
    logger = _log()
    filas = construir_filas(
        df,
        predicciones,
        batch_id=batch_id,
        particion=particion.etiqueta,
        model_name=metadatos["model_name"],
        model_version=metadatos["model_version"],
        model_alias=metadatos["model_alias"],
        model_uri=metadatos["model_uri"],
    )
    escritas = escribir(filas)
    logger.info("Escritas %d filas en %s (batch %s)", escritas, destino(), batch_id)
    return escritas


@flow(
    name="batch-predicciones-beijing",
    description="Predice sobre una particion de produccion y persiste con trazabilidad.",
    log_prints=True,
)
def batch_flow(
    particion: Particion | None = None,
    nombre_modelo: str = MODELO_REGISTRADO,
    alias: str = MODELO_ALIAS,
) -> dict[str, Any]:
    """Pipeline batch: leer -> cargar modelo -> predecir -> persistir.

    Args:
        particion: particion a predecir. Por defecto, la primera de
            ``PARTICIONES_PRODUCCION``.
        nombre_modelo: modelo registrado a usar.
        alias: alias a resolver. ``champion`` en produccion; ``candidate`` sirve
            para hacer un shadow run del candidato sin promoverlo.

    Returns:
        Dict con ``batch_id``, la version usada, filas escritas y el destino.
    """
    logger = _log()
    objetivo = particion or PARTICIONES_PRODUCCION[0]
    batch_id = f"{objetivo.etiqueta}-{uuid.uuid4().hex[:8]}"
    logger.info("Batch %s sobre la particion %s", batch_id, objetivo.etiqueta)

    df = leer_produccion(objetivo)
    modelo, metadatos = cargar_modelo(nombre_modelo, alias)
    predicciones: Any = predecir(modelo, df)
    escritas = persistir(df, predicciones, metadatos, objetivo, batch_id)

    serie = pd.Series([float(p) for p in predicciones])
    tabla: list[dict[str, Any]] = [
        {"campo": "batch_id", "valor": batch_id},
        {"campo": "particion", "valor": objetivo.etiqueta},
        {"campo": "model_version", "valor": metadatos["model_version"]},
        {"campo": "model_alias", "valor": metadatos["model_alias"]},
        {"campo": "filas_escritas", "valor": escritas},
        {"campo": "prediccion_media_pm25", "valor": round(float(serie.mean()), 3)},
        {"campo": "prediccion_p95_pm25", "valor": round(float(serie.quantile(0.95)), 3)},
    ]
    create_table_artifact(
        key="resumen-batch",
        table=tabla,
        description="Resumen del batch, con la version de modelo que produjo las predicciones.",
    )
    return {
        "batch_id": batch_id,
        "particion": objetivo.etiqueta,
        "model_version": metadatos["model_version"],
        "model_alias": metadatos["model_alias"],
        "filas_escritas": escritas,
        "destino": destino(),
    }


if __name__ == "__main__":
    resultado = batch_flow()
    print(
        f"batch={resultado['batch_id']} filas={resultado['filas_escritas']} "
        f"version={resultado['model_version']} destino={resultado['destino']}"
    )
