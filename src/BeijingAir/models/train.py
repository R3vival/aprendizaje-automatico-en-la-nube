"""Entrena y registra experimentos de PM2.5 en MLflow.

Uso:
    make mlflow  # en una terminal
    make train   # en otra terminal
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from BeijingAir.config import (
    FILAS_POR_PARTICION,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODELO_REGISTRADO,
    PARTICIONES_TRAIN,
    PARTICION_VALID,
    PROJECT_ROOT,
    RAW_DIR,
    SEMILLA,
)
from BeijingAir.data.loaders import preparar_particion, preparar_particiones
from BeijingAir.features import contract as fc
from BeijingAir.models.pipeline import (
    ResultadoEvaluacion,
    TipoModelo,
    crear_pipeline,
    evaluar_regresion,
    separar_features_target,
)


def _commit_actual() -> str:
    """Devuelve el commit para asociar cada corrida con una version de codigo."""
    try:
        resultado = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "desconocido"
    return resultado.stdout.strip()


def _hash_dataset() -> str:
    """Lee el hash registrado en metadata.json sin cargar los datos completos."""
    ruta_metadata = RAW_DIR / "metadata.json"
    if not ruta_metadata.exists():
        return "no-disponible"
    metadata = json.loads(ruta_metadata.read_text(encoding="utf-8"))
    primer_archivo = next(iter(metadata.values()), {})
    return str(primer_archivo.get("sha256", "no-disponible"))


def _parametros_modelo(modelo: TipoModelo, n_estimators: int) -> dict[str, str | int]:
    """Devuelve solo los parametros que distinguen una corrida de otra."""
    parametros: dict[str, str | int] = {"modelo": modelo, "semilla": SEMILLA}
    if modelo == "bosque":
        parametros["n_estimators"] = n_estimators
        parametros["min_samples_leaf"] = 3
    return parametros


def _loggear_corrida(
    modelo: TipoModelo,
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    estaciones_valid: pd.Series,
    n_estimators: int,
    registrar: bool,
) -> ResultadoEvaluacion:
    """Entrena un candidato y guarda modelo, metricas y artefactos en MLflow."""
    pipeline = crear_pipeline(modelo, n_estimators=n_estimators)
    pipeline.fit(x_train, y_train)
    predicciones = pipeline.predict(x_valid)
    evaluacion = evaluar_regresion(y_valid, predicciones, estaciones_valid)

    mlflow.log_params(_parametros_modelo(modelo, n_estimators))
    mlflow.log_metrics(
        {
            "mae_valid": evaluacion.mae,
            "rmse_valid": evaluacion.rmse,
            "r2_valid": evaluacion.r2,
            "peor_mae_estacion_valid": evaluacion.peor_mae_estacion,
        }
    )
    mlflow.log_dict(evaluacion.como_dict(), "evaluacion/metricas_por_estacion.json")

    ejemplo_entrada = x_train.head(3)
    firma = infer_signature(ejemplo_entrada, pipeline.predict(ejemplo_entrada))
    kwargs_modelo: dict[str, object] = {
        "artifact_path": "model",
        "input_example": ejemplo_entrada,
        "signature": firma,
    }
    if registrar:
        kwargs_modelo["registered_model_name"] = MODELO_REGISTRADO
    mlflow.sklearn.log_model(pipeline, **kwargs_modelo)
    return evaluacion


def entrenar_y_registrar(
    *,
    filas: int | None = FILAS_POR_PARTICION,
    n_estimators: int = 300,
    registrar: bool = False,
) -> dict[str, ResultadoEvaluacion]:
    """Compara baseline y bosque sobre el split temporal fijo.

    El conjunto de validacion es posterior al train y nunca participa en el
    ``fit``. MLflow recibe un run padre con los datos de procedencia y un run
    anidado para cada candidato.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    train = preparar_particiones(PARTICIONES_TRAIN, filas=filas)
    valid = preparar_particion(PARTICION_VALID, filas=filas)
    x_train, y_train = separar_features_target(train)
    x_valid, y_valid = separar_features_target(valid)

    tags = {
        "dataset_sha256": _hash_dataset(),
        "git_commit": _commit_actual(),
        "particion_train": ",".join(str(particion) for particion in PARTICIONES_TRAIN),
        "particion_valid": str(PARTICION_VALID),
        "target": fc.TARGET,
    }
    resultados: dict[str, ResultadoEvaluacion] = {}

    with mlflow.start_run(run_name="comparacion-pm25"):
        mlflow.set_tags(tags)
        mlflow.log_params({"filas_por_particion": filas or "completa"})
        mlflow.log_metrics({"filas_train": float(len(train)), "filas_valid": float(len(valid))})

        modelos: tuple[TipoModelo, ...] = ("baseline", "bosque")
        for modelo in modelos:
            with mlflow.start_run(run_name=modelo, nested=True):
                resultados[modelo] = _loggear_corrida(
                    modelo,
                    x_train=x_train,
                    y_train=y_train,
                    x_valid=x_valid,
                    y_valid=y_valid,
                    estaciones_valid=valid[fc.COL_SUBGRUPO],
                    n_estimators=n_estimators,
                    registrar=registrar and modelo == "bosque",
                )

    return resultados


def _argumentos(argumentos: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena PM2.5 y registra las corridas en MLflow.")
    parser.add_argument(
        "--filas",
        type=int,
        default=FILAS_POR_PARTICION,
        help="Filas por particion; usa 0 para procesar la particion completa.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Numero de arboles del Random Forest.",
    )
    parser.add_argument(
        "--registrar",
        action="store_true",
        help="Registra el bosque en el Model Registry de MLflow.",
    )
    return parser.parse_args(argumentos)


def main(argumentos: Sequence[str] | None = None) -> None:
    """Punto de entrada de ``python -m BeijingAir.models.train``."""
    opciones = _argumentos(argumentos)
    filas = None if opciones.filas == 0 else opciones.filas
    resultados = entrenar_y_registrar(
        filas=filas,
        n_estimators=opciones.n_estimators,
        registrar=opciones.registrar,
    )
    for modelo, evaluacion in resultados.items():
        print(
            f"{modelo}: MAE={evaluacion.mae:.3f}, RMSE={evaluacion.rmse:.3f}, "
            f"peor estacion={evaluacion.peor_mae_estacion:.3f}"
        )


if __name__ == "__main__":
    main()
