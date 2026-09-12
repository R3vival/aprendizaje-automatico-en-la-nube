"""Entrena y registra experimentos de PM2.5 en MLflow.

Uso:
    make mlflow  # en una terminal
    make train   # en otra terminal
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any, Final

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from mlflow.models import infer_signature
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from BeijingAir.config import (
    FILAS_POR_PARTICION,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODELO_ALIAS_CANDIDATO,
    MODELO_REGISTRADO,
    PARTICION_TEST,
    PARTICION_VALID,
    PARTICIONES_TRAIN,
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


def hash_dataset() -> str:
    """Lee el hash registrado en metadata.json sin cargar los datos completos."""
    ruta_metadata = RAW_DIR / "metadata.json"
    if not ruta_metadata.exists():
        return "no-disponible"
    metadata = json.loads(ruta_metadata.read_text(encoding="utf-8"))
    primer_archivo: dict[str, object] = next(iter(metadata.values()), {})
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
    x_test: pd.DataFrame,
    y_test: pd.Series,
    estaciones_test: pd.Series,
    n_estimators: int,
    registrar: bool,
) -> ResultadoEvaluacion:
    """Entrena un candidato y guarda modelo, metricas y artefactos en MLflow."""
    pipeline = crear_pipeline(modelo, n_estimators=n_estimators)
    pipeline.fit(x_train, y_train)
    predicciones = pipeline.predict(x_valid)
    evaluacion = evaluar_regresion(y_valid, predicciones, estaciones_valid)
    predicciones_test = pipeline.predict(x_test)
    evaluacion_test = evaluar_regresion(y_test, predicciones_test, estaciones_test)

    mlflow.log_params(_parametros_modelo(modelo, n_estimators))
    mlflow.log_metrics(
        {
            "mae_valid": evaluacion.mae,
            "rmse_valid": evaluacion.rmse,
            "r2_valid": evaluacion.r2,
            "peor_mae_estacion_valid": evaluacion.peor_mae_estacion,
            "mae_test": evaluacion_test.mae,
            "rmse_test": evaluacion_test.rmse,
            "r2_test": evaluacion_test.r2,
            "peor_mae_estacion_test": evaluacion_test.peor_mae_estacion,
        }
    )
    mlflow.log_dict(evaluacion.como_dict(), "evaluacion/metricas_por_estacion.json")
    mlflow.log_dict(evaluacion_test.como_dict(), "evaluacion/metricas_test_por_estacion.json")

    ejemplo_entrada = x_train.head(3)
    firma = infer_signature(ejemplo_entrada, pipeline.predict(ejemplo_entrada))
    kwargs_modelo: dict[str, object] = {
        "name": "model",
        "input_example": ejemplo_entrada,
        "signature": firma,
        "serialization_format": "cloudpickle",
    }
    if registrar:
        kwargs_modelo["registered_model_name"] = MODELO_REGISTRADO
    informacion_modelo = mlflow.sklearn.log_model(pipeline, **kwargs_modelo)
    version_registrada = getattr(informacion_modelo, "registered_model_version", None)
    if registrar and version_registrada:
        mlflow.MlflowClient().set_registered_model_alias(
            MODELO_REGISTRADO, MODELO_ALIAS_CANDIDATO, str(version_registrada)
        )
    return evaluacion


def entrenar_y_registrar(
    *,
    filas: int | None = FILAS_POR_PARTICION,
    n_estimators: int = 300,
    registrar: bool = False,
    tags_adicionales: Mapping[str, str] | None = None,
    datos_train: pd.DataFrame | None = None,
    datos_valid: pd.DataFrame | None = None,
    datos_test: pd.DataFrame | None = None,
) -> dict[str, ResultadoEvaluacion]:
    """Compara baseline y bosque sobre el split temporal fijo.

    El conjunto de validacion es posterior al train y nunca participa en el
    ``fit``. MLflow recibe un run padre con los datos de procedencia y un run
    anidado para cada candidato.
    """
    datasets_preparados = (datos_train, datos_valid, datos_test)
    if any(dataset is None for dataset in datasets_preparados) and not all(
        dataset is None for dataset in datasets_preparados
    ):
        raise ValueError("datos_train, datos_valid y datos_test se deben proporcionar juntos.")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    if datos_train is None or datos_valid is None or datos_test is None:
        train = preparar_particiones(PARTICIONES_TRAIN, filas=filas)
        valid = preparar_particion(PARTICION_VALID, filas=filas)
        test = preparar_particion(PARTICION_TEST, filas=filas)
    else:
        train = datos_train
        valid = datos_valid
        test = datos_test
    x_train, y_train = separar_features_target(train)
    x_valid, y_valid = separar_features_target(valid)
    x_test, y_test = separar_features_target(test)

    tags = {
        "dataset_sha256": hash_dataset(),
        "git_commit": _commit_actual(),
        "particion_train": ",".join(str(particion) for particion in PARTICIONES_TRAIN),
        "particion_valid": str(PARTICION_VALID),
        "particion_test": str(PARTICION_TEST),
        "target": fc.TARGET,
    }
    if tags_adicionales:
        tags.update(tags_adicionales)
    resultados: dict[str, ResultadoEvaluacion] = {}

    with mlflow.start_run(run_name="comparacion-pm25"):
        mlflow.set_tags(tags)
        mlflow.log_params({"filas_por_particion": filas or "completa"})
        mlflow.log_metrics(
            {
                "filas_train": float(len(train)),
                "filas_valid": float(len(valid)),
                "filas_test": float(len(test)),
            }
        )

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
                    x_test=x_test,
                    y_test=y_test,
                    estaciones_test=test[fc.COL_SUBGRUPO],
                    n_estimators=n_estimators,
                    registrar=registrar and modelo == "bosque",
                )

    return resultados


# =============================================================================
# Busqueda de hiperparametros (Optuna) — Sesion 3 del curso
# =============================================================================
#: Espacio de busqueda del Random Forest. ``n_estimators`` se maneja aparte
#: porque ``crear_pipeline`` lo recibe como parametro nombrado; el resto entra
#: por ``kwargs_extra``. Definirlo como dato (no como codigo) permite loguear el
#: espacio explorado junto al study.
ESPACIO_RANDOM_FOREST: Final[dict[str, tuple[str, int | float, int | float]]] = {
    "n_estimators": ("int", 100, 500),
    "max_depth": ("int", 5, 30),
    "min_samples_leaf": ("int", 1, 10),
    "min_samples_split": ("int", 2, 15),
    "max_features": ("float", 0.3, 1.0),
}


def _sugerir(trial: optuna.Trial) -> dict[str, Any]:
    """Traduce ``ESPACIO_RANDOM_FOREST`` a llamadas ``suggest_*`` de Optuna."""
    propuesta: dict[str, Any] = {}
    for nombre, (clase, bajo, alto) in ESPACIO_RANDOM_FOREST.items():
        if clase == "int":
            propuesta[nombre] = trial.suggest_int(nombre, int(bajo), int(alto))
        else:
            propuesta[nombre] = trial.suggest_float(nombre, float(bajo), float(alto))
    return propuesta


def _loggear_trial(
    trial: optuna.Trial,
    *,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    estaciones_valid: pd.Series,
) -> float:
    """Entrena un bosque con la propuesta del trial y devuelve su MAE de valid.

    El fit ocurre dentro del run anidado: cada trial queda asociado a su
    combinacion de hiperparametros y a su metrica. El objetivo se mide en
    ``valid``, nunca en ``test`` (el holdout es el juez del gate de S06).
    """
    propuesta = _sugerir(trial)
    n_estim = int(propuesta.pop("n_estimators"))
    with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
        pipeline = crear_pipeline("bosque", n_estimators=n_estim, **propuesta)
        pipeline.fit(x_train, y_train)
        evaluacion = evaluar_regresion(y_valid, pipeline.predict(x_valid), estaciones_valid)
        mlflow.log_params(
            {
                "modelo": "bosque",
                "semilla": SEMILLA,
                "trial": trial.number,
                "n_estimators": n_estim,
                **propuesta,
            }
        )
        mlflow.log_metrics(
            {
                "mae_valid": evaluacion.mae,
                "rmse_valid": evaluacion.rmse,
                "r2_valid": evaluacion.r2,
                "peor_mae_estacion_valid": evaluacion.peor_mae_estacion,
            }
        )
    return evaluacion.mae


def optimizar_hiperparametros(
    *,
    trials: int = 20,
    filas: int | None = FILAS_POR_PARTICION,
) -> dict[str, Any]:
    """Busca hiperparametros del bosque con Optuna sobre ``valid``.

    Estructura de runs: un **parent run** para el study y un **child run por
    trial** (``nested=True``), de modo que la UI muestra el arbol parent/child en
    lugar de runs sueltos. El sampler usa la semilla global para que dos corridas
    del mismo estudio sobre los mismos datos exploren las mismas combinaciones.

    El objetivo se calcula SIEMPRE sobre ``PARTICION_VALID``; ``PARTICION_TEST``
    no participa en la seleccion de hiperparametros.

    Returns:
        ``study.best_params``, el diccionario de mejores hiperparametros.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    train = preparar_particiones(PARTICIONES_TRAIN, filas=filas)
    valid = preparar_particion(PARTICION_VALID, filas=filas)
    x_train, y_train = separar_features_target(train)
    x_valid, y_valid = separar_features_target(valid)
    estaciones_valid = valid[fc.COL_SUBGRUPO]

    study = optuna.create_study(
        study_name=f"hpo-bosque-{trials}",
        direction="minimize",
        sampler=TPESampler(seed=SEMILLA),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1),
    )

    def objetivo(trial: optuna.Trial) -> float:
        return _loggear_trial(
            trial,
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            estaciones_valid=estaciones_valid,
        )

    with mlflow.start_run(run_name=f"hpo-bosque-{trials}-trials"):
        mlflow.set_tags({"tipo": "hpo-parent", "sampler": "TPESampler", "pruner": "MedianPruner"})
        mlflow.log_params(
            {
                "trials": trials,
                "espacio": {k: list(v) for k, v in ESPACIO_RANDOM_FOREST.items()},
                "semilla": SEMILLA,
            }
        )
        study.optimize(objetivo, n_trials=trials, catch=())
        completados = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        mlflow.log_metrics(
            {
                "mejor_mae_valid": float(study.best_value),
                "trials_completados": float(len(completados)),
            }
        )
        mlflow.log_dict(dict(study.best_params), "hpo/best_params.json")
        print(f"HPO: mejor mae_valid={study.best_value:.3f} con {study.best_params}")
    return dict(study.best_params)


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
    parser.add_argument(
        "--hpo",
        action="store_true",
        help="Busca hiperparametros del bosque con Optuna (runs anidados).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Numero de trials de la busqueda de hiperparametros.",
    )
    return parser.parse_args(argumentos)


def main(argumentos: Sequence[str] | None = None) -> None:
    """Punto de entrada de ``python -m BeijingAir.models.train``."""
    opciones = _argumentos(argumentos)
    filas = None if opciones.filas == 0 else opciones.filas
    if opciones.hpo:
        optimizar_hiperparametros(trials=opciones.trials, filas=filas)
        return
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
