"""Configuracion central del proyecto BeijingAir.

Las decisiones que afectan a mas de un modulo viven aqui, en un solo lugar.
Regla: si un valor aparece dos veces en el proyecto, sube aqui.

Sin tildes: convencion del proyecto para archivos .py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = [
    "ALFA_DRIFT",
    "API_PORT",
    "ARCHIVO_ZIP",
    "COL_TIEMPO",
    "DATA_DIR",
    "ESTADO_ENTRENAMIENTO",
    "FILAS_POR_PARTICION",
    "FUENTE",
    "LICENCIA",
    "LICENCIA_URL",
    "MAX_EMPEORAMIENTO_MAE",
    "MAX_MAE_TEST",
    "MIN_R2_TEST",
    "MLFLOW_EXPERIMENT",
    "MLFLOW_PORT",
    "MLFLOW_TRACKING_URI",
    "MODELO_ALIAS",
    "MODELO_ALIAS_CANDIDATO",
    "MODELO_REGISTRADO",
    "MODELO_URI",
    "PARTICIONES_PRODUCCION",
    "PARTICIONES_TRAIN",
    "PARTICION_TEST",
    "PARTICION_VALID",
    "PREFECT_PORT",
    "PREFECT_SCHEDULE_CRON",
    "PREFECT_TIMEZONE",
    "PROCESSED_DIR",
    "PROJECT_ROOT",
    "PROYECTO",
    "RAW_DIR",
    "REPORTS_DIR",
    "SEMILLA",
    "TAG_VALIDACION",
    "TODAS_LAS_PARTICIONES",
    "UMBRAL_DRIFT_COLUMNAS",
    "URL_DATASET",
    "VERSION",
    "Particion",
    "asegurar_directorios",
]

# =============================================================================
# Identidad del proyecto
# =============================================================================
PROYECTO: str = "beijing-air"
VERSION: str = "0.1.0"

# =============================================================================
# Rutas
# =============================================================================
# PROJECT_ROOT se deriva de la ubicacion de este archivo, nunca de una ruta
# absoluta escrita a mano: una ruta como C:\Users\... garantiza que el proyecto
# no corre en la maquina de nadie mas.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DATA_DIR: Final[Path] = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
ESTADO_ENTRENAMIENTO: Final[Path] = PROCESSED_DIR / "ultimo_entrenamiento.json"

# =============================================================================
# Fuente del dato
# =============================================================================
URL_DATASET: Final[str] = (
    "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
)
ARCHIVO_ZIP: Final[str] = "beijing-multi-site-air-quality.zip"

FUENTE: Final[str] = "UCI Machine Learning Repository, dataset 501"
LICENCIA: Final[str] = "CC BY 4.0"
LICENCIA_URL: Final[str] = "https://creativecommons.org/licenses/by/4.0/"


# =============================================================================
# Particiones - FIJAS. No usar datetime.now().
# =============================================================================
@dataclass(frozen=True)
class Particion:
    """Un rango de fechas del dataset.

    Beijing viene en UN solo archivo que cubre de 2013-03 a 2017-02, asi que
    las particiones son rangos temporales y no archivos separados.
    """

    etiqueta: str
    desde: str
    hasta: str

    def __str__(self) -> str:
        return self.etiqueta


#: Eje temporal. Se construye a partir de year/month/day/hour del CSV crudo.
COL_TIEMPO: Final[str] = "datetime"

PARTICIONES_TRAIN: Final[tuple[Particion, ...]] = (Particion("train", "2013-03-01", "2015-06-30"),)
PARTICION_VALID: Final[Particion] = Particion("valid", "2015-07-01", "2015-12-31")
PARTICION_TEST: Final[Particion] = Particion("test", "2016-01-01", "2016-06-30")
PARTICIONES_PRODUCCION: Final[tuple[Particion, ...]] = (
    Particion("produccion", "2016-07-01", "2017-02-28"),
)

TODAS_LAS_PARTICIONES: Final[tuple[Particion, ...]] = (
    *PARTICIONES_TRAIN,
    PARTICION_VALID,
    PARTICION_TEST,
    *PARTICIONES_PRODUCCION,
)

# =============================================================================
# Muestreo y determinismo
# =============================================================================
#: Un entrenamiento de 3 segundos se itera; uno de 20 minutos se ejecuta una
#: vez y uno se cree lo que diga.
FILAS_POR_PARTICION: Final[int] = 50_000
#: Semilla global. Se pasa explicitamente a cada componente.
SEMILLA: Final[int] = 42

# =============================================================================
# Tracking de experimentos
# =============================================================================
#: El servidor local se ejecuta con ``make mlflow``. Se puede reemplazar por
#: una URI remota sin cambiar el codigo, mediante MLFLOW_TRACKING_URI.
MLFLOW_PORT: Final[int] = int(os.getenv("MLFLOW_PORT", "5001"))
MLFLOW_TRACKING_URI: Final[str] = os.getenv(
    "MLFLOW_TRACKING_URI", f"http://127.0.0.1:{MLFLOW_PORT}"
)
MLFLOW_EXPERIMENT: Final[str] = "beijing-air"
MODELO_REGISTRADO: Final[str] = "beijing-air-pm25"

# =============================================================================
# Serving
# =============================================================================
#: La API consulta el Registry; nunca carga un .pkl copiado en la imagen.
#: ``champion`` se asignara mediante el gate de promocion de CI/CD. Mientras no
#: exista, la API informa estado degradado sin caerse ni servir ``candidate``.
MODELO_ALIAS: Final[str] = os.getenv("MODELO_ALIAS", "champion")
MODELO_ALIAS_CANDIDATO: Final[str] = os.getenv("MODELO_ALIAS_CANDIDATO", "candidate")
MODELO_URI: Final[str] = os.getenv("MODELO_URI", f"models:/{MODELO_REGISTRADO}@{MODELO_ALIAS}")
API_PORT: Final[int] = int(os.getenv("API_PORT", "8000"))

# =============================================================================
# Gate de promocion
# =============================================================================
#: Limites iniciales para el holdout temporal ``test``. Se pueden ajustar por
#: variables de entorno sin cambiar el codigo; el gate falla cerrado si faltan
#: las metricas del candidato o si empeora frente a champion.
MAX_MAE_TEST: Final[float] = float(os.getenv("MAX_MAE_TEST", "45.0"))
MIN_R2_TEST: Final[float] = float(os.getenv("MIN_R2_TEST", "0.0"))
MAX_EMPEORAMIENTO_MAE: Final[float] = float(os.getenv("MAX_EMPEORAMIENTO_MAE", "0.05"))

# =============================================================================
# Orquestacion
# =============================================================================
#: Prefect se usa localmente; la URI se configura por entorno.
PREFECT_PORT: Final[int] = int(os.getenv("PREFECT_PORT", "4200"))
PREFECT_SCHEDULE_CRON: Final[str] = "0 3 1 * *"
PREFECT_TIMEZONE: Final[str] = "America/Bogota"
#: Tag que el gate escribe ANTES de mover el alias, para dejar registrado
#: por que se promovio (o por que no).
TAG_VALIDACION: Final[str] = "validation_status"


def asegurar_directorios() -> None:
    """Crea los directorios de trabajo si no existen."""
    for directorio in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR):
        directorio.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Monitoreo de drift
# =============================================================================
#: Fraccion de columnas con drift que dispara la alerta.
#: TODO: justificar este numero en docs/politica-de-reentrenamiento.md.
UMBRAL_DRIFT_COLUMNAS: Final[float] = 0.30

#: Nivel de significancia de los tests por columna. Ojo: con 420.768 filas TODO
#: sale significativo, asi que el p-valor NO decide solo; el codigo exige
#: ademas un tamano de efecto minimo.
ALFA_DRIFT: Final[float] = 0.05
