"""Configuracion central del proyecto BeijingAir.

Las decisiones que afectan a mas de un modulo viven aqui, en un solo lugar.
Regla: si un valor aparece dos veces en el proyecto, sube aqui.

Sin tildes: convencion del curso para archivos .py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = [
    "PROYECTO",
    "VERSION",
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "REPORTS_DIR",
    "URL_DATASET",
    "ARCHIVO_ZIP",
    "FUENTE",
    "LICENCIA",
    "LICENCIA_URL",
    "Particion",
    "COL_TIEMPO",
    "PARTICIONES_TRAIN",
    "PARTICION_VALID",
    "PARTICION_TEST",
    "PARTICIONES_PRODUCCION",
    "TODAS_LAS_PARTICIONES",
    "FILAS_POR_PARTICION",
    "SEMILLA",
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

# =============================================================================
# Fuente del dato
# =============================================================================
URL_DATASET: Final[str] = (
    "https://archive.ics.uci.edu/static/public/501/"
    "beijing+multi+site+air+quality+data.zip"
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

PARTICIONES_TRAIN: Final[tuple[Particion, ...]] = (
    Particion("train", "2013-03-01", "2015-06-30"),
)
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


def asegurar_directorios() -> None:
    """Crea los directorios de trabajo si no existen."""
    for directorio in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR):
        directorio.mkdir(parents=True, exist_ok=True)