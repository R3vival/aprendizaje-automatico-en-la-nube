"""Descarga, verificacion por hash y preparacion del dataset Beijing.

Cuatro decisiones que conviene copiar tal cual:

1. **Particiones fijas** (``config.py``), nunca ``datetime.now()``. Beijing viene
   en UN archivo que cubre 2013-03 a 2017-02, asi que una particion es un rango
   temporal (ver ``config.Particion``), no un archivo separado.
2. **Verificacion por hash**. Se registra el SHA-256 del ZIP en
   ``data/raw/metadata.json``. Si el proveedor republica el archivo, la metrica
   que reportaste deja de ser comparable y quieres enterarte por un aviso.
3. **Muestreo determinista** a un tamano fijo, para que entrenar tome segundos.
4. **Imputacion por columna, con indicador de ausencia.** Nunca ``fillna(0)``
   sobre un contaminante: afirmaria aire limpio donde el sensor callo.

El crudo concatenado se cachea en ``data/processed/beijing_crudo.parquet`` para
no re-leer los 12 CSV (uno por estacion) en cada particion.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from BeijingAir.config import (
    FILAS_POR_PARTICION,
    PROCESSED_DIR,
    SEMILLA,
    Particion,
)
from BeijingAir.data import contract as dc
from BeijingAir.data.descarga import cargar_crudo, descargar, extraer
from BeijingAir.features import contract as fc

logger = logging.getLogger(__name__)

CACHE_CRUDO = PROCESSED_DIR / "beijing_crudo.parquet"


def asegurar_crudo() -> pd.DataFrame:
    """Devuelve el crudo completo cacheado, descargandolo si hace falta.

    El cache existe para no re-leer los 12 CSV en cada particion. El dato en si
    no se versiona; solo ``data/raw/metadata.json`` registra su procedencia.
    """
    if CACHE_CRUDO.exists():
        return pd.read_parquet(CACHE_CRUDO)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = descargar()
    extraer(zip_path)
    df = cargar_crudo()
    df.to_parquet(CACHE_CRUDO, index=False)
    logger.info("Crudo cacheado en %s (%d filas)", CACHE_CRUDO, len(df))
    return df


def leer_particion(particion: Particion) -> pd.DataFrame:
    """Lee una particion (rango temporal) del crudo cacheado."""
    crudo = asegurar_crudo()
    inicio = pd.Timestamp(particion.desde)
    fin_exclusivo = pd.Timestamp(particion.hasta) + pd.Timedelta(days=1)
    mascara = (crudo[fc.COL_TIEMPO] >= inicio) & (crudo[fc.COL_TIEMPO] < fin_exclusivo)
    return crudo.loc[mascara].reset_index(drop=True)


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la estrategia de imputacion y los filtros de negocio.

    Estrategia documentada (ver docs/dataset-card.md):

    - El **target** (``PM2.5``) NO se imputa: inventar una etiqueta ensena al
      modelo a replicar un sensor que no existio. Las filas con target nulo se
      descartan (2,1 % del total).
    - Las **numericas** (contaminantes y meteorologia) se imputan con la mediana
      de la columna, y se deja constancia en ``<col>_era_nulo``.
    - La **categorica** ``wd`` se rellena con ``desconocido``.

    ``<col>_era_nulo`` NO es una feature: no esta en ``features.contract.FEATURES``
    y el modelo no la ve. Es trazabilidad de la imputacion, y la consume el
    contrato de ``RegistrosProcesados`` para saltarse la regla ``DEWP <= TEMP``
    donde alguna de las dos magnitudes fue imputada. El motivo de no promoverla a
    feature esta en docs/dataset-card.md: la API exige las trece magnitudes, asi
    que el indicador seria constante en serving y variable en entrenamiento.

    ``fillna(0)`` sobre un contaminante seria un error: afirmaria aire limpio en
    las horas en que el sensor callo, y el modelo aprenderia esos ceros como
    reales.
    """
    out = df.copy()
    out = out[out[fc.TARGET].notna()].copy()
    for col in fc.CRUDAS_NUMERICAS:
        if col in out.columns:
            out[f"{col}_era_nulo"] = out[col].isna().astype("int8")
            out[col] = out[col].astype("float64").fillna(out[col].median())
    for col in fc.CRUDAS_CATEGORICAS:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("desconocido").astype(str)
    return out.reset_index(drop=True)


def preparar_particion(
    particion: Particion,
    *,
    filas: int | None = FILAS_POR_PARTICION,
    validar: bool = True,
) -> pd.DataFrame:
    """Lee, valida, limpia, muestrea y deriva features de una particion.

    El orden importa y es deliberado:

    1. leer el crudo
    2. **validar el contrato del crudo** (falla temprano, con el proveedor)
    3. imputar/filtrar segun la estrategia declarada
    4. muestrear (determinista)
    5. derivar features
    6. **validar el contrato del procesado** (falla tarde, con tu pipeline)

    Args:
        particion: particion a preparar.
        filas: tamano de la muestra. ``None`` usa la particion completa.
        validar: desactivarlo solo tiene sentido para demostrar en clase.
    """
    df = leer_particion(particion)
    if validar:
        df = dc.validar_crudos(df)
    df = limpiar(df)
    if filas is not None and len(df) > filas:
        df = df.sample(n=filas, random_state=SEMILLA).reset_index(drop=True)
    df = fc.construir_features(df)
    if validar:
        df = dc.validar_procesados(df)
    return df


def preparar_particiones(
    particiones: tuple[Particion, ...] | list[Particion],
    **kwargs: object,
) -> pd.DataFrame:
    """Concatena varias particiones preparadas, ordenadas por el eje temporal.

    El orden temporal no es cosmetico: es lo que hace que un split por posicion
    sea un split honesto.
    """
    marcos = [preparar_particion(p, **kwargs) for p in particiones]  # type: ignore[arg-type]
    return pd.concat(marcos, ignore_index=True).sort_values(fc.COL_TIEMPO).reset_index(drop=True)


def cachear(df: pd.DataFrame, nombre: str) -> Path:
    """Guarda un dataframe procesado en ``data/processed/``."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    destino = PROCESSED_DIR / f"{nombre}.parquet"
    df.to_parquet(destino, index=False)
    logger.info("Cacheado %d filas en %s", len(df), destino)
    return destino


def cargar_cache(nombre: str) -> pd.DataFrame | None:
    """Lee un dataframe procesado del cache, o ``None`` si no existe."""
    ruta = PROCESSED_DIR / f"{nombre}.parquet"
    return pd.read_parquet(ruta) if ruta.exists() else None


def split_temporal(
    df: pd.DataFrame, *, fraccion_train: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split por posicion sobre datos ya ordenados en el tiempo.

    ``train_test_split(shuffle=True)`` sobre datos con eje temporal es un bug:
    mezcla el futuro dentro del entrenamiento y produce una metrica optimista.
    """
    if not df[fc.COL_TIEMPO].is_monotonic_increasing:
        raise ValueError(
            f"El dataframe debe venir ordenado por {fc.COL_TIEMPO} antes de "
            "hacer un split temporal."
        )
    corte = int(len(df) * fraccion_train)
    return df.iloc[:corte].copy(), df.iloc[corte:].copy()
