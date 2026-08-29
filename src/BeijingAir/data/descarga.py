"""Descarga, verificacion por hash y carga del dataset Beijing Multi-Site.

El dato NO se versiona. Se descarga con este modulo y se verifica por hash.
Lo que si va al repositorio es data/raw/metadata.json: la procedencia del dato.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from BeijingAir.config import (
    ARCHIVO_ZIP,
    COL_TIEMPO,
    FUENTE,
    LICENCIA,
    LICENCIA_URL,
    RAW_DIR,
    URL_DATASET,
    Particion,
)

METADATA_PATH = RAW_DIR / "metadata.json"
DIR_EXTRAIDO = RAW_DIR / "beijing"
_CHUNK = 1 << 20


def sha256(path: Path) -> str:
    """SHA-256 del archivo, leido por bloques para no cargarlo en memoria."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def descargar(*, forzar: bool = False) -> Path:
    """Descarga el ZIP y registra su procedencia en metadata.json."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    destino = RAW_DIR / ARCHIVO_ZIP

    if destino.exists() and not forzar:
        print(f"Ya existe: {destino}")
    else:
        print(f"Descargando {URL_DATASET} ...")
        urllib.request.urlretrieve(URL_DATASET, destino)

    huella = sha256(destino)
    meta: dict[str, dict] = {}
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    anterior = meta.get(ARCHIVO_ZIP, {}).get("sha256")
    if anterior and anterior != huella:
        print("AVISO: el hash cambio. El proveedor republico el archivo.")
        print(f"  registrado: {anterior}")
        print(f"  actual:     {huella}")

    meta[ARCHIVO_ZIP] = {
        "url": URL_DATASET,
        "sha256": huella,
        "bytes": destino.stat().st_size,
        "fuente": FUENTE,
        "licencia": LICENCIA,
        "licencia_url": LICENCIA_URL,
    }
    METADATA_PATH.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Procedencia registrada en {METADATA_PATH}")
    print(f"SHA-256: {huella}")
    return destino


def extraer(zip_path: Path) -> Path:
    """Descomprime. Beijing trae otro ZIP adentro: hay que hacerlo dos veces."""
    if list(DIR_EXTRAIDO.rglob("PRSA_Data_*.csv")):
        return DIR_EXTRAIDO
    DIR_EXTRAIDO.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DIR_EXTRAIDO)
    for interno in DIR_EXTRAIDO.rglob("*.zip"):
        with zipfile.ZipFile(interno) as z:
            z.extractall(DIR_EXTRAIDO)
    return DIR_EXTRAIDO


def cargar_crudo() -> pd.DataFrame:
    """Junta los 12 CSV (uno por estacion) y construye el eje temporal."""
    csvs = sorted(DIR_EXTRAIDO.rglob("PRSA_Data_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No hay CSV de Beijing en {DIR_EXTRAIDO}.")
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df[COL_TIEMPO] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    return df.sort_values(COL_TIEMPO).reset_index(drop=True)


def filtrar(df: pd.DataFrame, particion: Particion) -> pd.DataFrame:
    """Recorta el dataframe al rango de fechas de una particion."""
    inicio = pd.Timestamp(particion.desde)
    fin_exclusivo = pd.Timestamp(particion.hasta) + pd.Timedelta(days=1)
    mascara = (df[COL_TIEMPO] >= inicio) & (df[COL_TIEMPO] < fin_exclusivo)
    return df.loc[mascara].reset_index(drop=True)


def main() -> None:
    """Descarga, extrae y describe el dataset."""
    zip_path = descargar()
    extraer(zip_path)
    df = cargar_crudo()
    print(f"\nFilas: {len(df):,}   Columnas: {len(df.columns)}")
    print(f"Rango temporal: {df[COL_TIEMPO].min()} a {df[COL_TIEMPO].max()}")
    print(f"Estaciones: {df['station'].nunique()}")
    print("\nNulos por columna:")
    print(df.isna().sum()[lambda s: s > 0])


if __name__ == "__main__":
    main()
