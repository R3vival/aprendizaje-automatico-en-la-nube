"""Valida las particiones reales del dataset Beijing desde la línea de comandos."""

from __future__ import annotations

from BeijingAir.config import TODAS_LAS_PARTICIONES
from BeijingAir.data.contract import validar_crudos
from BeijingAir.data.loaders import leer_particion


def main() -> None:
    """Descarga si hace falta y valida cada lote temporal completo."""
    for particion in TODAS_LAS_PARTICIONES:
        dataframe = leer_particion(particion)
        validar_crudos(dataframe)
        print(f"{particion}: {len(dataframe):,} filas validadas")


if __name__ == "__main__":
    main()
