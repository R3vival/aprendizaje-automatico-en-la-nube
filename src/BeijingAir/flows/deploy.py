"""Deployments del flow de entrenamiento (Prefect 3).

Un flow que solo se ejecuta cuando alguien escribe ``python ...`` no esta
orquestado: esta automatizado a medias. El deployment es lo que convierte al flow
en algo que el servidor conoce, puede programar y puede lanzar sin nadie delante.

``serve`` vs ``deploy`` + work pool
-----------------------------------
| Criterio | ``flow.serve(...)`` | ``flow.deploy(..., work_pool_name=...)`` |
|---|---|---|
| Quien ejecuta | el proceso que quedo vivo | un **worker** que toma trabajo del work pool |
| Infraestructura | estatica: la maquina donde corriste el comando | dinamica: contenedor/pod por corrida |
| Aislamiento | ninguno (mismo entorno del proceso) | por corrida (imagen propia) |
| Setup | cero | work pool + worker |
| Cuando usarlo | clase, laboratorio | produccion, cargas heterogeneas |

En clase se usa ``serve`` (cero infraestructura, resultado visible en la UI en
treinta segundos). En produccion se usa ``deploy`` con work pool, y ``deploy()``
**exige** ``work_pool_name``: sin work pool no hay quien ejecute. Los **agents
fueron eliminados** en Prefect 3 (``prefect agent start`` no existe); el modelo
es workers + work pools.

El trigger del reentrenamiento es **por llegada de datos** (hash del dataset,
ver ``flows/training.py``), aproximado con un schedule mensual. Un cron de
minutos reentrenando el modelo completo es un anti-patron: no aporta senal,
cuesta y ensucia el registry. Ver ``docs/politica-de-reentrenamiento.md``.
"""

from __future__ import annotations

import argparse
from typing import Any, Final

from prefect import serve
from prefect.schedules import Cron, Schedule

from BeijingAir.config import (
    FILAS_POR_PARTICION,
    PREFECT_SCHEDULE_CRON,
    PREFECT_TIMEZONE,
)
from BeijingAir.flows.training import flujo_entrenamiento

#: Nombre convencional del work pool del proyecto.
WORK_POOL: Final[str] = "beijing-air-pool"


def _schedule(cron: str) -> Schedule:
    """Construye el schedule con ``prefect.schedules.Cron`` y su zona.

    Se pasa siempre dentro de ``schedules=[...]`` (plural). Sin timezone un cron
    se interpreta en UTC y el "3 a.m." ocurre a otra hora local.
    """
    return Cron(cron, timezone=PREFECT_TIMEZONE)


def servir() -> None:
    """Sirve el flow desde un solo proceso (modo clase).

    El proceso queda vivo y hace polling de las corridas programadas. Ctrl+C lo
    detiene y los deployments desaparecen: ``serve`` no persiste infraestructura.
    """
    # `to_deployment` devuelve una union (soporta flows async); se anota Any.
    entrenamiento: Any = flujo_entrenamiento.to_deployment(
        name="entrenamiento-mensual",
        schedules=[_schedule(PREFECT_SCHEDULE_CRON)],
        tags=["s04", "entrenamiento"],
        description="Reentrena y registra el candidato. No promueve.",
        parameters={"filas": FILAS_POR_PARTICION, "n_estimators": 300, "forzar": False},
    )
    serve(entrenamiento)


def desplegar(*, work_pool_name: str = WORK_POOL, imagen: str | None = None) -> None:
    """Crea un deployment persistente contra un work pool.

    Requiere el work pool creado y un worker corriendo:

    ```bash
    prefect work-pool create beijing-air-pool --type process
    prefect worker start --pool beijing-air-pool
    ```

    Args:
        work_pool_name: obligatorio para ``deploy()`` en Prefect 3.
        imagen: imagen a construir/usar. Solo aplica a work pools que ejecutan
            contenedores; con un pool de tipo ``process`` se deja en None.
    """
    flujo_entrenamiento.deploy(
        name="entrenamiento-mensual",
        work_pool_name=work_pool_name,
        image=imagen,
        # Sin imagen no hay nada que construir ni publicar. Un ``build=True``
        # por defecto contra un pool ``process`` falla de forma confusa.
        build=imagen is not None,
        push=imagen is not None,
        schedules=[_schedule(PREFECT_SCHEDULE_CRON)],
        tags=["s04", "entrenamiento"],
        description="Reentrena y registra el candidato. No promueve.",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m BeijingAir.flows.deploy serve|deploy``."""
    parser = argparse.ArgumentParser(description="Deployments del flow de entrenamiento.")
    parser.add_argument(
        "modo", choices=["serve", "deploy"], help="serve = clase; deploy = work pool"
    )
    parser.add_argument("--work-pool", default=WORK_POOL, help="Work pool (solo para deploy)")
    parser.add_argument("--imagen", default=None, help="Imagen del contenedor (solo para deploy)")
    args = parser.parse_args(argv)

    if args.modo == "serve":
        servir()
    else:
        desplegar(work_pool_name=args.work_pool, imagen=args.imagen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
