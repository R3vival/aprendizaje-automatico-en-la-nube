#!/usr/bin/env python
"""Genera docs/model-card.md desde el modelo registrado en MLflow.

Problema que resuelve
---------------------
Una model card escrita a mano es documentacion, y la documentacion escrita a
mano miente en cuanto el sistema cambia. Nadie actualiza el MAE del markdown
cuando promueve la version 8. A los tres meses el documento dice cosas falsas
con total autoridad, que es peor que no tener documento.

Esta model card se **genera**. Los numeros salen del Model Registry, los hashes
del ``metadata.json`` de la ingesta y el contrato de features del propio codigo
(``resumen_contrato()``). Lo unico escrito a mano es lo que no se puede derivar:
uso previsto, limitaciones y consideraciones eticas, que son juicios humanos.

Modo degradado
--------------
El script funciona sin MLflow disponible: emite la card con un aviso visible en
lugar de metricas, en vez de fallar o de inventar ceros.

Uso
---
    python scripts/model_card.py
    python scripts/model_card.py --salida docs/model-card.md
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from BeijingAir import config
from BeijingAir.data.contract import resumen_contrato
from BeijingAir.features import contract as fc

logger = logging.getLogger(__name__)

RUTA_SALIDA_DEFECTO = config.PROJECT_ROOT / "docs" / "model-card.md"
RUTA_METADATA = config.RAW_DIR / "metadata.json"

AVISO_SIN_MLFLOW = (
    "> **AVISO — modo degradado.** No se pudo consultar el Model Registry en "
    "`{uri}`, asi que las secciones de identificacion y metricas estan "
    "incompletas. Esta card documenta el **contrato** del sistema (features, "
    "particiones, limitaciones), no una version concreta del modelo. Levanta "
    "MLflow (`make mlflow`) y vuelve a generarla antes de usarla como evidencia."
)


def leer_metadata_particiones() -> dict[str, dict[str, Any]]:
    """Lee ``data/raw/metadata.json``, que la ingesta escribe con el SHA-256."""
    if not RUTA_METADATA.exists():
        logger.warning("No existe %s: corre `make data` para generarlo", RUTA_METADATA)
        return {}
    try:
        return json.loads(RUTA_METADATA.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("%s no es JSON valido", RUTA_METADATA)
        return {}


def _formatear_epoch(milisegundos: int | None) -> str:
    if not milisegundos:
        return "desconocida"
    return datetime.fromtimestamp(milisegundos / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def _fallar_rapido() -> None:
    """Baja el timeout y los reintentos HTTP de MLflow.

    El cliente de MLflow reintenta 7 veces con espera creciente y 120 s de
    timeout por defecto. Si el registry esta caido, eso convierte el modo
    degradado en un proceso que cuelga varios minutos. Con 3 s y 1 reintento el
    fallo se reporta en segundos, que es lo que hace util la card sin server.
    """
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")


def recolectar_version(nombre: str, alias: str) -> tuple[dict[str, Any] | None, dict[str, float]]:
    """Consulta el registry y devuelve (info de la version, metricas del run).

    Nunca lanza: si MLflow no responde, devuelve ``(None, {})`` y el llamador
    entra en modo degradado.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        _fallar_rapido()
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        cliente = MlflowClient()
        mv = cliente.get_model_version_by_alias(nombre, alias)
        corrida = cliente.get_run(mv.run_id)
        metricas = {k: float(v) for k, v in corrida.data.metrics.items()}
        info: dict[str, Any] = {
            "nombre": nombre,
            "alias": alias,
            "version": str(mv.version),
            "run_id": mv.run_id or "desconocido",
            "creada": _formatear_epoch(getattr(mv, "creation_timestamp", None)),
            "tags": dict(getattr(mv, "tags", {}) or {}),
            "uri": f"models:/{nombre}@{alias}",
        }
        return info, metricas
    except Exception as exc:
        logger.warning("MLflow no disponible (%s): %s", type(exc).__name__, exc)
        return None, {}


def _tabla(filas: Sequence[tuple[str, ...]], encabezados: tuple[str, ...]) -> str:
    if not filas:
        return "_Sin datos disponibles._\n"
    lineas = [
        "| " + " | ".join(encabezados) + " |",
        "|" + "|".join(["---"] * len(encabezados)) + "|",
    ]
    lineas += ["| " + " | ".join(f) + " |" for f in filas]
    return "\n".join(lineas) + "\n"


def _seccion_identificacion(info: dict[str, Any] | None) -> str:
    if info is None:
        return (
            "| campo | valor |\n|---|---|\n"
            f"| Nombre registrado | `{config.MODELO_REGISTRADO}` |\n"
            "| Version | **no determinada** (registry no disponible) |\n"
            f"| Referencia de produccion | `{config.MODELO_URI}` |\n"
        )
    filas = [
        ("Nombre registrado", f"`{info['nombre']}`"),
        ("Version", f"**{info['version']}**"),
        ("Alias", f"`@{info['alias']}`"),
        ("URI de referencia", f"`{info['uri']}`"),
        ("Run de MLflow", f"`{info['run_id']}`"),
        ("Registrada el", info["creada"]),
        (f"`{config.TAG_VALIDACION}`", info["tags"].get(config.TAG_VALIDACION, "_ausente_")),
    ]
    return _tabla(filas, ("campo", "valor"))


def _seccion_datos(metadata: Mapping[str, dict[str, Any]]) -> str:
    grupos = [
        ("Entrenamiento", list(config.PARTICIONES_TRAIN)),
        ("Validacion (seleccion de hiperparametros)", [config.PARTICION_VALID]),
        ("Holdout (juez del gate de promocion)", [config.PARTICION_TEST]),
        ("Produccion simulada (monitoreo)", list(config.PARTICIONES_PRODUCCION)),
    ]
    filas: list[tuple[str, ...]] = []
    for rol, particiones in grupos:
        for particion in particiones:
            registro = next((v for v in metadata.values() if isinstance(v, dict)), {})
            sha = str(registro.get("sha256", ""))
            filas.append(
                (
                    rol,
                    particion.etiqueta,
                    f"`{particion.desde}` a `{particion.hasta}`",
                    f"`{sha[:16]}...`" if sha else "_no descargada_",
                )
            )
    return _tabla(filas, ("rol", "particion", "rango", "SHA-256 (16 primeros)"))


def _seccion_contrato() -> str:
    contrato = resumen_contrato()
    filas = [(clave, ", ".join(f"`{c}`" for c in columnas)) for clave, columnas in contrato.items()]
    return _tabla(filas, ("grupo", "columnas"))


def _separar_metricas(metricas: Mapping[str, float]) -> dict[str, float]:
    """Queda con las metricas globales de train/valid/test; excluye per-estacion."""
    excluidas = ("mae_por_estacion",)
    return {k: v for k, v in metricas.items() if k not in excluidas}


def _seccion_metricas(metricas: Mapping[str, float]) -> str:
    if not metricas:
        return (
            "_No disponibles: el registry no respondio. Las metricas de esta "
            "seccion se leen del run que produjo la version registrada._\n"
        )
    globales = _separar_metricas(metricas)
    return _tabla(
        [(f"`{k}`", f"{v:.4f}") for k, v in sorted(globales.items())],
        ("metrica", "valor"),
    )


def construir_card(
    info: dict[str, Any] | None,
    metricas: Mapping[str, float],
    metadata: Mapping[str, dict[str, Any]],
) -> str:
    """Arma el Markdown completo de la model card."""
    generada = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    aviso = "" if info else AVISO_SIN_MLFLOW.format(uri=config.MLFLOW_TRACKING_URI) + "\n\n"
    valid = config.PARTICION_VALID.etiqueta
    test = config.PARTICION_TEST.etiqueta
    filas_por_particion = f"{config.FILAS_POR_PARTICION:,}".replace(",", " ")

    return f"""# Model Card — {config.MODELO_REGISTRADO}

<!-- ARCHIVO GENERADO. No lo edites a mano: `make model-card` lo sobrescribe.
     Si necesitas cambiar el texto, edita scripts/model_card.py. -->

_Generada automaticamente el {generada} por `scripts/model_card.py`._

{aviso}## 1. Identificacion y version

{_seccion_identificacion(info)}
El modelo se referencia siempre por **alias**, nunca por numero de version ni
por ruta de archivo. Mover el alias es la operacion de despliegue y, en sentido
inverso, la de rollback.

## 2. Uso previsto

Estimar la **concentracion horaria de PM2.5** (ug/m3) en las 12 estaciones de
monitoreo de Beijing, a partir de la meteorologia y los demas contaminantes
registrados en la misma hora.

Casos de uso contemplados:

- Vigilancia de la calidad del aire a nivel horario por estacion.
- Estudios de correlacion entre contaminantes y meteorologia.
- Caso de estudio del curso de MLOps: es el objeto sobre el que se practica
  tracking, registry, promocion, despliegue y monitoreo.

## 3. Uso NO previsto

Esta seccion es la mas importante de la card y la que mas se omite. Un modelo
usado fuera de su contexto de entrenamiento falla de forma silenciosa.

- **No es un modelo de pronostico.** Predice el valor *concurrente* a partir de
  las magnitudes de esa misma hora; no proyecta el futuro.
- **No sirve para otra ciudad ni para otra red de sensores.** Fue entrenado con
  12 estaciones de Beijing en un periodo de 2013-2017; otra ciudad o un sensor
  distinto tienen distribuciones y calibraciones distintas.
- **No debe usarse para decisiones sobre personas.** No fue disenado, medido ni
  auditado para asignar responsabilidades, ventilar culpas ni tomar medidas
  administrativas sobre individuos u organizaciones.
- **No es un sistema de alerta en tiempo real.** No recibe lecturas en vivo ni
  estados de falla del sensor; ante un episodio de contaminacion atipico su
  error crece y el modelo no lo sabe.

## 4. Datos de entrenamiento

Fuente: **{config.FUENTE}** ({config.URL_DATASET}), licencia {config.LICENCIA}
({config.LICENCIA_URL}). Las particiones son **fijas y del pasado** por decision
de diseno: un pipeline que calcula el periodo con `datetime.now()` se rompe y
rompe la comparabilidad entre corridas.

{_seccion_datos(metadata)}
Muestreo determinista de {filas_por_particion} filas por particion con semilla
`{config.SEMILLA}`. La division es **temporal**, no aleatoria: se entrena con
periodos anteriores y se evalua con periodos posteriores, porque en produccion
el modelo siempre predice sobre el futuro.

## 5. Contrato de features

{_seccion_contrato()}
El consumidor envia las columnas crudas (`station`, `wd`, contaminantes y
meteorologia); las features de calendario (`hora`, `dia_semana`, `mes`,
`temporada`) se derivan dentro del pipeline con el mismo codigo que el
entrenamiento. Eso es lo que evita el train/serving skew.

Target: `{config.MODELO_REGISTRADO}` predice `{fc.TARGET}` en ug/m3
(regresion).

## 6. Metricas

Leidas del run de MLflow que produjo esta version. Los prefijos indican sobre
que particion se midio cada una (`*_valid` sobre {valid}, `*_test` sobre {test}).
El gate de promocion decide con `mae_test` y `r2_test`.

{_seccion_metricas(metricas)}
Las metricas por estacion se guardan como artifact del run en
`evaluacion/metricas_por_estacion.json`.

## 7. Limitaciones conocidas

- **Valores imputados.** Cuando un sensor calla, la magnitud se imputa con la
  mediana y se deja el indicador `<col>_era_nulo`. En horas con muchos sensores
  caidos, la entrada al modelo no describe el aire real.
- **Estaciones no vistas.** Si una estacion nueva no estaba en el entrenamiento,
  `OneHotEncoder(handle_unknown="ignore")` la codifica como cero y la prediccion
  se apoya solo en las demas features; el modelo no lo senala.
- **Deriva temporal y estructural.** Entrenado con datos de 2013-2017. Cambios
  de regulacion o de medicion (p. ej. la tendencia a la baja de SO2) degradan el
  desempeno con el tiempo. Es el problema que se monitorea en el drift.
- **El PM2.5 no se imputa.** Las horas con target nulo se descartan: el modelo
  no puede aprender a "inventar" un sensor que no existio.
- **Representatividad.** Una estacion con mas lecturas aporta mas a la metrica
  agregada; el error por estacion no es uniforme (ver `peor_mae_estacion`).

## 8. Consideraciones eticas

- **Sesgo geografico.** El error no se distribuye igual entre las 12 estaciones.
  Las estaciones con menos lecturas o con episodios extremos tienen peor
  estimacion. Cualquier uso que asocie la prediccion a un barrio debe tenerlo en
  cuenta.
- **Datos publicos, sin datos personales.** El dataset es agregado por estacion
  y hora; no contiene identificadores de personas.
- **Transparencia.** El artefacto es trazable de punta a punta: SHA-256 de los
  datos, run de MLflow, version del registry y tag de validacion.
- **Uso no regulatorio.** La prediccion no sustituye a un instrumento de
  medicion de referencia ni debe usarse para certificar la calidad del aire.

## 9. Clasificacion tentativa bajo el EU AI Act

**Clasificacion propuesta: riesgo minimo** (fuera de las categorias de riesgo
alto del Anexo III). Estimar una concentracion de contaminante no decide sobre
acceso a empleo, educacion, credito, servicios esenciales, migracion ni
justicia; no es identificacion biometrica ni infraestructura critica.

> Aviso: esta clasificacion es un **ejercicio didactico** del curso, no
> asesoramiento legal. Una clasificacion vinculante requiere analisis juridico
> del caso de uso concreto y del rol de quien lo opera.
"""


def generar(
    *,
    nombre_modelo: str = config.MODELO_REGISTRADO,
    alias: str = config.MODELO_ALIAS,
    salida: Path = RUTA_SALIDA_DEFECTO,
) -> tuple[Path, dict[str, Any] | None, dict[str, dict[str, Any]]]:
    """Escribe la model card y devuelve (ruta, info de version, metadata)."""
    info, metricas = recolectar_version(nombre_modelo, alias)
    metadata = leer_metadata_particiones()
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(construir_card(info, metricas, metadata), encoding="utf-8")
    logger.info("Model card escrita en %s", salida)
    return salida, info, metadata


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--alias", default=config.MODELO_ALIAS, show_default=True)
@click.option(
    "--salida",
    type=click.Path(path_type=Path),
    default=RUTA_SALIDA_DEFECTO,
    show_default=True,
    help="Ruta del Markdown a generar.",
)
@click.option("--verbose", "-v", is_flag=True, help="Log en nivel INFO.")
def main(alias: str, salida: Path, verbose: bool) -> None:
    """Genera la model card del modelo registrado."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)-8s %(name)s: %(message)s",
    )
    ruta, info, _ = generar(alias=alias, salida=salida)
    try:
        destino = ruta.relative_to(config.PROJECT_ROOT)
    except ValueError:
        destino = ruta
    if info is None:
        click.secho(
            f"Generada {destino} en MODO DEGRADADO: "
            f"el registry en {config.MLFLOW_TRACKING_URI} no respondio. Levanta "
            "MLflow y vuelve a generarla antes de usarla como evidencia.",
            fg="yellow",
        )
    else:
        click.secho(
            f"Generada {destino} para {info['nombre']} v{info['version']} (@{alias}).",
            fg="green",
        )


if __name__ == "__main__":
    main()
