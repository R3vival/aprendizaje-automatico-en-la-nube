"""Deteccion de drift como check ejecutable, no como PDF que nadie abre.

Que se puede medir y que no
---------------------------
=========================  ==================================================
Fenomeno                   Observable sin labels?
=========================  ==================================================
data drift                 Si. Cambia P(X)
prediction drift           Si. Cambia P(y_hat)
concept drift              No directamente. Cambia P(y|X)
degradacion de performance Solo cuando llegan los labels (*label lag*)
=========================  ==================================================

Por eso el monitoreo de un sistema de ML no es "medir accuracy en produccion":
la mayor parte del tiempo NO tienes el label. Lo que si tienes es la
distribucion de entrada y la de salida.

La trampa de los p-valores
--------------------------
Con n grande, TODO sale significativo. Con 500 000 filas, un cambio de la media
en un 0.1% da p < 1e-10 y no le importa a nadie. Por eso este modulo reporta
p-valor **y** tamano del efecto, y la decision usa la **fraccion de columnas con
drift** contra un umbral que el estudiante justifica, no un `p < 0.05` suelto.

Por que scipy y no solo Evidently
---------------------------------
La politica se implementa con ``scipy.stats`` en funciones puras: se testea sin
red, sin HTML y sin depender de la API de una libreria que cambia. Evidently se
usa para el **reporte** (que es donde aporta: presets, HTML navegable,
comparacion visual). Separar politica de presentacion es lo que permite que el
check corra en CI en dos segundos.

Uso:

    python -m BeijingAir.monitoring.check_drift                    # exit 1 si hay drift
    python -m BeijingAir.monitoring.check_drift --referencia train --produccion train

La segunda forma compara una particion consigo misma y por definicion no puede
dar drift: es la manera de demostrar que el ``exit 0`` funciona, y de verificar
que el detector no inventa alertas sobre datos identicos.

"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from BeijingAir.config import (
    ALFA_DRIFT,
    PARTICION_TEST,
    PARTICION_VALID,
    PARTICIONES_PRODUCCION,
    PARTICIONES_TRAIN,
    REPORTS_DIR,
    UMBRAL_DRIFT_COLUMNAS,
    Particion,
)
from BeijingAir.features import contract as fc

logger = logging.getLogger(__name__)

EXITO_SIN_DRIFT = 0
EXITO_CON_DRIFT = 1
ERROR_INFRA = 2

#: Tamano de efecto minimo para considerar el drift accionable, medido como
#: estadistico KS (distancia maxima entre las dos acumuladas, 0 a 1). Un KS de
#: 0.05 es un cambio que casi nunca justifica reentrenar; uno de 0.30 si.
EFECTO_MINIMO_KS: float = 0.10


@dataclass
class ResultadoColumna:
    """Drift de una columna: el test, el p-valor y el tamano del efecto."""

    columna: str
    test: str
    p_valor: float
    efecto: float
    hay_drift: bool


@dataclass
class ResultadoDrift:
    """Veredicto global, con el detalle por columna para poder auditarlo."""

    columnas: list[ResultadoColumna] = field(default_factory=list)
    umbral: float = UMBRAL_DRIFT_COLUMNAS

    @property
    def fraccion_con_drift(self) -> float:
        if not self.columnas:
            return 0.0
        return sum(c.hay_drift for c in self.columnas) / len(self.columnas)

    @property
    def hay_drift(self) -> bool:
        return self.fraccion_con_drift > self.umbral

    @property
    def columnas_con_drift(self) -> list[str]:
        return [c.columna for c in self.columnas if c.hay_drift]

    def a_markdown(self) -> str:
        """Tabla para pegar en el PR o en un artifact del orquestador."""
        lineas = [
            "| columna | test | p-valor | efecto | drift |",
            "|---|---|---:|---:|---|",
        ]
        for c in sorted(self.columnas, key=lambda x: -x.efecto):
            lineas.append(
                f"| {c.columna} | {c.test} | {c.p_valor:.2e} | {c.efecto:.3f} | "
                f"{'SI' if c.hay_drift else 'no'} |"
            )
        lineas.append("")
        lineas.append(
            f"**{self.fraccion_con_drift:.0%}** de las columnas con drift "
            f"(umbral: {self.umbral:.0%}) -> "
            f"{'ACCIONAR' if self.hay_drift else 'sin accion'}"
        )
        return "\n".join(lineas)


def drift_numerico(
    referencia: pd.Series, produccion: pd.Series, *, alfa: float = ALFA_DRIFT
) -> ResultadoColumna:
    """Kolmogorov-Smirnov de dos muestras sobre una columna numerica.

    KS mide la distancia maxima entre las dos funciones de distribucion
    acumuladas. El estadistico ES el tamano del efecto (0 = identicas,
    1 = disjuntas), lo que lo hace comodo: no hay que calcular el efecto aparte.
    """
    a = pd.to_numeric(referencia, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(produccion, errors="coerce").dropna().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return ResultadoColumna(str(referencia.name), "ks", 1.0, 0.0, False)
    resultado = stats.ks_2samp(a, b)
    efecto = float(resultado.statistic)
    return ResultadoColumna(
        columna=str(referencia.name),
        test="ks",
        p_valor=float(resultado.pvalue),
        efecto=efecto,
        # Las DOS condiciones: significativo Y con efecto que importa.
        hay_drift=bool(resultado.pvalue < alfa and efecto >= EFECTO_MINIMO_KS),
    )


def drift_categorico(
    referencia: pd.Series, produccion: pd.Series, *, alfa: float = ALFA_DRIFT
) -> ResultadoColumna:
    """Chi-cuadrado sobre la tabla de frecuencias, con V de Cramer como efecto.

    Las categorias que aparecen en una sola de las dos muestras se conservan con
    frecuencia 0: una categoria NUEVA en produccion es justo el tipo de cambio
    que hay que detectar, y descartarla lo esconderia.
    """
    nombre = str(referencia.name)
    ref = referencia.astype(str).value_counts()
    pro = produccion.astype(str).value_counts()
    categorias = sorted(set(ref.index) | set(pro.index))
    tabla = np.array(
        [[ref.get(c, 0) for c in categorias], [pro.get(c, 0) for c in categorias]],
        dtype=float,
    )
    # Se descartan las columnas todo-cero (imposibles) para que chi2 no divida
    # por cero.
    tabla = tabla[:, tabla.sum(axis=0) > 0]
    if tabla.shape[1] < 2 or tabla.sum() == 0:
        return ResultadoColumna(nombre, "chi2", 1.0, 0.0, False)

    chi2, p, _, _ = stats.chi2_contingency(tabla)
    n = tabla.sum()
    # V de Cramer: normaliza chi2 a [0, 1] y NO crece con n, que es exactamente
    # lo que le falta al p-valor.
    v_cramer = float(np.sqrt(chi2 / (n * (min(tabla.shape) - 1))))
    return ResultadoColumna(
        columna=nombre,
        test="chi2",
        p_valor=float(p),
        efecto=v_cramer,
        hay_drift=bool(p < alfa and v_cramer >= EFECTO_MINIMO_KS),
    )


def evaluar_drift(
    referencia: pd.DataFrame,
    produccion: pd.DataFrame,
    *,
    numericas: list[str] | None = None,
    categoricas: list[str] | None = None,
    umbral: float = UMBRAL_DRIFT_COLUMNAS,
    alfa: float = ALFA_DRIFT,
) -> ResultadoDrift:
    """Compara referencia vs produccion columna por columna. Funcion pura."""
    numericas = numericas if numericas is not None else fc.FEATURES_NUMERICAS
    categoricas = categoricas if categoricas is not None else fc.FEATURES_CATEGORICAS

    resultado = ResultadoDrift(umbral=umbral)
    for col in numericas:
        if col in referencia.columns and col in produccion.columns:
            resultado.columnas.append(drift_numerico(referencia[col], produccion[col], alfa=alfa))
    for col in categoricas:
        if col in referencia.columns and col in produccion.columns:
            resultado.columnas.append(drift_categorico(referencia[col], produccion[col], alfa=alfa))
    return resultado


def reporte_html(
    referencia: pd.DataFrame,
    produccion: pd.DataFrame,
    *,
    nombre: str = "drift-report.html",
) -> str:
    """Genera el reporte navegable con Evidently 0.7.x y devuelve su ruta.

    Se aisla aqui, fuera de la politica, para que la API de una libreria de
    presentacion no pueda romper el check de CI.

    Ojo con el orden de los argumentos: ``report.run(current, reference)``.
    Invertirlos no lanza error, solo produce un reporte que dice lo contrario de
    lo que crees.
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    destino = REPORTS_DIR / nombre

    esquema = DataDefinition(
        numerical_columns=[c for c in fc.FEATURES_NUMERICAS if c in referencia.columns],
        categorical_columns=[c for c in fc.FEATURES_CATEGORICAS if c in referencia.columns],
    )
    ref = Dataset.from_pandas(referencia, data_definition=esquema)
    cur = Dataset.from_pandas(produccion, data_definition=esquema)

    reporte = Report([DataDriftPreset(), DataSummaryPreset()])
    evaluacion = reporte.run(cur, ref)
    evaluacion.save_html(str(destino))
    logger.info("reporte de drift en %s", destino)
    return str(destino)


#: Columnas que se excluyen del check. `mes` y `temporada` driftean por
#: construccion: dos ventanas temporales distintas siempre tienen mezcla
#: distinta de meses, asi que medirlas es medir que el calendario avanzo.
#: `hora`, `dia_semana` y `station` SI se quedan: no driftean por construccion,
#: y que den efecto ~0 verifica que la particion esta bien armada.
EXCLUIDAS_POR_CONSTRUCCION: frozenset[str] = frozenset({"mes", "temporada"})


def particiones_por_etiqueta(etiqueta: str) -> tuple[Particion, ...]:
    """Resuelve el nombre de una particion declarada en ``config.py``.

    Se resuelve por etiqueta y no por fechas en la linea de comandos a
    proposito: un rango escrito a mano en un comando no queda versionado, y el
    dia que alguien reporte un resultado nadie sabria contra que lo midio.

    Raises:
        KeyError: si la etiqueta no existe, listando las validas.
    """
    disponibles: dict[str, tuple[Particion, ...]] = {
        "train": PARTICIONES_TRAIN,
        "valid": (PARTICION_VALID,),
        "test": (PARTICION_TEST,),
        "produccion": PARTICIONES_PRODUCCION,
    }
    if etiqueta not in disponibles:
        raise KeyError(f"Particion '{etiqueta}' desconocida. Validas: {sorted(disponibles)}")
    return disponibles[etiqueta]


def _argumentos(argumentos: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara dos particiones y sale con codigo != 0 si hay drift accionable."
    )
    parser.add_argument(
        "--referencia",
        default="train",
        help="Particion de referencia declarada en config.py (por defecto: train).",
    )
    parser.add_argument(
        "--produccion",
        default="produccion",
        help="Particion a vigilar (por defecto: produccion).",
    )
    parser.add_argument(
        "--sin-reporte",
        action="store_true",
        help="Omite el HTML de Evidently; util en CI, donde solo importa el exit code.",
    )
    return parser.parse_args(argumentos)


def main(argumentos: Sequence[str] | None = None) -> int:
    """Check de CI: exit 1 si el drift supera el umbral.

    Por defecto compara ``train`` contra ``produccion``, las particiones
    declaradas en ``config.py``. Los argumentos permiten comparar cualquier par
    y, en particular, una particion consigo misma para demostrar el ``exit 0``.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    opciones = _argumentos(argumentos)

    from BeijingAir.data.descarga import cargar_crudo, filtrar

    try:
        particiones_ref = particiones_por_etiqueta(opciones.referencia)
        particiones_pro = particiones_por_etiqueta(opciones.produccion)
    except KeyError as error:
        logger.error("%s", error)
        return ERROR_INFRA

    df = fc.construir_features(cargar_crudo())
    numericas = [c for c in fc.FEATURES_NUMERICAS if c not in EXCLUIDAS_POR_CONSTRUCCION]
    categoricas = [c for c in fc.FEATURES_CATEGORICAS if c not in EXCLUIDAS_POR_CONSTRUCCION]

    referencia = pd.concat([filtrar(df, p) for p in particiones_ref], ignore_index=True)
    produccion = pd.concat([filtrar(df, p) for p in particiones_pro], ignore_index=True)

    logger.info(
        "referencia '%s': %d filas | produccion '%s': %d filas",
        opciones.referencia,
        len(referencia),
        opciones.produccion,
        len(produccion),
    )
    if referencia.empty or produccion.empty:
        logger.error("Alguna particion quedo vacia. Revisa los rangos en config.py")
        return ERROR_INFRA

    resultado = evaluar_drift(referencia, produccion, numericas=numericas, categoricas=categoricas)
    print(resultado.a_markdown())

    if not opciones.sin_reporte:
        logger.info("reporte HTML en %s", reporte_html(referencia, produccion))

    return EXITO_CON_DRIFT if resultado.hay_drift else EXITO_SIN_DRIFT


if __name__ == "__main__":
    sys.exit(main())
