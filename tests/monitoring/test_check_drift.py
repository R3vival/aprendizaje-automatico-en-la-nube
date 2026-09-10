"""Pruebas de las funciones puras de deteccion de drift (sin red ni MLflow)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from BeijingAir.monitoring.check_drift import (
    EXCLUIDAS_POR_CONSTRUCCION,
    ResultadoDrift,
    drift_categorico,
    drift_numerico,
    evaluar_drift,
    particiones_por_etiqueta,
)


def test_drift_numerico_no_detecta_distribuciones_identicas() -> None:
    rng = np.random.default_rng(0)
    referencia = pd.Series(rng.normal(100, 5, 1000), name="TEMP")
    produccion = pd.Series(rng.normal(100, 5, 1000), name="TEMP")
    resultado = drift_numerico(referencia, produccion)
    assert not resultado.hay_drift


def test_drift_numerico_detecta_desplazamiento_grande() -> None:
    rng = np.random.default_rng(1)
    referencia = pd.Series(rng.normal(100, 5, 1000), name="TEMP")
    produccion = pd.Series(rng.normal(150, 5, 1000), name="TEMP")
    resultado = drift_numerico(referencia, produccion)
    assert resultado.hay_drift
    assert resultado.efecto > 0.5


def test_drift_numerico_devuelve_sin_drift_con_datos_insuficientes() -> None:
    referencia = pd.Series([10.0], name="TEMP")
    produccion = pd.Series([20.0], name="TEMP")
    resultado = drift_numerico(referencia, produccion)
    assert not resultado.hay_drift


def test_drift_categorico_detecta_categoria_nueva() -> None:
    referencia = pd.Series(["N"] * 500, name="wd")
    produccion = pd.Series(["N"] * 400 + ["NE"] * 100, name="wd")  # NE es nueva
    resultado = drift_categorico(referencia, produccion)
    assert resultado.hay_drift


def test_drift_categorico_no_detecta_distribuciones_identicas() -> None:
    referencia = pd.Series(["N", "N", "S", "S"], name="wd")
    produccion = pd.Series(["N", "N", "S", "S"], name="wd")
    resultado = drift_categorico(referencia, produccion)
    assert not resultado.hay_drift


def test_evaluar_drift_computa_la_fraccion_y_el_veredicto() -> None:
    rng = np.random.default_rng(3)
    referencia = pd.DataFrame({"TEMP": rng.normal(100, 5, 500), "wd": ["N"] * 500})
    produccion = pd.DataFrame({"TEMP": rng.normal(150, 5, 500), "wd": ["S"] * 500})
    resultado = evaluar_drift(
        referencia,
        produccion,
        numericas=["TEMP"],
        categoricas=["wd"],
        umbral=0.0,
    )
    assert isinstance(resultado, ResultadoDrift)
    assert resultado.fraccion_con_drift == 1.0
    assert resultado.hay_drift
    assert set(resultado.columnas_con_drift) == {"TEMP", "wd"}


def test_evaluar_drift_ignora_columnas_ausentes() -> None:
    referencia = pd.DataFrame({"TEMP": [1.0, 2.0]})
    produccion = pd.DataFrame({"TEMP": [1.0, 2.0]})
    resultado = evaluar_drift(
        referencia,
        produccion,
        numericas=["TEMP", "PRES"],
        categoricas=[],
    )
    assert resultado.fraccion_con_drift == 0.0
    assert not resultado.hay_drift


def test_markdown_incluye_el_veredicto() -> None:
    from BeijingAir.monitoring.check_drift import ResultadoColumna

    resultado = ResultadoDrift(umbral=0.50)
    resultado.columnas.append(
        ResultadoColumna("TEMP", "ks", p_valor=0.0, efecto=0.9, hay_drift=True)
    )
    assert "ACCIONAR" in resultado.a_markdown()


def test_las_etiquetas_declaradas_resuelven_a_particiones() -> None:
    """Las cuatro particiones de config.py se pueden nombrar desde la CLI."""
    for etiqueta in ("train", "valid", "test", "produccion"):
        assert particiones_por_etiqueta(etiqueta)


def test_una_etiqueta_inventada_falla_diciendo_las_validas() -> None:
    """El error tiene que decir que escribir, no solo que algo salio mal."""
    with pytest.raises(KeyError, match="Validas"):
        particiones_por_etiqueta("el-mes-pasado")


def test_comparar_una_particion_consigo_misma_no_da_drift() -> None:
    """Invariante del detector: datos identicos no pueden producir una alerta.

    Es lo que respalda el ``exit 0`` que se demuestra con
    ``--referencia train --produccion train``.
    """
    rng = np.random.default_rng(11)
    datos = pd.DataFrame({"TEMP": rng.normal(15, 8, 800), "wd": list("NSEW") * 200})

    resultado = evaluar_drift(datos, datos.copy(), numericas=["TEMP"], categoricas=["wd"])

    assert resultado.fraccion_con_drift == 0.0
    assert not resultado.hay_drift


def test_las_columnas_de_calendario_se_excluyen_por_construccion() -> None:
    """`mes` y `temporada` driftean siempre entre ventanas temporales distintas.

    Medirlas seria medir que el calendario avanzo. `hora`, `dia_semana` y
    `station` NO se excluyen: dan efecto ~0 y sirven de control de la particion.
    """
    assert {"mes", "temporada"} == EXCLUIDAS_POR_CONSTRUCCION
    assert "hora" not in EXCLUIDAS_POR_CONSTRUCCION
    assert "station" not in EXCLUIDAS_POR_CONSTRUCCION
