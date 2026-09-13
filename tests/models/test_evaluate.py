"""Pruebas de la politica del gate de promocion (funciones puras, sin MLflow).
 
La politica vive en ``models/evaluate.py`` como funciones que reciben metricas y
devuelven un veredicto; por eso se puede testear en milisegundos y sin levantar
infraestructura. La capa que habla con MLflow se prueba en ``test_promote.py``.
"""
 
from __future__ import annotations
 
import pandas as pd
 
from BeijingAir.models import evaluate
 
 
def test_primer_modelo_se_promueve(holdout_valido: pd.DataFrame) -> None:
    """Sin champion previo, un candidato que cumple los limites es el primero."""
    decision = evaluate.decidir_promocion(
        holdout_valido,
        metricas_candidato={"mae": 20.0, "rmse": 30.0, "r2": 0.4},
        subgrupos_candidato={"mae_temporada_invierno": 21.0},
        metricas_champion=None,
        subgrupos_champion=None,
    )
    assert decision.promover
    assert decision.es_primer_modelo
 
 
def test_rechaza_candidato_que_no_mejora_al_champion(holdout_valido: pd.DataFrame) -> None:
    """Si el champion es mejor, el candidato no supera el margen y se rechaza."""
    decision = evaluate.decidir_promocion(
        holdout_valido,
        metricas_candidato={"mae": 20.0, "rmse": 30.0, "r2": 0.4},
        subgrupos_candidato={"mae_temporada_invierno": 21.0},
        metricas_champion={"mae": 15.0, "rmse": 25.0, "r2": 0.6},
        subgrupos_champion={"mae_temporada_invierno": 16.0},
    )
    assert not decision.promover
    assert any(c.nombre == "mejora_global" and not c.aprobado for c in decision.criterios)
 
 
def test_rechaza_mejora_menor_al_margen(holdout_valido: pd.DataFrame) -> None:
    """Mejorar por debajo del margen (1%) es churn: se rechaza a proposito."""
    decision = evaluate.decidir_promocion(
        holdout_valido,
        metricas_candidato={"mae": 19.9, "rmse": 30.0, "r2": 0.4},
        subgrupos_candidato={"mae_temporada_invierno": 21.0},
        metricas_champion={"mae": 20.0, "rmse": 30.0, "r2": 0.4},
        subgrupos_champion={"mae_temporada_invierno": 21.0},
        mejora_minima=0.01,
    )
    # 19.9 no es <= 20*0.99 = 19.8
    assert not decision.promover
    assert decision.motivo == "criterios no superados: mejora_global"
 
 
def test_rechaza_regresion_por_subgrupo(holdout_valido: pd.DataFrame) -> None:
    """El global mejora pero un subgrupo se degrada mas del umbral: se rechaza."""
    decision = evaluate.decidir_promocion(
        holdout_valido,
        metricas_candidato={"mae": 15.0, "rmse": 25.0, "r2": 0.6},
        subgrupos_candidato={
            "mae_temporada_invierno": 30.0,  # empeora 50% sobre 20
            "mae_temporada_verano": 10.0,
        },
        metricas_champion={"mae": 20.0, "rmse": 30.0, "r2": 0.4},
        subgrupos_champion={
            "mae_temporada_invierno": 20.0,
            "mae_temporada_verano": 12.0,
        },
    )
    assert not decision.promover
    assert any(
        c.nombre == "sin_regresion_por_subgrupo" and not c.aprobado for c in decision.criterios
    )
 
 
def test_criterio_contrato_rechaza_holdout_roto(holdout_valido: pd.DataFrame) -> None:
    """Si el holdout viola el contrato, no se miran metricas (corto circuito)."""
    roto = holdout_valido
    roto.loc[0, "PM2.5"] = 5000.0  # fuera de rango
    decision = evaluate.decidir_promocion(
        roto,
        metricas_candidato={"mae": 15.0, "rmse": 25.0, "r2": 0.6},
        subgrupos_candidato={"mae_temporada_invierno": 16.0},
        metricas_champion={"mae": 20.0, "rmse": 30.0, "r2": 0.4},
        subgrupos_champion={"mae_temporada_invierno": 20.0},
    )
    assert not decision.promover
    estados = [c.estado for c in decision.criterios]
    assert estados[0] == "FALLA"
    # Los criterios posteriores se marcan NO EVALUADO, no FALLA.
    assert estados[1:] == ["NO EVALUADO", "NO EVALUADO"]
 
 
def test_metricas_por_subgrupo_separa_temporada_y_hora(holdout_valido: pd.DataFrame) -> None:
    """Los subgrupos de negocio se calculan sin modelo (solo con metricas)."""
    y_true = holdout_valido["PM2.5"].to_numpy(dtype=float)
    y_pred = y_true + 1.0  # error constante
    sub = evaluate.metricas_por_subgrupo(holdout_valido, y_true, y_pred, min_filas=1)
    assert "mae_temporada_invierno" in sub
    assert "mae_temporada_verano" in sub
    assert "mae_hora_madrugada" in sub
    assert abs(sub["mae_temporada_invierno"] - 1.0) < 1e-9
 
