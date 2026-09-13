"""Pruebas de la capa que habla con MLflow (``models/promote.py``).
 
La politica pura se prueba en ``test_evaluate.py``. Aqui se prueban los efectos
secundarios, que es donde estan los errores caros: que el alias solo se mueva
cuando la decision aprueba, que los tags queden escritos tambien al rechazar,
que ``--dry-run`` no escriba nada, y que un fallo de MLflow no se confunda con
"no hay champion con quien comparar".
 
MLflow se sustituye por dobles: no hace falta un registry para verificar que el
gate escribe lo que dice escribir.
"""
 
from __future__ import annotations
 
from collections.abc import Callable

import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
 
from BeijingAir.features import contract as fc
from BeijingAir.models import promote
 
NOMBRE = "beijing-air-pm25"
VERSION_CANDIDATO = "7"
VERSION_CHAMPION = "1"
 
 
class _ModeloFalso:
    """Modelo cuyo error es un numero elegido a mano.
 
    Predice el target mas un desfase constante, asi que su MAE global y el de
    cada subgrupo valen exactamente ``desfase``. Eso permite escribir tests
    donde la mejora (o la regresion) frente al champion es una cifra conocida,
    sin entrenar nada y sin depender de datos reales.
    """
 
    def __init__(self, holdout: pd.DataFrame, desfase: float) -> None:
        self._predicciones = holdout[fc.TARGET].to_numpy(dtype=float) + desfase
 
    def predict(self, _entrada: pd.DataFrame) -> np.ndarray:
        return self._predicciones
 
 
class _VersionFalsa:
    def __init__(self, version: str) -> None:
        self.version = version
 
 
class _ClienteFalso:
    """MlflowClient minimo que apunta todo lo que se le pide escribir."""
 
    def __init__(
        self,
        *,
        hay_champion: bool,
        error_champion: MlflowException | None = None,
    ) -> None:
        self.hay_champion = hay_champion
        self.error_champion = error_champion
        self.tags: dict[str, str] = {}
        self.alias_asignado: tuple[str, str, str] | None = None
 
    def get_model_version_by_alias(self, _nombre: str, alias: str) -> _VersionFalsa:
        if alias != promote.MODELO_ALIAS:
            return _VersionFalsa(VERSION_CANDIDATO)
        if self.error_champion is not None:
            raise self.error_champion
        if not self.hay_champion:
            raise MlflowException("alias champion ausente", error_code=RESOURCE_DOES_NOT_EXIST)
        return _VersionFalsa(VERSION_CHAMPION)
 
    def get_model_version(self, _nombre: str, version: str) -> _VersionFalsa:
        return _VersionFalsa(version)
 
    def set_model_version_tag(self, _n: str, _v: str, clave: str, valor: str) -> None:
        self.tags[clave] = valor
 
    def set_registered_model_alias(self, nombre: str, alias: str, version: str) -> None:
        self.alias_asignado = (nombre, alias, version)
 
 
@pytest.fixture
def montar(
    monkeypatch: pytest.MonkeyPatch, holdout_valido: pd.DataFrame
) -> Callable[..., _ClienteFalso]:
    """Sustituye MLflow y el holdout por dobles; devuelve el cliente falso."""
 
    def _montar(
        *,
        mae_candidato: float,
        mae_champion: float | None,
        error_champion: MlflowException | None = None,
    ) -> _ClienteFalso:
        cliente = _ClienteFalso(
            hay_champion=mae_champion is not None,
            error_champion=error_champion,
        )
        modelos = {promote.MODELO_ALIAS_CANDIDATO: _ModeloFalso(holdout_valido, mae_candidato)}
        if mae_champion is not None:
            modelos[promote.MODELO_ALIAS] = _ModeloFalso(holdout_valido, mae_champion)
 
        monkeypatch.setattr(promote, "MlflowClient", lambda *_a, **_k: cliente)
        monkeypatch.setattr(promote.mlflow, "set_tracking_uri", lambda *_a: None)
        monkeypatch.setattr(
            mlflow.pyfunc, "load_model", lambda uri: modelos[uri.rsplit("@", 1)[-1]]
        )
        monkeypatch.setattr(promote, "_cargar_holdout", lambda: holdout_valido)
        return cliente
 
    return _montar
 
 
def test_promueve_cuando_el_candidato_mejora_al_champion(
    montar: Callable[..., _ClienteFalso],
) -> None:
    """El camino feliz con champion presente: es el unico que mueve el alias.
 
    Sin este test la suite quedaria verde aunque el gate rechazara el 100% de
    los candidatos, que es justo el modo de fallo silencioso de un gate.
    """
    cliente = montar(mae_candidato=5.0, mae_champion=10.0)
 
    resultado = promote.evaluar_candidato(nombre_modelo=NOMBRE)
 
    assert resultado.decision.promover
    assert resultado.promovido
    assert cliente.alias_asignado == (NOMBRE, promote.MODELO_ALIAS, VERSION_CANDIDATO)
    assert cliente.tags["validation_status"] == "passed"
 
 
def test_un_rechazo_no_toca_el_alias_pero_deja_evidencia(
    montar: Callable[..., _ClienteFalso],
) -> None:
    """Rechazar tambien es un resultado que hay que poder auditar despues."""
    cliente = montar(mae_candidato=12.0, mae_champion=10.0)
 
    resultado = promote.evaluar_candidato(nombre_modelo=NOMBRE)
 
    assert not resultado.promovido
    assert cliente.alias_asignado is None
    assert cliente.tags["validation_status"] == "failed"
    assert cliente.tags["gate_motivo"]
 
 
def test_dry_run_evalua_sin_escribir_nada(montar: Callable[..., _ClienteFalso]) -> None:
    """El ensayo informa la decision sin tocar alias ni tags."""
    cliente = montar(mae_candidato=5.0, mae_champion=10.0)
 
    resultado = promote.evaluar_candidato(nombre_modelo=NOMBRE, dry_run=True)
 
    assert resultado.decision.promover
    assert not resultado.promovido
    assert cliente.alias_asignado is None
    assert cliente.tags == {}
 
 
def test_sin_champion_previo_se_promueve_el_primero(
    montar: Callable[..., _ClienteFalso],
) -> None:
    """Un alias champion inexistente es "no hay con quien comparar", no un error."""
    cliente = montar(mae_candidato=5.0, mae_champion=None)
 
    resultado = promote.evaluar_candidato(nombre_modelo=NOMBRE)
 
    assert resultado.version_champion is None
    assert resultado.promovido
    assert cliente.alias_asignado == (NOMBRE, promote.MODELO_ALIAS, VERSION_CANDIDATO)
 
 
def test_un_fallo_de_mlflow_no_se_confunde_con_ausencia_de_champion(
    montar: Callable[..., _ClienteFalso],
) -> None:
    """Si el champion existe pero no se puede cargar, el gate se detiene.
 
    Traducir ese fallo a "no hay champion" haria que el criterio de mejora se
    aprobara solo y que el candidato pisara a un champion real justo cuando el
    registry esta en problemas. El gate tiene que fallar cerrado.
    """
    cliente = montar(
        mae_candidato=5.0,
        mae_champion=10.0,
        error_champion=MlflowException("el registry no responde"),
    )
 
    with pytest.raises(MlflowException):
        promote.evaluar_candidato(nombre_modelo=NOMBRE)
 
    assert cliente.alias_asignado is None
    assert cliente.tags == {}
 