# =============================================================================
# Makefile — Aprendizaje Automatico en la Nube
# =============================================================================
# Interfaz unica del repositorio. El CI usa exactamente estos mismos targets,
# de modo que "pasa en mi maquina" y "pasa en CI" significan lo mismo.
#
#   make            muestra esta ayuda
#   make setup      instala todo y configura los hooks
# =============================================================================

.DEFAULT_GOAL := help
SHELL := /bin/bash
UV := uv
PY := $(UV) run

# PYTHONUTF8: MLflow imprime emojis. Sin esto, Windows cae a cp1252 y revienta
# con UnicodeEncodeError cuando la salida se redirige. `export` es directiva de
# make, no del shell: funciona igual en bash, cmd y PowerShell.
export PYTHONUTF8 = 1

.PHONY: help setup data smoke test test-fast lint format typecheck check validate-data mlflow prefect-server train hpo model-card flow serve-flow deploy-flow work-pool worker batch serve promote promote-check drift up down clean
# =============================================================================
help: ## Muestra los targets disponibles
	@echo ""
	@echo "  Aprendizaje Automatico en la Nube — targets disponibles"
	@echo "  ------------------------------------------------------"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo ""

# =============================================================================
# Entorno
# =============================================================================
setup: ## Instala dependencias y los hooks de pre-commit
	$(UV) sync --group dev
	@# core.hooksPath NO se configura: pre-commit es el unico sistema de hooks.
	@git config --get core.hooksPath >/dev/null 2>&1 \
	  && (echo "Limpiando core.hooksPath heredado..."; git config --unset-all core.hooksPath) \
	  || true
	$(PY) pre-commit install --install-hooks
	$(MAKE) data
	$(MAKE) smoke
	@echo ""
	@echo "Listo."

data: ## Descarga y extrae el dataset Beijing
	$(PY) python -m BeijingAir.data.descarga

smoke: ## Verifica que el entorno esta listo para entrenar y servir
	$(PY) python scripts/smoke_test.py
# =============================================================================
# Calidad — el CI corre exactamente esto
# =============================================================================
test: ## Corre todos los tests
	$(PY) pytest

test-fast: ## Corre solo los tests que no requieren red ni servicios
	$(PY) pytest -m "not slow and not integration"

lint: ## Revisa estilo y errores con ruff
	$(PY) ruff check .

format: ## Formatea el codigo con ruff
	$(PY) ruff format .

typecheck: ## Verifica tipos con mypy
	$(PY) mypy

check: lint typecheck test-fast ## Todo lo que el CI verifica, en local

validate-data: ## Descarga (si falta) y valida las particiones reales
	$(PY) python -m BeijingAir.data.validate

mlflow: ## Inicia el servidor local de tracking en http://127.0.0.1:5001
	$(PY) mlflow server --backend-store-uri sqlite:///mlflow.db \
	  --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5001

train: ## Entrena baseline y bosque, y registra ambos en MLflow
	$(PY) python -m BeijingAir.models.train

hpo: ## Busca hiperparametros del bosque con Optuna (runs anidados)
	$(PY) python -m BeijingAir.models.train --hpo --trials 20

model-card: ## Genera docs/model-card.md desde el modelo registrado
	$(PY) python scripts/model_card.py

# =============================================================================
# Monitoreo
# =============================================================================
drift: ## Reporte de drift: referencia vs produccion simulada
	$(PY) python -m BeijingAir.monitoring.check_drift

batch: ## Predice sobre la particion de produccion y persiste con trazabilidad
	$(PY) python -m BeijingAir.flows.batch

flow: ## Pipeline de entrenamiento orquestado con Prefect
	$(PY) python -m BeijingAir.flows.training

prefect-server: ## Inicia Prefect en http://127.0.0.1:4200
	$(PY) prefect server start

serve-flow: ## Deja servido el schedule mensual de entrenamiento en Prefect (modo clase)
	$(PY) python -m BeijingAir.flows.training --serve

deploy-flow: ## Crea un deployment persistente contra el work pool (Prefect 3)
	$(PY) python -m BeijingAir.flows.deploy deploy

work-pool: ## Crea el work pool de tipo process (si no existe)
	$(PY) prefect work-pool create beijing-air-pool --type process || true

worker: ## Levanta un worker del work pool (dejar esta terminal abierta)
	$(PY) prefect worker start --pool beijing-air-pool

serve: ## Inicia la API local de prediccion en http://127.0.0.1:8000
	$(PY) uvicorn BeijingAir.api.main:app --host 127.0.0.1 --port 8000

promote-check: ## Evalua candidate contra el gate sin mover el alias champion
	$(PY) python -m BeijingAir.models.promote --dry-run

promote: ## Promueve el candidato a champion si pasa el gate
	$(PY) python -m BeijingAir.models.promote

# =============================================================================
# Stack local con Docker
# =============================================================================
up: ## Levanta el stack local: MLflow + API
	docker compose up -d --build
	@echo ""
	@echo "  MLflow  http://127.0.0.1:5001"
	@echo "  API     http://127.0.0.1:8000/docs"

down: ## Detiene el stack
	docker compose down
# =============================================================================
# Limpieza
# =============================================================================
clean: ## Borra caches y artefactos temporales
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
