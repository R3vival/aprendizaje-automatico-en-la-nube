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

.PHONY: help setup data test test-fast lint format typecheck check validate-data mlflow prefect-server train flow serve-flow clean

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
	@echo ""
	@echo "Listo."

data: ## Descarga y extrae el dataset Beijing
	$(PY) python -m BeijingAir.data.descarga
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

prefect-server: ## Inicia Prefect en http://127.0.0.1:4200
	$(PY) prefect server start

flow: ## Ejecuta una corrida orquestada de validacion y entrenamiento
	$(PY) python -m BeijingAir.flows.training

serve-flow: ## Deja servido el schedule mensual de entrenamiento en Prefect
	$(PY) python -m BeijingAir.flows.training --serve

# =============================================================================
# Limpieza
# =============================================================================
clean: ## Borra caches y artefactos temporales
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
