# ADR 0003 — Servir solo el modelo promovido desde MLflow Registry

**Estado:** Aceptado  
**Fecha:** 2026-09-06

## Contexto

El flow de entrenamiento registra un bosque como alias `candidate`; ese alias
solo indica que una corrida termino, no que el modelo este listo para usuarios.
Copiar un archivo `.pkl` al repositorio o a la imagen de Docker haria imposible
saber que version se sirve y podria desalinear entrenamiento y produccion.

## Decision

Se expone una API FastAPI para estimaciones individuales de PM2.5:

1. `POST /predict` recibe una lectura cruda, valida rangos y deriva hora, dia,
   mes y temporada mediante `features.contract`, exactamente como entrenamiento;
2. al iniciar, la API resuelve `models:/beijing-air-pm25@champion` en MLflow
   Model Registry (la URI puede cambiarse por la variable `MODELO_URI`);
3. `GET /health` conserva HTTP 200 en estado `degradado` si falta MLflow o el
   alias. En ese caso `/predict` responde 503 y nunca sustituye `champion` por
   `candidate`;
4. la respuesta incluye nombre, version resuelta y latencia. `/modelo` publica
   el contrato de entrada y `/metrics` queda listo para Prometheus.

La imagen de Docker contiene codigo y dependencias bloqueadas por `uv.lock`,
pero no datasets ni artefactos de modelos. El gate de CI/CD de la siguiente
sesion sera quien asigne el alias `champion` despues de evaluar un candidato.

## Consecuencias

- Una prediccion usa las mismas transformaciones que el entrenamiento y evita
  train/serving skew por calculos duplicados en el cliente.
- Un despliegue nuevo no requiere reconstruir la imagen para cambiar de modelo:
  se mueve un alias auditable en el Registry.
- El servicio necesita conectividad con MLflow al arrancar. Si no la tiene,
  sigue diagnosticable por `/health`, pero no entrega predicciones engañosas.
