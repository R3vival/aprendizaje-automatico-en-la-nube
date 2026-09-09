# ADR 0004 — Promover modelos con un gate manual y reproducible

**Estado:** Aceptado  
**Fecha:** 2026-09-06

## Contexto

El flow registra el bosque como `candidate`, pero una corrida terminada no es
sinónimo de un modelo apto para producción. Promover automáticamente desde el
flow permitiría que un dataset nuevo, una regresión o un error de configuración
cambie la API sin revisión. Además, las métricas de `valid` ya participaron en
la selección durante entrenamiento y no son evidencia independiente para la
promoción.

## Decisión

Cada modelo registra métricas separadas sobre la partición temporal `test`.
El módulo `BeijingAir.models.promote` recupera el candidato desde MLflow,
obtiene esas métricas de su corrida y aplica estas reglas:

1. `mae_test` no puede superar `MAX_MAE_TEST` (45 inicialmente);
2. `r2_test` debe ser al menos `MIN_R2_TEST` (0 inicialmente);
3. si existe `champion`, el MAE del candidato no puede empeorar más de 5 %;
4. si una métrica falta, no es finita o una regla falla, el alias no cambia.

`promote-model.yml` se ejecuta únicamente de forma manual y usa el entorno de
GitHub `production`. Requiere el secreto `MLFLOW_TRACKING_URI`, que debe
apuntar a un Registry remoto accesible desde GitHub Actions. El URI local
`127.0.0.1` no funciona desde los runners de GitHub. El workflow permite
`dry-run` para auditar la decisión sin efectuar la mutación.

## Consecuencias

- `candidate` y `champion` conservan significados distintos: entrenado frente a
  aprobado para servir.
- Una corrida antigua sin métricas de test no se puede promover por accidente;
  hay que ejecutar el flow actualizado para producir un candidato completo.
- Los límites están centralizados y se pueden ajustar con variables de entorno
  tras revisar las primeras corridas, sin reescribir el gate.
