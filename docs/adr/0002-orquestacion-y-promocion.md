# ADR 0002 — Orquestar el entrenamiento sin promover automáticamente

**Estado:** Aceptado  
**Fecha:** 2026-09-06

## Contexto

El proyecto ya valida datos y registra experimentos, pero ejecutar un script a
mano no deja una vista operacional de sus pasos ni permite reintentos selectivos.
Además, el dataset Beijing llega como una fuente versionada por hash: entrenar
en cada ejecución de un cron no añade señal cuando el hash no cambió.

## Decisión

Usar Prefect para el flow `entrenamiento-beijing-pm25`. El flow:

1. asegura el origen con reintentos para fallos transitorios de descarga;
2. valida las particiones, con caché cuyo key incluye hash y tamaño de muestra;
3. entrena y registra en MLflow el bosque como alias `candidate`;
4. guarda la huella del dataset solo si toda la corrida fue exitosa.

El schedule mensual despierta el flow, pero el trigger real es un hash nuevo de
`metadata.json`; `--forzar` deja repetir una corrida por un cambio deliberado de
código o parámetros. El flow **no** mueve ningún modelo a producción: esa
promoción será un gate versionado de CI/CD.

## Consecuencias

- Prefect muestra qué paso falló y permite reintentar la descarga sin repetir la
  lógica de orquestación.
- MLflow conserva qué modelo, métricas y artefactos produjo cada flow mediante
  el tag `prefect_flow_run_id`.
- Una ejecución programada con datos idénticos termina correctamente, pero no
  gasta recursos ni llena el registry con copias del mismo candidato.
- Para servir el schedule se necesita el servidor local de Prefect; ejecutar el
  flow una vez no requiere un deployment ni promueve modelos.
