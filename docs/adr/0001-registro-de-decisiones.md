# ADR 0001 — Registro de Decisiones de Arquitectura

**Estado:** Aceptado
**Fecha:** 2026-08-25

## Contexto

Necesitamos un lugar para registrar las decisiones de diseño importantes de este
repositorio, de modo que el *por qué* de cada elección quede documentado junto al
código. Este archivo es la plantilla para las ADR futuras.

## Decisión

Cada ADR vive en `docs/adr/` con nombre `NNNN-titulo-breve.md`. Cada ADR responde
tres preguntas: el contexto que motivó la decisión, la decisión misma y las
consecuencias (lo que gana y lo que pierde).

## Consecuencias

- Las decisiones de diseño quedan trazables y no dependen de la memoria de quien
  las tomó.
- Cambiar una decisión requiere una ADR nueva, no una edición silenciosa.
