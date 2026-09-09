# Política de reentrenamiento

> Una página. Es el documento que cierra el ciclo: el monitoreo detecta, esta
> política decide y el gate de promoción ejecuta. Sin ella, el reentrenamiento es
> algo que alguien hace cuando se acuerda.
>
> TODO(estudiante) 32: reemplaza cada TODO. El criterio de calificación es si un
> compañero podría **operar** tu sistema leyendo solo este archivo.

## 1. Trigger: qué dispara un reentrenamiento

| Estrategia | Cuándo conviene | Riesgo |
|---|---|---|
| Periódico (semanal / mensual) | los datos cambian gradualmente | reentrena sin necesidad; cuesta |
| Por llegada de datos | llegan particiones nuevas | hay que detectar la disponibilidad |
| Por `drift` detectado | cambios impredecibles | falsos positivos → churn de modelos |
| Por caída de performance | hay labels en producción | los labels llegan tarde (*label lag*) |

**Elegido:** Por llegada de datos (cambio de hash del dataset).

**Por qué:** Beijing se descarga como un solo archivo con particiones temporales fijas; reentrenar por cron con el mismo dato no aporta señal. Detectar un SHA-256 distinto en data/raw/metadata.json es la señal honesta de "hay dato nuevo".

**Frecuencia máxima:** una vez al día (cron `0 3 1 * *`), nunca cada pocos minutos.

Un anti-patrón concreto que hay que evitar: `cron="*/2 * * * *"` reentrenando el
modelo completo. Cuesta dinero, no aporta señal y enseña el hábito contrario al
correcto.

## 2. Umbrales y su justificación

| Señal | Umbral | Ventana | Por qué **este** número |
|---|---|---|---|
| Fracción de columnas con `drift` | 0.30 | partición completa | UMBRAL_DRIFT_COLUMNAS en config.py; con ≤30% de columnas el sistema sigue sirviendo. |
| Tamaño de efecto por columna (KS / V de Cramer) | KS ≥ 0.10 (num), V de Cramer ≥ 0.10 (cat) | partición completa | EFECTO_MINIMO_KS; un KS de 0.05 casi nunca justifica reentrenar, uno de 0.30 sí. |
| Degradación de la métrica de negocio | MAE +5% vs champion | holdout test | MAX_EMPEORAMIENTO_MAE=0.05; no se promueve un candidato que empeora el error más del 5%. |

Dos advertencias que el umbral tiene que respetar:

- **`p < 0.05` no es un criterio de `drift`.** Con n grande todo sale
  significativo: con 500 000 filas, un cambio de la media del 0.1 % da
  p < 1e-10 y no le importa a nadie. Hay que mirar el tamaño del efecto.
- **La estacionalidad no es `drift` accionable.** Si diciembre siempre se comporta
  distinto, la respuesta puede ser incluir la estacionalidad como feature, no
  reentrenar cada diciembre.

## 3. Datos que se usan al reentrenar

Se reentrena sobre las particiones fijas de entrenamiento (PARTICIONES_TRAIN) que
registre config.py en ese momento; el hash en metadata.json garantiza que es el
mismo dato con el que se midió. No se reponderan datos recientes ni se excluye el
período con drift: la partición que disparó el drift se incluye, porque lo que se
quiere es que el modelo aprenda la distribución nueva.

## 4. Quién aprueba y qué queda registrado

| Situación | Decide | Evidencia que queda |
|---|---|---|
| El gate aprueba el candidato | automático | tag `validation_status=passed` + `gate_motivo` en la versión |
| El gate rechaza | automático (no se promueve) | tag `validation_status=failed` + motivo |
| Se quiere promover pese al rechazo | Alejandro Taborda |  comentario en el PR + tag `gate_motivo` en la versión |

Que la evidencia del rechazo se guarde importa tanto como la de la aprobación:
tres semanas después, "por qué no se promovió aquel modelo" es una pregunta real.

## 5. Rollback

Rollback de modelo: mover el alias `@champion` a la versión anterior (ver códigoabajo). Es una escritura de metadatos y no requiere reentrenar ni redeploy.

Volver atrás es mover el alias `@champion` a la versión anterior:

```python
from mlflow import MlflowClient

MlflowClient().set_registered_model_alias("beijing-air-pm25", "champion", "6")
```

Es una escritura de metadatos —sub-segundo, sin reentrenar, sin rebuild de imagen,
sin redeploy— y funciona porque las versiones del `registry` son inmutables: el
artefacto de la versión 6 sigue siendo bit a bit el que estaba sirviendo. Esa
propiedad es la razón principal para referenciar el modelo por alias en lugar de
copiarlo a un directorio.

**Criterio para hacer rollback:**  el modelo en producción empeora el MAE de test más del 5% frente a su antecesor, o degrada el error en una estación crítica. **Quién puede ejecutarlo:** el responsable del modelo, moviendo el alias `@champion` a la versión anterior registrada.

## 6. Alertas

Una alerta necesita cuatro cosas o es ruido: **umbral, ventana, destinatario y
acción esperada**. Una alerta sin acción esperada se ignora, y la fatiga de
alertas es un problema de operación tan real como el `drift`.

| Alerta | Umbral | Destinatario | Acción esperada |
|---|---|---|---|
| Drift de datos | >30% de columnas con drift (KS ≥ 0.10) | responsable del modelo / canal del equipo | revisar `reports/drift-report.html` y decidir si se reentrena |
| Degradación de métrica | MAE de test > umbral o +5% vs `champion` | responsable del modelo | rechazar el candidato o hacer rollback del alias |
