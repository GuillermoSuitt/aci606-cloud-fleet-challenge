# Reporte del modelo

- Filas totales: 18000 (6 flotas)
- Flotas de entrenamiento: [3, 4, 5, 6] · Flotas de prueba: [1, 2]

## Distribución de clases

| Clase | Cantidad |
|---|---|
| CRITICAL | 12720 |
| NORMAL | 2749 |
| WARNING | 2531 |

## Comparación de modelos (F1 macro en flotas de prueba)

| Modelo | F1 macro |
|---|---|
| baseline_mayoritaria | 0.290 |
| logistic_regression | 0.905 |
| random_forest | 0.907 |

## Matriz de confusión (random_forest)

| Real \ Predicho | NORMAL | WARNING | CRITICAL |
|---|---|---|---|
| NORMAL | 513 | 80 | 0 |
| WARNING | 53 | 710 | 18 |
| CRITICAL | 0 | 100 | 4526 |

Modelo final: **random_forest**, reentrenado con todas las flotas y guardado en `models\model.joblib`.


## Justificación técnica

**Métrica: F1 macro en vez de accuracy.** Las clases están desbalanceadas: CRITICAL es el 71% de las muestras. Un modelo que siempre responde CRITICAL obtiene 77% de accuracy en las flotas de prueba, pero nunca detecta un robot NORMAL ni WARNING (F1 macro 0,290). El F1 macro promedia el F1 de las tres clases con el mismo peso, así que solo es alto si el modelo reconoce bien todas las clases. Además, ambos clasificadores usan `class_weight="balanced"` para compensar el desbalance.

**Validación por flota.** La separación entrenamiento/prueba se hizo con `GroupShuffleSplit` por `fleet_id` (flotas 3–6 para entrenar, 1–2 para probar). Así el modelo se evalúa con flotas que nunca vio, igual que en la evaluación docente. Si se mezclaran filas de una misma flota en ambos conjuntos, el resultado saldría artificialmente alto.

**Elección del modelo.** Se compararon una regresión logística (F1 macro 0,905) y un Random Forest (0,907), ambos dentro de un pipeline con `StandardScaler`. Se eligió Random Forest por tener el mejor F1 macro y porque sus errores ocurren solo entre clases vecinas: en la matriz de confusión nunca confunde un robot NORMAL con uno CRITICAL. Tras elegirlo, se reentrenó con las seis flotas para aprovechar todos los datos.

**100 árboles en vez de 300.** Con 300 árboles el F1 macro era 0,909, pero en Render (plan gratuito, 0,1 CPU) cada predicción tardaba 82 ms. Con 100 árboles el F1 baja solo a 0,907 y la predicción tarda 13 ms. Además, en `/predict` se evalúan los árboles directamente en lugar de usar `predict_proba`, con resultados idénticos (diferencia 0,0) y 2,4 veces más capacidad bajo carga (prueba de 200 solicitudes concurrentes, 0 errores).

**Limitaciones.** El modelo solo conoce los rangos de los datos de entrenamiento (por ejemplo, temperatura del motor hasta 84 °C). Con valores fuera de ese rango, como 90 °C, la predicción pierde confianza (51%), por eso el servicio devuelve siempre la confianza junto con la clase. La clase más difícil es WARNING (F1 0,85), porque está entre las otras dos y se confunde con ambas.
