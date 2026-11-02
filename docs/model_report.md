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