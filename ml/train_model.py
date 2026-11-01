"""Exercise 4 training starter.

Students must complete this script. The goal is not to copy a fixed model but to
build a reproducible training pipeline and justify the chosen model.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data")
MODEL_PATH = Path("models/model.joblib")
REPORT_PATH = Path("docs/model_report.md")

FEATURES = [
    "speed",
    "battery",
    "motor_temp",
    "vibration",
    "motor_current",
    "obstacle_distance",
]
TARGET = "status"
LABELS = ["NORMAL", "WARNING", "CRITICAL"]


def main() -> None:
    # 1. Cargar todas las flotas y marcar a cuál pertenece cada fila
    frames = []
    for i, path in enumerate(sorted(DATA_DIR.glob("fleet_*.csv")), start=1):
        df_fleet = pd.read_csv(path)
        df_fleet["fleet_id"] = i
        frames.append(df_fleet)
    df = pd.concat(frames, ignore_index=True)

    print("Filas totales:", len(df))
    print("\nDistribución de clases:")
    print(df[TARGET].value_counts())

    # 2. Separar train/test por flota (las flotas de test el modelo NUNCA las ve)
    X = df[FEATURES]
    y = df[TARGET]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=df["fleet_id"]))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print("\nFlotas en entrenamiento:", sorted(df.loc[train_idx, "fleet_id"].unique()))
    print("Flotas en prueba:", sorted(df.loc[test_idx, "fleet_id"].unique()))

    # 3. Comparar modelos
    candidatos = {
        "baseline_mayoritaria": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        "random_forest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
        ),
    }

    resultados = {}
    for nombre, modelo in candidatos.items():
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        resultados[nombre] = f1_score(y_test, pred, average="macro")
        print(f"\n=== {nombre} === F1 macro: {resultados[nombre]:.3f}")
        print(classification_report(y_test, pred, zero_division=0))

    # 4. Elegir el mejor (sin contar la línea base)
    mejor = max(
        (n for n in resultados if n != "baseline_mayoritaria"),
        key=resultados.get,
    )
    cm = confusion_matrix(y_test, candidatos[mejor].predict(X_test), labels=LABELS)

    print(f"\nModelo elegido: {mejor}")
    print("Matriz de confusión (filas = real, columnas = predicho):")
    print("          ", LABELS)
    for label, fila in zip(LABELS, cm):
        print(f"{label:>10}", fila)

    # 5. Reentrenar el mejor con TODAS las flotas y guardarlo
    modelo_final = candidatos[mejor].fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo_final, MODEL_PATH)
    print(f"\nModelo guardado en {MODEL_PATH}")
    # 6. Guardar evidencia en docs/model_report.md
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    flotas_train = sorted(df.loc[train_idx, "fleet_id"].unique().tolist())
    flotas_test = sorted(df.loc[test_idx, "fleet_id"].unique().tolist())

    lineas = [
        "# Reporte del modelo",
        "",
        f"- Filas totales: {len(df)} ({df['fleet_id'].nunique()} flotas)",
        f"- Flotas de entrenamiento: {flotas_train} · Flotas de prueba: {flotas_test}",
        "",
        "## Distribución de clases",
        "",
        "| Clase | Cantidad |",
        "|---|---|",
    ]
    for clase, n in df[TARGET].value_counts().items():
        lineas.append(f"| {clase} | {n} |")

    lineas += ["", "## Comparación de modelos (F1 macro en flotas de prueba)", "",
               "| Modelo | F1 macro |", "|---|---|"]
    for nombre, f1 in resultados.items():
        lineas.append(f"| {nombre} | {f1:.3f} |")

    lineas += ["", f"## Matriz de confusión ({mejor})", "",
               "| Real \\ Predicho | " + " | ".join(LABELS) + " |", "|---|---|---|---|"]
    for label, fila in zip(LABELS, cm):
        lineas.append(f"| {label} | " + " | ".join(str(v) for v in fila) + " |")

    lineas += ["", f"Modelo final: **{mejor}**, reentrenado con todas las flotas y guardado en `{MODEL_PATH}`."]
    REPORT_PATH.write_text("\n".join(lineas), encoding="utf-8")
    print(f"Reporte guardado en {REPORT_PATH}")


if __name__ == "__main__":
    main()
