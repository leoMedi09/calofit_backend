"""
Evaluación de modelos ML de CaloFit:
  - RF (perfil_adherencia.pkl): accuracy en dataset Kaggle + análisis de sesgo
  - KNN (recomendador_knn.pkl): diversidad de recomendaciones entre los 15 usuarios de prueba

Ejecutar: docker exec calofit_backend python scripts/evaluar_modelos_ml.py
"""
import asyncio
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = Path("app/ml_models")
DATA_DIR  = Path("scripts/data")


# ─── RF ──────────────────────────────────────────────────────────────────────

def evaluar_rf():
    print("\n" + "="*65)
    print("EVALUACIÓN RF — perfil_adherencia.pkl")
    print("="*65)

    clf_path = MODEL_DIR / "perfil_adherencia.pkl"
    if not clf_path.exists():
        print(f"  ERROR: no se encontró {clf_path}")
        return

    clf = joblib.load(clf_path)
    csv_path = DATA_DIR / "gym_members_exercise_tracking.csv"

    if not csv_path.exists():
        print(f"  Modelo cargado ✅ ({clf.n_estimators} árboles, max_depth={clf.max_depth})")
        print(f"  Dataset Kaggle no encontrado en {csv_path} — omitiendo evaluación de accuracy.")
        print("  Nota: modelo entrenado sobre Kaggle (n=973, usuarios internacionales).")
        print("  Riesgo de sesgo: patrones de gimnasio peruano pueden diferir.")
        print("  Recomendación: reentrenar con retrain_rf_calofit.py cuando haya ≥100 usuarios reales.")
        return

    df = pd.read_csv(csv_path)

    # Replicar feature engineering de entrenar_perfil_adherencia.py
    df = df.dropna(subset=["Experience_Level"])
    df["Perfil"] = df["Experience_Level"].map({1: "PERFIL_C", 2: "PERFIL_B", 3: "PERFIL_A"})
    df = df.dropna(subset=["Perfil"])

    # One-hot workout type si existe
    if "Workout_Type" in df.columns:
        dummies = pd.get_dummies(df["Workout_Type"], prefix="wt")
        df = pd.concat([df, dummies], axis=1)

    feature_cols = [c for c in clf.feature_names_in_ if c in df.columns]
    missing = set(clf.feature_names_in_) - set(feature_cols)
    if missing:
        print(f"  Advertencia: features faltantes en CSV — {missing}. Rellenas con 0.")
        for m in missing:
            df[m] = 0
        feature_cols = list(clf.feature_names_in_)

    X = df[feature_cols].fillna(0).values
    y = df["Perfil"].values

    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_score

    y_pred = clf.predict(X)
    print(f"\n  Registros evaluados: {len(y)}")
    print(f"  Distribución real: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"\n  Classification Report:")
    print(classification_report(y, y_pred, target_names=["PERFIL_A", "PERFIL_B", "PERFIL_C"]))
    print(f"  Matriz de confusión:")
    cm = confusion_matrix(y, y_pred, labels=["PERFIL_A", "PERFIL_B", "PERFIL_C"])
    print(f"  {cm}")
    print("\n  CONCLUSIÓN:")
    acc = (y == y_pred).mean()
    if acc >= 0.80:
        print(f"  Accuracy={acc:.2%} ✅ — RF funciona bien sobre Kaggle.")
        print("  Recomendación: validar con datos reales CaloFit cuando haya ≥100 usuarios.")
    else:
        print(f"  Accuracy={acc:.2%} ⚠ — Considerar reentrenar (ver retrain_rf_calofit.py).")


# ─── KNN ─────────────────────────────────────────────────────────────────────

async def _obtener_recos_cliente(asistente, client_id, db):
    try:
        r = await asistente.obtener_recomendaciones(client_id=client_id, db=db)
        return [x["nombre"] for x in r.get("recomendaciones_alimentos", [])]
    except Exception as e:
        print(f"  Client {client_id}: ERROR — {e}")
        return []


def evaluar_knn():
    print("\n" + "="*65)
    print("EVALUACIÓN KNN — recomendador_knn.pkl")
    print("="*65)

    knn_path = MODEL_DIR / "recomendador_knn.pkl"
    scaler_path = MODEL_DIR / "scaler_knn.pkl"
    if not knn_path.exists():
        print(f"  ERROR: no se encontró {knn_path}")
        return

    knn    = joblib.load(knn_path)
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    print(f"  KNN cargado ✅ (n_neighbors={knn.n_neighbors}, metric={knn.metric})")
    if scaler:
        print(f"  Scaler cargado ✅")

    from app.core.database import SessionLocal
    from app.services.asistente_recomendaciones import AsistenteRecomendaciones

    db = SessionLocal()
    asistente = AsistenteRecomendaciones()
    todos_recos = []
    recos_por_cliente = {}

    async def _run():
        for cid in range(55, 70):
            recos = await _obtener_recos_cliente(asistente, cid, db)
            recos_por_cliente[cid] = recos
            todos_recos.extend(recos)
            perfil = "A" if cid < 60 else ("B" if cid < 65 else "C")
            print(f"  Client {cid} (Perfil {perfil}): {recos[:3]}")

    try:
        asyncio.run(_run())
    finally:
        db.close()

    unicos = len(set(todos_recos))
    total  = len(todos_recos)
    ratio  = unicos / max(total, 1)

    print(f"\n  Diversidad total: {unicos} alimentos únicos / {total} recomendaciones")
    print(f"  Ratio de diversidad: {ratio:.2%}")

    if ratio < 0.30:
        print("  ⚠ Diversidad baja — clientes con déficit similar reciben recos idénticas.")
    elif ratio < 0.60:
        print("  ~ Diversidad media — hay variación entre perfiles pero no entre individuos.")
    else:
        print("  ✅ Diversidad alta — buena variación entre clientes.")

    print("\n  LIMITACIONES DOCUMENTADAS:")
    print("  - Personalización SOLO por exclusión (últimas 48h via HistorialRecomendacion).")
    print("  - NO usa preferencias positivas ni historial de aceptación del usuario.")
    print("  - Clientes con mismo déficit calórico → mismas recomendaciones.")
    print("  Recomendación futura: agregar tabla preferencias_usuario (likes/dislikes).")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluar_rf()
    evaluar_knn()
    print("\n" + "="*65)
    print("Evaluación completa.")
