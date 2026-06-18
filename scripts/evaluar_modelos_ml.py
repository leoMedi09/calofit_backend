"""
Evaluación de modelos ML de CaloFit — metodología CRISP-DM (Fase 5: Evaluation)

  RF  (perfil_adherencia.pkl) — modelo SUPERVISADO:
      accuracy, F1 y matriz de confusión sobre el dataset de entrenamiento (Kaggle).

  KNN (recomendador_knn.pkl) — modelo NO SUPERVISADO (similitud coseno):
      no existe una etiqueta "correcta" a predecir, por lo que NO aplican
      accuracy/F1/recall clásicos. Se evalúa con:
        1. Estructura del espacio vectorial (distancia a vecinos + Silhouette).
        2. Diversidad/cobertura del catálogo bajo escenarios simulados.
        3. Validación con datos reales de los 15 usuarios de prueba (IDs 55-69).

Ejecutar (dentro del contenedor backend):
  docker exec calofit_backend python scripts/evaluar_modelos_ml.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = Path("app/models/ai_models")
DATA_DIR = Path("scripts/data")


# ─── RF (supervisado) ────────────────────────────────────────────────────────

def evaluar_rf():
    print("\n" + "=" * 65)
    print("EVALUACIÓN RF — perfil_adherencia.pkl (modelo SUPERVISADO)")
    print("=" * 65)

    clf_path = MODEL_DIR / "perfil_adherencia.pkl"
    if not clf_path.exists():
        print(f"  ERROR: no se encontró {clf_path}")
        return

    bundle = joblib.load(clf_path)
    if hasattr(bundle, "modelo"):  # soporte legacy (clase serializada)
        rf = bundle.modelo
        features = bundle.features
        label_map = {1: "PERFIL_C", 2: "PERFIL_B", 3: "PERFIL_A"}
    else:  # formato dict (recomendado)
        rf = bundle["rf_model"]
        features = bundle["features"]
        label_map = bundle["label_map"]

    csv_path = DATA_DIR / "gym_members_exercise_tracking.csv"
    if not csv_path.exists():
        print(f"  Modelo cargado ✅ ({rf.n_estimators} árboles, max_depth={rf.max_depth})")
        print(f"  Dataset Kaggle no encontrado en {csv_path} — omitiendo evaluación de accuracy.")
        return

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Experience_Level"])
    df["Perfil"] = df["Experience_Level"].map({1: "PERFIL_C", 2: "PERFIL_B", 3: "PERFIL_A"})
    df = df.dropna(subset=["Perfil"])

    # Reconstruir las features exactamente como en entrenar_perfil_adherencia.py
    df["Gender_Enc"] = (df["Gender"] == "Male").astype(int)
    df["Height_cm"] = df["Height (m)"] * 100
    if "Workout_Type" in df.columns:
        dummies = pd.get_dummies(df["Workout_Type"], prefix="Workout")
        df = pd.concat([df, dummies], axis=1)

    faltantes = [c for c in features if c not in df.columns]
    if faltantes:
        print(f"  Advertencia: features faltantes en CSV — {faltantes}. Rellenas con 0.")
        for c in faltantes:
            df[c] = 0

    X = df[features].fillna(0).values
    y = df["Perfil"].values

    from sklearn.metrics import classification_report, confusion_matrix

    y_pred_num = rf.predict(X)
    y_pred = np.array([label_map[p] for p in y_pred_num])
    print(f"\n  Registros evaluados: {len(y)}")
    print(f"  Distribución real: {dict(zip(*np.unique(y, return_counts=True)))}")
    print("\n  Classification Report:")
    print(classification_report(y, y_pred, target_names=["PERFIL_A", "PERFIL_B", "PERFIL_C"]))
    print("  Matriz de confusión (filas=real, columnas=predicho):")
    cm = confusion_matrix(y, y_pred, labels=["PERFIL_A", "PERFIL_B", "PERFIL_C"])
    print(f"  {cm}")

    acc = (y == y_pred).mean()
    print(f"\n  Accuracy global: {acc:.2%}")
    if acc >= 0.80:
        print("  ✅ RF dentro del umbral esperado (≥80%) sobre Kaggle.")
    else:
        print("  ⚠ Por debajo del umbral — considerar reentrenar (retrain_rf_calofit.py).")


# ─── KNN (no supervisado) ──────────────────────────────────────────────────────

def _cargar_knn():
    knn_path = MODEL_DIR / "recomendador_knn.pkl"
    if not knn_path.exists():
        print(f"  ERROR: no se encontró {knn_path}")
        return None
    paquete = joblib.load(knn_path)
    return paquete["modelo_knn"], paquete["scaler"], paquete["df_alimentos"]


def _evaluar_estructura_espacio(knn, scaler, df):
    """
    Métricas de estructura del espacio vectorial — NO requieren etiquetas.
    Validan que el KNN organiza los alimentos en un espacio matemáticamente
    coherente (alimentos con proporciones de macros parecidas quedan cerca).
    """
    print("\n  --- 1. Estructura del espacio vectorial ---")

    features = ["calorias_100g", "proteina_100g", "carbohindratos_100g", "grasas_100g"]
    X_scaled = scaler.transform(df[features].values)

    # Distancia al vecino más cercano (columna 0 es el propio punto, dist=0)
    distancias, _ = knn.kneighbors(X_scaled, n_neighbors=2)
    dist_vecino = distancias[:, 1]
    print(f"  Distancia coseno promedio al vecino más cercano: {dist_vecino.mean():.4f}")
    print(f"  Distancia coseno mediana al vecino más cercano:  {np.median(dist_vecino):.4f}")
    print("  (cercano a 0 = el catálogo tiene 'vecinos' nutricionales densos;")
    print("   cercano a 1 = alimentos aislados, sin equivalentes cercanos)")

    # Silhouette score sobre clusters K-Means en el mismo espacio escalado.
    # No usa las predicciones del KNN — solo valida si las features
    # (kcal, prot, carb, gras escalados) tienen estructura separable.
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    k = 4
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, labels, metric="cosine")
    print(f"\n  Silhouette Score (k={k} clusters, distancia coseno): {sil:.3f}")
    print("  Interpretación: >0.5 = buena separación, >0.25 = razonable, <0.1 = débil")
    if sil >= 0.50:
        print("  ✅ Buena estructura — clusters de macros bien diferenciados.")
    elif sil >= 0.25:
        print("  ✅ Estructura razonable — el espacio de macros separa grupos de alimentos.")
    elif sil >= 0.10:
        print("  ~ Estructura débil pero presente — esperable en datos nutricionales reales.")
    else:
        print("  ⚠ Estructura muy débil — revisar escalado/features.")

    db = davies_bouldin_score(X_scaled, labels)
    print(f"\n  Davies-Bouldin Index (k={k} clusters): {db:.3f}")
    print("  Interpretación: <1.0 = buena separación, ~1.5 = aceptable, >2.0 = clusters solapados")
    if db < 1.0:
        print("  ✅ Clusters bien separados — el espacio de macros tiene fronteras claras.")
    elif db < 1.5:
        print("  ✅ Separación aceptable — clusters con algo de solapamiento (normal en nutrición).")
    else:
        print("  ⚠ Clusters con alto solapamiento — alimentos muy similares entre grupos.")


def _vectores_deficit_sinteticos(n: int, seed: int = 42) -> list[list[float]]:
    """
    Genera N vectores de déficit [kcal, prot, carb, gras] dentro de rangos
    realistas (mismos límites que aplica obtener_recomendaciones()).
    """
    rng = np.random.default_rng(seed)
    return [
        [rng.uniform(50, 900), rng.uniform(0, 60), rng.uniform(0, 120), rng.uniform(0, 40)]
        for _ in range(n)
    ]


def _evaluar_diversidad_cobertura(n_consultas: int = 100):
    """
    Cobertura: de los alimentos del catálogo, ¿cuántos distintos aparecen
    alguna vez en el top-3 al simular N escenarios de déficit variados?
    Un modelo que siempre recomienda los mismos 5 alimentos tiene baja
    utilidad práctica aunque su similitud individual sea alta.
    """
    print("\n  --- 2. Diversidad y cobertura (simulación) ---")

    from app.services.ml_service import ml_recomendador

    if not ml_recomendador.modelo_activo:
        print("  ⚠ ml_recomendador no está activo — omitiendo.")
        return

    from sklearn.metrics.pairwise import cosine_distances

    features_knn = ["calorias_100g", "proteina_100g", "carbohindratos_100g", "grasas_100g"]
    df_knn  = ml_recomendador._df
    scaler  = ml_recomendador._scaler

    total_catalogo = len(df_knn)
    vistos: set[str] = set()
    todas_recos: list[str] = []
    ild_scores: list[float] = []

    for vector in _vectores_deficit_sinteticos(n_consultas):
        recos = ml_recomendador.obtener_recomendaciones(*vector, n_recomendaciones=3)
        for r in recos:
            vistos.add(r["alimento"].lower().strip())
            todas_recos.append(r["alimento"])

        # ILD — distancia coseno promedio entre pares dentro de la lista
        if len(recos) >= 2:
            item_vecs = []
            for r in recos:
                row = df_knn[df_knn["alimento"] == r["alimento"]]
                if not row.empty:
                    item_vecs.append(scaler.transform(row[features_knn].values)[0])
            if len(item_vecs) >= 2:
                dmat = cosine_distances(item_vecs)
                n = len(item_vecs)
                pares = [(i, j) for i in range(n) for j in range(i + 1, n)]
                ild_scores.append(float(np.mean([dmat[i][j] for i, j in pares])))

    cobertura = len(vistos) / total_catalogo
    print(f"  Catálogo total: {total_catalogo} alimentos")
    print(f"  Alimentos distintos recomendados en {n_consultas} escenarios simulados: {len(vistos)}")
    print(f"  Cobertura: {cobertura:.1%}")

    ratio_unicidad = len(set(todas_recos)) / max(len(todas_recos), 1)
    print(f"  Ratio de unicidad por recomendación: {ratio_unicidad:.1%}")

    if cobertura >= 0.15:
        print("  ✅ Cobertura razonable — el modelo explora una porción amplia del catálogo.")
    else:
        print("  ⚠ Cobertura baja — pocas combinaciones de macros dominan las recomendaciones.")

    ild_mean = float(np.mean(ild_scores)) if ild_scores else 0.0
    print(f"\n  ILD — Intra-list Diversity (diversidad intra-lista): {ild_mean:.4f}")
    print("  Interpretación: 0 = ítems nutricionalmente idénticos, 1 = completamente distintos")
    if ild_mean >= 0.10:
        print("  ✅ Las listas de 3 recomendaciones tienen diversidad interna razonable.")
    elif ild_mean >= 0.03:
        print("  ~ Diversidad interna moderada — esperado en recomendadores basados en déficit.")
    else:
        print("  ⚠ Ítems muy similares entre sí dentro de cada lista.")


def _evaluar_con_datos_reales():
    """
    Validación con los registros reales de progreso_calorias de los 15
    usuarios de prueba (IDs 55-69). Para cada día registrado se calcula el
    déficit restante del usuario (meta del plan - consumido) y se consulta
    el KNN.

    No mide "accuracy" (no hay un alimento "correcto" etiquetado) — reporta
    la distribución real de similitudes y un chequeo de coherencia: ¿la
    primera recomendación cabe dentro de un margen razonable del déficit
    calórico restante?
    """
    print("\n  --- 3. Validación con datos reales (progreso_calorias, IDs 55-69) ---")

    from datetime import datetime

    from app.core.database import SessionLocal
    from app.models.client import Client
    from app.models.historial import ProgresoCalorias
    from app.services.asistente.asistente_plan import obtener_plan_hoy
    from app.services.ml_service import ml_recomendador

    if not ml_recomendador.modelo_activo:
        print("  ⚠ ml_recomendador no está activo — omitiendo.")
        return

    db = SessionLocal()
    try:
        similitudes: list[float] = []
        coherentes = 0
        evaluados = 0

        for cid in range(55, 70):
            cliente = db.query(Client).filter(Client.id == cid).first()
            if not cliente:
                continue
            edad = 25
            if cliente.birth_date:
                edad = datetime.now().year - cliente.birth_date.year
            try:
                _, plan, _ = obtener_plan_hoy(cliente, edad, db)
            except Exception:
                continue

            registros = (
                db.query(ProgresoCalorias)
                .filter(ProgresoCalorias.client_id == cid)
                .all()
            )
            for r in registros:
                rest_kcal = max(plan["calorias_dia"] - (r.calorias_consumidas or 0), 50)
                rest_prot = max(plan["proteinas_g"] - (r.proteinas_consumidas or 0), 0)
                rest_carb = max(plan["carbohidratos_g"] - (r.carbohidratos_consumidos or 0), 0)
                rest_gras = max(plan["grasas_g"] - (r.grasas_consumidas or 0), 0)

                recos = ml_recomendador.obtener_recomendaciones(
                    rest_kcal, rest_prot, rest_carb, rest_gras, n_recomendaciones=3
                )
                if not recos:
                    continue
                evaluados += 1
                similitudes.extend(rec["similitud"] for rec in recos)
                if recos[0]["calorias_100g"] <= rest_kcal * 1.5:
                    coherentes += 1

        if not evaluados:
            print("  ⚠ No se encontraron registros evaluables.")
            return

        sims = np.array(similitudes)
        print(f"  Días evaluados: {evaluados}")
        print(f"  Similitud promedio top-3: {sims.mean():.1f}%")
        print(f"  Similitud mínima / máxima: {sims.min():.1f}% / {sims.max():.1f}%")
        print(
            "  Coherencia calórica (top-1 dentro de 1.5x del déficit restante): "
            f"{coherentes}/{evaluados} ({coherentes / evaluados:.1%})"
        )
    finally:
        db.close()


def evaluar_knn():
    print("\n" + "=" * 65)
    print("EVALUACIÓN KNN — recomendador_knn.pkl (modelo NO SUPERVISADO)")
    print("=" * 65)
    print(
        "\n"
        "  El KNN no tiene una etiqueta 'correcta' que predecir (no es un\n"
        "  clasificador), por lo que NO aplican accuracy/F1/recall clásicos.\n"
        "  Se evalúa con:\n"
        "    1. Estructura del espacio vectorial (distancia a vecinos + Silhouette).\n"
        "    2. Diversidad/cobertura del catálogo bajo escenarios simulados.\n"
        "    3. Validación con datos reales de los 15 usuarios de prueba.\n"
    )

    cargado = _cargar_knn()
    if cargado is None:
        return
    knn, scaler, df = cargado
    print(f"  Modelo cargado ✅ — {len(df)} alimentos, n_neighbors={knn.n_neighbors}, metric={knn.metric}")

    _evaluar_estructura_espacio(knn, scaler, df)
    _evaluar_diversidad_cobertura()
    _evaluar_con_datos_reales()

    print("\n  ROL ACTUAL Y LIMITACIONES:")
    print("  - Modelo no supervisado: sin accuracy/recall clásicos, solo métricas de estructura.")
    print("  - Rol en el sistema: en respuesta_recomendacion_llm() (modo comida), el KNN")
    print("    calcula 2-3 alimentos del catálogo INS/CENAN por similitud coseno con el")
    print("    déficit real de macros del usuario, excluyendo lo recomendado en las últimas")
    print("    48h (HistorialRecomendacion). Esos candidatos se inyectan al prompt del LLM")
    print("    como opciones a considerar; el LLM decide si los usa, los combina o los")
    print("    ignora — no genera el texto que ve el usuario (eso es Llama-3).")
    print("  - Anti-repetición: los 3 platos finales del LLM se persisten en")
    print("    HistorialRecomendacion (plato_id=NULL) para excluirlos en las próximas 48h.")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluar_rf()
    evaluar_knn()
    print("\n" + "=" * 65)
    print("Evaluación completa.")
    print("=" * 65)
