"""
╔══════════════════════════════════════════════════════════════════════╗
║     CaloFit — Sistema de Recomendación de Alimentos Peruanos        ║
║                Metodología CRISP-DM Documentada                     ║
║                                                                      ║
║  Algoritmo : K-Nearest Neighbors (KNN) con Similitud Coseno         ║
║  Dataset   : Tabla Peruana de Alimentos (INS/CENAN 2017) + OFF      ║
║  Output    : app/models/ai_models/recomendador_knn.pkl              ║
║                                                                      ║
║  Función   : Al recibir un déficit nutricional (ej: faltan 30g pro),║
║              busca los K alimentos más similares matemáticamente    ║
║              en la base de datos nacional para sugerirlos.          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, "app", "data")
INS_JSON    = os.path.join(DATA_DIR, "alimentos_peru_ins.json")
OFF_JSON    = os.path.join(DATA_DIR, "alimentos_peru_off.json")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "app", "models", "ai_models")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "recomendador_knn.pkl")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "═" * 65)
print("  FASE 1: BUSINESS UNDERSTANDING")
print("═" * 65)
print("""
  Problema de negocio (Tesis Objetivo 3):
  El asistente necesita brindar recomendaciones *dinámicas* y precisas 
  basadas en la correlación de datos físicos y dietéticos del cliente.
  
  Solución:
  Implementar un recomendador Content-Based basado en Nearest Neighbors (KNN)
  que procese los macros faltantes del día del usuario y cruce esa matriz con 
  la Tabla Peruana de Composición de Alimentos (INS/CENAN 2017).
""")

print("═" * 65)
print("  FASE 2 & 3: DATA UNDERSTANDING Y DATA PREPARATION")
print("═" * 65)

data_comida = []
for file_path in [INS_JSON, OFF_JSON]:
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            data_comida.extend(data)
            print(f"  📂 Cargado {os.path.basename(file_path)}: {len(data)} registros.")
    else:
        print(f"  ⚠️ No se encontró: {file_path}")

if not data_comida:
    print("  ❌ ERROR: No hay datos para entrenar el modelo.")
    sys.exit(1)

df_raw = pd.DataFrame(data_comida)
print(f"\n  Total registros iniciales: {len(df_raw)}")

df = df_raw.copy()
columnas_ml = [
    "alimento", "calorias_100g", "proteina_100g", 
    "carbohindratos_100g", "grasas_100g"
]
for col in columnas_ml:
    if col not in df.columns:
        df[col] = 0.0

df = df[columnas_ml]
df.fillna(0, inplace=True)

df["nombre_lower"] = df["alimento"].str.lower().str.strip()
df.drop_duplicates(subset=["nombre_lower"], inplace=True)
df.drop(columns=["nombre_lower"], inplace=True)

df = df[(df["calorias_100g"] > 0) & (df["calorias_100g"] < 900)]

df.reset_index(drop=True, inplace=True)

print(f"  ✅ Dataset procesado y limpio: {len(df)} alimentos peruanos aptos.")
print(f"  Variables de entrada (Features): Calorías, Proteína, Carbos, Grasas")

features_matriz = ["calorias_100g", "proteina_100g", "carbohindratos_100g", "grasas_100g"]
X = df[features_matriz].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  ✅ Aplicado StandardScaler para normalización matemática de la matriz.")

print("\n" + "═" * 65)
print("  FASE 4: MODELING")
print("═" * 65)
print("""
  Algoritmo: Nearest Neighbors (No supervisado)
  Métrica Espacial: Distancia Coseno (Cosine Similarity)
  Justificación: La distancia coseno mide el ÁNGULO entre vectores. 
  Esto es ideal para nutrición, porque busca perfiles con *proporciones* 
  similares de macros, sin importar el peso total (volumen).
""")

knn = NearestNeighbors(n_neighbors=5, algorithm="brute", metric="cosine")
knn.fit(X_scaled)

print("  ⚙️  Modelo KNN ajustado sobre el espacio vectorial nutricional.")

print("\n" + "═" * 65)
print("  FASE 5: EVALUATION (Pruebas Dinámicas)")
print("═" * 65)

def evaluar_recomendacion(deficit_vector, descripcion):
    print(f"\n  Caso de prueba: {descripcion}")
    print(f"  Déficit buscado: Cal={deficit_vector[0]} | Prot={deficit_vector[1]}g | Carb={deficit_vector[2]}g | Gras={deficit_vector[3]}g")
    
    entrada_scaled = scaler.transform([deficit_vector])
    
    distancias, indices = knn.kneighbors(entrada_scaled, n_neighbors=3)
    
    print("  Resultados principales recomendados:")
    for i, idx in enumerate(indices[0]):
        row = df.iloc[idx]
        similitud = round((1 - distancias[0][i]) * 100, 1)
        nombre = row["alimento"][:40] + "..." if len(row["alimento"]) > 40 else row["alimento"]
        print(f"   {i+1}. {nombre:<40} (Similitud: {similitud}%)")
        print(f"      Cal: {row['calorias_100g']} | Pro: {row['proteina_100g']}g | Ca: {row['carbohindratos_100g']}g | Gr: {row['grasas_100g']}g")

evaluar_recomendacion([150, 30, 0, 5], "Cliente necesita PURA proteína baja en calorías y sin carbos.")
evaluar_recomendacion([350, 5, 60, 2], "Cliente necesita ENERGÍA/CARBOS para antes de entrenar.")
evaluar_recomendacion([400, 20, 20, 20], "Cliente busca una comida BALANCEADA completa.")

print("\n" + "═" * 65)
print("  FASE 6: DEPLOYMENT")
print("═" * 65)

objeto_exportable = {
    "modelo_knn": knn,
    "scaler": scaler,
    "df_alimentos": df,
    "version": "1.0_MINSA2017"
}

joblib.dump(objeto_exportable, OUTPUT_PATH)
size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)

print(f"  💾 Recomendador exportado en: {OUTPUT_PATH}")
print(f"     Tamaño del paquete     : {size_mb:.2f} MB")
print("\n  ✅ CRISP-DM completado con éxito.")
print("  Próximo paso: Integrarlo en ml_service.py para que el IA Engine lo use.\n")
