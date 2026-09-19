"""
╔══════════════════════════════════════════════════════════════════════╗
║     CaloFit — ML Service                                            ║
║                                                                      ║
║  Modelos activos (cargados al iniciar el servidor):                 ║
║    ml_perfil       → predecir_perfil(datos) → PERFIL_A/B/C         ║
║    ml_recomendador → obtener_recomendaciones() → [alimentos KNN]   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import joblib
import numpy as np
from typing import List, Optional

from app.core.alimentos_ux_filters import es_alimento_bloqueado_ia, nombre_coincide_exclusion
from app.core.logging_config import get_logger
from app.core.utils import get_peru_now

logger = get_logger("ml_service")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "ai_models")
PERFIL_MODEL = os.path.join(MODELS_DIR, "perfil_adherencia.pkl")
RECOMENDADOR_MODEL = os.path.join(MODELS_DIR, "recomendador_knn.pkl")


class ClasificadorPerfil:
    """
    Random Forest entrenado con Gym Members Exercise Dataset (Kaggle).
    Clasifica al cliente en PERFIL_A / PERFIL_B / PERFIL_C para que
    el Asistente personalice el tono de sus respuestas vía el LLM.
    """

    TONO_ASISTENTE = {
        "PERFIL_A": (
            "El cliente es DISCIPLINADO (Perfil A). Usa tono desafiante y técnico. "
            "Propón metas ambiciosas y datos avanzados. Reconoce su esfuerzo y empújalo más."
        ),
        "PERFIL_B": (
            "El cliente está EN DESARROLLO (Perfil B). Usa tono motivador y positivo. "
            "Celebra avances, da pasos concretos. Ayúdalo a construir consistencia."
        ),
        "PERFIL_C": (
            "El cliente NECESITA GUÍA (Perfil C). Usa tono empático, simple y alentador. "
            "Propón objetivos pequeños y fáciles. Evita datos complejos. "
            "Prioriza el hábito sobre la perfección."
        ),
    }

    def __init__(self):
        self._rf = None
        self._features = None
        self._workout_types = None
        self._label_map = {1: "PERFIL_C", 2: "PERFIL_B", 3: "PERFIL_A"}
        self._activo = False
        self._cargar_modelo()

    def _cargar_modelo(self):
        if os.path.exists(PERFIL_MODEL):
            try:
                data = joblib.load(PERFIL_MODEL)
                if hasattr(data, "modelo"):
                    self._rf = data.modelo
                    self._features = data.features
                    self._workout_types = data.workout_types
                else:
                    self._rf = data["rf_model"]
                    self._features = data["features"]
                    self._workout_types = data["workout_types"]
                self._activo = True
                print("[ML Perfil] perfil_adherencia.pkl cargado - Personalizacion activa.")
            except Exception as e:
                print(f"[ML Perfil] Error: {e} - usando PERFIL_B por defecto.")
        else:
            print("[ML Perfil] .pkl no encontrado.\n   Ejecuta: python scripts/entrenar_perfil_adherencia.py")

    def predecir_perfil(self, datos_cliente: dict) -> tuple:
        """
        Predice el perfil de adherencia del cliente.

        Parámetros del dict (todos opcionales con defaults sensibles):
          age, gender, weight, height, workout_freq, session_hours,
          calories, fat_pct, water, avg_bpm, resting_bpm, workout_type

        Retorna: ("PERFIL_A"|"PERFIL_B"|"PERFIL_C", confianza_float)
        """
        if not (self._activo and self._rf is not None):
            return "PERFIL_B", 0.0

        try:
            import pandas as pd

            datos = datos_cliente

            gender_enc = 1 if str(datos.get("gender", "M")).upper() in ["M", "MALE"] else 0
            height_cm = float(datos.get("height", 170))
            weight_kg = float(datos.get("weight", 70))
            bmi = weight_kg / ((height_cm / 100) ** 2)

            sess_h = float(datos.get("session_hours", 1)) or 1
            cal = float(datos.get("calories", 500))
            cal_hora = cal / sess_h

            wt_data = {col: 0 for col in self._workout_types}
            wt_col = f"Workout_{datos.get('workout_type', '')}"
            if wt_col in wt_data:
                wt_data[wt_col] = 1

            row = {
                "Age": float(datos.get("age", 30)),
                "Gender_Enc": gender_enc,
                "Weight (kg)": weight_kg,
                "Height_cm": height_cm,
                "BMI": round(bmi, 2),
                "Workout_Frequency (days/week)": float(datos.get("workout_freq", 3)),
                "Session_Duration (hours)": float(datos.get("session_hours", 1)),
                "Calories_Burned": cal,
                "Cal_por_hora": cal_hora,
                "Fat_Percentage": float(datos.get("fat_pct", 25)),
                "Water_Intake (liters)": float(datos.get("water", 2)),
                "Avg_BPM": float(datos.get("avg_bpm", 140)),
                "Resting_BPM": float(datos.get("resting_bpm", 60)),
                **wt_data,
            }

            df_input = pd.DataFrame([row])[self._features]
            pred = self._rf.predict(df_input.values)[0]
            proba = self._rf.predict_proba(df_input.values)[0]
            conf = round(float(max(proba)) * 100, 1)
            perfil = self._label_map[pred]

            print(f"[ML Perfil] -> {perfil} ({conf}% confianza)")
            return perfil, conf

        except Exception as e:
            print(f"[ML Perfil] Error prediccion: {e} - usando PERFIL_B.")
            return "PERFIL_B", 0.0

    def predecir_perfil_desde_progreso(
        self,
        registros_semana: int = 3,
        adherencia_pct: float = 60.0,
        edad: int = 30,
        genero: str = "M",
        peso: float = 70.0,
        altura: float = 170.0,
    ) -> tuple:
        """
        Versión ligera para usar con datos reales de la BD CaloFit.
        Estima el perfil a partir de ProgresoCalorias + datos del Client.
        """
        datos = {
            "age": edad,
            "gender": genero,
            "weight": peso,
            "height": altura,
            "workout_freq": registros_semana,
            "session_hours": round(0.5 + registros_semana * 0.2, 1),
            "calories": round(400 + adherencia_pct * 10, 0),
            "fat_pct": round(max(10, 35 - adherencia_pct * 0.3), 1),
            "water": round(1.5 + registros_semana * 0.2, 1),
            "avg_bpm": 140,
            "resting_bpm": 65,
            "workout_type": "",
        }
        return self.predecir_perfil(datos)

    def get_tono_asistente(self, perfil: str) -> str:
        """Bloque de texto a inyectar en el prompt del LLM para personalizar el tono."""
        return self.TONO_ASISTENTE.get(perfil, self.TONO_ASISTENTE["PERFIL_B"])

    @property
    def modelo_activo(self) -> bool:
        return self._activo


class RecomendadorAlimentosKNN:
    """
    KNN con Similitud Coseno entrenado con datos del MINSA/CENAN 2017.
    Recomienda los alimentos matemáticamente más ideales para cubrir
    el déficit de macronutrientes del usuario en su día actual.
    """

    def __init__(self):
        self._knn = None
        self._scaler = None
        self._df = None
        self._activo = False
        self._cargar_modelo()

    def _cargar_modelo(self):
        if os.path.exists(RECOMENDADOR_MODEL):
            try:
                paquete = joblib.load(RECOMENDADOR_MODEL)
                self._knn = paquete["modelo_knn"]
                self._scaler = paquete["scaler"]
                self._df = paquete["df_alimentos"]
                self._activo = True
                print("[ML Recomendador] recomendador_knn.pkl cargado.")
            except Exception as e:
                print(f"[ML Recomendador] Error: {e} - modelo inactivo.")
        else:
            print("[ML Recomendador] .pkl no encontrado.\n   Ejecuta: python scripts/entrenar_recomendador.py")

    _OMEGA3_ESPECIES = frozenset(
        {
            "caballa",
            "lisa",
            "mero",
            "ojo de uva",
            "tollo",
            "cabrilla",
            "anchoveta",
            "trucha",
            "salmon",
            "sardina",
        }
    )

    _MMR_LAMBDA = 0.5

    _MMR_TEMPERATURE = 0.20

    _FEATURES_COLS = ["calorias_100g", "proteina_100g", "carbohindratos_100g", "grasas_100g"]

    def obtener_recomendaciones(
        self,
        calorias_faltantes: float,
        prote_faltante: float,
        carbo_faltante: float,
        grasa_faltante: float,
        n_recomendaciones: int = 3,
        excluir_nombres: Optional[List[str]] = None,
        contexto: Optional[str] = None,
    ) -> list:
        """
        Recibe el déficit del día (kcal + macros) y retorna una lista de
        alimentos peruanos óptimos ordenados por similitud coseno.

        Filtra nombres bloqueados (UX), sugerencias recientes del usuario y
        selecciona el top-N final por MMR (relevancia vs. diversidad).
        """
        if not self._activo:
            return []

        calorias_faltantes = max(50, min(calorias_faltantes, 900))
        prote_faltante = max(0, prote_faltante)
        carbo_faltante = max(0, carbo_faltante)
        grasa_faltante = max(0, grasa_faltante)

        vector = [calorias_faltantes, prote_faltante, carbo_faltante, grasa_faltante]
        excluir_nombres = excluir_nombres or []

        try:
            if self._df is None or len(self._df) == 0:
                return []
            vector_scaled = self._scaler.transform([vector])
            n_pool = min(max(n_recomendaciones * 20, 60), len(self._df))
            n_pool = max(1, n_pool)
            distancias, idx = self._knn.kneighbors(vector_scaled, n_neighbors=n_pool)
            vecs_scaled = self._scaler.transform(self._df.iloc[idx[0]][self._FEATURES_COLS].values)

            candidatos = []
            for i, row_idx in enumerate(idx[0]):
                row = self._df.iloc[row_idx]
                nombre = str(row["alimento"])
                if es_alimento_bloqueado_ia(nombre):
                    continue
                if nombre_coincide_exclusion(nombre, excluir_nombres):
                    continue
                similitud = round((1 - distancias[0][i]) * 100, 1)
                candidatos.append(
                    {
                        "alimento": nombre,
                        "calorias_100g": row["calorias_100g"],
                        "proteina_100g": row["proteina_100g"],
                        "carbohindratos_100g": row["carbohindratos_100g"],
                        "grasas_100g": row["grasas_100g"],
                        "similitud": similitud,
                        "_vec": vecs_scaled[i],
                    }
                )

            if not candidatos:
                return []

            _ctx = (contexto or "").lower()
            _omega_activo = any(kw in _ctx for kw in ("omega", "marino", "pescado", "mariscos")) or grasa_faltante > 3.0
            if _omega_activo:
                for c in candidatos:
                    if any(esp in c["alimento"].lower() for esp in self._OMEGA3_ESPECIES):
                        c["similitud"] = min(99.9, round(c["similitud"] * 1.25, 1))

            _bucket_temporal = int(get_peru_now().timestamp() // 60)
            _seed = (
                _bucket_temporal * 10007
                + int(calorias_faltantes)
                + int(prote_faltante * 10)
                + int(carbo_faltante * 10)
                + int(grasa_faltante * 10)
                + len(excluir_nombres) * 17
            ) % (2**31)
            _rng = np.random.default_rng(_seed)

            vistos = set()
            resultados: list = []
            restantes = candidatos.copy()
            while restantes and len(resultados) < n_recomendaciones:
                scores, cands = [], []
                for c in restantes:
                    key = c["alimento"].lower().strip()
                    if key in vistos:
                        continue
                    rel = c["similitud"] / 100.0
                    if not resultados:
                        div_penalty = 0.0
                    else:
                        div_penalty = max(
                            float(
                                np.dot(c["_vec"], s["_vec"])
                                / (np.linalg.norm(c["_vec"]) * np.linalg.norm(s["_vec"]) + 1e-9)
                            )
                            for s in resultados
                        )
                    scores.append(self._MMR_LAMBDA * rel - (1 - self._MMR_LAMBDA) * div_penalty)
                    cands.append(c)
                if not cands:
                    break
                if self._MMR_TEMPERATURE <= 0:
                    elegido = cands[int(np.argmax(scores))]
                else:
                    scores_arr = np.array(scores)
                    logits = (scores_arr - scores_arr.max()) / self._MMR_TEMPERATURE
                    probs = np.exp(logits)
                    probs /= probs.sum()
                    elegido = cands[_rng.choice(len(cands), p=probs)]
                resultados.append(elegido)
                vistos.add(elegido["alimento"].lower().strip())
                restantes.remove(elegido)

            for c in resultados:
                c.pop("_vec", None)

            if resultados:
                logger.info("1ra opcion: %s (%.1f%%)", resultados[0]["alimento"], resultados[0]["similitud"])
            return resultados

        except Exception as e:
            logger.error("Error en inferencia KNN: %s", e)
            return []

    @property
    def modelo_activo(self) -> bool:
        return self._activo


ml_perfil = ClasificadorPerfil()
ml_recomendador = RecomendadorAlimentosKNN()
