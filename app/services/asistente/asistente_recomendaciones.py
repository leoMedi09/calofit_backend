"""
Motor de recomendaciones de CaloFit.

Integra:
  - ML #1 Random Forest (Perfil de Adherencia) — prepara las 14 features desde BD.
  - ML #2 KNN Similitud Coseno — recomienda alimentos para cubrir déficit del día.
  - historial_recomendaciones — evita repetir platos sugeridos recientemente.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.core.utils import get_peru_date
from app.models.historial import ProgresoCalorias
from app.models.historial_recomendacion import HistorialRecomendacion
from app.services.ml_service import ml_perfil, ml_recomendador


class RecomendacionesHandler:
    """Orquesta las recomendaciones nutricionales usando KNN + historial."""

    # ── API pública ──────────────────────────────────────────────────────────

    def preparar_features_rf(self, perfil, db: Session) -> Dict[str, Any]:
        """
        Construye el dict de 14 features requeridas por el Random Forest.

        Features:
          Age, Gender_Enc, Weight(kg), Height_cm, BMI,
          Workout_Frequency(days/week), Session_Duration(hours),
          Calories_Burned, Cal_por_hora, Fat_Percentage,
          Water_Intake(liters), Avg_BPM, Resting_BPM,
          Workout_CARDIO/HIIT/STRENGTH/YOGA (one-hot)
        """
        hoy     = get_peru_date()
        semana  = hoy - timedelta(days=7)

        # Frecuencia de entrenamiento real (registros de progreso con quemadas > 0)
        registros_activos = (
            db.query(ProgresoCalorias)
            .filter(
                ProgresoCalorias.client_id == perfil.id,
                ProgresoCalorias.fecha >= semana,
                ProgresoCalorias.calorias_quemadas > 0,
            )
            .count()
        )

        # Promedio de calorías quemadas por día activo
        from sqlalchemy import func as _f
        avg_quemadas = (
            db.query(_f.avg(ProgresoCalorias.calorias_quemadas))
            .filter(
                ProgresoCalorias.client_id == perfil.id,
                ProgresoCalorias.fecha >= semana,
                ProgresoCalorias.calorias_quemadas > 0,
            )
            .scalar()
            or 0.0
        )

        edad        = (datetime.now().year - perfil.birth_date.year) if perfil.birth_date else 30
        peso        = float(getattr(perfil, "weight", None) or 70.0)
        altura_cm   = float(getattr(perfil, "height", None) or 170.0)
        session_h   = float(getattr(perfil, "session_duration", None) or 1.0)
        _wt_raw     = str(getattr(perfil, "workout_type", "") or "").strip()
        # Traducir valores en español a los nombres exactos del dataset Kaggle
        _WT_MAP = {
            "fuerza": "Strength", "pesas": "Strength", "musculacion": "Strength",
            "cardio": "Cardio",
            "hiit": "HIIT", "funcional": "HIIT", "crossfit": "HIIT",
            "yoga": "Yoga", "pilates": "Yoga",
        }
        workout_type = _WT_MAP.get(_wt_raw.lower(), _wt_raw)

        return {
            "age":           edad,
            "gender":        getattr(perfil, "gender", "M"),
            "weight":        peso,
            "height":        altura_cm,
            "workout_freq":  registros_activos,
            "session_hours": session_h,
            "calories":      round(float(avg_quemadas), 1),
            "fat_pct":       25.0,   # no disponible en BD → default seguro
            "water":         2.0,    # no disponible en BD → default seguro
            "avg_bpm":       140.0,  # no disponible en BD → default seguro
            "resting_bpm":   65.0,   # no disponible en BD → default seguro
            "workout_type":  workout_type,
        }

    def predecir_perfil(self, perfil, db: Session) -> tuple:
        """Devuelve (PERFIL_A|B|C, confianza_float) usando datos reales de la BD."""
        features = self.preparar_features_rf(perfil, db)
        return ml_perfil.predecir_perfil(features)

    def obtener_recomendaciones_knn(
        self,
        perfil,
        plan_hoy_data: dict,
        db: Session,
        n: int = 3,
        condiciones_dieta: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta el KNN Similitud Coseno con el déficit real del día.

        Vector entrada: [calorias_faltantes, proteinas_faltantes, carbos_faltantes, grasas_faltantes]
        Filtra alimentos del historial reciente y los bloqueados por el nutricionista.
        """
        hoy      = get_peru_date()
        progreso = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == perfil.id,
            ProgresoCalorias.fecha     == hoy,
        ).first()

        consumidas     = float((progreso.calorias_consumidas     if progreso else 0) or 0)
        prote_cons     = float((progreso.proteinas_consumidas     if progreso else 0) or 0)
        carbos_cons    = float((progreso.carbohidratos_consumidos if progreso else 0) or 0)
        grasas_cons    = float((progreso.grasas_consumidas        if progreso else 0) or 0)

        cal_meta   = float(plan_hoy_data.get("calorias_dia")    or 0)
        prot_meta  = float(plan_hoy_data.get("proteinas_g")     or 0)
        carbo_meta = float(plan_hoy_data.get("carbohidratos_g") or 0)
        grasa_meta = float(plan_hoy_data.get("grasas_g")        or 0)

        deficit = {
            "calorias":    max(0.0, cal_meta   - consumidas),
            "proteinas":   max(0.0, prot_meta  - prote_cons),
            "carbos":      max(0.0, carbo_meta - carbos_cons),
            "grasas":      max(0.0, grasa_meta - grasas_cons),
        }

        excluir = self._nombres_historial_reciente(perfil.id, db, dias=2)
        excluir += list(getattr(perfil, "forbidden_foods", None) or [])

        recommended = list(getattr(perfil, "recommended_foods", None) or [])
        # Fetch extra candidates: 4× when dietary filter active, 2× for boosting preferred
        _hay_dieta = bool(condiciones_dieta)
        n_fetch = n * 4 if _hay_dieta else (n * 2 if recommended else n)

        recs = ml_recomendador.obtener_recomendaciones(
            calorias_faltantes = deficit["calorias"],
            prote_faltante     = deficit["proteinas"],
            carbo_faltante     = deficit["carbos"],
            grasa_faltante     = deficit["grasas"],
            n_recomendaciones  = n_fetch,
            excluir_nombres    = excluir,
        )

        # Filtrar por restricciones dietéticas (Vegano / Vegetariano / Lactosa / Celíaco / Diabetes)
        if _hay_dieta:
            try:
                from app.services.recomendador_platos import _tokens_prohibidos
                _tok = _tokens_prohibidos(condiciones_dieta)
                if _tok:
                    recs = [
                        r for r in recs
                        if not any(
                            t in (r.get("alimento", "") or "").lower()
                            for t in _tok
                        )
                    ]
            except Exception:
                pass

        # Boost foods the nutritionist explicitly recommended → move to front
        if recommended and len(recs) > n:
            rec_norm = {r.lower() for r in recommended}
            preferred = [r for r in recs if any(
                token in (r.get("alimento") or "").lower() for token in rec_norm
            )]
            rest = [r for r in recs if r not in preferred]
            recs = (preferred + rest)[:n]
        else:
            recs = recs[:n]

        return recs

    def guardar_recomendacion(
        self,
        client_id: int,
        plato_id: Optional[int],
        nombre_plato: str,
        macros: dict,
        momento_dia: str,
        db: Session,
    ) -> None:
        """Persiste en historial_recomendaciones para evitar repeticiones futuras."""
        db.add(
            HistorialRecomendacion(
                client_id        = client_id,
                plato_id         = plato_id,
                nombre_plato     = nombre_plato[:200],
                calorias         = macros.get("calorias", 0),
                proteinas_g      = macros.get("proteinas_g", 0),
                carbohidratos_g  = macros.get("carbohidratos_g", 0),
                grasas_g         = macros.get("grasas_g", 0),
                momento_dia      = momento_dia,
                fue_consumido    = False,
            )
        )
        db.commit()

    # ── Privados ─────────────────────────────────────────────────────────────

    def _nombres_historial_reciente(
        self, client_id: int, db: Session, dias: int = 2
    ) -> List[str]:
        """Devuelve nombres de platos ya recomendados en las últimas N días."""
        desde = datetime.now() - timedelta(days=dias)
        rows  = (
            db.query(HistorialRecomendacion.nombre_plato)
            .filter(
                HistorialRecomendacion.client_id  == client_id,
                HistorialRecomendacion.created_at >= desde,
            )
            .all()
        )
        return [r[0] for r in rows if r[0]]


recomendaciones_handler = RecomendacionesHandler()
