"""
Helpers para obtener el plan nutricional activo del cliente.

Exporta:
  obtener_plan_hoy(perfil, edad, db) → (plan_maestro, plan_hoy_data, usa_fallback)

Lógica de recalculo dinámico:
  - Si el usuario NO tiene plan en BD           → calcula Mifflin-St Jeor en tiempo real.
  - Si el plan fue auto-generado (≠ 'validado') → siempre recalcula (refleja peso/actividad actual).
  - Si el plan es 'validado' por nutricionista  → usa valores del nutricionista, PERO si el
    cliente cambió su `goal` respecto al plan, recalcula (el objetivo del cliente es señal explícita).
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.core.utils import get_peru_date
from app.models.nutricion import PlanDiario, PlanNutricional


# La normalización de objetivos se centralizó en objetivo_utils.
# _OBJ_CANON fue eliminado — los 5 valores controlados del frontend
# se mapean ahora mediante normalizar_objetivo() a DEFICIT / MANTENIMIENTO / SUPERAVIT.
from app.core.objetivo_utils import normalizar_objetivo as _norm_obj

_NIVEL_MAP = {
    "Sedentario": 1.2, "Ligero": 1.375, "Moderado": 1.55,
    "Intenso": 1.725, "Muy intenso": 1.9,
}


def _calcular_macros_dinamicos(perfil, edad: int) -> dict:
    """
    Calcula calorías y macros en tiempo real desde el perfil actual.
    Usa Mifflin-St Jeor vía ia_engine (sin depender de la BD de planes).
    """
    from app.services.ia_service import ia_engine
    from app.core.macros_diarios import macros_desde_calorias_peso_objetivo

    obj    = (getattr(perfil, "goal", "Mantenimiento") or "Mantenimiento").strip()
    nivel  = _NIVEL_MAP.get(getattr(perfil, "activity_level", "Moderado"), 1.55)
    genero = 1 if str(getattr(perfil, "gender", "M")).upper() == "M" else 2
    peso   = float(perfil.weight or 70)
    talla  = float(perfil.height or 170)

    cal = ia_engine.calcular_requerimiento(genero, edad, peso, talla, nivel, obj)
    m   = macros_desde_calorias_peso_objetivo(cal, obj, peso)
    return {
        "calorias_dia":    round(cal),
        "proteinas_g":     m["proteinas_g"],
        "carbohidratos_g": m["carbohidratos_g"],
        "grasas_g":        m["grasas_g"],
    }


def obtener_plan_hoy(perfil, edad: int, db: Session):
    """
    Devuelve (plan_maestro, plan_hoy_data, usa_fallback bool).

    plan_hoy_data contiene siempre los macros alineados con el perfil actual:
      - Si goal cambió respecto al plan guardado → recalcula.
      - Si el plan no es 'validado' → recalcula.
      - Sin plan en BD → fallback dinámico (sin crash).
    """
    plan_maestro = (
        db.query(PlanNutricional)
        .filter(PlanNutricional.client_id == perfil.id)
        .order_by(PlanNutricional.fecha_creacion.desc())
        .first()
    )

    # ── Sin plan en BD → fallback dinámico ───────────────────────────────────
    if not plan_maestro:
        macros = _calcular_macros_dinamicos(perfil, edad)

        class _PlanFallback:
            def __init__(self, objetivo):
                self.objetivo       = objetivo
                self.status         = "calculado_ia"
                self.id             = None
                self.fecha_creacion = datetime.now()

        return (
            _PlanFallback(objetivo=perfil.goal),
            {
                **macros,
                "sugerencia_entrenamiento_ia": "Plan calculado automáticamente por IA",
            },
            True,
        )

    # ── Plan existe: leer el día de la semana ─────────────────────────────────
    dia_semana = get_peru_date().isoweekday()
    plan_hoy   = (
        db.query(PlanDiario)
        .filter(PlanDiario.plan_id == plan_maestro.id, PlanDiario.dia_numero == dia_semana)
        .first()
        or db.query(PlanDiario).filter(PlanDiario.plan_id == plan_maestro.id).first()
    )
    if not plan_hoy:
        raise ValueError("Tu plan nutricional está incompleto.")

    # Datos base del plan guardado (usados para sugerencia de entrenamiento)
    plan_base = {
        "calorias_dia":              plan_hoy.calorias_dia,
        "proteinas_g":               plan_hoy.proteinas_g,
        "carbohidratos_g":           plan_hoy.carbohidratos_g,
        "grasas_g":                  plan_hoy.grasas_g,
        "sugerencia_entrenamiento_ia": plan_hoy.sugerencia_entrenamiento_ia,
    }

    # ── Decidir si recalcular calorías/macros ────────────────────────────────
    status_plan  = (plan_maestro.status or "").strip().lower()

    # Detectar cambio de objetivo usando normalización canónica — cubre los
    # 5 valores controlados del frontend, incluyendo ganar_leve y perder_leve
    # que el antiguo _OBJ_CANON no tenía.
    plan_obj_canon  = _norm_obj(plan_maestro.objetivo)
    perf_obj_canon  = _norm_obj(perfil.goal)
    objetivo_cambio = plan_obj_canon != perf_obj_canon

    # Recalcular si:
    #   A) El plan NO fue validado por nutricionista (fue generado por IA)
    #   B) El objetivo del perfil cambió respecto al plan (señal explícita del cliente)
    necesita_recalculo = (status_plan != "validado") or objetivo_cambio

    if necesita_recalculo:
        macros_din = _calcular_macros_dinamicos(perfil, edad)
        return (
            plan_maestro,
            {
                **macros_din,
                "sugerencia_entrenamiento_ia": plan_hoy.sugerencia_entrenamiento_ia,
            },
            False,
        )

    return (plan_maestro, plan_base, False)
