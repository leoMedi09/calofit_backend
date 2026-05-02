"""
Helpers para obtener el plan nutricional activo del cliente.

Exporta:
  obtener_plan_hoy(perfil, edad, db) → (plan_maestro, plan_hoy_data, usa_fallback)
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.core.utils import get_peru_date
from app.models.nutricion import PlanDiario, PlanNutricional


def obtener_plan_hoy(perfil, edad: int, db: Session):
    """
    Devuelve (plan_maestro, plan_hoy_data dict, usa_fallback bool).

    Si el cliente no tiene plan validado, calcula TMB + TDEE con Mifflin-St Jeor
    y devuelve un objeto PlanFallback compatible con los campos que usa el asistente.
    """
    plan_maestro = (
        db.query(PlanNutricional)
        .filter(PlanNutricional.client_id == perfil.id)
        .order_by(PlanNutricional.fecha_creacion.desc())
        .first()
    )

    if not plan_maestro:
        from app.services.calculador_dieta import calculador_dieta

        _nivel_map = {
            "Sedentario": 1.2, "Ligero": 1.375, "Moderado": 1.55,
            "Intenso": 1.725, "Muy intenso": 1.9,
        }
        genero     = 1 if getattr(perfil, "gender", "M") == "M" else 2
        nivel      = _nivel_map.get(getattr(perfil, "activity_level", "Moderado"), 1.55)
        macros     = calculador_dieta.calcular_macros(
            genero=genero, edad=edad,
            peso=float(perfil.weight or 70), talla=float(perfil.height or 170),
            nivel_actividad=nivel,
            objetivo=getattr(perfil, "goal", "Mantenimiento") or "Mantenimiento",
        )

        class _PlanFallback:
            def __init__(self, objetivo):
                self.objetivo       = objetivo
                self.status         = "calculado_ia"
                self.id             = None
                self.fecha_creacion = datetime.now()

        return (
            _PlanFallback(objetivo=perfil.goal),
            {
                "calorias_dia":              macros.get("calorias_objetivo", 2000),
                "proteinas_g":               macros["proteinas_g"],
                "carbohidratos_g":           macros["carbohidratos_g"],
                "grasas_g":                  macros["grasas_g"],
                "sugerencia_entrenamiento_ia": "Plan calculado automáticamente por IA",
            },
            True,
        )

    dia_semana = get_peru_date().isoweekday()
    plan_hoy   = (
        db.query(PlanDiario)
        .filter(PlanDiario.plan_id == plan_maestro.id, PlanDiario.dia_numero == dia_semana)
        .first()
        or db.query(PlanDiario).filter(PlanDiario.plan_id == plan_maestro.id).first()
    )
    if not plan_hoy:
        raise ValueError("Tu plan nutricional está incompleto.")

    return (
        plan_maestro,
        {
            "calorias_dia":              plan_hoy.calorias_dia,
            "proteinas_g":               plan_hoy.proteinas_g,
            "carbohidratos_g":           plan_hoy.carbohidratos_g,
            "grasas_g":                  plan_hoy.grasas_g,
            "sugerencia_entrenamiento_ia": plan_hoy.sugerencia_entrenamiento_ia,
        },
        False,
    )
