"""
Construye el contexto del cliente para pasarlo al LLM y handlers.
"""
from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Reúne el contexto relevante del cliente desde la BD.

    Retorna un dict plano que los handlers pueden pasar al LLM
    sin necesidad de hacer más queries.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    def construir(self, client_id: int) -> Dict[str, Any]:
        """
        Construye el contexto completo del cliente.

        Returns:
            dict con claves: perfil, plan_hoy, progreso_hoy, historial_reciente, metas
        """
        ctx: Dict[str, Any] = {
            "client_id": client_id,
            "perfil": self._perfil(client_id),
            "plan_hoy": self._plan_hoy(client_id),
            "progreso_hoy": self._progreso_hoy(client_id),
            "historial_reciente": self._historial_reciente(client_id, dias=7),
            "metas": self._metas(client_id),
        }
        return ctx

    # ──────────────────────────────────────────────────────────────────
    # Privados
    # ──────────────────────────────────────────────────────────────────

    def _perfil(self, client_id: int) -> Dict[str, Any]:
        try:
            from app.models import Client
            c = self.db.query(Client).filter(Client.id == client_id).first()
            if not c:
                return {}
            edad = None
            if c.birth_date:
                edad = (date.today() - c.birth_date).days // 365
            return {
                "nombre": f"{c.first_name or ''} {c.last_name_paternal or ''}".strip(),
                "genero": c.gender,
                "edad": edad,
                "peso_kg": c.weight,
                "altura_cm": c.height,
                "nivel_actividad": c.activity_level,
                "objetivo": c.goal,
                "workout_type": c.workout_type,
                "session_duration_h": c.session_duration,
                "condiciones_medicas": c.medical_conditions or [],
                "lista_negra": c.forbidden_foods or [],
                "lista_blanca": c.recommended_foods or [],
                "nota_nutri": c.nutri_weekly_note,
            }
        except Exception as exc:
            logger.warning("ContextBuilder._perfil error: %s", exc)
            return {}

    def _plan_hoy(self, client_id: int) -> Optional[Dict[str, Any]]:
        try:
            from app.models import PlanNutricional, PlanDiario
            from sqlalchemy import func
            hoy = date.today()
            dia_semana = hoy.weekday()  # 0=lunes

            plan = (
                self.db.query(PlanNutricional)
                .filter(
                    PlanNutricional.cliente_id == client_id,
                    PlanNutricional.estado == "validado",
                )
                .order_by(PlanNutricional.id.desc())
                .first()
            )
            if not plan:
                return None

            plan_diario = (
                self.db.query(PlanDiario)
                .filter(
                    PlanDiario.plan_id == plan.id,
                    PlanDiario.dia_semana == dia_semana,
                )
                .first()
            )
            if not plan_diario:
                return None

            return {
                "plan_id": plan.id,
                "dia_semana": dia_semana,
                "kcal_objetivo": getattr(plan_diario, "calorias_objetivo", None),
                "sugerencia_entrenamiento": getattr(plan_diario, "sugerencia_entrenamiento", None),
            }
        except Exception as exc:
            logger.warning("ContextBuilder._plan_hoy error: %s", exc)
            return None

    def _progreso_hoy(self, client_id: int) -> Dict[str, Any]:
        try:
            from app.models import ProgresoCalorias
            hoy = date.today()
            reg = (
                self.db.query(ProgresoCalorias)
                .filter(
                    ProgresoCalorias.cliente_id == client_id,
                    ProgresoCalorias.fecha == hoy,
                )
                .first()
            )
            if not reg:
                return {"kcal_consumidas": 0, "kcal_quemadas": 0}
            return {
                "kcal_consumidas": float(reg.calorias_consumidas or 0),
                "kcal_quemadas": float(reg.calorias_quemadas or 0),
            }
        except Exception as exc:
            logger.warning("ContextBuilder._progreso_hoy error: %s", exc)
            return {"kcal_consumidas": 0, "kcal_quemadas": 0}

    def _historial_reciente(self, client_id: int, dias: int = 7) -> List[Dict[str, Any]]:
        try:
            from app.models import ProgresoCalorias
            from sqlalchemy import desc
            registros = (
                self.db.query(ProgresoCalorias)
                .filter(ProgresoCalorias.cliente_id == client_id)
                .order_by(desc(ProgresoCalorias.fecha))
                .limit(dias)
                .all()
            )
            return [
                {
                    "fecha": str(r.fecha),
                    "kcal_consumidas": float(r.calorias_consumidas or 0),
                    "kcal_quemadas": float(r.calorias_quemadas or 0),
                }
                for r in registros
            ]
        except Exception as exc:
            logger.warning("ContextBuilder._historial_reciente error: %s", exc)
            return []

    def _metas(self, client_id: int) -> Dict[str, Any]:
        try:
            from app.models import MetaUsuario
            meta = (
                self.db.query(MetaUsuario)
                .filter(MetaUsuario.client_id == client_id)
                .first()
            )
            if not meta:
                return {}
            return {
                "calorias_objetivo": getattr(meta, "calorias_objetivo", None),
                "proteina_objetivo": getattr(meta, "proteina_objetivo", None),
                "carbohidratos_objetivo": getattr(meta, "carbohidratos_objetivo", None),
                "grasas_objetivo": getattr(meta, "grasas_objetivo", None),
            }
        except Exception as exc:
            logger.warning("ContextBuilder._metas error: %s", exc)
            return {}
