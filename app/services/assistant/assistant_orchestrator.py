"""
Orquestador del Asistente CaloFit (nueva capa modular).

Flujo:
  1. Construir contexto del cliente (ContextBuilder)
  2. Predecir perfil ML (MLServiceWrapper)
  3. Clasificar intención (IntentClassifier)
  4. Despachar al handler correspondiente
  5. Retornar dict normalizado (ResponseParser)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.services.ai.llm_service import LLMService
from app.services.ai.ml_service import MLServiceWrapper
from app.services.assistant.intent_classifier import (
    IntentClassifier,
    INTENT_NUTRICION,
    INTENT_EJERCICIO,
    INTENT_INTEGRADO,
    INTENT_PLAN,
    INTENT_PROGRESO,
    INTENT_CHAT,
)
from app.services.assistant.context_builder import ContextBuilder
from app.services.assistant.response_parser import ResponseParser
from app.services.assistant.handlers.nutrition_handler import NutritionHandler
from app.services.assistant.handlers.exercise_handler import ExerciseHandler
from app.services.assistant.handlers.integrated_handler import IntegratedHandler
from app.services.assistant.handlers.chat_handler import ChatHandler

logger = logging.getLogger(__name__)


class AssistantOrchestrator:
    """
    Punto de entrada único para el nuevo módulo de asistente.

    Compatible con el endpoint existente `/asistente/consultar`.
    Puede coexistir con `asistente_service.py` mientras se migra gradualmente.

    Uso:
        orchestrator = AssistantOrchestrator(db)
        respuesta = await orchestrator.consultar(
            mensaje="¿Qué puedo cenar hoy?",
            client_id=55,
        )
    """

    def __init__(self, db: Session) -> None:
        self.db = db
        self._llm: Optional[LLMService] = None
        self._ml  = MLServiceWrapper()
        self._intent = IntentClassifier()

    # ──────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────

    async def consultar(
        self,
        mensaje: str,
        client_id: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Procesa el mensaje del cliente y retorna una respuesta estructurada.

        Args:
            mensaje:   texto enviado por el cliente
            client_id: ID del cliente en la BD
            metadata:  dict opcional con info extra (device, timestamp, etc.)

        Returns:
            dict normalizado compatible con Flutter (ver ResponseParser)
        """
        llm = self._obtener_llm()

        # 1. Construir contexto
        ctx_builder = ContextBuilder(self.db)
        contexto = ctx_builder.construir(client_id)

        # 2. Predecir perfil ML
        perfil_features = self._extraer_features(contexto)
        perfil_ml, confianza_ml = self._ml.predecir_perfil(perfil_features)
        contexto["perfil_ml"]          = perfil_ml
        contexto["perfil_ml_confianza"] = confianza_ml

        # 3. Clasificar intención (heurística, sin LLM por defecto)
        intent = self._intent.clasificar_heuristico(mensaje)
        contexto["intent"] = intent

        logger.info(
            "AssistantOrchestrator: client=%s, intent=%s, perfil=%s (%.0f%%)",
            client_id, intent, perfil_ml, confianza_ml * 100,
        )

        # 4. Despachar al handler
        try:
            handler = self._handler_para(intent, llm)
            return await handler.manejar(
                mensaje=mensaje,
                contexto=contexto,
                client_id=client_id,
            )
        except Exception as exc:
            logger.error("AssistantOrchestrator.consultar error: %s", exc, exc_info=True)
            return ResponseParser.texto(
                mensaje=(
                    "Lo siento, ocurrió un error procesando tu consulta. "
                    "Por favor intenta de nuevo en unos segundos."
                )
            )

    # ──────────────────────────────────────────────────────────────────
    # Privados
    # ──────────────────────────────────────────────────────────────────

    def _obtener_llm(self) -> Optional[LLMService]:
        if self._llm is not None:
            return self._llm
        try:
            self._llm = LLMService()
        except Exception as exc:
            logger.warning("LLMService no disponible: %s", exc)
            self._llm = None
        return self._llm

    def _handler_para(self, intent: str, llm) -> Any:
        kwargs = {"db": self.db, "llm_service": llm, "ml_service": self._ml}
        mapping = {
            INTENT_NUTRICION: NutritionHandler,
            INTENT_EJERCICIO: ExerciseHandler,
            INTENT_INTEGRADO: IntegratedHandler,
            INTENT_PLAN:      IntegratedHandler,
            INTENT_PROGRESO:  NutritionHandler,
            INTENT_CHAT:      ChatHandler,
        }
        cls = mapping.get(intent, ChatHandler)
        return cls(**kwargs)

    @staticmethod
    def _extraer_features(contexto: Dict[str, Any]) -> Dict[str, Any]:
        """Mapea el contexto del cliente a las features del RF."""
        perfil = contexto.get("perfil", {})
        historial = contexto.get("historial_reciente", [])

        # Promedios de los últimos 7 días
        avg_kcal = 0.0
        if historial:
            avg_kcal = sum(r.get("kcal_consumidas", 0) for r in historial) / len(historial)

        return {
            "Weight (kg)":              perfil.get("peso_kg") or 70,
            "Height (m)":               (perfil.get("altura_cm") or 170) / 100,
            "Age":                      perfil.get("edad") or 30,
            "Gender":                   1 if perfil.get("genero") == "M" else 0,
            "Session_Duration (hours)": perfil.get("session_duration_h") or 1.0,
            "Workout_Frequency (days/week)": 3,
            "Workout_Type":             perfil.get("workout_type") or "Cardio",
            "avg_kcal_7d":              avg_kcal,
        }
