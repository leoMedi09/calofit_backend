"""
Clasificador de intención del mensaje del usuario.
"""
from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# ─── Categorías válidas ────────────────────────────────────────────────────────
INTENT_NUTRICION  = "nutricion"
INTENT_EJERCICIO  = "ejercicio"
INTENT_INTEGRADO  = "integrado"
INTENT_PLAN       = "plan"
INTENT_PROGRESO   = "progreso"
INTENT_CHAT       = "chat"

_INTENTS_VALIDOS = {
    INTENT_NUTRICION, INTENT_EJERCICIO, INTENT_INTEGRADO,
    INTENT_PLAN, INTENT_PROGRESO, INTENT_CHAT,
}

# ─── Reglas heurísticas (fallback sin LLM) ───────────────────────────────────
_REGEX_NUTRICION = re.compile(
    r"\b(comer|comida|alimento|plato|desayuno|almuerzo|cena|merienda|snack|"
    r"proteína|carbohidrato|grasa|caloría|kcal|registr[ao]|ingeri|come|comí)\b",
    re.IGNORECASE,
)
_REGEX_EJERCICIO = re.compile(
    r"\b(ejercicio|rutina|entrenamiento|serie|repetición|repeticion|rep|"
    r"gimnasio|workout|cardio|fuerza|HIIT|estiram|calentamiento|músculo|musculo|"
    r"levantar|pesa|corr[ei]|bicicleta|natación|nadar|"
    r"sentadilla|dominada|flexion|press|curl|remo|peso muerto|"
    r"entrené|entrenaste|hice|hiciste)\b"
    r"|(?<!a la )(?<!la )\bplancha\b",  # 'plancha' solo si NO es 'a la plancha' (cocción)
    re.IGNORECASE,
)
_REGEX_PLAN = re.compile(
    r"\b(plan|planif|semana|semanal|dieta|programa|objetivo|meta)\b",
    re.IGNORECASE,
)
_REGEX_PROGRESO = re.compile(
    r"\b(progress|progress|avance|historial|cuánto|cuanto|evolución|evolucion|"
    r"resumen|análisis|analisis|seguimiento|balance|peso|imc|bmi)\b",
    re.IGNORECASE,
)


class IntentClassifier:
    """
    Clasifica la intención del mensaje.

    Primero aplica heurísticas locales (O(1), sin LLM).
    Si el LLMService está disponible y la heurística no es concluyente,
    delega al LLM para mayor precisión.
    """

    def __init__(self, llm_service=None) -> None:
        self._llm = llm_service

    def clasificar_heuristico(self, mensaje: str) -> str:
        """Clasificación rápida sin LLM."""
        tiene_nutri  = bool(_REGEX_NUTRICION.search(mensaje))
        tiene_ej     = bool(_REGEX_EJERCICIO.search(mensaje))
        tiene_plan   = bool(_REGEX_PLAN.search(mensaje))
        tiene_prog   = bool(_REGEX_PROGRESO.search(mensaje))

        if tiene_nutri and tiene_ej:
            return INTENT_INTEGRADO
        if tiene_plan:
            return INTENT_PLAN
        if tiene_prog:
            return INTENT_PROGRESO
        if tiene_nutri:
            return INTENT_NUTRICION
        if tiene_ej:
            return INTENT_EJERCICIO
        return INTENT_CHAT

    async def clasificar(self, mensaje: str) -> str:
        """
        Clasificación completa: heurística primero, LLM si está disponible
        y la respuesta heurística es CHAT (poco informativa).
        """
        intent_local = self.clasificar_heuristico(mensaje)

        # Si la heurística ya es determinante, no consultar LLM
        if intent_local != INTENT_CHAT:
            return intent_local

        # Fallback al LLM para mensajes ambiguos
        if self._llm is not None:
            try:
                from app.services.ai.prompts.system_prompts import SystemPrompts
                intent_llm = await self._llm.analizar_intencion(
                    mensaje=mensaje,
                    opciones=list(_INTENTS_VALIDOS),
                )
                if intent_llm in _INTENTS_VALIDOS:
                    return intent_llm
            except Exception as exc:
                logger.warning("IntentClassifier LLM fallback error: %s", exc)

        return intent_local
