"""
Handler de chat general — saludos, preguntas abiertas, conversación libre.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from app.services.assistant.handlers.base_handler import BaseHandler
from app.services.assistant.response_parser import ResponseParser

logger = logging.getLogger(__name__)

_SALUDOS = frozenset({
    "hola", "buenos dias", "buenas tardes", "buenas noches",
    "buenas", "hey", "hi", "hello", "qué tal", "como estas",
})


class ChatHandler(BaseHandler):
    """
    Maneja mensajes de chat general: saludos, preguntas abiertas,
    motivación y cualquier consulta que no encaje en otros handlers.
    """

    async def manejar(
        self,
        mensaje: str,
        contexto: Dict[str, Any],
        client_id: int,
    ) -> Dict[str, Any]:

        # Saludo rápido sin LLM
        if self._es_saludo(mensaje):
            return self._respuesta_saludo(contexto)

        # Respuesta LLM para conversación general
        perfil_ml = contexto.get("perfil_ml", "PERFIL_B")
        from app.services.ai.prompts.system_prompts import SystemPrompts
        tono   = self.ml.tono_para_perfil(perfil_ml) if self.ml else ""
        system = SystemPrompts.con_perfil(perfil_ml, tono)

        texto = await self._texto_llm(
            prompt=mensaje,
            system=system,
            max_tokens=300,
        )
        if not texto:
            texto = "Estoy aquí para ayudarte con tu nutrición y entrenamiento. ¿En qué te puedo ayudar?"

        return ResponseParser.texto(mensaje=ResponseParser.limpiar_texto(texto))

    # ──────────────────────────────────────────────────────────────────

    def _es_saludo(self, mensaje: str) -> bool:
        norm = mensaje.lower().strip().rstrip("!¡?¿. ")
        return norm in _SALUDOS

    def _respuesta_saludo(self, contexto: Dict[str, Any]) -> Dict[str, Any]:
        perfil = contexto.get("perfil", {})
        nombre = perfil.get("nombre", "").split()[0] if perfil.get("nombre") else ""
        kcal_cons = self._kcal_consumidas(contexto)
        kcal_obj  = self._kcal_objetivo(contexto)

        saludo = f"¡Hola{', ' + nombre if nombre else ''}! 👋" if False else \
                 f"¡Hola{', ' + nombre if nombre else ''}!"

        balance_str = ""
        if kcal_cons > 0:
            balance_str = (
                f" Hoy llevas {kcal_cons:.0f} de {kcal_obj:.0f} kcal."
            )

        texto = (
            f"{saludo} Soy tu asistente CaloFit.{balance_str} "
            "¿En qué te puedo ayudar hoy: nutrición, ejercicio o tu plan?"
        )
        return ResponseParser.texto(mensaje=texto)
