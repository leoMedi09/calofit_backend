"""
Handler integrado — consultas que combinan nutrición y ejercicio.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from app.services.assistant.handlers.base_handler import BaseHandler
from app.services.assistant.response_parser import ResponseParser

logger = logging.getLogger(__name__)


class IntegratedHandler(BaseHandler):
    """
    Maneja consultas que mezclan nutrición y ejercicio en un mismo mensaje.

    Ejemplos:
      "¿Qué como antes de entrenar piernas?"
      "Hice cardio y quiero saber cuánto puedo comer"
      "Dame un plan con rutina y dieta para bajar de peso"
    """

    async def manejar(
        self,
        mensaje: str,
        contexto: Dict[str, Any],
        client_id: int,
    ) -> Dict[str, Any]:

        kcal_obj    = self._kcal_objetivo(contexto)
        kcal_cons   = self._kcal_consumidas(contexto)
        kcal_quem   = float((contexto.get("progreso_hoy") or {}).get("kcal_quemadas", 0))
        deficit     = kcal_obj - kcal_cons + kcal_quem
        perfil_ml   = contexto.get("perfil_ml", "PERFIL_B")
        perfil_obj  = contexto.get("perfil", {})
        objetivo    = perfil_obj.get("objetivo", "Mantener peso")

        from app.services.ai.prompts.system_prompts import SystemPrompts
        tono   = self.ml.tono_para_perfil(perfil_ml) if self.ml else ""
        system = SystemPrompts.con_perfil(perfil_ml, tono)

        contexto_str = (
            f"Objetivo: {objetivo}. "
            f"Calorías hoy: {kcal_cons:.0f}/{kcal_obj:.0f} kcal consumidas. "
            f"Quemadas en ejercicio: {kcal_quem:.0f} kcal. "
            f"Balance disponible: {deficit:.0f} kcal."
        )
        prompt = f"{contexto_str}\n\nConsulta: {mensaje}"

        texto = await self._texto_llm(prompt=prompt, system=system, max_tokens=450)
        if not texto:
            texto = (
                "Para integrar nutrición y ejercicio: "
                f"te quedan {max(0, deficit):.0f} kcal disponibles hoy. "
                "¿Quieres que te sugiera qué comer o qué entrenar?"
            )

        return ResponseParser.texto(
            mensaje=ResponseParser.limpiar_texto(texto)
        )
