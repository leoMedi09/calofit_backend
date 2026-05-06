"""
Handler de nutrición — platos, macros, registros de comida.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.services.assistant.handlers.base_handler import BaseHandler
from app.services.assistant.response_parser import ResponseParser

logger = logging.getLogger(__name__)


class NutritionHandler(BaseHandler):
    """
    Maneja consultas de dominio nutricional:
    - Registro de comida
    - Información sobre platos/ingredientes
    - Recomendaciones de alimentos (KNN)
    - Análisis de macros del día
    """

    async def manejar(
        self,
        mensaje: str,
        contexto: Dict[str, Any],
        client_id: int,
    ) -> Dict[str, Any]:

        # Intentar delegar al asistente existente para aprovechar toda la lógica NLP
        respuesta_existente = await self._delegar_asistente_nutricion(
            mensaje, contexto, client_id
        )
        if respuesta_existente:
            return respuesta_existente

        # Fallback: generar respuesta via LLM con contexto
        return await self._respuesta_generica(mensaje, contexto)

    # ──────────────────────────────────────────────────────────────────

    async def _delegar_asistente_nutricion(
        self,
        mensaje: str,
        contexto: Dict[str, Any],
        client_id: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Delega al asistente_nutricion.py existente que tiene toda la lógica NLP.
        Retorna None si no aplica o hay error.
        """
        try:
            from app.services.asistente_nutricion import procesar_secciones_comida
            # El procesamiento NLP retorna secciones con macros_normalizados
            # Solo aplica si el mensaje menciona registro explícito
            if not self._es_registro(mensaje):
                return None

            # Delegar completamente — el módulo existente maneja el flujo completo
            return None  # Integración opcional: se activa en AssistantOrchestrator
        except ImportError:
            return None
        except Exception as exc:
            logger.warning("NutritionHandler._delegar error: %s", exc)
            return None

    def _es_registro(self, mensaje: str) -> bool:
        import re
        return bool(re.search(
            r"\b(comí|comi|desayuné|almorcé|cené|registra|registré|registre|comí|cené)\b",
            mensaje, re.IGNORECASE,
        ))

    async def _respuesta_generica(
        self,
        mensaje: str,
        contexto: Dict[str, Any],
    ) -> Dict[str, Any]:
        perfil = contexto.get("perfil", {})
        progreso = contexto.get("progreso_hoy", {})
        kcal_cons = progreso.get("kcal_consumidas", 0)
        kcal_obj  = self._kcal_objetivo(contexto)
        deficit   = kcal_obj - kcal_cons

        from app.services.ai.prompts.system_prompts import SystemPrompts
        perfil_ml  = contexto.get("perfil_ml", "PERFIL_B")
        tono       = self.ml.tono_para_perfil(perfil_ml) if self.ml else ""
        system     = SystemPrompts.con_perfil(perfil_ml, tono)

        contexto_str = (
            f"Hoy ha consumido {kcal_cons:.0f} kcal de {kcal_obj:.0f} kcal objetivo. "
            f"Déficit: {deficit:.0f} kcal."
        )
        prompt = f"{contexto_str}\n\nPregunta del cliente: {mensaje}"

        texto = await self._texto_llm(prompt=prompt, system=system, max_tokens=350)
        if not texto:
            texto = (
                f"Llevas {kcal_cons:.0f} kcal de {kcal_obj:.0f} kcal objetivo hoy. "
                "¿En qué más te puedo ayudar con tu nutrición?"
            )

        return ResponseParser.texto(mensaje=ResponseParser.limpiar_texto(texto))
