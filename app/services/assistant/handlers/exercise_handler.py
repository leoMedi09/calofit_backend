"""
Handler de ejercicio — rutinas, series, registros de entrenamiento.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.services.assistant.handlers.base_handler import BaseHandler
from app.services.assistant.response_parser import ResponseParser

logger = logging.getLogger(__name__)


class ExerciseHandler(BaseHandler):
    """
    Maneja consultas de dominio de ejercicio:
    - Generación de rutinas adaptativas
    - Registro de sesiones
    - Información sobre ejercicios y grupos musculares
    - Sustituciones por lesión
    """

    async def manejar(
        self,
        mensaje: str,
        contexto: Dict[str, Any],
        client_id: int,
    ) -> Dict[str, Any]:

        # Detectar si pide una rutina
        if self._solicita_rutina(mensaje):
            return await self._generar_rutina(mensaje, contexto, client_id)

        # Detectar si registra ejercicio
        if self._es_registro_ejercicio(mensaje):
            return await self._registrar_ejercicio(mensaje, contexto, client_id)

        # Respuesta informativa general
        return await self._respuesta_generica(mensaje, contexto)

    # ──────────────────────────────────────────────────────────────────

    def _solicita_rutina(self, mensaje: str) -> bool:
        import re
        return bool(re.search(
            r"\b(rutina|programa|ejercicios para|entrenamiento|workout|"
            r"quiero entrenar|diseña|dame una rutina|plan de ejercicio)\b",
            mensaje, re.IGNORECASE,
        ))

    def _es_registro_ejercicio(self, mensaje: str) -> bool:
        import re
        return bool(re.search(
            r"\b(hice|realicé|entren[eé]|corrí|levanté|series de|reps de|kg de)\b",
            mensaje, re.IGNORECASE,
        ))

    async def _generar_rutina(
        self,
        mensaje: str,
        contexto: Dict[str, Any],
        client_id: int,
    ) -> Dict[str, Any]:
        try:
            from app.services.rutina_service import generar_rutina_inteligente
            perfil_tipo = contexto.get("perfil_ml", "PERFIL_B")
            perfil_obj  = contexto.get("perfil", {})
            condiciones = perfil_obj.get("condiciones_medicas", [])

            # Inferir zonas del mensaje
            zonas = self._inferir_zonas(mensaje)

            resultado = await generar_rutina_inteligente(
                user_id=client_id,
                zonas_objetivo=zonas,
                tiempo_min=60,
                db=self.db,
            )

            texto_llm = await self._texto_llm(
                prompt=(
                    f"El sistema generó una rutina llamada '{resultado.get('nombre', 'Sin nombre')}' "
                    f"con {resultado.get('total_ejercicios', 0)} ejercicios para {', '.join(zonas)}. "
                    "Preséntala de forma motivadora en 2 oraciones."
                ),
                system=self._system(contexto),
                max_tokens=200,
            )

            return ResponseParser.rutina(
                mensaje=texto_llm or f"Rutina '{resultado.get('nombre')}' lista.",
                rutina_data=resultado,
            )
        except Exception as exc:
            logger.warning("ExerciseHandler._generar_rutina error: %s", exc)
            return await self._respuesta_generica(mensaje, contexto)

    async def _registrar_ejercicio(
        self,
        mensaje: str,
        contexto: Dict[str, Any],
        client_id: int,
    ) -> Dict[str, Any]:
        # Delegar al módulo existente
        try:
            from app.services.asistente_registro_ejercicio import procesar_registro_ejercicio
            result = await procesar_registro_ejercicio(
                mensaje=mensaje,
                client_id=client_id,
                db=self.db,
            )
            texto = result.get("mensaje", "Ejercicio registrado.")
            return ResponseParser.texto(mensaje=texto)
        except Exception as exc:
            logger.warning("ExerciseHandler._registrar_ejercicio error: %s", exc)
            return ResponseParser.texto("Ejercicio anotado. ¿Qué peso y series usaste?")

    async def _respuesta_generica(
        self,
        mensaje: str,
        contexto: Dict[str, Any],
    ) -> Dict[str, Any]:
        system = self._system(contexto)
        texto = await self._texto_llm(
            prompt=mensaje,
            system=system,
            max_tokens=350,
        )
        return ResponseParser.texto(
            mensaje=ResponseParser.limpiar_texto(texto) if texto
            else "Cuéntame más sobre tu entrenamiento y te ayudo."
        )

    def _inferir_zonas(self, mensaje: str) -> List[str]:
        import re
        zonas = []
        mapa = {
            "pierna": "Piernas",
            "glúteo": "Glúteos",
            "gluteo": "Glúteos",
            "pecho": "Pecho",
            "espalda": "Espalda",
            "hombro": "Hombros",
            "brazo": "Brazos",
            "bícep": "Brazos",
            "trícep": "Brazos",
            "abdomen": "Core",
            "cardio": "Cardio",
            "funcional": "Funcional",
        }
        msg_lower = mensaje.lower()
        for clave, zona in mapa.items():
            if clave in msg_lower and zona not in zonas:
                zonas.append(zona)
        return zonas or ["Cuerpo completo"]

    def _system(self, contexto: Dict[str, Any]) -> str:
        from app.services.ai.prompts.system_prompts import SystemPrompts
        perfil_ml = contexto.get("perfil_ml", "PERFIL_B")
        tono = self.ml.tono_para_perfil(perfil_ml) if self.ml else ""
        return SystemPrompts.con_perfil(perfil_ml, tono)
