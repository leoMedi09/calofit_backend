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
        restantes   = max(0, kcal_obj - kcal_cons)  # igual que la app (sin sumar quemadas)
        perfil_ml   = contexto.get("perfil_ml", "PERFIL_B")
        perfil_obj  = contexto.get("perfil", {})
        objetivo    = perfil_obj.get("objetivo", "Mantener peso")

        # ── Restricciones dietéticas ──────────────────────────────────────────
        condiciones  = perfil_obj.get("condiciones_medicas", []) or []
        lista_negra  = perfil_obj.get("lista_negra", []) or []
        bloque_dieta = self._bloque_restricciones(condiciones, lista_negra)

        from app.services.ai.prompts.system_prompts import SystemPrompts
        tono   = self.ml.tono_para_perfil(perfil_ml) if self.ml else ""
        system = SystemPrompts.con_perfil(perfil_ml, tono)
        if bloque_dieta:
            system = system + "\n\n" + bloque_dieta

        contexto_str = (
            f"Objetivo: {objetivo}. "
            f"Calorías hoy: {kcal_cons:.0f}/{kcal_obj:.0f} kcal consumidas. "
            f"Quemadas en ejercicio: {kcal_quem:.0f} kcal (dato informativo). "
            f"Calorías disponibles restantes: {restantes:.0f} kcal."
        )
        prompt = f"{contexto_str}\n\nConsulta: {mensaje}"

        texto = await self._texto_llm(prompt=prompt, system=system, max_tokens=200)
        if not texto:
            texto = (
                "Para integrar nutrición y ejercicio: "
                f"te quedan {max(0, restantes):.0f} kcal disponibles hoy. "
                "¿Quieres que te sugiera qué comer o qué entrenar?"
            )

        return ResponseParser.texto(
            mensaje=ResponseParser.limpiar_texto(texto)
        )

    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def _bloque_restricciones(condiciones: list, lista_negra: list) -> str:
        """Genera reglas PROHIBIDO para el system prompt a partir del perfil del usuario."""
        reglas: list[str] = []
        conds_lower = [c.lower() for c in condiciones]

        if any("vegano" in c for c in conds_lower):
            reglas.append(
                "⛔ DIETA VEGANA (OBLIGATORIO): PROHIBIDO ABSOLUTAMENTE recomendar "
                "pollo, carne, res, cerdo, pato, pavo, cabrito, pescado, camarón, mariscos, "
                "huevo, leche, queso, yogur, mantequilla ni ningún derivado animal. "
                "Solo ingredientes 100%% vegetales. Fuentes de proteína: lentejas, garbanzos, "
                "frijoles, tofu, tempeh, quinua, edamame, maní, almendras."
            )
        elif any("vegetariano" in c for c in conds_lower):
            reglas.append(
                "⛔ DIETA VEGETARIANA: PROHIBIDO recomendar carne, pollo, pescado, "
                "camarón ni mariscos. Huevos y lácteos permitidos."
            )

        if any("diabetes" in c for c in conds_lower):
            reglas.append(
                "⛔ DIABETES: PROHIBIDO azucar refinada, miel, gaseosas, jugos azucarados, "
                "chocolates, pasteles, arroz blanco en exceso. "
                "Preferir alimentos de bajo índice glucémico."
            )
        if any("intolerancia" in c for c in conds_lower):
            reglas.append("⛔ INTOLERANCIA A LA LACTOSA: PROHIBIDO leche, queso, yogur, crema.")
        if any("celiac" in c or "celí" in c for c in conds_lower):
            reglas.append("⛔ CELÍACO: PROHIBIDO trigo, avena, cebada, pan de trigo, pasta.")
        if any("hipertensi" in c for c in conds_lower):
            reglas.append("⛔ HIPERTENSIÓN: PROHIBIDO alimentos muy salados, embutidos, frituras.")

        if lista_negra:
            reglas.append(f"⛔ ALIMENTOS PROHIBIDOS POR EL NUTRICIONISTA: {', '.join(lista_negra)}. NUNCA los sugieras.")

        if not reglas:
            return ""
        return (
            "RESTRICCIONES DIETÉTICAS OBLIGATORIAS (INCUMPLIR ES ERROR GRAVE):\n"
            + "\n".join(f"   {r}" for r in reglas)
        )
