"""
Handler base abstracto para todos los handlers del asistente.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class BaseHandler(ABC):
    """
    Clase base para handlers del asistente.

    Cada handler recibe el contexto del cliente y el mensaje,
    ejecuta su lógica de dominio, y retorna un dict normalizado
    compatible con ResponseParser.
    """

    def __init__(
        self,
        db: Session,
        llm_service=None,
        ml_service=None,
    ) -> None:
        self.db = db
        self.llm = llm_service
        self.ml = ml_service

    @abstractmethod
    async def manejar(
        self,
        mensaje: str,
        contexto: Dict[str, Any],
        client_id: int,
    ) -> Dict[str, Any]:
        """
        Procesa el mensaje en el contexto de dominio del handler.

        Returns:
            dict normalizado (ver ResponseParser)
        """

    # ──────────────────────────────────────────────────────────────────
    # Helpers compartidos
    # ──────────────────────────────────────────────────────────────────

    def _perfil_tipo(self, contexto: Dict[str, Any]) -> str:
        """Retorna el perfil ML desde el contexto, o PERFIL_B por defecto."""
        return contexto.get("perfil_ml", "PERFIL_B")

    def _kcal_objetivo(self, contexto: Dict[str, Any]) -> float:
        """Retorna las kcal objetivo del plan o de las metas."""
        plan = contexto.get("plan_hoy") or {}
        metas = contexto.get("metas") or {}
        return (
            plan.get("kcal_objetivo")
            or metas.get("calorias_objetivo")
            or 2000.0
        )

    def _kcal_consumidas(self, contexto: Dict[str, Any]) -> float:
        return float((contexto.get("progreso_hoy") or {}).get("kcal_consumidas", 0))

    def _deficit(self, contexto: Dict[str, Any]) -> float:
        return self._kcal_objetivo(contexto) - self._kcal_consumidas(contexto)

    async def _texto_llm(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 400,
    ) -> str:
        """Genera texto vía LLM, retorna string vacío si LLM no disponible."""
        if self.llm is None:
            return ""
        try:
            return await self.llm.completar(
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.warning("%s._texto_llm error: %s", self.__class__.__name__, exc)
            return ""
