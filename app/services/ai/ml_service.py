"""
Wrapper ligero de MLService para el módulo ai/.
Delega a app.services.ml_service (ClasificadorPerfil + RecomendadorAlimentosKNN).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MLServiceWrapper:
    """
    Fachada sobre ClasificadorPerfil y RecomendadorAlimentosKNN.

    Importa lazy para no fallar si los modelos .pkl no están presentes.
    """

    def __init__(self) -> None:
        self._clasificador = None
        self._recomendador = None
        self._cargado = False

    def _cargar(self) -> None:
        if self._cargado:
            return
        try:
            from app.services.ml_service import ClasificadorPerfil, RecomendadorAlimentosKNN
            self._clasificador = ClasificadorPerfil()
            self._recomendador = RecomendadorAlimentosKNN()
            self._cargado = True
        except Exception as exc:
            logger.error("MLServiceWrapper: no se pudieron cargar modelos: %s", exc)

    # ──────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────

    def predecir_perfil(self, features: Dict[str, Any]) -> Tuple[str, float]:
        """
        Predice el perfil de adherencia (PERFIL_A/B/C) y su confianza.

        Args:
            features: diccionario de features del cliente (compatible con ClasificadorPerfil)

        Returns:
            (perfil: str, confianza: float)  — ej. ("PERFIL_A", 0.93)
        """
        self._cargar()
        if self._clasificador is None:
            return ("PERFIL_B", 0.50)
        try:
            perfil = self._clasificador.predecir(features)
            confianza = self._clasificador.confianza(features)
            return (perfil, float(confianza))
        except Exception as exc:
            logger.warning("MLServiceWrapper.predecir_perfil error: %s", exc)
            return ("PERFIL_B", 0.50)

    def recomendar_alimentos(
        self,
        deficit_kcal: float,
        excluir_ids: Optional[List[int]] = None,
        n: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Recomienda alimentos por similitud coseno KNN.

        Args:
            deficit_kcal: déficit calórico del cliente
            excluir_ids: IDs de alimentos a excluir (ya vistos hoy)
            n: cantidad de recomendaciones

        Returns:
            Lista de dicts con campos 'alimento_id', 'nombre', 'similitud'
        """
        self._cargar()
        if self._recomendador is None:
            return []
        try:
            return self._recomendador.recomendar(
                deficit_kcal=deficit_kcal,
                excluir_ids=excluir_ids or [],
                n=n,
            )
        except Exception as exc:
            logger.warning("MLServiceWrapper.recomendar_alimentos error: %s", exc)
            return []

    def tono_para_perfil(self, perfil: str) -> str:
        """Retorna la instrucción de tono para el LLM según el perfil."""
        self._cargar()
        if self._clasificador is None:
            return ""
        return self._clasificador.TONO_ASISTENTE.get(perfil, "")
