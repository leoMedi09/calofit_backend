"""
Wrapper ligero de MLService para el módulo ai/.
Delega a los singletons de app.services.ml_service (ya cargados al iniciar el servidor).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MLServiceWrapper:
    """
    Fachada sobre ClasificadorPerfil y RecomendadorAlimentosKNN.

    Usa los singletons ml_perfil / ml_recomendador que el servidor ya cargó
    al arrancar — no instancia nuevos objetos ni vuelve a leer los .pkl.
    """

    def __init__(self) -> None:
        self._cargado = False
        self._perfil = None
        self._recomendador = None

    def _cargar(self) -> None:
        if self._cargado:
            return
        try:
            # Reusar singletons ya inicializados — evita doble carga de .pkl
            from app.services.ml_service import ml_perfil, ml_recomendador
            self._perfil      = ml_perfil
            self._recomendador = ml_recomendador
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
            features: diccionario compatible con ClasificadorPerfil.predecir_perfil()

        Returns:
            (perfil: str, confianza: float)  — ej. ("PERFIL_A", 93.0)
        """
        self._cargar()
        if self._perfil is None:
            return ("PERFIL_B", 50.0)
        try:
            # predecir_perfil() devuelve (perfil_str, confianza_float)
            perfil, confianza = self._perfil.predecir_perfil(features)
            return (perfil, float(confianza))
        except Exception as exc:
            logger.warning("MLServiceWrapper.predecir_perfil error: %s", exc)
            return ("PERFIL_B", 50.0)

    def recomendar_alimentos(
        self,
        deficit_kcal: float,
        excluir_nombres: Optional[List[str]] = None,
        n: int = 5,
        prote_faltante: float = 0.0,
        carbo_faltante: float = 0.0,
        grasa_faltante: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Recomienda alimentos por similitud coseno KNN.

        Args:
            deficit_kcal:    déficit calórico del cliente
            excluir_nombres: nombres de alimentos a excluir (ya vistos hoy)
            n:               cantidad de recomendaciones
            prote_faltante:  déficit de proteínas en gramos
            carbo_faltante:  déficit de carbohidratos en gramos
            grasa_faltante:  déficit de grasas en gramos

        Returns:
            Lista de dicts con campos 'alimento', 'calorias_100g', 'proteina_100g',
            'carbohindratos_100g', 'grasas_100g', 'similitud'
        """
        self._cargar()
        if self._recomendador is None:
            return []
        try:
            return self._recomendador.obtener_recomendaciones(
                calorias_faltantes = deficit_kcal,
                prote_faltante     = prote_faltante,
                carbo_faltante     = carbo_faltante,
                grasa_faltante     = grasa_faltante,
                n_recomendaciones  = n,
                excluir_nombres    = excluir_nombres or [],
            )
        except Exception as exc:
            logger.warning("MLServiceWrapper.recomendar_alimentos error: %s", exc)
            return []

    def tono_para_perfil(self, perfil: str) -> str:
        """Retorna la instrucción de tono para el LLM según el perfil."""
        self._cargar()
        if self._perfil is None:
            return ""
        return self._perfil.TONO_ASISTENTE.get(perfil, "")
