"""
Parser de respuestas del asistente → estructura normalizada para Flutter.
"""
from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ResponseParser:
    """
    Normaliza la respuesta final del asistente a la estructura que espera Flutter:

    {
        "tipo":    str,           # "texto" | "plato" | "rutina" | "recomendacion" | "plan"
        "mensaje": str,           # texto principal
        "datos":   dict | None,   # payload estructurado (opcional)
        "advertencias": [str],    # advertencias clínicas
        "acciones":     [dict],   # botones/acciones sugeridas para Flutter
    }
    """

    # ──────────────────────────────────────────────────────────────────
    # Constructores de respuestas tipadas
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def texto(mensaje: str, advertencias: Optional[List[str]] = None) -> Dict[str, Any]:
        return {
            "tipo": "texto",
            "mensaje": mensaje,
            "datos": None,
            "advertencias": advertencias or [],
            "acciones": [],
        }

    @staticmethod
    def plato(
        mensaje: str,
        plato_data: Dict[str, Any],
        advertencias: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "tipo": "plato",
            "mensaje": mensaje,
            "datos": plato_data,
            "advertencias": advertencias or [],
            "acciones": [
                {"etiqueta": "Registrar comida", "accion": "registrar_comida"},
                {"etiqueta": "Ver ingredientes",  "accion": "ver_ingredientes"},
            ],
        }

    @staticmethod
    def rutina(
        mensaje: str,
        rutina_data: Dict[str, Any],
        advertencias: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "tipo": "rutina",
            "mensaje": mensaje,
            "datos": rutina_data,
            "advertencias": advertencias or [],
            "acciones": [
                {"etiqueta": "Iniciar rutina",      "accion": "iniciar_rutina"},
                {"etiqueta": "Registrar ejercicio",  "accion": "registrar_ejercicio"},
            ],
        }

    @staticmethod
    def recomendacion(
        mensaje: str,
        alimentos: List[Dict[str, Any]],
        advertencias: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "tipo": "recomendacion",
            "mensaje": mensaje,
            "datos": {"alimentos": alimentos},
            "advertencias": advertencias or [],
            "acciones": [
                {"etiqueta": "Ver detalles", "accion": "ver_alimento"},
            ],
        }

    @staticmethod
    def plan(
        mensaje: str,
        plan_data: Dict[str, Any],
        advertencias: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "tipo": "plan",
            "mensaje": mensaje,
            "datos": plan_data,
            "advertencias": advertencias or [],
            "acciones": [
                {"etiqueta": "Ver plan completo", "accion": "ver_plan"},
            ],
        }

    # ──────────────────────────────────────────────────────────────────
    # Utilidades
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def limpiar_texto(texto: str) -> str:
        """Elimina artefactos del LLM (bloques ```, comillas excesivas)."""
        texto = re.sub(r"```[\w]*\n?", "", texto)
        texto = texto.strip().strip('"').strip("'")
        return texto

    @staticmethod
    def truncar(texto: str, max_chars: int = 800) -> str:
        if len(texto) <= max_chars:
            return texto
        corte = texto.rfind(". ", 0, max_chars)
        return (texto[: corte + 1] if corte > 0 else texto[:max_chars]) + "…"
