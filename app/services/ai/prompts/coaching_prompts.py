"""
Prompts para coaching nutricional y deportivo.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class CoachingPrompts:
    """Prompts de coaching personalizado."""

    @staticmethod
    def resumen_diario(
        fecha: str,
        kcal_consumidas: float,
        kcal_objetivo: float,
        kcal_quemadas: float,
        perfil: str,
    ) -> str:
        balance = kcal_consumidas - kcal_objetivo + kcal_quemadas
        estado = "déficit" if balance < 0 else "superávit"
        return (
            f"El cliente (perfil {perfil}) tiene este balance al {fecha}:\n"
            f"  Consumidas: {kcal_consumidas:.0f} kcal\n"
            f"  Objetivo: {kcal_objetivo:.0f} kcal\n"
            f"  Quemadas en ejercicio: {kcal_quemadas:.0f} kcal\n"
            f"  Balance: {abs(balance):.0f} kcal en {estado}\n"
            "Genera un comentario de coaching breve (2-3 oraciones), positivo y accionable, "
            "adaptado al perfil. No repitas los números ya mostrados."
        )

    @staticmethod
    def recomendacion_alimento(
        alimentos: list,
        deficit_kcal: float,
        perfil: str,
    ) -> str:
        nombres = ", ".join(a.get("nombre", "") for a in alimentos[:5])
        return (
            f"El cliente (perfil {perfil}) tiene un déficit de {deficit_kcal:.0f} kcal.\n"
            f"El sistema recomienda estos alimentos por similitud nutricional: {nombres}.\n"
            "Escribe una sugerencia corta (1-2 oraciones) que presente estos alimentos "
            "de forma motivadora, mencionando el beneficio principal de cada uno. "
            "Usa el contexto cultural peruano/lambayecano."
        )

    @staticmethod
    def consejo_ejercicio(
        rutina_nombre: str,
        kcal_quemadas: float,
        duracion_min: int,
        perfil: str,
    ) -> str:
        return (
            f"El cliente (perfil {perfil}) completó la rutina '{rutina_nombre}' "
            f"({duracion_min} min, {kcal_quemadas:.0f} kcal quemadas).\n"
            "Escribe un mensaje de refuerzo positivo (1-2 oraciones) y un consejo práctico "
            "para la próxima sesión, adaptado al perfil."
        )

    @staticmethod
    def plan_semanal_intro(
        kcal_diarias: float,
        objetivo: str,
        perfil: str,
    ) -> str:
        return (
            f"Introduce brevemente el plan semanal para un cliente con:\n"
            f"  Objetivo: {objetivo}\n"
            f"  Calorías diarias objetivo: {kcal_diarias:.0f} kcal\n"
            f"  Perfil de adherencia: {perfil}\n"
            "Escribe 2-3 oraciones motivadoras que expliquen el enfoque del plan. "
            "Menciona la estrategia principal (déficit/mantenimiento/superávit calórico)."
        )
