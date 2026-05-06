"""
Prompts para generación y análisis de rutinas.
"""
from __future__ import annotations

from typing import List, Optional


class RoutinePrompts:
    """Prompts especializados en rutinas de ejercicio."""

    @staticmethod
    def generar_rutina(
        zonas: List[str],
        tiempo_min: int,
        perfil: str,
        nivel: str = "Intermedio",
        condiciones: Optional[List[str]] = None,
    ) -> str:
        zonas_str = ", ".join(zonas)
        cond_str = ""
        if condiciones:
            cond_str = f"\nCondiciones médicas a respetar: {', '.join(condiciones)}."
        return (
            f"Genera una rutina de ejercicio para un cliente con perfil {perfil}.\n"
            f"Zonas objetivo: {zonas_str}\n"
            f"Tiempo disponible: {tiempo_min} minutos\n"
            f"Nivel: {nivel}{cond_str}\n"
            "Reglas:\n"
            "  - Entre 4 y 8 ejercicios.\n"
            "  - Cada ejercicio debe existir en un gimnasio convencional.\n"
            "  - Incluir series, reps y descanso en segundos.\n"
            "  - Si hay condición de rodilla: omitir sentadillas y estocadas profundas.\n"
            "  - Si hay condición de espalda: omitir peso muerto y buenos días.\n"
            "Responde con JSON array:\n"
            "[{\"nombre\": str, \"series\": int, \"repeticiones\": int, "
            "\"descanso_segundos\": int, \"peso_kg\": number|null}, ...]"
        )

    @staticmethod
    def nombre_creativo(zonas: List[str], perfil: str) -> str:
        zonas_str = ", ".join(zonas)
        return (
            f"Crea un nombre corto y motivador (máx 5 palabras) para una rutina "
            f"de {zonas_str} diseñada para un cliente {perfil}. "
            "Responde SOLO con el nombre, sin comillas ni explicación."
        )

    @staticmethod
    def sustituto_ejercicio(ejercicio_bloqueado: str, zona: str, condicion: str) -> str:
        return (
            f"El ejercicio '{ejercicio_bloqueado}' está contraindicado "
            f"para la condición '{condicion}'.\n"
            f"Sugiere UN ejercicio alternativo para trabajar '{zona}' "
            "que sea seguro con esa condición.\n"
            "Responde con JSON: {\"nombre\": str, \"razon\": str}"
        )
