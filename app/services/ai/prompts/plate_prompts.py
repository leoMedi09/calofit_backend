"""
Prompts para construcción y análisis de platos.
"""
from __future__ import annotations

from typing import List


class PlatePrompts:
    """Prompts especializados en platos e ingredientes."""

    @staticmethod
    def generar_ingredientes(nombre_plato: str, max_ingredientes: int = 8) -> str:
        return (
            f"Genera la lista de ingredientes para el plato: '{nombre_plato}'.\n"
            f"Máximo {max_ingredientes} ingredientes. Peso total entre 400g y 750g.\n"
            "Reglas:\n"
            "  - USA SOLO gramos numéricos (NO '1 huevo', '1 plátano'). "
            "    Conversiones: 1 huevo→55g, 1 plátano→120g, 1 papa→150g, "
            "    1 taza arroz seco→185g, 1 cdta aceite→5g, 1 cda aceite→12g.\n"
            "  - Cebiche/tiradito → pescado FRESCO (nunca cocido/sancochado).\n"
            "  - Al horno/parrilla → NO sancochado ni hervido.\n"
            "  - Causa ferreñafana → pescado salpreso obligatorio.\n"
            "Responde con JSON array: [{\"nombre_es\": str, \"gramos\": number}, ...]"
        )

    @staticmethod
    def generar_preparacion(nombre_plato: str, ingredientes: List[str]) -> str:
        ingredientes_str = ", ".join(ingredientes)
        return (
            f"Genera los pasos de preparación para: '{nombre_plato}'.\n"
            f"Ingredientes: {ingredientes_str}\n"
            "Reglas:\n"
            "  - Entre 3 y 5 pasos concisos.\n"
            "  - TODOS los ingredientes deben aparecer en al menos un paso.\n"
            "  - Escribe en español, con acentos correctos.\n"
            "Responde con JSON array de strings: [\"Paso 1...\", \"Paso 2...\", ...]"
        )

    @staticmethod
    def analizar_coherencia(nombre_plato: str, ingredientes: List[str]) -> str:
        ingredientes_str = ", ".join(ingredientes)
        return (
            f"Evalúa si los ingredientes son coherentes con el nombre del plato.\n"
            f"Plato: '{nombre_plato}'\n"
            f"Ingredientes: {ingredientes_str}\n"
            "Responde con JSON: "
            "{\"coherente\": bool, \"problema\": str|null, \"sugerencia\": str|null}"
        )
