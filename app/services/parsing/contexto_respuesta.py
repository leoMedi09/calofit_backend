"""
Heurísticas ligeras sobre el mensaje del usuario (dominio comida vs ejercicio).
Usado por ``detectar_modo_funcion`` cuando no hay clasificación por LLM.
"""
from __future__ import annotations

from typing import Literal, Optional

_CTX_COMIDA = (
    "comí",
    "comi ",
    "desayun",
    "almorz",
    "almuerz",
    " cen",
    "cena",
    "pollo",
    "arroz",
    "ensalada",
    "sopa",
    "pasta",
    "snack",
    "comida",
    "kcal",
    "calor",
    "nutri",
)
_CTX_EJERCICIO = (
    "ejercicio",
    "entren",
    "gym",
    "gimnasio",
    "rutina",
    "sentadill",
    "press ",
    "pesas",
    "pierna",
    "piernas",
    "cardio",
    "series",
    "repes",
    "flexion",
    "flexión",
    "burpee",
    "plancha",
    "cuádriceps",
    "cuadriceps",
    "isquio",
    "isquios",
    "femoral",
    "gemelo",
    "gemelos",
    "pantorrilla",
    "vasto",
    "dorsal",
    "trapecio",
    "deltoides",
)


def inferir_forzar_por_mensaje_usuario(
    mensaje: str,
) -> Optional[Literal["comida", "ejercicio"]]:
    """
    Devuelve ``\"comida\"`` o ``\"ejercicio\"`` si hay más señales de un dominio que del otro;
    ``None`` si no hay señal clara o está empatado.
    """
    m = (mensaje or "").lower().strip()
    if len(m) < 2:
        return None
    fc = sum(1 for x in _CTX_COMIDA if x in m)
    fe = sum(1 for x in _CTX_EJERCICIO if x in m)
    if fc > fe:
        return "comida"
    if fe > fc:
        return "ejercicio"
    return None
