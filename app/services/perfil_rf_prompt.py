"""Perfil de adherencia (Random Forest A/B/C) como bloque ADICIONAL de los prompts de recomendación.

El perfil solo define el tono y la exigencia. No reemplaza los datos del cliente (objetivo, condiciones médicas, lesiones,
alimentos prohibidos, meta del plan, consumo del día): esos siguen entrando completos al prompt y tienen prioridad.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from app.services.ml_service import ml_perfil

logger = logging.getLogger("calofit.perfil_rf")

_NOMBRE = {"PERFIL_A": "A — disciplinado", "PERFIL_B": "B — en desarrollo", "PERFIL_C": "C — necesita guía"}

_REGLA_COMIDA = {
    "PERFIL_A": "Puedes proponer platos más elaborados y mencionar el aporte de proteína de cada uno.",
    "PERFIL_B": "Propón ideas prácticas y variadas, fáciles de repetir durante la semana.",
    "PERFIL_C": "Propón platos simples, de pocos ingredientes y preparación rápida. Evita tecnicismos.",
}


def obtener_perfil_rf(perfil, db) -> Optional[Tuple[str, float]]:
    """(PERFIL_A|B|C, confianza %) para el cliente, o None si no se puede predecir (sin BD, modelo inactivo o error)."""
    if db is None or not ml_perfil.modelo_activo:
        return None
    try:
        from app.services.asistente.asistente_recomendaciones import RecomendacionesHandler

        nombre, conf = RecomendacionesHandler().predecir_perfil(perfil, db)
        return (nombre, conf) if conf and conf > 0 else None
    except Exception as e:
        logger.warning("perfil RF no disponible: %s", e)
        return None


def _regla_ejercicio(nombre: str) -> str:
    try:
        from app.services.rutina_service import _CONFIG_PERFIL

        c = _CONFIG_PERFIL[nombre]
        return f"Si propones series, usa {c['series']}×{c['reps']} con {c['descanso_seg']} s de descanso (intensidad {c['intensidad']})."
    except Exception:
        return ""


def bloque_perfil_rf(rf: Optional[Tuple[str, float]], modo: str) -> str:
    """Texto para añadir al prompt ('' si no hay perfil). modo = 'comida' | 'ejercicio'."""
    if not rf or rf[0] not in _NOMBRE:
        return ""
    nombre, conf = rf
    regla = _regla_ejercicio(nombre) if modo == "ejercicio" else _REGLA_COMIDA[nombre]
    lineas = [
        f"PERFIL DE ADHERENCIA (modelo Random Forest, {conf:.0f}% de confianza): {_NOMBRE[nombre]}.",
        ml_perfil.get_tono_asistente(nombre),
    ]
    if regla:
        lineas.append(regla)
    lineas.append(
        "Este perfil solo define el tono y la exigencia. Las condiciones médicas, lesiones, restricciones y "
        "alimentos prohibidos del cliente tienen prioridad absoluta."
    )
    return "\n".join(lineas)
