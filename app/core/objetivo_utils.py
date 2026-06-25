"""
objetivo_utils.py — Normalización canónica del objetivo nutricional del usuario.

Los valores de clients.goal son un conjunto CERRADO de 5 strings controlados
por el dropdown del frontend (onboarding_profile_screen.dart y edit_profile_screen.dart).
Este módulo los mapea a 3 conceptos internos que toda la lógica del backend
debe usar en vez de comparar strings crudos contra el campo goal.

─── MODELO DE PRIORIDAD: perfil vs mensaje ──────────────────────────────────
El objetivo del perfil (clients.goal) es la fuente principal de verdad.
Representa la intención estable y declarada del usuario.

El mensaje actual del usuario puede expresar una intención TEMPORAL diferente
al perfil. Por ejemplo: perfil "ganar masa" pero el usuario dice "quiero bajar
grasa" en el chat. Esto NO modifica el balance calórico calculado — el sistema
no cambia de concepto (SUPERAVIT → DEFICIT) en tiempo real porque:
  1. El usuario no ha cambiado su objetivo en la app.
  2. Un mensaje aislado puede ser una consulta, no una declaración de cambio.

Comportamiento definido ante contradicción:
  - El balance (_calcular_balance_meta) usa el PERFIL como fuente principal.
  - El LLM recibe el contexto del perfil y puede responder informativamente
    ("para bajar grasa necesitarías cambiar tu objetivo en el perfil").
  - La única excepción es cuando el mensaje menciona explícitamente superávit
    (masa muscular, bulking, etc.) y el perfil NO es SUPERAVIT — en ese caso
    se activa temporalmente el tratamiento de superávit para ese turno.
    Esto NO aplica al inverso: si el perfil es SUPERAVIT, el mensaje "quiero
    bajar grasa" NO cambia el concepto a DEFICIT.

─── SEMÁNTICA DE SUPERAVIT ──────────────────────────────────────────────────
SUPERAVIT representa el estado energético de "ganar masa en cualquier ritmo".
Agrupa ganar_leve (Limpio, +250 kcal/día) y ganar masa (Volumen, +500 kcal/día).

No distingue entre ambas variantes porque la lógica de balance compartida
(exceso calórico = esperado, no advertir como error) aplica igual a las dos.
Si en el futuro se necesita distinguir intensidad del superávit, se puede
añadir SUPERAVIT_LEVE y SUPERAVIT_COMPLETO sin romper el contrato actual.

Uso:
    from app.core.objetivo_utils import normalizar_objetivo, es_superavit, SUPERAVIT
    if es_superavit(perfil.goal):
        ...  # ganar_leve o ganar masa — ambos son SUPERAVIT
"""
from __future__ import annotations

# ── Conceptos internos (3 estados energéticos) ────────────────────────────────
DEFICIT       = "DEFICIT"        # déficit calórico — perder peso en cualquier ritmo
MANTENIMIENTO = "MANTENIMIENTO"  # sin superávit ni déficit objetivo
SUPERAVIT     = "SUPERAVIT"      # superávit calórico — ganar masa (ganar_leve o ganar masa)
#                                  Ver nota en docstring sobre semántica y casos de extensión.

# ── Mapa cerrado: los 5 valores reales de la app ─────────────────────────────
# Fuente: onboarding_profile_screen.dart y edit_profile_screen.dart
_GOAL_MAP: dict[str, str] = {
    "perder peso":   DEFICIT,
    "perder_leve":   DEFICIT,
    "mantener peso": MANTENIMIENTO,
    "ganar_leve":    SUPERAVIT,
    "ganar masa":    SUPERAVIT,
}


def normalizar_objetivo(goal: str | None) -> str:
    """
    Convierte el valor raw de clients.goal al concepto interno.

    - Hace strip y lowercase antes de buscar en el mapa.
    - Soporta también valores ya normalizados (DEFICIT, SUPERAVIT, MANTENIMIENTO).
    - Fallback seguro: MANTENIMIENTO si el valor es None, vacío o desconocido.
      Esto garantiza que el sistema nunca asuma superávit sin confirmación explícita.

    Args:
        goal: Valor crudo de clients.goal (ej. "ganar_leve", "Perder peso", None)

    Returns:
        Uno de: DEFICIT | MANTENIMIENTO | SUPERAVIT
    """
    if not goal:
        return MANTENIMIENTO
    val = goal.strip()
    val_upper = val.upper()
    if val_upper in (DEFICIT, SUPERAVIT, MANTENIMIENTO):
        return val_upper
    return _GOAL_MAP.get(val.lower(), MANTENIMIENTO)


def es_superavit(goal: str | None) -> bool:
    """
    True si el objetivo del perfil implica un superávit calórico.
    Cubre tanto 'ganar_leve' (Ganar masa Limpio) como 'ganar masa' (Volumen).
    """
    return normalizar_objetivo(goal) == SUPERAVIT


def es_deficit(goal: str | None) -> bool:
    """
    True si el objetivo del perfil implica un déficit calórico.
    Cubre tanto 'perder peso' (Agresivo) como 'perder_leve' (Definición).
    """
    return normalizar_objetivo(goal) == DEFICIT


def es_mantenimiento(goal: str | None) -> bool:
    """True si el objetivo es mantener peso (sin ajuste calórico)."""
    return normalizar_objetivo(goal) == MANTENIMIENTO
