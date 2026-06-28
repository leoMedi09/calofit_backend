"""
MET aproximados para fuerza y cardio (referencia tipo Compendium).

Calorías (ACSM / compendio habitual):
  kcal = (MET × 3.5 × peso_kg / 200) × minutos
Equivalente: kcal/min = (MET × 3.5 × peso_kg) / 200

Fuente única de verdad: estos 4 dicts (CARDIO/FUERZA/FUNCIONAL/DEPORTES) son
los mismos números que ve el LLM en _PROMPT_EJERCICIO (llm_registro.py) — la
tabla de texto del prompt se genera desde aquí con tabla_prompt_texto(), así
que el registro manual (Constructor de Rutinas) y el registro por chat
SIEMPRE usan el mismo MET para el mismo ejercicio. Si agregas o cambias un
valor, hazlo aquí — nunca a mano en el prompt.
"""

from typing import Dict

# ── Cardio (con variantes de intensidad para que el LLM elija desde texto libre) ──
METS_CARDIO: Dict[str, float] = {
    "caminata lenta": 3.0,
    "caminata rápida": 4.5,
    "caminata rapida": 4.5,
    "trote suave": 8.3,
    "correr moderado": 10.0,
    "correr rápido": 11.5,
    "correr rapido": 11.5,
    "ciclismo moderado": 8.0,
    "natación recreativa": 6.0,
    "natacion recreativa": 6.0,
    "natación intensa": 10.0,
    "natacion intensa": 10.0,
    "bicicleta estática": 7.0,
    "bicicleta estatica": 7.0,
    "saltar cuerda": 12.0,
    "elíptica moderada": 5.0,
    "eliptica moderada": 5.0,
    "remo máquina": 7.0,
    "remo maquina": 7.0,
}

# ── Fuerza (gym) ──────────────────────────────────────────────────────────
METS_FUERZA: Dict[str, float] = {
    "press banca": 5.0,
    "press militar": 5.0,
    "press inclinado": 5.0,
    "sentadilla libre": 6.0,
    "prensa de piernas": 5.0,
    "peso muerto": 6.0,
    "dominadas": 8.0,
    "pull up": 8.0,
    "pull-ups": 8.0,
    "jalón al pecho": 5.0,
    "jalon al pecho": 5.0,
    "remo con barra": 6.0,
    "curl de bíceps": 3.5,
    "curl de biceps": 3.5,
    "extensión tríceps": 3.5,
    "extension triceps": 3.5,
    "elevaciones laterales": 3.0,
    "hip thrust": 5.0,
    "zancadas": 5.5,
    "lunges": 5.5,
    "extensión cuádriceps": 3.5,
    "extension cuadriceps": 3.5,
    "flexiones": 8.0,
    "push up": 8.0,
    "push-ups": 8.0,
    "fondos en paralelas": 8.0,
}

# ── Funcional / HIIT ──────────────────────────────────────────────────────
METS_FUNCIONAL: Dict[str, float] = {
    "burpees": 10.0,
    "box jumps": 10.0,
    "kettlebell swings": 12.0,
    "battle ropes": 10.0,
    "hiit circuito": 9.0,
    "crossfit wod": 12.0,
    "trx suspension": 7.0,
    "plancha isométrica": 4.0,
    "plancha isometrica": 4.0,
    "mountain climbers": 8.0,
}

# ── Deportes ──────────────────────────────────────────────────────────────
METS_DEPORTES: Dict[str, float] = {
    "fútbol": 7.0,
    "futbol": 7.0,
    "básquet": 8.0,
    "basquet": 8.0,
    "vóley": 4.0,
    "voley": 4.0,
    "tenis": 7.5,
    "boxeo sparring": 9.0,
}

# ── Alias y catálogo extendido — SOLO para matching del registro manual.
# No aparecen en el prompt del LLM (no hace falta, el LLM ya razona variantes
# y sinónimos); existen aquí para que el lookup por substring del Constructor
# de Rutinas reconozca nombres cortos o coloquiales ("nadar", "sentadilla",
# "remo", "gym", etc.) y caiga en el MET correcto de las tablas de arriba.
METS_EXTRA: Dict[str, float] = {
    "pesas": 5.0,
    "pesa": 5.0,
    "peso libre": 5.0,
    "mancuernas": 4.5,
    "barra": 5.0,
    "sentadilla": 6.0,
    "squat": 6.0,
    "press de banca": 5.0,
    "bench press": 5.0,
    "deadlift": 6.0,
    "jalón": 5.0,
    "jalon": 5.0,
    "remo": 6.0,
    "curl biceps": 3.5,
    "triceps": 3.5,
    "hombros": 4.5,
    "leg press": 5.0,
    "prensa": 5.0,
    "prensa de pierna": 5.0,
    "extensiones de piernas": 3.5,
    "extensiones de pierna": 3.5,
    "extensión de piernas": 3.5,
    "extensión de pierna": 3.5,
    "extension de piernas": 3.5,
    "extension de pierna": 3.5,
    "elevaciones de piernas": 3.8,
    "elevaciones de pierna": 3.8,
    "elevación de piernas": 3.8,
    "elevación de pierna": 3.8,
    "elevacion de piernas": 3.8,
    "elevacion de pierna": 3.8,
    "elevación lateral": 3.0,
    "elevacion lateral": 3.0,
    "patada de glúteo": 4.0,
    "patada trasera": 4.0,
    "curl femoral": 4.5,
    "curl de femoral": 4.5,
    "extensión": 3.5,
    "extensiones": 3.5,
    "elevaciones": 3.8,
    "femoral": 4.5,
    "glúteos": 4.0,
    "glúte": 4.0,
    "caminadora": 4.5,
    "trotadora": 8.3,
    "trotar caminadora": 8.3,
    "elíptica": 5.0,
    "spinning": 8.5,
    "rowing": 7.0,
    "escaladora": 8.0,
    "stepper": 6.0,
    "cardio": 6.0,
    "hiit": 9.0,
    "circuito": 7.0,
    "funcional": 6.5,
    "trotar": 8.3,
    "correr": 10.0,
    "saltar soga": 12.0,
    "cuerda": 12.0,
    "abdominales": 3.8,
    "abs": 3.8,
    "plancha": 4.0,
    "crunch": 3.8,
    "core": 4.0,
    "fondos en banco": 4.5,
    "fondos": 8.0,
    "paralelas": 8.0,
    "aperturas": 4.0,
    "rutina de pecho": 5.0,
    "rutina de espalda": 5.0,
    "rutina de pierna": 5.5,
    "rutina de hombros": 4.5,
    "rutina de brazo": 4.0,
    "rutina de brazos": 4.0,
    "día de pierna": 5.5,
    "día de pecho": 5.0,
    "día de espalda": 5.0,
    "entrené": 5.0,
    "entrene": 5.0,
    "entrenamiento": 5.0,
    "gym": 5.0,
    "yoga": 2.5,
    "pilates": 3.5,
    "stretching": 2.5,
    "estiramientos": 2.5,
    # ponytail: promedio recreativa(6.0)/intensa(10.0) — el registro manual no
    # captura intensidad como el LLM; si se agrega un selector, usar las claves
    # específicas de METS_CARDIO en su lugar.
    "natación": 7.0,
    "natacion": 7.0,
    "nadar": 7.0,
    "boxeo": 9.0,
    "zumba": 6.5,
    "leg extension": 3.5,
    "leg raise": 3.8,
    "leg curl": 4.5,
    "calf raise": 3.5,
    "lateral raise": 3.0,
}

METS_GYM: Dict[str, float] = {
    **METS_CARDIO, **METS_FUERZA, **METS_FUNCIONAL, **METS_DEPORTES, **METS_EXTRA,
}


def tabla_prompt_texto() -> str:
    """Genera el bloque de texto de la tabla MET para _PROMPT_EJERCICIO,
    a partir de las mismas 4 tablas que usa el registro manual — así el
    LLM y el Constructor de Rutinas nunca pueden desincronizarse."""

    def _linea(items: Dict[str, float]) -> str:
        return "  ".join(f"{nombre.title()}={met:g}" for nombre, met in items.items())

    return (
        "CARDIO:\n"
        f"  {_linea(METS_CARDIO)}\n\n"
        "FUERZA (gym):\n"
        f"  {_linea(METS_FUERZA)}\n\n"
        "FUNCIONAL / HIIT:\n"
        f"  {_linea(METS_FUNCIONAL)}\n\n"
        "DEPORTES:\n"
        f"  {_linea(METS_DEPORTES)}"
    )
