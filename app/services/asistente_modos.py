"""
Cuatro funciones del asistente cliente (una por conversación lógica):
  recomendar_nutricion | registrar_nutricion | recomendar_ejercicio | registrar_ejercicio
+ otro (saludo, progreso, dudas generales) sin forzar tarjetas estructuradas.
"""
from __future__ import annotations

from typing import Any, Optional

import re
from app.services.parsing.contexto_respuesta import inferir_forzar_por_mensaje_usuario

RECOMENDAR_NUTRICION = "recomendar_nutricion"
REGISTRAR_NUTRICION = "registrar_nutricion"
RECOMENDAR_EJERCICIO = "recomendar_ejercicio"
REGISTRAR_EJERCICIO = "registrar_ejercicio"
OTRO = "otro"

# Verbos imperativos que indican registro explícito — tienen prioridad sobre el LLM.
# Si el mensaje COMIENZA con alguno de estos, se fuerza REGISTRAR_NUTRICION sin LLM.
_VERBOS_IMPERATIVOS_REGISTRO: tuple[str, ...] = (
    "regístrame", "registrame", "registra que", "registra el", "registra la",
    "anota que", "anota el", "anota la", "anótame",
    "guarda que", "guarda el", "guarda la", "guárdame",
    "ponme", "ponme en", "agrégame", "agregame",
    "apunta el", "apunta la", "apunta que",
)


MODOS_ASISTENTE = frozenset(
    {
        RECOMENDAR_NUTRICION,
        REGISTRAR_NUTRICION,
        RECOMENDAR_EJERCICIO,
        REGISTRAR_EJERCICIO,
        OTRO,
    }
)


def intent_prioritario_para_parser(intent_ia: Optional[str], modo_funcion: Optional[str]) -> str:
    """
    Si el servidor ya clasificó el mensaje en uno de los 4 modos, el parseo usa un intent
    estable aunque el modelo olvide el tag [CALOFIT_INTENT:…].
    """
    modo = (modo_funcion or OTRO).strip().lower()
    if modo == RECOMENDAR_NUTRICION:
        return "RECIPE"
    if modo == REGISTRAR_NUTRICION:
        return "LOG"
    if modo == RECOMENDAR_EJERCICIO:
        return "POWER"
    if modo == REGISTRAR_EJERCICIO:
        return "LOG"
    return (intent_ia or "CHAT").upper()


def detectar_modo_funcion(mensaje: str, es_saludo: bool) -> str:
    """
    Elige una de las 4 funciones o OTRO. Prioriza registro sobre recomendación si hay señales claras.
    """
    if es_saludo:
        return OTRO
    m = (mensaje or "").lower().strip()
    if len(m) < 2:
        return OTRO

    # ── Prioridad máxima: verbo imperativo de registro ────────────────────────
    # "Regístrame el ceviche de caballa" → REGISTRAR_NUTRICION inmediato (sin LLM)
    if any(m.startswith(v) for v in _VERBOS_IMPERATIVOS_REGISTRO):
        return REGISTRAR_NUTRICION

    comida_ctx = (
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
        # Verbos de ingesta de líquidos
        "tomé",
        "tome ",
        "bebí",
        "bebi ",
        "me tomé",
        "me tome",
        "me bebí",
        "gaseosa",
        "limonada",
        "chicha",
        "jugo",
        "refresco",
        "cerveza",
    )
    ej_ctx = (
        "ejercicio",
        "entren",
        "gym",
        "gimnasio",
        "rutina",
        "pecho",
        "pector",
        "espalda",
        "bicep",
        "bícep",
        "tricep",
        "trícep",
        "hombro",
        "hombros",
        "gluteo",
        "glúteo",
        "abdomen",
        "abdominal",
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
        "core",
    )
    fc = sum(1 for x in comida_ctx if x in m)
    fe = sum(1 for x in ej_ctx if x in m)

    # Narración diaria de comidas: "Hoy desayuné X, almorcé Y, cené Z"
    # El LLM clasifica estos mensajes como recomendar_nutricion por error.
    # Señales: adverbio temporal + verbo pasado + alimento → siempre registro.
    _VERBOS_LOG = (
        "desayuné", "desayune", "almorcé", "almorce", "cené", "cene",
        "comí", "comi ", "tomé ", "tome ", "bebí ", "bebi ",
        "he desayunado", "he almorzado", "he cenado", "he comido", "he tomado",
        # Frases más naturales
        "acabo de comer", "acabo de tomar", "acabo de beber",
        "me comi", "me tomé", "me bebí", "me comí",
        "termine de comer", "terminé de comer",
        "ya comi", "ya comí", "ya almorcé", "ya desayuné",
        # "me hice un desayuno/almuerzo" — construcción coloquial de preparación+consumo
        "me hice un", "me hice una",
    )
    _TEMPORALES_LOG = ("hoy ", "hoy,", "esta mañana", "esta tarde", "esta noche",
                       "al mediodia", "al mediodía", "ayer ", "en la mañana", "en la noche")
    # Usamos límites de palabra (\b) para evitar coincidencias parciales (ej. darme comidas -> me comi)
    _verbos_log_count = sum(
        1 for v in _VERBOS_LOG
        if m.startswith(v) or re.search(rf"\b{re.escape(v.strip())}\b", m)
    )
    _tiene_temporal_log = any(t in m for t in _TEMPORALES_LOG)
    # Una de estas condiciones basta:
    # a) ≥2 verbos de consumo pasado en el mismo mensaje (resumen del día)
    # b) adverbio temporal + ≥1 verbo de consumo pasado + alimento
    # c) ≥1 verbo de consumo pasado + alimento (cualquier verbo de ingesta es señal de log)
    if _verbos_log_count >= 1 and fc > 0 and "?" not in m:
        return REGISTRAR_NUTRICION

    registro = any(
        x in m
        for x in (
            "registr",
            "anoté",
            "anote",
            "apunta",
            "ya comí",
            "ya comi",
            "acabo de comer",
            "acabo de entren",
            "terminé de entren",
            "termine de entren",
        )
    )
    if registro:
        if fc > fe:
            return REGISTRAR_NUTRICION
        if fe > fc:
            return REGISTRAR_EJERCICIO
        return OTRO

    rec_nut = any(
        x in m
        for x in (
            "qué como", "que como", "qué comer", "que comer",
            "ideas para comer", "sugerencias de comida",
            "opciones de almuerzo", "opciones de desayuno", "opciones de cena",
            "recetas", "receta ", "platos para",
            # Frases naturales de hambre/recomendación
            "tengo hambre", "tengo antojo", "qué puedo comer", "que puedo comer",
            "me recomiendas comer", "me da hambre", "quiero comer algo",
            "qué me como", "que me como", "qué hay para comer", "que hay para comer",
            "busco algo para comer", "dame algo de comer",
        )
    )
    # Ejercicio registrado: "entrené X", "realicé X", etc.
    # NOTA: "hice " (genérico) fue eliminado — matcheaba "me hice un desayuno".
    # Usar solo formas específicas de ejercicio.
    _EJERCICIO_REGISTRADO = (
        # Formas genéricas verificadas
        "hice cardio", "hice pesas", "hice gym", "hice ejercicio",
        "hice sentadillas", "hice flexiones", "hice abdominales",
        # Ejercicios específicos de gym (evita "hice " solo que matchea "me hice un desayuno")
        "hice press", "hice curl", "hice remo", "hice jalon", "hice jalón",
        "hice dominadas", "hice fondos", "hice burpees", "hice plancha",
        "hice peso muerto", "hice sentadilla", "hice extensiones",
        "tiré press", "tire press", "tiré sentadillas",
        "fui al gym", "fui al gimnasio", "entrené", "entrenei", "entrenei ",
        "realicé", "realize", "corrí", "corri ", "caminé", "camine ",
        "nadé", "nade ", "pedalié", "pedalee",
        "terminé de entrenar", "termine de entrenar",
        "acabo de entrenar", "acabo de hacer ejercicio",
        "sali a correr", "salí a correr", "sali a trotar", "salí a trotar",
        "sali a caminar", "salí a caminar", "sali a entrenar", "salí a entrenar",
        "correr durante", "trotar durante", "caminar durante",
    )
    if any(x in m for x in _EJERCICIO_REGISTRADO) and "?" not in m and fe >= fc:
        return REGISTRAR_EJERCICIO
    # Preguntas de permiso/capacidad sobre actividad física → OTRO informativo
    # "puedo realizar un trote", "se puede trotar", "es bueno correr" — NO son registros
    _PERMISO_EJ_SYNC = (
        "puedo realizar", "puedo trotar", "puedo correr", "puedo nadar",
        "puedo caminar", "puedo jugar", "puedo bailar", "puedo practicar",
        "puedo ir a ", "se puede trotar", "se puede correr", "se puede nadar",
        "es bueno trotar", "es bueno correr", "es bueno nadar", "es bueno caminar",
        "es malo trotar", "es malo correr",
    )
    if any(p in m for p in _PERMISO_EJ_SYNC):
        return OTRO

    rec_ex = any(
        x in m
        for x in (
            "qué ejercicio",
            "que ejercicio",
            "ejercicios puedo hacer",
            "ejercicios para",
            "ejercicio para",
            "para pecho",
            "pecho ",
            "rutina para",
            "rutina de",
            "cómo entren",
            "como entren",
            "qué hacer en el gym",
            "que hacer en el gym",
            "entreno de",
            "cuádriceps",
            "cuadriceps",
            "isquios",
            "isquio",
            "femoral",
            "gemelos",
            "gemelo",
            "pantorrilla",
            "vasto",
            "recto femoral",
        )
    )
    if rec_ex and not rec_nut:
        return RECOMENDAR_EJERCICIO
    if rec_nut and not rec_ex:
        return RECOMENDAR_NUTRICION
    if rec_ex and rec_nut:
        return OTRO

    # Si contiene intenciones claras de recetas o técnicas, forzar a OTRO
    _RECETA_O_TECNICA = (
        "como se hace", "como se prepara", "como hacer", "receta de", "como cocinar",
        "ingredientes de", "preparacion de", "tecnica de", "como realizar",
        "como ejecutar", "pasos para", "forma correcta", "preparar "
    )
    import unicodedata as _ud_rec
    _mn_rec = "".join(c for c in _ud_rec.normalize("NFD", m) if _ud_rec.category(c) != "Mn")
    if any(k in _mn_rec for k in _RECETA_O_TECNICA):
        return OTRO

    inf = inferir_forzar_por_mensaje_usuario(mensaje)
    if inf == "comida":
        return RECOMENDAR_NUTRICION
    if inf == "ejercicio":
        return RECOMENDAR_EJERCICIO
    return OTRO


async def resolver_modo_funcion(ia: Any, mensaje: str, es_saludo: bool) -> str:
    """
    Clasificación de intención: 4 pre-checks infalibles + LLM 70B para todo lo demás.
    Reemplaza los ~200 pre-checks anteriores por reglas mínimas y máxima confianza en el LLM.
    """
    _m = (mensaje or "").lower().strip()
    if len(_m) < 2:
        return OTRO

    # Si viene marcado como saludo, validar que sea un saludo puro (sin peticiones de comida/ejercicio)
    if es_saludo:
        _NO_SALUDO_PURO = (
            "comer", "cenar", "almorzar", "desayunar", "merendar", "comida", "cena", "almuerzo", "desayuno", "plato", "receta",
            "ejercicio", "entren", "rutina", "pecho", "espalda", "pierna", "hombro", "cardio", "gym", "gimnasio",
            "dime", "puedo", "quiero", "sugier", "recomiend", "dame", "verdura", "hacer",
            "correr", "corrí", "corri", "trotar", "troté", "trote", "caminar", "caminé", "camine",
            "nadar", "nadé", "nade", "bicicleta", "salí", "sali", "pesa", "pesas", "pesos", "comi", "comí", "tomar", "tome", "tomé"
        )
        if any(k in _m for k in _NO_SALUDO_PURO):
            es_saludo = False

    if es_saludo:
        return OTRO

    # ── Pre-check 1: verbos imperativos de registro ──────────────────────────────
    if any(_m.startswith(v) for v in _VERBOS_IMPERATIVOS_REGISTRO):
        return REGISTRAR_NUTRICION

    # ── Pre-check 1b: petición de recomendación de comida ───────────────────────
    # "qué puedo comer", "dime qué almorzar", "dame opciones" → RECOMENDAR_NUTRICION
    # DEBE ir ANTES del pre-check modal para no ser capturado por "puedo"
    import unicodedata as _ud
    _mn = "".join(c for c in _ud.normalize("NFD", _m) if _ud.category(c) != "Mn")
    _PEDIR_REC_NUT = (
        "que puedo comer", "que puedo almorzar", "que puedo cenar",
        "que puedo desayunar", "que puedo merendar",
        "dime que comer", "dime que almorzar", "dime que cenar",
        "que me recomiendas comer", "que me recomiendas almorzar",
        "que comer ahora", "que almorzar hoy", "que cenar hoy",
        "opciones de comida", "opciones para almorzar", "opciones para cenar",
        "que puedo comer en este momento",
    )
    _EJ_KW = ("ejercicio", "entren", "gym", "rutina", "deporte", "ejercicios")
    if any(p in _mn for p in _PEDIR_REC_NUT) and not any(e in _mn for e in _EJ_KW):
        return RECOMENDAR_NUTRICION
    # Petición de recomendación de ejercicio
    _PEDIR_REC_EJ = (
        "que ejercicios puedo", "que ejercicios hago", "dime que ejercicios",
        "ejercicios segun mi plan", "ejercicios según mi plan",
        "que ejercicios hacer", "ejercicios para hoy",
    )
    if any(p in _mn for p in _PEDIR_REC_EJ):
        return RECOMENDAR_EJERCICIO

    # ── Pre-check 2: modal de permiso = SIEMPRE pregunta, nunca registro ─────────
    _MODALES = (
        "puedo ", "se puede ", "podria ", "podria ",
        "es bueno ", "es malo ", "es buena ", "es mala ",
        "seria bueno ", "es recomendable ", "es posible ", "conviene ",
    )
    if any(_mn.startswith(p) or f" {p}" in _mn for p in _MODALES):
        return OTRO

    # ── Pre-check 2b: recetas y técnicas = SIEMPRE conversacional (OTRO) ─────────
    _RECETA_O_TECNICA = (
        "como se hace", "como se prepara", "como hacer", "receta de", "como cocinar",
        "ingredientes de", "preparacion de", "tecnica de", "como realizar",
        "como ejecutar", "pasos para", "forma correcta", "preparar "
    )
    if any(k in _mn for k in _RECETA_O_TECNICA):
        return OTRO

    # ── Pre-check 3: verbos de consumo pasado inequívocos ────────────────────────
    # Usamos límites de palabra (\b) y el texto normalizado _mn para evitar coincidencias parciales
    _CONSUMO_CLARO_NORM = (
        "comi", "desayune", "almorce", "cene", "bebi", "me jale",
        "me comi", "me tome", "acabo de comer", "acabo de tomar", "acabo de beber",
        "termine de comer", "he desayunado", "he almorzado", "he cenado",
        "he comido", "he bebido",
    )
    _tiene_consumo = any(
        _mn.startswith(v) or re.search(rf"\b{re.escape(v.strip())}\b", _mn)
        for v in _CONSUMO_CLARO_NORM
    )
    if _tiene_consumo and "?" not in _m:
        return REGISTRAR_NUTRICION

    # ── Pre-check 4: verbos de ejercicio pasado inequívocos ──────────────────────
    # Usamos límites de palabra (\b) y el texto normalizado _mn para evitar coincidencias parciales
    _EJ_CLARO_NORM = (
        "entrene", "entrenei", "corri", "camine", "nade", "fui al gym", "fui al gimnasio",
        "hice press", "hice sentadilla", "hice curl", "hice dominada", "hice peso muerto",
        "hice remo", "hice jalon", "hice fondos", "hice cardio", "hice pesas", "hice ejercicio",
        "hice flexiones", "hice burpees", "hice abdominales", "termine de entrenar",
        "acabo de entrenar", "acabo de hacer ejercicio", "tire press",
        "sali a correr", "salí a correr", "sali a trotar", "salí a trotar",
        "sali a caminar", "salí a caminar", "sali a entrenar", "salí a entrenar",
        "correr durante", "trotar durante", "caminar durante",
    )
    _tiene_ej = any(
        _mn.startswith(v) or re.search(rf"\b{re.escape(v.strip())}\b", _mn)
        for v in _EJ_CLARO_NORM
    )
    if _tiene_ej and "?" not in _m:
        return REGISTRAR_EJERCICIO

    # ── Pre-check 4b: patrón de volumen de ejercicio SIN verbo ───────────────────
    # "sentadillas 4x12 con 80kg", "press banca 3*10 70kg", "prensa 4×15 100kg"
    # El usuario reporta series×reps@peso sin usar "hice" explícitamente.
    import re as _re_vol
    _NOMBRES_EJ = (
        "sentadill", "press", "curl", "remo", "jalon", "jalón", "fondos",
        "prensa", "dominad", "peso muerto", "extensi", "elevaci", "hip thrust",
        "burpee", "plancha", "abdominal",
    )
    _tiene_nombre_ej = any(n in _mn for n in _NOMBRES_EJ)
    _tiene_volumen   = bool(_re_vol.search(r'\d+\s*[x\*×]\s*\d+', _mn))
    if _tiene_nombre_ej and _tiene_volumen and "?" not in _m:
        return REGISTRAR_EJERCICIO

    # ── LLM 70B: clasificación definitiva para todo lo demás ─────────────────────
    try:
        modo_ia = await ia.clasificar_modo_asistente(mensaje)
        if isinstance(modo_ia, str) and modo_ia in MODOS_ASISTENTE:
            return modo_ia
    except Exception:
        pass
    return detectar_modo_funcion(mensaje, False)

# ─── CÓDIGO ELIMINADO (reemplazado por LLM 70B) ──────────────────────────────
# Las ~200 líneas de pre-checks anteriores (VERBOS_CONSUMO_PASADO, FRASES_INGESTA_NATURAL,
# ALIMENTOS_RAPIDOS, QUERER_EJERCICIO, PREGUNTAS_PERMISO_EJ, PIDE_LISTA_EJ, etc.)
# generaban falsos positivos porque cada nuevo caso requería un parche manual.
# El LLM 70B con un prompt estructurado cubre todos esos casos sin mantenimiento.
# ─────────────────────────────────────────────────────────────────────────────

def bloque_prompt_modo_funcion(modo: str) -> str:
    """Texto inyectado al LLM para acotar la respuesta a una sola función."""
    modo = (modo or OTRO).strip().lower()
    if modo not in MODOS_ASISTENTE or modo == OTRO:
        return (
            "\n\n══ FUNCIÓN DEL ASISTENTE (MODO CONVERSACIONAL) ══\n"
            "Responde como un coach amigo: directo, cálido, MUY BREVE.\n\n"
            "⛔ REGLA #1 — LONGITUD MÁXIMA ABSOLUTA: 2 frases. Máximo 50 palabras en total.\n"
            "  Cuenta tus palabras. Si llegas a 51, borra hasta quedarte en 50 o menos.\n"
            "  PROHIBIDO: párrafos largos, listas, explicaciones extensas, múltiples puntos.\n\n"
            "⛔ REGLA #2 — CIERRE SIN PREGUNTA: la última frase NUNCA es una pregunta.\n"
            "  PROHIBIDO terminar con '¿...?' de ningún tipo.\n"
            "  Termina siempre con afirmación, dato concreto o recomendación corta.\n\n"
            "⛔ REGLA #3 — SIN TAGS: NO escribas [CALOFIT_INTENT], [CALOFIT_HEADER] ni ningún corchete.\n\n"
            "PLANTILLA SEGÚN TIPO DE PREGUNTA:\n"
            "• '¿puedo comer X?' → '[Sí/No], [razón en 5 palabras]. [alternativa o tip en 10 palabras].'\n"
            "  Ejemplo: 'No, el pollo no es vegano. Prueba con tofu o lentejas que tienen buena proteína.'\n"
            "• '¿puedo trotar/nadar/correr?' → '[Sí/claro/depende]. [beneficio o condición en 10 palabras].'\n"
            "  Ejemplo: 'Claro, el trote mejora el cardio. Empieza 20 min a ritmo cómodo.'\n"
            "• Saludo / respuesta corta → 1 frase amigable. Sin preguntas de vuelta.\n"
            "• Consulta de parámetros de ejercicio → dato puntual + tip. Sin tarjetas.\n"
        )

    instrucciones = {
        RECOMENDAR_NUTRICION: (
            "\n\n══ FUNCIÓN ACTUAL (OBLIGATORIA, ÚNICA) ══\n"
            "FUNCION: RECOMENDAR_NUTRICIÓN.\n"
            "🚨 ANTES DE GENERAR CUALQUIER PLATO — LEE ESTO:\n"
            "• Revisa DIETA del perfil (bloque ALERGIAS/DIETA/CONDICIONES arriba).\n"
            "• Si DIETA = 'Vegano': PROHIBIDO ABSOLUTO en TODOS los platos: carne, res, pollo, pechuga, cerdo, "
            "pato, pavo, pescado, mariscos, huevo, leche, queso, yogur, mantequilla. "
            "SOLO ingredientes 100% vegetales. Verifica cada ingrediente de cada plato antes de escribirlo.\n"
            "• Si DIETA = 'Vegetariano': PROHIBIDO: carne, pollo, pescado, mariscos. Huevos y lácteos permitidos.\n"
            "• kcal por opción ≤ kcal RESTANTES del día (ver STATUS DEL DÍA → Restante). "
            "Si restante=648 kcal, NINGUNA opción puede superar 648 kcal.\n"
            "• INCUMPLIR CUALQUIERA DE ESTAS REGLAS ES ERROR GRAVE — invalida toda la respuesta.\n"
            "• DISEÑO ÚNICO: todas las opciones (2–3) con la MISMA plantilla CALOFIT, en este orden por opción:\n"
            "  [CALOFIT_HEADER]…[/CALOFIT_HEADER] → [CALOFIT_JUSTIF]…[/CALOFIT_JUSTIF] → [CALOFIT_LIST]…[/CALOFIT_LIST] → "
            "[CALOFIT_ACTION]…[/CALOFIT_ACTION] → [CALOFIT_STATS]…[/CALOFIT_STATS].\n"
            "• TEXTO CONVERSACIONAL ANTES DE LAS TARJETAS — OBLIGATORIO y NATURAL:\n"
            "  Escribe 1-2 frases que combinen de forma FLUIDA (no como plantilla):\n"
            "    - El momento del día (desayuno/almuerzo/cena/snack del contexto «MOMENTO DEL DÍA»)\n"
            "    - Las kcal disponibles (de «PROGRESO DEL DÍA» → Restante)\n"
            "    - Las condiciones del perfil relevantes SI son importantes para las opciones\n"
            "  CLAVE: varía el inicio y el tono. NO uses siempre el mismo patrón.\n"
            "  EJEMPLOS de cómo puede sonar (NO los copies literalmente, varía):\n"
            "  - 'Todavía tienes 1500 kcal para tu almuerzo. Pensé en opciones que te llenan bien sin pasarte.'\n"
            "  - 'Para el desayuno de hoy te quedan 2000 kcal. Con tu objetivo de ganar masa, estos platos te vienen perfectos.'\n"
            "  - '450 kcal para la cena — suficiente para algo rico y ligero. Acá van mis sugerencias:'\n"
            "  - 'Buena pregunta. Para tu snack de ahora te recomiendo algo que te sostenga hasta la cena sin pasarte:'\n"
            "  - 'Teniendo diabetes, elegí opciones de bajo índice glucémico para tu almuerzo. Te quedan 800 kcal:'\n"
            "  PROHIBIDO: texto genérico sin datos reales, o que solo repita el objetivo sin números.\n"
            "  PROHIBIDO para snack: mencionar el déficit total del día (ej: 'te faltan 2355 kcal').\n"
            "• [CALOFIT_JUSTIF] OBLIGATORIO: 1 frase corta vinculando el plato AL DÉFICIT Y CONDICIONES del día "
            "(ej. 'Cubre ~35g de proteína vegetal que te faltan hoy, apto para diabetes', "
            "'Carbos complejos de bajo índice glucémico — ideal para diabetes y tus X kcal restantes'). "
            "Usa los números reales del contexto — no justificaciones genéricas.\n"
            "• PLATOS COMPLETOS OBLIGATORIO: cada opción debe ser un plato preparado con al menos 2 componentes "
            "(proteína + carbohidrato, o proteína + vegetal, etc.). "
            "PROHIBIDO sugerir un ingrediente solo como opción (ej. 'Arroz Integral' solo, 'Pollo' solo). "
            "Mínimo: 'Arroz Integral con Pollo a la Plancha', 'Ensalada de Lentejas con Tomate', etc.\n"
            "• PROHIBIDO mezclar formatos: ninguna receta en markdown, emojis sueltos ni párrafo tipo "
            "«Opción 1: …» fuera de tags si el resto va en CALOFIT. El usuario solo verá tarjetas coherentes si TODO va en tags.\n"
            "• Debes usar [CALOFIT_INTENT:RECIPE] al inicio (y [CALOFIT_QUESTION_TYPE:ABIERTA] si aplica).\n"
            "• EJEMPLO CORRECTO (sigue EXACTAMENTE este patrón, sin variaciones):\n"
            "[CALOFIT_INTENT:RECIPE]\n"
            "Para tu desayuno, te faltan 1800 kcal. Como tienes diabetes y tu objetivo es mantener el peso, elegí platos con proteína completa y carbos de bajo índice glucémico para completar tu balance sin elevar el azúcar.\n"
            "[CALOFIT_HEADER]Tortilla de Huevo con Pan Integral[/CALOFIT_HEADER]\n"
            "[CALOFIT_JUSTIF]Cubre ~18g de proteína que te faltan hoy — sin azúcares añadidos, apto para diabetes.[/CALOFIT_JUSTIF]\n"
            "[CALOFIT_LIST]\n"
            "2 huevos (140 kcal)\n"
            "2 rebanadas pan integral (160 kcal)\n"
            "1 cdta aceite de oliva (40 kcal)\n"
            "[/CALOFIT_LIST]\n"
            "[CALOFIT_ACTION]\n"
            "1. Bate los huevos con sal y pimienta.\n"
            "2. Calienta el aceite en sartén a fuego medio.\n"
            "3. Cocina la tortilla 3 min por lado.\n"
            "4. Tuesta el pan y sirve junto a la tortilla.\n"
            "[/CALOFIT_ACTION]\n"
            "[CALOFIT_STATS]P: 18g | C: 32g | G: 8g | Cal: 340kcal[/CALOFIT_STATS]\n"
            "• EJEMPLO INCORRECTO (NUNCA hagas esto):\n"
            "Aquí opciones: Tortilla de Huevo 2 huevos (140 kcal) 2 panes (160 kcal)\n"
            "[CALOFIT_HEADER]Sugerencia 1[/CALOFIT_HEADER] ← PROHIBIDO: nombre genérico y opción anterior en texto libre\n"
            "• ⚠️ LÍMITE CALÓRICO ESTRICTO POR OPCIÓN: antes de escribir cada [CALOFIT_STATS], "
            "suma las kcal de todos los ingredientes del [CALOFIT_LIST]. "
            "Si esa suma supera las kcal RESTANTES del día (STATUS DEL DÍA → Restante), "
            "REDUCE las porciones o cambia el plato hasta que la suma sea ≤ Restante. "
            "NUNCA muestres una opción cuyas kcal totales superen el Restante. "
            "Ejemplo: si Restante=648 kcal, el total de ingredientes debe ser ≤648 kcal. "
            "Si Restante<150 kcal, sugiere solo fruta pequeña, infusión o snack mínimo.\n"
            "• [CALOFIT_LIST] obligatorio: 4–8 líneas, una por ingrediente, con gramos y SOLO kcal entre paréntesis. "
            "  Ejemplo: \"150g pechuga de pollo (240 kcal)\". PROHIBIDO meter macros por ingrediente (ej. \"37g proteína\"). "
            "[CALOFIT_STATS] obligatorio en una sola línea: P: Xg | C: Yg | G: Zg | Cal: Wkcal. "
            "Verifica coherencia: 4×P + 4×C + 9×G ≈ W (±8%). Si hay pollo, pescado, huevo, lácteos, lentejas o "
            "legumbres en LIST o en el nombre del plato, P no puede ser 0. W debe ser la suma de kcal de LIST (±5%).\n"
            "• [CALOFIT_ACTION] obligatorio y útil: 3–6 pasos numerados (1., 2., 3. ...), "
            "cada paso debe tener AL MENOS 8–12 palabras y debe incluir detalle mínimo "
            "(tiempo aproximado, orden lógico, calor medio/alto, hervir/sancochar/saltear, etc.). "
            "PROHIBIDO: 1 solo paso genérico tipo \"Prepara y sirve\".\n"
            "• PERÚ + HORA + RANGO CALÓRICO OBLIGATORIO: usa el bloque «MOMENTO DEL DÍA» del contexto y respeta "
            "ESTRICTAMENTE el rango calórico por momento. NUNCA propongas un plato fuera de ese rango:\n"
            "  - DESAYUNO        → 250–550 kcal por opción. Platos ligeros: tostadas, avena, huevos, frutas con proteína.\n"
            "  - ALMUERZO        → 450–900 kcal por opción. Platos completos: arroz+proteína, sopas, guisos.\n"
            "  - CENA            → 200–500 kcal por opción. Ligero: ensaladas, sopas, huevos, verduras con proteína.\n"
            "  - SNACK/MERIENDA  → 80–300 kcal por opción. SOLO snacks reales y satisfactorios:\n"
            "      Ejemplos válidos: fruta fresca, avena con fruta, frutos secos, tostada con palta o mantequilla de maní, "
            "yogur de soja, barritas de avena, smoothie de frutas, plátano con chía.\n"
            "      PROHIBIDO como snack: ensaladas de verduras solas (lechuga+pepino), arroz integral completo, "
            "caldo completo, lentejas con arroz, sopas de fondo, platos principales >300 kcal.\n"
            "      El [CALOFIT_JUSTIF] del snack NO debe decir 'Cubre X gramos que te faltan hoy' — "
            "debe decir por qué ESE snack te sostendrá hasta la siguiente comida "
            "(ej: 'Energía natural para sostenerte hasta el almuerzo sin elevar el azúcar', "
            "'Fibra y carbos lentos que mantienen la saciedad hasta el mediodía').\n"
            "  Si el momento es SNACK o MERIENDA, el TEXTO CONVERSACIONAL NO menciona el déficit total del día "
            "(2000+ kcal es confuso para un snack) — en cambio di cuánto aporta el snack y para qué lo preparaste:\n"
            "      Ejemplo CORRECTO para snack: 'Para tu snack de las 10am, elegí opciones de 100–300 kcal "
            "que te sostendrán hasta el almuerzo sin elevar el azúcar en sangre.'\n"
            "      PROHIBIDO para snack: 'Te faltan 2355 kcal. Elegí platos...' ← confuso, no aplica.\n"
            "• Cocina peruana reconocible; ingredientes y técnica deben tener sentido juntos "
            "(no mezclas imposibles solo por sumar macros).\n"
            "• COHERENCIA NOCTURNA: si MOMENTO DEL DÍA indica NOCTURNO, las opciones deben ser realmente ligeras "
            "(snacks o platos suaves ≤400 kcal). PROHIBIDO proponer pollería (\"pollo a la brasa\") o platos pesados.\n"
            "• PROHIBIDO ingredientes poco comunes en Perú (ej. estragón/tarragón). Usa hierbas típicas: culantro, "
            "perejil, orégano, ají amarillo/panca, kion, ajo, cebolla.\n"
            "• VARIEDAD: platos distintos entre mensajes; no repitas las mismas recetas salvo que lo pidan.\n"
            "• TEXTO fuera de tags: 1–2 frases con los datos reales del déficit (ver regla de TEXTO arriba). "
            "NO repitas el nombre del usuario dos veces. Sin ingredientes ni macros sueltos ahí. "
            "No termines con «: 1.» ni «sugerencias: 2.» (enumeración huérfana); las opciones van solo en CALOFIT.\n"
            "• NOMBRES DE PLATOS: deben sonar a comida real y natural (Perú), no descripciones raras "
            "tipo \"lentejas cocidas con verduras\". Mejor: \"sopa de lentejas ligera\", \"crema de verduras\", "
            "\"tortilla de huevo con pan tostado\", etc.\n"
            "• PROHIBIDO rutinas de gym en esta respuesta.\n"
            "• PROHIBIDO simular registro de lo ya comido.\n"
            "• Si falta un solo dato importante para recomendar (ej. momento del día ya implícito en el contexto no cuenta), "
            "haz como máximo UNA pregunta corta; si el mensaje ya es suficiente, responde directo sin encadenar más preguntas.\n"
        ),
        REGISTRAR_NUTRICION: (
            "\n\n══ FUNCIÓN ACTUAL (OBLIGATORIA, ÚNICA) ══\n"
            "FUNCION: REGISTRAR_NUTRICIÓN.\n"
            "• Debes usar [CALOFIT_INTENT:LOG] al inicio.\n"
            "• Resume lo que el usuario comió, estima macros y deja listo para confirmar; "
            "puedes usar secciones tipo comida si ayudan.\n"
            "• PROHIBIDO rutinas de ejercicio o bloques POWER en esta respuesta.\n"
        ),
        RECOMENDAR_EJERCICIO: (
            "\n\n══ FUNCIÓN ACTUAL (OBLIGATORIA, ÚNICA) ══\n"
            "FUNCION: RECOMENDAR_EJERCICIO.\n"
            "• OBLIGATORIO: Ya sea que el usuario pida una 'rutina', 'entrenamiento' o simplemente 'ejercicios de pecho', "
            "SIEMPRE debes usar [CALOFIT_INTENT:POWER] al inicio y estructurar tarjetas interactivas.\n"
            "• REGLA DE SESIÓN ÚNICA: NUNCA generes 'rutinas de 4 días' ni planes semanales. Diseña SIEMPRE y ÚNICAMENTE el entrenamiento "
            "para la sesión de HOY (un solo día). Si piden un plan largo, dales solo la rutina del Día 1.\n"
            "• DEBES usar [CALOFIT_HEADER], [CALOFIT_LIST] (volumen), [CALOFIT_ACTION] (técnica detallada) y [CALOFIT_STATS] "
            "(duración/calorías) POR CADA EJERCICIO INDIVIDUALMENTE.\n"
            "• TÉCNICA UNIVERSAL: El bloque [CALOFIT_ACTION] debe tener pasos enumerados claros (1, 2, 3...) "
            "con explicaciones que sirvan tanto para un principiante como para un profesional. "
            "NUNCA uses palabras como 'Preparación:', 'Movimiento:' o 'Respiración:' al inicio de los pasos. "
            "Simplemente enumera las instrucciones de postura, ejecución y respiración de forma fluida. Termina siempre con un 'Tip:' clave.\n"
            "      • EJEMPLO CORRECTO OBLIGATORIO PARA CUALQUIER SUGERENCIA O RUTINA (UN BLOQUE POR EJERCICIO):\n"
            "[CALOFIT_INTENT:POWER]\n"
            "Aquí tienes excelentes opciones para tu entrenamiento de hoy:\n"
            "[CALOFIT_HEADER]Press Banca Plano[/CALOFIT_HEADER]\n"
            "[CALOFIT_LIST]\n"
            "Músculo: Pectoral mayor\n"
            "Equipo: Banco plano y barra\n"
            "Volumen: 4 series de 10 reps\n"
            "[/CALOFIT_LIST]\n"
            "[CALOFIT_ACTION]\n"
            "1. Acuéstate en el banco asegurando que tus pies estén firmes en el suelo y junta tus omóplatos hacia atrás para estabilizar los hombros.\n"
            "2. Inhala profundo y baja la barra de forma controlada (aprox. 2 segundos) hasta que roce ligeramente la parte media de tu pecho.\n"
            "3. Empuja la barra hacia arriba de forma explosiva mientras exhalas con fuerza, sin despegar los glúteos del banco.\n"
            "Tip: Mantener los codos ligeramente metidos hacia el cuerpo (a unos 45 grados) protege tus hombros de lesiones.\n"
            "[/CALOFIT_ACTION]\n"
            "[CALOFIT_STATS]Dur: 15 min | Cal: 120 kcal[/CALOFIT_STATS]\n"
            "\n"
            "[CALOFIT_HEADER]Sentadilla Libre[/CALOFIT_HEADER]\n"
            "[CALOFIT_LIST]\n"
            "Músculo: Cuádriceps y glúteos\n"
            "Equipo: Rack de sentadillas y barra\n"
            "Volumen: 4 series de 12 reps\n"
            "[/CALOFIT_LIST]\n"
            "[CALOFIT_ACTION]\n"
            "1. Apoya la barra firmemente sobre tus trapecios (espalda alta, no el cuello) y separa tus pies al ancho de tus hombros.\n"
            "2. Toma aire, aprieta el abdomen y desciende empujando la cadera hacia atrás, como si fueras a sentarte, manteniendo la espalda recta.\n"
            "3. Baja hasta que tus muslos rompan la línea paralela al suelo, luego sube exhalando y empujando con fuerza desde tus talones.\n"
            "Tip: Asegúrate de que durante todo el movimiento tus rodillas apunten en la misma dirección que las puntas de tus pies.\n"
            "[/CALOFIT_ACTION]\n"
            "[CALOFIT_STATS]Dur: 20 min | Cal: 200 kcal[/CALOFIT_STATS]\n"
            "• REGLA DE TIEMPO (IMPORTANTE): Siempre asigna un tiempo realista en [CALOFIT_STATS] a cada ejercicio.\n"
            "• VARIEDAD: alterna ejercicios y patrones entre respuestas.\n"
            "• PROHIBIDO: NUNCA respondas con simples viñetas Markdown (como '- Press Banca'). SIEMPRE usa las etiquetas de tarjetas interactivas.\n"
            "• PROHIBIDO recetas de cocina, ingredientes o menús en esta respuesta.\n"
        ),
        REGISTRAR_EJERCICIO: (
            "\n\n══ FUNCIÓN ACTUAL (OBLIGATORIA, ÚNICA) ══\n"
            "FUNCION: REGISTRAR_EJERCICIO.\n"
            "• Debes usar [CALOFIT_INTENT:LOG] al inicio.\n"
            "• Centra en anotar entrenamiento (ejercicio, series, tiempo o kcal si los da); "
            "formato claro para confirmación.\n"
            "• PROHIBIDO sugerir platos o recetas en esta respuesta.\n"
        ),
    }
    return instrucciones.get(modo, "")
