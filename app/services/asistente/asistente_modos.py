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
    # Usamos límites de palabra (\b) para evitar coincidencias parciales (ej. darme comidas -> me comi)
    _verbos_log_count = sum(
        1 for v in _VERBOS_LOG
        if m.startswith(v) or re.search(rf"\b{re.escape(v.strip())}\b", m)
    )
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
            # Verbos de alimentación en infinitivo → intención de comer
            "almorzar", "desayunar", "merendar",
            # Frases de recomendación de comida
            "para el almuerzo", "para la cena", "para el desayuno", "para la merienda",
            "qué recomiendas", "que recomiendas", "dame ideas", "dame opciones",
            "qué me recomiendas", "que me recomiendas",
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
    # NOTA 2: si el mensaje tiene intención clara de comida (rec_nut), no registrar ejercicio
    # aunque mencione "entrené" de forma incidental ("ya entrené, qué almuerzo?").
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
    if any(x in m for x in _EJERCICIO_REGISTRADO) and "?" not in m and fe >= fc and not rec_nut:
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


async def resolver_modo_funcion(ia: Any, mensaje: str, es_saludo: bool, historial: list = None) -> str:
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
        # Señales directas de hambre/snack (antes dependían del LLM clasificador)
        "tengo hambre", "tengo antojo",
        "una merienda", "un snack",
        "ideas para comer", "ideas para el desayuno", "ideas para el almuerzo",
        "ideas para la cena", "ideas para la merienda",
        "que deberia comer", "que deberia almorzar", "que deberia cenar",
        "que deberia desayunar",
    )
    # Solo bloquean recomendación de COMIDA los keywords que indican contexto de EJERCICIO puro.
    # "entren" y "ejercicio" (singular) se eliminan: "qué puedo comer después de entrenar"
    # es RECOMENDAR_NUTRICION aunque mencione entrenar.
    _EJ_KW = ("gym", "rutina", "ejercicios")
    # Guard: si el mensaje contiene verbo de consumo pasado ("comí un snack", "cené pollo")
    # NO es recomendación aunque contenga "un snack" u otras palabras de _PEDIR_REC_NUT.
    _PASADO_BLOQUEO_REC = ("comi", "desayune", "almorce", "cene", "bebi", "me comi", "me tome", "me bebi", "ya comi", "ya desayune")
    _tiene_pasado_bloqueo = any(
        _mn.startswith(v) or re.search(rf"\b{re.escape(v.strip())}\b", _mn)
        for v in _PASADO_BLOQUEO_REC
    )
    # Consulta de CÁLCULO nutricional ("¿cuánta proteína necesito?") es distinta
    # de pedir un plato ("qué puedo comer") — antes ambas caían en
    # RECOMENDAR_NUTRICION y el usuario recibía un plato en vez de un número.
    # Va a OTRO, donde respuesta_chat_llm puede responder con un cálculo real.
    _RX_CALCULO_NUTRICIONAL = re.compile(
        r"\bcuant[oa]s?\b.{0,15}\b(proteina|proteína|calorias|calorías|"
        r"carbohidratos|grasas|macros)\b"
    )
    if _RX_CALCULO_NUTRICIONAL.search(_mn):
        return OTRO
    if any(p in _mn for p in _PEDIR_REC_NUT) and not any(e in _mn for e in _EJ_KW) and not _tiene_pasado_bloqueo:
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
    # Excepción: "qué puedo hacer (hoy/para X)" es una pregunta ABIERTA pidiendo
    # una recomendación (ejercicio o comida) — no es una pregunta de permiso
    # como "¿puedo correr?" o "¿puedo comer X?". Sin esta excepción, cualquier
    # petición de recomendación que use "qué puedo hacer" cae aquí y se pierde
    # el formato/contexto específico de RECOMENDAR_EJERCICIO o RECOMENDAR_NUTRICION.
    _es_pregunta_abierta_que_hacer = bool(re.search(r"\bque\s+puedo\s+hacer\b", _mn))
    if not _es_pregunta_abierta_que_hacer and any(
        _mn.startswith(p) or f" {p}" in _mn for p in _MODALES
    ):
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
    # Guard: si el mensaje tiene intención de COMIDA explícita, el ejercicio es mención
    # incidental ("ya entrené en la mañana, necesito almorzar") — no registrar ejercicio.
    _COMIDA_PRIORITARIA_NUT = (
        "almorzar", "desayunar", "cenar", "merendar",
        "que como", "dame opciones", "dame ideas",
        "que recomiendas", "que me recomiendas",
        "necesito comer", "quiero comer",
        "para el almuerzo", "para la cena", "para el desayuno",
    )
    _comida_es_prioritaria = any(p in _mn for p in _COMIDA_PRIORITARIA_NUT)
    if _tiene_ej and "?" not in _m and not _comida_es_prioritaria:
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
        modo_ia = await ia.clasificar_modo_asistente(mensaje, historial=historial)
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

