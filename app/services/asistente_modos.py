"""
Cuatro funciones del asistente cliente (una por conversación lógica):
  recomendar_nutricion | registrar_nutricion | recomendar_ejercicio | registrar_ejercicio
+ otro (saludo, progreso, dudas generales) sin forzar tarjetas estructuradas.
"""
from __future__ import annotations

from typing import Any, Optional

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
            "qué como",
            "que como",
            "qué comer",
            "que comer",
            "ideas para comer",
            "sugerencias de comida",
            "opciones de almuerzo",
            "opciones de desayuno",
            "opciones de cena",
            "recetas",
            "receta ",
            "platos para",
        )
    )
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

    inf = inferir_forzar_por_mensaje_usuario(mensaje)
    if inf == "comida":
        return RECOMENDAR_NUTRICION
    if inf == "ejercicio":
        return RECOMENDAR_EJERCICIO
    return OTRO


async def resolver_modo_funcion(ia: Any, mensaje: str, es_saludo: bool) -> str:
    """
    Prioriza clasificación semántica vía LLM (Groq); si falla o no hay API, usa reglas por palabras clave.
    Los verbos imperativos de registro se resuelven antes del LLM para evitar degradación a RECIPE.
    """
    if es_saludo or len((mensaje or "").strip()) < 2:
        return OTRO
    _m = (mensaje or "").lower().strip()
    # Pre-check 1: verbos imperativos → nunca pasar al LLM (evita que clasifique como RECIPE)
    if any(_m.startswith(v) for v in _VERBOS_IMPERATIVOS_REGISTRO):
        return REGISTRAR_NUTRICION
    # Pre-check 2: verbos de consumo pasado ("tomé", "bebí", "comí"…) + alimento o bebida →
    # son registros afirmativos, no consultas. Resolverlos antes del LLM evita que los
    # clasifique como INFO cuando el usuario simplemente informa lo que consumió.
    _VERBOS_CONSUMO_PASADO = (
        "comí", "comi ", "tomé ", "tome ", "bebí ", "bebi ",
        "desayuné", "desayune", "almorcé", "almorce", "cené", "cene",
        "me tomé", "me tome", "me bebí", "me bebi",
    )
    _ALIMENTOS_RAPIDOS = frozenset({
        "gaseosa", "jugo", "chicha", "limonada", "refresco", "cerveza",
        "agua", "bebida", "pollo", "arroz", "sopa", "ensalada", "pan",
        "ceviche", "cebiche", "lomo", "causa", "papa", "avena", "fruta",
    })
    _es_consumo = any(_m.startswith(v) or _m == v.strip() for v in _VERBOS_CONSUMO_PASADO)
    _tiene_alimento = any(ak in _m for ak in _ALIMENTOS_RAPIDOS)
    if _es_consumo and _tiene_alimento and "?" not in _m:
        return REGISTRAR_NUTRICION
    # También: si el mensaje empieza con verbo de consumo pero el alimento está implícito
    # (ej. "Tomé un refresco" — "refresco" puede no estar en _ALIMENTOS_RAPIDOS)
    if any(_m.startswith(v) for v in _VERBOS_CONSUMO_PASADO) and "?" not in _m:
        return REGISTRAR_NUTRICION
    try:
        modo_ia = await ia.clasificar_modo_asistente(mensaje)
        if isinstance(modo_ia, str) and modo_ia in MODOS_ASISTENTE:
            return modo_ia
    except Exception:
        pass
    return detectar_modo_funcion(mensaje, False)


def bloque_prompt_modo_funcion(modo: str) -> str:
    """Texto inyectado al LLM para acotar la respuesta a una sola función."""
    modo = (modo or OTRO).strip().lower()
    if modo not in MODOS_ASISTENTE or modo == OTRO:
        return (
            "\n\n══ FUNCIÓN DEL ASISTENTE (MODO GENERAL) ══\n"
            "No es una petición clara de las 4 funciones principales. "
            "Responde con [CALOFIT_INTENT:INFO] o [CALOFIT_INTENT:PROGRESS] según corresponda, "
            "solo texto conversacional. No generes bloques [CALOFIT_HEADER] de recetas ni rutinas "
            "salvo que el usuario pida explícitamente comida o entrenamiento en este mismo mensaje.\n"
            "• Si no entiendes o el mensaje es muy ambiguo: dilo en una frase natural "
            "(ej. \"No estoy seguro de entenderte bien\"), reformula en tus palabras lo que crees que quiso decir "
            "y ofrece como máximo 2 interpretaciones concretas en una sola línea (¿querías A o B?). "
            "No inventes recetas ni rutinas completas si no está claro el pedido."
        )

    instrucciones = {
        RECOMENDAR_NUTRICION: (
            "\n\n══ FUNCIÓN ACTUAL (OBLIGATORIA, ÚNICA) ══\n"
            "FUNCION: RECOMENDAR_NUTRICIÓN.\n"
            "• DISEÑO ÚNICO: todas las opciones (2–3) con la MISMA plantilla CALOFIT, en este orden por opción:\n"
            "  [CALOFIT_HEADER]…[/CALOFIT_HEADER] → [CALOFIT_LIST]…[/CALOFIT_LIST] → "
            "[CALOFIT_ACTION]…[/CALOFIT_ACTION] → [CALOFIT_STATS]…[/CALOFIT_STATS].\n"
            "• PROHIBIDO mezclar formatos: ninguna receta en markdown, emojis sueltos ni párrafo tipo "
            "«Opción 1: …» fuera de tags si el resto va en CALOFIT. El usuario solo verá tarjetas coherentes si TODO va en tags.\n"
            "• Debes usar [CALOFIT_INTENT:RECIPE] al inicio (y [CALOFIT_QUESTION_TYPE:ABIERTA] si aplica).\n"
            "• EJEMPLO CORRECTO (sigue EXACTAMENTE este patrón, sin variaciones):\n"
            "[CALOFIT_INTENT:RECIPE]\n"
            "Aquí tienes opciones para la tarde.\n"
            "[CALOFIT_HEADER]Tortilla de Huevo con Pan Tostado[/CALOFIT_HEADER]\n"
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
            "• [CALOFIT_LIST] obligatorio: 4–8 líneas, una por ingrediente, con gramos y SOLO kcal entre paréntesis. "
            "  Ejemplo: \"150g pechuga de pollo (240 kcal)\". PROHIBIDO meter macros por ingrediente (ej. \"37g proteína\"). "
            "[CALOFIT_STATS] obligatorio en una sola línea: P: Xg | C: Yg | G: Zg | Cal: Wkcal. "
            "Verifica coherencia: 4×P + 4×C + 9×G ≈ W (±8%). Si hay pollo, pescado, huevo, lácteos, lentejas o "
            "legumbres en LIST o en el nombre del plato, P no puede ser 0. W debe ser la suma de kcal de LIST (±5%).\n"
            "• [CALOFIT_ACTION] obligatorio y útil: 3–6 pasos numerados (1., 2., 3. ...), "
            "cada paso debe tener AL MENOS 8–12 palabras y debe incluir detalle mínimo "
            "(tiempo aproximado, orden lógico, calor medio/alto, hervir/sancochar/saltear, etc.). "
            "PROHIBIDO: 1 solo paso genérico tipo \"Prepara y sirve\".\n"
            "• PERÚ + HORA: respeta el bloque «MOMENTO DEL DÍA (hora Perú)» del contexto: nombres de plato y porciones acordes "
            "a desayuno/almuerzo/cena/snack. Cocina peruana reconocible; ingredientes y técnica deben tener sentido junto "
            "(no mezclas imposibles solo por sumar macros).\n"
            "• COHERENCIA NOCTURNA: si MOMENTO DEL DÍA indica NOCTURNO, las opciones deben ser realmente ligeras "
            "(snacks o platos suaves). PROHIBIDO proponer pollería (\"pollo a la brasa\") o platos pesados. "
            "PROHIBIDO inventar combinaciones raras tipo \"lentejas con leche en polvo\" como plato principal.\n"
            "• PROHIBIDO ingredientes poco comunes en Perú (ej. estragón/tarragón). Usa hierbas típicas: culantro, "
            "perejil, orégano, ají amarillo/panca, kion, ajo, cebolla.\n"
            "• VARIEDAD: platos distintos entre mensajes; no repitas las mismas recetas salvo que lo pidan.\n"
            "• TEXTO fuera de tags: máximo 2–3 frases. NO repitas el nombre del usuario dos veces. "
            "Sin ingredientes ni macros ahí. "
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
