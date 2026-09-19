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

_VERBOS_IMPERATIVOS_REGISTRO: tuple[str, ...] = (
    "regístrame",
    "registrame",
    "registra que",
    "registra el",
    "registra la",
    "anota que",
    "anota el",
    "anota la",
    "anótame",
    "guarda que",
    "guarda el",
    "guarda la",
    "guárdame",
    "ponme",
    "ponme en",
    "agrégame",
    "agregame",
    "apunta el",
    "apunta la",
    "apunta que",
)

RX_CORREGIR_REGISTRO = re.compile(
    r"\b(agr[eé]ga(?:lo|la|le)?|agregalo|agregale|a[ñn]ade(?:lo|la)?|"
    r"s[uú]male|incl[uú]yelo|olvid[eé]|me\s+olvid[eé]|"
    r"falt[oó]\s+(?:poner|agregar|incluir|sumar))\b"
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
        "hamburguesa",
        "lenteja",
        "piqueo",
        "manzana",
        "platano",
        "plátano",
        "leche",
        "avena",
        "huevo",
        "pan",
        "fruta",
        "carne",
        "pescado",
        "ceviche",
        "queso",
        "yogur",
        "metí",
        "meti",
        "zampé",
        "zampe",
        "chupé",
        "chupe",
        "probé",
        "probe",
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

    _VERBOS_LOG = (
        "desayuné",
        "desayune",
        "almorcé",
        "almorce",
        "cené",
        "cene",
        "comí",
        "comi ",
        "tomé ",
        "tome ",
        "bebí ",
        "bebi ",
        "he desayunado",
        "he almorzado",
        "he cenado",
        "he comido",
        "he tomado",
        "acabo de comer",
        "acabo de tomar",
        "acabo de beber",
        "me comi",
        "me tomé",
        "me bebí",
        "me comí",
        "termine de comer",
        "terminé de comer",
        "ya comi",
        "ya comí",
        "ya almorcé",
        "ya desayuné",
        "me hice un",
        "me hice una",
        "me metí",
        "me meti",
        "metí",
        "meti",
        "me zampé",
        "me zampe",
        "zampé",
        "zampe",
        "me chupé",
        "me chupe",
        "chupé",
        "chupe",
        "piqueo",
        "pique",
        "piqué",
        "piqueando",
        "probé",
        "probe",
        "me probé",
        "me probe",
    )
    _verbos_log_count = sum(1 for v in _VERBOS_LOG if m.startswith(v) or re.search(rf"\b{re.escape(v.strip())}\b", m))
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
            "almorzar",
            "desayunar",
            "merendar",
            "para el almuerzo",
            "para la cena",
            "para el desayuno",
            "para la merienda",
            "qué recomiendas",
            "que recomiendas",
            "dame ideas",
            "dame opciones",
            "qué me recomiendas",
            "que me recomiendas",
            "tengo hambre",
            "tengo antojo",
            "qué puedo comer",
            "que puedo comer",
            "me recomiendas comer",
            "me da hambre",
            "quiero comer algo",
            "qué me como",
            "que me como",
            "qué hay para comer",
            "que hay para comer",
            "busco algo para comer",
            "dame algo de comer",
        )
    )
    _EJERCICIO_REGISTRADO = (
        "hice cardio",
        "hice pesas",
        "hice gym",
        "hice ejercicio",
        "hice sentadillas",
        "hice flexiones",
        "hice abdominales",
        "hice press",
        "hice curl",
        "hice remo",
        "hice jalon",
        "hice jalón",
        "hice dominadas",
        "hice fondos",
        "hice burpees",
        "hice plancha",
        "hice peso muerto",
        "hice sentadilla",
        "hice extensiones",
        "tiré press",
        "tire press",
        "tiré sentadillas",
        "fui al gym",
        "fui al gimnasio",
        "entrené",
        "entrenei",
        "entrenei ",
        "realicé",
        "realize",
        "corrí",
        "corri ",
        "caminé",
        "camine ",
        "nadé",
        "nade ",
        "pedalié",
        "pedalee",
        "terminé de entrenar",
        "termine de entrenar",
        "acabo de entrenar",
        "acabo de hacer ejercicio",
        "sali a correr",
        "salí a correr",
        "sali a trotar",
        "salí a trotar",
        "sali a caminar",
        "salí a caminar",
        "sali a entrenar",
        "salí a entrenar",
        "correr durante",
        "trotar durante",
        "caminar durante",
    )
    if any(x in m for x in _EJERCICIO_REGISTRADO) and "?" not in m and fe >= fc and not rec_nut:
        return REGISTRAR_EJERCICIO
    _PERMISO_EJ_SYNC = (
        "puedo realizar",
        "puedo trotar",
        "puedo correr",
        "puedo nadar",
        "puedo caminar",
        "puedo jugar",
        "puedo bailar",
        "puedo practicar",
        "puedo ir a ",
        "se puede trotar",
        "se puede correr",
        "se puede nadar",
        "es bueno trotar",
        "es bueno correr",
        "es bueno nadar",
        "es bueno caminar",
        "es malo trotar",
        "es malo correr",
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

    _RECETA_O_TECNICA = (
        "como se hace",
        "como se prepara",
        "como hacer",
        "receta de",
        "como cocinar",
        "ingredientes de",
        "preparacion de",
        "tecnica de",
        "como realizar",
        "como ejecutar",
        "pasos para",
        "forma correcta",
        "preparar ",
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

    if es_saludo:
        _NO_SALUDO_PURO = (
            "comer",
            "cenar",
            "almorzar",
            "desayunar",
            "merendar",
            "comida",
            "cena",
            "almuerzo",
            "desayuno",
            "plato",
            "receta",
            "ejercicio",
            "entren",
            "rutina",
            "pecho",
            "espalda",
            "pierna",
            "hombro",
            "cardio",
            "gym",
            "gimnasio",
            "dime",
            "puedo",
            "quiero",
            "sugier",
            "recomiend",
            "dame",
            "verdura",
            "hacer",
            "correr",
            "corrí",
            "corri",
            "trotar",
            "troté",
            "trote",
            "caminar",
            "caminé",
            "camine",
            "nadar",
            "nadé",
            "nade",
            "bicicleta",
            "salí",
            "sali",
            "pesa",
            "pesas",
            "pesos",
            "comi",
            "comí",
            "tomar",
            "tome",
            "tomé",
        )
        if any(k in _m for k in _NO_SALUDO_PURO):
            es_saludo = False

    if es_saludo:
        return OTRO

    if any(_m.startswith(v) for v in _VERBOS_IMPERATIVOS_REGISTRO):
        return REGISTRAR_NUTRICION

    import unicodedata as _ud

    _mn = "".join(c for c in _ud.normalize("NFD", _m) if _ud.category(c) != "Mn")
    _RX_CALCULO_NUTRICIONAL = re.compile(
        r"\bcuant[oa]s?\b.{0,15}\b(proteina|proteína|calorias|calorías|"
        r"carbohidratos|grasas|macros)\b"
    )
    if _RX_CALCULO_NUTRICIONAL.search(_mn):
        return OTRO

    _MODALES = (
        "puedo ",
        "se puede ",
        "podria ",
        "podria ",
        "es bueno ",
        "es malo ",
        "es buena ",
        "es mala ",
        "seria bueno ",
        "es recomendable ",
        "es posible ",
        "conviene ",
    )
    _es_pregunta_abierta_que_hacer = bool(
        re.search(r"\bque\s+(?:\w+\s+){0,2}puedo\s+(hacer|comer|almorzar|cenar|desayunar|merendar)\b", _mn)
    )
    if not _es_pregunta_abierta_que_hacer and any(_mn.startswith(p) or f" {p}" in _mn for p in _MODALES):
        return OTRO

    _RECETA_O_TECNICA = (
        "como se hace",
        "como se prepara",
        "como hacer",
        "receta de",
        "como cocinar",
        "ingredientes de",
        "preparacion de",
        "tecnica de",
        "como realizar",
        "como ejecutar",
        "pasos para",
        "forma correcta",
        "preparar ",
    )
    if any(k in _mn for k in _RECETA_O_TECNICA):
        return OTRO

    _CONSUMO_CLARO_NORM = (
        "comi",
        "desayune",
        "almorce",
        "cene",
        "bebi",
        "me jale",
        "me comi",
        "me tome",
        "acabo de comer",
        "acabo de tomar",
        "acabo de beber",
        "termine de comer",
        "he desayunado",
        "he almorzado",
        "he cenado",
        "he comido",
        "he bebido",
        "me meti",
        "meti",
        "me zampe",
        "zampe",
        "chupe",
        "me chupe",
        "piqueo",
        "probe",
        "tome",
    )
    _tiene_consumo = any(
        _mn.startswith(v) or re.search(rf"\b{re.escape(v.strip())}\b", _mn) for v in _CONSUMO_CLARO_NORM
    )
    if _tiene_consumo and "?" not in _m:
        return REGISTRAR_NUTRICION


    try:
        modo_ia = await ia.clasificar_modo_asistente(mensaje, historial=historial)
        if isinstance(modo_ia, str) and modo_ia in MODOS_ASISTENTE:
            return modo_ia
    except Exception:
        pass
    return detectar_modo_funcion(mensaje, False)
