"""
AsistenteService — Orquestador del asistente del cliente.

Todos los bloques de lógica de dominio viven en módulos especializados:
  asistente_plan.py              → obtener_plan_hoy
  asistente_registro_comida.py   → flujo 5 capas NLP + manual + registrar_desde_cache
  asistente_registro_ejercicio.py→ MET + workout_logs
  asistente_recomendaciones.py   → KNN + 14 features RF
  asistente_prompt.py            → prompt builder, rescue_nlp_log, helpers respuesta
"""
from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Guardia anti-no-alimento ──────────────────────────────────────────────────
# Capa 1: palabras que NUNCA son comida — bloqueo inmediato sin importar el verbo.
#   Ejemplos: "pan con caca", "jugo de orina", "caldo de veneno", "pan con puchaina"
_BLOQUEO_ABSOLUTO: frozenset[str] = frozenset({
    # Desechos corporales
    "caca", "orina", "excremento", "heces", "vomito", "feces", "moco",
    # Sustancias peligrosas
    "veneno", "toxico", "explosivo", "bomba", "gasolina", "cloro",
    # Jerga / insultos peruanos y latinos (no son alimentos)
    "puchaina", "pucha", "carajo", "mierda", "huevada", "wevada",
    "cojuda", "cojudo", "idiota", "estupido", "imbecil", "pendejo",
    "chingada", "verga", "puta", "cabron", "basura", "bosta",
    # Palabras sin sentido en contexto de comida
    "broma", "chiste", "jajaja", "jaja", "lol", "xd",
})

# Capa 2: palabras no-comestibles que se bloquean cuando van acompañadas
# de un verbo explícito de ingesta ("quiero comer papel", "me como una piedra").
_RE_INTENTO_COMER = re.compile(
    r"\b(quiero\s+comer|quiero\s+tomar|voy\s+a\s+comer|voy\s+a\s+tomar|"
    r"puedo\s+comer|puedo\s+tomar|me\s+voy\s+a\s+comer|me\s+como\s+un|"
    r"me\s+como\s+una|me\s+como|registra\s+que\s+com[ií])\b",
    re.IGNORECASE,
)

def _deaccent(s: str) -> str:
    """Quita tildes para comparación con NO_ALIMENTOS (sin tildes)."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


# ── Guardia off-topic ─────────────────────────────────────────────────────────
# Señales de que el mensaje SÍ es sobre nutrición/ejercicio/salud.
# Si el mensaje contiene alguna de estas palabras → pasa al pipeline normal.
_SEÑALES_NUTRICION: frozenset[str] = frozenset({
    # Acciones de comida
    "comer", "comi", "como", "tomar", "tome", "beber", "bebi",
    "desayunar", "almorzar", "cenar", "merendar",
    # Comidas y grupos
    "comida", "alimento", "plato", "receta", "ingrediente", "dieta",
    "desayuno", "almuerzo", "cena", "merienda", "snack", "postre",
    # Macros y nutrición
    "caloria", "kcal", "proteina", "carbohidrato", "grasa", "fibra",
    "vitamina", "mineral", "nutricion", "macro",
    # Alimentos peruanos comunes
    "arroz", "pollo", "ceviche", "lomo", "causa", "sopa", "papa",
    "quinua", "verdura", "fruta", "leche", "queso", "huevo", "pan",
    "pescado", "carne", "ensalada", "yogur", "avena", "menestra",
    # Ejercicio y gym
    "ejercicio", "entren", "gym", "gimnasio", "rutina", "cardio",
    "pesas", "correr", "caminar", "nadar", "trotar", "sentadilla",
    "press", "musculo", "fuerza", "cardio", "series", "reps",
    # Salud y metas
    "peso", "bajar", "subir", "adelgazar", "engordar", "masa",
    "salud", "plan", "meta", "progreso", "balance", "calorias",
    "diabetes", "hipertension", "vegano", "vegetariano", "alergia",
    # Verbos relacionados al asistente
    "registra", "anota", "guarda", "recomienda", "sugiere",
})

# Patrones que delatan preguntas de conocimiento general (off-topic)
_RE_OFFTOPIC = re.compile(
    r"\b(capital\s+de|presidente\s+de|quien\s+(es|fue|invento|descubrio|gano)|"
    r"cuando\s+(naci|fue\s+fundad|ocurri|empezo|termino)|historia\s+de\s+\w+\s+(pais|ciudad|guerra|mundo)|"
    r"cuanto\s+(es|son)\s+\d+\s*([\+\-\*\/x]|\s*por\s*|\s*entre\s*|\s*mas\s*|\s*menos\s*)\s*\d+|"
    r"en\s+que\s+(ano|pais|ciudad|continente)\s+(queda|esta|nacio|se\s+ubica)|"
    r"cual\s+es\s+(la\s+)?(capital|moneda|bandera|idioma|poblacion)\s+de|"
    r"(pelicula|cancion|serie|actor|actriz|director|libro|autor|novela)\s+(de|del|que)|"
    r"como\s+se\s+llama\s+el\s+(pais|presidente|rey|lider)|"
    r"dime\s+(algo\s+sobre|un\s+chiste|un\s+poema|una\s+historia))\b",
    re.IGNORECASE,
)

_RESPUESTA_OFFTOPIC = (
    "Solo puedo ayudarte con nutrición, alimentación y ejercicio. "
    "¿Tienes alguna consulta sobre tu dieta o entrenamiento?"
)

def _es_offtopic(msg_norm: str) -> bool:
    """Devuelve True si el mensaje es claramente off-topic (sin señales nutricionales)."""
    tokens = set(msg_norm.split())
    if tokens & _SEÑALES_NUTRICION:
        return False  # tiene señal de nutrición → on-topic
    return bool(_RE_OFFTOPIC.search(msg_norm))


# Verbos de ingesta en pasado — redirigen al handler directo (sin LLM)
# cuando el modo ya fue clasificado como REGISTRAR_NUTRICION.
# Ejemplos: "Temprano comí pan con pollo", "Almorcé lomo saltado con su gaseosa"
_RE_PASADO_COMER = re.compile(
    r"\b(com[ií]|desayun[eé]|almor[cz][eaé]|cen[eé]|tom[eé]|beb[ií]|inger[ií]"
    r"|acabo\s+de\s+(?:comer|cenar|desayunar|almorzar|tomar|beber)"
    r"|me\s+acab[oó]\s+de\s+(?:comer|cenar|desayunar|almorzar|tomar|beber)"
    r"|me\s+com[ií]|me\s+tom[eé]|me\s+beb[ií]"
    r"|termin[eé]\s+de\s+(?:comer|cenar|almorzar|desayunar)"
    r"|ya\s+com[ií]|reci[eé]n\s+com[ií]"
    r"|me\s+jal[eé]|jal[eé]\s+un|jal[eé]\s+dos|jal[eé]\s+tres)\b",
    re.IGNORECASE,
)

# ── Limpieza de historial para el LLM ────────────────────────────────────────
# Los mensajes del asistente contienen emojis, desgloses y advertencias
# formateados (✅ 📊 • | P:7.8g | C:77.6g) que Llama-3 lee literalmente
# y repite, produciendo respuestas incoherentes. Solo se conserva el hecho
# principal ("Registré: X — 650 kcal.").
_RE_EMOJI_HIST = re.compile(
    r"[\U0001F300-\U0001FFFF]"  # emoji amplio (📊 🥗 ✅ ❌ etc.)
    r"|[☀-➿]"          # símbolos misc (⚠️ → ✅)
    r"|[⬀-⯿]"          # flechas extendidas
    r"|[•]"                 # bullet •
    r"|[—–]"           # em dash — / en dash –
    r"|[️‍]",          # variation selector / ZWJ
    re.UNICODE,
)
_RE_MACROS_INLINE = re.compile(
    r"\s*\|\s*[PCGpcg][a-zA-Z]*\s*:\s*[\d.,]+\s*g",
    re.IGNORECASE,
)
# Prefijos de sección que marcan el inicio de bloques de desglose/advertencia
_CORTES_HISTORIAL = (
    "\n\n📊", "\n\n⚠️", "\n\n•", "\n\n🥗",
    "\n\nTotal:", "\n\n[CALOFIT",
    "\n\nDesglose", "\n\nDetalle",
    "\n\nRegistrado igualmente",
    "\n\nAlimentos no compatibles",
)


def _resumir_para_historial(content: str, role: str) -> str:
    """
    Extrae solo el hecho principal de los mensajes del asistente antes de
    pasarlos a Llama-3 como historial de contexto.

    - Mensajes de usuario  → se pasan intactos (texto natural, máx 300 chars).
    - Mensajes del asistente → se conserva solo la primera línea útil;
      se eliminan desgloses nutricionales, advertencias, emojis, bullets y
      los patrones '| P:Xg | C:Xg | G:Xg' que confunden al modelo.
    """
    if not content:
        return ""
    texto = str(content).strip()

    if role != "assistant":
        return texto[:300]

    # 1. Cortar en el primer bloque de desglose o advertencia
    for patron in _CORTES_HISTORIAL:
        idx = texto.find(patron)
        if idx > 0:
            texto = texto[:idx]

    # 2. Quitar emojis, bullets y guiones especiales
    texto = _RE_EMOJI_HIST.sub("", texto)

    # 3. Quitar macros inline "| P:7.8g | C:77.6g | G:0.7g"
    texto = _RE_MACROS_INLINE.sub("", texto)

    # 4. Pipes y flechas restantes → coma
    texto = re.sub(r"\s*[|→]\s*", ", ", texto)

    # 5. Normalizar espacios
    texto = re.sub(r"\s{2,}", " ", texto).strip()
    texto = re.sub(r",\s*$", ".", texto)

    # 6. Máximo 400 chars — preservar preguntas de seguimiento completas
    # (200 era demasiado corto: cortaba "¿Cómo te quedaste...?" y el LLM la regeneraba)
    if len(texto) > 400:
        texto = texto[:400].rsplit(" ", 1)[0] + "."

    return texto

from sqlalchemy.orm import Session

from app.core.cache import get_consulta_cached
from app.core.utils import get_peru_date
from app.models.client import Client
from app.models.historial import AlertaSalud, ProgresoCalorias
from app.services.asistente.asistente_ejercicio import (
    es_payload_ejercicio,
    procesar_secciones_ejercicio,
    registrar_ejercicio_desde_payload_tarjeta,
)
from app.services.asistente.asistente_modos import (
    OTRO, RECOMENDAR_EJERCICIO, REGISTRAR_EJERCICIO, REGISTRAR_NUTRICION,
    RX_CORREGIR_REGISTRO, _VERBOS_IMPERATIVOS_REGISTRO, resolver_modo_funcion,
)
from app.services.asistente.asistente_nutricion import (
    procesar_secciones_comida,
    registrar_comida_desde_payload_tarjeta,
)
from app.services.asistente.asistente_plan import obtener_plan_hoy
from app.services.asistente.asistente_prompt import (
    clasificar_intencion_respuesta,
    detectar_intencion_principal,
    limpiar_tags_calofit,
    rescue_nlp_log,
    respuesta_fallo_llm,
)
from app.services.asistente.asistente_registro_comida import (
    registrar_desde_cache,
    registro_comida_handler,
)
from app.services.asistente.asistente_registro_ejercicio import registro_ejercicio_handler
from app.services.asistente.asistente_respuesta_normalize import enriquecer_respuesta_estructurada
from app.services.ia_service import ia_engine
from app.services.response_parser import parsear_respuesta_para_frontend


# ── Helpers auto-rutina ────────────────────────────────────────────────────────

def _extraer_minutos(texto: str) -> int:
    """Extrae minutos de '45 min', '1 hora', etc. Devuelve 0 si no encuentra."""
    m = re.search(r"(\d+)\s*(h|hr|hora|horas)", texto)
    if m:
        return int(m.group(1)) * 60
    m = re.search(r"(\d+)\s*(min|mins|minutos)", texto)
    return int(m.group(1)) if m else 0


def _rutina_a_texto(zonas: list, rutina: dict) -> str:
    """
    Genera UNA card por ejercicio con el protocolo CALOFIT.
    El parser divide en secciones al encontrar cada [CALOFIT_HEADER].
    """
    nombre_rutina = rutina.get("nombre_rutina", "Rutina Personalizada")
    intensidad    = rutina.get("intensidad", "Media")
    series        = rutina.get("series", 3)
    reps          = rutina.get("reps", 12)
    descanso      = rutina.get("descanso_seg", 90)
    ejercs        = rutina.get("ejercicios", [])
    tiempo        = rutina.get("tiempo_estimado_min", 60)
    adv           = rutina.get("advertencias", [])

    adv_txt = f"\n⚠️ {' | '.join(adv)}" if adv else ""

    # Zonas con "y" antes de la última para sonar más natural
    if len(zonas) == 1:
        zonas_txt = zonas[0]
    elif len(zonas) == 2:
        zonas_txt = f"{zonas[0]} y {zonas[1]}"
    else:
        zonas_txt = ", ".join(zonas[:-1]) + f" y {zonas[-1]}"

    intro = (
        f"[CALOFIT_INTENT:POWER]\n"
        f"Tu entrenamiento de hoy está listo: **{nombre_rutina}**, {tiempo} min "
        f"de {intensidad.lower()} intensidad enfocados en {zonas_txt}.{adv_txt}\n"
    )

    dur_x_ejerc = round(tiempo / max(len(ejercs), 1))

    bloques = []
    for e in ejercs:
        musculo = e.get("musculo_principal") or e.get("grupo_padre") or ""
        lista_items = [
            f"- {series} series × {reps} repeticiones",
            f"- Descanso: {descanso} segundos",
        ]
        if musculo:
            lista_items.append(f"- Músculo: {musculo}")

        # Construir bloque de técnica en [CALOFIT_ACTION] para que el parser lo asigne
        # a seccion["tecnica"] y Flutter muestre los pasos numerados.
        instrucciones = (e.get("instrucciones") or "").strip()
        accion_block = ""
        if instrucciones:
            if re.match(r"^\d+\.", instrucciones):
                # Pre-numerado ("1. Paso uno. 2. Paso dos.") → separar cada número en su propia línea
                partes = [p.strip() for p in re.split(r"\s+(?=\d+\.\s)", instrucciones) if p.strip()]
            else:
                # Texto plano → dividir en oraciones y numerar
                oraciones = [s.strip() for s in re.split(r"(?<=[.!?])\s+", instrucciones) if s.strip()]
                partes = [f"{i+1}. {s}" for i, s in enumerate(oraciones[:5])]
            pasos_txt = "\n".join(partes[:5])
            accion_block = f"[CALOFIT_ACTION]\n{pasos_txt}\n[/CALOFIT_ACTION]\n"

        bloque = (
            f"[CALOFIT_HEADER]{e['nombre']}[/CALOFIT_HEADER]\n"
            f"[CALOFIT_LIST]\n"
            + "\n".join(lista_items)
            + f"\n[/CALOFIT_LIST]\n"
            + accion_block
            + f"[CALOFIT_STATS]Intensidad: {intensidad} | {dur_x_ejerc} min[/CALOFIT_STATS]"
        )
        bloques.append(bloque)

    return intro + "\n".join(bloques)


# ──────────────────────────────────────────────────────────────────────────────

def _build_response(perfil, intencion, tipo_pregunta, meta, consumido, quemado, plan_hoy, mensaje_txt):
    """Helper para construir respuesta estándar sin registro."""
    restante = max(0.0, meta - consumido + quemado)  # igual que la UI: suma quemadas
    return {
        "asistente": "CaloFit IA", "usuario": perfil.first_name,
        "intencion": intencion, "tipo_pregunta": tipo_pregunta, "alerta_salud": False,
        "data_cientifica": {
            "progreso_diario": {
                "consumido": round(consumido, 1), "meta": round(meta, 1),
                "restante": round(restante, 1), "quemado": round(quemado, 1),
            },
            "macros": {
                "proteinas_meta": plan_hoy.get("proteinas_g", 0),
                "carbohidratos_meta": plan_hoy.get("carbohidratos_g", 0),
                "grasas_meta": plan_hoy.get("grasas_g", 0),
            },
        },
        "respuesta_ia": mensaje_txt,
        "respuesta_estructurada": {
            "intent": intencion, "texto_conversacional": mensaje_txt, "secciones": [],
        },
    }


class AsistenteService:
    """Punto de entrada único del asistente del cliente."""

    def __init__(self):
        self.ia = ia_engine

    # ── 1. Chat principal ─────────────────────────────────────────────────────

    async def consultar(
        self,
        mensaje: str,
        db: Session,
        current_user,
        historial: list = None,
        contexto_manual: str = None,  # bloque extra del copiloto staff
        override_ia: str = None,
        consulta_id: str = None,  # confirmar card directamente desde el chat
    ):
        perfil = db.query(Client).filter(Client.email.ilike(current_user.email)).first()
        if not perfil:
            raise ValueError("Perfil de cliente no encontrado")
        edad = (datetime.now().year - perfil.birth_date.year) if perfil.birth_date else 25

        # Confirmar card sin abrir el endpoint dedicado
        if consulta_id and consulta_id.strip():
            payload = get_consulta_cached(consulta_id.strip())
            if payload:
                return registrar_desde_cache(payload, perfil, db)

        _ctx_extra = f"\n\nCONTEXTO ADICIONAL:\n{contexto_manual}" if contexto_manual else ""

        # Plan del día
        _, plan_hoy_data, _ = obtener_plan_hoy(perfil, edad, db)

        # Progreso y adherencia
        hoy           = get_peru_date()
        prog          = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == perfil.id, ProgresoCalorias.fecha == hoy
        ).first()
        consumo_real  = prog.calorias_consumidas if prog else 0
        
        # Calorías quemadas: fuente autoritativa = workout_logs (cubre todos los paths de registro)
        from sqlalchemy import text as _sql_wl
        _dialect = getattr(getattr(db, "bind", None), "dialect", None)
        _dname = getattr(_dialect, "name", "") or ""
        if _dname == "postgresql":
            quemadas_real = float(db.execute(_sql_wl(
                "SELECT COALESCE(SUM(calorias_quemadas), 0) FROM workout_logs "
                "WHERE client_id = :cid "
                "  AND (created_at AT TIME ZONE 'UTC' AT TIME ZONE 'America/Lima')::date = :hoy"
            ), {"cid": perfil.id, "hoy": hoy}).scalar() or 0)
        else:
            quemadas_real = float(db.execute(_sql_wl(
                "SELECT COALESCE(SUM(calorias_quemadas), 0) FROM workout_logs "
                "WHERE client_id = :cid AND date(created_at) = :hoy"
            ), {"cid": perfil.id, "hoy": hoy}).scalar() or 0)
        calorias_meta = plan_hoy_data["calorias_dia"]
        progreso_pct  = min(100, (consumo_real / calorias_meta) * 100) if calorias_meta > 0 else 0
        adherencia_pct = round(progreso_pct * 0.7 + 20, 1)

        alerta_fuzzy  = self.ia.generar_alerta_fuzzy(adherencia_pct, progreso_pct)
        mensaje_fuzzy = alerta_fuzzy.get("mensaje", "")
        msg_limpio    = mensaje.lower().strip()
        # Detecta saludos puros — inicio de conversación, no respuestas en medio del hilo.
        # Regla: es saludo SI contiene keyword de apertura Y no hay verbo de acción
        #        Y no es una respuesta de seguimiento ("bien gracias", "sí claro", "ok").
        _KW_SALUDO  = ("hola", "hey", "saludos", "buenas", "que tal", "qué tal",
                       "cómo estás", "como estas", "cómo te va", "como te va")
        _KW_ACCION  = ("comí", "comi", "hice", "fui al", "corrí", "corri",
                       "almorcé", "almorce", "desayuné", "desayune", "cené", "cene",
                       "registra", "anota", "tomé", "tome ", "bebí", "bebi",
                       "entrené", "entrenei",
                       "correr", "trotar", "caminar", "nadar", "salí", "sali", "ejercicio", "entrenar")
        # Patrones de respuesta conversacional — NO son saludos aunque contengan "gracias"
        _KW_RESPUESTA = (
            "bien gracias", "sí gracias", "si gracias", "ok gracias",
            "muchas gracias", "gracias igual", "gracias por", "de nada",
            "entendido", "ya entendí", "perfecto gracias", "claro gracias",
            "bueno gracias", "ok", "sí", "no gracias", "ya",
        )
        _es_respuesta = any(r in msg_limpio for r in _KW_RESPUESTA)
        es_saludo = (
            not _es_respuesta
            and any(s in msg_limpio for s in _KW_SALUDO)
            and not any(kw in msg_limpio for kw in _KW_ACCION)
        )
        if not es_saludo:
            asyncio.create_task(self._analizar_salud_background(mensaje, perfil, db))

        # ── Saludo puro ("hola", sin nada más) → saludo correcto según la hora
        # real de Perú, sin pasar por el LLM. Antes el LLM generaba saludos como
        # "estás a punto de empezar tu día" sin saber la hora real — a las 11pm
        # eso no tiene sentido. Solo se intercepta si NO queda contenido aparte
        # del saludo (si dice "Hola, ¿cuánta proteína necesito?" debe seguir el
        # flujo normal, no cortarse en un saludo).
        if es_saludo:
            _resto_saludo = msg_limpio
            for _s in _KW_SALUDO:
                _resto_saludo = _resto_saludo.replace(_s, "")
            _resto_saludo = re.sub(r'[^\wáéíóúñ]+', '', _resto_saludo)
            if len(_resto_saludo) <= 2:
                from app.core.utils import get_peru_now
                _hora_actual = get_peru_now().hour
                if 5 <= _hora_actual < 12:
                    _saludo_hora = "Buenos días"
                elif 12 <= _hora_actual < 19:
                    _saludo_hora = "Buenas tardes"
                else:
                    _saludo_hora = "Buenas noches"
                return _build_response(
                    perfil, "CHAT", "OTRO",
                    calorias_meta, consumo_real, quemadas_real, plan_hoy_data,
                    f"{_saludo_hora}, {perfil.first_name}. ¿En qué te ayudo hoy?"
                )

        # ── Guardia anti-no-alimento (2 capas) ───────────────────────────────
        _tokens_norm = set(_deaccent(msg_limpio).split())

        # Capa 1: bloqueo absoluto — palabras que nunca son comida (caca, veneno, etc.)
        _item_abs = next((t for t in _tokens_norm if t in _BLOQUEO_ABSOLUTO), None)

        # Capa 2: no-alimentos + verbo explícito de ingesta (papel, madera, hierro...)
        _item_verb = None
        if not _item_abs and _RE_INTENTO_COMER.search(msg_limpio):
            from app.services.nlp_food_extractor import NO_ALIMENTOS
            _item_verb = next((t for t in _tokens_norm if t in NO_ALIMENTOS), None)

        _item_no_comida = _item_abs or _item_verb
        if _item_no_comida:
            _resp = _build_response(
                perfil, "CHAT", "OTRO",
                calorias_meta, consumo_real, quemadas_real, plan_hoy_data,
                f"'{_item_no_comida.capitalize()}' no es un alimento. "
                f"Solo puedo ayudarte con comidas y bebidas reales. "
                f"¿Qué comida o ejercicio puedo registrar por ti?"
            )
            _resp["_blocked"] = True  # señal para no guardar en historial BD
            return _resp

        # ── Guardia off-topic (Python puro, sin llamar al LLM) ───────────────
        _msg_norm = _deaccent(msg_limpio)
        if not es_saludo and _es_offtopic(_msg_norm):
            _resp_ot = _build_response(
                perfil, "INFO", "OTRO",
                calorias_meta, consumo_real, quemadas_real, plan_hoy_data,
                _RESPUESTA_OFFTOPIC,
            )
            _resp_ot["_blocked"] = True  # no guardar en historial BD
            return _resp_ot

        # ── Pre-check de seguridad COMÚN, antes de decidir el modo ───────────────
        # Si se menciona una lesión sin especificar zona (rodilla/espalda/hombro/
        # codo), no hay info suficiente para recomendar nada seguro — sin importar
        # a qué modo iba a ir el mensaje (antes esto solo protegía dentro de
        # respuesta_chat_llm, así que RECOMENDAR_EJERCICIO podía saltárselo).
        from app.services.llm_registro import _lesion_mencionada_sin_tipo
        if _lesion_mencionada_sin_tipo(mensaje, historial):
            _resp_lesion = _build_response(
                perfil, "INFO", "OTRO",
                calorias_meta, consumo_real, quemadas_real, plan_hoy_data,
                "¿Qué lesión tienes exactamente? ¿Es en la rodilla, espalda, "
                "hombro, codo u otra zona? Así te doy un consejo seguro y específico.",
            )
            return _resp_lesion

        # Modo funcional + guard rails
        modo_funcion = await resolver_modo_funcion(self.ia, mensaje, es_saludo, historial=historial)

        # ══════════════════════════════════════════════════════════════════════════
        # NUEVA ARQUITECTURA: LLM estima macros directo, sin lookup de BD
        # ── REGISTRO COMIDA ────────────────────────────────────────────────────
        # ── REGISTRO COMIDA (LLM directo) ─────────────────────────────────────
        if modo_funcion == REGISTRAR_NUTRICION:
            # Verbo sin alimento → pedir qué comió
            _solo_verbo = bool(re.search(
                r"^acabo\s+de\s+(?:comer|cenar|desayunar|almorzar|tomar|beber)\s*$",
                msg_limpio, re.IGNORECASE
            )) or len(msg_limpio.split()) <= 1
            if _solo_verbo:
                return _build_response(
                    perfil, "INFO", "OTRO", calorias_meta, consumo_real, quemadas_real,
                    plan_hoy_data,
                    f"¡Qué bien! ¿Qué comiste exactamente, {perfil.first_name}? "
                    f"Dime el plato o los alimentos para anotarlo.",
                )
            from app.services.llm_registro import registrar_comida_llm
            # No pasar historial — las recomendaciones previas confunden los macros.
            # La consistencia viene de la tabla de referencia en el prompt.
            _com = await registrar_comida_llm(mensaje, perfil, plan_hoy_data, db, self.ia)
            # "Agrégalo en el registro" / "olvidé la palta" no nombra el alimento
            # en SU PROPIO mensaje (vive en el turno anterior, ej. "Te faltó la
            # palta") — pero "Agrega que comí un huevo frito" SÍ se basta solo.
            # No decidir por estructura del mensaje (regex) si hay que combinar
            # con el turno anterior — decidir según si la extracción del mensaje
            # actual SOLO ya encontró algo real. Combinar siempre (aunque el
            # mensaje ya tuviera su propio alimento) duplicaba lo ya registrado
            # del turno anterior (encontrado en pruebas reales: arroz/gelatina
            # se volvían a registrar y se contaban doble).
            if (
                not _com.get("success")
                and RX_CORREGIR_REGISTRO.search(msg_limpio)
                and historial
            ):
                _turnos_usuario_prev = [
                    h.get("content", "") for h in historial if h.get("role") == "user"
                ]
                if _turnos_usuario_prev:
                    _mensaje_combinado = f"{_turnos_usuario_prev[-1]} {mensaje}"
                    _com = await registrar_comida_llm(_mensaje_combinado, perfil, plan_hoy_data, db, self.ia)
            _bal = _com.get("balance_actualizado", {})
            return {
                "asistente":    "CaloFit IA",
                "usuario":      perfil.first_name,
                "intencion":    "SUCCESS" if _com.get("success") else "INFO",
                "tipo_pregunta": "LOG" if _com.get("success") else "OTRO",
                "alerta_salud": False,
                "data_cientifica": {
                    "progreso_diario": {
                        "consumido": _bal.get("consumido", round(consumo_real, 1)),
                        "meta":      _bal.get("meta", round(calorias_meta, 1)),
                        "restante":  _bal.get("restante", 0.0),
                        "quemado":   _bal.get("quemado", round(quemadas_real, 1)),
                    },
                    "macros": {
                        "proteinas_meta":     plan_hoy_data.get("proteinas_g", 0),
                        "carbohidratos_meta": plan_hoy_data.get("carbohidratos_g", 0),
                        "grasas_meta":        plan_hoy_data.get("grasas_g", 0),
                    },
                },
                "respuesta_ia": _com["mensaje"],
                "respuesta_estructurada": {
                    "intent": "SUCCESS" if _com.get("success") else "INFO",
                    "texto_conversacional": _com["mensaje"],
                    "secciones": [],
                },
                "tipo_detectado": _com.get("tipo_detectado", "nutricion"),
                "datos": _com.get("datos", {}),
                "balance_actualizado": _bal,
            }

        # ── REGISTRO EJERCICIO (LLM directo) ──────────────────────────────────
        if modo_funcion == REGISTRAR_EJERCICIO:
            from app.services.llm_registro import registrar_ejercicio_llm
            _ej = await registrar_ejercicio_llm(mensaje, perfil, db, self.ia)
            _bal_ej = _ej.get("balance_actualizado", {})
            _hoy_e = get_peru_date()
            _p_e = db.query(ProgresoCalorias).filter(
                ProgresoCalorias.client_id == perfil.id,
                ProgresoCalorias.fecha == _hoy_e,
            ).first()
            _c_e = float(_p_e.calorias_consumidas if _p_e else consumo_real)
            
            # Calorías quemadas de ejercicio registrado: fuente autoritativa = workout_logs
            from sqlalchemy import text as _sql_wl
            _dialect = getattr(getattr(db, "bind", None), "dialect", None)
            _dname = getattr(_dialect, "name", "") or ""
            if _dname == "postgresql":
                _q_e = float(db.execute(_sql_wl(
                    "SELECT COALESCE(SUM(calorias_quemadas), 0) FROM workout_logs "
                    "WHERE client_id = :cid "
                    "  AND (created_at AT TIME ZONE 'UTC' AT TIME ZONE 'America/Lima')::date = :hoy"
                ), {"cid": perfil.id, "hoy": _hoy_e}).scalar() or 0)
            else:
                _q_e = float(db.execute(_sql_wl(
                    "SELECT COALESCE(SUM(calorias_quemadas), 0) FROM workout_logs "
                    "WHERE client_id = :cid AND date(created_at) = :hoy"
                ), {"cid": perfil.id, "hoy": _hoy_e}).scalar() or 0)
            
            return {
                "asistente":    "CaloFit IA",
                "usuario":      perfil.first_name,
                "intencion":    "SUCCESS" if _ej.get("success") else "INFO",
                "tipo_pregunta": "LOG" if _ej.get("success") else "OTRO",
                "alerta_salud": False,
                "data_cientifica": {
                    "progreso_diario": {
                        "consumido": round(_c_e, 1),
                        "meta":      round(calorias_meta, 1),
                        "restante":  round(max(0.0, calorias_meta - _c_e + _q_e), 1),
                        "quemado":   round(_q_e, 1),
                    },
                    "macros": {
                        "proteinas_meta":     plan_hoy_data.get("proteinas_g", 0),
                        "carbohidratos_meta": plan_hoy_data.get("carbohidratos_g", 0),
                        "grasas_meta":        plan_hoy_data.get("grasas_g", 0),
                    },
                },
                "respuesta_ia": _ej["mensaje"],
                "respuesta_estructurada": {
                    "intent": "SUCCESS" if _ej.get("success") else "INFO",
                    "texto_conversacional": _ej["mensaje"],
                    "secciones": [],
                },
                "tipo_detectado": _ej.get("tipo_detectado", "ejercicio"),
                "datos": _ej.get("datos", {}),
                "balance_actualizado": _bal_ej,
            }

        # ══════════════════════════════════════════════════════════════════════
        # NUEVA ARQUITECTURA: Recomendación y Chat → LLM directo (sin CALOFIT)
        # ══════════════════════════════════════════════════════════════════════
        from app.services.llm_registro import (
            respuesta_recomendacion_llm,
            respuesta_chat_llm,
        )
        from app.services.asistente.asistente_modos import RECOMENDAR_NUTRICION, RECOMENDAR_EJERCICIO

        _hist_limpio = [
            {"role": m.get("role", "user"),
             "content": _resumir_para_historial(m.get("content", ""), m.get("role", "user"))}
            for m in (historial or [])[-6:]
        ]

        if modo_funcion in (RECOMENDAR_NUTRICION, RECOMENDAR_EJERCICIO):
            _modo_rec = "ejercicio" if modo_funcion == RECOMENDAR_EJERCICIO else "comida"
            _texto_rec = await respuesta_recomendacion_llm(
                mensaje, perfil, consumo_real, calorias_meta, quemadas_real, self.ia,
                modo=_modo_rec,
                historial=historial,
                db=db,
                plan_macros=plan_hoy_data,
                consumido_macros={
                    "proteinas":     prog.proteinas_consumidas if prog else 0,
                    "carbohidratos": prog.carbohidratos_consumidos if prog else 0,
                    "grasas":        prog.grasas_consumidas if prog else 0,
                },
            )
            _intent_rec = "RECIPE" if modo_funcion == RECOMENDAR_NUTRICION else "POWER"
            return _build_response(
                perfil, _intent_rec, modo_funcion.upper(),
                calorias_meta, consumo_real, quemadas_real, plan_hoy_data, _texto_rec
            )

        # OTRO / saludo / consulta informativa → respuesta conversacional
        _texto_chat = await respuesta_chat_llm(
            mensaje, perfil, consumo_real, calorias_meta,
            quemadas_real, _hist_limpio, self.ia,
            plan_macros=plan_hoy_data,
        )
        # Guardia: si el LLM falló o devolvió vacío, usar fallback
        if not _texto_chat or len(_texto_chat.strip()) < 5 or _texto_chat.startswith("["):
            _texto_chat = f"¿En qué te puedo ayudar, {perfil.first_name}? Puedes preguntarme sobre nutrición, ejercicio o qué comer hoy."
        return _build_response(
            perfil, "INFO", "OTRO",
            calorias_meta, consumo_real, quemadas_real, plan_hoy_data, _texto_chat
        )

        print("\n=== RAW LLM RESPONSE ===")
        print(respuesta_ia)
        print("========================\n")

        if self.ia.es_fallo_respuesta_llm(respuesta_ia):
            return respuesta_fallo_llm(perfil, consumo_real, calorias_meta, quemadas_real, respuesta_ia, modo_funcion)

        # Parsear y post-procesar
        resp_est = parsear_respuesta_para_frontend(respuesta_ia, mensaje_usuario=mensaje, modo_funcion=modo_funcion)
        resp_est["modo_funcion"] = modo_funcion
        await procesar_secciones_comida(resp_est, perfil, db=db, mensaje_original=mensaje)
        procesar_secciones_ejercicio(resp_est, perfil)

        # ── Guard LOG: para REGISTRAR_NUTRICION nunca mostrar tarjetas RECIPE ──
        # El LLM a veces genera secciones de comida en el fallback del registro.
        # Para un LOG el usuario solo necesita ver la confirmación textual, no tarjetas.
        if modo_funcion == REGISTRAR_NUTRICION:
            resp_est["secciones"] = [
                s for s in (resp_est.get("secciones") or [])
                if s.get("tipo") != "comida"
            ]

        # ── Guard dietético: eliminar secciones con ingredientes prohibidos ──
        # Filtro Python duro — independiente del LLM. Si el nombre del plato o
        # sus ingredientes contienen términos prohibidos por la dieta del usuario,
        # la sección se elimina antes de mostrarse al usuario.
        _conds_guard = list(getattr(perfil, "medical_conditions", None) or [])
        _tokens_guard: set[str] = set()
        if "Vegano" in _conds_guard:
            _tokens_guard = {
                "carne", "pollo", "pechuga", "gallina", "pato", "pavo", "cabrito",
                "cerdo", "chancho", "res", "bistec", "lomo", "chicharron",
                "pescado", "salmon", "atun", "trucha", "caballa", "corvina",
                "camaron", "camarón", "mariscos", "pulpo", "calamar",
                "huevo", "leche", "queso", "yogur", "mantequilla",
            }
        elif "Vegetariano" in _conds_guard:
            _tokens_guard = {
                "carne", "pollo", "pechuga", "gallina", "pato", "pavo", "cabrito",
                "cerdo", "chancho", "res", "bistec", "lomo",
                "pescado", "salmon", "atun", "trucha", "caballa",
                "camaron", "camarón", "mariscos",
            }
        if _tokens_guard and modo_funcion == "recomendar_nutricion":
            def _contiene_prohibido(sec: dict) -> bool:
                texto = (sec.get("nombre", "") + " " +
                         " ".join(sec.get("ingredientes", []))).lower()
                return any(t in texto for t in _tokens_guard)
            secciones_antes = resp_est.get("secciones") or []
            resp_est["secciones"] = [
                s for s in secciones_antes
                if not (s.get("tipo") == "comida" and _contiene_prohibido(s))
            ]
            n_filtradas = len(secciones_antes) - len(resp_est.get("secciones", []))
            if n_filtradas:
                logger.warning(
                    "[DietGuard] %d sección(es) eliminadas por contener "
                    "ingredientes prohibidos (%s)",
                    n_filtradas, list(_tokens_guard)[:5],
                )

        # ── Guard calórico: escalar secciones que superan el restante ────────
        _restantes_actual = calorias_meta - consumo_real + quemadas_real
        if _restantes_actual > 0 and modo_funcion == "recomendar_nutricion":
            for _sec in (resp_est.get("secciones") or []):
                if _sec.get("tipo") != "comida":
                    continue
                _mn = _sec.get("macros_normalizados") or {}
                _kcal_sec = float(_mn.get("kcal") or 0)
                if _kcal_sec > _restantes_actual and _kcal_sec > 0:
                    _factor = _restantes_actual / _kcal_sec
                    _mn["kcal"]            = round(_kcal_sec * _factor, 1)
                    _mn["proteinas_g"]     = round(float(_mn.get("proteinas_g", 0)) * _factor, 1)
                    _mn["carbohidratos_g"] = round(float(_mn.get("carbohidratos_g", 0)) * _factor, 1)
                    _mn["grasas_g"]        = round(float(_mn.get("grasas_g", 0)) * _factor, 1)
                    _sec["macros_normalizados"] = _mn

        # ── Guard: meta calórica cumplida o casi cumplida → eliminar tarjetas RECIPE ──
        # Se omite si el usuario insiste explícitamente en comer ("igual quiero", "de todas formas", etc.)
        _msg_insiste = any(p in mensaje.lower() for p in (
            "igual quiero", "igual quiero comer", "de todas formas", "de igual manera",
            "aun quiero", "aún quiero", "igualmente quiero", "quiero comer igual",
            "pero quiero", "aunque", "de todas maneras",
        ))
        # Umbral 50 kcal: con menos de 50 kcal no tiene sentido mostrar platos reales
        if _restantes_actual <= 50 and modo_funcion == "recomendar_nutricion" and not _msg_insiste:
            resp_est["secciones"] = [
                s for s in (resp_est.get("secciones") or [])
                if s.get("tipo") != "comida"
            ]
            resp_est["intent"] = "INFO"
            # Reemplazar el texto conversacional con un mensaje claro de meta cumplida
            _restantes_str = f"{int(_restantes_actual)} kcal" if _restantes_actual > 0 else "0 kcal"
            resp_est["texto_conversacional"] = (
                f"¡Excelente! Tu meta calórica del día está prácticamente completada "
                f"— solo te faltan {_restantes_str}. "
                f"Puedes tomar agua, una infusión sin azúcar o simplemente esperar "
                f"a la siguiente comida. ¡Buen trabajo hoy! 💪"
            )
            if isinstance(resp_est.get("respuesta_estructurada"), dict):
                resp_est["respuesta_estructurada"]["texto_conversacional"] = (
                    resp_est["texto_conversacional"]
                )

        await rescue_nlp_log(resp_est, mensaje, perfil, self.ia, db)

        # ── Guard LOG final: eliminar tarjetas de comida en modo registro ──────
        # rescue_nlp_log puede crear secciones nuevas — este guard las elimina.
        if modo_funcion == REGISTRAR_NUTRICION:
            resp_est["secciones"] = [
                s for s in (resp_est.get("secciones") or [])
                if s.get("tipo") != "comida"
            ]

        clasificar_intencion_respuesta(resp_est, mensaje)
        limpiar_tags_calofit(resp_est)
        enriquecer_respuesta_estructurada(resp_est, getattr(perfil, "goal", None))
        intencion_principal = detectar_intencion_principal(resp_est, mensaje)
        restantes = max(0.0, calorias_meta - consumo_real + quemadas_real)

        # ── Guard OTRO: forzar intent=INFO y eliminar tarjetas de ejercicio ──
        # El LLM emite [CALOFIT_INTENT:POWER] aunque esté en modo conversacional
        # (p.ej. "puedo ir a nadar"). Corrección server-side triple:
        #   1) intent=INFO en resp_est (para parsers internos)
        #   2) override intencion_principal (ya calculada por detectar_intencion_principal)
        #   3) limpieza de brackets residuales en texto_conversacional
        if modo_funcion == OTRO:
            resp_est["intent"] = "INFO"
            resp_est["intent_ai"] = "INFO"
            resp_est["secciones"] = [
                s for s in (resp_est.get("secciones") or [])
                if s.get("tipo") not in ("ejercicio", "rutina")
            ]
            # Limpiar brackets residuales — el LLM a veces embebe [CALOFIT_INTENT:INFO]
            # dentro del texto y el parser lo elimina dejando "]" suelto.
            import re as _re_m
            _tc = resp_est.get("texto_conversacional", "")
            _tc = _re_m.sub(r'\s*\]\s*', ' ', _tc)   # elimina ] sueltos
            _tc = _re_m.sub(r'\[(?!CALOFIT)[^\]]*\]', '', _tc)  # elimina [otros tags]
            resp_est["texto_conversacional"] = _tc.strip()
            # Override: intencion_principal fue calculada ANTES del guard —
            # hay que pisarla explícitamente para que Flutter muestre bubble, no tarjeta.
            intencion_principal = "INFO"

        # Guard tipo_pregunta: si el modo no es de registro pero el parser asignó "LOG"
        # o "POWER", corregir para que Flutter muestre el chip correcto.
        _tipo_q_raw = (resp_est.get("tipo_pregunta") or modo_funcion or "otro").upper()
        _NO_SON_LOG  = {
            "recomendar_nutricion", "recomendar_ejercicio", "otro",
            "RECOMENDAR_NUTRICION", "RECOMENDAR_EJERCICIO", "OTRO",
        }
        if _tipo_q_raw == "LOG" and modo_funcion in _NO_SON_LOG:
            _tipo_q_raw = modo_funcion.upper()
        # OTRO nunca puede ser POWER o LOG — el usuario no pidió acción de gym
        if modo_funcion == OTRO and _tipo_q_raw in {"POWER", "LOG", "SUCCESS"}:
            _tipo_q_raw = "OTRO"

        return {
            "asistente":    "CaloFit IA",
            "usuario":      perfil.first_name,
            "intencion":    intencion_principal,
            "tipo_pregunta": _tipo_q_raw,
            "alerta_salud": False,
            "data_cientifica": {
                "progreso_diario": {
                    "consumido": round(consumo_real, 1), "meta": round(calorias_meta, 1),
                    "restante":  round(restantes, 1),   "quemado": round(quemadas_real, 1),
                },
                "macros": {
                    "proteinas_meta":     plan_hoy_data.get("proteinas_g", 0),
                    "carbohidratos_meta": plan_hoy_data.get("carbohidratos_g", 0),
                    "grasas_meta":        plan_hoy_data.get("grasas_g", 0),
                },
            },
            "respuesta_ia":           respuesta_ia,
            "respuesta_estructurada": resp_est,
        }

    # ── 2. Registro por NLP ───────────────────────────────────────────────────

    async def registrar_por_nlp(
        self, mensaje: str, db: Session, current_user, consulta_id: str = None
    ):
        perfil = db.query(Client).filter(Client.email == current_user.email).first()
        if not perfil:
            raise ValueError("Perfil de cliente no encontrado")

        if consulta_id and consulta_id.strip():
            payload = get_consulta_cached(consulta_id.strip())
            if payload:
                return registrar_desde_cache(payload, perfil, db)

        edad = (datetime.now().year - perfil.birth_date.year) if perfil.birth_date else 25
        try:
            _, plan_hoy_data, _ = obtener_plan_hoy(perfil, edad, db)
        except Exception:
            plan_hoy_data = {"calorias_dia": 0, "proteinas_g": 0, "carbohidratos_g": 0, "grasas_g": 0}

        from app.services.asistente.asistente_ejercicio import frase_registro_actividad_fisica, frase_vocabulario_gimnasio
        if frase_registro_actividad_fisica(mensaje) or frase_vocabulario_gimnasio(mensaje):
            return await registro_ejercicio_handler.registrar(mensaje, perfil, db, self.ia)
        return await registro_comida_handler.registrar(mensaje, perfil, plan_hoy_data, db, self.ia)

    # ── 2b. Registro manual ───────────────────────────────────────────────────

    async def registrar_manual_alimento(self, body: dict, db: Session, current_user):
        perfil = db.query(Client).filter(Client.email == current_user.email).first()
        if not perfil:
            raise ValueError("Perfil de cliente no encontrado")
        return await registro_comida_handler.registrar_manual(body, perfil, db)

    async def calcular_ejercicio_manual(
        self, nombre: str, series: int, reps: int, peso_kg: float,
        db: Session, current_user
    ):
        """
        Calcula kcal para un ejercicio de fuerza definido por series×reps×peso.
        Fórmula: MET × peso_corporal × 3.5 / 200 × duracion_min
        Duración estimada: series × (reps × 4s + 90s descanso) / 60
        """
        perfil = db.query(Client).filter(Client.email == current_user.email).first()
        if not perfil:
            raise ValueError("Perfil no encontrado")
        peso_corporal = float(getattr(perfil, "weight", None) or 70.0)

        # Buscar MET en el catálogo interno
        from app.services.asistente.asistente_ejercicio import resolver_met_mets_gym
        _, met = resolver_met_mets_gym(nombre.lower())
        if not met:
            met = 5.0  # fuerza genérica

        # Estimar duración: 4 seg/rep + 90 seg descanso por serie
        dur_min = max(series * (reps * 4 + 90) / 60, 3.0)

        from app.services.ejercicios_service import ejercicios_service
        kcal = round(ejercicios_service.calcular_calorias(met, peso_corporal, dur_min), 1)

        return {
            "ejercicio": {
                "nombre": nombre.title(),
                "series":      series,
                "reps":        reps,
                "peso_kg":     peso_kg,
                "duracion_min": round(dur_min, 1),
                "calorias":    kcal,
                "met":         met,
            }
        }

    async def registrar_rutina_manual(self, ejercicios: list, db: Session, current_user):
        """
        Registra una lista de ejercicios con series×reps×peso en workout_logs.
        Cada ítem: {name, series, reps, peso_kg, kcal, met, duracion_min}
        """
        perfil = db.query(Client).filter(Client.email == current_user.email).first()
        if not perfil:
            raise ValueError("Perfil no encontrado")

        total_kcal = 0.0
        for ex in ejercicios:
            kcal    = float(ex.get('kcal', 0.0))
            dur_min = float(ex.get('duracion_min', ex.get('duration_min', 15.0)))
            series  = int(ex.get('series', 1))
            reps    = int(ex.get('reps', 1))
            peso_kg = float(ex.get('peso_kg', 0.0)) or None

            registro_ejercicio_handler._registrar_workout_log_completo(
                client_id=perfil.id,
                ejercicio=ex.get('name', 'Ejercicio'),
                series=series, reps=reps, peso_kg=peso_kg,
                calorias_quemadas=kcal,
                session_duration_min=dur_min,
                met=float(ex.get('met', 5.0)),
                db=db,
            )
            registro_ejercicio_handler._sumar_calorias_progreso(perfil.id, kcal, db)
            total_kcal += kcal

        return {
            "success": True,
            "mensaje": f"Rutina registrada: {len(ejercicios)} ejercicios — {total_kcal:.0f} kcal quemadas",
            "total_kcal": round(total_kcal, 1),
        }

    # ── 3. Confirmar desde card ───────────────────────────────────────────────

    async def confirmar_registro(self, consulta_id: str, db: Session, current_user):
        perfil = db.query(Client).filter(Client.email == current_user.email).first()
        if not perfil:
            raise ValueError("Perfil de cliente no encontrado")
        payload = get_consulta_cached(consulta_id)
        if not payload:
            raise ValueError("Registro expirado. El consulta_id solo es válido 10 minutos.")
        hoy = get_peru_date()
        progreso = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == perfil.id, ProgresoCalorias.fecha == hoy
        ).first()
        if not progreso:
            progreso = ProgresoCalorias(client_id=perfil.id, fecha=hoy)
            db.add(progreso)
        if es_payload_ejercicio(payload):
            meta = registrar_ejercicio_desde_payload_tarjeta(payload, perfil, progreso, db)
        else:
            meta = registrar_comida_desde_payload_tarjeta(payload, perfil, progreso, db)
        db.commit()
        return {
            "success": True,
            "mensaje": f"Registrado: {meta['nombre']} — {meta['calorias']:.0f} kcal (mismos valores del chat).",
            "tipo_detectado": meta["tipo_detectado"],
            "balance_actualizado": {
                "consumido":       progreso.calorias_consumidas,
                "quemado":         progreso.calorias_quemadas,
                "proteinas_g":     progreso.proteinas_consumidas,
                "carbohidratos_g": progreso.carbohidratos_consumidos,
                "grasas_g":        progreso.grasas_consumidas,
            },
        }

    # ── Privados ──────────────────────────────────────────────────────────────

    async def _analizar_salud_background(self, mensaje: str, perfil, db: Session):
        try:
            if self.ia.identificar_intencion_salud(mensaje) == "ALERT":
                db.add(AlertaSalud(
                    client_id=perfil.id, tipo="sintoma",
                    descripcion=mensaje[:200], severidad="medio", estado="pendiente",
                ))
                db.commit()
        except Exception as e:
            print(f"[SaludBG] Error: {e}")


asistente_service = AsistenteService()
