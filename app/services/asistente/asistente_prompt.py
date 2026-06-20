"""
Post-proceso de la respuesta del asistente.

Funciones exportadas:
  clasificar_intencion_respuesta() — decide tarjeta vs texto plano
  limpiar_tags_calofit()           — elimina residuos de tags CALOFIT
  detectar_intencion_principal()   — tema visual para Flutter (RECIPE, POWER, etc.)
"""
from __future__ import annotations

import re

from app.services.response_parser import sanear_texto_conversacional_recipe


# ── Post-procesado de respuesta ───────────────────────────────────────────────

def clasificar_intencion_respuesta(respuesta_estructurada: dict, mensaje: str) -> None:
    """Clasifica si la respuesta debe mostrarse como tarjeta (card) o texto plano."""
    msg_low    = mensaje.lower()
    texto_ai   = respuesta_estructurada.get("texto_conversacional", "").lower()
    intent_ai  = str(respuesta_estructurada.get("intent") or "INFO").upper().strip()

    m = re.search(r'\[\s*CALOFIT_INTENT\s*:\s*(.*?)\s*\]', texto_ai)
    if m and intent_ai not in ("RECIPE", "POWER", "LOG"):
        intent_ai = m.group(1).upper().strip()

    tipo_pregunta = str(respuesta_estructurada.get("modo_funcion") or "otro").upper().strip()
    respuesta_estructurada["intent_ai"]    = intent_ai
    respuesta_estructurada["tipo_pregunta"] = tipo_pregunta

    secciones_comida = [s for s in respuesta_estructurada.get("secciones", []) if s.get("tipo") == "comida"]
    _verbos_log = ("comi", "comí", "almorcé", "almorce", "desayuné", "desayune",
                   "cené", "cene", "tomé", "tome", "bebí", "bebi")
    _es_log_verb = any(v in msg_low for v in _verbos_log)

    es_info_directa = (
        intent_ai in ["INFO", "PROGRESS", "NORMAL"] or
        (
            intent_ai != "LOG" and not _es_log_verb and
            len(secciones_comida) == 1 and
            not any(k in msg_low for k in [
                "opcion", "opciones", "receta", "menú", "menu", "cena",
                "almuerzo", "desayuno", "suger", "dame", "recomienda",
            ])
        )
    )

    secciones_conservar = []
    for sec in respuesta_estructurada.get("secciones", []):
        tipo = sec.get("tipo")
        if tipo == "comida":
            if intent_ai in ["INFO", "PROGRESS"] and not any(
                k in msg_low for k in
                ["como", "comer", "opcion", "opciones", "receta", "menú", "menu",
                 "cena", "almuerzo", "desayuno", "suger", "dame", "recomienda", "plan"]
            ):
                continue
            tiene_pasos = bool(sec.get("pasos") or sec.get("preparacion"))
            if not tiene_pasos and es_info_directa:
                titulo     = re.sub(r'\[/?[A-Z_]+.*$', '', sec.get("nombre", "Alimento")).strip()
                lista      = "\n".join(f"• {ing}" for ing in sec.get("ingredientes", []))
                stats_raw  = sec.get("macros", "")
                stats      = stats_raw.replace("P:", "🥚 P:").replace("C:", "🍞 C:").replace("G:", "🥑 G:")
                texto_extra = f"\n\n🍏 **{titulo}**\n{lista}"
                if stats.strip():
                    texto_extra += f"\n\n📊 {stats}"
                actual = respuesta_estructurada.get("texto_conversacional", "")
                respuesta_estructurada["texto_conversacional"] = (actual + texto_extra).strip()
                continue
            secciones_conservar.append(sec)
        else:
            secciones_conservar.append(sec)

    respuesta_estructurada["secciones"] = secciones_conservar


def limpiar_tags_calofit(respuesta_estructurada: dict) -> None:
    """Elimina residuos de tags CALOFIT del texto y secciones."""
    # Con corchetes: [CALOFIT_INTENT:LOG] / [/CALOFIT_HEADER]
    _re = re.compile(r'\[/?CALOFIT_[A-Z_:]*.*?\]', re.IGNORECASE)
    # Sin corchetes: cuando el LLM filtra el texto dentro de un HEADER
    _re_bare = re.compile(
        r'\bCALOFIT_(?:INTENT|HEADER|LIST|ACTION|STATS|QUESTION_TYPE)'
        r'(?:\s*[:/]\s*\w+)?\b',
        re.IGNORECASE,
    )

    def _limpiar(t: str) -> str:
        return _re_bare.sub("", _re.sub("", t)).strip()

    texto = respuesta_estructurada.get("texto_conversacional", "")
    respuesta_estructurada["texto_conversacional"] = sanear_texto_conversacional_recipe(
        _limpiar(texto)
    )
    for s in respuesta_estructurada.get("secciones", []):
        for k in ["nombre", "macros", "gasto_calorico_estimado", "nota"]:
            if s.get(k):
                s[k] = _limpiar(str(s[k]))
        for k in ["ingredientes", "ejercicios", "preparacion", "tecnica", "instrucciones"]:
            if s.get(k) and isinstance(s[k], list):
                s[k] = [_limpiar(str(item)) for item in s[k]]


def detectar_intencion_principal(respuesta_estructurada: dict, mensaje: str) -> str:
    """Devuelve el tema visual para Flutter: RECIPE, POWER, PROGRESS, SUCCESS, DANGER, INFO."""
    secciones  = respuesta_estructurada.get("secciones", [])
    intent_ai  = respuesta_estructurada.get("intent_ai", "INFO")
    msg_low    = mensaje.lower()
    modo_fn    = (respuesta_estructurada.get("modo_funcion") or "").strip().lower()
    tipo_p     = (respuesta_estructurada.get("tipo_pregunta") or "").upper()
    texto_full = (respuesta_estructurada.get("texto_conversacional", "") + mensaje).lower()
    tipos      = [s.get("tipo") for s in secciones]

    if any(s.get("tipo") == "alerta" for s in secciones):
        return "DANGER"
    if intent_ai == "PROGRESS" or "balance" in msg_low or "progreso" in msg_low:
        return "PROGRESS"
    if "anotado" in texto_full or "registrado" in texto_full or intent_ai == "LOG":
        return "SUCCESS"
    if modo_fn == "recomendar_nutricion" or "RECOMENDAR_NUTRICION" in tipo_p:
        return "RECIPE"
    if "ejercicio" in tipos or any(k in texto_full for k in ["entren", "ejercicio", "rutina"]) or intent_ai == "POWER":
        return "POWER"
    if "comida" in tipos and "ejercicio" not in tipos:
        return "RECIPE"
    if any(
        k in texto_full for k in
        ["receta", "sugerencia", "opcion", "menú", "puedo comer", "qué comer", "cena",
         "almuerzo", "desayuno", "plato", "según mi plan"]
    ) or intent_ai == "RECIPE":
        return "RECIPE"
    if any(k in msg_low for k in ["cuántas", "qué tiene", "qué es", "dime sobre"]):
        return "INFO"
    return intent_ai if intent_ai in ("INFO", "RECIPE", "POWER", "PROGRESS", "SUCCESS", "DANGER") else "INFO"


# ── Rescue NLP (LOG sin tarjeta) ─────────────────────────────────────────────

async def rescue_nlp_log(
    resp_est: dict, mensaje: str, perfil, ia_engine, db
) -> None:
    """
    Si el LLM declaró LOG pero no generó ninguna tarjeta de comida,
    ejecuta NLPFoodExtractor y añade la sección resultante.
    """
    if str(resp_est.get("intent") or "").upper() != "LOG":
        return
    # Saltar solo si alguna sección realmente proviene del mensaje del usuario
    # (marcada con _origen_usuario en procesar_secciones_comida).
    # Sugerencias proactivas del LLM (ej. "Palta ensalada" cuando el usuario
    # dijo "comí pan con pollo") no tienen _origen_usuario → rescue sigue.
    if any(
        s.get("tipo") == "comida" and s.get("_origen_usuario")
        for s in (resp_est.get("secciones") or [])
    ):
        return
    try:
        import uuid
        from app.services.asistente.asistente_nutricion import (
            _limpiar_nombre_plato_bd,
            add_user_recent_meal,
            set_consulta_cached,
        )
        from app.services.nlp_food_extractor import NLPFoodExtractor

        res = await NLPFoodExtractor(ia_engine, db).extraer(mensaje)
        if not (res and res.calorias_total > 0):
            return
        nombre  = (_limpiar_nombre_plato_bd(res.nombres[0] if res.nombres else "Comida")).title()
        cid     = str(uuid.uuid4())
        payload = {
            "calorias":        round(res.calorias_total, 1),
            "proteinas_g":     round(res.proteinas_total, 1),
            "carbohidratos_g": round(res.carbohidratos_total, 1),
            "grasas_g":        round(res.grasas_total, 1),
            "nombre": nombre, "ingredientes": [],
        }
        set_consulta_cached(cid, payload)
        try:
            add_user_recent_meal(perfil.id, payload)
        except Exception:
            pass
        mcn = (
            f"Cal: {payload['calorias']}kcal | P: {payload['proteinas_g']}g | "
            f"C: {payload['carbohidratos_g']}g | G: {payload['grasas_g']}g"
        )
        resp_est.setdefault("secciones", []).append({
            "tipo": "comida", "nombre": nombre,
            "macros": mcn, "macros_cache": mcn,
            "ingredientes": getattr(res, "ingredientes", []),
            "preparacion": [], "consulta_id": cid,
        })
        print(f"[NLP-Rescue] LOG sin card → rescatado: {nombre} {payload['calorias']} kcal")
    except Exception as e:
        print(f"[NLP-Rescue] Error: {e}")


def respuesta_fallo_llm(
    perfil, consumo_real: float, calorias_meta: float,
    quemadas_real: float, respuesta_ia: str, modo_funcion: str,
) -> dict:
    """Devuelve el payload de error cuando el LLM falla o está offline."""
    return {
        "asistente": "CaloFit IA", "usuario": perfil.first_name,
        "intencion": "INFO", "tipo_pregunta": "INFO", "alerta_salud": False,
        "data_cientifica": {
            "progreso_diario": {
                "consumido": round(consumo_real, 1),
                "meta":      round(calorias_meta, 1),
                "restante":  round(max(0.0, calorias_meta - consumo_real + quemadas_real), 1),
                "quemado":   round(quemadas_real, 1),
            },
            "macros": {},
        },
        "respuesta_ia": respuesta_ia,
        "respuesta_estructurada": {
            "intent": "INFO", "modo_funcion": modo_funcion,
            "tipo_pregunta": "INFO",
            "texto_conversacional": respuesta_ia,
            "secciones": [],
        },
    }
