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
import re
from datetime import datetime

# Verbos de ingesta en pasado — redirigen al handler directo (sin LLM)
# cuando el modo ya fue clasificado como REGISTRAR_NUTRICION.
# Ejemplos: "Temprano comí pan con pollo", "Almorcé lomo saltado con su gaseosa"
_RE_PASADO_COMER = re.compile(
    r"\b(com[ií]|desayun[eé]|almor[cz][eaé]|cen[eé]|tom[eé]|beb[ií]|inger[ií])\b",
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

    # 6. Máximo 200 chars — el LLM solo necesita el hecho central
    if len(texto) > 200:
        texto = texto[:200].rsplit(" ", 1)[0] + "."

    return texto

from sqlalchemy.orm import Session

from app.core.cache import get_consulta_cached
from app.core.utils import get_peru_date
from app.models.client import Client
from app.models.historial import AlertaSalud, ProgresoCalorias
from app.services.asistente_ejercicio import (
    es_payload_ejercicio,
    procesar_secciones_ejercicio,
    registrar_ejercicio_desde_payload_tarjeta,
)
from app.services.asistente_modos import (
    RECOMENDAR_EJERCICIO, REGISTRAR_NUTRICION,
    _VERBOS_IMPERATIVOS_REGISTRO, resolver_modo_funcion,
)
from app.services.asistente_nutricion import (
    procesar_secciones_comida,
    registrar_comida_desde_payload_tarjeta,
)
from app.services.asistente_plan import obtener_plan_hoy
from app.services.asistente_prompt import (
    clasificar_intencion_respuesta,
    construir_prompt_cliente,
    detectar_intencion_principal,
    enriquecer_prompt_con_bd,
    limpiar_tags_calofit,
    rescue_nlp_log,
    respuesta_fallo_llm,
    respuesta_info_faltante,
)
from app.services.asistente_registro_comida import (
    registrar_desde_cache,
    registro_comida_handler,
)
from app.services.asistente_registro_ejercicio import registro_ejercicio_handler
from app.services.asistente_respuesta_normalize import enriquecer_respuesta_estructurada
from app.services.ia_service import ia_engine
from app.services.missing_data_guard import (
    detectar_faltantes,
    detectar_faltantes_recomendar_ejercicio,
    strict_ask_missing_enabled,
)
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

class AsistenteService:
    """Punto de entrada único del asistente del cliente (< 300 líneas)."""

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
        quemadas_real = prog.calorias_quemadas   if prog else 0
        calorias_meta = plan_hoy_data["calorias_dia"]
        progreso_pct  = min(100, (consumo_real / calorias_meta) * 100) if calorias_meta > 0 else 0
        adherencia_pct = round(progreso_pct * 0.7 + 20, 1)

        alerta_fuzzy  = self.ia.generar_alerta_fuzzy(adherencia_pct, progreso_pct)
        mensaje_fuzzy = alerta_fuzzy.get("mensaje", "")
        msg_limpio    = mensaje.lower().strip()
        es_saludo     = len(msg_limpio) < 20 and any(
            s in msg_limpio for s in ["hola", "buen", "hey", "salu", "que tal", "qué tal", "gracias"]
        )
        if not es_saludo:
            asyncio.create_task(self._analizar_salud_background(mensaje, perfil, db))

        # Modo funcional + guard rails
        modo_funcion = await resolver_modo_funcion(self.ia, mensaje, es_saludo)

        # ── Redirección registro comida → NLP handler (sin pasar por LLM) ─────────
        # Cubre dos patrones cuando modo ya clasificado como REGISTRAR_NUTRICION:
        #   A) Imperativo:  "Regístrame / Anota / Guarda el ceviche"
        #   B) Pasado:      "Temprano comí pan con pollo, almorcé lomo saltado"
        # Evita que el flujo consultar() muestre una tarjeta RECIPE sin persistir.
        if modo_funcion == REGISTRAR_NUTRICION and (
            any(msg_limpio.startswith(v) for v in _VERBOS_IMPERATIVOS_REGISTRO)
            or bool(_RE_PASADO_COMER.search(msg_limpio))
        ):
            _com = await registro_comida_handler.registrar(
                mensaje, perfil, plan_hoy_data, db, self.ia
            )
            # Envolver en el formato completo que Flutter espera (respuesta_ia + respuesta_estructurada)
            _hoy_c = get_peru_date()
            _p_c   = db.query(ProgresoCalorias).filter(
                ProgresoCalorias.client_id == perfil.id,
                ProgresoCalorias.fecha == _hoy_c,
            ).first()
            _con_c = float(_p_c.calorias_consumidas if _p_c else consumo_real)
            _que_c = float(_p_c.calorias_quemadas   if _p_c else quemadas_real)
            _msg_c = _com.get("mensaje", "")
            return {
                "asistente":    "CaloFit IA",
                "usuario":      perfil.first_name,
                "intencion":    "SUCCESS" if _com.get("success") else "ERROR",
                "tipo_pregunta": "LOG",
                "alerta_salud": False,
                "data_cientifica": {
                    "progreso_diario": {
                        "consumido": round(_con_c, 1),
                        "meta":      round(calorias_meta, 1),
                        "restante":  round(max(0, calorias_meta - _con_c + _que_c), 1),
                        "quemado":   round(_que_c, 1),
                    },
                    "macros": {
                        "proteinas_meta":     plan_hoy_data.get("proteinas_g", 0),
                        "carbohidratos_meta": plan_hoy_data.get("carbohidratos_g", 0),
                        "grasas_meta":        plan_hoy_data.get("grasas_g", 0),
                    },
                },
                "respuesta_ia": _msg_c,
                "respuesta_estructurada": {
                    "intent":               "SUCCESS" if _com.get("success") else "ERROR",
                    "texto_conversacional": _msg_c,
                    "secciones":            [],
                },
                "tipo_detectado":      _com.get("tipo_detectado", "comida"),
                "datos":               _com.get("datos", {}),
                "balance_actualizado": _com.get("balance_actualizado", {}),
                # Advertencias para el frontend
                "advertencia_dieta":     _com.get("advertencia_dieta"),
                "advertencia_prohibido": _com.get("advertencia_prohibido"),
                "advertencia_horario":   _com.get("advertencia_horario"),
                "advertencia_temporal":  _com.get("advertencia_temporal"),
                "advertencia_gula":      _com.get("advertencia_gula"),
                "advertencia_gramaje":   _com.get("advertencia_gramaje"),
                "alerta_macros":         _com.get("alerta_macros"),
            }

        # ── Redirección ejercicio: "Hoy realice / Hice X series…" → handler directo ──
        # Evita que el LLM genere texto con [CALOFIT_INTENT:LOG] sin registrar nada en BD.
        from app.services.asistente_ejercicio import frase_registro_actividad_fisica as _fraf
        if _fraf(mensaje):
            _ej = await registro_ejercicio_handler.registrar(mensaje, perfil, db, self.ia)
            if _ej.get("success"):
                _hoy2 = get_peru_date()
                _p2   = db.query(ProgresoCalorias).filter(
                    ProgresoCalorias.client_id == perfil.id,
                    ProgresoCalorias.fecha == _hoy2,
                ).first()
                _q2 = float(_p2.calorias_quemadas   if _p2 else 0)
                _c2 = float(_p2.calorias_consumidas  if _p2 else consumo_real)
                return {
                    "asistente":    "CaloFit IA",
                    "usuario":      perfil.first_name,
                    "intencion":    "SUCCESS",
                    "tipo_pregunta": "LOG",
                    "alerta_salud": False,
                    "data_cientifica": {
                        "progreso_diario": {
                            "consumido": round(_c2, 1),
                            "meta":      round(calorias_meta, 1),
                            "restante":  round(max(0, calorias_meta - _c2 + _q2), 1),
                            "quemado":   round(_q2, 1),
                        },
                        "macros": {
                            "proteinas_meta":     plan_hoy_data.get("proteinas_g", 0),
                            "carbohidratos_meta": plan_hoy_data.get("carbohidratos_g", 0),
                            "grasas_meta":        plan_hoy_data.get("grasas_g", 0),
                        },
                    },
                    "respuesta_ia": _ej.get("mensaje", ""),
                    "respuesta_estructurada": {
                        "intent":                "SUCCESS",
                        "texto_conversacional":  _ej.get("mensaje", ""),
                        "secciones":             [],
                    },
                    "tipo_detectado": _ej.get("tipo_detectado", "ejercicio"),
                    "datos":          _ej.get("datos", {}),
                    "balance_actualizado": _ej.get("balance_actualizado", {}),
                }

        # ── Guard: vocab gym + "por/durante X min" sin verbo → registro implícito ──
        # Cubre "fondos en paralelas por 20min", "sentadillas durante 15 min", etc.
        # El usuario omitió el verbo "hice" pero la intención es reportar lo que hizo.
        # Se añade "hice " para que el NLP lo procese igual que el flujo estándar.
        from app.services.asistente_ejercicio import frase_vocabulario_gimnasio as _fvg
        from app.services.asistente_registro_ejercicio import _RE_DURACION as _RE_DUR_EJ
        _MARCADORES_FUTURO = (
            "quiero", "voy a", "necesito", "puedo hacer",
            "debo", "cómo hacer", "como hacer", "cómo se", "como se", "me recomiend",
            "ejercicio",  # "ejercicios de pecho por 30 min" = recomendación, no registro
        )
        if (
            not _fraf(mensaje)
            and "?" not in msg_limpio
            and _fvg(mensaje)
            and _RE_DUR_EJ.search(msg_limpio)
            and not any(p in msg_limpio for p in _MARCADORES_FUTURO)
        ):
            _ej2 = await registro_ejercicio_handler.registrar("hice " + mensaje, perfil, db, self.ia)
            if _ej2.get("success"):
                _hoy3 = get_peru_date()
                _p3   = db.query(ProgresoCalorias).filter(
                    ProgresoCalorias.client_id == perfil.id,
                    ProgresoCalorias.fecha == _hoy3,
                ).first()
                _q3 = float(_p3.calorias_quemadas   if _p3 else 0)
                _c3 = float(_p3.calorias_consumidas  if _p3 else consumo_real)
                return {
                    "asistente":    "CaloFit IA",
                    "usuario":      perfil.first_name,
                    "intencion":    "SUCCESS",
                    "tipo_pregunta": "LOG",
                    "alerta_salud": False,
                    "data_cientifica": {
                        "progreso_diario": {
                            "consumido": round(_c3, 1),
                            "meta":      round(calorias_meta, 1),
                            "restante":  round(max(0, calorias_meta - _c3 + _q3), 1),
                            "quemado":   round(_q3, 1),
                        },
                        "macros": {
                            "proteinas_meta":     plan_hoy_data.get("proteinas_g", 0),
                            "carbohidratos_meta": plan_hoy_data.get("carbohidratos_g", 0),
                            "grasas_meta":        plan_hoy_data.get("grasas_g", 0),
                        },
                    },
                    "respuesta_ia": _ej2.get("mensaje", ""),
                    "respuesta_estructurada": {
                        "intent":               "SUCCESS",
                        "texto_conversacional": _ej2.get("mensaje", ""),
                        "secciones":            [],
                    },
                    "tipo_detectado":      _ej2.get("tipo_detectado", "ejercicio"),
                    "datos":               _ej2.get("datos", {}),
                    "balance_actualizado": _ej2.get("balance_actualizado", {}),
                }

        if strict_ask_missing_enabled():
            falt = detectar_faltantes(modo_funcion, mensaje)
            if falt:
                return respuesta_info_faltante(perfil, modo_funcion, falt)
        if not override_ia and not es_saludo and modo_funcion == RECOMENDAR_EJERCICIO:
            falt_ex = detectar_faltantes_recomendar_ejercicio(mensaje)
            if falt_ex:
                return respuesta_info_faltante(perfil, modo_funcion, falt_ex)

        # ── Auto-rutina desde perfil cuando el usuario no especifica zona ─────
        if not override_ia and not es_saludo and modo_funcion == RECOMENDAR_EJERCICIO:
            from app.services.missing_data_guard import _RX_FOCO_MUSCULAR, _RX_DURACION
            from app.services.rutina_service import (
                generar_rutina_inteligente, zonas_desde_workout_type,
            )
            if not _RX_FOCO_MUSCULAR.search(msg_limpio):
                _zonas_auto = zonas_desde_workout_type(getattr(perfil, "workout_type", None))
                _mins_msg   = _extraer_minutos(msg_limpio) if _RX_DURACION.search(msg_limpio) else 0
                _mins_auto  = _mins_msg or int((getattr(perfil, "session_duration", 1.0) or 1.0) * 60)
                _rutina     = await generar_rutina_inteligente(perfil.id, _zonas_auto, _mins_auto, db)
                override_ia = _rutina_a_texto(_zonas_auto, _rutina)
        # ─────────────────────────────────────────────────────────────────────

        # Construir prompt
        ctx = construir_prompt_cliente(
            perfil, edad, plan_hoy_data, calorias_meta,
            consumo_real, quemadas_real, adherencia_pct, progreso_pct,
            mensaje_fuzzy, es_saludo=es_saludo, db=db, modo_funcion=modo_funcion,
            mensaje_usuario=mensaje,
        )

        # Llamar IA
        if override_ia:
            respuesta_ia = override_ia
        else:
            ctx_hist = ""
            if historial:
                # Limpiar mensajes del asistente antes de pasarlos al LLM:
                # se eliminan desgloses nutricionales, emojis, bullets y
                # patrones "| P:Xg" que Llama-3 interpretaría literalmente.
                ctx_hist = "\n\nHISTORIAL RECIENTE:\n" + "\n".join(
                    "{}: {}".format(
                        m.get("role", "user").upper(),
                        _resumir_para_historial(m.get("content", ""), m.get("role", "user")),
                    )
                    for m in historial[-4:]
                )
            prompt_final = await enriquecer_prompt_con_bd(
                ctx + _ctx_extra + ctx_hist + f"\n\nMENSAJE DEL USUARIO: {mensaje}",
                mensaje, modo_funcion, db,
            )
            temp = 0.42 if modo_funcion == "recomendar_nutricion" else \
                   0.45 if modo_funcion == "recomendar_ejercicio" else 0.55
            respuesta_ia = await self.ia._llamar_groq(prompt=prompt_final, max_tokens=1200, temp=temp)

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

        # ── Guard: meta calórica cumplida → eliminar tarjetas RECIPE ─────────
        # El LLM a veces genera tarjetas de comida aunque se le diga que no.
        # Este guard las borra en código, independiente de lo que haya generado el LLM.
        _restantes_actual = calorias_meta - consumo_real + quemadas_real
        if _restantes_actual <= 0 and modo_funcion == "recomendar_nutricion":
            resp_est["secciones"] = [
                s for s in (resp_est.get("secciones") or [])
                if s.get("tipo") != "comida"
            ]
            resp_est["intent"] = "INFO"

        await rescue_nlp_log(resp_est, mensaje, perfil, self.ia, db)
        clasificar_intencion_respuesta(resp_est, mensaje)
        limpiar_tags_calofit(resp_est)
        enriquecer_respuesta_estructurada(resp_est, getattr(perfil, "goal", None))
        intencion_principal = detectar_intencion_principal(resp_est, mensaje)
        restantes = max(0, calorias_meta - consumo_real + quemadas_real)

        return {
            "asistente":    "CaloFit IA",
            "usuario":      perfil.first_name,
            "intencion":    intencion_principal,
            "tipo_pregunta": (resp_est.get("tipo_pregunta") or modo_funcion or "otro").upper(),
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

        from app.services.asistente_ejercicio import frase_registro_actividad_fisica, frase_vocabulario_gimnasio
        if frase_registro_actividad_fisica(mensaje) or frase_vocabulario_gimnasio(mensaje):
            return await registro_ejercicio_handler.registrar(mensaje, perfil, db, self.ia)
        return await registro_comida_handler.registrar(mensaje, perfil, plan_hoy_data, db, self.ia)

    # ── 2b. Registro manual ───────────────────────────────────────────────────

    async def registrar_manual_alimento(self, body: dict, db: Session, current_user):
        perfil = db.query(Client).filter(Client.email == current_user.email).first()
        if not perfil:
            raise ValueError("Perfil de cliente no encontrado")
        return await registro_comida_handler.registrar_manual(body, perfil, db)

    async def calcular_ejercicio_manual(self, texto: str, db: Session, current_user):
        perfil = db.query(Client).filter(Client.email == current_user.email).first()
        if not perfil:
            raise ValueError("Perfil no encontrado")
        peso_kg = float(getattr(perfil, "weight", None) or 70.0)
        
        extraccion = registro_ejercicio_handler._extraer_ejercicio_nlp(texto, texto.lower(), peso_kg, self.ia)
        if not extraccion or not extraccion.get("es_ejercicio"):
            return {"ejercicio": None}
            
        nombre = extraccion["ejercicios_detectados"][0]
        if " (" in nombre:
            nombre = nombre.split(" (")[0]
            
        return {
            "ejercicio": {
                "nombre": nombre,
                "duracion": extraccion.get("duracion_min", 15),
                "calorias": extraccion.get("calorias", 0.0),
                "met": extraccion.get("met", 5.0)
            }
        }

    async def registrar_rutina_manual(self, ejercicios: list, db: Session, current_user):
        perfil = db.query(Client).filter(Client.email == current_user.email).first()
        if not perfil:
            raise ValueError("Perfil no encontrado")
            
        for ex in ejercicios:
            dur_str = str(ex.get('duration', '15')).replace(' min', '')
            dur_min = float(dur_str) if dur_str.replace('.', '', 1).isdigit() else 15.0
            kcal = float(ex.get('kcal', 0.0))
            
            registro_ejercicio_handler._registrar_workout_log_completo(
                client_id=perfil.id,
                ejercicio=ex.get('name', 'Ejercicio'),
                series=1, reps=1, peso_kg=None,
                calorias_quemadas=kcal,
                session_duration_min=dur_min,
                met=float(ex.get('met', 5.0)),
                db=db
            )
            registro_ejercicio_handler._sumar_calorias_progreso(perfil.id, kcal, db)
            
        return {"success": True, "mensaje": "Rutina registrada correctamente"}

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
