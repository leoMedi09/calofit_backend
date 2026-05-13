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
from datetime import datetime

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

        # ── Redirección imperativa: "Regístrame / Anota / Guarda" → NLP handler ──
        # Evita que el flujo consultar() muestre una tarjeta RECIPE sin persistir.
        if modo_funcion == REGISTRAR_NUTRICION and any(
            msg_limpio.startswith(v) for v in _VERBOS_IMPERATIVOS_REGISTRO
        ):
            return await registro_comida_handler.registrar(
                mensaje, perfil, plan_hoy_data, db, self.ia
            )

        if strict_ask_missing_enabled():
            falt = detectar_faltantes(modo_funcion, mensaje)
            if falt:
                return respuesta_info_faltante(perfil, modo_funcion, falt)
        if not override_ia and not es_saludo and modo_funcion == RECOMENDAR_EJERCICIO:
            falt_ex = detectar_faltantes_recomendar_ejercicio(mensaje)
            if falt_ex:
                return respuesta_info_faltante(perfil, modo_funcion, falt_ex)

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
                ctx_hist = "\n\nHISTORIAL RECIENTE:\n" + "\n".join(
                    f"{m.get('role','user').upper()}: {m.get('content','')}" for m in historial[-4:]
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
