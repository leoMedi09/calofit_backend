"""
Servicio del Asistente del Cliente — "CaloFit Coach Personal"

Funcionalidades:
  ✅ Consultas nutricionales (info sobre alimentos, macros)
  ✅ Recomendar comidas/recetas con tarjetas interactivas
  ✅ Recomendar rutinas/ejercicios con tarjetas de entrenamiento
  ✅ Registrar comida por texto/voz (NLP + parsing de macros)
  ✅ Registrar ejercicio por texto/voz (METs + parsing)
  ✅ Consultar progreso diario (calorías, macros, adherencia)
  ✅ Tono adaptativo por lógica difusa
  ✅ Detección de alertas de salud en background
  ✅ Auto-aprendizaje de preferencias alimentarias
  ✅ Confirmar registro desde card (consulta_id)

  🚫 No puede ver otros pacientes
  🚫 No puede modificar planes nutricionales
  🚫 No puede ver alertas del sistema
"""

import uuid
import re
import asyncio
import difflib
import traceback
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func as sql_func

from app.models.client import Client
from app.models.nutricion import PlanNutricional, PlanDiario
from app.models.historial import ProgresoCalorias, AlertaSalud
from app.models.preferencias import PreferenciaAlimento, PreferenciaEjercicio
from app.services.ia_service import ia_engine
from app.services.ml_service import ml_perfil, ml_recomendador          # ML #2 (Perfil) y ML #3 (Recomendador)
from app.services.response_parser import parsear_respuesta_para_frontend
from app.core.utils import parsear_macros_de_texto, get_peru_date
from app.core.cache import set_consulta_cached, get_consulta_cached, add_user_recent_meal, get_user_recent_meals


class AsistenteService:
    """Servicio de negocio para el asistente del cliente."""

    def __init__(self):
        self.ia = ia_engine

    # ─────────────────────────────────────────────────
    # 1. CONSULTAR (Chat principal)
    # ─────────────────────────────────────────────────
    async def consultar(
        self,
        mensaje: str,
        db: Session,
        current_user,
        historial: list = None,
        contexto_manual: str = None,
        override_ia: str = None,
        consulta_id: str = None,
    ):
        """
        Procesa una consulta del cliente al asistente IA.
        Devuelve respuesta estructurada con intención, secciones y datos científicos.
        """
        print(f"🤖 >>> INICIO CONSULTA ASISTENTE <<<")
        print(f"🤖 Usuario Token: {current_user.email} (ID: {current_user.id})")

        # 1. Obtener perfil del cliente autenticado
        perfil = db.query(Client).filter(Client.email.ilike(current_user.email)).first()
        if not perfil:
            print(f"❌ ERROR: Cliente no encontrado en tabla 'clients' para email: {current_user.email}")
            raise ValueError("Perfil de cliente no encontrado")

        print(f"✅ PERFIL CLIENTE: {perfil.first_name} {perfil.last_name_paternal} (ID: {perfil.id})")

        # 2. Calcular edad
        edad = (datetime.now().year - perfil.birth_date.year) if perfil.birth_date else 25

        # 3. Obtener el plan semanal vigente o calcular fallback
        plan_maestro, plan_hoy_data, usa_fallback = self._obtener_plan_hoy(perfil, edad, db)

        # 4. Calcular adherencia y progreso para lógica difusa
        hoy = get_peru_date()
        progreso_hoy = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == perfil.id,
            ProgresoCalorias.fecha == hoy
        ).first()

        consumo_real = progreso_hoy.calorias_consumidas if progreso_hoy else 0
        quemadas_real = progreso_hoy.calorias_quemadas if progreso_hoy else 0

        calorias_meta = plan_hoy_data['calorias_dia']
        if calorias_meta and calorias_meta > 0:
            progreso_pct = min(100, (consumo_real / calorias_meta) * 100)
            adherencia_pct = 85
        else:
            progreso_pct = 50
            adherencia_pct = 85

        # 5. Aplicar lógica difusa para personalizar tono
        alerta_fuzzy = self.ia.generar_alerta_fuzzy(adherencia_pct, progreso_pct)
        mensaje_fuzzy = alerta_fuzzy.get("mensaje", "")

        if "Excelente" in mensaje_fuzzy:
            tono_instruccion = "Usa un tono muy motivador y celebratorio."
        elif "mejorar" in mensaje_fuzzy:
            tono_instruccion = "Usa un tono alentador pero firme."
        else:
            tono_instruccion = "Usa un tono empático pero directo."

        # 6. Detección Inteligente de Salud (Fire-and-Forget)
        msg_limpio = mensaje.lower().strip()
        es_saludo = len(msg_limpio) < 20 and any(
            sal in msg_limpio for sal in ["hola", "buen", "hey", "salu", "que tal", "qué tal", "gracias"]
        )

        async def _analizar_salud_background():
            try:
                # identificar_intencion_salud es SÍNCRONO (devuelve str, no coroutine)
                intencion_salud = self.ia.identificar_intencion_salud(mensaje)
                if intencion_salud == "ALERT":
                    nueva_alerta = AlertaSalud(
                        client_id=perfil.id,
                        tipo="sintoma",
                        descripcion=mensaje[:200],
                        severidad="medio",
                        estado="pendiente"
                    )
                    db.add(nueva_alerta)
                    db.commit()
                    print(f"🚨 Alerta de salud guardada en background para: {perfil.first_name}")
            except Exception as e:
                print(f"⚠️ Error en análisis de salud background: {e}")


        if not es_saludo:
            asyncio.create_task(_analizar_salud_background())

        # 7. Obtener especialistas
        nombre_nutri = "tu nutricionista"
        if perfil.nutritionist:
            nombre_nutri = f"tu nutricionista {perfil.nutritionist.first_name}"

        # 8. Construcción del Prompt con Identidad del Coach Personal
        contexto_asistente = self._construir_prompt_cliente(
            perfil, edad, plan_hoy_data, calorias_meta,
            consumo_real, quemadas_real, adherencia_pct, progreso_pct, mensaje_fuzzy,
            es_saludo=es_saludo, db=db
        )

        # 9. Respuesta de la IA
        if override_ia:
            respuesta_ia = override_ia
        else:
            # Insertar historial reciente dentro del prompt (txt) si existe
            ctx_historial = ""
            if historial:
                ctx_historial = "\n\nHISTORIAL RECIENTE:\n" + "\n".join(
                    f"{m.get('role','user').upper()}: {m.get('content', '')}"
                    for m in historial[-4:]
                )

            prompt_final = (
                contexto_asistente
                + ctx_historial
                + f"\n\nMENSAJE DEL USUARIO: {mensaje}"
            )
            respuesta_ia = await self.ia._llamar_groq(
                prompt=prompt_final,
                max_tokens=1200,
            )

        # 10. Parsear respuesta para Frontend
        respuesta_estructurada = parsear_respuesta_para_frontend(respuesta_ia, mensaje_usuario=mensaje)

        # 11. Flujo de consistencia (cache de consultas)
        self._procesar_secciones_comida(respuesta_estructurada, perfil)

        # 12. Clasificar intención de diseño (RecipeCard vs texto plano)
        self._clasificar_intencion_respuesta(respuesta_estructurada, mensaje)

        # 13. Limpieza de tags CALOFIT
        self._limpiar_tags_calofit(respuesta_estructurada)

        # 14. Detectar intención principal para el Front
        restantes = max(0, calorias_meta - consumo_real + quemadas_real)
        intencion_principal = self._detectar_intencion_principal(
            respuesta_estructurada, mensaje
        )

        return {
            "asistente": "CaloFit IA",
            "usuario": perfil.first_name,
            "intencion": intencion_principal,
            "tipo_pregunta": respuesta_estructurada.get("tipo_pregunta", "ABIERTA"),
            "alerta_salud": bool(
                any(s.get("tipo") == "alerta" for s in respuesta_estructurada.get("secciones", []))
            ),
            "data_cientifica": {
                "progreso_diario": {
                    "consumido": round(consumo_real, 1),
                    "meta": round(calorias_meta, 1),
                    "restante": round(restantes, 1),
                    "quemado": round(quemadas_real, 1)
                },
                "macros": {
                    "P": plan_hoy_data['proteinas_g'],
                    "C": plan_hoy_data['carbohidratos_g'],
                    "G": plan_hoy_data['grasas_g']
                }
            },
            "respuesta_ia": respuesta_ia,
            "respuesta_estructurada": respuesta_estructurada
        }

    # ─────────────────────────────────────────────────
    # 2. REGISTRAR POR NLP (Texto/Voz)
    # ─────────────────────────────────────────────────
    async def registrar_por_nlp(
        self,
        mensaje: str,
        db: Session,
        current_user,
        consulta_id: str = None,
    ):
        """
        Endpoint para registrar comida o ejercicio por voz/texto.
        Si envías consulta_id, se registran los MISMOS valores de la card.
        """
        perfil = db.query(Client).filter(Client.email == current_user.email).first()
        if not perfil:
            raise ValueError("Perfil de cliente no encontrado")

        # Obtener plan del día para verificación de macros
        edad = (datetime.now().year - perfil.birth_date.year) if perfil.birth_date else 25
        try:
            _, plan_hoy_data, _ = self._obtener_plan_hoy(perfil, edad, db)
        except Exception:
            plan_hoy_data = {"calorias_dia": 0, "proteinas_g": 0, "carbohidratos_g": 0, "grasas_g": 0}


        if consulta_id and consulta_id.strip():
            payload = get_consulta_cached(consulta_id.strip())
            if payload:
                return self._registrar_desde_cache(payload, perfil, db)

        # Fuzzy matching contra comidas recientes
        match_exacto = self._fuzzy_match_comidas_recientes(mensaje, perfil)

        # Extraer macros
        peso_usuario = perfil.weight if (perfil.weight and perfil.weight > 0) else 70.0
        extraccion = None

        if match_exacto:
            print(f"🎯 MATCH EXACTO ENCONTRADO (Fuzzy): '{mensaje}' -> '{match_exacto['nombre']}'")
            extraccion = {
                "calorias": match_exacto.get("calorias", 0),
                "proteinas_g": match_exacto.get("proteinas_g", 0),
                "carbohidratos_g": match_exacto.get("carbohidratos_g", 0),
                "grasas_g": match_exacto.get("grasas_g", 0),
                "fibra_g": match_exacto.get("fibra_g", 0),
                "azucar_g": match_exacto.get("azucar_g", 0),
                "sodio_mg": match_exacto.get("sodio_mg", 0),
                "es_comida": True,
                "es_ejercicio": False,
                "alimentos_detectados": [match_exacto.get("nombre", "Comida registrada")],
                "ejercicios_detectados": [],
                "calidad_nutricional": "Alta"
            }
        else:
            # ═══ FEATURE 6: Guía de Porciones ═══
            # Si el usuario no especifica cantidad, pedir orientación
            msg_low = mensaje.lower().strip()
            tiene_cantidad = bool(re.search(r'\d+\s*(g|gr|gramos|ml|kg|litro|unidad|porcion|porción|plato|taza|vaso|cuchara)', msg_low))
            if not tiene_cantidad and not match_exacto:
                # Enviar instrucción de porción estándar a la IA
                mensaje_con_porcion = f"{mensaje} (NOTA INTERNA: el usuario no especificó cantidad; asume 1 porción estándar según USDA)"
                extraccion = await self.ia.extraer_macros_de_texto(mensaje_con_porcion, peso_usuario_kg=peso_usuario)
            else:
                extraccion = await self.ia.extraer_macros_de_texto(mensaje, peso_usuario_kg=peso_usuario)

        if not extraccion or (extraccion.get("calorias", 0) == 0):
            return {
                "success": False,
                "mensaje": "No pude identificar alimentos o ejercicios en tu mensaje. ¿Podrías ser más específico?"
            }

        # ═══ FEATURE: Detección de Alimentos Prohibidos ═══
        advertencia_prohibido = None
        if extraccion.get("es_comida"):
            alimentos_detectados = extraccion.get("alimentos_detectados", [])
            prohibidos = [f.lower().strip() for f in (perfil.forbidden_foods or [])]
            if prohibidos and alimentos_detectados:
                coincidencias = []
                for alimento in alimentos_detectados:
                    al_low = alimento.lower().strip()
                    for prohib in prohibidos:
                        if prohib in al_low or al_low in prohib:
                            coincidencias.append(alimento)
                            break
                if coincidencias:
                    nombres = ", ".join(coincidencias)
                    advertencia_prohibido = (
                        f"⚠️ Atención: '{nombres}' está en tu lista de alimentos prohibidos "
                        f"definida por tu nutricionista. Se registró, pero te recomiendo "
                        f"consultarlo con tu profesional de salud."
                    )
                    print(f"🚫 ALIMENTO PROHIBIDO DETECTADO: {nombres} para cliente {perfil.id}")

        # Actualizar ProgresoCalorias
        hoy = get_peru_date()
        progreso = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == perfil.id, ProgresoCalorias.fecha == hoy
        ).first()

        if not progreso:
            progreso = ProgresoCalorias(client_id=perfil.id, fecha=hoy)
            db.add(progreso)

        calorias = extraccion.get("calorias", 0) or 0
        if extraccion.get("es_comida"):
            progreso.calorias_consumidas = (progreso.calorias_consumidas or 0) + calorias
            progreso.proteinas_consumidas = (progreso.proteinas_consumidas or 0) + (extraccion.get("proteinas_g", 0) or 0)
            progreso.carbohidratos_consumidos = (progreso.carbohidratos_consumidos or 0) + (extraccion.get("carbohidratos_g", 0) or 0)
            progreso.grasas_consumidas = (progreso.grasas_consumidas or 0) + (extraccion.get("grasas_g", 0) or 0)
        else:
            progreso.calorias_quemadas = (progreso.calorias_quemadas or 0) + calorias

        progreso_consumidas = progreso.calorias_consumidas
        progreso_quemadas = progreso.calorias_quemadas

        # Auto-aprendizaje: Registrar preferencias
        self._registrar_preferencias(extraccion, perfil, db)

        db.commit()

        tipo = "comida" if extraccion.get("es_comida") else "ejercicio"
        nombre_item = extraccion.get("alimentos_detectados", extraccion.get("ejercicios_detectados", []))
        nombre_str = ", ".join(nombre_item) if nombre_item else "tu registro"

        # ═══ FEATURE: Conflicto de Macros ═══
        alerta_macros = None
        if extraccion.get("es_comida"):
            alerta_macros = self._verificar_conflicto_macros(
                progreso, plan_hoy_data, perfil
            )

        mensaje_final = f"✅ Registré: {nombre_str} — {extraccion.get('calorias', 0)} kcal. ¡Buen trabajo!".replace("*", "")
        if advertencia_prohibido:
            mensaje_final += f"\n\n{advertencia_prohibido}"
        if alerta_macros:
            mensaje_final += f"\n\n{alerta_macros}"

        return {
            "success": True,
            "tipo_detectado": tipo,
            "alimentos": extraccion.get("alimentos_detectados", []),
            "advertencia_prohibido": advertencia_prohibido,
            "alerta_macros": alerta_macros,
            "balance_actualizado": {
                "consumido": progreso_consumidas,
                "quemado": progreso_quemadas,
            },
            "datos": {
                "calorias": extraccion.get("calorias", 0),
                "proteinas_g": extraccion.get("proteinas_g", 0),
                "carbohidratos_g": extraccion.get("carbohidratos_g", 0),
                "grasas_g": extraccion.get("grasas_g", 0),
                "azucar_g": extraccion.get("azucar_g", 0),
                "fibra_g": extraccion.get("fibra_g", 0),
                "sodio_mg": extraccion.get("sodio_mg", 0),
                "calidad": extraccion.get("calidad_nutricional", "Media"),
            },
            "mensaje": mensaje_final
        }

    # ─────────────────────────────────────────────────
    # 3. CONFIRMAR REGISTRO (desde card con consulta_id)
    # ─────────────────────────────────────────────────
    async def confirmar_registro(
        self,
        consulta_id: str,
        db: Session,
        current_user,
    ):
        """
        Registra una comida usando los valores exactos mostrados en el chat.
        """
        perfil = db.query(Client).filter(Client.email == current_user.email).first()
        if not perfil:
            raise ValueError("Perfil de cliente no encontrado")

        payload = get_consulta_cached(consulta_id)
        if not payload:
            raise ValueError(
                "Registro expirado o no encontrado. El consulta_id solo es válido 10 minutos."
            )

        hoy = get_peru_date()
        progreso = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == perfil.id, ProgresoCalorias.fecha == hoy
        ).first()

        if not progreso:
            progreso = ProgresoCalorias(client_id=perfil.id, fecha=hoy)
            db.add(progreso)

        calorias = payload.get("calorias", 0) or payload.get("calorias_quemadas", 0) or 0
        proteinas_g = payload.get("proteinas_g", 0) or 0
        carbohidratos_g = payload.get("carbohidratos_g", 0) or 0
        grasas_g = payload.get("grasas_g", 0) or 0
        nombre_plato = (payload.get("nombre") or "Actividad").strip()

        # Detectar si es comida o ejercicio
        es_ejercicio = "ejercicios" in payload or "gasto_calorico" in payload or "duracion" in payload
        
        if es_ejercicio:
            progreso.calorias_quemadas = (progreso.calorias_quemadas or 0) + calorias
            pref = db.query(PreferenciaEjercicio).filter(
                PreferenciaEjercicio.client_id == perfil.id,
                sql_func.lower(PreferenciaEjercicio.ejercicio) == nombre_plato.lower(),
            ).first()
            if pref:
                pref.frecuencia += 1
                pref.ultima_vez = datetime.now()
                pref.calorias_quemadas = calorias
            else:
                db.add(PreferenciaEjercicio(
                    client_id=perfil.id, ejercicio=nombre_plato.lower(),
                    frecuencia=1, puntuacion=1.0, ultima_vez=datetime.now(),
                    calorias_quemadas=calorias
                ))
            tipo_resp = "ejercicio"
        else:
            progreso.calorias_consumidas = (progreso.calorias_consumidas or 0) + calorias
            progreso.proteinas_consumidas = (progreso.proteinas_consumidas or 0) + proteinas_g
            progreso.carbohidratos_consumidos = (progreso.carbohidratos_consumidos or 0) + carbohidratos_g
            progreso.grasas_consumidas = (progreso.grasas_consumidas or 0) + grasas_g
            
            pref = db.query(PreferenciaAlimento).filter(
                PreferenciaAlimento.client_id == perfil.id,
                sql_func.lower(PreferenciaAlimento.alimento) == nombre_plato.lower(),
            ).first()
            if pref:
                pref.frecuencia += 1
                pref.ultima_vez = datetime.now()
                pref.calorias = calorias
                pref.proteinas = proteinas_g
                pref.carbohidratos = carbohidratos_g
                pref.grasas = grasas_g
            else:
                db.add(PreferenciaAlimento(
                    client_id=perfil.id, alimento=nombre_plato.lower(),
                    frecuencia=1, puntuacion=1.0, ultima_vez=datetime.now(),
                    calorias=calorias, proteinas=proteinas_g,
                    carbohidratos=carbohidratos_g, grasas=grasas_g
                ))
            tipo_resp = "comida"

        db.commit()

        return {
            "success": True,
            "mensaje": f"Registrado: {nombre_plato} — {calorias} kcal (mismos valores del chat).",
            "tipo_detectado": tipo_resp,
            "balance_actualizado": {
                "consumido": progreso.calorias_consumidas,
                "quemado": progreso.calorias_quemadas,
                "proteinas_g": progreso.proteinas_consumidas,
                "carbohidratos_g": progreso.carbohidratos_consumidos,
                "grasas_g": progreso.grasas_consumidas,
            },
        }

    # ═══════════════════════════════════════════════════
    #  MÉTODOS INTERNOS PRIVADOS
    # ═══════════════════════════════════════════════════

    def _verificar_conflicto_macros(self, progreso, plan_hoy_data, perfil):
        """
        Verifica si el progreso actual supera los límites diarios del plan.
        Retorna un string de advertencia o None si todo está bien.
        """
        alertas = []
        consumidas = progreso.calorias_consumidas or 0
        meta_cal = plan_hoy_data.get("calorias_dia", 0) or 0

        # Verificar calorías totales
        if meta_cal > 0 and consumidas > meta_cal:
            exceso = consumidas - meta_cal
            alertas.append(f"🔥 Calorías: llevas {consumidas:.0f}/{meta_cal:.0f} kcal (+{exceso:.0f} de exceso)")

        # Verificar proteínas
        prot_consumidas = progreso.proteinas_consumidas or 0
        prot_meta = plan_hoy_data.get("proteinas_g", 0) or 0
        if prot_meta > 0 and prot_consumidas > prot_meta * 1.15:
            alertas.append(f"🥚 Proteínas: {prot_consumidas:.0f}g de {prot_meta:.0f}g meta")

        # Verificar carbohidratos
        carbs_consumidos = progreso.carbohidratos_consumidos or 0
        carbs_meta = plan_hoy_data.get("carbohidratos_g", 0) or 0
        if carbs_meta > 0 and carbs_consumidos > carbs_meta * 1.15:
            alertas.append(f"🍞 Carbohidratos: {carbs_consumidos:.0f}g de {carbs_meta:.0f}g meta")

        # Verificar grasas
        grasas_consumidas = progreso.grasas_consumidas or 0
        grasas_meta = plan_hoy_data.get("grasas_g", 0) or 0
        if grasas_meta > 0 and grasas_consumidas > grasas_meta * 1.15:
            alertas.append(f"🥑 Grasas: {grasas_consumidas:.0f}g de {grasas_meta:.0f}g meta")

        if not alertas:
            return None

        detalle = "\n".join(alertas)
        return (
            f"⚠️ Has superado algunos límites de tu plan de hoy:\n"
            f"{detalle}\n"
            f"💡 Considera compensar con actividad física o elegir opciones más ligeras en tu próxima comida."
        )

    def _obtener_plan_hoy(self, perfil, edad, db):
        """Obtiene el plan del día o calcula fallback con IA."""
        print(f"🔍 Buscando plan maestro para cliente ID: {perfil.id}...")
        plan_maestro = db.query(PlanNutricional).filter(
            PlanNutricional.client_id == perfil.id
        ).order_by(PlanNutricional.fecha_creacion.desc()).first()

        usa_fallback = False
        plan_hoy_data = {}

        if not plan_maestro:
            print(f"⚠️ Plan Maestro no encontrado para cliente {perfil.id}. Usando fallback IA...")
            usa_fallback = True

            genero_map = {"M": 1, "F": 2}
            genero = genero_map.get(perfil.gender, 1)

            nivel_map = {
                "Sedentario": 1.2, "Ligero": 1.375, "Moderado": 1.55,
                "Intenso": 1.725, "Muy intenso": 1.9
            }
            nivel_actividad = nivel_map.get(perfil.activity_level, 1.2)

            objetivo_map = {
                "Perder peso": "perder", "Mantener peso": "mantener", "Ganar masa": "ganar"
            }
            objetivo = objetivo_map.get(perfil.goal, "mantener")

            calorias_fallback = self.ia.calcular_requerimiento(
                genero=genero, edad=edad, peso=perfil.weight,
                talla=perfil.height, nivel_actividad=nivel_actividad, objetivo=objetivo
            )

            condiciones_medicas = ", ".join(perfil.medical_conditions) if perfil.medical_conditions else ""
            macros_data = self.ia.calcular_macros_optimizados(
                peso=perfil.weight, objetivo=objetivo,
                calorias=calorias_fallback, condiciones_medicas=condiciones_medicas
            )

            plan_hoy_data = {
                "calorias_dia": calorias_fallback,
                "proteinas_g": macros_data['proteinas_g'],
                "carbohidratos_g": macros_data['carbohidratos_g'],
                "grasas_g": macros_data['grasas_g'],
                "sugerencia_entrenamiento_ia": "Plan calculado automáticamente por IA"
            }

            class PlanFallback:
                def __init__(self, objetivo):
                    self.objetivo = objetivo
                    self.status = "calculado_ia"
                    self.id = None
                    self.fecha_creacion = datetime.now()

            plan_maestro = PlanFallback(objetivo=perfil.goal)
            print(f"✅ FALLBACK IA: {calorias_fallback:.0f} kcal | P:{macros_data['proteinas_g']}g C:{macros_data['carbohidratos_g']}g G:{macros_data['grasas_g']}g")
        else:
            print(f"✅ PLAN MAESTRO: ID {plan_maestro.id} (Status: {plan_maestro.status})")
            dia_semana = datetime.now().isoweekday()
            plan_hoy = db.query(PlanDiario).filter(
                PlanDiario.plan_id == plan_maestro.id,
                PlanDiario.dia_numero == dia_semana
            ).first()

            if not plan_hoy:
                plan_hoy = db.query(PlanDiario).filter(PlanDiario.plan_id == plan_maestro.id).first()

            if not plan_hoy:
                raise ValueError("Tu plan nutricional está incompleto.")

            plan_hoy_data = {
                "calorias_dia": plan_hoy.calorias_dia,
                "proteinas_g": plan_hoy.proteinas_g,
                "carbohidratos_g": plan_hoy.carbohidratos_g,
                "grasas_g": plan_hoy.grasas_g,
                "sugerencia_entrenamiento_ia": plan_hoy.sugerencia_entrenamiento_ia
            }

        return plan_maestro, plan_hoy_data, usa_fallback

    def _construir_prompt_cliente(
        self, perfil, edad, plan_hoy_data, calorias_meta,
        consumo_real, quemadas_real, adherencia_pct, progreso_pct, mensaje_fuzzy,
        es_saludo=False, db=None
    ):
        """Construye el prompt del Coach Personal del Cliente."""
        # Formatear condiciones médicas
        alergias, preferencias_dieta, condiciones = [], [], []
        if perfil.medical_conditions:
            for cond in perfil.medical_conditions:
                cond_l = cond.lower()
                if any(p in cond_l for p in ["alérgico", "alergia", "intolerancia"]):
                    alergias.append(cond)
                elif any(p in cond_l for p in ["vegano", "vegetariano", "pescetariano"]):
                    preferencias_dieta.append(cond)
                else:
                    condiciones.append(cond)

        texto_alergias = ", ".join(alergias) if alergias else "Ninguna"
        texto_dieta = ", ".join(preferencias_dieta) if preferencias_dieta else "Omnívoro"
        texto_condiciones = ", ".join(condiciones) if condiciones else "Ninguna"

        restantes = max(0, calorias_meta - consumo_real + quemadas_real)
        foco = perfil.ai_strategic_focus or "Bienestar General"
        alimentos_pro = perfil.forbidden_foods or []
        alimentos_rec = perfil.recommended_foods or []

        # ═══ FEATURE 4: Sugerencias por Hora del Día ═══
        try:
            from app.core.utils import get_peru_now
            hora_actual = get_peru_now().hour
        except Exception:
            hora_actual = datetime.now().hour

        if 5 <= hora_actual < 10:
            momento_comida = "DESAYUNO"
            sugerencia_horaria = "SI el usuario pide sugerencias de comida, prioriza opciones de desayuno con proteína y fibra."
        elif 10 <= hora_actual < 12:
            momento_comida = "SNACK MAÑANA"
            sugerencia_horaria = "SI el usuario pide sugerencias de comida, recomienda un snack ligero de media mañana."
        elif 12 <= hora_actual < 15:
            momento_comida = "ALMUERZO"
            sugerencia_horaria = "SI el usuario pide sugerencias de comida, sugiere opciones de almuerzo completo y balanceado."
        elif 15 <= hora_actual < 18:
            momento_comida = "SNACK TARDE"
            sugerencia_horaria = "SI el usuario pide sugerencias de comida, recomienda snacks o meriendas saludables para la tarde."
        elif 18 <= hora_actual < 21:
            momento_comida = "CENA"
            sugerencia_horaria = "SI el usuario pide sugerencias de comida, recomienda cenas ligeras y digestivas."
        else:
            momento_comida = "NOCTURNO"
            sugerencia_horaria = "SI el usuario pide sugerencias de comida, advierte sobre la hora nocturna y sugiere solo opciones extremadamente ligeras."

        bloque_hora = (
            f"\nMOMENTO DEL DÍA: {momento_comida} (son las {hora_actual}:00). "
            f"{sugerencia_horaria}"
        )

        # ═══ FEATURE 5: Memoria de Favoritos ═══
        bloque_favoritos = ""
        if db:
            try:
                top_favoritos = db.query(PreferenciaAlimento).filter(
                    PreferenciaAlimento.client_id == perfil.id
                ).order_by(PreferenciaAlimento.frecuencia.desc()).limit(5).all()

                if top_favoritos:
                    lista_favs = ", ".join([f"{p.alimento} ({p.frecuencia}x)" for p in top_favoritos])
                    bloque_favoritos = (
                        f"\nCOMIDAS FAVORITAS DEL USUARIO: {lista_favs}. "
                        f"Si el usuario pide sugerencias generales, prioriza platos basados en sus favoritos."
                    )
            except Exception:
                pass

        # ═══ FEATURE 1: Resumen Proactivo de Progreso al saludar ═══
        bloque_saludo = ""
        if es_saludo:
            pct_dia = min(100, (consumo_real / calorias_meta * 100)) if calorias_meta > 0 else 0
            bloque_saludo = (
                f"\n\nINSTRUCCIÓN ESPECIAL DE SALUDO: "
                f"El usuario acaba de saludar. Responde con calidez E INCLUYE un mini-resumen de su día: "
                f"'Hoy llevas {consumo_real:.0f} de {calorias_meta:.0f} kcal ({pct_dia:.0f}%), "
                f"te faltan {restantes:.0f} kcal. "
                f"Has quemado {quemadas_real:.0f} kcal con ejercicio.' "
                f"Luego pregunta qué necesita: ¿registrar comida, consultar algo, o ver opciones para su próxima comida?"
            )

        # ═══ ML #2: Perfil de Adherencia ═══
        # Predice el perfil del cliente con datos reales de su progreso
        # y genera el bloque de tono personalizado para el LLM.
        bloque_perfil_ml = ""
        try:
            nivel_map = {"Sedentario": 1.2, "Ligero": 1.375, "Moderado": 1.55,
                         "Intenso": 1.725, "Muy intenso": 1.9}
            registros_sem = min(7, max(1, int(adherencia_pct / 15)))  # estimación desde adherencia
            genero_str = "M" if getattr(perfil, 'gender', 'M') == "M" else "F"
            perfil_ml, conf_ml = ml_perfil.predecir_perfil_desde_progreso(
                registros_semana = registros_sem,
                adherencia_pct   = adherencia_pct,
                edad             = edad,
                genero           = genero_str,
                peso             = float(perfil.weight or 70),
                altura           = float(perfil.height or 170),
            )
            tono_perfil = ml_perfil.get_tono_asistente(perfil_ml)
            bloque_perfil_ml = (
                f"\n\nPERFIL ML DEL CLIENTE ({perfil_ml}, confianza {conf_ml}%): "
                f"{tono_perfil}"
            )
        except Exception:
            bloque_perfil_ml = ""

        # ═══ ML #3: Recomendador KNN de Alimentos Peruanos ═══
        # Si al usuario le faltan calorías significativas (> 100 kcal), usamos 
        # el KNN para encontrar los alimentos peruanos óptimos que cubren el déficit.
        bloque_recomendacion_ml = ""
        if restantes > 100:
            try:
                # Calculamos el déficit de proteínas, carbos y grasas aproximado
                # basándonos en la distribución de la meta actual.
                prote_meta = plan_hoy_data.get("proteinas_g", 0)
                carbo_meta = plan_hoy_data.get("carbohidratos_g", 0)
                grasa_meta = plan_hoy_data.get("grasas_g", 0)
                
                # Estimación simple del consumo real de macros (asumiendo distribución equilibrada)
                # En un caso ideal se tomarían de los ProgresoCalorias detallados.
                pct_consumido = (consumo_real / calorias_meta) if calorias_meta > 0 else 0
                prote_faltante = max(0, prote_meta - (prote_meta * pct_consumido))
                carbo_faltante = max(0, carbo_meta - (carbo_meta * pct_consumido))
                grasa_faltante = max(0, grasa_meta - (grasa_meta * pct_consumido))

                # Obtener Top 3 del KNN
                recos = ml_recomendador.obtener_recomendaciones(
                    calorias_faltantes=restantes,
                    prote_faltante=prote_faltante,
                    carbo_faltante=carbo_faltante,
                    grasa_faltante=grasa_faltante,
                    n_recomendaciones=3
                )
                
                if recos:
                    lista_recos = ""
                    for i, r in enumerate(recos):
                        lista_recos += f"{i+1}. {r['alimento']} ({r['calorias_100g']} kcal, {r['proteina_100g']}g proteína por 100g) [Similitud: {r['similitud']}%]\n"
                    
                    bloque_recomendacion_ml = (
                        f"\n\nSUGERENCIAS DEL MODELO ML KNN (BASADO EN DÉFICIT ACTUAL):\n"
                        f"Para cubrir las {restantes:.0f} kcal restantes, el algoritmo recomienda "
                        f"los siguientes alimentos peruanos óptimos:\n{lista_recos}"
                        f"INSTRUCCIÓN: Si el cliente pide qué comer, OBLIGATORIAMENTE incorpora "
                        f"estos alimentos en tus recetas propuestas, porque son los que matemáticamente le ayudan más."
                    )
            except Exception as e:
                print(f"Error en ML Recomendador: {e}")
                bloque_recomendacion_ml = ""

        # ═══ BLOQUE NUTRI — VA PRIMERO (máxima prioridad para el LLM) ═══
        nota_semanal = getattr(perfil, 'nutri_weekly_note', None)
        bloque_nutri_header = (
            f"\n⚠️ INSTRUCCIONES OBLIGATORIAS DEL NUTRICIONISTA (NO NEGOCIABLES):\n"
            f"• FOCO SEMANAL: {foco}\n"
            f"• ALIMENTOS QUE DEBES PRIORIZAR: {', '.join(alimentos_rec) if alimentos_rec else 'Sin restricción especial'}\n"
            f"• ALIMENTOS TERMINANTEMENTE PROHIBIDOS: {', '.join(alimentos_pro) if alimentos_pro else 'Ninguno'} "
            f"— NUNCA los sugieras, aunque el cliente los pida directamente.\n"
            + (f"• META/MENSAJE SEMANAL DEL NUTRICIONISTA: \"{nota_semanal}\"\n"
               f"  → Cuando el cliente salude o pregunte cómo le va, MENCIONAA esta meta como contexto motivacional.\n"
               if nota_semanal else "")
            + "━" * 60 + "\n"
        )

        return (
            bloque_nutri_header
            + f"Eres el coach personal de {perfil.first_name}. "
            f"PERFIL: {perfil.weight}kg, {perfil.height}cm, {edad} años, {perfil.gender}. "
            f"ALERGIAS: {texto_alergias}. "
            f"PREFERENCIAS DIETÉTICAS: {texto_dieta}. "
            f"CONDICIONES MÉDICAS/LESIONES: {texto_condiciones}. "
            f"{bloque_hora}"
            f"{bloque_favoritos}"
            f"{bloque_perfil_ml}"
            f"{bloque_recomendacion_ml}"
            f"\nSTATUS DEL DÍA: "
            f"Meta: {calorias_meta} kcal. Consumido: {consumo_real} kcal. Restante: {restantes} kcal. "
            f"Adherencia: {adherencia_pct:.0f}%, Progreso: {progreso_pct:.0f}%. "
            f"{mensaje_fuzzy}."
            f"\n\nREGLAS DE INTENCIÓN Y FORMATO (OBLIGATORIO):"
            f"\n1. Preguntas sobre PROGRESO/BALANCE o DUDAS GENERALES: responde en MODO PROGRESS o MODO INFO usando SOLO texto conversacional. ¡PROHIBIDO generar la clave 'secciones' estructurada con recetas a menos que te pidan comida directamente!"
            f"\n2. Si piden opciones o recetas (ej: 'qué como ahora'), responde en MODO RECIPE. CRÍTICO: SIEMPRE ofrece 2 o 3 opciones distintas. El texto conversacional DEBE SER SOLO una introducción breve. ¡La receta y preparación van EXCLUSIVAMENTE dentro del objeto 'secciones'!"
            f"\n3. Indica tu intención al inicio de la respuesta con el tag: [CALOFIT_INTENT:CATEGORIA] donde CATEGORIA es (INFO, RECIPE, PROGRESS, POWER, LOG, ALERT)."
            f"\n4. Indica también el TIPO DE PREGUNTA con el tag: [CALOFIT_QUESTION_TYPE:TIPO] donde TIPO es ABIERTA (requiere opciones, pasos o explicación detallada) o RAPIDA (respuesta directa a una duda concreta o registro)."
            f"\n5. En MODO RECIPE, CADA ÍTEM en 'ingredientes' DEBE incluir sus calorías estimadas al lado (ej: '100g de Pollo (165 kcal)')."
            f"\n6. REGLA DE CONSISTENCIA: La preparación (pasos) SOLO debe usar ingredientes listados en la sección de ingredientes. NO inventes ingredientes nuevos en los pasos."
            f"\n7. RESTRICCIÓN TOTAL: NUNCA sugieras los ALIMENTOS PROHIBIDOS del Nutricionista, aunque el cliente lo pida explícitamente. Si lo pide, explica amablemente que tu nutricionista no lo recomienda. Al sugerir rutinas, diseña solo RUTINAS DE GYM/FITNESS, no deportes recreativos."
            f"\n8. REGLA DE MACROS OBLIGATORIA: En MODO RECIPE, al final de cada opción/receta DEBES incluir una línea de macros EXACTA con el formato: 'macros: Xcal, Xg proteína, Xg carbohidratos, Xg grasa'. Calcula sumando los ingredientes. Esta línea ES CRÍTICA para el registro consistente."
            f"{bloque_saludo}"
        )


    def _procesar_secciones_comida(self, respuesta_estructurada, perfil):
        """Cachea secciones de comida para consistencia de registro."""
        for seccion in respuesta_estructurada.get("secciones", []):
            if seccion.get("tipo") != "comida":
                continue

            macros_parsed = parsear_macros_de_texto(seccion.get("macros") or "")

            # Fallback: si los macros no se encontraron en la línea "macros:",
            # sumar las calorías mencionadas en cada ingrediente (formato "xxx kcal")
            if not macros_parsed or macros_parsed.get("calorias", 0) == 0:
                cal_suma = 0.0
                prot_suma = 0.0
                for ing in seccion.get("ingredientes", []):
                    # Detectar patrones "NNN kcal" o "NNN cal"
                    m_cal  = re.search(r'(\d+(?:\.\d+)?)\s*kcal', ing, re.IGNORECASE)
                    m_prot = re.search(r'(\d+(?:\.\d+)?)\s*g\s*prote', ing, re.IGNORECASE)
                    if m_cal:
                        cal_suma  += float(m_cal.group(1))
                    if m_prot:
                        prot_suma += float(m_prot.group(1))
                if cal_suma > 0:
                    # Estimar carbs y grasas desde la calificación de proteínas
                    gras_estimada = cal_suma * 0.25 / 9
                    carb_estimada = (cal_suma - (prot_suma * 4) - (gras_estimada * 9)) / 4
                    macros_parsed = {
                        "calorias":       round(cal_suma, 1),
                        "proteinas_g":    round(prot_suma, 1),
                        "carbohidratos_g": round(max(0, carb_estimada), 1),
                        "grasas_g":       round(gras_estimada, 1),
                    }
                else:
                    macros_parsed = {"calorias": 0, "proteinas_g": 0, "carbohidratos_g": 0, "grasas_g": 0}

            nombre_bruto = seccion.get("nombre") or "Comida"
            nombre_limpio = re.sub(r'\[.*?\]', '', nombre_bruto).split('[')[0].strip()

            consulta_id = str(uuid.uuid4())
            payload = {
                "calorias":        round(macros_parsed["calorias"], 1),
                "proteinas_g":     round(macros_parsed["proteinas_g"], 1),
                "carbohidratos_g": round(macros_parsed["carbohidratos_g"], 1),
                "grasas_g":        round(macros_parsed["grasas_g"], 1),
                "nombre":          nombre_limpio,
                "ingredientes":    seccion.get("ingredientes", [])
            }
            set_consulta_cached(consulta_id, payload)
            seccion["consulta_id"] = consulta_id
            seccion["macros_cache"] = f"{payload['calorias']} kcal | P:{payload['proteinas_g']}g C:{payload['carbohidratos_g']}g G:{payload['grasas_g']}g"
            add_user_recent_meal(perfil.id, payload)


    def _clasificar_intencion_respuesta(self, respuesta_estructurada, mensaje):
        """Clasifica si la respuesta debe mostrarse como tarjeta o texto plano."""
        msg_low = mensaje.lower()
        texto_ai = respuesta_estructurada.get("texto_conversacional", "").lower()
        
        # Detectar tag de intención generado por la IA
        intent_ai = "INFO"
        # Permitir espacios alrededor de los dos puntos o el valor
        intent_match = re.search(r'\[\s*CALOFIT_INTENT\s*:\s*(.*?)\s*\]', texto_ai)
        if intent_match:
            intent_ai = intent_match.group(1).upper().strip()

        # Detectar tipo de pregunta (ABIERTA o RAPIDA)
        tipo_pregunta = "ABIERTA"
        type_match = re.search(r'\[\s*CALOFIT_QUESTION_TYPE\s*:\s*(.*?)\s*\]', texto_ai)
        if type_match:
            tipo_pregunta = type_match.group(1).upper().strip()
            
        # Guardar en la respuesta estructurada para uso posterior
        respuesta_estructurada["intent_ai"] = intent_ai
        respuesta_estructurada["tipo_pregunta"] = tipo_pregunta

        secciones_comida = [s for s in respuesta_estructurada.get("secciones", []) if s.get("tipo") == "comida"]

        # Heurística de Info Directa (Texto Plano)
        es_info_directa = (
            intent_ai in ["INFO", "PROGRESS", "NORMAL"] or
            (len(secciones_comida) == 1 and
             not any(k in msg_low for k in [
                 "opcion", "opciones", "receta", "menú", "menu", "cena",
                 "almuerzo", "desayuno", "suger", "dame", "recomienda", "cocinar", "¿qué", "que como"
             ]))
        )

        secciones_conservar = []
        for seccion in respuesta_estructurada.get("secciones", []):
            tipo = seccion.get("tipo")
            if tipo == "comida":
                # Filtro extremo antimanchas: si es una consulta de avance, descartar la sección "comida" fantasma generada
                if intent_ai in ["INFO", "PROGRESS"] and not any(k in msg_low for k in ["como", "comer", "opcion", "opciones", "receta", "menú", "menu", "cena", "almuerzo", "desayuno", "suger", "dame", "recomienda", "plan"]):
                    continue # Literalmente el usuario no pidió comida activa
                
                tiene_pasos = bool(seccion.get("pasos") or seccion.get("preparacion"))
                
                # REGLA DE COMPLEJIDAD: Si no tiene pasos y se detectó como INFO, forzar texto plano
                if not tiene_pasos and es_info_directa:
                    # Convertir a texto plano concatenado al principal
                    nombre_raw = seccion.get("nombre", "Alimento")
                    titulo = re.sub(r'\[/?[A-Z_]+.*$', '', nombre_raw).replace("[CALOFIT_HEADER]", "").strip()
                    lista = "\n".join([f"• {ing}" for ing in seccion.get("ingredientes", [])])
                    stats_raw = seccion.get("macros", "")
                    stats = stats_raw.replace("P:", "🥚 P:").replace("C:", "🍞 C:").replace("G:", "🥑 G:")
                    stats = stats.replace("Cals:", "🔥 Calorías:").replace("Cal:", "🔥 Calorías:")

                    texto_extra = f"\n\n🍏 **{titulo}**\n{lista}"
                    if stats.strip():
                        texto_extra += f"\n\n📊 {stats}"

                    texto_actual = respuesta_estructurada.get("texto_conversacional", "")
                    respuesta_estructurada["texto_conversacional"] = (texto_actual + texto_extra).strip()
                    continue
                
                # Si llegamos aquí, se conserva como sección (Card)
                secciones_conservar.append(seccion)
            else:
                secciones_conservar.append(seccion)

        respuesta_estructurada["secciones"] = secciones_conservar

    def _limpiar_tags_calofit(self, respuesta_estructurada):
        """Limpieza de residuos de tags CALOFIT del texto y secciones."""
        final_text = respuesta_estructurada.get("texto_conversacional", "")
        cleaned_text = re.sub(r'\[/?CALOFIT_[A-Z_:]*.*?\]', '', final_text, flags=re.IGNORECASE).strip()
        respuesta_estructurada["texto_conversacional"] = cleaned_text

        for s in respuesta_estructurada.get("secciones", []):
            for k_str in ["nombre", "macros", "gasto_calorico_estimado", "nota"]:
                if s.get(k_str):
                    s[k_str] = re.sub(r'\[/?CALOFIT_[A-Z_:]*.*?\]', '', str(s[k_str]), flags=re.IGNORECASE).strip()
            for k_list in ["ingredientes", "ejercicios", "preparacion", "tecnica", "instrucciones"]:
                if s.get(k_list) and isinstance(s[k_list], list):
                    s[k_list] = [
                        re.sub(r'\[/?CALOFIT_[A-Z_:]*.*?\]', '', str(item), flags=re.IGNORECASE).strip()
                        for item in s[k_list]
                    ]

    def _detectar_intencion_principal(self, respuesta_estructurada, mensaje):
        """Detecta la intención principal para tema visual del frontend."""
        secciones = respuesta_estructurada.get("secciones", [])
        intent_ai = respuesta_estructurada.get("intent_ai", "INFO")
        alertas_activas = [s for s in secciones if s.get("tipo") == "alerta"]
        msg_low = mensaje.lower()

        if alertas_activas:
            return "DANGER"
        elif intent_ai == "PROGRESS" or "balance" in msg_low or "progreso" in msg_low:
            return "PROGRESS"
        elif "anotado" in respuesta_estructurada.get("texto_conversacional", "").lower() or \
             "registrado" in respuesta_estructurada.get("texto_conversacional", "").lower() or \
             intent_ai == "LOG":
            return "SUCCESS"
        else:
            tipos = [s.get("tipo") for s in secciones]
            texto_full = (respuesta_estructurada.get("texto_conversacional", "") + mensaje).lower()
            
            if "ejercicio" in tipos or any(k in texto_full for k in ["entren", "ejercicio", "rutina", "rutina"]) or intent_ai == "POWER":
                return "POWER"
            elif any(s.get("pasos") or s.get("preparacion") for s in secciones if s.get("tipo") == "comida") or \
                 any(k in texto_full for k in ["receta", "sugerencia", "opcion", "menu"]) or intent_ai == "RECIPE":
                return "RECIPE"
            elif any(k in msg_low for k in ["cuántas", "cuantas", "qué tiene", "que tiene", "qué es", "que es", "dime sobre"]):
                return "INFO"
        
        return intent_ai if intent_ai in ["INFO", "RECIPE", "POWER", "PROGRESS", "SUCCESS", "DANGER"] else "INFO"

    def _registrar_desde_cache(self, payload, perfil, db):
        """Registra una comida o ejercicio usando datos cacheados de una card."""
        hoy = get_peru_date()
        progreso = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == perfil.id, ProgresoCalorias.fecha == hoy
        ).first()
        if not progreso:
            progreso = ProgresoCalorias(client_id=perfil.id, fecha=hoy)
            db.add(progreso)

        calorias = payload.get("calorias", 0) or payload.get("calorias_quemadas", 0) or 0
        proteinas_g = payload.get("proteinas_g", 0) or 0
        carbohidratos_g = payload.get("carbohidratos_g", 0) or 0
        grasas_g = payload.get("grasas_g", 0) or 0
        nombre = (payload.get("nombre") or "Actividad").strip()
        
        # Detectar si es comida o ejercicio por las claves del payload
        es_ejercicio = "ejercicios" in payload or "gasto_calorico" in payload or "duracion" in payload
        if not es_ejercicio and calorias > 0 and proteinas_g == 0 and carbohidratos_g == 0:
             # Si solo hay calorias, podria ser ejercicio, pero si viene de una card de comida tendria macros.
             # Por seguridad, si el payload no tiene macros, checkemos si es_ejercicio
             pass

        if es_ejercicio:
            progreso.calorias_quemadas = (progreso.calorias_quemadas or 0) + calorias
            pref = db.query(PreferenciaEjercicio).filter(
                PreferenciaEjercicio.client_id == perfil.id,
                sql_func.lower(PreferenciaEjercicio.ejercicio) == nombre.lower(),
            ).first()
            if pref:
                pref.frecuencia += 1
                pref.ultima_vez = datetime.now()
                pref.calorias_quemadas = calorias
            else:
                db.add(PreferenciaEjercicio(
                    client_id=perfil.id, ejercicio=nombre.lower(),
                    frecuencia=1, puntuacion=1.0, ultima_vez=datetime.now(),
                    calorias_quemadas=calorias
                ))
            tipo_detectado = "ejercicio"
            mensaje = f"✅ Registré: {nombre} — {calorias} kcal quemadas."
        else:
            progreso.calorias_consumidas = (progreso.calorias_consumidas or 0) + calorias
            progreso.proteinas_consumidas = (progreso.proteinas_consumidas or 0) + proteinas_g
            progreso.carbohidratos_consumidos = (progreso.carbohidratos_consumidos or 0) + carbohidratos_g
            progreso.grasas_consumidas = (progreso.grasas_consumidas or 0) + grasas_g
            
            pref = db.query(PreferenciaAlimento).filter(
                PreferenciaAlimento.client_id == perfil.id,
                sql_func.lower(PreferenciaAlimento.alimento) == nombre.lower(),
            ).first()
            if pref:
                pref.frecuencia += 1
                pref.ultima_vez = datetime.now()
                pref.calorias = calorias
                pref.proteinas = proteinas_g
                pref.carbohidratos = carbohidratos_g
                pref.grasas = grasas_g
            else:
                db.add(PreferenciaAlimento(
                    client_id=perfil.id, alimento=nombre.lower(),
                    frecuencia=1, puntuacion=1.0, ultima_vez=datetime.now(),
                    calorias=calorias, proteinas=proteinas_g,
                    carbohidratos=carbohidratos_g, grasas=grasas_g
                ))
            tipo_detectado = "comida"
            mensaje = f"✅ Registré: {nombre} — {calorias} kcal."

        db.commit()
        return {
            "success": True,
            "tipo_detectado": tipo_detectado,
            "alimentos" if tipo_detectado == "comida" else "ejercicios": [nombre],
            "balance_actualizado": {
                "consumido": progreso.calorias_consumidas,
                "quemado": progreso.calorias_quemadas,
            },
            "datos": {
                "calorias": calorias, "proteinas_g": proteinas_g,
                "carbohidratos_g": carbohidratos_g, "grasas_g": grasas_g,
            } if tipo_detectado == "comida" else {
                "calorias_quemadas": calorias,
                "duracion_estimada": payload.get("duracion", 0)
            },
            "mensaje": mensaje.replace("*", ""),
        }

    def _fuzzy_match_comidas_recientes(self, mensaje, perfil):
        """Busca coincidencias de comidas recientes por similaridad."""
        msg_texto = mensaje.lower().strip()
        comidas_recientes = get_user_recent_meals(perfil.id)
        if not comidas_recientes:
            return None

        search_candidates = []
        for m in comidas_recientes:
            for ing in m.get("ingredientes", []):
                ing_match = re.search(r'de\s+(.*?)\s*\(', ing, re.IGNORECASE)
                if not ing_match:
                    ing_match = re.search(r'\*\s*\d+.*?\s+(.*?)\s*\(', ing)
                if ing_match:
                    nombre_ing = ing_match.group(1).strip().lower()
                    search_candidates.append((nombre_ing, m, ing))
            search_candidates.append((m["nombre"].lower(), m, None))

        # Limpiar palabras ruidosas
        msg_limpio = msg_texto
        for ruido in [
            "registra que me ", "registra que ", "registra me ", "registra ",
            "comí ", "comi ", "cómo ", "como ", "cené ", "almorcé ", "desayuné ",
            "cene ", "almorce ", "desayune ", "he comido ", "he cenado ",
            "un ", "una ", "unos ", "unas "
        ]:
            msg_limpio = msg_limpio.replace(ruido, "")
        msg_limpio = msg_limpio.strip()

        nombres_para_difflib = [c[0] for c in search_candidates]
        coincidencias = difflib.get_close_matches(msg_limpio, nombres_para_difflib, n=1, cutoff=0.75)

        if not coincidencias:
            return None

        match_nombre = coincidencias[0]
        for c_nombre, m_payload, item_str in search_candidates:
            if c_nombre == match_nombre:
                if item_str:
                    cals_match = re.search(r'\((\d+)\s*kcal', item_str)
                    p_match = re.search(r'P:\s*(\d+[,.]?\d*)g', item_str)
                    c_match = re.search(r'C:\s*(\d+[,.]?\d*)g', item_str)
                    g_match = re.search(r'G:\s*(\d+[,.]?\d*)g', item_str)
                    if cals_match:
                        try:
                            return {
                                "nombre": match_nombre.capitalize(),
                                "calorias": float(cals_match.group(1)),
                                "proteinas_g": float(p_match.group(1).replace(',', '.')) if p_match else 0,
                                "carbohidratos_g": float(c_match.group(1).replace(',', '.')) if c_match else 0,
                                "grasas_g": float(g_match.group(1).replace(',', '.')) if g_match else 0,
                            }
                        except:
                            pass
                return m_payload
        return None

    def _registrar_preferencias(self, extraccion, perfil, db):
        """Auto-aprendizaje: registra preferencias alimentarias y de ejercicio."""
        if extraccion.get("es_comida"):
            alimentos_raw = extraccion.get("alimentos_detectados", [])
            alimentos = [a.split('[')[0].strip() for a in alimentos_raw]
            extraccion["alimentos_detectados"] = alimentos

            for alimento in alimentos:
                pref_existente = db.query(PreferenciaAlimento).filter(
                    PreferenciaAlimento.client_id == perfil.id,
                    sql_func.lower(PreferenciaAlimento.alimento) == alimento.lower()
                ).first()

                if pref_existente:
                    pref_existente.frecuencia += 1
                    pref_existente.ultima_vez = datetime.now()
                    pref_existente.puntuacion = min(5.0, pref_existente.puntuacion + 0.1)
                else:
                    db.add(PreferenciaAlimento(
                        client_id=perfil.id, alimento=alimento.lower(),
                        frecuencia=1, puntuacion=1.0, ultima_vez=datetime.now()
                    ))

        elif extraccion.get("es_ejercicio"):
            ejercicios_detectados = extraccion.get("ejercicios_detectados", [])
            if not ejercicios_detectados:
                ejercicios_detectados = extraccion.get("alimentos_detectados", [])
            for ejercicio in ejercicios_detectados:
                pref_existente = db.query(PreferenciaEjercicio).filter(
                    PreferenciaEjercicio.client_id == perfil.id,
                    sql_func.lower(PreferenciaEjercicio.ejercicio) == ejercicio.lower()
                ).first()

                if pref_existente:
                    pref_existente.frecuencia += 1
                    pref_existente.ultima_vez = datetime.now()
                    pref_existente.puntuacion = min(5.0, pref_existente.puntuacion + 0.1)
                else:
                    db.add(PreferenciaEjercicio(
                        client_id=perfil.id, ejercicio=ejercicio.lower(),
                        frecuencia=1, puntuacion=1.0, ultima_vez=datetime.now()
                    ))


# Singleton
asistente_service = AsistenteService()
