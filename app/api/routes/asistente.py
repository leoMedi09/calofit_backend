import uuid
import re
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import asyncio
from app.core.database import get_db
from app.api.routes.auth import get_current_user
from app.models.nutricion import PlanNutricional, PlanDiario
from app.models.client import Client
import difflib
import traceback
from pydantic import BaseModel
from sqlalchemy import func as sql_func
from app.models.historial import ProgresoCalorias, AlertaSalud
from app.models.preferencias import PreferenciaAlimento
from app.core.utils import parsear_macros_de_texto, get_peru_date
from app.services.ia_service import ia_engine
from app.core.cache import set_consulta_cached, get_consulta_cached, add_user_recent_meal, get_user_recent_meals

router = APIRouter()

class ChatRequest(BaseModel):
    mensaje: str
    historial: list = None # Opcional: [{"role": "user", "content": "..."}, ...]
    contexto_manual: str = None # 🛠️ Para pruebas en Postman (Sobrescribe datos de BD)
    override_ia: str = None # 🛠️ DEBUG: Envía una respuesta de IA manual para probar el parser
    consulta_id: str = None  # Si envías esto al registrar, se usan los MISMOS valores de la card (evita 112 vs 360)


class ConfirmarRegistroRequest(BaseModel):
    consulta_id: str

@router.post("/consultar")
async def consultar_asistente(
    request: ChatRequest, 
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    try:
        print(f"🤖 >>> INICIO CONSULTA ASISTENTE <<<")
        print(f"🤖 Usuario Token: {current_user.email} (ID: {current_user.id})")
        
        # 1. Obtener perfil del cliente autenticado
        perfil = db.query(Client).filter(Client.email.ilike(current_user.email)).first()
        
        if not perfil:
            print(f"❌ ERROR: Cliente no encontrado en tabla 'clients' para email: {current_user.email}")
            raise HTTPException(status_code=404, detail="Perfil de cliente no encontrado")

        print(f"✅ PERFIL CLIENTE: {perfil.first_name} {perfil.last_name_paternal} (ID: {perfil.id})")

        # 2. Calcular edad una sola vez al inicio
        edad = (datetime.now().year - perfil.birth_date.year) if perfil.birth_date else 25

        # 3. Obtener el plan semanal vigente o calcular fallback
        print(f"🔍 Buscando plan maestro para cliente ID: {perfil.id}...")
        plan_maestro = db.query(PlanNutricional).filter(
            PlanNutricional.client_id == perfil.id
        ).order_by(PlanNutricional.fecha_creacion.desc()).first()

        # 🆕 FALLBACK: Si no hay plan, calcular con IA
        usa_fallback = False
        plan_hoy_data = {}
        
        if not plan_maestro:
            print(f"⚠️ Plan Maestro no encontrado para cliente {perfil.id}. Usando fallback IA...")
            usa_fallback = True
            
            # Mapear datos del cliente
            genero_map = {"M": 1, "F": 2}
            genero = genero_map.get(perfil.gender, 1)
            
            nivel_map = {
                "Sedentario": 1.2,
                "Ligero": 1.375,
                "Moderado": 1.55,
                "Intenso": 1.725,
                "Muy intenso": 1.9
            }
            nivel_actividad = nivel_map.get(perfil.activity_level, 1.2)
            
            objetivo_map = {
                "Perder peso": "perder",
                "Mantener peso": "mantener",
                "Ganar masa": "ganar"
            }
            objetivo = objetivo_map.get(perfil.goal, "mantener")
            
            # Calcular calorías con el modelo ML
            calorias_fallback = ia_engine.calcular_requerimiento(
                genero=genero,
                edad=edad,
                peso=perfil.weight,
                talla=perfil.height,
                nivel_actividad=nivel_actividad,
                objetivo=objetivo
            )
            
            # Calcular macros usando la lógica centralizada de la IA
            condiciones_medicas = ", ".join(perfil.medical_conditions) if perfil.medical_conditions else ""
            macros_data = ia_engine.calcular_macros_optimizados(
                peso=perfil.weight,
                objetivo=objetivo,
                calorias=calorias_fallback,
                condiciones_medicas=condiciones_medicas
            )
            
            proteinas_g = macros_data['proteinas_g']
            carbohidratos_g = macros_data['carbohidratos_g']
            grasas_g = macros_data['grasas_g']
            
            # Crear objeto de datos simulado
            plan_hoy_data = {
                "calorias_dia": calorias_fallback,
                "proteinas_g": proteinas_g,
                "carbohidratos_g": carbohidratos_g,
                "grasas_g": grasas_g,
                "sugerencia_entrenamiento_ia": "Plan calculado automáticamente por IA"
            }
            
            # Simulamos un objeto de plan
            class PlanFallback:
                def __init__(self, objetivo):
                    self.objetivo = objetivo
                    self.status = "calculado_ia"
                    self.id = None
                    self.fecha_creacion = datetime.now()  # Fecha actual como creación
            
            plan_maestro = PlanFallback(objetivo=perfil.goal)
            
            print(f"✅ FALLBACK IA: {calorias_fallback:.0f} kcal | P:{proteinas_g}g C:{carbohidratos_g}g G:{grasas_g}g")
        else:
            print(f"✅ PLAN MAESTRO: ID {plan_maestro.id} (Status: {plan_maestro.status})")
            
            # 3. Obtener el detalle del día actual
            dia_semana = datetime.now().isoweekday() 
            print(f"🔍 Buscando plan diario para día {dia_semana}...")
            plan_hoy = db.query(PlanDiario).filter(
                PlanDiario.plan_id == plan_maestro.id,
                PlanDiario.dia_numero == dia_semana
            ).first()

            if not plan_hoy:
                print(f"⚠️ Plan diario no encontrado para hoy (día {dia_semana}). Buscando primer día disponible...")
                plan_hoy = db.query(PlanDiario).filter(PlanDiario.plan_id == plan_maestro.id).first()
                
            if not plan_hoy:
                print(f"❌ ERROR FATAL: El plan maestro {plan_maestro.id} no tiene detalles diarios.")
                raise HTTPException(status_code=404, detail="Tu plan nutricional está incompleto.")

            print(f"✅ PLAN HOY: ID {plan_hoy.id} ({plan_hoy.calorias_dia} kcal)")
            
            # Extraer datos del plan
            plan_hoy_data = {
                "calorias_dia": plan_hoy.calorias_dia,
                "proteinas_g": plan_hoy.proteinas_g,
                "carbohidratos_g": plan_hoy.carbohidratos_g,
                "grasas_g": plan_hoy.grasas_g,
                "sugerencia_entrenamiento_ia": plan_hoy.sugerencia_entrenamiento_ia
            }
        
        # 4. 🧠 CALCULAR ADHERENCIA Y PROGRESO PARA LÓGICA DIFUSA
        hoy = get_peru_date()
        progreso_hoy = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == perfil.id,
            ProgresoCalorias.fecha == hoy
        ).first()

        consumo_real = progreso_hoy.calorias_consumidas if progreso_hoy else 0
        quemadas_real = progreso_hoy.calorias_quemadas if progreso_hoy else 0
        
        if plan_hoy and plan_hoy.calorias_dia > 0:
            progreso_pct = min(100, (consumo_real / plan_hoy.calorias_dia) * 100)
            adherencia_pct = 85
        else:
            progreso_pct = 50
            adherencia_pct = 85
        
        # 5. 🎯 APLICAR LÓGICA DIFUSA PARA PERSONALIZAR EL TONO
        alerta_fuzzy = ia_engine.generar_alerta_fuzzy(adherencia_pct, progreso_pct)
        mensaje_fuzzy = alerta_fuzzy.get("mensaje", "")
        
        if "Excelente" in mensaje_fuzzy:
            tono_instruccion = "Usa un tono muy motivador y celebratorio."
        elif "mejorar" in mensaje_fuzzy:
            tono_instruccion = "Usa un tono alentador pero firme."
        else:
            tono_instruccion = "Usa un tono empático pero directo."

        # 6. Detección Inteligente de Salud (Fire-and-Forget, NO bloqueante)
        msg_limpio = request.mensaje.lower().strip()
        es_saludo = len(msg_limpio) < 20 and any(sal in msg_limpio for sal in ["hola", "buen", "hey", "salu", "que tal", "qué tal", "gracias"])
        
        async def _analizar_salud_background():
            try:
                resultado = await ia_engine.identificar_intencion_salud(request.mensaje)
                if resultado and resultado.get("tiene_alerta"):
                    nueva_alerta = AlertaSalud(
                        client_id=perfil.id,
                        tipo=resultado.get("tipo", "otro"),
                        descripcion=resultado.get("descripcion_resumida", request.mensaje),
                        severidad=resultado.get("severidad", "bajo"),
                        estado="pendiente"
                    )
                    db.add(nueva_alerta)
                    db.commit()
                    print(f"🚨 Alerta de salud guardada en background: {resultado.get('tipo')}")
            except Exception as e:
                print(f"⚠️ Error en análisis de salud background: {e}")
        
        if not es_saludo:
            asyncio.create_task(_analizar_salud_background())

        # 7. Obtener especialistas
        nombre_nutri = "tu nutricionista"
        if perfil.nutritionist:
            nombre_nutri = f"tu nutricionista {perfil.nutritionist.first_name}"
            
        # 8. 🚀 Construcción del Prompt con Identidad Completa
        es_provisional = getattr(plan_maestro, 'status', 'provisional_ia') == 'provisional_ia' or not getattr(plan_maestro, 'validado_nutri', False)
        
        # Formatear condiciones médicas
        alergias = []
        preferencias_dieta = []
        condiciones_medicas = []
        
        if perfil.medical_conditions:
            for cond in perfil.medical_conditions:
                cond_l = cond.lower()
                if any(p in cond_l for p in ["alérgico", "alergia", "intolerancia"]): alergias.append(cond)
                elif any(p in cond_l for p in ["vegano", "vegetariano", "pescetariano"]): preferencias_dieta.append(cond)
                else: condiciones_medicas.append(cond)
        
        texto_alergias = ", ".join(alergias) if alergias else "Ninguna"
        texto_dieta = ", ".join(preferencias_dieta) if preferencias_dieta else "Omnívoro"
        texto_condiciones = ", ".join(condiciones_medicas) if condiciones_medicas else "Ninguna"

        # Variables already defined above as 0 if bypassed or queried if successful
        calorias_meta = plan_hoy_data['calorias_dia']
        restantes = max(0, calorias_meta - consumo_real + quemadas_real)

        contexto_asistente = (
            f"Eres el coach de {perfil.first_name}. "
            f"PERFIL: {perfil.weight}kg, {perfil.height}cm, {edad} años. "
            f"ALERGIAS: {texto_alergias}. "
            f"PREFERENCIAS DIETÉTICAS: {texto_dieta}. "
            f"CONDICIONES MÉDICAS: {texto_condiciones}. "
            f"\nSTATUS DEL DÍA: "
            f"Meta: {calorias_meta} kcal. Consumido: {consumo_real} kcal. Restante: {restantes} kcal. "
            f"Adherencia: {adherencia_pct:.0f}%, Progreso: {progreso_pct:.0f}%. "
            f"{mensaje_fuzzy}."
        )
        
        # 9. Respuesta de la IA
        if request.override_ia:
            respuesta_ia = request.override_ia
        else:
            respuesta_ia = await ia_engine.asistir_cliente(
                contexto=contexto_asistente, 
                mensaje_usuario=request.mensaje, 
                historial=request.historial,
                tono_aplicado=tono_instruccion
            )

        # 10. Parsear respuesta para Frontend
        from app.services.response_parser import parsear_respuesta_para_frontend
        respuesta_estructurada = parsear_respuesta_para_frontend(respuesta_ia, mensaje_usuario=request.mensaje)

        # 11. Flujo consistencia
        for seccion in respuesta_estructurada.get("secciones", []):
            if seccion.get("tipo") != "comida":
                continue
            
            # v71.4: NO descartar la sección si falla el parsing de macros.
            # Simplemente asignarle valores por defecto si no se pudo parsear.
            macros_parsed = parsear_macros_de_texto(seccion.get("macros") or "")
            if not macros_parsed:
                macros_parsed = {"calorias": 0, "proteinas_g": 0, "carbohidratos_g": 0, "grasas_g": 0}
            

            nombre_bruto = seccion.get("nombre") or "Comida"
            # v70.4: Limpieza agresiva de tags para evitar "[/CALOFIT_HEADE"
            nombre_limpio = re.sub(r'\[.*?\]', '', nombre_bruto).split('[')[0].strip()
            
            consulta_id = str(uuid.uuid4())
            payload = {
                "calorias": round(macros_parsed["calorias"], 1),
                "proteinas_g": round(macros_parsed["proteinas_g"], 1),
                "carbohidratos_g": round(macros_parsed["carbohidratos_g"], 1),
                "grasas_g": round(macros_parsed["grasas_g"], 1),
                "nombre": nombre_limpio,
                "ingredientes": seccion.get("ingredientes", [])
            }
            set_consulta_cached(consulta_id, payload)
            seccion["consulta_id"] = consulta_id
            
            # Guardar para fuzzy matching
            add_user_recent_meal(perfil.id, payload)

        # v71.5: PROCESAMIENTO INTELIGENTE - DISEÑOS POR INTENCIÓN
        # Restaurado el comportamiento dual:
        # - Recetas / Opciones múltiples -> Tarjetas Interactivas (RecipeCard)
        # - Consulta de macro directa (1 solo alimento) -> Texto limpio en chat
        msg_low = request.mensaje.lower()
        secciones_comida = [s for s in respuesta_estructurada.get("secciones", []) if s.get("tipo") == "comida"]
        
        es_info_directa = (
            len(secciones_comida) == 1 and
            not any(k in msg_low for k in ["opcion", "opciones", "receta", "menú", "menu", "cena", "almuerzo", "desayuno", "suger", "dame", "recomienda"])
        )
        
        print(f"🔍 v71.5: es_info_directa={es_info_directa}, secciones_comida={len(secciones_comida)}")
        secciones_conservar = []
        for seccion in respuesta_estructurada.get("secciones", []):
            tipo = seccion.get("tipo")
            
            if tipo == "comida":
                tiene_pasos = bool(seccion.get("pasos") or seccion.get("preparacion"))
                
                # CASO 1: Tiene pasos de preparación => RECETA COMPLETA => Tarjeta
                if tiene_pasos:
                    secciones_conservar.append(seccion)
                    continue

                # CASO 2: Son múltiples opciones o el usuario pidió opciones => Tarjeta
                if not es_info_directa:
                    secciones_conservar.append(seccion)
                    continue
                
                # CASO 3: Es consulta de info directa (ej. "¿qué tiene la manzana?") => TEXTO PLANO
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
            
            else:
                secciones_conservar.append(seccion)
                
        respuesta_estructurada["secciones"] = secciones_conservar
        print(f"✅ v71.5: Secciones a enviar al celular = {len(secciones_conservar)}")

        
        # v73.0: Limpieza de tags CALOFIT del texto conversacional
        final_text = respuesta_estructurada.get("texto_conversacional", "")
        # 🛡️ LIMPIEZA TOTAL: Eliminar cualquier residuo de [CALOFIT...]
        cleaned_text = re.sub(r'\[/?CALOFIT_[A-Z_:]*.*?\]', '', final_text, flags=re.IGNORECASE).strip()
        respuesta_estructurada["texto_conversacional"] = cleaned_text
        
        # 🛡️ LIMPIEZA AGRESIVA DE SECCIONES antes de enviar
        for s in secciones_conservar:
            for k_str in ["nombre", "macros", "gasto_calorico_estimado", "nota"]:
                if s.get(k_str):
                    s[k_str] = re.sub(r'\[/?CALOFIT_[A-Z_:]*.*?\]', '', str(s[k_str]), flags=re.IGNORECASE).strip()
            
            for k_list in ["ingredientes", "ejercicios", "preparacion", "tecnica", "instrucciones"]:
                if s.get(k_list) and isinstance(s[k_list], list):
                    s[k_list] = [re.sub(r'\[/?CALOFIT_[A-Z_:]*.*?\]', '', str(item), flags=re.IGNORECASE).strip() for item in s[k_list]]
        
        # v70.8: DETECCIÓN DE INTENCIÓN PRINCIPAL para el Front
        # v70.9: Fix NameError definined seccion_peligro
        alertas_activas = [s for s in secciones_conservar if s.get("tipo") == "alerta"]
        seccion_peligro = len(alertas_activas) > 0
        
        intencion_principal = "INFO"
        msg_low = request.mensaje.lower()
        
        # Prioridad 1: DANGER (Alertas)
        if seccion_peligro:
            intencion_principal = "DANGER"
        # Prioridad 2: SUCCESS (Si el texto indica confirmación de base de datos)
        elif "anotado" in respuesta_estructurada.get("texto_conversacional", "").lower() or "registrado" in respuesta_estructurada.get("texto_conversacional", "").lower():
            intencion_principal = "SUCCESS"
        # Prioridad 3: RECIPE/POWER según secciones
        else:
            tipos_en_secciones = [s.get("tipo") for s in respuesta_estructurada.get("secciones", [])]
            if "ejercicio" in tipos_en_secciones:
                intencion_principal = "POWER"
            elif any(s.get("pasos") or s.get("preparacion") for s in respuesta_estructurada.get("secciones", []) if s.get("tipo") == "comida"):
                intencion_principal = "RECIPE"
            elif any(k in msg_low for k in ["cuántas", "cuantas", "qué tiene", "que tiene", "qué es", "que es"]):
                intencion_principal = "INFO"

        return {
            "asistente": "CaloFit IA",
            "usuario": perfil.first_name,
            "intencion": intencion_principal,
            "alerta_salud": bool(seccion_peligro),
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
    except Exception as e:
        print(f"❌ ERROR EN /consultar: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/log-inteligente")
async def registro_inteligente_nlp(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Endpoint para registrar comida o ejercicio por voz/texto.
    Si envías consulta_id (de la card que mostró el chat), se registran los MISMOS valores
    que se mostraron (ej. 112 kcal), sin recalcular. Así se evita inconsistencia 112 vs 360.
    """
    perfil = db.query(Client).filter(Client.email == current_user.email).first()
    if not perfil:
        raise HTTPException(status_code=404, detail="Perfil de cliente no encontrado")

    # CONSISTENCIA: Si el front envía consulta_id (ej. al pulsar AÑADIR en una card), usar esos valores
    if request.consulta_id and request.consulta_id.strip():
        payload = get_consulta_cached(request.consulta_id.strip())
        if payload:
            hoy = get_peru_date()
            progreso = db.query(ProgresoCalorias).filter(
                ProgresoCalorias.client_id == perfil.id, ProgresoCalorias.fecha == hoy
            ).first()
            if not progreso:
                progreso = ProgresoCalorias(client_id=perfil.id, fecha=hoy)
                db.add(progreso)
            
            calorias = payload.get("calorias", 0) or 0
            proteinas_g = payload.get("proteinas_g", 0) or 0
            carbohidratos_g = payload.get("carbohidratos_g", 0) or 0
            grasas_g = payload.get("grasas_g", 0) or 0
            
            progreso.calorias_consumidas = (progreso.calorias_consumidas or 0) + calorias
            progreso.proteinas_consumidas = (progreso.proteinas_consumidas or 0) + proteinas_g
            progreso.carbohidratos_consumidos = (progreso.carbohidratos_consumidos or 0) + carbohidratos_g
            progreso.grasas_consumidas = (progreso.grasas_consumidas or 0) + grasas_g
            nombre_plato = (payload.get("nombre") or "Comida").strip()
            if nombre_plato:
                pref = db.query(PreferenciaAlimento).filter(
                    PreferenciaAlimento.client_id == perfil.id,
                    sql_func.lower(PreferenciaAlimento.alimento) == nombre_plato.lower(),
                ).first()
                if pref:
                    pref.frecuencia += 1
                    pref.ultima_vez = datetime.now()
                else:
                    db.add(PreferenciaAlimento(
                        client_id=perfil.id,
                        alimento=nombre_plato.lower(),
                        frecuencia=1,
                        puntuacion=1.0,
                        ultima_vez=datetime.now(),
                    ))
            db.commit()
            return {
                "success": True,
                "tipo_detectado": "comida",
                "alimentos": [nombre_plato],
                "balance_actualizado": {
                    "consumido": progreso.calorias_consumidas,
                    "quemado": progreso.calorias_quemadas,
                },
                "datos": {
                    "calorias": calorias,
                    "proteinas_g": proteinas_g,
                    "carbohidratos_g": carbohidratos_g,
                    "grasas_g": grasas_g,
                    "calidad": "Alta",
                },
                "mensaje": f"✅ Registré: {nombre_plato} — {calorias} kcal (mismos valores de la card).".replace("*", ""),
            }
        # Si consulta_id expiró o no existe, seguimos con el flujo normal (Groq)

    # 0.5 FALLBACK DE CONSISTENCIA: Fuzzy matching contra comidas recientes
    # Si el usuario mandó texto (ej. "Comí Ensalada de Quinoa con Pepitas de Chía"), 
    # revisamos si se parece mucho a algo que le sugerimos recientemente.
    msg_texto = request.mensaje.lower().strip()
    comidas_recientes = get_user_recent_meals(perfil.id)
    match_exacto = None
    
    if comidas_recientes:
        # v70.2: Búsqueda expandida (Título + Ítems Individuales)
        search_candidates = [] # [(nombre_para_buscar, payload_original, item_string_or_none)]
        
        for m in comidas_recientes:
            # v70.5: PRIORIZAR ÍTEMS individuales sobre el título total
            for ing in m.get("ingredientes", []):
                # Limpiar nombre del ingrediente: "1 unidad de manzana (68 kcal...)" -> "manzana"
                ing_match = re.search(r'de\s+(.*?)\s*\(', ing, re.IGNORECASE)
                if not ing_match:
                    ing_match = re.search(r'\*\s*\d+.*?\s+(.*?)\s*\(', ing)
                
                if ing_match:
                    nombre_ing = ing_match.group(1).strip().lower()
                    search_candidates.append((nombre_ing, m, ing))
            
            # Título al final como fallback
            search_candidates.append((m["nombre"].lower(), m, None))
        
        # Palabras ruidosas
        msg_limpio = msg_texto.replace("registra que me ", "").replace("registra que ", "").replace("registra me ", "").replace("registra ", "")
        msg_limpio = msg_limpio.replace("comí ", "").replace("comi ", "").replace("cómo ", "").replace("como ", "").replace("cené ", "").replace("almorcé ", "").replace("desayuné ", "")
        msg_limpio = msg_limpio.replace("cene ", "").replace("almorce ", "").replace("desayune ", "").replace("he comido ", "").replace("he cenado ", "")
        msg_limpio = msg_limpio.replace("un ", "").replace("una ", "").replace("unos ", "").replace("unas ", "").strip()
        
        nombres_para_difflib = [c[0] for c in search_candidates]
        print(f"DEBUG Fuzzy: Buscando '{msg_limpio}' en {len(nombres_para_difflib)} candidatos")
        
        coincidencias = difflib.get_close_matches(msg_limpio, nombres_para_difflib, n=1, cutoff=0.75)
        print(f"DEBUG Fuzzy coincidencias: {coincidencias}")
        
        if coincidencias:
            match_nombre = coincidencias[0]
            # Encontrar el candidato ganador
            for c_nombre, m_payload, item_str in search_candidates:
                if c_nombre == match_nombre:
                    # SI ES UN ITEM INDIVIDUAL, EXTRAER SUS MACROS ESPECÍFICOS (v70.2)
                    if item_str:
                        # Formato esperado: (68 kcal | P: 0,3g | C: 17g | G: 0,2g)
                        cals_match = re.search(r'\((\d+)\s*kcal', item_str)
                        p_match = re.search(r'P:\s*(\d+[,.]?\d*)g', item_str)
                        c_match = re.search(r'C:\s*(\d+[,.]?\d*)g', item_str)
                        g_match = re.search(r'G:\s*(\d+[,.]?\d*)g', item_str)
                        
                        if cals_match:
                            try:
                                match_exacto = {
                                    "nombre": match_nombre.capitalize(),
                                    "calorias": float(cals_match.group(1)),
                                    "proteinas_g": float(p_match.group(1).replace(',', '.')) if p_match else 0,
                                    "carbohidratos_g": float(c_match.group(1).replace(',', '.')) if c_match else 0,
                                    "grasas_g": float(g_match.group(1).replace(',', '.')) if g_match else 0,
                                }
                                break
                            except: pass
                    
                    # Si no es item o falló la extracción, usar el payload del card (total)
                    match_exacto = m_payload
                    break
    
    # 1. Extraer macros
    peso_usuario = perfil.weight if (perfil.weight and perfil.weight > 0) else 70.0
    extraccion = None
    
    if match_exacto:
        print(f"🎯 MATCH EXACTO ENCONTRADO (Fuzzy): '{request.mensaje}' -> '{match_exacto['nombre']}'")
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
        # Extraer con Groq (await) - Pasando peso del usuario para fórmula METs
        extraccion = await ia_engine.extraer_macros_de_texto(request.mensaje, peso_usuario_kg=peso_usuario)
        
    if not extraccion or (extraccion.get("calorias", 0) == 0):
        return {
            "success": False,
            "mensaje": "No pude identificar alimentos o ejercicios en tu mensaje. ¿Podrías ser más específico?"
        }
        
    # 2. Actualizar ProgresoCalorias
    from app.models.historial import ProgresoCalorias
    from app.core.utils import get_peru_date
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
    # 3. 🧠 AUTO-APRENDIZAJE: Registrar preferencias
    from app.models.preferencias import PreferenciaAlimento, PreferenciaEjercicio
    from sqlalchemy import func as sql_func
    
    if extraccion.get("es_comida"):
        # Registrar cada alimento detectado + Limpieza de tags AI (v70.2)
        alimentos_raw = extraccion.get("alimentos_detectados", [])
        alimentos = [a.split('[')[0].strip() for a in alimentos_raw]
        extraccion["alimentos_detectados"] = alimentos # Actualizar para el mensaje final
        
        for alimento in alimentos:
            # Buscar si ya existe preferencia
            pref_existente = db.query(PreferenciaAlimento).filter(
                PreferenciaAlimento.client_id == perfil.id,
                sql_func.lower(PreferenciaAlimento.alimento) == alimento.lower()
            ).first()
            
            if pref_existente:
                # Incrementar frecuencia
                pref_existente.frecuencia += 1
                pref_existente.ultima_vez = datetime.now()
                # Aumentar puntuación ligeramente
                pref_existente.puntuacion = min(5.0, pref_existente.puntuacion + 0.1)
            else:
                # Crear nueva preferencia
                nueva_pref = PreferenciaAlimento(
                    client_id=perfil.id,
                    alimento=alimento.lower(),
                    frecuencia=1,
                    puntuacion=1.0,
                    ultima_vez=datetime.now()
                )
                db.add(nueva_pref)
    
    elif extraccion.get("es_ejercicio"):
        # Registrar cada ejercicio detectado
        ejercicios_detectados = extraccion.get("ejercicios_detectados", [])
        if not ejercicios_detectados:
            # Fallback a alimentos_detectados si la IA se confundió
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
                nueva_pref = PreferenciaEjercicio(
                    client_id=perfil.id,
                    ejercicio=ejercicio.lower(),
                    frecuencia=1,
                    puntuacion=1.0,
                    ultima_vez=datetime.now()
                )
                db.add(nueva_pref)
        
    db.commit()
    
    tipo = "comida" if extraccion.get("es_comida") else "ejercicio"
    nombre_item = extraccion.get("alimentos_detectados", extraccion.get("ejercicios_detectados", []))
    nombre_str = ", ".join(nombre_item) if nombre_item else "tu registro"

    return {
        "success": True,
        "tipo_detectado": tipo,
        "alimentos": extraccion.get("alimentos_detectados", []),
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
        "mensaje": f"✅ Registré: {nombre_str} — {extraccion.get('calorias', 0)} kcal. ¡Buen trabajo!".replace("*", "")
    }


@router.post("/confirmar-registro")
async def confirmar_registro_con_consulta_id(
    body: ConfirmarRegistroRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """
    Registra una comida usando los valores exactos mostrados en el chat (consulta_id).
    Evita inconsistencia: mismo valor que vio el usuario, sin recalcular.
    El front debe enviar el consulta_id que recibió en respuesta_estructurada.secciones[].consulta_id.
    """
    perfil = db.query(Client).filter(Client.email == current_user.email).first()
    if not perfil:
        raise HTTPException(status_code=404, detail="Perfil de cliente no encontrado")

    payload = get_consulta_cached(body.consulta_id)
    if not payload:
        raise HTTPException(
            status_code=404,
            detail="Registro expirado o no encontrado. El consulta_id solo es válido 10 minutos. Registra de nuevo desde el chat.",
        )

    # 2. Actualizar ProgresoCalorias
    from app.models.historial import ProgresoCalorias
    from app.core.utils import get_peru_date
    hoy = get_peru_date()
    progreso = db.query(ProgresoCalorias).filter(
        ProgresoCalorias.client_id == perfil.id, ProgresoCalorias.fecha == hoy
    ).first()
    
    if not progreso:
        progreso = ProgresoCalorias(client_id=perfil.id, fecha=hoy)
        db.add(progreso)
        
    calorias = payload.get("calorias", 0) or 0
    proteinas_g = payload.get("proteinas_g", 0) or 0
    carbohidratos_g = payload.get("carbohidratos_g", 0) or 0
    grasas_g = payload.get("grasas_g", 0) or 0
    
    progreso.calorias_consumidas = (progreso.calorias_consumidas or 0) + calorias
    progreso.proteinas_consumidas = (progreso.proteinas_consumidas or 0) + proteinas_g
    progreso.carbohidratos_consumidos = (progreso.carbohidratos_consumidos or 0) + carbohidratos_g
    progreso.grasas_consumidas = (progreso.grasas_consumidas or 0) + grasas_g
    
    progreso_consumidas = progreso.calorias_consumidas
    progreso_proteinas = progreso.proteinas_consumidas
    progreso_carbs = progreso.carbohidratos_consumidos
    progreso_grasas = progreso.grasas_consumidas
    
    # Aprendizaje: preferencia del alimento
    nombre_plato = (payload.get("nombre") or "Comida").strip()
    if nombre_plato:
        from app.models.preferencias import PreferenciaAlimento
        from sqlalchemy import func as sql_func
        pref = db.query(PreferenciaAlimento).filter(
            PreferenciaAlimento.client_id == perfil.id,
            sql_func.lower(PreferenciaAlimento.alimento) == nombre_plato.lower(),
        ).first()
        if pref:
            pref.frecuencia += 1
            pref.ultima_vez = datetime.now()
        else:
            db.add(PreferenciaAlimento(
                client_id=perfil.id,
                alimento=nombre_plato.lower(),
                frecuencia=1,
                puntuacion=1.0,
                ultima_vez=datetime.now(),
            ))
    db.commit()

    calorias = payload.get("calorias", 0) or 0
    return {
        "success": True,
        "mensaje": f"Registrado: {nombre_plato} — {calorias} kcal (mismos valores del chat).",
        "balance_actualizado": {
            "consumido": progreso_consumidas,
            "proteinas_g": progreso_proteinas,
            "carbohidratos_g": progreso_carbs,
            "grasas_g": progreso_grasas,
        },
    }

