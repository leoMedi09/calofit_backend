from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from app.core.database import get_db
from app.api.routes.auth import get_current_user
from app.models.nutricion import PlanNutricional, PlanDiario
from app.models.client import Client
from app.models.historial import ProgresoCalorias, AlertaSalud
from app.services.ia_service import ia_engine
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    mensaje: str
    historial: list = None # Opcional: [{"role": "user", "content": "..."}, ...]
    contexto_manual: str = None # 🛠️ Para pruebas en Postman (Sobrescribe datos de BD)
    override_ia: str = None # 🛠️ DEBUG: Envía una respuesta de IA manual para probar el parser

@router.post("/consultar")
async def consultar_asistente(
    request: ChatRequest, 
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    print(f"🤖 >>> INICIO CONSULTA ASISTENTE <<<")
    print(f"🤖 Usuario Token: {current_user.email} (ID: {current_user.id})")
    
    # 1. Obtener perfil del cliente autenticado
    perfil = db.query(Client).filter(Client.email.ilike(current_user.email)).first()
    
    if not perfil:
        print(f"❌ ERROR: Cliente no encontrado en tabla 'clients' para email: {current_user.email}")
        raise HTTPException(status_code=404, detail="Perfil de cliente no encontrado")

    print(f"✅ PERFIL CLIENTE: {perfil.first_name} {perfil.last_name_paternal} (ID: {perfil.id})")

    # 2. Obtener el plan semanal vigente o calcular fallback
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
        edad = (datetime.now().year - perfil.birth_date.year) if perfil.birth_date else 25
        
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
    
    # 4. Lógica difusa
    print(f"🧠 Calculando lógica difusa...")
    # ... resto del código ...

    # 4. 🧠 CALCULAR ADHERENCIA Y PROGRESO PARA LÓGICA DIFUSA
    # Obtener progreso de calorías de hoy
    hoy = datetime.now().date()
    progreso_hoy = db.query(ProgresoCalorias).filter(
        ProgresoCalorias.client_id == perfil.id,
        ProgresoCalorias.fecha == hoy
    ).first()
    
    # Calcular adherencia (qué tan cerca está de su meta calórica)
    if progreso_hoy and progreso_hoy.calorias_consumidas:
        calorias_objetivo = plan_hoy_data["calorias_dia"]
        calorias_consumidas = progreso_hoy.calorias_consumidas
        diferencia_pct = abs(calorias_consumidas - calorias_objetivo) / calorias_objetivo * 100
        # Adherencia: 100% si está perfecto, baja si se desvía mucho
        adherencia_pct = max(0, 100 - diferencia_pct)
    else:
        # Si no ha registrado nada, adherencia baja (asume que no está siguiendo el plan)
        adherencia_pct = 30
    
    # Calcular progreso (simulado: basado en cuántos días lleva con el plan)
    if usa_fallback:
        # Si es fallback, progreso inicial bajo
        progreso_pct = 40
    else:
        dias_con_plan = (datetime.now() - plan_maestro.fecha_creacion).days
        # Progreso simulado: mejora gradualmente con el tiempo
        progreso_pct = min(100, 50 + (dias_con_plan * 5))  # Empieza en 50%, sube 5% por día
    
    # 5. 🎯 APLICAR LÓGICA DIFUSA PARA PERSONALIZAR EL TONO
    mensaje_fuzzy = ia_engine.generar_alerta_fuzzy(adherencia_pct, progreso_pct)
    
    # Determinar tono basado en la salida fuzzy
    if "Excelente" in mensaje_fuzzy:
        tono_instruccion = "Usa un tono muy motivador y celebratorio. El cliente está haciendo un trabajo excelente."
    elif "mejorar" in mensaje_fuzzy:
        tono_instruccion = "Usa un tono alentador pero firme. El cliente necesita un pequeño empujón."
    else:
        tono_instruccion = "Usa un tono empático pero directo. El cliente necesita más compromiso, pero sé comprensivo."

    # 6. Detección Inteligente de Salud (Reactividad Dinámica)
    analisis_salud = ia_engine.identificar_intencion_salud(request.mensaje)
    
    # Validar que analisis_salud no sea None
    if analisis_salud is None:
        analisis_salud = {"tiene_alerta": False}
    
    if analisis_salud.get("tiene_alerta"):
        # Registrar alerta en la base de datos
        nueva_alerta = AlertaSalud(
            client_id=perfil.id,
            tipo=analisis_salud.get("tipo", "otro"),
            descripcion=analisis_salud.get("descripcion_resumida", request.mensaje),
            severidad=analisis_salud.get("severidad", "bajo"),
            estado="pendiente"
        )
        db.add(nueva_alerta)
        db.commit()
    
    # 7. Obtener nombres de especialistas asignados
    nombre_nutri = "tu nutricionista"
    if perfil.nutritionist:
        nombre_nutri = f"tu nutricionista {perfil.nutritionist.first_name}"
        
    # 8. 🚀 Construcción del Prompt con Personalización Total
    es_provisional = getattr(plan_maestro, 'estado', 'provisional_ia') == 'provisional_ia' or not getattr(plan_maestro, 'validado_nutri', False)
    
    # Calcular edad de forma precisa
    edad = (datetime.now().year - perfil.birth_date.year) if perfil.birth_date else 25
    
    # Formatear condiciones médicas separando alergias, dieta y condiciones
    alergias = []
    preferencias_dieta = []
    condiciones_medicas = []
    
    if perfil.medical_conditions:
        for cond in perfil.medical_conditions:
            cond_lower = cond.lower()
            # Detectar alergias
            if any(palabra in cond_lower for palabra in ["alérgico", "alergia", "intolerancia"]):
                alergias.append(cond)
            # Detectar preferencias dietéticas
            elif any(palabra in cond_lower for palabra in ["vegano", "vegetariano", "pescetariano", "halal", "kosher"]):
                preferencias_dieta.append(cond)
            # El resto son condiciones médicas
            else:
                condiciones_medicas.append(cond)
    
    # Formatear para el prompt
    texto_alergias = ", ".join(alergias) if alergias else "Ninguna"
    texto_dieta = ", ".join(preferencias_dieta) if preferencias_dieta else "Omnívoro"
    texto_condiciones = ", ".join(condiciones_medicas) if condiciones_medicas else "Ninguna"

    # Datos de consumo real
    consumo_real = progreso_hoy.calorias_consumidas if (progreso_hoy and progreso_hoy.calorias_consumidas) else 0.0
    quemadas_real = progreso_hoy.calorias_quemadas if (progreso_hoy and progreso_hoy.calorias_quemadas) else 0.0
    calorias_meta = plan_hoy_data['calorias_dia']
    restantes = max(0, calorias_meta - consumo_real + quemadas_real)

    # Inferir la meta REAL basándose en los macros (más confiable que el label 'goal')
    proteinas_g = plan_hoy_data['proteinas_g']
    if calorias_meta >= 2800 and proteinas_g >= 140:
        objetivo_real = "GANAR MASA MUSCULAR (Bulking)"
    elif calorias_meta <= 1800:
        objetivo_real = "PERDER PESO / DEFINIR (Cutting)"
    else:
        objetivo_real = "MANTENER PESO (Mantenimiento)"

    contexto_asistente = (
        f"Eres el coach experto personal de {perfil.first_name}. CONÓCELO A FONDO: "
        f"- Perfil: {perfil.weight}kg, {perfil.height}cm, {edad} años. "
        f"- Nivel de Actividad: {perfil.activity_level}. "
        f"- Objetivo REAL (inferido de macros): {objetivo_real}. "
        f"\n🚨 RESTRICCIONES ABSOLUTAS (CRÍTICO - FALLO = PENALIZACIÓN): "
        f"- PREFERENCIA DIETÉTICA: {texto_dieta} (SI ES VEGANO: PROHIBIDO sugerir carne, pollo, pescado, huevos, lácteos, miel). "
        f"- ALERGIAS: {texto_alergias} (PROHIBIDO usar estos ingredientes o derivados). "
        f"- CONDICIONES MÉDICAS: {texto_condiciones}. "
        f"\nSTATUS DEL DÍA DE HOY: "
        f"- Meta: {calorias_meta} kcal (P: {plan_hoy_data['proteinas_g']}g, C: {plan_hoy_data['carbohidratos_g']}g, G: {plan_hoy_data['grasas_g']}g). "
        f"- Consumidas: {consumo_real} kcal. "
        f"- Restantes: {restantes} kcal. "
        f"\n📊 Adherencia actual: {adherencia_pct:.0f}%, Progreso histórico: {progreso_pct:.0f}%. "
        f"{mensaje_fuzzy}. "
        f"\n⚡ REGLAS SEGURIDAD/PERSONALIZACIÓN: "
        f"1. NUNCA violar PREFERENCIAS DIETÉTICAS ({texto_dieta}) ni ALERGIAS ({texto_alergias}). "
        f"2. Si tiene CONDICIONES MÉDICAS ({texto_condiciones}), evita alimentos/ejercicios prohibidos. "
        f"3. PROACTIVIDAD LOCAL: Sugiere alimentos PERUANOS (Pollo a la Brasa, Lomo, Ceviche, Quinua, Camote) y ejercicios conocidos (Pichanga, Vóley). "
        f"4. Adapta porciones a su peso de {perfil.weight}kg y nivel {perfil.activity_level}. "
        f"5. Sé empático con su historial de {progreso_pct:.0f}% de progreso. "
    )
    
    if es_provisional or usa_fallback:
        contexto_asistente += (
            f"\n⚠️ NOTA MÉDICA: Su plan de {calorias_meta} kcal es PROVISIONAL (IA). "
            f"Recuérdale que el nutricionista lo validará pronto."
        )
    else:
        contexto_asistente += f"\n✅ Plan profesional ya validado por {nombre_nutri}."
    
    # 🛠️ DEBUG/TESTING: Sobrescribir contexto si viene manual (para Postman)
    if request.contexto_manual:
        print(f"🛠️ [DEBUG] Usando contexto manual desde Postman: {request.contexto_manual[:50]}...")
        contexto_asistente = request.contexto_manual

    # 9. Respuesta de la IA usando Groq con contexto adaptativo y memoria
    if request.override_ia:
        print(f"🛠️ [DEBUG] Modo OVERRIDE activado. Usando respuesta manual para probar el parser.")
        respuesta_ia = request.override_ia
    else:
        respuesta_ia = ia_engine.asistir_cliente(
            contexto=contexto_asistente, 
            mensaje_usuario=request.mensaje, 
            historial=request.historial,
            tono_aplicado=tono_instruccion
        )

    # 10. Parsear respuesta para Flutter (estructurada en secciones)
    from app.services.response_parser import parsear_respuesta_para_frontend
    respuesta_estructurada = parsear_respuesta_para_frontend(respuesta_ia, mensaje_usuario=request.mensaje)

    return {
        "asistente": "CaloFit IA",
        "usuario": perfil.first_name,
        "dia_seguimiento": datetime.now().isoweekday(),
        "usa_fallback_ia": usa_fallback,
        "alerta_salud": analisis_salud.get("tiene_alerta", False),
        "control_adaptativo": {
            "adherencia_pct": round(adherencia_pct, 1),
            "progreso_pct": round(progreso_pct, 1),
            "mensaje_fuzzy": mensaje_fuzzy,
            "tono_aplicado": tono_instruccion
        },
        "data_cientifica": {
            "calorias_calculadas": plan_hoy_data["calorias_dia"],
            "macros": {
                "P": plan_hoy_data["proteinas_g"], 
                "C": plan_hoy_data["carbohidratos_g"], 
                "G": plan_hoy_data["grasas_g"]
            },
            "progreso_diario": {
                "consumido": round(consumo_real, 1),
                "meta": round(calorias_meta, 1),
                "restante": round(restantes, 1),
                "quemado": round(quemadas_real, 1)
            },
            "fuente_calorica": "Modelo Regresión Gradient Boosting" if usa_fallback else ("Plan Provisional IA" if es_provisional else "Plan Nutricional Validado")
        },
        "respuesta_ia": respuesta_ia,  # Texto completo (legacy)
        "respuesta_estructurada": respuesta_estructurada  # 🆕 JSON estructurado para Flutter
    }

@router.post("/log-inteligente")
async def registro_inteligente_nlp(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Endpoint para registrar comida o ejercicio por voz/texto.
    Usa Groq para extraer macros y actualizar el progreso diario.
    
    🧠 SISTEMA DE APRENDIZAJE: Registra automáticamente preferencias del usuario.
    """
    perfil = db.query(Client).filter(Client.email == current_user.email).first()
    if not perfil:
        raise HTTPException(status_code=404, detail="Perfil de cliente no encontrado")
        
    # 1. Extraer macros con Groq
    extraccion = ia_engine.extraer_macros_de_texto(request.mensaje)
    
    if not extraccion or (extraccion.get("calorias", 0) == 0):
        return {
            "success": False,
            "mensaje": "No pude identificar alimentos o ejercicios en tu mensaje. ¿Podrías ser más específico?"
        }
        
    # 2. Actualizar ProgresoCalorias
    hoy = datetime.now().date()
    progreso = db.query(ProgresoCalorias).filter(
        ProgresoCalorias.client_id == perfil.id,
        ProgresoCalorias.fecha == hoy
    ).first()
    
    if not progreso:
        progreso = ProgresoCalorias(client_id=perfil.id, fecha=hoy)
        db.add(progreso)
        
    if extraccion.get("es_comida"):
        progreso.calorias_consumidas = (progreso.calorias_consumidas or 0) + extraccion.get("calorias", 0)
    elif extraccion.get("es_ejercicio"):
        progreso.calorias_quemadas = (progreso.calorias_quemadas or 0) + extraccion.get("calorias", 0)
    
    # 3. 🧠 AUTO-APRENDIZAJE: Registrar preferencias
    from app.models.preferencias import PreferenciaAlimento, PreferenciaEjercicio
    from sqlalchemy import func as sql_func
    
    if extraccion.get("es_comida"):
        # Registrar cada alimento detectado
        alimentos = extraccion.get("alimentos_detectados", [])
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
                    puntuacion=1.0
                )
                db.add(nueva_pref)
    
    elif extraccion.get("es_ejercicio"):
        # Similar para ejercicios
        ejercicios_detectados = extraccion.get("alimentos_detectados", [])  # Reutiliza el campo
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
                    puntuacion=1.0
                )
                db.add(nueva_pref)
        
    db.commit()
    
    tipo = "comida" if extraccion.get("es_comida") else "ejercicio"
    return {
        "success": True,
        "tipo_detectado": tipo,
        "alimentos": extraccion.get("alimentos_detectados"),
        "datos": {
            "calorias": extraccion.get("calorias"),
            "proteinas": extraccion.get("proteinas_g"),
            "carbos": extraccion.get("carbohidratos_g"),
            "grasas": extraccion.get("grasas_g")
        },
        "mensaje": f"He registrado tu {tipo} exitosamente. ¡Sigue así!"
    }

