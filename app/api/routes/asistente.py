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
        
        # Calcular macros dinámicos según objetivo
        if objetivo == "perder":
            pct_proteina, pct_carbohidratos, pct_grasas = 0.35, 0.30, 0.35
        elif objetivo == "ganar":
            pct_proteina, pct_carbohidratos, pct_grasas = 0.30, 0.45, 0.25
        else:
            pct_proteina, pct_carbohidratos, pct_grasas = 0.30, 0.40, 0.30
        
        proteinas_g = round((calorias_fallback * pct_proteina) / 4, 1)
        carbohidratos_g = round((calorias_fallback * pct_carbohidratos) / 4, 1)
        grasas_g = round((calorias_fallback * pct_grasas) / 9, 1)
        
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
        
    # 8. 🚀 Construcción del Prompt con Lógica Difusa Adaptativa
    contexto_asistente = (
        f"Eres CaloFit IA, asistente del Gimnasio World Light. "
        f"Usuario: {perfil.first_name}. Objetivo: {plan_maestro.objetivo}. "
        f"Tu meta calórica hoy: {plan_hoy_data['calorias_dia']} kcal (Proteínas: {plan_hoy_data['proteinas_g']}g). "
        f"\n📊 Tu adherencia: {adherencia_pct:.0f}%, Progreso: {progreso_pct:.0f}%. "
        f"{mensaje_fuzzy}. "
        f"\n🎯 Tono: {tono_instruccion} "
        f"\n\n⚡ REGLAS IMPORTANTES: "
        f"1. Responde en MÁXIMO 3-4 oraciones cortas. "
        f"2. Sé directo, claro y amigable. NO uses jerga técnica excesiva. "
        f"3. Ve al punto rápidamente. "
        f"4. Usa emojis solo cuando sea necesario (máximo 2). "
        f"5. Si el usuario saluda, responde brevemente y pregunta en qué puedes ayudar. "
    )
    
    if analisis_salud.get("tiene_alerta"):
        contexto_asistente += f" ¡ALERTA DE SALUD DETECTADA!: {analisis_salud.get('tipo')}. "
        contexto_asistente += f"Instrucción: Sé empático y sugiere: {analisis_salud.get('recomendacion_contingencia')}. "
    
    # Instrucciones específicas según el tipo de plan
    if usa_fallback:
        contexto_asistente += (
            f"PLAN CALCULADO POR IA (no validado por nutricionista aún). "
            f"Menciona brevemente que es temporal y sugiere consultar nutricionista. "
        )
    else:
        contexto_asistente += (
            f"Plan validado por {nombre_nutri}. Menciónalo solo si es relevante. "
        )
    
    # 9. Respuesta de la IA usando Groq con contexto adaptativo
    respuesta_ia = ia_engine.asistir_cliente(contexto_asistente, request.mensaje)

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
            "fuente_calorica": "Modelo Regresión Gradient Boosting" if usa_fallback else "Plan Nutricional Validado"
        },
        "respuesta_ia": respuesta_ia
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

