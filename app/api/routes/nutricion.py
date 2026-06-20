from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
# Asegúrate de importar PlanDiario
from app.models.nutricion import PlanNutricional, PlanDiario
from app.schemas.nutricion import PlanNutricionalCreate, PlanNutricionalResponse, TestIARequest
from typing import List, Optional, Any, Dict

from app.api.routes.auth import get_current_staff, get_current_user
from app.services.ia_service import ia_engine 

router = APIRouter()

# Endpoint temporal para probar IA (Solo para testing/desarrollo)
@router.post("/test-ia")
async def test_ia(
    request: TestIARequest,
    current_user = Depends(get_current_staff)
):
    """
    Endpoint de prueba para verificar el modelo de IA.
    🔒 REQUIERE AUTH STAFF: Solo personal autorizado puede probar.
    """
    print(f"🛠️ Testing IA por: {current_user.email}")
    ia_engine = get_ia_engine()
    if not ia_engine:
        raise HTTPException(status_code=500, detail="Servicio de IA no disponible")

    try:
        calorias = ia_engine.calcular_requerimiento(
            genero=request.genero, edad=request.edad, peso=request.peso, talla=request.talla,
            nivel_actividad=request.nivel_actividad, objetivo=request.objetivo
        )
        return {"calorias_recomendadas": calorias, "mensaje": "Prueba exitosa"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en IA: {str(e)}")

@router.post("/", response_model=PlanNutricionalResponse)
async def crear_plan_nutricional(
    plan_data: PlanNutricionalCreate, 
    db: Session = Depends(get_db),
    current_user = Depends(get_current_staff)
):
    # 1. Seguridad
    if current_user.role_name not in ["nutritionist", "admin"]:
        raise HTTPException(status_code=403, detail="No autorizado")

    # 2. IA: Cálculo de Calorías Base
    try:
        calorias_base = ia_engine.calcular_requerimiento(
            genero=plan_data.genero, edad=plan_data.edad,
            peso=plan_data.peso, talla=plan_data.talla,
            nivel_actividad=plan_data.nivel_actividad, objetivo=plan_data.objetivo
        )
    except Exception as e:
        print(f"Error en calcular_requerimiento: {str(e)}")
        # Fallback: Usar fórmula de Harris-Benedict
        if plan_data.genero == 1:  # Masculino
            tmb = 88.362 + (13.397 * plan_data.peso) + (4.799 * plan_data.talla) - (5.677 * plan_data.edad)
        else:  # Femenino
            tmb = 447.593 + (9.247 * plan_data.peso) + (3.098 * plan_data.talla) - (4.330 * plan_data.edad)
        
        calorias_mantenimiento = tmb * plan_data.nivel_actividad
        
        if plan_data.objetivo == "ganar":
            calorias_base = calorias_mantenimiento + 500
        elif plan_data.objetivo == "perder":
            calorias_base = calorias_mantenimiento - 500
        else:
            calorias_base = calorias_mantenimiento
        
        calorias_base = round(calorias_base, 2)
        print(f"Usando cálculo alternativo: {calorias_base} kcal")

    # 3. IA Avanzada: Recomendaciones con Groq + CBF
    perfil_usuario = {
        "edad": plan_data.edad,
        "genero": plan_data.genero,
        "peso": plan_data.peso,
        "talla": plan_data.talla,
        "objetivo": plan_data.objetivo,
        "nivel_actividad": plan_data.nivel_actividad
    }
    try:
        recomendacion_groq = ia_engine.recomendar_alimentos_con_groq(perfil_usuario)
    except Exception as e:
        recomendacion_groq = "Error al generar recomendación avanzada. Usa plan básico."

    # 4. Guardar Plan Maestro (Encabezado)
    nuevo_plan = PlanNutricional(
        client_id=plan_data.client_id,
        nutricionista_id=current_user.id,
        genero=plan_data.genero, edad=plan_data.edad,
        peso=plan_data.peso, talla=plan_data.talla,
        nivel_actividad=plan_data.nivel_actividad,
        objetivo=plan_data.objetivo,
        calorias_ia_base=calorias_base,
        es_contingencia_ia=False, # Plan oficial creado en consulta
        observaciones=plan_data.observaciones
    )

    try:
        db.add(nuevo_plan)
        db.flush() 

        # 4. Generación Semanal Inteligente
        for i in range(1, 8):
            # Diferenciamos carga: Días 1-5 (Entreno) vs 6-7 (Descanso)
            factor = 1.1 if i <= 5 else 0.9
            cals_dia = round(calorias_base * factor, 2)
            
            # IA genera consejos para el Coach y el Cliente
            sugerencia_entreno = ia_engine.generar_sugerencia_entrenamiento(plan_data.objetivo, i)
            nota_ia = "Plan generado automáticamente para dar continuidad a tu progreso."

            dia = PlanDiario(
                plan_id=nuevo_plan.id,
                dia_numero=i,
                calorias_dia=cals_dia,
                # Repartición de macros basada en calorías del día
                proteinas_g=round((cals_dia * 0.25) / 4, 1),
                carbohidratos_g=round((cals_dia * 0.50) / 4, 1),
                grasas_g=round((cals_dia * 0.25) / 9, 1),
                # Campos de asistencia
                sugerencia_entrenamiento_ia=sugerencia_entreno,
                nota_asistente_ia=nota_ia,
                estado="sugerencia_ia", # Disponible para el cliente al instante
                validado_nutri=True
            )
            db.add(dia)

        db.commit()
        db.refresh(nuevo_plan)
        
        # Devolver el plan completo con sus relaciones cargadas
        return nuevo_plan

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

# Endpoint temporal para probar NLP y Fuzzy Logic (Solo para testing/desarrollo)
@router.post("/test-nlp-fuzzy")
async def test_nlp_fuzzy(
    request: dict,
    current_user = Depends(get_current_staff)
):
    """
    Endpoint de prueba para las nuevas funcionalidades de NLP y Fuzzy Logic.
    🔒 REQUIERE AUTH STAFF: Solo personal autorizado puede probar.
    
    Parámetros esperados en request:
    - comando_texto: str (opcional) - Comando en lenguaje natural
    - perfil_usuario: dict - Perfil del usuario (edad, genero, objetivo, etc.)
    - adherencia_pct: int (0-100) - Porcentaje de adherencia
    - progreso_pct: int (0-100) - Porcentaje de progreso
    """
    print(f"🛠️ Testing NLP/Fuzzy por: {current_user.email}")
    try:
        comando_texto = request.get("comando_texto")
        perfil_usuario = request.get("perfil_usuario", {})
        adherencia_pct = request.get("adherencia_pct", 50)
        progreso_pct = request.get("progreso_pct", 50)

        # Probar NLP si hay comando
        nlp_result = None
        if comando_texto:
            nlp_result = ia_engine.interpretar_comando_nlp(comando_texto)

        # Probar Fuzzy Logic
        alerta_fuzzy = ia_engine.generar_alerta_fuzzy(adherencia_pct, progreso_pct)

        # Generar recomendación completa con las nuevas features
        recomendacion = ia_engine.recomendar_alimentos_con_groq(
            perfil_usuario=perfil_usuario,
            comando_texto=comando_texto,
            adherencia_pct=adherencia_pct,
            progreso_pct=progreso_pct
        )

        return {
            "nlp_resultado": nlp_result,
            "alerta_fuzzy": alerta_fuzzy,
            "recomendacion_completa": recomendacion,
            "mensaje": "Prueba de NLP y Fuzzy Logic exitosa"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en NLP/Fuzzy: {str(e)}")

# =================================================================
# 🍎 NUEVOS ENDPOINTS: GESTIÓN DE PLANES (FLUJO GYM REAL)
# =================================================================

@router.get("/planes/pendientes", response_model=list[PlanNutricionalResponse])
async def listar_planes_pendientes(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_staff)
):
    """
    Lista los planes en estado 'draft_ia' (generados por IA)
    que pertenecen a los clientes asignados al nutricionista logueado.
    """
    from app.models.client import Client
    
    query = db.query(PlanNutricional).filter(PlanNutricional.status == "draft_ia")
    
    # Si el usuario es nutricionista, solo ver sus asignados
    if current_user.role_name == "nutritionist":
        query = query.join(Client).filter(Client.assigned_nutri_id == current_user.id)
    
    return query.all()

@router.put("/planes/{plan_id}/validar")
async def validar_plan_nutricional(
    plan_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_staff)
):
    """
    El nutricionista revisa y aprueba el plan generado por la IA.
    1. Cambia el estado a 'validado'.
    2. Registra quién y cuándo lo validó.
    3. Cambia el estado de los detalles diarios a 'oficial'.
    """
    plan = db.query(PlanNutricional).filter(PlanNutricional.id == plan_id).first()
    
    if not plan:
        raise HTTPException(status_code=404, detail="Plan no encontrado")
        
    # Seguridad: Un nutri solo puede validar si el cliente está asignado a él
    # o si es administrador
    from app.models.client import Client
    cliente = db.query(Client).filter(Client.id == plan.client_id).first()
    
    if current_user.role_name == "nutritionist" and cliente.assigned_nutri_id != current_user.id:
        raise HTTPException(status_code=403, detail="No tienes permiso para validar planes de este cliente")

    # Actualizar cabecera del plan
    from app.core.utils import get_peru_now
    plan.status = "validado"
    plan.validated_by_id = current_user.id
    plan.validated_at = get_peru_now().replace(tzinfo=None)
    plan.nutricionista_id = current_user.id # Asignar formalmente al plan
    
    # Actualizar todos los días del plan a oficial
    for dia in plan.detalles_diarios:
        dia.estado = "oficial"
        dia.validado_nutri = True
        
    db.commit()
    
    return {
        "message": "Plan validado exitosamente",
        "plan_id": plan.id,
        "validado_por": f"{current_user.first_name} {current_user.last_name_paternal}",
        "fecha": plan.validated_at.isoformat()
    }


@router.get("/recomendaciones")
async def obtener_recomendaciones_personalizadas(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    🧠 SISTEMA DE APRENDIZAJE: Recomendaciones personalizadas de alimentos

    - Usuario NUEVO → Top alimentos según objetivo y condiciones dietéticas
    - Usuario CON historial → Sus favoritos filtrados por restricciones dietéticas
    """
    from app.models.client import Client
    from app.models.preferencias import PreferenciaAlimento
    from app.services.recomendador_platos import _tokens_prohibidos

    # Obtener cliente
    cliente = db.query(Client).filter(Client.email == current_user.email).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    # Tokens prohibidos según condiciones dietéticas del perfil
    _conds = list(cliente.medical_conditions or [])
    tokens_prohib = _tokens_prohibidos(_conds)

    def _es_apto(nombre: str) -> bool:
        """True si el nombre del alimento no contiene tokens prohibidos."""
        if not tokens_prohib:
            return True
        return not any(t in nombre.lower() for t in tokens_prohib)

    def _justificacion_cold(nombre: str, categoria: str) -> str:
        """Genera una justificación breve para cold-start según categoría."""
        _map = {
            "proteina":         "fuente de proteína de calidad para tu meta",
            "proteina_vegetal": "proteína 100% vegetal, apta para tu dieta",
            "verduras":         "bajo en calorías, rico en fibra y micronutrientes",
            "carbohidratos":    "carbohidrato complejo para energía sostenida",
            "completo":         "plato equilibrado con macros balanceados",
            "frutas":           "vitaminas, fibra y azúcares naturales",
        }
        return _map.get(categoria, "opción equilibrada según tu objetivo")

    # Consultar preferencias del cliente
    preferencias = db.query(PreferenciaAlimento).filter(
        PreferenciaAlimento.client_id == cliente.id
    ).order_by(PreferenciaAlimento.frecuencia.desc()).limit(20).all()

    if len(preferencias) < 3:  # Cold start — usuario nuevo
        # Catálogo base omnívoro por objetivo
        _base_omnivoro = {
            "Perder peso": [
                {"nombre": "Pollo a la plancha",  "categoria": "proteina",      "calorias_aprox": 165},
                {"nombre": "Ensalada verde",       "categoria": "verduras",      "calorias_aprox": 50},
                {"nombre": "Pescado blanco",       "categoria": "proteina",      "calorias_aprox": 100},
            ],
            "Ganar masa": [
                {"nombre": "Arroz integral",       "categoria": "carbohidratos", "calorias_aprox": 215},
                {"nombre": "Pollo con piel",       "categoria": "proteina",      "calorias_aprox": 230},
                {"nombre": "Camote al horno",      "categoria": "carbohidratos", "calorias_aprox": 180},
            ],
            "Mantener peso": [
                {"nombre": "Arroz con pollo",      "categoria": "completo",      "calorias_aprox": 350},
                {"nombre": "Pescado a la plancha", "categoria": "proteina",      "calorias_aprox": 140},
                {"nombre": "Quinua cocida",        "categoria": "carbohidratos", "calorias_aprox": 222},
            ],
        }
        # Alternativas veganas/vegetarianas por objetivo
        _base_vegetal = {
            "Perder peso": [
                {"nombre": "Ensalada de quinua con verduras", "categoria": "proteina_vegetal", "calorias_aprox": 180},
                {"nombre": "Ensalada verde mixta",            "categoria": "verduras",         "calorias_aprox": 50},
                {"nombre": "Sopa de lentejas",                "categoria": "proteina_vegetal", "calorias_aprox": 150},
            ],
            "Ganar masa": [
                {"nombre": "Arroz con lentejas",         "categoria": "proteina_vegetal", "calorias_aprox": 320},
                {"nombre": "Tofu salteado con verduras", "categoria": "proteina_vegetal", "calorias_aprox": 180},
                {"nombre": "Camote al horno",            "categoria": "carbohidratos",    "calorias_aprox": 180},
            ],
            "Mantener peso": [
                {"nombre": "Quinua con verduras salteadas", "categoria": "completo",         "calorias_aprox": 280},
                {"nombre": "Ensalada de garbanzos",         "categoria": "proteina_vegetal", "calorias_aprox": 220},
                {"nombre": "Avena con frutas frescas",      "categoria": "carbohidratos",    "calorias_aprox": 200},
            ],
        }

        objetivo = cliente.goal or "Mantener peso"
        _es_vegano = "Vegano" in _conds or "Vegetariano" in _conds
        catalogo = _base_vegetal if _es_vegano else _base_omnivoro
        recomendaciones_raw = catalogo.get(objetivo, catalogo["Mantener peso"])

        # Filtrar cualquier residuo que incumpla restricciones + añadir justificación
        recomendaciones = [
            {**item, "justificacion": _justificacion_cold(item["nombre"], item["categoria"])}
            for item in recomendaciones_raw
            if _es_apto(item["nombre"])
        ]

        return {
            "tipo": "cold_start",
            "mensaje": "Recomendaciones según tu objetivo y perfil dietético",
            "recomendaciones": recomendaciones,
            "nota": "El sistema aprenderá tus preferencias a medida que registres tus comidas",
        }

    else:  # Usuario con historial
        favoritos = []
        for pref in preferencias:
            nombre = pref.alimento.capitalize()
            if not _es_apto(nombre):
                continue   # Excluir favorito que incumple restricción dietética actual
            favoritos.append({
                "nombre":       nombre,
                "frecuencia":   pref.frecuencia,
                "puntuacion":   round(pref.puntuacion, 2),
                "ultima_vez":   pref.ultima_vez.strftime("%Y-%m-%d"),
                "justificacion": "uno de tus alimentos más frecuentes, compatible con tu perfil",
            })
            if len(favoritos) == 10:
                break

        return {
            "tipo": "personalizado",
            "mensaje": f"Basado en tus {len(favoritos)} alimentos favoritos compatibles con tu dieta",
            "favoritos": favoritos,
            "nota": "Estas son tus elecciones más frecuentes que respetan tus restricciones dietéticas",
        }
