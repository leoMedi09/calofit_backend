from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.models.client import Client
from app.models.user import User
from app.api.routes.auth import get_current_user
from app.services.ia_service import ia_service
from app.models.nutricion import PlanNutricional, PlanDiario
from app.models.historial import AlertaSalud, HistorialPeso
from app.schemas.nutricion import PlanNutricionalResponse, PlanNutricionalUpdate
from app.schemas.client import StrategicGuideUpdate
from datetime import datetime, timedelta
from app.core.utils import calcular_metabolismo_basal, obtener_macros_desglosados

router = APIRouter()

def check_is_nutri(current_user: User):
    role = str(getattr(current_user, "role_name", "")).lower()
    if role not in ["nutricionista", "nutritionist", "admin", "administrador"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operación permitida solo para Nutricionistas o Administradores"
        )
    return current_user

def calcular_progreso_paciente(client: Client) -> float:
    """
    Calcula el progreso real (%) basado en la tendencia de peso y el objetivo.
    """
    if not client.historial_peso or len(client.historial_peso) < 1:
        return 50.0  # Punto neutro si no hay historial

    # Encontramos el peso más antiguo para comparar
    historial_ordenado = sorted(client.historial_peso, key=lambda x: x.fecha_registro)
    peso_inicial = historial_ordenado[0].peso_kg
    peso_actual = client.weight or peso_inicial
    objetivo = (client.goal or "Mantener peso").lower()

    if "perder" in objetivo:
        # Si bajó de peso respecto al inicio, progreso > 50
        cambio = peso_inicial - peso_actual
        return min(100.0, max(0.0, 50.0 + (cambio * 2)))
    elif "ganar" in objetivo:
        cambio = peso_actual - peso_inicial
        return min(100.0, max(0.0, 50.0 + (cambio * 2)))
    else:
        # Mantener: Estabilidad (variación < 1kg es 100%)
        variacion = abs(peso_actual - peso_inicial)
        return max(0.0, 100.0 - (variacion * 5))

@router.get("/clientes", response_model=List[dict])
def get_assigned_patients(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    check_is_nutri(current_user)
    
    # Si es Admin, ve todos los clientes. Si es Nutri, solo los suyos.
    query = db.query(Client)
    role = str(getattr(current_user, "role_name", "")).lower()
    if role in ["nutricionista", "nutritionist"]:
        query = query.filter(Client.assigned_nutri_id == current_user.id)
    
    clients = query.all()
    
    from app.core.utils import get_peru_now
    now = get_peru_now()
    seven_days_ago = now - timedelta(days=7)
    
    result = []
    for c in clients:
        # Lógica de adherencia real: Contar días con registros en los últimos 7 días
        registros_recientes = [r for r in c.progreso_calorias if r.fecha >= seven_days_ago.date()]
        num_registros = len(registros_recientes)
        
        adherencia = round((num_registros / 7) * 100, 1)
        progreso = calcular_progreso_paciente(c)
        
        alerta_data = ia_service.generar_alerta_fuzzy(adherencia, progreso)
        
        result.append({
            "id": c.id,
            "full_name": f"{c.first_name} {c.last_name_paternal} {c.last_name_maternal}",
            "email": c.email,
            "goal": c.goal,
            "weight": c.weight,
            "nutri_id": c.assigned_nutri_id, # ✅ Agregamos el ID para que Admin sepa si ya tiene uno
            "adherencia": adherencia,
            "alerta": alerta_data.get("mensaje", ""),
            "alerta_nivel": alerta_data.get("nivel", "Bajo"),
            "gender": c.gender,
            "is_validated": c.is_strategic_guide_validated,
            "semana_status": _calcular_semana_status(c, db) # ✅ Nuevo: Estado semántico para el nutri
        })
    
    return result

def _calcular_semana_status(c: Client, db: Session) -> str:
    """
    Determina el estado de la semana del paciente:
    - 'validado': Plan aprobado después del último sábado.
    - 'pendiente': Hizo check-in pero falta validación.
    - 'falta_checkin': No ha registrado peso desde el último sábado.
    """
    from app.core.utils import get_peru_now
    now = get_peru_now()
    # Encontrar el último sábado (o hoy si es sábado)
    # weekday(): 0=Mon, 5=Sat, 6=Sun
    days_since_sat = (now.weekday() - 5) % 7
    last_saturday = (now - timedelta(days=days_since_sat)).date()
    
    # 1. ¿Hizo check-in (registró peso) desde el último sábado?
    hizo_checkin = any(r.fecha_registro >= last_saturday for r in c.historial_peso)
    
    if not hizo_checkin:
        return "falta_checkin"
        
    # 2. ¿Tiene un plan validado después de ese check-in?
    # Buscamos el plan más reciente
    ultimo_plan = db.query(PlanNutricional).filter(
        PlanNutricional.client_id == c.id
    ).order_by(PlanNutricional.fecha_creacion.desc()).first()
    
    if ultimo_plan and ultimo_plan.status == "validado" and ultimo_plan.validated_at:
        # Si la validación fue después del último sábado, está OK
        if ultimo_plan.validated_at.date() >= last_saturday:
            return "validado"
            
    return "pendiente"

@router.get("/cliente/{id}/progreso")
def get_patient_progress(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    check_is_nutri(current_user)
    
    client = db.query(Client).filter(Client.id == id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
        
    # Verificar que el nutri tenga acceso
    if current_user.role_name.upper() in ["NUTRICIONISTA", "NUTRITIONIST"] and client.assigned_nutri_id != current_user.id:
        raise HTTPException(status_code=403, detail="No tienes permiso para ver este paciente")

    # Obtener historial de peso, imc y progreso calórico
    historial_peso = [{"fecha": h.fecha_registro, "valor": h.peso_kg} for h in client.historial_peso]
    historial_imc = [{"fecha": h.fecha_registro, "valor": h.imc} for h in client.historial_imc]
    
    # Obtener alertas de salud (v80.0)
    alertas = [{
        "id": a.id,
        "tipo": a.tipo,
        "descripcion": a.descripcion,
        "severidad": a.severidad,
        "estado": a.estado,
        "fecha": a.fecha_deteccion
    } for a in client.alertas_salud]

    # ✨ Sincronización Metabólica (v80.0)
    # Proporcionamos la misma base que ve el cliente en su dashboard
    tmb_estimada = calcular_metabolismo_basal(client)
    
    # Calorias ajustadas según objetivo
    calorias_ajustadas = tmb_estimada
    if client.goal == "Perder peso":
        calorias_ajustadas *= 0.85
    elif client.goal == "Ganar masa":
        calorias_ajustadas *= 1.1

    recomendacion_ia = obtener_macros_desglosados(calorias_ajustadas, client.goal)

    return {
        "id": client.id,
        "nombre": f"{client.first_name} {client.last_name_paternal} {client.last_name_maternal}", # Adjusted to match existing full_name format
        "objetivo": client.goal,
        "focus_objetivo": client.ai_strategic_focus, # Renamed from client.focus_objetivo to client.ai_strategic_focus
        "semana_status": _calcular_semana_status(client, db),
        "historial_peso": historial_peso,
        "historial_imc": historial_imc,
        "alertas_salud": alertas,
        # Guía Estratégica (Misión Semanal)
        "ai_strategic_focus": client.ai_strategic_focus,
        "is_strategic_guide_validated": client.is_strategic_guide_validated,
        "is_validated": client.is_strategic_guide_validated,
        # Sincronización v80.0
        "metabolismo_estimado": {
            "tmb": round(tmb_estimada),
            "calorias_objetivo": recomendacion_ia["calorias"],
            "proteinas_g": recomendacion_ia["proteinas_g"],
            "carbohidratos_g": recomendacion_ia["carbohidratos_g"],
            "grasas_g": recomendacion_ia["grasas_g"],
            "distribucion": recomendacion_ia["pct"]
        },
        "current_weight": client.weight,
        "current_height": client.height,
        "recommended_foods": client.recommended_foods,
        "forbidden_foods": client.forbidden_foods,
        "medical_conditions": client.medical_conditions
    }

@router.get("/cliente/{id}/sugerir-estrategia")
async def suggest_strategic_guide(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    check_is_nutri(current_user)
    
    client = db.query(Client).filter(Client.id == id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
        
    # Recopilar alertas recientes (últimos 15 días) para contexto
    date_limit = datetime.utcnow() - timedelta(days=15)
    alertas_recent = [a for a in client.alertas_salud if a.fecha_deteccion >= date_limit]
    
    alertas_list = [{
        "tipo": a.tipo,
        "descripcion": a.descripcion,
        "severidad": a.severidad
    } for a in alertas_recent]
    
    # Calcular IMC actual
    imc = 0
    if client.weight and client.height:
        height_m = client.height / 100
        imc = round(client.weight / (height_m * height_m), 1)

    # Calcular edad
    edad = 0
    if client.birth_date:
        today = datetime.now()
        edad = today.year - client.birth_date.year - ((today.month, today.day) < (client.birth_date.month, client.birth_date.day))

    # Obtener historial de peso para tendencia
    historial_peso = [{"fecha": h.fecha_registro, "valor": h.peso_kg} for h in client.historial_peso]

    perfil = {
        "full_name": f"{client.first_name} {client.last_name_paternal}",
        "gender": "Hombre" if client.gender == 'M' else "Mujer",
        "age": edad,
        "current_weight": client.weight,
        "current_height": client.height,
        "imc": imc,
        "activity_level": client.activity_level or "Moderado",
        "goal": client.goal,
        "medical_conditions": client.medical_conditions or [],
        "weight_history": historial_peso
    }
    
    sugerencia = await ia_service.sugerir_guia_estrategica(perfil, alertas_list)
    return sugerencia

@router.post("/actualizar-guia-estrategica/{id}")
def update_strategic_guide(
    id: int,
    guide: StrategicGuideUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    check_is_nutri(current_user)
    
    client = db.query(Client).filter(Client.id == id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
        
    if current_user.role_name.upper() in ["NUTRICIONISTA", "NUTRITIONIST"] and client.assigned_nutri_id != current_user.id:
        raise HTTPException(status_code=403, detail="No tienes permiso para modificar este paciente")

    # Actualización estratégica (v80.0)
    if guide.ai_strategic_focus is not None:
        client.ai_strategic_focus = guide.ai_strategic_focus
    if guide.recommended_foods is not None:
        client.recommended_foods = guide.recommended_foods
    if guide.forbidden_foods is not None:
        client.forbidden_foods = guide.forbidden_foods
    if guide.medical_conditions is not None:
        client.medical_conditions = guide.medical_conditions
    
    # ✅ (v80.0) No marcamos como validado globalmente al editar alimentos.
    # La validación oficial es un proceso semanal por separado (validate_plan).
         
    db.commit()
    return {"status": "success", "message": "Guía estratégica actualizada para la IA"}
@router.post("/validar-plan/{id}")
def validate_plan(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    check_is_nutri(current_user)
    
    client = db.query(Client).filter(Client.id == id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
        
    status_semana = _calcular_semana_status(client, db)
    if status_semana == "falta_checkin":
        raise HTTPException(
            status_code=400, 
            detail="No se puede validar el plan si el paciente aún no ha realizado su Check-in semanal."
        )

    # Lógica para cambiar status de plan a 'validado'
    plan = db.query(PlanNutricional).filter(PlanNutricional.client_id == id).order_by(PlanNutricional.fecha_creacion.desc()).first()
    if plan:
        plan.status = "validado"
        plan.validated_by_id = current_user.id
        plan.validated_at = datetime.utcnow()
        db.commit()
    return {"message": f"Plan del cliente {id} validado correctamente por Nutricionista {current_user.first_name}"}

@router.get("/cliente/{id}/plan", response_model=PlanNutricionalResponse)
def get_client_plan(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    check_is_nutri(current_user)
    client = db.query(Client).filter(Client.id == id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
        
    if current_user.role_name.upper() in ["NUTRICIONISTA", "NUTRITIONIST"] and client.assigned_nutri_id != current_user.id:
        raise HTTPException(status_code=403, detail="No tienes permiso para ver este paciente")
    
    plan = db.query(PlanNutricional).filter(PlanNutricional.client_id == id).order_by(PlanNutricional.fecha_creacion.desc()).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Este paciente aún no tiene un plan nutricional asignado")
    return plan

@router.put("/cliente/{id}/plan")
def update_client_plan(
    id: int,
    plan_update: PlanNutricionalUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    check_is_nutri(current_user)
    client = db.query(Client).filter(Client.id == id).first()
    if not client:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
        
    if current_user.role_name.upper() in ["NUTRICIONISTA", "NUTRITIONIST"] and client.assigned_nutri_id != current_user.id:
        raise HTTPException(status_code=403, detail="No tienes permiso")
    
    plan = db.query(PlanNutricional).filter(PlanNutricional.client_id == id).order_by(PlanNutricional.fecha_creacion.desc()).first()
    if not plan:
        raise HTTPException(status_code=404, detail="No hay plan para actualizar")
    
    if plan_update.objetivo:
        plan.objetivo = plan_update.objetivo
    if plan_update.observaciones:
        plan.observaciones = plan_update.observaciones
    if plan_update.status:
        plan.status = plan_update.status
        
    if plan_update.detalles_diarios:
        for i, daily_update in enumerate(plan_update.detalles_diarios):
            if i < len(plan.detalles_diarios):
                daily = plan.detalles_diarios[i]
                if daily_update.calorias_dia is not None:
                    daily.calorias_dia = daily_update.calorias_dia
                if daily_update.proteinas_g is not None:
                    daily.proteinas_g = daily_update.proteinas_g
                if daily_update.carbohidratos_g is not None:
                    daily.carbohidratos_g = daily_update.carbohidratos_g
                if daily_update.grasas_g is not None:
                    daily.grasas_g = daily_update.grasas_g
                daily.estado = daily_update.estado or "oficial"
    
    db.commit()
    return {"message": "Plan actualizado exitosamente"}
@router.get("/stats")
def get_nutri_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    check_is_nutri(current_user)
    
    # 1. Filtro base de pacientes asignados
    query = db.query(Client)
    role = str(getattr(current_user, "role_name", "")).lower()
    if role in ["nutricionista", "nutritionist"]:
        query = query.filter(Client.assigned_nutri_id == current_user.id)
    
    pacientes = query.all()
    total_pacientes = len(pacientes)
    
    if total_pacientes == 0:
        return {
            "total_pacientes": 0,
            "validaciones_pendientes": 0,
            "alertas_criticas": 0,
            "adherencia_media": 0.0,
            "tendencia_adherencia": [0,0,0,0,0,0,0]
        }

    # 2. Validaciones Pendientes (Planes en status provisional_ia)
    validaciones_pendientes = db.query(PlanNutricional).join(Client).filter(
        Client.assigned_nutri_id == current_user.id if role in ["nutricionista", "nutritionist"] else True,
        PlanNutricional.status == "provisional_ia"
    ).count()

    # 3. Alertas Críticas (IA + Alertas de Salud Pendientes)
    alertas_criticas = 0
    seven_days_ago = datetime.now() - timedelta(days=7)
    
    # 3.1 Alertas desde Tabla AlertaSalud (Pendientes)
    alertas_db_query = db.query(AlertaSalud).filter(
        AlertaSalud.estado == "pendiente"
    ).join(Client).filter(
        Client.assigned_nutri_id == current_user.id if role in ["nutricionista", "nutritionist"] else True
    )
    alertas_db_count = alertas_db_query.count()
    alertas_recientes_objs = alertas_db_query.order_by(AlertaSalud.fecha_deteccion.desc()).limit(3).all()
    
    alertas_formateadas = [
        {
            "id": a.id,
            "paciente": f"{a.cliente.first_name} {a.cliente.last_name_paternal}",
            "problema": a.descripcion,
            "urgencia": a.severidad.capitalize(),
            "tipo": a.tipo
        } for a in alertas_recientes_objs
    ]
    
    # 3.2 Alertas detectadas por IA (Lógica Fuzzy)
    for c in pacientes:
        # Calcular adherencia simplificada para el conteo de alertas
        registros_recientes = [r for r in c.progreso_calorias if r.fecha >= seven_days_ago.date()]
        adh = round((len(registros_recientes) / 7) * 100, 1)
        prog = calcular_progreso_paciente(c)
        
        alerta_data = ia_service.generar_alerta_fuzzy(adh, prog)
        if alerta_data.get("nivel") == "Alto":
            alertas_criticas += 1
            # Si no hay muchas alertas en DB, podemos agregar alertas de IA como sugerencias
            if len(alertas_formateadas) < 5:
                # Evitar duplicados si ya hay una alerta real para este paciente
                if not any(al['paciente'].startswith(c.first_name) for al in alertas_formateadas):
                    alertas_formateadas.append({
                        "id": 0,
                        "paciente": f"{c.first_name} {c.last_name_paternal}",
                        "problema": "Baja adherencia/progreso (Detectado por IA)",
                        "urgencia": "Media",
                        "tipo": "progreso"
                    })
            
    # Combinar ambas fuentes
    total_alertas = alertas_criticas + alertas_db_count

    # 4. Adherencia Media y Tendencia (Últimos 7 días)
    # Calculamos el promedio de registros realizados por todo el grupo por día
    tendencia = []
    total_adh_sum = 0
    for i in range(6, -1, -1):
        target_date = (datetime.now() - timedelta(days=i)).date()
        conteo_dia = 0
        for c in pacientes:
            if any(r.fecha == target_date for r in c.progreso_calorias):
                conteo_dia += 1
        
        adh_dia = round((conteo_dia / total_pacientes) * 100, 1) if total_pacientes > 0 else 0
        tendencia.append(adh_dia)
        total_adh_sum += adh_dia

    adherencia_media = round(total_adh_sum / 7, 1)

    return {
        "total_pacientes": total_pacientes,
        "validaciones_pendientes": validaciones_pendientes,
        "alertas_criticas": total_alertas,
        "alertas_recientes": alertas_formateadas,
        "adherencia_media": adherencia_media,
        "tendencia_adherencia": tendencia
    }
