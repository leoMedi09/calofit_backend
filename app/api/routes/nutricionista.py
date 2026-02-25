from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.models.client import Client
from app.models.user import User
from app.api.routes.auth import get_current_user
from app.services.ia_service import IAService
from app.models.nutricion import PlanNutricional, PlanDiario
from app.models.historial import AlertaSalud, HistorialPeso
from app.schemas.nutricion import PlanNutricionalResponse, PlanNutricionalUpdate
from datetime import datetime, timedelta

router = APIRouter()
ia_service = IAService()

def check_is_nutri(current_user: User):
    role = str(getattr(current_user, "role_name", "")).lower()
    if role not in ["nutricionista", "admin", "administrador"]:
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
    if role == "nutricionista":
        query = query.filter(Client.assigned_nutri_id == current_user.id)
    
    clients = query.all()
    
    from datetime import datetime, timedelta
    seven_days_ago = datetime.now() - timedelta(days=7)
    
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
            "gender": c.gender
        })
    
    return result

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
    if current_user.role_name.upper() == "NUTRICIONISTA" and client.assigned_nutri_id != current_user.id:
        raise HTTPException(status_code=403, detail="No tienes permiso para ver este paciente")

    # Obtener historial de peso, imc y progreso calórico
    historial_peso = [{"fecha": h.fecha, "valor": h.peso} for h in client.historial_peso]
    historial_imc = [{"fecha": h.fecha, "valor": h.imc} for h in client.historial_imc]
    
    return {
        "client_id": client.id,
        "full_name": f"{client.first_name} {client.last_name_paternal}",
        "historial_peso": historial_peso,
        "historial_imc": historial_imc,
        "current_weight": client.weight,
        "current_height": client.height,
    }

@router.post("/validar-plan/{id}")
def validate_plan(
    id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    check_is_nutri(current_user)
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
        
    if current_user.role_name.upper() == "NUTRICIONISTA" and client.assigned_nutri_id != current_user.id:
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
        
    if current_user.role_name.upper() == "NUTRICIONISTA" and client.assigned_nutri_id != current_user.id:
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
    if role == "nutricionista":
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
        Client.assigned_nutri_id == current_user.id if role == "nutricionista" else True,
        PlanNutricional.status == "provisional_ia"
    ).count()

    # 3. Alertas Críticas (IA + Alertas de Salud Pendientes)
    alertas_criticas = 0
    seven_days_ago = datetime.now() - timedelta(days=7)
    
    # 3.1 Alertas desde Tabla AlertaSalud (Pendientes)
    alertas_db_query = db.query(AlertaSalud).filter(
        AlertaSalud.estado == "pendiente"
    ).join(Client).filter(
        Client.assigned_nutri_id == current_user.id if role == "nutricionista" else True
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
