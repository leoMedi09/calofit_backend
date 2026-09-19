from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from datetime import datetime
from typing import List, Optional

from app.core.database import get_db
from app.models.historial import AlertaSalud
from app.models.client import Client
from app.models.user import User
from app.schemas.alerta import AlertaSaludResponse, AlertaUpdateRequest, AlertaAtenderRequest
from app.api.routes.auth import get_current_staff

router = APIRouter()


@router.get("/mis-clientes", response_model=list[AlertaSaludResponse])
def listar_alertas_mis_clientes(
    estado: Optional[str] = Query(None, description="Filtrar por estado: pendiente, en_proceso, atendida"),
    severidad: Optional[str] = Query(None, description="Filtrar por severidad: bajo, medio, alto"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_staff),
):
    """
    Lista todas las alertas de salud de los clientes asignados al staff actual.

    - **Autenticación:** Solo staff (Nutritionist/Trainer)
    - **Filtros opcionales:** estado, severidad
    """
    staff = db.query(User).filter(User.email == current_user.email).first()
    if not staff:
        raise HTTPException(status_code=404, detail="Usuario staff no encontrado")

    if staff.role_name == "nutritionist":
        clientes_ids = db.query(Client.id).filter(Client.assigned_nutri_id == staff.id).all()
    elif staff.role_name == "coach":
        clientes_ids = db.query(Client.id).filter(Client.assigned_coach_id == staff.id).all()
    else:
        clientes_ids = db.query(Client.id).all()

    clientes_ids = [c[0] for c in clientes_ids]

    if not clientes_ids:
        return []

    query = db.query(AlertaSalud).filter(AlertaSalud.client_id.in_(clientes_ids))

    if estado:
        query = query.filter(AlertaSalud.estado == estado)
    if severidad:
        query = query.filter(AlertaSalud.severidad == severidad)

    alertas = query.order_by(AlertaSalud.fecha_deteccion.desc()).all()

    clientes = {c.id: c for c in db.query(Client).filter(Client.id.in_(clientes_ids)).all()}
    atendido_ids = {a.atendido_por_id for a in alertas if a.atendido_por_id}
    usuarios = {u.id: u for u in db.query(User).filter(User.id.in_(atendido_ids)).all()} if atendido_ids else {}

    resultado = []
    for alerta in alertas:
        cliente = clientes.get(alerta.client_id)
        atendido_por = usuarios.get(alerta.atendido_por_id) if alerta.atendido_por_id else None

        alerta_dict = {
            "id": alerta.id,
            "client_id": alerta.client_id,
            "tipo": alerta.tipo,
            "descripcion": alerta.descripcion,
            "severidad": alerta.severidad,
            "estado": alerta.estado,
            "atendido_por_id": alerta.atendido_por_id,
            "notas": alerta.notas,
            "fecha_deteccion": alerta.fecha_deteccion,
            "fecha_atencion": alerta.fecha_atencion,
            "created_at": alerta.created_at,
            "cliente_nombre": f"{cliente.first_name} {cliente.last_name_paternal}" if cliente else "Desconocido",
            "atendido_por_nombre": f"{atendido_por.first_name} {atendido_por.last_name_paternal}"
            if atendido_por
            else None,
        }
        resultado.append(alerta_dict)

    return resultado


@router.get("/{alerta_id}", response_model=AlertaSaludResponse)
def obtener_detalle_alerta(
    alerta_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_staff)
):
    """
    Obtiene el detalle completo de una alerta específica.

    - **Autenticación:** Solo staff
    - **Validación:** El staff debe tener acceso al cliente
    """
    alerta = db.query(AlertaSalud).filter(AlertaSalud.id == alerta_id).first()
    if not alerta:
        raise HTTPException(status_code=404, detail="Alerta no encontrada")

    staff = db.query(User).filter(User.email == current_user.email).first()
    cliente = db.query(Client).filter(Client.id == alerta.client_id).first()

    if staff.role_name != "admin":
        tiene_acceso = False
        if staff.role_name == "nutritionist" and cliente.assigned_nutri_id == staff.id:
            tiene_acceso = True
        elif staff.role_name == "coach" and cliente.assigned_coach_id == staff.id:
            tiene_acceso = True

        if not tiene_acceso:
            raise HTTPException(status_code=403, detail="No tienes acceso a esta alerta")

    atendido_por = None
    if alerta.atendido_por_id:
        atendido_por = db.query(User).filter(User.id == alerta.atendido_por_id).first()

    return {
        "id": alerta.id,
        "client_id": alerta.client_id,
        "tipo": alerta.tipo,
        "descripcion": alerta.descripcion,
        "severidad": alerta.severidad,
        "estado": alerta.estado,
        "atendido_por_id": alerta.atendido_por_id,
        "notas": alerta.notas,
        "fecha_deteccion": alerta.fecha_deteccion,
        "fecha_atencion": alerta.fecha_atencion,
        "created_at": alerta.created_at,
        "cliente_nombre": f"{cliente.first_name} {cliente.last_name_paternal}",
        "atendido_por_nombre": f"{atendido_por.first_name} {atendido_por.last_name_paternal}" if atendido_por else None,
    }


@router.put("/{alerta_id}/actualizar")
def actualizar_alerta(
    alerta_id: int,
    update_data: AlertaUpdateRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_staff),
):
    """
    Actualiza el estado y/o notas de una alerta.

    - **Autenticación:** Solo staff
    - **Body:** estado (opcional), notas (opcional)
    """
    alerta = db.query(AlertaSalud).filter(AlertaSalud.id == alerta_id).first()
    if not alerta:
        raise HTTPException(status_code=404, detail="Alerta no encontrada")

    staff = db.query(User).filter(User.email == current_user.email).first()
    cliente = db.query(Client).filter(Client.id == alerta.client_id).first()

    if staff.role_name != "admin":
        tiene_acceso = False
        if staff.role_name == "nutritionist" and cliente.assigned_nutri_id == staff.id:
            tiene_acceso = True
        elif staff.role_name == "coach" and cliente.assigned_coach_id == staff.id:
            tiene_acceso = True

        if not tiene_acceso:
            raise HTTPException(status_code=403, detail="No tienes acceso a esta alerta")

    if update_data.estado:
        alerta.estado = update_data.estado
        if update_data.estado == "en_proceso" and not alerta.atendido_por_id:
            alerta.atendido_por_id = staff.id

    if update_data.notas:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        nueva_nota = f"[{timestamp} - {staff.first_name}] {update_data.notas}"
        if alerta.notas:
            alerta.notas += f"\n\n{nueva_nota}"
        else:
            alerta.notas = nueva_nota

    db.commit()
    db.refresh(alerta)

    return {
        "success": True,
        "message": "Alerta actualizada correctamente",
        "alerta_id": alerta.id,
        "nuevo_estado": alerta.estado,
    }


@router.put("/{alerta_id}/atender")
def marcar_alerta_atendida(
    alerta_id: int,
    atencion_data: AlertaAtenderRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_staff),
):
    """
    Marca una alerta como atendida y registra las notas finales.

    - **Autenticación:** Solo staff
    - **Body:** notas (requerido, mínimo 10 caracteres)
    """
    alerta = db.query(AlertaSalud).filter(AlertaSalud.id == alerta_id).first()
    if not alerta:
        raise HTTPException(status_code=404, detail="Alerta no encontrada")

    staff = db.query(User).filter(User.email == current_user.email).first()
    cliente = db.query(Client).filter(Client.id == alerta.client_id).first()

    if staff.role_name != "admin":
        tiene_acceso = False
        if staff.role_name == "nutritionist" and cliente.assigned_nutri_id == staff.id:
            tiene_acceso = True
        elif staff.role_name == "coach" and cliente.assigned_coach_id == staff.id:
            tiene_acceso = True

        if not tiene_acceso:
            raise HTTPException(status_code=403, detail="No tienes acceso a esta alerta")

    alerta.estado = "atendida"
    alerta.atendido_por_id = staff.id
    alerta.fecha_atencion = datetime.now()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    nota_final = f"[{timestamp} - {staff.first_name}] ATENDIDA: {atencion_data.notas}"
    if alerta.notas:
        alerta.notas += f"\n\n{nota_final}"
    else:
        alerta.notas = nota_final

    db.commit()
    db.refresh(alerta)

    return {
        "success": True,
        "message": "Alerta marcada como atendida correctamente",
        "alerta_id": alerta.id,
        "atendido_por": f"{staff.first_name} {staff.last_name_paternal}",
        "fecha_atencion": alerta.fecha_atencion,
    }


@router.get("/cliente/{cliente_id}", response_model=list[AlertaSaludResponse])
def listar_alertas_por_cliente(
    cliente_id: int, db: Session = Depends(get_db), current_user=Depends(get_current_staff)
):
    """
    Lista todas las alertas de un cliente específico (historial completo).

    - **Autenticación:** Solo staff con acceso al cliente
    """
    staff = db.query(User).filter(User.email == current_user.email).first()
    cliente = db.query(Client).filter(Client.id == cliente_id).first()

    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    if staff.role_name != "admin":
        tiene_acceso = False
        if staff.role_name == "nutritionist" and cliente.assigned_nutri_id == staff.id:
            tiene_acceso = True
        elif staff.role_name == "coach" and cliente.assigned_coach_id == staff.id:
            tiene_acceso = True

        if not tiene_acceso:
            raise HTTPException(status_code=403, detail="No tienes acceso a este cliente")

    alertas = (
        db.query(AlertaSalud)
        .filter(AlertaSalud.client_id == cliente_id)
        .order_by(AlertaSalud.fecha_deteccion.desc())
        .all()
    )

    atendido_ids = {a.atendido_por_id for a in alertas if a.atendido_por_id}
    usuarios = {u.id: u for u in db.query(User).filter(User.id.in_(atendido_ids)).all()} if atendido_ids else {}

    resultado = []
    for alerta in alertas:
        atendido_por = usuarios.get(alerta.atendido_por_id) if alerta.atendido_por_id else None

        alerta_dict = {
            "id": alerta.id,
            "client_id": alerta.client_id,
            "tipo": alerta.tipo,
            "descripcion": alerta.descripcion,
            "severidad": alerta.severidad,
            "estado": alerta.estado,
            "atendido_por_id": alerta.atendido_por_id,
            "notas": alerta.notas,
            "fecha_deteccion": alerta.fecha_deteccion,
            "fecha_atencion": alerta.fecha_atencion,
            "created_at": alerta.created_at,
            "cliente_nombre": f"{cliente.first_name} {cliente.last_name_paternal}",
            "atendido_por_nombre": f"{atendido_por.first_name} {atendido_por.last_name_paternal}"
            if atendido_por
            else None,
        }
        resultado.append(alerta_dict)

    return resultado
