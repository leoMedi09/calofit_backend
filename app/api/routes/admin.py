from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.core.database import get_db
from app.models.user import User
from app.models.client import Client
from app.models.auditoria import AuditoriaAdmin
from app.schemas.user import UserCreate, UserResponse, PasswordUpdate, UserUpdate
from app.core.security import security
from app.api.routes.auth import get_current_user

router = APIRouter()

def check_is_admin(current_user):
    role = str(getattr(current_user, "role_name", "")).lower()
    if role not in ["admin", "administrador"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Solo el administrador puede realizar esta acción"
        )
    return True

def _log_admin_action(db: Session, admin_id: int, accion: str, descripcion: str, tabla: str = None, reg_id: int = None):
    log = AuditoriaAdmin(
        admin_id=admin_id,
        accion=accion,
        descripcion=descripcion,
        tabla_afectada=tabla,
        registro_id=reg_id
    )
    db.add(log)
    db.commit()

@router.post("/usuarios", response_model=UserResponse)
async def crear_personal_staff(
    usuario_data: UserCreate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Permite al administrador crear cuentas para nutricionistas y entrenadores.
    """
    check_is_admin(current_user)
    
    usuario_existente = db.query(User).filter(User.email == usuario_data.email).first()
    if usuario_existente:
        raise HTTPException(status_code=400, detail="El correo electrónico ya está registrado")

    hashed_pwd = security.hash_password(usuario_data.password)
    
    nuevo_usuario = User(
        first_name=usuario_data.first_name,
        last_name_paternal=usuario_data.last_name_paternal,
        last_name_maternal=usuario_data.last_name_maternal,
        email=usuario_data.email,
        hashed_password=hashed_pwd,
        role_name=usuario_data.role, 
        role_id=usuario_data.role_id 
    )
    
    db.add(nuevo_usuario)
    db.commit()
    db.refresh(nuevo_usuario)
    
    _log_admin_action(
        db, current_user.id, "REGISTRO_STAFF", 
        f"Se registró a {nuevo_usuario.first_name} ({nuevo_usuario.role_name})",
        "users", nuevo_usuario.id
    )
    
    return nuevo_usuario

@router.get("/staff")
async def listar_personal_staff(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Lista el personal del staff (nutricionistas, entrenadores y otros administradores).
    """
    # 1. Verificación de permisos
    check_is_admin(current_user)
    
    try:
        # 2. Obtener usuarios filtrados (Incluyendo todas las variantes de admin y staff)
        usuarios_db = db.query(User).filter(
            User.role_name.ilike("%admin%"), # Captura admin, administrador, Administrador, etc.
            User.id != current_user.id
        ).all()
        
        # También incluir nutricionistas y entrenadores
        especialistas = db.query(User).filter(
            User.role_name.in_(["nutritionist", "coach", "trainer"]),
            User.id != current_user.id
        ).all()
        
        usuarios_db.extend(especialistas)
        
        # 3. Mapeo manual a diccionario
        return [
            {
                "id": u.id,
                "first_name": u.first_name if u.first_name else "N/A",
                "last_name_paternal": u.last_name_paternal if u.last_name_paternal else "",
                "last_name_maternal": u.last_name_maternal if u.last_name_maternal else "",
                "email": u.email if u.email else "sin@email.com",
                "role_name": u.role_name if u.role_name else "staff",
                "is_active": u.is_active
            } 
            for u in usuarios_db
        ]
    except Exception as e:
        print(f"❌ ERROR LISTAR STAFF: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error en el servidor al obtener personal: {str(e)}"
        )

@router.put("/clientes/{cliente_id}/asignar")
async def asignar_especialistas_a_cliente(
    cliente_id: int,
    nutri_id: int = None,
    trainer_id: int = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Vincula a un cliente con un nutricionista y un entrenador específico.
    """
    check_is_admin(current_user)
    
    cliente = db.query(Client).filter(Client.id == cliente_id).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")
        
    if nutri_id:
        nutri = db.query(User).filter(User.id == nutri_id, User.role_name == "nutritionist").first()
        if not nutri:
            raise HTTPException(status_code=400, detail="ID de nutricionista no válido")
        cliente.assigned_nutri_id = nutri_id
        
    if trainer_id:
        trainer = db.query(User).filter(User.id == trainer_id, User.role_name.in_(["coach", "trainer"])).first()
        if not trainer:
            raise HTTPException(status_code=400, detail="ID de entrenador no válido")
        cliente.assigned_coach_id = trainer_id
        
    db.commit()
    return {"message": f"Especialistas asignados correctamente al cliente {cliente.first_name}"}

@router.put("/staff/{user_id}/password")
async def cambiar_password_staff(
    user_id: int,
    password_data: PasswordUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Permite al administrador cambiar la contraseña de cualquier miembro del staff.
    """
    check_is_admin(current_user)
    usuario = db.query(User).filter(User.id == user_id).first()
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario del staff no encontrado")
        
    usuario.hashed_password = security.hash_password(password_data.new_password)
    db.commit()
    
    _log_admin_action(
        db, current_user.id, "CAMBIO_PASSWORD", 
        f"Se cambió la contraseña de {usuario.first_name} ({usuario.email})",
        "users", usuario.id
    )
    
    return {"message": f"Contraseña de {usuario.first_name} actualizada correctamente"}

@router.put("/staff/{user_id}")
async def actualizar_personal_staff(
    user_id: int,
    usuario_data: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Permite al administrador actualizar los datos básicos de un miembro del staff.
    """
    check_is_admin(current_user)
    
    usuario = db.query(User).filter(User.id == user_id).first()
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario del staff no encontrado")

    # Validar duplicación de email si se intenta cambiar
    if usuario_data.email and usuario_data.email != usuario.email:
        email_existente = db.query(User).filter(User.email == usuario_data.email).first()
        if email_existente:
            raise HTTPException(status_code=400, detail="El nuevo correo electrónico ya está registrado")
        usuario.email = usuario_data.email

    # Actualizar campos opcionales
    if usuario_data.first_name:
        usuario.first_name = usuario_data.first_name
    if usuario_data.last_name_paternal:
        usuario.last_name_paternal = usuario_data.last_name_paternal
    if usuario_data.last_name_maternal:
        usuario.last_name_maternal = usuario_data.last_name_maternal
    if usuario_data.role_name:
        usuario.role_name = usuario_data.role_name
    if usuario_data.role_id:
        usuario.role_id = usuario_data.role_id

    db.commit()
    db.refresh(usuario)
    
    _log_admin_action(
        db, current_user.id, "ACTUALIZACION_STAFF", 
        f"Se actualizaron los datos de {usuario.first_name} ({usuario.email})",
        "users", usuario.id
    )
    
    return {"message": f"Datos de {usuario.first_name} actualizados correctamente"}

@router.put("/staff/{user_id}/status")
async def alternar_estado_staff(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Permite al administrador suspender (dar de baja) o reactivar a un miembro del staff.
    """
    check_is_admin(current_user)
    
    usuario = db.query(User).filter(User.id == user_id).first()
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario del staff no encontrado")

    # Invertir estado
    nuevo_estado = not usuario.is_active
    usuario.is_active = nuevo_estado
    db.commit()
    
    accion = "PERSONAL_SUSPENDIDO" if not nuevo_estado else "PERSONAL_REACTIVADO"
    descripcion = f"Se {'suspendió' if not nuevo_estado else 'reactivó'} la cuenta de {usuario.first_name} ({usuario.email})"
    
    _log_admin_action(
        db, current_user.id, accion, descripcion,
        "users", usuario.id
    )
    
    return {
        "message": descripcion,
        "is_active": nuevo_estado
    }

@router.get("/logs")
async def listar_logs_admin(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Lista los eventos de auditoría administrativa.
    """
    check_is_admin(current_user)
    
    logs = db.query(AuditoriaAdmin).order_by(AuditoriaAdmin.fecha_evento.desc()).limit(100).all()
    
    return [
        {
            "id": log.id,
            "accion": log.accion,
            "descripcion": log.descripcion,
            "fecha": log.fecha_evento,
            "admin_id": log.admin_id
        }
        for log in logs
    ]
