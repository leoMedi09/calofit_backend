from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.user import User
from app.core.security import security
from app.schemas.user import UserCreate
from app.api.routes.auth import get_current_user

router = APIRouter()

@router.post("/registrar", status_code=201)
async def registrar_usuario(usuario_data: UserCreate, db: Session = Depends(get_db)):
    
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
    return nuevo_usuario
    
    
@router.get("/me")
async def leer_mi_perfil(current_user: User = Depends(get_current_user)):
    """Retorna los datos del usuario logueado usando su Token JWT"""
    return {
        "identidad": {
            "nombres": current_user.first_name,
            "apellido_paterno": current_user.last_name_paternal,
            "apellido_materno": current_user.last_name_maternal,
            "email": current_user.email
        },
        "fisico": {
            "edad": current_user.age,
            "peso_kg": current_user.weight,
            "talla_m": current_user.height,
            "condiciones": current_user.medical_conditions
        }
    }