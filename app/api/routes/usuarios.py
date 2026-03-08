from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.models.user import User
from app.core.security import security
from app.schemas.user import UserCreate
from app.api.routes.auth import get_current_user
from app.core.local_storage import local_storage
from datetime import datetime 

router = APIRouter()

@router.post("/perfil/foto")
async def subir_foto_perfil(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Sube una foto de perfil localmente y actualiza la URL en la base de datos"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    # 1. Leer archivo
    file_bytes = await file.read()
    
    # 2. Borrar foto anterior
    if current_user.profile_picture_url:
        local_storage.delete_file(current_user.profile_picture_url)
    
    # 3. Guardar Localmente
    relative_path = local_storage.save_file(file_bytes, file.filename)
    public_url = local_storage.get_public_url(relative_path)
    
    # 4. Actualizar base de datos
    current_user.profile_picture_url = public_url
    db.commit()
    
    return {"message": "Foto de perfil actualizada exitosamente", "url": public_url}

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
            "email": current_user.email,
            "foto_perfil": current_user.profile_picture_url # ✅ Añadido para el rediseño premium
        },
        "fisico": {
            "edad": getattr(current_user, "age", None),
            "peso_kg": getattr(current_user, "weight", None),
            "talla_m": getattr(current_user, "height", None),
            "condiciones": getattr(current_user, "medical_conditions", [])
        }
    }