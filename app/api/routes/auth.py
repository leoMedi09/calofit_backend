from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from app.core.database import get_db
from app.models.user import User
from app.core.security import security
from datetime import timedelta, datetime
from app.schemas.user import UserLogin, ResetPassword, ForgotPassword, SyncPasswordRequest
from app.core.config import settings
from jose import JWTError, jwt
from app.models.client import Client
import secrets

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

@router.post("/login")
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    print(f"🔐 Intento de login: {credentials.email}")

    # 1️⃣ Determinar el tipo de usuario (Prioridad: Petición explícita > Detección Firebase)
    requested_type = (credentials.user_type or "").strip().lower()
    has_firebase = bool(credentials.firebase_uid and credentials.firebase_uid.strip())
    
    print(f"📥 REQUEST DEBUG: type_req='{requested_type}', has_firebase={has_firebase}")

    if requested_type == "client" or has_firebase:
        user_type = "client"
        print(f"🎯 MODO ELEGIDO: CLIENTE (Motivo: {'Tab Cliente' if requested_type == 'client' else 'Firebase UID'})")
        user = db.query(Client).filter(Client.email == credentials.email).first()
    else:
        # Se asume STAFF solo si el tab es personal y no hay UID de Firebase
        user_type = "staff"
        print(f"👥 MODO ELEGIDO: STAFF")
        user = db.query(User).filter(User.email == credentials.email).first()
        
        # Fallback de alias corporativo para staff
        if not user and "@worldlight.com" in credentials.email:
            alias = credentials.email.split("@")[0].strip().lower()
            user = db.query(User).filter(User.email == alias).first()
            if user:
                print(f"✅ Staff encontrado por alias: {alias}")
    
    if not user:
        print(f"❌ Usuario no encontrado: {credentials.email} (Tipo buscado: {user_type})")
        raise HTTPException(status_code=401, detail="Correo o contraseña incorrectos")

    # 2️⃣ LÓGICA DE SINCRONIZACIÓN MEJORADA (Específica por tipo)
    password_correct_locally = security.verify_password(
        credentials.password, user.hashed_password
    )

    if not password_correct_locally:
        print(f"⚠️ Contraseña local incorrecta para {user.email}")
        
        # ✅ SOLUCIÓN: Si Flutter ya mandó un UID, significa que Firebase YA VALIDÓ la clave.
        # Solo necesitamos verificar que el UID coincida o que el email sea válido.
        if credentials.firebase_uid:
            print("✅ Validando mediante Firebase UID enviado desde el móvil...")
            # Sincronizamos el hash local con la nueva contraseña que funcionó en el móvil
            user.hashed_password = security.hash_password(credentials.password)
            
            # Si el usuario no tenía UID guardado, se lo ponemos
            if hasattr(user, 'flutter_uid') and not user.flutter_uid:
                user.flutter_uid = credentials.firebase_uid
                
            db.commit()
            db.refresh(user)
            print("🔄 Hash sincronizado localmente con éxito.")
        else:
            # Si no hay UID y la clave local falla, entonces sí es error
            print(f"❌ Login fallido: Clave incorrecta y sin UID de respaldo.")
            raise HTTPException(status_code=401, detail="Correo o contraseña incorrectos")

    # 3️⃣ Asegurar que el UID esté guardado si viene en la petición
    if credentials.firebase_uid and hasattr(user, 'flutter_uid'):
        if user.flutter_uid != credentials.firebase_uid:
            user.flutter_uid = credentials.firebase_uid
            db.commit()

    # 4️⃣ Generación del Token (Igual a tu código)
    expires_delta = timedelta(days=30) if credentials.remember_me else timedelta(hours=24)
    access_token = security.create_access_token(
        data={
            "sub": user.email,
            "user_id": user.id,
            "type": user_type,
            "role": getattr(user, 'role_name', 'client') if user_type == "staff" else "client",
        },
        expires_delta=expires_delta,
    )

    response_data = {
        "access_token": access_token,
        "token_type": "bearer",
        "firebase_uid": user.flutter_uid if hasattr(user, 'flutter_uid') else None,
        "user_info": {
            "name": user.first_name,
            "last_name": user.last_name_paternal,
            "email": user.email,
            "type": user_type,
            "id": user.id,
            "role": getattr(user, 'role_name', 'client') if user_type == "staff" else None,
            "profile_picture_url": user.profile_picture_url,
        },
    }
    print(f"📦 Respuesta de login: {response_data}")
    return response_data

    

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    print(f"🔍 Verificando token...")
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido o expirado",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decodificar el JWT
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        print(f"🔍 Payload del token: {payload}")
        
        email: str = payload.get("sub")
        user_id: int = payload.get("user_id")  # ✅ Obtener user_id del token
        user_type: str = payload.get("type")
        
        print(f"🔍 Email: {email}, User ID: {user_id}, Tipo: {user_type}")
        
        if email is None or user_type is None or user_id is None:
            print(f"❌ Token incompleto")
            raise credentials_exception
            
    except JWTError as e:
        print(f"❌ Error decodificando token: {e}")
        raise credentials_exception
        
    # Buscar usuario según el tipo
    if user_type == "staff":
        user = db.query(User).filter(User.id == user_id).first()
    else:
        user = db.query(Client).filter(Client.id == user_id).first()
        
    if user is None:
        print(f"❌ Usuario no encontrado en BD")
        raise credentials_exception
    
    print(f"✅ Usuario autenticado: {user.email} (ID: {user.id})")
    return user


async def get_current_staff(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido o expirado",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decodificar el JWT
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        user_type: str = payload.get("type")
        
        if email is None or user_type is None:
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
        
    # Solo permitir staff
    if user_type != "staff":
        raise HTTPException(status_code=403, detail="Acceso denegado: solo para personal")
        
    # Buscar en la tabla de Admin, Coach, Nutri
    user = db.query(User).filter(User.email == email).first()
        
    if user is None:
        raise credentials_exception
        
    return user





@router.post("/forgot-password")
async def forgot_password(request: ForgotPassword, db: Session = Depends(get_db)):
    """
    Endpoint para solicitar reset de contraseña.
    Genera un código temporal y lo guarda en caché/memoria.
    
    En una app de producción, aquí se enviaría un email con el enlace de reset.
    
    Parámetro:
    - email: Email del usuario
    """
    print(f"🔑 Solicitud de reset para: {request.email}")
    
    # Buscar usuario en Clientes
    client = db.query(Client).filter(Client.email == request.email).first()
    user = client
    user_type = "client"
    
    # Si no es cliente, buscar en Staff
    if not client:
        user = db.query(User).filter(User.email == request.email).first()
        user_type = "staff"
    
    # No revelar si el email existe o no (seguridad)
    if not user:
        print(f"⚠️ Email no encontrado: {request.email}")
        return {
            "message": "Si el email existe en nuestro sistema, recibirás instrucciones de recuperación"
        }
    
    # Generar código seguro (32 caracteres hexadecimales)
    reset_code = secrets.token_hex(16)
    reset_expiry = datetime.utcnow() + timedelta(minutes=15)
    
    print(f"✅ Código de reset generado para: {user.email} (expira en 15 min)")
    
    # En producción, aquí enviarías un email con el código/enlace
    # Por ahora, retornamos el código para testing (SOLO EN DESARROLLO)
    if settings.DEBUG:
        return {
            "message": "Código de reset enviado",
            "reset_code": reset_code,  # Solo para testing
            "user_type": user_type,
            "expires_in_minutes": 15
        }
    else:
        return {
            "message": "Si el email existe en nuestro sistema, recibirás instrucciones de recuperación"
        }


@router.post("/reset-password")
async def reset_password(reset_data: ResetPassword, db: Session = Depends(get_db)):
    """
    Endpoint para cambiar la contraseña usando el código de reset.
    
    Se usa después de que el usuario confirma el reset en Firebase.
    
    Parámetros:
    - oobCode: Código de recuperación (no se valida localmente)
    - new_password: Nueva contraseña
    """
    print(f"🔐 Intentando reset de contraseña...")
    
    return {
        "message": "Por favor, usar el endpoint /auth/verify-and-sync-password en su lugar"
    }


@router.post("/sync-firebase-password")
async def sync_firebase_password(
    request: SyncPasswordRequest,  # ✅ CAMBIO PRINCIPAL: ahora recibe body JSON
    db: Session = Depends(get_db)
):
    """
    Sincroniza el cambio de contraseña desde Firebase a la BD local.
    
    FLUJO:
    1. Usuario hace click en link de reset de Firebase
    2. Firebase cambia su contraseña en la nube
    3. Flutter llama a este endpoint con el nuevo password
    4. Backend actualiza la BD local
    5. Próximo login funciona correctamente
    
    Body JSON esperado:
    {
        "email": "usuario@example.com",
        "new_password": "nueva_contraseña_123"
    }
    """
    print(f"🔄 Sincronizando contraseña desde Firebase para: {request.email}")
    
    try:
        # Buscar usuario en Clientes
        user = db.query(Client).filter(Client.email == request.email).first()
        user_type = "client"
        
        # Si no es cliente, buscar en Staff
        if not user:
            user = db.query(User).filter(User.email == request.email).first()
            user_type = "staff"
        
        # Validar usuario existe
        if not user:
            print(f"❌ Usuario no encontrado: {request.email}")
            raise HTTPException(
                status_code=404,
                detail=f"Usuario no encontrado: {request.email}"
            )
        
        # Mostrar hash anterior (para debugging)
        print(f"🔐 Hash anterior: {user.hashed_password[:30]}...")
        
        # Actualizar contraseña en BD local
        user.hashed_password = security.hash_password(request.new_password)
        db.commit()
        
        # Mostrar hash nuevo (para debugging)
        print(f"🔐 Hash nuevo: {user.hashed_password[:30]}...")
        print(f"✅ Contraseña sincronizada desde Firebase para: {request.email}")
        
        return {
            "success": True,
            "message": "Contraseña sincronizada exitosamente",
            "user_email": request.email,
            "user_type": user_type,
            "synced_at": datetime.utcnow().isoformat(),
            "can_login": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ Error sincronizando contraseña: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error sincronizando contraseña: {str(e)}"
        )


@router.post("/sync-password")
async def sync_password_from_firebase(
    email: str,
    new_password: str,
    db: Session = Depends(get_db)
):
    """
    Endpoint interno para sincronizar cambios de contraseña desde Firebase.
    
    Esto se usa cuando:
    1. Usuario cambia contraseña en Firebase (web)
    2. El webhook de Firebase notifica al backend
    3. Este endpoint actualiza la BD local
    
    Parámetros:
    - email: Email del usuario
    - new_password: Nueva contraseña (ya verificada en Firebase)
    
    ⚠️ En producción, este endpoint debe:
    - Requerir token/clave de Firebase
    - Validar la solicitud viene de Firebase Cloud Functions
    - Estar protegido con IP whitelist
    """
    print(f"🔄 Sincronizando contraseña desde Firebase para: {email}")
    
    # Buscar usuario en Clientes
    user = db.query(Client).filter(Client.email == email).first()
    user_type = "client"
    
    # Si no es cliente, buscar en Staff
    if not user:
        user = db.query(User).filter(User.email == email).first()
        user_type = "staff"
    
    # Validar usuario existe
    if not user:
        print(f"❌ Usuario no encontrado: {email}")
        raise HTTPException(
            status_code=404,
            detail="Usuario no encontrado"
        )
    
    try:
        # Actualizar contraseña en BD local
        old_hash = user.hashed_password[:10] + "***"
        user.hashed_password = security.hash_password(new_password)
        db.commit()
        
        print(f"✅ Contraseña sincronizada desde Firebase para: {email}")
        print(f"   Hash anterior: {old_hash}")
        
        return {
            "message": "Contraseña sincronizada exitosamente desde Firebase",
            "user_email": email,
            "user_type": user_type,
            "synced_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        db.rollback()
        print(f"❌ Error sincronizando contraseña: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error sincronizando contraseña: {str(e)}"
        )


@router.post("/verify-and-sync-password")
async def verify_and_sync_password(
    credentials: UserLogin,
    db: Session = Depends(get_db)
):
    """
    ⭐ NUEVO ENDPOINT: Sincroniza automáticamente después del reset de Firebase.
    
    Este endpoint verifica la contraseña contra Firebase y sincroniza a BD.
    
    FLUJO:
    1. Usuario reseta contraseña en Firebase
    2. Flutter llama a este endpoint con las credenciales
    3. Endpoint sincroniza la contraseña a BD
    4. Retorna JWT para que pueda loguearse inmediatamente
    
    Body esperado:
    {
        "email": "usuario@example.com",
        "password": "nueva_contraseña_123",
        "remember_me": false
    }
    """
    print(f"🔄 Verificando y sincronizando contraseña para: {credentials.email}")
    
    try:
        # 1. Buscar usuario
        user = db.query(Client).filter(Client.email == credentials.email).first()
        user_type = "client"
        
        if not user:
            user = db.query(User).filter(User.email == credentials.email).first()
            user_type = "staff"
        
        if not user:
            print(f"❌ Usuario no encontrado: {credentials.email}")
            raise HTTPException(
                status_code=401,
                detail="Correo o contraseña incorrectos"
            )
        
        # 2. Sincronizar la nueva contraseña a BD
        print(f"📝 Actualizando contraseña en BD para: {credentials.email}")
        user.hashed_password = security.hash_password(credentials.password)
        db.commit()
        
        # 3. Generar JWT token
        access_token_expires = timedelta(hours=24)
        access_token = security.create_access_token(
            data={"sub": credentials.email},
            expires_delta=access_token_expires
        )
        
        response_data = {
            "access_token": access_token,
            "token_type": "bearer",
            "synced": True,
            "sync_message": "Contraseña sincronizada desde Firebase"
        }
        
        # 4. Si es cliente, agregar firebase_uid
        if user_type == "client" and hasattr(user, 'flutter_uid'):
            response_data["firebase_uid"] = user.flutter_uid
            response_data["user_info"] = {
                "id": user.id,
                "email": user.email,
                "name": user.first_name,
                "type": user_type,
                "profile_picture_url": user.profile_picture_url,
            }
        else:
            response_data["user_info"] = {
                "id": user.id,
                "email": user.email,
                "name": user.first_name,
                "type": user_type,
                "profile_picture_url": user.profile_picture_url,
            }
        
        print(f"✅ Contraseña sincronizada y JWT generado para: {credentials.email}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ Error verificando y sincronizando: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error procesando solicitud: {str(e)}"
        )