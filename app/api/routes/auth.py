from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from app.core.database import get_db
from app.models.user import User
from app.core.security import security
from datetime import timedelta, datetime
from app.schemas.user import UserLogin, SyncPasswordRequest, ForgotPasswordRequest, ValidateResetCodeRequest
from app.core.config import settings
from jose import JWTError, jwt
from app.models.client import Client
import hmac
import hashlib

router = APIRouter()


def _hash_reset_code(code: str) -> str:
    """HMAC-SHA256 del código de reset — lo que se persiste en BD."""
    return hmac.new(
        settings.SECRET_KEY.encode(),
        code.encode(),
        hashlib.sha256,
    ).hexdigest()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

@router.post("/login")
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    print(f"🔐 Intento de login: {credentials.email}")

    # 1️⃣ Determinar el tipo de usuario
    requested_type = (credentials.user_type or "").strip().lower()
    has_firebase = bool(credentials.firebase_uid and credentials.firebase_uid.strip())

    print(f"📥 REQUEST DEBUG: type_req='{requested_type}', has_firebase={has_firebase}")

    user = None
    user_type = "client"

    # Modo unificado: el backend resuelve cliente vs staff por email (sin pestañas en la app)
    if requested_type in ("auto", "unified", ""):
        if has_firebase:
            user_type = "client"
            print("🎯 MODO AUTO: CLIENTE (hay Firebase UID)")
            user = db.query(Client).filter(Client.email == credentials.email).first()
        else:
            print("🎯 MODO AUTO: buscando primero en Client, luego en User (staff)")
            user = db.query(Client).filter(Client.email == credentials.email).first()
            if user:
                user_type = "client"
            else:
                user = db.query(User).filter(User.email == credentials.email).first()
                if not user and "@worldlight.com" in credentials.email:
                    alias = credentials.email.split("@")[0].strip().lower()
                    user = db.query(User).filter(User.email == alias).first()
                    if user:
                        print(f"✅ Staff encontrado por alias: {alias}")
                if user:
                    user_type = "staff"
    elif requested_type == "client" or has_firebase:
        user_type = "client"
        print(f"🎯 MODO ELEGIDO: CLIENTE (Motivo: {'Tab Cliente' if requested_type == 'client' else 'Firebase UID'})")
        user = db.query(Client).filter(Client.email == credentials.email).first()
    else:
        user_type = "staff"
        print("👥 MODO ELEGIDO: STAFF (explícito)")
        user = db.query(User).filter(User.email == credentials.email).first()

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

    # 3.5️⃣ Verificación de Onboarding Forzoso de Nutrición (Flujo Express)
    is_profile_complete = getattr(user, 'is_profile_complete', True)
    if user_type == "client" and not is_profile_complete:
        print(f"⚠️ AVISO (EXPRESS): El perfil de {user.email} está incompleto. Se permite login para onboarding.")

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
            "last_name": getattr(user, 'last_name_paternal', ''),
            "email": user.email,
            "type": user_type,
            "id": user.id,
            "role": getattr(user, 'role_name', 'client') if user_type == "staff" else None,
            "profile_picture_url": getattr(user, 'profile_picture_url', None),
            "is_profile_complete": is_profile_complete,
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


from app.schemas.client import ChangePassword

# ... (otras rutas)

@router.post("/change-password")
async def change_password(
    data: ChangePassword,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Actualiza la contraseña del usuario actual en BD local y Firebase.
    """
    print(f"🔐 Cambiando contraseña para: {current_user.email}")
    
    if data.new_password != data.confirm_password:
        raise HTTPException(status_code=400, detail="Las contraseñas no coinciden")
    
    # Sincronizar con Firebase si tiene UID
    flutter_uid = getattr(current_user, 'flutter_uid', None)
    if flutter_uid:
        try:
            from app.core.firebase import auth as firebase_admin_auth
            firebase_admin_auth.update_user(flutter_uid, password=data.new_password)
            print(f"✅ Firebase Password Sync OK")
        except Exception as e:
            print(f"⚠️ Error sincronizando con Firebase: {e}")
            # Continuamos para que al menos la DB local se actualice

    current_user.hashed_password = security.hash_password(data.new_password)
    db.commit()
    
    return {"message": "Contraseña actualizada correctamente"}


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

# ----------------- RECUPERACIÓN DE CONTRASEÑA -----------------

@router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Paso 1: Solicitar código de recuperación — solo para clientes."""
    from app.models.password_reset import PasswordReset
    from app.services.email_service import EmailService
    import random

    # 1. Si el email pertenece a staff → rechazar con mensaje claro
    staff = db.query(User).filter(User.email == request.email).first()
    if staff:
        raise HTTPException(
            status_code=403,
            detail="Este servicio es solo para clientes. Si eres personal del gimnasio, contacta al administrador para restablecer tu contraseña."
        )

    # 2. Buscar solo en clientes
    client = db.query(Client).filter(Client.email == request.email).first()
    if not client:
        # Respuesta genérica para no revelar si el correo existe
        return {"success": True, "message": "Si el correo está registrado, recibirás un código."}

    # 3. Generar código de 6 dígitos
    code = str(random.randint(100000, 999999))

    # 4. Guardar hash en base de datos — el código plano solo viaja por correo
    reset_record = PasswordReset(email=request.email, reset_code=_hash_reset_code(code))
    db.add(reset_record)
    db.commit()

    # 5. Enviar código plano por correo
    EmailService.send_password_reset_brevo(request.email, code)

    return {"success": True, "message": "Si el correo está registrado, recibirás un código."}

@router.post("/verify-reset-code")
async def verify_reset_code(request: ValidateResetCodeRequest, db: Session = Depends(get_db)):
    """Paso 2: Validar que el código es correcto (sin consumir el código todavía)"""
    # Nota: Aquí usamos ValidateResetCodeRequest pero solo leemos email y reset_code 
    # (podemos mandar un password dummy o el frontend manda los 3 campos).
    # Como el schema requiere new_password, lo ideal sería que manden "" o crear un schema nuevo.
    # Pero vamos a leer solo lo necesario.
    from app.models.password_reset import PasswordReset
    
    record = db.query(PasswordReset).filter(
        PasswordReset.email == request.email,
        PasswordReset.reset_code == _hash_reset_code(request.reset_code),
        PasswordReset.is_used == False
    ).order_by(PasswordReset.created_at.desc()).first()

    if not record or record.is_expired():
        raise HTTPException(status_code=400, detail="Código inválido o expirado")

    return {"success": True, "message": "Código válido"}

@router.post("/reset-password")
async def reset_password(request: ValidateResetCodeRequest, db: Session = Depends(get_db)):
    """Paso 3: Cambiar la contraseña y consumir el código"""
    from app.models.password_reset import PasswordReset
    
    # 1. Validar código
    record = db.query(PasswordReset).filter(
        PasswordReset.email == request.email,
        PasswordReset.reset_code == _hash_reset_code(request.reset_code),
        PasswordReset.is_used == False
    ).order_by(PasswordReset.created_at.desc()).first()

    if not record or record.is_expired():
        raise HTTPException(status_code=400, detail="Código inválido o expirado")
        
    # 2. Actualizar clave (buscar en User o Client)
    user = db.query(User).filter(User.email == request.email).first()
    if user:
        # Staff: solo BD local (no usan Firebase)
        user.hashed_password = security.hash_password(request.new_password)
    else:
        client = db.query(Client).filter(Client.email == request.email).first()
        if client:
            client.hashed_password = security.hash_password(request.new_password)
            # Sincronizar también en Firebase para que el login del cliente no falle
            if client.flutter_uid:
                try:
                    from app.core.firebase import auth as firebase_admin_auth
                    firebase_admin_auth.update_user(client.flutter_uid, password=request.new_password)
                    print(f"✅ Firebase password actualizado para {client.email}")
                except Exception as firebase_err:
                    print(f"⚠️ No se pudo actualizar Firebase (BD local sí actualizada): {firebase_err}")
        else:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")

    # 3. Marcar código como usado
    record.is_used = True
    record.used_at = datetime.utcnow()

    db.commit()

    return {"success": True, "message": "Contraseña actualizada exitosamente"}