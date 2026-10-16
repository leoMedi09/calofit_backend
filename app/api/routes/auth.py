from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from app.core.database import get_db
from app.core.rate_limit import limitar
from app.models.user import User
from app.core.security import security
from datetime import timedelta, datetime
from app.schemas.user import UserLogin, ForgotPasswordRequest, ValidateResetCodeRequest
from app.core.firebase_rest import password_valida_en_firebase
from app.core.config import settings
from jose import JWTError, jwt
from app.models.client import Client
import hmac
import hashlib
from app.core.logging_config import get_logger

logger = get_logger("api.auth")

router = APIRouter()


MAX_INTENTOS_CODIGO = 5


def _validar_codigo_reset(db, email: str, code: str):
    """Devuelve el codigo vigente del correo si coincide. Solo vale el ultimo codigo pedido y cada fallo
    cuenta: al 5.o intento fallido el codigo queda inutilizable (hay que pedir otro)."""
    from app.models.password_reset import PasswordReset

    record = (
        db.query(PasswordReset)
        .filter(PasswordReset.email == email, PasswordReset.is_used == False)
        .order_by(PasswordReset.created_at.desc())
        .first()
    )
    if not record or record.is_expired() or (record.intentos or 0) >= MAX_INTENTOS_CODIGO:
        raise HTTPException(status_code=400, detail="Código inválido o expirado")
    if not hmac.compare_digest(record.reset_code, _hash_reset_code(code)):
        record.intentos = (record.intentos or 0) + 1
        db.commit()
        raise HTTPException(status_code=400, detail="Código inválido o expirado")
    return record


def _hash_reset_code(code: str) -> str:
    """HMAC-SHA256 del código de reset — lo que se persiste en BD."""
    return hmac.new(
        settings.SECRET_KEY.encode(),
        code.encode(),
        hashlib.sha256,
    ).hexdigest()


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")


@router.post("/login", dependencies=[Depends(limitar("login", 30, 60))])
def login(credentials: UserLogin, db: Session = Depends(get_db)):

    requested_type = (credentials.user_type or "").strip().lower()
    has_firebase = bool(credentials.firebase_uid and credentials.firebase_uid.strip())

    print(f"📥 REQUEST DEBUG: type_req='{requested_type}', has_firebase={has_firebase}")

    user = None
    user_type = "client"

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
        raise HTTPException(status_code=401, detail="Correo o contraseña incorrectos")

    password_correct_locally = security.verify_password(credentials.password, user.hashed_password)

    if not password_correct_locally:
        # Contrasena cambiada en Firebase y desincronizada: se acepta SOLO si Firebase la confirma para ese uid.
        uid_ok = isinstance(user, Client) and (not user.flutter_uid or user.flutter_uid == credentials.firebase_uid)
        if uid_ok and credentials.firebase_uid and password_valida_en_firebase(user.email, credentials.password, credentials.firebase_uid):
            user.hashed_password = security.hash_password(credentials.password)
            if not user.flutter_uid:
                user.flutter_uid = credentials.firebase_uid
            db.commit()
            db.refresh(user)
        else:
            raise HTTPException(status_code=401, detail="Correo o contraseña incorrectos")

    if credentials.firebase_uid and hasattr(user, "flutter_uid"):
        if user.flutter_uid != credentials.firebase_uid:
            user.flutter_uid = credentials.firebase_uid
            db.commit()

    is_profile_complete = getattr(user, "is_profile_complete", True)
    if user_type == "client" and not is_profile_complete:
        pass

    expires_delta = timedelta(days=7) if credentials.remember_me else timedelta(hours=24)
    access_token = security.create_access_token(
        data={
            "sub": user.email,
            "user_id": user.id,
            "type": user_type,
            "role": getattr(user, "role_name", "client") if user_type == "staff" else "client",
        },
        expires_delta=expires_delta,
    )

    response_data = {
        "access_token": access_token,
        "token_type": "bearer",
        "firebase_uid": user.flutter_uid if hasattr(user, "flutter_uid") else None,
        "user_info": {
            "name": user.first_name,
            "last_name": getattr(user, "last_name_paternal", ""),
            "email": user.email,
            "type": user_type,
            "id": user.id,
            "role": getattr(user, "role_name", "client") if user_type == "staff" else None,
            "profile_picture_url": getattr(user, "profile_picture_url", None),
            "is_profile_complete": is_profile_complete,
        },
    }
    logger.info("login ok id=%s tipo=%s", user.id, user_type)
    return response_data


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido o expirado",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])

        email: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        user_type: str = payload.get("type")


        if email is None or user_type is None or user_id is None:
            raise credentials_exception

    except JWTError as e:
        raise credentials_exception

    if user_type == "staff":
        user = db.query(User).filter(User.id == user_id).first()
    else:
        user = db.query(Client).filter(Client.id == user_id).first()

    if user is None:
        print(f"❌ Usuario no encontrado en BD")
        raise credentials_exception

    return user


def get_current_staff(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token inválido o expirado",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        user_type: str = payload.get("type")

        if email is None or user_type is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    if user_type != "staff":
        raise HTTPException(status_code=403, detail="Acceso denegado: solo para personal")

    user = db.query(User).filter(User.email == email).first()

    if user is None:
        raise credentials_exception

    return user






from app.schemas.client import ChangePassword


@router.post("/change-password")
def change_password(data: ChangePassword, current_user=Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Actualiza la contraseña del usuario actual en BD local y Firebase.
    """

    if data.new_password != data.confirm_password:
        raise HTTPException(status_code=400, detail="Las contraseñas no coinciden")

    flutter_uid = getattr(current_user, "flutter_uid", None)
    if flutter_uid:
        try:
            from app.core.firebase import auth as firebase_admin_auth

            firebase_admin_auth.update_user(flutter_uid, password=data.new_password)
        except Exception as e:
            print(f"⚠️ Error sincronizando con Firebase: {e}")

    current_user.hashed_password = security.hash_password(data.new_password)
    db.commit()

    return {"message": "Contraseña actualizada correctamente"}




@router.post("/forgot-password", dependencies=[Depends(limitar("forgot", 10, 600))])
def forgot_password(request: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Paso 1: Solicitar código de recuperación — solo para clientes."""
    from app.models.password_reset import PasswordReset
    from app.services.email_service import EmailService
    import random

    staff = db.query(User).filter(User.email == request.email).first()
    if staff:
        raise HTTPException(
            status_code=403,
            detail="Este servicio es solo para clientes. Si eres personal del gimnasio, contacta al administrador para restablecer tu contraseña.",
        )

    client = db.query(Client).filter(Client.email == request.email).first()
    if not client:
        return {"success": True, "message": "Si el correo está registrado, recibirás un código."}

    code = str(random.randint(100000, 999999))

    reset_record = PasswordReset(email=request.email, reset_code=_hash_reset_code(code))
    db.add(reset_record)
    db.commit()

    EmailService.send_password_reset_brevo(request.email, code)

    return {"success": True, "message": "Si el correo está registrado, recibirás un código."}


@router.post("/verify-reset-code", dependencies=[Depends(limitar("verify", 20, 600))])
def verify_reset_code(request: ValidateResetCodeRequest, db: Session = Depends(get_db)):
    """Paso 2: Validar que el código es correcto (sin consumir el código todavía)"""
    from app.models.password_reset import PasswordReset

    record = _validar_codigo_reset(db, request.email, request.reset_code)

    return {"success": True, "message": "Código válido"}


@router.post("/reset-password", dependencies=[Depends(limitar("reset", 20, 600))])
def reset_password(request: ValidateResetCodeRequest, db: Session = Depends(get_db)):
    """Paso 3: Cambiar la contraseña y consumir el código"""
    from app.models.password_reset import PasswordReset

    record = _validar_codigo_reset(db, request.email, request.reset_code)

    user = db.query(User).filter(User.email == request.email).first()
    if user:
        user.hashed_password = security.hash_password(request.new_password)
    else:
        client = db.query(Client).filter(Client.email == request.email).first()
        if client:
            client.hashed_password = security.hash_password(request.new_password)
            if client.flutter_uid:
                try:
                    from app.core.firebase import auth as firebase_admin_auth

                    firebase_admin_auth.update_user(client.flutter_uid, password=request.new_password)
                except Exception as firebase_err:
                    print(f"⚠️ No se pudo actualizar Firebase (BD local sí actualizada): {firebase_err}")
        else:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")

    record.is_used = True
    record.used_at = datetime.utcnow()

    db.commit()

    return {"success": True, "message": "Contraseña actualizada exitosamente"}
