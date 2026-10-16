"""Verifica una contrasena contra Firebase Auth desde el servidor (no confia en lo que diga el cliente)."""
import httpx

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger("core.firebase_rest")
_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"


def password_valida_en_firebase(email: str, password: str, uid: str) -> bool:
    """True solo si Firebase acepta email+contrasena Y el usuario devuelto tiene ese uid.
    Sin FIREBASE_API_KEY (o ante cualquier error) devuelve False: falla cerrado."""
    key = getattr(settings, "FIREBASE_API_KEY", "")
    if not key or not uid or not email or not password:
        return False
    try:
        r = httpx.post(_URL, params={"key": key}, json={"email": email, "password": password, "returnSecureToken": False}, timeout=10.0)
    except httpx.HTTPError as e:
        logger.warning("No se pudo consultar Firebase: %s", type(e).__name__)
        return False
    return r.status_code == 200 and r.json().get("localId") == uid
