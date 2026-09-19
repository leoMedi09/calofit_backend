import os
import warnings
from dotenv import load_dotenv

_orig_db_url = os.getenv("DATABASE_URL")
load_dotenv(override=True)
if _orig_db_url is not None:
    os.environ["DATABASE_URL"] = _orig_db_url

_WEAK_DEFAULTS = {"TU_CLAVE_PARA_LEY_29733", "changeme", "secret", ""}

class Settings:
    PROJECT_NAME: str = "CaloFit - Gimnasio World Light"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:leomeflo09@localhost/BD_Calofit")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "TU_CLAVE_PARA_LEY_29733")

    def __init__(self) -> None:
        if self.SECRET_KEY in _WEAK_DEFAULTS:
            warnings.warn(
                "SECRET_KEY usa un valor por defecto inseguro. "
                "Generar con: python -c \"import secrets; print(secrets.token_hex(32))\" "
                "y definir en .env antes del despliegue.",
                stacklevel=2,
            )
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_BACKUP_API_KEY: str = os.getenv("GROQ_BACKUP_API_KEY", "")
    GROQ_TIMEOUT_SEC: float = float(os.getenv("GROQ_TIMEOUT_SEC", "180"))
    GROQ_MAX_RETRIES: int = int(os.getenv("GROQ_MAX_RETRIES", "2"))
    CALOFIT_DISABLE_CLASIFICAR_MODO_LLM: bool = os.getenv(
        "CALOFIT_DISABLE_CLASIFICAR_MODO_LLM", ""
    ).strip().lower() in ("1", "true", "yes", "on")
    FATSECRET_CLIENT_ID: str = os.getenv("FATSECRET_CLIENT_ID", "")
    FATSECRET_CLIENT_SECRET: str = os.getenv("FATSECRET_CLIENT_SECRET", "")
    USDA_API_KEY: str = os.getenv("USDA_API_KEY", "")
    DISABLE_FATSECRET: bool = os.getenv("DISABLE_FATSECRET", "").lower() in ("1", "true", "yes")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    RESEND_API_KEY: str = os.getenv("RESEND_API_KEY", "")
    SENDER_EMAIL: str = os.getenv("SENDER_EMAIL", "onboarding@resend.dev")
    
    FIREBASE_API_KEY: str = os.getenv("FIREBASE_API_KEY", "")
    FIREBASE_PROJECT_ID: str = os.getenv("FIREBASE_PROJECT_ID", "calofit-c8c24")
    CLOUDINARY_CLOUD_NAME: str = os.getenv("CLOUDINARY_CLOUD_NAME", "")
    CLOUDINARY_API_KEY: str = os.getenv("CLOUDINARY_API_KEY", "")
    CLOUDINARY_API_SECRET: str = os.getenv("CLOUDINARY_API_SECRET", "")
    FIREBASE_STORAGE_BUCKET: str = os.getenv("FIREBASE_STORAGE_BUCKET", "calofit-c8c24.appspot.com")

settings = Settings()