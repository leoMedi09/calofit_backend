import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

class Settings:
    PROJECT_NAME: str = "CaloFit - Gimnasio World Light"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:leomeflo09@localhost/BD_Calofit")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "TU_CLAVE_PARA_LEY_29733")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # Email Configuration (Resend)
    RESEND_API_KEY: str = os.getenv("RESEND_API_KEY", "")
    SENDER_EMAIL: str = os.getenv("SENDER_EMAIL", "onboarding@resend.dev")
    
    # Firebase Configuration
    FIREBASE_API_KEY: str = os.getenv("FIREBASE_API_KEY", "")
    FIREBASE_PROJECT_ID: str = os.getenv("FIREBASE_PROJECT_ID", "calofit-c8c24")
    FIREBASE_STORAGE_BUCKET: str = os.getenv("FIREBASE_STORAGE_BUCKET", "calofit-c8c24.appspot.com")

settings = Settings()