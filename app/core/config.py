import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

class Settings:
    PROJECT_NAME: str = "CaloFit - Gimnasio World Light"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:leomeflo09@localhost/BD_Calofit")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
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
    # Cloudinary Configuration
    CLOUDINARY_CLOUD_NAME: str = os.getenv("CLOUDINARY_CLOUD_NAME", "")
    CLOUDINARY_API_KEY: str = os.getenv("CLOUDINARY_API_KEY", "")
    CLOUDINARY_API_SECRET: str = os.getenv("CLOUDINARY_API_SECRET", "")
    FIREBASE_STORAGE_BUCKET: str = os.getenv("FIREBASE_STORAGE_BUCKET", "calofit-c8c24.appspot.com")

settings = Settings()