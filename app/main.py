from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from app.core.database import engine, Base
from app.core import firebase

from app.models import user, client, role, historial 
from app.api import api_router
from app.api.routes.websockets import router as websocket_router


# from app.api.routes.clientes import router as clientes_router

Base.metadata.create_all(bind=engine) 

# MIGRACIONES MANUALES: Columnas añadidas post-creación inicial
from sqlalchemy import text
with engine.connect() as connection:
    try:
        connection.execute(text("ALTER TABLE clients ADD COLUMN IF NOT EXISTS dni VARCHAR UNIQUE;"))
        connection.execute(text("ALTER TABLE clients ADD COLUMN IF NOT EXISTS workout_type VARCHAR DEFAULT 'Cardio';"))
        connection.execute(text("ALTER TABLE clients ADD COLUMN IF NOT EXISTS session_duration FLOAT DEFAULT 1.0;"))
        connection.execute(text("ALTER TABLE clients ADD COLUMN IF NOT EXISTS nutri_weekly_note TEXT;"))
        connection.execute(text("DROP TABLE IF EXISTS platos_recomendados CASCADE;"))
        # Memoria conversacional persistida
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS chat_historial (
                id SERIAL PRIMARY KEY,
                client_id INTEGER NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
                rol VARCHAR(20) NOT NULL,
                contenido TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """))
        connection.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_chat_historial_client "
            "ON chat_historial(client_id, created_at DESC)"
        ))
        connection.commit()
        print("Migraciones manuales aplicadas correctamente.")
    except Exception as e:
        print(f"Error en migración manual: {e}")

app = FastAPI(title="CaloFit - Gimnasio World Light API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(websocket_router, tags=["WebSockets"])

# Incluir router general de API v1 (PASO 6)
from app.api.v1 import router as api_v1_router
app.include_router(api_v1_router)

# Crear directorio de subidas si no existe
UPLOAD_DIR = "app/uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Servir archivos estáticos
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# ✅ REMOVER REGISTRO DIRECTO - YA ESTÁ EN api_router
# app.include_router(clientes_router, prefix="/clientes", tags=["clientes"])

@app.get("/")
def read_root():
    return {"message": "Asistente CaloFit Operativo en Gimnasio World Light"}


@app.api_route("/health", methods=["GET", "HEAD"])
def health_check_root():
    return {"status": "OK", "version": "1.0.0"}


@app.get("/test")
def test_endpoint():
    return {"status": "OK", "birth_date_field": "working"}