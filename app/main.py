from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import os
import time
from app.core.database import engine, Base
from app.core import firebase

from app.models import user, client, role, historial
from app.api import api_router


Base.metadata.create_all(bind=engine)

from sqlalchemy import text

with engine.connect() as connection:
    try:
        connection.execute(text("ALTER TABLE clients ADD COLUMN IF NOT EXISTS dni VARCHAR UNIQUE;"))
        connection.execute(text("ALTER TABLE clients ADD COLUMN IF NOT EXISTS workout_type VARCHAR DEFAULT 'Cardio';"))
        connection.execute(text("ALTER TABLE clients ADD COLUMN IF NOT EXISTS session_duration FLOAT DEFAULT 1.0;"))
        connection.execute(text("ALTER TABLE clients ADD COLUMN IF NOT EXISTS nutri_weekly_note TEXT;"))
        connection.execute(text("ALTER TABLE clients ADD COLUMN IF NOT EXISTS fcm_token VARCHAR;"))
        connection.execute(
            text("ALTER TABLE clients ADD COLUMN IF NOT EXISTS notificaciones_activas BOOLEAN DEFAULT TRUE;")
        )
        connection.execute(text("ALTER TABLE clients ADD COLUMN IF NOT EXISTS terms_accepted_at TIMESTAMP;"))
        connection.execute(text("DROP TABLE IF EXISTS platos_recomendados CASCADE;"))
        connection.execute(
            text("""
            CREATE TABLE IF NOT EXISTS chat_historial (
                id SERIAL PRIMARY KEY,
                client_id INTEGER NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
                rol VARCHAR(20) NOT NULL,
                contenido TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
        )
        connection.execute(
            text("CREATE INDEX IF NOT EXISTS idx_chat_historial_client ON chat_historial(client_id, created_at DESC)")
        )
        connection.commit()
        print("Migraciones manuales aplicadas correctamente.")
    except Exception as e:
        print(f"Error en migración manual: {e}")

app = FastAPI(title="CaloFit - Gimnasio World Light API")


@app.middleware("http")
async def _log_lentas(request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = time.perf_counter() - t0
    if dt > 1.0 and request.url.path != "/health":
        logging.getLogger("calofit.timing").warning("%s %s %.2fs", request.method, request.url.path, dt)
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

from app.api.v1 import router as api_v1_router

app.include_router(api_v1_router)

UPLOAD_DIR = "app/uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.on_event("startup")
def iniciar_notificaciones():
    from app.core.notification_scheduler import iniciar_scheduler

    iniciar_scheduler()


@app.get("/")
def read_root():
    return {"message": "Asistente CaloFit Operativo en Gimnasio World Light"}


@app.api_route("/health", methods=["GET", "HEAD"])
def health_check_root():
    return {"status": "OK", "version": "1.0.0"}


@app.get("/test")
def test_endpoint():
    return {"status": "OK", "birth_date_field": "working"}
