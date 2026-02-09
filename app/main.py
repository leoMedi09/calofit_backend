from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import engine, Base
from app.core import firebase

from app.models import user, client, role, historial 
from app.api import api_router
from app.api.routes.websockets import router as websocket_router

# ✅ REMOVER IMPORTACIÓN DIRECTA - YA ESTÁ EN api_router
# from app.api.routes.clientes import router as clientes_router

Base.metadata.create_all(bind=engine) 

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

# ✅ REMOVER REGISTRO DIRECTO - YA ESTÁ EN api_router
# app.include_router(clientes_router, prefix="/clientes", tags=["clientes"])

@app.get("/")
def read_root():
    return {"message": "Asistente CaloFit Operativo en Gimnasio World Light"}

@app.get("/test")
def test_endpoint():
    return {"status": "OK", "birth_date_field": "working"}