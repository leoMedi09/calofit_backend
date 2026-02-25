from fastapi import APIRouter
from .routes import usuarios, ejercicios, nutricion, auth, clientes, asistente, dashboard, admin, alertas, balance, alimentos, nutricionista, copiloto

api_router = APIRouter()


api_router.include_router(auth.router, prefix="/auth", tags=["Autenticación"])
api_router.include_router(usuarios.router, prefix="/usuarios", tags=["Usuarios"])
api_router.include_router(clientes.router, prefix="/clientes", tags=["Clientes"])
api_router.include_router(ejercicios.router, prefix="/ejercicios", tags=["Ejercicios"])
api_router.include_router(nutricion.router, prefix="/nutricion", tags=["Nutrición"])
api_router.include_router(asistente.router, prefix="/asistente", tags=["Asistente Cliente"])
api_router.include_router(copiloto.router, prefix="/copiloto", tags=["Asistente Staff (Copiloto)"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["Dashboard"])
api_router.include_router(admin.router, prefix="/admin", tags=["Administración"])
api_router.include_router(alertas.router, prefix="/alertas", tags=["Alertas de Salud"])
api_router.include_router(balance.router, prefix="/balance", tags=["Mi Balance"])
api_router.include_router(alimentos.router, prefix="/alimentos", tags=["Detalle de Alimentos"])
api_router.include_router(nutricionista.router, prefix="/nutricionista", tags=["Panel Nutricionista"])