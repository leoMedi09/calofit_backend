"""
API v1 - Endpoints principales.
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["v1"])

# Importar routers de cada módulo
from app.api.v1.nutrition import plates, progress, parser
from app.api.v1.exercise import routines, workouts
from app.api.v1.assistant import chat
from app.api.v1.audit import reports
from app.api.v1 import health

# Registrar routers
router.include_router(plates.router)
router.include_router(progress.router)
router.include_router(parser.router, prefix="/nutrition")
router.include_router(routines.router)
router.include_router(workouts.router)
router.include_router(chat.router)
router.include_router(reports.router)
router.include_router(health.router)

__all__ = ["router"]
