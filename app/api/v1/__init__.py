"""
API v1 - Endpoints principales.
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["v1"])

# Importar routers de cada módulo
from app.api.v1.nutrition import parser

# Registrar routers
router.include_router(parser.router, prefix="/nutrition")

__all__ = ["router"]
