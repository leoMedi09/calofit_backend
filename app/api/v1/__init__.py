"""
API v1 - Endpoints principales.
"""
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["v1"])

from app.api.v1.nutrition import parser

router.include_router(parser.router, prefix="/nutrition")

__all__ = ["router"]
