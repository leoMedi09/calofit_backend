"""
Endpoints de recomendación de rutinas.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, Field
import logging

from app.api.dependencies import get_db, get_assistant_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/exercise", tags=["exercise"])


class RoutineRecommendationRequest(BaseModel):
    """Request para recomendación de rutina."""
    objetivo: str = Field(..., description="hipertrofia|resistencia|definicion")
    duracion_minutos: int = Field(..., ge=15, le=180)
    grupos_musculares: Optional[List[str]] = None
    nivel: str = Field(default="intermedio", description="principiante|intermedio|avanzado")


class RoutineRecommendationResponse(BaseModel):
    """Response de recomendación."""
    exito: bool
    rutina: dict
    ejercicios: List[dict]
    kcal_estimadas: float
    mensaje: str


@router.post(
    "/routines",
    response_model=RoutineRecommendationResponse,
    name="Recomendación de Rutina",
)
async def recomendar_rutina(
    request: RoutineRecommendationRequest,
    client_id: int,
    db: Session = Depends(get_db),
    assistant = Depends(get_assistant_orchestrator),
):
    """
    Recomienda rutina personalizada.
    
    - **objetivo**: hipertrofia, resistencia, definición
    - **duracion_minutos**: 15-180 minutos
    - **grupos_musculares**: lista de grupos a trabajar
    - **nivel**: principiante, intermedio, avanzado
    """
    try:
        logger.info(f"Recomendando rutina para cliente {client_id}")
        
        # Generar via orquestador
        resultado = await assistant.procesar_mensaje(
            client_id=client_id,
            mensaje=f"Rutina de {request.objetivo} en {request.duracion_minutos} minutos",
        )
        
        return RoutineRecommendationResponse(
            exito=resultado['exito'],
            rutina={},
            ejercicios=[],
            kcal_estimadas=0,
            mensaje=resultado['respuesta'],
        )
    
    except Exception as e:
        logger.error(f"Error recomendando rutina: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
