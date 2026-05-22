"""
Endpoints de recomendación y consumo de platos.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import date
import logging

from app.api.dependencies import (
    get_db,
    get_plate_builder,
    get_assistant_orchestrator,
)
from app.models import ProgresoCalorias
from app.models.historial_recomendacion import HistorialRecomendacion
# from app.schemas.nutrition import NutricionResponse # Ajustado para evitar error si no existe

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/nutrition", tags=["nutrition"])


class PlateRecommendationRequest(BaseModel):
    """Request para recomendación de platos."""
    objetivo: str = Field(..., description="alto_proteína|bajo_calorías|balanceado")
    momento_dia: str = Field(..., description="desayuno|almuerzo|cena|snack")
    restricciones: Optional[List[str]] = None
    cantidad: int = Field(default=3, ge=1, le=5)


class PlateRecommendationResponse(BaseModel):
    """Response de recomendación."""
    exito: bool
    platos: List[dict]
    mensaje: str


class ConsumeRequest(BaseModel):
    """Request para registrar consumo."""
    plato_id: Optional[int] = None
    nombre_plato: Optional[str] = None
    calorias: float = Field(..., gt=0)
    proteina: float = Field(..., ge=0)
    carbohidratos: float = Field(..., ge=0)
    grasas: float = Field(..., ge=0)
    momento_dia: str = Field(...)


class ConsumeResponse(BaseModel):
    """Response de registro."""
    exito: bool
    mensaje: str
    progreso: dict


@router.post(
    "/plates",
    response_model=PlateRecommendationResponse,
    name="Recomendación de Platos",
)
async def recomendar_platos(
    request: PlateRecommendationRequest,
    client_id: int,
    db: Session = Depends(get_db),
    plate_builder = Depends(get_plate_builder),
):
    """
    Recomienda platos con valores nutricionales verificados desde la BD.

    - **objetivo**: alto_proteina|bajo_calorias|balanceado
    - **momento_dia**: desayuno|almuerzo|cena|snack|cualquiera
    - **restricciones**: alimentos a evitar
    - **cantidad**: 1-5 propuestas (default 3)

    Los macros vienen SIEMPRE de ingredientes reales en la BD,
    no de estimaciones del LLM. Si un ingrediente no está en BD,
    se estima y se persiste para consistencia futura.
    """
    try:
        from app.services.recomendador_platos import RecomendadorPlatosConfiables
        from app.models import ProgresoCalorias, MetaUsuario
        from app.core.utils import get_peru_date

        logger.info(f"Recomendando platos BD para cliente {client_id}")

        # Obtener déficit real del día
        hoy = get_peru_date()
        progreso = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == client_id,
            ProgresoCalorias.fecha == hoy,
        ).first()

        meta = db.query(MetaUsuario).filter(
            MetaUsuario.client_id == client_id
        ).first()

        kcal_obj = float(getattr(meta, 'calorias_objetivo', 0) or 2000)
        prot_obj = float(getattr(meta, 'proteinas_g', 0) or 120)
        carb_obj = float(getattr(meta, 'carbohidratos_g', 0) or 250)
        gras_obj = float(getattr(meta, 'grasas_g', 0) or 65)

        kcal_cons = float(getattr(progreso, 'calorias_consumidas', 0) or 0)
        prot_cons = float(getattr(progreso, 'proteinas_consumidas', 0) or 0)
        carb_cons = float(getattr(progreso, 'carbohidratos_consumidos', 0) or 0)
        gras_cons = float(getattr(progreso, 'grasas_consumidas', 0) or 0)

        deficit_kcal = max(100.0, kcal_obj - kcal_cons)
        deficit_prot = max(0.0, prot_obj - prot_cons)
        deficit_carb = max(0.0, carb_obj - carb_cons)
        deficit_gras = max(0.0, gras_obj - gras_cons)

        # Ajustar déficit según objetivo
        if request.objetivo == "alto_proteina":
            deficit_prot *= 1.4
        elif request.objetivo == "bajo_calorias":
            deficit_kcal *= 0.7

        recomendador = RecomendadorPlatosConfiables(db, plate_builder=plate_builder)
        platos = recomendador.recomendar(
            client_id=client_id,
            deficit_kcal=deficit_kcal,
            deficit_proteina=deficit_prot,
            deficit_carb=deficit_carb,
            deficit_grasas=deficit_gras,
            momento_dia=request.momento_dia,
            n=request.cantidad,
            excluir_nombres=request.restricciones,
        )

        if not platos:
            return PlateRecommendationResponse(
                exito=False,
                platos=[],
                mensaje=(
                    "No se encontraron platos disponibles en este momento. "
                    "Intenta más tarde o ajusta tus restricciones."
                ),
            )

        return PlateRecommendationResponse(
            exito=True,
            platos=platos,
            mensaje=(
                f"Aquí tienes {len(platos)} opción{'es' if len(platos) > 1 else ''} "
                f"para {request.momento_dia}. "
                f"Déficit restante: {deficit_kcal:.0f} kcal."
            ),
        )

    except Exception as e:
        logger.error(f"Error recomendando platos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/consume",
    response_model=ConsumeResponse,
    name="Registrar Consumo",
)
async def registrar_consumo(
    request: ConsumeRequest,
    client_id: int,
    db: Session = Depends(get_db),
):
    """
    Registra consumo de alimento.
    
    Actualiza:
    • progreso_calorias
    • historial_recomendaciones
    """
    try:
        logger.info(f"Registrando consumo para cliente {client_id}")
        
        # Obtener o crear progreso del día
        progreso = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == client_id,
            ProgresoCalorias.fecha == date.today(),
        ).first()
        
        if not progreso:
            progreso = ProgresoCalorias(
                client_id=client_id,
                fecha=date.today(),
                calorias_consumidas=0,
                proteinas_consumidas=0,
                carbohidratos_consumidos=0,
                grasas_consumidas=0,
            )
            db.add(progreso)
        
        # Actualizar macros
        progreso.calorias_consumidas += request.calorias
        progreso.proteinas_consumidas += request.proteina
        progreso.carbohidratos_consumidos += request.carbohidratos
        progreso.grasas_consumidas += request.grasas
        
        db.commit()
        
        # Registrar en historial
        if request.plato_id:
            historial = HistorialRecomendacion(
                client_id=client_id,
                plato_id=request.plato_id,
                fue_consumido=True,
            )
            db.add(historial)
            db.commit()
        
        return ConsumeResponse(
            exito=True,
            mensaje="Consumo registrado exitosamente",
            progreso={
                'fecha': date.today().isoformat(),
                'calorias_consumidas': progreso.calorias_consumidas,
                'proteinas': progreso.proteinas_consumidas,
                'carbohidratos': progreso.carbohidratos_consumidos,
                'grasas': progreso.grasas_consumidas,
            },
        )
    
    except Exception as e:
        logger.error(f"Error registrando consumo: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
