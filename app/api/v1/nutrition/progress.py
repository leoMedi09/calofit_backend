"""
Endpoints de progreso nutricional.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from datetime import date, timedelta
import logging

from app.api.dependencies import get_db
from app.models import ProgresoCalorias, MetaUsuario

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/nutrition", tags=["nutrition"])


class ProgressResponse(BaseModel):
    """Response de progreso."""
    fecha: date
    calorias_consumidas: float
    calorias_quemadas: float
    balance_neto: float
    objetivo_kcal: float
    disponible: float
    macros: dict


@router.get(
    "/progress",
    response_model=List[ProgressResponse],
    name="Progreso Nutricional",
)
async def obtener_progreso(
    client_id: int,
    dias: int = 7,
    db: Session = Depends(get_db),
):
    """
    Obtiene progreso de nutrición de los últimos N días.
    
    - **dias**: 1-30 días (default 7)
    """
    try:
        logger.info(f"Obteniendo progreso de {dias} días para cliente {client_id}")
        
        # Obtener metas
        metas = db.query(MetaUsuario).filter(
            MetaUsuario.client_id == client_id
        ).first()
        
        objetivo_kcal = metas.calorias_objetivo if metas else 2000
        
        # Obtener progreso
        fecha_inicio = date.today() - timedelta(days=dias)
        
        progresos = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == client_id,
            ProgresoCalorias.fecha >= fecha_inicio,
        ).order_by(ProgresoCalorias.fecha.desc()).all()
        
        resultados = []
        for prog in progresos:
            balance = prog.calorias_consumidas - prog.calorias_quemadas
            disponible = objetivo_kcal - balance
            
            resultados.append(ProgressResponse(
                fecha=prog.fecha,
                calorias_consumidas=prog.calorias_consumidas,
                calorias_quemadas=prog.calorias_quemadas,
                balance_neto=balance,
                objetivo_kcal=objetivo_kcal,
                disponible=disponible,
                macros={
                    'proteina': prog.proteinas_consumidas,
                    'carbohidratos': prog.carbohidratos_consumidos,
                    'grasas': prog.grasas_consumidas,
                },
            ))
        
        return resultados
    
    except Exception as e:
        logger.error(f"Error obteniendo progreso: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/progress/today",
    response_model=ProgressResponse,
    name="Progreso Hoy",
)
async def obtener_progreso_hoy(
    client_id: int,
    db: Session = Depends(get_db),
):
    """Obtiene progreso del día actual."""
    try:
        # Obtener metas
        metas = db.query(MetaUsuario).filter(
            MetaUsuario.client_id == client_id
        ).first()
        
        objetivo_kcal = metas.calorias_objetivo if metas else 2000
        
        # Obtener progreso hoy
        progreso = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == client_id,
            ProgresoCalorias.fecha == date.today(),
        ).first()
        
        if not progreso:
            return ProgressResponse(
                fecha=date.today(),
                calorias_consumidas=0,
                calorias_quemadas=0,
                balance_neto=0,
                objetivo_kcal=objetivo_kcal,
                disponible=objetivo_kcal,
                macros={'proteina': 0, 'carbohidratos': 0, 'grasas': 0},
            )
        
        balance = progreso.calorias_consumidas - progreso.calorias_quemadas
        disponible = objetivo_kcal - balance
        
        return ProgressResponse(
            fecha=progreso.fecha,
            calorias_consumidas=progreso.calorias_consumidas,
            calorias_quemadas=progreso.calorias_quemadas,
            balance_neto=balance,
            objetivo_kcal=objetivo_kcal,
            disponible=disponible,
            macros={
                'proteina': progreso.proteinas_consumidas,
                'carbohidratos': progreso.carbohidratos_consumidos,
                'grasas': progreso.grasas_consumidas,
            },
        )
    
    except Exception as e:
        logger.error(f"Error obteniendo progreso hoy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
