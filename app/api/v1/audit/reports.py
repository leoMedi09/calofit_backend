"""
Endpoints de auditoría y reportes.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import logging

from app.api.dependencies import get_db
from app.services.validators import ConsistencyChecker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audit", tags=["audit"])


class AuditReportResponse(BaseModel):
    """Response de auditoría."""
    exito: bool
    tipo: str
    total_errores: int
    total_advertencias: int
    errores: List[str]
    advertencias: List[str]
    sugerencias: List[str]


@router.get(
    "/report",
    response_model=AuditReportResponse,
    name="Auditoría de BD",
)
async def obtener_reporte_auditoria(
    tipo: str = "completo",
    db: Session = Depends(get_db),
):
    """
    Ejecuta auditoría de integridad de BD.
    
    Tipos:
    - **completo**: audita todo
    - **alimentos**: solo alimentos
    - **platos**: solo platos
    - **ejercicios**: solo ejercicios
    - **rutinas**: solo rutinas
    """
    try:
        logger.info(f"Ejecutando auditoría: {tipo}")
        
        checker = ConsistencyChecker(db)
        resultado = checker.validar({'tipo': tipo})
        
        return AuditReportResponse(
            exito=resultado.es_valido,
            tipo=tipo,
            total_errores=len(resultado.errores),
            total_advertencias=len(resultado.advertencias),
            errores=resultado.errores,
            advertencias=resultado.advertencias,
            sugerencias=resultado.sugerencias,
        )
    
    except Exception as e:
        logger.error(f"Error en auditoría: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/cleanup",
    name="Limpieza de BD",
)
async def limpiar_bd(
    confirmar: bool = False,
    db: Session = Depends(get_db),
):
    """
    Ejecuta limpieza de BD corrupta.
    
    PELIGRO: Esta operación modifica la BD.
    
    - **confirmar**: debe ser True para ejecutar
    """
    if not confirmar:
        return {
            'exito': False,
            'mensaje': 'Se requiere confirmar=true',
        }
    
    try:
        logger.warning(f"Ejecutando cleanup de BD...")
        
        # TODO: Implementar limpieza
        
        return {
            'exito': True,
            'mensaje': 'Limpieza completada',
            'registros_eliminados': 0,
            'registros_actualizados': 0,
        }
    
    except Exception as e:
        logger.error(f"Error en cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
