"""
Endpoints de registro de ejercicio.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import date
import logging

from app.api.dependencies import get_db
from app.models import WorkoutSession, WorkoutSessionEjercicio, ProgresoCalorias

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/exercise", tags=["exercise"])


class WorkoutEjercicioRequest(BaseModel):
    """Request para ejercicio en sesión."""
    ejercicio_id: int
    series_realizadas: int = Field(..., ge=1)
    reps_realizadas: int = Field(..., ge=1)
    peso_kg_realizado: Optional[float] = None
    duracion_minutos: Optional[int] = None


class WorkoutSessionRequest(BaseModel):
    """Request para registrar sesión."""
    duracion_minutos: int = Field(..., ge=5)
    ejercicios: List[WorkoutEjercicioRequest]
    notas: Optional[str] = None


class WorkoutSessionResponse(BaseModel):
    """Response de sesión."""
    exito: bool
    session_id: int
    kcal_quemadas: float
    mensaje: str


@router.post(
    "/workouts",
    response_model=WorkoutSessionResponse,
    name="Registrar Sesión",
)
async def registrar_sesion(
    request: WorkoutSessionRequest,
    client_id: int,
    db: Session = Depends(get_db),
):
    """
    Registra sesión de ejercicio.
    
    Actualiza:
    • workout_sessions
    • workout_session_ejercicios
    • progreso_calorias
    """
    try:
        logger.info(f"Registrando sesión para cliente {client_id}")
        
        # Crear sesión
        sesion = WorkoutSession(
            client_id=client_id,
            fecha=date.today(),
            duracion_minutos=request.duracion_minutos,
            completada=True,
            notas=request.notas,
        )
        db.add(sesion)
        db.flush()
        
        # Calcular kcal totales
        kcal_totales = 0
        
        for ejercicio in request.ejercicios:
            # Buscar MET del ejercicio
            from app.models import Ejercicio
            ej = db.query(Ejercicio).filter(
                Ejercicio.id == ejercicio.ejercicio_id
            ).first()
            
            if ej and ej.met:
                # Obtener peso del cliente
                from app.models import Client
                cliente = db.query(Client).filter(Client.id == client_id).first()
                peso = cliente.weight if cliente else 70
                
                # Calcular kcal: MET × peso × 3.5 / 200 × minutos
                duracion = ejercicio.duracion_minutos or (request.duracion_minutos // len(request.ejercicios))
                kcal = (ej.met * peso * 3.5 / 200) * duracion
                kcal_totales += kcal
                
                # Registrar ejercicio
                ej_sesion = WorkoutSessionEjercicio(
                    session_id=sesion.id,
                    ejercicio_id=ejercicio.ejercicio_id,
                    series_realizadas=ejercicio.series_realizadas,
                    reps_realizadas=ejercicio.reps_realizadas,
                    peso_kg_realizado=ejercicio.peso_kg_realizado,
                    duracion_minutos=duracion,
                    kcal_quemadas=kcal,
                )
                db.add(ej_sesion)
        
        # Actualizar sesión con kcal
        sesion.kcal_quemadas_estimadas = kcal_totales
        
        # Actualizar progreso del día
        progreso = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == client_id,
            ProgresoCalorias.fecha == date.today(),
        ).first()
        
        if not progreso:
            progreso = ProgresoCalorias(
                client_id=client_id,
                fecha=date.today(),
                calorias_quemadas=kcal_totales,
            )
            db.add(progreso)
        else:
            progreso.calorias_quemadas += kcal_totales
        
        db.commit()
        
        return WorkoutSessionResponse(
            exito=True,
            session_id=sesion.id,
            kcal_quemadas=kcal_totales,
            mensaje=f"Sesión registrada. Quemaste {kcal_totales:.0f} kcal.",
        )
    
    except Exception as e:
        logger.error(f"Error registrando sesión: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
