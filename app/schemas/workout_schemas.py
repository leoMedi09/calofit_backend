"""
Schemas Pydantic para sesiones de entrenamiento.
"""
from datetime import datetime, date
from typing import Optional, List
from pydantic import BaseModel, Field


# ─── WorkoutSessionEjercicio ─────────────────────────────────────────────────

class WorkoutSessionEjercicioCreate(BaseModel):
    ejercicio_id:       str
    orden:              int   = Field(default=1, ge=1)
    series_completadas: Optional[int]   = None
    reps_completadas:   Optional[int]   = None
    peso_kg:            Optional[float] = None
    duracion_s:         Optional[int]   = None
    calorias_quemadas:  Optional[float] = None
    notas:              Optional[str]   = None


class WorkoutSessionEjercicioResponse(WorkoutSessionEjercicioCreate):
    id:         int
    session_id: int

    class Config:
        from_attributes = True


# ─── WorkoutSession ───────────────────────────────────────────────────────────

class WorkoutSessionCreate(BaseModel):
    client_id:         int
    rutina_id:         Optional[int]   = None
    nombre_rutina:     Optional[str]   = None
    fecha:             date
    duracion_min:      Optional[int]   = None
    calorias_quemadas: Optional[float] = None
    intensity:         Optional[str]   = None
    notas:             Optional[str]   = None
    ejercicios:        List[WorkoutSessionEjercicioCreate] = []


class WorkoutSessionUpdate(BaseModel):
    duracion_min:      Optional[int]   = None
    calorias_quemadas: Optional[float] = None
    intensity:         Optional[str]   = None
    notas:             Optional[str]   = None


class WorkoutSessionResponse(BaseModel):
    id:                int
    client_id:         int
    rutina_id:         Optional[int]   = None
    nombre_rutina:     Optional[str]   = None
    fecha:             date
    duracion_min:      Optional[int]   = None
    calorias_quemadas: Optional[float] = None
    intensity:         Optional[str]   = None
    notas:             Optional[str]   = None
    created_at:        datetime
    ejercicios:        List[WorkoutSessionEjercicioResponse] = []

    class Config:
        from_attributes = True


class WorkoutSessionSummary(BaseModel):
    id:                int
    fecha:             date
    nombre_rutina:     Optional[str]   = None
    duracion_min:      Optional[int]   = None
    calorias_quemadas: Optional[float] = None
    total_ejercicios:  int = 0
    intensity:         Optional[str]   = None

    class Config:
        from_attributes = True
