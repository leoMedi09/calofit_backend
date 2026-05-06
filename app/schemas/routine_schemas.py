"""
Schemas Pydantic para rutinas de ejercicio.
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


# ─── RutinaEjercicio ─────────────────────────────────────────────────────────

class RutinaEjercicioCreate(BaseModel):
    ejercicio_id:      str
    orden:             int = Field(default=1, ge=1)
    series:            int = Field(default=3,  ge=1)
    reps:              int = Field(default=12, ge=1)
    descanso_s:        int = Field(default=60, ge=0)
    peso_sugerido_kg:  Optional[float] = None
    notas:             Optional[str]   = None


class RutinaEjercicioResponse(RutinaEjercicioCreate):
    id:        int
    rutina_id: int

    class Config:
        from_attributes = True


# ─── Rutina ───────────────────────────────────────────────────────────────────

class RutinaCreate(BaseModel):
    nombre:         str
    descripcion:    Optional[str] = None
    perfil_tipo:    Optional[str] = None
    nivel:          Optional[str] = None
    grupo_muscular: Optional[str] = None
    tiempo_min:     Optional[int] = None
    series_config:  Optional[str] = None
    origen:         str = "llm"
    ejercicios:     List[RutinaEjercicioCreate] = []


class RutinaUpdate(BaseModel):
    nombre:         Optional[str] = None
    descripcion:    Optional[str] = None
    perfil_tipo:    Optional[str] = None
    nivel:          Optional[str] = None
    grupo_muscular: Optional[str] = None
    tiempo_min:     Optional[int] = None


class RutinaResponse(BaseModel):
    id:             int
    nombre:         str
    descripcion:    Optional[str] = None
    perfil_tipo:    Optional[str] = None
    nivel:          Optional[str] = None
    grupo_muscular: Optional[str] = None
    tiempo_min:     Optional[int] = None
    origen:         str
    created_at:     datetime
    ejercicios:     List[RutinaEjercicioResponse] = []

    class Config:
        from_attributes = True
