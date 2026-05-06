"""
DTOs para rutinas.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class EjercicioEnRutinaDTO(BaseModel):
    """Ejercicio dentro de una rutina."""
    ejercicio_id: int
    nombre: str
    series: int = Field(gt=0)
    repeticiones: int = Field(gt=0)
    peso_kg_recomendado: Optional[float] = None
    descanso_segundos: int = Field(ge=0)
    orden: int = Field(gt=0)
    met: Optional[float] = None
    kcal_por_minuto: Optional[float] = None


class RoutineConstructionResultDTO(BaseModel):
    """Resultado de construcción de rutina."""
    exito: bool
    rutina_id: Optional[int] = None
    nombre: str
    tipo_entrenamiento: str
    intensidad: str
    objetivo: str
    duracion_estimada_minutos: int
    
    # Detalles
    ejercicios: List[EjercicioEnRutinaDTO]
    total_ejercicios: int
    
    # Métricas
    kcal_quemadas_estimadas: float
    intensidad_total: float = Field(ge=0, le=100)
    
    # Validación
    fingerprint: str
    confianza_global: int = Field(ge=0, le=100)
    cached: bool = False
    timestamp: Optional[str] = None
    
    class Config:
        from_attributes = True
