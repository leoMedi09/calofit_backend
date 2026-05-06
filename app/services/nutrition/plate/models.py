"""
DTOs (Data Transfer Objects) para platos.
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from app.services.validators.base_validator import ValidationResult


class IngredienteDTO(BaseModel):
    """DTO para un ingrediente en un plato."""
    nombre: str
    gramos: float = Field(gt=0)
    alimento_id: Optional[int] = None
    macros_100g: Optional[Dict[str, float]] = None
    macros_totales: Optional[Dict[str, float]] = None
    source: Optional[str] = None
    confianza: int = Field(default=80, ge=0, le=100)
    fingerprint: Optional[str] = None


# Alias para compatibilidad con código que importe ValidationResultDTO
ValidationResultDTO = ValidationResult


class MacrosDTO(BaseModel):
    """Macros totales."""
    calorias: float
    proteina: float
    carbohidratos: float
    grasas: float
    fibra: Optional[float] = None
    azucar: Optional[float] = None


class PlatoConstructionResultDTO(BaseModel):
    """Resultado de construcción de plato."""
    exito: bool
    plato_id: Optional[int] = None
    nombre: str
    peso_total_gramos: float
    ingredientes: List[IngredienteDTO]
    macros_totales: MacrosDTO

    # Validaciones — usa ValidationResult directamente (sin conversión)
    validacion_semantica: ValidationResult
    validacion_nutricional: ValidationResult

    # Metadata
    fingerprint: str
    confianza_global: int = Field(ge=0, le=100)
    cached: bool = False
    timestamp: Optional[str] = None

    class Config:
        from_attributes = True
        arbitrary_types_allowed = True
