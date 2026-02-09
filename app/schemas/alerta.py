from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class AlertaSaludResponse(BaseModel):
    """Schema para respuesta de alerta de salud"""
    id: int
    client_id: int
    tipo: str
    descripcion: str
    severidad: str
    estado: str
    atendido_por_id: Optional[int] = None
    notas: Optional[str] = None
    fecha_deteccion: datetime
    fecha_atencion: Optional[datetime] = None
    created_at: datetime
    
    # Información adicional del cliente
    cliente_nombre: Optional[str] = None
    atendido_por_nombre: Optional[str] = None
    
    class Config:
        from_attributes = True


class AlertaUpdateRequest(BaseModel):
    """Schema para actualizar una alerta"""
    estado: Optional[str] = Field(None, pattern="^(pendiente|en_proceso|atendida)$")
    notas: Optional[str] = None


class AlertaAtenderRequest(BaseModel):
    """Schema para marcar una alerta como atendida"""
    notas: str = Field(..., min_length=10, description="Notas sobre cómo se atendió la alerta")
