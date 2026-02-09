from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

# --- Esquema para los detalles de cada día ---
class PlanDiarioResponse(BaseModel):
    id: int
    dia_numero: int
    calorias_dia: float
    proteinas_g: float
    carbohidratos_g: float
    grasas_g: float
    # Estos campos son los que "hablan" con el cliente y el coach
    sugerencia_entrenamiento_ia: Optional[str] = None
    nota_asistente_ia: Optional[str] = None
    estado: str # 'sugerencia_ia', 'oficial', 'ajustado_ia'

    class Config:
        from_attributes = True

# --- Esquema para crear el plan (lo que envía el Nutricionista) ---
class PlanNutricionalCreate(BaseModel):
    client_id: int
    genero: int
    edad: int
    peso: float
    talla: float
    nivel_actividad: float = Field(1.2, ge=1.2, le=2.0)
    objetivo: str = Field("mantener", pattern="^(ganar|perder|mantener)$")
    
    # ✨ OPCIONAL: Si el nutricionista quiere distribuir manualmente los macros
    # De lo contrario, la IA los calculará automáticamente
    proteinas_g: Optional[float] = None
    carbohidratos_g: Optional[float] = None
    grasas_g: Optional[float] = None
    observaciones: Optional[str] = None

# --- Esquema de respuesta completa (lo que recibe la App Móvil) ---
class PlanNutricionalResponse(BaseModel):
    id: int
    client_id: int
    calorias_ia_base: Optional[float] = None
    objetivo: str
    fecha_creacion: datetime
    es_contingencia_ia: bool = False
    
    # CLAVE: Aquí incluimos los 7 días generados por la IA
    detalles_diarios: List[PlanDiarioResponse] 
    
    # --- Nuevos campos de validación ---
    status: str = "draft_ia"
    validated_by_id: Optional[int] = None
    validated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# --- Esquema para probar IA ---
class TestIARequest(BaseModel):
    genero: str = Field(..., description="Género: 1=Masculino, 2=Femenino")
    edad: int = Field(..., ge=1, le=120, description="Edad en años")
    peso: float = Field(..., gt=0, description="Peso en kg")
    talla: float = Field(..., gt=0, description="Talla en cm")
    nivel_actividad: float = Field(1.2, description="Nivel de actividad (ej. 1.2=sedentario, 1.55=moderado)")
    objetivo: str = Field("mantener", description="Objetivo: mantener, ganar, perder")

# --- Esquemas para Alertas de Salud ---
class AlertaSaludBase(BaseModel):
    tipo: str = Field(..., description="Tipo: fatiga, lesion, desanimo, otro")
    descripcion: str
    severidad: str = "bajo"

class AlertaSaludCreate(AlertaSaludBase):
    pass

class AlertaSaludResponse(AlertaSaludBase):
    id: int
    client_id: int
    estado: str
    atendido_por_id: Optional[int] = None
    fecha_deteccion: datetime
    
    class Config:
        from_attributes = True

class AlertaSaludUpdate(BaseModel):
    estado: str
    atendido_por_id: Optional[int] = None