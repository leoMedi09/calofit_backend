"""
📋 Schemas para respuestas de perfil con recomendaciones de dieta
"""

from pydantic import BaseModel
from typing import Optional, List
from datetime import date

class MacronutrientesRecomendados(BaseModel):
    """Recomendaciones de macronutrientes"""
    calorias_diarias: float
    proteinas_g: float
    carbohidratos_g: float
    grasas_g: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "calorias_diarias": 2000,
                "proteinas_g": 150,
                "carbohidratos_g": 200,
                "grasas_g": 67
            }
        }


class RecomendacionDietaCompleta(BaseModel):
    """Recomendación completa de dieta automática"""
    calorias_diarias: float
    proteinas_g: float
    carbohidratos_g: float
    grasas_g: float
    imc: float
    categoria_imc: str
    gasto_metabolico_basal: float
    objetivo_recomendado: str
    alimentos_recomendados: List[str]
    alimentos_a_evitar: List[str]
    frecuencia_comidas: str
    notas: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "calorias_diarias": 2000,
                "proteinas_g": 150,
                "carbohidratos_g": 200,
                "grasas_g": 67,
                "imc": 24.5,
                "categoria_imc": "Peso normal",
                "gasto_metabolico_basal": 1800,
                "objetivo_recomendado": "Mantenimiento de peso actual",
                "alimentos_recomendados": ["Pollo sin piel", "Pescado", "Vegetales verdes"],
                "alimentos_a_evitar": ["Azúcares refinados", "Frituras"],
                "frecuencia_comidas": "3 comidas principales + 1-2 meriendas",
                "notas": "✅ Tu IMC es normal. Mantén hábitos saludables."
            }
        }


class ClientResponseConDieta(BaseModel):
    """Respuesta de cliente con recomendación de dieta automática"""
    id: int
    first_name: str
    last_name_paternal: str
    last_name_maternal: str
    email: str
    flutter_uid: Optional[str]
    birth_date: Optional[date]
    weight: float
    height: float
    goal: Optional[str]
    activity_level: Optional[str]
    assigned_coach_id: Optional[int]
    assigned_nutri_id: Optional[int]
    medical_conditions: List[str] = []

    # ✅ NUEVO: Recomendación de dieta automática
    dieta_recomendada: RecomendacionDietaCompleta
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 3,
                "first_name": "Juan",
                "last_name_paternal": "García",
                "last_name_maternal": "López",
                "email": "juan@email.com",
                "flutter_uid": "abc123xyz",
                "birth_date": "1990-05-15",
                "weight": 75.5,
                "height": 180,
                "goal": "Mantener peso",
                "activity_level": "Moderada",
                "assigned_coach_id": 1,
                "assigned_nutri_id": 2,
                "dieta_recomendada": {
                    "calorias_diarias": 2100,
                    "proteinas_g": 157,
                    "carbohidratos_g": 210,
                    "grasas_g": 70,
                    "imc": 23.3,
                    "categoria_imc": "Peso normal",
                    "gasto_metabolico_basal": 1850,
                    "objetivo_recomendado": "Mantenimiento de peso actual",
                    "alimentos_recomendados": ["Pollo", "Pescado", "Verduras"],
                    "alimentos_a_evitar": ["Azúcares", "Frituras"],
                    "frecuencia_comidas": "3+2",
                    "notas": "✅ Normal | 💡 Consulta nutricionista"
                }
            }
        }
