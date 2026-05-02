from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, List
from datetime import date

# --- TUS ESQUEMAS EXISTENTES ---

class ClientCreate(BaseModel):
    first_name: str
    last_name_paternal: str
    last_name_maternal: str
    email: EmailStr
    password: str = Field(..., min_length=6)
    birth_date: date
    weight: float = Field(..., gt=0, description="Peso en kilogramos")
    height: float = Field(..., gt=0, description="Altura en centímetros")
    gender: str = Field(..., pattern="^[MF]$", description="Género: M (Masculino) o F (Femenino)")
    
    medical_conditions: List[str] = Field(default=[], description="Lista de condiciones médicas")
    
    activity_level: Optional[str] = Field(default="Sedentario", description="Nivel de actividad física")
    goal: Optional[str] = Field(default="Mantener peso", description="Objetivo de salud")
    workout_type: Optional[str] = Field(default=None, description="Tipo de entrenamiento preferido (para ML)")
    session_duration: Optional[float] = Field(default=None, description="Duración de sesión en horas (para ML)")
    flutter_uid: str = Field(..., min_length=10, description="Firebase UID único del usuario")
    assigned_coach_id: Optional[int] = None
    assigned_nutri_id: Optional[int] = None

    class Config:
        from_attributes = True


class ClientResponse(BaseModel):
    id: int
    first_name: str
    last_name_paternal: str
    last_name_maternal: str
    email: EmailStr
    flutter_uid: Optional[str]
    birth_date: Optional[date]
    weight: float
    height: float
    gender: Optional[str] = 'M'
    activity_level: Optional[str] = 'Sedentario'
    goal: Optional[str] = 'Mantener peso'
    workout_type: Optional[str] = 'Cardio'        # 🆕 Para ML Random Forest
    session_duration: Optional[float] = 1.0        # 🆕 Para ML Random Forest (en horas)
    medical_conditions: List[str] = []
    assigned_coach_id: Optional[int]
    assigned_nutri_id: Optional[int]
    profile_picture_url: Optional[str] = None
    is_profile_complete: bool = False

    class Config:
        from_attributes = True

class ClientUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name_paternal: Optional[str] = None
    last_name_maternal: Optional[str] = None
    email: Optional[EmailStr] = None
    flutter_uid: Optional[str] = None
    birth_date: Optional[date] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    gender: Optional[str] = None
    medical_conditions: Optional[List[str]] = None
    activity_level: Optional[str] = None
    goal: Optional[str] = None
    workout_type: Optional[str] = None             # 🆕 Para ML Random Forest
    session_duration: Optional[float] = None       # 🆕 Para ML Random Forest (en horas)
    profile_picture_url: Optional[str] = None
    is_profile_complete: Optional[bool] = None  # 🆕 El onboarding lo marca como True al terminar


class AdminCreateClient(BaseModel):
    """Schema simplificado para que el Admin cree un usuario: solo email + contraseña"""
    email: EmailStr
    password: str = Field(..., min_length=6)
    flutter_uid: str = Field(..., description="Firebase UID generado por el Admin al crear el usuario en Firebase")
    assigned_nutri_id: Optional[int] = None
    assigned_coach_id: Optional[int] = None

class ClientExpressCreate(BaseModel):
    """Schema para la creación B2B de un paciente solo usando DNI y Correo."""
    email: EmailStr
    dni: str = Field(..., min_length=7, max_length=15, description="El DNI será usado como clave temporal")
    assigned_nutri_id: Optional[int] = None

class ChangePassword(BaseModel):
    new_password: str = Field(..., min_length=6)
    confirm_password: str = Field(..., min_length=6)

# --- 🚀 NUEVOS ESQUEMAS ESTRATÉGICOS (v80.0) ---

class StrategicGuideUpdate(BaseModel):
    """Para que el nutri guíe a la IA desde el expediente"""
    ai_strategic_focus: Optional[str] = None
    recommended_foods: Optional[List[str]] = None
    forbidden_foods: Optional[List[str]] = None
    medical_conditions: Optional[List[str]] = None  # Nutri también puede ajustar
    is_strategic_guide_validated: Optional[bool] = None
    nutri_weekly_note: Optional[str] = None          # 🆕 Mensaje/meta semanal al cliente

class ResetPasswordRequest(BaseModel):
    """Para cuando el usuario ingresa su email para recibir el código"""
    email: EmailStr

class ResetPasswordVerify(BaseModel):
    """Para cuando el usuario ingresa el código de 6 dígitos y su nueva clave"""
    email: EmailStr
    code: str = Field(..., min_length=6, max_length=6)
    new_password: str = Field(..., min_length=6)
    confirm_password: str = Field(..., min_length=6)

    @field_validator('confirm_password')
    @classmethod
    def passwords_match_reset(cls, v, info):
        if 'new_password' in info.data and v != info.data['new_password']:
            raise ValueError('Las contraseñas no coinciden')
        return v