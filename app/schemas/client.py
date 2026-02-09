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
    medical_conditions: List[str] = []
    assigned_coach_id: Optional[int]
    assigned_nutri_id: Optional[int]

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
    medical_conditions: Optional[List[str]] = None
    activity_level: Optional[str] = None
    goal: Optional[str] = None

class ChangePassword(BaseModel):
    new_password: str = Field(..., min_length=6)
    confirm_password: str = Field(..., min_length=6)

# --- 🛡️ NUEVOS ESQUEMAS PARA EL SISTEMA DE CÓDIGO OTP ---

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