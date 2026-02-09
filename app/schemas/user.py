from pydantic import BaseModel, EmailStr
from typing import Optional, List

class UserLogin(BaseModel):
    email: str
    password: str
    remember_me: bool = False
    firebase_uid: Optional[str] = None
    user_type: Optional[str] = None # 'client' o 'staff'


class ResetPassword(BaseModel):
    oobCode: str
    new_password: str

class ForgotPassword(BaseModel):
    email: EmailStr
    
class SyncPasswordRequest(BaseModel):
    email: EmailStr
    new_password: str

class ForgotPasswordRequest(BaseModel):
    """Schema para solicitar reset de contraseña"""
    email: EmailStr

class ValidateResetCodeRequest(BaseModel):
    """Schema para validar código y cambiar contraseña"""
    email: EmailStr
    reset_code: str  # Código de 6 dígitos
    new_password: str

class PasswordUpdate(BaseModel):
    new_password: str

class ResetCodeResponse(BaseModel):
    """Schema para respuesta de validación de código"""
    message: str
    success: bool

class UserCreate(BaseModel):
    first_name: str
    last_name_paternal: str
    last_name_maternal: str
    email: EmailStr
    password: str
    role: str 
    role_id: int

class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name_paternal: Optional[str] = None
    last_name_maternal: Optional[str] = None
    email: Optional[EmailStr] = None
    role_name: Optional[str] = None
    role_id: Optional[int] = None
    is_active: Optional[bool] = None

class UserResponse(BaseModel):
    id: int
    first_name: Optional[str] = None
    last_name_paternal: Optional[str] = None
    last_name_maternal: Optional[str] = None
    email: Optional[str] = None
    role_name: Optional[str] = "staff"
    is_active: bool = True
    
    class Config:
        from_attributes = True