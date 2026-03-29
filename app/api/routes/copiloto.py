from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.api.routes.auth import get_current_user
from app.services.nutricionista_service import nutricionista_ia_service
from app.services.admin_service import admin_ia_service
from pydantic import BaseModel
import traceback

router = APIRouter()

class CopilotoRequest(BaseModel):
    mensaje: str
    historial: list = None

@router.post("/consultar")
async def consultar_copiloto(
    request: CopilotoRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Endpoint único para el Staff (Nutri/Admin). 
    Enruta automáticamente al servicio correspondiente según el rol.
    """
    user_role = str(current_user.role_name).lower().strip() if hasattr(current_user, 'role_name') else "client"
    
    print(f"🩺 >>> CONSULTA COPILOTO STAFF <<<")
    print(f"🩺 Usuario: {current_user.email} | Rol: {user_role}")

    try:
        if user_role == "admin":
             # Usar Servicio de Administración Gerencial
             resultado = await admin_ia_service.consultar(
                 mensaje=request.mensaje,
                 db=db,
                 current_user=current_user,
                 historial=request.historial
             )
        elif user_role in ["nutricionista", "nutritionist", "coach"]:
             # Usar Servicio de Nutrición Clínica
             resultado = await nutricionista_ia_service.consultar(
                 mensaje=request.mensaje,
                 db=db,
                 current_user=current_user,
                 historial=request.historial
             )
        else:
            raise HTTPException(
                status_code=403, 
                detail=f"Acceso denegado: El rol '{user_role}' no tiene permisos para el Copiloto Staff."
            )
        
        return resultado

    except Exception as e:
        print(f"❌ ERROR EN /copiloto/consultar: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
