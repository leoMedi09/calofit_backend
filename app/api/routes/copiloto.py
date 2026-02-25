from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.api.routes.auth import get_current_user
from app.services.copiloto_service import copiloto_service
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
    Endpoint dedicado para el Copiloto del Staff.
    Protegido para roles admin y nutricionista.
    """
    # 🛡️ Seguridad: Solo Staff
    user_role = str(current_user.role_name).lower().strip() if hasattr(current_user, 'role_name') else "client"
    print(f"🕵️ DEBUG: Usuario '{current_user.email}' intentando usar Copiloto. Rol detectado: '{user_role}'")
    
    if user_role not in ["admin", "nutricionista", "coach"]:
        raise HTTPException(
            status_code=403, 
            detail=f"Acceso denegado: El rol '{user_role}' no tiene permisos clínicos."
        )

    try:
        print(f"🩺 >>> CONSULTA COPILOTO CLÍNICO <<<")
        print(f"🩺 Staff: {current_user.email} (Rol: {current_user.role_name})")
        
        resultado = await copiloto_service.consultar_copiloto(
            mensaje=request.mensaje,
            db=db,
            current_user=current_user,
            historial=request.historial
        )
        
        return resultado

    except Exception as e:
        print(f"❌ ERROR EN /copiloto/consultar: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
