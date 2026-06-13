from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.core.database import get_db
from app.models.client import Client
from app.api.routes.auth import get_current_user

router = APIRouter()


class FcmTokenRequest(BaseModel):
    fcm_token: str


class NotificacionesPrefRequest(BaseModel):
    activas: bool


@router.post("/fcm-token")
async def registrar_fcm_token(
    payload: FcmTokenRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """
    Registra (o actualiza) el token FCM del dispositivo del cliente autenticado,
    usado para enviar notificaciones push (recordatorios de registro diario).
    """
    cliente = db.query(Client).filter(Client.id == current_user.id).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    cliente.fcm_token = payload.fcm_token
    db.commit()

    return {"status": "ok", "mensaje": "Token FCM registrado correctamente"}


@router.put("/preferencias")
async def actualizar_preferencia_notificaciones(
    payload: NotificacionesPrefRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """
    Activa o desactiva los recordatorios diarios push para el cliente autenticado.
    """
    cliente = db.query(Client).filter(Client.id == current_user.id).first()
    if not cliente:
        raise HTTPException(status_code=404, detail="Cliente no encontrado")

    cliente.notificaciones_activas = payload.activas
    db.commit()

    return {"status": "ok", "notificaciones_activas": cliente.notificaciones_activas}
