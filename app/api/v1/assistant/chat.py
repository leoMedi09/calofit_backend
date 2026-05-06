"""
Endpoint principal de chat del asistente.
"""
from fastapi import APIRouter, Depends, HTTPException, WebSocket
from sqlalchemy.orm import Session
from typing import Optional, List
from pydantic import BaseModel
import logging
from datetime import datetime

from app.api.dependencies import get_db, get_assistant_orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assistant", tags=["assistant"])


class ChatMessage(BaseModel):
    """Mensaje en el chat."""
    role: str  # "user" | "assistant"
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    """Request de chat."""
    mensaje: str
    historial: Optional[List[ChatMessage]] = None


class ChatResponse(BaseModel):
    """Response de chat."""
    exito: bool
    respuesta: str
    intent: str
    siguiente_accion: Optional[str] = None
    datos: dict


@router.post(
    "/chat",
    response_model=ChatResponse,
    name="Chat Conversacional",
)
async def chat(
    request: ChatRequest,
    client_id: int,
    db: Session = Depends(get_db),
    assistant = Depends(get_assistant_orchestrator),
):
    """
    Chat conversacional principal del asistente.
    
    Soporta:
    • Recomendación de platos
    • Recomendación de rutinas
    • Registro de consumo
    • Consultas generales
    • Coaching personalizado
    
    Returns:
      - **exito**: operación exitosa
      - **respuesta**: texto de respuesta
      - **intent**: intención detectada
      - **siguiente_accion**: acción sugerida
      - **datos**: datos adicionales (platos, rutinas, etc)
    """
    try:
        logger.info(f"Chat de cliente {client_id}: {request.mensaje[:50]}...")
        
        # Convertir historial
        historial = None
        if request.historial:
            historial = [
                {
                    "role": msg.role,
                    "content": msg.content,
                }
                for msg in request.historial
            ]
        
        # Procesar mensaje
        resultado = await assistant.procesar_mensaje(
            client_id=client_id,
            mensaje=request.mensaje,
            historial=historial,
        )
        
        return ChatResponse(
            exito=resultado.get('exito', True),
            respuesta=resultado.get('respuesta', ''),
            intent=resultado.get('intent', 'unknown'),
            siguiente_accion=resultado.get('siguiente_accion'),
            datos=resultado.get('datos', {}),
        )
    
    except Exception as e:
        logger.error(f"Error en chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/history",
    response_model=List[ChatMessage],
    name="Historial de Chat",
)
async def obtener_historial(
    client_id: int,
    limite: int = 20,
    db: Session = Depends(get_db),
):
    """
    Obtiene historial de chat del cliente.
    
    - **limite**: máximo de mensajes (default 20)
    """
    try:
        # TODO: Implementar tabla de historial de chat en BD
        # Por ahora retornar lista vacía
        return []
    
    except Exception as e:
        logger.error(f"Error obteniendo historial: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
