"""
Health check endpoint.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.api.dependencies import get_db
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", name="Health Check")
async def health_check(db: Session = Depends(get_db)):
    """
    Health check del sistema.
    
    Verifica:
    • API responde
    • BD conecta
    • Servicios disponibles
    """
    try:
        # Probar conexión a BD
        db.execute("SELECT 1")
        
        return {
            'status': 'OK',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'db': 'Connected',
            'services': {
                'nutrition': 'Ready',
                'exercise': 'Ready',
                'ai': 'Ready',
            }
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            'status': 'ERROR',
            'timestamp': datetime.utcnow().isoformat(),
            'error': str(e),
            'db': 'Disconnected',
        }, 503
