"""Limpieza diaria de datos que no deben conservarse. NO toca registros de comida, ejercicio, progreso ni el chat."""
from sqlalchemy import text

from app.core.database import SessionLocal
from app.core.logging_config import get_logger

logger = get_logger("core.limpieza")

_SENTENCIAS = {
    "codigos de recuperacion (>7 dias)": "DELETE FROM password_resets WHERE created_at < NOW() - INTERVAL '7 days'",
    "cache de alimentos vencida": "DELETE FROM app_cache_alimentos WHERE expires_at IS NOT NULL AND expires_at < NOW()",
    "cache de platos vencida": "DELETE FROM app_cache_platos WHERE expires_at IS NOT NULL AND expires_at < NOW()",
    "historial de recomendaciones (>90 dias)": "DELETE FROM historial_recomendaciones WHERE created_at < NOW() - INTERVAL '90 days'",
}


def limpiar_datos_vencidos() -> dict:
    db = SessionLocal()
    resumen = {}
    try:
        for nombre, sql in _SENTENCIAS.items():
            resumen[nombre] = db.execute(text(sql)).rowcount
        db.commit()
        logger.info("Limpieza diaria: %s", resumen)
    except Exception as e:
        db.rollback()
        logger.error("Error en la limpieza diaria: %s", e)
    finally:
        db.close()
    return resumen
