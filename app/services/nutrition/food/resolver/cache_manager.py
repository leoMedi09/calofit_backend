"""
Gestor de caché inteligente para alimentos.

AppCacheAlimentos tiene: food_normalized, user_id, alimento_id, source,
raw_response (TEXT), hit_count, expires_at, created_at.
Los macros se serializan como JSON en raw_response.
"""
import json
from typing import Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from app.models import AppCacheAlimentos
import logging

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Maneja caché inteligente de alimentos resueltos.

    • Expira después de 60 días
    • Macros almacenados como JSON en raw_response
    """

    CACHE_EXPIRY_DAYS = 60

    def __init__(self, db: Session):
        self.db = db

    def obtener_del_cache(
        self,
        food_normalized: str,
        user_id: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Obtiene macros del caché.

        Returns:
            Dict con claves calorias_100g, proteina_100g, … o None si expiró/no existe.
        """
        try:
            entry = self.db.query(AppCacheAlimentos).filter(
                AppCacheAlimentos.food_normalized == food_normalized.lower().strip(),
                AppCacheAlimentos.user_id == user_id,
            ).first()

            if not entry:
                return None

            if entry.expires_at and datetime.now(timezone.utc) > entry.expires_at:
                logger.info("Cache expirado para %s", food_normalized)
                self.db.delete(entry)
                self.db.commit()
                return None

            if not entry.raw_response:
                return None

            data = json.loads(entry.raw_response)
            return data.get("macros")

        except Exception as exc:
            logger.error("CacheManager.obtener_del_cache error: %s", exc)
            return None

    def guardar_en_cache(
        self,
        food_normalized: str,
        user_id: int,
        macros: Dict[str, float],
        source: str,
        alimento_id: Optional[int] = None,
    ) -> bool:
        """
        Guarda macros de un alimento en caché.

        Args:
            food_normalized: nombre normalizado
            user_id: ID del cliente
            macros: {calorias_100g, proteina_100g, carbohidratos_100g, grasas_100g, …}
            source: BD|USDA|FatSecret|Groq
            alimento_id: FK a alimentos si se conoce
        """
        try:
            payload = json.dumps({"macros": macros, "source": source}, ensure_ascii=False)
            expires = datetime.now(timezone.utc) + timedelta(days=self.CACHE_EXPIRY_DAYS)
            norm = food_normalized.lower().strip()

            existing = self.db.query(AppCacheAlimentos).filter(
                AppCacheAlimentos.food_normalized == norm,
                AppCacheAlimentos.user_id == user_id,
            ).first()

            if existing:
                existing.raw_response = payload
                existing.source = source
                existing.alimento_id = alimento_id
                existing.expires_at = expires
                existing.hit_count = (existing.hit_count or 0) + 1
                logger.info("Cache actualizado: %s", food_normalized)
            else:
                entry = AppCacheAlimentos(
                    food_normalized=norm,
                    user_id=user_id,
                    alimento_id=alimento_id,
                    source=source,
                    raw_response=payload,
                    expires_at=expires,
                )
                self.db.add(entry)
                logger.info("Cache creado: %s", food_normalized)

            self.db.commit()
            return True

        except Exception as exc:
            logger.error("CacheManager.guardar_en_cache error: %s", exc)
            self.db.rollback()
            return False

    def invalidar_cache(self, food_normalized: str, user_id: int) -> bool:
        """Elimina entrada de caché específica."""
        try:
            self.db.query(AppCacheAlimentos).filter(
                AppCacheAlimentos.food_normalized == food_normalized.lower().strip(),
                AppCacheAlimentos.user_id == user_id,
            ).delete()
            self.db.commit()
            return True
        except Exception as exc:
            logger.error("CacheManager.invalidar_cache error: %s", exc)
            return False

    def limpiar_cache_expirado(self, user_id: Optional[int] = None) -> int:
        """Elimina entradas expiradas."""
        try:
            q = self.db.query(AppCacheAlimentos).filter(
                AppCacheAlimentos.expires_at < datetime.now(timezone.utc)
            )
            if user_id:
                q = q.filter(AppCacheAlimentos.user_id == user_id)
            count = q.delete()
            self.db.commit()
            return count
        except Exception as exc:
            logger.error("CacheManager.limpiar_cache_expirado error: %s", exc)
            return 0
