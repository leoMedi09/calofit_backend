"""
Inyección de dependencias para endpoints.
"""
from fastapi import Depends
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.services.nutrition.food.resolver.source_resolver import FoodSourceResolver
from app.services.nutrition.food.resolver.cache_manager import CacheManager
from app.services.nutrition.plate.plate_builder import PlatoBuilder
from app.services.exercise.routine.routine_builder import RoutineBuilder
from app.services.assistant.assistant_orchestrator import AssistantOrchestrator
import logging

logger = logging.getLogger(__name__)


def get_db():
    """Obtiene sesión de BD."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



def get_cache_manager(db: Session = Depends(get_db)) -> CacheManager:
    """Obtiene gestor de caché."""
    return CacheManager(db)


def get_food_resolver(
    db: Session = Depends(get_db),
    cache_manager: CacheManager = Depends(get_cache_manager),
) -> FoodSourceResolver:
    """Obtiene resolvedor de alimentos."""
    return FoodSourceResolver(db, cache_manager)


def get_plate_builder(
    db: Session = Depends(get_db),
    cache_manager: CacheManager = Depends(get_cache_manager),
) -> PlatoBuilder:
    """Obtiene constructor de platos."""
    food_resolver = FoodSourceResolver(db, cache_manager)
    return PlatoBuilder(db, food_resolver, cache_manager)


def get_routine_builder(db: Session = Depends(get_db)) -> RoutineBuilder:
    """Obtiene constructor de rutinas."""
    return RoutineBuilder(db)


def get_assistant_orchestrator(
    db: Session = Depends(get_db),
) -> AssistantOrchestrator:
    """Obtiene orquestador del asistente."""
    return AssistantOrchestrator(db)
