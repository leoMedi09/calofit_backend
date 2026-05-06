"""
Resolvedores de alimentos (ingredientes).
"""
from app.services.nutrition.food.resolver.source_resolver import FoodSourceResolver
from app.services.nutrition.food.resolver.cache_manager import CacheManager
from app.services.nutrition.food.resolver.api_clients import (
    USDAClient,
    FatSecretClient,
)

__all__ = [
    "FoodSourceResolver",
    "CacheManager",
    "USDAClient",
    "FatSecretClient",
]
