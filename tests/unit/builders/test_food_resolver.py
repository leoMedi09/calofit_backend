"""
Tests para FoodSourceResolver.
"""
import pytest
from app.services.nutrition.food.resolver.source_resolver import FoodSourceResolver
from app.services.nutrition.food.resolver.cache_manager import CacheManager

@pytest.mark.unit
class TestFoodSourceResolver:
    @pytest.fixture
    def resolver(self, db, mock_usda_client, mock_fatsecret_client):
        return FoodSourceResolver(
            db=db, 
            cache_manager=CacheManager(db),
            usda_client=mock_usda_client,
            fatsecret_client=mock_fatsecret_client
        )
        
    def test_resolver_init(self, resolver):
        assert resolver is not None
