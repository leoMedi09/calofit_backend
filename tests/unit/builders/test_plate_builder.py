"""
Tests para PlatoBuilder.
"""
import pytest
from app.services.nutrition.plate.plate_builder import PlatoBuilder
from app.services.nutrition.food.resolver.source_resolver import FoodSourceResolver
from app.services.nutrition.food.resolver.cache_manager import CacheManager


@pytest.mark.unit
class TestPlatoBuilder:
    """Tests del constructor de platos."""

    @pytest.fixture
    def builder(self, db, sample_alimentos, mock_usda_client, mock_fatsecret_client):
        cache_manager = CacheManager(db)
        resolver = FoodSourceResolver(
            db=db,
            cache_manager=cache_manager,
            usda_client=mock_usda_client,
            fatsecret_client=mock_fatsecret_client,
        )
        return PlatoBuilder(db, resolver, cache_manager)

    def test_construir_plato_valido(self, builder, sample_client):
        """Construye plato válido exitosamente (con ingredientes en BD)."""
        resultado = builder.construir_plato(
            nombre_plato="Arroz con pollo",
            ingredientes=[
                {'nombre': 'arroz blanco cocido', 'gramos': 200},
                {'nombre': 'pollo pechuga cocida', 'gramos': 150},
            ],
            client_id=sample_client.id,
        )

        # Debe haber resuelto los ingredientes y calculado macros
        assert resultado.macros_totales is not None
        assert resultado.macros_totales.calorias > 0
        assert resultado.confianza_global >= 60

    def test_plato_en_cache(self, builder, sample_client):
        """
        Construir el mismo plato dos veces.
        La segunda llamada debe usar caché o al menos tener ingredientes resueltos.
        Usa solo ingredientes que existen en la BD de test (sin acentos problemáticos).
        """
        ingredientes = [
            {'nombre': 'arroz blanco cocido', 'gramos': 200},
            {'nombre': 'pollo pechuga cocida', 'gramos': 150},
        ]

        resultado1 = builder.construir_plato(
            nombre_plato="Arroz con pollo test",
            ingredientes=ingredientes,
            client_id=sample_client.id,
        )

        resultado2 = builder.construir_plato(
            nombre_plato="Arroz con pollo test",
            ingredientes=ingredientes,
            client_id=sample_client.id,
        )

        # Ambas construcciones deben tener macros calculadas
        assert resultado1.macros_totales.calorias > 0
        assert resultado2.macros_totales.calorias > 0
        # Las macros deben ser idénticas (determinismo)
        assert resultado1.macros_totales.calorias == resultado2.macros_totales.calorias

    def test_ingrediente_no_resuelto_es_advertencia(self, builder, sample_client):
        """
        Ingrediente desconocido no debe romper la construcción,
        sino registrarse como no resuelto con confianza reducida.
        """
        resultado = builder.construir_plato(
            nombre_plato="Plato con ingrediente raro",
            ingredientes=[
                {'nombre': 'arroz blanco cocido', 'gramos': 200},
                {'nombre': 'ingrediente_inexistente_xyz', 'gramos': 50},
            ],
            client_id=sample_client.id,
        )

        # No debe explotar — debe retornar un resultado con advertencias
        assert resultado is not None
        assert resultado.macros_totales is not None
        # Los ingredientes resueltos deben tener macros
        ingredientes_resueltos = [
            i for i in resultado.ingredientes
            if i.macros_totales is not None
        ]
        assert len(ingredientes_resueltos) >= 1
