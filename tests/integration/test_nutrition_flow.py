"""
Tests de integración: flujo completo de nutrición.
"""
import pytest
from app.services.nutrition.plate.plate_builder import PlatoBuilder
from app.services.nutrition.food.resolver.source_resolver import FoodSourceResolver
from app.services.nutrition.food.resolver.cache_manager import CacheManager
from app.models import ProgresoCalorias
from datetime import date


@pytest.mark.integration
class TestNutritionFlow:
    """Tests del flujo completo de nutrición."""
    
    @pytest.fixture
    def setup(self, db, sample_client, sample_alimentos, mock_usda_client, mock_fatsecret_client):
        cache_manager = CacheManager(db)
        resolver = FoodSourceResolver(
            db=db,
            cache_manager=cache_manager,
            usda_client=mock_usda_client,
            fatsecret_client=mock_fatsecret_client,
        )
        builder = PlatoBuilder(db, resolver, cache_manager)
        return {'builder': builder, 'db': db, 'client': sample_client}
    
    def test_flujo_completo_recomendacion_consumo(self, setup):
        """
        Flujo completo:
        1. Recomendar plato
        2. Usuario confirma
        3. Registrar consumo
        4. Actualizar progreso
        """
        builder = setup['builder']
        db = setup['db']
        client = setup['client']
        
        # 1. Recomendar
        resultado = builder.construir_plato(
            nombre_plato="Almuerzo balanceado",
            ingredientes=[
                {'nombre': 'arroz blanco cocido', 'gramos': 200},
                {'nombre': 'pollo pechuga cocida', 'gramos': 150},
                {'nombre': 'brocoli cocido', 'gramos': 100},
            ],
            client_id=client.id,
        )
        
        assert resultado.exito
        plato_id = resultado.plato_id
        
        # 2. Registrar consumo (simulado)
        progreso = ProgresoCalorias(
            client_id=client.id,
            fecha=date.today(),
            calorias_consumidas=resultado.macros_totales.calorias,
            proteinas_consumidas=resultado.macros_totales.proteina,
            carbohidratos_consumidos=resultado.macros_totales.carbohidratos,
            grasas_consumidas=resultado.macros_totales.grasas,
        )
        db.add(progreso)
        db.commit()
        
        # 3. Verificar actualización
        progreso_verificar = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == client.id,
            ProgresoCalorias.fecha == date.today(),
        ).first()
        
        assert progreso_verificar is not None
        assert progreso_verificar.calorias_consumidas > 0
        assert progreso_verificar.proteinas_consumidas > 0
