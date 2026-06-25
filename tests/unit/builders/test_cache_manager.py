"""
Tests para CacheManager — caché inteligente de alimentos.
"""
import pytest
from datetime import datetime, timedelta, timezone
from app.services.nutrition.food.resolver.cache_manager import CacheManager


@pytest.mark.unit
class TestCacheManager:
    """Tests del gestor de caché de alimentos."""

    @pytest.fixture
    def cache(self, db):
        return CacheManager(db)

    @pytest.fixture
    def macros_arroz(self):
        return {
            "calorias_100g": 130.0,
            "proteina_100g": 2.7,
            "carbohidratos_100g": 28.0,
            "grasas_100g": 0.3,
            "fibra_100g": 0.4,
            "azucar_100g": 0.0,
        }

    def test_guardar_y_recuperar(self, cache, sample_client, macros_arroz):
        """Guardar en caché y recuperar correctamente."""
        ok = cache.guardar_en_cache(
            food_normalized="arroz blanco",
            user_id=sample_client.id,
            macros=macros_arroz,
            source="BD",
        )
        assert ok is True

        resultado = cache.obtener_del_cache("arroz blanco", sample_client.id)
        assert resultado is not None
        # obtener_del_cache() devuelve {"macros": {...}, "alimento_id": ...}
        # (contrato actual de cache_manager.py) — no los macros planos.
        assert resultado["macros"]["calorias_100g"] == 130.0
        assert resultado["macros"]["proteina_100g"] == 2.7

    def test_cache_inexistente_retorna_none(self, cache, sample_client):
        """Alimento no cacheado retorna None."""
        resultado = cache.obtener_del_cache("unicornio asado", sample_client.id)
        assert resultado is None

    def test_normaliza_nombre(self, cache, sample_client, macros_arroz):
        """El caché normaliza el nombre (lowercase, sin espacios)."""
        cache.guardar_en_cache("ARROZ BLANCO  ", sample_client.id, macros_arroz, "BD")
        resultado = cache.obtener_del_cache("arroz blanco", sample_client.id)
        assert resultado is not None

    def test_actualizar_cache_existente(self, cache, sample_client, macros_arroz):
        """Guardar dos veces el mismo alimento actualiza sin duplicar."""
        cache.guardar_en_cache("pollo", sample_client.id, macros_arroz, "BD")

        macros_actualizados = {**macros_arroz, "calorias_100g": 165.0}
        cache.guardar_en_cache("pollo", sample_client.id, macros_actualizados, "USDA")

        resultado = cache.obtener_del_cache("pollo", sample_client.id)
        assert resultado["macros"]["calorias_100g"] == 165.0

    def test_invalidar_cache(self, cache, sample_client, macros_arroz):
        """Invalidar entrada elimina del caché."""
        cache.guardar_en_cache("tomate", sample_client.id, macros_arroz, "BD")
        
        ok = cache.invalidar_cache("tomate", sample_client.id)
        assert ok is True

        resultado = cache.obtener_del_cache("tomate", sample_client.id)
        assert resultado is None

    def test_limpiar_expirado(self, cache, sample_client, macros_arroz, db):
        """Entradas expiradas se eliminan al limpiar."""
        from app.models import AppCacheAlimentos
        import json

        # Insertar entrada con expiración pasada
        entry = AppCacheAlimentos(
            food_normalized="alimento_viejo",
            user_id=sample_client.id,
            source="BD",
            raw_response=json.dumps({"macros": macros_arroz, "source": "BD"}),
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),  # Expirado
        )
        db.add(entry)
        db.commit()

        eliminados = cache.limpiar_cache_expirado(sample_client.id)
        assert eliminados >= 1

    def test_cache_diferente_por_usuario(self, cache, sample_client, macros_arroz, db):
        """El caché es por usuario — otro usuario no ve el mismo caché."""
        from app.models import Client
        
        otro_client = Client(
            first_name="Otro",
            last_name_paternal="Test",
            last_name_maternal="Demo",
            dni="87654321",
            email=f"otro_{datetime.utcnow().timestamp()}@test.com",
            hashed_password="pwd",
            birth_date=datetime(1995, 5, 5).date(),
            weight=65.0,
            height=170,
            gender="F",
            medical_conditions=[],
            activity_level="bajo",
            goal="mantenimiento",
            is_profile_complete=True,
        )
        db.add(otro_client)
        db.commit()
        db.refresh(otro_client)

        cache.guardar_en_cache("manzana", sample_client.id, macros_arroz, "BD")

        resultado = cache.obtener_del_cache("manzana", otro_client.id)
        assert resultado is None
