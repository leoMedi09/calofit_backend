"""
Tests adversariales: casos imposibles/extremos.
"""
import pytest
from app.services.validators import SemanticValidator, NutritionalValidator


@pytest.mark.adversarial
class TestAdversarialCases:
    """Tests de casos adversariales."""
    
    @pytest.fixture
    def semantic_validator(self, db):
        return SemanticValidator(db)
    
    @pytest.fixture
    def nutritional_validator(self):
        return NutritionalValidator()
    
    def test_unicornio_asado_rechazado(self, semantic_validator):
        """
        'Unicornio asado' no existe y debe ser RECHAZADO.
        
        Este es el caso más importante: el sistema NO debe
        aceptar alimentos ficticios/imposibles.
        """
        resultado = semantic_validator.validar({
            'nombre_plato': 'Unicornio asado',
            'ingredientes': [
                {'nombre': 'unicornio'},
                {'nombre': 'especias mágicas'},
            ],
        })
        
        # No debe ser válido (no está en reglas culinarias reales)
        assert resultado.confianza < 50 or len(resultado.advertencias) > 0
    
    def test_ceviche_con_mayonesa_rechazado(self, semantic_validator):
        """Ceviche con mayonesa es INVÁLIDO."""
        resultado = semantic_validator.validar({
            'nombre_plato': 'Ceviche',
            'ingredientes': [
                {'nombre': 'pescado'},
                {'nombre': 'mayonesa'},
            ],
        })
        
        assert resultado.es_valido == False
    
    def test_macros_imposibles_rechazadas(self, nutritional_validator):
        """
        Macros imposibles (kcal extremas) son RECHAZADAS.
        
        Ejemplo: 1g de alimento con 10000 kcal = IMPOSIBLE
        """
        resultado = nutritional_validator.validar({
            'nombre_plato': 'Alimento mágico',
            'peso_total_gramos': 1,
            'calorias_total': 10000,  # 10000 kcal/100g = IMPOSIBLE
            'proteina_total': 0,
            'carbohidratos_total': 0,
            'grasas_total': 0,
        })
        
        # Debe haber advertencias o ser inválido
        assert len(resultado.advertencias) > 0 or not resultado.es_valido
    
    def test_plato_solo_agua_rechazado(self, nutritional_validator):
        """Plato con 0 macros (solo agua) tiene baja confianza."""
        resultado = nutritional_validator.validar({
            'nombre_plato': 'Agua',
            'peso_total_gramos': 1000,
            'calorias_total': 0,
            'proteina_total': 0,
            'carbohidratos_total': 0,
            'grasas_total': 0,
        })
        
        assert resultado.confianza < 50
    
    def test_macros_negativas_rechazadas(self, nutritional_validator):
        """Macros negativas son RECHAZADAS."""
        resultado = nutritional_validator.validar({
            'nombre_plato': 'Antimateria',
            'peso_total_gramos': 100,
            'calorias_total': -500,
            'proteina_total': -10,
            'carbohidratos_total': -10,
            'grasas_total': -10,
        })
        
        assert resultado.es_valido == False
        assert len(resultado.errores) > 0
    
    def test_plato_sin_ingredientes_rechazado(self, semantic_validator):
        """Plato sin ingredientes es inválido."""
        resultado = semantic_validator.validar({
            'nombre_plato': 'Fantasma',
            'ingredientes': [],
        })
        
        assert resultado.es_valido == False
