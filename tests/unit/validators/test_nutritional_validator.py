"""
Tests para NutritionalValidator.
"""
import pytest
from app.services.validators import NutritionalValidator


@pytest.mark.unit
class TestNutritionalValidator:
    """Tests del validador nutricional."""
    
    @pytest.fixture
    def validator(self):
        return NutritionalValidator()
    
    def test_macros_validas_atwater(self, validator):
        """Macros coherentes con Atwater son válidas."""
        # Arroz con pollo: 541.5 kcal
        # Atwater: 54.7*4 + 63*4 + 6.2*9 = 568.8 (diferencia ~5%)
        resultado = validator.validar({
            'nombre_plato': 'Arroz con pollo',
            'peso_total_gramos': 550,
            'calorias_total': 541.5,
            'proteina_total': 54.7,
            'carbohidratos_total': 63,
            'grasas_total': 6.2,
        })
        
        assert resultado.es_valido == True
        assert resultado.confianza >= 70
    
    def test_macros_invalidas_atwater(self, validator):
        """Macros inconsistentes con Atwater generan advertencia."""
        resultado = validator.validar({
            'nombre_plato': 'Plato raro',
            'peso_total_gramos': 100,
            'calorias_total': 1000,  # Muy alto para 100g
            'proteina_total': 5,
            'carbohidratos_total': 5,
            'grasas_total': 5,
        })
        
        # Atwater: 5*4 + 5*4 + 5*9 = 85 kcal (vs 1000)
        # Diferencia ENORME = advertencia o error
        assert len(resultado.advertencias) > 0 or not resultado.es_valido
    
    def test_densidad_calorica_normal(self, validator):
        """Densidad calórica normal (50-300) es OK."""
        resultado = validator.validar({
            'nombre_plato': 'Arroz cocido',
            'peso_total_gramos': 400,
            'calorias_total': 520,  # 130 kcal/100g (normal)
            'proteina_total': 10,
            'carbohidratos_total': 115,
            'grasas_total': 2,
        })
        
        assert resultado.es_valido == True
    
    def test_densidad_calorica_extrema_baja(self, validator):
        """Densidad calórica muy baja (<20) genera advertencia."""
        resultado = validator.validar({
            'nombre_plato': 'Agua con aire',
            'peso_total_gramos': 1000,
            'calorias_total': 10,  # 1 kcal/100g (imposible)
            'proteina_total': 0,
            'carbohidratos_total': 0,
            'grasas_total': 0,
        })
        
        assert len(resultado.advertencias) > 0
    
    def test_macros_negativos(self, validator):
        """Macros negativas son inválidas."""
        resultado = validator.validar({
            'nombre_plato': 'Plato imposible',
            'peso_total_gramos': 100,
            'calorias_total': -50,
            'proteina_total': 0,
            'carbohidratos_total': 0,
            'grasas_total': 0,
        })
        
        assert resultado.es_valido == False
        assert len(resultado.errores) > 0
