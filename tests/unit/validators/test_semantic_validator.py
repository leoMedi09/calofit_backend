"""
Tests para SemanticValidator.
"""
import pytest
from app.services.validators import SemanticValidator


@pytest.mark.unit
class TestSemanticValidator:
    """Tests del validador semántico."""
    
    @pytest.fixture
    def validator(self, db):
        return SemanticValidator(db)
    
    def test_ceviche_valido(self, validator):
        """Ceviche con pescado y limón es válido."""
        resultado = validator.validar({
            'nombre_plato': 'Ceviche de atún',
            'ingredientes': [
                {'nombre': 'atún'},
                {'nombre': 'limón'},
            ],
        })
        
        assert resultado.es_valido == True
        assert resultado.confianza >= 80
    
    def test_ceviche_invalido_con_mayonesa(self, validator):
        """Ceviche con mayonesa es INVÁLIDO."""
        resultado = validator.validar({
            'nombre_plato': 'Ceviche',
            'ingredientes': [
                {'nombre': 'pescado'},
                {'nombre': 'mayonesa'},
            ],
        })
        
        assert resultado.es_valido == False
        assert len(resultado.errores) > 0
        assert any('mayonesa' in err.lower() for err in resultado.errores)
    
    def test_plato_sin_ingredientes(self, validator):
        """Plato sin ingredientes es inválido."""
        resultado = validator.validar({
            'nombre_plato': 'Arroz con pollo',
            'ingredientes': [],
        })
        
        assert resultado.es_valido == False
    
    def test_ensalada_valida(self, validator):
        """Ensalada con verduras es válida."""
        resultado = validator.validar({
            'nombre_plato': 'Ensalada mixta',
            'ingredientes': [
                {'nombre': 'lechuga'},
                {'nombre': 'tomate'},
            ],
        })
        
        assert resultado.es_valido == True
    
    def test_restricciones_cliente(self, validator, sample_client, db):
        """Respeta restricciones del cliente."""
        # Agregar restricción
        sample_client.forbidden_foods = ['lácteos', 'gluten']
        db.commit()
        
        resultado = validator.validar({
            'nombre_plato': 'Pasta con queso',
            'ingredientes': [
                {'nombre': 'pasta'},
                {'nombre': 'queso'},
            ],
            'client_id': sample_client.id,
        })
        
        assert resultado.es_valido == False
