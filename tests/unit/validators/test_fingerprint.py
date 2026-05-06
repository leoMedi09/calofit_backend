"""
Tests para FingerprintGenerator.
"""
import pytest
from app.services.validators import FingerprintGenerator


@pytest.mark.unit
class TestFingerprintGenerator:
    """Tests del generador de fingerprint."""
    
    def test_fingerprint_deterministico(self):
        """Mismo plato = mismo fingerprint siempre."""
        fp1 = FingerprintGenerator.generar_fingerprint_plato(
            nombre="Arroz con pollo",
            ingredientes=[
                {'nombre': 'arroz', 'gramos': 200},
                {'nombre': 'pollo', 'gramos': 150},
            ],
            macros={'calorias': 500, 'proteina': 45, 'carbohidratos': 50, 'grasas': 10},
        )
        
        fp2 = FingerprintGenerator.generar_fingerprint_plato(
            nombre="Arroz con pollo",
            ingredientes=[
                {'nombre': 'arroz', 'gramos': 200},
                {'nombre': 'pollo', 'gramos': 150},
            ],
            macros={'calorias': 500, 'proteina': 45, 'carbohidratos': 50, 'grasas': 10},
        )
        
        assert fp1 == fp2
        assert len(fp1) == 64  # SHA256 = 64 caracteres hex
    
    def test_fingerprint_cambio_ingrediente(self):
        """Cambio en ingrediente = fingerprint diferente."""
        fp1 = FingerprintGenerator.generar_fingerprint_plato(
            nombre="Arroz con pollo",
            ingredientes=[
                {'nombre': 'arroz', 'gramos': 200},
                {'nombre': 'pollo', 'gramos': 150},
            ],
            macros={'calorias': 500, 'proteina': 45, 'carbohidratos': 50, 'grasas': 10},
        )
        
        fp2 = FingerprintGenerator.generar_fingerprint_plato(
            nombre="Arroz con pollo",
            ingredientes=[
                {'nombre': 'arroz', 'gramos': 200},
                {'nombre': 'pollo', 'gramos': 200},  # Cambio: 150 → 200
            ],
            macros={'calorias': 500, 'proteina': 45, 'carbohidratos': 50, 'grasas': 10},
        )
        
        assert fp1 != fp2
    
    def test_comparar_fingerprints(self):
        """Comparación de fingerprints."""
        fp1 = "abc123"
        fp2 = "abc123"
        fp3 = "xyz789"
        
        assert FingerprintGenerator.comparar_fingerprints(fp1, fp2) == True
        assert FingerprintGenerator.comparar_fingerprints(fp1, fp3) == False
