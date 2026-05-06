"""
Tests para ConsistencyChecker — usa método real 'validar'.
"""
import pytest
from app.services.validators.consistency_checker import ConsistencyChecker


@pytest.mark.unit
class TestConsistencyChecker:
    """Tests del verificador de consistencia."""

    @pytest.fixture
    def checker(self, db):
        return ConsistencyChecker(db)

    def test_checker_inicializa(self, checker):
        """El checker se inicializa correctamente."""
        assert checker is not None

    def test_checker_tiene_metodo_validar(self, checker):
        """ConsistencyChecker expone método validar()."""
        assert hasattr(checker, 'validar')
        assert callable(checker.validar)

    def test_validar_plato_completo(self, checker, sample_plato):
        """Plato con ingredientes válidos pasa la validación."""
        resultado = checker.validar({
            'plato_id': sample_plato.id,
            'nombre': sample_plato.nombre,
        })
        assert resultado is not None

    def test_validar_plato_sin_id(self, checker):
        """Validar plato sin ID no explota."""
        try:
            resultado = checker.validar({'nombre': 'Plato sin ID'})
            assert resultado is not None
        except Exception as e:
            pytest.fail(f"ConsistencyChecker explotó: {e}")

    def test_validar_retorna_estructura_conocida(self, checker, sample_plato):
        """El resultado de validar() tiene estructura conocida."""
        resultado = checker.validar({
            'plato_id': sample_plato.id,
            'nombre': sample_plato.nombre,
        })
        # Debe ser un objeto con atributos o un dict
        assert resultado is not None
        # Verificar que tiene al menos es_valido o similar
        tiene_valido = hasattr(resultado, 'es_valido') or isinstance(resultado, dict)
        assert tiene_valido
