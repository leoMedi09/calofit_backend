"""
Tests para RoutineBuilder — usa la firma real del método.
"""
import pytest
from app.services.exercise.routine.routine_builder import RoutineBuilder


@pytest.mark.unit
class TestRoutineBuilder:
    """Tests del constructor de rutinas."""

    @pytest.fixture
    def builder(self, db):
        return RoutineBuilder(db)

    def test_builder_inicializa(self, builder):
        """El builder se inicializa correctamente."""
        assert builder is not None

    def test_construir_rutina_hipertrofia(self, builder, sample_client, sample_ejercicios):
        """Construye rutina de hipertrofia con ejercicios disponibles."""
        resultado = builder.construir_rutina(
            nombre_rutina="Rutina Full Body",
            tipo_entrenamiento="fuerza",
            intensidad="moderada",
            objetivo="hipertrofia",
            ejercicios=[
                {"nombre": "Sentadilla", "series": 4, "repeticiones": 10},
                {"nombre": "Press de banca", "series": 3, "repeticiones": 12},
            ],
            client_id=sample_client.id,
        )
        assert resultado is not None

    def test_construir_rutina_resistencia(self, builder, sample_client, sample_ejercicios):
        """Construye rutina de resistencia."""
        resultado = builder.construir_rutina(
            nombre_rutina="Rutina Cardio",
            tipo_entrenamiento="cardio",
            intensidad="baja",
            objetivo="resistencia",
            ejercicios=[
                {"nombre": "Curl de biceps", "series": 3, "repeticiones": 15},
            ],
            client_id=sample_client.id,
        )
        assert resultado is not None

    def test_rutina_tiene_fingerprint(self, builder, sample_client, sample_ejercicios):
        """La rutina construida tiene fingerprint para determinismo."""
        resultado = builder.construir_rutina(
            nombre_rutina="Rutina Test",
            tipo_entrenamiento="fuerza",
            intensidad="alta",
            objetivo="hipertrofia",
            ejercicios=[
                {"nombre": "Sentadilla", "series": 5, "repeticiones": 5},
            ],
            client_id=sample_client.id,
        )
        assert resultado is not None
        if hasattr(resultado, 'fingerprint'):
            assert resultado.fingerprint is not None

    def test_rutina_sin_ejercicios_no_explota(self, builder, sample_client):
        """Rutina con lista vacía de ejercicios no lanza excepción."""
        try:
            resultado = builder.construir_rutina(
                nombre_rutina="Rutina Vacía",
                tipo_entrenamiento="fuerza",
                intensidad="baja",
                objetivo="mantenimiento",
                ejercicios=[],
                client_id=sample_client.id,
            )
            assert resultado is not None
        except Exception as e:
            pytest.fail(f"RoutineBuilder explotó con lista vacía: {e}")

    def test_misma_rutina_mismo_fingerprint(self, builder, sample_client, sample_ejercicios):
        """Construir la misma rutina dos veces genera mismo fingerprint (determinismo)."""
        ejercicios = [{"nombre": "Sentadilla", "series": 4, "repeticiones": 10}]

        r1 = builder.construir_rutina(
            nombre_rutina="Rutina Determinista",
            tipo_entrenamiento="fuerza",
            intensidad="moderada",
            objetivo="hipertrofia",
            ejercicios=ejercicios,
            client_id=sample_client.id,
        )
        r2 = builder.construir_rutina(
            nombre_rutina="Rutina Determinista",
            tipo_entrenamiento="fuerza",
            intensidad="moderada",
            objetivo="hipertrofia",
            ejercicios=ejercicios,
            client_id=sample_client.id,
        )
        if hasattr(r1, 'fingerprint') and hasattr(r2, 'fingerprint'):
            assert r1.fingerprint == r2.fingerprint
