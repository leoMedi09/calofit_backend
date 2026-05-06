"""
Tests de integración: flujo completo de ejercicio.
Usa columnas reales de WorkoutSession: duracion_min, calorias_quemadas.
"""
import pytest
from datetime import date
from app.models import WorkoutSession, ProgresoCalorias


@pytest.mark.integration
class TestExerciseFlow:
    """Tests del flujo completo de ejercicio."""

    def test_registrar_sesion_persiste_en_bd(self, db, sample_client):
        """Registrar sesión de ejercicio persiste en BD."""
        sesion = WorkoutSession(
            client_id=sample_client.id,
            fecha=date.today(),
            duracion_min=45,
            calorias_quemadas=350.0,
            notas="Sesión de prueba",
        )
        db.add(sesion)
        db.commit()
        db.refresh(sesion)
        assert sesion.id is not None
        assert sesion.calorias_quemadas == 350.0
        assert sesion.duracion_min == 45

    def test_progreso_calorico_se_guarda(self, db, sample_client):
        """El progreso calórico del día se persiste correctamente."""
        progreso = ProgresoCalorias(
            client_id=sample_client.id,
            fecha=date.today(),
            calorias_consumidas=2000,
            calorias_quemadas=350,
            proteinas_consumidas=150,
            carbohidratos_consumidos=250,
            grasas_consumidas=60,
        )
        db.add(progreso)
        db.commit()

        guardado = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == sample_client.id,
            ProgresoCalorias.fecha == date.today(),
        ).first()

        assert guardado is not None
        assert guardado.calorias_quemadas == 350
        assert guardado.calorias_consumidas == 2000

    def test_balance_neto_correcto(self, db, sample_client):
        """Balance neto = consumidas - quemadas."""
        p = ProgresoCalorias(
            client_id=sample_client.id,
            fecha=date.today(),
            calorias_consumidas=2200,
            calorias_quemadas=400,
            proteinas_consumidas=160,
            carbohidratos_consumidos=260,
            grasas_consumidas=70,
        )
        db.add(p)
        db.commit()
        assert (p.calorias_consumidas - p.calorias_quemadas) == 1800

    def test_multiples_sesiones_mismo_dia(self, db, sample_client):
        """Se pueden registrar múltiples sesiones el mismo día."""
        s1 = WorkoutSession(
            client_id=sample_client.id,
            fecha=date.today(),
            duracion_min=30,
            calorias_quemadas=200.0,
        )
        s2 = WorkoutSession(
            client_id=sample_client.id,
            fecha=date.today(),
            duracion_min=20,
            calorias_quemadas=150.0,
        )
        db.add_all([s1, s2])
        db.commit()

        sesiones = db.query(WorkoutSession).filter(
            WorkoutSession.client_id == sample_client.id,
            WorkoutSession.fecha == date.today(),
        ).all()
        assert len(sesiones) == 2
        total_kcal = sum(s.calorias_quemadas for s in sesiones)
        assert total_kcal == 350.0
