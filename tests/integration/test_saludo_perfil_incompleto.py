"""
Prueba del saludo cuando el perfil no tiene first_name (usuario nuevo /
onboarding incompleto). Encontrado en la auditoría final pre-demo: el saludo
quedaba con una coma colgando ("Buenos días, . ¿En qué te ayudo hoy?") cuando
el nombre estaba vacío. Ver asistente_service.py, bloque de saludo puro.
"""
from datetime import datetime

import pytest

from app.models.client import Client
from app.services.asistente.asistente_service import AsistenteService


@pytest.mark.integration
class TestSaludoPerfilIncompleto:

    @pytest.fixture
    def cliente_sin_nombre(self, db):
        client = Client(
            first_name=None,
            last_name_paternal=None,
            last_name_maternal=None,
            email=f"sin_nombre_{datetime.utcnow().timestamp()}@test.com",
            hashed_password="hashedpwd123",
            gender="M",
            is_profile_complete=False,
        )
        db.add(client)
        db.commit()
        db.refresh(client)
        return client

    @pytest.mark.asyncio
    async def test_saludo_sin_nombre_no_deja_coma_colgando(self, db, cliente_sin_nombre):
        class MockUser:
            email = cliente_sin_nombre.email

        asistente = AsistenteService()
        resp = await asistente.consultar(
            mensaje="Hola", db=db, current_user=MockUser(), historial=[]
        )
        texto = resp["respuesta_ia"]
        assert ", ." not in texto
        assert ",." not in texto
        assert texto.split(".")[0].strip() in ("Buenos días", "Buenas tardes", "Buenas noches")
        # No debe inventar un nombre que no existe en el perfil.
        assert "None" not in texto
