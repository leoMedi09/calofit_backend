"""
Tests E2E: Endpoints del Asistente.

Nota: los tests contra /api/v1/assistant/chat, /api/v1/nutrition/plates y
/api/v1/exercise/routines se eliminaron — esas rutas fueron retiradas a
propósito el 2026-06-13 (ver CLAUDE.md, "Eliminación de la arquitectura
nueva"). Flutter usa /asistente/consultar (legacy), no /api/v1/assistant.
Mantenerlos generaba falsos negativos (404 esperado, no un bug).
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.mark.e2e
class TestAssistantAPI:
    """Tests E2E del asistente conversacional."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_check(self, client):
        """GET /health retorna 200."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "OK"
