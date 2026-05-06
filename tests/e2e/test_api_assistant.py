"""
Tests E2E: Endpoints del Asistente.
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
    
    def test_chat_endpoint_exists(self, client):
        """POST /api/v1/assistant/chat existe."""
        response = client.post(
            "/api/v1/assistant/chat",
            json={"mensaje": "Hola"},
            headers={"X-Client-ID": "1"},
        )
        
        # Puede fallar por auth pero endpoint debe existir
        assert response.status_code in [200, 401, 422]
    
    def test_nutrition_plates_endpoint(self, client):
        """POST /api/v1/nutrition/plates existe."""
        response = client.post(
            "/api/v1/nutrition/plates",
            json={
                "objetivo": "alto_proteína",
                "momento_dia": "almuerzo",
                "cantidad": 3,
            },
            headers={"X-Client-ID": "1"},
        )
        
        assert response.status_code in [200, 401, 422]
    
    def test_exercise_routines_endpoint(self, client):
        """POST /api/v1/exercise/routines existe."""
        response = client.post(
            "/api/v1/exercise/routines",
            json={
                "objetivo": "hipertrofia",
                "duracion_minutos": 45,
                "nivel": "intermedio",
            },
            headers={"X-Client-ID": "1"},
        )
        
        assert response.status_code in [200, 401, 422]
