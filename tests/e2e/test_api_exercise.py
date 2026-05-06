"""
Tests E2E: Endpoints de Ejercicio.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.mark.e2e
class TestExerciseAPI:
    @pytest.fixture
    def client(self):
        return TestClient(app)
        
    def test_placeholder(self):
        assert True
