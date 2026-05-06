"""
Tests para AssistantOrchestrator — flujo completo del asistente.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from app.services.assistant.assistant_orchestrator import AssistantOrchestrator


@pytest.mark.unit
class TestAssistantOrchestrator:
    """Tests del orquestador del asistente."""

    @pytest.fixture
    def orchestrator(self, db):
        return AssistantOrchestrator(db)

    def test_orchestrator_inicializa(self, orchestrator):
        """El orquestador se inicializa con sus dependencias."""
        assert orchestrator is not None
        assert orchestrator.db is not None
        assert orchestrator._intent is not None
        assert orchestrator._ml is not None

    def test_extraer_features_con_perfil_completo(self, orchestrator):
        """_extraer_features extrae correctamente del contexto."""
        contexto = {
            "perfil": {
                "peso_kg": 75,
                "altura_cm": 180,
                "edad": 30,
                "genero": "M",
            },
            "historial_reciente": [
                {"kcal_consumidas": 2000},
                {"kcal_consumidas": 2200},
            ],
        }
        features = orchestrator._extraer_features(contexto)

        assert features["Weight (kg)"] == 75
        assert features["Height (m)"] == 1.80
        assert features["Age"] == 30
        assert features["Gender"] == 1  # M → 1
        assert features["avg_kcal_7d"] == 2100.0

    def test_extraer_features_con_perfil_vacio(self, orchestrator):
        """_extraer_features usa defaults si el perfil está vacío."""
        features = orchestrator._extraer_features({})

        assert features["Weight (kg)"] == 70  # default
        assert features["Age"] == 30          # default
        assert features["Gender"] == 0        # default F

    def test_extraer_features_genero_femenino(self, orchestrator):
        """Género F → 0."""
        contexto = {"perfil": {"genero": "F"}, "historial_reciente": []}
        features = orchestrator._extraer_features(contexto)
        assert features["Gender"] == 0

    def test_handler_para_nutricion(self, orchestrator):
        """_handler_para retorna NutritionHandler para intent nutricion."""
        from app.services.assistant.handlers.nutrition_handler import NutritionHandler
        from app.services.assistant.intent_classifier import INTENT_NUTRICION

        handler = orchestrator._handler_para(INTENT_NUTRICION, None)
        assert isinstance(handler, NutritionHandler)

    def test_handler_para_ejercicio(self, orchestrator):
        """_handler_para retorna ExerciseHandler para intent ejercicio."""
        from app.services.assistant.handlers.exercise_handler import ExerciseHandler
        from app.services.assistant.intent_classifier import INTENT_EJERCICIO

        handler = orchestrator._handler_para(INTENT_EJERCICIO, None)
        assert isinstance(handler, ExerciseHandler)

    def test_handler_para_chat(self, orchestrator):
        """_handler_para retorna ChatHandler para intent chat."""
        from app.services.assistant.handlers.chat_handler import ChatHandler
        from app.services.assistant.intent_classifier import INTENT_CHAT

        handler = orchestrator._handler_para(INTENT_CHAT, None)
        assert isinstance(handler, ChatHandler)

    def test_handler_intent_desconocido(self, orchestrator):
        """_handler_para con intent desconocido usa ChatHandler como fallback."""
        from app.services.assistant.handlers.chat_handler import ChatHandler

        handler = orchestrator._handler_para("intent_inexistente_xyz", None)
        assert isinstance(handler, ChatHandler)

    def test_consultar_retorna_dict(self, orchestrator, sample_client):
        """consultar() siempre retorna un dict estructurado."""
        resultado = asyncio.run(
            orchestrator.consultar(
                mensaje="hola",
                client_id=sample_client.id,
            )
        )
        assert isinstance(resultado, dict)
        # Debe tener al menos una clave de respuesta
        assert len(resultado) > 0

    def test_consultar_mensaje_nutricion(self, orchestrator, sample_client):
        """consultar() con mensaje de nutrición no explota."""
        resultado = asyncio.run(
            orchestrator.consultar(
                mensaje="dame opciones de desayuno alto en proteína",
                client_id=sample_client.id,
            )
        )
        assert isinstance(resultado, dict)

    def test_consultar_con_client_inexistente(self, orchestrator):
        """consultar() con client_id inexistente maneja el error graciosamente."""
        resultado = asyncio.run(
            orchestrator.consultar(
                mensaje="hola",
                client_id=999999,  # No existe
            )
        )
        # No debe explotar — debe retornar dict con mensaje de error o respuesta
        assert isinstance(resultado, dict)
