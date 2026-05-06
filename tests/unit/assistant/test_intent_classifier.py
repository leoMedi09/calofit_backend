"""
Tests para IntentClassifier — clasificador de intenciones del asistente.
"""
import pytest
import asyncio
from app.services.assistant.intent_classifier import (
    IntentClassifier,
    INTENT_NUTRICION,
    INTENT_EJERCICIO,
    INTENT_INTEGRADO,
    INTENT_PLAN,
    INTENT_PROGRESO,
    INTENT_CHAT,
)


@pytest.mark.unit
class TestIntentClassifier:
    """Tests del clasificador de intenciones."""

    @pytest.fixture
    def classifier(self):
        return IntentClassifier(llm_service=None)

    # ─── Casos de nutrición ───────────────────────────────────────────────────

    def test_detecta_comer(self, classifier):
        """'qué puedo comer' → nutricion."""
        result = classifier.clasificar_heuristico("qué puedo comer hoy")
        assert result == INTENT_NUTRICION

    def test_detecta_desayuno(self, classifier):
        """'dame opciones de desayuno' → nutricion."""
        result = classifier.clasificar_heuristico("dame opciones de desayuno")
        assert result == INTENT_NUTRICION

    def test_detecta_proteina(self, classifier):
        """'necesito más proteína' → nutricion."""
        result = classifier.clasificar_heuristico("necesito más proteína hoy")
        assert result == INTENT_NUTRICION

    def test_detecta_calorias(self, classifier):
        """'caloría' dispara intent nutricion."""
        result = classifier.clasificar_heuristico("necesito controlar mi caloría diaria")
        assert result == INTENT_NUTRICION

    def test_detecta_registrar(self, classifier):
        """'registra mi almuerzo' → nutricion."""
        result = classifier.clasificar_heuristico("registra mi almuerzo de hoy")
        assert result == INTENT_NUTRICION

    # ─── Casos de ejercicio ───────────────────────────────────────────────────

    def test_detecta_rutina(self, classifier):
        """'dame una rutina' → ejercicio."""
        result = classifier.clasificar_heuristico("dame una rutina para hoy")
        assert result == INTENT_EJERCICIO

    def test_detecta_entrenamiento(self, classifier):
        """'entrenamiento' sin 'plan' → ejercicio (con plan activa INTENT_PLAN)."""
        result = classifier.clasificar_heuristico("quiero ver mi rutina de entrenamiento")
        assert result == INTENT_EJERCICIO

    def test_detecta_sentadilla(self, classifier):
        """'sentadilla' → ejercicio (verificar ortografía exacta del regex)."""
        result = classifier.clasificar_heuristico("voy a hacer sentadilla hoy")
        assert result == INTENT_EJERCICIO

    def test_detecta_cardio(self, classifier):
        """'cardio' explícito → ejercicio."""
        result = classifier.clasificar_heuristico("quiero hacer cardio hoy")
        assert result == INTENT_EJERCICIO

    # ─── Casos integrados ─────────────────────────────────────────────────────

    def test_detecta_integrado(self, classifier):
        """Mensaje con comida + ejercicio → integrado."""
        result = classifier.clasificar_heuristico("qué comer antes del entrenamiento")
        assert result == INTENT_INTEGRADO

    # ─── Casos de plan ────────────────────────────────────────────────────────

    def test_detecta_plan_semanal(self, classifier):
        """'plan semanal' → plan."""
        result = classifier.clasificar_heuristico("dame un plan semanal")
        assert result == INTENT_PLAN

    def test_detecta_dieta(self, classifier):
        """'quiero una dieta' → plan."""
        result = classifier.clasificar_heuristico("quiero una dieta para bajar de peso")
        assert result == INTENT_PLAN

    # ─── Casos de progreso ────────────────────────────────────────────────────

    def test_detecta_historial(self, classifier):
        """'muéstrame mi historial' → progreso."""
        result = classifier.clasificar_heuristico("muéstrame mi historial")
        assert result == INTENT_PROGRESO

    def test_detecta_balance(self, classifier):
        """'cuál es mi balance' → progreso."""
        result = classifier.clasificar_heuristico("cuál es mi balance de hoy")
        assert result == INTENT_PROGRESO

    def test_detecta_peso(self, classifier):
        """'actualiza mi peso' → progreso."""
        result = classifier.clasificar_heuristico("registra mi peso de hoy")
        assert result == INTENT_PROGRESO

    # ─── Caso chat (fallback) ─────────────────────────────────────────────────

    def test_fallback_chat(self, classifier):
        """Mensaje sin palabras clave → chat."""
        result = classifier.clasificar_heuristico("hola, ¿cómo estás?")
        assert result == INTENT_CHAT

    def test_clasificar_async_heuristico(self, classifier):
        """clasificar() sin LLM retorna mismo resultado que heurístico."""
        result = asyncio.run(classifier.clasificar("quiero comer sano"))
        assert result == INTENT_NUTRICION
