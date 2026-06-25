"""
UBICACIÓN (tests/external/): usa Groq real, sin mock — depende de la API
externa, está excluida de la corrida por defecto (`pytest tests/` usa
-m "not external", ver tests/pytest.ini). Ejecutar a mano con:
    pytest tests/external/test_conversacion_natural.py -v
"""
import pytest
import re
from datetime import date
from app.services.asistente.asistente_service import AsistenteService
from app.models.client import Client
from app.models.historial import ProgresoCalorias
from app.models import MetaUsuario

@pytest.mark.integration
@pytest.mark.external
class TestConversacionNatural:
    """Batería de pruebas de conversación natural e integración para demo."""

    @pytest.fixture
    def setup_user(self, db, sample_client):
        class MockUser:
            email = sample_client.email
        return {'client': sample_client, 'user': MockUser(), 'db': db}

    @pytest.mark.asyncio
    async def test_caso1_chat_completo_nutricional(self, setup_user):
        """
        Caso 1 — Chat completo nutricional:
        Registrar comida → declarar restricción en chat → recomendar cena respetando la restricción.
        """
        client = setup_user['client']
        user = setup_user['user']
        db = setup_user['db']
        
        # Inicializar condiciones limpias
        client.medical_conditions = []
        db.commit()
        
        asistente = AsistenteService()
        historial = []
        
        # Turno 1: Hoy desayuné avena con leche
        resp1 = await asistente.consultar(
            mensaje="Hoy desayuné avena con leche",
            db=db,
            current_user=user,
            historial=historial,
        )
        assert resp1["intencion"] == "SUCCESS"
        historial.append({"role": "user", "content": "Hoy desayuné avena con leche"})
        historial.append({"role": "assistant", "content": resp1["respuesta_ia"]})
        
        # Turno 2: Pero soy intolerante a la lactosa
        resp2 = await asistente.consultar(
            mensaje="Pero soy intolerante a la lactosa",
            db=db,
            current_user=user,
            historial=historial,
        )
        historial.append({"role": "user", "content": "Pero soy intolerante a la lactosa"})
        historial.append({"role": "assistant", "content": resp2["respuesta_ia"]})
        
        # Simulamos que la app actualiza el perfil con la nueva condición declarada
        client.medical_conditions = ["Intolerancia a la Lactosa"]
        db.commit()
        
        # Turno 3: ¿Qué puedo cenar?
        resp3 = await asistente.consultar(
            mensaje="¿Qué puedo cenar?",
            db=db,
            current_user=user,
            historial=historial,
        )
        
        resp_text = resp3["respuesta_ia"].lower()
        # Verificar que recuerda la restricción y no recomienda lácteos
        assert "leche" not in resp_text
        assert "yogur" not in resp_text
        assert "queso" not in resp_text
        # Pero sigue recomendando cena (debería sugerir opciones sin lácteos)
        assert len(resp_text) > 10

    @pytest.mark.asyncio
    async def test_caso2_registro_correccion(self, setup_user):
        """
        Caso 2 — Registro + corrección:
        Registrar comida y corregir la cantidad en el siguiente turno.
        """
        client = setup_user['client']
        user = setup_user['user']
        db = setup_user['db']
        
        asistente = AsistenteService()
        historial = []
        
        # Turno 1: Comí dos huevos con arroz
        resp1 = await asistente.consultar(
            mensaje="Comí dos huevos con arroz",
            db=db,
            current_user=user,
            historial=historial,
        )
        assert resp1["intencion"] == "SUCCESS"
        historial.append({"role": "user", "content": "Comí dos huevos con arroz"})
        historial.append({"role": "assistant", "content": resp1["respuesta_ia"]})
        
        # Turno 2: ah no, eran tres huevos
        resp2 = await asistente.consultar(
            mensaje="ah no, eran tres huevos",
            db=db,
            current_user=user,
            historial=historial,
        )
        
        resp_text = resp2["respuesta_ia"].lower()
        # Verificar que se corrige correctamente
        assert "3" in resp_text or "tres" in resp_text
        assert "huevo" in resp_text

    @pytest.mark.asyncio
    async def test_caso3_lesion(self, setup_user):
        """
        Caso 3 — Lesión:
        Declarar dolor de rodilla → pedir rutina de pierna → evitar sentadillas u otros ejercicios lesivos.
        """
        client = setup_user['client']
        user = setup_user['user']
        db = setup_user['db']
        
        asistente = AsistenteService()
        historial = []
        
        # Turno 1: Me duele la rodilla
        resp1 = await asistente.consultar(
            mensaje="Me duele la rodilla",
            db=db,
            current_user=user,
            historial=historial,
        )
        historial.append({"role": "user", "content": "Me duele la rodilla"})
        historial.append({"role": "assistant", "content": resp1["respuesta_ia"]})
        
        # Turno 2: Dame rutina de pierna
        resp2 = await asistente.consultar(
            mensaje="Dame rutina de pierna",
            db=db,
            current_user=user,
            historial=historial,
        )
        
        resp_text = resp2["respuesta_ia"].lower()
        # Evitar ejercicios problemáticos (como sentadillas)
        assert "sentadilla" not in resp_text

    @pytest.mark.asyncio
    async def test_caso4_masa_muscular(self, setup_user):
        """
        Caso 4 — Masa muscular:
        Perfil ganar_leve → Exceso calórico → Informar de forma neutral sin tratar como error clínico.
        """
        client = setup_user['client']
        user = setup_user['user']
        db = setup_user['db']
        
        # Configurar perfil de ganancia
        client.goal = "ganar_leve"
        db.commit()
        
        # Asegurar MetaUsuario de ganancia
        meta = db.query(MetaUsuario).filter(MetaUsuario.client_id == client.id).first()
        if not meta:
            meta = MetaUsuario(
                client_id=client.id,
                genero="M", edad=25, peso_kg=70, talla_cm=175,
                nivel_actividad="Moderado", objetivo="ganar_leve",
                tmb=1700, get=2500, calorias_objetivo=2800,
                proteinas_g=150, carbohidratos_g=300, grasas_g=70
            )
            db.add(meta)
            db.commit()
            
        # Simular exceso de calorías consumidas hoy
        prog = db.query(ProgresoCalorias).filter(
            ProgresoCalorias.client_id == client.id,
            ProgresoCalorias.fecha == date.today()
        ).first()
        if not prog:
            prog = ProgresoCalorias(client_id=client.id, fecha=date.today())
            db.add(prog)
        prog.calorias_consumidas = 3100  # Excede la meta de 2800
        db.commit()
        
        asistente = AsistenteService()
        resp = await asistente.consultar(
            mensaje="Ya pasé mi meta de calorías, ¿como algo más?",
            db=db,
            current_user=user,
            historial=[]
        )
        
        resp_text = resp["respuesta_ia"].lower()
        # Debe informar contextualmente del objetivo sin tratarlo como error o prohibición de comer
        assert "ganar" in resp_text or "músculo" in resp_text or "masa" in resp_text or "superávit" in resp_text

    @pytest.mark.asyncio
    async def test_caso5_perfil_restrictivo_recomendacion(self, setup_user):
        """
        Caso 5 — Perfil restrictivo + recomendación:
        Vegano + Diabetes + Ganar Masa → Equilibrar proteína, veganismo y control glucémico.
        """
        client = setup_user['client']
        user = setup_user['user']
        db = setup_user['db']
        
        client.goal = "ganar masa"
        client.medical_conditions = ["Vegano", "Diabetes"]
        db.commit()
        
        meta = db.query(MetaUsuario).filter(MetaUsuario.client_id == client.id).first()
        if not meta:
            meta = MetaUsuario(
                client_id=client.id,
                genero="M", edad=25, peso_kg=70, talla_cm=175,
                nivel_actividad="Moderado", objetivo="ganar masa",
                tmb=1700, get=2500, calorias_objetivo=3000,
                proteinas_g=150, carbohidratos_g=300, grasas_g=70
            )
            db.add(meta)
            db.commit()
            
        asistente = AsistenteService()
        resp = await asistente.consultar(
            mensaje="Qué puedo comer después de entrenar",
            db=db,
            current_user=user,
            historial=[]
        )
        
        resp_text = resp["respuesta_ia"].lower()
        # 1. Vegano: no productos animales
        assert "pollo" not in resp_text
        assert "carne" not in resp_text
        assert "pescado" not in resp_text
        assert "huevo" not in resp_text
        assert "leche" not in resp_text
        # 2. Proteína + Control glucémico: sugiere tofu, quinua, legumbres
        assert any(w in resp_text for w in ["tofu", "quinua", "lenteja", "soya", "frijol", "legumbre", "garbanzo", "proteína"])
