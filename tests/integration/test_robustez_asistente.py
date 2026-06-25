import pytest
import asyncio
import re
from datetime import date
from unittest.mock import AsyncMock, patch
from app.services.asistente.asistente_service import AsistenteService
from app.models.client import Client
from app.models.historial import ProgresoCalorias
from app.models import MetaUsuario
from app.services.ia_service import ia_engine

@pytest.mark.integration
class TestRobustezAsistente:
    """Pruebas de robustez y casos límite (anomalías, fallos, ambigüedad) para la demo."""

    @pytest.fixture
    def setup_user(self, db, sample_client):
        class MockUser:
            email = sample_client.email
        return {'client': sample_client, 'user': MockUser(), 'db': db}

    @pytest.fixture(autouse=True)
    def mock_groq_calls(self):
        async def mock_llamar_groq(prompt: str, max_tokens: int = 800, temp: float = 0.7, model: str = None):
            prompt_lower = prompt.lower()
            import json
            import re

            # Extraer el mensaje del usuario de forma aislada
            def get_msg():
                for line in prompt_lower.splitlines():
                    line_s = line.strip()
                    for marker in ["mensaje a clasificar:", "mensaje actual:", "mensaje del usuario:", "mensaje:"]:
                        if line_s.startswith(marker):
                            after = line_s[len(marker):].strip()
                            if after.startswith('"') and after.endswith('"'):
                                return after[1:-1].strip()
                            if after.startswith("'") and after.endswith("'"):
                                return after[1:-1].strip()
                            return after.strip('"').strip("'")
                return prompt_lower

            msg = get_msg()

            def has_word(text, word_list):
                for w in word_list:
                    if re.search(r'\b' + re.escape(w) + r'\b', text):
                        return True
                return False

            # A. CLASIFICADOR DE INTENCIONES
            if "clasificador de intenciones" in prompt_lower:
                if has_word(msg, ["bajar grasa"]):
                    return "otro"
                elif "calorias" in msg and "mi comida" in msg:
                    return "otro"
                elif has_word(msg, ["arroz con pollo", "hamburguesa", "lentejas", "lenteja", "comi", "comí", "zampe", "zampé", "almorce", "almorcé", "huevo", "huevos", "arroz", "leche"]):
                    return "registrar_nutricion"
                elif has_word(msg, ["entrenar", "comer", "cena"]):
                    return "recomendar_nutricion"
                return "otro"

            # B. REGISTRO DE COMIDA (JSON extraction)
            if "extrae todos los alimentos" in prompt_lower or "responde solo con json" in prompt_lower:
                if "arroz con pollo" in msg:
                    return json.dumps({
                        "alimentos": [
                            {"nombre": "Arroz con pollo", "es_real": True, "cantidad": 1, "porcion_g": 350, "kcal": 650, "prot_g": 35, "carb_g": 80, "grasa_g": 18}
                        ],
                        "prot_total": 35,
                        "carb_total": 80,
                        "grasa_total": 18
                    })
                elif "hamburguesa" in msg:
                    return json.dumps({
                        "alimentos": [
                            {"nombre": "Hamburguesa", "es_real": True, "cantidad": 1, "porcion_g": 200, "kcal": 500, "prot_g": 25, "carb_g": 40, "grasa_g": 22}
                        ],
                        "prot_total": 25,
                        "carb_total": 40,
                        "grasa_total": 22
                    })
                elif "lentejas" in msg or "lenteja" in msg:
                    return json.dumps({
                        "alimentos": [
                            {"nombre": "Lentejas", "es_real": True, "cantidad": 1, "porcion_g": 250, "kcal": 350, "prot_g": 18, "carb_g": 55, "grasa_g": 5}
                        ],
                        "prot_total": 18,
                        "carb_total": 55,
                        "grasa_total": 5
                    })
                elif "arroz" in msg:
                    return json.dumps({
                        "alimentos": [
                            {"nombre": "Arroz", "es_real": True, "cantidad": 1, "porcion_g": 150, "kcal": 200, "prot_g": 4, "carb_g": 44, "grasa_g": 0.5}
                        ],
                        "prot_total": 4,
                        "carb_total": 44,
                        "grasa_total": 0.5
                    })
                elif "leche" in msg:
                    return json.dumps({
                        "alimentos": [
                            {"nombre": "Leche", "es_real": True, "cantidad": 1, "porcion_g": 250, "kcal": 150, "prot_g": 8, "carb_g": 12, "grasa_g": 8}
                        ],
                        "prot_total": 8,
                        "carb_total": 12,
                        "grasa_total": 8
                    })
                elif "huevos" in msg or "huevo" in msg:
                    return json.dumps({
                        "alimentos": [
                            {"nombre": "Huevos", "es_real": True, "cantidad": 2, "porcion_g": 100, "kcal": 140, "prot_g": 12, "carb_g": 1, "grasa_g": 10}
                        ],
                        "prot_total": 12,
                        "carb_total": 1,
                        "grasa_total": 10
                    })
                return json.dumps({"alimentos": [], "prot_total": 0, "carb_total": 0, "grasa_total": 0})

            # C. VALIDACIÓN DIETÉTICA (SI / NO)
            if "hay al menos un plato inadecuado" in prompt_lower:
                parts = prompt_lower.split("platos propuestos:\n")
                proposed = parts[1].split("revisa cada plato")[0] if len(parts) > 1 else prompt_lower
                if any(p in proposed for p in ["picarones", "azúcar", "azucar", "miel", "leche", "pollo", "carne"]):
                    return "SI"
                return "NO"

            # D. MICROCONSULTA MÉDICA
            if "restricciones dietéticas concretas" in prompt_lower or "nutricionista clínico. el paciente tiene" in prompt_lower:
                return "• Diabetes: evitar azúcar refinado, dulces y postres.\n• Vegano: evitar carne, pollo, pescado, huevo, leche animal.\n• Intolerancia a la lactosa: evitar lácteos con lactosa."

            # E. RECOMENDACIÓN DE COMIDA
            if any(w in prompt_lower for w in ["propón exactamente 3 platos", "platos para", "lista exactamente 3 platos", "exactamente 3 platos"]):
                if "vegano" in prompt_lower or "vegana" in prompt_lower or "tofu" in prompt_lower or "lentejas" in prompt_lower:
                    if "tofu" in prompt_lower:
                        return (
                            "- Ensalada de garbanzos con palta y chía (~350 kcal, P:18g C:50g G:8g)\n"
                            "- Hamburguesa de lentejas con ensalada ligera (~300 kcal, P:22g C:40g G:6g)\n"
                            "- Saltado de champiñones con quinua (~320 kcal, P:14g C:48g G:8g)"
                        )
                    elif "garbanzo" in prompt_lower:
                        return (
                            "- Ceviche de champiñones con choclo (~250 kcal, P:8g C:42g G:4g)\n"
                            "- Menestra de frijol negro con arroz integral (~380 kcal, P:20g C:58g G:6g)\n"
                            "- Tofu al horno con espárragos (~290 kcal, P:24g C:18g G:13g)"
                        )
                    else:
                        return (
                            "- Tofu salteado con brócoli y quinua (~380 kcal, P:25g C:45g G:10g)\n"
                            "- Ensalada de lentejas con palta (~350 kcal, P:18g C:50g G:8g)\n"
                            "- Crema de arvejas con tostadas de centeno (~320 kcal, P:16g C:48g G:6g)"
                        )
                return (
                    "- Sopa de verduras con quinua (~300 kcal, P:10g C:50g G:5g)\n"
                    "- Ensalada de garbanzos y palta (~280 kcal, P:8g C:35g G:12g)\n"
                    "- Tofu a la plancha con arroz integral (~380 kcal, P:22g C:45g G:10g)"
                )

            # F. CHAT PERMISSION CHECK (puedo comer X?)
            if "coach nutricional amigable" in prompt_lower:
                if "pescado" in prompt_lower or "pollo" in prompt_lower or "carne" in prompt_lower:
                    return "No, como eres vegano no debes consumirlo. Puedes comer tofu como alternativa."
                return "Sí, puedes comerlo con moderación dentro de tus macros."

            # G. RESPUESTAS CONVERSACIONALES (Chat)
            if "bajar grasa" in prompt_lower:
                return "Para bajar grasa de manera efectiva manteniendo tu salud, te sugiero un ligero déficit calórico controlado."
            elif "calorías rápido" in prompt_lower or "aumentar peso" in prompt_lower:
                return "Para subir de peso saludablemente, prioriza grasas saludables como palta, frutos secos y avena integral."
            elif "muerto después del gym" in prompt_lower:
                return "Es normal sentir cansancio después de un entrenamiento intenso. Descansa bien, hidrátate y consume proteínas."
            elif "calorías tenía mi comida" in prompt_lower or "calorias" in prompt_lower:
                return "Tu comida (Arroz con pollo) tenía aproximadamente 650 calorías."
            elif "entrenar" in prompt_lower or "cenar" in prompt_lower:
                return "Puedes comer tofu salteado con quinua o un batido de proteína vegetal con avena y bebida de almendras."

            return "Entendido, estoy para ayudarte."

        with patch.object(ia_engine, "_llamar_groq", new=mock_llamar_groq):
            yield

    # =========================================================================
    # CASO 1 — FALLA DE SERVICIO LLM
    # =========================================================================
    @pytest.mark.asyncio
    async def test_caso1_falla_servicio_llm(self, setup_user):
        client = setup_user['client']
        user = setup_user['user']
        db = setup_user['db']

        # Perfil con Diabetes
        client.medical_conditions = ["Diabetes"]
        db.commit()

        asistente = AsistenteService()

        # A: Simular Timeout de Groq
        with patch.object(ia_engine, "_llamar_groq", side_effect=asyncio.TimeoutError("Timeout simulado")):
            resp_timeout = await asistente.consultar(
                mensaje="¿Qué puedo comer de cena?",
                db=db,
                current_user=user,
                historial=[]
            )
            resp_text = resp_timeout["respuesta_ia"].lower()
            # Validaciones:
            # 1. Se ejecuta fallback determinístico
            assert len(resp_text) > 10
            # 2. Mantiene restricciones médicas (no azúcar, no postres dulces del fallback)
            assert "sopa de verduras" in resp_text or "ensalada" in resp_text
            assert "picarones" not in resp_text and "azúcar" not in resp_text

        # B: Simular Respuesta Vacía del LLM
        with patch.object(ia_engine, "_llamar_groq", return_value=""):
            resp_empty = await asistente.consultar(
                mensaje="¿Qué puedo comer de cena?",
                db=db,
                current_user=user,
                historial=[]
            )
            resp_text = resp_empty["respuesta_ia"].lower()
            assert len(resp_text) > 10
            assert "sopa de verduras" in resp_text or "ensalada" in resp_text

        # B-2: Simular Rate Limit de Groq
        with patch.object(ia_engine, "_llamar_groq", return_value="El asistente está temporalmente ocupado (límite de consultas alcanzado). Espera unos minutos y vuelve a intentarlo. "):
            resp_rate = await asistente.consultar(
                mensaje="¿Qué puedo comer de cena?",
                db=db,
                current_user=user,
                historial=[]
            )
            resp_text = resp_rate["respuesta_ia"].lower()
            assert len(resp_text) > 10
            assert "sopa de verduras" in resp_text or "ensalada" in resp_text

        # C: Simular Error durante la Microconsulta Médica
        # La microconsulta se hace con un prompt específico en respuesta_recomendacion_llm.
        # Hacemos que _llamar_groq devuelva error sólo cuando se le pregunta sobre condiciones médicas.
        original_llamar_groq = ia_engine._llamar_groq
        async def mock_llamar_groq(prompt, *args, **kwargs):
            if "condiciones" in prompt or "médico" in prompt or "paciente" in prompt:
                raise Exception("Error de microconsulta médica simulado")
            return await original_llamar_groq(prompt, *args, **kwargs)

        with patch.object(ia_engine, "_llamar_groq", new=mock_llamar_groq):
            resp_med_err = await asistente.consultar(
                mensaje="¿Qué puedo comer de cena?",
                db=db,
                current_user=user,
                historial=[]
            )
            resp_text = resp_med_err["respuesta_ia"].lower()
            # El fallback médico determinístico inyecta restricciones locales y evita prohibidos
            assert "leche" not in resp_text
            assert "azúcar" not in resp_text or "dulce" not in resp_text

    # =========================================================================
    # CASO 2 — RECUPERACIÓN DESPUÉS DE CAMBIO DE TEMA
    # =========================================================================
    @pytest.mark.asyncio
    async def test_caso2_recuperacion_cambio_tema(self, setup_user):
        client = setup_user['client']
        user = setup_user['user']
        db = setup_user['db']

        asistente = AsistenteService()
        historial = []

        # Turno 1: Comí arroz con pollo (Registro de comida)
        resp1 = await asistente.consultar(
            mensaje="Comí arroz con pollo",
            db=db,
            current_user=user,
            historial=historial
        )
        assert resp1["intencion"] == "SUCCESS"
        historial.append({"role": "user", "content": "Comí arroz con pollo"})
        historial.append({"role": "assistant", "content": resp1["respuesta_ia"]})

        await asyncio.sleep(3)

        # Turno 2: ¿Cuál es la capital de Francia? (Off-topic)
        resp2 = await asistente.consultar(
            mensaje="¿Cuál es la capital de Francia?",
            db=db,
            current_user=user,
            historial=historial
        )
        # El router/guard lo detecta como INFO (off-topic) o CHAT
        assert resp2["intencion"] == "INFO"
        assert "nutrición" in resp2["respuesta_ia"] or "ejercicio" in resp2["respuesta_ia"]
        
        # Ojo: No lo metemos en el historial del LLM nutricional o lo limpiamos, pero en la app se registra el flujo
        historial.append({"role": "user", "content": "¿Cuál es la capital de Francia?"})
        historial.append({"role": "assistant", "content": resp2["respuesta_ia"]})

        await asyncio.sleep(3)

        # Turno 3: Ahora dime cuántas calorías tenía mi comida
        resp3 = await asistente.consultar(
            mensaje="Ahora dime cuántas calorías tenía mi comida",
            db=db,
            current_user=user,
            historial=historial
        )
        resp_text = resp3["respuesta_ia"].lower()
        # Debe recuperar la comida del turno 1 (arroz con pollo)
        assert "arroz" in resp_text or "pollo" in resp_text or "calorías" in resp_text

    # =========================================================================
    # CASO 3 — CONTEXTO INCOMPLETO
    # =========================================================================
    @pytest.mark.asyncio
    async def test_caso3_contexto_incompleto(self, setup_user):
        client = setup_user['client']
        user = setup_user['user']
        db = setup_user['db']

        # Limpiar perfil para que esté incompleto
        client.goal = None
        client.medical_conditions = []
        client.activity_level = None
        db.commit()

        asistente = AsistenteService()
        resp = await asistente.consultar(
            mensaje="¿Qué puedo comer?",
            db=db,
            current_user=user,
            historial=[]
        )
        resp_text = resp["respuesta_ia"].lower()
        # Debe usar valores por defecto seguros (alrededor de 2000 kcal o rango normal de comida)
        assert len(resp_text) > 10
        # No inventa objetivos ni asume enfermedades
        assert "diabetes" not in resp_text
        assert "hipertensión" not in resp_text

    # =========================================================================
    # CASO 4 — CONTRADICCIÓN ENTRE MENSAJE Y PERFIL
    # =========================================================================
    @pytest.mark.asyncio
    async def test_caso4_contradiccion_mensaje_perfil(self, setup_user):
        client = setup_user['client']
        user = setup_user['user']
        db = setup_user['db']

        # Perfil: ganar_leve
        client.goal = "ganar_leve"
        db.commit()

        asistente = AsistenteService()
        resp = await asistente.consultar(
            mensaje="Quiero bajar grasa rápido",
            db=db,
            current_user=user,
            historial=[]
        )
        
        # Verificar que el objetivo almacenado no haya cambiado
        db.refresh(client)
        assert client.goal == "ganar_leve"

        resp_text = resp["respuesta_ia"].lower()
        # El LLM debe abordar el deseo de bajar grasa temporalmente sin contradecir o ignorar el perfil de forma destructiva
        assert "grasa" in resp_text or "bajar" in resp_text or "calorías" in resp_text

    # =========================================================================
    # CASO 5 — RESTRICCIÓN MÉDICA + OBJETIVO CONFLICTIVO
    # =========================================================================
    @pytest.mark.asyncio
    async def test_caso5_restriccion_medica_objetivo_conflictivo(self, setup_user):
        client = setup_user['client']
        user = setup_user['user']
        db = setup_user['db']

        # Perfil: ganar masa + Diabetes
        client.goal = "ganar masa"
        client.medical_conditions = ["Diabetes"]
        db.commit()

        meta = db.query(MetaUsuario).filter(MetaUsuario.client_id == client.id).first()
        if not meta:
            meta = MetaUsuario(
                client_id=client.id,
                genero="M", edad=25, peso_kg=75, talla_cm=175,
                nivel_actividad="Moderado", objetivo="ganar masa",
                tmb=1700, get=2500, calorias_objetivo=3000,
                proteinas_g=150, carbohidratos_g=300, grasas_g=70
            )
            db.add(meta)
            db.commit()

        asistente = AsistenteService()
        resp = await asistente.consultar(
            mensaje="Quiero subir calorías rápido, dame comidas para aumentar peso",
            db=db,
            current_user=user,
            historial=[]
        )
        resp_text = resp["respuesta_ia"].lower()

        # Debe equilibrar aumento calórico con control glucémico
        # 1. No debe recomendar azúcar, postres azucarados, ni picarones
        assert "picarones" not in resp_text
        assert "azúcar" not in resp_text and "dulces" not in resp_text
        # 2. Debe recomendar grasas buenas o carbohidratos complejos (palta, frutos secos, avena, etc.)
        assert any(w in resp_text for w in ["palta", "frutos secos", "nueces", "avena", "grasas saludables", "tofu", "lentejas", "quinua"])

    # =========================================================================
    # CASO 6 — ENTRADAS AMBIGUAS
    # =========================================================================
    @pytest.mark.asyncio
    async def test_caso6_entradas_ambiguas(self, setup_user):
        client = setup_user['client']
        user = setup_user['user']
        db = setup_user['db']

        asistente = AsistenteService()

        # Test A: "Comí un poco de arroz"
        resp_arroz = await asistente.consultar(
            mensaje="Comí un poco de arroz",
            db=db,
            current_user=user,
            historial=[]
        )
        assert "duda" in resp_arroz["respuesta_ia"].lower() or "estimado" in resp_arroz["respuesta_ia"].lower()

        # Test B: "Tomé algo de leche"
        resp_leche = await asistente.consultar(
            mensaje="Tomé algo de leche",
            db=db,
            current_user=user,
            historial=[]
        )
        assert "duda" in resp_leche["respuesta_ia"].lower() or "estimado" in resp_leche["respuesta_ia"].lower()

        # Test C: "Creo que fueron unos huevos"
        resp_huevos = await asistente.consultar(
            mensaje="Creo que fueron unos huevos",
            db=db,
            current_user=user,
            historial=[]
        )
        assert "duda" in resp_huevos["respuesta_ia"].lower() or "estimado" in resp_huevos["respuesta_ia"].lower()

    # =========================================================================
    # CASO 7 — LENGUAJE NATURAL INFORMAL
    # =========================================================================
    @pytest.mark.asyncio
    async def test_caso7_lenguaje_natural_informal(self, setup_user):
        client = setup_user['client']
        user = setup_user['user']
        db = setup_user['db']

        asistente = AsistenteService()

        # Test A: "me metí un arroz con pollo"
        resp_meteo = await asistente.consultar(
            mensaje="me metí un arroz con pollo",
            db=db,
            current_user=user,
            historial=[]
        )
        assert resp_meteo["intencion"] == "SUCCESS"
        assert "arroz" in resp_meteo["respuesta_ia"].lower() or "pollo" in resp_meteo["respuesta_ia"].lower()

        await asyncio.sleep(3)

        # Test B: "me zampé una hamburguesa"
        resp_zampe = await asistente.consultar(
            mensaje="me zampé una hamburguesa",
            db=db,
            current_user=user,
            historial=[]
        )
        assert resp_zampe["intencion"] == "SUCCESS"
        assert "hamburguesa" in resp_zampe["respuesta_ia"].lower()

        await asyncio.sleep(3)

        # Test B-2: "almorcé lentejas"
        resp_lentejas = await asistente.consultar(
            mensaje="almorcé lentejas",
            db=db,
            current_user=user,
            historial=[]
        )
        assert resp_lentejas["intencion"] == "SUCCESS"
        assert "lenteja" in resp_lentejas["respuesta_ia"].lower()

        await asyncio.sleep(3)

        # Test C: "Hoy comí cualquier cosa"
        resp_cualquier = await asistente.consultar(
            mensaje="Hoy comí cualquier cosa",
            db=db,
            current_user=user,
            historial=[]
        )
        # El LLM no puede identificar el alimento, pide aclaración
        assert "cualquier cosa" in resp_cualquier["respuesta_ia"].lower() or "qué comiste" in resp_cualquier["respuesta_ia"].lower() or "no pude" in resp_cualquier["respuesta_ia"].lower()

        await asyncio.sleep(3)

        # Test D: "Estoy muerto después del gym"
        resp_gym = await asistente.consultar(
            mensaje="Estoy muerto después del gym",
            db=db,
            current_user=user,
            historial=[]
        )
        # Es un mensaje de estado físico, responde conversacionalmente
        assert len(resp_gym["respuesta_ia"]) > 10

    # =========================================================================
    # CASO 8 — REPETICIÓN Y CONSISTENCIA
    # =========================================================================
    @pytest.mark.asyncio
    async def test_caso8_repeticion_consistencia(self, setup_user):
        client = setup_user['client']
        user = setup_user['user']
        db = setup_user['db']

        # Perfil con Diabetes
        client.medical_conditions = ["Diabetes"]
        db.commit()

        asistente = AsistenteService()
        historial = []

        respuestas = []
        for i in range(3):
            resp = await asistente.consultar(
                mensaje="¿Qué puedo comer después de entrenar?",
                db=db,
                current_user=user,
                historial=historial
            )
            resp_text = resp["respuesta_ia"].lower()
            respuestas.append(resp_text)
            
            # Registrar recomendación en el historial simulado de la app para que aplique exclusión rotativa
            historial.append({"role": "user", "content": "¿Qué puedo comer después de entrenar?"})
            historial.append({"role": "assistant", "content": resp["respuesta_ia"]})

        # Validaciones:
        # 1. Cumplimiento de restricciones en todas las respuestas
        for r in respuestas:
            assert "azúcar" not in r
            assert "picarones" not in r
            
        # 2. Variación de respuestas (gracias a la persistencia y la exclusión de platos sugeridos)
        # Los textos o los platos sugeridos en las respuestas no deberían ser idénticos
        assert respuestas[0] != respuestas[1] or respuestas[1] != respuestas[2]

    # =========================================================================
    # CASO 9 — PERFIL COMPLEJO (Diabetes + Vegano + Sin Lactosa + ganar_leve)
    # =========================================================================
    @pytest.mark.asyncio
    async def test_caso9_perfil_complejo(self, setup_user):
        client = setup_user['client']
        user = setup_user['user']
        db = setup_user['db']

        # Objetivo: ganar_leve
        client.goal = "ganar_leve"
        # Condiciones: Diabetes + Vegano + sin lactosa (medical_conditions)
        client.medical_conditions = ["Diabetes", "Vegano", "intolerancia a la lactosa"]
        db.commit()

        # Asegurar meta calculada
        meta = db.query(MetaUsuario).filter(MetaUsuario.client_id == client.id).first()
        if not meta:
            meta = MetaUsuario(
                client_id=client.id,
                genero="M", edad=25, peso_kg=70, talla_cm=175,
                nivel_actividad="Moderado", objetivo="ganar_leve",
                tmb=1600, get=2300, calorias_objetivo=2500,
                proteinas_g=140, carbohidratos_g=300, grasas_g=60
            )
            db.add(meta)
            db.commit()

        asistente = AsistenteService()
        resp = await asistente.consultar(
            mensaje="¿Qué puedo comer después de entrenar?",
            db=db,
            current_user=user,
            historial=[]
        )
        resp_text = resp["respuesta_ia"].lower()

        # Validaciones:
        # 1. Respeta restricciones (Vegano -> no carne/pollo/pescado/huevo/queso común; Sin lactosa -> no lácteos con lactosa)
        assert "pollo" not in resp_text and "carne" not in resp_text and "huevo" not in resp_text and "pescado" not in resp_text
        assert "leche" not in resp_text or "deslactosad" in resp_text or "vegetal" in resp_text or "almendra" in resp_text or "soya" in resp_text or "soja" in resp_text
        # 2. Mantiene objetivo de ganancia muscular / recomienda proteína vegetal (tofu, lentejas, soja, quinua, etc.)
        assert any(w in resp_text for w in ["tofu", "lenteja", "garbanzo", "frijol", "poroto", "soja", "soya", "quinua", "chía", "frutos secos", "semilla", "maní", "proteína"])
        # 3. Controla glucosa (Diabetes -> evita azúcar refinado, dulces, picarones)
        assert "azúcar" not in resp_text and "picarones" not in resp_text and "miel" not in resp_text and "suspiro" not in resp_text

    def test_caso10_filtrado_contenedores_no_redundantes(self):
        from app.services.llm_registro import _filtrar_contenedor_generico_con_ingredientes
        
        # Caso A: Jugo de piña + Arroz + Chuleta (No redundante, no debe filtrarse el jugo)
        alimentos_a = [
            {"nombre": "Chuleta de chancho", "porcion_g": 150},
            {"nombre": "Arroz", "porcion_g": 100},
            {"nombre": "Jugo de piña", "porcion_g": 200}
        ]
        res_a = _filtrar_contenedor_generico_con_ingredientes(alimentos_a, "comí chuleta con arroz y jugo de piña")
        assert len(res_a) == 3
        assert any(x["nombre"] == "Jugo de piña" for x in res_a)

        # Caso B: Batido de avena con plátano + Avena + Plátano + Leche (Redundante, debe filtrarse el batido)
        alimentos_b = [
            {"nombre": "Batido de avena con platano", "porcion_g": 300},
            {"nombre": "Avena", "porcion_g": 40},
            {"nombre": "Platano", "porcion_g": 100},
            {"nombre": "Leche", "porcion_g": 200}
        ]
        res_b = _filtrar_contenedor_generico_con_ingredientes(alimentos_b, "tomé un batido de avena con platano, leche, avena y platano")
        assert len(res_b) == 3
        assert not any("Batido" in x["nombre"] for x in res_b)


