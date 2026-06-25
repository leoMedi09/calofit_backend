"""
Pruebas de los 3 fixes confirmados en la auditoría de diagnóstico
(tests/integration/test_auditoria_registro_real.py):

1. Contenedor genérico ("Batido"/"Jugo"/"Sopa"...) + ingredientes sueltos
   -> se descarta el contenedor, se conservan los ingredientes.
2. Alimento alucinado ("umas") -> se rechaza, nunca se registran macros
   inventados. Bebidas reales de ~0 kcal ("café") se mantienen.
3. Palabra de momento del día ("Almuerzo") como único "alimento" -> 0
   registros, no se inventan macros para la palabra.

Groq mockeado a propósito (a diferencia del archivo de diagnóstico): aquí
se valida la LÓGICA determinista del fix con datos de entrada controlados
y reproducibles, no el comportamiento variable del LLM real.
"""
import json

import pytest
from unittest.mock import patch

from app.core.utils import get_peru_date
from app.services.llm_registro import registrar_comida_llm
from app.services.ia_service import ia_engine
from app.models.comida_registro import ComidaRegistro


def _get_mensaje(prompt: str) -> str:
    marker = 'mensaje: "'
    low = prompt.lower()
    idx = low.find(marker)
    if idx == -1:
        return low
    resto = prompt[idx + len(marker):]
    return resto.split('"', 1)[0].lower()


@pytest.fixture
def plan_hoy():
    return {"calorias_dia": 2616, "proteinas_g": 160, "carbohidratos_g": 240, "grasas_g": 60}


def _filas_de_hoy(db, client_id):
    return (
        db.query(ComidaRegistro)
        .filter(ComidaRegistro.client_id == client_id, ComidaRegistro.fecha == get_peru_date())
        .all()
    )


@pytest.mark.integration
class TestFixContenedorGenerico:
    """TEST 1 — 'Batido de avena, leche, dos plátanos, miel' no debe duplicar
    contenedor + ingredientes."""

    @pytest.fixture(autouse=True)
    def mock_groq(self):
        async def mock_llamar_groq(prompt, max_tokens=800, temp=0.7, model=None):
            prompt_lower = prompt.lower()
            if "extrae todos los alimentos" in prompt_lower or "responde solo con json" in prompt_lower:
                msg = _get_mensaje(prompt)
                if "batido" in msg or "avena" in msg:
                    # Reproduce EXACTAMENTE lo que devolvió Groq real en el
                    # diagnóstico: el contenedor Y los 4 ingredientes a la vez.
                    return json.dumps({"alimentos": [
                        {"nombre": "Batido de avena", "es_real": True, "cantidad": 1,
                         "porcion_g": 250, "kcal": 252, "prot_g": 8, "carb_g": 40, "grasa_g": 5},
                        {"nombre": "Leche", "es_real": True, "cantidad": 1,
                         "porcion_g": 200, "kcal": 112, "prot_g": 6, "carb_g": 10, "grasa_g": 5},
                        {"nombre": "Plátano", "es_real": True, "cantidad": 2,
                         "porcion_g": 120, "kcal": 104, "prot_g": 1, "carb_g": 27, "grasa_g": 0.3},
                        {"nombre": "Miel", "es_real": True, "cantidad": 1,
                         "porcion_g": 21, "kcal": 60, "prot_g": 0, "carb_g": 16, "grasa_g": 0},
                    ]})
                return json.dumps({"alimentos": []})
            return json.dumps({"alimentos": []})

        with patch.object(ia_engine, "_llamar_groq", new=mock_llamar_groq):
            yield

    @pytest.mark.asyncio
    async def test_batido_no_duplica_contenedor_e_ingredientes(self, db, sample_client, plan_hoy):
        resultado = await registrar_comida_llm(
            "Me hice un batido de avena, leche, dos plátanos, miel",
            sample_client, plan_hoy, db, ia_engine,
        )
        assert resultado["success"] is True
        nombres_norm = [n.lower() for n in resultado["alimentos"]]

        assert not any("batido" in n for n in nombres_norm), (
            f"El contenedor 'Batido' no debió quedar registrado: {resultado['alimentos']}"
        )
        assert any("leche" in n for n in nombres_norm)
        assert any("plátano" in n or "platano" in n for n in nombres_norm)
        assert any("miel" in n for n in nombres_norm)

        filas = _filas_de_hoy(db, sample_client.id)
        assert not any("batido" in f.nombre_alimento.lower() for f in filas)
        # Ninguna fila debe corresponder a las ~252 kcal que el LLM le asignó
        # al contenedor — esa cantidad no debe aparecer sumada en el total.
        assert all(abs(f.kcal - 252) > 1 for f in filas), (
            f"Parece que las kcal del contenedor (252) se filtraron igual: {[(f.nombre_alimento, f.kcal) for f in filas]}"
        )


@pytest.mark.integration
class TestFixAlimentoAlucinado:
    """TEST 2 — 'Comí tres umas y dos tazas de café': umas se rechaza, café
    se mantiene, nunca se inventan macros."""

    @pytest.fixture(autouse=True)
    def mock_groq(self):
        async def mock_llamar_groq(prompt, max_tokens=800, temp=0.7, model=None):
            prompt_lower = prompt.lower()
            if "extrae todos los alimentos" in prompt_lower or "responde solo con json" in prompt_lower:
                msg = _get_mensaje(prompt)
                if "umas" in msg or "café" in msg or "cafe" in msg:
                    # Reproduce EXACTAMENTE lo que devolvió Groq real:
                    # "umas" marcado es_real=true con macros en 0, café
                    # también con macros en 0 (bebida real ~0 kcal).
                    return json.dumps({"alimentos": [
                        {"nombre": "Tres umas", "es_real": True, "cantidad": 3,
                         "porcion_g": 50, "kcal": 0, "prot_g": 0, "carb_g": 0, "grasa_g": 0},
                        {"nombre": "Café", "es_real": True, "cantidad": 2,
                         "porcion_g": 200, "kcal": 0, "prot_g": 0, "carb_g": 0, "grasa_g": 0},
                    ]})
                return json.dumps({"alimentos": []})
            return json.dumps({"alimentos": []})

        with patch.object(ia_engine, "_llamar_groq", new=mock_llamar_groq):
            yield

    @pytest.mark.asyncio
    async def test_umas_rechazado_cafe_se_mantiene(self, db, sample_client, plan_hoy):
        resultado = await registrar_comida_llm(
            "Comí tres umas y dos tazas de café", sample_client, plan_hoy, db, ia_engine,
        )
        nombres_norm = [n.lower() for n in resultado["alimentos"]]

        assert not any("uma" in n for n in nombres_norm), (
            f"'umas' no debió quedar registrado: {resultado['alimentos']}"
        )
        assert any("café" in n or "cafe" in n for n in nombres_norm), (
            f"'café' debió mantenerse (bebida real, aunque sea ~0 kcal): {resultado['alimentos']}"
        )
        assert resultado["success"] is True

    @pytest.mark.asyncio
    async def test_umas_sola_sin_nada_mas_pide_aclaracion(self, db, sample_client, plan_hoy):
        """Si el ÚNICO ítem extraído es la alucinación, no debe quedar
        ningún registro — debe degradar al flujo de 'no identificado'."""
        async def mock_solo_umas(prompt, max_tokens=800, temp=0.7, model=None):
            if "extrae todos los alimentos" in prompt.lower():
                return json.dumps({"alimentos": [
                    {"nombre": "Umas", "es_real": True, "cantidad": 1,
                     "porcion_g": 50, "kcal": 0, "prot_g": 0, "carb_g": 0, "grasa_g": 0},
                ]})
            return json.dumps({"alimentos": []})

        with patch.object(ia_engine, "_llamar_groq", new=mock_solo_umas):
            resultado = await registrar_comida_llm(
                "Comí una uma", sample_client, plan_hoy, db, ia_engine,
            )
        assert resultado["success"] is False
        assert resultado["tipo_detectado"] == "no_identificado"

        filas = _filas_de_hoy(db, sample_client.id)
        assert len(filas) == 0


@pytest.mark.integration
class TestFixPalabraMomentoDia:
    """TEST 3 — 'Registré mi almuerzo' no debe registrar 'Almuerzo' como
    alimento, ni con macros inventados."""

    @pytest.fixture(autouse=True)
    def mock_groq(self):
        async def mock_llamar_groq(prompt, max_tokens=800, temp=0.7, model=None):
            if "extrae todos los alimentos" in prompt.lower():
                # Reproduce EXACTAMENTE lo que devolvió Groq real: macros NO
                # nulos (790/30/100/30) — el fix de momento-del-día debe
                # bloquearlo por el NOMBRE, sin depender de que los macros
                # sean sospechosos.
                return json.dumps({"alimentos": [
                    {"nombre": "Almuerzo", "es_real": True, "cantidad": 1,
                     "porcion_g": 400, "kcal": 790, "prot_g": 30, "carb_g": 100, "grasa_g": 30},
                ]})
            return json.dumps({"alimentos": []})

        with patch.object(ia_engine, "_llamar_groq", new=mock_llamar_groq):
            yield

    @pytest.mark.asyncio
    async def test_almuerzo_no_se_registra_como_alimento(self, db, sample_client, plan_hoy):
        resultado = await registrar_comida_llm(
            "Registré mi almuerzo", sample_client, plan_hoy, db, ia_engine,
        )
        assert resultado["success"] is False
        assert resultado["tipo_detectado"] == "no_identificado"

        filas = _filas_de_hoy(db, sample_client.id)
        assert len(filas) == 0, f"No debió registrarse nada, hay: {[f.nombre_alimento for f in filas]}"


@pytest.mark.integration
class TestNoRegresionCantidad:
    """TEST 5 — confirma que los fixes nuevos no rompen lo ya corregido:
    'dos mandarinas' (cantidad).

    La regresión de 'Arroz con lentejas' (plato compuesto) se consolidó en
    tests/integration/test_registro_plato_compuesto.py — es la misma
    aserción exacta (mismo mensaje, mismo mock, mismo resultado esperado)
    que estaba duplicada aquí. Esa otra prueba es la fuente única de verdad
    para ese caso: además de verificar `resultado["alimentos"]`, también
    confirma el contenido real de la fila en `comida_registros`."""

    @pytest.fixture(autouse=True)
    def mock_groq(self):
        async def mock_llamar_groq(prompt, max_tokens=800, temp=0.7, model=None):
            prompt_lower = prompt.lower()
            if "extrae todos los alimentos" in prompt_lower:
                msg = _get_mensaje(prompt)
                if "mandarina" in msg:
                    return json.dumps({"alimentos": [
                        {"nombre": "Mandarina", "es_real": True, "cantidad": 2,
                         "porcion_g": 80, "kcal": 52, "prot_g": 0.7, "carb_g": 12, "grasa_g": 0.2},
                    ]})
                return json.dumps({"alimentos": []})
            return json.dumps({"alimentos": []})

        with patch.object(ia_engine, "_llamar_groq", new=mock_llamar_groq):
            yield

    @pytest.mark.asyncio
    async def test_dos_mandarinas_mantiene_multiplicador(self, db, sample_client, plan_hoy):
        resultado = await registrar_comida_llm(
            "comí dos mandarinas", sample_client, plan_hoy, db, ia_engine,
        )
        assert resultado["success"] is True
        assert any("×2" in n for n in resultado["alimentos"]), (
            f"Regresión: 'dos mandarinas' perdió el multiplicador x2: {resultado['alimentos']}"
        )
        filas = _filas_de_hoy(db, sample_client.id)
        assert len(filas) == 2
