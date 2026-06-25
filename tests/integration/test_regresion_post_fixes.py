"""
Auditoría de regresión post-fixes — consistencia matemática y aislamiento de
caché. Groq mockeado: lo que se valida aquí es aritmética exacta y estado
determinista, no creatividad del LLM.

Los Casos A-G (filtros vs. registros normales, sin mock de Groq) se movieron
a tests/external/test_filtros_no_rompen_registros_normales.py — dependen de
la API real y están excluidos de la corrida por defecto.
"""
import json

import pytest
from unittest.mock import patch

from app.core.utils import get_peru_date
from app.services.llm_registro import registrar_comida_llm
from app.services.ia_service import ia_engine
from app.models.comida_registro import ComidaRegistro
from app.models.historial import ProgresoCalorias


def _get_mensaje(prompt: str) -> str:
    marker = 'mensaje: "'
    low = prompt.lower()
    idx = low.find(marker)
    if idx == -1:
        return low
    resto = prompt[idx + len(marker):]
    return resto.split('"', 1)[0].lower()


def _filas_de_hoy(db, client_id):
    return (
        db.query(ComidaRegistro)
        .filter(ComidaRegistro.client_id == client_id, ComidaRegistro.fecha == get_peru_date())
        .all()
    )


def _progreso_de_hoy(db, client_id):
    return (
        db.query(ProgresoCalorias)
        .filter(ProgresoCalorias.client_id == client_id, ProgresoCalorias.fecha == get_peru_date())
        .first()
    )


@pytest.fixture
def plan_hoy():
    return {"calorias_dia": 2616, "proteinas_g": 160, "carbohidratos_g": 240, "grasas_g": 60}


# ════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Consistencia matemática (Groq mockeado: necesita números
# controlados, no creatividad del LLM).
# ════════════════════════════════════════════════════════════════════════
@pytest.mark.integration
class TestConsistenciaMatematica:

    @pytest.mark.asyncio
    async def test_progreso_calorias_igual_a_suma_de_comida_registros(self, db, sample_client, plan_hoy):
        async def mock_groq(prompt, max_tokens=800, temp=0.7, model=None):
            if "extrae todos los alimentos" in prompt.lower():
                return json.dumps({"alimentos": [
                    {"nombre": "Pollo a la plancha", "es_real": True, "cantidad": 1,
                     "porcion_g": 150, "kcal": 250, "prot_g": 40, "carb_g": 0, "grasa_g": 9},
                    {"nombre": "Arroz blanco", "es_real": True, "cantidad": 1,
                     "porcion_g": 150, "kcal": 195, "prot_g": 4, "carb_g": 42, "grasa_g": 0.5},
                ]})
            return json.dumps({"alimentos": []})

        with patch.object(ia_engine, "_llamar_groq", new=mock_groq):
            resultado = await registrar_comida_llm(
                "Comí pollo a la plancha con arroz blanco", sample_client, plan_hoy, db, ia_engine,
            )
        assert resultado["success"] is True

        filas = _filas_de_hoy(db, sample_client.id)
        prog = _progreso_de_hoy(db, sample_client.id)
        suma_filas = sum(f.kcal for f in filas)
        assert abs(prog.calorias_consumidas - suma_filas) < 1.5, (
            f"progreso_calorias ({prog.calorias_consumidas}) != suma de comida_registros "
            f"({suma_filas}) — filas: {[(f.nombre_alimento, f.kcal) for f in filas]}"
        )

    @pytest.mark.asyncio
    async def test_cantidad_multiplica_macros_correctamente(self, db, sample_client, plan_hoy):
        async def mock_groq(prompt, max_tokens=800, temp=0.7, model=None):
            if "extrae todos los alimentos" in prompt.lower():
                # 3 huevos, macros POR UNIDAD (convención ya documentada en
                # el código: prot_g/carb_g/grasa_g son por unidad cuando
                # cantidad > 1).
                return json.dumps({"alimentos": [
                    {"nombre": "Huevo", "es_real": True, "cantidad": 3,
                     "porcion_g": 50, "kcal": 70, "prot_g": 6, "carb_g": 0.5, "grasa_g": 5},
                ]})
            return json.dumps({"alimentos": []})

        with patch.object(ia_engine, "_llamar_groq", new=mock_groq):
            resultado = await registrar_comida_llm(
                "Comí tres huevos", sample_client, plan_hoy, db, ia_engine,
            )
        assert resultado["success"] is True

        filas = _filas_de_hoy(db, sample_client.id)
        assert len(filas) == 3, f"Se esperaban 3 filas (una por huevo), hay {len(filas)}"

        prog = _progreso_de_hoy(db, sample_client.id)
        # 3 huevos × (4*6 + 4*0.5 + 9*5) = 3 × 71 = 213 kcal
        kcal_esperado_por_unidad = 4 * 6 + 4 * 0.5 + 9 * 5
        assert abs(prog.calorias_consumidas - kcal_esperado_por_unidad * 3) < 2, (
            f"El total no refleja 3 huevos multiplicados: {prog.calorias_consumidas} "
            f"(esperado ~{kcal_esperado_por_unidad * 3})"
        )

    @pytest.mark.asyncio
    async def test_correccion_reemplaza_no_acumula(self, db, sample_client, plan_hoy):
        async def mock_groq(prompt, max_tokens=800, temp=0.7, model=None):
            prompt_lower = prompt.lower()
            if "extrae todos los alimentos" in prompt_lower:
                msg = _get_mensaje(prompt)
                if "huevos" in msg and ("tres" in msg or " 3" in msg):
                    return json.dumps({"alimentos": [
                        {"nombre": "Huevo", "es_real": True, "cantidad": 3,
                         "porcion_g": 50, "kcal": 70, "prot_g": 6, "carb_g": 0.5, "grasa_g": 5},
                    ]})
                if "huevos" in msg and "dos" in msg:
                    return json.dumps({"alimentos": [
                        {"nombre": "Huevo", "es_real": True, "cantidad": 2,
                         "porcion_g": 50, "kcal": 70, "prot_g": 6, "carb_g": 0.5, "grasa_g": 5},
                    ]})
                return json.dumps({"alimentos": []})
            return json.dumps({"alimentos": []})

        with patch.object(ia_engine, "_llamar_groq", new=mock_groq):
            await registrar_comida_llm("Comí dos huevos", sample_client, plan_hoy, db, ia_engine)
            resultado2 = await registrar_comida_llm(
                "Corrección, fueron tres huevos", sample_client, plan_hoy, db, ia_engine,
            )
        assert resultado2["success"] is True

        filas = _filas_de_hoy(db, sample_client.id)
        assert len(filas) == 3, (
            f"Tras la corrección debe haber EXACTAMENTE 3 filas (no 2+3=5): "
            f"{[(f.nombre_alimento, f.kcal) for f in filas]}"
        )
        prog = _progreso_de_hoy(db, sample_client.id)
        suma_filas = sum(f.kcal for f in filas)
        assert abs(prog.calorias_consumidas - suma_filas) < 1.5, (
            f"progreso_calorias ({prog.calorias_consumidas}) no coincide con la suma "
            f"post-corrección ({suma_filas}) — quedaron calorías viejas acumuladas."
        )


# ════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Aislamiento de _macro_cache entre usuarios distintos.
# ════════════════════════════════════════════════════════════════════════
@pytest.mark.integration
class TestAislamientoCache:

    @pytest.mark.asyncio
    async def test_macro_cache_es_global_no_aislado_por_usuario(self, db, sample_client, sample_user):
        """Documenta el comportamiento REAL de _macro_cache: la key es solo
        el nombre normalizado del alimento (_cache_key = _normalizar_nombre),
        sin client_id. Esto es así por diseño (consistencia recomendación↔
        registro dentro de una sesión), pero significa que SÍ puede haber
        bleed entre usuarios distintos si mencionan el mismo alimento dentro
        del TTL de 2h. Este test no afirma que sea un bug — confirma el
        comportamiento actual para que quede documentado, no asumido."""
        from app.services.llm_registro import cache_macros, _buscar_en_cache

        # Cliente A registra con un valor MUY específico (ej. de una etiqueta)
        cache_macros("Yogur griego marca X", {
            "nombre": "Yogur griego marca X", "kcal": 999, "prot_g": 50, "carb_g": 1, "grasa_g": 1,
        })

        # Si otro "usuario" (en código, simplemente otra llamada) menciona
        # el mismo nombre normalizado, antes de que expire el TTL...
        encontrado = _buscar_en_cache("Comí un yogur griego marca x")
        if encontrado and abs(encontrado.get("kcal", 0) - 999) < 1:
            pytest.xfail(
                "CONFIRMADO (no es bug nuevo, es diseño actual): _macro_cache no "
                "distingue usuarios — el valor de 999 kcal cacheado por una sesión "
                "se reutilizaría para cualquier otra mención del mismo nombre "
                "normalizado dentro del TTL de 2h, sin importar el client_id."
            )
        else:
            pytest.fail(
                "El comportamiento de _macro_cache cambió respecto a lo documentado "
                "(ya no es un cache global por nombre) — revisar si fue intencional."
            )

    @pytest.mark.asyncio
    async def test_misma_comida_no_cambia_macros_por_orden_de_ejecucion(self, db, sample_client, plan_hoy):
        """Dos registros consecutivos del MISMO alimento (mismo client_id,
        mismo día) deben dar macros consistentes entre sí — no depender de
        qué orden se llamó la función."""
        llamadas = {"n": 0}

        async def mock_groq(prompt, max_tokens=800, temp=0.7, model=None):
            if "extrae todos los alimentos" in prompt.lower():
                llamadas["n"] += 1
                return json.dumps({"alimentos": [
                    {"nombre": "Quinua con leche", "es_real": True, "cantidad": 1,
                     "porcion_g": 200, "kcal": 220, "prot_g": 8, "carb_g": 35, "grasa_g": 4},
                ]})
            return json.dumps({"alimentos": []})

        with patch.object(ia_engine, "_llamar_groq", new=mock_groq):
            r1 = await registrar_comida_llm(
                "Comí quinua con leche", sample_client, plan_hoy, db, ia_engine,
            )
            r2 = await registrar_comida_llm(
                "Comí quinua con leche", sample_client, plan_hoy, db, ia_engine,
            )
        assert r1["datos"]["calorias"] == r2["datos"]["calorias"], (
            f"La misma comida dio kcal distintas en la 2da llamada: "
            f"{r1['datos']['calorias']} vs {r2['datos']['calorias']}"
        )
