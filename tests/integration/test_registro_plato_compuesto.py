"""
Prueba end-to-end: registrar un plato compuesto en lenguaje informal no debe
duplicar sus ingredientes como registros separados.

Fuente única de verdad para el caso "Arroz con lentejas" — había una
aserción duplicada en test_fix_alimentos_alucinados_y_contenedores.py
(misma entrada, mismo mock); se consolidó aquí porque esta versión además
verifica el contenido real de la fila en `comida_registros`, no solo la
lista devuelta.
"""
import json

import pytest
from unittest.mock import patch

from app.core.utils import get_peru_date
from app.services.llm_registro import registrar_comida_llm
from app.services.ia_service import ia_engine
from app.models.comida_registro import ComidaRegistro


@pytest.mark.integration
class TestRegistroPlatoCompuesto:

    @pytest.fixture(autouse=True)
    def mock_groq_combo_y_componentes(self):
        """Simula exactamente el comportamiento problemático encontrado en la
        auditoría: el LLM devuelve el plato completo Y sus componentes en la
        misma respuesta JSON."""
        async def mock_llamar_groq(prompt, max_tokens=800, temp=0.7, model=None):
            prompt_lower = prompt.lower()
            if "extrae todos los alimentos" in prompt_lower or "responde solo con json" in prompt_lower:
                if "arroz" in prompt_lower and "lentejas" in prompt_lower:
                    return json.dumps({
                        "alimentos": [
                            {"nombre": "Arroz con lentejas", "es_real": True, "cantidad": 1,
                             "porcion_g": 350, "kcal": 450, "prot_g": 16, "carb_g": 75, "grasa_g": 6},
                            {"nombre": "lentejas", "es_real": True, "cantidad": 1,
                             "porcion_g": 150, "kcal": 200, "prot_g": 12, "carb_g": 30, "grasa_g": 1},
                            {"nombre": "arroz", "es_real": True, "cantidad": 1,
                             "porcion_g": 200, "kcal": 260, "prot_g": 4, "carb_g": 56, "grasa_g": 0.5},
                        ]
                    })
                return json.dumps({"alimentos": []})
            return json.dumps({"alimentos": []})

        with patch.object(ia_engine, "_llamar_groq", new=mock_llamar_groq):
            yield

    @pytest.mark.asyncio
    async def test_arroz_con_lentejas_no_duplica_ingredientes(self, db, sample_client):
        plan_hoy = {"calorias_dia": 2000, "proteinas_g": 150, "carbohidratos_g": 220, "grasas_g": 60}
        resultado = await registrar_comida_llm(
            "Me metí un poco de arroz con lentejas", sample_client, plan_hoy, db, ia_engine
        )
        assert resultado["success"] is True
        # Solo el plato completo, nunca sus componentes por separado.
        assert resultado["alimentos"] == ["Arroz con lentejas"]

        filas = (
            db.query(ComidaRegistro)
            .filter(
                ComidaRegistro.client_id == sample_client.id,
                ComidaRegistro.fecha == get_peru_date(),
            )
            .all()
        )
        assert len(filas) == 1, f"Se esperaba 1 registro, hay {len(filas)}: {[f.nombre_alimento for f in filas]}"
        assert filas[0].nombre_alimento == "Arroz con lentejas"
