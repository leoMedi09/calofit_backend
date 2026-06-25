"""
Pruebas del flujo de corrección de un registro nutricional ya hecho.

Caso encontrado en la auditoría final pre-demo: "Comí dos huevos" seguido de
"Corrección, fueron tres" no actualizaba el registro — pedía el alimento de
nuevo y/o podía duplicar filas en comida_registros. Ver llm_registro.py:
_RX_CORRECCION_REGISTRO + bloque de corrección al inicio de registrar_comida_llm.
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
    """Extrae el texto entre `Mensaje: "..."` que _PROMPT_COMIDA siempre incluye."""
    marker = 'mensaje: "'
    low = prompt.lower()
    idx = low.find(marker)
    if idx == -1:
        return low
    resto = prompt[idx + len(marker):]
    return resto.split('"', 1)[0].lower()


@pytest.mark.integration
class TestCorreccionRegistroNutricional:

    @pytest.fixture(autouse=True)
    def mock_groq_extraccion(self):
        async def mock_llamar_groq(prompt, max_tokens=800, temp=0.7, model=None):
            prompt_lower = prompt.lower()
            if "extrae todos los alimentos" in prompt_lower or "responde solo con json" in prompt_lower:
                msg = _get_mensaje(prompt)
                if "huevos" in msg and ("tres" in msg or " 3" in msg):
                    # Corrección ya resuelta a "Huevos - ..." por el código,
                    # con la nueva cantidad mencionada en el mensaje.
                    return json.dumps({
                        "alimentos": [
                            {"nombre": "Huevos", "es_real": True, "cantidad": 3,
                             "porcion_g": 50, "kcal": 75, "prot_g": 6.3,
                             "carb_g": 0.6, "grasa_g": 5.0}
                        ]
                    })
                if "huevos" in msg and ("dos" in msg or " 2" in msg):
                    return json.dumps({
                        "alimentos": [
                            {"nombre": "Huevos", "es_real": True, "cantidad": 2,
                             "porcion_g": 50, "kcal": 75, "prot_g": 6.3,
                             "carb_g": 0.6, "grasa_g": 5.0}
                        ]
                    })
                return json.dumps({"alimentos": [], "prot_total": 0, "carb_total": 0, "grasa_total": 0})
            # Microconsultas/otros prompts no relevantes para este test.
            return json.dumps({"alimentos": []})

        with patch.object(ia_engine, "_llamar_groq", new=mock_llamar_groq):
            yield

    @pytest.fixture
    def plan_hoy(self):
        return {"calorias_dia": 2000, "proteinas_g": 150, "carbohidratos_g": 220, "grasas_g": 60}

    @pytest.mark.asyncio
    async def test_correccion_actualiza_sin_duplicar(self, db, sample_client, plan_hoy):
        hoy = get_peru_date()

        # Turno 1: "Comí dos huevos"
        r1 = await registrar_comida_llm("Comí dos huevos", sample_client, plan_hoy, db, ia_engine)
        assert r1["success"] is True

        filas_antes = (
            db.query(ComidaRegistro)
            .filter(ComidaRegistro.client_id == sample_client.id, ComidaRegistro.fecha == hoy)
            .all()
        )
        assert len(filas_antes) == 2  # 2 huevos = 2 filas (una por unidad)
        assert all(f.nombre_alimento == "Huevos" for f in filas_antes)

        prog_antes = (
            db.query(ProgresoCalorias)
            .filter(ProgresoCalorias.client_id == sample_client.id, ProgresoCalorias.fecha == hoy)
            .first()
        )
        assert prog_antes is not None
        kcal_antes = prog_antes.calorias_consumidas

        # Turno 2: corrección — NO repite el nombre del alimento a propósito,
        # es justo el caso que fallaba antes del fix.
        r2 = await registrar_comida_llm("Corrección, fueron tres", sample_client, plan_hoy, db, ia_engine)
        assert r2["success"] is True
        assert "huevo" in r2["mensaje"].lower()
        assert "no identifiqué" not in r2["mensaje"].lower()

        filas_despues = (
            db.query(ComidaRegistro)
            .filter(ComidaRegistro.client_id == sample_client.id, ComidaRegistro.fecha == hoy)
            .all()
        )
        # Las 2 filas viejas se reemplazan por 3 nuevas — nunca deben coexistir
        # las 2 viejas + las 3 nuevas (eso sería duplicación).
        assert len(filas_despues) == 3, (
            f"Se esperaban exactamente 3 filas de 'Huevos' tras la corrección, "
            f"hay {len(filas_despues)} — posible duplicación o reemplazo incompleto."
        )
        assert all(f.nombre_alimento == "Huevos" for f in filas_despues)

        prog_despues = (
            db.query(ProgresoCalorias)
            .filter(ProgresoCalorias.client_id == sample_client.id, ProgresoCalorias.fecha == hoy)
            .first()
        )
        kcal_filas_despues = sum(f.kcal for f in filas_despues)
        # El total diario debe reflejar SOLO las filas actuales (3 huevos),
        # no las 2 viejas + las 3 nuevas sumadas.
        assert abs(prog_despues.calorias_consumidas - kcal_filas_despues) < 2.0
        # El total cambió respecto al de antes de la corrección.
        assert prog_despues.calorias_consumidas != kcal_antes

    @pytest.mark.asyncio
    async def test_correccion_sin_registro_previo_no_falla(self, db, sample_client, plan_hoy):
        """Si no hay nada que corregir (primer mensaje del día), el mensaje de
        corrección no debe causar una excepción — debe degradar al flujo normal
        de 'no identifiqué ningún alimento'."""
        r = await registrar_comida_llm("Corrección, fueron tres", sample_client, plan_hoy, db, ia_engine)
        assert r["success"] is False
        assert "tipo_detectado" in r
