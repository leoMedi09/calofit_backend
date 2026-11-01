"""El perfil del Random Forest (A/B/C) se SUMA a los prompts de recomendación; los datos del cliente no se tocan."""
from unittest.mock import patch

import pytest

from app.services.ia_service import ia_engine
from app.services.llm_registro import respuesta_recomendacion_llm
from app.services.ml_service import ml_perfil

MARCA = "PERFIL DE ADHERENCIA"
PERFILES = ["PERFIL_A", "PERFIL_B", "PERFIL_C"]


async def _prompts(perfil, db, modo, mensaje, rf):
    """Prompts enviados al LLM con el perfil RF fijado en ``rf`` (None = modelo sin perfil)."""
    capturados = []

    async def falso(prompt, *a, **k):
        capturados.append(prompt)
        return "PLATOS: Quinua con verduras (~300 kcal,P:10g,C:40g,G:8g)|Lentejas (~320 kcal,P:18g,C:45g,G:4g)|Sopa de verduras (~200 kcal,P:6g,C:30g,G:4g)\nTe propongo tres opciones."

    with patch.object(ia_engine, "_llamar_groq", new=falso), patch("app.services.llm_registro.obtener_perfil_rf", return_value=rf):
        await respuesta_recomendacion_llm(
            mensaje=mensaje, perfil=perfil, consumido=500.0, meta=2000.0, quemado=0.0,
            ia_engine=ia_engine, modo=modo, historial=[], db=db,
        )
    return capturados


def _con_marca(prompts):
    return [p for p in prompts if MARCA in p]


@pytest.fixture
def cliente(db, sample_client):
    sample_client.medical_conditions = ["Diabetes", "Intolerancia a la Lactosa"]
    sample_client.forbidden_foods = ["palta"]
    sample_client.goal = "Perder peso"
    db.commit()
    return sample_client


@pytest.mark.asyncio
@pytest.mark.parametrize("modo,mensaje", [("comida", "¿qué ceno hoy?"), ("ejercicio", "¿qué ejercicios hago de pierna?")])
async def test_cada_perfil_agrega_su_bloque_y_solo_el_suyo(db, cliente, modo, mensaje):
    tonos = {p: ml_perfil.get_tono_asistente(p) for p in PERFILES}
    for p in PERFILES:
        prompts = _con_marca(await _prompts(cliente, db, modo, mensaje, (p, 88.0)))
        assert len(prompts) == 1, f"{modo} {p}: el bloque debe estar en exactamente un prompt"
        assert tonos[p] in prompts[0]
        for otro in PERFILES:
            if otro != p:
                assert tonos[otro] not in prompts[0]
        assert "prioridad absoluta" in prompts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("modo,mensaje", [("comida", "¿qué ceno hoy?"), ("ejercicio", "¿qué ejercicios hago de pierna?")])
async def test_el_perfil_no_quita_ningun_dato_del_cliente(db, cliente, modo, mensaje):
    """Sin perfil RF y con perfil RF el prompt es el mismo salvo el bloque nuevo (todas las líneas previas siguen)."""
    sin = await _prompts(cliente, db, modo, mensaje, None)
    con = await _prompts(cliente, db, modo, mensaje, ("PERFIL_B", 88.0))
    assert not _con_marca(sin)
    principal_con = _con_marca(con)[0]
    # el prompt equivalente sin perfil: el que comparte su primera línea
    base = next(p for p in sin if p.splitlines()[0] == principal_con.splitlines()[0])
    for linea in (l for l in base.splitlines() if l.strip()):
        assert linea in principal_con, f"se perdió la línea del cliente: {linea[:80]!r}"
    # los datos del cliente siguen visibles
    assert cliente.first_name in principal_con
    if modo == "ejercicio":
        assert "Diabetes" in principal_con  # el objetivo ya se verifica con la comparación línea por línea


@pytest.mark.asyncio
async def test_rutina_usa_series_y_descanso_del_perfil(db, cliente):
    esperado = {"PERFIL_A": "4×10 con 60 s", "PERFIL_B": "3×12 con 90 s", "PERFIL_C": "3×15 con 120 s"}
    for p, texto in esperado.items():
        prompts = _con_marca(await _prompts(cliente, db, "ejercicio", "¿qué hago de pierna?", (p, 90.0)))
        assert texto in prompts[0]


@pytest.mark.asyncio
async def test_sin_perfil_el_prompt_queda_como_antes(db, cliente):
    for modo, msg in (("comida", "¿qué ceno?"), ("ejercicio", "¿qué hago de pecho?")):
        assert not _con_marca(await _prompts(cliente, db, modo, msg, None))
