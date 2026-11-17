"""«Dame otra opción» / «no me gusta, cámbialo» tras una recomendación: vuelve a recomendar, sin repetir y respetando el momento."""
from unittest.mock import patch

import pytest

from app.services.asistente.asistente_modos import (
    RECOMENDAR_EJERCICIO,
    RECOMENDAR_NUTRICION,
    OTRO,
    detectar_modo_funcion,
    modo_por_cambio_de_opcion,
    resolver_modo_funcion,
)
from app.services.ia_service import ia_engine
from app.services.llm_registro import respuesta_recomendacion_llm

HIST_COMIDA = [
    {"role": "user", "content": "¿qué ceno hoy?"},
    {"role": "assistant", "content": "Para tu cena te propongo Pollo a la plancha (~400 kcal), Sopa de quinua (~300 kcal) y Tortilla (~350 kcal)."},
]
HIST_EJERCICIO = [
    {"role": "user", "content": "rutina de pecho"},
    {"role": "assistant", "content": "Press de banca 4 series de 10 reps, aperturas 3 series de 12 reps."},
]
FRASES = ["dame otra opción", "no me gusta, cámbialo", "dame otro plato", "algo diferente por favor", "No me gusta el pollo"]


@pytest.mark.parametrize("frase", FRASES)
def test_cambio_tras_recomendacion_de_comida(frase):
    assert modo_por_cambio_de_opcion(frase, HIST_COMIDA) == RECOMENDAR_NUTRICION
    assert detectar_modo_funcion(frase, False, HIST_COMIDA) == RECOMENDAR_NUTRICION


def test_cambio_tras_rutina_recomienda_ejercicio():
    assert modo_por_cambio_de_opcion("dame otra opción", HIST_EJERCICIO) == RECOMENDAR_EJERCICIO


@pytest.mark.parametrize("hist", [None, [], [{"role": "assistant", "content": "Hola, ¿en qué te ayudo?"}], [{"role": "user", "content": "x"}]])
def test_sin_recomendacion_previa_no_se_activa(hist):
    assert modo_por_cambio_de_opcion("dame otra opción", hist) is None
    assert detectar_modo_funcion("dame otra opción", False, hist) == OTRO


@pytest.mark.parametrize("frase", ["¿cuál tiene más proteína?", "gracias", "me gusta, anótalo", "puedo comer palta de noche"])
def test_otros_mensajes_no_se_desvian(frase):
    assert modo_por_cambio_de_opcion(frase, HIST_COMIDA) is None


@pytest.mark.asyncio
async def test_resolver_modo_no_depende_del_llm():
    class IAQueFallaria:
        async def clasificar_modo_asistente(self, *a, **k):  # si se llamara, devolvería 'otro'
            return "otro"

    assert await resolver_modo_funcion(IAQueFallaria(), "dame otra opción", False, historial=HIST_COMIDA) == RECOMENDAR_NUTRICION


async def _prompt(perfil, db, mensaje, historial):
    capturados = []

    async def falso(prompt, *a, **k):
        capturados.append(prompt)
        return "PLATOS: Quinua con verduras (~300 kcal,P:10g,C:40g,G:8g)|Lentejas (~320 kcal,P:18g,C:45g,G:4g)|Sopa de verduras (~200 kcal,P:6g,C:30g,G:4g)\nTe propongo tres opciones."

    with patch.object(ia_engine, "_llamar_groq", new=falso):
        await respuesta_recomendacion_llm(
            mensaje=mensaje, perfil=perfil, consumido=500.0, meta=2000.0, quemado=0.0,
            ia_engine=ia_engine, modo="comida", historial=historial, db=db,
        )
    return next(p for p in capturados if "PLATOS:" in p and "dice:" in p)


@pytest.mark.asyncio
async def test_mantiene_el_momento_de_la_recomendacion_anterior(db, sample_client):
    # sin importar la hora del servidor, «otra opción» tras una CENA sigue siendo CENA
    p = await _prompt(sample_client, db, "dame otra opción", HIST_COMIDA)
    assert "disponibles para cena" in p


@pytest.mark.asyncio
async def test_no_me_gusta_un_alimento_lo_excluye_del_prompt(db, sample_client):
    p = await _prompt(sample_client, db, "no me gusta el pollo, cámbialo", HIST_COMIDA)
    assert "NO quiere comer: **pollo**" in p


@pytest.mark.asyncio
async def test_no_me_gusta_sin_alimento_no_inventa_exclusion(db, sample_client):
    for frase in ("no me gusta, cámbialo", "no me gusta eso"):
        p = await _prompt(sample_client, db, frase, HIST_COMIDA)
        assert "NO quiere comer" not in p


HIST_VIÑETAS = [
    {"role": "user", "content": "¿qué ceno hoy?"},
    {"role": "assistant", "content": "Para tu cena:\n- Tortilla de huevo con verduras (~300 kcal)\n- Sopa de quinua (~250 kcal)"},
]
REPETIDO = "PLATOS: Tortilla de huevo con verduras (~300 kcal,P:20g,C:10g,G:15g)|Sopa de quinua (~250 kcal,P:9g,C:30g,G:5g)|Pescado a la plancha (~350 kcal,P:30g,C:5g,G:10g)\nTe propongo tres opciones."
NUEVO = "PLATOS: Crema de zapallo (~200 kcal,P:5g,C:30g,G:6g)|Pescado a la plancha (~350 kcal,P:30g,C:5g,G:10g)|Menestra de lentejas (~380 kcal,P:20g,C:50g,G:6g)\nTe propongo Crema de zapallo (~200 kcal), Pescado a la plancha (~350 kcal) y Menestra de lentejas (~380 kcal)."


async def _consultar(perfil, db, mensaje, historial, respuestas):
    prompts, it = [], iter(respuestas)

    async def falso(prompt, *a, **k):
        if "dice:" not in prompt:  # evaluador del KNN u otras llamadas auxiliares
            return respuestas[-1]
        prompts.append(prompt)
        return next(it, respuestas[-1])

    with patch.object(ia_engine, "_llamar_groq", new=falso):
        texto = await respuesta_recomendacion_llm(
            mensaje=mensaje, perfil=perfil, consumido=500.0, meta=2000.0, quemado=0.0,
            ia_engine=ia_engine, modo="comida", historial=historial, db=db,
        )
    return texto, prompts


@pytest.mark.asyncio
async def test_si_el_llm_repite_platos_se_reintenta_y_se_usa_la_respuesta_nueva(db, sample_client):
    texto, prompts = await _consultar(sample_client, db, "dame otra opción", HIST_VIÑETAS, [REPETIDO, NUEVO])
    assert "Crema de zapallo" in texto and "Tortilla" not in texto
    assert any("PROHIBIDO repetir" in p and "Tortilla de huevo con verduras" in p for p in prompts)


@pytest.mark.asyncio
async def test_si_no_repite_no_hay_reintento(db, sample_client):
    _, prompts = await _consultar(sample_client, db, "dame otra opción", HIST_VIÑETAS, [NUEVO])
    assert not any("PROHIBIDO repetir" in p for p in prompts)


@pytest.mark.asyncio
async def test_si_el_reintento_tambien_repite_se_queda_con_el_que_repite_menos(db, sample_client):
    peor = REPETIDO.replace("Pescado a la plancha", "Sopa de quinua")  # 3 repetidos/duplicados
    texto, _ = await _consultar(sample_client, db, "dame otra opción", HIST_VIÑETAS, [REPETIDO, peor])
    assert "Pescado a la plancha" in texto  # conserva la primera (2 repetidos < 3)
