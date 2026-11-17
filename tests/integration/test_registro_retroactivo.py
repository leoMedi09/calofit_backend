"""«Ayer comí…», «anteayer cené…», «anoche hice…»: el registro se guarda en ese día (máx. 2 atrás), no en hoy."""
import json
from datetime import date, timedelta
from unittest.mock import patch

import pytest
from sqlalchemy import text

from app.core.fecha_retroactiva import extraer_fecha_retroactiva
from app.core.utils import get_peru_date
from app.models.comida_registro import ComidaRegistro
from app.models.historial import ProgresoCalorias
from app.services.ia_service import ia_engine
from app.services.llm_registro import registrar_comida_llm, registrar_ejercicio_llm

HOY = date(2026, 11, 16)
PLAN = {"calorias_dia": 2000, "proteinas_g": 150, "carbohidratos_g": 220, "grasas_g": 60}


@pytest.mark.parametrize(
    "mensaje,dias,limpio",
    [
        ("ayer comí lomo saltado", 1, "comí lomo saltado"),
        ("Ayer cené arroz con pollo", 1, "cené arroz con pollo"),
        ("anteayer almorcé ceviche", 2, "almorcé ceviche"),
        ("antes de ayer desayuné avena", 2, "desayuné avena"),
        ("antier comí pizza", 2, "comí pizza"),
        ("anoche comí pollo", 1, "comí pollo"),
        ("comí ayer una manzana", 1, "comí una manzana"),
        ("AYER, comí papa", 1, "comí papa"),
    ],
)
def test_detecta_dia_pasado_y_limpia_el_mensaje(mensaje, dias, limpio):
    fecha, msg = extraer_fecha_retroactiva(mensaje, hoy=HOY)
    assert fecha == HOY - timedelta(days=dias)
    assert msg == limpio


@pytest.mark.parametrize(
    "mensaje",
    ["comí lomo saltado", "hoy comí arroz", "ayer comí pollo y hoy arroz", "hace una semana comí pizza", ""],
)
def test_sin_fecha_pasada_o_ambiguo_no_cambia_nada(mensaje):
    fecha, msg = extraer_fecha_retroactiva(mensaje, hoy=HOY)
    assert fecha is None and msg == mensaje


@pytest.fixture
def llm_comida():
    async def falso(prompt, max_tokens=800, temp=0.7, model=None):
        return json.dumps(
            {"alimentos": [{"nombre": "Lomo saltado", "es_real": True, "cantidad": 1, "porcion_g": 350, "kcal": 600,
                            "prot_g": 30, "carb_g": 60, "grasa_g": 25}]}
        )

    with patch.object(ia_engine, "_llamar_groq", new=falso):
        yield


@pytest.mark.asyncio
async def test_comida_de_ayer_se_guarda_en_ayer_y_no_toca_hoy(db, sample_client, llm_comida):
    hoy = get_peru_date()
    ayer = hoy - timedelta(days=1)
    r = await registrar_comida_llm("comí lomo saltado", sample_client, PLAN, db, ia_engine, fecha=ayer)
    assert r["success"] is True and r["es_retroactivo"] is True
    assert f"para el {ayer.strftime('%d/%m')}" in r["mensaje"] and "hoy" not in r["mensaje"].lower().split("kcal")[-1]

    filas = db.query(ComidaRegistro).filter(ComidaRegistro.client_id == sample_client.id).all()
    assert [f.fecha for f in filas] == [ayer]
    prog = {p.fecha: p for p in db.query(ProgresoCalorias).filter(ProgresoCalorias.client_id == sample_client.id)}
    assert prog[ayer].calorias_consumidas > 0
    assert hoy not in prog  # hoy queda intacto


@pytest.mark.asyncio
async def test_comida_sin_fecha_sigue_siendo_de_hoy(db, sample_client, llm_comida):
    r = await registrar_comida_llm("comí lomo saltado", sample_client, PLAN, db, ia_engine)
    assert r["es_retroactivo"] is False and "kcal hoy" in r["mensaje"]
    filas = db.query(ComidaRegistro).filter(ComidaRegistro.client_id == sample_client.id).all()
    assert [f.fecha for f in filas] == [get_peru_date()]


@pytest.mark.asyncio
async def test_ejercicio_de_anteayer_cae_en_ese_dia_en_hora_de_lima(db, sample_client):
    async def falso(prompt, max_tokens=800, temp=0.7, model=None):
        return json.dumps([{"encontrado": True, "ejercicio": "Correr", "duracion_min": 30, "met": 8.0, "kcal_quemadas": 280}])

    hoy = get_peru_date()
    anteayer = hoy - timedelta(days=2)
    with patch.object(ia_engine, "_llamar_groq", new=falso):
        r = await registrar_ejercicio_llm("hice correr 30 minutos", sample_client, db, ia_engine, fecha=anteayer)
    assert r["success"] is True and r["es_retroactivo"] is True and f"para el {anteayer.strftime('%d/%m')}" in r["mensaje"]

    dias = db.execute(
        text("SELECT (created_at AT TIME ZONE 'UTC' AT TIME ZONE 'America/Lima')::date FROM workout_logs WHERE client_id=:c"),
        {"c": sample_client.id},
    ).scalars().all()
    assert dias == [anteayer]
    prog = {p.fecha: p for p in db.query(ProgresoCalorias).filter(ProgresoCalorias.client_id == sample_client.id)}
    assert prog[anteayer].calorias_quemadas > 0 and hoy not in prog


# --- validación de fechas y día elegido en el calendario del Balance ------------------------------------------------------
from app.core.fecha_retroactiva import MAX_DIAS_RETRO, validar_fecha  # noqa: E402
from app.services.asistente.asistente_service import AsistenteService  # noqa: E402


def test_validar_fecha_acepta_hasta_el_limite_y_rechaza_futuro_y_antiguo():
    assert MAX_DIAS_RETRO == 7
    assert validar_fecha("2026-11-15", HOY) == (date(2026, 11, 15), None)
    assert validar_fecha("2026-11-09", HOY) == (date(2026, 11, 9), None)  # justo 7 días
    f, e = validar_fecha("2026-11-08", HOY)  # 8 días
    assert f is None and "hasta 7 días" in e and "08/11" in e
    f, e = validar_fecha("2026-11-20", HOY)
    assert f is None and "todavía no llega" in e


@pytest.mark.parametrize("valor", [None, "", "null", "None", "mañana", "32/13/2026", "2026-13-40", 12345])
def test_validar_fecha_ignora_lo_que_no_es_fecha_y_hoy_no_es_retroactivo(valor):
    assert validar_fecha(valor, HOY) == (None, None)
    assert validar_fecha("2026-11-16", HOY) == (None, None)  # hoy


def test_fecha_de_fin_de_ano_escrita_en_enero_se_corrige_al_ano_anterior():
    assert validar_fecha("2027-12-31", date(2027, 1, 2)) == (date(2026, 12, 31), None)  # 2 días atrás
    f, e = validar_fecha("2027-11-21", date(2026, 11, 16))  # 5 días en el futuro: no es del año pasado
    assert f is None and "todavía no llega" in e


def test_el_calendario_manda_sobre_las_palabras_del_mensaje():
    svc = AsistenteService
    cal = date(2026, 11, 12)
    assert svc._fecha_del_registro("ayer comí arroz", cal) == (cal, "ayer comí arroz")
    f, msg = svc._fecha_del_registro("ayer comí arroz", None)
    assert f is not None and msg == "comí arroz"
    assert svc._fecha_del_registro("comí arroz", None) == (None, "comí arroz")


async def _consultar(db, cliente, mensaje, fecha):
    return await AsistenteService().consultar(mensaje=mensaje, db=db, current_user=cliente, historial=[], fecha=fecha)


def _filas(db, cliente):
    return [f.fecha for f in db.query(ComidaRegistro).filter(ComidaRegistro.client_id == cliente.id)]


@pytest.mark.asyncio
async def test_chat_con_dia_del_calendario_registra_en_ese_dia_y_no_cambia_hoy(db, sample_client, llm_comida):
    ayer = get_peru_date() - timedelta(days=1)
    r = await _consultar(db, sample_client, "comí lomo saltado", ayer.isoformat())
    assert f"para el {ayer.strftime('%d/%m')}" in r["respuesta_ia"]
    assert _filas(db, sample_client) == [ayer]
    assert r["data_cientifica"]["progreso_diario"]["consumido"] == 0  # el balance de hoy no cambia


@pytest.mark.asyncio
async def test_chat_en_dia_pasado_solo_registra_no_recomienda(db, sample_client, llm_comida):
    ayer = get_peru_date() - timedelta(days=1)

    async def clasificador(mensaje, historial=None):  # el clasificador real distingue pedir ideas de registrar
        return "recomendar_nutricion"

    with patch.object(ia_engine, "clasificar_modo_asistente", new=clasificador):
        r = await _consultar(db, sample_client, "¿qué ceno hoy?", ayer.isoformat())
    assert "solo puedo registrar" in r["respuesta_ia"] and "Vuelve a hoy" in r["respuesta_ia"]
    assert _filas(db, sample_client) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("dias,trozo", [(-3, "todavía no llega"), (10, "hasta 7 días")])
async def test_chat_con_fecha_futura_o_muy_antigua_no_registra_nada(db, sample_client, llm_comida, dias, trozo):
    fecha = get_peru_date() - timedelta(days=dias)
    r = await _consultar(db, sample_client, "comí lomo saltado", fecha.isoformat())
    assert trozo in r["respuesta_ia"] and "No registré nada" in r["respuesta_ia"]
    assert _filas(db, sample_client) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("fecha", [None, "", "abc", "hoy"])
async def test_chat_sin_fecha_valida_o_con_basura_registra_hoy(db, sample_client, llm_comida, fecha):
    r = await _consultar(db, sample_client, "comí lomo saltado", fecha)
    assert "kcal hoy" in r["respuesta_ia"]
    assert _filas(db, sample_client) == [get_peru_date()]


@pytest.mark.asyncio
async def test_chat_con_fecha_de_hoy_del_calendario_es_registro_normal(db, sample_client, llm_comida):
    r = await _consultar(db, sample_client, "comí lomo saltado", get_peru_date().isoformat())
    assert "kcal hoy" in r["respuesta_ia"] and _filas(db, sample_client) == [get_peru_date()]
