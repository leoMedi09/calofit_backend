"""«hice N minutos de <ejercicio>» (pasado o presente) es un registro de ejercicio, no una recomendación."""
import pytest

from app.services.asistente.asistente_modos import (
    OTRO,
    RECOMENDAR_EJERCICIO,
    REGISTRAR_EJERCICIO,
    REGISTRAR_NUTRICION,
    detectar_modo_funcion,
)


@pytest.mark.parametrize(
    "mensaje",
    [
        "hice 20 minutos de press banca",
        "hice 30 min de cardio",
        "corrí 45 minutos",
        "entrené piernas una hora",
        "caminé 1 hora",
        "hago 40 minutos de bicicleta",
        "entreno pecho 1.5 horas",
        "hice press banca 20 minutos",
    ],
)
def test_registro_ejercicio_con_duracion(mensaje):
    assert detectar_modo_funcion(mensaje, False) == REGISTRAR_EJERCICIO


@pytest.mark.parametrize(
    "mensaje, esperado, es_saludo",
    [
        ("qué ejercicios hago hoy", RECOMENDAR_EJERCICIO, False),
        ("hola", OTRO, True),
        ("regístrame 200 g de arroz", REGISTRAR_NUTRICION, False),
        ("comí arroz con pollo", REGISTRAR_NUTRICION, False),
        ("¿qué rutina hago 30 minutos?", RECOMENDAR_EJERCICIO, False),
    ],
)
def test_no_rompe_otros_modos(mensaje, esperado, es_saludo):
    assert detectar_modo_funcion(mensaje, es_saludo) == esperado
