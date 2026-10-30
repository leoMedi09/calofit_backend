"""El fallback de recomendaciones no debe ofrecer opciones incompatibles con las alergias/restricciones del cliente.

Cada caso se ejecuta 200 veces por momento del día para cubrir el azar de random.sample. La comprobación de palabras
prohibidas es independiente de las etiquetas del módulo (si una opción estuviera mal etiquetada, la prueba lo detecta).
"""
import re

import pytest

from app.core.fallback_recomendaciones import MENSAJE_SIN_OPCION_SEGURA, OPCIONES, seleccionar_opciones
from app.services.llm_registro import _FALLBACKS_OPCIONES, _fallback_recomendacion, obtener_fallback_aleatorio

MOMENTOS = ["DESAYUNO", "ALMUERZO", "CENA", "MERIENDA"]
REPETICIONES = 200

# palabras que NO pueden aparecer en ninguna opción devuelta (comprobación por texto, independiente de las etiquetas)
FRUTOS_SECOS = r"almendra|nuez|nueces|mani\b|maní|cacahuate|frutos secos"
HUEVO = r"huevo|tortilla"
GLUTEN = r"tostada|avena|\bpan\b|trigo|cebada|fideo|pasta"
LACTEOS = r"crema|leche|queso|yogur|mantequilla"
AZUCAR = r"batido|plátano|azúcar|azucar|dulce|miel|mermelada"


def _opciones(momento, condiciones=None, prohibidos=None, aleatorio=True):
    texto = obtener_fallback_aleatorio(momento, aleatorio, condiciones, prohibidos)
    return [] if texto == MENSAJE_SIN_OPCION_SEGURA else texto.split(chr(10))


def _sin(patron, condiciones, prohibidos=None):
    rx = re.compile(patron, re.IGNORECASE)
    for m in MOMENTOS:
        for _ in range(REPETICIONES):
            ops = _opciones(m, condiciones, prohibidos)
            assert ops, f"{m} {condiciones}: se esperaba al menos una opción segura"
            assert len(ops) == len(set(ops)), f"opciones repetidas: {ops}"
            for o in ops:
                assert not rx.search(o), f"{m} {condiciones}: '{o}' contiene un alérgeno ({patron})"


@pytest.mark.parametrize("condicion", ["Alergia al maní", "alergia a frutos secos", "Alérgico a las nueces"])
def test_alergia_frutos_secos(condicion):
    _sin(FRUTOS_SECOS, [condicion])


def test_alergia_huevo():
    _sin(HUEVO, ["Alergia al huevo"])


@pytest.mark.parametrize("condicion", ["Celíaco", "celiaquía", "Sin gluten"])
def test_celiaco_sin_gluten(condicion):
    _sin(GLUTEN, [condicion])


def test_intolerancia_lacteos():
    _sin(LACTEOS, ["Intolerancia a la Lactosa"])


def test_diabetico():
    _sin(AZUCAR, ["Diabetes"])


def test_vegano_sin_huevo_ni_lacteos():
    _sin(HUEVO + "|" + LACTEOS, ["Vegano"])


def test_varias_restricciones_a_la_vez():
    _sin(
        "|".join([FRUTOS_SECOS, HUEVO, GLUTEN, LACTEOS, AZUCAR]),
        ["Celíaco", "Alergia a frutos secos", "Intolerancia a la Lactosa", "Diabetes", "Alergia al huevo"],
    )


def test_lista_de_alimentos_prohibidos_del_nutricionista():
    _sin(r"palta", [], prohibidos=["palta"])
    _sin(r"huevo|tortilla", [], prohibidos=["huevos"])  # exclusión pedida en el mensaje ("no quiero huevos")


def test_completa_con_opciones_seguras_si_quedan_menos_de_tres():
    # DESAYUNO: solo "Fruta picada con chía" cumple celíaco + frutos secos + diabetes; el resto lo completan las seguras.
    cond = ["Celíaco", "Alergia a frutos secos", "Diabetes"]
    for aleatorio in (True, False):
        ops = _opciones("DESAYUNO", cond, aleatorio=aleatorio)
        assert len(ops) == 3 and len(set(ops)) == 3
        assert any("Fruta picada" in o for o in ops)
        assert any("Sopa de verduras" in o or "Ensalada" in o for o in ops)


def test_alergia_no_interpretable_no_recomienda_nada_especifico():
    for m in MOMENTOS:
        assert obtener_fallback_aleatorio(m, True, ["Alergia a la fresa"]) == MENSAJE_SIN_OPCION_SEGURA
        assert seleccionar_opciones(m, ["Alergia a la fresa"]) == []
    assert "nutricionista" in MENSAJE_SIN_OPCION_SEGURA.lower()


def test_intro_no_acompana_al_mensaje_sin_opcion():
    sin = _fallback_recomendacion("Tuve un problema técnico:", "CENA", ["Alergia a la fresa"], [])
    assert sin == MENSAJE_SIN_OPCION_SEGURA
    con = _fallback_recomendacion("Tuve un problema técnico:", "CENA", [], [])
    assert con.startswith("Tuve un problema técnico:") and len(con.split(chr(10))) == 4


def test_condiciones_sin_efecto_no_cambian_nada():
    # asma / hipertensión / 'Ninguna' no excluyen opciones: se comporta como sin restricciones
    for cond in (["Asma"], ["Hipertensión"], ["Ninguna"], ["Enfermedad Cardiovascular"]):
        for m in MOMENTOS:
            assert _opciones(m, cond, aleatorio=False) == _FALLBACKS_OPCIONES[m][:3]


@pytest.mark.parametrize("momento", MOMENTOS)
def test_sin_restricciones_comportamiento_actual(momento):
    base = _FALLBACKS_OPCIONES[momento]
    vistos = set()
    for _ in range(REPETICIONES):
        ops = _opciones(momento, [], [])
        assert len(ops) == 3 and len(set(ops)) == 3 and set(ops) <= set(base)  # 3 distintas del momento
        vistos.add(tuple(ops))
    assert len(vistos) > 1  # sigue siendo aleatorio
    assert _opciones(momento, None, None, aleatorio=False) == base[:3]  # aleatorio=False: siempre las 3 primeras
    assert obtener_fallback_aleatorio(momento, False) == chr(10).join(base[:3])


def test_toda_opcion_tiene_texto_con_formato_y_momentos_completos():
    assert set(OPCIONES) == set(MOMENTOS)
    for ops in OPCIONES.values():
        for o in ops:
            assert o.texto.startswith("- ") and "(~" in o.texto
