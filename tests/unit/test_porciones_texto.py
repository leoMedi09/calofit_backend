from app.services.llm_registro import _porciones_texto


def test_porciones_texto():
    r = _porciones_texto(
        [
            {"nombre": "Avena", "porcion_g": 250, "cantidad": 1},
            {"nombre": "Huevo", "porcion_g": 55, "cantidad": 2},
            {"nombre": "Sin porcion", "cantidad": 1},
            {"nombre": "Pan", "porcion_g": "x"},
        ],
        factor=0.5,
    )
    assert r == ["Avena — 125 g", "2 × Huevo — 55 g"]
    assert _porciones_texto(None) == []


def test_medida_del_modelo_o_del_usuario():
    r = _porciones_texto(
        [
            {"nombre": "Ají de gallina", "porcion_g": 150, "cantidad": 1, "medida": "1 plato"},
            {"nombre": "Pollo", "porcion_g": 200, "cantidad": 1, "medida": "200 g"},
            {"nombre": "Huevo", "porcion_g": 55, "cantidad": 2, "medida": "2 unidades"},
            {"nombre": "Palta", "porcion_g": 60, "cantidad": 1, "medida": "1 porción", "_factor_acumulado": 0.5},
        ]
    )
    assert r == ["1 plato de Ají de gallina (~150 g)", "Pollo — 200 g", "2 unidades de Huevo (~110 g)", "Palta — 60 g"]
    # con topes de kcal aplicados, la medida del modelo ya no es fiable
    assert _porciones_texto([{"nombre": "Arroz", "porcion_g": 300, "medida": "1 plato"}], factor=0.5) == ["Arroz — 150 g"]


def test_bebidas_en_ml():
    r = _porciones_texto(
        [
            {"nombre": "Limonada", "porcion_g": 250, "medida": "1 vaso"},
            {"nombre": "Chicha morada", "porcion_g": 250},
            {"nombre": "Arroz con leche", "porcion_g": 200},
            {"nombre": "Avena", "porcion_g": 250, "medida": "1 taza"},
        ]
    )
    assert r == ["1 vaso de Limonada (~250 ml)", "Chicha morada — 250 ml", "Arroz con leche — 200 g", "1 taza de Avena (~250 g)"]


def test_quita_etiqueta_de_restriccion_salvo_productos():
    from app.services.llm_registro import _quitar_etiqueta_restriccion as q

    assert q("Caldo de gallina sin lácteos") == "Caldo de gallina"
    assert q("Para tu cena te propongo Caldo de gallina sin lácteos (~280 kcal) y Seco sin gluten (~600 kcal).") == (
        "Para tu cena te propongo Caldo de gallina (~280 kcal) y Seco (~600 kcal)."
    )
    assert q("Pan sin gluten con palta") == "Pan sin gluten con palta"
    assert q("Queso sin lactosa") == "Queso sin lactosa"


def test_calculo_ejercicio_texto():
    from app.services.llm_registro import _calculo_ejercicio_texto as c

    assert c(15.0, 86.9057, 5.0, False) == "15 min · 87 kg · MET 5 (esfuerzo moderado)"
    assert c(15.0, 70.0, 8.0, True) == "15 min · 70 kg · MET 8 (esfuerzo alto) · duración supuesta"
    assert "ligero" in c(10.0, 70.0, 2.5, False)


def test_fuente_ejercicio_solo_cita_el_compendium_si_todos_lo_tienen():
    from app.core.mets_gym import COMPENDIUM_CODIGO, METS_GYM
    from app.services.llm_registro import _fuente_ejercicio as f

    assert f([{"compendium": "02054"}]).startswith("Compendium of Physical Activities 2024")
    assert f([{"compendium": "02054"}, {"compendium": None}]).startswith("MET de referencia estimado")
    assert f([]).startswith("MET de referencia estimado")
    # el press de banca queda en el valor del Compendium (3.5) y con su codigo
    assert METS_GYM["press banca"] == 3.5 and COMPENDIUM_CODIGO["press banca"] == "02054"
    assert METS_GYM["sentadilla libre"] == 5.0 and METS_GYM["correr moderado"] == 9.3
