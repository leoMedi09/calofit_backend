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
