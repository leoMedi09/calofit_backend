from app.core.fuente_alimentos import contrastar, texto_fuente


def test_contraste_con_la_tabla_ins_cenan():
    assert contrastar("Pollo", 170) == "INS/CENAN"
    assert contrastar("Arroz", 130) == "INS/CENAN"
    assert contrastar("Pollo", 900) is None  # densidad imposible para un pollo
    assert contrastar("Pan con huevo", 230) is None  # plato preparado: "pan de huevo" es otro alimento
    assert contrastar("Lomo saltado", 150) is None
    assert contrastar("", 100) is None and contrastar("Pollo", 0) is None


def test_texto_fuente():
    assert texto_fuente([("Arroz", "INS/CENAN"), ("Pollo", "INS/CENAN"), ("Lomo saltado", None)]) == (
        "Contrastado con INS/CENAN: Arroz, Pollo · Valor referencial: Lomo saltado"
    )
    assert texto_fuente([("Lomo saltado", None)]) == "Valor referencial: Lomo saltado"
    assert texto_fuente([]) == ""
