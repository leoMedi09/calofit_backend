"""Filtros del KNN: casos reales que el KNN devolvía mal (ficha de validación del nutricionista, 02/10/2026)."""
from app.core.knn_filtros import excluir, familia


def test_lactosa_bloquea_formula_infantil_y_cereal_con_leche():
    for n in ("Nan maternizada", "Cerelac trigo", "Milo - nestle", "Queso mantecoso"):
        assert excluir(n, ["Intolerancia a la Lactosa"]), n
    assert not excluir("Frejol canario", ["Intolerancia a la Lactosa"])


def test_hipertension_bloquea_sodio_alto():
    for n in ("Pescado boquichico, salado deshidratado", "Pollo, pechuga frita de**", "Camaroncito seco (chino)", "Pescado merluza, seco"):
        assert excluir(n, ["Hipertensión"]), n
    assert not excluir("Quinua", ["Hipertensión"])


def test_lista_evitar_se_interpreta_como_categoria():
    assert excluir("Cerdo, churrasco c/piel*", [], ["Carnes rojas con grasa"])
    assert excluir("Club social integral tradicional - nabisco", [], ["Alimentos procesados"])
    assert excluir("Milo - nestle", [], ["Bebidas azucaradas"])


def test_nunca_recomendar_condimentos_ni_suplementos():
    for n in ("Maca, almidón", "Levadura seca**", "Palillo", "Ají dulce seco, molido (paprika)", "Bollito de yema de huánuco"):
        assert excluir(n), n


def test_desayuno_sin_pescado_ni_legumbres_y_familias():
    assert excluir("Pescado caballa, en conserva", momento="DESAYUNO")
    assert not excluir("Quinua", momento="DESAYUNO")
    assert familia("Frejol chavín") == familia("Pallar seco") == "legumbre"


def test_sodio_en_lista_evitar_y_sardina_en_hipertension():
    assert excluir("Camaroncito seco (chino)", [], ["Alimentos con alto contenido de sodio"])
    assert excluir("Pescado sardina, con tomate", ["Hipertensión"])


def test_solo_alimentos_peruanos_conocidos():
    for n in ("Frejol bush bean", "Frejol red kidney", "Quinua, variedad cica-127 (puno)", "Papa variedad nativa huayro", "Res, hígado frito**",
              "Pescado boquichico, salado deshidratado", "Hiwicto marfil oscuro", "Original - club social", "Inka Cola", "Leche materna"):
        assert excluir(n), n
    for n in ("Frejol canario", "Quinua", "Lentejas chicas", "Papa blanca", "Pescado jurel, fresco", "Pollo, pechuga cocida de**", "Pan francés", "Maní tostado"):
        assert not excluir(n), n


def test_nombre_legible_sin_asteriscos_ni_de_final():
    from app.core.knn_filtros import nombre_legible
    esperado = {"Pollo, pechuga cocida de**": "Pechuga cocida de pollo", "Quinua, hojuelas de": "Hojuelas de quinua", "Res, cuadril, churrasco***": "Churrasco de res (cuadril)",
                "Pescado tilapia, cruda*": "Pescado tilapia cruda", "Semilla de ajonjolí*": "Semilla de ajonjolí", "Pato, carne de": "Carne de pato", "Frejol canario": "Frejol canario"}
    for k, v in esperado.items():
        assert nombre_legible(k) == v, (k, nombre_legible(k))


def test_alimentos_poco_conocidos_y_ultraprocesados():
    for n in ("Sorgo", "Pecana", "Maíz, morado sin coronta", "Codorniz, carne cocida de **", "7 semillas", "Cerdo, churrasco c/piel*", "Arvejón"):
        assert excluir(n), n
    assert excluir("Pan de molde", [], ["Alimentos procesados"])
    assert not excluir("Pan de molde")


def test_frejol_caballero_no_se_recomienda():
    assert excluir("Frejol caballero", [])  # poco conocido: no entra al KNN
    assert not excluir("Frejol canario", [])
