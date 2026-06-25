"""
Prueba de la deduplicación de ingredientes cuando el LLM extrae tanto un
plato compuesto reconocido como sus propios ingredientes por separado.

Caso encontrado en la auditoría final pre-demo: "Me metí un poco de arroz con
lentejas" registraba "Arroz con lentejas" + "lentejas" + "arroz" como 3 ítems,
duplicando las kcal. Ver llm_registro.py::_filtrar_componentes_de_plato_compuesto.
"""
import pytest

from app.services.llm_registro import _filtrar_componentes_de_plato_compuesto


@pytest.mark.unit
class TestFiltrarComponentesDePlatoCompuesto:

    def test_descarta_ingredientes_ya_cubiertos_por_el_plato_completo(self):
        alimentos = [
            {"nombre": "Arroz con lentejas", "kcal": 500},
            {"nombre": "lentejas", "kcal": 200},
            {"nombre": "arroz", "kcal": 180},
        ]
        resultado = _filtrar_componentes_de_plato_compuesto(alimentos)
        assert len(resultado) == 1
        assert resultado[0]["nombre"] == "Arroz con lentejas"

    def test_no_descarta_alimentos_independientes_reales(self):
        alimentos = [
            {"nombre": "Arroz con pollo", "kcal": 600},
            {"nombre": "Gelatina", "kcal": 90},
        ]
        resultado = _filtrar_componentes_de_plato_compuesto(alimentos)
        assert len(resultado) == 2

    def test_lista_de_un_solo_item_no_se_toca(self):
        alimentos = [{"nombre": "Huevos", "kcal": 138}]
        assert _filtrar_componentes_de_plato_compuesto(alimentos) == alimentos

    def test_lista_vacia(self):
        assert _filtrar_componentes_de_plato_compuesto([]) == []

    def test_dos_platos_compuestos_distintos_se_mantienen(self):
        alimentos = [
            {"nombre": "Pollo con arroz", "kcal": 600},
            {"nombre": "Tallarines con pollo", "kcal": 550},
        ]
        # Ninguno es subconjunto completo del otro ("arroz" vs "tallarines"
        # no coinciden) — ambos deben conservarse.
        resultado = _filtrar_componentes_de_plato_compuesto(alimentos)
        assert len(resultado) == 2
