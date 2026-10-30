"""El metabolismo basal sale de una sola fórmula (Mifflin-St Jeor) en todos los módulos."""
from types import SimpleNamespace

from app.core.macros_diarios import tmb_mifflin_st_jeor
from app.core.utils import calcular_metabolismo_basal
from app.services.calculador_dieta import CalculadorDietaAutomatica
from app.services.ia_service import ia_engine


def test_valores_de_referencia():
    assert tmb_mifflin_st_jeor(70, 175, 25, True) == 1673.75
    assert tmb_mifflin_st_jeor(70, 175, 25, False) == 1507.75


def test_los_tres_modulos_coinciden_con_ia_service():
    ia = ia_engine.calcular_requerimiento(1, 25, 70, 175, 1.0, "mantener")
    assert ia == 1673.75
    assert CalculadorDietaAutomatica.calcular_gasto_metabolico_basal(70, 175, 25, "M") == round(ia)
    from datetime import date

    hoy = date.today()
    c = SimpleNamespace(birth_date=date(hoy.year - 25, 1, 1) if (hoy.month, hoy.day) >= (1, 1) else None,
                        gender="M", weight=70, height=175, activity_level="Sedentario")
    assert abs(calcular_metabolismo_basal(c) - 1673.75 * 1.2) < 1e-6
