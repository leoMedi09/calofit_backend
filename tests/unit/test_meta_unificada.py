from types import SimpleNamespace

from app.services.asistente import asistente_plan as ap


def test_meta_unificada_usa_la_regla_del_asistente(monkeypatch):
    perfil = SimpleNamespace(birth_date=None)
    monkeypatch.setattr(
        ap,
        "obtener_plan_hoy",
        lambda p, e, db: (None, {"calorias_dia": 2587, "proteinas_g": 190, "carbohidratos_g": 250, "grasas_g": 70}, False),
    )
    assert ap.obtener_macros_plan_hoy(perfil, None) == {
        "calorias_dia": 2587.0,
        "proteinas_g": 190.0,
        "carbohidratos_g": 250.0,
        "grasas_g": 70.0,
    }


def test_meta_unificada_devuelve_none_si_falla(monkeypatch):
    def _falla(*a):
        raise ValueError("plan incompleto")

    monkeypatch.setattr(ap, "obtener_plan_hoy", _falla)
    assert ap.obtener_macros_plan_hoy(SimpleNamespace(birth_date=None), None) is None
