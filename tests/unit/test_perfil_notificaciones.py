"""GET /clientes/perfil debe devolver el valor guardado de notificaciones_activas (antes siempre devolvia True)."""
import pytest

from app.api.routes.clientes import obtener_perfil_cliente
from app.models.client import Client


def _cliente(valor):
    return Client(id=1, first_name="A", last_name_paternal="B", last_name_maternal="C", email="a@b.com", flutter_uid="u",
                  weight=70.0, height=170.0, is_profile_complete=False, notificaciones_activas=valor)


@pytest.mark.parametrize("guardado, esperado", [(False, False), (True, True), (None, True)])
def test_perfil_devuelve_preferencia_guardada(guardado, esperado):
    assert obtener_perfil_cliente(current_user=_cliente(guardado)).notificaciones_activas is esperado
