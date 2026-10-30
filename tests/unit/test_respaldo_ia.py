import asyncio
from types import SimpleNamespace

from app.services.ia_service import IAService


def _cliente(resultado=None, error=None):
    async def create(**kw):
        if error:
            raise error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=resultado))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _servicio(principal, respaldo):
    s = IAService.__new__(IAService)
    s.groq_client, s.groq_backup_client = principal, respaldo
    return s


def test_sin_credito_usa_el_respaldo():
    s = _servicio(_cliente(error=Exception("Error en OpenRouter API (402): requires more credits")), _cliente("ok-respaldo"))
    assert asyncio.run(s._llamar_groq("hola", model="llama-3.3-70b-versatile")) == "ok-respaldo"


def test_un_error_normal_no_gasta_el_respaldo():
    s = _servicio(_cliente(error=Exception("500 internal")), _cliente("ok-respaldo"))
    assert asyncio.run(s._llamar_groq("hola", model="llama-3.3-70b-versatile")).startswith("[Error:")


def test_principal_ok():
    s = _servicio(_cliente("ok-principal"), _cliente("ok-respaldo"))
    assert asyncio.run(s._llamar_groq("hola", model="llama-3.3-70b-versatile")) == "ok-principal"
