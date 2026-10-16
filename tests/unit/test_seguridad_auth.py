"""Plan de seguridad (oct-2026): rutas sin sesion eliminadas, login que no confia en firebase_uid,
limite de intentos del codigo de recuperacion y borrado de clientes solo por admin/nutricionista asignado."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import HTTPException

from app.api.routes import auth as auth_mod
from app.api.routes.nutricionista import delete_client
from app.core import firebase_rest
from app.core.security import security
from app.models.client import Client
from app.schemas.user import UserLogin


def _rutas():
    from app.main import app

    return {getattr(r, "path", "") for r in app.routes}


def test_rutas_que_cambiaban_contrasenas_sin_sesion_ya_no_existen():
    rutas = _rutas()
    for fin in ("/sync-firebase-password", "/sync-password", "/verify-and-sync-password", "/forgot-password/request", "/forgot-password/verify"):
        assert not any(r.endswith(fin) and ("auth" in r or "clientes" in r) for r in rutas), fin


# ---------------------------------------------------------------- verificacion en Firebase
def _resp(codigo, cuerpo):
    return SimpleNamespace(status_code=codigo, json=lambda: cuerpo)


def test_firebase_falla_cerrado_sin_clave(monkeypatch):
    monkeypatch.setattr(firebase_rest.settings, "FIREBASE_API_KEY", "", raising=False)
    monkeypatch.setattr(firebase_rest.httpx, "post", lambda *a, **k: pytest.fail("no debe llamar a Firebase sin clave"))
    assert firebase_rest.password_valida_en_firebase("a@b.com", "x", "uid1") is False


def test_firebase_acepta_solo_si_el_uid_coincide(monkeypatch):
    monkeypatch.setattr(firebase_rest.settings, "FIREBASE_API_KEY", "k", raising=False)
    monkeypatch.setattr(firebase_rest.httpx, "post", lambda *a, **k: _resp(200, {"localId": "uid1"}))
    assert firebase_rest.password_valida_en_firebase("a@b.com", "x", "uid1") is True
    assert firebase_rest.password_valida_en_firebase("a@b.com", "x", "otro") is False


def test_firebase_rechaza_contrasena_incorrecta_y_errores_de_red(monkeypatch):
    monkeypatch.setattr(firebase_rest.settings, "FIREBASE_API_KEY", "k", raising=False)
    monkeypatch.setattr(firebase_rest.httpx, "post", lambda *a, **k: _resp(400, {"error": {}}))
    assert firebase_rest.password_valida_en_firebase("a@b.com", "x", "uid1") is False

    def caida(*a, **k):
        raise httpx.ConnectError("sin red")

    monkeypatch.setattr(firebase_rest.httpx, "post", caida)
    assert firebase_rest.password_valida_en_firebase("a@b.com", "x", "uid1") is False


# ---------------------------------------------------------------- login
def _cliente(uid="uid1"):
    return Client(id=7, first_name="Ana", last_name_paternal="P", last_name_maternal="M", email="ana@x.com",
                  hashed_password=security.hash_password("la-real"), flutter_uid=uid, weight=60.0, height=160.0,
                  is_profile_complete=True)


def _db(user):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = user
    return db


def test_login_no_acepta_cualquier_firebase_uid_con_contrasena_falsa(monkeypatch):
    """El hueco original: contrasena equivocada + cualquier firebase_uid = entrar y reescribir la contrasena."""
    user = _cliente()
    hash_original = user.hashed_password
    monkeypatch.setattr(auth_mod, "password_valida_en_firebase", lambda *a, **k: False)
    with pytest.raises(HTTPException) as e:
        auth_mod.login(UserLogin(email="ana@x.com", password="atacante", firebase_uid="uid1", user_type="client"), _db(user))
    assert e.value.status_code == 401
    assert user.hashed_password == hash_original


def test_login_resincroniza_si_firebase_confirma(monkeypatch):
    user = _cliente()
    monkeypatch.setattr(auth_mod, "password_valida_en_firebase", lambda *a, **k: True)
    r = auth_mod.login(UserLogin(email="ana@x.com", password="nueva-en-firebase", firebase_uid="uid1", user_type="client"), _db(user))
    assert r["access_token"] and security.verify_password("nueva-en-firebase", user.hashed_password)


def test_login_rechaza_uid_distinto_al_guardado(monkeypatch):
    user = _cliente(uid="uid1")
    monkeypatch.setattr(auth_mod, "password_valida_en_firebase", lambda *a, **k: True)
    with pytest.raises(HTTPException) as e:
        auth_mod.login(UserLogin(email="ana@x.com", password="x", firebase_uid="uid-ajeno", user_type="client"), _db(user))
    assert e.value.status_code == 401


def test_login_normal_con_contrasena_correcta():
    r = auth_mod.login(UserLogin(email="ana@x.com", password="la-real", user_type="client"), _db(_cliente()))
    assert r["token_type"] == "bearer"


# ---------------------------------------------------------------- codigo de recuperacion
def _registro(codigo, intentos=0, usado=False, vigente=True):
    return SimpleNamespace(reset_code=auth_mod._hash_reset_code(codigo), intentos=intentos, is_used=usado, is_expired=lambda: not vigente)


def _db_reset(registro):
    db = MagicMock()
    db.query.return_value.filter.return_value.order_by.return_value.first.return_value = registro
    return db


def test_codigo_correcto_pasa():
    reg = _registro("123456")
    assert auth_mod._validar_codigo_reset(_db_reset(reg), "a@b.com", "123456") is reg


def test_codigo_se_bloquea_al_quinto_intento_fallido():
    reg = _registro("123456")
    db = _db_reset(reg)
    for _ in range(auth_mod.MAX_INTENTOS_CODIGO):
        with pytest.raises(HTTPException):
            auth_mod._validar_codigo_reset(db, "a@b.com", "000000")
    assert reg.intentos == auth_mod.MAX_INTENTOS_CODIGO
    with pytest.raises(HTTPException):  # ya bloqueado: ni el codigo correcto sirve
        auth_mod._validar_codigo_reset(db, "a@b.com", "123456")


# ---------------------------------------------------------------- borrado de clientes
@pytest.mark.parametrize("rol", ["entrenador", "coach"])
def test_entrenador_no_puede_eliminar_clientes(rol):
    cliente = SimpleNamespace(id=5, assigned_nutri_id=2, flutter_uid=None)
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = cliente
    with pytest.raises(HTTPException) as e:
        delete_client(5, db=db, current_user=SimpleNamespace(id=9, role_name=rol))
    assert e.value.status_code == 403
    db.delete.assert_not_called()


def test_nutricionista_no_asignado_no_puede_eliminar():
    cliente = SimpleNamespace(id=5, assigned_nutri_id=2, flutter_uid=None)
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = cliente
    with pytest.raises(HTTPException) as e:
        delete_client(5, db=db, current_user=SimpleNamespace(id=3, role_name="nutricionista"))
    assert e.value.status_code == 403
