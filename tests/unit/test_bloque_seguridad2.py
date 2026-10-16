"""Plan de seguridad, bloque 2: borrado completo de un cliente, acceso por asignacion, logs enmascarados y limite de peticiones."""
import asyncio
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.api.routes import nutricionista as nut
from app.core import log_masking, rate_limit
from app.services.eliminacion_cliente import eliminar_datos_cliente


# ------------------------------------------------------------------ borrado completo
def test_eliminacion_borra_todo_y_el_cliente_va_al_final():
    db = MagicMock()
    eliminar_datos_cliente(db, 5, "a@b.com")
    sqls = [str(c.args[0]) for c in db.execute.call_args_list]
    for tabla in ("workout_logs", "metas_usuario", "historial_recomendaciones", "chat_historial", "comida_registros",
                  "alimentos_sin_resolver", "password_resets"):
        assert any(tabla in s for s in sqls), tabla
    assert "planes_diarios" in sqls[0]  # los planes diarios antes que los planes
    assert "DELETE FROM clients" in sqls[-1]  # el cliente, lo ultimo


def _url_local():
    pw = os.getenv("POSTGRES_PASSWORD")
    if not pw:
        env = Path(__file__).resolve().parents[3] / ".env"
        if env.exists():
            for linea in env.read_text(encoding="utf-8").splitlines():
                if linea.startswith("POSTGRES_PASSWORD="):
                    pw = linea.split("=", 1)[1].strip()
    return f"postgresql://postgres:{pw}@127.0.0.1:5433/BD_Calofit" if pw else None


def test_eliminacion_contra_la_bd_local_deja_cero_filas():
    """Crea un cliente de prueba con datos en las tablas problematicas, lo elimina y revierte todo (nada persiste)."""
    url = _url_local()
    if not url:
        pytest.skip("sin POSTGRES_PASSWORD")
    try:
        engine = create_engine(url, connect_args={"connect_timeout": 3})
        conn = engine.connect()
    except Exception:
        pytest.skip("BD local de Docker no disponible")
    trans = conn.begin()
    db = Session(bind=conn)
    try:
        email = "prueba-eliminacion@calofit.test"
        cid = db.execute(
            text("INSERT INTO clients(first_name,last_name_paternal,last_name_maternal,email,hashed_password,flutter_uid,gender) "
                 "VALUES ('T','T','T',:e,'x','uid-prueba-eliminacion','M') RETURNING id"),
            {"e": email},
        ).scalar()
        p = {"c": cid}
        db.execute(text("INSERT INTO workout_logs(client_id,ejercicio,series,reps,created_at) VALUES (:c,'Press',3,10,NOW())"), p)
        db.execute(text("INSERT INTO metas_usuario(client_id,genero,edad,peso_kg,talla_cm,nivel_actividad,objetivo,tmb,get,"
                        "calorias_objetivo,proteinas_g,carbohidratos_g,grasas_g) VALUES (:c,'M',30,70,170,'x','x',1,1,1,1,1,1)"), p)
        db.execute(text("INSERT INTO historial_recomendaciones(client_id) VALUES (:c)"), p)
        db.execute(text("INSERT INTO chat_historial(client_id,rol,contenido) VALUES (:c,'user','hola')"), p)
        db.execute(text("INSERT INTO alimentos_sin_resolver(nombre_original,user_id) VALUES ('xyz',:c)"), p)
        db.execute(text("INSERT INTO password_resets(email,reset_code) VALUES (:e,'h')"), {"e": email})

        conteo = eliminar_datos_cliente(db, cid, email)

        assert conteo["clients"] == 1 and conteo["workout_logs"] == 1
        assert conteo["metas_usuario"] == 1 and conteo["historial_recomendaciones"] == 1
        for tabla, col, val in [("clients", "id", cid), ("workout_logs", "client_id", cid), ("metas_usuario", "client_id", cid),
                                ("historial_recomendaciones", "client_id", cid), ("chat_historial", "client_id", cid),
                                ("alimentos_sin_resolver", "user_id", cid), ("password_resets", "email", email)]:
            assert db.execute(text(f"SELECT COUNT(*) FROM {tabla} WHERE {col} = :v"), {"v": val}).scalar() == 0, tabla
    finally:
        trans.rollback()
        conn.close()


# ------------------------------------------------------------------ acceso por asignacion en el panel
FUNCIONES = ["get_patient_progress", "get_patient_daily_log", "suggest_strategic_guide", "update_strategic_guide",
             "validate_plan", "get_client_plan", "update_client_plan", "save_coach_note"]


def _llamar(nombre, usuario):
    fn = getattr(nut, nombre)
    cliente = SimpleNamespace(id=5, assigned_nutri_id=2, assigned_coach_id=3, flutter_uid=None, email="c@x.com")
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = cliente
    kw = {"id": 5, "db": db, "current_user": usuario}
    for nombre_p, par in inspect.signature(fn).parameters.items():
        if nombre_p not in kw and par.default is inspect.Parameter.empty:
            kw[nombre_p] = MagicMock()
    r = fn(**kw)
    return asyncio.run(r) if inspect.iscoroutine(r) else r


@pytest.mark.parametrize("nombre", FUNCIONES)
@pytest.mark.parametrize("usuario", [
    SimpleNamespace(id=9, role_name="entrenador"),     # entrenador NO asignado
    SimpleNamespace(id=8, role_name="nutricionista"),  # nutricionista NO asignado
])
def test_staff_no_asignado_recibe_403(nombre, usuario):
    with pytest.raises(HTTPException) as e:
        _llamar(nombre, usuario)
    assert e.value.status_code == 403


# ------------------------------------------------------------------ logs enmascarados
def test_enmascarar_correo_token_y_clave():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhYmMifQ.firmafirmafirma"
    m = log_masking.enmascarar(f"login ana.perez@gmail.com password=Secreto123 token={jwt} Authorization: Bearer abcdefghijklmnop1234")
    assert "ana.perez@gmail.com" not in m and "a***@gmail.com" in m
    assert "Secreto123" not in m and jwt not in m and "abcdefghijklmnop1234" not in m


def test_enmascarar_no_toca_texto_normal():
    assert log_masking.enmascarar("Registré: Arroz con pollo — 650 kcal") == "Registré: Arroz con pollo — 650 kcal"


def test_flujo_enmascarado_envuelve_stdout():
    import sys

    log_masking.instalar()
    try:
        assert isinstance(sys.stdout, log_masking._FlujoEnmascarado)
        assert isinstance(sys.stderr, log_masking._FlujoEnmascarado)
    finally:
        sys.stdout = sys.stdout._flujo
        sys.stderr = sys.stderr._flujo


# ------------------------------------------------------------------ limite de peticiones
def _req(ip, cf=None):
    cabeceras = {"x-forwarded-for": f"1.1.1.1, {ip}, 10.9.9.9"}  # cliente falsificado, IP real, IP que agrega el proxy
    if cf:
        cabeceras["cf-connecting-ip"] = cf
    return SimpleNamespace(headers=cabeceras, client=SimpleNamespace(host="10.0.0.1"))


def test_limite_bloquea_al_pasar_el_maximo_y_es_por_ip():
    rate_limit._golpes.clear()
    dep = rate_limit.limitar("prueba", 3, 60)
    for _ in range(3):
        dep(_req("2.2.2.2"))
    with pytest.raises(HTTPException) as e:
        dep(_req("2.2.2.2"))
    assert e.value.status_code == 429
    dep(_req("3.3.3.3"))  # otra IP no se ve afectada


def test_limite_prefiere_cf_connecting_ip_y_no_se_evade_falsificando_la_cabecera():
    rate_limit._golpes.clear()
    dep = rate_limit.limitar("prueba2", 1, 60)
    dep(SimpleNamespace(headers={"cf-connecting-ip": "8.8.8.8", "x-forwarded-for": "9.9.9.1, 8.8.8.8, 10.0.0.5"}, client=None))
    with pytest.raises(HTTPException):  # el atacante cambia el primer valor de la cabecera y sigue siendo la misma IP
        dep(SimpleNamespace(headers={"cf-connecting-ip": "8.8.8.8", "x-forwarded-for": "9.9.9.2, 8.8.8.8, 10.0.0.5"}, client=None))


def test_limite_sin_cloudflare_usa_el_penultimo_valor():
    rate_limit._golpes.clear()
    dep = rate_limit.limitar("prueba3", 1, 60)
    dep(_req("4.4.4.4"))
    with pytest.raises(HTTPException):
        dep(SimpleNamespace(headers={"x-forwarded-for": "7.7.7.7, 4.4.4.4, 10.9.9.9"}, client=None))


# ------------------------------------------------------------------ borrado de cuenta por el propio usuario
def test_eliminar_mi_cuenta_exige_cliente_y_contrasena_correcta(monkeypatch):
    from app.api.routes import clientes as cl
    from app.models.client import Client

    with pytest.raises(HTTPException) as e:  # personal del gimnasio: no es Client
        cl.eliminar_mi_cuenta(cl.EliminarCuenta(password="x"), SimpleNamespace(id=1), MagicMock())
    assert e.value.status_code == 403

    cliente = MagicMock(spec=Client)
    cliente.id, cliente.email, cliente.flutter_uid, cliente.hashed_password = 5, "c@x.com", "uid", "h"
    monkeypatch.setattr(cl.security, "verify_password", lambda p, h: False)
    monkeypatch.setattr("app.core.firebase_rest.password_valida_en_firebase", lambda *a: False)
    llamado = []
    monkeypatch.setattr("app.services.eliminacion_cliente.eliminar_datos_cliente", lambda *a: llamado.append(a))
    db = MagicMock()
    with pytest.raises(HTTPException) as e:
        cl.eliminar_mi_cuenta(cl.EliminarCuenta(password="mala"), cliente, db)
    assert e.value.status_code == 401 and not llamado and not db.commit.called

    monkeypatch.setattr(cl.security, "verify_password", lambda p, h: True)
    borrados = []
    monkeypatch.setattr(cl.firebase_admin_auth, "delete_user", lambda uid: borrados.append(uid))
    assert cl.eliminar_mi_cuenta(cl.EliminarCuenta(password="ok"), cliente, db)["status"] == "success"
    assert llamado and borrados == ["uid"] and db.commit.called
