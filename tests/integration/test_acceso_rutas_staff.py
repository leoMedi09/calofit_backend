"""Control de acceso del personal: contraseñas de clientes/empleados, creación de planes, límite de peticiones y cabecera.

Reproduce lo que halló la revisión de seguridad (un coach cambiaba la clave del admin, un nutricionista no asignado
operaba sobre clientes ajenos, POST /nutricion/ fallaba siempre por un método inexistente).
"""
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from app.api.routes.auth import get_current_staff
from app.core import rate_limit
from app.core.database import get_db
from app.main import app
from app.models import Client, Role, User
from app.services.ia_service import ia_engine

NUEVA = {"new_password": "Cambiada2024!", "confirm_password": "Cambiada2024!"}
PLAN = {"genero": 1, "edad": 25, "peso": 70, "talla": 170, "nivel_actividad": 1.2, "objetivo": "mantener"}


def _staff(db, email, rol):
    r = db.query(Role).filter(Role.name == rol).first() or Role(name=rol, description=rol)
    db.add(r)
    db.flush()
    u = User(first_name=rol, last_name_paternal="P", last_name_maternal="M", email=email, hashed_password="hash-original",
             role_id=r.id, role_name=rol, is_active=True)
    db.add(u)
    db.flush()
    return u


@pytest.fixture
def escenario(db):
    rate_limit._golpes.clear()
    sufijo = datetime.utcnow().timestamp()
    admin = _staff(db, f"admin_{sufijo}@t.com", "admin")
    nutri_a = _staff(db, f"na_{sufijo}@t.com", "nutricionista")
    nutri_b = _staff(db, f"nb_{sufijo}@t.com", "nutricionista")
    coach = _staff(db, f"co_{sufijo}@t.com", "coach")
    cliente = Client(first_name="C", last_name_paternal="P", last_name_maternal="M", email=f"c_{sufijo}@t.com",
                     hashed_password="hash-original", birth_date=datetime(1990, 1, 1).date(), weight=70.0, height=175,
                     gender="M", medical_conditions=[], activity_level="moderado", goal="perdida_de_peso",
                     is_profile_complete=True, assigned_nutri_id=nutri_a.id)
    db.add(cliente)
    db.commit()

    actual = {}
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_staff] = lambda: actual["u"]

    class E:
        pass

    e = E()
    e.db, e.cliente = db, cliente
    e.admin, e.nutri_a, e.nutri_b, e.coach = admin, nutri_a, nutri_b, coach
    e.http = TestClient(app)
    e.como = lambda u: actual.__setitem__("u", u)
    yield e
    app.dependency_overrides.clear()
    rate_limit._golpes.clear()


def _clave(db, tabla, id_):
    db.expire_all()
    return db.get(tabla, id_).hashed_password


# --- cambiar la contraseña de un CLIENTE ---------------------------------------------------------------------------
@pytest.mark.parametrize("quien", ["coach", "nutri_b"])
def test_staff_sin_asignacion_no_cambia_clave_de_cliente(escenario, quien):
    escenario.como(getattr(escenario, quien))
    r = escenario.http.put(f"/clientes/{escenario.cliente.id}/cambiar-contrasena", json=NUEVA)
    assert r.status_code == 403
    assert _clave(escenario.db, Client, escenario.cliente.id) == "hash-original"


@pytest.mark.parametrize("quien", ["nutri_a", "admin"])
def test_nutricionista_asignado_y_admin_si_cambian_clave_de_cliente(escenario, quien):
    escenario.como(getattr(escenario, quien))
    r = escenario.http.put(f"/clientes/{escenario.cliente.id}/cambiar-contrasena", json=NUEVA)
    assert r.status_code == 200
    assert _clave(escenario.db, Client, escenario.cliente.id) != "hash-original"


# --- cambiar la contraseña de un EMPLEADO (solo admin) -----------------------------------------------------------------
@pytest.mark.parametrize("quien,objetivo", [("coach", "admin"), ("nutri_b", "nutri_a"), ("nutri_a", "coach")])
def test_solo_el_admin_cambia_clave_de_empleados(escenario, quien, objetivo):
    escenario.como(getattr(escenario, quien))
    destino = getattr(escenario, objetivo)
    r = escenario.http.put(f"/clientes/usuario/{destino.id}/cambiar-contrasena", json=NUEVA)
    assert r.status_code == 403
    assert _clave(escenario.db, User, destino.id) == "hash-original"


def test_admin_si_cambia_clave_de_empleado(escenario):
    escenario.como(escenario.admin)
    r = escenario.http.put(f"/clientes/usuario/{escenario.coach.id}/cambiar-contrasena", json=NUEVA)
    assert r.status_code == 200
    assert _clave(escenario.db, User, escenario.coach.id) != "hash-original"


# --- crear plan nutricional ---------------------------------------------------------------------------------------------
def test_coach_no_crea_plan(escenario):
    escenario.como(escenario.coach)
    assert escenario.http.post("/nutricion/", json={**PLAN, "client_id": escenario.cliente.id}).status_code == 403


def test_nutricionista_no_asignado_no_crea_plan(escenario):
    escenario.como(escenario.nutri_b)
    r = escenario.http.post("/nutricion/", json={**PLAN, "client_id": escenario.cliente.id})
    assert r.status_code == 403


def test_cliente_inexistente_da_404(escenario):
    escenario.como(escenario.admin)
    assert escenario.http.post("/nutricion/", json={**PLAN, "client_id": 99999999}).status_code == 404


@pytest.mark.parametrize("quien", ["nutri_a", "admin"])
def test_nutricionista_asignado_y_admin_crean_plan_con_7_dias(escenario, quien):
    escenario.como(getattr(escenario, quien))
    r = escenario.http.post("/nutricion/", json={**PLAN, "client_id": escenario.cliente.id})
    assert r.status_code == 200, r.text
    dias = r.json()["detalles_diarios"]
    assert len(dias) == 7
    assert dias[0]["sugerencia_entrenamiento_ia"] and "Recuperación" in dias[6]["sugerencia_entrenamiento_ia"]


def test_sugerencia_de_entrenamiento_por_objetivo():
    assert ia_engine.generar_sugerencia_entrenamiento("ganar", 1).startswith("Fuerza")
    assert ia_engine.generar_sugerencia_entrenamiento("perder", 3).startswith("Gasto")
    assert ia_engine.generar_sugerencia_entrenamiento("mantener", 5).startswith("Tonificación")
    assert ia_engine.generar_sugerencia_entrenamiento("ganar", 6).startswith("Día de Recuperación")


# --- límite de peticiones y cabecera -----------------------------------------------------------------------------------
def test_cambio_de_clave_tiene_limite_de_peticiones(escenario):
    escenario.como(escenario.admin)
    codigos = [escenario.http.put(f"/clientes/{escenario.cliente.id}/cambiar-contrasena", json=NUEVA).status_code for _ in range(22)]
    assert codigos.count(200) == 20 and codigos[-1] == 429


def test_cabecera_nosniff_en_todas_las_respuestas():
    r = TestClient(app).get("/health")
    assert r.headers.get("x-content-type-options") == "nosniff"
