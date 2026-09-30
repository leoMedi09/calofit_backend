"""Nombres de rol del staff — fuente única de verdad.

La BD tiene variantes históricas de texto libre en `users.role_name`
(NUTRI, nutritionist, nutricionista, coach, entrenador...). Estas funciones
son el único lugar que decide qué variante cuenta como qué rol; todo el
resto del código debe llamarlas en vez de comparar strings a mano.
"""

_ADMIN = {"admin", "administrador"}
_NUTRI = {"nutricionista", "nutritionist", "nutri"}
_COACH = {"coach", "entrenador", "trainer"}


def _norm(role_name) -> str:
    return str(role_name or "").strip().lower()


def es_admin(role_name) -> bool:
    return _norm(role_name) in _ADMIN


def es_nutricionista(role_name) -> bool:
    return _norm(role_name) in _NUTRI


def es_entrenador(role_name) -> bool:
    return _norm(role_name) in _COACH


def es_staff(role_name) -> bool:
    return es_admin(role_name) or es_nutricionista(role_name) or es_entrenador(role_name)


def verificar_acceso_cliente(current_user, cliente) -> None:
    """Un cliente solo ve su propio perfil; el staff ve el de sus pacientes asignados
    (o cualquiera, si es admin). Se usa en todo endpoint que reciba un cliente_id
    en la URL, para que no cualquiera pueda leer los datos de otro cliente.
    """
    from fastapi import HTTPException

    denegado = HTTPException(status_code=403, detail="No tienes permiso para acceder a este perfil")

    role_name = getattr(current_user, "role_name", None)
    if role_name is None:
        if current_user.id != cliente.id:
            raise denegado
        return

    if es_admin(role_name):
        return
    if es_nutricionista(role_name) and cliente.assigned_nutri_id == current_user.id:
        return
    if es_entrenador(role_name) and cliente.assigned_coach_id == current_user.id:
        return
    raise denegado


def verificar_admin(current_user) -> None:
    from fastapi import HTTPException

    if not es_admin(getattr(current_user, "role_name", "")):
        raise HTTPException(status_code=403, detail="Solo el administrador puede realizar esta acción")
