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
