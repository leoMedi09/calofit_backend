"""Eliminacion COMPLETA de los datos de un cliente (la base no borra en cascada la mayoria de las tablas)."""
from sqlalchemy import text

# tablas con columna client_id (nombres fijos, no vienen del usuario)
_TABLAS_CLIENT_ID = (
    "workout_logs",
    "metas_usuario",
    "historial_recomendaciones",
    "chat_historial",
    "alertas_salud",
    "comida_registros",
    "historial_imc",
    "historial_peso",
    "preferencias_alimentos",
    "preferencias_ejercicios",
    "progreso_calorias",
    "sugerencias_guardadas",
)
# tablas con user_id (la base solo les pone NULL, pero conservarian el texto del usuario)
_TABLAS_USER_ID = ("alimentos_sin_resolver", "app_cache_alimentos", "app_cache_platos")


def eliminar_datos_cliente(db, client_id: int, email: str | None = None) -> dict:
    """Borra todo lo del cliente y al cliente, dentro de la transaccion actual (no hace commit). Devuelve filas por tabla."""
    p = {"c": client_id}
    conteo = {}
    conteo["planes_diarios"] = db.execute(
        text("DELETE FROM planes_diarios WHERE plan_id IN (SELECT id FROM planes_nutricionales WHERE client_id = :c)"), p
    ).rowcount
    for tabla in _TABLAS_CLIENT_ID + ("planes_nutricionales",):
        conteo[tabla] = db.execute(text(f"DELETE FROM {tabla} WHERE client_id = :c"), p).rowcount
    for tabla in _TABLAS_USER_ID:
        conteo[tabla] = db.execute(text(f"DELETE FROM {tabla} WHERE user_id = :c"), p).rowcount
    if email:
        conteo["password_resets"] = db.execute(text("DELETE FROM password_resets WHERE email = :e"), {"e": email}).rowcount
    conteo["clients"] = db.execute(text("DELETE FROM clients WHERE id = :c"), p).rowcount
    return conteo
