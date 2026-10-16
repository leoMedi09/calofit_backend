"""Limite de peticiones por IP, en memoria (una sola instancia del servidor). Se usa como dependencia de FastAPI:

    @router.post("/login", dependencies=[Depends(limitar("login", 30, 60))])

La IP real sale de CF-Connecting-IP o del penultimo valor de X-Forwarded-For (ver _ip); sin proxy, la del socket.
Al reiniciarse el servidor los contadores se reinician; no sustituye el limite de 5 intentos del codigo de recuperacion."""
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

_lock = threading.Lock()
_golpes: dict = defaultdict(deque)
_MAX_CLAVES = 5000


def _ip(request: Request) -> str:
    """IP real del usuario detras de Cloudflare + Render. El proxy de Render AGREGA la IP al final de X-Forwarded-For sin borrar
    lo que mande el cliente, asi que el primer valor es falsificable. Orden: CF-Connecting-IP (lo escribe Cloudflare) y, si no
    llega, el penultimo valor de X-Forwarded-For (el anterior a la IP que agrega el proxy de Render)."""
    cf = request.headers.get("cf-connecting-ip")
    if cf:
        return cf.strip()
    xff = [x.strip() for x in request.headers.get("x-forwarded-for", "").split(",") if x.strip()]
    if len(xff) >= 2:
        return xff[-2]
    if xff:
        return xff[0]
    return request.client.host if request.client else "desconocida"


def limitar(nombre: str, maximo: int, segundos: int):
    def dependencia(request: Request) -> None:
        ahora = time.monotonic()
        clave = (nombre, _ip(request))
        with _lock:
            if len(_golpes) > _MAX_CLAVES:  # limpieza para que la memoria no crezca sin limite
                for k in [k for k, q in _golpes.items() if not q or ahora - q[-1] > segundos]:
                    del _golpes[k]
            cola = _golpes[clave]
            while cola and ahora - cola[0] > segundos:
                cola.popleft()
            if len(cola) >= maximo:
                raise HTTPException(
                    status_code=429,
                    detail="Demasiados intentos. Espera unos minutos e inténtalo de nuevo.",
                    headers={"Retry-After": str(segundos)},
                )
            cola.append(ahora)

    return dependencia
