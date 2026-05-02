"""
Caché en proceso (sin Redis).

Misma API que antes: consultas del asistente, comidas recientes, alimentos en nutrición_unificado.
En multi-worker cada proceso tiene su propia caché; para un solo uvicorn es suficiente.
"""
import json
import threading
import time
from typing import Any, Optional

from app.core.config import settings

_lock = threading.Lock()
# key -> (expiry_epoch_seconds, json_str)
_store: dict[str, tuple[float, str]] = {}

_CACHE_PREFIX = "calofit"
_CONSULTA_TTL = 600
_RECENT_MEALS_TTL = 3600
_ALIMENTO_TTL = 86400 * 7
_EJERCICIO_TTL = 86400 * 7


def _cache_debug(msg: str) -> None:
    if settings.DEBUG:
        print(msg)


def _full_key(key: str) -> str:
    return f"{_CACHE_PREFIX}:{key}"


def _purge_expired_unlocked(now: float) -> None:
    dead = [k for k, (exp, _) in _store.items() if exp <= now]
    for k in dead:
        del _store[k]


def get_cached(key: str) -> Optional[Any]:
    rkey = _full_key(key)
    now = time.time()
    with _lock:
        _purge_expired_unlocked(now)
        tup = _store.get(rkey)
        if not tup:
            _cache_debug(f"CACHE MISS [{rkey}]")
            return None
        exp, raw = tup
        if exp <= now:
            del _store[rkey]
            return None
        try:
            val = json.loads(raw)
            _cache_debug(f"CACHE HIT [{rkey}]")
            return val
        except Exception as e:
            print(f"CACHE GET JSON [{rkey}]: {e}")
            return None


def set_cached(key: str, value: Any, ttl_seconds: int = _ALIMENTO_TTL) -> bool:
    rkey = _full_key(key)
    now = time.time()
    try:
        raw = json.dumps(value, ensure_ascii=False)
    except Exception as e:
        print(f"CACHE SET JSON [{rkey}]: {e}")
        return False
    with _lock:
        _purge_expired_unlocked(now)
        _store[rkey] = (now + float(ttl_seconds), raw)
        _cache_debug(f"CACHE SAVE [{rkey}] TTL={ttl_seconds}s")
    return True


def get_consulta_cached(consulta_id: str) -> Optional[dict]:
    return get_cached(f"consulta:{consulta_id}")


def set_consulta_cached(consulta_id: str, payload: dict) -> bool:
    return set_cached(f"consulta:{consulta_id}", payload, ttl_seconds=_CONSULTA_TTL)


def cache_key_alimento(nombre_normalizado: str) -> str:
    return f"alimento:{nombre_normalizado.lower().strip()}"


def cache_key_ejercicio(nombre_normalizado: str) -> str:
    return f"ejercicio:{nombre_normalizado.lower().strip()}"


def get_user_recent_meals(user_id: int) -> list:
    val = get_cached(f"recent_meals:{user_id}")
    _cache_debug(f"CACHE DEBUG: get_user_recent_meals({user_id}) -> {val}")
    return val if val else []


def add_user_recent_meal(user_id: int, payload: dict) -> bool:
    meals = get_user_recent_meals(user_id)
    meals = [m for m in meals if m.get("nombre", "").lower() != payload.get("nombre", "").lower()]
    meals.insert(0, payload)
    if len(meals) > 10:
        meals = meals[:10]
    return set_cached(f"recent_meals:{user_id}", meals, ttl_seconds=_RECENT_MEALS_TTL)
