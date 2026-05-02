"""
Cliente FatSecret Platform API (OAuth 2.0 client_credentials).

Uso: búsqueda `foods.search` + macros desde `food_description` (p. ej. Per 100g).
Las credenciales vienen de ``settings`` / variables de entorno, nunca hardcodeadas.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Union

import httpx

from app.core.config import settings

_TOKEN_URL = "https://oauth.fatsecret.com/connect/token"
_API_URL = "https://platform.fatsecret.com/rest/server.api"


def _parse_food_description(desc: str) -> tuple[float, Dict[str, float]]:
    """
    Interpreta líneas tipo:
    ``Per 100g - Calories: 22kcal | Fat: 0.34g | Carbs: 3.28g | Protein: 3.09g``
    Devuelve (gramos_base_referencia, {calorias, proteinas_g, carbohidratos_g, grasas_g} por esa base).
    """
    if not desc:
        return 100.0, {"calorias": 0.0, "proteinas_g": 0.0, "carbohidratos_g": 0.0, "grasas_g": 0.0}

    base_g = 100.0
    m_per = re.search(r"per\s+([\d.]+)\s*g\b", desc, re.IGNORECASE)
    if m_per:
        base_g = float(m_per.group(1))

    def _f(pat: str) -> float:
        x = re.search(pat, desc, re.IGNORECASE)
        if not x:
            return 0.0
        return float(x.group(1).replace(",", "."))

    cal = _f(r"calories:\s*([\d.,]+)\s*kcal")
    if cal == 0:
        cal = _f(r"([\d.,]+)\s*kcal")

    prot = _f(r"protein:\s*([\d.,]+)\s*g")
    carb = _f(r"carbs:\s*([\d.,]+)\s*g")
    fat = _f(r"fat:\s*([\d.,]+)\s*g")

    return base_g, {
        "calorias": cal,
        "proteinas_g": prot,
        "carbohidratos_g": carb,
        "grasas_g": fat,
    }


def _normalize_food_list(food_block: Union[Dict, List, None]) -> List[Dict[str, Any]]:
    if food_block is None:
        return []
    if isinstance(food_block, list):
        return [f for f in food_block if isinstance(f, dict)]
    if isinstance(food_block, dict):
        return [food_block]
    return []


class FatSecretClient:
    """Cliente mínimo: token + foods.search."""

    def __init__(self, client_id: str, client_secret: str):
        self._client_id = (client_id or "").strip()
        self._client_secret = (client_secret or "").strip()
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0

    def configured(self) -> bool:
        return bool(self._client_id and self._client_secret)

    def _refresh_token(self) -> None:
        with httpx.Client(timeout=25.0) as client:
            r = client.post(
                _TOKEN_URL,
                auth=(self._client_id, self._client_secret),
                data={
                    "grant_type": "client_credentials",
                    "scope": "basic",
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            r.raise_for_status()
            data = r.json()
        self._access_token = data.get("access_token")
        if not self._access_token:
            raise RuntimeError("FatSecret: respuesta sin access_token")
        ttl = int(data.get("expires_in", 3500))
        self._token_expires_at = time.time() + max(60, ttl - 120)

    def _ensure_token(self) -> None:
        if not self.configured():
            raise RuntimeError("FatSecret: faltan FATSECRET_CLIENT_ID / FATSECRET_CLIENT_SECRET")
        if self._access_token and time.time() < self._token_expires_at:
            return
        self._refresh_token()

    def foods_search(self, search_expression: str, max_results: int = 5) -> Dict[str, Any]:
        self._ensure_token()
        expr = (search_expression or "").strip()[:200]
        if len(expr) < 2:
            return {}
        with httpx.Client(timeout=30.0) as client:
            r = client.post(
                _API_URL,
                headers={
                    "Authorization": f"Bearer {self._access_token}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "method": "foods.search",
                    "search_expression": expr,
                    "format": "json",
                    "max_results": str(min(50, max(1, max_results))),
                },
            )
            r.raise_for_status()
            return r.json()

    def lookup_macros(
        self,
        search_expression: str,
        porcion_g: float,
        max_results: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """
        Busca un alimento y devuelve macros escalados a ``porcion_g`` según la referencia
        del texto (p. ej. Per 100g → factor porcion_g/100).
        Formato compatible con ``extraer_macros_de_texto`` (nutricion_service / IA).
        """
        raw = self.foods_search(search_expression, max_results=max_results)
        foods = raw.get("foods") or {}
        food_block = foods.get("food")
        candidates = _normalize_food_list(food_block)
        if not candidates:
            return None

        first = candidates[0]
        desc = str(first.get("food_description") or "")
        base_g, per = _parse_food_description(desc)
        if base_g <= 0:
            base_g = 100.0
        cal = float(per.get("calorias") or 0)
        if cal <= 0:
            return None

        factor = float(porcion_g or 100.0) / base_g
        nombre = str(first.get("food_name") or search_expression).strip()
        brand = (first.get("brand_name") or "").strip()
        if brand:
            nombre = f"{nombre} ({brand})"

        return {
            "nombre": nombre,
            "alimento": nombre,
            "origen": "FatSecret API",
            "calorias": round(cal * factor, 1),
            "proteinas": round(float(per["proteinas_g"]) * factor, 1),
            "carbohidratos": round(float(per["carbohidratos_g"]) * factor, 1),
            "grasas": round(float(per["grasas_g"]) * factor, 1),
            "fibra": 0.0,
            "azucares": 0.0,
            "sodio": 0.0,
            "_fatsecret_food_id": first.get("food_id"),
            "_fatsecret_description": desc,
            "_fatsecret_base_g": base_g,
        }


_fs_instance: Optional[FatSecretClient] = None
_fs_negative_cache: bool = False


def get_fatsecret_client() -> Optional[FatSecretClient]:
    """Singleton; ``None`` si no hay credenciales o están vacías."""
    global _fs_instance, _fs_negative_cache
    if _fs_negative_cache:
        return None
    if _fs_instance is not None:
        return _fs_instance
    cid = (getattr(settings, "FATSECRET_CLIENT_ID", None) or "").strip()
    sec = (getattr(settings, "FATSECRET_CLIENT_SECRET", None) or "").strip()
    if not cid or not sec:
        _fs_negative_cache = True
        return None
    _fs_instance = FatSecretClient(cid, sec)
    return _fs_instance


def simplify_text_for_fatsecret_query(texto: str) -> str:
    """Reduce ruido del mensaje del usuario para la búsqueda."""
    t = (texto or "").strip()
    if not t:
        return ""
    low = t.lower()
    for pref in (
        "registra ",
        "registrá ",
        "anota ",
        "apunta ",
        "ya comí ",
        "ya comi ",
        "comí ",
        "comi ",
        "mi almuerzo:",
        "mi desayuno:",
        "mi cena:",
    ):
        if low.startswith(pref):
            t = t[len(pref) :].strip()
            low = t.lower()
    # Primera línea o segmento antes de coma larga
    t = t.split("\n")[0].strip()
    if "," in t and len(t) > 40:
        t = t.split(",")[0].strip()
    return t[:120].strip()
