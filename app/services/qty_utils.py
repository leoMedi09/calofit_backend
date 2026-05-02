from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Optional, Tuple


def _norm_simple(s: str) -> str:
    s = (s or "").strip().lower()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_qty_token(token: str) -> float:
    """
    Cantidades simples en español/numéricas:
    - "2", "1.5", "1,5"
    - "una", "dos", ...
    - "media/medio" -> 0.5
    """
    p = (token or "").strip().lower()
    if not p:
        return 1.0
    m = re.match(r"^\s*(\d+(?:[.,]\d+)?)\b", p)
    if m:
        return float(m.group(1).replace(",", "."))
    mapa = {
        "un": 1.0,
        "uno": 1.0,
        "una": 1.0,
        "dos": 2.0,
        "tres": 3.0,
        "cuatro": 4.0,
        "cinco": 5.0,
        "media": 0.5,
        "medio": 0.5,
    }
    for k, v in mapa.items():
        if p.startswith(k):
            return float(v)
    return 1.0


def singularizar_es(palabra: str) -> str:
    """
    Singularización naive para UX:
    - hamburguesas -> hamburguesa
    - panes -> pan (aprox.)
    No pretende ser lingüísticamente perfecta; solo ayuda a mejorar hits en BD/IA.
    """
    w = _norm_simple(palabra)
    if len(w) < 3:
        return w
    # Casos comunes en comida peruana
    # "ceviches"/"cebiches" -> "ceviche"/"cebiche"
    if w.endswith("ches") and len(w) > 5:
        return w[:-1]  # quitar solo "s"
    if w.endswith("es") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    return w


_RX_QTY_COUNT = re.compile(
    r"(?i)^\s*(?P<qty>\d+(?:[.,]\d+)?|media|medio|un|una|uno|dos|tres|cuatro|cinco)\s+(?P<name>.+?)\s*$"
)


_RX_QTY_UNIT_DE = re.compile(
    r"(?i)^\s*(?P<qty>\d+(?:[.,]\d+)?|media|medio|un|una|uno|dos|tres|cuatro|cinco)\s+"
    r"(?P<unit>botellas?|vasos?|tazas?|latas?|tarros?|botes?)\s+de\s+(?P<name>.+?)\s*$"
)


def parse_count_qty_phrase(texto: str) -> Optional[Tuple[float, str]]:
    """
    Detecta frases tipo:
      - "dos hamburguesas"
      - "media hamburguesa de pollo"
    Retorna (qty, name_part) sin los prefijos de "comi/..." (debe limpiarlo el caller).
    """
    t = (texto or "").strip()
    if not t:
        return None
    m = _RX_QTY_COUNT.match(t)
    if not m:
        return None
    qty = parse_qty_token(m.group("qty"))
    name = (m.group("name") or "").strip()
    if qty <= 0 or not name:
        return None
    return qty, name


def parse_qty_unit_de_phrase(texto: str) -> Optional[Tuple[float, str, str]]:
    """
    Detecta: "media botella de gaseosa", "2 botellas de gaseosa", "1 vaso de agua".
    Retorna (qty, unit, name).
    """
    t = (texto or "").strip()
    if not t:
        return None
    m = _RX_QTY_UNIT_DE.match(t)
    if not m:
        return None
    qty = parse_qty_token(m.group("qty"))
    unit = _norm_simple(m.group("unit"))
    name = (m.group("name") or "").strip()
    if qty <= 0 or not unit or not name:
        return None
    return qty, unit, name


def scale_extraccion(extraccion: Dict[str, Any], qty: float) -> Dict[str, Any]:
    """
    Escala una extracción comida (kcal y macros) por qty.
    Mantiene flags y listas; ajusta origen para trazabilidad.
    """
    if not extraccion or qty is None:
        return extraccion
    q = float(qty)
    if q <= 0:
        return extraccion
    out = dict(extraccion)
    for k in ("calorias", "proteinas_g", "carbohidratos_g", "grasas_g", "fibra_g", "azucar_g", "sodio_mg"):
        if k in out and out[k] is not None:
            try:
                out[k] = round(float(out[k]) * q, 1)
            except Exception:
                pass
    # Porción: si existe, también escalar
    if "porcion_g" in out and out["porcion_g"] is not None:
        try:
            out["porcion_g"] = round(float(out["porcion_g"]) * q, 1)
        except Exception:
            pass
    origen = str(out.get("origen") or "").strip()
    out["origen"] = (origen + " + scaled_qty").strip(" +")
    return out

