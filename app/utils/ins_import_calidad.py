"""
Criterios para conservar filas INS/CENAN 2017 (Importado): nombre legible y macros plausibles.
"""
from __future__ import annotations

import re
from typing import Any

# Nombres INS que no queremos volver a importar (cultivar muy específico, etc.).
_NOMBRES_RAW_EXCLUIDOS: frozenset[str] = frozenset(
    {
        "al 110 maternizada",
    }
)
_SUBCADENAS_NOMBRE_EXCLUIDAS: tuple[str, ...] = (
    "ajos variedad",  # barranquino, napuri, molidos en aceite/agua…
)


def _carb_key(item: dict[str, Any]) -> str | None:
    if "carbohindratos_100g" in item:
        return "carbohindratos_100g"
    if "carbohidratos_100g" in item:
        return "carbohidratos_100g"
    return None


def ins_nombre_raw_formato_bueno(raw: str) -> bool:
    """Nombre tal como viene en JSON: sin notas *, poca puntuación tipo catálogo, UTF-8 sano."""
    s = (raw or "").strip()
    if len(s) < 3 or len(s) > 220:
        return False
    low = s.lower()
    # Catálogo INS muy granular / poco útil en app (acuerdo producto).
    if low in _NOMBRES_RAW_EXCLUIDOS:
        return False
    if any(sub in low for sub in _SUBCADENAS_NOMBRE_EXCLUIDAS):
        return False
    if "?" in s or "\ufffd" in s or "�" in s:
        return False
    if "*" in s:
        return False
    if any(ch in s for ch in "\t\n\r"):
        return False
    if "  " in s:
        return False
    # Comas de catálogo (no contar decimales tipo 12,0)
    sin_decimal = re.sub(r"\d+,\d+", "", s)
    if sin_decimal.count(",") > 1:
        return False
    if s.count("(") > 2:
        return False
    return True


def ins_macros_formato_bueno(item: dict[str, Any]) -> bool:
    """Valores por 100 g finitos y en rangos razonables para alimentos."""
    ck = _carb_key(item)
    if not ck:
        return False
    try:
        cal = float(item.get("calorias_100g", -1) or -1)
        prot = float(item.get("proteina_100g", -1) or -1)
        carb = float(item.get(ck, -1) or -1)
        gras = float(item.get("grasas_100g", -1) or -1)
        fib = float(item.get("fibra_100g") or 0)
        azu = float(item.get("azucar_100g") or 0)
    except (TypeError, ValueError):
        return False
    for v in (cal, prot, carb, gras, fib, azu):
        if v != v:  # NaN
            return False
    if cal < 0 or cal > 950:
        return False
    if prot < 0 or prot > 100 or carb < 0 or carb > 100 or gras < 0 or gras > 100:
        return False
    if fib < 0 or fib > 150 or azu < 0 or azu > 100:
        return False
    if cal == 0 and prot == 0 and carb == 0 and gras == 0:
        return False
    return True


def ins_item_formato_bueno(item: dict[str, Any]) -> bool:
    raw = item.get("alimento")
    if not raw or not isinstance(raw, str):
        return False
    if not ins_nombre_raw_formato_bueno(raw):
        return False
    if not ins_macros_formato_bueno(item):
        return False
    return True
