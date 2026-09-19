"""
Nombres legibles para alimentos INS/CENAN y clave normalizada para búsqueda/resolución.

La clave incluye la letra «ñ»; antes se perdía con patrones ASCII [a-z0-9] solamente.
"""

from __future__ import annotations

import re
import unicodedata

_LETTERS = "a-z0-9ñ"
_PH_N = "\ue000"


def norm_alimento_key(s: str) -> str:
    """Minúsculas, sin tildes, ñ preservada, solo letras dígitos y espacio."""
    s = (s or "").strip().lower()
    if not s:
        return ""
    s = s.replace("ñ", _PH_N)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace(_PH_N, "ñ")
    s = re.sub(rf"[^{_LETTERS}\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


_PRETTY_OVERRIDES: dict[str, str] = {
    "abridores": "Abridores (fruta — código INS C-1)",
}


def pretty_nombre_ins(raw: str) -> str:
    """
    Convierte la denominación tipo tabla (muchas comas, notas *) en texto más legible en UI.
    No altera macros ni id; solo nombre de presentación.
    """
    s = (raw or "").strip()
    if not s:
        return s
    low = s.lower()
    if low in _PRETTY_OVERRIDES:
        return _PRETTY_OVERRIDES[low]

    s = re.sub(r"\*+\s*$", "", s).strip()
    s = re.sub(r",\s*\(", " (", s)
    s = re.sub(r",\s+", " · ", s)
    s = re.sub(r"\s+", " ", s).strip()

    parts = [p.strip() for p in s.split(" · ") if p.strip()]
    titled: list[str] = []
    for p in parts:
        titled.append(p[0].upper() + p[1:] if len(p) > 1 else p.upper())
    return " · ".join(titled)
