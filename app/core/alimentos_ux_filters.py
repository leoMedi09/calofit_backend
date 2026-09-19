"""
Filtros de UX para nombres de alimentos (INS/CENAN con errores de OCR o poco conocidos).

No sustituye revisión nutricional: solo evita sugerencias confusas en IA y KNN.
"""

from __future__ import annotations

from typing import Iterable, Optional

BLOQUE_SUBCADENAS_ALIMENTO: tuple[str, ...] = (
    "eledón",
    "eledon",
    "llama",
    "alpaca",
    "venado",
    "sajino",
    "majaz",
    "motelo",
    "taricaya",
    "charapa",
    "tortuga",
    "paiche",
    "aska",
    "ziqui",
    "cuy, carne",
    "rana, carne",
    "cushuro",
    "nostoc",
    "sesos",
    "criadilla",
    "sangre",
    "ubre",
    "bofe",
    "poroto de cumbasa",
    "frejol ucayalino",
    "frejol shimpe",
    "frejol tarhui",
    "frejol vacapaleta",
    "frejol terciopelo",
    "frejol zarandaja",
    "frejol nucya",
    "hemico leguminoso",
    "chocho",
)

BLOQUE_EXACTO_ALIMENTO: frozenset[str] = frozenset(
    {
        "cuy",
    }
)


def es_alimento_bloqueado_ia(nombre: Optional[str]) -> bool:
    if not nombre:
        return False
    n = str(nombre).lower().strip()
    if n in BLOQUE_EXACTO_ALIMENTO:
        return True
    return any(b in n for b in BLOQUE_SUBCADENAS_ALIMENTO)


def nombre_coincide_exclusion(nombre: str, exclusiones: Iterable[str]) -> bool:
    """True si el nombre debe excluirse por coincidencia con lista reciente (normalizado)."""
    n = (nombre or "").lower().strip()
    if not n:
        return False
    for ex in exclusiones:
        e = (ex or "").lower().strip()
        if not e:
            continue
        if e in n or n in e:
            return True
        for part in e.split():
            if len(part) > 4 and part in n:
                return True
    return False
