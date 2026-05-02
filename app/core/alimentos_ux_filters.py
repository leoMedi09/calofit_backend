"""
Filtros de UX para nombres de alimentos (INS/CENAN con errores de OCR o poco conocidos).

No sustituye revisión nutricional: solo evita sugerencias confusas en IA y KNN.
"""
from __future__ import annotations

from typing import Iterable, Optional

# Subcadenas en nombre de alimento → no usar en matching ni en sugerencias ML.
# "eledón" aparece por datos INS importados; no es un alimento cotidiano reconocible.
BLOQUE_SUBCADENAS_ALIMENTO: tuple[str, ...] = (
    "eledón",
    "eledon",
)


def es_alimento_bloqueado_ia(nombre: Optional[str]) -> bool:
    if not nombre:
        return False
    n = str(nombre).lower()
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
