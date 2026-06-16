"""
Filtros de UX para nombres de alimentos (INS/CENAN con errores de OCR o poco conocidos).

No sustituye revisión nutricional: solo evita sugerencias confusas en IA y KNN.
"""
from __future__ import annotations

from typing import Iterable, Optional

# Subcadenas: si alguna aparece en el nombre del alimento (lowercased) → bloquear.
# Criterios: artifacts OCR de INS/CENAN, fauna exótica andina/amazónica no consumida
# en la costa lambayecana, algas andinas, y vísceras extremas inapropiadas para un
# contexto de nutrición deportiva.
BLOQUE_SUBCADENAS_ALIMENTO: tuple[str, ...] = (
    # OCR artifacts INS/CENAN
    "eledón",
    "eledon",
    # Fauna exótica andina/amazónica — no disponible en Lambayeque
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
    "cuy, carne",   # "cuy" solo false-positiva con "maracuyá" → ver BLOQUE_EXACTO_ALIMENTO
    "rana, carne",  # "rana" solo false-positiva con "granada"/"granadilla"
    # Algas andinas
    "cushuro",
    "nostoc",
    # Vísceras extremas — confusas en recomendaciones deportivas
    "sesos",
    "criadilla",
    "sangre",
    "ubre",
    "bofe",
    # Legumbres/granos de regiones amazónicas o andinas remotas — no reconocibles
    # en Lambayeque y generan nombres de plato extraños cuando el LLM los usa literalmente
    "poroto de cumbasa",   # Ucayali
    "frejol ucayalino",    # Amazonía
    "frejol shimpe",       # variedad amazónica
    "frejol tarhui",       # Andino (chocho crudo/harina)
    "frejol vacapaleta",   # variedad regional remota
    "frejol terciopelo",   # Mucuna pruriens — legumbre forrajera, no alimentaria cotidiana
    "frejol zarandaja",    # variedad andina poco conocida
    "frejol nucya",        # variedad andina poco conocida
    "hemico leguminoso",   # nombre técnico de laboratorio, no es un alimento cotidiano
    "chocho",              # tarwi crudo — amargo, requiere preparación especial
)

# Nombres exactos (lowercased, stripped) que no son capturables de forma segura
# por BLOQUE_SUBCADENAS sin producir falsos positivos.
BLOQUE_EXACTO_ALIMENTO: frozenset[str] = frozenset({
    "cuy",  # entrada standalone en catálogo KNN; "cuy" sola matchea "maracuyá"
})


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
