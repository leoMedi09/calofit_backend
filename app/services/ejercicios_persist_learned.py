"""
Persistencia de ejercicios "aprendidos".

Objetivo: cuando el NLP detecta un ejercicio pero no existe en la BD (o se estimó con fallback),
guardarlo en ``ejercicios`` (con MET) para que futuras consultas/registro sean consistentes
y DB-first pueda resolverlo sin recalcular distinto.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.models.ejercicio import Ejercicio


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\\s]+", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s


def upsert_ejercicio_aprendido_desde_extraccion(
    db: Session,
    extraccion: Dict[str, Any],
) -> Optional[int]:
    if not extraccion or not extraccion.get("es_ejercicio") or extraccion.get("es_comida"):
        return None

    origen = (extraccion.get("origen") or "")
    olow = re.sub(r"\\s+", " ", str(origen).lower())
    # Guardar cuando sea estimado/fallback (no para cada ejercicio estándar ya conocido).
    if "fallback" not in olow and "llm" not in olow and "groq" not in olow:
        return None

    met = float(extraccion.get("met") or 0)
    if met <= 0:
        return None

    ejercicios = extraccion.get("ejercicios_detectados") or []
    nombre = str(ejercicios[0] if ejercicios else "").strip()
    if not nombre:
        return None
    # Normalizar rótulos como "X (45 min)" → "X"
    nombre = re.sub(r"\\(\\s*\\d+\\s*min\\s*\\)", "", nombre, flags=re.IGNORECASE).strip()
    nombre = nombre[:255]
    nn = _norm(nombre)
    if not nn:
        return None

    existing = db.query(Ejercicio).filter(Ejercicio.nombre_normalizado == nn).first()
    if existing:
        return int(existing.id)

    row = Ejercicio(
        nombre=nombre,
        nombre_normalizado=nn,
        alias=None,
        descripcion=None,
        met=met,
        grupo_muscular=None,
        origen="aprendido_fallback",
    )
    db.add(row)
    db.flush()
    return int(row.id)

