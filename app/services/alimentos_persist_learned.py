"""
Tras un registro por NLP con estimación (Groq/LLM), persiste un fila mínima en
``alimentos`` (macros por 100g) para que búsquedas futuras (AlimentosDBService,
CENAN, etc.) resuelvan el mismo alimento y no haya cifras distintas entre
consultas. No sustituye filas existentes (mismo ``nombre_normalizado``).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from app.models.alimento import Alimento
from app.services.alimentos_db_service import _norm


def upsert_alimento_aprendido_desde_extraccion(
    db: Session,
    extraccion: Dict[str, Any],
) -> Optional[int]:
    """
    Si la extracción viene de LLM y es comida con kcal>0, inserta alimento
    con macros por 100g inferidos desde la porción estimada.
    """
    if not extraccion or not extraccion.get("es_comida"):
        return None
    origen = (extraccion.get("origen") or "")
    olow = re.sub(r"\s+", " ", str(origen).lower())
    if "llm" not in olow and "llama" not in olow and "groq" not in olow:
        return None

    cals = float(extraccion.get("calorias") or 0)
    if cals < 0.5:
        return None

    nombres = extraccion.get("alimentos_detectados") or []
    nombre = (nombres[0] if nombres else None) or "Alimento (IA)"
    nombre = str(nombre).strip()[:255]
    nn = _norm(nombre)
    if not nn:
        return None

    existing = db.query(Alimento).filter(Alimento.nombre_normalizado == nn).first()
    if existing:
        return int(existing.id)

    porcion = float(extraccion.get("porcion_g") or 0)
    if porcion < 1.0:
        porcion = 100.0
    f_100 = 100.0 / porcion

    cal_100 = round(cals * f_100, 2)
    p_100 = round(float(extraccion.get("proteinas_g") or 0) * f_100, 2)
    c_100 = round(float(extraccion.get("carbohidratos_g") or 0) * f_100, 2)
    g_100 = round(float(extraccion.get("grasas_g") or 0) * f_100, 2)
    if cal_100 < 0.1:
        return None

    fib = extraccion.get("fibra_g")
    fib_100 = round(float(fib) * f_100, 2) if fib is not None else None
    azu = extraccion.get("azucar_g")
    azu_100 = round(float(azu) * f_100, 2) if azu is not None else None

    row = Alimento(
        nombre=nombre,
        nombre_normalizado=nn,
        calorias_100g=cal_100,
        proteina_100g=p_100,
        carbohidratos_100g=c_100,
        grasas_100g=g_100,
        fibra_100g=fib_100,
        azucar_100g=azu_100,
        categoria="aprendido_ia",
        fuente=origen[:200] if origen else "llm_estimado",
        id_externo=None,
    )
    db.add(row)
    db.flush()
    return int(row.id)
